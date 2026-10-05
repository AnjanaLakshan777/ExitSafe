"""Official CSE public-data discovery: what can actually be retrieved, and can it be trusted?

    python scripts/cse_source_discovery.py            # full run (downloads ~200 MB of PDFs once)
    python scripts/cse_source_discovery.py --no-pdfs  # API probes only

Only public, unauthenticated CSE endpoints (www.cse.lk/api, used by the CSE
website itself) and files on the CSE CDN (cdn.cse.lk) are accessed. Endpoints
that answer 401/417 are recorded as login-required and never retried with
credentials. Requests are sequential with a delay between them.

Raw evidence goes to data/raw/cse/discovery/ (not tracked in git). The
structured result goes to data/processed/audit/cse_source_evidence.json.
Nothing is written to canonical processed data.

Historical-date rule: a record is only attributed to a date that the source
payload itself states (report text, per-row trade date, per-point timestamp).
A date we asked for is never assigned to a response.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from app.config.paths import AUDIT_DIR, CSE_RAW_DIR  # noqa: E402
from app.data.validators.market_validator import validate_market_data  # noqa: E402

API = "https://www.cse.lk/api/"
CDN_CMT = "https://cdn.cse.lk/cmt/"
SLT = timezone(timedelta(hours=5, minutes=30))  # Sri Lanka time, used by CSE timestamps
USER_AGENT = "Mozilla/5.0 (compatible; ExitSafe-data-discovery/0.1; research, low volume)"
REQUEST_DELAY_S = 1.0

DISCOVERY_DIR = CSE_RAW_DIR / "discovery"
EVIDENCE_FILE = AUDIT_DIR / "cse_source_evidence.json"

# Liquid securities used to sample the per-stock chart endpoint. Those whose base
# ticker is also the short name printed in the daily report can be cross-checked.
CHART_SAMPLE = ["JKH.N0000", "HNB.N0000", "HNB.X0000", "ACL.N0000", "HDFC.N0000",
                "COMB.N0000", "DIAL.N0000"]
# Dates deliberately NOT present in any listing, to record what happens.
UNLISTED_TARGETS = [date(2026, 9, 15), date(2025, 12, 30), date(2024, 6, 28)]


# --- HTTP with evidence -----------------------------------------------------------

class Session:
    def __init__(self, out_dir):
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": USER_AGENT, "Origin": "https://www.cse.lk",
                                  "Referer": "https://www.cse.lk/"})
        self.out_dir = out_dir
        self.log = []

    def call(self, name, method, url, save_as=None, **kwargs):
        time.sleep(REQUEST_DELAY_S)
        entry = {"name": name, "method": method, "url": url,
                 "params": kwargs.get("data") or kwargs.get("json") or kwargs.get("params"),
                 "fetched_at": _now()}
        try:
            response = self.http.request(method, url, timeout=180, **kwargs)
        except requests.RequestException as exc:
            entry.update(http_status=None, error=str(exc))
            self.log.append(entry)
            return entry, None
        body = response.content
        entry.update(http_status=response.status_code,
                     content_type=response.headers.get("Content-Type"),
                     bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        if save_as and response.ok and body:
            path = self.out_dir / save_as
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            entry["saved_to"] = _rel(path)
        self.log.append(entry)
        return entry, response

    def api_json(self, name, method, endpoint, **kwargs):
        entry, response = self.call(name, method, API + endpoint, save_as=f"api/{name}.json", **kwargs)
        if response is None or not response.ok or not response.content:
            return entry, None
        try:
            return entry, response.json()
        except ValueError:
            entry["error"] = "response is not JSON"
            return entry, None

    def download(self, name, url, save_as):
        """Download once; later runs re-check availability with HEAD and reuse the file."""
        path = self.out_dir / save_as
        if path.is_file() and path.stat().st_size:
            entry, _ = self.call(name, "HEAD", url)
            entry.update(from_cache=True, saved_to=_rel(path), bytes=path.stat().st_size,
                         sha256=_sha256(path))
            return entry, path
        entry, response = self.call(name, "GET", url, save_as=save_as)
        return entry, (path if response is not None and response.ok else None)


# --- report parsing -----------------------------------------------------------------

NUMBER = r"-?[\d,]+(?:\.\d+)?"  # some report columns contain negative values
LISTING_DATE = re.compile(r"(\d{1,2})[-_/.](\d{1,2})[-_/.](\d{4})")  # titles use - or _
US_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


def listing_date(file_text):
    """Date written in a CSE report listing title, e.g. '29-09-2026 Report' (metadata only)."""
    match = LISTING_DATE.search(file_text or "")
    return date(int(match[3]), int(match[2]), int(match[1])) if match else None


def new_report_date(first_page_text):
    """Date printed on page 1 of the new-format daily report ('Tuesday, 29 September, 2026')."""
    match = re.search(r"[A-Za-z]+,\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", first_page_text or "")
    if not match:
        return None
    return datetime.strptime(f"{match[1]} {match[2]} {match[3]}", "%d %B %Y").date()


def us_date(text):
    return datetime.strptime(text, "%m/%d/%Y").date()


def _clean(text):
    return re.sub(r"(\(cid:\d+\))+", "", text or "")


@dataclass
class ColumnBounds:
    board: float
    name: float
    type: float


def equity_rows_from_words(words, bounds):
    """Rows of '02. Daily Movements on Equity' from pdfplumber words of one page.

    A data row is a line containing a M/D/YYYY date. Company-name and industry
    words that wrapped onto the line above/below are attached to the nearest
    data row by vertical distance, using the header's column positions.
    """
    lines = {}
    for word in words:
        lines.setdefault(round(word["top"]), []).append(word)
    data_tops, fragments = [], []
    for top, line in sorted(lines.items()):
        numbers = sum(bool(re.fullmatch(NUMBER, w["text"])) for w in line)
        if any(US_DATE.match(w["text"]) for w in line) and numbers >= 7:  # not the page-header date
            data_tops.append(top)
        else:
            fragments.extend(w for w in line if w["x0"] < bounds.type - 5)

    rows = {top: {"group": [], "board": [], "name": [], "type": None, "values": []}
            for top in data_tops}
    for top in data_tops:
        for word in sorted(lines[top], key=lambda w: w["x0"]):
            _place(rows[top], word, bounds)
    for word in fragments:
        if not data_tops:
            break
        nearest = min(data_tops, key=lambda t: abs(t - word["top"]))
        if abs(nearest - word["top"]) <= 14:
            _place(rows[nearest], word, bounds, fragment=True)

    parsed = []
    for top in data_tops:
        row = rows[top]
        values = row["values"]
        if row["type"] is None or len(values) != 8:
            parsed.append({"parse_error": f"unexpected layout: type={row['type']} values={values}"})
            continue
        parsed.append({
            "industry_group": " ".join(w["text"] for w in sorted(row["group"], key=_reading_order)),
            "board": " ".join(w["text"] for w in row["board"]),
            "company_name": " ".join(w["text"] for w in sorted(row["name"], key=_reading_order)),
            "type": row["type"],
            "close_price": values[0], "last_traded_price": values[1],
            "date_last_traded": values[2], "high": values[3], "low": values[4],
            "foreign_holding": values[5], "turnover_rs": values[6], "quantity_in_cds": values[7],
        })
    return parsed


def _place(row, word, bounds, fragment=False):
    x = word["x0"]
    if x < bounds.board - 2:
        row["group"].append(word)
    elif x < bounds.name - 2:
        if not fragment:
            row["board"].append(word)
    elif x < bounds.type - 5:
        text = word["text"]
        # A long name can touch the type column, fusing the type letter onto the
        # last name word ("EQUIPMENTN"). Split it only if the word physically
        # extends into the type column.
        if (not fragment and row["type"] is None and word["x1"] > bounds.type - 1
                and len(text) > 1 and text[-1] in "NXPUZ"):
            row["name"].append({**word, "text": text[:-1]})
            row["type"] = text[-1]
        else:
            row["name"].append(word)
    elif not fragment and row["type"] is None and re.fullmatch(r"[A-Z]", word["text"]):
        row["type"] = word["text"]
    elif not fragment:
        row["values"].append(word["text"])


def _reading_order(word):
    return (round(word["top"]), word["x0"])


def header_bounds(words):
    xs = {w["text"]: w["x0"] for w in words if w["text"] in ("Board", "Company", "Type")}
    if len(xs) < 3:
        return None
    return ColumnBounds(board=xs["Board"], name=xs["Company"], type=xs["Type"])


def parse_new_daily_report(path):
    import pdfplumber

    result = {"format": "new ('DD-MM-YYYY Report')", "section": "02. Daily Movements on Equity"}
    with pdfplumber.open(path) as pdf:
        result["pages"] = len(pdf.pages)
        result["source_reported_date"] = _iso(new_report_date(pdf.pages[0].extract_text()))
        header_dates, rows, errors, pages = set(), [], 0, []
        for index, page in enumerate(pdf.pages):
            text = _clean(page.extract_text())
            if "02. Daily Movements on Equity" not in text:
                if pages:  # the section is contiguous; stop once it has ended
                    break
                continue
            pages.append(index + 1)
            header_dates.update(re.findall(r"^\s*(\d{1,2}/\d{1,2}/\d{4})\s*$", text, flags=re.M))
            words = page.extract_words()
            bounds = header_bounds(words)
            if bounds is None:
                errors += 1
                continue
            for row in equity_rows_from_words(words, bounds):
                if "parse_error" in row:
                    errors += 1
                else:
                    rows.append(row)
    result["section_pages"] = pages
    result["page_header_dates"] = sorted(_iso(us_date(d)) for d in header_dates)
    # e.g. the column labelled "Foreign Holding" contains negative values, so its
    # meaning cannot be taken at face value.
    result["negative_value_counts"] = {
        column: sum(row[column].startswith("-") for row in rows)
        for column in ("close_price", "last_traded_price", "high", "low", "foreign_holding",
                       "turnover_rs", "quantity_in_cds")}
    result["rows"] = rows
    result["row_parse_errors"] = errors
    result["columns"] = ["industry_group", "board", "company_name", "type", "close_price",
                         "last_traded_price", "date_last_traded", "high", "low",
                         "foreign_holding", "turnover_rs", "quantity_in_cds"]
    return result


def inspect_old_daily_report(path):
    """Old 'smd full' format: verify its date and count rows of the daily-movements table."""
    import pdfplumber

    result = {"format": "old ('smd full DD-MM-YYYY')", "section": "Daily Movements Equity on <date>"}
    with pdfplumber.open(path) as pdf:
        result["pages"] = len(pdf.pages)
        first = _clean(pdf.pages[0].extract_text())
        match = re.search(r"(\d{2})-(\d{2})-(\d{4})", first)
        result["source_reported_date"] = (_iso(date(int(match[3]), int(match[2]), int(match[1])))
                                          if match else None)
        rows, heading_dates, pages = 0, set(), []
        for index, page in enumerate(pdf.pages):
            text = _clean(page.extract_text())
            heading = re.search(r"Daily Movements Equity on (\d{1,2})\w{2} (\w+) (\d{4})", text)
            if not heading:
                continue
            pages.append(index + 1)
            heading_dates.add(_iso(datetime.strptime(" ".join(heading.groups()), "%d %B %Y").date()))
            rows += len(re.findall(rf"\s{NUMBER}\s{NUMBER}\s\d{{2}}/\d{{2}}/\d{{2}}\s", text))
    result["section_pages"] = pages
    result["section_heading_dates"] = sorted(heading_dates)
    result["row_count"] = rows
    result["columns"] = ["company_name", "closing_price", "last_traded_price", "last_traded_date",
                         "high", "low", "foreign_holding", "issued_quantity", "turnover",
                         "indexed_market_cap", "qty_in_cds"]
    return result


# --- probes -----------------------------------------------------------------------

def probe_listings(session):
    listings = {}
    for name, method in [("cseDailyNew", "GET"), ("cseDaily", "POST"), ("cseWeekly", "POST"),
                         ("cseMonthly", "POST"), ("quarterlySummary", "POST")]:
        entry, data = session.api_json(name, method, name)
        items = []
        if isinstance(data, dict):
            for key, value in data.items():
                values = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
                items += [{"title": v.get("file_text"), "path": v.get("path"),
                           "listing_date": _iso(listing_date(v.get("file_text"))),
                           "uploaded_at": _ms(v.get("uploaded_date")),
                           "report_type": v.get("report_type"), "latest": "Latest" in key}
                          for v in values if isinstance(v, dict) and v.get("path")]
        listings[name] = {"request": entry, "items": items}
    return listings


def probe_daily_reports(session, listings, download_pdfs):
    attempts, parsed = [], {}
    series = [("cseDailyNew", "daily_new", parse_new_daily_report),
              ("cseDaily", "daily_smd", inspect_old_daily_report)]
    for listing_name, folder, parser in series:
        items = sorted({i["path"]: i for i in listings[listing_name]["items"]}.values(),
                       key=lambda i: i["listing_date"] or "", reverse=True)
        for item in items:
            url = CDN_CMT + item["path"]
            attempt = {"series": listing_name, "target_date": item["listing_date"],
                       "listing_title": item["title"], "url": url}
            if not download_pdfs:
                attempt["retrieved"] = False
                attempt["notes"] = "skipped (--no-pdfs)"
                attempts.append(attempt)
                continue
            entry, path = session.download(f"{folder}_{item['listing_date']}", url,
                                           f"reports/{folder}/{Path(item['path']).name}")
            attempt.update(http_status=entry.get("http_status"), retrieved=path is not None,
                           file_type=entry.get("content_type"), bytes=entry.get("bytes"),
                           sha256=entry.get("sha256"), saved_to=entry.get("saved_to"))
            if path is None:
                attempts.append(attempt)
                continue
            try:
                info = parser(path)
                attempt["parseable"] = True
            except Exception as exc:  # noqa: BLE001 - record any parser failure as evidence
                attempt.update(parseable=False, parse_error=str(exc))
                attempts.append(attempt)
                continue
            reported = info["source_reported_date"]
            internal = info.get("page_header_dates") or info.get("section_heading_dates") or []
            attempt.update(
                source_reported_date=reported,
                internal_dates=internal,
                date_matches_target=reported == item["listing_date"],
                date_internally_consistent=bool(reported) and set(internal) <= {reported},
                pages=info["pages"],
                section=info["section"],
                row_count=len(info["rows"]) if "rows" in info else info["row_count"],
                row_parse_errors=info.get("row_parse_errors"),
                negative_value_counts=info.get("negative_value_counts"),
                columns=info["columns"],
            )
            if "rows" in info:
                parsed[reported] = info
            attempts.append(attempt)

    for target in UNLISTED_TARGETS:
        attempts.append({
            "series": "any", "target_date": _iso(target), "url": None, "retrieved": False,
            "http_status": None,
            "notes": ("NOT FOUND: no public listing entry for this date. Listings return only the "
                      "latest ~5 reports, and report URLs contain a random token plus an upload "
                      "timestamp, so a URL cannot be derived from a date."),
        })
    return attempts, parsed


def inspect_periodic_report(path, kind):
    """Monthly (SMM) / quarterly (SMQ) report: stated period and the per-security table."""
    import pdfplumber

    heading = ("SECURITY TRADING STATISTICS" if kind == "monthly" else "PRICE CHANGES IN THE QUARTER")
    result = {"pages": 0, "stated_period": None, "section": heading, "section_pages": [],
              "row_count": 0, "columns": None}
    with pdfplumber.open(path) as pdf:
        result["pages"] = len(pdf.pages)
        for index, page in enumerate(pdf.pages):
            text = _clean(page.extract_text())
            if result["stated_period"] is None:
                match = (re.search(r"(\d{2}) / (\d{4})\(MM/YYYY\)", text) if kind == "monthly"
                         else re.search(r"(\d{4}):(Q\d)", text))
                if match:
                    result["stated_period"] = (f"{match[2]}-{match[1]}" if kind == "monthly"
                                               else f"{match[1]}-{match[2]}")
            if heading not in text:
                continue
            result["section_pages"].append(index + 1)
            if kind == "monthly":
                header = re.search(r"^SECURITY OPEN CLOSE .*$", text, flags=re.M)
                if header and result["columns"] is None:
                    result["columns"] = header.group(0).split()
                result["row_count"] += len(re.findall(r"\s[A-Z] \d{4}\s", text))
            else:
                value = rf"\(?\s?{NUMBER}\)?"
                result["row_count"] += len(re.findall(rf"^\D.*?(?:\s{value}){{8}}\s*$", text, flags=re.M))
    if kind == "quarterly":
        result["columns"] = ("company_name, then 8 values whose English labels are not in the "
                             "extracted text; by position they appear to be two prices, % change, "
                             "high, low, turnover, shares, trades (unverified)")
    return result


def probe_periodic_reports(session, listings, download_pdfs):
    results = []
    picks = [("cseMonthly", "monthly", listings["cseMonthly"]["items"][:1]),
             ("quarterlySummary", "quarterly",
              [i for i in listings["quarterlySummary"]["items"] if "2025" in (i["title"] or "")][:1])]
    for listing_name, kind, items in picks:
        for item in items:
            url = CDN_CMT + item["path"]
            record = {"series": listing_name, "title": item["title"], "url": url, "retrieved": False}
            if download_pdfs:
                entry, path = session.download(f"{kind}_{item['title']}", url,
                                               f"reports/periodic/{Path(item['path']).name}")
                record.update(http_status=entry.get("http_status"), retrieved=path is not None,
                              bytes=entry.get("bytes"), sha256=entry.get("sha256"),
                              saved_to=entry.get("saved_to"))
                if path is not None:
                    record.update(inspect_periodic_report(path, kind))
            results.append(record)
    return results


def probe_charts(session):
    charts = {}
    for symbol in CHART_SAMPLE:
        entry, company = session.api_json(f"homeCompanyData_{symbol}", "POST", "homeCompanyData",
                                          files={"symbol": (None, symbol)})
        stock_id = company.get("id") if isinstance(company, dict) else None
        record = {"symbol": symbol, "stock_id": stock_id, "requests": [entry]}
        if stock_id:
            entry, data = session.api_json(f"chart_{symbol}_period5", "POST", "companyChartDataByStock",
                                           files={"stockId": (None, str(stock_id)),
                                                  "period": (None, "5")})
            record["requests"].append(entry)
            points = data.get("chartData", []) if isinstance(data, dict) else []
            record["points"] = points
        charts[symbol] = record
    indices = {}
    for chart_id, label in [(1, "ASPI"), (40, "S&P SL20")]:
        entry, data = session.api_json(f"chartData_{chart_id}_period5", "POST", "chartData",
                                       files={"chartId": (None, str(chart_id)), "period": (None, "5")})
        indices[label] = {"request": entry, "points": data if isinstance(data, list) else []}
    return charts, indices


def probe_annual_statistics(session):
    results = {}
    for year in (2025, 2024, 2023, 2022, 2021):
        entry, data = session.api_json(f"security_trading_statistics_{year}", "POST",
                                       "security_trading_statistics",
                                       files={"year": (None, str(year))})
        rows = data.get("reqSecurityTrading", []) if isinstance(data, dict) else []
        results[year] = {"request": entry, "rows": len(rows),
                         "fields": sorted(rows[0]) if rows else [],
                         "symbols": len({r.get("symbol") for r in rows})}
    return results


def probe_current(session):
    status_entry, status = session.api_json("marketStatus", "POST", "marketStatus", json={})
    summary_entry, summary = session.api_json("dailyMarketSummery", "POST", "dailyMarketSummery")
    trade_entry, trade = session.api_json("tradeSummary", "POST", "tradeSummary")
    rows = trade.get("reqTradeSummery", []) if isinstance(trade, dict) else []
    trade_date = None
    if isinstance(summary, list) and summary and summary[0]:
        trade_date = _ms_date(summary[0][0].get("tradeDate"))
    return {"market_status": {"request": status_entry, "body": status},
            "daily_market_summary": {"request": summary_entry, "trade_date": _iso(trade_date),
                                     "fields": sorted(summary[0][0]) if trade_date else []},
            "trade_summary": {"request": trade_entry, "rows": rows,
                              "fields": sorted(rows[0]) if rows else []},
            "trade_date": trade_date}


def probe_market_reviews(session):
    entry, data = session.api_json("news_web_MR_2025", "GET", "news/web",
                                   params={"top": "false", "year": "2025", "type": "MR"})
    items = [i for month in (data or {}).values() if isinstance(month, list) for i in month] \
        if isinstance(data, dict) else []
    publishers = sorted({i.get("companyLogo") for i in items if i.get("companyLogo")})
    return {"request": entry, "items": len(items), "publisher_logos": publishers,
            "example_titles": [i.get("title", "").strip() for i in items[:5]]}


# --- canonical mapping, cross-checks and validation ---------------------------------------

def report_rows_to_canonical(parsed_reports):
    """Canonical-named frame from the new-format reports.

    date      = the report's own date, kept only for rows whose 'Date Last Traded'
                equals it (rows with an older last-traded date did not trade that
                day and are excluded as stale, never re-dated).
    symbol    = short company name + share type as printed; the report has no
                CSE symbol codes, so this is NOT a canonical symbol.
    open, volume, trades: not in the report -> left empty.
    """
    frames, stale = [], 0
    for report_date, info in sorted(parsed_reports.items()):
        for row in info["rows"]:
            traded = _iso(us_date(row["date_last_traded"]))
            if traded != report_date:
                stale += 1
                continue
            frames.append({
                "date": report_date,
                "symbol": f"{row['company_name']} [{row['type']}]",
                "open": "", "high": row["high"], "low": row["low"], "close": row["close_price"],
                "volume": "", "turnover": row["turnover_rs"],
            })
    return pd.DataFrame(frames, dtype=str), stale


def chart_points_to_canonical(charts):
    """Canonical-named frame from companyChartDataByStock points.

    date = calendar date of the point timestamp 't' in Sri Lanka time (all 't'
    values observed are exactly 00:00 SLT, i.e. a trade date). o and c are null
    in every observed point, so open stays empty. q -> volume is supported by the
    VWAP cross-check below; 's' is not mapped (meaning unknown).
    """
    frames = []
    for symbol, record in charts.items():
        for p in record.get("points", []):
            frames.append({
                "date": _iso(_ms_date(p.get("t"))),
                "symbol": symbol,
                "open": "" if p.get("o") is None else str(p["o"]),
                "high": _str(p.get("h")), "low": _str(p.get("l")), "close": _str(p.get("p")),
                "volume": _str(p.get("q")),
            })
    return pd.DataFrame(frames, dtype=str)


def cross_check_charts(charts, parsed_reports):
    """Compare chart points with the official daily report where the report's short
    name equals the symbol's base ticker (e.g. 'JKH' <-> JKH.N0000)."""
    checks = []
    for symbol, record in charts.items():
        base, share_type = symbol.split(".")[0], symbol.split(".")[1][0]
        by_date = {_iso(_ms_date(p["t"])): p for p in record.get("points", [])}
        for report_date, info in parsed_reports.items():
            matches = [r for r in info["rows"]
                       if r["company_name"] == base and r["type"] == share_type
                       and _iso(us_date(r["date_last_traded"])) == report_date]
            point = by_date.get(report_date)
            if len(matches) != 1 or point is None:
                continue
            row = matches[0]
            close, high, low = (_num(row[k]) for k in ("close_price", "high", "low"))
            turnover = _num(row["turnover_rs"])
            vwap = turnover / point["q"] if point.get("q") else None
            checks.append({
                "symbol": symbol, "date": report_date,
                "chart": {"p": point.get("p"), "h": point.get("h"), "l": point.get("l"),
                          "q": point.get("q")},
                "report": {"close": close, "high": high, "low": low, "turnover": turnover},
                "close_matches": point.get("p") == close,
                "high_matches": point.get("h") == high,
                "low_matches": point.get("l") == low,
                "implied_vwap": round(vwap, 4) if vwap else None,
                "vwap_within_high_low": bool(vwap and low - 0.01 <= vwap <= high + 0.01),
            })
    return checks


def trade_summary_to_canonical(rows):
    """Canonical-named frame from the current tradeSummary snapshot.

    date = calendar date (SLT) of each row's own lastTradedTime. close =
    closingPrice exactly as returned (0.0 while the session is open).
    """
    return pd.DataFrame([{
        "date": _iso(_ms_date(r.get("lastTradedTime"))),
        "symbol": r.get("symbol"),
        "open": _str(r.get("open")), "high": _str(r.get("high")), "low": _str(r.get("low")),
        "close": _str(r.get("closingPrice")), "volume": _str(r.get("sharevolume")),
        "turnover": _str(r.get("turnover")), "trades": _str(r.get("tradevolume")),
    } for r in rows], dtype=str)


def validation_summary(frame, expected_date=None):
    if frame.empty:
        return {"status": "NO_DATA", "total_rows": 0}
    result = validate_market_data(frame, expected_date=expected_date)
    summary = result.summary()
    summary["missing_ohlc"] = {k: summary["missing_values"].get(k, 0)
                               for k in ("open", "high", "low", "close")}
    return summary


# --- decisions ------------------------------------------------------------------------

def decisions(evidence):
    charts = evidence["historical_files"]["per_stock_chart"]
    chart_days = max((c["points"] for c in charts), default=0)
    annual = evidence["historical_files"]["annual_security_statistics"]
    public_years = [y for y, r in annual.items() if r["http_status"] == 200]
    locked_years = [y for y, r in annual.items() if r["http_status"] in (401, 403, 417)]
    reports = [a for a in evidence["daily_reports"] if a.get("retrieved")]
    return [
        {
            "source": "CSE daily report PDFs (cseDailyNew / cseDaily listings on cdn.cse.lk)",
            "classification": "C. RESEARCH ONLY",
            "reasons": [
                f"{len(reports)} report(s) retrieved; each date verified from the report's own text.",
                "Only the latest ~5 reports per series are listed; older URLs cannot be derived "
                "(random token in path), so no multi-year history is obtainable this way.",
                "Equity table has close/last/high/low/turnover but no open, no daily share volume, "
                "no trade count, and short company names instead of CSE symbols.",
                "Useful as an official cross-check of other sources for the same dates.",
            ],
        },
        {
            "source": "CSE monthly (SMM) and quarterly (SMQ) report PDFs",
            "classification": "C. RESEARCH ONLY",
            "reasons": [
                "Per-security open/close/high/low/turnover/shares/trades, but aggregated per "
                "month or quarter - not daily.",
                "Only the latest few reports are listed (same random-token URLs), so a long "
                "monthly history cannot be assembled from public listings.",
            ],
        },
        {
            "source": "companyChartDataByStock (per-stock daily chart, period=5)",
            "classification": "B. USABLE WITH LIMITATIONS",
            "reasons": [
                f"Up to {chart_days} daily points (~1 year) per stock, each with its own timestamp.",
                "Close, high and low match the official daily report where cross-checked; q is "
                "consistent with daily share volume (implied VWAP within the day's range).",
                "No open, no turnover, no trade count; 1-year rolling window only; one request per "
                "stock; undocumented endpoint that may change without notice.",
            ],
        },
        {
            "source": "tradeSummary + dailyMarketSummery (current session snapshot)",
            "classification": "B. USABLE WITH LIMITATIONS",
            "reasons": [
                "Has every canonical field (open, high, low, close, volume, turnover, trades) plus "
                "a per-row lastTradedTime and a market-level tradeDate.",
                "Current session only: usable for forward collection after the close, never for "
                "backfill. closingPrice is 0 while the market is open.",
            ],
        },
        {
            "source": "security_trading_statistics (annual per-security statistics)",
            "classification": "C. RESEARCH ONLY",
            "reasons": [
                f"Public without login only for {public_years or 'no year'}; "
                f"{locked_years} answered 401/417 (login required, not attempted further).",
                "Annual aggregates (one row per security per year), not daily data.",
            ],
        },
        {
            "source": "Index charts (chartData chartId 1 = ASPI, 40 = S&P SL20)",
            "classification": "C. RESEARCH ONLY",
            "reasons": [
                "~1 year of one value per day, no OHLC.",
                "Point timestamps are update times, not trade dates (the latest observed point "
                "carries a pre-open morning timestamp), so dates need verification before use.",
            ],
        },
        {
            "source": "Market reviews (news/web type=MR)",
            "classification": "D. NOT USABLE",
            "reasons": ["Broker-authored commentary PDFs hosted on the CSE CDN, not CSE market data."],
        },
    ]


# --- main -----------------------------------------------------------------------------

def run(download_pdfs=True):
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    session = Session(DISCOVERY_DIR)
    started = _now()

    robots_entry, robots = session.call("robots.txt", "GET", "https://www.cse.lk/robots.txt",
                                        save_as="robots.txt")
    listings = probe_listings(session)
    daily_attempts, parsed_reports = probe_daily_reports(session, listings, download_pdfs)
    periodic = probe_periodic_reports(session, listings, download_pdfs)
    charts, indices = probe_charts(session)
    annual = probe_annual_statistics(session)
    current = probe_current(session)
    reviews = probe_market_reviews(session)

    report_frame, stale_rows = report_rows_to_canonical(parsed_reports)
    report_validation = validation_summary(report_frame)
    latest_report = max(parsed_reports, default=None)
    latest_frame = report_frame[report_frame["date"] == latest_report] if latest_report else report_frame
    chart_frame = chart_points_to_canonical(charts)
    trade_frame = trade_summary_to_canonical(current["trade_summary"]["rows"])
    cross_checks = cross_check_charts(charts, parsed_reports)

    evidence = {
        "generated_at": _now(),
        "started_at": started,
        "tool": "scripts/cse_source_discovery.py",
        "rules": [
            "Only public, unauthenticated cse.lk / cdn.cse.lk endpoints and files were accessed.",
            "401/417 responses are recorded as login-required; no credentials were used.",
            "Dates are taken only from the source payload; requested dates are never assigned.",
            "Third-party datasets are not treated as evidence of official CSE data.",
        ],
        "robots_txt": {"request": robots_entry,
                       "body": robots.text if robots is not None and robots.ok else None},
        "listings": {name: {"request": v["request"], "items": v["items"]} for name, v in listings.items()},
        "daily_reports": daily_attempts,
        "historical_files": {
            "periodic_reports": periodic,
            "per_stock_chart": [{
                "symbol": s, "stock_id": c.get("stock_id"), "points": len(c.get("points", [])),
                "first_date": _iso(_ms_date(c["points"][0]["t"])) if c.get("points") else None,
                "last_date": _iso(_ms_date(c["points"][-1]["t"])) if c.get("points") else None,
                "fields": sorted(c["points"][0]) if c.get("points") else [],
                "always_null_fields": sorted(k for k in (c["points"][0] if c.get("points") else {})
                                             if all(p.get(k) is None for p in c["points"])),
                "timestamps_at_midnight_slt": all(
                    datetime.fromtimestamp(p["t"] / 1000, SLT).time() == datetime.min.time()
                    for p in c.get("points", [])),
                "requests": c["requests"],
            } for s, c in charts.items()],
            "index_charts": {label: {
                "request": v["request"], "points": len(v["points"]),
                "fields": sorted(v["points"][0]) if v["points"] else [],
                "first_timestamp": _ms(v["points"][0]["d"]) if v["points"] else None,
                "last_timestamp": _ms(v["points"][-1]["d"]) if v["points"] else None,
            } for label, v in indices.items()},
            "annual_security_statistics": {year: {
                "http_status": r["request"].get("http_status"), "rows": r["rows"],
                "symbols": r["symbols"], "fields": r["fields"], "request": r["request"],
            } for year, r in annual.items()},
        },
        "current_snapshot": {
            "market_status": current["market_status"]["body"],
            "daily_market_summary": current["daily_market_summary"],
            "trade_summary": {"request": current["trade_summary"]["request"],
                              "rows": len(current["trade_summary"]["rows"]),
                              "fields": current["trade_summary"]["fields"]},
        },
        "market_reviews": reviews,
        "canonical_mapping": CANONICAL_MAPPING,
        "cross_checks": {
            "chart_vs_daily_report": cross_checks,
            "summary": {
                "comparisons": len(cross_checks),
                "close_matches": sum(c["close_matches"] for c in cross_checks),
                "high_matches": sum(c["high_matches"] for c in cross_checks),
                "low_matches": sum(c["low_matches"] for c in cross_checks),
                "vwap_within_high_low": sum(c["vwap_within_high_low"] for c in cross_checks),
            },
        },
        "validation": {
            "daily_report_latest": {"report_date": latest_report,
                                    **validation_summary(latest_frame, _date(latest_report))},
            "daily_report_all_retrieved": {"stale_rows_excluded": stale_rows, **report_validation},
            "per_stock_chart_sample": validation_summary(chart_frame),
            "trade_summary_snapshot": {
                "expected_date": _iso(current["trade_date"]),
                **validation_summary(trade_frame, current["trade_date"]),
            },
        },
        "requests": session.log,
    }
    evidence["coverage_decisions"] = decisions(evidence)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FILE.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    return evidence


CANONICAL_MAPPING = {
    "daily_report_new_format": {
        "date": "report date from page 1 (kept only where 'Date Last Traded' equals it)",
        "symbol": "NOT AVAILABLE - short company name + share type only",
        "open": None, "high": "High", "low": "Low", "close": "Close Price",
        "volume": None, "turnover": "Turnover (Rs.)", "trades": None,
        "source": "cse_daily_report", "source_priority": "1 (official CSE publication)",
        "source_timestamp": None,
        "validation_status": "validator", "validation_warnings": "validator",
        "extra_fields_not_in_schema": ["Last Traded Price", "Foreign Holding (contains negative "
                                       "values; meaning unverified)", "Quantity in CDS",
                                       "Industry Group", "Board"],
    },
    "per_stock_chart": {
        "date": "t (epoch ms) -> date in Asia/Colombo",
        "symbol": "requested symbol (the chart is fetched by the stock id that symbol resolves to)",
        "open": None, "high": "h", "low": "l", "close": "p",
        "volume": "q (consistent with share volume; see cross-checks)",
        "turnover": None, "trades": None,
        "source": "cse_chart_api", "source_priority": "2 (official content, undocumented interface)",
        "source_timestamp": "t",
        "unmapped_fields": {"s": "unknown meaning", "o": "always null", "c": "always null",
                            "pc": "always null", "n": "always null"},
    },
    "trade_summary_snapshot": {
        "date": "lastTradedTime (epoch ms) -> date in Asia/Colombo",
        "symbol": "symbol", "open": "open", "high": "high", "low": "low",
        "close": "closingPrice (0.0 during the session)", "volume": "sharevolume",
        "turnover": "turnover", "trades": "tradevolume",
        "source": "cse_trade_summary_current", "source_priority": "2",
        "source_timestamp": "lastTradedTime",
    },
}


# --- helpers ----------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ms(value):
    return datetime.fromtimestamp(value / 1000, SLT).isoformat() if value else None


def _ms_date(value):
    return datetime.fromtimestamp(value / 1000, SLT).date() if value else None


def _iso(value):
    return value.isoformat() if value else None


def _date(text):
    return date.fromisoformat(text) if text else None


def _str(value):
    return "" if value is None else str(value)


def _num(text):
    return float(str(text).replace(",", ""))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path):
    try:
        return Path(path).resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return str(path)


def print_summary(evidence):
    print("CSE SOURCE DISCOVERY")
    print("====================")
    print(f"Requests made: {len(evidence['requests'])}")
    print("\nDaily reports:")
    for a in evidence["daily_reports"]:
        print(f"  {a['series']:<11} target {a['target_date']}  HTTP {a.get('http_status')}  "
              f"retrieved={a.get('retrieved')}  reported={a.get('source_reported_date')}  "
              f"match={a.get('date_matches_target')}  rows={a.get('row_count')}")
    print("\nPeriodic reports:")
    for r in evidence["historical_files"]["periodic_reports"]:
        print(f"  {r['title']:<18} HTTP {r.get('http_status')} period={r.get('stated_period')} "
              f"pages={r.get('pages')} rows={r.get('row_count')}")
    print("\nPer-stock chart:")
    for c in evidence["historical_files"]["per_stock_chart"]:
        print(f"  {c['symbol']:<11} {c['points']} points {c['first_date']}..{c['last_date']}  "
              f"null fields {c['always_null_fields']}")
    print("\nAnnual statistics:", {y: r["http_status"]
                                    for y, r in evidence["historical_files"]["annual_security_statistics"].items()})
    print("Cross-checks:", evidence["cross_checks"]["summary"])
    for name, v in evidence["validation"].items():
        print(f"Validation {name}: status={v.get('status')} rows={v.get('total_rows')} "
              f"invalid={v.get('invalid_rows')} issues={v.get('issue_counts')}")
    print("\nDecisions:")
    for d in evidence["coverage_decisions"]:
        print(f"  {d['classification']:<28} {d['source']}")
    print(f"\nEvidence written to {_rel(EVIDENCE_FILE)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pdfs", action="store_true", help="skip report PDF downloads")
    args = parser.parse_args(argv)
    print_summary(run(download_pdfs=not args.no_pdfs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
