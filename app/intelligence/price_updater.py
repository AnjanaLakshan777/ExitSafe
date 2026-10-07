"""Append each new trading day's prices to the user's tracked market-data CSV.

Rows are written in the file's own layout (columns, delimiter, date format) so
the dashboard importer reads them like the original rows. Each new row names its
source in a Source column (added, empty for older rows, if the file has none),
so CSE prices and secondary Gemini prices are never mixed up. Existing values are
never changed: a backup is taken before every write, and a (symbol, date) that is
already in the file is skipped. Every written row is also logged to
``price_updates.jsonl``.

The quote collection and the file-layout writing are shared with the database
tracking in ``app/intelligence/market_tracking.py``.
"""

import csv
import io
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from app.data.loaders.csv_market_loader import detect_columns
from app.data.source_catalog import get_source
from app.intelligence.collectors.http import CollectionError
from app.intelligence.collectors.price_collector import (
    COLOMBO,
    collect_cse_quotes,
    collect_gemini_quotes,
    cse_market_closed,
)
from app.intelligence.settings import PRICE_LOG_FILE

# Tried in order; a format is used only if every existing date parses with it.
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
                "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y")
DELIMITERS = ",;\t|"
SOURCE_COLUMN = "Source"
NO_TRADES = "no trades on CSE in the latest session"
NO_PRICE = "no price found"
MARKET_OPEN = "CSE session is still open; today's prices are not final yet"


class TrackedCsvError(ValueError):
    """The tracked CSV can't be updated safely; nothing was written."""


@dataclass
class PriceUpdateResult:
    csv_path: str
    added: list = field(default_factory=list)           # PriceQuote
    skipped: dict = field(default_factory=dict)         # symbol -> reason
    errors: list = field(default_factory=list)


@dataclass
class QuoteCollection:
    """Latest-session quotes for a set of symbols, and why the others have none."""
    quotes: dict = field(default_factory=dict)          # symbol -> PriceQuote
    skipped: dict = field(default_factory=dict)         # symbol -> reason
    errors: list = field(default_factory=list)
    market_open: bool = False
    session: object = None                              # latest CSE session date, if fetched
    cse_failed: bool = False
    gemini_symbols: tuple = ()                          # symbols looked up with Gemini
    gemini_failed: bool = False


def collect_latest_quotes(symbols, settings, *, cse_snapshot=collect_cse_quotes,
                          market_closed=cse_market_closed, gemini_quotes=collect_gemini_quotes,
                          now=None, latest_dates=None):
    """Official CSE quotes first; Gemini only for symbols CSE doesn't list.

    Nothing is collected while the CSE session is open. ``latest_dates`` (symbol ->
    newest date already stored) lets symbols that are already current skip Gemini.
    """
    now = now or datetime.now(timezone.utc)
    result = QuoteCollection()
    remaining = set(symbols)
    if "cse" in settings.price_sources:
        try:
            if not market_closed():
                result.errors.append(MARKET_OPEN)
                result.market_open = True
                remaining.clear()        # try again after the close rather than use Gemini now
            else:
                snapshot = cse_snapshot()
                result.session = snapshot.session
                for symbol in symbols:
                    if symbol in snapshot.quotes:
                        result.quotes[symbol] = snapshot.quotes[symbol]
                    elif symbol in snapshot.listed:
                        result.skipped[symbol] = NO_TRADES
                remaining -= set(result.quotes) | snapshot.listed
        except CollectionError as exc:
            result.errors.append(f"CSE: {exc}")
            result.cse_failed = True

    if remaining and latest_dates:
        newest_possible = result.session or now.astimezone(COLOMBO).date()
        for symbol in sorted(remaining):
            last = latest_dates.get(symbol)
            if last is not None and last >= newest_possible:
                result.skipped[symbol] = f"already has {last}"
                remaining.discard(symbol)

    if remaining and "gemini" in settings.price_sources:
        if not settings.gemini_configured:
            result.errors.append("GEMINI_API_KEY is not set; can't look up "
                                 + ", ".join(sorted(remaining)))
        else:
            result.gemini_symbols = tuple(sorted(remaining))
            try:
                result.quotes.update(gemini_quotes(sorted(remaining), settings,
                                                   today=now.astimezone(COLOMBO).date()))
            except CollectionError as exc:
                result.errors.append(f"Gemini: {exc}")
                result.gemini_failed = True
    for symbol in sorted(remaining - set(result.quotes)):
        result.skipped.setdefault(symbol, NO_PRICE)
    return result


def _read_text(text):
    try:
        delimiter = csv.Sniffer().sniff(text.splitlines()[0], DELIMITERS).delimiter
    except (csv.Error, IndexError):
        delimiter = ","
    frame = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    return frame, delimiter


def read_tracked_csv(path):
    """``(frame, delimiter)``; all cells kept as text, exactly as written."""
    return _read_text(path.read_text(encoding="utf-8-sig"))


def detect_date_format(values):
    """The one format all dates parse with. Refuses when day/month order is ambiguous."""
    values = [v.strip() for v in values if v and v.strip()]
    if not values:
        return "%Y-%m-%d"
    readings = {}
    for fmt in DATE_FORMATS:
        try:
            readings[fmt] = tuple(datetime.strptime(v, fmt).date() for v in values)
        except ValueError:
            continue
    if not readings:
        raise TrackedCsvError(f"Can't recognise the date format (e.g. {values[-1]!r})")
    if len(set(readings.values())) > 1:
        raise TrackedCsvError(
            "Dates could be read as day/month or month/day; new rows can't be written "
            f"without guessing. Formats that fit: {', '.join(readings)}")
    return next(iter(readings))


@dataclass(frozen=True)
class FileLayout:
    """How a market-data file is laid out, so new rows can be written the same way."""
    frame: pd.DataFrame
    delimiter: str
    column_for: dict           # canonical name -> the file's column
    date_format: str
    symbols: list              # upper-case
    spelling: dict             # upper-case symbol -> as written in the file
    existing: frozenset        # (upper-case symbol, date text) already in the file


def file_layout(content, tracked_symbol=None):
    """Read the layout of CSV bytes. Raises TrackedCsvError if rows can't be added safely."""
    frame, delimiter = _read_text(content.decode("utf-8-sig"))
    assignments, _ = detect_columns(frame.columns)
    column_for = {canonical: columns[0] for canonical, columns in assignments.items()}
    for required in ("date", "close"):
        if required not in column_for:
            raise TrackedCsvError(f"The tracked CSV has no {required} column")
    date_format = detect_date_format(frame[column_for["date"]].tolist())

    if "symbol" in column_for:
        written = frame[column_for["symbol"]].str.strip()
        spelling = {s.upper(): s for s in written if s}
        symbols = sorted(spelling)
        row_symbols = written.str.upper()
    elif tracked_symbol:
        symbols = [tracked_symbol.strip().upper()]
        spelling = {symbols[0]: symbols[0]}
        row_symbols = pd.Series(symbols * len(frame), index=frame.index, dtype=str)
    else:
        raise TrackedCsvError("The tracked CSV has no Symbol column; set TRACKED_SYMBOL in .env")
    existing = frozenset(zip(row_symbols, frame[column_for["date"]].str.strip()))
    return FileLayout(frame, delimiter, column_for, date_format, symbols, spelling, existing)


def _number(value):
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _row(columns, column_for, quote, date_format, symbol):
    values = {"date": quote.day.strftime(date_format), "symbol": symbol,
              "open": _number(quote.open), "high": _number(quote.high), "low": _number(quote.low),
              "close": _number(quote.close), "volume": _number(quote.volume),
              "turnover": _number(quote.turnover), "source": quote.source}
    row = dict.fromkeys(columns, "")
    for canonical, value in values.items():
        if canonical in column_for:
            row[column_for[canonical]] = value
    return [row[c] for c in columns]


def _with_source_column(raw, delimiter):
    """Add an empty Source column to older file bytes; existing values stay as they are."""
    bom = raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    out = []
    for number, line in enumerate(lines):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if body.strip():
            body += delimiter + (SOURCE_COLUMN if number == 0 else "")
        out.append(body + ending)
    return (b"\xef\xbb\xbf" if bom else b"") + "".join(out).encode("utf-8")


def append_quotes(content, layout, quotes):
    """``(new bytes, added quotes, skipped)``: quotes for a (symbol, date) already in the
    file are skipped, so the file's own rows always win. ``content`` is not changed."""
    added, skipped = [], {}
    for quote in quotes:
        if (quote.symbol, quote.day.strftime(layout.date_format)) in layout.existing:
            skipped[quote.symbol] = f"{quote.day} already in file"
        else:
            added.append(quote)
    if not added:
        return content, added, skipped

    columns = list(layout.frame.columns)
    column_for = dict(layout.column_for)
    raw = content
    if "source" not in column_for:
        raw = _with_source_column(raw, layout.delimiter)
        columns.append(SOURCE_COLUMN)
        column_for["source"] = SOURCE_COLUMN
    rows = io.StringIO()
    writer = csv.writer(rows, delimiter=layout.delimiter, lineterminator="\n")
    for quote in added:
        writer.writerow(_row(columns, column_for, quote, layout.date_format,
                             layout.spelling.get(quote.symbol, quote.symbol)))
    if raw and not raw.endswith((b"\n", b"\r")):
        raw += b"\n"
    return raw + rows.getvalue().encode("utf-8"), added, skipped


def save_tracked_csv(content, path):
    """Store uploaded CSV bytes as the tracked file (the previous one is kept as ``.bak``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_bytes(content)


def analysis_use(source):
    if source.is_ai_generated:
        return "excluded by default: secondary AI-sourced data, not exchange data"
    return "included"


def _log(quotes, now):
    PRICE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PRICE_LOG_FILE.open("a", encoding="utf-8") as log:
        for q in quotes:
            source = get_source(q.source)
            log.write(json.dumps({"written_time": now.isoformat(), "symbol": q.symbol,
                                  "date": q.day.isoformat(), "close": q.close,
                                  "source": q.source, "source_priority": source.source_priority,
                                  "source_url": q.source_url,
                                  "quantitative_analysis": analysis_use(source)}) + "\n")


def update_tracked_csv(settings, *, cse_snapshot=collect_cse_quotes,
                       market_closed=cse_market_closed, gemini_quotes=collect_gemini_quotes,
                       now=None):
    """Fetch the latest session for every symbol in the tracked CSV and append new rows."""
    now = now or datetime.now(timezone.utc)
    path = settings.tracked_csv
    result = PriceUpdateResult(csv_path=str(path))
    if not path.exists():
        raise TrackedCsvError(f"No tracked CSV at {path}. Save one from the dashboard first, "
                              "or set TRACKED_CSV_PATH.")

    content = path.read_bytes()
    layout = file_layout(content, settings.tracked_symbol)
    collection = collect_latest_quotes(layout.symbols, settings, cse_snapshot=cse_snapshot,
                                       market_closed=market_closed, gemini_quotes=gemini_quotes,
                                       now=now)
    result.errors.extend(collection.errors)
    result.skipped.update(collection.skipped)

    updated, added, skipped = append_quotes(
        content, layout, [quote for _, quote in sorted(collection.quotes.items())])
    result.skipped.update(skipped)
    if not added:
        return result
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_bytes(updated)
    _log(added, now)
    result.added = added
    return result
