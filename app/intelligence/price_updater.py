"""Append each new trading day's prices to the user's tracked market-data CSV.

Rows are written in the file's own layout (columns, delimiter, date format) so
the dashboard importer reads them like the original rows. Existing rows are
never changed: a backup is taken before every write, and a (symbol, date) that
is already in the file is skipped. Where every row came from is logged to
``price_updates.jsonl``.
"""

import csv
import io
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from app.data.loaders.csv_market_loader import detect_columns
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


class TrackedCsvError(ValueError):
    """The tracked CSV can't be updated safely; nothing was written."""


@dataclass
class PriceUpdateResult:
    csv_path: str
    added: list = field(default_factory=list)           # PriceQuote
    skipped: dict = field(default_factory=dict)         # symbol -> reason
    errors: list = field(default_factory=list)


def read_tracked_csv(path):
    """``(frame, delimiter)``; all cells kept as text, exactly as written."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        delimiter = csv.Sniffer().sniff(text.splitlines()[0], DELIMITERS).delimiter
    except (csv.Error, IndexError):
        delimiter = ","
    frame = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    return frame, delimiter


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


def _number(value):
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _row(columns, column_for, quote, date_format):
    values = {"date": quote.day.strftime(date_format), "symbol": quote.symbol,
              "open": _number(quote.open), "high": _number(quote.high), "low": _number(quote.low),
              "close": _number(quote.close), "volume": _number(quote.volume),
              "turnover": _number(quote.turnover)}
    row = dict.fromkeys(columns, "")
    for canonical, value in values.items():
        if canonical in column_for:
            row[column_for[canonical]] = value
    return [row[c] for c in columns]


def save_tracked_csv(content, path):
    """Store uploaded CSV bytes as the tracked file (the previous one is kept as ``.bak``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_bytes(content)


def _log(quotes, now):
    PRICE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PRICE_LOG_FILE.open("a", encoding="utf-8") as log:
        for q in quotes:
            log.write(json.dumps({"written_time": now.isoformat(), "symbol": q.symbol,
                                  "date": q.day.isoformat(), "close": q.close,
                                  "source": q.source, "source_url": q.source_url}) + "\n")


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

    frame, delimiter = read_tracked_csv(path)
    assignments, _ = detect_columns(frame.columns)
    column_for = {canonical: columns[0] for canonical, columns in assignments.items()}
    for required in ("date", "close"):
        if required not in column_for:
            raise TrackedCsvError(f"The tracked CSV has no {required} column")
    date_format = detect_date_format(frame[column_for["date"]].tolist())

    if "symbol" in column_for:
        symbols = sorted({s.strip().upper() for s in frame[column_for["symbol"]] if s.strip()})
        row_symbols = frame[column_for["symbol"]].str.strip().str.upper()
    elif settings.tracked_symbol:
        symbols = [settings.tracked_symbol.strip().upper()]
        row_symbols = pd.Series(symbols * len(frame), index=frame.index, dtype=str)
    else:
        raise TrackedCsvError("The tracked CSV has no Symbol column; set TRACKED_SYMBOL in .env")
    existing = set(zip(row_symbols, frame[column_for["date"]].str.strip()))

    quotes, remaining = {}, set(symbols)
    if "cse" in settings.price_sources:
        try:
            if not market_closed():
                result.errors.append("CSE session is still open; today's prices are not final yet")
                remaining.clear()        # try again after the close rather than use Gemini now
            else:
                snapshot = cse_snapshot()
                for symbol in symbols:
                    if symbol in snapshot.quotes:
                        quotes[symbol] = snapshot.quotes[symbol]
                    elif symbol in snapshot.listed:
                        result.skipped[symbol] = "no trades on CSE in the latest session"
                remaining -= set(quotes) | snapshot.listed
        except CollectionError as exc:
            result.errors.append(f"CSE: {exc}")

    if remaining and "gemini" in settings.price_sources:
        if not settings.gemini_configured:
            result.errors.append("GEMINI_API_KEY is not set; can't look up "
                                 + ", ".join(sorted(remaining)))
        else:
            try:
                quotes.update(gemini_quotes(sorted(remaining), settings,
                                            today=now.astimezone(COLOMBO).date()))
            except CollectionError as exc:
                result.errors.append(f"Gemini: {exc}")
    for symbol in sorted(remaining - set(quotes)):
        result.skipped.setdefault(symbol, "no price found")

    new_quotes = []
    for symbol, quote in sorted(quotes.items()):
        if (symbol, quote.day.strftime(date_format)) in existing:
            result.skipped[symbol] = f"{quote.day} already in file"
        else:
            new_quotes.append(quote)
    if not new_quotes:
        return result

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    raw = path.read_bytes()
    with path.open("a", encoding="utf-8", newline="") as handle:
        if raw and not raw.endswith((b"\n", b"\r")):
            handle.write("\n")
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        for quote in new_quotes:
            writer.writerow(_row(list(frame.columns), column_for, quote, date_format))
    _log(new_quotes, now)
    result.added = new_quotes
    return result
