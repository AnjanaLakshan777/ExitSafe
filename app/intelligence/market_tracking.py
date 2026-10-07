"""Tracked market data: upload a CSV once, and ExitSafe keeps those companies up to date.

Start: the upload is checked with the normal importer and kept byte for byte in the
database. Every symbol in it becomes a tracked symbol.

Update (the scheduled job and "Update now" both call ``run_tracking_update``): after
the CSE close, the latest session's prices are collected with the same code as the
tracked-CSV updater (official CSE first, Gemini only for symbols CSE doesn't list),
validated, and stored once per symbol, date and source. The CSE source only gives the
latest session, so days that were missed stay missing; nothing is filled in.

Analysis: the original file plus the newer stored days, written in the file's own
layout, goes through the normal importer, so the provenance selection still decides
what reaches the analytics (Gemini rows are left out by default).
"""

import hashlib
import logging
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from app.data.loaders.csv_market_loader import (MarketDataImportError, SymbolRequiredError,
                                                load_csv_market_data)
from app.data.repositories.tracking_repository import NOT_YET_UPDATED, TrackingRepository
from app.data.schemas.market_schema import ValidationStatus
from app.data.source_catalog import DataSourceType, find_source, get_source
from app.data.validators.market_validator import DatasetStatus, validate_market_data
from app.intelligence.collectors.price_collector import (
    COLOMBO,
    PriceQuote,
    collect_cse_quotes,
    collect_gemini_quotes,
    cse_market_closed,
)
from app.intelligence.price_updater import (NO_TRADES, TrackedCsvError, analysis_use,
                                            append_quotes, collect_latest_quotes, file_layout)
from app.intelligence.settings import load_settings

log = logging.getLogger("exitsafe.tracking")

# CSE closes at 14:30 Colombo time; prices are collected from this hour on.
UPDATE_HOUR = 15

START_INVESTING = "START_INVESTING"
ALREADY_INVESTED = "ALREADY_INVESTED"
PURPOSES = (START_INVESTING, ALREADY_INVESTED)

SCHEDULED = "scheduled"
MANUAL = "manual"

# Dataset status after an update
UP_TO_DATE = "UP_TO_DATE"
UPDATED = "UPDATED"
PARTIALLY_UPDATED = "PARTIALLY_UPDATED"
WAITING_FOR_MARKET_CLOSE = "WAITING_FOR_MARKET_CLOSE"
UPDATE_FAILED = "UPDATE_FAILED"
NO_NEW_DATA = "NO_NEW_DATA"
STATUS_LABELS = {
    NOT_YET_UPDATED: "Not updated yet",
    UP_TO_DATE: "Up to date",
    UPDATED: "Updated",
    PARTIALLY_UPDATED: "Partially updated",
    WAITING_FOR_MARKET_CLOSE: "Waiting for market close",
    UPDATE_FAILED: "Update failed",
    NO_NEW_DATA: "No new data",
}
# A completed check, even when there was nothing new.
SUCCESSFUL = frozenset({UP_TO_DATE, UPDATED, PARTIALLY_UPDATED, NO_NEW_DATA})

# Whole-run status (besides WAITING_FOR_MARKET_CLOSE)
COMPLETED = "COMPLETED"
COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
FAILED = "FAILED"
ALREADY_RUNNING = "ALREADY_RUNNING"
NO_TRACKED_DATA = "NO_TRACKED_DATA"
DATABASE_NOT_CONFIGURED = "DATABASE_NOT_CONFIGURED"
DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"

# What happened to one tracked symbol
ADDED = "ADDED"
ADDED_SECONDARY = "ADDED_SECONDARY"
NO_TRADES_TODAY = "NO_TRADES"
NOT_FOUND = "NOT_FOUND"
REJECTED = "REJECTED"
WAITING = "WAITING"
SYMBOL_LABELS = {
    ADDED: "Updated",
    ADDED_SECONDARY: "Updated (secondary source)",
    UP_TO_DATE: "Up to date",
    NO_TRADES_TODAY: "No trades",
    NOT_FOUND: "Not updated",
    REJECTED: "Not updated",
    WAITING: "Waiting for market close",
}


class TrackingSetupError(ValueError):
    """Tracking can't start; nothing was stored. The message is meant for the user."""


def is_ai_source(name):
    source = find_source(name) if name else None
    return bool(source and source.is_ai_generated)


def source_description(name):
    """Plain-language source of a row: official CSE, secondary AI, or the user's file."""
    if not name:
        return "Your uploaded file"
    source = find_source(name)
    if source is None:
        return name
    if source.is_ai_generated:
        return "Secondary AI-sourced (Gemini)"
    if source.source_type is DataSourceType.OFFICIAL_CSE:
        return "Official CSE"
    return "Your uploaded file"


# --- starting to track an upload ---

def _import(file_name, content, symbol):
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / (Path(file_name).name or "market_data.csv")
            path.write_bytes(content)
            return load_csv_market_data(path, symbol=symbol)
    except SymbolRequiredError:
        raise TrackingSetupError("This data has no Symbol column. Enter the stock symbol so "
                                 "ExitSafe knows which company to track.") from None
    except MarketDataImportError as exc:
        raise TrackingSetupError(f"The file can't be read as market data: {exc}") from None


def start_tracking(repository, client_id, file_name, content, *, name=None, purpose=None,
                   symbol=None, settings=None, now=None):
    """Validate an upload, keep it, and start tracking its symbols. Returns the dataset."""
    settings = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    if purpose is not None and purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    symbol = (symbol or "").strip().upper() or None
    if not content:
        raise TrackingSetupError("The file is empty.")

    digest = hashlib.sha256(content).hexdigest()
    existing = repository.find_by_upload(client_id, digest)
    if existing is not None:
        raise TrackingSetupError(f"This file is already tracked as “{existing.name}”.")

    result = _import(file_name, content, symbol)
    if result.data is None:
        raise TrackingSetupError("Required market-data columns are missing: "
                                 + ", ".join(result.validation.missing_columns) + ".")
    if result.validation.status is DatasetStatus.FAIL:
        raise TrackingSetupError("The file didn't pass validation, so it can't be tracked: "
                                 + "; ".join(result.validation.failure_reasons) + ".")

    data = result.data[result.data["date"].notna() & result.data["symbol"].notna()]
    upper = data["symbol"].astype(str).str.strip().str.upper()
    last_dates = {s: d.date() for s, d in data.groupby(upper)["date"].max().items() if s}
    if not last_dates:
        raise TrackingSetupError("No company symbols were found in the file.")
    today = now.astimezone(COLOMBO).date()
    if max(last_dates.values()) > today:
        raise TrackingSetupError(f"The file has dates after today ({max(last_dates.values())}); "
                                 "ExitSafe won't track prices dated in the future.")
    try:
        file_layout(content, symbol)
    except TrackedCsvError as exc:
        raise TrackingSetupError(f"New trading days can't be added to this file safely: {exc}") \
            from None

    report = result.report
    upload = {
        "file_name": Path(file_name).name[:255], "content": content, "sha256": digest,
        "size_bytes": len(content), "uploaded_at": now, "rows_read": report.rows_read,
        "valid_rows": report.valid_rows,
        "date_min": date.fromisoformat(report.date_min) if report.date_min else None,
        "date_max": date.fromisoformat(report.date_max) if report.date_max else None,
        "validation_status": result.validation.status.value,
        "validation_summary": result.validation.summary(),
        "provenance": report.source_metadata,
    }
    name = (name or "").strip()[:120] or Path(file_name).stem[:120] or "My market data"
    dataset_id = repository.create_dataset(
        client_id, name=name, purpose=purpose, price_sources=",".join(settings.price_sources),
        symbol_override=symbol if report.symbol_source == "parameter" else None,
        upload=upload, symbols=last_dates)
    log.info("Client %s started tracking dataset %s (%d symbols)", client_id, dataset_id,
             len(last_dates))
    return repository.dataset_for(client_id, dataset_id)


# --- the data the analytics see ---

@dataclass(frozen=True)
class TrackedAnalysisInput:
    file_name: str
    content: bytes            # original upload + newer collected days, in the file's layout
    symbol: str | None        # for a file without a Symbol column
    original_content: bytes   # the upload alone (the backtest keeps using only this)
    collected: tuple          # MarketObservation rows added after the upload
    superseded: tuple         # less trusted rows for a day that has a better one


def _quote(row):
    return PriceQuote(symbol=row.symbol, day=row.date, open=row.open, high=row.high, low=row.low,
                      close=row.close, volume=int(row.volume), turnover=row.turnover,
                      source=row.source, source_url=row.source_url)


def analysis_input(repository, client_id, dataset_id):
    """The client's dataset as file bytes for the normal importer, or None if it isn't theirs.

    Only days after the upload's own last day are added, so uploaded values are never
    replaced, and only up to the last day this dataset's own updates accepted (stored
    prices are shared, so a paused dataset doesn't pick up days collected for others).
    When two sources have the same day, the more trusted one is used (official CSE
    before Gemini) and the other is listed in ``superseded``.
    """
    dataset = repository.dataset_for(client_id, dataset_id)
    if dataset is None:
        return None
    original = repository.upload_content(client_id, dataset_id)
    symbols = {s.symbol: s for s in dataset.symbols}
    chosen, superseded = {}, []
    for row in repository.observations_for(symbols):       # most trusted first for each day
        tracked = symbols[row.symbol]
        if tracked.upload_last_date is not None and row.date <= tracked.upload_last_date:
            continue
        if tracked.last_market_date is None or row.date > tracked.last_market_date:
            continue
        if (row.symbol, row.date) in chosen:
            superseded.append(row)
        else:
            chosen[(row.symbol, row.date)] = row
    layout = file_layout(original, dataset.symbol_override)
    content, _, _ = append_quotes(original, layout, [_quote(r) for r in chosen.values()])
    return TrackedAnalysisInput(dataset.upload.file_name, content, dataset.symbol_override,
                                original, tuple(chosen.values()), tuple(superseded))


# --- the update job ---

@dataclass
class SymbolOutcome:
    symbol: str
    outcome: str
    message: str
    day: date | None = None
    source: str | None = None


@dataclass
class DatasetOutcome:
    dataset_id: int
    client_id: int
    status: str
    message: str
    rows_added: int
    symbols: list


@dataclass
class TrackingUpdateResult:
    status: str
    run_id: int | None = None
    session: date | None = None
    rows_added: int = 0                     # new observations stored by this run
    datasets: dict = field(default_factory=dict)   # dataset_id -> DatasetOutcome
    messages: list = field(default_factory=list)
    cse_failed: bool = False

    @property
    def retry_later(self):
        """Today's update should run again (waiting for the close, or a source/database failed)."""
        return self.cse_failed or self.status in (WAITING_FOR_MARKET_CLOSE, ALREADY_RUNNING,
                                                  FAILED, DATABASE_UNAVAILABLE)


def _redact(message, settings):
    for secret in (settings.gemini_api_key, settings.smtp_password):
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def _plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word.replace('company', 'companie')}s"


def _validate(quotes, today):
    """(accepted {symbol: (quote, status, warnings)}, rejected {symbol: reason})."""
    accepted, rejected = {}, {}
    current = {}
    for symbol, quote in quotes.items():
        if quote.day > today:
            rejected[symbol] = f"The price is dated {quote.day}, after today, so it was rejected."
        else:
            current[symbol] = quote
    if not current:
        return accepted, rejected
    frame = pd.DataFrame([{"date": q.day.isoformat(), "symbol": q.symbol, "open": q.open,
                           "high": q.high, "low": q.low, "close": q.close, "volume": q.volume,
                           "turnover": q.turnover} for q in current.values()])
    validation = validate_market_data(frame)
    for position, (symbol, quote) in enumerate(current.items()):
        status = validation.row_status.iloc[position]
        issues = validation.row_issues.iloc[position]
        if status == ValidationStatus.INVALID:
            rejected[symbol] = f"The price failed validation ({', '.join(issues)}) and was rejected."
        else:
            accepted[symbol] = (quote, status, ";".join(issues))
    return accepted, rejected


def _observation(quote, status, warnings, run_id, now):
    source = get_source(quote.source)
    return {"symbol": quote.symbol, "date": quote.day, "open": quote.open, "high": quote.high,
            "low": quote.low, "close": quote.close, "volume": float(quote.volume),
            "turnover": quote.turnover,
            "estimated_traded_value": (quote.close * quote.volume
                                       if quote.turnover is None else None),
            "source": source.source_name, "source_priority": source.source_priority,
            "source_url": quote.source_url, "validation_status": str(status),
            "validation_warnings": warnings, "collected_at": now, "update_run_id": run_id}


def _gap_note(last, day):
    if last is None or day <= last:
        return ""
    missing = int(np.busday_count(last + timedelta(days=1), day))
    if not missing:
        return ""
    return (f" Earlier weekdays without a price: {missing} (market holidays, or days the update "
            "didn't run; CSE only provides the latest session, so they're left empty).")


def _not_found_message(collection, settings):
    if collection.cse_failed:
        return ("Official CSE data couldn't be reached and no other price was found. "
                "It will be tried again.")
    if collection.gemini_failed:
        return ("Not in the CSE price list, and the secondary source (Gemini) is unavailable "
                "right now.")
    if "gemini" in settings.price_sources and not settings.gemini_configured:
        return "Not in the CSE price list, and no secondary source is set up."
    if "gemini" not in settings.price_sources:
        return "Not in the CSE price list."
    return "Not in the CSE price list, and the secondary source found no price."


def _symbol_outcome(tracked, collection, stored, rejected, repository, settings):
    symbol = tracked.symbol
    if collection.market_open:
        return SymbolOutcome(symbol, WAITING, "Waiting for the market to close.")
    if symbol in rejected:
        return SymbolOutcome(symbol, REJECTED, rejected[symbol])
    if symbol in stored:
        day = stored[symbol].day
        source = repository.preferred_source(symbol, day) or stored[symbol].source
        if tracked.upload_last_date is not None and day <= tracked.upload_last_date:
            return SymbolOutcome(symbol, UP_TO_DATE, f"Your uploaded file already covers {day}.",
                                 tracked.last_market_date, tracked.last_source)
        last = tracked.last_market_date
        official_now = (day == last and is_ai_source(tracked.last_source)
                        and not is_ai_source(source))
        if last is None or day > last or official_now:
            if is_ai_source(source):
                return SymbolOutcome(symbol, ADDED_SECONDARY,
                                     f"Added {day} from a secondary AI source (Gemini). It is "
                                     "not used in the analysis by default." + _gap_note(last, day),
                                     day, source)
            message = (f"The official CSE price for {day} now replaces the secondary one."
                       if official_now else f"Added {day} from the official CSE.")
            return SymbolOutcome(symbol, ADDED, message + _gap_note(last, day), day, source)
        return SymbolOutcome(symbol, UP_TO_DATE, f"Already up to date ({last}).", last,
                             tracked.last_source)
    reason = collection.skipped.get(symbol, "")
    if reason == NO_TRADES:
        return SymbolOutcome(symbol, NO_TRADES_TODAY,
                             "No trades on CSE in the latest session, so there's no new price "
                             "(the day is left empty, not filled in).")
    if reason.startswith("already has"):
        return SymbolOutcome(symbol, UP_TO_DATE,
                             f"Already up to date ({tracked.last_market_date}).",
                             tracked.last_market_date, tracked.last_source)
    return SymbolOutcome(symbol, NOT_FOUND, _not_found_message(collection, settings))


def _dataset_status(outcomes):
    total = len(outcomes)
    if any(o.outcome == WAITING for o in outcomes):
        return WAITING_FOR_MARKET_CLOSE, ("Waiting for the market to close. The day's prices are "
                                          "collected after the CSE session ends.")
    added = [o for o in outcomes if o.outcome in (ADDED, ADDED_SECONDARY)]
    failed = [o for o in outcomes if o.outcome in (NOT_FOUND, REJECTED)]
    secondary = sum(o.outcome == ADDED_SECONDARY for o in outcomes)
    parts = []
    if added:
        parts.append(f"{len(added)} of {_plural(total, 'company')} updated.")
    if failed:
        parts.append(f"{_plural(len(failed), 'company')} could not be updated.")
    if secondary:
        parts.append(f"{secondary} used a secondary AI source and is left out of the analysis "
                     "by default.")
    if added:
        return (PARTIALLY_UPDATED if failed else UPDATED), " ".join(parts)
    if failed:
        return UPDATE_FAILED, " ".join(parts)
    if all(o.outcome == UP_TO_DATE for o in outcomes):
        return UP_TO_DATE, f"All {_plural(total, 'company')} are up to date."
    quiet = sum(o.outcome == NO_TRADES_TODAY for o in outcomes)
    return NO_NEW_DATA, (f"No new market data: {_plural(quiet, 'company')} had no trades in the "
                         "latest session.")


def _oldest_dates(targets):
    """Per symbol, the oldest last date over the datasets tracking it (None if any has none)."""
    oldest = {}
    for tracked in targets:
        current = oldest.get(tracked.symbol, date.max)
        last = tracked.last_market_date
        oldest[tracked.symbol] = None if current is None or last is None else min(current, last)
    return oldest


def _update(repository, settings, run_id, client_id, trigger, sources, now):
    targets = repository.active_tracked_symbols(client_id)
    if not targets:
        repository.finish_run(run_id, status=NO_TRACKED_DATA)
        return TrackingUpdateResult(NO_TRACKED_DATA, run_id=run_id,
                                    messages=["There is no active tracked market data to update."])

    symbols = sorted({t.symbol for t in targets})
    collection = collect_latest_quotes(symbols, settings, **sources, now=now,
                                       latest_dates=_oldest_dates(targets))
    errors = [_redact(e, settings) for e in collection.errors if not collection.market_open]
    today = now.astimezone(COLOMBO).date()
    accepted, rejected = _validate(collection.quotes, today)

    stored, details, inserted = {}, [], 0
    for symbol, (quote, status, warnings) in accepted.items():
        source = get_source(quote.source)
        if source.is_ai_generated and repository.has_observation(symbol, quote.day):
            new = False         # an official (or earlier) row already covers this day
        else:
            new = repository.add_observation(**_observation(quote, status, warnings, run_id, now))
        inserted += new
        stored[symbol] = quote
        details.append({"symbol": symbol, "date": quote.day.isoformat(),
                        "source": source.source_name, "stored": new,
                        "quantitative_analysis": analysis_use(source)})
    details += [{"symbol": s, "rejected": reason} for s, reason in rejected.items()]
    details += [{"symbol": s, "skipped": reason} for s, reason in collection.skipped.items()]

    result = TrackingUpdateResult(COMPLETED, run_id=run_id, session=collection.session,
                                  rows_added=inserted, messages=errors,
                                  cse_failed=collection.cse_failed)
    by_dataset = defaultdict(list)
    for tracked in targets:
        by_dataset[tracked.dataset_id].append(tracked)
    for dataset_id, rows in by_dataset.items():
        outcomes, symbol_updates = [], {}
        for tracked in rows:
            outcome = _symbol_outcome(tracked, collection, stored, rejected, repository, settings)
            outcomes.append(outcome)
            changes = {"last_status": outcome.outcome, "last_message": outcome.message}
            if outcome.outcome in (ADDED, ADDED_SECONDARY):
                changes.update(last_market_date=outcome.day, last_source=outcome.source)
            symbol_updates[tracked.id] = changes
        status, message = _dataset_status(outcomes)
        added = sum(o.outcome in (ADDED, ADDED_SECONDARY) for o in outcomes)
        repository.record_dataset_update(
            dataset_id, run_id, trigger, status=status, message=message, rows_added=added,
            details=[{**asdict(o), "day": o.day.isoformat() if o.day else None}
                     for o in outcomes],
            symbol_updates=symbol_updates, successful=status in SUCCESSFUL, now=now)
        result.datasets[dataset_id] = DatasetOutcome(dataset_id, rows[0].dataset.client_id,
                                                     status, message, added, outcomes)

    if collection.market_open:
        result.status = WAITING_FOR_MARKET_CLOSE
        result.messages = ["The CSE session is still open; prices are collected after the close."]
    elif errors or rejected:
        result.status = COMPLETED_WITH_ERRORS
    repository.finish_run(run_id, status=result.status, session_date=collection.session,
                          symbols_processed=len(symbols), rows_added=inserted,
                          rows_skipped=len(accepted) - inserted + len(rejected),
                          gemini_used=bool(collection.gemini_symbols),
                          details={"errors": errors, "symbols": details})
    log.info("Tracked market data (%s): %s, %d symbol(s), %d new row(s)", trigger, result.status,
             len(symbols), inserted)
    for message in errors:
        log.warning("  %s", message)
    return result


def run_tracking_update(repository, settings=None, *, client_id=None, trigger=SCHEDULED,
                        cse_snapshot=None, market_closed=None, gemini_quotes=None, now=None):
    """Collect and store the latest session for every active tracked symbol.

    ``client_id`` limits the update to one client's datasets ("Update now"). Safe to run
    any number of times: one update runs at a time, and stored rows are never duplicated.
    A failing symbol or source doesn't undo the symbols that worked.
    """
    settings = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    sources = {"cse_snapshot": cse_snapshot or collect_cse_quotes,
               "market_closed": market_closed or cse_market_closed,
               "gemini_quotes": gemini_quotes or collect_gemini_quotes}
    with repository.update_lock() as acquired:
        if not acquired:
            return TrackingUpdateResult(ALREADY_RUNNING, messages=[
                "Another update is already running. Try again in a minute."])
        run_id = repository.start_run(trigger, client_id)
        try:
            return _update(repository, settings, run_id, client_id, trigger, sources, now)
        except Exception as exc:  # noqa: BLE001 - recorded, and retried by the next run
            message = _redact(f"{type(exc).__name__}: {exc}", settings)
            log.error("Tracked market-data update failed: %s", message)
            repository.finish_run(run_id, status=FAILED, details={"errors": [message]})
            return TrackingUpdateResult(FAILED, run_id=run_id, messages=[
                f"The update stopped unexpectedly ({message}). It will be tried again."])


def run_scheduled_update(settings=None, repository=None):
    """The scheduled job: update every client's tracked data, if the database is set up."""
    settings = settings or load_settings()
    own = repository is None
    try:
        repository = repository or TrackingRepository()
    except ValueError as exc:            # DB_NAME isn't set
        log.info("Tracked market data: skipped, database not configured (%s)", exc)
        return TrackingUpdateResult(DATABASE_NOT_CONFIGURED, messages=[str(exc)])
    try:
        repository.create_tables()
        return run_tracking_update(repository, settings, trigger=SCHEDULED)
    except SQLAlchemyError as exc:
        log.warning("Tracked market data: database unavailable (%s)", type(exc).__name__)
        return TrackingUpdateResult(DATABASE_UNAVAILABLE,
                                    messages=[f"Database unavailable ({type(exc).__name__})"])
    finally:
        if own:
            repository.engine.dispose()


def next_update_time(now=None, last_success=None):
    """When the update service next collects prices: after the close on the next weekday.

    Returns today's time (already passed) if today's update is still due. Market
    holidays aren't known here; on those days the update finds nothing new.
    """
    local = (now or datetime.now(timezone.utc)).astimezone(COLOMBO)
    today = local.replace(hour=UPDATE_HOUR, minute=0, second=0, microsecond=0)
    if local.weekday() < 5 and local >= today and (
            last_success is None or last_success.astimezone(COLOMBO) < today):
        return today
    candidate = today if local < today else today + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate
