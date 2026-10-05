"""Generic CSV market-data import: provider-specific layouts -> canonical market data.

Users can import historical price CSVs from different providers without editing
the file. For example::

    Date,Price,Open,High,Low,Vol.,Change %
    12/31/2025,660.09,664.75,665,659.44,7.94M,-0.88%

becomes canonical rows with ``close`` (from Price), ``volume`` (7940000) and
``change_pct`` (-0.88).

Pipeline (each step reports instead of guessing):
  1. read the file unchanged and attach provenance  (load_raw_market_file)
  2. detect columns from aliases; conflicting duplicates are flagged, not chosen
  3. normalize values to plain text: K/M/B volumes, thousands separators,
     percent signs, and dates to ISO (ambiguous dates are rejected, never swapped)
  4. validate with the shared canonical validator (+ importer-specific row checks)
  5. build the canonical frame and an import report

Analytical returns are always computed from close prices elsewhere
(app.analytics.returns); a supplied "Change %" is kept for reference and only
compared against the closes to warn about inconsistencies.
"""

import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from app.data.loaders.cse_market_loader import load_raw_market_file
from app.data.schemas.market_schema import (
    MISSING_TOKENS,
    derived_fields_used,
    is_missing,
    parse_dates,
    parse_numbers,
    to_canonical,
)
from app.data.validators.market_validator import ValidationResult, validate_market_data

USER_CSV_SOURCE = "user_csv_upload"

# Canonical field -> accepted source headers, highest priority first. Headers
# match ignoring case and whitespace, so "Change %" == "change%" and
# "Vol." == "vol.". When several headers for one field are present, the first
# listed wins and the others are compared against it.
COLUMN_ALIASES = {
    "date": ["Date"],
    "symbol": ["Symbol"],
    "open": ["Open", "Opening Price"],
    "high": ["High"],
    "low": ["Low"],
    "close": ["Close", "Closing Price", "Price"],
    "volume": ["Volume", "Share Volume", "Vol.", "Vol"],
    "change_pct": ["Change %", "change_pct", "Change"],
    "turnover": ["Turnover", "Turnover (Rs.)", "Value Traded"],
    "trades": ["Trades", "No. of Trades"],
}

# Fields whose values may be abbreviated with K / M / B suffixes.
SUFFIX_FIELDS = frozenset({"volume", "turnover"})
MULTIPLIERS = {"K": Decimal(10) ** 3, "M": Decimal(10) ** 6, "B": Decimal(10) ** 9}

# Maximum gap, in percentage points, between a supplied Change % and the change
# implied by consecutive closes before a CHANGE_PCT_MISMATCH warning.
CHANGE_PCT_TOLERANCE = 0.05

# A file name can stand in for a symbol only if it is exactly a symbol (JKH.N0000.csv).
SYMBOL_FILENAME = re.compile(r"[A-Za-z0-9]{1,12}(?:\.[A-Za-z0-9]{1,8})?")

_NUMBER = re.compile(r"([+-]?)(\d{1,3}(?:,\d{3})+|\d*)(\.\d+)?\s*([KMB])?", re.IGNORECASE)
_DAY_MONTH_YEAR = r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$"


class MarketDataImportError(ValueError):
    """The file cannot be imported at all (row-level problems are reported, not raised)."""


# --- value parsing --------------------------------------------------------------------------

def parse_number_text(value, allow_suffix=False):
    """Normalize one numeric cell to plain decimal text.

    Returns "" for a missing value, the normalized text for a readable number
    ("7.94M" -> "7940000", "250,000" -> "250000"), or None when the value is
    not a number this importer can read safely. "." is always the decimal point
    and "," only a thousands separator in groups of three, so "1.234,5" or "1,23"
    are rejected instead of misread. Decimal arithmetic avoids float artefacts.
    """
    if value is None or isinstance(value, bool):
        return "" if value is None else None
    if isinstance(value, (int, float, Decimal)):
        return "" if value != value else _decimal_text(Decimal(str(value)))
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return ""
    match = _NUMBER.fullmatch(text)
    if not match or not (match[2] or match[3]) or (match[4] and not allow_suffix):
        return None
    number = Decimal(match[1] + (match[2] or "0").replace(",", "") + (match[3] or ""))
    if match[4]:
        number *= MULTIPLIERS[match[4].upper()]
    return _decimal_text(number)


def parse_percent_text(value, percent_units=False):
    """'-0.88%' -> '-0.88' (percent units). A value without a % sign is only
    accepted when the column is known to hold percentages (header contains % or
    'pct'); a bare 'Change' column could equally be an absolute price change."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if not isinstance(value, str):
        return parse_number_text(value) if percent_units else None
    text = value.strip()
    if text.lower() in MISSING_TOKENS:
        return ""
    if text.endswith("%"):
        return parse_number_text(text[:-1].strip())
    return parse_number_text(text) if percent_units else None


def _decimal_text(number):
    text = format(number.normalize(), "f")
    return "0" if text == "-0" else text


def normalize_dates(values, date_format=None):
    """Convert dates to ISO text without ever swapping day and month silently.

    Returns (iso_values, ambiguous_mask, convention). With ``date_format`` the
    given strptime format is applied strictly. Otherwise ISO dates (2025-12-31)
    pass through, and D/M/YYYY-style dates use the file's convention: if any
    first part is > 12 the file is day-first, if any second part is > 12 it is
    month-first. Without such evidence (or with contradictory evidence) a row is
    only accepted when it is unambiguous on its own (a part > 12, or day ==
    month); otherwise it is flagged AMBIGUOUS_DATE. Unreadable values are kept
    as-is so the validator reports INVALID_DATE.
    """
    text = values.astype("string").str.strip()
    present = ~is_missing(values)
    result = text.astype(object).where(present, "")
    ambiguous = pd.Series(False, index=values.index)

    if date_format:
        for index in text[present].index:
            try:
                result[index] = datetime.strptime(text[index], date_format).date().isoformat()
            except ValueError:
                pass
        return result, ambiguous, f"explicit format {date_format}"

    parts = text.str.extract(_DAY_MONTH_YEAR).astype("Float64")
    numeric = parts[0].notna() & present
    first, second = parts[0][numeric], parts[1][numeric]
    day_first, month_first = bool((first > 12).any()), bool((second > 12).any())
    convention = ("DD/MM/YYYY" if day_first and not month_first else
                  "MM/DD/YYYY" if month_first and not day_first else None)

    for index in parts[numeric].index:
        a, b, year = (int(v) for v in parts.loc[index])
        if convention == "DD/MM/YYYY" or (convention is None and a > 12):
            day, month = a, b
        elif convention == "MM/DD/YYYY" or (convention is None and (b > 12 or a == b)):
            month, day = a, b
        else:
            ambiguous[index] = True
            continue
        try:
            result[index] = date(year, month, day).isoformat()
        except ValueError:
            pass  # e.g. 02/30/2025 -> stays raw -> INVALID_DATE

    labels = []
    if text[present].str.fullmatch(r"\d{4}-\d{2}-\d{2}").any():
        labels.append("YYYY-MM-DD")
    if numeric.any():
        labels.append(convention or ("mixed day-first and month-first" if day_first and month_first
                                     else "undetermined (all day and month parts <= 12)"))
    return result, ambiguous, ", ".join(labels) or "none recognised"


# --- column detection and normalization ----------------------------------------------------------

def _key(name):
    return re.sub(r"\s+", "", str(name)).casefold()


_ALIAS_INDEX = {_key(alias): (canonical, rank)
                for canonical, aliases in COLUMN_ALIASES.items()
                for rank, alias in enumerate(aliases)}


def detect_columns(columns, column_map=None):
    """({canonical: [source columns, priority order]}, [unrecognised columns]).

    An explicit catalog ``column_map`` entry outranks the generic aliases."""
    found, unrecognised = {}, []
    for column in columns:
        if column_map and column in column_map:
            canonical, rank = column_map[column], -1
        elif _key(column) in _ALIAS_INDEX:
            canonical, rank = _ALIAS_INDEX[_key(column)]
        else:
            unrecognised.append(column)
            continue
        found.setdefault(canonical, []).append((rank, column))
    return ({canonical: [c for _, c in sorted(items, key=lambda item: item[0])]
             for canonical, items in found.items()}, unrecognised)


@dataclass
class NormalizedMarketFrame:
    data: pd.DataFrame                    # canonical column names, normalized text values
    column_mapping: dict[str, str]        # source column -> canonical column
    ignored_columns: dict[str, str]       # source column -> reason
    symbol_source: str                    # "column", "parameter" or "file name"
    date_convention: str
    extra_errors: dict[str, pd.Series] = field(default_factory=dict)
    extra_warnings: dict[str, pd.Series] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def normalize_market_frame(raw, symbol=None, *, column_map=None, date_format=None, file_name=None,
                           infer_symbol_from_filename=False,
                           change_pct_tolerance=CHANGE_PCT_TOLERANCE):
    """Map and normalize a raw provider frame. ``raw`` is never modified."""
    assignments, unrecognised = detect_columns(raw.columns, column_map)
    for canonical in ("date", "symbol"):
        if len(assignments.get(canonical, [])) > 1:
            raise MarketDataImportError(
                f"Several columns could be the {canonical}: {', '.join(assignments[canonical])}. "
                "ExitSafe will not choose between them.")

    out = pd.DataFrame(index=raw.index)
    result = NormalizedMarketFrame(
        data=out, column_mapping={}, ignored_columns={c: "not a recognised market-data column"
                                                      for c in unrecognised},
        symbol_source="", date_convention="none")

    for canonical, columns in assignments.items():
        primary, *others = columns
        if canonical == "date":
            out["date"], ambiguous, result.date_convention = normalize_dates(raw[primary], date_format)
            if ambiguous.any():
                result.extra_errors["AMBIGUOUS_DATE"] = ambiguous
                result.notes.append(
                    f"{int(ambiguous.sum())} date(s) could be read as day/month or month/day and "
                    "were rejected; pass date_format (e.g. '%d/%m/%Y') to resolve them.")
        elif canonical == "symbol":
            out["symbol"] = raw[primary]
        elif canonical == "change_pct":
            if not _add_change_pct(raw, primary, out, result):
                continue
        else:
            out[canonical] = _normalized_numbers(raw[primary], canonical in SUFFIX_FIELDS)
            _note_abbreviations(raw[primary], primary, canonical, result)
        result.column_mapping[primary] = canonical
        for other in others:
            _compare_duplicate(raw, primary, other, canonical, out, result)

    _apply_symbol(out, result, symbol, file_name, infer_symbol_from_filename)

    if {"change_pct", "close", "date", "symbol"} <= set(out.columns):
        mismatch = change_pct_mismatches(out, change_pct_tolerance)
        if mismatch.any():
            result.extra_warnings["CHANGE_PCT_MISMATCH"] = mismatch
    return result


def _normalized_numbers(values, allow_suffix):
    normalized = values.map(lambda v: parse_number_text(v, allow_suffix))
    # Unreadable values stay as the raw text so the validator reports INVALID_<COL>.
    return normalized.where(normalized.notna(), values.astype(object))


def _add_change_pct(raw, column, out, result):
    percent_units = "%" in str(column) or "pct" in str(column).lower()
    normalized = raw[column].map(lambda v: parse_percent_text(v, percent_units))
    present = ~is_missing(raw[column])
    unreadable = present & normalized.isna()
    if not percent_units and present.any() and unreadable[present].all():
        result.ignored_columns[column] = (
            "change values have no % sign and the header does not say they are percentages; "
            "units are ambiguous (could be an absolute price change), so the column was ignored")
        return False
    out["change_pct"] = normalized.where(normalized.notna(), "")
    if unreadable.any():
        result.extra_warnings["CHANGE_PCT_UNREADABLE"] = unreadable
    return True


def _note_abbreviations(values, column, canonical, result):
    if canonical not in SUFFIX_FIELDS:
        return
    abbreviated = values.astype("string").str.strip().str.contains(r"\d\s*[KMBkmb]$", na=False)
    if abbreviated.any():
        result.notes.append(
            f"{int(abbreviated.sum())} {canonical} value(s) in '{column}' use K/M/B abbreviations; "
            "they are only as precise as the digits shown (7.94M = 7,940,000 +/- 5,000).")


def _compare_duplicate(raw, primary, other, canonical, out, result):
    if canonical == "change_pct":
        units = "%" in str(other) or "pct" in str(other).lower()
        other_values = parse_numbers(raw[other].map(lambda v: parse_percent_text(v, units)).fillna(""))
    else:
        other_values = parse_numbers(_normalized_numbers(raw[other], canonical in SUFFIX_FIELDS))
    primary_values = parse_numbers(out[canonical])
    conflict = primary_values.notna() & other_values.notna() & (primary_values != other_values)
    only_other = primary_values.isna() & other_values.notna()
    reason = f"duplicate of '{primary}' for {canonical}; '{primary}' was used"
    if conflict.any():
        code = f"{canonical.upper()}_SOURCE_CONFLICT"
        result.extra_errors[code] = conflict
        reason += f"; {int(conflict.sum())} row(s) differ and are marked INVALID ({code})"
    else:
        reason += "; values identical wherever both are present"
    if only_other.any():
        reason += f"; {int(only_other.sum())} row(s) had a value only in '{other}' (not used)"
    result.ignored_columns[other] = reason


def _apply_symbol(out, result, symbol, file_name, infer_symbol_from_filename):
    if symbol is not None:
        symbol = str(symbol).strip().upper()
        if not symbol:
            raise MarketDataImportError("The symbol parameter is empty.")
    if "symbol" in out:
        if symbol:
            given = out["symbol"].astype("string").str.strip().str.upper()
            differing = ~is_missing(out["symbol"]) & (given != symbol)
            if differing.any():
                raise MarketDataImportError(
                    f"symbol={symbol!r} conflicts with the file's Symbol column "
                    f"({', '.join(sorted(given[differing].unique()))}). ExitSafe will not choose.")
        result.symbol_source = "column"
    elif symbol:
        out["symbol"] = symbol
        result.symbol_source = "parameter"
    elif infer_symbol_from_filename:
        stem = Path(file_name).stem if file_name else ""
        if not SYMBOL_FILENAME.fullmatch(stem):
            raise MarketDataImportError(
                f"Cannot infer a symbol from the file name {file_name!r}: the name must be exactly "
                "a symbol, e.g. 'JKH.N0000.csv'. Pass symbol=... instead.")
        out["symbol"] = stem.upper()
        result.symbol_source = "file name"
    else:
        raise MarketDataImportError(
            "The file has no Symbol column and no symbol was given. Pass the security, e.g. "
            "load_csv_market_data(path, symbol='JKH.N0000'). ExitSafe does not guess the company.")


def change_pct_mismatches(frame, tolerance=CHANGE_PCT_TOLERANCE):
    """Rows whose supplied Change % differs from the change implied by consecutive
    closes (same symbol, previous usable date) by more than ``tolerance`` points.

    A data-consistency check only: it never replaces the analytical return.
    """
    dates = parse_dates(frame["date"])
    close = parse_numbers(frame["close"])
    change = parse_numbers(frame["change_pct"])
    symbol = frame["symbol"].astype("string").str.strip().str.upper()
    usable = dates.notna() & (close > 0) & symbol.notna()
    keys = pd.DataFrame({"symbol": symbol, "date": dates})
    usable &= ~keys.duplicated(keep=False)

    ordered = pd.DataFrame({"symbol": symbol, "date": dates, "close": close})[usable]
    ordered = ordered.sort_values(["symbol", "date"], kind="stable")
    implied = (ordered["close"] / ordered.groupby("symbol")["close"].shift(1) - 1) * 100
    gap = (change.reindex(ordered.index) - implied).abs()
    return (gap > tolerance).reindex(frame.index, fill_value=False).astype(bool)


# --- import entry point --------------------------------------------------------------------------

@dataclass
class ImportReport:
    file_name: str
    source_name: str
    symbols: list[str]
    symbol_source: str
    detected_columns: list[str]
    column_mapping: dict[str, str]
    ignored_columns: dict[str, str]
    normalized_columns: list[str]
    date_convention: str
    rows_read: int
    valid_rows: int
    invalid_rows: int
    status: str
    date_min: str | None
    date_max: str | None
    missing_values: dict[str, int]
    issue_counts: dict[str, int]
    warnings: list[str]
    notes: list[str]
    failure_reasons: list[str]
    source_metadata: dict

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MarketImportResult:
    data: pd.DataFrame | None       # canonical frame; None when required columns are missing
    validation: ValidationResult
    provenance: object              # Provenance
    report: ImportReport


def load_csv_market_data(file_path, symbol=None, *, source_name=USER_CSV_SOURCE,
                         infer_symbol_from_filename=False, date_format=None, retrieval_time=None,
                         source_url=None, source_version=None,
                         change_pct_tolerance=CHANGE_PCT_TOLERANCE):
    """Import a provider CSV into canonical market data.

    symbol: required when the file has no Symbol column (unless
        ``infer_symbol_from_filename`` is True and the file name is exactly a symbol).
    date_format: strptime format to use instead of detection, e.g. "%d/%m/%Y".
    source_name: catalog entry describing the file's origin (default: unverified user upload).

    Raises MarketDataImportError when the file cannot be imported at all;
    row-level problems are reported in the result instead.
    """
    path = Path(file_path)
    if path.suffix.lower() != ".csv":
        raise MarketDataImportError(f"Expected a .csv file, got {path.name!r}")
    try:
        raw = load_raw_market_file(path, source_name, retrieval_time=retrieval_time,
                                   source_url=source_url, source_version=source_version)
    except pd.errors.EmptyDataError as exc:
        raise MarketDataImportError(f"{path.name} is empty") from exc

    normalized = normalize_market_frame(
        raw.data, symbol, column_map=raw.source.column_map, date_format=date_format,
        file_name=path.name, infer_symbol_from_filename=infer_symbol_from_filename,
        change_pct_tolerance=change_pct_tolerance)
    validation = validate_market_data(normalized.data, extra_errors=normalized.extra_errors,
                                      extra_warnings=normalized.extra_warnings)

    canonical, provenance = None, raw.provenance
    if not validation.missing_columns:
        canonical = to_canonical(normalized.data, raw.source, validation)
        provenance = replace(provenance, derived_fields=derived_fields_used(canonical))

    report = ImportReport(
        file_name=path.name,
        source_name=raw.source.source_name,
        symbols=sorted(normalized.data["symbol"][~is_missing(normalized.data["symbol"])]
                       .astype(str).str.strip().unique()),
        symbol_source=normalized.symbol_source,
        detected_columns=[str(c) for c in raw.data.columns],
        column_mapping=normalized.column_mapping,
        ignored_columns=normalized.ignored_columns,
        normalized_columns=([c for c in canonical.columns if canonical[c].notna().any()]
                            if canonical is not None else list(normalized.data.columns)),
        date_convention=normalized.date_convention,
        rows_read=len(raw.data),
        valid_rows=validation.valid_rows,
        invalid_rows=validation.invalid_rows,
        status=validation.status.value,
        date_min=_day(validation.date_min),
        date_max=_day(validation.date_max),
        missing_values=validation.missing_values,
        issue_counts=validation.issue_counts,
        warnings=validation.warnings,
        notes=normalized.notes + validation.notes,
        failure_reasons=validation.failure_reasons,
        source_metadata=provenance.to_dict(),
    )
    return MarketImportResult(data=canonical, validation=validation, provenance=provenance,
                              report=report)


def _day(value):
    return None if value is None or pd.isna(value) else value.date().isoformat()
