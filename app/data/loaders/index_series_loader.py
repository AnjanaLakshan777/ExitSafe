"""Market-index price series (e.g. ASPI, S&P SL20) -> validated index data.

Stock files go through csv_market_loader, which needs open/high/low/volume. An
index series often has only one value per day (ASPI files commonly carry no
volume), so it gets this small loader with the same conventions:

  columns   date, index_name, close (+ validation_status, validation_issues)
  headers   Date | Index, Index Name, index_name | Close, Price, Value, Index Value
            (matched ignoring case and spaces); a file without an Index column
            needs ``index_name`` from the caller - it is never guessed
  dates     the importer's normalize_dates: day and month are never swapped
            silently; an ambiguous date is an error
  numbers   the importer's parse_number_text ("12,345.67" -> 12345.67)

Row status (nothing is repaired or dropped; analytics skip INVALID rows)
  INVALID  MISSING_DATE, INVALID_DATE, AMBIGUOUS_DATE, MISSING_INDEX_NAME,
           MISSING_CLOSE, INVALID_CLOSE, NON_POSITIVE_CLOSE,
           DUPLICATE_INDEX_DATE (every copy: conflicting values cannot be resolved)
  WARNING  WEEKEND_DATE
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.loaders.cse_market_loader import TEXT_SUFFIXES, detect_delimiter
from app.data.loaders.csv_market_loader import normalize_dates, parse_number_text
from app.data.schemas.market_schema import ValidationStatus, is_missing

INDEX_COLUMN_ALIASES = {
    "date": ["Date"],
    "index_name": ["index_name", "Index", "Index Name"],
    "close": ["Close", "Price", "Value", "Index Value", "Closing Value"],
}
INDEX_SERIES_COLUMNS = ["date", "index_name", "close", "validation_status", "validation_issues"]
_ERROR_CODES = ("MISSING_DATE", "INVALID_DATE", "AMBIGUOUS_DATE", "MISSING_INDEX_NAME",
                "MISSING_CLOSE", "INVALID_CLOSE", "NON_POSITIVE_CLOSE", "DUPLICATE_INDEX_DATE")


class IndexImportError(ValueError):
    """The index file cannot be imported at all."""


class IndexNameRequiredError(IndexImportError):
    """The file has no Index column and no index name was supplied (never guessed)."""


@dataclass(frozen=True)
class IndexImportResult:
    data: pd.DataFrame                 # INDEX_SERIES_COLUMNS, sorted by index_name and date
    file_name: str
    index_names: list
    rows_read: int
    invalid_rows: int
    warning_rows: int
    issue_counts: dict
    date_convention: str
    column_mapping: dict               # canonical -> source header


def load_index_csv(file_path, index_name=None, date_format=None):
    """Read a CSV/TSV/TXT index file into validated index data (see module docstring)."""
    path = Path(file_path)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise IndexImportError(f"Expected a .csv, .tsv or .txt file, got {path.name!r}")
    try:
        raw = pd.read_csv(path, sep=detect_delimiter(path), dtype=str, keep_default_na=False,
                          encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise IndexImportError(f"{path.name} is empty") from exc
    if raw.empty:
        raise IndexImportError(f"{path.name} has no data rows")

    mapping = _detect_columns(raw.columns)
    missing = [c for c in ("date", "close") if c not in mapping]
    if missing:
        raise IndexImportError(f"{path.name} is missing column(s): {', '.join(missing)} "
                               f"(found: {', '.join(map(str, raw.columns))})")
    given = (index_name or "").strip().upper() or None
    if "index_name" in mapping:
        names = raw[mapping["index_name"]].astype("string").str.strip().str.upper()
        if given is not None and (names.dropna() != given).any():
            raise IndexImportError(f"The file's Index column does not match the selected "
                                   f"index {given!r}")
    elif given is None:
        raise IndexNameRequiredError(f"{path.name} has no Index column: choose the index "
                                     "name (e.g. ASPI or S&P SL20).")
    else:
        names = pd.Series(given, index=raw.index, dtype="string")

    dates, ambiguous, convention = normalize_dates(raw[mapping["date"]], date_format)
    closes_text = raw[mapping["close"]].map(parse_number_text)
    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(dates, index=raw.index).astype("string"),
                               format="ISO8601", errors="coerce"),
        "index_name": names,
        "close": pd.to_numeric(closes_text.where(closes_text != "", None),
                               errors="coerce").astype("float64"),
    })
    frame = validate_index_series(frame, extra_errors={
        "AMBIGUOUS_DATE": ambiguous,
        "MISSING_DATE": is_missing(raw[mapping["date"]]),
        "MISSING_CLOSE": is_missing(raw[mapping["close"]]),
        "INVALID_CLOSE": closes_text.isna(),
    })
    frame = frame.sort_values(["index_name", "date"], kind="stable").reset_index(drop=True)
    issues = frame["validation_issues"].str.split("; ").explode()
    return IndexImportResult(
        data=frame, file_name=path.name,
        index_names=sorted(frame["index_name"].dropna().unique()),
        rows_read=len(raw),
        invalid_rows=int((frame["validation_status"] == ValidationStatus.INVALID.value).sum()),
        warning_rows=int((frame["validation_status"] == ValidationStatus.WARNING.value).sum()),
        issue_counts={k: int(v) for k, v in issues[issues != ""].value_counts().items()},
        date_convention=convention,
        column_mapping={k: str(v) for k, v in mapping.items()})


def validate_index_series(data, extra_errors=None):
    """Add validation_status / validation_issues to a date, index_name, close frame.

    Returns a new frame; the input is not modified. ``extra_errors`` maps an
    issue code to a boolean mask (aligned to ``data``) of rows with that error.
    """
    frame = data[["date", "index_name", "close"]].copy()
    errors = {code: pd.Series(False, index=frame.index) for code in _ERROR_CODES}
    for code, mask in (extra_errors or {}).items():
        errors[code] = errors[code] | mask.reindex(frame.index, fill_value=False).astype(bool)
    errors["INVALID_DATE"] |= frame["date"].isna() & ~errors["MISSING_DATE"] & ~errors["AMBIGUOUS_DATE"]
    errors["MISSING_INDEX_NAME"] |= is_missing(frame["index_name"])
    close = frame["close"].astype("float64")
    errors["MISSING_CLOSE"] |= close.isna() & ~errors["INVALID_CLOSE"]
    errors["INVALID_CLOSE"] |= np.isinf(close)
    errors["NON_POSITIVE_CLOSE"] |= np.isfinite(close) & (close <= 0)
    keyed = frame["date"].notna() & ~errors["MISSING_INDEX_NAME"]
    errors["DUPLICATE_INDEX_DATE"] |= keyed & frame[keyed].duplicated(
        subset=["index_name", "date"], keep=False).reindex(frame.index, fill_value=False)
    warnings = {"WEEKEND_DATE": frame["date"].dt.dayofweek >= 5}

    has_error = pd.DataFrame(errors).any(axis=1)
    has_warning = pd.DataFrame(warnings).fillna(False).any(axis=1)
    codes = pd.DataFrame({**errors, **{k: v.fillna(False) for k, v in warnings.items()}})
    frame["validation_status"] = np.where(
        has_error, ValidationStatus.INVALID.value,
        np.where(has_warning, ValidationStatus.WARNING.value, ValidationStatus.VALID.value))
    frame["validation_issues"] = codes.apply(lambda row: "; ".join(row.index[row]), axis=1)
    return frame[INDEX_SERIES_COLUMNS]


def _key(name):
    return re.sub(r"\s+", "", str(name)).casefold()


def _detect_columns(columns):
    """canonical -> source header; when several aliases are present the first listed wins."""
    by_key = {}
    for column in columns:
        by_key.setdefault(_key(column), column)
    mapping = {}
    for canonical, aliases in INDEX_COLUMN_ALIASES.items():
        for alias in aliases:
            if _key(alias) in by_key:
                mapping[canonical] = by_key[_key(alias)]
                break
    return mapping
