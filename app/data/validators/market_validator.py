"""Validate market data without repairing it.

Each row is marked VALID, WARNING or INVALID with the issues found, and the
file as a whole gets PASS, WARNING or FAIL.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from app.data.schemas.market_schema import (
    CANONICAL_COLUMNS,
    OPTIONAL_MARKET_COLUMNS,
    PRICE_COLUMNS,
    REQUIRED_MARKET_COLUMNS,
    ValidationStatus,
    is_missing,
    parse_dates,
    parse_numbers,
)

OHLC_CHECKS = ["LOW_ABOVE_OPEN", "HIGH_BELOW_OPEN", "HIGH_BELOW_LOW"]
DEFAULT_MAX_INVALID_SHARE = 0.05


class DatasetStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class ValidationResult:
    status: DatasetStatus
    total_rows: int
    valid_rows: int               # rows without errors (includes WARNING rows)
    invalid_rows: int
    duplicate_rows: int           # rows sharing a symbol/date with another row
    exact_duplicate_rows: int     # rows identical to an earlier row
    missing_values: dict[str, int]
    invalid_ohlc: dict[str, int]
    issue_counts: dict[str, int]
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None
    trading_days: int
    symbol_count: int
    missing_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    row_status: pd.Series = field(default_factory=lambda: pd.Series(dtype="string"), repr=False)
    row_issues: pd.Series = field(default_factory=lambda: pd.Series(dtype=object), repr=False)

    def invalid_row_report(self, data):
        """The invalid rows of ``data`` with an ``issues`` column explaining why."""
        mask = self.row_status == ValidationStatus.INVALID
        report = data.loc[mask].copy()
        report["issues"] = self.row_issues[mask].map(", ".join)
        return report

    def summary(self):
        """JSON-friendly summary (no per-row data)."""
        return {
            "status": self.status.value,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "duplicate_rows": self.duplicate_rows,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "missing_values": self.missing_values,
            "invalid_ohlc": self.invalid_ohlc,
            "issue_counts": self.issue_counts,
            "date_min": _iso(self.date_min),
            "date_max": _iso(self.date_max),
            "trading_days": self.trading_days,
            "symbol_count": self.symbol_count,
            "missing_columns": self.missing_columns,
            "warnings": self.warnings,
            "notes": self.notes,
            "failure_reasons": self.failure_reasons,
        }


def validate_market_data(data, expected_date=None, date_format="ISO8601",
                         max_invalid_share=DEFAULT_MAX_INVALID_SHARE,
                         extra_errors=None, extra_warnings=None):
    """Validate canonical-named market data.

    expected_date is the date the source itself states for the file, never the
    date you requested.
    """
    total = len(data)
    notes = _column_notes(data)
    missing_columns = [c for c in REQUIRED_MARKET_COLUMNS if c not in data.columns]
    if missing_columns:
        return _missing_columns_result(data, missing_columns, notes)

    errors, warnings_by_code, missing_values = {}, {}, {}

    date_missing = is_missing(data["date"])
    dates = parse_dates(data["date"], date_format)
    missing_values["date"] = int(date_missing.sum())
    errors["MISSING_DATE"] = date_missing
    errors["INVALID_DATE"] = ~date_missing & dates.isna()

    symbol_missing = is_missing(data["symbol"])
    missing_values["symbol"] = int(symbol_missing.sum())
    errors["MISSING_SYMBOL"] = symbol_missing

    numbers = {}
    for column in PRICE_COLUMNS + ["volume"] + [c for c in OPTIONAL_MARKET_COLUMNS if c in data]:
        missing = is_missing(data[column])
        numbers[column] = parse_numbers(data[column])
        missing_values[column] = int(missing.sum())
        if column in REQUIRED_MARKET_COLUMNS:
            errors[f"MISSING_{column.upper()}"] = missing
        errors[f"INVALID_{column.upper()}"] = ~missing & numbers[column].isna()

    for column in PRICE_COLUMNS:
        errors[f"NON_POSITIVE_{column.upper()}"] = numbers[column] <= 0

    o, h, l, c = (numbers[col] for col in PRICE_COLUMNS)
    ohlc = dict(zip(OHLC_CHECKS, [l > o, h < o, h < l]))
    errors.update(ohlc)

    for column in ["volume", "turnover", "trades"]:
        if column in numbers:
            errors[f"NEGATIVE_{column.upper()}"] = numbers[column] < 0

    symbol_key = data["symbol"].astype("string").str.strip().str.upper()
    has_key = ~symbol_missing & dates.notna()
    keys = pd.DataFrame({"symbol": symbol_key, "date": dates})
    errors["DUPLICATE_SYMBOL_DATE"] = has_key & keys.duplicated(keep=False)

    if expected_date is not None:
        errors["DATE_MISMATCH"] = dates.notna() & (dates.dt.date != expected_date)

    warnings_by_code["CLOSE_OUTSIDE_HIGH_LOW"] = (c < l) | (c > h)
    warnings_by_code["ZERO_VOLUME_WITH_PRICE_RANGE"] = (numbers["volume"] == 0) & (h > l)
    warnings_by_code["NON_INTEGER_VOLUME"] = (numbers["volume"] % 1).fillna(0) != 0
    warnings_by_code["WEEKEND_DATE"] = dates.dt.dayofweek >= 5

    for target, extra in ((errors, extra_errors), (warnings_by_code, extra_warnings)):
        for code, mask in (extra or {}).items():
            target[code] = mask.reindex(data.index, fill_value=False).astype(bool)

    error_frame = pd.DataFrame(errors, index=data.index).astype(bool)
    warning_frame = pd.DataFrame(warnings_by_code, index=data.index).astype(bool)
    row_status, row_issues = _row_outcomes(error_frame, warning_frame)

    invalid_rows = int(error_frame.any(axis=1).sum())
    exact_duplicates = int(data.duplicated(keep="first").sum())
    issue_counts = {code: int(count)
                    for code, count in pd.concat([error_frame, warning_frame], axis=1).sum().items()
                    if count}

    warnings = []
    for code, mask in warnings_by_code.items():
        if mask.any():
            warnings.append(f"{int(mask.sum())} row(s) flagged {code}")
    if exact_duplicates:
        warnings.append(f"{exact_duplicates} row(s) are exact copies of an earlier row")

    failure_reasons = []
    if total == 0:
        failure_reasons.append("Dataset has no rows")
    if issue_counts.get("DATE_MISMATCH"):
        failure_reasons.append(
            f"{issue_counts['DATE_MISMATCH']} row(s) do not carry the source's stated date "
            f"{expected_date}; the data cannot be attributed to that date")
    if total and invalid_rows / total > max_invalid_share:
        failure_reasons.append(
            f"{invalid_rows / total:.2%} of rows are invalid "
            f"(threshold {max_invalid_share:.0%})")

    if failure_reasons:
        status = DatasetStatus.FAIL
    elif invalid_rows or warnings:
        status = DatasetStatus.WARNING
    else:
        status = DatasetStatus.PASS

    valid_dates = dates.dropna()
    return ValidationResult(
        status=status,
        total_rows=total,
        valid_rows=total - invalid_rows,
        invalid_rows=invalid_rows,
        duplicate_rows=int(errors["DUPLICATE_SYMBOL_DATE"].sum()),
        exact_duplicate_rows=exact_duplicates,
        missing_values=missing_values,
        invalid_ohlc={code: int(mask.sum()) for code, mask in ohlc.items()},
        issue_counts=issue_counts,
        date_min=valid_dates.min() if len(valid_dates) else None,
        date_max=valid_dates.max() if len(valid_dates) else None,
        trading_days=int(valid_dates.dt.normalize().nunique()),
        symbol_count=int(symbol_key[~symbol_missing].nunique()),
        warnings=warnings,
        notes=notes,
        failure_reasons=failure_reasons,
        row_status=row_status,
        row_issues=row_issues,
    )


def _row_outcomes(error_frame, warning_frame):
    combined = pd.concat([error_frame, warning_frame], axis=1)
    codes = np.array(combined.columns)
    row_issues = pd.Series([tuple(codes[row]) for row in combined.to_numpy()],
                           index=combined.index, dtype=object)
    has_error = error_frame.any(axis=1).to_numpy()
    has_warning = warning_frame.any(axis=1).to_numpy()
    status = np.where(has_error, ValidationStatus.INVALID.value,
                      np.where(has_warning, ValidationStatus.WARNING.value,
                               ValidationStatus.VALID.value))
    return pd.Series(status, index=combined.index, dtype="string"), row_issues


def _column_notes(data):
    notes = []
    absent = [c for c in OPTIONAL_MARKET_COLUMNS if c not in data.columns]
    if absent:
        notes.append(f"Optional column(s) not provided by source: {', '.join(absent)}")
    extra = [str(c) for c in data.columns if c not in CANONICAL_COLUMNS]
    if extra:
        notes.append(f"Columns outside the canonical schema (not validated): {', '.join(extra)}")
    return notes


def _missing_columns_result(data, missing_columns, notes):
    total = len(data)
    return ValidationResult(
        status=DatasetStatus.FAIL,
        total_rows=total,
        valid_rows=0,
        invalid_rows=total,
        duplicate_rows=0,
        exact_duplicate_rows=0,
        missing_values={},
        invalid_ohlc={},
        issue_counts={},
        date_min=None,
        date_max=None,
        trading_days=0,
        symbol_count=0,
        missing_columns=missing_columns,
        notes=notes,
        failure_reasons=[f"Missing required column(s): {', '.join(missing_columns)}"],
        row_status=pd.Series(ValidationStatus.INVALID.value, index=data.index, dtype="string"),
        row_issues=pd.Series([("MISSING_REQUIRED_COLUMNS",)] * total, index=data.index, dtype=object),
    )


def _iso(value):
    return None if value is None or pd.isna(value) else value.date().isoformat()
