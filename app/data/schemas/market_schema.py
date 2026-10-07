"""The canonical market-data columns every source is mapped onto. Missing values stay empty."""

from dataclasses import dataclass
from app._compat import StrEnum

import pandas as pd

from app.data.source_catalog import find_source


class ValidationStatus(StrEnum):
    """Per-row validation outcome."""

    VALID = "VALID"
    WARNING = "WARNING"    # usable, but something looks questionable
    INVALID = "INVALID"    # must not be used in calculations


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    required: bool
    description: str


MARKET_FIELDS = (
    FieldSpec("date", "datetime64", True, "Trading date, as stated by the source"),
    FieldSpec("symbol", "string", True, "Security symbol as published by the source, e.g. JKH.N0000"),
    FieldSpec("open", "float64", True, "Opening price (LKR)"),
    FieldSpec("high", "float64", True, "Highest price (LKR)"),
    FieldSpec("low", "float64", True, "Lowest price (LKR)"),
    FieldSpec("close", "float64", True, "Closing price (LKR)"),
    FieldSpec("volume", "float64", True, "Shares traded"),
    FieldSpec("change_pct", "float64", False,
              "Source-reported change in percent (-0.88 = -0.88%); informational only, "
              "never used as ExitSafe's return"),
    FieldSpec("turnover", "float64", False, "Value traded (LKR) as reported; null if the source has none"),
    FieldSpec("estimated_traded_value", "float64", False,
              "DERIVED estimate close * volume, only for rows without reported turnover; "
              "never turnover"),
    FieldSpec("trades", "float64", False, "Number of trades; null if the source has none"),
    FieldSpec("source", "string", True, "source_name from the source catalog"),
    FieldSpec("source_priority", "int64", True, "Lower = more trusted (from the catalog trust level)"),
    FieldSpec("source_timestamp", "datetime64 (UTC)", False,
              "Timestamp the source itself attaches to the record; null if none"),
    FieldSpec("validation_status", "string", True, "VALID / WARNING / INVALID"),
    FieldSpec("validation_warnings", "string", True, "Semicolon-separated issue codes; empty if none"),
)

CANONICAL_COLUMNS = [f.name for f in MARKET_FIELDS]
REQUIRED_MARKET_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
OPTIONAL_MARKET_COLUMNS = ["turnover", "trades"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume", "turnover", "trades"]

# Derived columns and the rule that produced them (recorded in provenance).
DERIVED_FIELD_RULES = {
    "estimated_traded_value": ("close * volume, computed only for rows with no reported "
                               "turnover; an estimate, not turnover"),
}

# Cell values treated as "no value" (compared case-insensitively after stripping).
MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "-"})


def is_missing(values):
    """Boolean mask of missing cells (null or a MISSING_TOKENS string)."""
    missing = values.isna()
    if not (pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values)):
        text = values.astype("string").str.strip().str.lower()
        missing |= text.isin(MISSING_TOKENS).fillna(False).astype(bool)
    return missing.astype(bool)


def parse_numbers(values):
    """Parse numbers (thousands separators allowed); anything unreadable becomes NaN."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return values.astype("float64")
    text = values.astype("string").str.strip().str.replace(",", "", regex=False)
    text = text.mask(is_missing(values))
    return pd.to_numeric(text, errors="coerce").astype("float64")


def parse_dates(values, date_format="ISO8601"):
    """Parse dates; missing or unparseable cells become NaT. Never guesses a format."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    text = values.astype("string").str.strip().mask(is_missing(values))
    return pd.to_datetime(text, format=date_format, errors="coerce")


def standardize_columns(data, column_map):
    """Rename source columns to canonical names. Values are not touched."""
    for source_column, canonical in column_map.items():
        if canonical not in CANONICAL_COLUMNS:
            raise ValueError(f"column_map target {canonical!r} is not a canonical column")
        if source_column not in data.columns:
            continue
        if canonical in data.columns and canonical != source_column:
            raise ValueError(f"Both {source_column!r} and {canonical!r} are present; "
                             "mapping would overwrite a column")
    targets = [column_map.get(c, c) for c in data.columns]
    duplicates = sorted({t for t in targets if targets.count(t) > 1})
    if duplicates:
        raise ValueError(f"column_map maps several columns onto: {', '.join(duplicates)}")
    return data.rename(columns=column_map)


def to_canonical(data, source, validation, date_format="ISO8601", source_timestamp_column=None):
    """Build the canonical DataFrame. Invalid rows are kept and flagged, not dropped."""
    result = pd.DataFrame(index=data.index)
    result["date"] = parse_dates(data["date"], date_format)
    symbols = data["symbol"].astype("string").str.strip()
    result["symbol"] = symbols.mask(is_missing(data["symbol"]))
    for column in NUMERIC_COLUMNS + ["change_pct"]:
        result[column] = parse_numbers(data[column]) if column in data.columns else float("nan")
    derivable = result["turnover"].isna() & (result["close"] > 0) & (result["volume"] >= 0)
    result["estimated_traded_value"] = (result["close"] * result["volume"]).where(derivable)
    result["source"] = source.source_name
    result["source_priority"] = source.source_priority
    if "source" in data.columns:
        # A row may name its own registered source (the price updater does this). Blank or
        # unknown labels keep the file's source, so a source is never made up.
        row_sources = [find_source(v.strip()) if isinstance(v, str) else None
                       for v in data["source"]]
        result["source"] = [s.source_name if s else source.source_name for s in row_sources]
        result["source_priority"] = [s.source_priority if s else source.source_priority
                                     for s in row_sources]
    if source_timestamp_column:
        result["source_timestamp"] = pd.to_datetime(data[source_timestamp_column],
                                                    errors="coerce", utc=True)
    else:
        result["source_timestamp"] = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    result["validation_status"] = validation.row_status.astype("string")
    result["validation_warnings"] = validation.row_issues.map(";".join).astype("string")
    return result[CANONICAL_COLUMNS].reset_index(drop=True)


def derived_fields_used(canonical):
    """{column: rule} for derived columns that actually hold values."""
    return {name: rule for name, rule in DERIVED_FIELD_RULES.items()
            if name in canonical and canonical[name].notna().any()}
