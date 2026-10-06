"""Load and clean a daily market-data CSV (the original Phase 1 loader).

Rows that can't be trusted are dropped, and every removal is logged.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATE = "Date"
SYMBOL = "Symbol"
OPEN = "Open"
HIGH = "High"
LOW = "Low"
CLOSE = "Close"
VOLUME = "Volume"
VALUE_TRADED = "Value Traded"

REQUIRED_COLUMNS = [DATE, SYMBOL, OPEN, HIGH, LOW, CLOSE, VOLUME, VALUE_TRADED]
PRICE_COLUMNS = [OPEN, HIGH, LOW, CLOSE]
NUMERIC_COLUMNS = PRICE_COLUMNS + [VOLUME, VALUE_TRADED]

# A row without these cannot be used in any calculation.
ESSENTIAL_COLUMNS = [DATE, SYMBOL, CLOSE]


def load_market_data(file_path):
    """Load a market-data CSV and return a clean DataFrame.

    Drops rows with missing essentials, bad prices or negative volumes, and keeps
    the last copy of duplicate Symbol/Date rows. Raises FileNotFoundError or
    ValueError if the file can't be used.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Market data file not found: {path}")

    try:
        # Read as text and convert each column explicitly below.
        raw = pd.read_csv(path, dtype=str, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Market data file is empty: {path}") from exc

    raw.columns = raw.columns.str.strip()
    _validate_columns(raw, path)

    data = raw[REQUIRED_COLUMNS].copy()
    data = _convert_types(data)
    data = _drop_missing_essentials(data)
    data = _drop_invalid_values(data)
    data = _sort_and_deduplicate(data)
    return data.reset_index(drop=True)


def _validate_columns(data, path):
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(
            f"Market data file {path} is missing required column(s): "
            f"{', '.join(missing)}. Required columns are: {', '.join(REQUIRED_COLUMNS)}."
        )


def _convert_types(data):
    data[DATE] = pd.to_datetime(
        data[DATE].str.strip(), format="ISO8601", errors="coerce"
    ).dt.normalize()

    symbols = data[SYMBOL].str.strip().str.upper()
    data[SYMBOL] = symbols.mask(symbols == "")

    for col in NUMERIC_COLUMNS:
        cleaned = data[col].str.strip().str.replace(",", "", regex=False)
        data[col] = pd.to_numeric(cleaned, errors="coerce")

    return data


def _drop_missing_essentials(data):
    missing = data[ESSENTIAL_COLUMNS].isna().any(axis=1)
    _log_dropped(missing, "missing or unparseable Date, Symbol or Close")
    return data[~missing]


def _drop_invalid_values(data):
    # Missing Open/High/Low are tolerated (compare as False); only present,
    # non-positive prices are rejected.
    bad_price = (data[PRICE_COLUMNS] <= 0).any(axis=1)
    _log_dropped(bad_price, "zero or negative price")

    bad_activity = (data[[VOLUME, VALUE_TRADED]] < 0).any(axis=1)
    _log_dropped(bad_activity & ~bad_price, "negative Volume or Value Traded")

    return data[~(bad_price | bad_activity)]


def _sort_and_deduplicate(data):
    # The sort is stable, so keep="last" keeps the row that came last in the file.
    data = data.sort_values([SYMBOL, DATE], kind="stable")
    duplicated = data.duplicated(subset=[SYMBOL, DATE], keep="last")
    _log_dropped(duplicated, "duplicate Symbol/Date record")
    return data[~duplicated]


def _log_dropped(mask, reason):
    count = int(mask.sum())
    if count:
        logger.warning("Dropped %d market-data row(s): %s", count, reason)
