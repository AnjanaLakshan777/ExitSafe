"""Daily returns from closing prices.

Works with both the canonical column names (date, symbol, close) and the older
loader's names (Date, Symbol, Close).
"""

DAILY_RETURN = "Daily Return"            # Phase 1 loader layout
CANONICAL_DAILY_RETURN = "daily_return"  # canonical layout

_REQUIRED_COLUMNS = ["Date", "Symbol", "Close"]
CANONICAL_REQUIRED_COLUMNS = ["date", "symbol", "close"]
_INVALID = "INVALID"


def calculate_daily_returns(data):
    """Add each stock's daily return (today's close / previous close - 1).

    A return next to an INVALID row is left empty instead of skipping over the bad
    row. The input is not changed.
    """
    if all(col in data.columns for col in CANONICAL_REQUIRED_COLUMNS):
        return _canonical_daily_returns(data)

    missing = [col for col in _REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        canonical_missing = [c for c in CANONICAL_REQUIRED_COLUMNS if c not in data.columns]
        raise ValueError(
            f"Cannot calculate daily returns, missing column(s): {', '.join(canonical_missing)} "
            f"(canonical layout) or {', '.join(missing)} (Phase 1 loader layout)"
        )

    result = data.sort_values(["Symbol", "Date"], kind="stable").reset_index(drop=True)
    result[DAILY_RETURN] = _simple_returns(result, "Symbol", "Close")
    return result


def _canonical_daily_returns(data):
    result = data.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    returns = _simple_returns(result, "symbol", "close")
    # Rows without a date or marked INVALID can't take part in a return.
    usable = result["date"].notna()
    if "validation_status" in result.columns:
        usable &= result["validation_status"].astype("string").ne(_INVALID).fillna(True)
    previous_usable = usable.groupby(result["symbol"], sort=False).shift(1)
    both_usable = usable & previous_usable.astype("boolean").fillna(False)
    result[CANONICAL_DAILY_RETURN] = returns.where(both_usable)
    return result


def _simple_returns(data, symbol_column, close_column):
    previous_close = data.groupby(symbol_column, sort=False)[close_column].shift(1)
    return data[close_column] / previous_close - 1
