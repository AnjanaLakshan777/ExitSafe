"""Return calculations on market data.

Two input layouts are supported, with one formula:

* canonical market data (``date``, ``symbol``, ``close`` and, when present,
  ``validation_status``), as produced by app.data.schemas.market_schema;
  the result column is ``daily_return``.
* the Phase 1 loader layout (``Date``, ``Symbol``, ``Close``) from
  app.data.loaders.market_data; the result column is ``Daily Return``.
"""

DAILY_RETURN = "Daily Return"            # Phase 1 loader layout
CANONICAL_DAILY_RETURN = "daily_return"  # canonical layout

_REQUIRED_COLUMNS = ["Date", "Symbol", "Close"]
CANONICAL_REQUIRED_COLUMNS = ["date", "symbol", "close"]
_INVALID = "INVALID"


def calculate_daily_returns(data):
    """Add a daily-return column computed separately for each stock.

    Daily Return = (today's Close / previous Close) - 1

    "Previous" means the stock's previous trading record, so the first record
    of each stock has no return (NaN). Returns are never computed across two
    different symbols.

    Canonical data (``date``/``symbol``/``close``) gets a ``daily_return``
    column. If it carries ``validation_status``, a return is only kept when
    both of its rows are usable (not INVALID); otherwise it is NaN. Invalid
    rows are never dropped and bridged over, which would silently turn two
    trading days into one return.

    The input is not modified; a new DataFrame sorted by symbol and date is
    returned.
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
    # A row is usable if it has a date and is not INVALID. Rows without a date
    # sort last and must not get a return against the last dated row (validated
    # data already marks them INVALID; this also covers unvalidated frames).
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
