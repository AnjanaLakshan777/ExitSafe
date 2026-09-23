"""Return calculations on cleaned market data."""

DAILY_RETURN = "Daily Return"

_REQUIRED_COLUMNS = ["Date", "Symbol", "Close"]


def calculate_daily_returns(data):
    """Add a ``Daily Return`` column computed separately for each stock.

    Daily Return = (today's Close / previous Close) - 1

    "Previous" means the stock's previous trading record, so the first record
    of each stock has no return (NaN). Returns are never computed across two
    different symbols.

    The input is not modified; a new DataFrame sorted by Symbol and Date is
    returned.
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(
            f"Cannot calculate daily returns, missing column(s): {', '.join(missing)}"
        )

    result = data.sort_values(["Symbol", "Date"], kind="stable").reset_index(drop=True)
    previous_close = result.groupby("Symbol", sort=False)["Close"].shift(1)
    result[DAILY_RETURN] = result["Close"] / previous_close - 1
    return result
