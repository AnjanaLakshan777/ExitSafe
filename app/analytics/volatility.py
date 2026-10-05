"""Historical volatility per symbol, from simple daily returns.

daily_volatility      = sample standard deviation (ddof=1) of daily returns
annualized_volatility = daily_volatility * sqrt(periods_per_year)   (default 252)

Returns come from app.analytics.returns.calculate_daily_returns on canonical
market data, so rows marked INVALID never contribute. Only finite returns are
used. Fewer than two usable returns gives NaN volatility (never a misleading
zero); the ``observations`` column shows how many returns were used.

Values are returned at full precision; rounding belongs to presentation.
Volatility measures how much returns varied historically. It is not, by
itself, an expected loss.
"""

import numpy as np
import pandas as pd

from app.analytics.returns import (
    CANONICAL_DAILY_RETURN,
    CANONICAL_REQUIRED_COLUMNS,
    calculate_daily_returns,
)

TRADING_DAYS_PER_YEAR = 252


def calculate_daily_volatility(data):
    """Per-symbol daily volatility.

    Input: canonical market data with ``date``, ``symbol``, ``close`` (and
    optionally ``validation_status``). The input is not modified.

    Output: one row per symbol, sorted by symbol, with columns
      symbol            security symbol
      observations      number of usable daily returns
      daily_volatility  sample standard deviation of those returns (NaN if < 2)
    """
    missing = [col for col in CANONICAL_REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Cannot calculate volatility, missing column(s): {', '.join(missing)}")

    returns = calculate_daily_returns(data)
    usable = returns[CANONICAL_DAILY_RETURN].where(np.isfinite(returns[CANONICAL_DAILY_RETURN]))
    grouped = usable.groupby(returns["symbol"], sort=True)
    result = pd.DataFrame({
        "observations": grouped.count().astype("int64"),
        "daily_volatility": grouped.std(ddof=1).astype("float64"),
    })
    result.index.name = "symbol"
    return result.reset_index()


def calculate_annualized_volatility(data, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Per-symbol daily and annualized volatility.

    annualized_volatility = daily_volatility * sqrt(periods_per_year)

    The default of 252 assumes 252 trading days per year; calendar gaps
    (weekends, holidays, missing days) are not adjusted for.

    Output: the columns of ``calculate_daily_volatility`` plus
    ``annualized_volatility`` and ``periods_per_year``.
    """
    if (isinstance(periods_per_year, bool) or not isinstance(periods_per_year, (int, float))
            or not periods_per_year > 0):
        raise ValueError(f"periods_per_year must be a positive number, got {periods_per_year!r}")

    result = calculate_daily_volatility(data)
    result["annualized_volatility"] = result["daily_volatility"] * np.sqrt(periods_per_year)
    result["periods_per_year"] = periods_per_year
    return result
