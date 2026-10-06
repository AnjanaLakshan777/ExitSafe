"""Historical volatility of each stock's daily returns."""

import numpy as np
import pandas as pd

from app.analytics.common import (  # noqa: F401  (TRADING_DAYS_PER_YEAR re-exported)
    TRADING_DAYS_PER_YEAR,
    require_canonical_columns,
    validate_periods_per_year,
)
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns


def calculate_daily_volatility(data):
    """Daily volatility (sample standard deviation of returns) per stock."""
    require_canonical_columns(data, "volatility")

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
    """Daily and annualized volatility per stock (annualized = daily x sqrt(periods))."""
    validate_periods_per_year(periods_per_year)

    result = calculate_daily_volatility(data)
    result["annualized_volatility"] = result["daily_volatility"] * np.sqrt(periods_per_year)
    result["periods_per_year"] = periods_per_year
    return result
