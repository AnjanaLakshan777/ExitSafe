"""Risk-adjusted return ratios per symbol: Sharpe and Sortino.

All inputs come from the existing analytics: daily simple returns from
app.analytics.returns.calculate_daily_returns (INVALID rows never contribute
and are never bridged; WARNING rows count) and annualized volatility from
app.analytics.volatility. Only finite returns are used; the first return of
each symbol (NaN) never counts.

Conventions (P = periods_per_year, default 252; rf = annual risk-free rate)
  daily risk-free rate      rf_d = (1 + rf) ** (1 / P) - 1        (compounding-consistent)
  daily excess return       e_t  = r_t - rf_d
  annualized excess return  mean(e) * P                           (arithmetic)
  annualized volatility     std(r, ddof=1) * sqrt(P)              (existing volatility module)
  Sharpe ratio              annualized excess return / annualized volatility
  downside deviation        sqrt(mean(min(e_t, 0) ** 2)) * sqrt(P)
                            (target = rf_d; the mean is over ALL n observations,
                             with non-negative excess returns counting as 0)
  Sortino ratio             annualized excess return / downside deviation
  annualized return         (prod(1 + r_t)) ** (P / n) - 1          (geometric; reported
                            for information, NOT the ratio numerator)

The risk-free rate is an explicit input. The default 0.0 is only a calculation
default: meaningful analysis should supply an appropriate rate.

Undefined results are NaN, never 0 or infinity:
  * fewer than MIN_OBSERVATIONS usable returns: every computed field is NaN
  * Sharpe when daily volatility <= CONSTANT_RETURN_TOLERANCE (no variation)
  * Sortino when the daily downside deviation <= CONSTANT_RETURN_TOLERANCE
    (no return below the risk-free target)
Values are full precision; rounding belongs to presentation.
"""

import math

import numpy as np
import pandas as pd

from app.analytics.common import (
    CONSTANT_RETURN_TOLERANCE,
    TRADING_DAYS_PER_YEAR,
    require_canonical_columns,
    validate_periods_per_year,
)
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns
from app.analytics.volatility import calculate_annualized_volatility

MIN_OBSERVATIONS = 2

RATIO_COLUMNS = ["symbol", "observations", "risk_free_rate", "daily_risk_free_rate",
                 "periods_per_year", "annualized_return", "annualized_volatility",
                 "annualized_excess_return", "sharpe_ratio", "downside_deviation",
                 "sortino_ratio"]
SHARPE_COLUMNS = ["symbol", "observations", "risk_free_rate", "annualized_excess_return",
                  "annualized_volatility", "sharpe_ratio"]
SORTINO_COLUMNS = ["symbol", "observations", "risk_free_rate", "annualized_excess_return",
                   "downside_deviation", "sortino_ratio"]


def daily_risk_free_rate(risk_free_rate, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Convert an annual rate to a per-period rate: (1 + rf) ** (1 / P) - 1."""
    validate_risk_free_rate(risk_free_rate)
    validate_periods_per_year(periods_per_year)
    return (1 + risk_free_rate) ** (1 / periods_per_year) - 1


def validate_risk_free_rate(risk_free_rate):
    """A finite annual rate above -100% (negative rates are allowed)."""
    if (isinstance(risk_free_rate, bool) or not isinstance(risk_free_rate, (int, float))
            or not math.isfinite(risk_free_rate) or not risk_free_rate > -1):
        raise ValueError("risk_free_rate must be a finite annual rate above -1 (-100%), "
                         f"got {risk_free_rate!r}")


def calculate_risk_adjusted_ratios(data, risk_free_rate=0.0,
                                   periods_per_year=TRADING_DAYS_PER_YEAR):
    """Sharpe, Sortino and their components, one row per symbol (sorted).

    Columns: RATIO_COLUMNS. The input is not modified.
    """
    require_canonical_columns(data, "risk-adjusted ratios")
    rf_daily = daily_risk_free_rate(risk_free_rate, periods_per_year)

    volatility = calculate_annualized_volatility(data, periods_per_year).set_index("symbol")
    returns = calculate_daily_returns(data)
    usable = returns[np.isfinite(returns[CANONICAL_DAILY_RETURN])]
    by_symbol = {symbol: group[CANONICAL_DAILY_RETURN].to_numpy()
                 for symbol, group in usable.groupby("symbol", sort=True)}

    rows = []
    for symbol, vol in volatility.iterrows():
        row = {"symbol": symbol, "observations": int(vol["observations"]),
               "risk_free_rate": risk_free_rate, "daily_risk_free_rate": rf_daily,
               "periods_per_year": periods_per_year}
        row.update(_ratios(by_symbol.get(symbol, np.array([])), rf_daily, periods_per_year,
                           vol["daily_volatility"], vol["annualized_volatility"]))
        rows.append(row)
    return pd.DataFrame(rows, columns=RATIO_COLUMNS).astype(
        {"observations": "int64", **{c: "float64" for c in RATIO_COLUMNS[2:] if c != "periods_per_year"}})


def calculate_sharpe_ratio(data, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Per-symbol Sharpe ratio with its numerator and denominator (SHARPE_COLUMNS)."""
    return calculate_risk_adjusted_ratios(data, risk_free_rate, periods_per_year)[SHARPE_COLUMNS]


def calculate_sortino_ratio(data, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Per-symbol Sortino ratio with its numerator and denominator (SORTINO_COLUMNS)."""
    return calculate_risk_adjusted_ratios(data, risk_free_rate, periods_per_year)[SORTINO_COLUMNS]


def _ratios(daily_returns, rf_daily, periods, daily_vol, annual_vol):
    nan = {"annualized_return": np.nan, "annualized_volatility": np.nan,
           "annualized_excess_return": np.nan, "sharpe_ratio": np.nan,
           "downside_deviation": np.nan, "sortino_ratio": np.nan}
    n = len(daily_returns)
    if n < MIN_OBSERVATIONS:
        return nan

    excess = daily_returns - rf_daily
    annual_excess = excess.mean() * periods
    daily_downside = math.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    growth = np.prod(1 + daily_returns)

    return {
        "annualized_return": growth ** (periods / n) - 1,
        "annualized_volatility": annual_vol,
        "annualized_excess_return": annual_excess,
        "sharpe_ratio": (annual_excess / annual_vol
                         if daily_vol > CONSTANT_RETURN_TOLERANCE else np.nan),
        "downside_deviation": daily_downside * math.sqrt(periods),
        "sortino_ratio": (annual_excess / (daily_downside * math.sqrt(periods))
                          if daily_downside > CONSTANT_RETURN_TOLERANCE else np.nan),
    }
