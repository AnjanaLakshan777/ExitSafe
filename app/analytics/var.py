"""Stock-level 1-day Value at Risk (VaR): historical and parametric (Normal).

Sign convention: VaR is a POSITIVE LOSS MAGNITUDE. 0.032 means a 3.2% one-day
loss threshold. (If even the lower tail of the returns is a gain, VaR is
negative; it is reported as calculated, not floored at zero.)

Interpretation: at confidence C, the 1-day VaR is the loss threshold that,
under the chosen method and data, is expected to be exceeded on only about
(1 - C) of days. It is not the maximum possible loss, and it says nothing about
how large losses are once the threshold is exceeded (that is CVaR).

For confidence C, alpha = 1 - C, and the daily returns r of one symbol:
  historical VaR = -quantile(r, alpha)            empirical distribution
  parametric VaR = -(mean(r) + z_alpha * std(r))  Normal approximation
                   z_alpha = scipy.stats.norm.ppf(alpha) (negative for alpha < 0.5)
                   std uses ddof = 1, as in the volatility module

Quantile convention: numpy.quantile(method="linear") (Hyndman & Fan type 7):
on the sorted returns x_0..x_{n-1}, h = (n - 1) * alpha and the quantile is
x_floor(h) + (h - floor(h)) * (x_floor(h)+1 - x_floor(h)). The method is passed
explicitly (QUANTILE_METHOD), never left to a library default.

Data: daily simple returns from app.analytics.returns.calculate_daily_returns
(INVALID rows give no returns and are never bridged; WARNING rows count; the
first return of each symbol never counts). Only finite returns are used, and
both methods use exactly the same returns for a symbol.

Sample size: a symbol needs at least ``min_observations`` usable returns
(default 20); otherwise both VaR values are NaN (never 0), and
``observations`` shows how many were available. A small empirical sample says
little about rare tail events (e.g. 99% VaR from 20 returns).

The parametric method ASSUMES returns are approximately normally distributed;
real returns often have fatter tails. Historical VaR makes fewer
distributional assumptions but depends entirely on the sample. Values are full
precision; rounding belongs to presentation.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.analytics.common import require_canonical_columns
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MIN_OBSERVATIONS = 20
QUANTILE_METHOD = "linear"

SUMMARY_COLUMNS = ["symbol", "observations", "min_observations", "sufficient_data",
                   "confidence_level", "historical_var", "parametric_var"]
HISTORICAL_COLUMNS = ["symbol", "observations", "confidence_level", "historical_var"]
PARAMETRIC_COLUMNS = ["symbol", "observations", "confidence_level", "parametric_var"]


def validate_confidence_level(confidence_level):
    """A finite number strictly between 0 and 1 (e.g. 0.95, 0.99)."""
    if (isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float))
            or not math.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError("confidence_level must be a number strictly between 0 and 1, "
                         f"got {confidence_level!r}")


def validate_min_observations(min_observations):
    """A whole number >= 2 (the parametric method needs a standard deviation)."""
    if (isinstance(min_observations, bool) or not isinstance(min_observations, (int, np.integer))
            or min_observations < 2):
        raise ValueError(f"min_observations must be a whole number >= 2, got {min_observations!r}")


def calculate_var_summary(data, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                          min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Historical and parametric 1-day VaR per symbol (sorted), on the same returns.

    Columns: symbol, observations, min_observations, sufficient_data,
    confidence_level, historical_var, parametric_var. The input is not modified.
    """
    require_canonical_columns(data, "VaR")
    validate_confidence_level(confidence_level)
    validate_min_observations(min_observations)
    alpha = 1 - confidence_level
    z_alpha = norm.ppf(alpha)

    rows = []
    for symbol, sample in returns_by_symbol(data).items():
        sufficient = len(sample) >= min_observations
        rows.append({
            "symbol": symbol,
            "observations": len(sample),
            "min_observations": min_observations,
            "sufficient_data": sufficient,
            "confidence_level": confidence_level,
            "historical_var": historical_var(sample, alpha) if sufficient else np.nan,
            "parametric_var": parametric_var(sample, alpha, z_alpha) if sufficient else np.nan,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).astype(
        {"observations": "int64", "min_observations": "int64", "sufficient_data": "bool",
         "confidence_level": "float64", "historical_var": "float64", "parametric_var": "float64"})


def historical_var(returns, alpha):
    """Historical VaR (positive loss) of one return sample: -quantile(r, alpha), linear."""
    return -np.quantile(np.asarray(returns, dtype="float64"), alpha, method=QUANTILE_METHOD)


def parametric_var(returns, alpha, z_alpha=None):
    """Parametric Normal VaR (positive loss) of one return sample: -(mean + z * std)."""
    sample = np.asarray(returns, dtype="float64")
    z = norm.ppf(alpha) if z_alpha is None else z_alpha
    return -(sample.mean() + z * sample.std(ddof=1))


def returns_by_symbol(data):
    """{symbol: array of usable daily returns}, every input symbol included (sorted).

    The single source of the return sample for VaR and CVaR, so both always use
    exactly the same observations: finite returns from calculate_daily_returns.
    """
    returns = calculate_daily_returns(data)
    usable = returns[np.isfinite(returns[CANONICAL_DAILY_RETURN])]
    by_symbol = {s: g[CANONICAL_DAILY_RETURN].to_numpy() for s, g in usable.groupby("symbol")}
    return {s: by_symbol.get(s, np.array([])) for s in sorted(returns["symbol"].dropna().unique())}


def calculate_historical_var(data, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                             min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Historical 1-day VaR per symbol (HISTORICAL_COLUMNS)."""
    return calculate_var_summary(data, confidence_level, min_observations)[HISTORICAL_COLUMNS]


def calculate_parametric_var(data, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                             min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Parametric (Normal) 1-day VaR per symbol (PARAMETRIC_COLUMNS)."""
    return calculate_var_summary(data, confidence_level, min_observations)[PARAMETRIC_COLUMNS]
