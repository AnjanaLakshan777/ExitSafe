"""Stock-level 1-day Value at Risk, historical and parametric.

VaR is reported as a positive loss: 0.03 means a 3% one-day loss threshold.
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
    """Historical and parametric VaR for each stock, from the same returns."""
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
    """Usable daily returns per stock, shared by VaR and CVaR so both use the same data."""
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
