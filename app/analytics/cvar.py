"""Stock-level 1-day CVaR (expected shortfall), historical and parametric.

CVaR is the average loss beyond the VaR level, reported as a positive number.
It uses exactly the same returns as the VaR module.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.analytics.common import require_canonical_columns
from app.analytics.var import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MIN_OBSERVATIONS,
    calculate_var_summary,
    returns_by_symbol,
)

TAIL_MASS_DECIMALS = 9

SUMMARY_COLUMNS = ["symbol", "observations", "min_observations", "sufficient_data",
                   "confidence_level", "tail_mass", "historical_var", "historical_cvar",
                   "parametric_var", "parametric_cvar"]
HISTORICAL_COLUMNS = ["symbol", "observations", "confidence_level", "historical_cvar"]
PARAMETRIC_COLUMNS = ["symbol", "observations", "confidence_level", "parametric_cvar"]


def historical_expected_shortfall(returns, alpha):
    """Average of the worst alpha share of returns, as a positive loss."""
    losses = np.sort(-np.asarray(returns, dtype="float64"))[::-1]
    tail_mass = round(alpha * len(losses), TAIL_MASS_DECIMALS)
    whole = math.floor(tail_mass)
    fraction = tail_mass - whole
    total = losses[:whole].sum() + (fraction * losses[whole] if fraction > 0 else 0.0)
    return total / tail_mass


def parametric_expected_shortfall(returns, alpha):
    """Normal expected shortfall (positive loss): -(mu - sigma * pdf(z_alpha) / alpha)."""
    sample = np.asarray(returns, dtype="float64")
    return -(sample.mean() - sample.std(ddof=1) * norm.pdf(norm.ppf(alpha)) / alpha)


def calculate_cvar_summary(data, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                           min_observations=DEFAULT_MIN_OBSERVATIONS):
    """VaR and CVaR for each stock, historical and parametric."""
    require_canonical_columns(data, "CVaR")
    summary = calculate_var_summary(data, confidence_level, min_observations)   # validates inputs
    alpha = 1 - confidence_level
    samples = returns_by_symbol(data)

    historical, parametric, tail_mass = [], [], []
    for row in summary.itertuples():
        sample = samples[row.symbol]
        tail_mass.append(round(alpha * len(sample), TAIL_MASS_DECIMALS))
        historical.append(historical_expected_shortfall(sample, alpha)
                          if row.sufficient_data else np.nan)
        parametric.append(parametric_expected_shortfall(sample, alpha)
                          if row.sufficient_data else np.nan)

    summary["tail_mass"] = pd.Series(tail_mass, index=summary.index, dtype="float64")
    summary["historical_cvar"] = pd.Series(historical, index=summary.index, dtype="float64")
    summary["parametric_cvar"] = pd.Series(parametric, index=summary.index, dtype="float64")
    return summary[SUMMARY_COLUMNS]


def calculate_historical_cvar(data, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                              min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Historical 1-day CVaR per symbol (HISTORICAL_COLUMNS)."""
    return calculate_cvar_summary(data, confidence_level, min_observations)[HISTORICAL_COLUMNS]


def calculate_parametric_cvar(data, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                              min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Parametric (Normal) 1-day CVaR per symbol (PARAMETRIC_COLUMNS)."""
    return calculate_cvar_summary(data, confidence_level, min_observations)[PARAMETRIC_COLUMNS]
