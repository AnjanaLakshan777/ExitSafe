"""Stock-level 1-day Conditional Value at Risk (CVaR), also called Expected Shortfall (ES).

VaR marks where the loss tail begins; CVaR is the AVERAGE loss within that
tail, i.e. how severe the worst (1 - C) of outcomes are on average. It is not
the maximum possible loss.

Sign convention (same as VaR): a POSITIVE LOSS MAGNITUDE. 0.047 means an
average tail loss of 4.7%. If the tail of the sample contains gains, the value
can be smaller or even negative; it is reported as calculated, never clamped.

For confidence C, alpha = 1 - C, and one symbol's daily returns r_1..r_n:

Historical CVaR (empirical expected shortfall with fractional tail weighting)
  1. losses l_i = -r_i, sorted from largest to smallest: d_1 >= d_2 >= ... >= d_n
  2. tail mass m = alpha * n   (rounded to 9 decimals, removing the
                                floating-point noise of 1 - C, e.g. 1.0000000000000009)
  3. k = floor(m), f = m - k
  4. CVaR = (d_1 + ... + d_k + f * d_{k+1}) / m
  Each observation carries probability 1/n, so this averages exactly the worst
  alpha of the probability mass: e.g. n = 30 at 95% gives m = 1.5, i.e. the worst
  loss with weight 1 and the second worst with weight 0.5, divided by 1.5. When
  m < 1 (e.g. 99% with 20 returns) the result is the worst loss.
  With historical VaR's linear (type-7) quantile this always gives
  historical CVaR >= historical VaR.

Parametric CVaR (Normal, same mean/sd/confidence as parametric VaR)
  mu = mean(r), sigma = std(r, ddof=1), z = norm.ppf(alpha)
  ES_return = mu - sigma * norm.pdf(z) / alpha       (lower-tail conditional mean)
  CVaR      = -ES_return = -(mu - sigma * norm.pdf(z) / alpha)
  Since pdf(z) / alpha > -z for a Normal lower tail, parametric CVaR >= parametric
  VaR; with sigma = 0 both equal -mu.

Data, confidence validation, minimum observations (default 20) and the return
sample are shared with app.analytics.var, so VaR and CVaR always use exactly the
same returns. Below the minimum both CVaRs are NaN (never 0). Values are full
precision; rounding belongs to presentation.
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
    """Empirical expected shortfall (positive loss) of one return sample.

    Averages the worst ``alpha`` fraction of probability mass, weighting the
    boundary observation fractionally (see module docstring).
    """
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
    """Historical and parametric 1-day VaR and CVaR per symbol (sorted).

    Columns: SUMMARY_COLUMNS. ``tail_mass`` is alpha * observations, the number
    of observations' worth of probability averaged by historical CVaR.
    """
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
