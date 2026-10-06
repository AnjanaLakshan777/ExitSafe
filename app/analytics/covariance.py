"""Covariance and correlation of daily returns across symbols.

Returns come from app.analytics.returns.calculate_daily_returns on canonical
market data (no second return definition), so INVALID rows never contribute
and are never bridged over.

Alignment (missing-data policy)
  A date is a *common observation* only if every symbol has a usable (finite)
  daily return on it AND all those returns start from the same previous date,
  i.e. they cover the same period. A symbol that skipped a day has a two-day
  return there; pairing it with another symbol's one-day return would mix
  periods, so such dates are excluded. Every matrix element is computed from
  the same set of common dates; nothing is filled in. ``describe_return_alignment``
  reports how many dates were used and why others were left out.

Formulas
  daily covariance      = sample covariance of daily returns (ddof = 1)
                        = sum((x - mean_x)(y - mean_y)) / (n - 1)
  annualized covariance = daily covariance * periods_per_year   (default 252;
                          NOT sqrt(252), which is for volatility)
  correlation           = Pearson correlation of daily returns (not annualized)

Insufficient data
  Fewer than 2 common observations gives an all-NaN matrix (never zeros). A
  symbol whose common returns are constant (range <= CONSTANT_RETURN_TOLERANCE)
  has NaN correlations, including its diagonal, because correlation is
  undefined without variation. Values are full precision; rounding belongs to
  presentation.
"""

import numpy as np
import pandas as pd

from app.analytics.common import (
    TRADING_DAYS_PER_YEAR,
    require_canonical_columns,
    validate_periods_per_year,
)
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns

# Return series varying by less than this are treated as constant for correlation.
CONSTANT_RETURN_TOLERANCE = 1e-12


def calculate_return_matrix(data):
    """Date x symbol matrix of usable daily returns, NaN where a symbol has none.

    Rows: every date on which at least one symbol has a usable return (sorted).
    Columns: every symbol in the input (sorted), even one with no usable returns.
    """
    return _alignment(data, "a return matrix")[0]


def calculate_aligned_return_matrix(data):
    """The return matrix restricted to common observations (no missing values)."""
    return _alignment(data, "a return matrix")[1]


def describe_return_alignment(data):
    """How much data the covariance/correlation matrices are based on.

    Returns a dict with:
      symbols              symbols in the matrices (sorted)
      observations         number of common observations used
      start_date, end_date first and last common observation (None if none)
      returns_per_symbol   usable daily returns available per symbol
      candidate_dates      dates on which at least one symbol has a usable return
      excluded_missing     candidate dates left out because a symbol had no return
      excluded_misaligned  candidate dates left out because returns covered
                           different periods (different previous dates)
    """
    return _alignment(data, "a return alignment")[2]


def calculate_covariance_matrix(data):
    """Square symbol x symbol matrix of daily sample covariances (ddof = 1).

    Diagonal = each symbol's return variance over the common observations.
    """
    aligned = _alignment(data, "covariance")[1]
    symbols = list(aligned.columns)
    if len(aligned) < 2:
        return _nan_matrix(symbols)
    return _square(aligned.cov(ddof=1), symbols)


def calculate_annualized_covariance_matrix(data, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Daily covariance matrix * periods_per_year (default 252, not its square root)."""
    validate_periods_per_year(periods_per_year)
    return calculate_covariance_matrix(data) * periods_per_year


def calculate_correlation_matrix(data):
    """Square symbol x symbol matrix of Pearson correlations of daily returns.

    Values lie in [-1, 1] (floating-point overshoot is clipped). The diagonal
    is 1 for symbols with varying returns and NaN for constant ones.
    """
    aligned = _alignment(data, "correlation")[1]
    symbols = list(aligned.columns)
    if len(aligned) < 2:
        return _nan_matrix(symbols)

    correlation = _square(aligned.corr(method="pearson"), symbols)
    constant = (aligned.max() - aligned.min()) <= CONSTANT_RETURN_TOLERANCE
    correlation.loc[constant, :] = np.nan
    correlation.loc[:, constant] = np.nan
    correlation = correlation.clip(lower=-1.0, upper=1.0)
    for symbol in constant.index[~constant]:
        correlation.loc[symbol, symbol] = 1.0
    return correlation


def _alignment(data, purpose):
    require_canonical_columns(data, purpose)
    returns = calculate_daily_returns(data)
    returns["previous_date"] = returns.groupby("symbol", sort=False)["date"].shift(1)
    symbols = sorted(returns["symbol"].dropna().unique())

    usable = returns[np.isfinite(returns[CANONICAL_DAILY_RETURN])
                     & returns["date"].notna() & returns["symbol"].notna()]
    duplicated = usable.duplicated(subset=["date", "symbol"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Cannot align returns: duplicate symbol/date rows "
            f"({', '.join(sorted(set(usable.loc[duplicated, 'symbol'].astype(str))))}). "
            "Validate the data so duplicates are marked INVALID.")

    matrix = _pivot(usable, CANONICAL_DAILY_RETURN, symbols)
    previous = _pivot(usable, "previous_date", symbols)
    complete = matrix.notna().all(axis=1) if symbols else pd.Series(False, index=matrix.index)
    same_period = previous.nunique(axis=1) == 1
    common = complete & same_period
    aligned = matrix[common]

    info = {
        "symbols": symbols,
        "observations": len(aligned),
        "start_date": aligned.index.min() if len(aligned) else None,
        "end_date": aligned.index.max() if len(aligned) else None,
        "returns_per_symbol": {s: int(n) for s, n in matrix.count().items()},
        "candidate_dates": len(matrix),
        "excluded_missing": int((~complete).sum()),
        "excluded_misaligned": int((complete & ~same_period).sum()),
    }
    return matrix, aligned, info


def _pivot(usable, values, symbols):
    if usable.empty:
        matrix = pd.DataFrame(columns=symbols, index=pd.DatetimeIndex([], name="date"), dtype=float)
    else:
        matrix = usable.pivot(index="date", columns="symbol", values=values)
    matrix = matrix.reindex(columns=symbols).sort_index()
    matrix.index.name, matrix.columns.name = "date", "symbol"
    return matrix


def _square(matrix, symbols):
    matrix = matrix.reindex(index=symbols, columns=symbols).astype("float64")
    matrix.index.name = matrix.columns.name = "symbol"
    return matrix


def _nan_matrix(symbols):
    return _square(pd.DataFrame(np.nan, index=symbols, columns=symbols), symbols)
