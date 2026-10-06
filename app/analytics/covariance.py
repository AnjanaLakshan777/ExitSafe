"""Covariance and correlation of daily returns between stocks.

Only dates where every stock has a return over the same period are used, so a
two-day return after a gap is never paired with a one-day return.
"""

import numpy as np
import pandas as pd

from app.analytics.common import (
    CONSTANT_RETURN_TOLERANCE,
    TRADING_DAYS_PER_YEAR,
    require_canonical_columns,
    validate_periods_per_year,
)
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns


def calculate_return_matrix(data):
    """Daily returns as a date x symbol table, with NaN where a stock has no return."""
    return _alignment(data, "a return matrix")[0]


def calculate_aligned_return_matrix(data):
    """The return matrix restricted to common observations (no missing values)."""
    return _alignment(data, "a return matrix")[1]


def describe_return_alignment(data):
    """Summarise how many dates the matrices use and why other dates were dropped."""
    return _alignment(data, "a return alignment")[2]


def calculate_covariance_matrix(data):
    """Daily sample covariance matrix (all NaN with fewer than two common dates)."""
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
    """Pearson correlation matrix; a stock with constant returns gets NaN."""
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
        # the date the first common return starts from (its previous close)
        "base_date": (pd.Timestamp(previous[common].iloc[0, 0]) if len(aligned) else None),
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
