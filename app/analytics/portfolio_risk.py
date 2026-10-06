"""Portfolio risk for user-supplied, fixed weights (no optimization).

Portfolio risk considers how the holdings behave TOGETHER, so stock-level
measures are never weighted and added: volatility comes from the covariance
matrix, and drawdown, VaR and CVaR come from the portfolio's own return series.

Weights
  {symbol: weight}, a DataFrame with ``symbol`` and ``weight`` columns, or
  (symbol, weight) pairs. Every weight must be finite and >= 0 (no short
  selling), symbols must be unique and present in the data, and the weights
  must sum to 1 within WEIGHT_SUM_TOLERANCE. Nothing is normalized.

Common-date alignment
  Returns come from app.analytics.covariance.calculate_aligned_return_matrix:
  a date is used only if every holding (with weight > 0) has a usable return
  covering the same period. INVALID rows are never bridged; missing returns are
  never filled with zero. Zero-weight holdings are listed but do not restrict
  the dates.

Formulas (P = periods_per_year, default 252; w = weights; R_i(t) daily returns)
  portfolio return        R_p(t) = sum_i w_i * R_i(t)
  cumulative return       prod(1 + R_p) - 1
  annualized arithmetic   mean(R_p) * P
  annualized geometric    (1 + cumulative) ** (P / n) - 1
  daily variance          w' Sigma_daily w        (calculate_covariance_matrix)
  daily volatility        sqrt(w' Sigma w);  annualized: sqrt(w' Sigma w * P)
  drawdown                value path starting at 1.0 on the base date (the close
                          before the first common return), V(t) = V(t-1) * (1 + R_p(t));
                          same peak/trough/recovery rules as stock drawdown
  VaR / CVaR              historical and parametric, from the R_p sample itself,
                          same conventions as the stock-level modules
  concentration           maximum weight; HHI = sum(w_i ** 2)

Statistics need at least 2 common observations (otherwise NaN); VaR/CVaR need
``min_observations`` (default 20). These are historical, sample-based figures
for a portfolio held at constant weights over the period, not forecasts.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analytics.common import (
    TRADING_DAYS_PER_YEAR,
    require_canonical_columns,
    validate_periods_per_year,
)
from app.analytics.covariance import (
    calculate_aligned_return_matrix,
    calculate_covariance_matrix,
    describe_return_alignment,
)
from app.analytics.cvar import historical_expected_shortfall, parametric_expected_shortfall
from app.analytics.drawdown import drawdown_path, maximum_drawdown_event
from app.analytics.liquidity import (
    DEFAULT_PARTICIPATION_RATE,
    calculate_position_liquidity,
    validate_participation_rate,
    validate_position_value,
)
from app.analytics.var import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MIN_OBSERVATIONS,
    historical_var,
    parametric_var,
    validate_confidence_level,
    validate_min_observations,
)

WEIGHT_SUM_TOLERANCE = 1e-6
MIN_STATISTIC_OBSERVATIONS = 2
RETURN_SERIES_COLUMNS = ["date", "portfolio_return"]
VALUE_PATH_COLUMNS = ["date", "value", "running_peak", "drawdown"]
HOLDING_LIQUIDITY_COLUMNS = ["position_value", "average_daily_traded_value",
                             "traded_value_source", "position_to_adtv",
                             "daily_executable_value", "estimated_liquidation_days"]


@dataclass(frozen=True)
class PortfolioRiskResult:
    weights: dict
    return_series: pd.DataFrame          # date, portfolio_return
    value_path: pd.DataFrame             # date, value, running_peak, drawdown (from 1.0)
    covariance: pd.DataFrame             # daily covariance of holdings with weight > 0
    observations: int
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    base_date: pd.Timestamp | None
    excluded_missing: int
    excluded_misaligned: int
    periods_per_year: float
    mean_daily_return: float
    annualized_arithmetic_return: float
    cumulative_return: float
    annualized_geometric_return: float
    daily_variance: float
    daily_volatility: float
    annualized_volatility: float
    maximum_drawdown: float
    peak_date: pd.Timestamp | None
    peak_value: float
    trough_date: pd.Timestamp | None
    trough_value: float
    recovery_date: pd.Timestamp | None
    confidence_level: float
    min_observations: int
    sufficient_tail_data: bool
    historical_var: float
    parametric_var: float
    historical_cvar: float
    parametric_cvar: float
    maximum_weight: float
    hhi: float
    holdings: pd.DataFrame               # symbol, weight (+ liquidity if portfolio_value)
    portfolio_value: float | None = None
    participation_rate: float | None = None
    most_illiquid_symbol: str | None = None
    maximum_estimated_liquidation_days: float = np.nan
    maximum_position_to_adtv: float = np.nan
    undefined_liquidation_symbols: tuple = ()


def validate_weights(weights, available_symbols=None):
    """Validated weights as {symbol: weight} sorted by symbol. Never normalizes."""
    pairs = _weight_pairs(weights)
    if not pairs:
        raise ValueError("The portfolio has no holdings")
    symbols = [s.strip() if isinstance(s, str) else s for s, _ in pairs]
    if any(not isinstance(s, str) or not s for s in symbols):
        raise ValueError("Every holding needs a non-empty symbol")
    duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
    if duplicates:
        raise ValueError(f"Duplicate symbol(s) in weights: {', '.join(duplicates)}")

    validated = {}
    for symbol, (_, weight) in zip(symbols, pairs):
        if (isinstance(weight, bool) or not isinstance(weight, (int, float, np.number))
                or not math.isfinite(weight)):
            raise ValueError(f"Weight for {symbol} must be a finite number, got {weight!r}")
        if weight < 0:
            raise ValueError(f"Weight for {symbol} is negative ({weight}); short selling is "
                             "not supported")
        validated[symbol] = float(weight)
    total = sum(validated.values())
    if abs(total - 1) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Weights must sum to 1 (100%), got {total:.6g}; they are not "
                         "normalized automatically")
    if available_symbols is not None:
        missing = sorted(set(validated) - set(available_symbols))
        if missing:
            raise ValueError(f"Symbol(s) not found in the market data: {', '.join(missing)}")
    return dict(sorted(validated.items()))


def calculate_portfolio_return_series(data, weights):
    """Portfolio daily returns on common dates: columns date, portfolio_return."""
    return _portfolio_returns(data, weights)[0]


def calculate_portfolio_risk_summary(data, weights, portfolio_value=None,
                                     confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                                     min_observations=DEFAULT_MIN_OBSERVATIONS,
                                     periods_per_year=TRADING_DAYS_PER_YEAR,
                                     participation_rate=DEFAULT_PARTICIPATION_RATE):
    """Portfolio-level risk view for fixed weights. Returns a PortfolioRiskResult."""
    validate_confidence_level(confidence_level)
    validate_min_observations(min_observations)
    validate_periods_per_year(periods_per_year)
    if portfolio_value is not None:
        validate_position_value(portfolio_value)
        validate_participation_rate(participation_rate)

    series, weights, active_data, info = _portfolio_returns(data, weights)
    r = series["portfolio_return"].to_numpy()
    n = len(r)
    enough = n >= MIN_STATISTIC_OBSERVATIONS
    active = [s for s, w in weights.items() if w > 0]
    w = np.array([weights[s] for s in active])

    covariance = calculate_covariance_matrix(active_data).reindex(index=active, columns=active)
    variance = max(float(w @ covariance.to_numpy() @ w), 0.0) if enough else np.nan
    cumulative = float(np.prod(1 + r) - 1) if enough else np.nan

    path = (drawdown_path([info["base_date"], *series["date"]], np.concatenate([[1.0], np.cumprod(1 + r)]))
            if n else pd.DataFrame(columns=VALUE_PATH_COLUMNS))
    event = maximum_drawdown_event(path) if enough else maximum_drawdown_event(path.iloc[0:0])

    alpha = 1 - confidence_level
    tail_ok = n >= min_observations

    result = dict(
        weights=weights, return_series=series, value_path=path, covariance=covariance,
        observations=n, start_date=info["start_date"], end_date=info["end_date"],
        base_date=info["base_date"], excluded_missing=info["excluded_missing"],
        excluded_misaligned=info["excluded_misaligned"], periods_per_year=periods_per_year,
        mean_daily_return=float(r.mean()) if enough else np.nan,
        annualized_arithmetic_return=float(r.mean() * periods_per_year) if enough else np.nan,
        cumulative_return=cumulative,
        annualized_geometric_return=((1 + cumulative) ** (periods_per_year / n) - 1
                                     if enough else np.nan),
        daily_variance=variance,
        daily_volatility=math.sqrt(variance) if enough else np.nan,
        annualized_volatility=math.sqrt(variance * periods_per_year) if enough else np.nan,
        maximum_drawdown=event["maximum_drawdown"], peak_date=event["peak_date"],
        peak_value=event["peak_value"], trough_date=event["trough_date"],
        trough_value=event["trough_value"], recovery_date=event["recovery_date"],
        confidence_level=confidence_level, min_observations=min_observations,
        sufficient_tail_data=tail_ok,
        historical_var=historical_var(r, alpha) if tail_ok else np.nan,
        parametric_var=parametric_var(r, alpha) if tail_ok else np.nan,
        historical_cvar=historical_expected_shortfall(r, alpha) if tail_ok else np.nan,
        parametric_cvar=parametric_expected_shortfall(r, alpha) if tail_ok else np.nan,
        maximum_weight=max(weights.values()),
        hhi=sum(x * x for x in weights.values()),
        holdings=pd.DataFrame({"symbol": list(weights), "weight": list(weights.values())}),
    )
    if portfolio_value is not None:
        result.update(_holdings_liquidity(data, weights, float(portfolio_value), participation_rate))
    return PortfolioRiskResult(**result)


def _portfolio_returns(data, weights):
    require_canonical_columns(data, "portfolio risk")
    weights = validate_weights(weights, set(data["symbol"].dropna()))
    active = [s for s, w in weights.items() if w > 0]
    active_data = data[data["symbol"].isin(active)]
    aligned = calculate_aligned_return_matrix(active_data).reindex(columns=active)
    info = describe_return_alignment(active_data)
    values = aligned.to_numpy() @ np.array([weights[s] for s in active])
    series = pd.DataFrame({"date": aligned.index, "portfolio_return": values},
                          columns=RETURN_SERIES_COLUMNS).reset_index(drop=True)
    return series, weights, active_data, info


def _holdings_liquidity(data, weights, portfolio_value, participation_rate):
    rows = []
    for symbol, weight in weights.items():
        holding = data[data["symbol"] == symbol]
        position = portfolio_value * weight
        # Existing stock-level formulas. A zero weight has nothing to sell: 0 when
        # ADTV is defined, NaN when it is not (as for any position).
        liquidity = calculate_position_liquidity(holding, position if weight > 0 else 1.0,
                                                 participation_rate).iloc[0]
        ratio, days = liquidity["position_to_adtv"], liquidity["estimated_liquidation_days"]
        if weight == 0:
            ratio = days = 0.0 if np.isfinite(ratio) else np.nan
        rows.append({"symbol": symbol, "weight": weight, "position_value": position,
                     "average_daily_traded_value": liquidity["average_daily_traded_value"],
                     "traded_value_source": liquidity["traded_value_source"],
                     "position_to_adtv": ratio,
                     "daily_executable_value": liquidity["daily_executable_value"],
                     "estimated_liquidation_days": days})
    holdings = pd.DataFrame(rows, columns=["symbol", "weight", *HOLDING_LIQUIDITY_COLUMNS])
    held = holdings[holdings["weight"] > 0]
    days = held["estimated_liquidation_days"]
    return {
        "holdings": holdings,
        "portfolio_value": portfolio_value,
        "participation_rate": participation_rate,
        "most_illiquid_symbol": (held.loc[days.idxmax(), "symbol"] if days.notna().any() else None),
        "maximum_estimated_liquidation_days": days.max() if days.notna().any() else np.nan,
        "maximum_position_to_adtv": (held["position_to_adtv"].max()
                                     if held["position_to_adtv"].notna().any() else np.nan),
        "undefined_liquidation_symbols": tuple(held.loc[days.isna(), "symbol"]),
    }


def _weight_pairs(weights):
    if isinstance(weights, pd.DataFrame):
        if not {"symbol", "weight"} <= set(weights.columns):
            raise ValueError("A weights table needs 'symbol' and 'weight' columns")
        return list(zip(weights["symbol"], weights["weight"]))
    if isinstance(weights, Mapping):
        return list(weights.items())
    try:
        return [(symbol, weight) for symbol, weight in weights]
    except (TypeError, ValueError):
        raise ValueError("weights must be a mapping, a symbol/weight table or "
                         "(symbol, weight) pairs") from None
