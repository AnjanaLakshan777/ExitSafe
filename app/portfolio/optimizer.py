"""Risk-aware portfolio optimisation with CVXPY (long-only, fully invested).

It minimises a weighted mix of variance, CVaR and negative expected return,
within weight limits and an optional liquidity limit. The CVaR term is built
from the portfolio's own historical returns.
"""

import math
from dataclasses import dataclass

import cvxpy as cp
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
from app.analytics.cvar import historical_expected_shortfall
from app.analytics.liquidity import calculate_liquidity_summary, validate_position_value
from app.analytics.portfolio_risk import WEIGHT_SUM_TOLERANCE
from app.analytics.var import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MIN_OBSERVATIONS,
    historical_var,
    validate_confidence_level,
    validate_min_observations,
)

SOLVER = cp.CLARABEL
# Tighter than the solver defaults, because daily-return numbers are small.
SOLVER_OPTIONS = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}
WEIGHT_TOLERANCE = WEIGHT_SUM_TOLERANCE          # 1e-6
OBJECTIVE_TOLERANCE = 1e-6                       # allowed gap when re-checking the objective
DEFAULT_RISK_AVERSION = 1.0
DEFAULT_CVAR_WEIGHT = 1.0
DEFAULT_RETURN_WEIGHT = 1.0

LIQUIDITY_NOT_APPLIED = "NOT_APPLIED"            # constraint disabled
LIQUIDITY_WITHIN_LIMIT = "WITHIN_LIMIT"
LIQUIDITY_AT_LIMIT = "AT_LIMIT"                  # the limit is binding
LIQUIDITY_NO_DATA = "NO_LIQUIDITY_DATA"          # no ADTV, so no limit applied
LIQUIDITY_COLUMNS = ["symbol", "weight", "position_value", "average_daily_traded_value",
                     "traded_value_source", "position_to_adtv", "liquidity_weight_cap",
                     "liquidity_constraint_status"]


class OptimizationError(ValueError):
    """The optimization could not produce a valid allocation."""


class InfeasibleConstraintsError(OptimizationError):
    """No allocation satisfies the constraints."""


class InsufficientObservationsError(ValueError):
    """Too few common return scenarios for the selected universe."""


@dataclass(frozen=True)
class OptimizationScenarios:
    symbols: list
    returns: pd.DataFrame                 # date x symbol daily returns, common dates only
    expected_daily_returns: pd.Series     # mean daily return per stock
    covariance: pd.DataFrame              # daily sample covariance (ddof = 1)
    observations: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    excluded_missing: int
    excluded_misaligned: int


@dataclass(frozen=True)
class PortfolioMetrics:
    weights: dict
    expected_daily_return: float
    expected_annual_return: float         # daily mean x periods per year
    daily_variance: float
    daily_volatility: float
    annualized_volatility: float
    historical_var: float                 # 1-day, as a positive loss
    historical_cvar: float
    hhi: float
    max_weight: float


@dataclass(frozen=True)
class OptimizationResult:
    optimized_weights: dict
    metrics: PortfolioMetrics
    equal_weight_metrics: PortfolioMetrics
    expected_annual_return: float
    daily_volatility: float
    annualized_volatility: float
    historical_var: float
    historical_cvar: float
    hhi: float
    max_weight: float
    objective_value: float
    variance_term: float                  # the three objective parts at the optimum
    cvar_term: float
    return_term: float
    solver: str
    solver_status: str
    observations: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    excluded_missing: int
    excluded_misaligned: int
    symbols: list
    risk_aversion: float
    cvar_weight: float
    return_weight: float
    confidence_level: float
    periods_per_year: float
    min_weight_limit: float
    max_weight_limit: float
    portfolio_value: float | None
    liquidity_constraint_enabled: bool
    max_position_to_adtv: float | None
    liquidity: pd.DataFrame | None        # LIQUIDITY_COLUMNS, when portfolio_value is given
    liquidity_unconstrained_symbols: tuple


def equal_weight_portfolio(symbols):
    """{symbol: 1 / n} for the given symbols (validated, sorted)."""
    symbols = validate_symbols(symbols)
    return {s: 1 / len(symbols) for s in symbols}


def validate_portfolio_weights(weights, symbols=None, min_weight=0.0, max_weight=1.0,
                               tolerance=WEIGHT_TOLERANCE):
    """Check that weights are finite, within the limits and sum to 1. Nothing is rescaled."""
    weights = {s: float(w) for s, w in dict(weights).items()}
    if symbols is not None and set(weights) != set(symbols):
        raise OptimizationError("The allocation does not cover exactly the selected symbols")
    values = np.array(list(weights.values()), dtype="float64")
    if not values.size or not np.isfinite(values).all():
        raise OptimizationError("The allocation contains non-finite weights")
    if (values < min_weight - tolerance).any():
        raise OptimizationError(f"A weight is below the minimum weight {min_weight:g}")
    if (values > max_weight + tolerance).any():
        raise OptimizationError(f"A weight is above the maximum weight {max_weight:g}")
    if abs(values.sum() - 1) > tolerance:
        raise OptimizationError(f"The weights sum to {values.sum():.9g}, not 1")
    return dict(sorted(weights.items()))


def build_optimization_scenarios(data, symbols, min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Common-date return scenarios, expected daily returns and covariance for a universe."""
    require_canonical_columns(data, "portfolio optimization")
    validate_min_observations(min_observations)
    symbols = validate_symbols(symbols)
    missing = sorted(set(symbols) - set(data["symbol"].dropna()))
    if missing:
        raise ValueError(f"Selected symbol(s) not found in the market data: {', '.join(missing)}")

    universe = data[data["symbol"].isin(symbols)]
    returns = calculate_aligned_return_matrix(universe).reindex(columns=symbols)
    info = describe_return_alignment(universe)
    if len(returns) < min_observations:
        raise InsufficientObservationsError(
            f"Only {len(returns)} common daily return observation(s) for {', '.join(symbols)}; "
            f"at least {min_observations} are required ({info['excluded_missing']} date(s) "
            f"without a return for every stock, {info['excluded_misaligned']} covering "
            "different periods). Select fewer stocks, provide more data or lower Minimum "
            "Observations.")
    covariance = calculate_covariance_matrix(universe).reindex(index=symbols, columns=symbols)
    return OptimizationScenarios(
        symbols=symbols, returns=returns, expected_daily_returns=returns.mean(),
        covariance=covariance, observations=len(returns),
        start_date=info["start_date"], end_date=info["end_date"],
        excluded_missing=info["excluded_missing"],
        excluded_misaligned=info["excluded_misaligned"])


def portfolio_metrics(scenarios, weights, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                      periods_per_year=TRADING_DAYS_PER_YEAR):
    """Risk/return of fixed weights on the given scenarios (same dates as the optimizer)."""
    validate_confidence_level(confidence_level)
    validate_periods_per_year(periods_per_year)
    w = np.array([weights[s] for s in scenarios.symbols], dtype="float64")
    series = scenarios.returns.to_numpy() @ w
    alpha = 1 - confidence_level
    variance = max(float(w @ scenarios.covariance.to_numpy() @ w), 0.0)
    daily_mean = float(scenarios.expected_daily_returns.to_numpy() @ w)
    return PortfolioMetrics(
        weights=dict(zip(scenarios.symbols, w.tolist())),
        expected_daily_return=daily_mean,
        expected_annual_return=daily_mean * periods_per_year,
        daily_variance=variance,
        daily_volatility=math.sqrt(variance),
        annualized_volatility=math.sqrt(variance * periods_per_year),
        historical_var=float(historical_var(series, alpha)),
        historical_cvar=float(historical_expected_shortfall(series, alpha)),
        hhi=float(np.sum(w ** 2)),
        max_weight=float(w.max()))


def optimize_portfolio(data, symbols,
                       risk_aversion=DEFAULT_RISK_AVERSION,
                       cvar_weight=DEFAULT_CVAR_WEIGHT,
                       return_weight=DEFAULT_RETURN_WEIGHT,
                       confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                       min_observations=DEFAULT_MIN_OBSERVATIONS,
                       periods_per_year=TRADING_DAYS_PER_YEAR,
                       min_weight=0.0, max_weight=1.0,
                       portfolio_value=None,
                       liquidity_constraint_enabled=False,
                       max_position_to_adtv=None):
    """Choose long-only, fully invested weights for the selected stocks."""
    coefficients = validate_objective_weights(risk_aversion, cvar_weight, return_weight)
    validate_confidence_level(confidence_level)
    validate_min_observations(min_observations)
    validate_periods_per_year(periods_per_year)
    validate_weight_bounds(min_weight, max_weight)
    if not isinstance(liquidity_constraint_enabled, bool):
        raise ValueError("liquidity_constraint_enabled must be True or False")
    if portfolio_value is not None:
        validate_position_value(portfolio_value)
    if liquidity_constraint_enabled:
        if portfolio_value is None:
            raise ValueError("The liquidity constraint needs a portfolio_value")
        _validate_positive(max_position_to_adtv, "max_position_to_adtv")
    elif max_position_to_adtv is not None:
        _validate_positive(max_position_to_adtv, "max_position_to_adtv")

    symbols = validate_symbols(symbols)
    n = len(symbols)
    stocks = "stock" if n == 1 else "stocks"
    if n * min_weight > 1 + 1e-12:
        raise InfeasibleConstraintsError(
            f"Infeasible constraints: {n} {stocks} x minimum weight {min_weight:.4g} = "
            f"{n * min_weight:.4g} > 100%.")
    if n * max_weight < 1 - 1e-12:
        raise InfeasibleConstraintsError(
            f"Infeasible constraints: {n} {stocks} x maximum weight {max_weight:.4g} = "
            f"{n * max_weight:.4g} < 100%, so the portfolio cannot be fully invested.")

    scenarios = build_optimization_scenarios(data, symbols, min_observations)
    adtv = (_adtv(data, symbols) if portfolio_value is not None else None)
    caps = (_liquidity_caps(adtv, symbols, float(portfolio_value), max_position_to_adtv,
                            min_weight, max_weight)
            if liquidity_constraint_enabled else None)

    weights, status, objective = _solve_problem(scenarios, coefficients, confidence_level,
                                                min_weight, max_weight, caps)
    metrics = portfolio_metrics(scenarios, weights, confidence_level, periods_per_year)
    terms = {"variance_term": metrics.daily_variance,
             "cvar_term": metrics.historical_cvar,
             "return_term": metrics.expected_daily_return}
    recomputed = (coefficients[0] * terms["variance_term"] + coefficients[1] * terms["cvar_term"]
                  - coefficients[2] * terms["return_term"])
    if not math.isfinite(objective) or abs(objective - recomputed) > OBJECTIVE_TOLERANCE:
        raise OptimizationError(
            f"The solver objective ({objective!r}) does not match the objective recomputed "
            f"from the weights ({recomputed!r}); the result is rejected.")

    liquidity = (_liquidity_table(weights, symbols, adtv, float(portfolio_value), caps,
                                  max_position_to_adtv, liquidity_constraint_enabled)
                 if portfolio_value is not None else None)
    unconstrained = (tuple(liquidity.loc[liquidity["liquidity_constraint_status"]
                                         == LIQUIDITY_NO_DATA, "symbol"])
                     if liquidity is not None else ())
    return OptimizationResult(
        optimized_weights=weights, metrics=metrics,
        equal_weight_metrics=portfolio_metrics(scenarios, equal_weight_portfolio(symbols),
                                               confidence_level, periods_per_year),
        expected_annual_return=metrics.expected_annual_return,
        daily_volatility=metrics.daily_volatility,
        annualized_volatility=metrics.annualized_volatility,
        historical_var=metrics.historical_var, historical_cvar=metrics.historical_cvar,
        hhi=metrics.hhi, max_weight=metrics.max_weight,
        objective_value=objective, **terms,
        solver=SOLVER, solver_status=status,
        observations=scenarios.observations, start_date=scenarios.start_date,
        end_date=scenarios.end_date, excluded_missing=scenarios.excluded_missing,
        excluded_misaligned=scenarios.excluded_misaligned, symbols=symbols,
        risk_aversion=coefficients[0], cvar_weight=coefficients[1],
        return_weight=coefficients[2], confidence_level=confidence_level,
        periods_per_year=periods_per_year, min_weight_limit=float(min_weight),
        max_weight_limit=float(max_weight),
        portfolio_value=None if portfolio_value is None else float(portfolio_value),
        liquidity_constraint_enabled=liquidity_constraint_enabled,
        max_position_to_adtv=(None if max_position_to_adtv is None
                              else float(max_position_to_adtv)),
        liquidity=liquidity, liquidity_unconstrained_symbols=unconstrained)


def _solve_problem(scenarios, coefficients, confidence_level, min_weight, max_weight, caps):
    risk_aversion, cvar_weight, return_weight = coefficients
    symbols = scenarios.symbols
    r = scenarios.returns.to_numpy()
    mu = scenarios.expected_daily_returns.to_numpy()
    sigma = scenarios.covariance.to_numpy()
    sigma = (sigma + sigma.T) / 2

    w = cp.Variable(len(symbols), name="weights")
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    objective = 0
    if risk_aversion > 0:
        objective += risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
    if cvar_weight > 0:
        alpha = 1 - confidence_level
        z = cp.Variable(name="var_threshold")
        u = cp.Variable(len(r), name="excess_loss")
        constraints += [u >= -r @ w - z, u >= 0]
        objective += cvar_weight * (z + cp.sum(u) / (alpha * len(r)))
    if return_weight > 0:
        objective -= return_weight * (mu @ w)
    if caps is not None:
        # position / ADTV <= k, written in weight terms so the solver stays well scaled.
        constrained = [i for i, s in enumerate(symbols) if math.isfinite(caps[s])]
        if constrained:
            constraints.append(w[constrained] <= caps.iloc[constrained].to_numpy())

    # Scaling the objective doesn't change the answer but improves solver precision.
    scale = 1 / max(risk_aversion * float(np.abs(np.diag(sigma)).max()),
                    cvar_weight * float(np.abs(r).max()),
                    return_weight * float(np.abs(mu).max()), 1e-12)
    problem = cp.Problem(cp.Minimize(scale * objective), constraints)
    status = _solve(problem)
    if status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        raise InfeasibleConstraintsError(
            "Infeasible constraints: the solver found no allocation that satisfies the "
            "weight" + (" and liquidity" if caps is not None else "") + " constraints.")
    if status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
        raise OptimizationError("The optimization problem is unbounded; no allocation is "
                                "reported.")
    if status != cp.OPTIMAL:
        raise OptimizationError(f"The solver did not reach an optimal solution (status: "
                                f"{status}); no allocation is reported.")

    values = np.full(len(symbols), np.nan) if w.value is None else np.asarray(w.value, float)
    upper = max_weight if caps is None else np.minimum(max_weight, caps.to_numpy())
    weights = validate_portfolio_weights(dict(zip(symbols, values)), symbols, min_weight,
                                         max_weight)
    values = np.array([weights[s] for s in symbols])
    if caps is not None and (values > caps.to_numpy() + WEIGHT_TOLERANCE).any():
        raise OptimizationError("The solver result breaks the liquidity constraint; the "
                                "result is rejected.")
    # Clip tiny solver noise (like -1e-10) back inside the bounds.
    values = np.clip(values, min_weight, upper)
    value = problem.value
    return dict(zip(symbols, values.tolist())), status, (float(value) / scale
                                                         if value is not None else math.nan)


def _solve(problem):
    """Run the solver and return its status string (separate so tests can simulate)."""
    try:
        problem.solve(solver=SOLVER, **SOLVER_OPTIONS)
    except cp.error.SolverError as exc:
        raise OptimizationError(f"The {SOLVER} solver failed: {exc}") from None
    return problem.status


def _adtv(data, symbols):
    """ADTV and its source for each symbol (NaN when there's no usable volume data)."""
    table = pd.DataFrame({"adtv": math.nan, "source": None}, index=symbols)
    if "volume" in data.columns:
        summary = calculate_liquidity_summary(data[data["symbol"].isin(symbols)])
        summary = summary.set_index("symbol").reindex(symbols)
        table["adtv"] = summary["average_daily_traded_value"].astype("float64")
        table["source"] = summary["traded_value_source"].astype(object).where(
            summary["traded_value_source"].notna(), None)
    return table


def _liquidity_caps(adtv, symbols, portfolio_value, k, min_weight, max_weight):
    """Highest weight per stock allowed by V * w <= k * ADTV (inf when not constrained)."""
    caps = (k * adtv["adtv"] / portfolio_value).fillna(math.inf)
    too_low = [s for s in symbols if caps[s] < min_weight - WEIGHT_TOLERANCE]
    if too_low:
        raise InfeasibleConstraintsError(
            "Infeasible liquidity constraint: at Position / ADTV <= "
            f"{k:g} and portfolio value {portfolio_value:,.2f}, "
            f"{', '.join(too_low)} cannot reach the minimum weight {min_weight:.4g}.")
    capacity = float(np.minimum(caps.to_numpy(), max_weight).sum())
    if capacity < 1 - WEIGHT_TOLERANCE:
        raise InfeasibleConstraintsError(
            "Infeasible liquidity constraint: at Position / ADTV <= "
            f"{k:g} and portfolio value {portfolio_value:,.2f}, the stocks can hold at most "
            f"{capacity:.2%} of the portfolio in total (with the maximum weight), so it "
            "cannot be fully invested. Raise the limit, lower the portfolio value or add "
            "more liquid stocks.")
    return caps


def _liquidity_table(weights, symbols, adtv, portfolio_value, caps, k, enabled):
    rows = []
    for s in symbols:
        position = portfolio_value * weights[s]
        a = adtv.loc[s, "adtv"]
        ratio = position / a if a > 0 else (0.0 if position == 0 and a == 0 else math.nan)
        if not enabled:
            status = LIQUIDITY_NOT_APPLIED
        elif math.isnan(a):
            status = LIQUIDITY_NO_DATA
        elif position >= k * a - WEIGHT_TOLERANCE * portfolio_value:
            status = LIQUIDITY_AT_LIMIT
        else:
            status = LIQUIDITY_WITHIN_LIMIT
        rows.append({"symbol": s, "weight": weights[s], "position_value": position,
                     "average_daily_traded_value": a, "traded_value_source": adtv.loc[s, "source"],
                     "position_to_adtv": ratio,
                     "liquidity_weight_cap": (float(caps[s]) if caps is not None
                                              and math.isfinite(caps[s]) else math.nan),
                     "liquidity_constraint_status": status})
    return pd.DataFrame(rows, columns=LIQUIDITY_COLUMNS)


def validate_symbols(symbols):
    if isinstance(symbols, str):
        raise ValueError("symbols must be a list of stock symbols, not a single string")
    try:
        cleaned = [s.strip() if isinstance(s, str) else s for s in symbols]
    except TypeError:
        raise ValueError("symbols must be a list of stock symbols") from None
    if not cleaned:
        raise ValueError("Select at least one stock for the optimization universe")
    if any(not isinstance(s, str) or not s for s in cleaned):
        raise ValueError("Every selected symbol must be a non-empty text value")
    duplicates = sorted({s for s in cleaned if cleaned.count(s) > 1})
    if duplicates:
        raise ValueError(f"Duplicate symbol(s) in the universe: {', '.join(duplicates)}")
    return sorted(cleaned)


def validate_objective_weights(risk_aversion, cvar_weight, return_weight):
    values = []
    for name, value in (("risk_aversion", risk_aversion), ("cvar_weight", cvar_weight),
                        ("return_weight", return_weight)):
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        values.append(float(value))
    if not any(v > 0 for v in values):
        raise ValueError("At least one of risk_aversion, cvar_weight and return_weight must "
                         "be positive")
    return tuple(values)


def validate_weight_bounds(min_weight, max_weight):
    for name, value in (("min_weight", min_weight), ("max_weight", max_weight)):
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be a number between 0 and 1, got {value!r}")
    if min_weight > max_weight:
        raise ValueError(f"min_weight ({min_weight:g}) is above max_weight ({max_weight:g})")


def _validate_positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(value) or not value > 0):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
