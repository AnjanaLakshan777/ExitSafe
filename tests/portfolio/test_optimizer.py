"""Tests for app.portfolio.optimizer (risk-aware, long-only, fully invested; CVXPY).

Expected values never come from the optimizer itself:
  * returns are rebuilt in plain Python from the generated closes
  * mean / covariance from ``statistics``; expected shortfall in exact fractions
  * optimality is checked by comparing the optimizer's objective with an
    independent objective evaluated at the equal-weight portfolio, random feasible
    portfolios and a fine grid (the optimum can never be worse)
  * closed forms where they exist: two-asset minimum variance
    w_A = (var_B - cov_AB) / (var_A + var_B - 2 cov_AB), and return-only LPs
    (all weight to the highest mean, up to the caps)
Properties guaranteed by the mathematics (monotonicity in the CVaR weight) are
tested instead of arbitrary exact weights.
"""

import math
import statistics
from fractions import Fraction

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from app.analytics.cvar import calculate_cvar_summary
from app.analytics.liquidity import ACTUAL_TURNOVER, ESTIMATED_TRADED_VALUE
from app.portfolio import optimizer
from app.portfolio.optimizer import (
    LIQUIDITY_AT_LIMIT,
    LIQUIDITY_NO_DATA,
    LIQUIDITY_NOT_APPLIED,
    LIQUIDITY_WITHIN_LIMIT,
    InfeasibleConstraintsError,
    InsufficientObservationsError,
    OptimizationError,
    build_optimization_scenarios,
    equal_weight_portfolio,
    optimize_portfolio,
    validate_portfolio_weights,
)

TOL = 1e-7          # solver accuracy allowance on weights / objective comparisons

RA = [0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.015, -0.01, 0.005, -0.03, 0.012, -0.008,
      0.025, -0.015, 0.004, -0.022, 0.018, -0.006, 0.009, -0.04, 0.011, -0.003, 0.007, 0.02, -0.012]
RB = [-0.01, 0.02, -0.005, 0.01, 0.0, 0.03, -0.02, 0.015, -0.01, 0.02, -0.004, 0.006,
      -0.01, 0.012, 0.003, 0.01, -0.015, 0.008, -0.002, 0.025, -0.006, 0.004, -0.009, -0.01, 0.007]
RC = [0.005, 0.01, -0.02, 0.004, -0.012, 0.008, 0.002, -0.006, 0.015, -0.01, 0.003, 0.0,
      -0.004, 0.009, -0.007, 0.012, 0.001, -0.003, 0.006, -0.015, 0.004, 0.002, -0.001, 0.005, 0.003]
RD = [0.01, 0.004, -0.008, 0.012, -0.003, -0.01, 0.006, 0.002, -0.005, 0.009, -0.002, 0.007,
      -0.011, 0.005, 0.001, -0.004, 0.008, -0.006, 0.003, 0.002, -0.007, 0.01, -0.001, 0.004, 0.0]

# Important test: H = higher mean, higher volatility, crashes together with L;
# L = lower mean, small tail.
H_BLOCK = [0.012, -0.004, 0.010, -0.002, 0.011, -0.003, 0.009, 0.0, 0.008, -0.03]
L_BLOCK = [0.002, -0.001, 0.0015, -0.0005, 0.002, -0.001, 0.0015, -0.0005, 0.002, -0.003]
RH, RL = H_BLOCK * 4, L_BLOCK * 4


def market(symbol, returns, start=100.0, volume=1_000.0, turnover=None, status=None, dates=None):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    frame = pd.DataFrame({
        "date": dates if dates is not None else pd.bdate_range("2026-01-01", periods=len(closes)),
        "symbol": symbol, "close": closes,
        "volume": volume if np.ndim(volume) == 0 else [float(v) for v in volume]})
    if turnover is not None:
        frame["turnover"] = [float(t) if t is not None else np.nan for t in turnover]
    if status is not None:
        frame["validation_status"] = status
    return frame


def combine(*frames):
    return pd.concat(frames, ignore_index=True)


def simple_returns(frame):
    closes = frame["close"].tolist()
    return [closes[i + 1] / closes[i] - 1 for i in range(len(closes) - 1)]


def es_exact(returns, confidence):
    alpha = Fraction(1) - Fraction(str(confidence))
    losses = sorted((-Fraction(float(r)) for r in returns), reverse=True)
    m = alpha * len(losses)
    k = math.floor(m)
    total = sum(losses[:k], Fraction(0)) + ((m - k) * losses[k] if m > k else 0)
    return float(total / m)


def quantile_type7(values, p):
    x = sorted(values)
    h = (len(x) - 1) * p
    lo = math.floor(h)
    return x[lo] + (h - lo) * (x[min(lo + 1, len(x) - 1)] - x[lo])


def series_of(weights, returns):
    n = len(next(iter(returns.values())))
    return [sum(weights[s] * returns[s][t] for s in weights) for t in range(n)]


def variance_of(weights, returns):
    return sum(weights[a] * weights[b] * statistics.covariance(returns[a], returns[b])
               for a in weights for b in weights)


def objective_of(weights, returns, risk_aversion=1.0, cvar_weight=1.0, return_weight=1.0,
                 confidence=0.95):
    series = series_of(weights, returns)
    return (risk_aversion * variance_of(weights, returns)
            + cvar_weight * es_exact(series, confidence)
            - return_weight * statistics.mean(series))


def random_feasible(symbols, count, seed=11):
    rng = np.random.RandomState(seed)
    return [dict(zip(symbols, rng.dirichlet(np.ones(len(symbols))))) for _ in range(count)]


FRAMES = {"A": market("A", RA), "B": market("B", RB), "C": market("C", RC), "D": market("D", RD)}
RETURNS = {s: simple_returns(f) for s, f in FRAMES.items()}


def universe(*symbols):
    return combine(*(FRAMES[s] for s in symbols)), {s: RETURNS[s] for s in symbols}


def assert_valid(result, symbols, min_weight=0.0, max_weight=1.0):
    w = result.optimized_weights
    assert sorted(w) == sorted(symbols)
    assert all(math.isfinite(v) for v in w.values())
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= min_weight - 1e-6 for v in w.values())
    assert all(v <= max_weight + 1e-6 for v in w.values())
    assert result.solver_status == "optimal" and result.solver == "CLARABEL"


# 1-5. universes of 2, 3 and 4 stocks; weights valid and optimal ---------------------------------

@pytest.mark.parametrize("symbols", [("A", "B"), ("A", "B", "C"), ("A", "B", "C", "D")])
def test_optimization_works_and_beats_every_alternative(symbols):
    data, returns = universe(*symbols)
    result = optimize_portfolio(data, list(symbols))
    assert_valid(result, symbols)
    assert result.observations == 25
    best = objective_of(result.optimized_weights, returns)
    assert result.objective_value == pytest.approx(best, abs=1e-9)
    for alternative in [equal_weight_portfolio(symbols), *random_feasible(symbols, 25)]:
        assert best <= objective_of(alternative, returns) + 1e-9


def test_weights_sum_to_one_and_are_never_negative():
    data, _ = universe("A", "B", "C", "D")
    for kwargs in ({}, {"cvar_weight": 0}, {"risk_aversion": 50}, {"return_weight": 0}):
        w = optimize_portfolio(data, ["A", "B", "C", "D"], **kwargs).optimized_weights
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        assert min(w.values()) >= 0.0


def test_minimum_weight_is_respected():
    data, _ = universe("A", "B", "C", "D")
    result = optimize_portfolio(data, ["A", "B", "C", "D"], min_weight=0.15)
    assert_valid(result, "ABCD", min_weight=0.15)
    assert result.min_weight_limit == 0.15


def test_maximum_weight_is_respected():
    data, returns = universe("A", "B", "C", "D")
    # return-only would put everything in the best stock; the cap must hold
    result = optimize_portfolio(data, ["A", "B", "C", "D"], risk_aversion=0, cvar_weight=0,
                                return_weight=1, max_weight=0.3)
    assert_valid(result, "ABCD", max_weight=0.3)
    ranked = sorted(returns, key=lambda s: statistics.mean(returns[s]), reverse=True)
    expected = {ranked[0]: 0.3, ranked[1]: 0.3, ranked[2]: 0.3, ranked[3]: 0.1}
    assert result.optimized_weights == pytest.approx(expected, abs=TOL)
    assert result.max_weight == pytest.approx(0.3, abs=TOL)


# 8-13. constraint and input validation ---------------------------------------------------------

def test_infeasible_minimum_weights_fail():
    data, _ = universe("A", "B", "C")
    with pytest.raises(InfeasibleConstraintsError, match="3 stocks x minimum weight 0.4"):
        optimize_portfolio(data, ["A", "B", "C"], min_weight=0.4)


def test_infeasible_maximum_weights_fail():
    data, _ = universe("A", "B", "C")
    with pytest.raises(InfeasibleConstraintsError, match="cannot be fully invested"):
        optimize_portfolio(data, ["A", "B", "C"], max_weight=0.3)


def test_inconsistent_bounds_fail():
    data, _ = universe("A", "B")
    with pytest.raises(ValueError, match="above max_weight"):
        optimize_portfolio(data, ["A", "B"], min_weight=0.6, max_weight=0.5)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            optimize_portfolio(data, ["A", "B"], max_weight=bad)


def test_unknown_symbols_fail():
    data, _ = universe("A", "B")
    with pytest.raises(ValueError, match="not found in the market data: ZZZ"):
        optimize_portfolio(data, ["A", "ZZZ"])


def test_duplicate_symbols_fail():
    data, _ = universe("A", "B")
    with pytest.raises(ValueError, match="Duplicate symbol.*A"):
        optimize_portfolio(data, ["A", "B", " A"])


def test_empty_or_malformed_universe_fails():
    data, _ = universe("A", "B")
    for bad in ([], "AB", [None, "A"], ["", "A"]):
        with pytest.raises(ValueError):
            optimize_portfolio(data, bad)


@pytest.mark.parametrize("kwargs", [
    {"risk_aversion": float("nan")}, {"cvar_weight": np.nan}, {"return_weight": float("nan")},
    {"min_weight": float("nan")}, {"max_weight": float("nan")},
    {"confidence_level": float("nan")}, {"portfolio_value": float("nan")},
    {"portfolio_value": 1e6, "liquidity_constraint_enabled": True,
     "max_position_to_adtv": float("nan")},
    {"risk_aversion": None}, {"cvar_weight": "1"}, {"risk_aversion": True},
])
def test_nan_or_non_numeric_inputs_fail_safely(kwargs):
    data, _ = universe("A", "B")
    with pytest.raises(ValueError):
        optimize_portfolio(data, ["A", "B"], **kwargs)


def test_nan_prices_are_excluded_not_propagated():
    a = FRAMES["A"].copy()
    a.loc[6, "close"] = np.nan
    result = optimize_portfolio(combine(a, FRAMES["B"]), ["A", "B"])
    assert all(math.isfinite(v) for v in result.optimized_weights.values())
    assert result.observations == 23                 # no return into or out of the NaN day


@pytest.mark.parametrize("kwargs", [
    {"risk_aversion": float("inf")}, {"cvar_weight": -np.inf}, {"return_weight": np.inf},
    {"max_weight": np.inf}, {"periods_per_year": float("inf")}, {"portfolio_value": np.inf},
    {"portfolio_value": 1e6, "liquidity_constraint_enabled": True, "max_position_to_adtv": np.inf},
])
def test_infinite_parameters_fail(kwargs):
    data, _ = universe("A", "B")
    with pytest.raises(ValueError):
        optimize_portfolio(data, ["A", "B"], **kwargs)


# 14-18. data handling ---------------------------------------------------------------------------

def test_unsorted_market_data_gives_the_same_result():
    data, _ = universe("A", "B", "C")
    shuffled = data.sample(frac=1, random_state=5).reset_index(drop=True)
    a = optimize_portfolio(shuffled, ["C", "A", "B"])
    b = optimize_portfolio(data, ["A", "B", "C"])
    assert a.optimized_weights == pytest.approx(b.optimized_weights, abs=1e-9)
    assert a.historical_cvar == pytest.approx(b.historical_cvar, abs=1e-12)


def test_invalid_rows_are_excluded_and_do_not_create_bridged_returns():
    status = ["VALID"] * 26
    status[3] = "INVALID"
    a = market("A", RA, status=status)
    data = combine(a, market("B", RB, status=["VALID"] * 26))
    scenarios = build_optimization_scenarios(data, ["A", "B"])
    dates = pd.bdate_range("2026-01-01", periods=26)
    assert dates[3] not in scenarios.returns.index           # the INVALID day
    assert dates[4] not in scenarios.returns.index           # would bridge day 2 -> day 4
    assert scenarios.observations == 23
    kept = [t for t in range(25) if t + 1 not in (3, 4)]
    assert scenarios.expected_daily_returns["A"] == pytest.approx(
        statistics.mean(RETURNS["A"][t] for t in kept), rel=1e-12)
    result = optimize_portfolio(data, ["A", "B"])
    assert result.observations == 23


def test_common_date_scenario_alignment():
    gappy = FRAMES["B"].drop(index=[10]).reset_index(drop=True)
    data = combine(FRAMES["A"], gappy, FRAMES["C"])
    scenarios = build_optimization_scenarios(data, ["A", "B", "C"])
    dates = pd.bdate_range("2026-01-01", periods=26)
    assert dates[10] not in scenarios.returns.index           # B has no row
    assert dates[11] not in scenarios.returns.index           # B's return spans two days
    assert scenarios.observations == 23
    assert scenarios.excluded_missing == 1 and scenarios.excluded_misaligned == 1
    assert not scenarios.returns.isna().any().any()
    assert list(scenarios.returns.columns) == ["A", "B", "C"]


def test_insufficient_observations_fail_clearly():
    data = combine(market("A", RA[:10]), market("B", RB[:10]))
    with pytest.raises(InsufficientObservationsError, match="Only 10 common daily return.*at least 20"):
        optimize_portfolio(data, ["A", "B"])
    assert optimize_portfolio(data, ["A", "B"], min_observations=10).observations == 10


# 19-23. expected return, variance and CVaR ---------------------------------------------------------

def test_expected_return_vector_matches_independent_calculation():
    data, returns = universe("A", "B", "C")
    scenarios = build_optimization_scenarios(data, ["A", "B", "C"])
    for s in "ABC":
        assert scenarios.expected_daily_returns[s] == pytest.approx(statistics.mean(returns[s]),
                                                                    rel=1e-12)
    result = optimize_portfolio(data, ["A", "B", "C"], periods_per_year=52)
    w = result.optimized_weights
    assert result.expected_annual_return == pytest.approx(
        sum(w[s] * statistics.mean(returns[s]) for s in w) * 52, rel=1e-9)


def test_portfolio_variance_equals_w_sigma_w():
    data, returns = universe("A", "B", "C")
    result = optimize_portfolio(data, ["A", "B", "C"], risk_aversion=20)
    variance = variance_of(result.optimized_weights, returns)
    assert result.variance_term == pytest.approx(variance, rel=1e-9)
    assert result.daily_volatility == pytest.approx(math.sqrt(variance), rel=1e-9)
    assert result.annualized_volatility == pytest.approx(math.sqrt(variance * 252), rel=1e-9)


def test_optimizer_uses_covariance_not_weighted_volatility():
    data, returns = universe("A", "B")
    var_a, var_b = statistics.variance(returns["A"]), statistics.variance(returns["B"])
    cov = statistics.covariance(returns["A"], returns["B"])
    w_a = (var_b - cov) / (var_a + var_b - 2 * cov)          # closed-form minimum variance
    assert 0 < w_a < 1
    result = optimize_portfolio(data, ["A", "B"], risk_aversion=1, cvar_weight=0, return_weight=0)
    assert result.optimized_weights["A"] == pytest.approx(w_a, abs=1e-8)
    # a weighted average of volatilities is linear in w and would pick a corner instead
    assert 0.05 < result.optimized_weights["A"] < 0.95
    weighted = w_a * math.sqrt(var_a) + (1 - w_a) * math.sqrt(var_b)
    assert result.daily_volatility < weighted


def test_portfolio_cvar_comes_from_portfolio_scenarios():
    data, returns = universe("A", "B")
    result = optimize_portfolio(data, ["A", "B"], risk_aversion=0, cvar_weight=1, return_weight=0)
    series = series_of(result.optimized_weights, returns)
    assert result.historical_cvar == pytest.approx(es_exact(series, 0.95), rel=1e-9)
    assert result.cvar_term == result.historical_cvar
    # the linear (Rockafellar-Uryasev) objective equals the scenario expected shortfall
    assert result.objective_value == pytest.approx(es_exact(series, 0.95), abs=1e-9)
    assert result.historical_var == pytest.approx(-quantile_type7(series, 0.05), rel=1e-9)
    # and it is the minimum over a fine grid of long-only two-stock portfolios
    grid = min(es_exact(series_of({"A": x, "B": 1 - x}, returns), 0.95)
               for x in np.linspace(0, 1, 401))
    assert result.historical_cvar <= grid + 1e-9


def test_portfolio_cvar_is_not_weighted_average_of_stock_cvars():
    data, returns = universe("A", "B")
    result = optimize_portfolio(data, ["A", "B"], risk_aversion=0, cvar_weight=1, return_weight=0)
    stock = calculate_cvar_summary(data).set_index("symbol")["historical_cvar"]
    w = result.optimized_weights
    weighted = w["A"] * stock["A"] + w["B"] * stock["B"]
    assert result.historical_cvar < weighted - 1e-4          # diversification in the tail


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_confidence_levels(confidence):
    data, returns = universe("A", "B", "C")
    result = optimize_portfolio(data, ["A", "B", "C"], confidence_level=confidence)
    assert_valid(result, "ABC")
    assert result.confidence_level == confidence
    series = series_of(result.optimized_weights, returns)
    assert result.historical_cvar == pytest.approx(es_exact(series, confidence), rel=1e-9)
    best = objective_of(result.optimized_weights, returns, confidence=confidence)
    assert result.objective_value == pytest.approx(best, abs=1e-9)
    for alternative in random_feasible("ABC", 25, seed=3):
        assert best <= objective_of(alternative, returns, confidence=confidence) + 1e-9


def test_confidence_99_uses_a_thinner_tail_than_95():
    data, returns = universe("A", "B")
    w = {"A": 0.5, "B": 0.5}
    series = series_of(w, returns)
    # 25 scenarios: 95% -> tail mass 1.25, 99% -> 0.25 (only the worst loss)
    r95 = optimize_portfolio(data, ["A", "B"], risk_aversion=0, return_weight=0)
    r99 = optimize_portfolio(data, ["A", "B"], risk_aversion=0, return_weight=0,
                             confidence_level=0.99)
    assert es_exact(series, 0.99) == pytest.approx(-min(series))
    s99 = series_of(r99.optimized_weights, returns)
    assert r99.historical_cvar == pytest.approx(-min(s99), rel=1e-9)
    assert r99.historical_cvar >= r95.historical_cvar - 1e-12


# 26-30. objective coefficients ------------------------------------------------------------------

def test_custom_risk_aversion():
    data, returns = universe("A", "B", "C")
    low = optimize_portfolio(data, ["A", "B", "C"], risk_aversion=0.1, cvar_weight=0)
    high = optimize_portfolio(data, ["A", "B", "C"], risk_aversion=1000, cvar_weight=0)
    assert high.risk_aversion == 1000
    assert high.variance_term <= low.variance_term + 1e-12
    assert high.expected_annual_return <= low.expected_annual_return + 1e-9
    for res, lam in ((low, 0.1), (high, 1000)):
        best = objective_of(res.optimized_weights, returns, lam, 0, 1)
        assert all(best <= objective_of(w, returns, lam, 0, 1) + 1e-9
                   for w in random_feasible("ABC", 20))


def test_custom_cvar_weight():
    data, returns = universe("A", "B", "C")
    result = optimize_portfolio(data, ["A", "B", "C"], risk_aversion=0, cvar_weight=3,
                                return_weight=1)
    assert result.cvar_weight == 3
    best = objective_of(result.optimized_weights, returns, 0, 3, 1)
    assert result.objective_value == pytest.approx(best, abs=1e-9)
    assert all(best <= objective_of(w, returns, 0, 3, 1) + 1e-9 for w in random_feasible("ABC", 25))


def test_custom_return_weight():
    data, returns = universe("A", "B", "C")
    result = optimize_portfolio(data, ["A", "B", "C"], risk_aversion=0, cvar_weight=0,
                                return_weight=2)
    best_symbol = max(returns, key=lambda s: statistics.mean(returns[s]) if s in "ABC" else -1)
    assert result.optimized_weights[best_symbol] == pytest.approx(1.0, abs=TOL)
    assert result.objective_value == pytest.approx(-2 * statistics.mean(returns[best_symbol]),
                                                   abs=1e-9)


@pytest.mark.parametrize("name", ["risk_aversion", "cvar_weight", "return_weight"])
def test_negative_objective_weights_fail(name):
    data, _ = universe("A", "B")
    with pytest.raises(ValueError, match=f"{name} must be a finite number >= 0"):
        optimize_portfolio(data, ["A", "B"], **{name: -0.5})


def test_all_objective_weights_zero_fails():
    data, _ = universe("A", "B")
    with pytest.raises(ValueError, match="At least one"):
        optimize_portfolio(data, ["A", "B"], risk_aversion=0, cvar_weight=0, return_weight=0)


# Important optimizer test: raising the CVaR weight moves away from the risky stock --------------

def test_raising_cvar_weight_shifts_allocation_away_from_tail_risk():
    data = combine(market("H", RH), market("L", RL))
    rh, rl = simple_returns(market("H", RH)), simple_returns(market("L", RL))
    assert statistics.mean(rh) > statistics.mean(rl)                       # higher return
    assert statistics.stdev(rh) > statistics.stdev(rl)                     # higher volatility
    assert es_exact(rh, 0.95) > es_exact(rl, 0.95)                         # worse tail
    crash_days = [t for t, r in enumerate(rh) if r < -0.02]
    assert all(rl[t] < 0 for t in crash_days)                              # crash together

    results = [optimize_portfolio(data, ["H", "L"], risk_aversion=0, cvar_weight=c,
                                  return_weight=1)
               for c in (0, 0.01, 0.05, 0.2, 1, 5)]
    w_h = [r.optimized_weights["H"] for r in results]
    cvars = [r.historical_cvar for r in results]
    rets = [r.expected_annual_return for r in results]
    assert w_h[0] == pytest.approx(1.0, abs=TOL)               # return only: the high-return stock
    assert all(b <= a + 1e-6 for a, b in zip(w_h, w_h[1:]))     # never increases with cvar_weight
    assert all(b <= a + 1e-9 for a, b in zip(cvars, cvars[1:]))
    assert all(b <= a + 1e-9 for a, b in zip(rets, rets[1:]))
    assert w_h[-1] < 0.5 and cvars[-1] < cvars[0] / 2


# 31-39. liquidity -------------------------------------------------------------------------------

def liquid_universe(turnover_a=None, volume_a=100.0):
    """A: higher return, ADTV ~ 10,000; B: lower return, ADTV ~ 100,000,000."""
    a = market("A", [abs(r) + 0.004 for r in RC], volume=volume_a, turnover=turnover_a)
    b = market("B", [r / 4 for r in RC], volume=1_000_000.0)
    return combine(a, b), a, b


def test_liquidity_constraint_disabled():
    data, a, _ = liquid_universe()
    result = optimize_portfolio(data, ["A", "B"], risk_aversion=0, cvar_weight=0,
                                return_weight=1, portfolio_value=100_000)
    assert result.optimized_weights["A"] == pytest.approx(1.0, abs=TOL)
    assert not result.liquidity_constraint_enabled
    assert set(result.liquidity["liquidity_constraint_status"]) == {LIQUIDITY_NOT_APPLIED}
    assert result.liquidity.set_index("symbol").loc["A", "position_to_adtv"] > 3


def test_liquidity_constraint_enabled_restricts_illiquid_stock():
    data, a, _ = liquid_universe()
    adtv_a = statistics.mean(c * 100.0 for c in a["close"])
    result = optimize_portfolio(data, ["A", "B"], risk_aversion=0, cvar_weight=0,
                                return_weight=1, portfolio_value=100_000,
                                liquidity_constraint_enabled=True, max_position_to_adtv=3)
    cap = 3 * adtv_a / 100_000
    assert 0.2 < cap < 0.5
    assert result.optimized_weights["A"] == pytest.approx(cap, abs=TOL)
    rows = result.liquidity.set_index("symbol")
    assert rows.loc["A", "liquidity_constraint_status"] == LIQUIDITY_AT_LIMIT
    assert rows.loc["B", "liquidity_constraint_status"] == LIQUIDITY_WITHIN_LIMIT
    assert rows.loc["A", "liquidity_weight_cap"] == pytest.approx(cap, rel=1e-12)


def test_max_position_to_adtv_is_respected():
    data, returns = universe("A", "B", "C")
    for k in (0.5, 2.0):
        # ADTV ~ Rs. 100,000 per stock: k = 0.5 caps each weight near 0.5 (binding)
        result = optimize_portfolio(data, ["A", "B", "C"], portfolio_value=100_000,
                                    liquidity_constraint_enabled=True, max_position_to_adtv=k)
        rows = result.liquidity.set_index("symbol")
        for s in "ABC":
            adtv = statistics.mean(c * 1_000.0 for c in FRAMES[s]["close"])
            position = 100_000 * result.optimized_weights[s]
            assert rows.loc[s, "average_daily_traded_value"] == pytest.approx(adtv, rel=1e-12)
            assert rows.loc[s, "position_value"] == pytest.approx(position, rel=1e-12)
            assert position / adtv <= k * (1 + 1e-6)


def test_missing_liquidity_data_leaves_stock_unconstrained_and_reported():
    data, _ = universe("A", "B")
    no_volume = data.drop(columns=["volume"])
    kwargs = dict(risk_aversion=0, cvar_weight=0, return_weight=1, portfolio_value=1e9,
                  liquidity_constraint_enabled=True, max_position_to_adtv=0.01)
    result = optimize_portfolio(no_volume, ["A", "B"], **kwargs)
    unconstrained = optimize_portfolio(no_volume, ["A", "B"], risk_aversion=0, cvar_weight=0,
                                       return_weight=1)
    assert result.optimized_weights == pytest.approx(unconstrained.optimized_weights, abs=TOL)
    assert result.liquidity_unconstrained_symbols == ("A", "B")
    assert set(result.liquidity["liquidity_constraint_status"]) == {LIQUIDITY_NO_DATA}
    assert result.liquidity["average_daily_traded_value"].isna().all()

    # one stock without usable volume: only that one is unconstrained
    partial = data.copy()
    partial.loc[partial["symbol"] == "B", "volume"] = np.nan
    result = optimize_portfolio(partial, ["A", "B"], **kwargs)
    assert result.liquidity_unconstrained_symbols == ("B",)
    assert result.optimized_weights["A"] <= 0.01 * statistics.mean(
        c * 1_000.0 for c in FRAMES["A"]["close"]) / 1e9 + 1e-9


def test_zero_adtv_forces_zero_weight_or_fails_with_minimum():
    data = combine(market("A", RA, volume=0.0), FRAMES["B"])
    kwargs = dict(portfolio_value=100_000, liquidity_constraint_enabled=True,
                  max_position_to_adtv=5)
    result = optimize_portfolio(data, ["A", "B"], **kwargs)
    assert result.optimized_weights["A"] == pytest.approx(0.0, abs=TOL)
    with pytest.raises(InfeasibleConstraintsError, match="A cannot reach the minimum weight"):
        optimize_portfolio(data, ["A", "B"], min_weight=0.1, **kwargs)


def test_actual_turnover_is_preferred():
    turnover = [20_000.0 + 100 * i for i in range(26)]
    data, a, _ = liquid_universe(turnover_a=turnover)
    data.loc[data["symbol"] == "B", "turnover"] = 1e8
    result = optimize_portfolio(data, ["A", "B"], risk_aversion=0, cvar_weight=0,
                                return_weight=1, portfolio_value=100_000,
                                liquidity_constraint_enabled=True, max_position_to_adtv=3)
    rows = result.liquidity.set_index("symbol")
    assert rows.loc["A", "traded_value_source"] == ACTUAL_TURNOVER
    assert rows.loc["A", "average_daily_traded_value"] == pytest.approx(statistics.mean(turnover))
    assert result.optimized_weights["A"] == pytest.approx(
        3 * statistics.mean(turnover) / 100_000, abs=TOL)


def test_estimated_traded_value_only_when_turnover_unavailable():
    turnover = [20_000.0] * 26
    turnover[7] = None                                       # incomplete -> estimate for every day
    data, a, _ = liquid_universe(turnover_a=turnover)
    result = optimize_portfolio(data, ["A", "B"], risk_aversion=0, cvar_weight=0,
                                return_weight=1, portfolio_value=100_000,
                                liquidity_constraint_enabled=True, max_position_to_adtv=3)
    row = result.liquidity.set_index("symbol").loc["A"]
    estimate = statistics.mean(c * 100.0 for c in a["close"])
    assert row["traded_value_source"] == ESTIMATED_TRADED_VALUE
    assert row["average_daily_traded_value"] == pytest.approx(estimate, rel=1e-12)
    assert result.optimized_weights["A"] == pytest.approx(3 * estimate / 100_000, abs=TOL)


def test_estimated_traded_value_is_never_labelled_as_turnover():
    data, _, _ = liquid_universe()
    result = optimize_portfolio(data, ["A", "B"], portfolio_value=100_000,
                                liquidity_constraint_enabled=True, max_position_to_adtv=3)
    sources = set(result.liquidity["traded_value_source"])
    assert sources == {ESTIMATED_TRADED_VALUE} and ACTUAL_TURNOVER not in sources


def test_portfolio_value_affects_liquidity_constraint():
    data, _, _ = liquid_universe()
    kwargs = dict(risk_aversion=0, cvar_weight=0, return_weight=1,
                  liquidity_constraint_enabled=True, max_position_to_adtv=3)
    small = optimize_portfolio(data, ["A", "B"], portfolio_value=100_000, **kwargs)
    large = optimize_portfolio(data, ["A", "B"], portfolio_value=200_000, **kwargs)
    assert large.optimized_weights["A"] == pytest.approx(small.optimized_weights["A"] / 2, abs=TOL)


def test_infeasible_liquidity_constraints_fail_clearly():
    data, _, _ = liquid_universe()
    with pytest.raises(InfeasibleConstraintsError, match="Infeasible liquidity constraint"):
        optimize_portfolio(data, ["A", "B"], portfolio_value=1e12,
                           liquidity_constraint_enabled=True, max_position_to_adtv=1)


def test_liquidity_parameters_are_validated():
    data, _ = universe("A", "B")
    with pytest.raises(ValueError, match="needs a portfolio_value"):
        optimize_portfolio(data, ["A", "B"], liquidity_constraint_enabled=True,
                           max_position_to_adtv=1)
    for bad in (None, 0, -1):
        with pytest.raises(ValueError, match="max_position_to_adtv must be a positive"):
            optimize_portfolio(data, ["A", "B"], portfolio_value=1e6,
                               liquidity_constraint_enabled=True, max_position_to_adtv=bad)
    with pytest.raises(ValueError, match="True or False"):
        optimize_portfolio(data, ["A", "B"], liquidity_constraint_enabled="yes")


# 40-41. solver status and result validation ---------------------------------------------------

@pytest.mark.parametrize("status, error, message", [
    (cp.INFEASIBLE, InfeasibleConstraintsError, "Infeasible"),
    (cp.INFEASIBLE_INACCURATE, InfeasibleConstraintsError, "Infeasible"),
    (cp.UNBOUNDED, OptimizationError, "unbounded"),
    (cp.UNBOUNDED_INACCURATE, OptimizationError, "unbounded"),
    (cp.OPTIMAL_INACCURATE, OptimizationError, "optimal_inaccurate"),
    (cp.SOLVER_ERROR, OptimizationError, "solver_error"),
])
def test_solver_status_is_checked(monkeypatch, status, error, message):
    data, _ = universe("A", "B")
    monkeypatch.setattr(optimizer, "_solve", lambda problem: status)
    with pytest.raises(error, match=message):
        optimize_portfolio(data, ["A", "B"])


def test_solver_exception_is_reported(monkeypatch):
    data, _ = universe("A", "B")

    def fail(self, *args, **kwargs):
        raise cp.error.SolverError("numerical trouble")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    with pytest.raises(OptimizationError, match="CLARABEL solver failed: numerical trouble"):
        optimize_portfolio(data, ["A", "B"])


def test_result_weights_are_validated(monkeypatch):
    data, _ = universe("A", "B")
    # "optimal" without an actual solve leaves no weights: must be rejected, not reported
    monkeypatch.setattr(optimizer, "_solve", lambda problem: cp.OPTIMAL)
    with pytest.raises(OptimizationError, match="non-finite"):
        optimize_portfolio(data, ["A", "B"])


@pytest.mark.parametrize("weights, max_weight, message", [
    ({"A": 0.5, "B": float("nan")}, 1.0, "non-finite"),
    ({"A": -0.01, "B": 1.01}, 1.0, "below the minimum"),
    ({"A": 0.3, "B": 0.7}, 0.65, "above the maximum"),
    ({"A": 0.5, "B": 0.4}, 1.0, "sum to 0.9"),
    ({"A": 1.0}, 1.0, "exactly the selected"),
])
def test_validate_portfolio_weights(weights, max_weight, message):
    with pytest.raises(OptimizationError, match=message):
        validate_portfolio_weights(weights, ["A", "B"], max_weight=max_weight)


def test_validate_portfolio_weights_accepts_noise_within_tolerance():
    ok = validate_portfolio_weights({"B": 0.6 + 5e-7, "A": 0.4 - 1e-9})
    assert list(ok) == ["A", "B"]


# 42-45. no mutation, baseline, HHI, reproducibility ---------------------------------------------

def test_input_is_not_mutated():
    data, _ = universe("A", "B", "C")
    data = data.sample(frac=1, random_state=2).reset_index(drop=True)
    before = data.copy(deep=True)
    symbols = ["C", "A", "B"]
    optimize_portfolio(data, symbols, portfolio_value=1e6, liquidity_constraint_enabled=True,
                       max_position_to_adtv=10)
    pd.testing.assert_frame_equal(data, before)
    assert symbols == ["C", "A", "B"]


def test_equal_weight_baseline():
    assert equal_weight_portfolio(["C", "A", "B"]) == pytest.approx({"A": 1 / 3, "B": 1 / 3,
                                                                     "C": 1 / 3})
    data, returns = universe("A", "B", "C", "D")
    result = optimize_portfolio(data, ["A", "B", "C", "D"])
    ew = result.equal_weight_metrics
    series = [sum(returns[s][t] for s in "ABCD") / 4 for t in range(25)]
    assert ew.weights == pytest.approx({s: 0.25 for s in "ABCD"})
    assert ew.expected_annual_return == pytest.approx(statistics.mean(series) * 252, rel=1e-9)
    assert ew.annualized_volatility == pytest.approx(statistics.stdev(series) * math.sqrt(252),
                                                     rel=1e-9)
    assert ew.historical_cvar == pytest.approx(es_exact(series, 0.95), rel=1e-9)
    assert ew.historical_var == pytest.approx(-quantile_type7(series, 0.05), rel=1e-9)
    assert ew.hhi == pytest.approx(0.25) and ew.max_weight == pytest.approx(0.25)
    with pytest.raises(ValueError):
        equal_weight_portfolio(["A", "A"])


def test_hhi():
    data, _ = universe("A", "B", "C")
    result = optimize_portfolio(data, ["A", "B", "C"])
    w = result.optimized_weights
    assert result.hhi == pytest.approx(sum(v * v for v in w.values()), rel=1e-12)
    assert result.max_weight == pytest.approx(max(w.values()), rel=1e-12)


def test_deterministic_on_the_same_input():
    data, _ = universe("A", "B", "C", "D")
    kwargs = dict(max_weight=0.4, portfolio_value=1e6, liquidity_constraint_enabled=True,
                  max_position_to_adtv=5)
    first = optimize_portfolio(data, ["A", "B", "C", "D"], **kwargs)
    second = optimize_portfolio(data, ["A", "B", "C", "D"], **kwargs)
    assert first.optimized_weights == second.optimized_weights
    assert first.objective_value == second.objective_value
