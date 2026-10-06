"""Tests for app.backtesting.backtest_engine (walk-forward backtest, no look-ahead).

Expected values never come from the engine: the schedule is rebuilt by
indexing the calendar, equal-weight buy-and-hold returns are simulated in plain
Python from the closes, metrics use statistics / exact fractions, and the
expected optimizer weights come from calling app.portfolio.optimizer directly
on the training rows only. The data is synthetic test data.
"""

import math
import statistics
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtesting import backtest_engine
from app.backtesting.backtest_engine import (
    CARRIED_FORWARD,
    EQUAL_WEIGHT,
    EXITSAFE,
    MEAN_VARIANCE,
    METRIC_ROWS,
    NOT_INVESTED,
    REBALANCED,
    UNAVAILABLE,
    AVAILABLE,
    BacktestConfig,
    BacktestError,
    build_walk_forward_schedule,
    calculate_backtest_metrics,
    create_equity_curve,
    run_strategy_backtest,
    run_walk_forward_backtest,
)
from app.data.loaders.csv_market_loader import load_csv_market_data
from app.portfolio.optimizer import OptimizationError, optimize_portfolio

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_backtest_data.csv"
N_DAYS = 120
SYMBOLS = ("A", "B", "C")
SPEC = {"A": (0.0006, 0.010, 400.0), "B": (0.0002, 0.005, 2_000.0), "C": (0.0010, 0.018, 50.0)}


def make_data(days=N_DAYS, seed=3, shock_after=None, shock=0.0, volume_after=None):
    """Synthetic closes (deterministic). ``shock_after``: add ``shock`` to every return
    from that day index on (used to change ONLY the future)."""
    rng = np.random.RandomState(seed)
    market = rng.normal(0, 0.006, days)
    frames = []
    for i, (symbol, (mean, sd, volume)) in enumerate(SPEC.items()):
        r = mean + (0.5 + 0.4 * i) * market + rng.normal(0, sd, days)
        if shock_after is not None:
            r[shock_after:] += shock
        closes = 100.0 * np.cumprod(np.concatenate([[1.0], 1 + r[1:]]))
        volumes = np.full(days, volume)
        if volume_after is not None:
            volumes[volume_after:] = 0.001
        frames.append(pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=days),
                                    "symbol": symbol, "close": closes, "volume": volumes}))
    return pd.concat(frames, ignore_index=True)


def config(**overrides):
    base = dict(symbols=SYMBOLS, training_window=40, test_window=10, rebalance_frequency=10,
                min_observations=20, max_weight=0.6, initial_capital=1_000_000.0)
    return BacktestConfig(**{**base, **overrides})


DATA = make_data()
CALENDAR = list(pd.bdate_range("2025-01-01", periods=N_DAYS))


@pytest.fixture(scope="module")
def result():
    return run_walk_forward_backtest(DATA, config())


def closes(data, symbol):
    rows = data[data["symbol"] == symbol].sort_values("date")
    return dict(zip(rows["date"], rows["close"]))


def equal_weight_reference(data, cfg, calendar):
    """Plain-Python equal-weight buy-and-hold over every complete test window."""
    prices = {s: closes(data, s) for s in cfg.symbols}
    returns, capital = {}, 1.0
    k = cfg.training_window
    while k + cfg.test_window <= len(calendar):
        holdings = {s: capital / len(cfg.symbols) for s in cfg.symbols}
        for j in range(k, k + cfg.test_window):
            day, prev = calendar[j], calendar[j - 1]
            r = {s: prices[s][day] / prices[s][prev] - 1 for s in cfg.symbols}
            total = sum(holdings.values())
            returns[day] = sum(holdings[s] * r[s] for s in cfg.symbols) / total
            holdings = {s: holdings[s] * (1 + r[s]) for s in cfg.symbols}
        capital = sum(holdings.values())
        k += cfg.rebalance_frequency
    return returns


def training_rows(data, w):
    return data[data["date"].isin(w.training_dates)]


def quantile_type7(values, p):
    x = sorted(values)
    h = (len(x) - 1) * p
    lo = math.floor(h)
    return x[lo] + (h - lo) * (x[min(lo + 1, len(x) - 1)] - x[lo])


def es_exact(returns, confidence):
    alpha = Fraction(1) - Fraction(str(confidence))
    losses = sorted((-Fraction(float(r)) for r in returns), reverse=True)
    m = alpha * len(losses)
    k = math.floor(m)
    total = sum(losses[:k], Fraction(0)) + ((m - k) * losses[k] if m > k else 0)
    return float(total / m)


def log_for(res, strategy):
    return [r for r in res.rebalances if r.strategy == strategy]


# 1-8. walk-forward schedule and timing -------------------------------------------------------

def test_walk_forward_ordering_and_window_sizes(result):
    cfg = config()
    expected_starts = list(range(40, N_DAYS - 10 + 1, 10))          # 40, 50, ..., 110
    assert [w.window for w in result.windows] == list(range(1, len(expected_starts) + 1))
    for w, k in zip(result.windows, expected_starts):
        assert list(w.training_dates) == CALENDAR[k - 40:k]          # correct training size
        assert list(w.test_dates) == CALENDAR[k:k + 10]              # correct test size
        assert max(w.training_dates) < min(w.test_dates)             # training precedes test
    starts = [CALENDAR.index(w.test_dates[0]) for w in result.windows]
    assert all(b - a == cfg.rebalance_frequency for a, b in zip(starts, starts[1:]))
    assert len(result.windows) == 8                                  # multiple rebalances


def test_rebalance_timing_decision_after_close_t_first_return_on_t_plus_1(result):
    first = log_for(result, EQUAL_WEIGHT)[0]
    t = CALENDAR[39]
    assert first.training_end == t and first.rebalance_date == CALENDAR[40] == first.test_start
    assert result.comparison_start == t                              # value 1.0 at T's close
    returns = result.returns.set_index("date")
    assert returns.index[0] == CALENDAR[40]                          # first return is T+1
    a, b, c = (closes(DATA, s) for s in SYMBOLS)
    expected = sum(p[CALENDAR[40]] / p[t] - 1 for p in (a, b, c)) / 3
    assert returns.loc[CALENDAR[40], "EqualWeight"] == pytest.approx(expected, rel=1e-12)
    assert t not in set(returns.index)                               # T's own return is training


def test_next_day_rule_close_on_t_plus_1_does_not_affect_that_rebalance():
    w = build_walk_forward_schedule(DATA, config())[0]
    changed = DATA.copy()
    changed.loc[changed["date"] == w.test_dates[0], "close"] *= 1.5   # the first test day
    base = run_strategy_backtest(DATA, EXITSAFE, config())
    moved = run_strategy_backtest(changed, EXITSAFE, config())
    assert moved.rebalances[0].weights == base.rebalances[0].weights
    assert moved.returns[w.test_dates[0]] != base.returns[w.test_dates[0]]


def test_training_uses_data_through_the_decision_close():
    w = build_walk_forward_schedule(DATA, config())[0]
    changed = DATA.copy()
    mask = (changed["date"] == w.training_dates[-1]) & (changed["symbol"] == "C")
    changed.loc[mask, "close"] *= 0.7                                # a crash at T's close
    base = run_strategy_backtest(DATA, EXITSAFE, config())
    moved = run_strategy_backtest(changed, EXITSAFE, config())
    assert moved.rebalances[0].weights != base.rebalances[0].weights


def test_test_window_shorter_than_rebalance_frequency_leaves_gaps():
    res = run_walk_forward_backtest(DATA, config(test_window=5, rebalance_frequency=10))
    dates = list(res.returns["date"])
    for w in res.windows:
        assert len(w.test_dates) == 5
    gap = CALENDAR[45:50]
    assert not set(gap) & set(dates) and res.observations == 5 * len(res.windows)


# 25-28. critical look-ahead tests ---------------------------------------------------------------

def test_critical_future_returns_cannot_change_past_weights():
    cut = 80                                       # day index where the future starts
    calm = make_data()
    wild = make_data(shock_after=cut, shock=0.25)  # +25% EVERY day from day 80 on
    cfg = config()
    for strategy in (EXITSAFE, MEAN_VARIANCE, EQUAL_WEIGHT):
        a = run_strategy_backtest(calm, strategy, cfg)
        b = run_strategy_backtest(wild, strategy, cfg)
        checked = 0
        for ra, rb in zip(a.rebalances, b.rebalances):
            if ra.training_end < CALENDAR[cut]:
                assert ra.weights == rb.weights, (strategy, ra.window)
                checked += 1
        # windows 1-5 decide at or before day 79's close; window 5's test period starts
        # exactly at the shock, so even it must keep its weights
        assert checked == 5
    later = [r for r in run_strategy_backtest(wild, EXITSAFE, cfg).rebalances
             if r.training_end >= CALENDAR[cut]]
    assert later and later[-1].weights != run_strategy_backtest(
        calm, EXITSAFE, cfg).rebalances[-1].weights              # the future itself does change


def test_future_liquidity_data_is_not_used_for_past_weights():
    # C's ADTV is about Rs. 5,000: at Rs. 1,000,000 and Position / ADTV <= 50 its cap
    # (about 25%) binds; the drained volume later shrinks every cap
    cfg = config(liquidity_constraint_enabled=True, max_position_to_adtv=50.0,
                 initial_capital=1_000_000.0)
    base = run_strategy_backtest(make_data(), EXITSAFE, cfg)
    dry = run_strategy_backtest(make_data(volume_after=80), EXITSAFE, cfg)   # volume vanishes
    for ra, rb in zip(base.rebalances, dry.rebalances):
        if ra.training_end < CALENDAR[80]:
            assert ra.status == rb.status == REBALANCED         # real decisions, not carries
            assert ra.weights == rb.weights
    assert any(ra.weights != rb.weights or ra.status != rb.status
               for ra, rb in zip(base.rebalances, dry.rebalances))       # the future still matters


def test_exitsafe_weights_equal_the_optimizer_on_training_rows_only(result):
    cfg = config()
    for w, logged in zip(result.windows, log_for(result, EXITSAFE)):
        expected = optimize_portfolio(training_rows(DATA, w), list(SYMBOLS), max_weight=0.6,
                                      min_observations=20).optimized_weights
        assert logged.status == REBALANCED and logged.solver_status == "optimal"
        assert logged.parameters["cvar_weight"] == cfg.cvar_weight
        if w.window == 1:
            assert logged.weights == pytest.approx(expected, abs=1e-12)


def test_mean_variance_is_the_optimizer_without_cvar(result):
    mv = log_for(result, MEAN_VARIANCE)
    for w, logged in zip(result.windows, mv):
        assert logged.parameters["cvar_weight"] == 0.0
        assert logged.parameters["liquidity_constraint_enabled"] is False
    w = result.windows[0]
    expected = optimize_portfolio(training_rows(DATA, w), list(SYMBOLS), cvar_weight=0.0,
                                  max_weight=0.6, min_observations=20).optimized_weights
    assert mv[0].weights == pytest.approx(expected, abs=1e-12)
    # minimizes risk_aversion * variance - mean over long-only bounded weights
    rows = training_rows(DATA, w)
    rets = {s: list(pd.Series(closes(rows, s)).pct_change().dropna()) for s in SYMBOLS}

    def objective(weights):
        var = sum(weights[a] * weights[b] * statistics.covariance(rets[a], rets[b])
                  for a in SYMBOLS for b in SYMBOLS)
        return var - sum(weights[s] * statistics.mean(rets[s]) for s in SYMBOLS)

    rng = np.random.RandomState(0)
    best = objective(mv[0].weights)
    for _ in range(200):
        x = rng.dirichlet(np.ones(3))
        if x.max() <= 0.6:
            assert best <= objective(dict(zip(SYMBOLS, x))) + 1e-12
    assert any(e.weights != m.weights for e, m in zip(log_for(result, EXITSAFE), mv))


def test_strategy_fairness_same_universe_windows_and_dates(result):
    by_window = {}
    for r in result.rebalances:
        by_window.setdefault(r.window, []).append(r)
    for rows in by_window.values():
        assert len(rows) == 3
        assert len({(r.training_start, r.training_end, r.test_start, r.test_end,
                     r.training_observations) for r in rows}) == 1
        assert all(set(r.weights) == set(SYMBOLS) for r in rows)
    returns = result.returns
    assert not returns[["ExitSafe", "EqualWeight", "MeanVariance"]].isna().any().any()


# 9-14. strategies and benchmarks ---------------------------------------------------------------

def test_equal_weight_strategy_returns(result):
    expected = equal_weight_reference(DATA, config(), CALENDAR)
    got = result.returns.set_index("date")["EqualWeight"]
    assert list(got.index) == sorted(expected)
    assert got.tolist() == pytest.approx([expected[d] for d in got.index], rel=1e-10)
    assert all(r.weights == pytest.approx({s: 1 / 3 for s in SYMBOLS})
               for r in log_for(result, EQUAL_WEIGHT))


def test_one_stock():
    res = run_walk_forward_backtest(DATA[DATA["symbol"] == "B"], config(symbols=("B",),
                                                                        max_weight=1.0))
    prices = closes(DATA, "B")
    returns = res.returns.set_index("date")
    for day in returns.index[:5]:
        prev = CALENDAR[CALENDAR.index(day) - 1]
        for label in ("ExitSafe", "EqualWeight", "MeanVariance"):
            assert returns.loc[day, label] == pytest.approx(prices[day] / prices[prev] - 1)


def test_multiple_stocks_long_only_and_bounds(result):
    for r in result.rebalances:
        if r.status == REBALANCED:
            assert sum(r.weights.values()) == pytest.approx(1.0, abs=1e-6)
            assert all(-1e-9 <= v <= 0.6 + 1e-6 for v in r.weights.values())
    res = run_walk_forward_backtest(DATA, config(min_weight=0.2, max_weight=0.5))
    for r in res.rebalances:
        assert all(0.2 - 1e-6 <= v <= 0.5 + 1e-6 for v in r.weights.values())


def index_frame(data=DATA, drop=None, name="ASPI"):
    # a synthetic index: the average close of the three stocks
    pivot = data.pivot(index="date", columns="symbol", values="close").mean(axis=1)
    frame = pd.DataFrame({"date": pivot.index, "index_name": name, "close": pivot.values})
    if drop is not None:
        frame = frame[~frame["date"].isin(drop)]
    return frame.reset_index(drop=True)


def test_market_index_benchmark_and_alignment():
    idx = index_frame()
    res = run_walk_forward_backtest(DATA, config(), idx, "ASPI")
    assert res.market_index_status == AVAILABLE and "MarketIndex" in res.strategies
    level = dict(zip(idx["date"], idx["close"]))
    returns = res.returns.set_index("date")
    for day in returns.index:
        prev = CALENDAR[CALENDAR.index(day) - 1]
        assert returns.loc[day, "MarketIndex"] == pytest.approx(level[day] / level[prev] - 1)
    assert math.isnan(res.metrics.loc["Rebalances", "MarketIndex"])
    # an index gap removes that day (and the next, which spans two days) for every strategy
    gap = [CALENDAR[60]]
    gapped = run_walk_forward_backtest(DATA, config(), index_frame(drop=gap), "ASPI")
    dates = set(gapped.returns["date"])
    assert CALENDAR[60] not in dates and CALENDAR[61] not in dates
    assert gapped.observations == res.observations - 2
    assert gapped.excluded_test_days == 2


def test_missing_market_index_is_unavailable(result):
    assert result.market_index_status == UNAVAILABLE
    assert list(result.metrics.columns) == ["ExitSafe", "EqualWeight", "MeanVariance"]
    assert "MarketIndex" not in result.equity_curve.columns
    far = index_frame()
    far["date"] = far["date"] + pd.Timedelta(days=3650)
    res = run_walk_forward_backtest(DATA, config(), far, "ASPI")
    assert res.market_index_status == UNAVAILABLE and "no same-period" in res.market_index_reason
    with pytest.raises(ValueError, match="Index S&P SL20 not found"):
        run_walk_forward_backtest(DATA, config(), index_frame(), "S&P SL20")


# 15-23. data problems -------------------------------------------------------------------------

def test_missing_selected_symbol():
    with pytest.raises(ValueError, match="not found in the market data: ZZZ"):
        run_walk_forward_backtest(DATA, config(symbols=("A", "ZZZ")))


def test_insufficient_training_data():
    with pytest.raises(BacktestError, match="insufficient training data: 9 common"):
        run_walk_forward_backtest(DATA, config(training_window=10, min_observations=20))
    status = ["VALID"] * len(DATA)
    data = DATA.assign(validation_status=status)
    first_training = CALENDAR[:40]
    # A INVALID on days 0, 2, ..., 20: returns of days 1-21 are lost, leaving window 1 with
    # 39 - 21 = 18 (< 20) common returns, while window 2 (days 10-49) keeps 28
    bad = data["date"].isin(first_training[:21:2]) & (data["symbol"] == "A")
    data.loc[bad, "validation_status"] = "INVALID"
    res = run_walk_forward_backtest(data, config())
    assert res.skipped_rebalances[0][0] == 1 and "insufficient training data" in res.skipped_rebalances[0][2]
    first_logged = [r for r in res.rebalances if r.window == 1]
    assert {r.status for r in first_logged} == {NOT_INVESTED}
    assert res.comparison_start == CALENDAR[49]                   # starts with window 2


def test_insufficient_test_data():
    short = DATA[DATA["date"] < CALENDAR[49]]                     # 49 days < 40 + 10
    with pytest.raises(BacktestError, match="needs 50 observations, the data has 49"):
        run_walk_forward_backtest(short, config())
    exactly = DATA[DATA["date"] < CALENDAR[50]]
    assert run_walk_forward_backtest(exactly, config()).observations == 10


def test_incomplete_final_test_window_is_not_used():
    data = DATA[DATA["date"] < CALENDAR[115]]                     # last window would be 110-114
    res = run_walk_forward_backtest(data, config())
    assert len(res.windows) == 7 and max(res.returns["date"]) == CALENDAR[109]


def test_invalid_training_rows_reduce_training_observations():
    data = DATA.assign(validation_status="VALID")
    data.loc[(data["date"] == CALENDAR[20]) & (data["symbol"] == "B"), "validation_status"] = "INVALID"
    windows = build_walk_forward_schedule(data, config())
    assert windows[0].training_observations == 39 - 2            # day 20 and day 21 lost
    assert windows[1].training_observations == 39 - 2
    assert windows[2].training_observations == 39 - 1             # day 20 starts the window:
    assert windows[3].training_observations == 39                 # only day 21's return is lost


def test_invalid_test_rows_are_excluded_not_zero_filled():
    data = DATA.assign(validation_status="VALID")
    data.loc[(data["date"] == CALENDAR[55]) & (data["symbol"] == "C"), "validation_status"] = "INVALID"
    res = run_walk_forward_backtest(data, config())
    dates = set(res.returns["date"])
    assert CALENDAR[55] not in dates and CALENDAR[56] not in dates   # no bridging over day 55
    assert res.excluded_test_days == 2 and res.observations == 80 - 2
    window2 = [r for r in res.rebalances if r.window == 2]
    assert {(r.evaluated_days, r.excluded_days) for r in window2} == {(8, 2)}
    assert 0.0 not in res.returns["EqualWeight"].tolist()


def test_warning_rows_are_usable():
    data = DATA.assign(validation_status="WARNING")
    res = run_walk_forward_backtest(data, config())
    assert res.observations == 80 and res.excluded_test_days == 0


def test_missing_dates():
    data = DATA[~((DATA["date"] == CALENDAR[70]) & (DATA["symbol"] == "A"))]
    res = run_walk_forward_backtest(data, config())
    dates = set(res.returns["date"])
    assert CALENDAR[70] not in dates and CALENDAR[71] not in dates   # 71 spans two days for A
    assert res.observations == 78


# 24-33. returns, value path and metrics -------------------------------------------------------

def test_portfolio_value_path_and_initial_capital(result):
    expected = equal_weight_reference(DATA, config(), CALENDAR)
    value, path = 1.0, [1.0]
    for day in sorted(expected):
        value *= 1 + expected[day]
        path.append(value)
    curve = result.equity_curve
    assert curve["date"].iloc[0] == CALENDAR[39] and (curve.iloc[0, 1:] == 1.0).all()
    assert curve["EqualWeight"].tolist() == pytest.approx(path, rel=1e-10)
    assert result.final_capital["EqualWeight"] == pytest.approx(1_000_000 * path[-1], rel=1e-10)


def test_metrics_match_independent_calculations(result):
    r = result.returns["ExitSafe"].tolist()
    n = len(r)
    m = result.metrics["ExitSafe"]
    cumulative = math.prod(1 + x for x in r) - 1
    vol = statistics.stdev(r) * math.sqrt(252)
    downside = math.sqrt(sum(min(x, 0) ** 2 for x in r) / n) * math.sqrt(252)
    value, peak, mdd = 1.0, 1.0, 0.0
    for x in r:
        value *= 1 + x
        peak = max(peak, value)
        mdd = min(mdd, value / peak - 1)
    assert m["Cumulative Return"] == pytest.approx(cumulative, rel=1e-9)
    assert m["Annualized Return (arithmetic)"] == pytest.approx(statistics.mean(r) * 252, rel=1e-9)
    assert m["Annualized Return (geometric)"] == pytest.approx((1 + cumulative) ** (252 / n) - 1,
                                                                 rel=1e-9)
    assert m["Annualized Volatility"] == pytest.approx(vol, rel=1e-9)
    assert m["Sharpe"] == pytest.approx(statistics.mean(r) * 252 / vol, rel=1e-9)
    assert m["Sortino"] == pytest.approx(statistics.mean(r) * 252 / downside, rel=1e-9)
    assert m["Maximum Drawdown"] == pytest.approx(mdd, rel=1e-9)
    assert m["Historical VaR (1-day)"] == pytest.approx(-quantile_type7(r, 0.05), rel=1e-9)
    assert m["Historical CVaR (1-day)"] == pytest.approx(es_exact(r, 0.95), rel=1e-9)
    assert m["Observations"] == n == 80 and m["Rebalances"] == 8


def test_risk_free_rate_is_explicit_and_shared():
    res = run_walk_forward_backtest(DATA, config(risk_free_rate=0.05))
    rf_d = 1.05 ** (1 / 252) - 1
    for label in res.strategies:
        r = res.returns[label].tolist()
        vol = statistics.stdev(r) * math.sqrt(252)
        assert res.metrics.loc["Sharpe", label] == pytest.approx(
            (statistics.mean(r) - rf_d) * 252 / vol, rel=1e-9)


def test_out_of_sample_metrics_use_only_test_returns(result):
    test_days = {d for w in result.windows for d in w.test_dates}
    training_only = set(CALENDAR[:40])
    dates = set(result.returns["date"])
    assert dates <= test_days and not dates & training_only
    assert result.metrics.loc["Observations"].tolist() == [80.0] * 3
    # metrics of an arbitrary series come from that series only
    series = pd.Series([0.01, -0.02, 0.03], index=CALENDAR[:3])
    metrics = calculate_backtest_metrics(series, config(min_observations=2))
    assert metrics["Cumulative Return"] == pytest.approx(1.01 * 0.98 * 1.03 - 1)
    assert set(metrics.index) == set(METRIC_ROWS)


def test_create_equity_curve():
    returns = pd.DataFrame({"X": [0.1, -0.5]}, index=CALENDAR[1:3])
    curve = create_equity_curve(returns, CALENDAR[0])
    assert curve["X"].tolist() == pytest.approx([1.0, 1.1, 0.55])
    assert curve["date"].tolist() == CALENDAR[:3]


# 34-44. log, comparison, failures, determinism, mutation, config -------------------------------

def test_rebalance_log(result):
    log = result.rebalance_log
    assert list(log.columns) == backtest_engine.LOG_COLUMNS
    assert len(log) == 8 * 3
    row = log[(log["window"] == 1) & (log["strategy"] == "ExitSafe")].iloc[0]
    assert row["training_start"] == CALENDAR[0] and row["training_end"] == CALENDAR[39]
    assert row["test_start"] == row["rebalance_date"] == CALENDAR[40]
    assert row["test_end"] == CALENDAR[49] and row["training_observations"] == 39
    assert row["solver_status"] == "optimal" and row["status"] == REBALANCED
    assert set(row["weights"]) == set(SYMBOLS) and row["parameters"]["max_weight"] == 0.6


def test_strategy_comparison_table(result):
    assert list(result.metrics.index) == METRIC_ROWS
    assert list(result.metrics.columns) == ["ExitSafe", "EqualWeight", "MeanVariance"]
    assert result.final_weights["EqualWeight"] == pytest.approx({s: 1 / 3 for s in SYMBOLS})
    assert set(result.final_weights["ExitSafe"]) == set(SYMBOLS)


def test_optimizer_failures_are_recorded_and_positions_kept(monkeypatch):
    real = backtest_engine.optimize_portfolio
    fail_on = CALENDAR[59]                                     # decision date of window 3

    def flaky(training, *args, **kwargs):
        if training["date"].max() == fail_on and kwargs.get("cvar_weight", 1.0) > 0:
            raise OptimizationError("simulated solver failure")
        return real(training, *args, **kwargs)

    monkeypatch.setattr(backtest_engine, "optimize_portfolio", flaky)
    res = run_walk_forward_backtest(DATA, config())
    third = [r for r in res.rebalances if r.window == 3 and r.strategy == EXITSAFE][0]
    assert third.status == CARRIED_FORWARD and "simulated solver failure" in third.reason
    assert "previous positions kept" in third.reason and third.solver_status is None
    second = [r for r in res.rebalances if r.window == 2 and r.strategy == EXITSAFE][0]
    assert set(third.weights) == set(second.weights) and third.weights != second.weights  # drifted
    assert res.metrics.loc["Rebalances", "ExitSafe"] == 7

    def broken_first(training, *args, **kwargs):
        if training["date"].max() == CALENDAR[39]:
            raise OptimizationError("first window fails")
        return real(training, *args, **kwargs)

    monkeypatch.setattr(backtest_engine, "optimize_portfolio", broken_first)
    res = run_walk_forward_backtest(DATA, config())
    assert [r.status for r in res.rebalances if r.window == 1 and r.strategy == EXITSAFE] == [NOT_INVESTED]
    assert res.comparison_start == CALENDAR[49] and res.observations == 70


def test_deterministic_repeated_run(result):
    again = run_walk_forward_backtest(DATA, config())
    pd.testing.assert_frame_equal(again.returns, result.returns)
    pd.testing.assert_frame_equal(again.metrics, result.metrics)
    assert [r.weights for r in again.rebalances] == [r.weights for r in result.rebalances]


def test_input_is_not_mutated():
    data = DATA.sample(frac=1, random_state=1).reset_index(drop=True)
    before = data.copy(deep=True)
    idx = index_frame()
    idx_before = idx.copy(deep=True)
    res = run_walk_forward_backtest(data, config(), idx, "ASPI")
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(idx, idx_before)
    assert res.observations == 80                                  # unsorted input works


@pytest.mark.parametrize("kwargs, message", [
    ({"training_window": 1}, "training_window"),
    ({"test_window": 0}, "test_window"),
    ({"rebalance_frequency": 2.5}, "rebalance_frequency"),
    ({"test_window": 20, "rebalance_frequency": 10}, "overlap"),
    ({"initial_capital": 0}, "position_value"),
    ({"risk_free_rate": float("nan")}, "risk"),
    ({"confidence_level": 1.0}, "confidence_level"),
    ({"max_weight": 0.3}, "Infeasible weight bounds"),
    ({"cvar_weight": -1}, "cvar_weight"),
    ({"liquidity_constraint_enabled": True}, "max_position_to_adtv"),
    ({"symbols": ("A", "A")}, "Duplicate"),
    ({"symbols": ()}, "at least one"),
])
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        config(**kwargs)


def test_config_is_required():
    with pytest.raises(ValueError, match="BacktestConfig"):
        run_walk_forward_backtest(DATA, {"symbols": SYMBOLS})


# synthetic fixture -------------------------------------------------------------------------

def test_synthetic_fixture_runs_a_multi_window_backtest():
    data = load_csv_market_data(FIXTURE).data
    assert set(data["symbol"]) == {"ALPHA", "BRAVO", "CHARLIE", "DELTA"}
    assert data.groupby("symbol").size().min() >= 300
    res = run_walk_forward_backtest(data, BacktestConfig(
        symbols=("ALPHA", "BRAVO", "CHARLIE", "DELTA"), max_weight=0.4))
    assert len(res.windows) == 13 and res.observations == 260
    assert res.metrics.loc["Rebalances"].tolist() == [13.0, 13.0, 13.0]
