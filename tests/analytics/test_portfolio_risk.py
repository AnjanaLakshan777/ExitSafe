"""Tests for app.analytics.portfolio_risk (fixed, user-supplied weights).

Expected values are computed independently of the module under test: the
portfolio return series is rebuilt in plain Python from the returns used to
generate the prices, covariance comes from statistics.covariance, quantiles
and expected shortfall from small reference implementations below, and the
Normal figures from statistics.NormalDist (not scipy).

Worked example (two stocks, 60/40)
  A returns: +2%, -1%, +3%      B returns: -1%, +2%, 0%
  R_p = 0.6 * A + 0.4 * B   =   0.008, 0.002, 0.018
  cumulative = 1.008 * 1.002 * 1.018 - 1 = 0.0281963...
"""

import math
import statistics
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics.drawdown import calculate_maximum_drawdown
from app.analytics.liquidity import ACTUAL_TURNOVER, ESTIMATED_TRADED_VALUE
from app.analytics.portfolio_risk import (
    RETURN_SERIES_COLUMNS,
    WEIGHT_SUM_TOLERANCE,
    PortfolioRiskResult,
    calculate_portfolio_return_series,
    calculate_portfolio_risk_summary,
    validate_weights,
)
from app.analytics.var import calculate_var_summary
from app.data.loaders.csv_market_loader import load_csv_market_data

SAMPLE_3_SYMBOLS = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"
ND = statistics.NormalDist()

RA = [0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.015, -0.01, 0.005, -0.03, 0.012, -0.008,
      0.025, -0.015, 0.004, -0.022, 0.018, -0.006, 0.009, -0.04, 0.011, -0.003, 0.007, 0.02, -0.012]
RB = [-0.01, 0.02, -0.005, 0.01, 0.0, 0.03, -0.02, 0.015, -0.01, 0.02, -0.004, 0.006,
      -0.01, 0.012, 0.003, 0.01, -0.015, 0.008, -0.002, 0.025, -0.006, 0.004, -0.009, -0.01, 0.007]
RC = [0.005, 0.01, -0.02, 0.004, -0.012, 0.008, 0.002, -0.006, 0.015, -0.01, 0.003, 0.0,
      -0.004, 0.009, -0.007, 0.012, 0.001, -0.003, 0.006, -0.015, 0.004, 0.002, -0.001, 0.005, 0.003]


def market(symbol, returns, start=100.0, volume=1_000.0, turnover=None, status=None, dates=None):
    """Prices generated from ``returns`` (first close = start)."""
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    n = len(closes)
    frame = pd.DataFrame({
        "date": dates if dates is not None else pd.bdate_range("2026-01-01", periods=n),
        "symbol": symbol, "close": closes,
        "volume": volume if np.ndim(volume) == 0 else [float(v) for v in volume]})
    if turnover is not None:
        frame["turnover"] = [float(t) if t is not None else np.nan for t in turnover]
    if status is not None:
        frame["validation_status"] = status
    return frame


def combine(*frames):
    return pd.concat(frames, ignore_index=True)


def expected_series(weights, returns):
    """R_p(t) = sum_i w_i R_i(t), in plain Python."""
    return [sum(weights[s] * returns[s][t] for s in weights) for t in range(len(next(iter(returns.values()))))]


def quantile_type7(values, p):
    x = sorted(values)
    h = (len(x) - 1) * p
    lo = math.floor(h)
    return x[lo] + (h - lo) * (x[min(lo + 1, len(x) - 1)] - x[lo])


def es_exact(returns, confidence):
    alpha = Fraction(1) - Fraction(str(confidence))
    losses = sorted((-Fraction(repr(r)) for r in returns), reverse=True)
    m = alpha * len(losses)
    k = math.floor(m)
    total = sum(losses[:k], Fraction(0)) + ((m - k) * losses[k] if m > k else 0)
    return float(total / m)


def portfolio_variance(weights, returns):
    """w' Sigma w with Sigma from statistics.covariance (ddof = 1)."""
    return sum(weights[a] * weights[b] * statistics.covariance(returns[a], returns[b])
               for a in weights for b in weights)


def summary(data, weights, **kwargs):
    return calculate_portfolio_risk_summary(data, weights, **kwargs)


TWO = combine(market("A", RA), market("B", RB))
THREE = combine(market("A", RA), market("B", RB), market("C", RC))
RETURNS = {"A": RA, "B": RB, "C": RC}


# 1-5. portfolio shapes and the return series --------------------------------------------------------

def test_two_stock_return_series_worked_example():
    data = combine(market("A", [0.02, -0.01, 0.03]), market("B", [-0.01, 0.02, 0.0]))
    series = calculate_portfolio_return_series(data, {"A": 0.6, "B": 0.4})
    assert list(series.columns) == RETURN_SERIES_COLUMNS
    assert list(series["date"]) == list(pd.bdate_range("2026-01-02", periods=3))
    assert series["portfolio_return"].tolist() == pytest.approx([0.008, 0.002, 0.018], abs=1e-12)
    result = summary(data, {"A": 0.6, "B": 0.4}, min_observations=2)
    assert result.cumulative_return == pytest.approx(1.008 * 1.002 * 1.018 - 1, rel=1e-9)


def test_three_stock_return_series():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    series = calculate_portfolio_return_series(THREE, weights)
    assert series["portfolio_return"].tolist() == pytest.approx(expected_series(weights, RETURNS),
                                                                 abs=1e-12)
    assert len(series) == 25


def test_single_stock_portfolio_matches_the_stock():
    result = summary(TWO, {"A": 1.0})
    assert result.return_series["portfolio_return"].tolist() == pytest.approx(RA, abs=1e-12)
    assert result.daily_volatility == pytest.approx(statistics.stdev(RA), rel=1e-9)
    stock_var = calculate_var_summary(market("A", RA)).iloc[0]
    assert result.historical_var == pytest.approx(stock_var["historical_var"], rel=1e-12)
    assert list(result.holdings["symbol"]) == ["A"]
    assert result.hhi == 1.0 and result.maximum_weight == 1.0


def test_equal_weights():
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    result = summary(THREE, weights)
    expected = [(a + b + c) / 3 for a, b, c in zip(RA, RB, RC)]
    assert result.return_series["portfolio_return"].tolist() == pytest.approx(expected, abs=1e-12)
    assert result.hhi == pytest.approx(1 / 3, rel=1e-12)


def test_unequal_weights():
    weights = {"A": 0.7, "B": 0.1, "C": 0.2}
    result = summary(THREE, weights)
    assert result.return_series["portfolio_return"].tolist() == pytest.approx(
        expected_series(weights, RETURNS), abs=1e-12)
    assert result.weights == {"A": 0.7, "B": 0.1, "C": 0.2}


def test_weights_accepted_as_table_or_pairs_and_symbols_trimmed():
    table = pd.DataFrame({"symbol": ["B", "A"], "weight": [0.4, 0.6]})
    pairs = [(" A ", 0.6), ("B", 0.4)]
    expected = calculate_portfolio_return_series(TWO, {"A": 0.6, "B": 0.4})
    for weights in (table, pairs):
        pd.testing.assert_frame_equal(calculate_portfolio_return_series(TWO, weights), expected)


# 6-12. weight validation ----------------------------------------------------------------------------

def test_weights_not_summing_to_one_raise_and_are_not_normalized():
    with pytest.raises(ValueError, match="sum to 1"):
        summary(TWO, {"A": 0.6, "B": 0.3})
    with pytest.raises(ValueError, match="sum to 1"):
        summary(TWO, {"A": 60, "B": 40})            # percentages are not silently rescaled


def test_weight_sum_tolerance_is_documented_and_applied():
    assert WEIGHT_SUM_TOLERANCE == 1e-6
    ok = validate_weights({"A": 0.6 + 0.5e-6, "B": 0.4})
    assert ok["A"] == 0.6 + 0.5e-6                    # kept as given, not normalized
    with pytest.raises(ValueError, match="sum to 1"):
        validate_weights({"A": 0.6 + 2e-6, "B": 0.4})


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="negative"):
        summary(TWO, {"A": 1.2, "B": -0.2})


def test_zero_weight_is_listed_but_does_not_restrict_dates():
    # B has a gap; with weight 0 it must not remove A's dates
    gappy = market("B", RB).drop(index=[5]).reset_index(drop=True)
    data = combine(market("A", RA), gappy)
    result = summary(data, {"A": 1.0, "B": 0.0})
    assert result.observations == 25
    assert result.return_series["portfolio_return"].tolist() == pytest.approx(RA, abs=1e-12)
    assert list(result.holdings["symbol"]) == ["A", "B"]
    assert result.hhi == 1.0


@pytest.mark.parametrize("bad", [np.nan, float("nan"), None, "0.5"])
def test_nan_or_non_numeric_weight_raises(bad):
    with pytest.raises(ValueError, match="finite number"):
        summary(TWO, {"A": 0.5, "B": bad})


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_weight_raises(bad):
    with pytest.raises(ValueError, match="finite number"):
        summary(TWO, {"A": 0.5, "B": bad})


def test_duplicate_symbol_raises():
    with pytest.raises(ValueError, match="Duplicate symbol.*A"):
        summary(TWO, [("A", 0.5), ("A", 0.5)])
    with pytest.raises(ValueError, match="Duplicate"):
        summary(TWO, pd.DataFrame({"symbol": ["A", "A "], "weight": [0.5, 0.5]}))


def test_missing_symbol_raises():
    with pytest.raises(ValueError, match="not found in the market data: ZZZ"):
        summary(TWO, {"A": 0.5, "ZZZ": 0.5})


def test_empty_or_malformed_weights_raise():
    with pytest.raises(ValueError, match="no holdings"):
        summary(TWO, {})
    with pytest.raises(ValueError, match="symbol"):
        summary(TWO, pd.DataFrame({"ticker": ["A"], "w": [1.0]}))
    with pytest.raises(ValueError, match="non-empty symbol"):
        summary(TWO, {"": 1.0})


def test_only_portfolio_symbols_participate():
    # C has a gap that would remove dates if it were (wrongly) included
    gappy_c = market("C", RC).drop(index=[3]).reset_index(drop=True)
    data = combine(market("A", RA), market("B", RB), gappy_c)
    result = summary(data, {"A": 0.5, "B": 0.5})
    assert result.observations == 25
    assert list(result.covariance.columns) == ["A", "B"]


# 13-16. data handling -------------------------------------------------------------------------------

def test_unsorted_data_gives_the_same_result():
    shuffled = THREE.sample(frac=1, random_state=7).reset_index(drop=True)
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    pd.testing.assert_frame_equal(calculate_portfolio_return_series(shuffled, weights),
                                  calculate_portfolio_return_series(THREE, weights))
    a, b = summary(shuffled, weights), summary(THREE, weights)
    assert a.annualized_volatility == b.annualized_volatility
    assert a.historical_cvar == b.historical_cvar


def test_invalid_rows_are_excluded_and_never_bridged():
    status = ["VALID"] * 26
    status[3] = "INVALID"
    data = combine(market("A", RA, status=status), market("B", RB, status=["VALID"] * 26))
    dates = pd.bdate_range("2026-01-01", periods=26)
    series = calculate_portfolio_return_series(data, {"A": 0.5, "B": 0.5})
    # A has no return on day 3 (INVALID) or day 4 (would bridge over it)
    assert dates[3] not in set(series["date"]) and dates[4] not in set(series["date"])
    assert len(series) == 23
    kept = [t for t in range(25) if t + 1 not in (3, 4)]
    assert series["portfolio_return"].tolist() == pytest.approx(
        [0.5 * RA[t] + 0.5 * RB[t] for t in kept], abs=1e-12)


def test_warning_rows_are_usable():
    data = combine(market("A", RA, status=["WARNING"] * 26), market("B", RB, status=["VALID"] * 26))
    assert len(calculate_portfolio_return_series(data, {"A": 0.5, "B": 0.5})) == 25


def test_missing_common_dates_are_excluded_without_filling():
    gappy = market("B", RB).drop(index=[10]).reset_index(drop=True)
    data = combine(market("A", RA), gappy)
    result = summary(data, {"A": 0.5, "B": 0.5})
    dates = pd.bdate_range("2026-01-01", periods=26)
    used = set(result.return_series["date"])
    assert dates[10] not in used        # B has no row: missing
    assert dates[11] not in used        # B's return spans two days: different period
    assert result.observations == 23
    assert result.excluded_missing == 1 and result.excluded_misaligned == 1
    assert 0.0 not in result.return_series["portfolio_return"].tolist()


def test_insufficient_observations_give_nan_with_count():
    data = combine(market("A", [0.01]), market("B", [0.02]))
    result = summary(data, {"A": 0.5, "B": 0.5})
    assert result.observations == 1
    for value in (result.annualized_volatility, result.daily_volatility, result.cumulative_return,
                  result.annualized_arithmetic_return, result.maximum_drawdown,
                  result.historical_var, result.parametric_cvar):
        assert math.isnan(value)

    short = combine(market("A", RA[:10]), market("B", RB[:10]))
    result = summary(short, {"A": 0.5, "B": 0.5})
    assert result.observations == 10 and not result.sufficient_tail_data
    assert math.isnan(result.historical_var) and math.isnan(result.parametric_var)
    assert math.isnan(result.historical_cvar) and math.isnan(result.parametric_cvar)
    assert not math.isnan(result.annualized_volatility)    # still has >= 2 observations


def test_no_common_dates_gives_empty_series_and_nan():
    a = market("A", RA[:3])
    b = market("B", RB[:3], dates=pd.bdate_range("2026-03-02", periods=4))
    result = summary(combine(a, b), {"A": 0.5, "B": 0.5})
    assert result.observations == 0 and result.return_series.empty
    assert result.start_date is None and math.isnan(result.annualized_volatility)


# 17-21. return, volatility and drawdown ------------------------------------------------------------

def test_return_summary():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    rp = expected_series(weights, RETURNS)
    result = summary(THREE, weights)
    cumulative = math.prod(1 + r for r in rp) - 1
    assert result.observations == 25
    assert result.start_date == pd.Timestamp("2026-01-02")
    assert result.end_date == pd.bdate_range("2026-01-01", periods=26)[-1]
    assert result.mean_daily_return == pytest.approx(statistics.mean(rp), rel=1e-9)
    assert result.annualized_arithmetic_return == pytest.approx(statistics.mean(rp) * 252, rel=1e-9)
    assert result.cumulative_return == pytest.approx(cumulative, rel=1e-9)
    assert result.annualized_geometric_return == pytest.approx(
        (1 + cumulative) ** (252 / 25) - 1, rel=1e-9)


def test_volatility_uses_covariance_not_weighted_average():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = summary(THREE, weights)
    variance = portfolio_variance(weights, RETURNS)
    weighted_average = sum(weights[s] * statistics.stdev(RETURNS[s]) for s in weights)

    assert result.daily_variance == pytest.approx(variance, rel=1e-9)
    assert result.daily_volatility == pytest.approx(math.sqrt(variance), rel=1e-9)
    assert result.annualized_volatility == pytest.approx(math.sqrt(variance * 252), rel=1e-9)
    # identical to the standard deviation of the portfolio's own return series
    assert result.daily_volatility == pytest.approx(
        statistics.stdev(expected_series(weights, RETURNS)), rel=1e-9)
    # and clearly not the weighted average of stock volatilities
    assert result.daily_volatility < weighted_average * 0.9


def test_perfectly_correlated_stocks_volatility_equals_weighted_average():
    rb = [2 * r for r in RA]                       # B moves exactly with A (correlation +1)
    data = combine(market("A", RA), market("B", rb))
    result = summary(data, {"A": 0.6, "B": 0.4})
    expected = 0.6 * statistics.stdev(RA) + 0.4 * statistics.stdev(rb)
    assert result.daily_volatility == pytest.approx(expected, rel=1e-9)


def test_negatively_correlated_stocks_diversify_away_volatility():
    rb = [-r for r in RA]                          # correlation -1
    data = combine(market("A", RA), market("B", rb))
    hedged = summary(data, {"A": 0.5, "B": 0.5})
    assert hedged.daily_volatility == pytest.approx(0.0, abs=1e-12)
    tilted = summary(data, {"A": 0.7, "B": 0.3})
    weighted_average = statistics.stdev(RA)        # both stocks have the same volatility
    assert tilted.daily_volatility == pytest.approx(0.4 * statistics.stdev(RA), rel=1e-9)
    assert tilted.daily_volatility < weighted_average


def test_drawdown_from_portfolio_value_path():
    ra = [0.10, -0.20, 0.05, 0.10, 0.20, -0.05]
    rb = [0.00, 0.10, -0.10, 0.00, 0.00, 0.02]
    data = combine(market("A", ra), market("B", rb))
    weights = {"A": 0.5, "B": 0.5}
    result = summary(data, weights, min_observations=2)

    rp = [0.5 * a + 0.5 * b for a, b in zip(ra, rb)]       # 0.05, -0.05, -0.025, 0.05, 0.10, -0.015
    values = [1.0]
    for r in rp:
        values.append(values[-1] * (1 + r))
    dates = list(pd.bdate_range("2026-01-01", periods=7))
    peak, worst, worst_at, peak_at = values[0], 0.0, None, None
    running_peak_at = 0
    for t, v in enumerate(values):
        if v >= peak:
            peak, running_peak_at = v, t
        if v / peak - 1 < worst:
            worst, worst_at, peak_at = v / peak - 1, t, running_peak_at
    recovery = next(t for t in range(worst_at + 1, 7) if values[t] >= values[peak_at])

    assert result.value_path["value"].tolist() == pytest.approx(values, rel=1e-12)
    assert result.value_path["date"].iloc[0] == dates[0]          # starts at 1.0 on the base date
    assert result.maximum_drawdown == pytest.approx(worst, rel=1e-9)
    assert result.peak_date == dates[peak_at] and result.peak_value == pytest.approx(values[peak_at])
    assert result.trough_date == dates[worst_at]
    assert result.trough_value == pytest.approx(values[worst_at])
    assert result.recovery_date == dates[recovery]


def test_drawdown_is_not_an_average_of_stock_drawdowns():
    weights = {"A": 0.5, "B": 0.5}
    result = summary(TWO, weights)
    stock = calculate_maximum_drawdown(TWO).set_index("symbol")["maximum_drawdown"]
    averaged = 0.5 * stock["A"] + 0.5 * stock["B"]
    assert result.maximum_drawdown != pytest.approx(averaged, rel=1e-3)
    assert result.maximum_drawdown > averaged       # B cushions A's losses


def test_drawdown_can_start_on_the_base_date():
    data = combine(market("A", [-0.1, 0.05]), market("B", [-0.1, 0.05]))
    result = summary(data, {"A": 0.5, "B": 0.5}, min_observations=2)
    assert result.peak_date == pd.Timestamp("2026-01-01") and result.peak_value == 1.0
    assert result.maximum_drawdown == pytest.approx(-0.1, rel=1e-9)
    assert pd.isna(result.recovery_date)


# 22-25. VaR / CVaR from the portfolio series --------------------------------------------------------

WEIGHTS_ABC = {"A": 0.5, "B": 0.3, "C": 0.2}
RP = expected_series(WEIGHTS_ABC, RETURNS)


def test_historical_var_from_independent_portfolio_series():
    result = summary(THREE, WEIGHTS_ABC)
    assert result.sufficient_tail_data
    assert result.historical_var == pytest.approx(-quantile_type7(RP, 0.05), rel=1e-9)
    stock = calculate_var_summary(THREE).set_index("symbol")["historical_var"]
    weighted = sum(WEIGHTS_ABC[s] * stock[s] for s in WEIGHTS_ABC)
    assert result.historical_var != pytest.approx(weighted, rel=1e-3)


def test_parametric_var_from_independent_portfolio_series():
    result = summary(THREE, WEIGHTS_ABC)
    expected = -(statistics.mean(RP) + ND.inv_cdf(0.05) * statistics.stdev(RP))
    assert result.parametric_var == pytest.approx(expected, rel=1e-9)


def test_historical_cvar_from_independent_portfolio_series():
    result = summary(THREE, WEIGHTS_ABC)
    assert result.historical_cvar == pytest.approx(es_exact(RP, 0.95), rel=1e-9)
    assert result.historical_cvar >= result.historical_var


def test_parametric_cvar_from_independent_portfolio_series():
    result = summary(THREE, WEIGHTS_ABC)
    z = ND.inv_cdf(0.05)
    expected = -(statistics.mean(RP) - statistics.stdev(RP) * ND.pdf(z) / 0.05)
    assert result.parametric_cvar == pytest.approx(expected, rel=1e-9)
    assert result.parametric_cvar >= result.parametric_var


def test_custom_confidence_level():
    result = summary(THREE, WEIGHTS_ABC, confidence_level=0.90)
    assert result.confidence_level == 0.90
    assert result.historical_var == pytest.approx(-quantile_type7(RP, 0.10), rel=1e-9)
    assert result.historical_cvar == pytest.approx(es_exact(RP, 0.90), rel=1e-9)
    expected = -(statistics.mean(RP) + ND.inv_cdf(0.10) * statistics.stdev(RP))
    assert result.parametric_var == pytest.approx(expected, rel=1e-9)


def test_custom_min_observations():
    assert not summary(THREE, WEIGHTS_ABC, min_observations=26).sufficient_tail_data
    assert summary(THREE, WEIGHTS_ABC, min_observations=25).sufficient_tail_data


def test_custom_periods_per_year():
    result = summary(THREE, WEIGHTS_ABC, periods_per_year=52)
    variance = portfolio_variance(WEIGHTS_ABC, RETURNS)
    assert result.annualized_volatility == pytest.approx(math.sqrt(variance * 52), rel=1e-9)
    assert result.annualized_arithmetic_return == pytest.approx(statistics.mean(RP) * 52, rel=1e-9)
    cumulative = math.prod(1 + r for r in RP) - 1
    assert result.annualized_geometric_return == pytest.approx(
        (1 + cumulative) ** (52 / 25) - 1, rel=1e-9)


@pytest.mark.parametrize("kwargs", [{"confidence_level": 1.0}, {"min_observations": 1},
                                    {"periods_per_year": 0}, {"portfolio_value": 0},
                                    {"portfolio_value": float("nan")},
                                    {"portfolio_value": 1e6, "participation_rate": 0}])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        summary(THREE, WEIGHTS_ABC, **kwargs)


# 26-27. concentration -------------------------------------------------------------------------------

def test_maximum_weight():
    assert summary(THREE, {"A": 0.2, "B": 0.45, "C": 0.35}).maximum_weight == 0.45


def test_hhi():
    result = summary(THREE, {"A": 0.2, "B": 0.45, "C": 0.35})
    assert result.hhi == pytest.approx(0.2 ** 2 + 0.45 ** 2 + 0.35 ** 2, rel=1e-12)   # 0.365
    assert summary(THREE, {"A": 0.25, "B": 0.25, "C": 0.5}).hhi == pytest.approx(0.375)


# 28-31. liquidity -----------------------------------------------------------------------------------

def test_portfolio_value_converts_to_position_values():
    result = summary(THREE, WEIGHTS_ABC, portfolio_value=20_000_000)
    positions = result.holdings.set_index("symbol")["position_value"]
    assert positions.to_dict() == pytest.approx({"A": 10_000_000, "B": 6_000_000, "C": 4_000_000})
    assert result.portfolio_value == 20_000_000 and result.participation_rate == 0.10


def test_no_liquidity_without_portfolio_value():
    result = summary(THREE, WEIGHTS_ABC)
    assert list(result.holdings.columns) == ["symbol", "weight"]
    assert result.most_illiquid_symbol is None
    assert math.isnan(result.maximum_estimated_liquidation_days)


def test_liquidity_with_actual_turnover():
    turnover = [5e6 + 1e5 * i for i in range(26)]
    a = market("A", RA, turnover=turnover)
    b = market("B", RB, turnover=[2e6] * 26)
    result = summary(combine(a, b), {"A": 0.5, "B": 0.5}, portfolio_value=10_000_000)
    rows = result.holdings.set_index("symbol")
    adtv_a = statistics.mean(turnover)                   # 6,250,000
    assert rows.loc["A", "traded_value_source"] == ACTUAL_TURNOVER
    assert rows.loc["A", "average_daily_traded_value"] == pytest.approx(adtv_a)
    assert rows.loc["A", "position_to_adtv"] == pytest.approx(5_000_000 / adtv_a)
    assert rows.loc["A", "estimated_liquidation_days"] == pytest.approx(5_000_000 / (adtv_a * 0.10))
    assert rows.loc["B", "estimated_liquidation_days"] == pytest.approx(5_000_000 / 200_000)  # 25


def test_liquidity_with_missing_turnover_uses_estimated_traded_value():
    turnover = [5e6] * 26
    turnover[4] = None                                    # one missing day -> never mixed
    volumes = [1_000 + 10 * i for i in range(26)]
    a = market("A", RA, volume=volumes, turnover=turnover)
    result = summary(combine(a, market("B", RB, turnover=[2e6] * 26)), {"A": 0.5, "B": 0.5},
                     portfolio_value=10_000_000)
    row = result.holdings.set_index("symbol").loc["A"]
    closes = a["close"].tolist()
    adtv = statistics.mean(c * v for c, v in zip(closes, volumes))
    assert row["traded_value_source"] == ESTIMATED_TRADED_VALUE
    assert row["average_daily_traded_value"] == pytest.approx(adtv, rel=1e-12)
    assert row["estimated_liquidation_days"] == pytest.approx(5_000_000 / (adtv * 0.10), rel=1e-12)


def test_liquidity_with_estimated_traded_value_no_turnover_column():
    data = combine(market("A", RA, volume=2_000.0), market("B", RB, volume=500.0))
    result = summary(data, {"A": 0.25, "B": 0.75}, portfolio_value=1_000_000, participation_rate=0.2)
    rows = result.holdings.set_index("symbol")
    assert set(rows["traded_value_source"]) == {ESTIMATED_TRADED_VALUE}
    adtv_b = statistics.mean(c * 500.0 for c in market("B", RB)["close"])
    assert rows.loc["B", "position_to_adtv"] == pytest.approx(750_000 / adtv_b, rel=1e-12)
    assert rows.loc["B", "estimated_liquidation_days"] == pytest.approx(
        750_000 / (adtv_b * 0.2), rel=1e-12)


def test_liquidity_summary_is_descriptive_not_summed():
    result = summary(THREE, WEIGHTS_ABC, portfolio_value=20_000_000)
    days = result.holdings.set_index("symbol")["estimated_liquidation_days"]
    ratios = result.holdings.set_index("symbol")["position_to_adtv"]
    assert result.most_illiquid_symbol == days.idxmax()
    assert result.maximum_estimated_liquidation_days == days.max()
    assert result.maximum_position_to_adtv == ratios.max()
    field_names = set(PortfolioRiskResult.__dataclass_fields__)
    assert not any("total" in f or "portfolio_liquidation" in f for f in field_names)


def test_zero_weight_holding_has_nothing_to_liquidate():
    result = summary(THREE, {"A": 0.6, "B": 0.4, "C": 0.0}, portfolio_value=1_000_000)
    row = result.holdings.set_index("symbol").loc["C"]
    assert row["position_value"] == 0 and row["estimated_liquidation_days"] == 0.0
    assert result.most_illiquid_symbol in {"A", "B"}


def test_zero_volume_holding_has_undefined_liquidation_days():
    data = combine(market("A", RA), market("B", RB, volume=0.0))
    result = summary(data, {"A": 0.5, "B": 0.5}, portfolio_value=1_000_000)
    assert math.isnan(result.holdings.set_index("symbol").loc["B", "estimated_liquidation_days"])
    assert result.undefined_liquidation_symbols == ("B",)
    assert result.most_illiquid_symbol == "A"


# 32. no mutation; sample data -----------------------------------------------------------------------

def test_inputs_are_not_mutated():
    data = THREE.sample(frac=1, random_state=3).reset_index(drop=True)
    before = data.copy(deep=True)
    weights = {"C": 0.2, "A": 0.5, "B": 0.3}
    table = pd.DataFrame({"symbol": ["A", "B", "C"], "weight": [0.5, 0.3, 0.2]})
    table_before = table.copy(deep=True)
    summary(data, weights, portfolio_value=1_000_000)
    summary(data, table)
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(table, table_before)
    assert weights == {"C": 0.2, "A": 0.5, "B": 0.3}


def test_sample_three_stock_dataset():
    data = load_csv_market_data(SAMPLE_3_SYMBOLS).data
    weights = {"ABC": 0.4, "LMN": 0.25, "XYZ": 0.35}
    result = summary(data, weights, portfolio_value=20_000_000)
    assert result.observations == 24 and result.sufficient_tail_data
    closes = {s: g.sort_values("date")["close"].tolist() for s, g in data.groupby("symbol")}
    returns = {s: [c[i + 1] / c[i] - 1 for i in range(len(c) - 1)] for s, c in closes.items()}
    rp = expected_series(weights, returns)
    assert result.return_series["portfolio_return"].tolist() == pytest.approx(rp, abs=1e-12)
    assert result.annualized_volatility == pytest.approx(
        math.sqrt(portfolio_variance(weights, returns) * 252), rel=1e-9)
    assert result.historical_var == pytest.approx(-quantile_type7(rp, 0.05), rel=1e-9)
    assert result.historical_cvar == pytest.approx(es_exact(rp, 0.95), rel=1e-9)
    assert set(result.holdings["traded_value_source"]) == {ACTUAL_TURNOVER}
