"""Tests for app.analytics.ratios (Sharpe and Sortino), checked against plain-Python calculations."""

import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics.ratios import (
    RATIO_COLUMNS,
    SHARPE_COLUMNS,
    SORTINO_COLUMNS,
    calculate_risk_adjusted_ratios,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    daily_risk_free_rate,
)
from app.data.loaders.csv_market_loader import load_csv_market_data

R = [0.02, -0.01, 0.03, -0.02, 0.01]
SAMPLE_3_SYMBOLS = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"


def prices(symbol, returns, start=100.0, status=None):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    frame = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=len(closes)),
                          "symbol": symbol, "close": closes})
    if status is not None:
        frame["validation_status"] = status
    return frame


def row(data, symbol="ABC", **kwargs):
    return calculate_risk_adjusted_ratios(data, **kwargs).set_index("symbol").loc[symbol]


def expected(returns, rf=0.0, periods=252):
    """Independent reference implementation of the documented conventions."""
    rf_d = (1 + rf) ** (1 / periods) - 1
    excess = [r - rf_d for r in returns]
    annual_excess = statistics.fmean(excess) * periods
    annual_vol = statistics.stdev(returns) * math.sqrt(periods)
    downside = math.sqrt(sum(min(e, 0) ** 2 for e in excess) / len(excess)) * math.sqrt(periods)
    growth = 1.0
    for r in returns:
        growth *= 1 + r
    return {"excess": annual_excess, "vol": annual_vol, "sharpe": annual_excess / annual_vol,
            "downside": downside, "sortino": annual_excess / downside,
            "annual_return": growth ** (periods / len(returns)) - 1}


# Worked example

def test_sharpe_with_zero_risk_free_rate_matches_hand_calculation():
    result = row(prices("ABC", R))
    assert result["annualized_excess_return"] == pytest.approx(1.512, rel=1e-12)
    assert result["annualized_volatility"] == pytest.approx(0.3291808013842849, rel=1e-12)
    assert result["sharpe_ratio"] == pytest.approx(4.593220484431881, rel=1e-12)


def test_sortino_with_known_upside_and_downside_returns():
    result = row(prices("ABC", R))
    assert result["downside_deviation"] == pytest.approx(0.01 * math.sqrt(252), rel=1e-12)
    assert result["sortino_ratio"] == pytest.approx(9.524704719832526, rel=1e-12)


def test_independent_reference_agrees_for_several_series():
    for returns in (R, [0.004, -0.012, 0.009, 0.0, -0.003, 0.011, -0.007],
                    [-0.02, -0.01, 0.005, -0.015]):
        result = row(prices("ABC", returns), risk_free_rate=0.07)
        ref = expected(returns, rf=0.07)
        assert result["annualized_excess_return"] == pytest.approx(ref["excess"], rel=1e-9)
        assert result["sharpe_ratio"] == pytest.approx(ref["sharpe"], rel=1e-9)
        assert result["downside_deviation"] == pytest.approx(ref["downside"], rel=1e-9)
        assert result["sortino_ratio"] == pytest.approx(ref["sortino"], rel=1e-9)
        assert result["annualized_return"] == pytest.approx(ref["annual_return"], rel=1e-9)


# Risk-free rate

def test_daily_risk_free_rate_conversion_compounds_back_to_the_annual_rate():
    rf_d = daily_risk_free_rate(0.05)
    assert rf_d == pytest.approx(1.05 ** (1 / 252) - 1, rel=1e-15)
    assert (1 + rf_d) ** 252 == pytest.approx(1.05, rel=1e-12)
    assert daily_risk_free_rate(0.0) == 0.0


def test_sharpe_with_non_zero_risk_free_rate():
    result = row(prices("ABC", R), risk_free_rate=0.05)
    rf_d = 1.05 ** (1 / 252) - 1
    assert result["daily_risk_free_rate"] == pytest.approx(rf_d, rel=1e-15)
    assert result["annualized_excess_return"] == pytest.approx((0.006 - rf_d) * 252, rel=1e-12)
    assert result["sharpe_ratio"] == pytest.approx((0.006 - rf_d) * 252 / 0.3291808013842849,
                                                   rel=1e-12)


def test_positive_risk_free_rate_reduces_excess_return_and_ratios():
    zero, positive = row(prices("ABC", R)), row(prices("ABC", R), risk_free_rate=0.10)
    shift = ((1.10 ** (1 / 252)) - 1) * 252
    assert positive["annualized_excess_return"] == pytest.approx(
        zero["annualized_excess_return"] - shift, rel=1e-12)
    assert positive["sharpe_ratio"] < zero["sharpe_ratio"]
    assert positive["sortino_ratio"] < zero["sortino_ratio"]
    assert positive["annualized_volatility"] == zero["annualized_volatility"]   # rf doesn't move risk


def test_negative_risk_free_rate_is_supported():
    result = row(prices("ABC", R), risk_free_rate=-0.01)
    assert result["daily_risk_free_rate"] < 0
    assert result["sharpe_ratio"] == pytest.approx(expected(R, rf=-0.01)["sharpe"], rel=1e-9)
    assert result["sharpe_ratio"] > row(prices("ABC", R))["sharpe_ratio"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, -1.5, "0.05", None, True])
def test_invalid_risk_free_rate_fails_clearly(bad):
    with pytest.raises(ValueError, match="risk_free_rate"):
        calculate_risk_adjusted_ratios(prices("ABC", R), risk_free_rate=bad)


# Denominators

def test_sharpe_uses_total_standard_deviation():
    result = row(prices("ABC", R))
    assert result["annualized_volatility"] == pytest.approx(statistics.stdev(R) * math.sqrt(252))
    assert result["sharpe_ratio"] == pytest.approx(
        result["annualized_excess_return"] / result["annualized_volatility"])


def test_sortino_uses_downside_deviation_not_total_volatility():
    # Same downside, much larger upside: total volatility rises, downside does not.
    calm = row(prices("ABC", [0.01, -0.01, 0.01, -0.01]))
    spiky = row(prices("ABC", [0.08, -0.01, 0.08, -0.01]))
    assert spiky["downside_deviation"] == pytest.approx(calm["downside_deviation"], rel=1e-12)
    assert spiky["annualized_volatility"] > calm["annualized_volatility"]
    assert spiky["downside_deviation"] != pytest.approx(spiky["annualized_volatility"])
    # mean over ALL observations: sqrt((0.01^2 + 0.01^2) / 4) * sqrt(252)
    assert calm["downside_deviation"] == pytest.approx(math.sqrt(0.0002 / 4) * math.sqrt(252))


# Periods per year

def test_custom_periods_per_year_is_used_everywhere():
    result = row(prices("ABC", R), risk_free_rate=0.05, periods_per_year=52)
    ref = expected(R, rf=0.05, periods=52)
    assert result["periods_per_year"] == 52
    assert result["annualized_excess_return"] == pytest.approx(ref["excess"], rel=1e-12)
    assert result["annualized_volatility"] == pytest.approx(ref["vol"], rel=1e-12)
    assert result["downside_deviation"] == pytest.approx(ref["downside"], rel=1e-12)
    assert result["sharpe_ratio"] == pytest.approx(ref["sharpe"], rel=1e-12)


@pytest.mark.parametrize("bad", [0, -252, "252", None, True, float("nan"), float("inf")])
def test_invalid_periods_per_year_fails(bad):
    with pytest.raises(ValueError, match="periods_per_year"):
        calculate_risk_adjusted_ratios(prices("ABC", R), periods_per_year=bad)


# Undefined ratios and negative returns

def test_zero_volatility_gives_nan_not_infinity():
    result = row(prices("ABC", [0.0, 0.0, 0.0]), risk_free_rate=0.05)
    assert math.isnan(result["sharpe_ratio"])
    assert not np.isinf(result["sharpe_ratio"])


def test_float_noise_volatility_gives_nan():
    # 100 -> 110 -> 121 -> 133.1: returns are 10% up to floating-point noise.
    data = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=4), "symbol": "ABC",
                         "close": [100, 110, 121, 133.1]})
    assert math.isnan(row(data)["sharpe_ratio"])


def test_no_downside_observations_makes_sortino_nan():
    result = row(prices("ABC", [0.01, 0.02, 0.015, 0.03]))
    assert result["downside_deviation"] == 0.0
    assert math.isnan(result["sortino_ratio"])
    assert result["sharpe_ratio"] > 0                       # Sharpe is still defined


def test_negative_returns_give_negative_ratios():
    falling = [-0.01, -0.02, 0.005, -0.015]
    result = row(prices("ABC", falling))
    ref = expected(falling)
    assert result["annualized_excess_return"] < 0
    assert result["sharpe_ratio"] == pytest.approx(ref["sharpe"], rel=1e-9) and result["sharpe_ratio"] < 0
    assert result["sortino_ratio"] == pytest.approx(ref["sortino"], rel=1e-9) and result["sortino_ratio"] < 0


# Symbols, ordering, validation status

def test_multiple_symbols_are_independent():
    xyz = [0.01, 0.004, -0.006, 0.012]
    data = pd.concat([prices("ABC", R), prices("XYZ", xyz, start=10)])
    result = calculate_risk_adjusted_ratios(data).set_index("symbol")

    assert list(result.index) == ["ABC", "XYZ"]
    assert result.loc["ABC", "sharpe_ratio"] == pytest.approx(expected(R)["sharpe"], rel=1e-9)
    assert result.loc["XYZ", "sharpe_ratio"] == pytest.approx(expected(xyz)["sharpe"], rel=1e-9)


def test_unsorted_data_is_handled():
    data = pd.concat([prices("ABC", R), prices("XYZ", [0.01, -0.01, 0.02])]).sample(frac=1, random_state=9)
    assert row(data)["sharpe_ratio"] == pytest.approx(4.593220484431881, rel=1e-12)


def test_invalid_rows_are_excluded_and_not_bridged():
    # Close 999 on day 3 is INVALID: both returns touching it are excluded, and
    # no return is calculated across it.
    closes = [100, 102, 101, 999, 103, 104, 102]
    status = ["VALID", "VALID", "VALID", "INVALID", "VALID", "VALID", "VALID"]
    data = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=7), "symbol": "ABC",
                         "close": closes, "validation_status": status})
    usable = [102 / 100 - 1, 101 / 102 - 1, 104 / 103 - 1, 102 / 104 - 1]

    result = row(data)

    assert result["observations"] == 4
    assert result["sharpe_ratio"] == pytest.approx(expected(usable)["sharpe"], rel=1e-9)
    assert result["sortino_ratio"] == pytest.approx(expected(usable)["sortino"], rel=1e-9)


def test_warning_rows_remain_usable():
    data = prices("ABC", R, status=["VALID", "WARNING", "VALID", "WARNING", "VALID", "VALID"])
    result = row(data)
    assert result["observations"] == 5
    assert result["sharpe_ratio"] == pytest.approx(4.593220484431881, rel=1e-12)


# Insufficient data, empty input, columns, mutation

@pytest.mark.parametrize("returns", [[], [0.01]])
def test_insufficient_observations_give_nan_not_zero(returns):
    result = row(prices("ABC", returns))
    assert result["observations"] == len(returns)
    for column in ("annualized_return", "annualized_excess_return", "sharpe_ratio",
                   "downside_deviation", "sortino_ratio"):
        assert math.isnan(result[column]), column


def test_empty_input_is_handled_clearly():
    empty = prices("ABC", R).iloc[0:0]
    assert list(calculate_risk_adjusted_ratios(empty).columns) == RATIO_COLUMNS
    assert calculate_sharpe_ratio(empty).empty and calculate_sortino_ratio(empty).empty


@pytest.mark.parametrize("dropped", ["date", "symbol", "close"])
def test_missing_required_columns_fail_clearly(dropped):
    with pytest.raises(ValueError, match=f"missing column\\(s\\): {dropped}"):
        calculate_sharpe_ratio(prices("ABC", R).drop(columns=dropped))


def test_input_is_not_mutated():
    data = prices("ABC", R, status="VALID").iloc[::-1]
    before = data.copy()
    calculate_risk_adjusted_ratios(data, risk_free_rate=0.05)
    calculate_sharpe_ratio(data)
    calculate_sortino_ratio(data)
    pd.testing.assert_frame_equal(data, before)


# Output shape and annualized return

def test_sharpe_and_sortino_outputs():
    assert list(calculate_sharpe_ratio(prices("ABC", R)).columns) == SHARPE_COLUMNS
    assert list(calculate_sortino_ratio(prices("ABC", R)).columns) == SORTINO_COLUMNS


def test_annualized_return_is_geometric_and_separate_from_excess_return():
    result = row(prices("ABC", R))
    growth = 1.02 * 0.99 * 1.03 * 0.98 * 1.01
    assert result["annualized_return"] == pytest.approx(growth ** (252 / 5) - 1, rel=1e-12)
    assert result["annualized_return"] != pytest.approx(result["annualized_excess_return"])


def test_three_symbol_sample_through_csv_importer():
    canonical = load_csv_market_data(SAMPLE_3_SYMBOLS).data
    result = calculate_risk_adjusted_ratios(canonical, risk_free_rate=0.08).set_index("symbol")
    assert list(result.index) == ["ABC", "LMN", "XYZ"]
    for symbol in result.index:
        closes = list(canonical[canonical["symbol"] == symbol].sort_values("date")["close"])
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        assert result.loc[symbol, "sharpe_ratio"] == pytest.approx(
            expected(returns, rf=0.08)["sharpe"], rel=1e-9)
        assert result.loc[symbol, "sortino_ratio"] == pytest.approx(
            expected(returns, rf=0.08)["sortino"], rel=1e-9)
