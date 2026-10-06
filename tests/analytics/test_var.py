"""Tests for app.analytics.var (stock-level 1-day VaR).

Expected values are computed independently of the module under test:
  * historical: a hand-written Hyndman & Fan type-7 (linear) quantile in plain
    Python (``quantile7``), not numpy
  * parametric: statistics.mean / statistics.stdev and
    statistics.NormalDist().inv_cdf for z, not scipy

Worked example (R below, 20 returns)
  95%: alpha = 0.05; sorted lowest returns -0.05, -0.04, ...
       h = (20 - 1) * 0.05 = 0.95 -> quantile = -0.05 + 0.95 * (-0.04 - -0.05) = -0.0405
       historical VaR = 0.0405
       mean = -0.063 / 20 = -0.00315, sample sd = 0.0218253..., z_0.05 = -1.6448536...
       parametric VaR = -(-0.00315 - 1.6448536 * 0.0218253) = 0.0390494...
  99%: h = 19 * 0.01 = 0.19 -> quantile = -0.05 + 0.19 * 0.01 = -0.0481 -> 0.0481
       parametric VaR = -(-0.00315 - 2.3263479 * 0.0218253) = 0.0539232...
"""

import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics import var as var_module
from app.analytics.var import (
    HISTORICAL_COLUMNS,
    PARAMETRIC_COLUMNS,
    SUMMARY_COLUMNS,
    calculate_historical_var,
    calculate_parametric_var,
    calculate_var_summary,
)
from app.data.loaders.csv_market_loader import load_csv_market_data

R = [0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.015, -0.01, 0.005, -0.03,
     0.012, -0.008, 0.025, -0.015, 0.004, -0.022, 0.018, -0.006, 0.009, -0.04]
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


def quantile7(values, alpha):
    """Independent Hyndman & Fan type-7 (linear interpolation) quantile."""
    xs = sorted(values)
    h = (len(xs) - 1) * alpha
    lo = math.floor(h)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def expected(returns, confidence):
    alpha = 1 - confidence
    z = statistics.NormalDist().inv_cdf(alpha)
    return {"historical": -quantile7(returns, alpha),
            "parametric": -(statistics.mean(returns) + z * statistics.stdev(returns))}


def row(data, symbol="ABC", **kwargs):
    return calculate_var_summary(data, **kwargs).set_index("symbol").loc[symbol]


# 1-6, 18 of the brief's list: worked example ---------------------------------------------------------

def test_historical_var_on_known_returns():
    result = row(prices("ABC", R))
    assert result["historical_var"] == pytest.approx(0.0405, rel=1e-9)
    assert result["observations"] == 20


def test_parametric_var_on_known_returns():
    result = row(prices("ABC", R))
    assert statistics.mean(R) == pytest.approx(-0.00315)
    assert statistics.stdev(R) == pytest.approx(0.02182532329295732, rel=1e-12)
    assert result["parametric_var"] == pytest.approx(expected(R, 0.95)["parametric"], rel=1e-9)
    assert result["parametric_var"] == pytest.approx(0.03904946217780928, rel=1e-9)


def test_99_percent_confidence():
    result = row(prices("ABC", R), confidence_level=0.99)
    assert result["historical_var"] == pytest.approx(0.0481, rel=1e-9)
    assert result["parametric_var"] == pytest.approx(0.0539232944428253, rel=1e-9)


def test_historical_var_is_the_negated_lower_tail_quantile():
    result = row(prices("ABC", R))
    lower_tail = quantile7(R, 0.05)
    assert lower_tail < 0                                      # a loss
    assert result["historical_var"] == pytest.approx(-lower_tail, rel=1e-9)
    assert -quantile7(R, 0.95) < 0                             # the upper tail would be a gain
    assert result["historical_var"] > -quantile7(R, 0.5)       # deeper in the loss tail than the median


def test_var_is_a_positive_loss_magnitude():
    result = row(prices("ABC", R))
    assert result["historical_var"] > 0 and result["parametric_var"] > 0


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.975, 0.99])
def test_any_confidence_level_matches_independent_calculation(confidence):
    result = row(prices("ABC", R), confidence_level=confidence)
    ref = expected(R, confidence)
    assert result["confidence_level"] == confidence
    assert result["historical_var"] == pytest.approx(ref["historical"], rel=1e-9)
    assert result["parametric_var"] == pytest.approx(ref["parametric"], rel=1e-9)


# 7. invalid confidence ---------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, 1, -0.5, 1.5, 95, float("nan"), float("inf"), "0.95", None, True])
def test_invalid_confidence_levels_fail(bad):
    with pytest.raises(ValueError, match="confidence_level"):
        calculate_var_summary(prices("ABC", R), confidence_level=bad)


@pytest.mark.parametrize("bad", [0, 1, -5, 2.5, "20", True])
def test_invalid_min_observations_fail(bad):
    with pytest.raises(ValueError, match="min_observations"):
        calculate_var_summary(prices("ABC", R), min_observations=bad)


# 8-10. parametric ingredients and quantile method ---------------------------------------------------------

def test_parametric_var_uses_sample_standard_deviation():
    result = row(prices("ABC", R))
    z = statistics.NormalDist().inv_cdf(0.05)
    population = -(statistics.mean(R) + z * statistics.pstdev(R))
    assert result["parametric_var"] != pytest.approx(population, rel=1e-6)
    assert result["parametric_var"] == pytest.approx(
        -(statistics.mean(R) + z * statistics.stdev(R)), rel=1e-12)


def test_parametric_var_uses_the_normal_quantile_not_a_rounded_z():
    result = row(prices("ABC", R), confidence_level=0.95)
    with_rounded_z = -(statistics.mean(R) - 1.65 * statistics.stdev(R))
    exact_z = statistics.NormalDist().inv_cdf(0.05)
    assert exact_z == pytest.approx(-1.6448536269514722, rel=1e-12)
    assert result["parametric_var"] != pytest.approx(with_rounded_z, rel=1e-6)
    # an unusual level has no textbook z-score, so it must come from the distribution
    odd = row(prices("ABC", R), confidence_level=0.937)
    assert odd["parametric_var"] == pytest.approx(expected(R, 0.937)["parametric"], rel=1e-9)


def test_historical_quantile_method_is_explicitly_linear():
    assert var_module.QUANTILE_METHOD == "linear"
    # Returns chosen so the lower/nearest/linear conventions disagree.
    returns = [-0.10, -0.02] + [0.01] * 18
    result = row(prices("ABC", returns))
    assert result["historical_var"] == pytest.approx(-quantile7(returns, 0.05), rel=1e-9)   # 0.024
    assert result["historical_var"] == pytest.approx(0.024, rel=1e-9)
    assert result["historical_var"] != pytest.approx(0.10)    # 'lower' would give the minimum


# 11, 24-25. same sample, differing methods, tail ordering ------------------------------------------------------

def test_both_methods_use_the_same_returns():
    data = prices("ABC", R + [0.01, -0.02])
    summary = row(data)
    assert summary["observations"] == 22
    assert summary["historical_var"] == pytest.approx(expected(R + [0.01, -0.02], 0.95)["historical"])
    assert summary["parametric_var"] == pytest.approx(expected(R + [0.01, -0.02], 0.95)["parametric"])
    assert (calculate_historical_var(data)["observations"]
            == calculate_parametric_var(data)["observations"]).all()


def test_methods_can_differ():
    result = row(prices("ABC", R))
    assert result["historical_var"] != pytest.approx(result["parametric_var"], rel=1e-3)


def test_higher_confidence_gives_larger_var_and_expected_tail_difference():
    v95, v99 = row(prices("ABC", R)), row(prices("ABC", R), confidence_level=0.99)
    assert v99["historical_var"] >= v95["historical_var"]
    assert v99["parametric_var"] >= v95["parametric_var"]
    # parametric difference is exactly (z_0.05 - z_0.01) * sd
    nd = statistics.NormalDist()
    assert v99["parametric_var"] - v95["parametric_var"] == pytest.approx(
        (nd.inv_cdf(0.05) - nd.inv_cdf(0.01)) * statistics.stdev(R), rel=1e-9)


# 12-13. symbols and ordering -----------------------------------------------------------------------------------

def test_multiple_symbols_are_independent():
    other = [r * 0.5 + 0.001 for r in R]
    data = pd.concat([prices("ABC", R), prices("XYZ", other, start=10)])
    summary = calculate_var_summary(data).set_index("symbol")

    assert list(summary.index) == ["ABC", "XYZ"]
    assert summary.loc["ABC", "historical_var"] == pytest.approx(expected(R, 0.95)["historical"])
    assert summary.loc["XYZ", "historical_var"] == pytest.approx(expected(other, 0.95)["historical"])
    assert summary.loc["XYZ", "parametric_var"] == pytest.approx(expected(other, 0.95)["parametric"])


def test_unsorted_data_is_handled():
    data = pd.concat([prices("ABC", R), prices("XYZ", R[:5])]).sample(frac=1, random_state=4)
    assert row(data)["historical_var"] == pytest.approx(0.0405, rel=1e-9)


# 14-17. validation status and non-finite returns -----------------------------------------------------------------

def test_invalid_rows_are_excluded_and_not_bridged():
    # Day 6's close is INVALID (an absurd 1). Both returns touching it vanish;
    # no return is calculated across it.
    data = prices("ABC", R, status="VALID")
    data.loc[6, ["close", "validation_status"]] = [1.0, "INVALID"]
    usable = R[:5] + R[7:]          # R[5] (into day 6) and R[6] (out of day 6) are gone

    result = row(data, min_observations=18)

    assert result["observations"] == 18
    assert result["historical_var"] == pytest.approx(expected(usable, 0.95)["historical"], rel=1e-9)
    assert result["parametric_var"] == pytest.approx(expected(usable, 0.95)["parametric"], rel=1e-9)


def test_warning_rows_remain_usable():
    data = prices("ABC", R, status="WARNING")
    assert row(data)["historical_var"] == pytest.approx(0.0405, rel=1e-9)


def test_non_finite_returns_are_excluded():
    # Without validation_status the data is assumed validated, but a zero close
    # makes the next return infinite; it must not enter VaR.
    data = prices("ABC", R)
    data.loc[21] = [pd.Timestamp("2025-01-30"), "ABC", 0.0]
    data.loc[22] = [pd.Timestamp("2025-01-31"), "ABC", 50.0]
    result = row(data)
    assert result["observations"] == 21                         # 20 + the -100% return; inf dropped
    assert np.isfinite(result["historical_var"]) and np.isfinite(result["parametric_var"])


def test_rows_without_date_symbol_or_close_do_not_enter_var():
    data = prices("ABC", R)
    extra = pd.DataFrame({"date": [pd.NaT, pd.Timestamp("2025-02-03"), pd.Timestamp("2025-02-04")],
                          "symbol": ["ABC", None, "ABC"], "close": [500.0, 500.0, np.nan]})
    result = row(pd.concat([data, extra], ignore_index=True))
    assert result["observations"] == 20
    assert result["historical_var"] == pytest.approx(0.0405, rel=1e-9)


# 18-19. insufficient data and constant returns -------------------------------------------------------------------------

@pytest.mark.parametrize("returns", [[], [0.01], R[:5], R[:19]])
def test_insufficient_observations_give_nan_not_zero(returns):
    result = row(prices("ABC", returns))
    assert result["observations"] == len(returns)
    assert not result["sufficient_data"]
    assert math.isnan(result["historical_var"]) and math.isnan(result["parametric_var"])


def test_exactly_the_minimum_is_enough_and_minimum_is_configurable():
    assert row(prices("ABC", R))["sufficient_data"]                    # 20 of 20
    small = row(prices("ABC", R[:5]), min_observations=5)
    assert small["sufficient_data"]
    assert small["historical_var"] == pytest.approx(expected(R[:5], 0.95)["historical"])


def test_constant_returns():
    flat = row(prices("ABC", [0.0] * 20))
    assert flat["historical_var"] == 0.0 and flat["parametric_var"] == 0.0
    # Constant 1% growth (closes compounded; returns equal up to float noise):
    # historical reflects the empirical distribution, parametric uses sigma ~ 0.
    growth = row(prices("ABC", [0.01] * 20))
    assert growth["historical_var"] == pytest.approx(-0.01, rel=1e-9)
    assert growth["parametric_var"] == pytest.approx(-0.01, rel=1e-6)


def test_all_positive_returns_give_negative_var():
    gains = [0.002 * (i + 1) for i in range(20)]
    result = row(prices("ABC", gains))
    assert result["historical_var"] == pytest.approx(expected(gains, 0.95)["historical"])
    assert result["historical_var"] < 0                              # even the tail is a gain


def test_all_negative_returns():
    losses = [-0.002 * (i + 1) for i in range(20)]
    result = row(prices("ABC", losses))
    ref = expected(losses, 0.95)
    assert result["historical_var"] == pytest.approx(ref["historical"], rel=1e-9) and ref["historical"] > 0
    assert result["parametric_var"] == pytest.approx(ref["parametric"], rel=1e-9)


# 20-22. empty input, columns, mutation ------------------------------------------------------------------------------

def test_empty_input_is_handled():
    empty = prices("ABC", R).iloc[0:0]
    assert calculate_var_summary(empty).empty
    assert list(calculate_var_summary(empty).columns) == SUMMARY_COLUMNS
    assert list(calculate_historical_var(empty).columns) == HISTORICAL_COLUMNS
    assert list(calculate_parametric_var(empty).columns) == PARAMETRIC_COLUMNS


@pytest.mark.parametrize("dropped", ["date", "symbol", "close"])
def test_missing_required_columns_fail_clearly(dropped):
    with pytest.raises(ValueError, match=f"missing column\\(s\\): {dropped}"):
        calculate_var_summary(prices("ABC", R).drop(columns=dropped))


def test_input_is_not_mutated():
    data = prices("ABC", R, status="VALID").iloc[::-1]
    before = data.copy()
    calculate_var_summary(data, confidence_level=0.99)
    calculate_historical_var(data)
    calculate_parametric_var(data)
    pd.testing.assert_frame_equal(data, before)


# end-to-end: 3-symbol synthetic sample via the CSV importer --------------------------------------------------------------

def test_three_symbol_sample_through_csv_importer():
    canonical = load_csv_market_data(SAMPLE_3_SYMBOLS).data
    summary = calculate_var_summary(canonical, confidence_level=0.99).set_index("symbol")
    assert list(summary.index) == ["ABC", "LMN", "XYZ"]
    for symbol in summary.index:
        closes = list(canonical[canonical["symbol"] == symbol].sort_values("date")["close"])
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        assert summary.loc[symbol, "observations"] == 24
        assert summary.loc[symbol, "historical_var"] == pytest.approx(expected(returns, 0.99)["historical"])
        assert summary.loc[symbol, "parametric_var"] == pytest.approx(expected(returns, 0.99)["parametric"])
