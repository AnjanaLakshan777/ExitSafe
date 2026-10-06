"""Tests for app.analytics.cvar (stock-level 1-day CVaR / Expected Shortfall).

Expected values are independent of the module under test:
  * historical: exact rational arithmetic (fractions.Fraction) in ``es_exact``
  * parametric: statistics.mean / stdev and statistics.NormalDist (inv_cdf, pdf),
    not scipy

Worked example: R30 (30 synthetic returns), 95% -> alpha = 0.05
  tail mass m = 30 * 0.05 = 1.5 -> k = 1 full observation, fraction f = 0.5
  worst losses: 0.05 (weight 1), 0.04 (weight 0.5)
  historical CVaR = (0.05 + 0.5 * 0.04) / 1.5 = 0.07 / 1.5 = 0.0466666...
  NOT the plain average of the worst two, (0.05 + 0.04) / 2 = 0.045
  mean = -0.0036666..., sample sd = 0.0202677..., z_0.05 = -1.6448536...,
  phi(z_0.05) = 0.1031356...
  parametric CVaR = -(mean - sd * phi(z) / alpha) = 0.0454732...
"""

import math
import statistics
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics import cvar as cvar_module
from app.analytics.cvar import (
    HISTORICAL_COLUMNS,
    PARAMETRIC_COLUMNS,
    SUMMARY_COLUMNS,
    calculate_cvar_summary,
    calculate_historical_cvar,
    calculate_parametric_cvar,
    historical_expected_shortfall,
)
from app.data.loaders.csv_market_loader import load_csv_market_data

R20 = [0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.015, -0.01, 0.005, -0.03,
       0.012, -0.008, 0.025, -0.015, 0.004, -0.022, 0.018, -0.006, 0.009, -0.04]
R30 = R20 + [0.007, -0.012, 0.011, -0.003, 0.016, -0.019, 0.002, -0.027, 0.013, -0.035]
SAMPLE_3_SYMBOLS = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"
ND = statistics.NormalDist()


def prices(symbol, returns, start=100.0, status=None):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    frame = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=len(closes)),
                          "symbol": symbol, "close": closes})
    if status is not None:
        frame["validation_status"] = status
    return frame


def es_exact(returns, confidence):
    """Independent empirical expected shortfall in exact fractions."""
    alpha = Fraction(1) - Fraction(str(confidence))
    losses = sorted((-Fraction(str(r)) for r in returns), reverse=True)
    m = alpha * len(losses)
    k = math.floor(m)
    total = sum(losses[:k], Fraction(0)) + ((m - k) * losses[k] if m > k else 0)
    return float(total / m)


def es_normal(returns, confidence):
    alpha = 1 - confidence
    z = ND.inv_cdf(alpha)
    return -(statistics.mean(returns) - statistics.stdev(returns) * ND.pdf(z) / alpha)


def row(data, symbol="ABC", **kwargs):
    return calculate_cvar_summary(data, **kwargs).set_index("symbol").loc[symbol]


# 1-4, 32. worked example and tail mass --------------------------------------------------------------

def test_historical_cvar_known_example_with_fractional_tail():
    result = row(prices("ABC", R30))
    assert result["tail_mass"] == 1.5
    assert result["historical_cvar"] == pytest.approx(0.07 / 1.5, rel=1e-9)
    assert result["historical_cvar"] == pytest.approx(es_exact(R30, 0.95), rel=1e-9)


def test_parametric_cvar_known_example():
    result = row(prices("ABC", R30))
    assert statistics.mean(R30) == pytest.approx(-0.11 / 30)
    assert ND.pdf(ND.inv_cdf(0.05)) == pytest.approx(0.10313564037537132, rel=1e-12)
    assert result["parametric_cvar"] == pytest.approx(es_normal(R30, 0.95), rel=1e-9)
    assert result["parametric_cvar"] == pytest.approx(0.04547321005597537, rel=1e-9)


@pytest.mark.parametrize("confidence", [0.95, 0.99, 0.90, 0.975, 0.937])
def test_confidence_levels_match_independent_calculation(confidence):
    result = row(prices("ABC", R30), confidence_level=confidence)
    assert result["confidence_level"] == confidence
    assert result["historical_cvar"] == pytest.approx(es_exact(R30, confidence), rel=1e-9)
    assert result["parametric_cvar"] == pytest.approx(es_normal(R30, confidence), rel=1e-9)


def test_99_percent_with_tail_mass_below_one_is_the_worst_loss():
    result = row(prices("ABC", R20), confidence_level=0.99)
    assert result["tail_mass"] == pytest.approx(0.2)
    assert result["historical_cvar"] == pytest.approx(0.05, rel=1e-9)


@pytest.mark.parametrize("n, expected_mass", [(20, 1.0), (30, 1.5), (40, 2.0), (25, 1.25)])
def test_tail_mass_is_alpha_times_n_without_float_noise(n, expected_mass):
    returns = (R30 * 2)[:n]
    assert row(prices("ABC", returns))["tail_mass"] == expected_mass


# 7-10. integer vs fractional tails ------------------------------------------------------------------

def test_exact_integer_tail_mass():
    # 20 returns at 95%: m = 1 -> the single worst loss
    assert row(prices("ABC", R20))["historical_cvar"] == pytest.approx(0.05, rel=1e-9)
    # 40 returns at 95%: m = 2 -> plain average of the worst two
    r40 = R20 + [r / 2 for r in R20]
    assert row(prices("ABC", r40))["historical_cvar"] == pytest.approx((0.05 + 0.04) / 2, rel=1e-9)


def test_fractional_boundary_is_weighted_not_averaged_equally():
    result = row(prices("ABC", R30))
    equal_average = (0.05 + 0.04) / 2
    assert result["historical_cvar"] != pytest.approx(equal_average)
    assert result["historical_cvar"] == pytest.approx((1.0 * 0.05 + 0.5 * 0.04) / 1.5, rel=1e-9)


def test_quarter_weight_boundary():
    # 25 returns at 95%: m = 1.25 -> worst loss full, second worst 0.25
    returns = (R30 * 2)[:25]
    losses = sorted((-r for r in returns), reverse=True)
    expected = (losses[0] + 0.25 * losses[1]) / 1.25
    assert row(prices("ABC", returns))["historical_cvar"] == pytest.approx(expected, rel=1e-9)


def test_not_simply_the_mean_below_the_interpolated_var_cutoff():
    result = row(prices("ABC", R30))
    cutoff = -result["historical_var"]                     # interpolated lower-tail return
    naive = -statistics.mean([r for r in R30 if r <= cutoff])
    assert naive != pytest.approx(result["historical_cvar"], rel=1e-6)


def test_helper_averages_the_worst_probability_mass():
    # direct check of the documented algorithm on a tiny sample
    assert historical_expected_shortfall([0.01, -0.02, -0.04, 0.03], 0.5) == pytest.approx(0.03)
    # m = 2: worst two losses 0.04 and 0.02 -> 0.03


# 5-6. validation (shared with VaR) ---------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, 1, -0.5, 1.5, 95, float("nan"), float("inf"), "0.95", None, True])
def test_invalid_confidence_levels(bad):
    with pytest.raises(ValueError, match="confidence_level"):
        calculate_cvar_summary(prices("ABC", R30), confidence_level=bad)


@pytest.mark.parametrize("bad", [0, 1, -5, 2.5, "20", True])
def test_minimum_observation_validation(bad):
    with pytest.raises(ValueError, match="min_observations"):
        calculate_cvar_summary(prices("ABC", R30), min_observations=bad)


# 11-17. parametric ingredients and VaR relationship ------------------------------------------------------

def test_parametric_cvar_uses_sample_mean_and_sample_sd():
    result = row(prices("ABC", R30))
    z = ND.inv_cdf(0.05)
    with_population_sd = -(statistics.mean(R30) - statistics.pstdev(R30) * ND.pdf(z) / 0.05)
    without_mean = statistics.stdev(R30) * ND.pdf(z) / 0.05
    assert result["parametric_cvar"] != pytest.approx(with_population_sd, rel=1e-6)
    assert result["parametric_cvar"] != pytest.approx(without_mean, rel=1e-6)
    assert result["parametric_cvar"] == pytest.approx(without_mean - statistics.mean(R30), rel=1e-9)


def test_parametric_cvar_uses_normal_quantile_and_pdf():
    # The tail factor pdf(z)/alpha has no textbook constant; check it at 93.7%.
    result = row(prices("ABC", R30), confidence_level=0.937)
    alpha = 1 - 0.937
    factor = ND.pdf(ND.inv_cdf(alpha)) / alpha
    assert result["parametric_cvar"] == pytest.approx(
        -(statistics.mean(R30) - statistics.stdev(R30) * factor), rel=1e-9)
    # pdf(z)/alpha differs from |z|: the CVaR factor is not the VaR factor
    assert factor != pytest.approx(-ND.inv_cdf(alpha))


def test_var_and_cvar_share_confidence_and_sample():
    result = row(prices("ABC", R30), confidence_level=0.975)
    alpha = 1 - 0.975
    assert result["observations"] == 30
    assert result["parametric_var"] == pytest.approx(
        -(statistics.mean(R30) + ND.inv_cdf(alpha) * statistics.stdev(R30)), rel=1e-9)
    assert result["parametric_cvar"] == pytest.approx(es_normal(R30, 0.975), rel=1e-9)
    assert (calculate_historical_cvar(prices("ABC", R30))["observations"]
            == calculate_parametric_cvar(prices("ABC", R30))["observations"]).all()


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.975, 0.99])
def test_cvar_is_at_least_var(confidence):
    rng = np.random.default_rng(21)
    for _ in range(25):
        returns = list(rng.standard_t(3, size=int(rng.integers(20, 80))) * 0.01)
        result = row(prices("ABC", returns), confidence_level=confidence)
        assert result["historical_cvar"] >= result["historical_var"] - 1e-12
        assert result["parametric_cvar"] >= result["parametric_var"] - 1e-12


# 18-19. symbols and ordering ---------------------------------------------------------------------------------

def test_multiple_symbols_are_independent():
    other = [r * 0.5 + 0.001 for r in R30]
    data = pd.concat([prices("ABC", R30), prices("XYZ", other, start=10), prices("LMN", R20)])
    summary = calculate_cvar_summary(data).set_index("symbol")

    assert list(summary.index) == ["ABC", "LMN", "XYZ"]
    assert summary.loc["ABC", "historical_cvar"] == pytest.approx(es_exact(R30, 0.95), rel=1e-9)
    assert summary.loc["XYZ", "historical_cvar"] == pytest.approx(es_exact(other, 0.95), rel=1e-9)
    assert summary.loc["LMN", "historical_cvar"] == pytest.approx(0.05, rel=1e-9)
    assert summary.loc["XYZ", "parametric_cvar"] == pytest.approx(es_normal(other, 0.95), rel=1e-9)


def test_unsorted_data_is_handled():
    data = pd.concat([prices("ABC", R30), prices("XYZ", R20)]).sample(frac=1, random_state=8)
    assert row(data)["historical_cvar"] == pytest.approx(0.07 / 1.5, rel=1e-9)


# 20-23. validation status and non-finite returns -----------------------------------------------------------------

def test_invalid_rows_are_excluded_and_not_bridged():
    data = prices("ABC", R30, status="VALID")
    data.loc[6, ["close", "validation_status"]] = [1.0, "INVALID"]   # an absurd crash
    usable = R30[:5] + R30[7:]          # R30[5] (into day 6) and R30[6] (out of it) are gone

    result = row(data)

    assert result["observations"] == 28
    assert result["historical_cvar"] == pytest.approx(es_exact(usable, 0.95), rel=1e-9)
    assert result["parametric_cvar"] == pytest.approx(es_normal(usable, 0.95), rel=1e-9)
    assert result["historical_cvar"] < 0.5                    # the 1.0 close never leaks in


def test_warning_rows_remain_usable():
    result = row(prices("ABC", R30, status="WARNING"))
    assert result["observations"] == 30
    assert result["historical_cvar"] == pytest.approx(0.07 / 1.5, rel=1e-9)


def test_non_finite_returns_are_excluded():
    data = prices("ABC", R30)
    data.loc[31] = [pd.Timestamp("2025-02-13"), "ABC", 0.0]
    data.loc[32] = [pd.Timestamp("2025-02-14"), "ABC", 50.0]
    result = row(data)
    assert result["observations"] == 31                      # inf return dropped
    assert np.isfinite(result["historical_cvar"]) and np.isfinite(result["parametric_cvar"])


def test_rows_without_date_symbol_or_close_do_not_enter():
    extra = pd.DataFrame({"date": [pd.NaT, pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-04")],
                          "symbol": ["ABC", None, "ABC"], "close": [500.0, 500.0, np.nan]})
    result = row(pd.concat([prices("ABC", R30), extra], ignore_index=True))
    assert result["observations"] == 30
    assert result["historical_cvar"] == pytest.approx(0.07 / 1.5, rel=1e-9)


# 24-26. insufficient data, constant returns, empty input ----------------------------------------------------------------

@pytest.mark.parametrize("returns", [[], [0.01], R20[:19]])
def test_insufficient_observations_give_nan_not_zero(returns):
    result = row(prices("ABC", returns))
    assert result["observations"] == len(returns)
    assert not result["sufficient_data"]
    for column in ("historical_var", "historical_cvar", "parametric_var", "parametric_cvar"):
        assert math.isnan(result[column]), column


def test_exactly_the_minimum_is_enough():
    result = row(prices("ABC", R20))
    assert result["sufficient_data"] and result["historical_cvar"] == pytest.approx(0.05)


def test_constant_returns_and_zero_sigma():
    flat = row(prices("ABC", [0.0] * 20))
    assert flat["historical_cvar"] == 0.0 and flat["parametric_cvar"] == 0.0
    growth = row(prices("ABC", [0.01] * 20))              # sigma ~ 0 (float noise only)
    assert growth["historical_cvar"] == pytest.approx(-0.01, rel=1e-9)
    assert growth["parametric_cvar"] == pytest.approx(-0.01, rel=1e-6)
    assert growth["parametric_cvar"] == pytest.approx(growth["parametric_var"], rel=1e-6)


def test_empty_input_is_handled():
    empty = prices("ABC", R30).iloc[0:0]
    assert calculate_cvar_summary(empty).empty
    assert list(calculate_cvar_summary(empty).columns) == SUMMARY_COLUMNS
    assert list(calculate_historical_cvar(empty).columns) == HISTORICAL_COLUMNS
    assert list(calculate_parametric_cvar(empty).columns) == PARAMETRIC_COLUMNS


# 27-31. columns, mutation, sign conventions ------------------------------------------------------------------------------

@pytest.mark.parametrize("dropped", ["date", "symbol", "close"])
def test_missing_columns_fail_clearly(dropped):
    with pytest.raises(ValueError, match=f"missing column\\(s\\): {dropped}"):
        calculate_cvar_summary(prices("ABC", R30).drop(columns=dropped))


def test_input_is_not_mutated():
    data = prices("ABC", R30, status="VALID").iloc[::-1]
    before = data.copy()
    calculate_cvar_summary(data, confidence_level=0.99)
    calculate_historical_cvar(data)
    calculate_parametric_cvar(data)
    pd.testing.assert_frame_equal(data, before)


def test_negative_returns_give_positive_loss():
    result = row(prices("ABC", R30))
    assert result["historical_cvar"] > 0 and result["parametric_cvar"] > 0


def test_all_positive_returns_follow_the_sign_convention():
    gains = [0.002 * (i + 1) for i in range(20)]        # worst day is a +0.2% gain
    result = row(prices("ABC", gains))
    assert result["historical_cvar"] == pytest.approx(-0.002, rel=1e-9)   # negative: a gain
    assert result["historical_cvar"] == pytest.approx(es_exact(gains, 0.95), rel=1e-9)


def test_all_negative_returns():
    losses = [-0.002 * (i + 1) for i in range(30)]
    result = row(prices("ABC", losses))
    assert result["historical_cvar"] == pytest.approx(es_exact(losses, 0.95), rel=1e-9)
    assert result["historical_cvar"] == pytest.approx((0.06 + 0.5 * 0.058) / 1.5, rel=1e-9)
    assert result["parametric_cvar"] == pytest.approx(es_normal(losses, 0.95), rel=1e-9)


def test_tail_containing_gains_is_not_clamped():
    # Only 1 loss in 40 returns; at 90% the tail mass of 4 includes three gains.
    returns = [-0.01] + [0.005 + 0.0001 * i for i in range(39)]
    result = row(prices("ABC", returns), confidence_level=0.90)
    assert result["tail_mass"] == pytest.approx(4.0)
    assert result["historical_cvar"] == pytest.approx(es_exact(returns, 0.90), rel=1e-9)
    assert result["historical_cvar"] < 0.01                # pulled down by the gains, not clamped


def test_tail_mass_rounding_constant_is_documented():
    assert cvar_module.TAIL_MASS_DECIMALS == 9


# end-to-end: 3-symbol sample through the CSV importer ------------------------------------------------------------------------

def test_three_symbol_sample_through_csv_importer():
    canonical = load_csv_market_data(SAMPLE_3_SYMBOLS).data
    summary = calculate_cvar_summary(canonical).set_index("symbol")
    assert list(summary.index) == ["ABC", "LMN", "XYZ"]
    for symbol in summary.index:
        closes = list(canonical[canonical["symbol"] == symbol].sort_values("date")["close"])
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        assert summary.loc[symbol, "tail_mass"] == pytest.approx(1.2)        # 24 * 0.05
        assert summary.loc[symbol, "historical_cvar"] == pytest.approx(es_exact(returns, 0.95), rel=1e-9)
        assert summary.loc[symbol, "parametric_cvar"] == pytest.approx(es_normal(returns, 0.95), rel=1e-9)
        assert summary.loc[symbol, "historical_cvar"] >= summary.loc[symbol, "historical_var"]
