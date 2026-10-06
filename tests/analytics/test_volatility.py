"""Tests for app.analytics.volatility, checked against statistics.stdev and hand-worked values."""

import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics.volatility import (
    TRADING_DAYS_PER_YEAR,
    calculate_annualized_volatility,
    calculate_daily_volatility,
)
from app.data.loaders.csv_market_loader import load_csv_market_data

# closes 100, 102, 101, 103 -> returns 0.02, -0.0098039216, 0.0198019802
# sample std (ddof=1) computed with exact fractions, then converted to float
EXAMPLE_DAILY_VOL = 0.017150424543468497
EXAMPLE_ANNUAL_VOL = 0.2722545493271767
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_date_price_vol_change.csv"


def prices(symbol, closes, start="2025-01-01", status=None):
    frame = pd.DataFrame({
        "date": pd.bdate_range(start, periods=len(closes)),
        "symbol": symbol,
        "close": [float(c) for c in closes],
    })
    if status is not None:
        frame["validation_status"] = status
    return frame


def simple_returns(closes):
    return [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]


def row(result, symbol):
    return result.set_index("symbol").loc[symbol]


# Known example

def test_known_returns_give_expected_daily_volatility():
    result = calculate_daily_volatility(prices("ABC", [100, 102, 101, 103]))

    assert list(result.columns) == ["symbol", "observations", "daily_volatility"]
    assert row(result, "ABC")["observations"] == 3
    assert row(result, "ABC")["daily_volatility"] == pytest.approx(EXAMPLE_DAILY_VOL, rel=1e-12)
    assert EXAMPLE_DAILY_VOL == pytest.approx(statistics.stdev([0.02, -1 / 102, 2 / 101]), rel=1e-12)


def test_sample_not_population_standard_deviation():
    vol = row(calculate_daily_volatility(prices("ABC", [100, 102, 101, 103])), "ABC")["daily_volatility"]
    population = statistics.pstdev(simple_returns([100, 102, 101, 103]))
    assert vol != pytest.approx(population)
    assert vol == pytest.approx(population * math.sqrt(3 / 2))


# Annualization

def test_annualized_is_daily_times_sqrt_252():
    result = row(calculate_annualized_volatility(prices("ABC", [100, 102, 101, 103])), "ABC")

    assert TRADING_DAYS_PER_YEAR == 252
    assert result["periods_per_year"] == 252
    assert result["annualized_volatility"] == pytest.approx(EXAMPLE_ANNUAL_VOL, rel=1e-12)
    assert result["annualized_volatility"] == pytest.approx(result["daily_volatility"] * math.sqrt(252),
                                                            rel=1e-15)


def test_custom_periods_per_year():
    result = row(calculate_annualized_volatility(prices("ABC", [100, 102, 101, 103]),
                                                 periods_per_year=260), "ABC")
    assert result["annualized_volatility"] == pytest.approx(EXAMPLE_DAILY_VOL * math.sqrt(260),
                                                            rel=1e-12)


@pytest.mark.parametrize("bad", [0, -252, "252", None, True])
def test_invalid_periods_per_year_is_rejected(bad):
    with pytest.raises(ValueError, match="periods_per_year"):
        calculate_annualized_volatility(prices("ABC", [100, 102, 101]), periods_per_year=bad)


# Per symbol, histories never mixed

def test_volatility_is_calculated_separately_for_each_symbol():
    abc, xyz = [100, 102, 101, 103, 104], [10, 9.5, 9.8, 9.1]
    data = pd.concat([prices("ABC", abc), prices("XYZ", xyz)])

    result = calculate_daily_volatility(data)

    assert list(result["symbol"]) == ["ABC", "XYZ"]
    assert row(result, "ABC")["daily_volatility"] == pytest.approx(statistics.stdev(simple_returns(abc)))
    assert row(result, "XYZ")["daily_volatility"] == pytest.approx(statistics.stdev(simple_returns(xyz)))


def test_interleaved_histories_are_not_mixed():
    # Same dates, very different price levels, rows shuffled together. Mixing
    # the two series would create returns like 10/100 - 1 = -90%.
    abc, xyz = [100, 101, 99, 102], [10, 10.2, 10.1, 10.4]
    data = pd.concat([prices("ABC", abc), prices("XYZ", xyz)]).sample(frac=1, random_state=7)

    result = calculate_daily_volatility(data)

    assert row(result, "ABC")["daily_volatility"] == pytest.approx(statistics.stdev(simple_returns(abc)))
    assert row(result, "XYZ")["daily_volatility"] == pytest.approx(statistics.stdev(simple_returns(xyz)))
    assert result["daily_volatility"].max() < 0.05


def test_unsorted_dates_are_ordered_before_returns():
    data = prices("ABC", [100, 102, 101, 103]).iloc[[2, 0, 3, 1]]
    assert row(calculate_daily_volatility(data), "ABC")["daily_volatility"] == pytest.approx(
        EXAMPLE_DAILY_VOL, rel=1e-12)


# No mutation

def test_input_dataframe_is_not_mutated():
    data = prices("ABC", [100, 102, 101, 103], status=["VALID", "INVALID", "VALID", "VALID"])
    before = data.copy()

    calculate_annualized_volatility(data)

    pd.testing.assert_frame_equal(data, before)


# First NaN return

def test_first_nan_return_does_not_break_calculation():
    result = row(calculate_daily_volatility(prices("ABC", [100, 102, 101, 103])), "ABC")
    assert result["observations"] == 3          # 4 closes -> first return NaN -> 3 used
    assert not math.isnan(result["daily_volatility"])


# Invalid rows

def test_invalid_rows_do_not_participate():
    # Row 3 is INVALID with an absurd close. Both returns that touch it are
    # excluded; it is not dropped and bridged into a two-day return.
    closes = [100, 102, 101, 5000, 103, 104]
    status = ["VALID", "VALID", "VALID", "INVALID", "VALID", "VALID"]

    result = row(calculate_daily_volatility(prices("ABC", closes, status=status)), "ABC")

    usable = [102 / 100 - 1, 101 / 102 - 1, 104 / 103 - 1]
    assert result["observations"] == 3
    assert result["daily_volatility"] == pytest.approx(statistics.stdev(usable))


def test_warning_rows_are_used():
    closes = [100, 102, 101, 103]
    status = ["VALID", "WARNING", "VALID", "WARNING"]
    result = row(calculate_daily_volatility(prices("ABC", closes, status=status)), "ABC")
    assert result["daily_volatility"] == pytest.approx(EXAMPLE_DAILY_VOL, rel=1e-12)


def test_non_finite_returns_are_ignored():
    # Without validation_status the data is assumed validated, but a zero close
    # would give an infinite return; it must not reach the standard deviation.
    result = row(calculate_daily_volatility(prices("ABC", [100, 0, 101, 103, 104])), "ABC")
    finite = [0 / 100 - 1, 104 / 103 - 1, 103 / 101 - 1]
    assert result["observations"] == 3
    assert result["daily_volatility"] == pytest.approx(statistics.stdev(finite))


# Insufficient observations

@pytest.mark.parametrize("closes, observations", [([100], 0), ([100, 101], 1)])
def test_insufficient_observations_give_nan_not_zero(closes, observations):
    result = row(calculate_annualized_volatility(prices("ABC", closes)), "ABC")
    assert result["observations"] == observations
    assert math.isnan(result["daily_volatility"])
    assert math.isnan(result["annualized_volatility"])


def test_insufficient_symbol_does_not_affect_others():
    data = pd.concat([prices("ABC", [100, 102, 101, 103]), prices("NEW", [50])])
    result = calculate_daily_volatility(data)
    assert math.isnan(row(result, "NEW")["daily_volatility"])
    assert row(result, "ABC")["daily_volatility"] == pytest.approx(EXAMPLE_DAILY_VOL, rel=1e-12)


def test_all_rows_invalid_gives_nan():
    data = prices("ABC", [100, 102, 101], status="INVALID")
    result = row(calculate_daily_volatility(data), "ABC")
    assert result["observations"] == 0
    assert math.isnan(result["daily_volatility"])


# Constant returns

def test_constant_prices_give_zero_volatility():
    result = row(calculate_annualized_volatility(prices("ABC", [100, 100, 100, 100])), "ABC")
    assert result["daily_volatility"] == 0.0
    assert result["annualized_volatility"] == 0.0


def test_constant_growth_gives_zero_volatility_within_rounding():
    result = row(calculate_daily_volatility(prices("ABC", [100, 110, 121, 133.1])), "ABC")
    assert result["daily_volatility"] == pytest.approx(0.0, abs=1e-12)


# Negative returns

def test_negative_returns_are_handled():
    falling = [100, 95, 92, 90, 84]
    result = row(calculate_daily_volatility(prices("ABC", falling)), "ABC")

    assert all(r < 0 for r in simple_returns(falling))
    assert result["daily_volatility"] > 0
    assert result["daily_volatility"] == pytest.approx(statistics.stdev(simple_returns(falling)))


def test_mixed_sign_returns():
    closes = [100, 90, 99, 89.1, 98.01]   # -10%, +10%, -10%, +10%
    result = row(calculate_daily_volatility(prices("ABC", closes)), "ABC")
    assert result["daily_volatility"] == pytest.approx(statistics.stdev([-0.1, 0.1, -0.1, 0.1]))


# Empty input and missing columns

def test_empty_input_returns_empty_result():
    empty = prices("ABC", [100]).iloc[0:0]
    result = calculate_annualized_volatility(empty)
    assert result.empty
    assert list(result.columns) == ["symbol", "observations", "daily_volatility",
                                    "annualized_volatility", "periods_per_year"]


@pytest.mark.parametrize("dropped", ["date", "symbol", "close"])
def test_missing_required_columns_fail_clearly(dropped):
    data = prices("ABC", [100, 102, 101]).drop(columns=dropped)
    with pytest.raises(ValueError, match=f"missing column\\(s\\): {dropped}"):
        calculate_daily_volatility(data)


def test_phase1_layout_is_not_accepted():
    legacy = pd.DataFrame({"Date": pd.bdate_range("2025-01-01", periods=3), "Symbol": "ABC",
                           "Close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="missing column"):
        calculate_daily_volatility(legacy)


# End-to-end with the CSV importer

def test_works_on_canonical_data_from_the_csv_importer():
    canonical = load_csv_market_data(FIXTURE, symbol="TEST.N0000").data
    closes = list(canonical.sort_values("date")["close"])

    result = row(calculate_annualized_volatility(canonical), "TEST.N0000")

    assert result["observations"] == len(closes) - 1
    assert result["daily_volatility"] == pytest.approx(statistics.stdev(simple_returns(closes)))
    assert np.isfinite(result["annualized_volatility"])
