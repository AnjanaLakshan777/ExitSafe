"""Tests for app.analytics.liquidity. Expected values are worked out by hand with exact fractions."""

import math
import statistics
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics.liquidity import (
    ACTUAL_TURNOVER,
    ESTIMATED_TRADED_VALUE,
    POSITION_COLUMNS,
    SUMMARY_COLUMNS,
    calculate_liquidity_summary,
    calculate_position_liquidity,
)
from app.data.loaders.csv_market_loader import load_csv_market_data

SAMPLE_3_SYMBOLS = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"
CLOSES = [100.0, 102.0, 101.0]
VOLUMES = [1_000_000.0, 1_200_000.0, 800_000.0]
EXACT_ADTV = Fraction(100 * 1_000_000 + 102 * 1_200_000 + 101 * 800_000, 3)


def market(symbol, closes, volumes, turnover=None, status=None, dates=None):
    frame = pd.DataFrame({
        "date": dates if dates is not None else pd.bdate_range("2026-01-05", periods=len(closes)),
        "symbol": symbol, "close": [float(c) for c in closes],
        "volume": [float(v) if v is not None else np.nan for v in volumes]})
    if turnover is not None:
        frame["turnover"] = [float(t) if t is not None else np.nan for t in turnover]
    if status is not None:
        frame["validation_status"] = status
    return frame


def row(data, symbol="ABC"):
    return calculate_liquidity_summary(data).set_index("symbol").loc[symbol]


def position(data, value, rate=0.10, symbol="ABC"):
    return calculate_position_liquidity(data, value, rate).set_index("symbol").loc[symbol]


# Worked example: ADV, estimated ADTV, position metrics

def test_worked_example_adv_and_estimated_adtv():
    result = row(market("ABC", CLOSES, VOLUMES))
    assert result["average_daily_volume"] == 1_000_000
    assert result["average_daily_traded_value"] == pytest.approx(float(EXACT_ADTV), rel=1e-12)
    assert result["traded_value_source"] == ESTIMATED_TRADED_VALUE
    assert result["observations"] == 3


def test_worked_example_position_liquidity():
    result = position(market("ABC", CLOSES, VOLUMES), 20_000_000)
    assert result["position_to_adtv"] == pytest.approx(float(Fraction(20_000_000) / EXACT_ADTV), rel=1e-12)
    assert result["daily_executable_value"] == pytest.approx(float(EXACT_ADTV / 10), rel=1e-12)
    assert result["estimated_liquidation_days"] == pytest.approx(
        float(Fraction(20_000_000) / (EXACT_ADTV / 10)), rel=1e-12)
    assert result["estimated_liquidation_days"] == pytest.approx(1.9788918205804749, rel=1e-12)


def test_descriptive_statistics():
    result = row(market("ABC", CLOSES, VOLUMES))
    assert (result["median_daily_volume"], result["min_daily_volume"],
            result["max_daily_volume"]) == (1_000_000, 800_000, 1_200_000)
    assert result["median_daily_traded_value"] == 100_000_000
    assert result["start_date"] == pd.Timestamp("2026-01-05")
    assert result["end_date"] == pd.Timestamp("2026-01-07")


# Actual vs estimated

def test_actual_turnover_is_used_when_every_row_has_it():
    turnover = [99_000_000, 125_000_000, 80_000_000]          # differs from close * volume
    result = row(market("ABC", CLOSES, VOLUMES, turnover=turnover))
    assert result["traded_value_source"] == ACTUAL_TURNOVER
    assert result["actual_turnover_days"] == 3
    assert result["average_daily_traded_value"] == pytest.approx(statistics.fmean(turnover))


def test_missing_turnover_column_falls_back_to_estimate():
    result = row(market("ABC", CLOSES, VOLUMES))
    assert result["traded_value_source"] == ESTIMATED_TRADED_VALUE
    assert result["actual_turnover_days"] == 0


def test_partial_turnover_is_not_mixed_with_estimates():
    # One day lacks turnover: the whole symbol uses close * volume, not a mix.
    turnover = [99_000_000, None, 80_000_000]
    result = row(market("ABC", CLOSES, VOLUMES, turnover=turnover))
    assert result["traded_value_source"] == ESTIMATED_TRADED_VALUE
    assert result["actual_turnover_days"] == 2
    assert result["average_daily_traded_value"] == pytest.approx(float(EXACT_ADTV), rel=1e-12)


def test_estimated_traded_value_equals_close_times_volume():
    closes, volumes = [10.5, 11.25, 9.75, 10.0], [1_234, 5_678, 0, 910]
    expected = statistics.fmean(c * v for c, v in zip(closes, volumes))
    assert row(market("ABC", closes, volumes))["average_daily_traded_value"] == pytest.approx(expected)


def test_actual_turnover_is_never_overwritten():
    data = market("ABC", CLOSES, VOLUMES, turnover=[1, 2, 3])
    before = data.copy()
    result = row(data)
    assert result["average_daily_traded_value"] == 2.0           # the reported values, not estimates
    pd.testing.assert_frame_equal(data, before)


def test_source_is_decided_per_symbol():
    data = pd.concat([market("ABC", CLOSES, VOLUMES, turnover=[1e8, 1e8, 1e8]),
                      market("XYZ", CLOSES, VOLUMES)])
    summary = calculate_liquidity_summary(data).set_index("symbol")
    assert summary.loc["ABC", "traded_value_source"] == ACTUAL_TURNOVER
    assert summary.loc["XYZ", "traded_value_source"] == ESTIMATED_TRADED_VALUE


# Zero-volume handling

def test_zero_volume_days_are_counted_and_averaged_as_zero():
    result = row(market("ABC", [100, 100, 101, 101], [1_000, 0, 3_000, 0]))
    assert result["observations"] == 4
    assert result["zero_volume_days"] == 2
    assert result["zero_volume_rate"] == 0.5
    assert result["average_daily_volume"] == 1_000                 # (1000 + 0 + 3000 + 0) / 4


def test_all_zero_volume_stock():
    data = market("ABC", [100, 100, 100], [0, 0, 0])
    summary = row(data)
    assert summary["zero_volume_rate"] == 1.0
    assert summary["average_daily_traded_value"] == 0.0
    result = position(data, 1_000_000)
    for column in ("position_to_adtv", "daily_executable_value", "estimated_liquidation_days"):
        assert math.isnan(result[column]), column                  # NaN, not infinity


def test_zero_adtv_from_reported_turnover_gives_nan():
    result = position(market("ABC", CLOSES, VOLUMES, turnover=[0, 0, 0]), 5_000_000)
    assert math.isnan(result["estimated_liquidation_days"])
    assert not np.isinf(result["position_to_adtv"])


# Position size and participation

def test_very_large_and_small_positions():
    data = market("ABC", CLOSES, VOLUMES)
    large = position(data, 5_000_000_000)
    small = position(data, 10_000)
    assert large["position_to_adtv"] == pytest.approx(float(Fraction(5_000_000_000) / EXACT_ADTV))
    assert large["estimated_liquidation_days"] == pytest.approx(
        float(Fraction(5_000_000_000) / (EXACT_ADTV / 10)))
    assert small["estimated_liquidation_days"] == pytest.approx(float(Fraction(10_000) / (EXACT_ADTV / 10)))
    assert small["estimated_liquidation_days"] < 1


def test_full_participation():
    result = position(market("ABC", CLOSES, VOLUMES), 20_000_000, rate=1.0)
    assert result["daily_executable_value"] == pytest.approx(float(EXACT_ADTV))
    assert result["estimated_liquidation_days"] == pytest.approx(result["position_to_adtv"])


@pytest.mark.parametrize("bad", [0, -0.1, 1.01, 10, float("nan"), float("inf"), "0.1", None, True])
def test_invalid_participation_rate(bad):
    with pytest.raises(ValueError, match="participation_rate"):
        calculate_position_liquidity(market("ABC", CLOSES, VOLUMES), 1_000_000, bad)


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), "1000", None, True])
def test_invalid_position_value(bad):
    with pytest.raises(ValueError, match="position_value"):
        calculate_position_liquidity(market("ABC", CLOSES, VOLUMES), bad)


# Symbols, ordering, gaps

def test_multiple_symbols_are_never_mixed():
    data = pd.concat([market("ABC", CLOSES, VOLUMES),
                      market("XYZ", [10, 11], [50_000, 70_000])]).sample(frac=1, random_state=2)
    summary = calculate_liquidity_summary(data).set_index("symbol")

    assert list(summary.index) == ["ABC", "XYZ"]
    assert summary.loc["ABC", "average_daily_volume"] == 1_000_000
    assert summary.loc["XYZ", "average_daily_volume"] == 60_000
    assert summary.loc["XYZ", "average_daily_traded_value"] == pytest.approx((10 * 50_000 + 11 * 70_000) / 2)
    positions = calculate_position_liquidity(data, 1_000_000).set_index("symbol")
    assert positions.loc["XYZ", "position_to_adtv"] == pytest.approx(1_000_000 / 635_000)


def test_unsorted_dates_and_gaps_are_not_filled():
    dates = pd.to_datetime(["2026-01-09", "2026-01-05", "2026-01-20"])   # gaps, unsorted
    result = row(market("ABC", CLOSES, VOLUMES, dates=dates))
    assert result["observations"] == 3                        # no invented trading days
    assert result["start_date"] == pd.Timestamp("2026-01-05")
    assert result["end_date"] == pd.Timestamp("2026-01-20")


# Invalid data

def test_invalid_rows_do_not_contribute():
    data = market("ABC", CLOSES + [100], VOLUMES + [999_000_000],
                  status=["VALID", "VALID", "VALID", "INVALID"])
    result = row(data)
    assert result["observations"] == 3
    assert result["average_daily_volume"] == 1_000_000            # no fake liquidity


def test_warning_rows_remain_usable():
    result = row(market("ABC", CLOSES, VOLUMES, status=["VALID", "WARNING", "WARNING"]))
    assert result["observations"] == 3 and result["average_daily_volume"] == 1_000_000


def test_rows_missing_date_volume_or_symbol_are_not_used():
    data = pd.concat([market("ABC", CLOSES, VOLUMES), pd.DataFrame({
        "date": [pd.NaT, pd.Timestamp("2026-02-02"), pd.Timestamp("2026-02-03")],
        "symbol": ["ABC", "ABC", None], "close": [100.0, 100.0, 100.0],
        "volume": [5e9, np.nan, 5e9]})])
    result = row(data)
    assert result["observations"] == 3 and result["average_daily_volume"] == 1_000_000


def test_unvalidated_duplicates_fail_clearly():
    data = pd.concat([market("ABC", CLOSES, VOLUMES), market("ABC", [100], [1])])
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        calculate_liquidity_summary(data)


def test_validated_duplicates_are_invalid_and_excluded():
    data = pd.concat([market("ABC", CLOSES, VOLUMES, status="VALID"),
                      market("ABC", [100], [1], status="INVALID")])
    data.iloc[0, data.columns.get_loc("validation_status")] = "INVALID"   # the other copy
    result = row(data)
    assert result["observations"] == 2
    assert result["average_daily_volume"] == 1_000_000            # (1.2M + 0.8M) / 2


# Empty input, columns, mutation

def test_empty_input():
    empty = market("ABC", CLOSES, VOLUMES).iloc[0:0]
    assert list(calculate_liquidity_summary(empty).columns) == SUMMARY_COLUMNS
    assert calculate_liquidity_summary(empty).empty
    assert list(calculate_position_liquidity(empty, 1_000).columns) == POSITION_COLUMNS


@pytest.mark.parametrize("dropped", ["date", "symbol", "close", "volume"])
def test_missing_required_columns_fail_clearly(dropped):
    with pytest.raises(ValueError, match=f"missing column\\(s\\): {dropped}"):
        calculate_liquidity_summary(market("ABC", CLOSES, VOLUMES).drop(columns=dropped))


def test_input_is_not_mutated():
    data = market("ABC", CLOSES, VOLUMES, turnover=[1e8, None, 1e8], status="VALID").iloc[::-1]
    before = data.copy()
    calculate_liquidity_summary(data)
    calculate_position_liquidity(data, 20_000_000, 0.25)
    pd.testing.assert_frame_equal(data, before)


# End-to-end: 3-symbol sample (reports actual turnover) via the CSV importer

def test_three_symbol_sample_uses_actual_turnover():
    canonical = load_csv_market_data(SAMPLE_3_SYMBOLS).data
    raw = pd.read_csv(SAMPLE_3_SYMBOLS)
    summary = calculate_liquidity_summary(canonical).set_index("symbol")

    assert list(summary.index) == ["ABC", "LMN", "XYZ"]
    assert (summary["traded_value_source"] == ACTUAL_TURNOVER).all()
    for symbol in summary.index:
        rows = raw[raw["Symbol"] == symbol]
        assert summary.loc[symbol, "average_daily_volume"] == pytest.approx(statistics.fmean(rows["Volume"]))
        assert summary.loc[symbol, "average_daily_traded_value"] == pytest.approx(
            statistics.fmean(rows["Value Traded"]))
        assert summary.loc[symbol, "zero_volume_days"] == int((rows["Volume"] == 0).sum())
    assert summary.loc["LMN", "zero_volume_rate"] == pytest.approx(5 / 25)
