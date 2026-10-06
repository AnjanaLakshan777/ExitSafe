"""Tests for app.analytics.drawdown.

Expected values are worked out by hand, not with the module under test.

Worked example (from the brief)
  closes        100   110   120   108    90    105
  running peak  100   110   120   120   120    120
  drawdown        0     0     0  -10%  -25%  -12.5%
  maximum drawdown = -0.25, peak 120 (day 3), trough 90 (day 5), not recovered

Recovery example
  closes 100, 120, 96, 110, 125 -> MDD = 96/120 - 1 = -0.20, peak 120 (day 2),
  trough 96 (day 3), recovery on day 5 (125 >= 120)
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from app.analytics.drawdown import (
    SERIES_COLUMNS,
    SUMMARY_COLUMNS,
    calculate_drawdown_series,
    calculate_maximum_drawdown,
)
from app.data.loaders.csv_market_loader import load_csv_market_data

BRIEF = [100, 110, 120, 108, 90, 105]
DAYS = pd.bdate_range("2025-01-01", periods=10)   # Jan 1, 2, 3, 6, 7, 8, 9, 10, 13, 14
SAMPLE_3_SYMBOLS = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"


def prices(symbol, closes, status=None, dates=None):
    frame = pd.DataFrame({"date": dates if dates is not None else DAYS[:len(closes)],
                          "symbol": symbol, "close": [float(c) for c in closes]})
    if status is not None:
        frame["validation_status"] = status
    return frame


def summary_row(data, symbol):
    return calculate_maximum_drawdown(data).set_index("symbol").loc[symbol]


# 1-5. worked example ------------------------------------------------------------------------------

def test_known_sequence_gives_expected_maximum_drawdown():
    row = summary_row(prices("ABC", BRIEF), "ABC")
    assert row["maximum_drawdown"] == pytest.approx(-0.25)
    assert row["observations"] == 6


def test_drawdown_series_and_running_peak():
    series = calculate_drawdown_series(prices("ABC", BRIEF))

    assert list(series.columns) == SERIES_COLUMNS
    assert list(series["running_peak"]) == [100, 110, 120, 120, 120, 120]
    assert list(series["drawdown"]) == pytest.approx([0, 0, 0, -0.10, -0.25, -0.125])


def test_peak_and_trough_dates_and_prices():
    row = summary_row(prices("ABC", BRIEF), "ABC")

    assert (row["peak_price"], row["peak_date"]) == (120, DAYS[2])
    assert (row["trough_price"], row["trough_date"]) == (90, DAYS[4])
    assert row["peak_date"] < row["trough_date"]
    assert row["trough_price"] / row["peak_price"] - 1 == pytest.approx(row["maximum_drawdown"])


def test_maximum_drawdown_is_negative_not_a_positive_percentage():
    assert summary_row(prices("ABC", BRIEF), "ABC")["maximum_drawdown"] < 0


def test_peak_is_chronological_not_the_overall_maximum():
    # The highest price (150) comes after the worst fall, so a naive
    # max-price / min-price approach would be wrong.
    row = summary_row(prices("ABC", [100, 80, 120, 150]), "ABC")
    assert row["maximum_drawdown"] == pytest.approx(-0.20)
    assert (row["peak_price"], row["peak_date"]) == (100, DAYS[0])
    assert (row["trough_price"], row["trough_date"]) == (80, DAYS[1])


def test_peak_is_the_latest_date_at_the_peak_level():
    # 120 is reached twice; the decline to 90 starts from the second one.
    row = summary_row(prices("ABC", [100, 120, 110, 120, 90]), "ABC")
    assert row["peak_date"] == DAYS[3]
    assert row["maximum_drawdown"] == pytest.approx(-0.25)


# 6-7. recovery ------------------------------------------------------------------------------------

def test_recovered_drawdown_has_recovery_date():
    row = summary_row(prices("ABC", [100, 120, 96, 110, 125]), "ABC")
    assert row["maximum_drawdown"] == pytest.approx(-0.20)
    assert (row["peak_date"], row["trough_date"]) == (DAYS[1], DAYS[2])
    assert row["recovery_date"] == DAYS[4]


def test_returning_exactly_to_the_peak_counts_as_recovery():
    row = summary_row(prices("ABC", [100, 120, 90, 105, 120, 100]), "ABC")
    assert row["recovery_date"] == DAYS[4]


def test_unrecovered_drawdown_has_no_recovery_date():
    row = summary_row(prices("ABC", BRIEF), "ABC")
    assert pd.isna(row["recovery_date"])


# 8-11. shapes of price paths ----------------------------------------------------------------------

def test_monotonically_increasing_series_has_zero_drawdown_and_no_event():
    row = summary_row(prices("ABC", [100, 101, 105, 110]), "ABC")
    assert row["maximum_drawdown"] == 0.0
    assert pd.isna(row["peak_date"]) and pd.isna(row["trough_date"]) and pd.isna(row["recovery_date"])


def test_monotonically_decreasing_series():
    row = summary_row(prices("ABC", [100, 90, 75, 60]), "ABC")
    assert row["maximum_drawdown"] == pytest.approx(-0.40)
    assert (row["peak_price"], row["peak_date"]) == (100, DAYS[0])
    assert (row["trough_price"], row["trough_date"]) == (60, DAYS[3])
    assert pd.isna(row["recovery_date"])


def test_constant_price_gives_zero():
    row = summary_row(prices("ABC", [50, 50, 50]), "ABC")
    assert row["maximum_drawdown"] == 0.0
    assert list(calculate_drawdown_series(prices("ABC", [50, 50, 50]))["drawdown"]) == [0, 0, 0]


def test_multiple_drawdowns_selects_the_worst():
    # first fall 120 -> 108 (-10%), second fall 125 -> 100 (-20%)
    row = summary_row(prices("ABC", [100, 120, 108, 125, 100, 130]), "ABC")
    assert row["maximum_drawdown"] == pytest.approx(-0.20)
    assert (row["peak_price"], row["peak_date"]) == (125, DAYS[3])
    assert (row["trough_price"], row["trough_date"]) == (100, DAYS[4])
    assert row["recovery_date"] == DAYS[5]


def test_equal_worst_drawdowns_report_the_earliest():
    row = summary_row(prices("ABC", [100, 80, 100, 80]), "ABC")
    assert row["maximum_drawdown"] == pytest.approx(-0.20)
    assert row["trough_date"] == DAYS[1]
    assert row["recovery_date"] == DAYS[2]


# 12-13. symbols and ordering -------------------------------------------------------------------------

def test_symbols_are_independent():
    data = pd.concat([prices("ABC", BRIEF), prices("XYZ", [10, 8, 12, 9, 6, 7])])
    summary = calculate_maximum_drawdown(data).set_index("symbol")

    assert list(summary.index) == ["ABC", "XYZ"]
    assert summary.loc["ABC", "maximum_drawdown"] == pytest.approx(-0.25)
    assert summary.loc["XYZ", "maximum_drawdown"] == pytest.approx(6 / 12 - 1)   # -50%
    assert summary.loc["XYZ", "peak_price"] == 12 and summary.loc["XYZ", "trough_price"] == 6
    # XYZ's lower prices never count against ABC's running peak
    assert calculate_drawdown_series(data).query("symbol == 'ABC'")["running_peak"].min() == 100


def test_unsorted_input_is_sorted_chronologically():
    data = pd.concat([prices("ABC", BRIEF), prices("XYZ", [10, 8, 12])]).sample(frac=1, random_state=5)
    series = calculate_drawdown_series(data)

    abc = series[series["symbol"] == "ABC"]
    assert abc["date"].is_monotonic_increasing
    assert list(abc["drawdown"]) == pytest.approx([0, 0, 0, -0.10, -0.25, -0.125])
    assert summary_row(data, "ABC")["maximum_drawdown"] == pytest.approx(-0.25)


# 14-16. validation status ------------------------------------------------------------------------------

def test_invalid_rows_are_excluded():
    # The INVALID 30 would otherwise be a -75% trough.
    data = prices("ABC", [100, 120, 30, 108, 115],
                  status=["VALID", "VALID", "INVALID", "VALID", "VALID"])

    series = calculate_drawdown_series(data)
    row = summary_row(data, "ABC")

    assert list(series["close"]) == [100, 120, 108, 115]
    assert row["maximum_drawdown"] == pytest.approx(108 / 120 - 1)      # -10%
    assert row["observations"] == 4


def test_invalid_rows_create_no_artificial_peak_or_bridge():
    # An INVALID spike must not become a peak, and no price is invented for
    # its date: the series simply skips it.
    data = prices("ABC", [100, 500, 95, 101],
                  status=["VALID", "INVALID", "VALID", "VALID"])

    series = calculate_drawdown_series(data)
    row = summary_row(data, "ABC")

    assert DAYS[1] not in set(series["date"])
    assert list(series["running_peak"]) == [100, 100, 101]
    assert row["maximum_drawdown"] == pytest.approx(-0.05)
    assert row["recovery_date"] == DAYS[3]


def test_warning_rows_remain_usable():
    data = prices("ABC", BRIEF, status=["VALID", "WARNING", "VALID", "VALID", "WARNING", "VALID"])
    row = summary_row(data, "ABC")
    assert row["maximum_drawdown"] == pytest.approx(-0.25)
    assert row["observations"] == 6


def test_missing_calendar_days_are_not_filled_in():
    # Prices on Jan 1, Jan 2, then nothing until Jan 9: no rows are invented.
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-09", "2025-01-10"])
    series = calculate_drawdown_series(prices("ABC", [100, 110, 99, 104.5], dates=dates))
    assert len(series) == 4
    assert list(series["drawdown"]) == pytest.approx([0, 0, -0.10, -0.05])


# 17. tiny inputs --------------------------------------------------------------------------------------

def test_one_observation_is_not_a_misleading_zero():
    row = summary_row(prices("ABC", [100]), "ABC")
    assert row["observations"] == 1
    assert math.isnan(row["maximum_drawdown"])


def test_two_observations():
    assert summary_row(prices("ABC", [100, 80]), "ABC")["maximum_drawdown"] == pytest.approx(-0.20)
    assert summary_row(prices("XYZ", [100, 120]), "XYZ")["maximum_drawdown"] == 0.0


# 18-21. empty input, columns, duplicates, mutation ----------------------------------------------------------

def test_empty_input_gives_empty_results_with_columns():
    empty = prices("ABC", BRIEF).iloc[0:0]
    assert list(calculate_drawdown_series(empty).columns) == SERIES_COLUMNS
    summary = calculate_maximum_drawdown(empty)
    assert summary.empty and list(summary.columns) == SUMMARY_COLUMNS


@pytest.mark.parametrize("dropped", ["date", "symbol", "close"])
def test_missing_required_columns_fail_clearly(dropped):
    with pytest.raises(ValueError, match=f"missing column\\(s\\): {dropped}"):
        calculate_maximum_drawdown(prices("ABC", BRIEF).drop(columns=dropped))


def test_rows_without_symbol_are_not_used():
    data = pd.concat([prices("ABC", BRIEF), prices(None, [1])])
    assert list(calculate_maximum_drawdown(data)["symbol"]) == ["ABC"]


def test_unvalidated_duplicate_rows_fail_clearly():
    data = pd.concat([prices("ABC", BRIEF), prices("ABC", [101])])
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        calculate_maximum_drawdown(data)


def test_validated_duplicates_are_invalid_and_excluded():
    # The canonical validator marks every copy of a duplicate INVALID.
    data = pd.concat([prices("ABC", BRIEF, status="VALID"),
                      prices("ABC", [70], status="INVALID")])
    data.loc[data["close"] == 100, "validation_status"] = "INVALID"
    row = summary_row(data, "ABC")
    assert row["observations"] == 5
    assert row["maximum_drawdown"] == pytest.approx(-0.25)


def test_input_is_not_mutated():
    data = prices("ABC", BRIEF, status="VALID").iloc[::-1]
    before = data.copy()
    calculate_drawdown_series(data)
    calculate_maximum_drawdown(data)
    pd.testing.assert_frame_equal(data, before)


# end-to-end: 3-symbol synthetic sample through the CSV importer --------------------------------------------------

def test_three_symbol_sample_through_csv_importer():
    canonical = load_csv_market_data(SAMPLE_3_SYMBOLS).data
    summary = calculate_maximum_drawdown(canonical).set_index("symbol")

    assert list(summary.index) == ["ABC", "LMN", "XYZ"]
    for symbol in summary.index:
        closes = list(canonical[canonical["symbol"] == symbol].sort_values("date")["close"])
        worst, peak = 0.0, closes[0]
        for close in closes:                              # independent running-peak loop
            peak = max(peak, close)
            worst = min(worst, close / peak - 1)
        assert summary.loc[symbol, "maximum_drawdown"] == pytest.approx(worst)
        assert summary.loc[symbol, "peak_date"] <= summary.loc[symbol, "trough_date"]
