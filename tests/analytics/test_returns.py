"""Tests for daily returns on canonical data (the older layout is covered in tests/test_market_data.py)."""

import math

import pandas as pd
import pytest

from app.analytics.returns import CANONICAL_DAILY_RETURN, DAILY_RETURN, calculate_daily_returns


def canonical(closes, symbol="ABC", status=None):
    frame = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=len(closes)),
                          "symbol": symbol, "close": [float(c) for c in closes]})
    if status is not None:
        frame["validation_status"] = status
    return frame


def test_canonical_layout_gets_daily_return_column():
    result = calculate_daily_returns(canonical([100, 102, 101]))

    assert CANONICAL_DAILY_RETURN in result.columns
    assert DAILY_RETURN not in result.columns
    assert math.isnan(result.loc[0, CANONICAL_DAILY_RETURN])
    assert result.loc[1, CANONICAL_DAILY_RETURN] == pytest.approx(0.02)
    assert result.loc[2, CANONICAL_DAILY_RETURN] == pytest.approx(101 / 102 - 1)


def test_canonical_returns_are_per_symbol():
    data = pd.concat([canonical([100, 105], "ABC"), canonical([10, 9], "XYZ")])
    result = calculate_daily_returns(data).set_index(["symbol", "date"])[CANONICAL_DAILY_RETURN]

    assert list(result.xs("ABC").dropna()) == pytest.approx([0.05])
    assert list(result.xs("XYZ").dropna()) == pytest.approx([-0.1])


def test_invalid_row_blocks_both_adjacent_returns():
    result = calculate_daily_returns(
        canonical([100, 102, 999, 103, 104], status=["VALID", "VALID", "INVALID", "VALID", "VALID"]))
    returns = list(result[CANONICAL_DAILY_RETURN])

    assert returns[1] == pytest.approx(0.02)
    assert math.isnan(returns[2])          # into the invalid row
    assert math.isnan(returns[3])          # out of the invalid row (no bridging 103/102)
    assert returns[4] == pytest.approx(104 / 103 - 1)


def test_warning_rows_and_missing_status_are_usable():
    with_warning = calculate_daily_returns(canonical([100, 105], status=["WARNING", "WARNING"]))
    without_status = calculate_daily_returns(canonical([100, 105]))
    assert with_warning.loc[1, CANONICAL_DAILY_RETURN] == pytest.approx(0.05)
    assert without_status.loc[1, CANONICAL_DAILY_RETURN] == pytest.approx(0.05)


def test_canonical_input_is_not_mutated():
    data = canonical([100, 102, 101], status="VALID")
    before = data.copy()
    calculate_daily_returns(data)
    pd.testing.assert_frame_equal(data, before)


def test_missing_columns_name_both_layouts():
    with pytest.raises(ValueError, match="close \\(canonical layout\\).*Close \\(Phase 1"):
        calculate_daily_returns(canonical([100, 101]).drop(columns="close"))


def test_rows_without_a_date_get_no_return():
    # Unvalidated data: the undated row sorts last and must not be compared
    # with the last dated close.
    data = canonical([100, 102, 999])
    data.loc[2, "date"] = pd.NaT
    result = calculate_daily_returns(data)
    assert result[CANONICAL_DAILY_RETURN].notna().sum() == 1
    assert result.loc[1, CANONICAL_DAILY_RETURN] == pytest.approx(0.02)
