from datetime import date

import pandas as pd
import pytest

from app.data.validators.market_validator import DatasetStatus, validate_market_data

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
GOOD_ROWS = [
    ["2026-01-05", "ABC.N0000", "100", "103", "99", "102", "150000"],
    ["2026-01-06", "ABC.N0000", "102", "105", "101", "104", "180000"],
    ["2026-01-05", "XYZ.N0000", "50", "51", "49", "50.5", "1000"],
]


def frame(*rows, columns=COLUMNS):
    """Raw-looking data: every cell is text, exactly as the CSV loader reads it."""
    return pd.DataFrame([list(r) for r in rows], columns=columns, dtype=str)


def issues_of(result, row):
    return set(result.row_issues.iloc[row])


# --- valid data -----------------------------------------------------------------

def test_valid_market_data_passes():
    result = validate_market_data(frame(*GOOD_ROWS))

    assert result.status is DatasetStatus.PASS
    assert (result.total_rows, result.valid_rows, result.invalid_rows) == (3, 3, 0)
    assert result.symbol_count == 2
    assert result.trading_days == 2
    assert result.date_min == pd.Timestamp("2026-01-05")
    assert result.date_max == pd.Timestamp("2026-01-06")
    assert set(result.row_status) == {"VALID"}
    assert result.missing_values == {c: 0 for c in COLUMNS}
    assert result.notes == ["Optional column(s) not provided by source: turnover, trades"]


def test_optional_columns_are_validated_when_present():
    data = frame(GOOD_ROWS[0] + ["15300000", "42"], GOOD_ROWS[1] + ["", "-3"],
                 columns=COLUMNS + ["turnover", "trades"])

    result = validate_market_data(data, max_invalid_share=1)

    assert result.missing_values["turnover"] == 1       # missing optional value is fine...
    assert "MISSING_TURNOVER" not in result.issue_counts
    assert issues_of(result, 1) == {"NEGATIVE_TRADES"}  # ...a negative one is not


def test_input_is_not_modified():
    data = frame(*GOOD_ROWS, ["bad-date", " ABC ", "0", "1", "2", "3", "-5"])
    before = data.copy()

    validate_market_data(data)

    pd.testing.assert_frame_equal(data, before)


# --- dates ----------------------------------------------------------------------

@pytest.mark.parametrize("bad_date", ["not-a-date", "2026-02-30", "05/01/2026"])
def test_invalid_date_is_reported(bad_date):
    result = validate_market_data(frame(*GOOD_ROWS, [bad_date, "ABC.N0000", "1", "1", "1", "1", "1"]),
                                  max_invalid_share=1)

    assert issues_of(result, 3) == {"INVALID_DATE"}
    assert result.row_status.iloc[3] == "INVALID"
    assert result.date_max == pd.Timestamp("2026-01-06")  # invalid date not counted


@pytest.mark.parametrize("blank", ["", "  ", "NA", "null"])
def test_missing_date_and_symbol_are_reported(blank):
    result = validate_market_data(frame(*GOOD_ROWS,
                                        [blank, "ABC.N0000", "1", "1", "1", "1", "1"],
                                        ["2026-01-07", blank, "1", "1", "1", "1", "1"]),
                                  max_invalid_share=1)

    assert issues_of(result, 3) == {"MISSING_DATE"}
    assert issues_of(result, 4) == {"MISSING_SYMBOL"}
    assert result.missing_values["date"] == 1
    assert result.missing_values["symbol"] == 1


def test_weekend_date_is_a_warning_not_an_error():
    result = validate_market_data(frame(["2026-01-03", "ABC.N0000", "1", "1", "1", "1", "1"]))

    assert result.row_status.iloc[0] == "WARNING"
    assert result.status is DatasetStatus.WARNING
    assert result.invalid_rows == 0


# --- prices ---------------------------------------------------------------------

@pytest.mark.parametrize("column, value, code", [
    ("close", "0", "NON_POSITIVE_CLOSE"),
    ("open", "-1", "NON_POSITIVE_OPEN"),
    ("close", "abc", "INVALID_CLOSE"),
    ("close", "", "MISSING_CLOSE"),
])
def test_invalid_price_is_reported(column, value, code):
    row = dict(zip(COLUMNS, ["2026-01-07", "ABC.N0000", "10", "10", "10", "10", "100"]))
    row[column] = value

    result = validate_market_data(frame(*GOOD_ROWS, list(row.values())), max_invalid_share=1)

    assert code in issues_of(result, 3)
    assert result.row_status.iloc[3] == "INVALID"


def test_thousands_separators_are_accepted():
    result = validate_market_data(frame(["2026-01-05", "ABC.N0000", "1,000", "1,050", "990", "1,020",
                                         "1,500,000"]))
    assert result.status is DatasetStatus.PASS


# --- OHLC relationships ---------------------------------------------------------

@pytest.mark.parametrize("o, h, l, c, expected", [
    ("100", "103", "101", "102", {"LOW_ABOVE_OPEN"}),
    ("104", "103", "99", "102", {"HIGH_BELOW_OPEN"}),
    ("100", "98", "99", "98.5", {"HIGH_BELOW_OPEN", "HIGH_BELOW_LOW", "CLOSE_OUTSIDE_HIGH_LOW"}),
])
def test_invalid_ohlc_relationship_is_reported(o, h, l, c, expected):
    result = validate_market_data(frame(["2026-01-05", "ABC.N0000", o, h, l, c, "100"]))

    assert issues_of(result, 0) == expected
    assert result.invalid_ohlc == {code: int(code in expected) for code in result.invalid_ohlc}
    assert result.status is DatasetStatus.FAIL  # 1 of 1 rows invalid


@pytest.mark.parametrize("c", ["100", "104"])  # below the low / above the high
def test_close_outside_high_low_is_only_a_warning(c):
    # The CSE's official closing price can fall outside the traded range for
    # thinly traded securities, so this is not universally an error.
    result = validate_market_data(frame(["2026-01-05", "ABC.N0000", "102", "103", "101", c, "100"]))

    assert issues_of(result, 0) == {"CLOSE_OUTSIDE_HIGH_LOW"}
    assert result.row_status.iloc[0] == "WARNING"
    assert result.invalid_rows == 0
    assert result.status is DatasetStatus.WARNING


def test_questionable_values_are_not_repaired():
    data = frame(["2026-01-05", "ABC.N0000", "100", "98", "99", "100", "100"])
    validate_market_data(data)
    assert data.loc[0, "high"] == "98"


# --- volume ---------------------------------------------------------------------

def test_negative_volume_is_reported():
    result = validate_market_data(frame(*GOOD_ROWS, ["2026-01-07", "ABC.N0000", "1", "1", "1", "1", "-5"]),
                                  max_invalid_share=1)
    assert issues_of(result, 3) == {"NEGATIVE_VOLUME"}


def test_missing_volume_is_an_error():
    result = validate_market_data(frame(["2026-01-05", "ABC.N0000", "1", "1", "1", "1", ""]))
    assert issues_of(result, 0) == {"MISSING_VOLUME"}


@pytest.mark.parametrize("o, h, l, c, volume, expected", [
    ("10", "10", "10", "10", "0", set()),                             # no trades, flat price: fine
    ("10", "11", "9", "10", "0", {"ZERO_VOLUME_WITH_PRICE_RANGE"}),   # price range without trades
    ("10", "10", "10", "10", "1500.5", {"NON_INTEGER_VOLUME"}),       # fractional shares
])
def test_volume_warnings(o, h, l, c, volume, expected):
    result = validate_market_data(frame(["2026-01-05", "ABC.N0000", o, h, l, c, volume]))

    assert issues_of(result, 0) == expected
    assert result.invalid_rows == 0


# --- duplicates -----------------------------------------------------------------

def test_duplicate_symbol_date_rows_are_all_flagged():
    data = frame(*GOOD_ROWS,
                 ["2026-01-05", "abc.n0000 ", "100", "103", "99", "101", "150000"])  # conflicting copy

    result = validate_market_data(data, max_invalid_share=1)

    # Neither copy is chosen automatically: both are flagged for review.
    assert result.duplicate_rows == 2
    assert issues_of(result, 0) == {"DUPLICATE_SYMBOL_DATE"}
    assert issues_of(result, 3) == {"DUPLICATE_SYMBOL_DATE"}
    assert result.exact_duplicate_rows == 0


def test_exact_duplicate_rows_are_counted():
    result = validate_market_data(frame(GOOD_ROWS[0], GOOD_ROWS[0]), max_invalid_share=1)
    assert result.exact_duplicate_rows == 1
    assert result.duplicate_rows == 2


# --- missing columns ------------------------------------------------------------

def test_missing_required_columns_fail_without_raising():
    data = frame(["2026-01-05", "ABC.N0000", "102"], columns=["date", "symbol", "close"])

    result = validate_market_data(data)

    assert result.status is DatasetStatus.FAIL
    assert result.missing_columns == ["open", "high", "low", "volume"]
    assert "Missing required column(s): open, high, low, volume" in result.failure_reasons
    assert result.invalid_rows == 1


# --- dataset status and source-date rule ----------------------------------------

def test_small_share_of_invalid_rows_is_a_warning():
    rows = [["2026-01-05", f"S{i}.N0000", "1", "1", "1", "1", "1"] for i in range(99)]
    rows.append(["2026-01-05", "BAD.N0000", "0", "1", "1", "1", "1"])

    result = validate_market_data(frame(*rows))

    assert result.invalid_rows == 1
    assert result.status is DatasetStatus.WARNING


def test_snapshot_must_carry_the_stated_source_date():
    # A "current" snapshot requested for 2026-01-05 but actually for 2026-01-06
    # must not be accepted as 2026-01-05 data.
    result = validate_market_data(frame(*GOOD_ROWS), expected_date=date(2026, 1, 5))

    assert issues_of(result, 1) == {"DATE_MISMATCH"}
    assert result.status is DatasetStatus.FAIL
    assert any("stated date" in reason for reason in result.failure_reasons)


def test_invalid_row_report_explains_each_row():
    data = frame(*GOOD_ROWS, ["2026-01-07", "ABC.N0000", "0", "1", "1", "1", "-5"])
    result = validate_market_data(data, max_invalid_share=1)

    report = result.invalid_row_report(data)

    assert list(report.index) == [3]
    assert report.loc[3, "issues"] == "NON_POSITIVE_OPEN, LOW_ABOVE_OPEN, NEGATIVE_VOLUME"


def test_summary_is_json_serializable():
    import json
    summary = validate_market_data(frame(*GOOD_ROWS)).summary()
    assert json.loads(json.dumps(summary))["date_min"] == "2026-01-05"
