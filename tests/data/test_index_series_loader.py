"""Tests for the market-index file loader."""

from pathlib import Path

import pandas as pd
import pytest

from app.data.loaders.index_series_loader import (
    INDEX_SERIES_COLUMNS,
    IndexImportError,
    IndexNameRequiredError,
    load_index_csv,
    validate_index_series,
)

SAMPLE_INDEX = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_index_data.csv"


def write(tmp_path, text, name="index.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_canonical_layout_with_two_indices(tmp_path):
    path = write(tmp_path, "date,index_name,close\n2026-01-06,ASPI,101\n2026-01-05,ASPI,100\n"
                           "2026-01-05,S&P SL20,50\n")
    result = load_index_csv(path)
    assert list(result.data.columns) == INDEX_SERIES_COLUMNS
    assert result.index_names == ["ASPI", "S&P SL20"]
    aspi = result.data[result.data["index_name"] == "ASPI"]
    assert list(aspi["close"]) == [100.0, 101.0]                       # sorted by date
    assert set(result.data["validation_status"]) == {"VALID"} and result.invalid_rows == 0


def test_provider_layout_needs_an_index_name_and_never_guesses(tmp_path):
    text = 'Date,Price,Open,High,Low,Vol.,Change %\n01/13/2026,"12,345.67",1,1,1,,0.1%\n'
    path = write(tmp_path, text, "ASPI.csv")
    with pytest.raises(IndexNameRequiredError, match="no Index column"):
        load_index_csv(path)
    result = load_index_csv(path, " aspi ")
    row = result.data.iloc[0]
    assert row["index_name"] == "ASPI" and row["close"] == 12345.67
    assert row["date"] == pd.Timestamp("2026-01-13")
    assert result.column_mapping == {"date": "Date", "close": "Price"}


def test_close_header_has_priority_over_price(tmp_path):
    path = write(tmp_path, "Date,Price,Close,Index\n2026-01-05,1,2,ASPI\n")
    assert load_index_csv(path).data["close"].iloc[0] == 2.0


def test_selected_name_must_match_the_index_column(tmp_path):
    path = write(tmp_path, "Date,Index,Close\n2026-01-05,ASPI,100\n")
    with pytest.raises(IndexImportError, match="does not match"):
        load_index_csv(path, "S&P SL20")
    assert load_index_csv(path, "aspi").index_names == ["ASPI"]


def test_row_problems_are_marked_not_dropped(tmp_path):
    text = ("Date,Index,Close\n2026-01-05,ASPI,100\n2026-01-06,ASPI,101\n2026-01-06,ASPI,102\n"
            "2026-01-07,ASPI,abc\n2026-01-08,ASPI,0\n,ASPI,100\nnot-a-date,ASPI,100\n"
            "2026-01-09,,100\n2026-01-10,ASPI,99\n2026-01-12,ASPI,\n")
    result = load_index_csv(write(tmp_path, text))
    assert result.rows_read == 10 and len(result.data) == 10
    assert result.issue_counts == {"DUPLICATE_INDEX_DATE": 2, "INVALID_CLOSE": 1,
                                   "NON_POSITIVE_CLOSE": 1, "MISSING_DATE": 1,
                                   "INVALID_DATE": 1, "MISSING_INDEX_NAME": 1,
                                   "WEEKEND_DATE": 1, "MISSING_CLOSE": 1}
    assert result.invalid_rows == 8 and result.warning_rows == 1
    saturday = result.data[result.data["date"] == pd.Timestamp("2026-01-10")].iloc[0]
    assert saturday["validation_status"] == "WARNING"
    assert saturday["validation_issues"] == "WEEKEND_DATE"


def test_ambiguous_dates_are_invalid_not_guessed(tmp_path):
    result = load_index_csv(write(tmp_path, "Date,Index,Close\n01/02/2026,ASPI,100\n"))
    assert result.data["validation_issues"].iloc[0] == "AMBIGUOUS_DATE"
    assert result.data["validation_status"].iloc[0] == "INVALID"
    explicit = load_index_csv(write(tmp_path, "Date,Index,Close\n01/02/2026,ASPI,100\n", "b.csv"),
                              date_format="%d/%m/%Y")
    assert explicit.data["date"].iloc[0] == pd.Timestamp("2026-02-01")


def test_tab_separated_text(tmp_path):
    result = load_index_csv(write(tmp_path, "Date\tIndex\tClose\n2026-01-05\tASPI\t1,234.5\n",
                                  "pasted.txt"))
    assert result.data["close"].iloc[0] == 1234.5


@pytest.mark.parametrize("name, text, message", [
    ("index.xlsx", "x", "Expected a .csv"),
    ("empty.csv", "", "empty"),
    ("header.csv", "Date,Index,Close\n", "no data rows"),
    ("noclose.csv", "Date,Index\n2026-01-05,ASPI\n", "missing column.*close"),
])
def test_unusable_files_raise(tmp_path, name, text, message):
    with pytest.raises(IndexImportError, match=message):
        load_index_csv(write(tmp_path, text, name))


def test_validate_index_series_does_not_mutate():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
                          "index_name": ["ASPI", "ASPI"], "close": [1.0, 2.0]})
    before = frame.copy()
    result = validate_index_series(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert list(result["validation_status"]) == ["INVALID", "INVALID"]


def test_sample_index_file_is_clean_and_synthetic_series_load():
    result = load_index_csv(SAMPLE_INDEX)
    assert result.index_names == ["ASPI", "S&P SL20"]
    assert result.invalid_rows == 0 and result.warning_rows == 0
    assert result.data.groupby("index_name").size().to_dict() == {"ASPI": 321, "S&P SL20": 321}
