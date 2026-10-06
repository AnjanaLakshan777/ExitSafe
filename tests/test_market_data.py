from pathlib import Path

import pandas as pd
import pytest

from app.analytics.returns import calculate_daily_returns
from app.data.loaders.market_data import REQUIRED_COLUMNS, load_market_data

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_market_data.csv"
HEADER = "Date,Symbol,Open,High,Low,Close,Volume,Value Traded"


def write_csv(tmp_path, *rows, header=HEADER):
    path = tmp_path / "market.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


# Test 1: valid CSV

def test_valid_csv_loads_successfully(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-03,XYZ,50,51,49,50.5,1000,50500",
        "2026-01-02,ABC,100,103,99,102,150000,15300000",
        "2026-01-02,XYZ,49,50,48,49.5,2000,99000",
        "2026-01-03,ABC,102,105,101,104,180000,18720000",
    )

    data = load_market_data(path)

    assert list(data.columns) == REQUIRED_COLUMNS
    assert len(data) == 4
    assert pd.api.types.is_datetime64_any_dtype(data["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume", "Value Traded"]:
        assert pd.api.types.is_numeric_dtype(data[col]), col
    # Sorted by Symbol, then Date
    assert list(data["Symbol"]) == ["ABC", "ABC", "XYZ", "XYZ"]
    assert list(data["Date"].dt.strftime("%Y-%m-%d")) == [
        "2026-01-02", "2026-01-03", "2026-01-02", "2026-01-03",
    ]
    assert data.loc[0, "Value Traded"] == 15_300_000


def test_sample_dataset_loads_cleanly():
    raw_rows = len(pd.read_csv(SAMPLE_FILE))

    data = load_market_data(SAMPLE_FILE)

    assert len(data) == raw_rows  # the sample has no bad rows
    assert data["Symbol"].nunique() >= 3
    assert data["Date"].nunique() > 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_market_data(tmp_path / "does_not_exist.csv")


# Test 2: missing required column

def test_missing_required_column_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-02,ABC,100,103,99,102,15300000",
        header="Date,Symbol,Open,High,Low,Close,Value Traded",
    )

    with pytest.raises(ValueError, match="missing required column.*Volume"):
        load_market_data(path)


# Test 3: invalid date

def test_invalid_date_row_is_removed(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-02,ABC,100,103,99,102,150000,15300000",
        "not-a-date,ABC,102,105,101,104,180000,18720000",
        "2026-02-30,ABC,102,105,101,104,180000,18720000",
    )

    data = load_market_data(path)

    assert len(data) == 1
    assert data["Date"].notna().all()
    assert data.loc[0, "Date"] == pd.Timestamp("2026-01-02")


# Test 4: invalid prices

def test_invalid_prices_are_removed(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-02,ABC,100,103,99,102,150000,15300000",
        "2026-01-03,ABC,102,105,101,0,180000,18720000",      # zero close
        "2026-01-04,ABC,104,104,98,-99,250000,24750000",     # negative close
        "2026-01-05,ABC,-1,104,98,99,250000,24750000",       # negative open
        "2026-01-06,ABC,104,104,98,abc,250000,24750000",     # non-numeric close
    )

    data = load_market_data(path)

    assert list(data["Date"].dt.strftime("%Y-%m-%d")) == ["2026-01-02"]
    assert (data[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_rows_missing_essential_data_are_removed(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-02,ABC,100,103,99,102,150000,15300000",
        ",ABC,102,105,101,104,180000,18720000",              # no date
        "2026-01-03,,102,105,101,104,180000,18720000",       # no symbol
        "2026-01-04,ABC,104,104,98,,250000,24750000",        # no close
    )

    data = load_market_data(path)

    assert len(data) == 1
    assert data[["Date", "Symbol", "Close"]].notna().all().all()


# Test 5: duplicates

def test_duplicate_stock_date_records_are_removed(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-02,ABC,100,103,99,102,150000,15300000",
        "2026-01-02,XYZ,50,51,49,50,1000,50000",
        "2026-01-02,ABC,100,103,99,101,150000,15150000",     # later correction
    )

    data = load_market_data(path)

    assert len(data) == 2
    assert not data.duplicated(subset=["Symbol", "Date"]).any()
    abc = data[data["Symbol"] == "ABC"].iloc[0]
    assert abc["Close"] == 101  # the last record in the file is kept


# Test 6: daily returns

def test_daily_return_calculation():
    data = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
        "Symbol": ["ABC", "ABC"],
        "Close": [100.0, 105.0],
    })

    result = calculate_daily_returns(data)

    assert pd.isna(result.loc[0, "Daily Return"])
    assert result.loc[1, "Daily Return"] == pytest.approx(0.05)


def test_daily_returns_are_calculated_per_stock():
    # Rows deliberately interleaved and unsorted.
    data = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-03", "2026-01-02", "2026-01-03", "2026-01-02"]),
        "Symbol": ["XYZ", "XYZ", "ABC", "ABC"],
        "Close": [55.0, 50.0, 90.0, 100.0],
    })

    result = calculate_daily_returns(data)
    returns = result.set_index(["Symbol", result["Date"].dt.strftime("%Y-%m-%d")])["Daily Return"]

    # First day of each stock has no return; it is never computed from another stock.
    assert pd.isna(returns[("ABC", "2026-01-02")])
    assert pd.isna(returns[("XYZ", "2026-01-02")])
    assert returns[("ABC", "2026-01-03")] == pytest.approx(-0.10)
    assert returns[("XYZ", "2026-01-03")] == pytest.approx(0.10)
    assert "Daily Return" not in data.columns  # input not modified


def test_daily_returns_on_sample_dataset():
    result = calculate_daily_returns(load_market_data(SAMPLE_FILE))

    for _, group in result.groupby("Symbol"):
        expected = group["Close"] / group["Close"].shift(1) - 1
        pd.testing.assert_series_equal(group["Daily Return"], expected, check_names=False)
        assert group["Daily Return"].isna().sum() == 1
