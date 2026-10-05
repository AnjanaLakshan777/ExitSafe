import hashlib
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from app.analytics.returns import calculate_daily_returns
from app.data.loaders.csv_market_loader import (
    MarketDataImportError,
    load_csv_market_data,
    normalize_dates,
    normalize_market_frame,
    parse_number_text,
    parse_percent_text,
)
from app.data.schemas.market_schema import CANONICAL_COLUMNS
from app.data.source_catalog import DataSourceType, get_source
from app.data.validators.market_validator import DatasetStatus

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_date_price_vol_change.csv"
HEADER = "Date,Price,Open,High,Low,Vol.,Change %"


def write_csv(tmp_path, *rows, header=HEADER, name="prices.csv"):
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def issues(result, row):
    return set(result.data.loc[row, "validation_warnings"].split(";")) - {""}


# --- 1. the Date/Price/Open/High/Low/Vol./Change % format ---------------------------------

def test_example_format_imports(tmp_path):
    path = write_csv(tmp_path,
                     "12/31/2025,660.09,664.75,665,659.44,7.94M,-0.88%",
                     "12/30/2025,665.95,658.69,672.22,657.84,9.19M,1.10%")

    result = load_csv_market_data(path, symbol="TEST.N0000")
    data = result.data.set_index("date")

    assert list(result.data.columns) == CANONICAL_COLUMNS
    assert data.loc["2025-12-31", "close"] == 660.09
    assert data.loc["2025-12-31", "open"] == 664.75
    assert data.loc["2025-12-31", "volume"] == 7_940_000
    assert data.loc["2025-12-31", "change_pct"] == -0.88
    assert set(result.data["validation_status"]) == {"VALID"}
    assert result.report.column_mapping == {
        "Date": "date", "Price": "close", "Open": "open", "High": "high", "Low": "low",
        "Vol.": "volume", "Change %": "change_pct"}


def test_synthetic_fixture_imports_cleanly():
    result = load_csv_market_data(FIXTURE, symbol="TEST.N0000")

    assert result.validation.status is DatasetStatus.PASS
    assert result.report.rows_read == result.report.valid_rows == 5
    assert (result.report.date_min, result.report.date_max) == ("2025-12-24", "2025-12-31")
    assert result.report.date_convention == "MM/DD/YYYY"
    assert list(result.data["volume"]) == [7_940_000, 9_190_000, 560_000, 8_510_000, 7_130_000]


# --- 2-3. close column mapping -------------------------------------------------------------

def test_price_maps_to_close(tmp_path):
    result = load_csv_market_data(write_csv(tmp_path, "2025-12-31,660.09,664.75,665,659.44,100,"),
                                  symbol="T")
    assert result.data.loc[0, "close"] == 660.09
    assert result.report.column_mapping["Price"] == "close"


@pytest.mark.parametrize("header", ["Close", "close", "Closing Price"])
def test_close_column_maps(tmp_path, header):
    path = write_csv(tmp_path, "2025-12-31,660.09,664.75,665,659.44,100",
                     header=f"Date,{header},Open,High,Low,Volume")
    result = load_csv_market_data(path, symbol="T")
    assert result.data.loc[0, "close"] == 660.09
    assert result.report.column_mapping[header] == "close"


def test_close_is_preferred_over_identical_price(tmp_path):
    path = write_csv(tmp_path, "2025-12-31,660.09,660.09,664.75,665,659.44,100",
                     header="Date,Price,Close,Open,High,Low,Volume")
    result = load_csv_market_data(path, symbol="T")

    assert result.report.column_mapping["Close"] == "close"
    assert "Price" in result.report.ignored_columns
    assert "identical" in result.report.ignored_columns["Price"]
    assert result.data.loc[0, "validation_status"] == "VALID"


def test_conflicting_price_and_close_are_not_silently_resolved(tmp_path):
    path = write_csv(tmp_path,
                     "2025-12-31,660.09,661.00,664.75,665,659.44,100",
                     "2025-12-30,665.95,665.95,658.69,672.22,657.84,100",
                     header="Date,Price,Close,Open,High,Low,Volume")
    result = load_csv_market_data(path, symbol="T")
    data = result.data.set_index("date")

    assert data.loc["2025-12-31", "close"] == 661.00                     # Close kept...
    assert data.loc["2025-12-31", "validation_status"] == "INVALID"      # ...but flagged
    assert "CLOSE_SOURCE_CONFLICT" in data.loc["2025-12-31", "validation_warnings"]
    assert data.loc["2025-12-30", "validation_status"] == "VALID"
    assert "1 row(s) differ" in result.report.ignored_columns["Price"]


# --- 4-7. volume parsing -------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("560K", "560000"),
    ("0.5k", "500"),
    ("7.94M", "7940000"),
    ("9.19M", "9190000"),
    ("1.2B", "1200000000"),
    ("250000", "250000"),
    ("250,000", "250000"),
    ("1,234,567.5", "1234567.5"),
    (250000, "250000"),
    (7940000.0, "7940000"),
    ("", ""),
    ("-", ""),
])
def test_volume_values_are_parsed(text, expected):
    assert parse_number_text(text, allow_suffix=True) == expected


@pytest.mark.parametrize("text", ["1.234,5", "1,23", "12,34,567", "7.94X", "abc", "1.2.3", "M"])
def test_ambiguous_or_malformed_numbers_are_rejected(text):
    assert parse_number_text(text, allow_suffix=True) is None


def test_suffixes_are_exact_not_floating_point():
    # 7.94 * 1e6 in floating point is 7940000.000000001
    assert Decimal(parse_number_text("7.94M", allow_suffix=True)) == Decimal(7_940_000)


def test_prices_do_not_accept_suffixes():
    assert parse_number_text("1.2K") is None


def test_comma_separated_volume_in_file(tmp_path):
    result = load_csv_market_data(write_csv(tmp_path, '2025-12-31,660.09,664.75,665,659.44,"1,250,000",'),
                                  symbol="T")
    assert result.data.loc[0, "volume"] == 1_250_000


def test_abbreviated_volume_precision_is_noted(tmp_path):
    result = load_csv_market_data(write_csv(tmp_path, "2025-12-31,660.09,664.75,665,659.44,7.94M,"),
                                  symbol="T")
    assert any("K/M/B abbreviations" in note for note in result.report.notes)


# --- 8-9. Change % ---------------------------------------------------------------------------

def test_missing_change_column_is_allowed(tmp_path):
    path = write_csv(tmp_path, "12/31/2025,660.09,664.75,665,659.44,7.94M",
                     header="Date,Price,Open,High,Low,Vol.")
    result = load_csv_market_data(path, symbol="T")

    assert result.validation.status is DatasetStatus.PASS
    assert result.data["change_pct"].isna().all()


@pytest.mark.parametrize("text, units, expected", [
    ("-0.88%", False, "-0.88"), ("1.10 %", False, "1.1"), ("+2%", False, "2"),
    ("-0.88", True, "-0.88"), ("-0.88", False, None), ("", False, ""),
])
def test_percent_parsing(text, units, expected):
    assert parse_percent_text(text, percent_units=units) == expected


def test_supplied_change_does_not_replace_calculated_return(tmp_path):
    path = write_csv(tmp_path,
                     "12/31/2025,105.00,100,106,99,1000,9.99%",   # true change is +5.00%
                     "12/30/2025,100.00,100,101,99,1000,0.00%")
    result = load_csv_market_data(path, symbol="T")
    data = result.data.set_index("date")

    assert data.loc["2025-12-31", "change_pct"] == 9.99            # kept as supplied
    assert "CHANGE_PCT_MISMATCH" in data.loc["2025-12-31", "validation_warnings"]
    assert data.loc["2025-12-31", "validation_status"] == "WARNING"

    legacy = result.data.rename(columns={"date": "Date", "symbol": "Symbol", "close": "Close"})
    returns = calculate_daily_returns(legacy).set_index("Date")["Daily Return"]
    assert returns.loc["2025-12-31"] == pytest.approx(0.05)        # from closes, not 9.99%


def test_consistent_change_raises_no_warning():
    result = load_csv_market_data(FIXTURE, symbol="T")
    assert "CHANGE_PCT_MISMATCH" not in result.validation.issue_counts


def test_bare_change_column_without_percent_sign_is_ignored(tmp_path):
    path = write_csv(tmp_path, "12/31/2025,660.09,664.75,665,659.44,100,-5.86",
                     header="Date,Price,Open,High,Low,Volume,Change")
    result = load_csv_market_data(path, symbol="T")

    assert "Change" in result.report.ignored_columns
    assert "ambiguous" in result.report.ignored_columns["Change"]
    assert result.data["change_pct"].isna().all()


# --- 10-11. symbol ---------------------------------------------------------------------------

def test_symbol_supplied_externally(tmp_path):
    result = load_csv_market_data(write_csv(tmp_path, "12/31/2025,660.09,664.75,665,659.44,100,"),
                                  symbol=" jkh.n0000 ")
    assert list(result.data["symbol"]) == ["JKH.N0000"]
    assert result.report.symbol_source == "parameter"
    assert result.report.symbols == ["JKH.N0000"]


def test_symbol_column_is_used(tmp_path):
    path = write_csv(tmp_path, "JKH.N0000,2025-12-31,18.9,18.8,19,18.7,100",
                     header="Symbol,Date,Close,Open,High,Low,Volume")
    result = load_csv_market_data(path)
    assert result.report.symbol_source == "column"
    assert result.data.loc[0, "symbol"] == "JKH.N0000"


def test_symbol_parameter_conflicting_with_column_fails(tmp_path):
    path = write_csv(tmp_path, "JKH.N0000,2025-12-31,18.9,18.8,19,18.7,100",
                     header="Symbol,Date,Close,Open,High,Low,Volume")
    with pytest.raises(MarketDataImportError, match="conflicts"):
        load_csv_market_data(path, symbol="HNB.N0000")


def test_missing_symbol_fails_clearly(tmp_path):
    with pytest.raises(MarketDataImportError, match="no Symbol column and no symbol was given"):
        load_csv_market_data(write_csv(tmp_path, "12/31/2025,660.09,664.75,665,659.44,100,"))


def test_symbol_from_filename_only_when_enabled(tmp_path):
    path = write_csv(tmp_path, "12/31/2025,660.09,664.75,665,659.44,100,", name="JKH.N0000.csv")

    with pytest.raises(MarketDataImportError):
        load_csv_market_data(path)
    result = load_csv_market_data(path, infer_symbol_from_filename=True)
    assert result.report.symbol_source == "file name"
    assert result.data.loc[0, "symbol"] == "JKH.N0000"


def test_symbol_is_not_guessed_from_a_descriptive_filename(tmp_path):
    path = write_csv(tmp_path, "12/31/2025,660.09,664.75,665,659.44,100,",
                     name="JKH Historical Data.csv")
    with pytest.raises(MarketDataImportError, match="exactly"):
        load_csv_market_data(path, infer_symbol_from_filename=True)


# --- 12-13. dates ----------------------------------------------------------------------------

@pytest.mark.parametrize("values, expected, convention", [
    (["12/31/2025", "01/02/2025"], ["2025-12-31", "2025-01-02"], "MM/DD/YYYY"),
    (["31/12/2025", "01/02/2025"], ["2025-12-31", "2025-02-01"], "DD/MM/YYYY"),
    (["2025-12-31", "2025-01-02"], ["2025-12-31", "2025-01-02"], "YYYY-MM-DD"),
])
def test_date_formats_are_detected(values, expected, convention):
    result, ambiguous, detected = normalize_dates(pd.Series(values, dtype="string"))
    assert list(result) == expected
    assert not ambiguous.any()
    assert detected == convention


def test_ambiguous_dates_are_rejected_not_swapped(tmp_path):
    # Every day and month part is <= 12, so 01/02/2025 could be Jan 2 or Feb 1.
    path = write_csv(tmp_path,
                     "01/02/2025,100,100,101,99,100,",
                     "05/05/2025,101,100,101,99,100,")   # day == month: safe either way
    result = load_csv_market_data(path, symbol="T")

    assert "AMBIGUOUS_DATE" in issues(result, 0)
    assert result.data.loc[0, "validation_status"] == "INVALID"
    assert pd.isna(result.data.loc[0, "date"])
    assert result.data.loc[1, "date"] == pd.Timestamp("2025-05-05")
    assert any("date_format" in note for note in result.report.notes)


def test_explicit_date_format_resolves_ambiguity(tmp_path):
    path = write_csv(tmp_path, "01/02/2025,100,100,101,99,100,")
    result = load_csv_market_data(path, symbol="T", date_format="%d/%m/%Y")
    assert result.data.loc[0, "date"] == pd.Timestamp("2025-02-01")


def test_impossible_and_unreadable_dates_are_invalid(tmp_path):
    path = write_csv(tmp_path,
                     "12/31/2025,100,100,101,99,100,",
                     "02/30/2025,100,100,101,99,100,",
                     "not-a-date,100,100,101,99,100,")
    result = load_csv_market_data(path, symbol="T")
    assert "INVALID_DATE" in issues(result, 1)
    assert "INVALID_DATE" in issues(result, 2)


# --- 14-16. prices, volume, duplicates (shared validator) -------------------------------------

@pytest.mark.parametrize("close, code", [("0", "NON_POSITIVE_CLOSE"), ("-5", "NON_POSITIVE_CLOSE"),
                                         ("abc", "INVALID_CLOSE"), ("", "MISSING_CLOSE")])
def test_invalid_price(tmp_path, close, code):
    result = load_csv_market_data(write_csv(tmp_path, f"12/31/2025,{close},100,101,99,100,"),
                                  symbol="T")
    assert code in issues(result, 0)
    assert result.data.loc[0, "validation_status"] == "INVALID"


@pytest.mark.parametrize("volume, code", [("-100", "NEGATIVE_VOLUME"), ("lots", "INVALID_VOLUME"),
                                          ("7.94X", "INVALID_VOLUME"), ("", "MISSING_VOLUME")])
def test_invalid_volume(tmp_path, volume, code):
    result = load_csv_market_data(write_csv(tmp_path, f"12/31/2025,100,100,101,99,{volume},"),
                                  symbol="T")
    assert code in issues(result, 0)


def test_high_below_low_is_an_error_but_close_outside_range_is_a_warning(tmp_path):
    path = write_csv(tmp_path,
                     "12/31/2025,100,100,99,101,100,",    # high < low
                     "12/30/2025,98,100,101,99,100,")     # close below low (CSE behaviour)
    result = load_csv_market_data(path, symbol="T")
    assert "HIGH_BELOW_LOW" in issues(result, 0)
    assert issues(result, 1) == {"CLOSE_OUTSIDE_HIGH_LOW"}
    assert result.data.loc[1, "validation_status"] == "WARNING"


def test_duplicate_symbol_date_rows_are_flagged(tmp_path):
    path = write_csv(tmp_path,
                     "12/31/2025,100,100,101,99,100,",
                     "2025-12-31,101,100,101,99,100,")    # same day, other format
    result = load_csv_market_data(path, symbol="T")
    assert result.validation.duplicate_rows == 2
    assert all("DUPLICATE_SYMBOL_DATE" in issues(result, i) for i in (0, 1))


def test_missing_required_column_fails_without_data(tmp_path):
    path = write_csv(tmp_path, "12/31/2025,100", header="Date,Price")
    result = load_csv_market_data(path, symbol="T")

    assert result.data is None
    assert result.validation.status is DatasetStatus.FAIL
    assert result.validation.missing_columns == ["open", "high", "low", "volume"]


# --- 17-18. traded value -----------------------------------------------------------------------

def test_estimated_traded_value_is_derived_and_marked(tmp_path):
    result = load_csv_market_data(write_csv(tmp_path, "12/31/2025,100.50,100,101,99,2K,"),
                                  symbol="T")

    assert result.data.loc[0, "estimated_traded_value"] == pytest.approx(201_000)
    assert pd.isna(result.data.loc[0, "turnover"])                     # never called turnover
    assert "estimated_traded_value" in result.provenance.derived_fields
    assert "estimated_traded_value" in result.report.source_metadata["derived_fields"]


def test_reported_turnover_is_untouched(tmp_path):
    path = write_csv(tmp_path, '12/31/2025,100.50,100,101,99,2000,"199,999.75"',
                     header="Date,Close,Open,High,Low,Volume,Turnover")
    result = load_csv_market_data(path, symbol="T")

    assert result.data.loc[0, "turnover"] == 199_999.75                # not close * volume
    assert pd.isna(result.data.loc[0, "estimated_traded_value"])
    assert result.provenance.derived_fields == {}


# --- 19-20. no mutation, provenance -----------------------------------------------------------

def test_file_and_dataframe_are_not_mutated(tmp_path):
    path = write_csv(tmp_path, "12/31/2025,660.09,664.75,665,659.44,7.94M,-0.88%")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    load_csv_market_data(path, symbol="T")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

    raw = pd.DataFrame({"Date": ["12/31/2025"], "Price": ["660.09"], "Open": ["664.75"],
                        "High": ["665"], "Low": ["659.44"], "Vol.": ["7.94M"],
                        "Change %": ["-0.88%"]})
    copy = raw.copy()
    normalized = normalize_market_frame(raw, symbol="T")
    pd.testing.assert_frame_equal(raw, copy)
    assert normalized.data.loc[0, "volume"] == "7940000"


def test_source_and_provenance_are_preserved():
    result = load_csv_market_data(FIXTURE, symbol="TEST.N0000", source_url="https://example.com/x.csv")
    p = result.provenance

    assert p.source_name == "user_csv_upload"
    assert p.source_type is DataSourceType.OTHER
    assert p.original_file_name == FIXTURE.name
    assert p.file_sha256 == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert p.source_url == "https://example.com/x.csv"
    assert (result.data["source"] == "user_csv_upload").all()
    assert (result.data["source_priority"] == get_source("user_csv_upload").source_priority).all()
    assert result.report.source_metadata["file_sha256"] == p.file_sha256


def test_other_catalog_source_can_be_named(tmp_path):
    path = write_csv(tmp_path, "2026-01-02,ABC,100,103,99,102,150000,15300000",
                     header="Date,Symbol,Open,High,Low,Close,Volume,Value Traded")
    result = load_csv_market_data(path, source_name="exitsafe_sample")
    assert result.data.loc[0, "turnover"] == 15_300_000
    assert result.data.loc[0, "source"] == "exitsafe_sample"


# --- other failures ----------------------------------------------------------------------------

def test_report_lists_columns_and_counts():
    report = load_csv_market_data(FIXTURE, symbol="T").report.to_dict()
    assert report["detected_columns"] == ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    assert {"close", "volume", "change_pct", "estimated_traded_value"} <= set(report["normalized_columns"])
    assert report["rows_read"] == 5 and report["invalid_rows"] == 0


def test_non_csv_and_empty_files_are_rejected(tmp_path):
    with pytest.raises(MarketDataImportError, match=".csv"):
        load_csv_market_data(tmp_path / "prices.xlsx", symbol="T")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(MarketDataImportError, match="empty"):
        load_csv_market_data(empty, symbol="T")


def test_unrecognised_columns_are_reported(tmp_path):
    path = write_csv(tmp_path, "12/31/2025,100,100,101,99,100,,x",
                     header=HEADER + ",Notes")
    result = load_csv_market_data(path, symbol="T")
    assert result.report.ignored_columns["Notes"] == "not a recognised market-data column"
