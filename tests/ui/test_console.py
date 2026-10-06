"""Tests for the non-Streamlit helpers behind the analytics test console."""

import math

import pandas as pd
import pytest

from app.ui.console import (
    EXAMPLE_CSV,
    EXAMPLE_FILE_NAME,
    SAMPLE_CSV,
    SAMPLE_SYMBOL,
    canonical_column_order,
    format_percent,
    import_summary,
    run_import,
    status_level,
    volatility_display,
)


def test_sample_csv_runs_through_import_returns_and_volatility():
    outcome = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL)

    assert outcome.error is None
    assert outcome.import_result.report.symbols == [SAMPLE_SYMBOL]
    assert list(outcome.returns.columns) == ["date", "symbol", "close", "validation_status",
                                             "daily_return"]
    assert outcome.returns["daily_return"].notna().sum() == 4
    vol = outcome.volatility.iloc[0]
    assert vol["observations"] == 4
    assert vol["annualized_volatility"] == pytest.approx(vol["daily_volatility"] * math.sqrt(252))


def test_example_csv_is_importable_and_keeps_its_file_name():
    outcome = run_import(EXAMPLE_FILE_NAME, EXAMPLE_CSV.encode("utf-8"), " ex.n0000 ")
    report = outcome.import_result.report

    assert outcome.error is None
    assert report.status == "PASS"
    assert report.column_mapping["Price"] == "close"
    assert report.column_mapping["Vol."] == "volume"
    assert report.symbols == ["EX.N0000"]
    assert outcome.import_result.provenance.original_file_name == EXAMPLE_FILE_NAME


def test_missing_symbol_asks_the_user_for_one():
    outcome = run_import("x.csv", EXAMPLE_CSV.encode("utf-8"), "  ")
    assert outcome.needs_symbol                    # the UI shows a Stock Symbol field
    assert outcome.error.startswith("Symbol is required")
    assert outcome.import_result is None
    assert "no Symbol column" in outcome.error_detail

    retried = run_import("x.csv", EXAMPLE_CSV.encode("utf-8"), "ENTERED.N0000")
    assert not retried.needs_symbol and retried.error is None
    assert retried.import_result.report.symbols == ["ENTERED.N0000"]


def test_symbol_column_in_data_needs_no_prompt():
    content = b"Symbol,Date,Close,Open,High,Low,Volume\nABC.N0000,2026-01-05,20.1,20,20.3,19.9,100\n"
    outcome = run_import("with_symbol.csv", content)
    assert not outcome.needs_symbol and outcome.error is None
    assert outcome.import_result.report.symbol_source == "column"
    assert outcome.import_result.report.symbols == ["ABC.N0000"]


def test_other_import_errors_do_not_ask_for_a_symbol():
    outcome = run_import("x.csv", b"Date,Close\n2025-12-31,100\n", "T")
    assert not outcome.needs_symbol
    assert outcome.error.startswith("Required market-data columns are missing")


def test_missing_required_columns_are_reported():
    outcome = run_import("x.csv", b"Date,Close\n2025-12-31,100\n", "T")
    assert outcome.error == "Required market-data columns are missing: open, high, low, volume."
    assert outcome.import_result is not None      # summary can still be shown
    assert outcome.returns is None and outcome.volatility is None


@pytest.mark.parametrize("content, expected", [
    (b"", "CSV could not be imported: the file is empty."),
    (b"\xff\xfe\x00\x01not,a\x00csv", "CSV could not be imported."),
])
def test_unreadable_files_give_short_errors(content, expected):
    outcome = run_import("x.csv", content, "T")
    assert outcome.error == expected
    assert outcome.error_detail                    # technical detail kept for the expander


def test_import_summary_rows():
    outcome = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL)
    summary = dict(import_summary(outcome.import_result))
    assert summary["Symbol"] == SAMPLE_SYMBOL
    assert summary["Rows read"] == 5
    assert summary["Date range"] == "2025-12-24 to 2025-12-31"
    assert summary["Validation status"] == "PASS"


@pytest.mark.parametrize("status, level", [("PASS", "success"), ("WARNING", "warning"),
                                           ("FAIL", "error")])
def test_status_level(status, level):
    assert status_level(status) == level


@pytest.mark.parametrize("value, text", [(0.0124, "1.24%"), (0.19694, "19.69%"), (0.0, "0.00%"),
                                         (float("nan"), "n/a"), (None, "n/a")])
def test_format_percent(value, text):
    assert format_percent(value) == text


def test_volatility_display_is_a_separate_formatted_copy():
    volatility = pd.DataFrame({"symbol": ["A", "B"], "observations": [4, 1],
                               "daily_volatility": [0.0123557, float("nan")],
                               "annualized_volatility": [0.1961409, float("nan")],
                               "periods_per_year": [252, 252]})
    before = volatility.copy()

    shown = volatility_display(volatility)

    pd.testing.assert_frame_equal(volatility, before)      # underlying values untouched
    assert list(shown["Daily Volatility"]) == ["1.24%", "n/a (needs at least 2 usable returns)"]
    assert list(shown["Annualized Volatility"])[0] == "19.61%"


def test_canonical_column_order_puts_key_columns_first():
    data = pd.DataFrame(columns=["source", "close", "date", "extra"])
    assert canonical_column_order(data) == ["date", "close", "source", "extra"]


def test_pasted_tab_separated_text_is_imported():
    from app.ui.console import PASTED_FILE_NAME, pasted_bytes

    text = ("Date\tPrice\tOpen\tHigh\tLow\tVol.\tChange %\n"
            "12/31/2025\t101.20\t102.10\t102.40\t100.90\t7.94M\t-0.78%\n"
            "12/30/2025\t102.00\t100.90\t102.30\t100.60\t9.19M\t1.24%\n")
    outcome = run_import(PASTED_FILE_NAME, pasted_bytes(text), "T.N0000")

    assert outcome.error is None
    summary = dict(import_summary(outcome.import_result))
    assert summary["Delimiter"] == "tab"
    assert summary["Validation status"] == "PASS"
    assert outcome.volatility.iloc[0]["observations"] == 1


@pytest.mark.parametrize("text", ["", "   \n  ", None])
def test_empty_paste_is_ignored(text):
    from app.ui.console import pasted_bytes
    assert pasted_bytes(text) is None


def test_three_symbol_sample_produces_covariance_and_correlation():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, alignment_summary

    outcome = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes())

    assert outcome.error is None
    for matrix in (outcome.covariance, outcome.annualized_covariance, outcome.correlation):
        assert list(matrix.index) == list(matrix.columns) == ["ABC", "LMN", "XYZ"]
    assert outcome.annualized_covariance.equals(outcome.covariance * 252)
    summary = dict(alignment_summary(outcome.alignment))
    assert summary["Symbols"] == "ABC, LMN, XYZ"
    assert summary["Common return observations used"] == 24
    assert summary["Date range used"] == "2026-01-05 to 2026-02-05"


def test_single_symbol_sample_gives_one_by_one_matrices():
    outcome = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL)
    assert outcome.covariance.shape == outcome.correlation.shape == (1, 1)
    assert outcome.correlation.iloc[0, 0] == 1.0


def test_matrix_display_formats_without_changing_values():
    from app.ui.console import matrix_display

    matrix = pd.DataFrame([[0.000123456789, float("nan")], [float("nan"), 1.0]],
                          index=["A", "B"], columns=["A", "B"])
    before = matrix.copy()

    html = matrix_display(matrix, 8).to_html()

    pd.testing.assert_frame_equal(matrix, before)
    assert "0.00012346" in html and "n/a" in html and "1.00000000" in html
