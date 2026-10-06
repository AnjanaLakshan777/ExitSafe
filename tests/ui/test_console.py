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


def test_three_symbol_sample_produces_separate_drawdowns():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, drawdown_chart_data, maximum_drawdown_display

    outcome = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes())

    assert list(outcome.maximum_drawdown["symbol"]) == ["ABC", "LMN", "XYZ"]
    assert (outcome.maximum_drawdown["maximum_drawdown"] <= 0).all()
    shown = maximum_drawdown_display(outcome.maximum_drawdown)
    assert all(text.endswith("%") and text.startswith("-") for text in shown["Maximum Drawdown"])
    assert list(drawdown_chart_data(outcome.drawdown_series).columns) == ["ABC", "LMN", "XYZ"]


def test_drawdown_display_marks_unrecovered_and_undetermined():
    from app.ui.console import maximum_drawdown_display

    summary = pd.DataFrame({
        "symbol": ["DOWN", "UP", "ONE"], "observations": [3, 3, 1],
        "maximum_drawdown": [-0.25, 0.0, float("nan")],
        "peak_date": pd.to_datetime(["2025-01-01", None, None]), "peak_price": [120.0, None, None],
        "trough_date": pd.to_datetime(["2025-01-03", None, None]), "trough_price": [90.0, None, None],
        "recovery_date": pd.to_datetime([None, None, None])})

    shown = maximum_drawdown_display(summary)

    assert list(shown["Maximum Drawdown"]) == ["-25.00%", "0.00%", "n/a"]
    assert list(shown["Recovery Date"]) == ["not recovered", "n/a", "n/a"]
    assert list(shown["Peak Price"]) == ["120.00", "n/a", "n/a"]


def test_risk_adjusted_ratios_take_percent_input():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, ratios_display, risk_adjusted_ratios

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    zero, error = risk_adjusted_ratios(data, 0.0, 252)
    five, _ = risk_adjusted_ratios(data, 5.0, 252)

    assert error is None and list(zero["symbol"]) == ["ABC", "LMN", "XYZ"]
    assert (five["risk_free_rate"] == 0.05).all()
    assert (five["annualized_excess_return"] < zero["annualized_excess_return"]).all()
    shown = ratios_display(five)
    assert list(shown.columns) == ["Symbol", "Observations", "Annualized Return",
                                   "Annualized Volatility", "Annualized Excess Return",
                                   "Sharpe Ratio", "Downside Deviation", "Sortino Ratio"]


def test_invalid_ratio_inputs_return_a_message():
    from app.ui.console import risk_adjusted_ratios

    data = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    ratios, error = risk_adjusted_ratios(data, -150.0, 252)
    assert ratios is None and "risk_free_rate" in error


def test_value_at_risk_takes_percent_input_and_flags_small_samples():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, value_at_risk, var_display

    three = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    summary, error = value_at_risk(three, 99.0, 20)
    assert error is None and summary["sufficient_data"].all()
    assert (summary["confidence_level"] == 0.99).all()
    shown = var_display(summary)
    assert list(shown["Confidence"]) == ["99%"] * 3
    assert all(v.endswith("%") for v in shown["Historical VaR"])

    one = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    small, _ = value_at_risk(one, 95.0, 20)
    assert not small["sufficient_data"].any()
    assert list(var_display(small)["Historical VaR"]) == ["n/a"]


def test_invalid_var_inputs_return_a_message():
    from app.ui.console import value_at_risk

    data = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    summary, error = value_at_risk(data, 100.0, 20)
    assert summary is None and "confidence_level" in error


def test_conditional_value_at_risk_uses_var_inputs_and_flags_small_samples():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, conditional_value_at_risk, cvar_display

    three = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    summary, error = conditional_value_at_risk(three, 95.0, 20)
    assert error is None and summary["sufficient_data"].all()
    assert (summary["historical_cvar"] >= summary["historical_var"]).all()
    shown = cvar_display(summary)
    assert list(shown.columns) == ["Symbol", "Observations", "Tail observations", "Historical VaR",
                                   "Historical CVaR", "Parametric VaR", "Parametric CVaR"]
    assert list(shown["Tail observations"]) == ["1.2"] * 3

    one = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    small, _ = conditional_value_at_risk(one, 95.0, 20)
    assert list(cvar_display(small)["Historical CVaR"]) == ["n/a"]


def test_invalid_cvar_inputs_return_a_message():
    from app.ui.console import conditional_value_at_risk

    data = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    summary, error = conditional_value_at_risk(data, 95.0, 1)
    assert summary is None and "min_observations" in error


def test_liquidity_tables_label_actual_and_estimated_sources():
    from app.ui.console import (MULTI_SYMBOL_SAMPLE_CSV, liquidity_display, liquidity_summary,
                                uses_estimated_traded_value)

    three = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    one = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data

    actual, estimated = liquidity_summary(three), liquidity_summary(one)
    assert not uses_estimated_traded_value(actual) and uses_estimated_traded_value(estimated)
    assert set(liquidity_display(actual)["Traded Value Source"]) == {"Actual turnover"}
    assert list(liquidity_display(estimated)["Traded Value Source"]) == ["Estimated (close × volume)"]
    assert list(liquidity_display(actual)["Zero Volume Days"]) == [0, 5, 0]


def test_position_liquidity_takes_percent_and_reports_errors():
    from app.ui.console import position_display, position_liquidity

    data = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    positions, error = position_liquidity(data, 20_000_000, 10.0)
    assert error is None and positions["participation_rate"].iloc[0] == pytest.approx(0.10)
    assert list(position_display(positions).columns)[-1] == "Estimated Liquidation Days"

    none, error = position_liquidity(data, 20_000_000, 150.0)
    assert none is None and "participation_rate" in error
