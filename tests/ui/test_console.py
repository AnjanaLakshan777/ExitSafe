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


def test_default_holdings_text_totals_exactly_100():
    from app.ui.console import default_holdings_text, parse_holdings

    text = default_holdings_text(["XYZ", "ABC", "LMN"])
    assert text.splitlines() == ["ABC, 33.33", "LMN, 33.33", "XYZ, 33.34"]
    pairs, error = parse_holdings(text)
    assert error is None and sum(w for _, w in pairs) == pytest.approx(1.0)
    assert default_holdings_text(["ONE"]) == "ONE, 100.00"


@pytest.mark.parametrize("text, expected", [
    ("ABC, 60\nLMN, 40", [("ABC", 0.6), ("LMN", 0.4)]),
    ("ABC\t60%\n\n  LMN 40 % ", [("ABC", 0.6), ("LMN", 0.4)]),
])
def test_parse_holdings_accepts_percent_lines(text, expected):
    from app.ui.console import parse_holdings

    pairs, error = parse_holdings(text)
    assert error is None
    assert [s for s, _ in pairs] == [s for s, _ in expected]
    assert [w for _, w in pairs] == pytest.approx([w for _, w in expected])


@pytest.mark.parametrize("text, message", [
    ("ABC, 60\nLMN, 30", "total 90%; they must total 100%"),
    ("ABC, 0.6\nLMN, 0.4", "total 1%"),                  # fractions are not rescaled
    ("ABC, sixty", "not a number"),
    ("ABC, 50, 50", "one holding per line"),
    ("ABC, nan", "finite"),
    ("   ", "at least one holding"),
])
def test_parse_holdings_reports_errors_without_normalizing(text, message):
    from app.ui.console import parse_holdings

    pairs, error = parse_holdings(text)
    assert pairs is None and message in error


def test_portfolio_risk_tables_for_three_stock_sample():
    from app.ui.console import (MULTI_SYMBOL_SAMPLE_CSV, parse_holdings,
                                portfolio_holdings_display, portfolio_risk,
                                portfolio_summary_display)

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    holdings, _ = parse_holdings("ABC, 40\nLMN, 25\nXYZ, 35")
    result, error = portfolio_risk(data, holdings, 20_000_000, 95.0, 20, 252, 10.0)
    assert error is None
    assert result.confidence_level == pytest.approx(0.95) and result.participation_rate == pytest.approx(0.10)

    summary = dict(zip(*portfolio_summary_display(result).T.values))
    assert summary["Annualized Volatility"] == format_percent(result.annualized_volatility)
    assert summary["Maximum Weight"] == "40.00%"
    assert summary["HHI (concentration)"] == "0.3450"
    assert summary["Most Illiquid Holding"] == "LMN"
    assert not any(word in " ".join(summary.values()).upper() for word in ("BUY", "SELL", "SAFE"))

    table = portfolio_holdings_display(result)
    assert list(table.columns) == ["Symbol", "Weight", "Position Value (Rs.)", "ADTV (Rs.)",
                                   "ADTV Source", "Position / ADTV", "Estimated Liquidation Days"]
    assert list(table["Position Value (Rs.)"]) == ["8,000,000.00", "5,000,000.00", "7,000,000.00"]
    assert set(table["ADTV Source"]) == {"Actual turnover"}


def test_portfolio_risk_reports_analytics_errors():
    from app.ui.console import portfolio_risk

    data = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    result, error = portfolio_risk(data, [("NOPE", 1.0)], 1_000_000, 95.0, 20, 252, 10.0)
    assert result is None and "not found" in error
    result, error = portfolio_risk(data, [(SAMPLE_SYMBOL, 0.5), (SAMPLE_SYMBOL, 0.5)],
                                   1_000_000, 95.0, 20, 252, 10.0)
    assert result is None and "Duplicate" in error
    result, error = portfolio_risk(data, [(SAMPLE_SYMBOL, 1.0)], 1_000_000, 95.0, 20, 252, 10.0)
    assert error is None and not result.sufficient_tail_data and result.observations == 4


def test_optimize_takes_percent_inputs_and_builds_tables():
    from app.ui.console import (MULTI_SYMBOL_SAMPLE_CSV, optimization_allocation_display,
                                optimization_comparison_display, optimization_metrics_display,
                                optimize)

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    result, error = optimize(data, ["ABC", "LMN", "XYZ"], 0.0, 40.0, 1.0, 1.0, 1.0, 95.0, 20, 252,
                             20_000_000, False)
    assert error is None
    assert result.max_weight_limit == pytest.approx(0.40) and result.confidence_level == pytest.approx(0.95)
    assert result.max_weight <= 0.40 + 1e-6

    allocation = optimization_allocation_display(result)
    assert list(allocation.columns) == ["Symbol", "Weight %", "Position (Rs.)"]
    metrics = dict(zip(*optimization_metrics_display(result).T.values))
    assert metrics["Solver Status"] == "optimal (CLARABEL)"
    assert metrics["Historical CVaR (1-day)"] == format_percent(result.historical_cvar)
    comparison = optimization_comparison_display(result)
    assert list(comparison.columns) == ["Metric", "Equal Weight", "ExitSafe Optimized"]
    assert list(comparison["Metric"]) == ["Annualized Return", "Annualized Volatility",
                                          "Historical VaR (1-day)", "Historical CVaR (1-day)", "HHI"]
    assert comparison.iloc[-1]["Equal Weight"] == "0.3333"
    text = " ".join(map(str, comparison.to_numpy().ravel())).upper()
    assert not any(word in text for word in ("BETTER", "BUY", "SELL", "SAFE"))


def test_optimize_with_liquidity_shows_labelled_liquidity_columns():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, optimization_allocation_display, optimize

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    result, error = optimize(data, ["ABC", "LMN", "XYZ"], 0.0, 60.0, 1.0, 1.0, 1.0, 95.0, 20, 252,
                             20_000_000, True, 3.0)
    assert error is None
    table = optimization_allocation_display(result).set_index("Symbol")
    assert list(table.columns)[-4:] == ["ADTV (Rs.)", "ADTV Source", "Position / ADTV",
                                        "Liquidity Constraint Status"]
    assert table.loc["LMN", "Liquidity Constraint Status"] == "At limit"
    assert set(table["ADTV Source"]) == {"Actual turnover"}

    one = run_import(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL).import_result.data
    estimated, _ = optimize(one, [SAMPLE_SYMBOL], 0.0, 100.0, 1.0, 1.0, 1.0, 95.0, 4, 252,
                            1_000, True, 10.0)
    assert list(optimization_allocation_display(estimated)["ADTV Source"]) == [
        "Estimated (close × volume)"]


@pytest.mark.parametrize("args, message", [
    ((["ABC", "LMN", "XYZ"], 0.0, 30.0), "cannot be fully invested"),
    ((["ABC", "LMN", "XYZ"], 40.0, 100.0), "minimum weight"),
    (([], 0.0, 40.0), "Select at least one stock"),
])
def test_optimize_reports_errors(args, message):
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, optimize

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    result, error = optimize(data, *args, 1.0, 1.0, 1.0, 95.0, 20, 252, 20_000_000, False)
    assert result is None and message in error


def test_optimize_reports_infeasible_liquidity_and_insufficient_data():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, optimize

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    result, error = optimize(data, ["ABC", "LMN", "XYZ"], 0.0, 40.0, 1.0, 1.0, 1.0, 95.0, 20, 252,
                             20_000_000, True, 1.0)
    assert result is None and "Infeasible liquidity constraint" in error
    result, error = optimize(data, ["ABC", "LMN", "XYZ"], 0.0, 40.0, 1.0, 1.0, 1.0, 95.0, 30, 252,
                             20_000_000, False)
    assert result is None and "Only 24 common daily return" in error


def test_market_regime_on_the_synthetic_sample_index():
    from app.ui.console import (INDEX_SAMPLE_CSV, load_index_data, market_regime,
                                regime_current_display, regime_history_display)

    imported, error = load_index_data(INDEX_SAMPLE_CSV.name, INDEX_SAMPLE_CSV.read_bytes())
    assert error is None and imported.index_names == ["ASPI", "S&P SL20"]
    result, error = market_regime(imported.data, "ASPI")
    assert error is None and result.required_observations == 141
    current = dict(zip(*regime_current_display(result).T.values))
    assert current["Regime"] in {"Normal", "High volatility", "Stress", "Recovery"}
    assert current["Rolling Volatility (20-day, daily)"] == format_percent(
        result.current.rolling_volatility)
    history = regime_history_display(result)
    assert list(history.columns) == ["Date", "Regime", "Rolling Volatility (20-day, daily)",
                                     "Current Drawdown", "Trend State"]
    assert list(history["Regime"][:140]) == ["N/A (warm-up)"] * 140
    assert {"Normal", "High volatility", "Stress", "Recovery"} <= set(history["Regime"])


def test_market_regime_reports_index_and_file_errors():
    from app.ui.console import INDEX_SAMPLE_CSV, load_index_data, market_regime

    imported, _ = load_index_data(INDEX_SAMPLE_CSV.name, INDEX_SAMPLE_CSV.read_bytes())
    result, error = market_regime(imported.data, "CSE ALL")
    assert result is None and "not found" in error
    none, error = load_index_data("aspi.csv", b"Date,Price\n2026-01-05,100\n")
    assert none is None and "no Index column" in error
    named, error = load_index_data("aspi.csv", b"Date,Price\n2026-01-05,100\n", "ASPI")
    assert error is None and named.index_names == ["ASPI"]
    short, error = market_regime(named.data, "ASPI")
    assert error is None and short.current.regime == "N/A"
    broken, error = load_index_data("aspi.csv", b"Date,Index\n2026-01-05,ASPI\n")
    assert broken is None and "missing column" in error


def test_chosen_index_name_only_applies_to_files_without_an_index_column():
    from app.ui.console import load_index_data

    named, error = load_index_data("x.csv", b"Date,Index,Close\n2026-01-05,ASPI,100\n",
                                   "S&P SL20")
    assert error is None and named.index_names == ["ASPI"]
    unnamed, error = load_index_data("x.csv", b"Date,Close\n2026-01-05,100\n", "S&P SL20")
    assert error is None and unnamed.index_names == ["S&P SL20"]


def test_parse_sector_mapping():
    from app.ui.console import parse_sector_mapping

    mapping, error = parse_sector_mapping("abc, Banking Finance & Insurance\nXYZ\tManufacturing\n\n")
    assert error is None
    assert mapping == {"ABC": "Banking Finance & Insurance", "XYZ": "Manufacturing"}
    assert parse_sector_mapping("  ") == (None, None)
    assert "SYMBOL, Sector" in parse_sector_mapping("ABC")[1]
    assert "two sectors" in parse_sector_mapping("ABC, Banking\nABC, Hotels")[1]


def test_custom_stress_scenario_uses_only_the_components_that_are_set():
    from app.ui.console import custom_stress_scenario

    assert custom_stress_scenario() == (None, None)
    market, error = custom_stress_scenario(market_percent=-15)
    assert error is None and market.scenario_type == "MARKET" and market.market_shock == -0.15
    combined, _ = custom_stress_scenario(-10, "Banking", -20, 0.0, 0.5)
    assert combined.scenario_type == "COMBINED"
    assert combined.components == ("market", "sector", "liquidity")
    assert custom_stress_scenario(0, "", -5)[1] == "Enter a sector name for the custom sector shock."


def test_stress_report_tables_for_the_sample_portfolio():
    from app.ui.console import (DEFAULT_SCENARIOS, MULTI_SYMBOL_SAMPLE_CSV,
                                custom_stress_scenario, parse_sector_mapping, stress_impact_display,
                                stress_liquidity_display, stress_report, stress_summary_display)

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    mapping, _ = parse_sector_mapping("ABC, Banking\nLMN, Banking\nXYZ, Manufacturing")
    custom, _ = custom_stress_scenario(-10, "banking", -20)
    report, error = stress_report(data, [("ABC", 0.40), ("XYZ", 0.35), ("LMN", 0.25)],
                                  20_000_000, list(DEFAULT_SCENARIOS) + [custom], mapping,
                                  10.0, "zero", 95.0, 20)
    assert error is None
    summary = stress_summary_display(report).set_index("Scenario")
    assert summary.loc["Market -10%", "Portfolio Return"] == "-10.00%"
    assert summary.loc["Market -10%", "Loss Amount (Rs.)"] == "2,000,000.00"
    assert summary.loc["Market -30%", "Stressed Portfolio Value (Rs.)"] == "14,000,000.00"
    assert summary.loc["Custom scenario", "Portfolio Loss"] == "23.00%"
    impact = stress_impact_display(report.results[-1]).set_index("Symbol")
    assert list(impact.columns) == ["Sector", "Weight", "Base Return", "Stressed Return",
                                    "Contribution"]
    assert impact.loc["XYZ", "Stressed Return"] == "-10.00%"
    liquidity = stress_liquidity_display(report.results[5])
    assert list(liquidity.columns) == ["Symbol", "Base ADTV (Rs.)", "Stressed ADTV (Rs.)",
                                       "ADTV Source", "Base Liquidation Days",
                                       "Stressed Liquidation Days", "Status"]
    words = " ".join(map(str, summary.to_numpy().ravel())).upper()
    assert not any(w in words.split() for w in ("BUY", "SELL", "SAFE"))


def test_stress_report_reports_errors():
    from app.ui.console import DEFAULT_SCENARIOS, MULTI_SYMBOL_SAMPLE_CSV, stress_report

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    report, error = stress_report(data, [("ABC", 0.5), ("QQQ", 0.5)], 1e6, DEFAULT_SCENARIOS,
                                  None, 10.0, "zero", 95.0, 20)
    assert report is None and "not found" in error


BACKTEST_SETTINGS = {"min_percent": 0.0, "max_percent": 40.0, "risk_aversion": 1.0,
                     "cvar_weight": 1.0, "return_weight": 1.0, "liquidity_enabled": False,
                     "max_position_to_adtv": None}


def test_run_backtest_on_the_synthetic_fixture_with_the_sample_index():
    from app.ui.console import (BACKTEST_SAMPLE_CSV, INDEX_SAMPLE_CSV,
                                backtest_comparison_display, backtest_final_weights_display,
                                backtest_log_display, load_index_data, run_backtest)

    data = run_import(BACKTEST_SAMPLE_CSV.name, BACKTEST_SAMPLE_CSV.read_bytes()).import_result.data
    index, _ = load_index_data(INDEX_SAMPLE_CSV.name, INDEX_SAMPLE_CSV.read_bytes())
    result, error = run_backtest(data, ["ALPHA", "BRAVO", "CHARLIE", "DELTA"], 1_000_000, 60, 20,
                                 20, 0.0, 95.0, 20, 252, BACKTEST_SETTINGS, index.data, "ASPI")
    assert error is None and result.config.max_weight == pytest.approx(0.40)
    table = backtest_comparison_display(result)
    assert list(table.columns) == ["Metric", "ExitSafe", "EqualWeight", "MeanVariance",
                                   "MarketIndex"]
    rows = table.set_index("Metric")
    assert rows.loc["Observations"].tolist() == ["260"] * 4
    assert rows.loc["Rebalances", "MarketIndex"] == "n/a"
    log = backtest_log_display(result)
    assert len(log) == 13 * 3 and set(log["Status"]) == {"REBALANCED"}
    assert log["Weights"].iloc[0].startswith("ALPHA 25.0%")
    weights = backtest_final_weights_display(result)
    assert list(weights.columns) == ["Symbol", "ExitSafe", "EqualWeight", "MeanVariance"]


def test_run_backtest_reports_errors():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, run_backtest

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    result, error = run_backtest(data, ["ABC", "LMN", "XYZ"], 1_000_000, 60, 20, 20, 0.0, 95.0,
                                 20, 252, BACKTEST_SETTINGS)
    assert result is None and "Not enough history" in error and "the data has 25" in error
    result, error = run_backtest(data, [], 1_000_000, 10, 5, 5, 0.0, 95.0, 20, 252,
                                 BACKTEST_SETTINGS)
    assert result is None and "at least one" in error
    result, error = run_backtest(data, ["ABC"], 1_000_000, 10, 10, 5, 0.0, 95.0, 20, 252,
                                 {**BACKTEST_SETTINGS, "max_percent": 100.0})
    assert result is None and "overlap" in error


EXIT_POLICY = {"max_safe_exit_days": 5.0, "max_caution_exit_days": 20.0,
               "cvar_caution_percent": 5.0, "cvar_risk_percent": 10.0,
               "stress_caution_percent": 10.0, "stress_risk_percent": 20.0,
               "coverage_minimum_percent": 100.0, "insufficient_coverage_percent": 50.0,
               "cvar_measure": "historical"}
EXIT_HOLDINGS = [("ABC", 0.40), ("XYZ", 0.35), ("LMN", 0.25)]


def test_exit_safety_tables_for_the_sample_portfolio():
    from app.ui.console import (MULTI_SYMBOL_SAMPLE_CSV, exit_plan_display, exit_safety,
                                exit_summary_display)

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    result, error = exit_safety(data, EXIT_HOLDINGS, 20_000_000, 5_000_000, 10.0, "Market -10%",
                                False, EXIT_POLICY, 95.0, 20, 252)
    assert error is None and result.overall_status == "CAUTION"
    summary = dict(zip(*exit_summary_display(result).T.values))
    assert summary["Target Exit Amount (Rs.)"] == "5,000,000.00"
    assert summary["Remaining Portfolio Value (Rs.)"] == "15,000,000.00"
    assert summary["Estimated Exit Horizon"] == "15.7 trading days"
    assert summary["Market Impact"] == "Not modelled in the current Exit Safety version."
    assert summary["Current Market Regime"].startswith("Unavailable")
    plan = exit_plan_display(result).set_index("Symbol")
    assert list(plan.columns) == ["Weight", "Holding Value (Rs.)", "Planned Exit (Rs.)",
                                  "ADTV (Rs.)", "ADTV Source", "Exit / ADTV",
                                  "Estimated Exit Days", "Liquidity Status"]
    assert plan.loc["LMN", "Planned Exit (Rs.)"] == "1,250,000.00"
    stressed, _ = exit_safety(data, EXIT_HOLDINGS, 20_000_000, 5_000_000, 10.0, "Market -10%",
                              True, EXIT_POLICY, 95.0, 20, 252)
    assert "Stressed Exit Days" in exit_plan_display(stressed).columns
    assert stressed.overall_status == "AT_RISK"                    # 31.5 days > 20


def test_exit_safety_reports_errors():
    from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, exit_safety

    data = run_import(MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data
    result, error = exit_safety(data, EXIT_HOLDINGS, 20_000_000, 25_000_000, 10.0, "Market -10%",
                                False, EXIT_POLICY, 95.0, 20, 252)
    assert result is None and "exceeds portfolio_value" in error
    result, error = exit_safety(data, EXIT_HOLDINGS, 20_000_000, 5_000_000, 10.0, "Market -10%",
                                False, {**EXIT_POLICY, "max_safe_exit_days": 30.0}, 95.0, 20, 252)
    assert result is None and "cannot exceed" in error
