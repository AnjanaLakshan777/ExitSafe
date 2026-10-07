"""Current portfolio value for I Already Invested: holdings × latest approved price.

No sample amount may stand in for the user's own value, and a holding without a price is
never given one.
"""

import ast
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ui import decision as d
from app.ui.console import MULTI_SYMBOL_SAMPLE_CSV, exit_safety, run_import
from app.ui.journeys import positions_from_investments

UI = Path(__file__).resolve().parents[2] / "app" / "ui"
CSV_HEADER = "Date,Symbol,Open,High,Low,Close,Volume,Source\n"


@pytest.fixture(scope="module")
def three():
    return run_import(MULTI_SYMBOL_SAMPLE_CSV.name,
                      MULTI_SYMBOL_SAMPLE_CSV.read_bytes()).import_result.data


def latest(data, symbol):
    rows = data[(data["symbol"] == symbol) & (data["validation_status"] != "INVALID")]
    return float(rows.sort_values("date")["close"].iloc[-1])


def test_no_sample_amount_is_a_default_in_the_dashboard():
    """No number input in the UI starts from a fixed money amount (the old 20M / 5M / 1M)."""
    for name in ("journeys.py", "streamlit_app.py"):
        tree = ast.parse((UI / name).read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "number_input":
                label = call.args[0].value if call.args and isinstance(call.args[0],
                                                                       ast.Constant) else ""
                if "Rs." not in label:
                    continue
                value = next((k.value for k in call.keywords if k.arg == "value"), None)
                assert value is None or not isinstance(value, ast.Constant) or \
                    value.value is None, f"{name}: '{label}' has a fixed default"


def test_current_value_is_the_sum_of_quantity_times_latest_price(three):
    valuation = d.value_holdings(three, [("ABC", 1000), ("XYZ", 2000), ("LMN", 500)])
    expected = {s: q * latest(three, s) for s, q in (("ABC", 1000), ("XYZ", 2000),
                                                     ("LMN", 500))}
    assert {r.symbol: r.value for r in valuation.rows} == pytest.approx(expected)
    assert valuation.total == pytest.approx(sum(expected.values()))
    assert valuation.complete and valuation.total != 20_000_000
    weights = dict(d.parse_holdings(valuation.holdings_text)[0])
    assert weights["ABC"] == pytest.approx(expected["ABC"] / valuation.total, abs=1e-6)


def test_current_value_follows_the_price(three):
    before = d.value_holdings(three, [("ABC", 1000)])
    newer = three.copy()
    last = newer[newer["symbol"] == "ABC"]["date"].idxmax()
    newer.loc[last, "close"] = newer.loc[last, "close"] * 1.10
    after = d.value_holdings(newer, [("ABC", 1000)])
    assert after.total == pytest.approx(before.total * 1.10)


def test_values_entered_directly_are_used_as_given(three):
    valuation = d.value_holdings(three, [("ABC", 8_000_000), ("XYZ", 7_000_000)], d.VALUES)
    assert valuation.total == 15_000_000
    assert [r.value for r in valuation.rows] == [8_000_000, 7_000_000]


def test_a_holding_without_a_price_is_flagged_not_priced(three):
    valuation = d.value_holdings(three, [("ABC", 1000), ("NOPRICE.N0000", 500)])
    missing = next(r for r in valuation.rows if r.symbol == "NOPRICE.N0000")
    assert missing.price is None and missing.value is None
    assert valuation.missing == ["NOPRICE.N0000"] and not valuation.complete
    assert valuation.total == pytest.approx(1000 * latest(three, "ABC"))
    assert "NOPRICE" not in valuation.holdings_text


def test_nothing_priced_gives_no_value(three):
    valuation = d.value_holdings(three, [("NOPE", 10)])
    assert valuation.total == 0 and valuation.holdings_text == ""


def test_invalid_rows_are_not_used_as_the_latest_price(tmp_path):
    rows = "".join(f"2026-09-{day:02d},AAA,10,11,9,10,100,\n" for day in range(1, 26))
    bad = "2026-09-29,AAA,10,9,11,999,100,\n"                    # high below low: INVALID
    content = (CSV_HEADER + rows + bad).encode()
    data = run_import("a.csv", content).import_result.data
    assert d.latest_prices(data)["AAA"][0] == 10.0


def test_secondary_ai_prices_are_not_used_unless_included():
    rows = "".join(f"2026-09-{day:02d},AAA,10,11,9,10,100,cse_trade_summary_current\n"
                   for day in range(1, 26))
    content = (CSV_HEADER + rows + "2026-09-29,AAA,50,51,49,50,100,gemini_web_search\n").encode()
    default = run_import("a.csv", content).import_result.data
    included = run_import("a.csv", content, allow_ai_sourced=True).import_result.data
    assert d.latest_prices(default)["AAA"][0] == 10.0
    price, _, source = d.latest_prices(included)["AAA"]
    assert price == 50.0 and source == "gemini_web_search"


def test_target_exit_is_compared_with_the_current_value(three):
    valuation = d.value_holdings(three, [("ABC", 1000), ("XYZ", 2000), ("LMN", 500)])
    value = valuation.total
    too_much = d.review_portfolio(three, valuation.holdings_text, value, value + 1)
    assert "larger than the current portfolio value" in too_much.error

    review = d.review_portfolio(three, valuation.holdings_text, value, value / 2)
    assert review.exit.portfolio_value == pytest.approx(value)
    assert review.exit.target_exit_value == pytest.approx(value / 2)


def test_exit_safety_is_unchanged_for_the_calculated_value(three):
    valuation = d.value_holdings(three, [("ABC", 1000), ("XYZ", 2000), ("LMN", 500)])
    review = d.review_portfolio(three, valuation.holdings_text, valuation.total,
                                valuation.total / 4)
    direct, error = exit_safety(three, d.parse_holdings(valuation.holdings_text)[0],
                                valuation.total, valuation.total / 4, d.PARTICIPATION,
                                d.STRESS_SCENARIO, False, d.POLICY, d.CONFIDENCE,
                                d.MIN_OBSERVATIONS, d.PERIODS_PER_YEAR)
    assert error is None
    assert review.exit.overall_status == direct.overall_status
    assert review.exit.assessed_exit_days == direct.assessed_exit_days
    assert [r.message for r in review.exit.reasons] == [r.message for r in direct.reasons]


@pytest.mark.parametrize("text, message", [
    ("", "at least one holding"), ("ABC", "SYMBOL, number"), ("ABC, ten", "not a number"),
    ("ABC, 0", "more than zero"), ("ABC, -5", "more than zero"),
])
def test_invalid_holdings_lines(text, message):
    positions, error = d.parse_positions(text)
    assert positions is None and message in error


def test_holdings_lines_accept_separators_and_repeats():
    assert d.parse_positions("ABC, 1,000\nxyz 250.5\nabc, 500")[0] == [("ABC", 1500.0),
                                                                     ("xyz", 250.5)]


def test_saved_investments_fill_shares_and_original_investment():
    saved = [SimpleNamespace(symbol="abc", quantity=Decimal("1000"), purchase_price=Decimal("10")),
             SimpleNamespace(symbol="ABC", quantity=Decimal("500.5"), purchase_price=Decimal("12")),
             SimpleNamespace(symbol="LMN", quantity=Decimal("200"), purchase_price=Decimal("50"))]
    text, original = positions_from_investments(saved)
    assert text == "ABC, 1500.5\nLMN, 200"
    assert original == pytest.approx(1000 * 10 + 500.5 * 12 + 200 * 50)
