"""The rule-based recommendation layer for Start Investing and I Already Invested.

It must only read results from the existing engines, so most checks compare it with
direct calls to those engines.
"""

import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

from app.exit_engine.exit_safety import ExitReason
from app.ui import decision as d
from app.ui.console import (BACKTEST_SAMPLE_CSV, MULTI_SYMBOL_SAMPLE_CSV, exit_safety,
                            format_percent, optimize, parse_holdings, portfolio_risk, run_import)

HOLDINGS = "ABC, 40\nXYZ, 35\nLMN, 25"


def sample(path):
    return run_import(path.name, path.read_bytes()).import_result.data


@pytest.fixture(scope="module")
def three():
    return sample(MULTI_SYMBOL_SAMPLE_CSV)


@pytest.fixture(scope="module")
def four():
    return sample(BACKTEST_SAMPLE_CSV)


@pytest.fixture(scope="module")
def plan(four):
    return d.plan_new_investment(four, ["ALPHA", "BRAVO", "CHARLIE", "DELTA"], 5_000_000,
                                 "MEDIUM", "BALANCED")


def engine_exit(data, holdings, value, target):
    pairs, _ = parse_holdings(holdings) if isinstance(holdings, str) else (holdings, None)
    result, error = exit_safety(data, pairs, value, target, d.PARTICIPATION, d.STRESS_SCENARIO,
                                False, d.POLICY, d.CONFIDENCE, d.MIN_OBSERVATIONS,
                                d.PERIODS_PER_YEAR)
    assert error is None
    return result


# --- Start Investing ---

def test_recommended_allocation_is_the_optimizer_output(four, plan):
    s = plan.settings
    direct, error = optimize(four, ["ALPHA", "BRAVO", "CHARLIE", "DELTA"], s["min_percent"],
                             s["max_percent"], s["risk_aversion"], s["cvar_weight"],
                             s["return_weight"], d.CONFIDENCE, d.MIN_OBSERVATIONS,
                             d.PERIODS_PER_YEAR, 5_000_000, s["liquidity_enabled"],
                             s["max_position_to_adtv"])
    assert error is None
    assert plan.optimization.optimized_weights == pytest.approx(direct.optimized_weights)
    assert dict(plan.holdings) == pytest.approx(direct.optimized_weights, abs=1e-6)
    assert sum(w for _, w in plan.holdings) == pytest.approx(1.0)


def test_plan_risk_checks_are_the_existing_engines(four, plan):
    risk, _ = portfolio_risk(four, plan.holdings, 5_000_000, d.CONFIDENCE, d.MIN_OBSERVATIONS,
                             d.PERIODS_PER_YEAR, d.PARTICIPATION)
    assert plan.risk.historical_cvar == risk.historical_cvar
    assert plan.risk.annualized_volatility == risk.annualized_volatility
    health = engine_exit(four, plan.holdings, 5_000_000, 5_000_000)
    assert plan.health.overall_status == health.overall_status
    assert plan.health.assessed_exit_days == health.assessed_exit_days


def test_start_investing_recommendation_and_its_evidence(plan):
    rec = plan.recommendation
    assert rec.action == d.INVEST_GRADUALLY               # DELTA needs more than 5 days to trade
    assert rec.status == d.STATUS_LABELS[plan.health.overall_status]
    assert 2 <= len(rec.reasons) <= 5
    text = " ".join(rec.reasons)
    assert f"{plan.health.assessed_exit_days:,.1f} trading days" in text
    assert format_percent(plan.risk.historical_cvar) in text
    assert format_percent(plan.optimization.expected_annual_return) in text


def test_risk_preference_changes_the_optimizer_settings(four):
    careful = d.plan_new_investment(four, ["ALPHA", "BRAVO", "CHARLIE", "DELTA"], 5_000_000,
                                    "LONG", "CONSERVATIVE")
    bold = d.plan_new_investment(four, ["ALPHA", "BRAVO", "CHARLIE", "DELTA"], 5_000_000,
                                 "LONG", "GROWTH")
    assert careful.settings["max_percent"] < bold.settings["max_percent"]
    assert careful.optimization.optimized_weights != bold.optimization.optimized_weights


def test_short_horizon_turns_on_the_existing_liquidity_limit():
    settings, _ = d.optimizer_settings("BALANCED", "SHORT", ["A", "B", "C"])
    assert settings["liquidity_enabled"] and settings["max_position_to_adtv"] == 2.0
    settings, _ = d.optimizer_settings("BALANCED", "LONG", ["A", "B", "C"])
    assert not settings["liquidity_enabled"]


def test_limit_per_company_is_raised_only_when_the_money_cannot_be_invested():
    settings, notes = d.optimizer_settings("CONSERVATIVE", "LONG", ["A", "B"])
    assert settings["max_percent"] == 50.0 and "raised from 30% to 50%" in notes[0]
    settings, notes = d.optimizer_settings("CONSERVATIVE", "LONG", ["A", "B", "C", "D"])
    assert settings["max_percent"] == 30.0 and notes == []


def test_losing_candidates_are_sent_back_for_review(three):
    plan = d.plan_new_investment(three, ["ABC", "LMN", "XYZ"], 5_000_000, "MEDIUM", "BALANCED")
    assert plan.optimization.expected_annual_return < 0
    assert plan.recommendation.action == d.REVIEW_CANDIDATES
    assert "lost value on average" in " ".join(plan.recommendation.reasons)


def test_an_impossible_plan_explains_the_optimizer_error(three):
    plan = d.plan_new_investment(three, ["ABC", "LMN", "XYZ"], 5_000_000, "SHORT", "CONSERVATIVE")
    assert plan.error and "liquidity" in plan.error.lower()
    assert plan.recommendation.action == d.REVIEW_CANDIDATES
    assert plan.recommendation.reasons[0].startswith("The optimizer couldn't build a portfolio")


@pytest.mark.parametrize("capital, candidates, message", [
    (0, ["ABC"], "more than zero"), (-5, ["ABC"], "more than zero"),
    (float("nan"), ["ABC"], "more than zero"), (1_000_000, [], "at least one company"),
])
def test_invalid_start_investing_input(three, capital, candidates, message):
    plan = d.plan_new_investment(three, candidates, capital, "MEDIUM", "BALANCED")
    assert message in plan.error and plan.recommendation is None


def test_holdings_text_totals_exactly_100():
    text = d.holdings_text({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    pairs, error = parse_holdings(text)
    assert error is None and sum(w for _, w in pairs) == pytest.approx(1.0)


# --- I Already Invested ---

def test_portfolio_status_is_the_exit_engines_status_for_a_full_exit(three):
    review = d.review_portfolio(three, HOLDINGS, 20_000_000)
    health = engine_exit(three, HOLDINGS, 20_000_000, 20_000_000)
    assert review.recommendation.status == d.STATUS_LABELS[health.overall_status] == "AT RISK"
    assert review.health.assessed_exit_days == health.assessed_exit_days
    assert review.exit is None


def test_withdrawal_reuses_the_exit_safety_result(three):
    review = d.review_portfolio(three, HOLDINGS, 20_000_000, 5_000_000)
    direct = engine_exit(three, HOLDINGS, 20_000_000, 5_000_000)
    assert review.exit.overall_status == direct.overall_status == "CAUTION"
    assert review.exit.assessed_exit_days == direct.assessed_exit_days
    assert review.exit.coverage_ratio == direct.coverage_ratio
    assert [r.message for r in review.exit.reasons] == [r.message for r in direct.reasons]


def test_recommendation_follows_the_evidence(three):
    small = d.review_portfolio(three, HOLDINGS, 20_000_000, 1_000_000)
    large = d.review_portfolio(three, HOLDINGS, 20_000_000, 5_000_000)
    assert small.exit.assessed_exit_days < d.POLICY["max_safe_exit_days"]
    assert small.recommendation.action == d.REVIEW_PORTFOLIO      # selling everything is slow
    assert large.recommendation.action == d.CONSIDER_STAGED_EXIT  # the withdrawal itself is slow
    assert "15.7 trading days" in large.recommendation.reasons[0]


def test_reasons_quote_the_calculated_numbers(three):
    review = d.review_portfolio(three, HOLDINGS, 20_000_000)
    text = " ".join(review.recommendation.reasons)
    assert f"{review.health.assessed_exit_days:,.1f} trading days" in text
    assert review.risk.most_illiquid_symbol in text
    assert format_percent(review.risk.historical_cvar) in text
    assert f"Rs. {review.risk.historical_cvar * 20_000_000:,.0f}" in text


@pytest.mark.parametrize("holdings, value, target, message", [
    ("ABC, 50\nXYZ, 30", 1_000_000, None, "must total 100%"),
    ("ABC fifty", 1_000_000, None, "is not a number"),
    (HOLDINGS, 0, None, "more than zero"),
    (HOLDINGS, 1_000_000, 2_000_000, "larger than the current portfolio value"),
    ("ABC, 50\nNOPE, 50", 1_000_000, None, "NOPE"),
])
def test_invalid_portfolio_input(three, holdings, value, target, message):
    review = d.review_portfolio(three, holdings, value, target)
    assert message in review.error and review.recommendation is None


# --- the rules, one at a time ---

def reason(factor, level):
    return ExitReason(factor, level, f"{factor} {level}")


def health(*reasons, status="SAFE", days=3.0):
    return SimpleNamespace(
        reasons=list(reasons), overall_status=status, assessed_exit_days=days,
        participation_rate=0.10, stress_loss=0.10, stress_loss_amount=100_000.0,
        stress_scenario="Market -10%", market_regime="NORMAL", regime_index="ASPI",
        portfolio_value=1_000_000.0, target_exit_value=1_000_000.0, coverage_ratio=1.0,
        holdings=[SimpleNamespace(symbol="AAA", liquidity_status="OK", estimated_exit_days=days)])


def risk(max_weight=0.4):
    return SimpleNamespace(
        sufficient_tail_data=True, historical_cvar=0.02, confidence_level=0.95,
        observations=100, min_observations=20, hhi=0.34, maximum_weight=max_weight,
        most_illiquid_symbol="AAA",
        holdings=pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"],
                               "weight": [max_weight, 0.35, 0.65 - max_weight]}))


CALM = (reason("LIQUIDITY", "OK"), reason("TAIL_RISK", "OK"), reason("STRESS", "CAUTION"))


@pytest.mark.parametrize("evidence, exit_result, max_weight, action", [
    (health(*CALM, status="CAUTION"), None, 0.40, d.HOLD),
    (health(*CALM, status="CAUTION"), None, 0.60, d.REVIEW_PORTFOLIO),
    (health(reason("LIQUIDITY", "AT_RISK"), status="AT_RISK", days=40), None, 0.4,
     d.REVIEW_PORTFOLIO),
    (health(reason("TAIL_RISK", "CAUTION"), status="CAUTION"), None, 0.4, d.REDUCE_EXPOSURE),
    (health(reason("STRESS", "AT_RISK"), status="AT_RISK"), None, 0.4, d.REDUCE_EXPOSURE),
    (health(*CALM, status="CAUTION"),
     health(reason("LIQUIDITY", "CAUTION"), status="CAUTION", days=9), 0.4,
     d.CONSIDER_STAGED_EXIT),
    (health(reason("COVERAGE", "INSUFFICIENT_DATA"), status="INSUFFICIENT_DATA"), None, 0.4,
     d.NO_DATA),
])
def test_existing_investor_rules(evidence, exit_result, max_weight, action):
    rec = d.recommend_existing(risk(max_weight), evidence, exit_result)
    assert rec.action == action
    assert 1 <= len(rec.reasons) <= 5


def test_a_market_wide_stress_caution_alone_does_not_change_the_action():
    rec = d.recommend_existing(risk(), health(*CALM, status="CAUTION"))
    assert rec.action == d.HOLD and rec.status == "CAUTION"


@pytest.mark.parametrize("levels, expected_return, action", [
    ((reason("LIQUIDITY", "OK"), reason("TAIL_RISK", "OK")), 0.08, d.PROCEED),
    ((reason("LIQUIDITY", "CAUTION"),), 0.08, d.INVEST_GRADUALLY),
    ((reason("TAIL_RISK", "AT_RISK"),), 0.08, d.REVIEW_CANDIDATES),
    ((reason("LIQUIDITY", "OK"),), -0.02, d.REVIEW_CANDIDATES),
])
def test_new_investor_rules(levels, expected_return, action):
    metrics = SimpleNamespace(annualized_volatility=0.1, historical_cvar=0.02)
    optimization = SimpleNamespace(expected_annual_return=expected_return, metrics=metrics,
                                   equal_weight_metrics=metrics,
                                   start_date=pd.Timestamp("2025-01-01"),
                                   end_date=pd.Timestamp("2026-01-01"))
    rec = d.recommend_new(optimization, risk(), health(*levels), 1_000_000)
    assert rec.action == action


# --- news and AI search never decide ---

def test_news_and_gemini_are_not_inputs_to_the_recommendation():
    for function in (d.plan_new_investment, d.review_portfolio, d.recommend_new,
                     d.recommend_existing):
        names = " ".join(inspect.signature(function).parameters).lower()
        assert not any(word in names for word in ("threat", "news", "gemini", "event"))
    source = inspect.getsource(d)
    assert "app.intelligence" not in source and "gemini" not in source.split('"""', 2)[2]


def test_the_same_evidence_always_gives_the_same_recommendation(three):
    first = d.review_portfolio(three, HOLDINGS, 20_000_000, 5_000_000).recommendation
    again = d.review_portfolio(three, HOLDINGS, 20_000_000, 5_000_000).recommendation
    assert (first.action, first.reasons, first.status) == (again.action, again.reasons,
                                                           again.status)
