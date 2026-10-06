"""Tests for the Exit Safety Engine. Expected values are worked out by hand from known ADTVs and shocks."""

import math
import statistics
from dataclasses import fields
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from app.exit_engine.exit_safety import (
    AT_RISK,
    CAUTION,
    EXIT_HORIZON_LABEL,
    INSUFFICIENT_DATA,
    LIQUIDITY_DATA_COMPLETE,
    LIQUIDITY_DATA_INCOMPLETE,
    LIQUIDITY_OK,
    MARKET_IMPACT_NOT_MODELLED,
    NO_LIQUIDITY_DATA,
    REGIME_DATA_UNAVAILABLE,
    SAFE,
    ExitAssessmentInputs,
    ExitSafetyPolicy,
    ExitSafetyResult,
    assess_exit_safety,
    classify_exit_safety,
)
from app.regime.regime_detector import detect_market_regime
from app.stress_testing.stress_engine import StressScenario

RA = [0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.015, -0.01, 0.005, -0.03, 0.012, -0.008,
      0.025, -0.015, 0.004, -0.022, 0.018, -0.006, 0.009, -0.04, 0.011, -0.003, 0.007, 0.02, -0.012]
RB = [-0.01, 0.02, -0.005, 0.01, 0.0, 0.03, -0.02, 0.015, -0.01, 0.02, -0.004, 0.006,
      -0.01, 0.012, 0.003, 0.01, -0.015, 0.008, -0.002, 0.025, -0.006, 0.004, -0.009, -0.01, 0.007]
RC = [0.005, 0.01, -0.02, 0.004, -0.012, 0.008, 0.002, -0.006, 0.015, -0.01, 0.003, 0.0,
      -0.004, 0.009, -0.007, 0.012, 0.001, -0.003, 0.006, -0.015, 0.004, 0.002, -0.001, 0.005, 0.003]
V, T5, T20 = 20_000_000.0, 5_000_000.0, 20_000_000.0


def stock(symbol, returns, turnover=None, volume=1_000.0, status=None, start=100.0):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    frame = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=len(closes)),
                          "symbol": symbol, "close": closes, "volume": volume})
    if turnover is not None:
        frame["turnover"] = turnover
    if status is not None:
        frame["validation_status"] = status
    return frame


def combine(*frames):
    return pd.concat(frames, ignore_index=True)


# ABC: ADTV Rs. 100M; LMN: ADTV Rs. 1M; XYZ: ADTV Rs. 50M (constant reported turnover)
ABC = stock("ABC", RA, turnover=100_000_000.0)
LMN = stock("LMN", RB, turnover=1_000_000.0)
XYZ = stock("XYZ", RC, turnover=50_000_000.0)
DATA = combine(ABC, LMN, XYZ)
W3 = {"ABC": 0.40, "XYZ": 0.35, "LMN": 0.25}
ADTV = {"ABC": 100e6, "LMN": 1e6, "XYZ": 50e6}
LOOSE = ExitSafetyPolicy(max_safe_exit_days=100, max_caution_exit_days=200,
                         stress_caution_threshold=0.15, stress_risk_threshold=0.25)


def returns_of(frame):
    c = frame["close"].tolist()
    return [c[i + 1] / c[i] - 1 for i in range(len(c) - 1)]


def es_exact(returns, confidence=0.95):
    alpha = Fraction(1) - Fraction(str(confidence))
    losses = sorted((-Fraction(float(r)) for r in returns), reverse=True)
    m = alpha * len(losses)
    k = math.floor(m)
    total = sum(losses[:k], Fraction(0)) + ((m - k) * losses[k] if m > k else 0)
    return float(total / m)


def portfolio_series(weights, frames):
    rets = {s: returns_of(f) for s, f in frames.items()}
    n = len(next(iter(rets.values())))
    return [sum(weights[s] * rets[s][t] for s in weights) for t in range(n)]


def table(result):
    return result.holdings_table().set_index("symbol")


# Critical: full and partial exit

def test_full_exit():
    result = assess_exit_safety(DATA, W3, V, T20)
    rows = table(result)
    assert result.remaining_portfolio_value == 0
    assert rows["planned_exit_value"].sum() == pytest.approx(20_000_000.0, abs=1e-6)
    assert rows["planned_exit_value"].to_dict() == pytest.approx(
        {"ABC": 8_000_000, "LMN": 5_000_000, "XYZ": 7_000_000})
    assert rows["holding_value"].to_dict() == rows["planned_exit_value"].to_dict()


def test_partial_exit():
    result = assess_exit_safety(DATA, W3, V, T5)
    rows = table(result)
    assert result.target_exit_value == 5_000_000 and result.portfolio_value == 20_000_000
    assert result.remaining_portfolio_value == 15_000_000
    assert rows["planned_exit_value"].sum() == pytest.approx(5_000_000.0, abs=1e-6)
    assert rows.loc["ABC", "planned_exit_value"] == pytest.approx(2_000_000)
    assert rows.loc["ABC", "holding_value"] == pytest.approx(8_000_000)


# Critical: the illiquid holding dominates; days are not summed

def test_illiquid_holding_dominates_the_exit_horizon():
    data = combine(ABC, LMN)
    result = assess_exit_safety(data, {"ABC": 0.5, "LMN": 0.5}, V, 10_000_000,
                                participation_rate=0.10)
    rows = table(result)
    abc_days = 5_000_000 / (100_000_000 * 0.10)            # 0.5
    lmn_days = 5_000_000 / (1_000_000 * 0.10)              # 50
    assert rows.loc["ABC", "estimated_exit_days"] == pytest.approx(abc_days)
    assert rows.loc["LMN", "estimated_exit_days"] == pytest.approx(lmn_days)
    assert rows.loc["LMN", "exit_value_to_adtv"] == pytest.approx(5.0)
    assert result.estimated_exit_days == pytest.approx(50.0)
    assert result.estimated_exit_days != pytest.approx(abc_days + lmn_days)      # not a sum
    assert result.estimated_exit_days != pytest.approx((abc_days + lmn_days) / 2)  # nor an average
    assert result.overall_status == AT_RISK
    assert any("50.0 trading days, above the configured caution threshold of 20 days" in t
               for t in result.reason_text)
    assert result.exit_horizon_label == EXIT_HORIZON_LABEL


# Critical: each status by changing only policy thresholds or the scenario

PARTIAL = combine(stock("ABC", RA, turnover=100e6), stock("XYZ", RC, turnover=50e6),
                  stock("LMN", RB, volume=np.nan))                 # LMN: no liquidity data
WP = {"ABC": 0.30, "XYZ": 0.30, "LMN": 0.40}


def test_status_policy_safe_caution_at_risk_insufficient():
    base = dict(coverage_minimum=0.6, insufficient_coverage_ratio=0.5,
                stress_caution_threshold=0.15, stress_risk_threshold=0.25)
    # coverage is exactly 60% (LMN has no data); CVaR is low; horizon is short
    safe = assess_exit_safety(PARTIAL, WP, V, T5, policy=ExitSafetyPolicy(**base))
    assert safe.coverage_ratio == pytest.approx(0.60) and safe.overall_status == SAFE
    caution = assess_exit_safety(PARTIAL, WP, V, T5, stress_scenario="Market -20%",
                                 policy=ExitSafetyPolicy(**base))           # 0.20 >= 0.15
    assert caution.overall_status == CAUTION
    at_risk = assess_exit_safety(PARTIAL, WP, V, T5, stress_scenario="Market -30%",
                                 policy=ExitSafetyPolicy(**base))           # 0.30 >= 0.25
    assert at_risk.overall_status == AT_RISK
    insufficient = assess_exit_safety(PARTIAL, WP, V, T5, policy=ExitSafetyPolicy(
        **{**base, "coverage_minimum": 0.7, "insufficient_coverage_ratio": 0.65}))
    assert insufficient.overall_status == INSUFFICIENT_DATA
    assert any(r.level == INSUFFICIENT_DATA and "60.0% of the target" in r.message
               and "LMN" in r.message for r in insufficient.reasons)


# Inputs and validation

@pytest.mark.parametrize("target, message", [
    (20_000_001, "exceeds portfolio_value"), (0, "positive"), (-5, "positive"),
    (float("nan"), "finite"), (None, "finite")])
def test_invalid_target(target, message):
    with pytest.raises(ValueError, match=message):
        assess_exit_safety(DATA, W3, V, target)


def test_missing_portfolio_value_and_weights():
    with pytest.raises(ValueError, match="portfolio_value"):
        assess_exit_safety(DATA, W3, None, T5)
    with pytest.raises(ValueError, match="weights are required"):
        assess_exit_safety(DATA, None, V, T5)
    with pytest.raises(ValueError, match="no holdings"):
        assess_exit_safety(DATA, {}, V, T5)


@pytest.mark.parametrize("weights, message", [
    ({"ABC": 0.6, "XYZ": 0.3}, "sum to 1"), ({"ABC": 1.2, "XYZ": -0.2}, "negative"),
    ([("ABC", 0.5), ("ABC", 0.5)], "Duplicate"), ({"ABC": 0.5, "QQQ": 0.5}, "not found")])
def test_invalid_weights_duplicates_and_missing_symbols(weights, message):
    with pytest.raises(ValueError, match=message):
        assess_exit_safety(DATA, weights, V, T5)


def test_one_stock_and_multiple_stocks():
    one = assess_exit_safety(DATA, {"XYZ": 1.0}, V, T5)
    assert len(one.holdings) == 1
    assert one.estimated_exit_days == pytest.approx(5e6 / (50e6 * 0.10))
    many = assess_exit_safety(DATA, W3, V, T5)
    assert [h.symbol for h in many.holdings] == ["ABC", "LMN", "XYZ"]


# Liquidity, coverage, participation

def test_full_liquidity_data_and_coverage_one():
    result = assess_exit_safety(DATA, W3, V, T5)
    rows = table(result)
    for s, w in W3.items():
        exit_value = T5 * w
        assert rows.loc[s, "average_daily_traded_value"] == pytest.approx(ADTV[s])
        assert rows.loc[s, "traded_value_source"] == "ACTUAL_TURNOVER"
        assert rows.loc[s, "exit_value_to_adtv"] == pytest.approx(exit_value / ADTV[s])
        assert rows.loc[s, "estimated_exit_days"] == pytest.approx(exit_value / (ADTV[s] * 0.10))
        assert rows.loc[s, "liquidity_status"] == LIQUIDITY_OK
    assert result.coverage_ratio == 1.0 and result.uncovered_exit_value == 0
    assert result.liquidity_data_status == LIQUIDITY_DATA_COMPLETE
    assert result.estimated_exit_days == pytest.approx(max(T5 * w / (ADTV[s] * 0.10)
                                                           for s, w in W3.items()))


def test_partial_liquidity_data_and_coverage_below_one():
    result = assess_exit_safety(PARTIAL, WP, V, T5)
    rows = table(result)
    assert rows.loc["LMN", "liquidity_status"] == NO_LIQUIDITY_DATA
    assert math.isnan(rows.loc["LMN", "estimated_exit_days"])
    assert result.covered_exit_value == pytest.approx(0.6 * T5)
    assert result.uncovered_exit_value == pytest.approx(0.4 * T5)
    assert result.coverage_ratio == pytest.approx(0.6)
    assert result.liquidity_data_status == LIQUIDITY_DATA_INCOMPLETE
    assert result.liquidity_data_missing == ("LMN",)
    # horizon from the covered holdings only, and the gap is stated (not ignored)
    assert result.estimated_exit_days == pytest.approx(max(1.5e6 / 10e6, 1.5e6 / 5e6))
    assert result.overall_status == CAUTION
    assert any("Liquidity coverage is 60.0%" in t and "LMN" in t for t in result.reason_text)


def test_no_liquidity_data_is_insufficient():
    result = assess_exit_safety(DATA.drop(columns=["volume", "turnover"]), W3, V, T5)
    assert result.overall_status == INSUFFICIENT_DATA
    assert result.coverage_ratio == 0 and math.isnan(result.estimated_exit_days)
    assert set(result.liquidity_data_missing) == {"ABC", "LMN", "XYZ"}


def test_zero_adtv_is_undefined_not_infinite():
    data = combine(ABC, XYZ, stock("LMN", RB, volume=0.0))
    result = assess_exit_safety(data, W3, V, T5)
    row = table(result).loc["LMN"]
    assert row["average_daily_traded_value"] == 0
    assert math.isnan(row["estimated_exit_days"]) and not math.isinf(row["exit_value_to_adtv"])
    assert row["liquidity_status"] == NO_LIQUIDITY_DATA and "ADTV is zero" in row["liquidity_note"]


def test_very_long_exit_horizon():
    data = combine(ABC, stock("LMN", RB, turnover=10_000.0))
    result = assess_exit_safety(data, {"ABC": 0.5, "LMN": 0.5}, V, T20)
    assert result.estimated_exit_days == pytest.approx(10_000_000 / (10_000 * 0.10))   # 10,000
    assert result.overall_status == AT_RISK


def test_participation_rate_changes_the_horizon():
    ten = assess_exit_safety(DATA, W3, V, T5, participation_rate=0.10)
    twenty_five = assess_exit_safety(DATA, W3, V, T5, participation_rate=0.25)
    assert twenty_five.estimated_exit_days == pytest.approx(ten.estimated_exit_days * 0.10 / 0.25)
    assert table(twenty_five)["participation_rate"].eq(0.25).all()
    with pytest.raises(ValueError, match="participation_rate"):
        assess_exit_safety(DATA, W3, V, T5, participation_rate=0)


def test_liquidity_stress_half_capacity():
    base = assess_exit_safety(DATA, W3, V, T5)
    stressed = assess_exit_safety(DATA, W3, V, T5, liquidity_stress_multiplier=0.5)
    assert stressed.stressed_exit_days == pytest.approx(2 * base.estimated_exit_days)
    assert stressed.estimated_exit_days == base.estimated_exit_days        # base still reported
    assert stressed.assessed_exit_days == stressed.stressed_exit_days      # stressed is assessed
    rows = table(stressed)
    for s, w in W3.items():
        assert rows.loc[s, "stressed_exit_days"] == pytest.approx(T5 * w / (ADTV[s] * 0.5 * 0.10))
    assert stressed.stress_loss == base.stress_loss                        # no price change
    assert math.isnan(base.stressed_exit_days)
    with pytest.raises(ValueError, match="liquidity_multiplier"):
        assess_exit_safety(DATA, W3, V, T5, liquidity_stress_multiplier=1.5)


# Tail risk and stress

def test_var_cvar_references_and_low_cvar():
    result = assess_exit_safety(DATA, W3, V, T5)
    series = portfolio_series(W3, {"ABC": ABC, "LMN": LMN, "XYZ": XYZ})
    assert result.historical_cvar == pytest.approx(es_exact(series), rel=1e-9)
    assert result.historical_cvar < 0.05                                    # low CVaR
    assert result.reference_cvar_loss_amount == pytest.approx(T5 * es_exact(series), rel=1e-9)
    mean, sd = statistics.mean(series), statistics.stdev(series)
    assert result.parametric_var == pytest.approx(-(mean + statistics.NormalDist().inv_cdf(0.05) * sd),
                                                  rel=1e-9)
    assert any(r.factor == "TAIL_RISK" and r.level == "OK" for r in result.reasons)


def test_high_cvar_is_at_risk():
    wild = [0.15, -0.14, 0.12, -0.16] * 6 + [0.1]
    data = combine(stock("ABC", wild, turnover=100e6), stock("XYZ", RC, turnover=50e6))
    result = assess_exit_safety(data, {"ABC": 0.9, "XYZ": 0.1}, V, T5, policy=LOOSE)
    series = portfolio_series({"ABC": 0.9, "XYZ": 0.1}, {"ABC": data[data["symbol"] == "ABC"],
                                                        "XYZ": data[data["symbol"] == "XYZ"]})
    assert result.historical_cvar == pytest.approx(es_exact(series), rel=1e-9) and result.historical_cvar >= 0.10
    assert result.overall_status == AT_RISK
    assert any(r.factor == "TAIL_RISK" and r.level == AT_RISK for r in result.reasons)


@pytest.mark.parametrize("scenario, loss", [("Market -10%", 0.10), ("Market -20%", 0.20),
                                            ("Market -30%", 0.30)])
def test_market_stress_scenarios(scenario, loss):
    result = assess_exit_safety(DATA, W3, V, T5, stress_scenario=scenario)
    assert result.stress_scenario == scenario
    assert result.stress_return == pytest.approx(-loss) and result.stress_loss == pytest.approx(loss)
    assert result.stress_loss_amount == pytest.approx(V * loss)
    assert result.reference_stress_loss_amount == pytest.approx(T5 * loss)


def test_low_and_high_stress_loss_and_volatility_scenario():
    sigma = {s: statistics.stdev(returns_of(f)) for s, f in (("ABC", ABC), ("LMN", LMN), ("XYZ", XYZ))}
    low = assess_exit_safety(DATA, W3, V, T5, stress_scenario="High Volatility 1.5x", policy=LOOSE)
    assert low.stress_loss == pytest.approx(1.5 * sum(W3[s] * sigma[s] for s in W3), rel=1e-9)
    assert low.stress_loss < 0.15 and low.overall_status == SAFE
    high = assess_exit_safety(DATA, W3, V, T5, stress_scenario="Market -30%", policy=LOOSE)
    assert high.overall_status == AT_RISK
    with pytest.raises(ValueError, match="Unknown stress scenario"):
        assess_exit_safety(DATA, W3, V, T5, stress_scenario="Liquidity -50%")
    with pytest.raises(ValueError, match="price scenario"):
        assess_exit_safety(DATA, W3, V, T5, stress_scenario=StressScenario(
            "Liq", "LIQUIDITY", liquidity_multiplier=0.5))


def test_insufficient_risk_metrics():
    short = combine(stock("ABC", RA[:10], turnover=100e6), stock("XYZ", RC[:10], turnover=50e6))
    result = assess_exit_safety(short, {"ABC": 0.5, "XYZ": 0.5}, V, T5,
                                stress_scenario="High Volatility 2.0x")
    assert result.overall_status == INSUFFICIENT_DATA
    assert math.isnan(result.historical_cvar) and result.stress_status == "UNAVAILABLE"
    text = " ".join(result.reason_text)
    assert "CVaR is unavailable: 10 common daily return(s), 20 required" in text
    assert "High Volatility 2.0x: Insufficient return history" in text


# Regime is context only

CALM = [0.004, -0.003]
SEQUENCE = (CALM * 90 + [0.025, -0.024] * 8
            + [-0.04, 0.01, -0.05, -0.03, 0.015, -0.045, -0.02, -0.03]
            + [0.006, 0.002, 0.007, 0.003] * 10)


def index(returns, name="ASPI"):
    closes = [1000.0]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=len(closes)),
                         "index_name": name, "close": closes})


@pytest.mark.parametrize("returns, regime", [
    (CALM * 100, "NORMAL"),
    (CALM * 80 + [0.02, -0.019] * 6, "HIGH_VOLATILITY"),
    (CALM * 80 + [-0.03, -0.04, 0.01, -0.05, -0.03, -0.02], "STRESS"),
    (SEQUENCE[:234], "RECOVERY"),
])
def test_regimes_are_reported_without_changing_the_status(returns, regime):
    idx = index(returns)
    expected = detect_market_regime(idx).current                      # existing module
    assert expected.regime == regime
    result = assess_exit_safety(DATA, W3, V, T5, index_data=idx, index_name="ASPI")
    assert result.market_regime == regime and result.trend_state == expected.trend_state
    assert result.current_drawdown == expected.current_drawdown
    assert result.rolling_volatility == expected.rolling_volatility
    no_index = assess_exit_safety(DATA, W3, V, T5)
    assert result.overall_status == no_index.overall_status          # no hidden regime penalty
    assert any(f"Current market regime is {regime}" in t for t in result.reason_text)


def test_regime_unavailable():
    none = assess_exit_safety(DATA, W3, V, T5)
    assert none.market_regime == REGIME_DATA_UNAVAILABLE and none.trend_state is None
    assert "no market-index data supplied" in none.regime_note
    warm = assess_exit_safety(DATA, W3, V, T5, index_data=index(CALM * 20), index_name="ASPI")
    assert warm.market_regime == REGIME_DATA_UNAVAILABLE and "141 are needed" in warm.regime_note
    wrong = assess_exit_safety(DATA, W3, V, T5, index_data=index(CALM * 100), index_name="S&P SL20")
    assert wrong.market_regime == REGIME_DATA_UNAVAILABLE and "not found" in wrong.regime_note


# Policy

def test_policy_thresholds_change_the_status_without_code_changes():
    data = combine(ABC, LMN)
    weights = {"ABC": 0.5, "LMN": 0.5}           # horizon 2.5M / 0.1M = 25 days at 10%
    strict = assess_exit_safety(data, weights, V, T5)
    assert strict.estimated_exit_days == pytest.approx(25.0) and strict.overall_status == AT_RISK
    relaxed = assess_exit_safety(data, weights, V, T5, policy=ExitSafetyPolicy(
        max_safe_exit_days=10, max_caution_exit_days=30))
    assert relaxed.overall_status == CAUTION
    assert relaxed.policy_parameters["max_caution_exit_days"] == 30


@pytest.mark.parametrize("kwargs, message", [
    ({"max_safe_exit_days": 30}, "cannot exceed"), ({"cvar_caution_threshold": 0}, "positive"),
    ({"stress_risk_threshold": float("nan")}, "finite"), ({"coverage_minimum": 1.5}, "at most 1"),
    ({"insufficient_coverage_ratio": 1.0, "coverage_minimum": 0.9}, "between 0 and coverage_minimum"),
    ({"cvar_measure": "max"}, "cvar_measure"), ({"stress_caution_threshold": 0.3}, "cannot exceed")])
def test_invalid_policy(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ExitSafetyPolicy(**kwargs)


def test_parametric_cvar_measure_can_be_selected():
    result = assess_exit_safety(DATA, W3, V, T5, policy=ExitSafetyPolicy(cvar_measure="parametric"))
    assert result.reference_cvar_loss_amount == pytest.approx(T5 * result.parametric_cvar)
    assert any("Parametric 1-day CVaR" in t for t in result.reason_text)


def test_decision_tree_by_hand():
    p = ExitSafetyPolicy()

    def status(**kw):
        base = dict(coverage_ratio=1.0, exit_days=1.0, cvar=0.01, stress_loss=0.05)
        return classify_exit_safety(ExitAssessmentInputs(**{**base, **kw}), p)[0]

    assert status() == SAFE
    assert status(exit_days=5.0) == SAFE and status(exit_days=5.01) == CAUTION
    assert status(exit_days=20.0) == CAUTION and status(exit_days=20.01) == AT_RISK
    assert status(cvar=0.05) == CAUTION and status(cvar=0.10) == AT_RISK
    assert status(stress_loss=0.10) == CAUTION and status(stress_loss=0.20) == AT_RISK
    assert status(coverage_ratio=0.99) == CAUTION and status(coverage_ratio=0.49) == INSUFFICIENT_DATA
    assert status(cvar=math.nan) == INSUFFICIENT_DATA
    assert status(stress_loss=math.nan) == INSUFFICIENT_DATA
    assert status(exit_days=math.nan) == INSUFFICIENT_DATA
    assert status(exit_days=100, cvar=math.nan) == INSUFFICIENT_DATA    # data gaps come first


# No invented market impact

def test_no_market_impact_value_is_invented():
    result = assess_exit_safety(DATA, W3, V, T5)
    assert result.market_impact == MARKET_IMPACT_NOT_MODELLED == (
        "Not modelled in the current Exit Safety version.")
    impact_fields = [f.name for f in fields(ExitSafetyResult) if "impact" in f.name]
    assert impact_fields == ["market_impact"]
    assert not any("impact" in c for c in result.holdings_table().columns)
    assert any(MARKET_IMPACT_NOT_MODELLED in t for t in result.reason_text)


def test_reasons_point_to_calculated_metrics():
    result = assess_exit_safety(DATA, W3, V, T5)
    text = " ".join(result.reason_text)
    assert f"{result.estimated_exit_days:.1f} trading days" in text
    assert f"{result.historical_cvar:.2%}" in text and "10.0% hypothetical portfolio loss" in text
    assert "Rs. 500,000.00" in text                                        # 5M x 10%
    assert result.overall_status == CAUTION          # Market -10% loss of 10% >= caution 10%


# Data handling

def test_invalid_rows_are_excluded():
    status = ["VALID"] * 26
    status[4] = "INVALID"
    turnover = [100e6] * 26
    turnover[4] = 1.0                                   # would distort ADTV if it were used
    data = combine(stock("ABC", RA, turnover=turnover, status=status),
                   stock("XYZ", RC, turnover=50e6, status=["VALID"] * 26))
    result = assess_exit_safety(data, {"ABC": 0.5, "XYZ": 0.5}, V, T5)
    assert table(result).loc["ABC", "average_daily_traded_value"] == pytest.approx(100e6)


def test_warning_rows_are_usable():
    data = combine(stock("ABC", RA, turnover=100e6, status=["WARNING"] * 26), XYZ)
    result = assess_exit_safety(data, {"ABC": 0.5, "XYZ": 0.5}, V, T5)
    assert table(result).loc["ABC", "liquidity_status"] == LIQUIDITY_OK
    assert not math.isnan(result.historical_cvar)


def test_unsorted_data_and_determinism():
    shuffled = DATA.sample(frac=1, random_state=4).reset_index(drop=True)
    a = assess_exit_safety(shuffled, W3, V, T5, liquidity_stress_multiplier=0.5)
    b = assess_exit_safety(DATA, W3, V, T5, liquidity_stress_multiplier=0.5)
    c = assess_exit_safety(DATA, W3, V, T5, liquidity_stress_multiplier=0.5)
    pd.testing.assert_frame_equal(a.holdings_table(), b.holdings_table())
    assert a.reason_text == b.reason_text == c.reason_text
    assert a.overall_status == b.overall_status == c.overall_status


def test_input_is_not_mutated():
    data = DATA.sample(frac=1, random_state=2).reset_index(drop=True)
    before = data.copy(deep=True)
    weights = {"LMN": 0.25, "ABC": 0.40, "XYZ": 0.35}
    idx = index(CALM * 100)
    idx_before = idx.copy(deep=True)
    assess_exit_safety(data, weights, V, T5, index_data=idx, index_name="ASPI",
                       liquidity_stress_multiplier=0.5)
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(idx, idx_before)
    assert weights == {"LMN": 0.25, "ABC": 0.40, "XYZ": 0.35}


def test_zero_weight_holding_has_nothing_to_exit():
    result = assess_exit_safety(DATA, {"ABC": 0.6, "XYZ": 0.4, "LMN": 0.0}, V, T5)
    row = table(result).loc["LMN"]
    assert row["planned_exit_value"] == 0 and row["liquidity_status"] == "NO_EXIT"
    assert result.coverage_ratio == pytest.approx(1.0)
