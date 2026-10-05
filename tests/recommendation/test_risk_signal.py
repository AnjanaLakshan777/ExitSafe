from datetime import datetime, timezone

import pytest

from app.data.schemas.event_schema import ImpactDirection
from app.intelligence.impact.event_impact import ImpactScenario
from app.intelligence.models import Evidence, EvidenceBasis
from app.recommendation.decision_engine import portfolio_exposure
from app.recommendation.explanation import DISCLAIMER, explain_signal
from app.recommendation.risk_signal import RiskLevel, RiskSignal

NOW = datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc)

FACT = Evidence("ABC Bank disclosed an IT security incident to the exchange",
                EvidenceBasis.CONFIRMED_FACT, "https://exchange.example/abc/123")
CLAIM = Evidence("A news outlet reports customer records may be affected",
                 EvidenceBasis.REPORTED_CLAIM)
INFERENCE = Evidence("Value traded is 3x its 20-day average", EvidenceBasis.MODEL_INFERENCE)
SCENARIO = ImpactScenario("Severe", -0.20, -0.08, "regulators impose penalties")


def make_signal(**overrides):
    values = dict(
        symbol="ABC",
        generated_time=NOW,
        risk_level=RiskLevel.HIGH,
        direction=ImpactDirection.NEGATIVE,
        confidence=0.7,
        evidence=(FACT, CLAIM, INFERENCE),
        scenarios=(SCENARIO,),
        portfolio_weight=0.125,
    )
    values.update(overrides)
    return RiskSignal(**values)


def test_valid_signal():
    assert make_signal().risk_level is RiskLevel.HIGH


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_signal_confidence_must_be_in_range(confidence):
    with pytest.raises(ValueError):
        make_signal(confidence=confidence)


def test_signal_requires_sourced_evidence():
    with pytest.raises(ValueError, match="at least one confirmed fact or reported claim"):
        make_signal(evidence=(INFERENCE,))
    with pytest.raises(ValueError):
        make_signal(evidence=())


def test_portfolio_weight_must_be_a_share():
    with pytest.raises(ValueError):
        make_signal(portfolio_weight=1.2)


def test_scenario_range_must_be_ordered():
    with pytest.raises(ValueError):
        ImpactScenario("Bad", -0.05, -0.20, "nothing")
    with pytest.raises(ValueError):
        ImpactScenario("Impossible", -1.5, -0.2, "price below zero")


def test_explanation_separates_facts_claims_inferences_and_assumptions():
    text = explain_signal(make_signal())

    sections = ["Confirmed facts:", "Reported claims (not officially confirmed):",
                "Model inferences:", "Scenario assumptions:"]
    positions = [text.index(section) for section in sections]
    assert positions == sorted(positions)

    assert FACT.statement in text.split("Reported claims")[0]
    assert "https://exchange.example/abc/123" in text
    assert "between -20% and -8%, assuming regulators impose penalties" in text
    assert "12.5% of portfolio value" in text
    assert text.endswith(DISCLAIMER)


def test_explanation_omits_empty_sections():
    text = explain_signal(make_signal(evidence=(CLAIM,), scenarios=(), portfolio_weight=None))
    assert "Confirmed facts" not in text
    assert "Scenario assumptions" not in text
    assert "Portfolio exposure" not in text


def test_portfolio_exposure():
    holdings = {"ABC": 25_000.0, "XYZ": 75_000.0}
    assert portfolio_exposure("abc", holdings) == pytest.approx(0.25)
    assert portfolio_exposure("LMN", holdings) == 0.0
    assert portfolio_exposure("ABC", {}) == 0.0
