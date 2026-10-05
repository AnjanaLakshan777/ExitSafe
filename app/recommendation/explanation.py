"""Plain-language explanation of a risk signal.

Evidence is always grouped by its basis, so an investor can see at a glance
what is confirmed, what is only reported, and what is ExitSafe's own inference
or assumption.
"""

from app.intelligence.models import EvidenceBasis

SECTION_TITLES = {
    EvidenceBasis.CONFIRMED_FACT: "Confirmed facts",
    EvidenceBasis.REPORTED_CLAIM: "Reported claims (not officially confirmed)",
    EvidenceBasis.MODEL_INFERENCE: "Model inferences",
    EvidenceBasis.SCENARIO_ASSUMPTION: "Scenario assumptions",
}

DISCLAIMER = ("This is a risk signal with scenario ranges, not a prediction "
              "of the stock's future price.")


def explain_signal(signal):
    lines = [
        f"{signal.symbol}: {signal.risk_level} risk "
        f"(direction: {signal.direction}, confidence: {signal.confidence:.0%})"
    ]
    if signal.portfolio_weight is not None:
        lines.append(f"Portfolio exposure: {signal.portfolio_weight:.1%} of portfolio value")

    for basis, title in SECTION_TITLES.items():
        items = [_format_evidence(e) for e in signal.evidence if e.basis is basis]
        if basis is EvidenceBasis.SCENARIO_ASSUMPTION:
            items += [_format_scenario(s) for s in signal.scenarios]
        if items:
            lines += ["", f"{title}:"] + [f"  - {item}" for item in items]

    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def _format_evidence(evidence):
    if evidence.source_url:
        return f"{evidence.statement} [{evidence.source_url}]"
    return evidence.statement


def _format_scenario(scenario):
    return (f"{scenario.name}: price change between {scenario.price_change_low:+.0%} "
            f"and {scenario.price_change_high:+.0%}, assuming {scenario.assumption}")
