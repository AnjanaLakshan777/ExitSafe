"""Decide whether a headline signals a threat to world markets, and how serious it is.

Keyword-based, like ``event_classifier``: every assessment lists the terms that
triggered it, so an alert can always be explained. A term matches as a whole
word plus common endings ("plunge" matches "plunges", "plunged", "plunging").
"""

import re
from dataclasses import dataclass

from app.data.schemas.event_schema import Severity

THREAT_TERMS = {
    Severity.CRITICAL: (
        "market crash", "stock market crash", "stocks crash", "financial crisis",
        "banking crisis", "bank collapse", "bank run", "sovereign default", "debt default",
        "circuit breaker", "trading halted", "war declared", "declares war", "nuclear attack",
        "nuclear strike", "martial law", "coup", "systemic risk",
    ),
    Severity.HIGH: (
        "recession", "sell-off", "selloff", "plunge", "plummet", "tumble", "market rout", "slump",
        "bear market", "war", "invasion", "invade", "airstrike", "missile", "military strike",
        "sanction", "debt crisis", "default", "downgrade", "bankruptcy", "insolvency",
        "contagion", "pandemic", "currency crisis", "devaluation", "emergency rate",
        "oil shock", "oil price surge", "oil spike", "credit crunch", "liquidity crisis",
        "capital controls", "trading halt", "stagflation", "hyperinflation", "terror attack",
    ),
    Severity.MEDIUM: (
        "volatility", "inflation", "tariff", "trade war", "geopolitical", "tension",
        "tariff war", "price war", "currency war", "bidding war",
        "slowdown", "layoff", "job cuts", "rate hike", "federal reserve", "opec",
        "market correction", "bond yield", "unrest", "protest", "supply shock",
        "trade deficit", "shutdown",
    ),
}

SEVERITY_ORDER = (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
# This many distinct threat terms in one headline raise its severity one level.
ESCALATE_AT_TERMS = 3
# Terms that appear only in the description cap the story at this level.
BODY_ONLY_CAP = Severity.MEDIUM
# Headline words saying markets are coping lower the severity one level.
CALMING_TERMS = ("record high", "all-time high", "shrug off", "shrugs off", "shrugged off",
                 "rebound", "rally", "rallies", "recover", "ease", "cool")


def severity_rank(severity):
    return SEVERITY_ORDER.index(Severity(severity))


def _pattern(term):
    stem = re.escape(term[:-1]) + r"(?:e|es|ed|ing)" if term.endswith("e") else (
        re.escape(term) + r"(?:s|es|ed|ing)?")
    return re.compile(rf"\b{stem}\b", re.IGNORECASE)


_PATTERNS = [(severity, term, _pattern(term))
             for severity, terms in THREAT_TERMS.items() for term in terms]
_CALMING = [_pattern(term) for term in CALMING_TERMS]


@dataclass(frozen=True)
class ThreatAssessment:
    severity: Severity | None          # None: no threat terms found
    matched_terms: tuple[str, ...] = ()

    @property
    def is_threat(self):
        return self.severity is not None


def _matches(text):
    """Matched ``(severity, term)``s; a longer phrase ("trade war") hides a term inside it ("war")."""
    found = [(severity, term) for severity, term, pattern in _PATTERNS if pattern.search(text)]
    terms = {term for _, term in found}
    return [(s, t) for s, t in found
            if not any(t != other and re.search(rf"\b{re.escape(t)}", other) for other in terms)]


def assess_threat(title, description=""):
    """Severity of a story from its threat terms.

    The headline decides: its most serious term sets the level, raised one step
    when it has several terms and lowered one step by calming words ("record
    high", "rebound"). Terms found only in the description are side mentions,
    so on their own they never rate above MEDIUM.
    """
    in_title = _matches(title)
    in_body = [m for m in _matches(f"{title} {description}") if m not in in_title]
    if not in_title and not in_body:
        return ThreatAssessment(None)

    rank = max((severity_rank(s) for s, _ in in_title), default=severity_rank(Severity.LOW))
    if len(in_title) >= ESCALATE_AT_TERMS:
        rank += 1
    if in_title and any(p.search(title) for p in _CALMING):
        rank -= 1
    if in_body:
        rank = max(rank, min(max(severity_rank(s) for s, _ in in_body),
                             severity_rank(BODY_ONLY_CAP)))
    rank = min(max(rank, 0), len(SEVERITY_ORDER) - 1)
    return ThreatAssessment(SEVERITY_ORDER[rank],
                            tuple(sorted({t for _, t in in_title + in_body})))
