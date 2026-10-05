"""Confidence that an event is real, based on source credibility.

Confidence here means "how sure are we the event happened as described", not
"how sure are we about the price impact".
"""

from app.data.schemas.event_schema import OFFICIAL_SOURCE_TYPES, SourceType

SOURCE_CREDIBILITY = {
    SourceType.OFFICIAL_DISCLOSURE: 0.95,
    SourceType.REGULATOR: 0.95,
    SourceType.COMPANY: 0.90,
    SourceType.NEWS: 0.60,
    SourceType.THREAT_INTELLIGENCE: 0.30,
    SourceType.SEARCH_RESULT: 0.20,
    SourceType.OTHER: 0.10,
}

CORROBORATION_BONUS = 0.10
# Without an official source, confidence never reaches near-certainty, no
# matter how many unofficial reports repeat the claim.
UNOFFICIAL_CAP = 0.80


def source_credibility(source_type):
    return SOURCE_CREDIBILITY[SourceType(source_type)]


def assess_confidence(source_types):
    """Confidence (0-1) from the source types of independent reports of one event.

    Starts from the most credible source and adds a bonus for each additional
    independent source, capped at ``UNOFFICIAL_CAP`` unless an official source
    is among them.
    """
    source_types = [SourceType(s) for s in source_types]
    if not source_types:
        raise ValueError("At least one source is required to assess confidence")

    best = max(source_credibility(s) for s in source_types)
    score = best + CORROBORATION_BONUS * (len(source_types) - 1)
    cap = 1.0 if any(s in OFFICIAL_SOURCE_TYPES for s in source_types) else UNOFFICIAL_CAP
    return round(min(score, cap), 4)
