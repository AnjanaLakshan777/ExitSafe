"""Keyword-based event classification. Each result records the words that triggered it."""

import re
from dataclasses import dataclass

from app.data.schemas.event_schema import EventType
from app.intelligence.classification.event_types import EVENT_KEYWORDS

# Keywords match at the start of a word, so "hack" matches "hacked"/"hackers"
# but not "shackle", and "irregularit" matches "irregularity"/"irregularities".
_PATTERNS = {
    event_type: [(term, re.compile(rf"\b{re.escape(term)}", re.IGNORECASE)) for term in terms]
    for event_type, terms in EVENT_KEYWORDS.items()
}


@dataclass(frozen=True)
class Classification:
    event_type: EventType
    matched_terms: tuple[str, ...] = ()


def classify_event(title, description=""):
    """Classify text into an ``EventType``; ``OTHER`` if no keyword matches."""
    text = f"{title} {description}"
    for event_type, patterns in _PATTERNS.items():
        matched = tuple(term for term, pattern in patterns if pattern.search(text))
        if matched:
            return Classification(event_type, matched)
    return Classification(EventType.OTHER)
