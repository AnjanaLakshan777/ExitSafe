"""Intelligence-specific types: raw collected items and evidence.

The event itself (``MarketEvent`` and its enums) is stored data and lives in
``app.data.schemas.event_schema``.

Every statement used as evidence carries an ``EvidenceBasis`` so confirmed
facts, reported claims, model inferences and scenario assumptions are never
mixed up.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from app.data.schemas.checks import require_aware, require_enum, require_text
from app.data.schemas.event_schema import SourceType


class EvidenceBasis(StrEnum):
    CONFIRMED_FACT = "CONFIRMED_FACT"
    REPORTED_CLAIM = "REPORTED_CLAIM"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    SCENARIO_ASSUMPTION = "SCENARIO_ASSUMPTION"


@dataclass(frozen=True)
class Evidence:
    statement: str
    basis: EvidenceBasis
    source_url: str | None = None

    def __post_init__(self):
        require_text(self.statement, "statement")
        require_enum(self.basis, EvidenceBasis, "basis")


@dataclass(frozen=True)
class RawItem:
    """A document as returned by a collector, before parsing/classification."""

    source_name: str
    source_type: SourceType
    title: str
    published_time: datetime
    fetched_time: datetime
    source_url: str | None = None
    body: str = ""

    def __post_init__(self):
        require_text(self.source_name, "source_name")
        require_enum(self.source_type, SourceType, "source_type")
        require_text(self.title, "title")
        require_aware(self.published_time, "published_time")
        require_aware(self.fetched_time, "fetched_time")


# Contract every collector implements: return items published at or after `since`.
Collector = Callable[[datetime], list[RawItem]]
