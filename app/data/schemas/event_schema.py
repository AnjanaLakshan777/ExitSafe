"""Stored market events.

Events are immutable: an assessment creates a new copy. Only an official source
(company, exchange or regulator) can mark an event as confirmed.
"""

import hashlib
from dataclasses import dataclass, fields
from datetime import datetime
from enum import StrEnum

from app.data.schemas.checks import require_aware, require_enum, require_text, validate_confidence


class EventType(StrEnum):
    CYBERSECURITY_INCIDENT = "CYBERSECURITY_INCIDENT"
    DATA_BREACH = "DATA_BREACH"
    REGULATORY_ACTION = "REGULATORY_ACTION"
    FINANCIAL_RESULT = "FINANCIAL_RESULT"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    FRAUD_OR_GOVERNANCE = "FRAUD_OR_GOVERNANCE"
    LEGAL_EVENT = "LEGAL_EVENT"
    CREDIT_EVENT = "CREDIT_EVENT"
    MACROECONOMIC_EVENT = "MACROECONOMIC_EVENT"
    SECTOR_EVENT = "SECTOR_EVENT"
    LIQUIDITY_EVENT = "LIQUIDITY_EVENT"
    OTHER = "OTHER"


class SourceType(StrEnum):
    """Kind of document an event was taken from."""

    OFFICIAL_DISCLOSURE = "OFFICIAL_DISCLOSURE"  # exchange (e.g. CSE) filings
    REGULATOR = "REGULATOR"                      # central bank, SEC, CERT, etc.
    COMPANY = "COMPANY"                          # company's own press release / site
    NEWS = "NEWS"
    THREAT_INTELLIGENCE = "THREAT_INTELLIGENCE"  # lawful public / licensed feeds only
    SEARCH_RESULT = "SEARCH_RESULT"
    OTHER = "OTHER"


# Only these sources can confirm that an event happened.
OFFICIAL_SOURCE_TYPES = frozenset(
    {SourceType.OFFICIAL_DISCLOSURE, SourceType.REGULATOR, SourceType.COMPANY}
)


class VerificationStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"      # single low-credibility claim (e.g. threat-intel post)
    REPORTED = "REPORTED"          # reported by a credible non-official source
    CORROBORATED = "CORROBORATED"  # reported by several independent sources
    CONFIRMED = "CONFIRMED"        # confirmed by an official source
    DENIED = "DENIED"              # officially denied
    RETRACTED = "RETRACTED"        # withdrawn by the original source


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ImpactDirection(StrEnum):
    """Direction of *risk*, not a price prediction."""

    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"
    MIXED = "MIXED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    event_type: EventType
    title: str
    source_name: str
    source_type: SourceType
    published_time: datetime
    detected_time: datetime
    symbol: str | None = None             # None until mapped (or for market-wide events)
    company_name: str | None = None
    description: str = ""
    source_url: str | None = None
    event_time: datetime | None = None    # when it happened; may be unknown or in the future
    severity: Severity | None = None      # None = not assessed yet
    confidence: float | None = None       # 0-1; None = not assessed yet
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    affected_sector: str | None = None
    potential_market_impact: ImpactDirection | None = None

    def __post_init__(self):
        require_text(self.event_id, "event_id")
        require_enum(self.event_type, EventType, "event_type")
        require_text(self.title, "title")
        require_text(self.source_name, "source_name")
        require_enum(self.source_type, SourceType, "source_type")
        require_enum(self.verification_status, VerificationStatus, "verification_status")
        require_enum(self.severity, Severity, "severity", optional=True)
        require_enum(self.potential_market_impact, ImpactDirection,
                     "potential_market_impact", optional=True)

        require_aware(self.published_time, "published_time")
        require_aware(self.detected_time, "detected_time")
        require_aware(self.event_time, "event_time", optional=True)
        if self.detected_time < self.published_time:
            raise ValueError("detected_time cannot be earlier than published_time")

        if self.confidence is not None:
            validate_confidence(self.confidence)

        if (self.verification_status is VerificationStatus.CONFIRMED
                and self.source_type not in OFFICIAL_SOURCE_TYPES):
            raise ValueError(
                f"An event from a {self.source_type} source cannot be CONFIRMED; "
                "confirmation requires an official disclosure, regulator or company source."
            )

        if self.symbol is not None:
            require_text(self.symbol, "symbol")
            # Same normalization as the market-data loader, so symbols join cleanly.
            object.__setattr__(self, "symbol", self.symbol.strip().upper())

    def to_dict(self):
        """JSON-friendly dict: enums as strings, datetimes as ISO-8601."""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, StrEnum):
                value = value.value
            result[field.name] = value
        return result

    @classmethod
    def from_dict(cls, data):
        """Inverse of ``to_dict``. Unknown enum values raise ``ValueError``."""
        values = dict(data)
        for name, enum_type in _ENUM_FIELDS.items():
            if values.get(name) is not None:
                values[name] = enum_type(values[name])
        for name in _DATETIME_FIELDS:
            if isinstance(values.get(name), str):
                values[name] = datetime.fromisoformat(values[name])
        return cls(**values)


_ENUM_FIELDS = {
    "event_type": EventType,
    "source_type": SourceType,
    "severity": Severity,
    "potential_market_impact": ImpactDirection,
    "verification_status": VerificationStatus,
}
_DATETIME_FIELDS = ("published_time", "detected_time", "event_time")


def make_event_id(source_name, published_time, source_url=None, title=""):
    """Deterministic id, so re-collecting the same item gives the same event_id."""
    key = "|".join([source_name.strip().lower(), published_time.isoformat(),
                    (source_url or title).strip().lower()])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
