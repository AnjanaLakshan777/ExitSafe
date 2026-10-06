"""The risk signal shown to investors. It never contains a predicted price."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from app.data.schemas.checks import validate_confidence
from app.data.schemas.event_schema import ImpactDirection
from app.intelligence.impact.event_impact import ImpactScenario
from app.intelligence.models import Evidence, EvidenceBasis

# Evidence that comes from a source (as opposed to ExitSafe's own reasoning).
SOURCED_BASES = frozenset({EvidenceBasis.CONFIRMED_FACT, EvidenceBasis.REPORTED_CLAIM})


class RiskLevel(StrEnum):
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


@dataclass(frozen=True)
class RiskSignal:
    symbol: str
    generated_time: datetime
    risk_level: RiskLevel
    direction: ImpactDirection
    confidence: float
    evidence: tuple[Evidence, ...]
    event_ids: tuple[str, ...] = ()
    scenarios: tuple[ImpactScenario, ...] = ()
    portfolio_weight: float | None = None  # share of the portfolio in this symbol, 0-1

    def __post_init__(self):
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.generated_time.tzinfo is None:
            raise ValueError("generated_time must be timezone-aware")
        if not isinstance(self.risk_level, RiskLevel):
            raise ValueError(f"risk_level must be a RiskLevel, got {self.risk_level!r}")
        if not isinstance(self.direction, ImpactDirection):
            raise ValueError(f"direction must be an ImpactDirection, got {self.direction!r}")
        validate_confidence(self.confidence)
        if not any(item.basis in SOURCED_BASES for item in self.evidence):
            raise ValueError(
                "A risk signal needs at least one confirmed fact or reported claim as "
                "evidence; model inferences and scenario assumptions alone are not enough"
            )
        if self.portfolio_weight is not None and not 0.0 <= self.portfolio_weight <= 1.0:
            raise ValueError("portfolio_weight must be between 0 and 1")
