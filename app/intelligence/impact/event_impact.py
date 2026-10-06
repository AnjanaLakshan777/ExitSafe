"""Possible market impact of an event, given as scenarios rather than a single predicted move."""

from dataclasses import dataclass

from app.data.schemas.event_schema import ImpactDirection
from app.intelligence.models import Evidence


@dataclass(frozen=True)
class ImpactScenario:
    name: str                 # e.g. "Contained", "Severe"
    price_change_low: float   # fractional, e.g. -0.15 for -15%
    price_change_high: float
    assumption: str           # what has to be true for this scenario

    def __post_init__(self):
        if not self.name.strip() or not self.assumption.strip():
            raise ValueError("Scenario name and assumption are required")
        if self.price_change_low > self.price_change_high:
            raise ValueError("price_change_low cannot exceed price_change_high")
        if self.price_change_low < -1.0:
            raise ValueError("A price cannot fall by more than 100%")


@dataclass(frozen=True)
class EventImpact:
    event_id: str
    direction: ImpactDirection
    scenarios: tuple[ImpactScenario, ...] = ()
    evidence: tuple[Evidence, ...] = ()


def assess_event_impact(event, market_data):
    """Estimate the possible impact of an event from market data."""
    raise NotImplementedError("Event impact assessment is not implemented yet")
