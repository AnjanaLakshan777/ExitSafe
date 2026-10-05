"""Potential financial/market impact of an event.

Impact is expressed as a risk direction plus a set of *scenarios* (ranges of
possible price change under stated assumptions), never as a single predicted
price move.
"""

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
    """Estimate the potential impact of ``event`` using cleaned ``market_data``.

    Planned inputs: event type/severity/confidence, the stock's recent
    volatility and liquidity (Value Traded) from the market-data domain, and
    historical reactions to similar events from data/intelligence/historical.
    """
    raise NotImplementedError("Event impact assessment is not implemented yet")
