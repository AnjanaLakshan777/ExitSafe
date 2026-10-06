"""Periodic jobs. No scheduler library is wired in yet; a runner only needs to loop over JOBS."""

from collections.abc import Callable
from dataclasses import dataclass

from app.intelligence.collectors.company_disclosure_collector import collect_company_disclosures
from app.intelligence.collectors.market_event_collector import collect_market_events
from app.intelligence.collectors.news_collector import collect_news
from app.intelligence.collectors.threat_intel_collector import collect_threat_intel
from app.intelligence.models import Collector

COLLECTORS: tuple[Collector, ...] = (
    collect_company_disclosures,
    collect_news,
    collect_market_events,
    collect_threat_intel,
)


@dataclass(frozen=True)
class Job:
    name: str
    run: Callable[[], None]
    interval_minutes: int


def collect_intelligence():
    """Run every collector, then parse, classify and store the new events."""
    raise NotImplementedError("Intelligence collection job is not implemented yet")


def refresh_market_data():
    """Load the latest market-data files into the processed store."""
    raise NotImplementedError("Market data refresh job is not implemented yet")


def archive_events():
    """Move processed events into INTELLIGENCE_HISTORICAL_DIR for backtesting."""
    raise NotImplementedError("Event archiving job is not implemented yet")


JOBS = (
    Job("collect_intelligence", collect_intelligence, interval_minutes=30),
    Job("refresh_market_data", refresh_market_data, interval_minutes=24 * 60),
    Job("archive_events", archive_events, interval_minutes=24 * 60),
)
