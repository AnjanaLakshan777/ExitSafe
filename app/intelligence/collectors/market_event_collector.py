"""Collect market-wide and macroeconomic events.

Intended sources: central-bank policy announcements, exchange notices (trading
halts, index changes), government economic releases and credit-rating agency
announcements. These events usually have no single ``symbol``; they are linked
to holdings later through sector or market-wide exposure.

Implements the ``Collector`` contract in app.intelligence.models.
Not implemented yet: no network access exists in this phase.
"""


def collect_market_events(since):
    """Return market-event ``RawItem``s published at or after ``since`` (tz-aware datetime)."""
    raise NotImplementedError("Market event collection is not implemented yet")
