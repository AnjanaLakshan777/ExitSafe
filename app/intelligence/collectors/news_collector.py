"""Collect financial news articles.

Intended sources: reputable financial news outlets via their public RSS feeds
or licensed APIs, respecting each site's terms of use and robots.txt.
Items are returned with ``SourceType.NEWS``; a news report is a *reported
claim*, never a confirmed fact on its own.

Implements the ``Collector`` contract in app.intelligence.models.
Not implemented yet: no network access exists in this phase.
"""


def collect_news(since):
    """Return news ``RawItem``s published at or after ``since`` (tz-aware datetime)."""
    raise NotImplementedError("News collection is not implemented yet")
