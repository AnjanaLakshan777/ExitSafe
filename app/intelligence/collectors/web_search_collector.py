"""Find world-market threat news with Gemini + Google Search.

Search answers are the least reliable source: they start as ``SEARCH_RESULT``
(credibility 0.2) and are only kept when Gemini gives a publish date.
"""

from datetime import date, datetime, time, timezone

from app.data.schemas.event_schema import SourceType
from app.intelligence.collectors.gemini import grounded_json
from app.intelligence.collectors.http import CollectionError
from app.intelligence.models import RawItem

MAX_RESULTS = 10

PROMPT = """Search the web for news published in the last {hours} hours about events that
threaten world financial markets: market crashes or sharp sell-offs, recession signals,
wars and military conflicts, sanctions, banking or debt crises, sovereign defaults or
downgrades, oil or currency shocks, emergency central-bank moves, trade wars and
pandemics. Include Sri Lanka and South Asia if relevant.

Reply with ONLY a JSON array (no prose) of at most {limit} objects, newest first:
[{{"title": "...", "summary": "one or two sentences", "source_name": "publisher",
   "url": "article url", "published": "YYYY-MM-DD"}}]
Only include stories you actually found in search results. Never invent a story,
date or URL. If nothing qualifies, reply with []."""


def _published(value, now):
    """Publish date as UTC midnight; None if missing, unreadable or in the future."""
    try:
        day = date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None
    if day > now.date():
        return None          # a story dated tomorrow can't have been found today
    return datetime.combine(day, time(), tzinfo=timezone.utc)


def collect_web_search(since, settings, now=None):
    """Return threat-news ``RawItem``s found by Gemini search, published on or after ``since``."""
    now = now or datetime.now(timezone.utc)
    hours = max(1, round((now - since).total_seconds() / 3600))
    data, cited = grounded_json(PROMPT.format(hours=hours, limit=MAX_RESULTS), settings)
    if not isinstance(data, list):
        raise CollectionError("Gemini search reply was not a JSON list")

    items = []
    for entry in data[:MAX_RESULTS]:
        if not isinstance(entry, dict) or not str(entry.get("title") or "").strip():
            continue
        published = _published(entry.get("published"), now)
        if published is None or published.date() < since.date():
            continue
        publisher = str(entry.get("source_name") or "").strip() or "unknown publisher"
        url = str(entry.get("url") or "").strip() or (cited[0] if cited else None)
        items.append(RawItem(
            source_name=f"Gemini web search ({publisher})",
            source_type=SourceType.SEARCH_RESULT,
            title=str(entry["title"]).strip(),
            body=str(entry.get("summary") or ""),
            source_url=url,
            published_time=published,
            fetched_time=now))
    return items
