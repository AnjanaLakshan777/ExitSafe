"""Collect financial news from RSS feeds and from listing pages whose robots.txt allows it.

A news report is a claim, not a confirmed fact.
"""

import calendar
import re
from datetime import datetime, timezone
from urllib.parse import urljoin

import feedparser
from bs4 import BeautifulSoup

from app.intelligence.collectors.http import CollectionError, fetch
from app.intelligence.models import RawItem
from app.intelligence.sources import HTML_PAGES, RSS_FEEDS

MIN_HEADLINE_CHARS = 25
MAX_HEADLINE_CHARS = 300
# IMF-style listings put the date in front of the headline ("October 7, 2026Coming Soon: ...").
_LEADING_DATE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}\s*")


def _entry_time(entry):
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    return None


def collect_feed(source, since, now=None):
    """Items from one RSS/Atom feed published at or after ``since``. Undated entries are skipped."""
    now = now or datetime.now(timezone.utc)
    feed = feedparser.parse(fetch(source.url).content)
    if feed.bozo and not feed.entries:
        raise CollectionError(f"{source.url}: not a readable feed ({feed.bozo_exception})")

    items = []
    for entry in feed.entries:
        published = _entry_time(entry)
        title = (entry.get("title") or "").strip()
        if published is None or not title or published < since:
            continue
        items.append(RawItem(
            source_name=source.name, source_type=source.source_type, title=title,
            body=entry.get("summary") or "", source_url=entry.get("link"),
            published_time=min(published, now),   # a feed clock running ahead is not "the future"
            fetched_time=now))
    return items


def collect_page(source, since, now=None):
    """Headline links on one listing page whose URL carries a publish date at or after ``since``."""
    now = now or datetime.now(timezone.utc)
    soup = BeautifulSoup(fetch(source.url, check_robots=True).text, "html.parser")
    pattern = re.compile(source.link_pattern, re.IGNORECASE)

    items, seen = [], set()
    for link in soup.find_all("a", href=True):
        url = urljoin(source.url, link["href"])
        title = _LEADING_DATE.sub("", link.get_text(" ", strip=True))
        match = pattern.search(url)
        if (not match or url in seen
                or not MIN_HEADLINE_CHARS <= len(title) <= MAX_HEADLINE_CHARS):
            continue
        try:
            published = datetime(int(match["year"]), int(match["month"]), int(match["day"]),
                                  tzinfo=timezone.utc)
        except (IndexError, ValueError):
            continue
        # A date-only URL means "some time that day"; only whole days before `since` are old.
        if published.date() < since.date():
            continue
        seen.add(url)
        items.append(RawItem(source_name=source.name, source_type=source.source_type,
                             title=title, source_url=url, published_time=min(published, now),
                             fetched_time=now))
    return items


def collect_news(since, feeds=RSS_FEEDS, pages=HTML_PAGES, errors=None):
    """Return news ``RawItem``s published at or after ``since`` (tz-aware datetime).

    A source that fails is skipped; its error is appended to ``errors`` if given.
    """
    items = []
    for collect, sources in ((collect_feed, feeds), (collect_page, pages)):
        for source in sources:
            try:
                items.extend(collect(source, since))
            except CollectionError as exc:
                if errors is not None:
                    errors.append(f"{source.name}: {exc}")
    return items
