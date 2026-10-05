"""Build normalized ``MarketEvent``s from parsed ``RawItem``s."""

import re

from app.data.schemas.event_schema import (
    OFFICIAL_SOURCE_TYPES,
    MarketEvent,
    SourceType,
    VerificationStatus,
    make_event_id,
)


def initial_status(source_type):
    """Verification status an event starts with, based only on its source.

    Official sources confirm, credible news reports, everything else
    (threat-intel claims, search results, unknown sources) is unverified.
    """
    if source_type in OFFICIAL_SOURCE_TYPES:
        return VerificationStatus.CONFIRMED
    if source_type is SourceType.NEWS:
        return VerificationStatus.REPORTED
    return VerificationStatus.UNVERIFIED


def find_mentioned_symbols(text, company_directory):
    """Return the sorted symbols whose company names/aliases appear in ``text``.

    ``company_directory`` maps symbol -> list of names, e.g.
    ``{"ABC": ["ABC Bank PLC", "ABC Bank"]}``. Matching is case-insensitive
    on whole words.
    """
    found = set()
    for symbol, names in company_directory.items():
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
                found.add(symbol.strip().upper())
                break
    return sorted(found)


def parse_event(item, event_type, detected_time, symbol=None, company_name=None,
                affected_sector=None):
    """Create an unassessed ``MarketEvent`` (no severity/confidence yet) from a ``RawItem``."""
    return MarketEvent(
        event_id=make_event_id(item.source_name, item.published_time,
                               source_url=item.source_url, title=item.title),
        event_type=event_type,
        title=item.title,
        description=item.body,
        source_name=item.source_name,
        source_url=item.source_url,
        source_type=item.source_type,
        published_time=item.published_time,
        detected_time=detected_time,
        symbol=symbol,
        company_name=company_name,
        affected_sector=affected_sector,
        verification_status=initial_status(item.source_type),
    )
