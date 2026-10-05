"""Collect official company disclosures and announcements.

Intended sources: stock-exchange disclosure pages (e.g. CSE company
announcements), regulator publications and companies' own investor-relations
releases. Items are returned with ``SourceType.OFFICIAL_DISCLOSURE``,
``SourceType.REGULATOR`` or ``SourceType.COMPANY`` - the only source types
allowed to confirm an event.

Implements the ``Collector`` contract in app.intelligence.models.
Not implemented yet: no network access exists in this phase.
"""


def collect_company_disclosures(since):
    """Return disclosure ``RawItem``s published at or after ``since`` (tz-aware datetime)."""
    raise NotImplementedError("Company disclosure collection is not implemented yet")
