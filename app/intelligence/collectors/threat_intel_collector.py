"""Collect cybersecurity and data-breach intelligence from lawful sources.

Allowed sources ONLY:
  * official cybersecurity alerts (national CERTs, e.g. Sri Lanka CERT|CC)
  * reputable, licensed threat-intelligence feeds used under their terms
  * public security research and public news reporting on incidents
  * company and regulatory breach disclosures

Explicitly out of scope, and must never be added: accessing private systems,
logging into restricted forums or marketplaces, downloading leaked data or
credentials, or any exploit / credential-theft functionality.

Items are returned with ``SourceType.THREAT_INTELLIGENCE``. A threat-intel
claim (e.g. a "dark web" listing reported by a feed) is an *unverified claim*:
events built from it start as ``VerificationStatus.UNVERIFIED`` and the event model
refuses to mark them CONFIRMED without an official source.

Implements the ``Collector`` contract in app.intelligence.models.
Not implemented yet: no network access exists in this phase.
"""


def collect_threat_intel(since):
    """Return threat-intel ``RawItem``s published at or after ``since`` (tz-aware datetime)."""
    raise NotImplementedError("Threat intelligence collection is not implemented yet")
