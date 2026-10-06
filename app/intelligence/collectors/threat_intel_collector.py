"""Collect cybersecurity and data-breach information from lawful sources only.

That means official alerts (e.g. Sri Lanka CERT|CC), licensed threat feeds,
public research and company or regulator disclosures. Accessing private
systems or restricted forums, downloading leaked data or anything like
credential theft must never be added. Claims from these sources start as
unverified. Not implemented yet.
"""


def collect_threat_intel(since):
    """Return threat-intel ``RawItem``s published at or after ``since`` (tz-aware datetime)."""
    raise NotImplementedError("Threat intelligence collection is not implemented yet")
