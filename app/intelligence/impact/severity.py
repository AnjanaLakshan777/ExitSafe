"""Severity of an event, starting from a simple default per event type."""

from app.data.schemas.event_schema import EventType, Severity

BASE_SEVERITY = {
    EventType.CYBERSECURITY_INCIDENT: Severity.HIGH,
    EventType.DATA_BREACH: Severity.HIGH,
    EventType.REGULATORY_ACTION: Severity.MEDIUM,
    EventType.FINANCIAL_RESULT: Severity.MEDIUM,
    EventType.MANAGEMENT_CHANGE: Severity.MEDIUM,
    EventType.FRAUD_OR_GOVERNANCE: Severity.CRITICAL,
    EventType.LEGAL_EVENT: Severity.MEDIUM,
    EventType.CREDIT_EVENT: Severity.HIGH,
    EventType.MACROECONOMIC_EVENT: Severity.MEDIUM,
    EventType.SECTOR_EVENT: Severity.MEDIUM,
    EventType.LIQUIDITY_EVENT: Severity.HIGH,
    EventType.OTHER: Severity.LOW,
}


def base_severity(event_type):
    """Prior severity for an event type."""
    return BASE_SEVERITY[EventType(event_type)]
