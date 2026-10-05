import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from app.data.schemas.event_schema import (
    EventType,
    ImpactDirection,
    MarketEvent,
    Severity,
    SourceType,
    VerificationStatus,
    make_event_id,
)
from app.intelligence.models import Evidence, EvidenceBasis, RawItem

PUBLISHED = datetime(2026, 3, 10, 9, 30, tzinfo=timezone.utc)
DETECTED = PUBLISHED + timedelta(minutes=5)


def make_event(**overrides):
    values = dict(
        event_id="evt-001",
        event_type=EventType.DATA_BREACH,
        title="ABC Bank investigates possible customer data exposure",
        source_name="Example Financial News",
        source_type=SourceType.NEWS,
        published_time=PUBLISHED,
        detected_time=DETECTED,
    )
    values.update(overrides)
    return MarketEvent(**values)


# --- valid creation -------------------------------------------------------------

def test_minimal_event_has_unassessed_defaults():
    event = make_event()

    assert event.verification_status is VerificationStatus.UNVERIFIED
    assert event.severity is None
    assert event.confidence is None
    assert event.symbol is None


def test_fully_populated_event():
    event = make_event(
        symbol=" abc ",
        company_name="ABC Bank PLC",
        description="A news report says customer records may have been exposed.",
        source_url="https://news.example.com/abc-bank",
        event_time=PUBLISHED - timedelta(days=1),
        severity=Severity.HIGH,
        confidence=0.6,
        affected_sector="Banking",
        potential_market_impact=ImpactDirection.NEGATIVE,
        verification_status=VerificationStatus.REPORTED,
    )

    assert event.symbol == "ABC"  # normalized like market-data symbols
    assert event.severity is Severity.HIGH
    assert event.confidence == 0.6


def test_events_are_immutable_and_replace_revalidates():
    event = make_event()

    with pytest.raises(AttributeError):
        event.confidence = 0.5  # type: ignore[misc]

    assert replace(event, confidence=0.5).confidence == 0.5
    with pytest.raises(ValueError):
        replace(event, confidence=2.0)


# --- required fields ------------------------------------------------------------

@pytest.mark.parametrize("field", ["event_id", "title", "source_name"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_required_text_fields_cannot_be_blank(field, value):
    with pytest.raises(ValueError, match=field):
        make_event(**{field: value})


def test_missing_required_argument_raises():
    with pytest.raises(TypeError):
        MarketEvent(event_id="evt-001", title="No type or source")  # type: ignore[call-arg]


def test_blank_symbol_is_rejected():
    with pytest.raises(ValueError, match="symbol"):
        make_event(symbol="  ")


# --- event type / source type / status -----------------------------------------

@pytest.mark.parametrize("bad_type", ["NOT_A_TYPE", "DATA_BREACH", None, 3])
def test_event_type_must_be_an_event_type(bad_type):
    # Even a valid name as a plain string is rejected: conversion from text
    # happens explicitly in from_dict, never silently in the constructor.
    with pytest.raises(ValueError, match="event_type"):
        make_event(event_type=bad_type)


def test_source_type_must_be_a_source_type():
    with pytest.raises(ValueError, match="source_type"):
        make_event(source_type="BLOG")


def test_all_required_event_types_exist():
    expected = {
        "CYBERSECURITY_INCIDENT", "DATA_BREACH", "REGULATORY_ACTION", "FINANCIAL_RESULT",
        "MANAGEMENT_CHANGE", "FRAUD_OR_GOVERNANCE", "LEGAL_EVENT", "CREDIT_EVENT",
        "MACROECONOMIC_EVENT", "SECTOR_EVENT", "LIQUIDITY_EVENT", "OTHER",
    }
    assert {t.value for t in EventType} == expected


@pytest.mark.parametrize("source_type", [
    SourceType.NEWS, SourceType.THREAT_INTELLIGENCE, SourceType.SEARCH_RESULT, SourceType.OTHER,
])
def test_unofficial_source_cannot_confirm_an_event(source_type):
    with pytest.raises(ValueError, match="cannot be CONFIRMED"):
        make_event(source_type=source_type, verification_status=VerificationStatus.CONFIRMED)


@pytest.mark.parametrize("source_type", [
    SourceType.OFFICIAL_DISCLOSURE, SourceType.REGULATOR, SourceType.COMPANY,
])
def test_official_source_can_confirm_an_event(source_type):
    event = make_event(source_type=source_type, verification_status=VerificationStatus.CONFIRMED)
    assert event.verification_status is VerificationStatus.CONFIRMED


# --- timestamps -----------------------------------------------------------------

@pytest.mark.parametrize("field", ["published_time", "detected_time", "event_time"])
def test_naive_timestamps_are_rejected(field):
    overrides = {field: datetime(2026, 3, 10, 9, 30)}
    if field == "published_time":
        overrides["detected_time"] = datetime(2026, 3, 10, 9, 35)
    with pytest.raises(ValueError, match="timezone-aware"):
        make_event(**overrides)


def test_non_datetime_timestamp_is_rejected():
    with pytest.raises(TypeError, match="published_time"):
        make_event(published_time="2026-03-10T09:30:00+00:00")


def test_detected_before_published_is_rejected():
    with pytest.raises(ValueError, match="detected_time"):
        make_event(detected_time=PUBLISHED - timedelta(seconds=1))


def test_timezones_are_compared_correctly():
    colombo = timezone(timedelta(hours=5, minutes=30))
    # 15:05 in Colombo is 09:35 UTC, i.e. after publication.
    event = make_event(detected_time=datetime(2026, 3, 10, 15, 5, tzinfo=colombo))
    assert event.detected_time > event.published_time


def test_future_event_time_is_allowed():
    # e.g. an announcement of an upcoming results release
    event = make_event(event_type=EventType.FINANCIAL_RESULT,
                       event_time=PUBLISHED + timedelta(days=14))
    assert event.event_time > event.published_time


# --- confidence and severity ----------------------------------------------------

@pytest.mark.parametrize("value", [0, 0.0, 0.35, 1, 1.0])
def test_confidence_within_range_is_accepted(value):
    assert make_event(confidence=value).confidence == value


@pytest.mark.parametrize("value", [-0.01, 1.01, 50])
def test_confidence_outside_range_is_rejected(value):
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_event(confidence=value)


@pytest.mark.parametrize("value", ["0.5", True])
def test_confidence_must_be_numeric(value):
    with pytest.raises(TypeError, match="confidence"):
        make_event(confidence=value)


@pytest.mark.parametrize("severity", list(Severity))
def test_valid_severities_are_accepted(severity):
    assert make_event(severity=severity).severity is severity


@pytest.mark.parametrize("value", ["EXTREME", "HIGH", 3])
def test_invalid_severity_is_rejected(value):
    with pytest.raises(ValueError, match="severity"):
        make_event(severity=value)


# --- serialization --------------------------------------------------------------

def test_event_round_trips_through_json():
    event = make_event(
        symbol="ABC",
        event_time=PUBLISHED - timedelta(hours=3),
        severity=Severity.HIGH,
        confidence=0.6,
        potential_market_impact=ImpactDirection.NEGATIVE,
        verification_status=VerificationStatus.REPORTED,
    )

    payload = json.dumps(event.to_dict())
    restored = MarketEvent.from_dict(json.loads(payload))

    assert restored == event
    assert json.loads(payload)["event_type"] == "DATA_BREACH"
    assert json.loads(payload)["published_time"] == "2026-03-10T09:30:00+00:00"


def test_optional_fields_round_trip_as_none():
    event = make_event()
    assert MarketEvent.from_dict(event.to_dict()) == event


def test_from_dict_rejects_unknown_event_type():
    data = make_event().to_dict()
    data["event_type"] = "ALIEN_INVASION"
    with pytest.raises(ValueError):
        MarketEvent.from_dict(data)


def test_from_dict_applies_the_same_validation():
    data = make_event().to_dict()
    data["source_type"] = "THREAT_INTELLIGENCE"
    data["verification_status"] = "CONFIRMED"
    with pytest.raises(ValueError, match="cannot be CONFIRMED"):
        MarketEvent.from_dict(data)


# --- ids, raw items, evidence ---------------------------------------------------

def test_event_id_is_deterministic():
    first = make_event_id("Example News", PUBLISHED, source_url="https://x.example/a")
    again = make_event_id("  example news ", PUBLISHED, source_url="https://X.example/a")
    other = make_event_id("Example News", PUBLISHED, source_url="https://x.example/b")

    assert first == again
    assert first != other


def test_raw_item_requires_aware_timestamps():
    with pytest.raises(ValueError, match="timezone-aware"):
        RawItem(source_name="Example", source_type=SourceType.NEWS, title="t",
                published_time=datetime(2026, 3, 10), fetched_time=DETECTED)


def test_evidence_requires_a_basis():
    assert Evidence("Bank confirmed the incident", EvidenceBasis.CONFIRMED_FACT)
    with pytest.raises(ValueError, match="basis"):
        Evidence("Bank confirmed the incident", "FACT")
