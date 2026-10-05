from datetime import datetime, timedelta, timezone

import pytest

from app.intelligence.classification.event_classifier import classify_event
from app.data.schemas.event_schema import EventType
from app.intelligence.classification.event_types import EVENT_KEYWORDS
from app.intelligence.impact.confidence import UNOFFICIAL_CAP, assess_confidence
from app.intelligence.impact.severity import base_severity
from app.data.schemas.event_schema import Severity, SourceType, VerificationStatus
from app.intelligence.models import RawItem
from app.intelligence.parsers.event_parser import find_mentioned_symbols, parse_event
from app.intelligence.parsers.news_parser import clean_text, parse_news_item

PUBLISHED = datetime(2026, 3, 10, 9, 30, tzinfo=timezone.utc)
DETECTED = PUBLISHED + timedelta(minutes=5)


# --- classification -------------------------------------------------------------

@pytest.mark.parametrize("title, expected", [
    ("ABC Bank investigates possible customer data exposure", EventType.DATA_BREACH),
    ("Ransomware attack disrupts XYZ Finance systems", EventType.CYBERSECURITY_INCIDENT),
    ("Hackers claim access to LMN Insurance network", EventType.CYBERSECURITY_INCIDENT),
    ("Regulator imposes penalty on ABC Bank", EventType.REGULATORY_ACTION),
    ("XYZ PLC announces quarterly results", EventType.FINANCIAL_RESULT),
    ("LMN Holdings CEO resigns", EventType.MANAGEMENT_CHANGE),
    ("Forensic audit uncovers accounting irregularities at XYZ", EventType.FRAUD_OR_GOVERNANCE),
    ("Customers file lawsuit against LMN", EventType.LEGAL_EVENT),
    ("Rating agency announces downgrade of ABC Finance", EventType.CREDIT_EVENT),
    ("Central bank raises policy rate by 50 basis points", EventType.MACROECONOMIC_EVENT),
    ("Trading halt imposed on XYZ shares", EventType.LIQUIDITY_EVENT),
    ("New tariff hits apparel exporters", EventType.SECTOR_EVENT),
    ("ABC opens a new branch in Kandy", EventType.OTHER),
])
def test_classify_event(title, expected):
    assert classify_event(title).event_type is expected


def test_classification_records_matched_terms():
    result = classify_event("Bank hit by ransomware", "Attackers deployed malware")
    assert result.matched_terms == ("ransomware", "malware")


def test_more_specific_type_wins():
    # Mentions both a cyber attack and a data breach -> the breach is the more useful label.
    result = classify_event("Cyber attack leads to data breach at ABC Bank")
    assert result.event_type is EventType.DATA_BREACH


def test_keywords_match_at_word_start_only():
    assert classify_event("Shackles removed from old warehouse").event_type is EventType.OTHER


def test_classification_uses_description():
    result = classify_event("ABC Bank statement", "The bank reported a data breach.")
    assert result.event_type is EventType.DATA_BREACH


def test_every_type_except_other_has_keywords():
    assert set(EVENT_KEYWORDS) == set(EventType) - {EventType.OTHER}


# --- severity -------------------------------------------------------------------

def test_every_event_type_has_a_base_severity():
    for event_type in EventType:
        assert isinstance(base_severity(event_type), Severity)


def test_fraud_is_rated_above_other():
    assert base_severity(EventType.FRAUD_OR_GOVERNANCE) is Severity.CRITICAL
    assert base_severity(EventType.OTHER) is Severity.LOW


# --- confidence -----------------------------------------------------------------

def test_official_sources_are_more_credible_than_news_and_threat_intel():
    official = assess_confidence([SourceType.OFFICIAL_DISCLOSURE])
    news = assess_confidence([SourceType.NEWS])
    threat = assess_confidence([SourceType.THREAT_INTELLIGENCE])
    assert official > news > threat


def test_corroboration_increases_confidence():
    assert assess_confidence([SourceType.NEWS, SourceType.NEWS]) > assess_confidence([SourceType.NEWS])


def test_unofficial_sources_never_reach_certainty():
    many_claims = [SourceType.THREAT_INTELLIGENCE] * 3 + [SourceType.NEWS] * 10
    assert assess_confidence(many_claims) == UNOFFICIAL_CAP


def test_confidence_is_always_between_zero_and_one():
    for source_type in SourceType:
        assert 0.0 <= assess_confidence([source_type] * 20) <= 1.0


def test_confidence_needs_a_source():
    with pytest.raises(ValueError):
        assess_confidence([])


# --- parsing --------------------------------------------------------------------

def make_raw_item(source_type, **overrides):
    values = dict(
        source_name="Example Source",
        source_type=source_type,
        title="ABC Bank investigates possible customer data exposure",
        body="Reports suggest customer records at ABC Bank PLC were exposed.",
        source_url="https://example.com/item/1",
        published_time=PUBLISHED,
        fetched_time=DETECTED,
    )
    values.update(overrides)
    return RawItem(**values)


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>ABC&nbsp;Bank &amp; Co\n\n  <b>update</b></p>") == "ABC Bank & Co update"
    assert clean_text(None) == ""


def test_parse_news_item_cleans_title_and_body():
    item = make_raw_item(SourceType.NEWS, title="<h1>ABC  Bank</h1>", body="a&amp;b")
    parsed = parse_news_item(item)
    assert (parsed.title, parsed.body) == ("ABC Bank", "a&b")


@pytest.mark.parametrize("source_type, expected_status", [
    (SourceType.OFFICIAL_DISCLOSURE, VerificationStatus.CONFIRMED),
    (SourceType.REGULATOR, VerificationStatus.CONFIRMED),
    (SourceType.COMPANY, VerificationStatus.CONFIRMED),
    (SourceType.NEWS, VerificationStatus.REPORTED),
    (SourceType.THREAT_INTELLIGENCE, VerificationStatus.UNVERIFIED),
    (SourceType.SEARCH_RESULT, VerificationStatus.UNVERIFIED),
    (SourceType.OTHER, VerificationStatus.UNVERIFIED),
])
def test_parsed_event_status_depends_on_source(source_type, expected_status):
    event = parse_event(make_raw_item(source_type), EventType.DATA_BREACH, DETECTED, symbol="abc")

    assert event.verification_status is expected_status
    assert event.symbol == "ABC"
    assert event.severity is None and event.confidence is None  # not assessed yet


def test_threat_intel_claim_is_never_confirmed():
    item = make_raw_item(SourceType.THREAT_INTELLIGENCE,
                         title="Actor claims to sell ABC Bank customer database")
    event = parse_event(item, EventType.DATA_BREACH, DETECTED)
    assert event.verification_status is VerificationStatus.UNVERIFIED


def test_same_item_gets_same_event_id():
    item = make_raw_item(SourceType.NEWS)
    first = parse_event(item, EventType.DATA_BREACH, DETECTED)
    later = parse_event(item, EventType.DATA_BREACH, DETECTED + timedelta(hours=1))
    assert first.event_id == later.event_id


def test_find_mentioned_symbols():
    directory = {"ABC": ["ABC Bank PLC", "ABC Bank"], "XYZ": ["XYZ Finance"], "LMN": ["LMN"]}

    assert find_mentioned_symbols("Outage at abc bank branches", directory) == ["ABC"]
    assert find_mentioned_symbols("ABC Bank and XYZ Finance merge", directory) == ["ABC", "XYZ"]
    assert find_mentioned_symbols("LMNOP Holdings results", directory) == []
