"""One scan: collect news, keep the stories that signal a world-market threat, store and alert."""

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from app.data.schemas.event_schema import ImpactDirection
from app.intelligence.classification.event_classifier import classify_event
from app.intelligence.classification.threat_keywords import assess_threat, severity_rank
from app.intelligence.collectors.http import CollectionError
from app.intelligence.collectors.news_collector import collect_news
from app.intelligence.collectors.web_search_collector import collect_web_search
from app.intelligence.impact.confidence import assess_confidence
from app.intelligence.notifier import send_alert_email
from app.intelligence.parsers.event_parser import parse_event
from app.intelligence.parsers.news_parser import parse_news_item
from app.intelligence.settings import BOT_STATUS_FILE
from app.intelligence.threat_store import ThreatRecord, ThreatStore


@dataclass
class ScanResult:
    started: datetime
    items_collected: int = 0
    new_threats: list = field(default_factory=list)      # ThreatRecord
    alerted: list = field(default_factory=list)          # ThreatRecord emailed this scan
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def threat_record(item):
    """A ``ThreatRecord`` for a ``RawItem``, or None when it signals no threat."""
    item = parse_news_item(item)
    threat = assess_threat(item.title, item.body)
    if not threat.is_threat:
        return None
    event = parse_event(item, classify_event(item.title, item.body).event_type,
                        detected_time=item.fetched_time)
    event = replace(event, severity=threat.severity,
                    confidence=assess_confidence([item.source_type]),
                    potential_market_impact=ImpactDirection.NEGATIVE)
    return ThreatRecord(event=event, threat_terms=threat.matched_terms)


def _default_collectors(settings):
    collectors = [("News feeds and pages", lambda since, errors: collect_news(since, errors=errors))]
    if settings.use_web_search and settings.gemini_configured:
        collectors.append(("Gemini web search",
                           lambda since, errors: collect_web_search(since, settings)))
    return collectors


def run_threat_scan(settings, *, store=None, collectors=None, send_email=send_alert_email,
                    now=None):
    """Collect, assess, store and (if configured) email new threats at or above the alert level."""
    now = now or datetime.now(timezone.utc)
    result = ScanResult(started=now)
    since = now - timedelta(hours=settings.lookback_hours)
    store = store if store is not None else ThreatStore()

    items = []
    for name, collect in collectors if collectors is not None else _default_collectors(settings):
        errors = []
        try:
            items.extend(collect(since, errors))
        except CollectionError as exc:
            errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - one broken source must not stop the scan
            errors.append(f"{type(exc).__name__}: {exc}")
        result.errors.extend(f"{name}: {e}" for e in errors)
    if settings.use_web_search and not settings.gemini_configured:
        result.notes.append("Gemini web search skipped: GEMINI_API_KEY is not set")
    result.items_collected = len(items)

    for item in items:
        record = threat_record(item)
        if record is not None and store.add(record):
            result.new_threats.append(record)

    floor = severity_rank(settings.alert_min_severity)
    due = sorted((r for r in store.records
                  if r.alerted_time is None and r.event.published_time >= since
                  and severity_rank(r.event.severity) >= floor),
                 key=lambda r: (-severity_rank(r.event.severity), r.event.published_time))
    if due and settings.email_configured:
        try:
            send_email(due, settings)
            store.mark_alerted([r.event.event_id for r in due], now)
            result.alerted = due
        except Exception as exc:  # noqa: BLE001 - keep threats unalerted; retry next scan
            result.errors.append(f"Email alert failed: {type(exc).__name__}: {exc}")
    elif due:
        result.notes.append(f"{len(due)} threat(s) at {settings.alert_min_severity.value}+ not "
                            "emailed: SMTP_USER, SMTP_PASSWORD or ALERT_EMAIL_TO is not set")
    store.save()
    write_status(result)
    return result


def write_status(result):
    """Summary of the last scan, shown in the dashboard."""
    BOT_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BOT_STATUS_FILE.write_text(json.dumps({
        "last_scan": result.started.isoformat(),
        "items_collected": result.items_collected,
        "new_threats": len(result.new_threats),
        "emailed": len(result.alerted),
        "errors": result.errors,
        "notes": result.notes,
    }, indent=2), encoding="utf-8")


def read_status():
    if not BOT_STATUS_FILE.exists():
        return None
    return json.loads(BOT_STATUS_FILE.read_text(encoding="utf-8"))
