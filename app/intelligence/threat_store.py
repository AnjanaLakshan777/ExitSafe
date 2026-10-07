"""Detected market threats, kept as JSON lines so the bot and the dashboard can share them."""

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime

from app.data.schemas.event_schema import MarketEvent
from app.intelligence.classification.threat_keywords import severity_rank
from app.intelligence.settings import THREATS_FILE

MAX_RECORDS = 2000
_NOT_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ThreatRecord:
    event: MarketEvent
    threat_terms: tuple[str, ...]
    alerted_time: datetime | None = None

    def to_dict(self):
        return {"event": self.event.to_dict(), "threat_terms": list(self.threat_terms),
                "alerted_time": self.alerted_time.isoformat() if self.alerted_time else None}

    @classmethod
    def from_dict(cls, data):
        alerted = data.get("alerted_time")
        return cls(event=MarketEvent.from_dict(data["event"]),
                   threat_terms=tuple(data.get("threat_terms") or ()),
                   alerted_time=datetime.fromisoformat(alerted) if alerted else None)


def headline_key(title):
    """Same story syndicated under slightly different punctuation/case counts once."""
    return _NOT_WORD.sub(" ", title.lower()).strip()


class ThreatStore:
    def __init__(self, path=THREATS_FILE, max_records=MAX_RECORDS):
        self.path = path
        self.max_records = max_records
        self.records = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(ThreatRecord.from_dict(json.loads(line)))

    def add(self, record):
        """Add a record; returns False if the same event or headline is already stored."""
        key = headline_key(record.event.title)
        if any(r.event.event_id == record.event.event_id or headline_key(r.event.title) == key
               for r in self.records):
            return False
        self.records.append(record)
        return True

    def mark_alerted(self, event_ids, when):
        ids = set(event_ids)
        self.records = [replace(r, alerted_time=when) if r.event.event_id in ids else r
                        for r in self.records]

    def recent(self, limit=50, min_severity=None):
        """Newest first, optionally only at or above ``min_severity``."""
        records = self.records
        if min_severity is not None:
            floor = severity_rank(min_severity)
            records = [r for r in records if severity_rank(r.event.severity) >= floor]
        return sorted(records, key=lambda r: r.event.published_time, reverse=True)[:limit]

    def save(self):
        self.records = sorted(self.records, key=lambda r: r.event.published_time)[-self.max_records:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in self.records),
                        encoding="utf-8")
        temp.replace(self.path)
