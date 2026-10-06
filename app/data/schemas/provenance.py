"""Where an imported file came from (source, hash, retrieval time), so results can be reproduced."""

import hashlib
import json
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timezone
from enum import StrEnum
from pathlib import Path

from app.data.schemas.checks import require_aware, require_enum, require_text
from app.data.source_catalog import DataSourceType


@dataclass(frozen=True)
class Provenance:
    source_name: str
    source_type: DataSourceType
    original_file_name: str
    file_sha256: str
    file_size_bytes: int
    loaded_time: datetime                 # when ExitSafe read the file
    retrieval_time: datetime | None = None  # when the file was downloaded, if known
    source_date: date | None = None       # as-of date stated by the source, if it states one
    source_url: str | None = None
    source_version: str | None = None     # e.g. dataset commit / file version
    # Columns ExitSafe derived rather than read from the source: {column: rule}.
    derived_fields: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        require_text(self.source_name, "source_name")
        require_enum(self.source_type, DataSourceType, "source_type")
        require_text(self.original_file_name, "original_file_name")
        require_text(self.file_sha256, "file_sha256")
        require_aware(self.loaded_time, "loaded_time")
        require_aware(self.retrieval_time, "retrieval_time", optional=True)
        if self.source_date is not None and (
                isinstance(self.source_date, datetime) or not isinstance(self.source_date, date)):
            raise TypeError("source_date must be a date (not a datetime)")

    def to_dict(self):
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (date, datetime)):
                value = value.isoformat()
            elif isinstance(value, StrEnum):
                value = value.value
            elif isinstance(value, dict):
                value = dict(value)
            result[item.name] = value
        return result


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(path, source, retrieval_time=None, source_date=None, source_url=None,
                     source_version=None, loaded_time=None):
    path = Path(path)
    return Provenance(
        source_name=source.source_name,
        source_type=source.source_type,
        original_file_name=path.name,
        file_sha256=file_sha256(path),
        file_size_bytes=path.stat().st_size,
        loaded_time=loaded_time or datetime.now(timezone.utc),
        retrieval_time=retrieval_time,
        source_date=source_date,
        source_url=source_url or source.location,
        source_version=source_version,
    )


def write_manifest(path, provenance, validation=None):
    """Write provenance (and the validation summary, if given) as JSON."""
    manifest = {"provenance": provenance.to_dict(),
                "validation": validation.summary() if validation is not None else None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
