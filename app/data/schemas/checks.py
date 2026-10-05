"""Small field-validation helpers shared by the dataclass schemas."""

from datetime import datetime


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required and must be a non-empty string")


def require_enum(value, enum_type, name, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, enum_type):
        raise ValueError(
            f"{name} must be a {enum_type.__name__}, got {value!r}. "
            f"Valid values: {', '.join(member.value for member in enum_type)}"
        )


def require_aware(value, name, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (e.g. UTC or Asia/Colombo)")


def validate_confidence(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"confidence must be a number, got {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {value}")
    return float(value)
