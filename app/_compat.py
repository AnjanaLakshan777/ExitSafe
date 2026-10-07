"""Compatibility helpers for running on Python versions older than 3.11."""

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of enum.StrEnum (Python 3.11+)."""

        def __new__(cls, *values):
            value = str(*values)
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return str.__str__(self)

        def __format__(self, format_spec: str) -> str:
            return str.__format__(str(self), format_spec)

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()


__all__ = ["StrEnum"]
