"""Small helpers shared by the analytics modules."""

import math

from app.analytics.returns import CANONICAL_REQUIRED_COLUMNS

TRADING_DAYS_PER_YEAR = 252

# Variation below this (in daily-return units) is treated as no variation at
# all, so floating-point noise (e.g. a steady growth path) is not mistaken for
# real volatility in correlations and risk-adjusted ratios.
CONSTANT_RETURN_TOLERANCE = 1e-12


def require_canonical_columns(data, purpose):
    """Raise ValueError naming any missing canonical column (date, symbol, close)."""
    missing = [col for col in CANONICAL_REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Cannot calculate {purpose}, missing column(s): {', '.join(missing)}")


def validate_periods_per_year(periods_per_year):
    """Raise ValueError unless periods_per_year is a positive, finite number (bool excluded)."""
    if (isinstance(periods_per_year, bool) or not isinstance(periods_per_year, (int, float))
            or not periods_per_year > 0 or not math.isfinite(periods_per_year)):
        raise ValueError(f"periods_per_year must be a positive number, got {periods_per_year!r}")
