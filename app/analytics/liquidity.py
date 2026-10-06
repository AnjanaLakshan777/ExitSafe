"""Stock-level liquidity: trading volume, traded value and position liquidation estimates.

Liquidity describes how easily an investor can buy or sell a position without
needing an unusually large share of the market's normal trading activity.

Usable rows (per symbol; nothing is filled in or interpolated)
  not INVALID (WARNING rows count), with a date, a symbol, a finite positive close
  and a finite non-negative volume. Zero-volume days are valid observations: they
  count as 0 in the averages and in ``zero_volume_days``. Missing trading days
  simply do not appear; ``observations`` is the number of usable rows.

Traded value: one source per symbol, never mixed
  ACTUAL_TURNOVER         reported ``turnover``, used only if EVERY usable row of
                          the symbol has a finite, non-negative turnover
  ESTIMATED_TRADED_VALUE  otherwise close * volume for every usable row (the same
                          definition as the canonical estimated_traded_value).
                          This is an estimate, NOT official turnover. Reported
                          turnover is never overwritten; ``actual_turnover_days``
                          shows how many rows had it.

Metrics
  average_daily_volume (ADV)        mean(volume)
  average_daily_traded_value (ADTV) mean(daily traded value from the chosen source)
  zero_volume_rate                  zero_volume_days / observations
  plus median/min/max daily volume and median daily traded value

Position liquidity (each symbol evaluated separately; not a portfolio)
  position_to_adtv            = position_value / ADTV
  daily_executable_value      = ADTV * participation_rate
  estimated_liquidation_days  = position_value / daily_executable_value
  The participation rate (default 10%) is an ASSUMPTION about how much of the
  typical daily traded value the investor would trade, not a market rule, and
  the result is a simplified estimate, not an execution guarantee. ADTV of zero
  or unavailable gives NaN (never infinity).

No liquidity score or HIGH/LOW classification is produced: any thresholds would
be arbitrary at this stage. Values are full precision.
"""

import math

import numpy as np
import pandas as pd

from app.analytics.common import require_canonical_columns

ACTUAL_TURNOVER = "ACTUAL_TURNOVER"
ESTIMATED_TRADED_VALUE = "ESTIMATED_TRADED_VALUE"
NO_DATA = "NO_DATA"
DEFAULT_PARTICIPATION_RATE = 0.10
_INVALID = "INVALID"

SUMMARY_COLUMNS = ["symbol", "observations", "start_date", "end_date",
                   "average_daily_volume", "median_daily_volume", "min_daily_volume",
                   "max_daily_volume", "average_daily_traded_value", "median_daily_traded_value",
                   "traded_value_source", "actual_turnover_days", "zero_volume_days",
                   "zero_volume_rate"]
POSITION_COLUMNS = ["symbol", "observations", "position_value", "participation_rate",
                    "average_daily_traded_value", "traded_value_source", "position_to_adtv",
                    "daily_executable_value", "estimated_liquidation_days"]


def calculate_liquidity_summary(data):
    """Stock-level liquidity metrics, one row per symbol (sorted). Columns: SUMMARY_COLUMNS."""
    rows = [_symbol_liquidity(symbol, group)
            for symbol, group in _usable_rows(data).groupby("symbol", sort=True)]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary.astype({"observations": "int64", "actual_turnover_days": "int64",
                           "zero_volume_days": "int64"})


def calculate_position_liquidity(data, position_value,
                                 participation_rate=DEFAULT_PARTICIPATION_RATE):
    """Liquidation estimate for a position of ``position_value`` in each symbol.

    Columns: POSITION_COLUMNS. See the module docstring for the formulas.
    """
    validate_position_value(position_value)
    validate_participation_rate(participation_rate)
    summary = calculate_liquidity_summary(data)

    adtv = summary["average_daily_traded_value"]
    usable_adtv = adtv.where(adtv > 0)                      # 0 / NaN -> NaN, never infinity
    daily_executable = usable_adtv * participation_rate
    result = summary[["symbol", "observations", "average_daily_traded_value",
                      "traded_value_source"]].copy()
    result["position_value"] = float(position_value)
    result["participation_rate"] = float(participation_rate)
    result["position_to_adtv"] = position_value / usable_adtv
    result["daily_executable_value"] = daily_executable
    result["estimated_liquidation_days"] = position_value / daily_executable
    return result[POSITION_COLUMNS]


def validate_position_value(position_value):
    if (isinstance(position_value, bool) or not isinstance(position_value, (int, float))
            or not math.isfinite(position_value) or not position_value > 0):
        raise ValueError(f"position_value must be a positive amount, got {position_value!r}")


def validate_participation_rate(participation_rate):
    if (isinstance(participation_rate, bool) or not isinstance(participation_rate, (int, float))
            or not math.isfinite(participation_rate) or not 0 < participation_rate <= 1):
        raise ValueError("participation_rate must be a fraction above 0 and at most 1, "
                         f"got {participation_rate!r}")


def _symbol_liquidity(symbol, group):
    volume = group["volume"]
    has_turnover = group["turnover"].notna() & np.isfinite(group["turnover"]) & (group["turnover"] >= 0)
    actual_days = int(has_turnover.sum())
    if actual_days == len(group):
        source, traded_value = ACTUAL_TURNOVER, group["turnover"]
    else:
        source, traded_value = ESTIMATED_TRADED_VALUE, group["close"] * volume
    zero_days = int((volume == 0).sum())
    return {
        "symbol": symbol,
        "observations": len(group),
        "start_date": group["date"].min(),
        "end_date": group["date"].max(),
        "average_daily_volume": volume.mean(),
        "median_daily_volume": volume.median(),
        "min_daily_volume": volume.min(),
        "max_daily_volume": volume.max(),
        "average_daily_traded_value": traded_value.mean(),
        "median_daily_traded_value": traded_value.median(),
        "traded_value_source": source,
        "actual_turnover_days": actual_days,
        "zero_volume_days": zero_days,
        "zero_volume_rate": zero_days / len(group),
    }


def _usable_rows(data):
    require_canonical_columns(data, "liquidity")
    if "volume" not in data.columns:
        raise ValueError("Cannot calculate liquidity, missing column(s): volume")

    rows = data[["date", "symbol", "close", "volume"]].copy()
    rows["close"] = rows["close"].astype("float64")
    rows["volume"] = rows["volume"].astype("float64")
    rows["turnover"] = (data["turnover"].astype("float64") if "turnover" in data.columns
                        else np.nan)

    usable = (rows["date"].notna() & rows["symbol"].notna()
              & np.isfinite(rows["close"]) & (rows["close"] > 0)
              & np.isfinite(rows["volume"]) & (rows["volume"] >= 0))
    if "validation_status" in data.columns:
        usable &= data["validation_status"].astype("string").ne(_INVALID).fillna(True)
    rows = rows[usable].sort_values(["symbol", "date"], kind="stable")

    duplicated = rows.duplicated(subset=["symbol", "date"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Cannot calculate liquidity: duplicate symbol/date rows "
            f"({', '.join(sorted(set(rows.loc[duplicated, 'symbol'].astype(str))))}). "
            "Validate the data so duplicates are marked INVALID.")
    return rows
