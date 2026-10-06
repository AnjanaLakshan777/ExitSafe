"""Drawdown and maximum drawdown per symbol, from closing prices.

running_peak(t) = max(close_1 ... close_t)          per symbol, chronological
drawdown(t)     = close_t / running_peak(t) - 1      (0 at a peak, negative below it)
maximum drawdown = min over t of drawdown(t)          (a loss measure: <= 0)

Each symbol is calculated independently. Only usable rows count: rows marked
INVALID are left out (WARNING rows are kept), as are rows without a date,
symbol or a finite positive close. Nothing is filled in: a missing day is
simply absent, and an excluded row is neither a peak nor a trough.

Maximum drawdown event (chronological, never min/max taken independently)
  trough   date of the lowest drawdown (the earliest one if tied)
  peak     the last date at or before the trough on which the close equalled
           the running peak at the trough, i.e. where that decline started
  recovery first date after the trough with close >= the peak price; NaT if
           the price has not recovered by the end of the data

Fewer than 2 usable prices gives NaN (a drawdown needs a price to fall from).
With no decline at all the maximum drawdown is 0 and there is no event, so the
peak/trough/recovery fields are empty. Values are full precision.
"""

import numpy as np
import pandas as pd

from app.analytics.common import require_canonical_columns

SERIES_COLUMNS = ["date", "symbol", "close", "running_peak", "drawdown"]
SUMMARY_COLUMNS = ["symbol", "observations", "maximum_drawdown", "peak_date", "peak_price",
                   "trough_date", "trough_price", "recovery_date"]
_INVALID = "INVALID"


def calculate_drawdown_series(data):
    """Per-row running peak and drawdown, sorted by symbol and date.

    Output columns: date, symbol, close, running_peak, drawdown. Only usable
    rows appear (see module docstring). The input is not modified.
    """
    prices = _usable_prices(data, "drawdown")
    peaks = prices.groupby("symbol", sort=False)["close"].cummax()
    prices["running_peak"] = peaks
    prices["drawdown"] = prices["close"] / peaks - 1
    return prices[SERIES_COLUMNS].reset_index(drop=True)


def calculate_maximum_drawdown(data):
    """One row per symbol (sorted) describing its maximum drawdown event.

    Columns: symbol, observations (usable prices), maximum_drawdown (<= 0, NaN
    if fewer than 2 prices), peak_date, peak_price, trough_date, trough_price,
    recovery_date (NaT if not recovered or if there was no decline).
    """
    series = calculate_drawdown_series(data)
    rows = [_maximum_drawdown_event(symbol, group.reset_index(drop=True))
            for symbol, group in series.groupby("symbol", sort=True)]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for column in ("peak_date", "trough_date", "recovery_date"):
        summary[column] = pd.to_datetime(summary[column])
    for column in ("maximum_drawdown", "peak_price", "trough_price"):
        summary[column] = summary[column].astype("float64")
    summary["observations"] = summary["observations"].astype("int64")
    return summary


def _maximum_drawdown_event(symbol, group):
    row = {"symbol": symbol, "observations": len(group), "maximum_drawdown": np.nan,
           "peak_date": pd.NaT, "peak_price": np.nan, "trough_date": pd.NaT,
           "trough_price": np.nan, "recovery_date": pd.NaT}
    if len(group) < 2:
        return row

    trough = int(group["drawdown"].idxmin())            # earliest minimum
    row["maximum_drawdown"] = group.loc[trough, "drawdown"]
    if row["maximum_drawdown"] >= 0:
        return row                                       # no decline: no event

    peak_price = group.loc[trough, "running_peak"]
    peak = int(group.index[(group.index <= trough) & (group["close"] == peak_price)].max())
    recovered = group.index[(group.index > trough) & (group["close"] >= peak_price)]
    row.update(peak_date=group.loc[peak, "date"], peak_price=peak_price,
               trough_date=group.loc[trough, "date"], trough_price=group.loc[trough, "close"],
               recovery_date=group.loc[recovered.min(), "date"] if len(recovered) else pd.NaT)
    return row


def _usable_prices(data, purpose):
    require_canonical_columns(data, purpose)
    columns = ["date", "symbol", "close"]
    prices = data[columns].copy()
    prices["close"] = prices["close"].astype("float64")

    usable = (prices["date"].notna() & prices["symbol"].notna()
              & np.isfinite(prices["close"]) & (prices["close"] > 0))
    if "validation_status" in data.columns:
        usable &= data["validation_status"].astype("string").ne(_INVALID).fillna(True)
    prices = prices[usable].sort_values(["symbol", "date"], kind="stable")

    duplicated = prices.duplicated(subset=["symbol", "date"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Cannot calculate drawdown: duplicate symbol/date rows "
            f"({', '.join(sorted(set(prices.loc[duplicated, 'symbol'].astype(str))))}). "
            "Validate the data so duplicates are marked INVALID.")
    return prices
