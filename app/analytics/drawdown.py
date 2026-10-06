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
    rows = []
    for symbol, group in series.groupby("symbol", sort=True):
        path = group.rename(columns={"close": "value"}).reset_index(drop=True)
        event = maximum_drawdown_event(path)
        rows.append({"symbol": symbol, "observations": event["observations"],
                     "maximum_drawdown": event["maximum_drawdown"],
                     "peak_date": event["peak_date"], "peak_price": event["peak_value"],
                     "trough_date": event["trough_date"], "trough_price": event["trough_value"],
                     "recovery_date": event["recovery_date"]})
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for column in ("peak_date", "trough_date", "recovery_date"):
        summary[column] = pd.to_datetime(summary[column])
    for column in ("maximum_drawdown", "peak_price", "trough_price"):
        summary[column] = summary[column].astype("float64")
    summary["observations"] = summary["observations"].astype("int64")
    return summary


def drawdown_path(dates, values):
    """Chronological path of one value series: date, value, running_peak, drawdown.

    ``values`` are prices or any value index (e.g. a portfolio value starting at 1.0).
    """
    path = pd.DataFrame({"date": list(dates), "value": np.asarray(values, dtype="float64")})
    path["running_peak"] = path["value"].cummax()
    path["drawdown"] = path["value"] / path["running_peak"] - 1
    return path


def maximum_drawdown_event(path):
    """The maximum drawdown event of a path (columns date, value, running_peak, drawdown,
    in date order, index 0..n-1), using the rules in the module docstring.

    Returns a dict: observations, maximum_drawdown, peak_date, peak_value,
    trough_date, trough_value, recovery_date.
    """
    event = {"observations": len(path), "maximum_drawdown": np.nan, "peak_date": pd.NaT,
             "peak_value": np.nan, "trough_date": pd.NaT, "trough_value": np.nan,
             "recovery_date": pd.NaT}
    if len(path) < 2:
        return event

    trough = int(path["drawdown"].idxmin())             # earliest minimum
    event["maximum_drawdown"] = path.loc[trough, "drawdown"]
    if event["maximum_drawdown"] >= 0:
        return event                                     # no decline: no event

    peak_value = path.loc[trough, "running_peak"]
    peak = int(path.index[(path.index <= trough) & (path["value"] == peak_value)].max())
    recovered = path.index[(path.index > trough) & (path["value"] >= peak_value)]
    event.update(peak_date=path.loc[peak, "date"], peak_value=peak_value,
                 trough_date=path.loc[trough, "date"], trough_value=path.loc[trough, "value"],
                 recovery_date=path.loc[recovered.min(), "date"] if len(recovered) else pd.NaT)
    return event


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
