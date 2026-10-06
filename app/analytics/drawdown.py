"""Drawdown and maximum drawdown from closing prices, one stock at a time."""

import numpy as np
import pandas as pd

from app.analytics.common import require_canonical_columns

SERIES_COLUMNS = ["date", "symbol", "close", "running_peak", "drawdown"]
SUMMARY_COLUMNS = ["symbol", "observations", "maximum_drawdown", "peak_date", "peak_price",
                   "trough_date", "trough_price", "recovery_date"]
_INVALID = "INVALID"


def calculate_drawdown_series(data):
    """Running peak and drawdown for every usable row."""
    prices = _usable_prices(data, "drawdown")
    peaks = prices.groupby("symbol", sort=False)["close"].cummax()
    prices["running_peak"] = peaks
    prices["drawdown"] = prices["close"] / peaks - 1
    return prices[SERIES_COLUMNS].reset_index(drop=True)


def calculate_maximum_drawdown(data):
    """The worst peak-to-trough fall of each stock, with its dates."""
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
    """Running peak and drawdown for one series of values (prices or a portfolio value)."""
    path = pd.DataFrame({"date": list(dates), "value": np.asarray(values, dtype="float64")})
    path["running_peak"] = path["value"].cummax()
    path["drawdown"] = path["value"] / path["running_peak"] - 1
    return path


def maximum_drawdown_event(path):
    """Find the deepest drawdown in a path and its peak, trough and recovery dates."""
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
