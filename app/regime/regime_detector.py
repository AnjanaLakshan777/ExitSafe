"""Rule-based market regime (NORMAL, HIGH_VOLATILITY, STRESS, RECOVERY) from an index.

It compares recent volatility with a longer baseline, looks at the drawdown from
the recent peak and the trend against a moving average. The thresholds are
model settings, and the regime describes the past; it doesn't predict prices.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.analytics.common import CONSTANT_RETURN_TOLERANCE
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns

UNDEFINED = "N/A"
NORMAL, HIGH_VOLATILITY, STRESS, RECOVERY = "NORMAL", "HIGH_VOLATILITY", "STRESS", "RECOVERY"
UPTREND, DOWNTREND, NEUTRAL = "UPTREND", "DOWNTREND", "NEUTRAL"
VOLATILITY_NORMAL, VOLATILITY_ELEVATED, VOLATILITY_HIGH = "NORMAL", "ELEVATED", "HIGH"
REGIMES = [NORMAL, HIGH_VOLATILITY, STRESS, RECOVERY]

DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_BASELINE_WINDOW = 100
DEFAULT_TREND_WINDOW = 50
DEFAULT_DRAWDOWN_LOOKBACK = 60

INDICATOR_COLUMNS = ["date", "index_name", "close", "daily_return", "rolling_volatility",
                     "volatility_baseline", "volatility_ratio", "running_peak",
                     "current_drawdown", "moving_average", "trend_ratio"]
CLASSIFICATION_COLUMNS = ["trend_state", "volatility_state", "recent_peak_volatility_ratio",
                          "regime", "regime_reason"]
HISTORY_COLUMNS = ["date", "regime", "rolling_volatility", "current_drawdown", "trend_state"]
_INVALID = "INVALID"


@dataclass(frozen=True)
class RegimeThresholds:
    """Model thresholds (assumptions, not market rules). See the module docstring."""
    neutral_band: float = 0.02
    elevated_volatility_ratio: float = 1.25
    high_volatility_ratio: float = 1.5
    stress_drawdown: float = -0.10
    strong_downtrend_ratio: float = 0.95
    recovery_lookback: int = 40
    recovery_drawdown: float = -0.02

    def __post_init__(self):
        _finite(self.neutral_band, "neutral_band")
        if not 0 <= self.neutral_band < 1:
            raise ValueError(f"neutral_band must be at least 0 and below 1, got {self.neutral_band!r}")
        _finite(self.elevated_volatility_ratio, "elevated_volatility_ratio")
        _finite(self.high_volatility_ratio, "high_volatility_ratio")
        if not self.elevated_volatility_ratio > 0:
            raise ValueError("elevated_volatility_ratio must be positive, got "
                             f"{self.elevated_volatility_ratio!r}")
        if not self.high_volatility_ratio > self.elevated_volatility_ratio:
            raise ValueError("high_volatility_ratio must be above elevated_volatility_ratio, got "
                             f"{self.high_volatility_ratio!r} <= {self.elevated_volatility_ratio!r}")
        _finite(self.stress_drawdown, "stress_drawdown")
        if not -1 < self.stress_drawdown < 0:
            raise ValueError(f"stress_drawdown must be between -1 and 0 (a loss), got "
                             f"{self.stress_drawdown!r}")
        _finite(self.strong_downtrend_ratio, "strong_downtrend_ratio")
        if not 0 < self.strong_downtrend_ratio <= 1 - self.neutral_band:
            raise ValueError("strong_downtrend_ratio must be above 0 and at most "
                             f"1 - neutral_band ({1 - self.neutral_band:g}), got "
                             f"{self.strong_downtrend_ratio!r}")
        _whole(self.recovery_lookback, "recovery_lookback", 1)
        _finite(self.recovery_drawdown, "recovery_drawdown")
        if not -1 < self.recovery_drawdown < 0:
            raise ValueError(f"recovery_drawdown must be between -1 and 0 (a loss), got "
                             f"{self.recovery_drawdown!r}")


@dataclass(frozen=True)
class RegimeSnapshot:
    date: pd.Timestamp
    current_close: float
    daily_return: float
    rolling_volatility: float
    volatility_baseline: float
    volatility_ratio: float
    running_peak: float
    current_drawdown: float
    moving_average: float
    trend_ratio: float
    trend_state: str
    volatility_state: str
    regime: str
    regime_reason: str


@dataclass(frozen=True)
class MarketRegimeResult:
    index_name: str
    current: RegimeSnapshot              # the latest usable date
    history: pd.DataFrame                # INDICATOR_COLUMNS + CLASSIFICATION_COLUMNS, by date
    thresholds: RegimeThresholds
    volatility_window: int
    baseline_window: int
    trend_window: int
    drawdown_lookback: int
    required_observations: int           # observations needed before the first regime
    observations: int                    # usable dates
    excluded_rows: int                   # rows of this index that were not usable
    warm_up_rows: int                    # usable dates with regime N/A
    first_classified_date: pd.Timestamp | None


def required_observations(volatility_window=DEFAULT_VOLATILITY_WINDOW,
                          baseline_window=DEFAULT_BASELINE_WINDOW,
                          trend_window=DEFAULT_TREND_WINDOW,
                          drawdown_lookback=DEFAULT_DRAWDOWN_LOOKBACK,
                          recovery_lookback=RegimeThresholds.recovery_lookback):
    """Usable observations (without gaps) before the first regime can be classified."""
    _validate_windows(volatility_window, baseline_window, trend_window, drawdown_lookback)
    return max(baseline_window + 1 + recovery_lookback, trend_window, drawdown_lookback)


def calculate_regime_indicators(data, index_name=None,
                                volatility_window=DEFAULT_VOLATILITY_WINDOW,
                                baseline_window=DEFAULT_BASELINE_WINDOW,
                                trend_window=DEFAULT_TREND_WINDOW,
                                drawdown_lookback=DEFAULT_DRAWDOWN_LOOKBACK):
    """Indicator table (INDICATOR_COLUMNS) for one index, one row per usable date."""
    return _indicators(data, index_name, volatility_window, baseline_window, trend_window,
                       drawdown_lookback)[0]


def classify_market_regime(indicators, thresholds=None):
    """Add the trend state, volatility state and regime to an indicator table."""
    t = thresholds if thresholds is not None else RegimeThresholds()
    if not isinstance(t, RegimeThresholds):
        raise ValueError("thresholds must be a RegimeThresholds")
    result = indicators.reset_index(drop=True).copy()
    ratio = result["volatility_ratio"].astype("float64")
    rolling = result["rolling_volatility"].astype("float64")
    baseline = result["volatility_baseline"].astype("float64")
    trend_ratio = result["trend_ratio"].astype("float64")
    drawdown = result["current_drawdown"].astype("float64")

    trend = np.select([trend_ratio.isna(), trend_ratio > 1 + t.neutral_band,
                       trend_ratio < 1 - t.neutral_band],
                      [UNDEFINED, UPTREND, DOWNTREND], NEUTRAL)
    flat = baseline.notna() & rolling.notna() & (baseline <= CONSTANT_RETURN_TOLERANCE)
    volatility = np.select(
        [flat, ratio.isna(), ratio <= t.elevated_volatility_ratio,
         ratio <= t.high_volatility_ratio],
        [VOLATILITY_NORMAL, UNDEFINED, VOLATILITY_NORMAL, VOLATILITY_ELEVATED], VOLATILITY_HIGH)
    # For the recovery check a flat day (no variation at all) counts as a ratio of 0.
    known = pd.Series(np.where(flat, 0.0, ratio), index=result.index).where(volatility != UNDEFINED)
    recent_peak = known.shift(1).rolling(t.recovery_lookback,
                                         min_periods=t.recovery_lookback).max()

    regimes, reasons = [], []
    for i in result.index:
        regime, reason = _decide(t, trend[i], volatility[i], ratio[i], recent_peak[i],
                                 drawdown[i], trend_ratio[i])
        regimes.append(regime)
        reasons.append(reason)
    result["trend_state"] = trend
    result["volatility_state"] = volatility
    result["recent_peak_volatility_ratio"] = recent_peak
    result["regime"] = regimes
    result["regime_reason"] = reasons
    return result


def detect_market_regime(data, index_name=None,
                         volatility_window=DEFAULT_VOLATILITY_WINDOW,
                         baseline_window=DEFAULT_BASELINE_WINDOW,
                         trend_window=DEFAULT_TREND_WINDOW,
                         drawdown_lookback=DEFAULT_DRAWDOWN_LOOKBACK,
                         thresholds=None):
    """Historical regime series and the current regime for one index (MarketRegimeResult)."""
    thresholds = thresholds if thresholds is not None else RegimeThresholds()
    if not isinstance(thresholds, RegimeThresholds):
        raise ValueError("thresholds must be a RegimeThresholds")
    indicators, name, excluded = _indicators(data, index_name, volatility_window,
                                             baseline_window, trend_window, drawdown_lookback)
    if indicators.empty:
        raise ValueError(f"Index {name} has no usable rows (a date, a positive close and not "
                         "INVALID)")
    history = classify_market_regime(indicators, thresholds)
    last = history.iloc[-1]
    classified = history.loc[history["regime"] != UNDEFINED, "date"]
    return MarketRegimeResult(
        index_name=name,
        current=RegimeSnapshot(
            date=last["date"], current_close=float(last["close"]),
            **{c: float(last[c]) for c in ("daily_return", "rolling_volatility",
                                           "volatility_baseline", "volatility_ratio",
                                           "running_peak", "current_drawdown",
                                           "moving_average", "trend_ratio")},
            trend_state=last["trend_state"], volatility_state=last["volatility_state"],
            regime=last["regime"], regime_reason=last["regime_reason"]),
        history=history, thresholds=thresholds, volatility_window=volatility_window,
        baseline_window=baseline_window, trend_window=trend_window,
        drawdown_lookback=drawdown_lookback,
        required_observations=required_observations(volatility_window, baseline_window,
                                                    trend_window, drawdown_lookback,
                                                    thresholds.recovery_lookback),
        observations=len(history), excluded_rows=excluded,
        warm_up_rows=int((history["regime"] == UNDEFINED).sum()),
        first_classified_date=classified.iloc[0] if len(classified) else None)


def thresholds_dict(thresholds):
    """Plain dict of a RegimeThresholds (for display)."""
    return asdict(thresholds)


def _decide(t, trend, volatility, ratio, recent_peak, drawdown, trend_ratio):
    if UNDEFINED in (trend, volatility) or math.isnan(drawdown) or math.isnan(recent_peak):
        return UNDEFINED, "Warm-up: not enough history for every indicator yet"
    if volatility == VOLATILITY_HIGH and (drawdown <= t.stress_drawdown
                                          or trend_ratio <= t.strong_downtrend_ratio):
        why = (f"drawdown {drawdown:.1%} <= {t.stress_drawdown:.0%}"
               if drawdown <= t.stress_drawdown else
               f"close / average {trend_ratio:.3f} <= {t.strong_downtrend_ratio:g}")
        return STRESS, f"High volatility (ratio {ratio:.2f} > {t.high_volatility_ratio:g}) and {why}"
    if (recent_peak > t.elevated_volatility_ratio and volatility != VOLATILITY_HIGH
            and not math.isnan(ratio) and ratio < recent_peak and trend != DOWNTREND
            and drawdown <= t.recovery_drawdown):
        return RECOVERY, (f"Volatility ratio {ratio:.2f} has fallen from a recent "
                          f"{recent_peak:.2f}, trend {trend.lower()}, still {drawdown:.1%} "
                          "below the peak")
    if volatility in (VOLATILITY_ELEVATED, VOLATILITY_HIGH):
        return HIGH_VOLATILITY, (f"Volatility ratio {ratio:.2f} > "
                                 f"{t.elevated_volatility_ratio:g} ({volatility.lower()})")
    return NORMAL, "No stress, recovery or elevated-volatility condition"


def _indicators(data, index_name, volatility_window, baseline_window, trend_window,
                drawdown_lookback):
    _validate_windows(volatility_window, baseline_window, trend_window, drawdown_lookback)
    series, name = prepare_index_series(data, index_name)
    usable = series["validation_status"] != _INVALID
    excluded = int((~usable).sum())

    returns = calculate_daily_returns(series.rename(columns={"index_name": "symbol"}))
    rows = returns[returns["validation_status"] != _INVALID].reset_index(drop=True)
    r = rows[CANONICAL_DAILY_RETURN].astype("float64")
    close = rows["close"].astype("float64")

    defined = r.dropna()                     # skip gaps rather than filling them
    rolling = defined.rolling(volatility_window, min_periods=volatility_window).std(ddof=1)
    baseline = defined.rolling(baseline_window, min_periods=baseline_window).std(ddof=1)
    rolling = rolling.reindex(rows.index).ffill()
    baseline = baseline.reindex(rows.index).ffill()
    peak = close.rolling(drawdown_lookback, min_periods=drawdown_lookback).max()
    average = close.rolling(trend_window, min_periods=trend_window).mean()

    out = pd.DataFrame({
        "date": rows["date"], "index_name": name, "close": close, "daily_return": r,
        "rolling_volatility": rolling, "volatility_baseline": baseline,
        "volatility_ratio": rolling / baseline.where(baseline > CONSTANT_RETURN_TOLERANCE),
        "running_peak": peak, "current_drawdown": close / peak - 1,
        "moving_average": average, "trend_ratio": close / average,
    })
    return out[INDICATOR_COLUMNS], name, excluded


def prepare_index_series(data, index_name=None):
    """Pick one index from the data and mark unusable rows INVALID."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Index data must be a DataFrame with date, index_name and close")
    missing = [c for c in ("date", "index_name", "close") if c not in data.columns]
    if missing:
        raise ValueError(f"Index data is missing column(s): {', '.join(missing)}")
    if data.empty:
        raise ValueError("Index data is empty")

    names = data["index_name"].astype("string").str.strip().str.upper()
    available = sorted(names.dropna().unique())
    if index_name is None:
        if len(available) != 1:
            raise ValueError("Several indices in the data; choose one with index_name "
                             f"(available: {', '.join(available) or 'none'})")
        name = available[0]
    else:
        if not isinstance(index_name, str) or not index_name.strip():
            raise ValueError("index_name must be a non-empty text value")
        name = index_name.strip().upper()
        if name not in available:
            raise ValueError(f"Index {name} not found in the index data "
                             f"(available: {', '.join(available) or 'none'})")

    selected = data[names == name]
    dates = selected["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates.astype("string"), format="ISO8601", errors="coerce")
    close = pd.to_numeric(selected["close"], errors="coerce").astype("float64")
    status = (selected["validation_status"].astype("string")
              if "validation_status" in selected.columns
              else pd.Series("VALID", index=selected.index, dtype="string"))
    unusable = dates.isna() | ~np.isfinite(close) | (close <= 0) | status.eq(_INVALID).fillna(False)
    series = pd.DataFrame({"date": dates, "index_name": name, "close": close,
                           "validation_status": status.where(~unusable, _INVALID)})
    usable = series[series["validation_status"] != _INVALID]
    duplicated = usable["date"].duplicated(keep=False)
    if duplicated.any():
        days = sorted({d.date().isoformat() for d in usable.loc[duplicated, "date"]})
        raise ValueError(f"Duplicate date rows for index {name} ({', '.join(days[:5])}"
                         f"{'...' if len(days) > 5 else ''}). Validate the data so duplicates "
                         "are marked INVALID.")
    return series.sort_values("date", kind="stable", na_position="last").reset_index(drop=True), name


def _validate_windows(volatility_window, baseline_window, trend_window, drawdown_lookback):
    _whole(volatility_window, "volatility_window", 2)
    _whole(baseline_window, "baseline_window", 3)
    if not baseline_window > volatility_window:
        raise ValueError(f"baseline_window ({baseline_window}) must be longer than "
                         f"volatility_window ({volatility_window})")
    _whole(trend_window, "trend_window", 2)
    _whole(drawdown_lookback, "drawdown_lookback", 2)


def _whole(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be a whole number >= {minimum}, got {value!r}")


def _finite(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
