"""Tests for app.regime.regime_detector (rule-based market regime, version 1).

Expected values never come from the module under test. ``reference`` below is
an independent plain-Python implementation of the documented indicators and
decision tree (statistics.stdev, max, mean over explicit windows), used to check
whole regime series; boundary tests feed hand-made indicator rows to
classify_market_regime with values exactly at, just below and just above each
configurable threshold.
"""

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from app.regime.regime_detector import (
    DOWNTREND,
    HIGH_VOLATILITY,
    HISTORY_COLUMNS,
    NEUTRAL,
    NORMAL,
    RECOVERY,
    STRESS,
    UNDEFINED,
    UPTREND,
    RegimeThresholds,
    calculate_regime_indicators,
    classify_market_regime,
    detect_market_regime,
    required_observations,
)

T0 = RegimeThresholds()
W, B, TW, L = 20, 100, 50, 60
REQUIRED = B + 1 + T0.recovery_lookback          # 141 with the defaults

CALM = [0.004, -0.003]
SEQUENCE = (CALM * 90 + [0.025, -0.024] * 8                                  # normal, volatile
            + [-0.04, 0.01, -0.05, -0.03, 0.015, -0.045, -0.02, -0.03]       # sharp fall
            + [0.006, 0.002, 0.007, 0.003] * 10 + [0.003, -0.002] * 40)      # rebound, calm


def closes_from(returns, start=1000.0):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return closes


def index_frame(returns, name="ASPI", dates=None, status=None, start=1000.0):
    closes = closes_from(returns, start)
    frame = pd.DataFrame({
        "date": dates if dates is not None else pd.bdate_range("2023-01-02", periods=len(closes)),
        "index_name": name, "close": closes})
    if status is not None:
        frame["validation_status"] = status
    return frame


def reference(frame, t=T0, w=W, b=B, tw=TW, lb=L):
    """Independent implementation of the documented rules: list of row dicts."""
    rows = frame.sort_values("date").to_dict("records")
    usable_rows, prev_usable, prev_close = [], False, None
    for row in rows:
        usable = (row.get("validation_status", "VALID") != "INVALID"
                  and math.isfinite(row["close"]) and row["close"] > 0)
        if usable:
            r = row["close"] / prev_close - 1 if prev_usable else None
            usable_rows.append({"date": row["date"], "close": row["close"], "r": r})
        prev_usable, prev_close = usable, row["close"]

    out, defined = [], []
    for k, row in enumerate(usable_rows):
        if row["r"] is not None:
            defined.append(row["r"])
        vol = statistics.stdev(defined[-w:]) if len(defined) >= w else None
        base = statistics.stdev(defined[-b:]) if len(defined) >= b else None
        closes = [x["close"] for x in usable_rows[:k + 1]]
        peak = max(closes[-lb:]) if len(closes) >= lb else None
        ma = statistics.mean(closes[-tw:]) if len(closes) >= tw else None
        flat = vol is not None and base is not None and base <= 1e-12
        ratio = vol / base if vol is not None and base is not None and not flat else None
        out.append({"date": row["date"], "close": row["close"], "r": row["r"], "vol": vol,
                    "base": base, "ratio": ratio, "flat": flat, "peak": peak,
                    "dd": row["close"] / peak - 1 if peak else None,
                    "ma": ma, "tr": row["close"] / ma if ma else None})

    for k, row in enumerate(out):
        tr = row["tr"]
        row["trend"] = (UNDEFINED if tr is None else UPTREND if tr > 1 + t.neutral_band
                        else DOWNTREND if tr < 1 - t.neutral_band else NEUTRAL)
        if row["flat"]:
            row["vstate"] = "NORMAL"
        elif row["ratio"] is None:
            row["vstate"] = UNDEFINED
        else:
            q = row["ratio"]
            row["vstate"] = ("NORMAL" if q <= t.elevated_volatility_ratio else
                             "ELEVATED" if q <= t.high_volatility_ratio else "HIGH")
        previous = out[max(0, k - t.recovery_lookback):k]
        known = [0.0 if p["flat"] else p["ratio"] for p in previous]
        peak_ratio = (max(known) if len(previous) == t.recovery_lookback
                      and all(v is not None for v in known) else None)
        if UNDEFINED in (row["trend"], row["vstate"]) or row["dd"] is None or peak_ratio is None:
            regime = UNDEFINED
        elif row["vstate"] == "HIGH" and (row["dd"] <= t.stress_drawdown
                                          or tr <= t.strong_downtrend_ratio):
            regime = STRESS
        elif (peak_ratio > t.elevated_volatility_ratio and row["vstate"] != "HIGH"
              and row["ratio"] is not None and row["ratio"] < peak_ratio
              and row["trend"] != DOWNTREND and row["dd"] <= t.recovery_drawdown):
            regime = RECOVERY
        elif row["vstate"] in ("ELEVATED", "HIGH"):
            regime = HIGH_VOLATILITY
        else:
            regime = NORMAL
        row["regime"] = regime
    return out


def runs(regimes):
    collapsed = []
    for r in regimes:
        if not collapsed or collapsed[-1] != r:
            collapsed.append(r)
    return collapsed


def assert_matches_reference(result, ref):
    h = result.history
    assert len(h) == len(ref)
    assert list(h["regime"]) == [r["regime"] for r in ref]
    assert list(h["trend_state"]) == [r["trend"] for r in ref]
    for column, key in (("rolling_volatility", "vol"), ("volatility_baseline", "base"),
                        ("volatility_ratio", "ratio"), ("running_peak", "peak"),
                        ("current_drawdown", "dd"), ("moving_average", "ma"),
                        ("trend_ratio", "tr"), ("daily_return", "r")):
        expected = [np.nan if r[key] is None else r[key] for r in ref]
        assert h[column].tolist() == pytest.approx(expected, rel=1e-9, abs=1e-15, nan_ok=True)


# Important regime test: NORMAL -> HIGH_VOLATILITY -> STRESS -> RECOVERY -----------------------

def test_regime_sequence_normal_high_volatility_stress_recovery():
    data = index_frame(SEQUENCE)
    result = detect_market_regime(data)
    ref = reference(data)
    assert_matches_reference(result, ref)
    regimes = list(result.history["regime"])
    assert runs(regimes) == [UNDEFINED, NORMAL, HIGH_VOLATILITY, STRESS, HIGH_VOLATILITY,
                             NORMAL, RECOVERY, NORMAL]
    first = {r: regimes.index(r) for r in (NORMAL, HIGH_VOLATILITY, STRESS, RECOVERY)}
    assert first[NORMAL] < first[HIGH_VOLATILITY] < first[STRESS] < first[RECOVERY]
    # each phase is classified where it is expected
    assert regimes[179] == NORMAL                      # end of the calm phase
    assert regimes[196] == HIGH_VOLATILITY             # the volatile spell, no big fall
    assert regimes[205] == STRESS                      # after the sharp fall
    assert regimes[235] == RECOVERY                    # rebound, still well below the peak
    assert regimes[-1] == NORMAL
    stress = result.history.iloc[205]
    assert stress["volatility_state"] == "HIGH" and stress["current_drawdown"] <= -0.10


# 1-7. market shapes --------------------------------------------------------------------------

def test_normal_market():
    data = index_frame(CALM * 100)
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    defined = result.history[result.history["regime"] != UNDEFINED]
    assert set(defined["regime"]) == {NORMAL}
    assert result.current.regime == NORMAL and result.current.volatility_state == "NORMAL"


def test_high_volatility_market():
    data = index_frame(CALM * 80 + [0.02, -0.019] * 6)
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    assert result.current.regime == HIGH_VOLATILITY
    assert result.current.current_drawdown > T0.stress_drawdown       # not a severe fall


def test_strong_market_decline_is_stress():
    data = index_frame(CALM * 80 + [-0.03, -0.04, 0.01, -0.05, -0.03, -0.02])
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    current = result.current
    assert current.regime == STRESS and current.volatility_state == "HIGH"
    assert current.current_drawdown <= -0.10 and current.trend_state == DOWNTREND


def test_recovery_after_drawdown():
    data = index_frame(SEQUENCE[:180 + 16 + 8 + 30])
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    current = result.current
    assert current.regime == RECOVERY
    assert current.current_drawdown <= T0.recovery_drawdown
    assert current.trend_state != DOWNTREND and current.volatility_state != "HIGH"


def test_stable_upward_trend():
    data = index_frame([0.006, -0.002] * 100)
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    defined = result.history[result.history["regime"] != UNDEFINED]
    assert set(defined["regime"]) == {NORMAL} and set(defined["trend_state"]) == {UPTREND}


def test_stable_downward_trend_is_normal_with_downtrend():
    data = index_frame([0.002, -0.006] * 100)
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    defined = result.history[result.history["regime"] != UNDEFINED]
    assert set(defined["regime"]) == {NORMAL} and set(defined["trend_state"]) == {DOWNTREND}
    assert result.current.current_drawdown < -0.10      # a deep but orderly decline


def test_flat_market():
    data = index_frame([0.0] * 200)
    result = detect_market_regime(data)
    current = result.current
    assert current.regime == NORMAL and current.trend_state == NEUTRAL
    assert current.rolling_volatility == 0 and math.isnan(current.volatility_ratio)
    assert current.volatility_state == "NORMAL" and current.current_drawdown == 0


# 8. warm-up -------------------------------------------------------------------------------------

def test_insufficient_warm_up_data_is_not_a_false_normal():
    assert required_observations() == REQUIRED == 141
    short = detect_market_regime(index_frame(CALM * 50))            # 101 observations
    assert set(short.history["regime"]) == {UNDEFINED}
    assert short.current.regime == UNDEFINED and short.first_classified_date is None
    assert short.warm_up_rows == 101 and short.required_observations == 141
    assert "Warm-up" in short.current.regime_reason


def test_first_regime_appears_exactly_after_required_observations():
    returns = CALM * 70                                             # 141 observations
    exact = detect_market_regime(index_frame(returns))
    assert exact.warm_up_rows == 140 and exact.current.regime == NORMAL
    one_less = detect_market_regime(index_frame(returns[:-1]))
    assert one_less.current.regime == UNDEFINED
    # each rolling value is NaN (never 0) until its own window is full
    h = exact.history
    assert h["rolling_volatility"].isna().sum() == W
    assert h["volatility_baseline"].isna().sum() == B
    assert h["moving_average"].isna().sum() == TW - 1
    assert h["running_peak"].isna().sum() == L - 1
    assert required_observations(10, 30, 80, 20, 5) == 80


# 9-12. data handling ----------------------------------------------------------------------------

def test_missing_dates_use_the_previous_available_close():
    data = index_frame(SEQUENCE[:220]).drop(index=[30, 31, 150]).reset_index(drop=True)
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    assert result.observations == 218 and result.excluded_rows == 0


def test_invalid_rows_are_excluded_and_never_bridged():
    status = ["VALID"] * 221
    status[100] = "INVALID"
    data = index_frame(SEQUENCE[:220], status=status)
    result = detect_market_regime(data)
    ref = reference(data)
    assert_matches_reference(result, ref)
    h = result.history
    assert result.excluded_rows == 1 and len(h) == 220
    invalid_date = data.loc[100, "date"]
    assert invalid_date not in set(h["date"])
    after = h.index[h["date"] == data.loc[101, "date"]][0]
    assert math.isnan(h.loc[after, "daily_return"])                 # no return across the gap
    # the volatility window skips the missing return instead of filling it with 0
    returns_up_to = [r["r"] for r in ref[:after + 1] if r["r"] is not None]
    assert h.loc[after, "rolling_volatility"] == pytest.approx(statistics.stdev(returns_up_to[-W:]))


def test_unusable_closes_are_treated_as_invalid():
    data = index_frame(SEQUENCE[:200])
    data.loc[50, "close"] = np.nan
    data.loc[60, "close"] = -1.0
    result = detect_market_regime(data)
    assert result.excluded_rows == 2
    marked = data.assign(validation_status=["INVALID" if i in (50, 60) else "VALID"
                                            for i in range(len(data))])
    assert_matches_reference(result, reference(marked))


def test_warning_rows_are_usable():
    data = index_frame(SEQUENCE[:200], status=["WARNING"] * 201)
    result = detect_market_regime(data)
    assert result.excluded_rows == 0 and result.observations == 201
    assert_matches_reference(result, reference(data))


def test_unsorted_data_gives_the_same_result():
    data = index_frame(SEQUENCE)
    shuffled = data.sample(frac=1, random_state=4).reset_index(drop=True)
    pd.testing.assert_frame_equal(detect_market_regime(shuffled).history,
                                  detect_market_regime(data).history)


def test_string_dates_are_parsed_as_iso():
    data = index_frame(CALM * 75)
    as_text = data.assign(date=data["date"].dt.strftime("%Y-%m-%d"))
    pd.testing.assert_frame_equal(detect_market_regime(as_text).history,
                                  detect_market_regime(data).history)


# 13-15. invalid input -----------------------------------------------------------------------------

def test_empty_input_fails():
    with pytest.raises(ValueError, match="empty"):
        detect_market_regime(pd.DataFrame(columns=["date", "index_name", "close"]))
    with pytest.raises(ValueError, match="DataFrame"):
        detect_market_regime([])


def test_no_usable_rows_fails():
    data = index_frame(CALM * 5, status=["INVALID"] * 11)
    with pytest.raises(ValueError, match="no usable rows"):
        detect_market_regime(data)


@pytest.mark.parametrize("column", ["date", "index_name", "close"])
def test_missing_index_columns_fail(column):
    with pytest.raises(ValueError, match=f"missing column.*{column}"):
        detect_market_regime(index_frame(CALM * 5).drop(columns=[column]))


def test_duplicate_date_rows_fail_unless_marked_invalid():
    data = index_frame(CALM * 80)
    duplicated = pd.concat([data, data.iloc[[30]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate date rows for index ASPI"):
        detect_market_regime(duplicated)
    status = ["VALID"] * len(duplicated)
    status[30] = status[-1] = "INVALID"
    result = detect_market_regime(duplicated.assign(validation_status=status))
    assert result.excluded_rows == 2 and result.observations == len(data) - 1


# 16-19. parameter validation ----------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"volatility_window": 1}, {"volatility_window": 0}, {"volatility_window": 2.5},
    {"volatility_window": True}, {"volatility_window": 100}, {"baseline_window": 20},
    {"baseline_window": None},
])
def test_invalid_rolling_window(kwargs):
    with pytest.raises(ValueError, match="window"):
        detect_market_regime(index_frame(CALM * 5), **kwargs)


@pytest.mark.parametrize("bad", [1, 0, -5, 10.0, "50", None])
def test_invalid_trend_window(bad):
    with pytest.raises(ValueError, match="trend_window"):
        detect_market_regime(index_frame(CALM * 5), trend_window=bad)


@pytest.mark.parametrize("bad", [1, 0, 3.5])
def test_invalid_drawdown_lookback(bad):
    with pytest.raises(ValueError, match="drawdown_lookback"):
        detect_market_regime(index_frame(CALM * 5), drawdown_lookback=bad)


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5, float("nan"), float("inf"), "0.02", True])
def test_invalid_neutral_band(bad):
    with pytest.raises(ValueError, match="neutral_band"):
        RegimeThresholds(neutral_band=bad)


@pytest.mark.parametrize("kwargs, message", [
    ({"elevated_volatility_ratio": 0}, "elevated_volatility_ratio must be positive"),
    ({"elevated_volatility_ratio": -1}, "elevated_volatility_ratio must be positive"),
    ({"high_volatility_ratio": 1.25}, "above elevated"),
    ({"high_volatility_ratio": 1.0}, "above elevated"),
    ({"elevated_volatility_ratio": float("nan")}, "finite"),
    ({"high_volatility_ratio": float("inf")}, "finite"),
    ({"stress_drawdown": 0.1}, "stress_drawdown"),
    ({"stress_drawdown": -1.0}, "stress_drawdown"),
    ({"strong_downtrend_ratio": 0.99}, "strong_downtrend_ratio"),
    ({"strong_downtrend_ratio": 0}, "strong_downtrend_ratio"),
    ({"recovery_lookback": 0}, "recovery_lookback"),
    ({"recovery_drawdown": 0}, "recovery_drawdown"),
])
def test_invalid_volatility_and_regime_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RegimeThresholds(**kwargs)


def test_thresholds_must_be_a_regime_thresholds():
    with pytest.raises(ValueError, match="RegimeThresholds"):
        detect_market_regime(index_frame(CALM * 5), thresholds={"neutral_band": 0.02})


# 20-22. long history, priority, determinism ----------------------------------------------------

def test_multiple_years_of_data():
    rng = np.random.RandomState(7)
    returns = list(np.concatenate([rng.normal(0.0004, 0.006, 300), rng.normal(0, 0.02, 30),
                                   rng.normal(-0.01, 0.02, 20), rng.normal(0.003, 0.006, 150),
                                   rng.normal(0.0003, 0.006, 280)]))
    data = index_frame(returns)
    result = detect_market_regime(data)
    assert_matches_reference(result, reference(data))
    assert (result.history["date"].iloc[-1] - result.history["date"].iloc[0]).days > 2 * 365
    assert result.observations == 781 and result.warm_up_rows == 140
    assert {NORMAL, HIGH_VOLATILITY, STRESS} <= set(result.history["regime"])


def indicator_rows(last, history_ratio=1.0, lookback=T0.recovery_lookback):
    """Hand-made indicator rows: ``lookback`` history rows, then the row under test."""
    rows = [{"rolling_volatility": 0.01 * history_ratio, "volatility_baseline": 0.01,
             "volatility_ratio": history_ratio, "current_drawdown": -0.01,
             "trend_ratio": 1.0}] * lookback
    base = {"rolling_volatility": 0.01, "volatility_baseline": 0.01, "volatility_ratio": 1.0,
            "current_drawdown": -0.01, "trend_ratio": 1.0}
    last = {**base, **last}
    if "volatility_ratio" in last:
        last["rolling_volatility"] = 0.01 * last["volatility_ratio"]
    return pd.DataFrame(rows + [last])


def regime_of(last, history_ratio=1.0, thresholds=T0, lookback=None):
    frame = indicator_rows(last, history_ratio, lookback or thresholds.recovery_lookback)
    return classify_market_regime(frame, thresholds).iloc[-1]


def test_regime_priority():
    # STRESS beats everything that could also apply (elevated history, uptrend-like values)
    stress = regime_of({"volatility_ratio": 2.0, "current_drawdown": -0.2, "trend_ratio": 1.05},
                       history_ratio=3.0)
    assert stress["regime"] == STRESS
    # RECOVERY beats HIGH_VOLATILITY: elevated (1.4) but falling from 2.0, not declining
    recovery = regime_of({"volatility_ratio": 1.4, "current_drawdown": -0.08,
                          "trend_ratio": 1.0}, history_ratio=2.0)
    assert recovery["volatility_state"] == "ELEVATED" and recovery["regime"] == RECOVERY
    # without the falling-volatility history the same row is HIGH_VOLATILITY
    assert regime_of({"volatility_ratio": 1.4, "current_drawdown": -0.08},
                     history_ratio=1.0)["regime"] == HIGH_VOLATILITY
    # HIGH volatility with a mild drawdown and no strong downtrend: HIGH_VOLATILITY, not STRESS
    assert regime_of({"volatility_ratio": 2.0, "current_drawdown": -0.05,
                      "trend_ratio": 0.97})["regime"] == HIGH_VOLATILITY
    # STRESS through a strong downtrend alone (drawdown above the stress level)
    assert regime_of({"volatility_ratio": 2.0, "current_drawdown": -0.05,
                      "trend_ratio": 0.94})["regime"] == STRESS
    # recovery needs volatility that is no longer HIGH (so it never overlaps STRESS)
    assert regime_of({"volatility_ratio": 1.6, "current_drawdown": -0.05, "trend_ratio": 1.0},
                     history_ratio=3.0)["regime"] == HIGH_VOLATILITY


def test_deterministic_classification():
    data = index_frame(SEQUENCE)
    first, second = detect_market_regime(data), detect_market_regime(data)
    pd.testing.assert_frame_equal(first.history, second.history)
    assert first.current == second.current


# Threshold transitions: both sides of every configurable threshold -----------------------------

def test_trend_neutral_band_both_sides():
    band = T0.neutral_band
    states = classify_market_regime(pd.DataFrame({
        "rolling_volatility": 0.01, "volatility_baseline": 0.01, "volatility_ratio": 1.0,
        "current_drawdown": 0.0,
        "trend_ratio": [1 + band + 1e-9, 1 + band, 1 + band - 1e-9, 1.0, 1 - band + 1e-9,
                        1 - band, 1 - band - 1e-9]}))["trend_state"]
    assert list(states) == [UPTREND, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, DOWNTREND]


def test_neutral_band_absorbs_small_fluctuations():
    wiggle = [1.0, 1.015, 0.985, 1.019, 0.981, 1.0, 1.0199, 0.9801]
    frame = pd.DataFrame({"rolling_volatility": 0.01, "volatility_baseline": 0.01,
                          "volatility_ratio": 1.0, "current_drawdown": -0.01,
                          "trend_ratio": wiggle})
    assert set(classify_market_regime(frame)["trend_state"]) == {NEUTRAL}
    no_band = classify_market_regime(frame, RegimeThresholds(neutral_band=0.0,
                                                             strong_downtrend_ratio=0.95))
    assert {UPTREND, DOWNTREND} <= set(no_band["trend_state"])


def test_volatility_thresholds_both_sides():
    e, h = T0.elevated_volatility_ratio, T0.high_volatility_ratio
    ratios = [e - 1e-9, e, e + 1e-9, h - 1e-9, h, h + 1e-9]
    frame = pd.DataFrame({"rolling_volatility": [0.01 * q for q in ratios],
                          "volatility_baseline": 0.01, "volatility_ratio": ratios,
                          "current_drawdown": -0.01, "trend_ratio": 1.0})
    states = list(classify_market_regime(frame)["volatility_state"])
    assert states == ["NORMAL", "NORMAL", "ELEVATED", "ELEVATED", "ELEVATED", "HIGH"]


def test_suggested_ratio_one_threshold_can_be_configured():
    t = RegimeThresholds(elevated_volatility_ratio=1.0)
    assert regime_of({"volatility_ratio": 1.1}, thresholds=t)["regime"] == HIGH_VOLATILITY
    assert regime_of({"volatility_ratio": 1.1})["regime"] == NORMAL       # default 1.25


def test_stress_drawdown_both_sides():
    d = T0.stress_drawdown
    assert regime_of({"volatility_ratio": 2.0, "current_drawdown": d})["regime"] == STRESS
    assert regime_of({"volatility_ratio": 2.0, "current_drawdown": d - 1e-9})["regime"] == STRESS
    assert regime_of({"volatility_ratio": 2.0,
                      "current_drawdown": d + 1e-9})["regime"] == HIGH_VOLATILITY


def test_strong_downtrend_ratio_both_sides():
    s = T0.strong_downtrend_ratio
    row = {"volatility_ratio": 2.0, "current_drawdown": -0.03}
    assert regime_of({**row, "trend_ratio": s})["regime"] == STRESS
    assert regime_of({**row, "trend_ratio": s - 1e-9})["regime"] == STRESS
    assert regime_of({**row, "trend_ratio": s + 1e-9})["regime"] == HIGH_VOLATILITY


def test_recovery_drawdown_both_sides():
    r = T0.recovery_drawdown
    row = {"volatility_ratio": 0.9, "trend_ratio": 1.0}
    assert regime_of({**row, "current_drawdown": r}, history_ratio=2.0)["regime"] == RECOVERY
    assert regime_of({**row, "current_drawdown": r - 1e-9}, history_ratio=2.0)["regime"] == RECOVERY
    assert regime_of({**row, "current_drawdown": r + 1e-9}, history_ratio=2.0)["regime"] == NORMAL


def test_recovery_requires_recent_elevated_falling_volatility_and_no_downtrend():
    row = {"volatility_ratio": 0.9, "current_drawdown": -0.05, "trend_ratio": 1.0}
    e = T0.elevated_volatility_ratio
    assert regime_of(row, history_ratio=e + 1e-9)["regime"] == RECOVERY
    assert regime_of(row, history_ratio=e)["regime"] == NORMAL            # never elevated
    assert regime_of({**row, "trend_ratio": 0.97}, history_ratio=2.0)["regime"] == NORMAL
    assert regime_of({**row, "volatility_ratio": 2.5},
                     history_ratio=2.0)["regime"] == HIGH_VOLATILITY      # rising, and HIGH


def test_recovery_lookback_both_sides():
    t = RegimeThresholds(recovery_lookback=5)
    calm = {"rolling_volatility": 0.009, "volatility_baseline": 0.01, "volatility_ratio": 0.9,
            "current_drawdown": -0.05, "trend_ratio": 1.0}
    spike = {**calm, "rolling_volatility": 0.02, "volatility_ratio": 2.0}
    inside = pd.DataFrame([calm] * 5 + [spike] + [calm] * 4 + [calm])   # spike 5 rows back
    outside = pd.DataFrame([calm] * 5 + [spike] + [calm] * 5 + [calm])  # spike 6 rows back
    assert classify_market_regime(inside, t)["regime"].iloc[-1] == RECOVERY
    assert classify_market_regime(outside, t)["regime"].iloc[-1] == NORMAL


# Index selection, outputs, no mutation ----------------------------------------------------------

def test_index_selection_and_errors():
    data = pd.concat([index_frame(CALM * 75, "ASPI"),
                      index_frame([0.01, -0.01] * 75, "S&P SL20", start=500.0)],
                     ignore_index=True)
    with pytest.raises(ValueError, match="choose one with index_name.*ASPI, S&P SL20"):
        detect_market_regime(data)
    with pytest.raises(ValueError, match="Index CSE ALL not found.*available: ASPI, S&P SL20"):
        detect_market_regime(data, "CSE ALL")
    sl20 = detect_market_regime(data, " s&p sl20 ")
    assert sl20.index_name == "S&P SL20" and sl20.current.current_close == pytest.approx(
        closes_from([0.01, -0.01] * 75, 500.0)[-1])
    aspi = detect_market_regime(data, "ASPI")
    assert aspi.current.current_close == pytest.approx(closes_from(CALM * 75)[-1])
    with pytest.raises(ValueError, match="non-empty"):
        detect_market_regime(data, "  ")


def test_result_fields_and_history_columns():
    result = detect_market_regime(index_frame(SEQUENCE))
    current, last = result.current, result.history.iloc[-1]
    assert set(HISTORY_COLUMNS) <= set(result.history.columns)
    assert current.date == last["date"] and current.current_close == last["close"]
    assert current.rolling_volatility == last["rolling_volatility"]   # daily, not annualized
    assert current.running_peak == pytest.approx(max(closes_from(SEQUENCE)[-L:]))
    assert current.moving_average == pytest.approx(statistics.mean(closes_from(SEQUENCE)[-TW:]))
    assert result.first_classified_date == result.history["date"].iloc[REQUIRED - 1]
    indicators = calculate_regime_indicators(index_frame(SEQUENCE))
    assert "regime" not in indicators.columns and len(indicators) == len(result.history)


def test_custom_windows_match_reference():
    data = index_frame(SEQUENCE)
    t = RegimeThresholds(neutral_band=0.01, recovery_lookback=10, stress_drawdown=-0.08)
    result = detect_market_regime(data, volatility_window=10, baseline_window=60,
                                  trend_window=30, drawdown_lookback=40, thresholds=t)
    assert result.required_observations == 71
    assert_matches_reference(result, reference(data, t, 10, 60, 30, 40))


def test_input_is_not_mutated():
    data = index_frame(SEQUENCE, status=["VALID"] * (len(SEQUENCE) + 1))
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    before = data.copy(deep=True)
    detect_market_regime(data)
    pd.testing.assert_frame_equal(data, before)
