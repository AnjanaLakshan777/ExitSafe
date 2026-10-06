"""Tests for app.stress_testing.stress_engine (deterministic scenario stress tests).

Expected values are computed independently of the engine: shocks and weights in
plain arithmetic, standard deviations with statistics.stdev over returns rebuilt
from the generated closes, ADTV with statistics.mean of turnover or close x
volume, and historical VaR / CVaR with small reference implementations.
"""

import math
import statistics
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analytics.liquidity import ACTUAL_TURNOVER, ESTIMATED_TRADED_VALUE
from app.analytics.portfolio_risk import calculate_portfolio_risk_summary
from app.data.loaders.csv_market_loader import load_csv_market_data
from app.stress_testing.stress_engine import (
    BASELINE_HISTORICAL_MEAN,
    COMBINED,
    DEFAULT_SCENARIOS,
    LIQUIDITY,
    LIQUIDITY_NO_DATA,
    LIQUIDITY_OK,
    LIQUIDITY_ZERO_ADTV,
    MARKET,
    SECTOR,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    VOLATILITY,
    StressScenario,
    run_combined_stress_test,
    run_liquidity_stress_test,
    run_market_stress_test,
    run_sector_stress_test,
    run_stress_scenario,
    run_stress_scenarios,
    run_volatility_stress_test,
    scenario_from_components,
)

SAMPLE_3 = Path(__file__).resolve().parents[2] / "data" / "sample" / "sample_market_data.csv"
W3 = {"ABC": 0.40, "XYZ": 0.35, "LMN": 0.25}
SECTORS = {"ABC": "Banking", "XYZ": "Manufacturing", "LMN": "Banking"}

RA = [0.02, -0.01, 0.03, -0.02, 0.01, -0.05, 0.015, -0.01, 0.005, -0.03, 0.012, -0.008,
      0.025, -0.015, 0.004, -0.022, 0.018, -0.006, 0.009, -0.04, 0.011, -0.003, 0.007, 0.02, -0.012]
RB = [-0.01, 0.02, -0.005, 0.01, 0.0, 0.03, -0.02, 0.015, -0.01, 0.02, -0.004, 0.006,
      -0.01, 0.012, 0.003, 0.01, -0.015, 0.008, -0.002, 0.025, -0.006, 0.004, -0.009, -0.01, 0.007]
RC = [0.005, 0.01, -0.02, 0.004, -0.012, 0.008, 0.002, -0.006, 0.015, -0.01, 0.003, 0.0,
      -0.004, 0.009, -0.007, 0.012, 0.001, -0.003, 0.006, -0.015, 0.004, 0.002, -0.001, 0.005, 0.003]


def market(symbol, returns, start=100.0, volume=1_000.0, turnover=None, status=None):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    frame = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=len(closes)),
                          "symbol": symbol, "close": closes,
                          "volume": volume if np.ndim(volume) == 0 else [float(v) for v in volume]})
    if turnover is not None:
        frame["turnover"] = [float(t) if t is not None else np.nan for t in turnover]
    if status is not None:
        frame["validation_status"] = status
    return frame


def combine(*frames):
    return pd.concat(frames, ignore_index=True)


def simple_returns(frame):
    c = frame["close"].tolist()
    return [c[i + 1] / c[i] - 1 for i in range(len(c) - 1)]


FRAMES = {"A": market("A", RA), "B": market("B", RB), "C": market("C", RC)}
DATA = combine(*FRAMES.values())
RET = {s: simple_returns(f) for s, f in FRAMES.items()}
WABC = {"A": 0.5, "B": 0.3, "C": 0.2}


@pytest.fixture(scope="module")
def sample():
    return load_csv_market_data(SAMPLE_3).data


def impacts(result):
    return result.symbol_impacts.set_index("symbol")


# Important test: ABC 40% / XYZ 35% / LMN 25%, market -10%, then Banking -20% --------------------

def test_market_shock_on_sample_portfolio_then_banking(sample):
    market10 = run_market_stress_test(sample, W3, -0.10, portfolio_value=20_000_000)
    assert market10.portfolio_return == pytest.approx(0.40 * -0.10 + 0.35 * -0.10 + 0.25 * -0.10)
    assert market10.portfolio_return == pytest.approx(-0.10)            # same shock throughout
    assert list(impacts(market10)["stressed_return"]) == pytest.approx([-0.10] * 3)

    combined = run_combined_stress_test(sample, W3, market_shock=-0.10, sector_name="Banking",
                                        sector_shock=-0.20, sector_mapping=SECTORS,
                                        portfolio_value=20_000_000)
    rows = impacts(combined)
    assert rows.loc["ABC", "stressed_return"] == pytest.approx(-0.30)   # Banking
    assert rows.loc["LMN", "stressed_return"] == pytest.approx(-0.30)   # Banking
    assert rows.loc["XYZ", "stressed_return"] == pytest.approx(-0.10)   # not Banking
    assert rows.loc["XYZ", "sector_component"] == 0.0
    assert combined.portfolio_return == pytest.approx(0.40 * -0.30 + 0.35 * -0.10 + 0.25 * -0.30)
    assert combined.portfolio_loss_amount == pytest.approx(20_000_000 * 0.23)


# 1-4. market shocks ---------------------------------------------------------------------------

@pytest.mark.parametrize("shock", [-0.10, -0.20, -0.30])
def test_market_shocks(shock):
    result = run_market_stress_test(DATA, WABC, shock)
    assert result.status == STATUS_OK and result.scenario_type == MARKET
    assert result.portfolio_return == pytest.approx(shock, abs=1e-15)
    assert result.portfolio_loss == pytest.approx(-shock, abs=1e-15)
    assert list(impacts(result)["stressed_return"]) == pytest.approx([shock] * 3)
    assert list(impacts(result)["base_return"]) == [0.0, 0.0, 0.0]


def test_default_scenarios_cover_the_brief():
    names = [s.name for s in DEFAULT_SCENARIOS]
    assert names == ["Market -10%", "Market -20%", "Market -30%", "High Volatility 1.5x",
                     "High Volatility 2.0x", "Liquidity -50%"]
    assert [s.market_shock for s in DEFAULT_SCENARIOS[:3]] == [-0.10, -0.20, -0.30]
    assert [s.volatility_multiplier for s in DEFAULT_SCENARIOS[3:5]] == [1.5, 2.0]
    assert DEFAULT_SCENARIOS[5].liquidity_multiplier == 0.5


def test_custom_market_shock_and_gains_are_not_clamped():
    down = run_market_stress_test(DATA, WABC, -0.07, name="Custom -7%")
    assert down.scenario_name == "Custom -7%" and down.portfolio_loss == pytest.approx(0.07)
    up = run_market_stress_test(DATA, WABC, 0.05)
    assert up.portfolio_return == pytest.approx(0.05)
    assert up.portfolio_loss == pytest.approx(-0.05)                 # a negative loss, not 0
    assert up.largest_negative_contributor is None
    assert math.isnan(up.largest_negative_contribution)


def test_historical_mean_baseline():
    result = run_market_stress_test(DATA, WABC, -0.10, baseline=BASELINE_HISTORICAL_MEAN)
    means = {s: statistics.mean(RET[s]) for s in "ABC"}
    rows = impacts(result)
    for s in "ABC":
        assert rows.loc[s, "base_return"] == pytest.approx(means[s], rel=1e-12)
        assert rows.loc[s, "stressed_return"] == pytest.approx(means[s] - 0.10, rel=1e-12)
    assert result.portfolio_return == pytest.approx(
        sum(WABC[s] * (means[s] - 0.10) for s in "ABC"), rel=1e-12)


# 5-6. sector shocks ---------------------------------------------------------------------------

def test_sector_shock():
    mapping = {"A": "Banking", "B": "Plantations", "C": "Banking"}
    result = run_sector_stress_test(DATA, WABC, "Banking", -0.30, sector_mapping=mapping)
    rows = impacts(result)
    assert result.scenario_type == SECTOR
    assert list(rows["stressed_return"]) == pytest.approx([-0.30, 0.0, -0.30])
    assert list(rows["sector"]) == ["Banking", "Plantations", "Banking"]
    assert result.portfolio_return == pytest.approx(0.5 * -0.30 + 0.2 * -0.30)


def test_custom_sector_shock_any_sector_name_case_insensitive():
    mapping = pd.DataFrame({"symbol": ["A", "B", "C"],
                            "sector": ["Telecom", "telecom ", "Hotels"]})
    result = run_sector_stress_test(DATA, WABC, "TELECOM", -0.15, sector_mapping=mapping)
    assert list(impacts(result)["sector_component"]) == pytest.approx([-0.15, -0.15, 0.0])
    assert result.portfolio_return == pytest.approx((0.5 + 0.3) * -0.15)
    none_in_sector = run_sector_stress_test(DATA, WABC, "Banking", -0.2, sector_mapping=mapping)
    assert none_in_sector.portfolio_return == 0.0
    assert "No holding is in sector Banking" in none_in_sector.warnings[0]


# 7-8. volatility shocks -----------------------------------------------------------------------

@pytest.mark.parametrize("k", [1.5, 2.0])
def test_volatility_multiplier(k):
    result = run_volatility_stress_test(DATA, WABC, k)
    sigma = {s: statistics.stdev(RET[s]) for s in "ABC"}
    rows = impacts(result)
    for s in "ABC":
        assert rows.loc[s, "stressed_return"] == pytest.approx(-k * sigma[s], rel=1e-12)
        assert rows.loc[s, "volatility_component"] == pytest.approx(-k * sigma[s], rel=1e-12)
    assert result.portfolio_return == pytest.approx(-k * sum(WABC[s] * sigma[s] for s in "ABC"),
                                                    rel=1e-12)
    assert result.scenario_type == VOLATILITY
    assert result.most_exposed_holding == max(sigma, key=sigma.get)


def test_volatility_with_historical_mean_baseline_is_mean_minus_k_sigma():
    result = run_volatility_stress_test(DATA, WABC, 2.0, baseline=BASELINE_HISTORICAL_MEAN)
    for s in "ABC":
        expected = statistics.mean(RET[s]) - 2.0 * statistics.stdev(RET[s])
        assert impacts(result).loc[s, "stressed_return"] == pytest.approx(expected, rel=1e-12)


def test_volatility_needs_enough_history():
    short = combine(market("A", RA[:10]), market("B", RB[:10]))
    result = run_volatility_stress_test(short, {"A": 0.5, "B": 0.5}, 2.0)
    assert result.status == STATUS_UNAVAILABLE and math.isnan(result.portfolio_return)
    assert "10 common daily return(s), at least 20 required" in result.message
    assert run_volatility_stress_test(short, {"A": 0.5, "B": 0.5}, 2.0,
                                      min_observations=10).status == STATUS_OK


# 9. liquidity shock --------------------------------------------------------------------------

def test_liquidity_multiplier_half():
    result = run_liquidity_stress_test(DATA, WABC, 0.5, portfolio_value=1_000_000)
    rows = result.liquidity_impacts.set_index("symbol")
    for s in "ABC":
        adtv = statistics.mean(c * 1_000.0 for c in FRAMES[s]["close"])
        position = 1_000_000 * WABC[s]
        assert rows.loc[s, "base_adtv"] == pytest.approx(adtv, rel=1e-12)
        assert rows.loc[s, "stressed_adtv"] == pytest.approx(adtv * 0.5, rel=1e-12)
        assert rows.loc[s, "base_position_to_adtv"] == pytest.approx(position / adtv, rel=1e-12)
        assert rows.loc[s, "stressed_position_to_adtv"] == pytest.approx(position / (adtv * 0.5),
                                                                         rel=1e-12)
        assert rows.loc[s, "base_liquidation_days"] == pytest.approx(position / (adtv * 0.10),
                                                                     rel=1e-12)
        assert rows.loc[s, "stressed_liquidation_days"] == pytest.approx(
            position / (adtv * 0.5 * 0.10), rel=1e-12)
    assert result.scenario_type == LIQUIDITY
    assert result.most_exposed_holding == rows["stressed_liquidation_days"].idxmax()


def test_liquidity_participation_rate_is_used():
    result = run_liquidity_stress_test(DATA, WABC, 0.5, portfolio_value=1_000_000,
                                       participation_rate=0.25)
    row = result.liquidity_impacts.set_index("symbol").loc["A"]
    adtv = statistics.mean(c * 1_000.0 for c in FRAMES["A"]["close"])
    assert row["stressed_liquidation_days"] == pytest.approx(500_000 / (adtv * 0.5 * 0.25))


# 10, 33. combined shocks, no double counting --------------------------------------------------

def test_combined_market_and_sector_shock():
    mapping = {"A": "Banking", "B": "Banking", "C": "Hotels"}
    result = run_combined_stress_test(DATA, WABC, market_shock=-0.10, sector_name="Banking",
                                      sector_shock=-0.20, sector_mapping=mapping)
    rows = impacts(result)
    assert result.scenario_type == COMBINED
    assert list(rows["market_component"]) == [-0.10, -0.10, -0.10]
    assert list(rows["sector_component"]) == [-0.20, -0.20, 0.0]
    assert list(rows["stressed_return"]) == pytest.approx([-0.30, -0.30, -0.10])
    assert result.portfolio_return == pytest.approx(0.8 * -0.30 + 0.2 * -0.10)


def test_combined_shocks_do_not_double_count():
    mapping = {"A": "Banking", "B": "Banking", "C": "Banking"}
    combined = run_combined_stress_test(DATA, WABC, market_shock=-0.10, sector_name="Banking",
                                        sector_shock=-0.20, sector_mapping=mapping)
    # each component once: -10 + -20 = -30, never -40 (market twice) or -50 (sector twice)
    assert combined.portfolio_return == pytest.approx(-0.30)
    rows = impacts(combined)
    assert (rows["market_component"] + rows["sector_component"]
            == rows["stressed_return"]).all()
    # a scenario's type must match the components it sets
    with pytest.raises(ValueError, match="MARKET scenario must set only its own shock"):
        StressScenario("x", MARKET, market_shock=-0.1, sector_name="Banking", sector_shock=-0.2)
    with pytest.raises(ValueError, match="at least two"):
        StressScenario("x", COMBINED, market_shock=-0.1)
    with pytest.raises(ValueError, match="sets no shock"):
        StressScenario("x", MARKET)


def test_combined_with_volatility_and_liquidity():
    result = run_combined_stress_test(DATA, WABC, market_shock=-0.05, volatility_multiplier=1.0,
                                      liquidity_multiplier=0.5, portfolio_value=1_000_000)
    for s in "ABC":
        expected = -0.05 - statistics.stdev(RET[s])
        assert impacts(result).loc[s, "stressed_return"] == pytest.approx(expected, rel=1e-12)
    assert result.liquidity_impacts is not None


# 11-17. portfolio impact ------------------------------------------------------------------------

def test_portfolio_return_loss_amount_and_stressed_value():
    mapping = {"A": "Banking", "B": "Hotels", "C": "Banking"}
    result = run_combined_stress_test(DATA, WABC, market_shock=-0.05, sector_name="Banking",
                                      sector_shock=-0.10, sector_mapping=mapping,
                                      portfolio_value=5_000_000)
    expected_return = 0.5 * -0.15 + 0.3 * -0.05 + 0.2 * -0.15             # -0.12
    assert result.portfolio_return == pytest.approx(expected_return)
    assert result.portfolio_loss == pytest.approx(0.12)
    assert result.portfolio_loss_amount == pytest.approx(5_000_000 * 0.12)
    assert result.stressed_portfolio_value == pytest.approx(5_000_000 * (1 - 0.12))


def test_per_stock_stressed_return_and_contribution():
    mapping = {"A": "Banking", "B": "Hotels", "C": "Banking"}
    result = run_combined_stress_test(DATA, WABC, market_shock=-0.05, sector_name="Banking",
                                      sector_shock=-0.10, sector_mapping=mapping)
    rows = impacts(result)
    expected = {"A": -0.15, "B": -0.05, "C": -0.15}
    for s, r in expected.items():
        assert rows.loc[s, "stressed_return"] == pytest.approx(r)
        assert rows.loc[s, "stress_contribution"] == pytest.approx(WABC[s] * r)
    assert rows["stress_contribution"].sum() == pytest.approx(result.portfolio_return)


def test_largest_negative_contributor_and_most_exposed():
    mapping = {"A": "Hotels", "B": "Hotels", "C": "Banking"}
    # C has the worst return (-0.40) but the smallest weight: 0.2 * -0.40 = -0.08;
    # A contributes 0.5 * -0.20 = -0.10
    result = run_combined_stress_test(DATA, WABC, market_shock=-0.20, sector_name="Banking",
                                      sector_shock=-0.20, sector_mapping=mapping)
    assert result.largest_negative_contributor == "A"
    assert result.largest_negative_contribution == pytest.approx(-0.10)
    assert result.most_exposed_holding == "C"


# 18-21. optional and missing data -------------------------------------------------------------

def test_portfolio_value_is_optional():
    result = run_market_stress_test(DATA, WABC, -0.10)
    assert result.portfolio_value is None
    assert result.portfolio_loss_amount is None and result.stressed_portfolio_value is None


def test_liquidity_shock_without_portfolio_value_is_unavailable():
    result = run_liquidity_stress_test(DATA, WABC, 0.5)
    assert result.status == STATUS_UNAVAILABLE and "needs a portfolio value" in result.message
    assert result.liquidity_impacts is None and math.isnan(result.portfolio_return)


def test_missing_sector_data_is_unavailable_and_never_guessed():
    result = run_sector_stress_test(DATA, WABC, "Banking", -0.20)
    assert result.status == STATUS_UNAVAILABLE
    assert result.message.startswith("Sector data unavailable")
    partial = run_sector_stress_test(DATA, WABC, "Banking", -0.20, sector_mapping={"A": "Banking"})
    assert partial.status == STATUS_UNAVAILABLE and "B, C" in partial.message
    # a market scenario does not need sectors
    assert run_market_stress_test(DATA, WABC, -0.1).status == STATUS_OK


def test_missing_liquidity_data_is_undefined_with_status():
    no_volume = DATA.drop(columns=["volume"])
    result = run_liquidity_stress_test(no_volume, WABC, 0.5, portfolio_value=1_000_000)
    rows = result.liquidity_impacts
    assert set(rows["liquidity_status"]) == {LIQUIDITY_NO_DATA}
    assert rows[["base_adtv", "stressed_adtv", "stressed_liquidation_days"]].isna().all().all()
    assert "Liquidity undefined" in result.warnings[0]
    assert result.most_exposed_holding is None


# 22-24. validation ----------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"market_shock": float("nan")}, "finite"),
    ({"market_shock": -1.5}, "between -1 and 1"),
    ({"market_shock": "0.1"}, "finite"),
    ({"volatility_multiplier": 0}, "positive"),
    ({"volatility_multiplier": -1.5}, "positive"),
    ({"volatility_multiplier": float("inf")}, "finite"),
    ({"liquidity_multiplier": 0}, "above 0 and at most 1"),
    ({"liquidity_multiplier": 1.5}, "above 0 and at most 1"),
    ({"sector_shock": -0.2}, "both sector_name and sector_shock"),
    ({"sector_name": "Banking"}, "both sector_name and sector_shock"),
    ({"sector_name": " ", "sector_shock": -0.2}, "non-empty"),
])
def test_invalid_scenario_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        scenario_from_components("Bad", **kwargs)


def test_invalid_scenario_definitions_and_options():
    with pytest.raises(ValueError, match="non-empty name"):
        StressScenario("", MARKET, market_shock=-0.1)
    with pytest.raises(ValueError, match="scenario_type"):
        StressScenario("x", "CRASH", market_shock=-0.1)
    with pytest.raises(ValueError, match="at least one shock"):
        scenario_from_components("Empty")
    with pytest.raises(ValueError, match="loss above 100%"):
        run_combined_stress_test(DATA, WABC, market_shock=-0.6, sector_name="Banking",
                                 sector_shock=-0.6, sector_mapping={s: "Banking" for s in "ABC"})
    with pytest.raises(ValueError, match="baseline"):
        run_market_stress_test(DATA, WABC, -0.1, baseline="latest")
    with pytest.raises(ValueError, match="participation_rate"):
        run_liquidity_stress_test(DATA, WABC, 0.5, portfolio_value=1e6, participation_rate=0)
    with pytest.raises(ValueError, match="position_value"):
        run_market_stress_test(DATA, WABC, -0.1, portfolio_value=-5)
    with pytest.raises(ValueError, match="two sectors"):
        run_sector_stress_test(DATA, WABC, "Banking", -0.2,
                               sector_mapping=pd.DataFrame({"symbol": ["A", "A"],
                                                            "sector": ["Banking", "Hotels"]}))
    with pytest.raises(ValueError, match="StressScenario"):
        run_stress_scenario(DATA, WABC, {"market_shock": -0.1})
    with pytest.raises(ValueError, match="unique"):
        run_stress_scenarios(DATA, WABC, [DEFAULT_SCENARIOS[0], DEFAULT_SCENARIOS[0]])


@pytest.mark.parametrize("weights, message", [
    ({"A": 0.6, "B": 0.3}, "sum to 1"),
    ({"A": 1.2, "B": -0.2}, "negative"),
    ({"A": 0.5, "B": float("nan")}, "finite"),
    ([("A", 0.5), ("A", 0.5)], "Duplicate"),
])
def test_invalid_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        run_market_stress_test(DATA, weights, -0.1)


def test_unknown_symbols():
    with pytest.raises(ValueError, match="not found in the market data: ZZZ"):
        run_market_stress_test(DATA, {"A": 0.5, "ZZZ": 0.5}, -0.1)


# 25-30. data handling ---------------------------------------------------------------------------

def test_invalid_rows_are_excluded_and_not_bridged():
    status = ["VALID"] * 26
    status[5] = "INVALID"
    a = market("A", RA, volume=[1000 + 10 * i for i in range(26)], status=status)
    data = combine(a, market("B", RB, status=["VALID"] * 26))
    vol = run_volatility_stress_test(data, {"A": 0.5, "B": 0.5}, 1.0)
    kept = [t for t in range(25) if t + 1 not in (5, 6)]       # no return into or out of row 5
    sigma_a = statistics.stdev([RET["A"][t] for t in kept])
    assert impacts(vol).loc["A", "stressed_return"] == pytest.approx(-sigma_a, rel=1e-12)
    liquid = run_liquidity_stress_test(data, {"A": 0.5, "B": 0.5}, 0.5, portfolio_value=1e6)
    closes, volumes = a["close"].tolist(), a["volume"].tolist()
    adtv = statistics.mean(c * v for i, (c, v) in enumerate(zip(closes, volumes)) if i != 5)
    assert liquid.liquidity_impacts.set_index("symbol").loc["A", "base_adtv"] == pytest.approx(adtv)


def test_warning_rows_are_usable():
    data = combine(market("A", RA, status=["WARNING"] * 26), market("B", RB, status=["VALID"] * 26))
    result = run_volatility_stress_test(data, {"A": 0.5, "B": 0.5}, 1.0)
    assert impacts(result).loc["A", "stressed_return"] == pytest.approx(-statistics.stdev(RET["A"]))


def test_unsorted_data_gives_the_same_results():
    shuffled = DATA.sample(frac=1, random_state=9).reset_index(drop=True)
    for runner in (lambda d: run_volatility_stress_test(d, WABC, 2.0, portfolio_value=1e6),
                   lambda d: run_liquidity_stress_test(d, WABC, 0.5, portfolio_value=1e6)):
        a, b = runner(shuffled), runner(DATA)
        pd.testing.assert_frame_equal(a.symbol_impacts, b.symbol_impacts)
        assert a.portfolio_return == b.portfolio_return


def test_empty_data():
    with pytest.raises(ValueError, match="empty"):
        run_market_stress_test(DATA.iloc[0:0], WABC, -0.1)
    with pytest.raises(ValueError, match="missing column"):
        run_market_stress_test(pd.DataFrame(), WABC, -0.1)
    with pytest.raises(ValueError, match="DataFrame"):
        run_market_stress_test(None, WABC, -0.1)


def test_multiple_and_single_stock_portfolios(sample):
    multiple = run_market_stress_test(sample, W3, -0.2, portfolio_value=1e6)
    assert len(multiple.symbol_impacts) == 3 and multiple.portfolio_return == pytest.approx(-0.2)
    single = run_volatility_stress_test(DATA, {"B": 1.0}, 2.0, portfolio_value=1e6)
    assert list(single.symbol_impacts["symbol"]) == ["B"]
    assert single.portfolio_return == pytest.approx(-2.0 * statistics.stdev(RET["B"]), rel=1e-12)
    assert single.most_exposed_holding == "B" and single.largest_negative_contributor == "B"


def test_input_is_not_mutated():
    data = DATA.sample(frac=1, random_state=2).reset_index(drop=True)
    before = data.copy(deep=True)
    weights = {"C": 0.2, "A": 0.5, "B": 0.3}
    mapping = {"A": "Banking", "B": "Hotels", "C": "Banking"}
    run_stress_scenarios(data, weights, list(DEFAULT_SCENARIOS) + [
        scenario_from_components("Combo", market_shock=-0.1, sector_name="Banking",
                                 sector_shock=-0.2)],
        portfolio_value=1e6, sector_mapping=mapping)
    pd.testing.assert_frame_equal(data, before)
    assert weights == {"C": 0.2, "A": 0.5, "B": 0.3}
    assert mapping == {"A": "Banking", "B": "Hotels", "C": "Banking"}


# 31-32. historical risk metrics untouched; determinism -----------------------------------------

def quantile_type7(values, p):
    x = sorted(values)
    h = (len(x) - 1) * p
    lo = math.floor(h)
    return x[lo] + (h - lo) * (x[min(lo + 1, len(x) - 1)] - x[lo])


def es_exact(returns, confidence):
    alpha = Fraction(1) - Fraction(str(confidence))
    losses = sorted((-Fraction(float(r)) for r in returns), reverse=True)
    m = alpha * len(losses)
    k = math.floor(m)
    total = sum(losses[:k], Fraction(0)) + ((m - k) * losses[k] if m > k else 0)
    return float(total / m)


def test_historical_var_and_cvar_are_reported_unchanged():
    report = run_stress_scenarios(DATA, WABC, portfolio_value=1e6)
    series = [sum(WABC[s] * RET[s][t] for s in "ABC") for t in range(25)]
    assert report.historical_var == pytest.approx(-quantile_type7(series, 0.05), rel=1e-9)
    assert report.historical_cvar == pytest.approx(es_exact(series, 0.95), rel=1e-9)
    risk = calculate_portfolio_risk_summary(DATA, WABC)
    assert (report.historical_var, report.historical_cvar) == (risk.historical_var,
                                                              risk.historical_cvar)
    harsher = run_stress_scenarios(DATA, WABC, [DEFAULT_SCENARIOS[2]], portfolio_value=1e6)
    assert harsher.historical_cvar == report.historical_cvar           # scenarios never alter it
    assert harsher.results[0].portfolio_loss == pytest.approx(0.30)
    assert harsher.results[0].portfolio_loss != report.historical_cvar


def test_scenario_results_are_deterministic():
    first = run_stress_scenarios(DATA, WABC, portfolio_value=1e6)
    second = run_stress_scenarios(DATA, WABC, portfolio_value=1e6)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    for a, b in zip(first.results, second.results):
        pd.testing.assert_frame_equal(a.symbol_impacts, b.symbol_impacts)


def test_run_stress_scenarios_skips_disabled_and_reports_unavailable():
    disabled = StressScenario("Off", MARKET, market_shock=-0.5, enabled=False)
    report = run_stress_scenarios(DATA, WABC, list(DEFAULT_SCENARIOS) + [disabled])
    assert list(report.summary["scenario"]) == [s.name for s in DEFAULT_SCENARIOS]
    status = dict(zip(report.summary["scenario"], report.summary["status"]))
    assert status["Liquidity -50%"] == STATUS_UNAVAILABLE              # no portfolio value
    assert status["Market -10%"] == STATUS_OK
    assert report.summary["portfolio_loss_amount"].isna().all()


# 34-38. liquidity details -----------------------------------------------------------------------

def test_liquidity_shock_changes_liquidation_not_price():
    base = run_market_stress_test(DATA, WABC, -0.10, portfolio_value=1e6)
    with_liquidity = run_combined_stress_test(DATA, WABC, market_shock=-0.10,
                                              liquidity_multiplier=0.5, portfolio_value=1e6)
    assert with_liquidity.portfolio_return == base.portfolio_return
    only_liquidity = run_liquidity_stress_test(DATA, WABC, 0.5, portfolio_value=1e6,
                                               baseline=BASELINE_HISTORICAL_MEAN)
    rows = impacts(only_liquidity)
    assert (rows["stressed_return"] == rows["base_return"]).all()
    assert only_liquidity.portfolio_return == pytest.approx(
        sum(WABC[s] * statistics.mean(RET[s]) for s in "ABC"), rel=1e-12)
    liquidity = only_liquidity.liquidity_impacts
    assert list(liquidity["stressed_liquidation_days"]) == pytest.approx(
        list(liquidity["base_liquidation_days"] * 2), rel=1e-12)


def test_actual_turnover_is_used(sample):
    result = run_liquidity_stress_test(sample, W3, 0.5, portfolio_value=20_000_000)
    rows = result.liquidity_impacts.set_index("symbol")
    assert set(rows["traded_value_source"]) == {ACTUAL_TURNOVER}
    usable = sample[sample["validation_status"] != "INVALID"]
    for s in W3:
        expected = statistics.mean(usable.loc[usable["symbol"] == s, "turnover"])
        assert rows.loc[s, "base_adtv"] == pytest.approx(expected, rel=1e-12)


def test_estimated_traded_value_when_turnover_is_incomplete():
    turnover = [5e5] * 26
    turnover[3] = None
    a = market("A", RA, volume=2_000.0, turnover=turnover)
    data = combine(a, market("B", RB, turnover=[1e5] * 26))
    rows = run_liquidity_stress_test(data, {"A": 0.5, "B": 0.5}, 0.5,
                                     portfolio_value=1e6).liquidity_impacts.set_index("symbol")
    assert rows.loc["A", "traded_value_source"] == ESTIMATED_TRADED_VALUE
    assert rows.loc["A", "base_adtv"] == pytest.approx(
        statistics.mean(c * 2_000.0 for c in a["close"]), rel=1e-12)
    assert rows.loc["B", "traded_value_source"] == ACTUAL_TURNOVER


def test_zero_adtv_is_undefined_not_infinite():
    data = combine(market("A", RA, volume=0.0), market("B", RB))
    result = run_liquidity_stress_test(data, {"A": 0.5, "B": 0.5}, 0.5, portfolio_value=1e6)
    rows = result.liquidity_impacts.set_index("symbol")
    assert rows.loc["A", "liquidity_status"] == LIQUIDITY_ZERO_ADTV
    assert rows.loc["A", "base_adtv"] == 0.0
    assert math.isnan(rows.loc["A", "stressed_liquidation_days"])
    assert math.isnan(rows.loc["A", "stressed_position_to_adtv"])
    assert rows.loc["B", "liquidity_status"] == LIQUIDITY_OK
    assert "A" in result.warnings[0] and result.most_exposed_holding == "B"


def test_zero_volume_days_count_in_adtv():
    volumes = [1_000.0] * 26
    for i in (2, 7, 11):
        volumes[i] = 0.0
    a = market("A", RA, volume=volumes)
    result = run_liquidity_stress_test(combine(a, FRAMES["B"]), {"A": 0.5, "B": 0.5}, 0.5,
                                       portfolio_value=1e6)
    expected = statistics.mean(c * v for c, v in zip(a["close"], volumes))   # zeros included
    row = result.liquidity_impacts.set_index("symbol").loc["A"]
    assert row["base_adtv"] == pytest.approx(expected, rel=1e-12)
    assert row["stressed_liquidation_days"] == pytest.approx(500_000 / (expected * 0.5 * 0.10))
