"""Scenario stress tests for a fixed-weight portfolio.

A scenario adds simple shocks (market, sector, volatility) to each holding's
return, or reduces trading capacity. Results are hypothetical: no probability
is attached to any scenario.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.analytics.common import require_canonical_columns
from app.analytics.covariance import calculate_aligned_return_matrix
from app.analytics.liquidity import (
    DEFAULT_PARTICIPATION_RATE,
    calculate_liquidity_summary,
    validate_participation_rate,
    validate_position_value,
)
from app.analytics.portfolio_risk import calculate_portfolio_risk_summary, validate_weights
from app.analytics.var import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MIN_OBSERVATIONS,
    validate_confidence_level,
    validate_min_observations,
)

MARKET, SECTOR, VOLATILITY, LIQUIDITY, COMBINED = (
    "MARKET", "SECTOR", "VOLATILITY", "LIQUIDITY", "COMBINED")
SCENARIO_TYPES = (MARKET, SECTOR, VOLATILITY, LIQUIDITY, COMBINED)
BASELINE_ZERO, BASELINE_HISTORICAL_MEAN = "zero", "historical_mean"
STATUS_OK, STATUS_UNAVAILABLE = "OK", "UNAVAILABLE"
LIQUIDITY_OK, LIQUIDITY_ZERO_ADTV, LIQUIDITY_NO_DATA = "OK", "ZERO_ADTV", "NO_LIQUIDITY_DATA"

SYMBOL_IMPACT_COLUMNS = ["symbol", "weight", "sector", "base_return", "market_component",
                         "sector_component", "volatility_component", "stressed_return",
                         "stress_contribution"]
LIQUIDITY_IMPACT_COLUMNS = ["symbol", "weight", "position_value", "traded_value_source",
                            "base_adtv", "stressed_adtv", "base_position_to_adtv",
                            "stressed_position_to_adtv", "base_liquidation_days",
                            "stressed_liquidation_days", "liquidity_status"]
SUMMARY_COLUMNS = ["scenario", "scenario_type", "status", "portfolio_return", "portfolio_loss",
                   "portfolio_loss_amount", "stressed_portfolio_value",
                   "largest_negative_contributor", "largest_negative_contribution",
                   "most_exposed_holding"]


def _finite(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


@dataclass(frozen=True)
class StressScenario:
    """One hypothetical scenario. Shocks are absolute returns (-0.10 = -10 points)."""
    name: str
    scenario_type: str
    description: str = ""
    market_shock: float | None = None
    sector_name: str | None = None
    sector_shock: float | None = None
    volatility_multiplier: float | None = None
    liquidity_multiplier: float | None = None
    enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A scenario needs a non-empty name")
        if self.scenario_type not in SCENARIO_TYPES:
            raise ValueError(f"scenario_type must be one of {', '.join(SCENARIO_TYPES)}, "
                             f"got {self.scenario_type!r}")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be True or False")
        for name in ("market_shock", "sector_shock"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
                if not -1 <= value <= 1:
                    raise ValueError(f"{name} must be between -1 and 1 (-100% to +100%), "
                                     f"got {value!r}")
        if (self.sector_name is None) != (self.sector_shock is None):
            raise ValueError("A sector shock needs both sector_name and sector_shock")
        if self.sector_name is not None and (not isinstance(self.sector_name, str)
                                             or not self.sector_name.strip()):
            raise ValueError("sector_name must be a non-empty text value")
        if self.volatility_multiplier is not None:
            _finite(self.volatility_multiplier, "volatility_multiplier")
            if not self.volatility_multiplier > 0:
                raise ValueError("volatility_multiplier must be positive, got "
                                 f"{self.volatility_multiplier!r}")
        if self.liquidity_multiplier is not None:
            _finite(self.liquidity_multiplier, "liquidity_multiplier")
            if not 0 < self.liquidity_multiplier <= 1:
                raise ValueError("liquidity_multiplier must be above 0 and at most 1 "
                                 f"(0.5 = half the trading capacity), got "
                                 f"{self.liquidity_multiplier!r}")
        components = self.components
        if not components:
            raise ValueError(f"Scenario {self.name!r} sets no shock")
        expected = {MARKET: ("market",), SECTOR: ("sector",), VOLATILITY: ("volatility",),
                    LIQUIDITY: ("liquidity",)}.get(self.scenario_type)
        if expected is not None and components != expected:
            raise ValueError(f"A {self.scenario_type} scenario must set only its own shock, "
                             f"got: {', '.join(components)}")
        if self.scenario_type == COMBINED and len(components) < 2:
            raise ValueError("A COMBINED scenario needs at least two shock components")

    @property
    def components(self):
        """The shock components this scenario sets, in a fixed order."""
        present = {"market": self.market_shock is not None,
                   "sector": self.sector_shock is not None,
                   "volatility": self.volatility_multiplier is not None,
                   "liquidity": self.liquidity_multiplier is not None}
        return tuple(name for name, on in present.items() if on)


DEFAULT_SCENARIOS = (
    StressScenario("Market -10%", MARKET, "Every holding's return falls by 10 percentage "
                   "points.", market_shock=-0.10),
    StressScenario("Market -20%", MARKET, "Every holding's return falls by 20 percentage "
                   "points.", market_shock=-0.20),
    StressScenario("Market -30%", MARKET, "Every holding's return falls by 30 percentage "
                   "points.", market_shock=-0.30),
    StressScenario("High Volatility 1.5x", VOLATILITY, "Every holding falls by 1.5 of its own "
                   "daily standard deviations on the same day.", volatility_multiplier=1.5),
    StressScenario("High Volatility 2.0x", VOLATILITY, "Every holding falls by 2.0 of its own "
                   "daily standard deviations on the same day.", volatility_multiplier=2.0),
    StressScenario("Liquidity -50%", LIQUIDITY, "Executable daily trading capacity (ADTV) is "
                   "halved; prices are unchanged.", liquidity_multiplier=0.5),
)


@dataclass(frozen=True)
class StressResult:
    scenario_name: str
    scenario_type: str
    description: str
    status: str                              # OK or UNAVAILABLE (see message)
    message: str
    portfolio_return: float                  # scenario return (not a forecast)
    portfolio_loss: float                    # -portfolio_return
    portfolio_value: float | None
    portfolio_loss_amount: float | None      # only with a portfolio value
    stressed_portfolio_value: float | None   # only with a portfolio value
    symbol_impacts: pd.DataFrame | None      # SYMBOL_IMPACT_COLUMNS
    liquidity_impacts: pd.DataFrame | None   # LIQUIDITY_IMPACT_COLUMNS (liquidity shocks)
    largest_negative_contributor: str | None
    largest_negative_contribution: float
    most_exposed_holding: str | None
    assumptions: tuple
    warnings: tuple = ()
    scenario: StressScenario | None = field(default=None, repr=False)


@dataclass(frozen=True)
class StressReport:
    results: list                            # StressResult per enabled scenario
    summary: pd.DataFrame                    # SUMMARY_COLUMNS
    historical_var: float                    # 1-day, from past returns (unchanged)
    historical_cvar: float
    confidence_level: float
    historical_observations: int
    portfolio_value: float | None


def run_stress_scenario(data, weights, scenario, portfolio_value=None, sector_mapping=None,
                        participation_rate=DEFAULT_PARTICIPATION_RATE, baseline=BASELINE_ZERO,
                        min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Apply one scenario to the portfolio. Missing optional data gives an UNAVAILABLE result."""
    if not isinstance(scenario, StressScenario):
        raise ValueError("scenario must be a StressScenario")
    weights, value = _validate_inputs(data, weights, portfolio_value, participation_rate,
                                      baseline, min_observations)
    symbols = list(weights)
    assumptions = _assumptions(scenario, baseline, participation_rate, min_observations)

    def unavailable(message):
        return _unavailable(scenario, message, value, assumptions)

    sectors = {s: None for s in symbols}
    if scenario.sector_shock is not None:
        if sector_mapping is None:
            return unavailable("Sector data unavailable: no sector mapping was supplied, and "
                               "sectors are never guessed.")
        mapping = _sector_mapping(sector_mapping)
        unmapped = [s for s in symbols if s not in mapping]
        if unmapped:
            return unavailable("Sector data unavailable for " + ", ".join(unmapped)
                               + ": every holding needs a sector for a sector shock.")
        sectors = {s: mapping[s] for s in symbols}
    elif sector_mapping is not None:
        mapping = _sector_mapping(sector_mapping)
        sectors = {s: mapping.get(s) for s in symbols}

    needs_history = baseline == BASELINE_HISTORICAL_MEAN or scenario.volatility_multiplier
    if needs_history:
        returns = calculate_aligned_return_matrix(
            data[data["symbol"].isin([s for s in symbols if weights[s] > 0])])
        if len(returns) < min_observations:
            return unavailable(f"Insufficient return history: {len(returns)} common daily "
                               f"return(s), at least {min_observations} required for "
                               + ("the volatility shock" if scenario.volatility_multiplier
                                  else "the historical-mean baseline") + ".")
        mean, sigma = returns.mean(), returns.std(ddof=1)

    rows = []
    for s in symbols:
        has_history = needs_history and s in mean.index
        base = float(mean[s]) if baseline == BASELINE_HISTORICAL_MEAN and has_history else 0.0
        market = scenario.market_shock or 0.0
        in_sector = (scenario.sector_shock is not None
                     and sectors[s].casefold() == scenario.sector_name.strip().casefold())
        sector = scenario.sector_shock if in_sector else 0.0
        volatility = (-scenario.volatility_multiplier * float(sigma[s])
                      if scenario.volatility_multiplier and has_history else 0.0)
        stressed = base + market + sector + volatility
        if stressed < -1 - 1e-12:
            raise ValueError(f"Scenario {scenario.name!r} gives {s} a return of "
                             f"{stressed:.2%}, a loss above 100%; reduce the shocks.")
        rows.append({"symbol": s, "weight": weights[s], "sector": sectors[s], "base_return": base,
                     "market_component": market, "sector_component": sector,
                     "volatility_component": volatility, "stressed_return": stressed,
                     "stress_contribution": weights[s] * stressed})
    impacts = pd.DataFrame(rows, columns=SYMBOL_IMPACT_COLUMNS)
    portfolio_return = float(impacts["stress_contribution"].sum())
    portfolio_loss = 0.0 - portfolio_return            # avoids showing a -0.0 loss

    warnings = []
    if scenario.sector_shock is not None and not any(
            sectors[s].casefold() == scenario.sector_name.strip().casefold() for s in symbols):
        warnings.append(f"No holding is in sector {scenario.sector_name.strip()}; the sector "
                        "shock changes nothing.")
    liquidity = None
    if scenario.liquidity_multiplier is not None:
        if value is None:
            return unavailable("A liquidity shock needs a portfolio value (positions are "
                               "compared with trading capacity).")
        liquidity = _liquidity_impacts(data, weights, value, scenario.liquidity_multiplier,
                                       participation_rate)
        missing = liquidity.loc[liquidity["liquidity_status"] != LIQUIDITY_OK, "symbol"]
        if len(missing):
            warnings.append("Liquidity undefined (no liquidity data or zero ADTV) for: "
                            + ", ".join(missing) + ".")

    negative = impacts[impacts["stress_contribution"] < 0]
    worst = negative.loc[negative["stress_contribution"].idxmin()] if len(negative) else None
    return StressResult(
        scenario_name=scenario.name, scenario_type=scenario.scenario_type,
        description=scenario.description, status=STATUS_OK, message="",
        portfolio_return=portfolio_return, portfolio_loss=portfolio_loss,
        portfolio_value=value,
        portfolio_loss_amount=None if value is None else value * portfolio_loss,
        stressed_portfolio_value=None if value is None else value * (1 + portfolio_return),
        symbol_impacts=impacts, liquidity_impacts=liquidity,
        largest_negative_contributor=None if worst is None else worst["symbol"],
        largest_negative_contribution=(math.nan if worst is None
                                       else float(worst["stress_contribution"])),
        most_exposed_holding=_most_exposed(scenario, impacts, liquidity),
        assumptions=assumptions, warnings=tuple(warnings), scenario=scenario)


def run_market_stress_test(data, weights, market_shock, portfolio_value=None, name=None,
                           **options):
    """Every holding's return shifted by ``market_shock`` (e.g. -0.10)."""
    scenario = StressScenario(name or f"Market {market_shock:+.0%}", MARKET,
                              "Every holding's return shifted by the market shock.",
                              market_shock=market_shock)
    return run_stress_scenario(data, weights, scenario, portfolio_value, **options)


def run_sector_stress_test(data, weights, sector_name, sector_shock, sector_mapping=None,
                           portfolio_value=None, name=None, **options):
    """Holdings mapped to ``sector_name`` shifted by ``sector_shock``; others unchanged."""
    scenario = StressScenario(name or f"{sector_name} {sector_shock:+.0%}", SECTOR,
                              f"Holdings in {sector_name} shifted by the sector shock.",
                              sector_name=sector_name, sector_shock=sector_shock)
    return run_stress_scenario(data, weights, scenario, portfolio_value,
                               sector_mapping=sector_mapping, **options)


def run_volatility_stress_test(data, weights, volatility_multiplier, portfolio_value=None,
                               name=None, **options):
    """Every holding falls by ``volatility_multiplier`` of its own daily standard deviations."""
    scenario = StressScenario(name or f"High Volatility {volatility_multiplier:g}x", VOLATILITY,
                              "Every holding falls by k of its own daily standard deviations.",
                              volatility_multiplier=volatility_multiplier)
    return run_stress_scenario(data, weights, scenario, portfolio_value, **options)


def run_liquidity_stress_test(data, weights, liquidity_multiplier, portfolio_value=None,
                              name=None, **options):
    """Trading capacity (ADTV) multiplied by ``liquidity_multiplier``; prices unchanged."""
    scenario = StressScenario(name or f"Liquidity {liquidity_multiplier - 1:+.0%}", LIQUIDITY,
                              "Executable daily trading capacity reduced; prices unchanged.",
                              liquidity_multiplier=liquidity_multiplier)
    return run_stress_scenario(data, weights, scenario, portfolio_value, **options)


def run_combined_stress_test(data, weights, market_shock=None, sector_name=None,
                             sector_shock=None, volatility_multiplier=None,
                             liquidity_multiplier=None, sector_mapping=None,
                             portfolio_value=None, name="Combined scenario", **options):
    """Several shock components at once, each applied exactly once."""
    scenario = StressScenario(name, COMBINED, "Several shock components applied together.",
                              market_shock=market_shock, sector_name=sector_name,
                              sector_shock=sector_shock,
                              volatility_multiplier=volatility_multiplier,
                              liquidity_multiplier=liquidity_multiplier)
    return run_stress_scenario(data, weights, scenario, portfolio_value,
                               sector_mapping=sector_mapping, **options)


def run_stress_scenarios(data, weights, scenarios=DEFAULT_SCENARIOS, portfolio_value=None,
                         sector_mapping=None, participation_rate=DEFAULT_PARTICIPATION_RATE,
                         baseline=BASELINE_ZERO, confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                         min_observations=DEFAULT_MIN_OBSERVATIONS):
    """Run every enabled scenario and report historical VaR/CVaR alongside (unchanged)."""
    validate_confidence_level(confidence_level)
    scenarios = list(scenarios)
    if any(not isinstance(s, StressScenario) for s in scenarios):
        raise ValueError("scenarios must be StressScenario objects")
    names = [s.name for s in scenarios if s.enabled]
    if len(set(names)) != len(names):
        raise ValueError("Scenario names must be unique")
    results = [run_stress_scenario(data, weights, s, portfolio_value, sector_mapping,
                                   participation_rate, baseline, min_observations)
               for s in scenarios if s.enabled]
    risk = calculate_portfolio_risk_summary(data, weights, confidence_level=confidence_level,
                                            min_observations=min_observations)
    summary = pd.DataFrame([{
        "scenario": r.scenario_name, "scenario_type": r.scenario_type, "status": r.status,
        "portfolio_return": r.portfolio_return, "portfolio_loss": r.portfolio_loss,
        "portfolio_loss_amount": r.portfolio_loss_amount,
        "stressed_portfolio_value": r.stressed_portfolio_value,
        "largest_negative_contributor": r.largest_negative_contributor,
        "largest_negative_contribution": r.largest_negative_contribution,
        "most_exposed_holding": r.most_exposed_holding} for r in results],
        columns=SUMMARY_COLUMNS)
    return StressReport(results=results, summary=summary, historical_var=risk.historical_var,
                        historical_cvar=risk.historical_cvar, confidence_level=confidence_level,
                        historical_observations=risk.observations,
                        portfolio_value=None if portfolio_value is None
                        else float(portfolio_value))


def scenario_from_components(name, market_shock=None, sector_name=None, sector_shock=None,
                             volatility_multiplier=None, liquidity_multiplier=None,
                             description="Custom scenario."):
    """A StressScenario whose type follows from the components set (custom scenarios)."""
    probe = {"market": market_shock is not None,
             "sector": sector_shock is not None or sector_name is not None,
             "volatility": volatility_multiplier is not None,
             "liquidity": liquidity_multiplier is not None}
    count = sum(probe.values())
    kind = (COMBINED if count > 1 else
            {"market": MARKET, "sector": SECTOR, "volatility": VOLATILITY,
             "liquidity": LIQUIDITY}[next(k for k, on in probe.items() if on)] if count else
            None)
    if kind is None:
        raise ValueError("A custom scenario needs at least one shock")
    return StressScenario(name, kind, description, market_shock=market_shock,
                          sector_name=sector_name, sector_shock=sector_shock,
                          volatility_multiplier=volatility_multiplier,
                          liquidity_multiplier=liquidity_multiplier)


def _validate_inputs(data, weights, portfolio_value, participation_rate, baseline,
                     min_observations):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Market data must be a DataFrame")
    require_canonical_columns(data, "stress tests")
    if data.empty:
        raise ValueError("Market data is empty")
    weights = validate_weights(weights, set(data["symbol"].dropna()))
    if portfolio_value is not None:
        validate_position_value(portfolio_value)
    validate_participation_rate(participation_rate)
    validate_min_observations(min_observations)
    if baseline not in (BASELINE_ZERO, BASELINE_HISTORICAL_MEAN):
        raise ValueError(f"baseline must be {BASELINE_ZERO!r} or {BASELINE_HISTORICAL_MEAN!r}, "
                         f"got {baseline!r}")
    return weights, None if portfolio_value is None else float(portfolio_value)


def _sector_mapping(mapping):
    if isinstance(mapping, pd.DataFrame):
        if not {"symbol", "sector"} <= set(mapping.columns):
            raise ValueError("A sector table needs 'symbol' and 'sector' columns")
        pairs = list(zip(mapping["symbol"], mapping["sector"]))
    elif isinstance(mapping, Mapping):
        pairs = list(mapping.items())
    else:
        raise ValueError("sector_mapping must be {symbol: sector} or a symbol/sector table")
    result = {}
    for symbol, sector in pairs:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Every sector-mapping row needs a symbol")
        if not isinstance(sector, str) or not sector.strip():
            raise ValueError(f"The sector for {symbol.strip()} is missing")
        symbol, sector = symbol.strip(), sector.strip()
        if symbol in result and result[symbol].casefold() != sector.casefold():
            raise ValueError(f"{symbol} is mapped to two sectors ({result[symbol]}, {sector})")
        result[symbol] = sector
    return result


def _liquidity_impacts(data, weights, value, multiplier, participation_rate):
    summary = (calculate_liquidity_summary(data[data["symbol"].isin(list(weights))])
               .set_index("symbol") if "volume" in data.columns else pd.DataFrame())
    rows = []
    for s, w in weights.items():
        known = s in summary.index
        adtv = float(summary.loc[s, "average_daily_traded_value"]) if known else math.nan
        status = (LIQUIDITY_NO_DATA if not known or math.isnan(adtv) else
                  LIQUIDITY_ZERO_ADTV if adtv == 0 else LIQUIDITY_OK)
        usable = adtv if status == LIQUIDITY_OK else math.nan
        stressed = usable * multiplier
        position = value * w
        rows.append({"symbol": s, "weight": w, "position_value": position,
                     "traded_value_source": summary.loc[s, "traded_value_source"] if known else None,
                     "base_adtv": adtv, "stressed_adtv": adtv * multiplier,
                     "base_position_to_adtv": position / usable,
                     "stressed_position_to_adtv": position / stressed,
                     "base_liquidation_days": position / (usable * participation_rate),
                     "stressed_liquidation_days": position / (stressed * participation_rate),
                     "liquidity_status": status})
    return pd.DataFrame(rows, columns=LIQUIDITY_IMPACT_COLUMNS)


def _most_exposed(scenario, impacts, liquidity):
    """Price scenarios: lowest stressed return. Liquidity-only: longest stressed exit."""
    if scenario.scenario_type == LIQUIDITY:
        days = liquidity["stressed_liquidation_days"]
        return liquidity.loc[days.idxmax(), "symbol"] if days.notna().any() else None
    held = impacts[impacts["weight"] > 0]
    return held.loc[held["stressed_return"].idxmin(), "symbol"] if len(held) else None


def _assumptions(scenario, baseline, participation_rate, min_observations):
    notes = ["Hypothetical scenario: a model assumption, not a forecast; no probability "
             "is attached.",
             "Shocks are absolute returns added to the base return (a -10% shock turns "
             "+2% into -8%); historical data is not modified.",
             "Base return: " + ("0 (scenario shock only)." if baseline == BASELINE_ZERO else
                                "mean daily return on the portfolio's common dates.")]
    if scenario.volatility_multiplier is not None:
        notes.append(f"Every holding falls by {scenario.volatility_multiplier:g} of its own "
                     f"daily standard deviations at once (no diversification), estimated from "
                     f"at least {min_observations} common daily returns.")
    if scenario.liquidity_multiplier is not None:
        notes.append(f"Trading capacity: ADTV x {scenario.liquidity_multiplier:g} at a "
                     f"{participation_rate:.0%} participation rate; prices are unchanged by "
                     "the liquidity shock.")
    return tuple(notes)


def _unavailable(scenario, message, value, assumptions):
    return StressResult(
        scenario_name=scenario.name, scenario_type=scenario.scenario_type,
        description=scenario.description, status=STATUS_UNAVAILABLE, message=message,
        portfolio_return=math.nan, portfolio_loss=math.nan, portfolio_value=value,
        portfolio_loss_amount=None, stressed_portfolio_value=None, symbol_impacts=None,
        liquidity_impacts=None, largest_negative_contributor=None,
        largest_negative_contribution=math.nan, most_exposed_holding=None,
        assumptions=assumptions, scenario=scenario)

