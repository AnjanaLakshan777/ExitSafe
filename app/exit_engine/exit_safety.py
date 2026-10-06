"""Exit Safety Engine: can a requested amount reasonably be sold from the portfolio?

The target is split across holdings by weight, and the existing liquidity, risk,
stress and regime modules are combined into a SAFE / CAUTION / AT_RISK /
INSUFFICIENT_DATA assessment. It supports decisions; it doesn't guarantee an exit.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.analytics.common import TRADING_DAYS_PER_YEAR
from app.analytics.liquidity import (
    DEFAULT_PARTICIPATION_RATE,
    calculate_position_liquidity,
    validate_participation_rate,
)
from app.analytics.portfolio_risk import calculate_portfolio_risk_summary, validate_weights
from app.analytics.var import DEFAULT_CONFIDENCE_LEVEL, DEFAULT_MIN_OBSERVATIONS
from app.regime.regime_detector import UNDEFINED as REGIME_UNDEFINED
from app.regime.regime_detector import detect_market_regime
from app.stress_testing.stress_engine import (
    DEFAULT_SCENARIOS,
    LIQUIDITY,
    MARKET,
    STATUS_OK,
    VOLATILITY,
    StressScenario,
    run_liquidity_stress_test,
    run_stress_scenario,
)

SAFE, CAUTION, AT_RISK, INSUFFICIENT_DATA = "SAFE", "CAUTION", "AT_RISK", "INSUFFICIENT_DATA"
LIQUIDITY_OK, NO_LIQUIDITY_DATA, NO_EXIT = "OK", "NO_LIQUIDITY_DATA", "NO_EXIT"
LIQUIDITY_DATA_COMPLETE, LIQUIDITY_DATA_INCOMPLETE = "COMPLETE", "LIQUIDITY_DATA_INCOMPLETE"
REGIME_DATA_UNAVAILABLE = "REGIME_DATA_UNAVAILABLE"
INFO, OK = "INFO", "OK"
MARKET_IMPACT_NOT_MODELLED = "Not modelled in the current Exit Safety version."
EXIT_HORIZON_LABEL = "Estimated exit horizon under proportional parallel liquidation"
REFERENCE_CVAR_LABEL = "Reference 1-day CVaR loss magnitude on target-exit amount"
REFERENCE_STRESS_LABEL = "Reference stress loss on target-exit amount"
PRICE_STRESS_SCENARIOS = {s.name: s for s in DEFAULT_SCENARIOS
                          if s.scenario_type in (MARKET, VOLATILITY)}
DEFAULT_STRESS_SCENARIO = "Market -10%"
CVAR_MEASURES = ("historical", "parametric")
HOLDING_COLUMNS = ["symbol", "weight", "holding_value", "planned_exit_value",
                   "average_daily_traded_value", "traded_value_source", "exit_value_to_adtv",
                   "participation_rate", "estimated_exit_days", "stressed_exit_days",
                   "liquidity_status", "liquidity_note"]


@dataclass(frozen=True)
class ExitSafetyPolicy:
    """Initial model settings (assumptions, not financial standards)."""
    max_safe_exit_days: float = 5.0
    max_caution_exit_days: float = 20.0
    cvar_caution_threshold: float = 0.05
    cvar_risk_threshold: float = 0.10
    stress_caution_threshold: float = 0.10
    stress_risk_threshold: float = 0.20
    coverage_minimum: float = 1.0
    insufficient_coverage_ratio: float = 0.5
    cvar_measure: str = "historical"

    def __post_init__(self):
        for name in ("max_safe_exit_days", "max_caution_exit_days", "cvar_caution_threshold",
                     "cvar_risk_threshold", "stress_caution_threshold", "stress_risk_threshold"):
            value = getattr(self, name)
            _finite(value, name)
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        for low, high in (("max_safe_exit_days", "max_caution_exit_days"),
                          ("cvar_caution_threshold", "cvar_risk_threshold"),
                          ("stress_caution_threshold", "stress_risk_threshold")):
            if getattr(self, low) > getattr(self, high):
                raise ValueError(f"{low} ({getattr(self, low)!r}) cannot exceed {high} "
                                 f"({getattr(self, high)!r})")
        _finite(self.coverage_minimum, "coverage_minimum")
        _finite(self.insufficient_coverage_ratio, "insufficient_coverage_ratio")
        if not 0 < self.coverage_minimum <= 1:
            raise ValueError(f"coverage_minimum must be above 0 and at most 1, got "
                             f"{self.coverage_minimum!r}")
        if not 0 <= self.insufficient_coverage_ratio <= self.coverage_minimum:
            raise ValueError("insufficient_coverage_ratio must be between 0 and coverage_minimum, "
                             f"got {self.insufficient_coverage_ratio!r}")
        if self.cvar_measure not in CVAR_MEASURES:
            raise ValueError(f"cvar_measure must be one of {', '.join(CVAR_MEASURES)}, got "
                             f"{self.cvar_measure!r}")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExitHoldingResult:
    symbol: str
    weight: float
    holding_value: float
    planned_exit_value: float
    average_daily_traded_value: float
    traded_value_source: str | None
    exit_value_to_adtv: float
    participation_rate: float
    estimated_exit_days: float
    stressed_exit_days: float          # NaN unless a liquidity stress is applied
    liquidity_status: str              # OK / NO_LIQUIDITY_DATA / NO_EXIT (zero weight)
    liquidity_note: str


@dataclass(frozen=True)
class ExitReason:
    factor: str                        # LIQUIDITY, COVERAGE, TAIL_RISK, STRESS, REGIME, ...
    level: str                         # INSUFFICIENT_DATA / AT_RISK / CAUTION / OK / INFO
    message: str


@dataclass(frozen=True)
class ExitAssessmentInputs:
    """The metrics the policy looks at (what classify_exit_safety needs)."""
    coverage_ratio: float
    exit_days: float                   # the horizon being assessed (stressed if applied)
    cvar: float                        # the policy's CVaR measure; NaN if unavailable
    stress_loss: float                 # NaN if the scenario is unavailable
    uncovered_symbols: tuple = ()
    cvar_unavailable_reason: str = ""
    stress_unavailable_reason: str = ""
    stress_scenario: str = ""


@dataclass(frozen=True)
class ExitSafetyResult:
    overall_status: str
    reasons: tuple                     # ExitReason, in a fixed factor order
    portfolio_value: float
    target_exit_value: float
    remaining_portfolio_value: float
    participation_rate: float
    holdings: tuple                    # ExitHoldingResult per holding
    covered_exit_value: float
    uncovered_exit_value: float
    coverage_ratio: float
    liquidity_data_status: str         # COMPLETE / LIQUIDITY_DATA_INCOMPLETE
    liquidity_data_missing: tuple      # symbols without usable liquidity data
    estimated_exit_days: float         # base horizon (max over holdings with data)
    stressed_exit_days: float          # NaN unless a liquidity stress is applied
    liquidity_stress_multiplier: float | None
    assessed_exit_days: float          # the horizon used by the policy
    historical_var: float
    historical_cvar: float
    parametric_var: float
    parametric_cvar: float
    confidence_level: float
    risk_observations: int
    reference_cvar_loss_amount: float
    stress_scenario: str
    stress_status: str
    stress_return: float
    stress_loss: float
    stress_loss_amount: float
    reference_stress_loss_amount: float
    market_regime: str                 # regime, or REGIME_DATA_UNAVAILABLE
    regime_index: str | None
    regime_date: pd.Timestamp | None
    trend_state: str | None
    current_drawdown: float
    rolling_volatility: float
    regime_note: str
    policy_parameters: dict
    market_impact: str = MARKET_IMPACT_NOT_MODELLED
    exit_horizon_label: str = EXIT_HORIZON_LABEL
    reference_cvar_label: str = REFERENCE_CVAR_LABEL
    reference_stress_label: str = REFERENCE_STRESS_LABEL

    @property
    def reason_text(self):
        return [r.message for r in self.reasons]

    def holdings_table(self):
        return pd.DataFrame([asdict(h) for h in self.holdings], columns=HOLDING_COLUMNS)


def assess_exit_safety(data, weights, portfolio_value, target_exit_value,
                       participation_rate=DEFAULT_PARTICIPATION_RATE,
                       stress_scenario=DEFAULT_STRESS_SCENARIO, liquidity_stress_multiplier=None,
                       policy=None, index_data=None, index_name=None,
                       confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                       min_observations=DEFAULT_MIN_OBSERVATIONS,
                       periods_per_year=TRADING_DAYS_PER_YEAR):
    """Assess selling target_exit_value from the portfolio. Invalid inputs raise ValueError."""
    policy = policy if policy is not None else ExitSafetyPolicy()
    if not isinstance(policy, ExitSafetyPolicy):
        raise ValueError("policy must be an ExitSafetyPolicy")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Market data must be a DataFrame")
    if weights is None:
        raise ValueError("Portfolio weights are required")
    weights = validate_weights(weights, set(data["symbol"].dropna()) if "symbol" in data else None)
    portfolio_value = _amount(portfolio_value, "portfolio_value")
    target = _amount(target_exit_value, "target_exit_value")
    if target > portfolio_value:
        raise ValueError(f"target_exit_value ({target:,.2f}) exceeds portfolio_value "
                         f"({portfolio_value:,.2f})")
    validate_participation_rate(participation_rate)
    scenario = _price_scenario(stress_scenario)

    holdings = _exit_plan(data, weights, portfolio_value, target, participation_rate)
    stressed = {}
    if liquidity_stress_multiplier is not None:
        liquidity = run_liquidity_stress_test(data, weights, liquidity_stress_multiplier,
                                              portfolio_value=target,
                                              participation_rate=participation_rate)
        rows = liquidity.liquidity_impacts.set_index("symbol")["stressed_liquidation_days"]
        stressed = {s: (0.0 if weights[s] == 0 else float(rows[s])) for s in weights}
        holdings = [_with_stressed(h, stressed[h.symbol]) for h in holdings]

    usable = [h for h in holdings if h.liquidity_status in (LIQUIDITY_OK, NO_EXIT)]
    missing = tuple(h.symbol for h in holdings if h.liquidity_status == NO_LIQUIDITY_DATA)
    covered = float(sum(h.planned_exit_value for h in usable))
    coverage = covered / target
    with_days = [h for h in holdings if h.liquidity_status == LIQUIDITY_OK]
    horizon = max((h.estimated_exit_days for h in with_days), default=math.nan)
    stressed_horizon = (max((h.stressed_exit_days for h in with_days), default=math.nan)
                        if stressed else math.nan)
    assessed = stressed_horizon if stressed else horizon

    risk = calculate_portfolio_risk_summary(data, weights, confidence_level=confidence_level,
                                            min_observations=min_observations,
                                            periods_per_year=periods_per_year)
    cvar = risk.historical_cvar if policy.cvar_measure == "historical" else risk.parametric_cvar
    stress = run_stress_scenario(data, weights, scenario, portfolio_value=portfolio_value,
                                 min_observations=min_observations)
    stress_ok = stress.status == STATUS_OK
    regime = _regime(index_data, index_name)

    inputs = ExitAssessmentInputs(
        coverage_ratio=coverage, exit_days=assessed, cvar=cvar,
        stress_loss=stress.portfolio_loss if stress_ok else math.nan,
        uncovered_symbols=missing,
        cvar_unavailable_reason=(
            "" if not math.isnan(cvar) else
            f"Portfolio {policy.cvar_measure} CVaR is unavailable: {risk.observations} common "
            f"daily return(s), {min_observations} required."),
        stress_unavailable_reason="" if stress_ok else f"{scenario.name}: {stress.message}",
        stress_scenario=scenario.name)
    status, reasons = classify_exit_safety(inputs, policy)

    context = []
    if stressed:
        context.append(ExitReason("LIQUIDITY_STRESS", INFO, (
            f"Liquidity stress applied: ADTV x {liquidity_stress_multiplier:g}. The estimated "
            f"exit horizon rises from {_days(horizon)} to {_days(stressed_horizon)} trading days; "
            "the stressed horizon is the one assessed. Prices are not changed by this scenario.")))
    context.append(ExitReason("TAIL_RISK", INFO, (
        f"{REFERENCE_CVAR_LABEL}: Rs. {_money(target * cvar)} (target x {policy.cvar_measure} "
        "CVaR; a 1-day proportional reference, not a forecast of the loss while exiting).")))
    if stress_ok:
        context.append(ExitReason("STRESS", INFO, (
            f"{REFERENCE_STRESS_LABEL}: Rs. {_money(target * stress.portfolio_loss)} "
            f"(scenario {scenario.name}; a hypothetical reference, not a forecast).")))
    context.append(ExitReason("REGIME", INFO, regime["note"]))
    context.append(ExitReason("MARKET_IMPACT", INFO, f"Market impact: {MARKET_IMPACT_NOT_MODELLED}"))
    context.append(ExitReason("ASSUMPTION", INFO, (
        f"Exit plan: the target is split across holdings in proportion to their weights and "
        f"sold in parallel at a {participation_rate:.0%} participation rate (an assumption, not "
        "an execution guarantee).")))

    return ExitSafetyResult(
        overall_status=status, reasons=tuple(reasons + context),
        portfolio_value=portfolio_value, target_exit_value=target,
        remaining_portfolio_value=portfolio_value - target,
        participation_rate=float(participation_rate), holdings=tuple(holdings),
        covered_exit_value=covered, uncovered_exit_value=target - covered,
        coverage_ratio=coverage,
        liquidity_data_status=LIQUIDITY_DATA_INCOMPLETE if missing else LIQUIDITY_DATA_COMPLETE,
        liquidity_data_missing=missing, estimated_exit_days=horizon,
        stressed_exit_days=stressed_horizon,
        liquidity_stress_multiplier=liquidity_stress_multiplier, assessed_exit_days=assessed,
        historical_var=risk.historical_var, historical_cvar=risk.historical_cvar,
        parametric_var=risk.parametric_var, parametric_cvar=risk.parametric_cvar,
        confidence_level=confidence_level, risk_observations=risk.observations,
        reference_cvar_loss_amount=target * cvar,
        stress_scenario=scenario.name, stress_status=stress.status,
        stress_return=stress.portfolio_return, stress_loss=stress.portfolio_loss,
        stress_loss_amount=(stress.portfolio_loss_amount if stress_ok else math.nan),
        reference_stress_loss_amount=(target * stress.portfolio_loss if stress_ok else math.nan),
        market_regime=regime["regime"], regime_index=regime["index"], regime_date=regime["date"],
        trend_state=regime["trend"], current_drawdown=regime["drawdown"],
        rolling_volatility=regime["volatility"], regime_note=regime["note"],
        policy_parameters=policy.as_dict())


def classify_exit_safety(inputs, policy=None):
    """(status, [ExitReason]) from the assessed metrics, by the documented decision tree."""
    p = policy if policy is not None else ExitSafetyPolicy()
    reasons = []
    insufficient = []
    if inputs.coverage_ratio < p.insufficient_coverage_ratio or math.isnan(inputs.exit_days):
        insufficient.append(ExitReason("COVERAGE", INSUFFICIENT_DATA, (
            f"Only {inputs.coverage_ratio:.1%} of the target exit amount has usable liquidity "
            f"data (without data: {', '.join(inputs.uncovered_symbols) or 'none'}); at least "
            f"{p.insufficient_coverage_ratio:.0%} with at least one holding is needed for an "
            "assessment.")))
    if math.isnan(inputs.cvar):
        insufficient.append(ExitReason("TAIL_RISK", INSUFFICIENT_DATA,
                                       inputs.cvar_unavailable_reason or "Portfolio CVaR is unavailable."))
    if math.isnan(inputs.stress_loss):
        insufficient.append(ExitReason("STRESS", INSUFFICIENT_DATA,
                                       inputs.stress_unavailable_reason or
                                       "The stress scenario is unavailable."))
    reasons.extend(insufficient)

    if not math.isnan(inputs.exit_days):
        days = inputs.exit_days
        if days > p.max_caution_exit_days:
            level, text = AT_RISK, f"above the configured caution threshold of {p.max_caution_exit_days:g} days"
        elif days > p.max_safe_exit_days:
            level, text = CAUTION, (f"above the configured safe threshold of {p.max_safe_exit_days:g} "
                                    f"days (at most {p.max_caution_exit_days:g} for caution)")
        else:
            level, text = OK, f"within the configured safe threshold of {p.max_safe_exit_days:g} days"
        reasons.append(ExitReason("LIQUIDITY", level,
                                  f"Estimated exit horizon is {_days(days)} trading days, {text}."))
    if not insufficient or inputs.coverage_ratio >= p.insufficient_coverage_ratio:
        if inputs.coverage_ratio < p.coverage_minimum:
            reasons.append(ExitReason("COVERAGE", CAUTION, (
                f"Liquidity coverage is {inputs.coverage_ratio:.1%}, below the configured minimum "
                f"of {p.coverage_minimum:.0%}: no usable liquidity data for "
                f"{', '.join(inputs.uncovered_symbols)}, so the exit horizon covers only the "
                "other holdings.")))
        else:
            reasons.append(ExitReason("COVERAGE", OK, (
                f"Liquidity coverage is {inputs.coverage_ratio:.1%} of the target exit amount.")))
    if not math.isnan(inputs.cvar):
        level = (AT_RISK if inputs.cvar >= p.cvar_risk_threshold else
                 CAUTION if inputs.cvar >= p.cvar_caution_threshold else OK)
        reasons.append(ExitReason("TAIL_RISK", level, (
            f"{p.cvar_measure.capitalize()} 1-day CVaR is {inputs.cvar:.2%} (caution at "
            f"{p.cvar_caution_threshold:.0%}, at risk at {p.cvar_risk_threshold:.0%}).")))
    if not math.isnan(inputs.stress_loss):
        level = (AT_RISK if inputs.stress_loss >= p.stress_risk_threshold else
                 CAUTION if inputs.stress_loss >= p.stress_caution_threshold else OK)
        reasons.append(ExitReason("STRESS", level, (
            f"Selected stress scenario {inputs.stress_scenario} implies a {inputs.stress_loss:.1%} "
            f"hypothetical portfolio loss (caution at {p.stress_caution_threshold:.0%}, at risk "
            f"at {p.stress_risk_threshold:.0%}).")))

    levels = {r.level for r in reasons}
    status = (INSUFFICIENT_DATA if INSUFFICIENT_DATA in levels else AT_RISK if AT_RISK in levels
              else CAUTION if CAUTION in levels else SAFE)
    order = {INSUFFICIENT_DATA: 0, AT_RISK: 1, CAUTION: 2, OK: 3}
    return status, sorted(reasons, key=lambda r: order[r.level])


def _exit_plan(data, weights, portfolio_value, target, participation_rate):
    holdings = []
    has_volume = "volume" in data.columns
    for symbol, w in weights.items():
        exit_value = target * w
        base = dict(symbol=symbol, weight=w, holding_value=portfolio_value * w,
                    planned_exit_value=exit_value, participation_rate=float(participation_rate),
                    stressed_exit_days=math.nan)
        if w == 0:
            holdings.append(ExitHoldingResult(**base, average_daily_traded_value=math.nan,
                                              traded_value_source=None, exit_value_to_adtv=0.0,
                                              estimated_exit_days=0.0, liquidity_status=NO_EXIT,
                                              liquidity_note="Nothing to exit (zero weight)."))
            continue
        row = None
        if has_volume:
            table = calculate_position_liquidity(data[data["symbol"] == symbol], exit_value,
                                                 participation_rate)
            row = table.iloc[0] if len(table) else None
        adtv = float(row["average_daily_traded_value"]) if row is not None else math.nan
        days = float(row["estimated_liquidation_days"]) if row is not None else math.nan
        ok = math.isfinite(days)
        note = ("" if ok else "ADTV is zero (no traded value)." if adtv == 0 else
                "No usable volume or turnover data.")
        holdings.append(ExitHoldingResult(
            **base, average_daily_traded_value=adtv,
            traded_value_source=row["traded_value_source"] if row is not None else None,
            exit_value_to_adtv=float(row["position_to_adtv"]) if row is not None else math.nan,
            estimated_exit_days=days, liquidity_status=LIQUIDITY_OK if ok else NO_LIQUIDITY_DATA,
            liquidity_note=note))
    return holdings


def _with_stressed(holding, days):
    values = asdict(holding)
    values["stressed_exit_days"] = days if holding.liquidity_status != NO_LIQUIDITY_DATA else math.nan
    return ExitHoldingResult(**values)


def _price_scenario(stress_scenario):
    if isinstance(stress_scenario, StressScenario):
        if stress_scenario.liquidity_multiplier is not None or stress_scenario.scenario_type == LIQUIDITY:
            raise ValueError("The exit stress scenario must be a price scenario; use "
                             "liquidity_stress_multiplier for a liquidity shock")
        return stress_scenario
    if stress_scenario not in PRICE_STRESS_SCENARIOS:
        raise ValueError(f"Unknown stress scenario {stress_scenario!r}; choose one of "
                         f"{', '.join(PRICE_STRESS_SCENARIOS)}")
    return PRICE_STRESS_SCENARIOS[stress_scenario]


def _regime(index_data, index_name):
    unavailable = {"regime": REGIME_DATA_UNAVAILABLE, "index": index_name, "date": None,
                   "trend": None, "drawdown": math.nan, "volatility": math.nan}
    if index_data is None:
        return {**unavailable, "note": "Market regime: REGIME_DATA_UNAVAILABLE (no market-index "
                                       "data supplied); the regime is never guessed."}
    try:
        result = detect_market_regime(index_data, index_name)
    except ValueError as exc:
        return {**unavailable, "note": f"Market regime: REGIME_DATA_UNAVAILABLE ({exc})."}
    c = result.current
    if c.regime == REGIME_UNDEFINED:
        return {**unavailable, "index": result.index_name,
                "note": (f"Market regime: REGIME_DATA_UNAVAILABLE ({result.index_name} has "
                         f"{result.observations} usable observations; "
                         f"{result.required_observations} are needed).")}
    return {"regime": c.regime, "index": result.index_name, "date": c.date,
            "trend": c.trend_state, "drawdown": c.current_drawdown,
            "volatility": c.rolling_volatility,
            "note": (f"Current market regime is {c.regime} ({result.index_name}, {c.date.date()}; "
                     f"trend {c.trend_state}, drawdown {c.current_drawdown:.1%}, "
                     f"{result.volatility_window}-day volatility {c.rolling_volatility:.2%}). "
                     "It is context only and does not change the status under this policy.")}


def _amount(value, name):
    _finite(value, name)
    if not value > 0:
        raise ValueError(f"{name} must be a positive amount, got {value!r}")
    return float(value)


def _finite(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _days(value):
    return "n/a" if math.isnan(value) else f"{value:.1f}"


def _money(value):
    return "n/a" if math.isnan(value) else f"{value:,.2f}"
