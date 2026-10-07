"""Plain-language recommendations for the two user journeys: Start Investing and I Already Invested.

This is a transparent, rule-based interpretation layer, not a prediction model. It runs the
existing engines (optimizer, portfolio risk, Exit Safety) through the same console helpers
the dashboard uses, with the dashboard's default settings, and turns their results into an
action and a few evidence points. Every rule reads a status or number those engines
produced. News, threat scans and Gemini results are never inputs here.

Rules (first match wins):

Start Investing
  INSUFFICIENT DATA     the optimizer or the Exit Safety checks lack data
  REVIEW CANDIDATES     no allocation meets the limits, the best mix lost money on average
                        in the data, or the risk of a large loss is at caution or worse
                        (CVaR check), or a stress loss is at risk
  INVEST GRADUALLY      the positions take longer to buy/sell than the safe horizon
                        (Exit Safety liquidity check at caution or worse)
  PROCEED               otherwise

I Already Invested
  INSUFFICIENT DATA     the Exit Safety checks lack data
  CONSIDER STAGED EXIT  a withdrawal was entered and its liquidity check is at caution or worse
  REDUCE EXPOSURE       risk of a large loss at caution or worse, or a stress loss at risk
  REVIEW PORTFOLIO      selling everything takes longer than the safe horizon, or one holding
                        is above the optimizer's default 40% limit
  HOLD                  otherwise

A stress result at CAUTION on its own doesn't change the action: the default scenario is a
market-wide fall, which costs any fully invested stock portfolio about the same.

"Portfolio status" is the Exit Safety Engine's own status for selling the whole portfolio,
with its default thresholds: SAFE is shown as GOOD.
"""

import math
import re
from dataclasses import dataclass, field

from app.analytics.volatility import TRADING_DAYS_PER_YEAR
from app.exit_engine.exit_safety import (AT_RISK, CAUTION, DEFAULT_STRESS_SCENARIO, INFO,
                                         INSUFFICIENT_DATA, OK, ExitSafetyPolicy)
from app.ui.console import (REGIME_LABELS, exit_safety, format_percent, optimize, parse_holdings,
                            portfolio_risk)

# The dashboard's default analysis settings (the calculation sections start from these too).
CONFIDENCE = 95.0
MIN_OBSERVATIONS = 20
PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR
PARTICIPATION = 10.0
_POLICY = ExitSafetyPolicy()
POLICY = {
    "max_safe_exit_days": _POLICY.max_safe_exit_days,
    "max_caution_exit_days": _POLICY.max_caution_exit_days,
    "cvar_caution_percent": _POLICY.cvar_caution_threshold * 100,
    "cvar_risk_percent": _POLICY.cvar_risk_threshold * 100,
    "stress_caution_percent": _POLICY.stress_caution_threshold * 100,
    "stress_risk_percent": _POLICY.stress_risk_threshold * 100,
    "coverage_minimum_percent": _POLICY.coverage_minimum * 100,
    "insufficient_coverage_percent": _POLICY.insufficient_coverage_ratio * 100,
    "cvar_measure": _POLICY.cvar_measure,
}
STRESS_SCENARIO = DEFAULT_STRESS_SCENARIO
DEFAULT_MAX_WEIGHT = 40.0           # the optimizer section's default maximum weight

HORIZONS = {"SHORT": "Less than 1 year", "MEDIUM": "1 to 3 years", "LONG": "More than 3 years"}
# Shorter horizons switch on the optimizer's existing liquidity limit (position / ADTV).
# At the 10% participation rate, 2x ADTV is about 20 trading days to sell, the Exit Safety
# caution horizon; 10x is about 100 days.
HORIZON_LIQUIDITY_LIMIT = {"SHORT": 2.0, "MEDIUM": 10.0, "LONG": None}
RISK_PREFERENCES = {"CONSERVATIVE": "Conservative: avoid large losses",
                    "BALANCED": "Balanced",
                    "GROWTH": "Growth: accept more risk for return"}
# Presets of the optimizer's existing settings; all can be changed under See Calculations.
RISK_PRESETS = {
    "CONSERVATIVE": {"risk_aversion": 2.0, "cvar_weight": 2.0, "return_weight": 0.5,
                     "max_percent": 30.0},
    "BALANCED": {"risk_aversion": 1.0, "cvar_weight": 1.0, "return_weight": 1.0,
                 "max_percent": DEFAULT_MAX_WEIGHT},
    "GROWTH": {"risk_aversion": 0.5, "cvar_weight": 0.5, "return_weight": 2.0,
               "max_percent": 50.0},
}

STATUS_LABELS = {"SAFE": "GOOD", CAUTION: "CAUTION", AT_RISK: "AT RISK",
                 INSUFFICIENT_DATA: "INSUFFICIENT DATA"}

PROCEED = "PROCEED"
INVEST_GRADUALLY = "INVEST_GRADUALLY"
REVIEW_CANDIDATES = "REVIEW_CANDIDATES"
HOLD = "HOLD"
REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
CONSIDER_STAGED_EXIT = "CONSIDER_STAGED_EXIT"
REVIEW_PORTFOLIO = "REVIEW_PORTFOLIO"
NO_DATA = "INSUFFICIENT_DATA"
ACTIONS = {
    PROCEED: ("PROCEED WITH THIS ALLOCATION",
              "The recommended mix passed the main risk and liquidity checks."),
    INVEST_GRADUALLY: ("INVEST GRADUALLY",
                       "Build the positions over several days: some of them are large compared "
                       "with how much of the stock trades each day."),
    REVIEW_CANDIDATES: ("REVIEW THE CANDIDATES",
                        "Look again at the companies or the limits before investing."),
    HOLD: ("HOLD", "No change is needed based on the current evidence."),
    REDUCE_EXPOSURE: ("REDUCE EXPOSURE",
                      "Consider lowering the share of the riskiest holdings."),
    CONSIDER_STAGED_EXIT: ("CONSIDER A STAGED EXIT",
                           "Sell over several days rather than all at once."),
    REVIEW_PORTFOLIO: ("REVIEW YOUR PORTFOLIO",
                       "Nothing is urgent, but part of the portfolio needs a closer look."),
    NO_DATA: ("INSUFFICIENT DATA",
              "There isn't enough price or trading history for a reliable recommendation."),
}
_SEVERITY = {INSUFFICIENT_DATA: 4, AT_RISK: 3, CAUTION: 2, OK: 1, INFO: 0}


@dataclass
class Recommendation:
    action: str
    reasons: list                       # 2-5 evidence points, each with numbers from the engines
    status: str | None = None           # GOOD / CAUTION / AT RISK / INSUFFICIENT DATA

    @property
    def title(self):
        return ACTIONS[self.action][0]

    @property
    def meaning(self):
        return ACTIONS[self.action][1]


@dataclass
class NewInvestmentPlan:
    settings: dict                      # the optimizer settings used
    notes: list = field(default_factory=list)
    optimization: object = None         # OptimizationResult
    holdings: list | None = None        # (symbol, weight) as used by the risk checks
    holdings_text: str = ""
    risk: object = None                 # PortfolioRiskResult
    health: object = None               # ExitSafetyResult for selling the whole portfolio
    error: str | None = None
    recommendation: Recommendation | None = None


@dataclass
class PortfolioReview:
    holdings: list | None = None
    risk: object = None
    health: object = None               # ExitSafetyResult for selling the whole portfolio
    exit: object = None                 # ExitSafetyResult for the withdrawal, if one was entered
    error: str | None = None
    recommendation: Recommendation | None = None


def worst_levels(result):
    """{factor: most severe level} over the Exit Safety reasons."""
    levels = {}
    for reason in result.reasons:
        if _SEVERITY.get(reason.level, 0) > _SEVERITY.get(levels.get(reason.factor), -1):
            levels[reason.factor] = reason.level
    return levels


def _at_least(level, threshold):
    return _SEVERITY.get(level, 0) >= _SEVERITY[threshold]


def optimizer_settings(risk_preference, horizon, candidates, max_percent=None):
    """Optimizer settings for a risk preference and horizon, and notes on any adjustment."""
    preset = RISK_PRESETS[risk_preference]
    limit = HORIZON_LIQUIDITY_LIMIT[horizon]
    notes = []
    max_percent = preset["max_percent"] if max_percent is None else float(max_percent)
    if candidates and len(candidates) * max_percent < 100:
        raised = math.ceil(10000 / len(candidates)) / 100
        notes.append(f"With {len(candidates)} companies the most per company was raised from "
                     f"{max_percent:g}% to {raised:g}%, so the whole amount can be invested.")
        max_percent = raised
    return {"min_percent": 0.0, "max_percent": max_percent,
            "risk_aversion": preset["risk_aversion"], "cvar_weight": preset["cvar_weight"],
            "return_weight": preset["return_weight"], "liquidity_enabled": limit is not None,
            "max_position_to_adtv": limit}, notes


def holdings_text(weights):
    """Weights as 'SYMBOL, weight %' lines totalling exactly 100 (the last line takes the
    rounding remainder), the format Portfolio Risk Analysis reads."""
    items = sorted(weights.items())
    if not items:
        return ""
    rounded = [round(w * 100, 4) for _, w in items[:-1]]
    rounded.append(round(100 - sum(rounded), 4))
    return "\n".join(f"{s}, {w:.4f}" for (s, _), w in zip(items, rounded))


def _assess(data, holdings, value, target, index_data, index_name):
    return exit_safety(data, holdings, value, target, PARTICIPATION, STRESS_SCENARIO, False,
                       POLICY, CONFIDENCE, MIN_OBSERVATIONS, PERIODS_PER_YEAR, index_data,
                       index_name)


def _risk(data, holdings, value):
    return portfolio_risk(data, holdings, value, CONFIDENCE, MIN_OBSERVATIONS, PERIODS_PER_YEAR,
                          PARTICIPATION)


# --- evidence points ---

def _money(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"Rs. {value:,.0f}"


def _days(value):
    return "n/a" if value is None or math.isnan(value) else f"{value:,.1f}"


def _bottleneck(result):
    rows = [h for h in result.holdings if h.liquidity_status == "OK"]
    return max(rows, key=lambda h: h.estimated_exit_days, default=None)


def liquidity_point(health, buying=False):
    action = "Buying or later selling these positions" if buying else "Selling the whole portfolio"
    if math.isnan(health.assessed_exit_days):
        return f"Estimated time to sell: not available ({action.lower()} lacks trading data)."
    text = (f"Estimated time to {'trade' if buying else 'sell'}: {action} would take about "
            f"{_days(health.assessed_exit_days)} trading days at "
            f"{health.participation_rate:.0%} of normal daily trading")
    slowest = _bottleneck(health)
    if slowest is not None:
        text += f"; {slowest.symbol} is the slowest"
    return text + (f" (up to {_POLICY.max_safe_exit_days:g} days counts as easy to sell; "
                   "estimated liquidation days).")


def tail_point(risk, value):
    if not risk.sufficient_tail_data or math.isnan(risk.historical_cvar):
        return ("Risk of large loss: not enough daily history to estimate it "
                f"({risk.observations} common days, {risk.min_observations} needed).")
    return (f"Risk of large loss: on the worst {1 - risk.confidence_level:.0%} of days in the "
            f"data, the portfolio lost {format_percent(risk.historical_cvar)} on average in a "
            f"day, about {_money(risk.historical_cvar * value)} (CVaR, Expected Shortfall).")


def stress_point(health):
    if math.isnan(health.stress_loss):
        return f"Stress test ({health.stress_scenario}): not available."
    return (f"Stress test: in a hypothetical '{health.stress_scenario}' scenario the portfolio "
            f"would lose {format_percent(health.stress_loss)}, about "
            f"{_money(health.stress_loss_amount)}. A what-if, not a forecast.")


def concentration_point(risk):
    weights = dict(zip(risk.holdings["symbol"], risk.holdings["weight"]))
    held = {s: w for s, w in weights.items() if w > 0}
    largest = max(held, key=held.get)
    text = (f"Diversification: {len(held)} companies; the largest is {largest} at "
            f"{format_percent(held[largest])}. Concentration (HHI) {risk.hhi:.2f}, where "
            f"{1 / len(held):.2f} would be an even split.")
    if risk.maximum_weight > DEFAULT_MAX_WEIGHT / 100 + 1e-9:
        text += f" That is above the {DEFAULT_MAX_WEIGHT:g}% per-company limit ExitSafe uses by default."
    return text


def regime_point(health):
    if health.market_regime not in REGIME_LABELS:
        return ("Market regime: not available (no market index chosen). It is context only "
                "and never changes the result.")
    return (f"Market regime: {REGIME_LABELS[health.market_regime]} ({health.regime_index}). "
            "Context only; it doesn't change the result.")


def exit_point(result):
    return (f"Withdrawing {_money(result.target_exit_value)}: Exit Safety says "
            f"{STATUS_LABELS[result.overall_status].replace('GOOD', 'SAFE')}; about "
            f"{_days(result.assessed_exit_days)} trading days, "
            f"{format_percent(result.coverage_ratio)} of it with trading data.")


def return_point(optimization):
    return (f"Historical average return of this mix: "
            f"{format_percent(optimization.expected_annual_return)} a year, from "
            f"{optimization.start_date.date()} to {optimization.end_date.date()}. Measured on "
            "past prices; it is not a forecast.")


def diversification_point(optimization):
    mix, even = optimization.metrics, optimization.equal_weight_metrics
    return (f"Compared with splitting the money equally: volatility "
            f"{format_percent(mix.annualized_volatility)} vs "
            f"{format_percent(even.annualized_volatility)}, risk of large loss (CVaR) "
            f"{format_percent(mix.historical_cvar)} vs {format_percent(even.historical_cvar)}.")


def _ordered(first, others, limit=5):
    points = [p for p in [*first, *others] if p]
    unique = list(dict.fromkeys(points))
    return unique[:limit]


# --- the two journeys ---

def recommend_new(optimization, risk, health, capital):
    """Start Investing: the action for the recommended mix and the evidence behind it."""
    levels = worst_levels(health)
    status = STATUS_LABELS[health.overall_status]
    points = {"liquidity": liquidity_point(health, buying=True),
              "tail": tail_point(risk, capital), "stress": stress_point(health),
              "return": return_point(optimization),
              "spread": diversification_point(optimization),
              "regime": regime_point(health)}
    if health.overall_status == INSUFFICIENT_DATA:
        insufficient = [r.message for r in health.reasons if r.level == INSUFFICIENT_DATA]
        return Recommendation(NO_DATA, _ordered(insufficient, [points["tail"]]), status)
    if optimization.expected_annual_return < 0:
        return Recommendation(REVIEW_CANDIDATES, _ordered(
            [points["return"], "Even the best mix of these companies lost value on average in "
             "the data, so there is no historical support for investing in them now."],
            [points["tail"], points["liquidity"], points["regime"]]), status)
    if _at_least(levels.get("TAIL_RISK"), CAUTION) or levels.get("STRESS") == AT_RISK:
        return Recommendation(REVIEW_CANDIDATES, _ordered(
            [points["tail"], points["stress"]],
            [points["return"], points["liquidity"], points["regime"]]), status)
    if _at_least(levels.get("LIQUIDITY"), CAUTION) or _at_least(levels.get("COVERAGE"), CAUTION):
        return Recommendation(INVEST_GRADUALLY, _ordered(
            [points["liquidity"]],
            [points["tail"], points["return"], points["spread"], points["regime"]]), status)
    return Recommendation(PROCEED, _ordered(
        [points["spread"], points["tail"]],
        [points["liquidity"], points["return"], points["regime"]]), status)


def recommend_existing(risk, health, exit_result=None):
    """I Already Invested: the action for the current portfolio and the evidence behind it."""
    levels = worst_levels(health)
    status = STATUS_LABELS[health.overall_status]
    value = health.portfolio_value
    points = {"liquidity": liquidity_point(health), "tail": tail_point(risk, value),
              "stress": stress_point(health), "spread": concentration_point(risk),
              "regime": regime_point(health),
              "exit": exit_point(exit_result) if exit_result is not None else None}
    if health.overall_status == INSUFFICIENT_DATA:
        insufficient = [r.message for r in health.reasons if r.level == INSUFFICIENT_DATA]
        return Recommendation(NO_DATA, _ordered(insufficient, [points["tail"]]), status)
    if exit_result is not None:
        if exit_result.overall_status == INSUFFICIENT_DATA:
            insufficient = [r.message for r in exit_result.reasons if r.level == INSUFFICIENT_DATA]
            return Recommendation(NO_DATA, _ordered([points["exit"], *insufficient], []), status)
        exit_levels = worst_levels(exit_result)
        if (_at_least(exit_levels.get("LIQUIDITY"), CAUTION)
                or _at_least(exit_levels.get("COVERAGE"), CAUTION)):
            slowest = _bottleneck(exit_result)
            return Recommendation(CONSIDER_STAGED_EXIT, _ordered(
                [points["exit"],
                 f"{slowest.symbol} is the slowest to sell: about "
                 f"{_days(slowest.estimated_exit_days)} trading days for its share of the "
                 "withdrawal." if slowest else None],
                [points["tail"], points["stress"], points["regime"]]), status)
    if _at_least(levels.get("TAIL_RISK"), CAUTION) or levels.get("STRESS") == AT_RISK:
        return Recommendation(REDUCE_EXPOSURE, _ordered(
            [points["tail"], points["stress"]],
            [points["spread"], points["exit"], points["regime"]]), status)
    concentrated = risk.maximum_weight > DEFAULT_MAX_WEIGHT / 100 + 1e-9
    if (_at_least(levels.get("LIQUIDITY"), CAUTION) or _at_least(levels.get("COVERAGE"), CAUTION)
            or concentrated):
        first = [points["liquidity"]] if _at_least(levels.get("LIQUIDITY"), CAUTION) else []
        return Recommendation(REVIEW_PORTFOLIO, _ordered(
            first + ([points["spread"]] if concentrated else []),
            [points["exit"], points["tail"], points["stress"], points["regime"]]), status)
    return Recommendation(HOLD, _ordered(
        [points["tail"], points["liquidity"]],
        [points["exit"], points["spread"], points["stress"], points["regime"]]), status)


def plan_new_investment(data, candidates, capital, horizon, risk_preference, max_percent=None,
                        index_data=None, index_name=None):
    """Run the optimizer and the risk checks for a new investor's candidates."""
    candidates = sorted(candidates)
    settings, notes = optimizer_settings(risk_preference, horizon, candidates, max_percent)
    plan = NewInvestmentPlan(settings=settings, notes=notes)
    if not (isinstance(capital, (int, float)) and math.isfinite(capital) and capital > 0):
        plan.error = "Enter the amount you want to invest (more than zero)."
        return plan
    if not candidates:
        plan.error = "Choose at least one company to consider."
        return plan

    s = settings
    plan.optimization, error = optimize(
        data, candidates, s["min_percent"], s["max_percent"], s["risk_aversion"],
        s["cvar_weight"], s["return_weight"], CONFIDENCE, MIN_OBSERVATIONS, PERIODS_PER_YEAR,
        capital, s["liquidity_enabled"], s["max_position_to_adtv"])
    if error:
        plan.error = error
        action = NO_DATA if "observation" in error.lower() else REVIEW_CANDIDATES
        plan.recommendation = Recommendation(action, [f"The optimizer couldn't build a portfolio: "
                                                      f"{error}"])
        return plan

    plan.holdings_text = holdings_text(plan.optimization.optimized_weights)
    plan.holdings, _ = parse_holdings(plan.holdings_text)
    plan.risk, error = _risk(data, plan.holdings, capital)
    if not error:
        plan.health, error = _assess(data, plan.holdings, capital, capital, index_data,
                                     index_name)
    if error:
        plan.error = error
        return plan
    plan.recommendation = recommend_new(plan.optimization, plan.risk, plan.health, capital)
    return plan


def review_portfolio(data, holdings_input, portfolio_value, target=None, index_data=None,
                     index_name=None):
    """Run portfolio risk and Exit Safety for an existing investor's holdings."""
    review = PortfolioReview()
    review.holdings, error = parse_holdings(holdings_input)
    if error:
        review.error = error
        return review
    if not (math.isfinite(portfolio_value) and portfolio_value > 0):
        review.error = "Enter the current value of your portfolio (more than zero)."
        return review
    if target is not None and target > portfolio_value:
        review.error = (f"The withdrawal ({_money(target)}) is larger than the current "
                        f"portfolio value ({_money(portfolio_value)}).")
        return review
    review.risk, error = _risk(data, review.holdings, portfolio_value)
    if not error:
        review.health, error = _assess(data, review.holdings, portfolio_value, portfolio_value,
                                       index_data, index_name)
    if not error and target:
        review.exit, error = _assess(data, review.holdings, portfolio_value, target, index_data,
                                     index_name)
    if error:
        review.error = error
        return review
    review.recommendation = recommend_existing(review.risk, review.health, review.exit)
    return review


# --- current value of an existing portfolio ---

SHARES = "SHARES"
VALUES = "VALUES"
HOLDING_UNITS = {SHARES: "Number of shares", VALUES: "Current value (Rs.)"}


@dataclass
class HoldingValue:
    symbol: str                         # as entered
    amount: float                       # shares, or Rs. when entered as values
    price: float | None                 # latest approved close, None if there is none
    price_date: object
    source: str | None
    value: float | None                 # None = missing current valuation data


@dataclass
class Valuation:
    rows: list
    total: float                        # current value of the holdings that could be valued
    missing: list                       # holdings without current valuation data
    holdings_text: str                  # their weights, for the existing risk engines

    @property
    def complete(self):
        return not self.missing


def parse_positions(text):
    """'SYMBOL, number' lines as [(symbol, number)], adding up repeated symbols."""
    totals = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = re.split(r"[\s,;]+", line.strip(), maxsplit=1)
        if len(parts) != 2:
            return None, f"Line {number}: enter one holding per line as 'SYMBOL, number'."
        symbol, amount_text = parts
        try:
            amount = float(amount_text.replace(",", "").replace(" ", ""))
        except ValueError:
            return None, f"Line {number}: '{amount_text}' is not a number."
        if not math.isfinite(amount) or amount <= 0:
            return None, f"Line {number}: the number for {symbol} must be more than zero."
        key = symbol.upper()
        first, total = totals.get(key, (symbol, 0.0))
        totals[key] = (first, total + amount)
    if not totals:
        return None, "Enter at least one holding."
    return list(totals.values()), None


def latest_prices(data):
    """{symbol: (close, date, source)} from the latest row the analysis uses for each symbol.

    ``data`` is the imported data after the provenance selection, so secondary AI-sourced
    prices are only here when the user chose to include them. INVALID rows never count.
    """
    usable = data[(data["validation_status"].astype("string") != "INVALID")
                  & data["date"].notna() & (data["close"] > 0)]
    latest = usable.sort_values(["symbol", "date"], kind="stable").groupby("symbol").tail(1)
    return {str(row.symbol): (float(row.close), row.date, row.source)
            for row in latest.itertuples()}


def value_holdings(data, positions, units=SHARES):
    """Current value of each holding: shares × latest approved close, or the value entered.

    A holding without market data gets no value (it is never given a made-up price) and is
    listed in ``missing``; the total and the weights cover the other holdings only.
    """
    prices = latest_prices(data)
    by_upper = {s.upper(): s for s in prices}
    rows, values = [], {}
    for symbol, amount in positions:
        match = by_upper.get(symbol.upper())
        price, day, source = prices[match] if match else (None, None, None)
        value = None if match is None else (amount * price if units == SHARES else amount)
        rows.append(HoldingValue(symbol, amount, price, day, source, value))
        if value is not None:
            values[match] = values.get(match, 0.0) + value
    total = sum(values.values())
    text = holdings_text({s: v / total for s, v in values.items()}) if total else ""
    return Valuation(rows, total, [r.symbol for r in rows if r.value is None], text)
