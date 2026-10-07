"""The two user journeys on the dashboard: Start Investing and I Already Invested.

Each one asks for a few plain inputs, then shows the recommendation, why, and the key
numbers. The numbers come from the existing engines through app/ui/decision.py; the
full calculations stay available under See Calculations.
"""

from decimal import Decimal

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.intelligence.market_tracking import ALREADY_INVESTED, START_INVESTING
from app.intelligence.market_tracking import source_description as tracking_source
from app.ui import auth, decision
from app.ui.console import format_percent

MODE_KEY = "mode"
MODE_LABELS = {START_INVESTING: "Start Investing", ALREADY_INVESTED: "I Already Invested"}
STATUS_STYLE = {"GOOD": (st.success, "✅"), "CAUTION": (st.warning, "⚠️"),
                "AT RISK": (st.error, "⛔"), "INSUFFICIENT DATA": (st.info, "ℹ️")}
ACTION_STYLE = {decision.PROCEED: st.success, decision.HOLD: st.success,
                decision.INVEST_GRADUALLY: st.warning, decision.REVIEW_PORTFOLIO: st.warning,
                decision.CONSIDER_STAGED_EXIT: st.warning, decision.REVIEW_CANDIDATES: st.error,
                decision.REDUCE_EXPOSURE: st.error, decision.NO_DATA: st.info}
EXIT_LABELS = {"SAFE": "SAFE", "CAUTION": "CAUTION", "AT_RISK": "AT RISK",
               "INSUFFICIENT_DATA": "INSUFFICIENT DATA"}


def choose_mode():
    """The landing question. Returns the chosen journey, or None while none is chosen."""
    mode = st.session_state.get(MODE_KEY)
    if mode is None:
        st.subheader("What are you trying to do?")
        left, right = st.columns(2)
        with left:
            if st.button("Start Investing", key="mode_start", type="primary",
                         width="stretch"):
                st.session_state[MODE_KEY] = START_INVESTING
                st.rerun()
            st.caption("I have money to invest and some companies in mind. Suggest how to split "
                       "it, and tell me the risks.")
        with right:
            if st.button("I Already Invested", key="mode_invested", type="primary",
                         width="stretch"):
                st.session_state[MODE_KEY] = ALREADY_INVESTED
                st.rerun()
            st.caption("I own shares. Tell me how healthy my portfolio is and whether I could "
                       "sell part of it safely.")
        st.caption("How it works: your information → a simple recommendation → why → the full "
                   "calculations, if you want them. ExitSafe supports decisions; it is not "
                   "financial advice.")
        return None

    left, right = st.columns([4, 1])
    left.markdown(f"**{MODE_LABELS[mode]}**")
    if right.button("Change", key="mode_change"):
        del st.session_state[MODE_KEY]
        st.rerun()
    return mode


def _step(title, caption=None):
    st.header(title)
    if caption:
        st.caption(caption)


def show_recommendation(recommendation, status_label=None):
    title = recommendation.title
    ACTION_STYLE[recommendation.action](f"Suggested action: **{title}** — {recommendation.meaning}")
    if status_label and recommendation.status:
        show, icon = STATUS_STYLE[recommendation.status]
        show(f"{status_label}: **{recommendation.status}**", icon=icon)


def show_why(recommendation):
    _step("Why are we saying this?")
    st.markdown("\n".join(f"- {reason}" for reason in recommendation.reasons))
    st.caption("This is a rule-based reading of ExitSafe's own calculations (optimizer, "
               "portfolio risk, stress test and Exit Safety), not a prediction. News and AI "
               "search results are never used to decide it. Thresholds are model settings, not "
               "guarantees.")


def _metrics(items):
    for column, (label, value, helptext) in zip(st.columns(len(items)), items):
        column.metric(label, value, help=helptext)


def _days(value):
    return "n/a" if value is None or pd.isna(value) else f"{value:,.1f} days"


CVAR_HELP = ("CVaR (Expected Shortfall), 1-day, 95%: the average loss on the worst 5% of days "
             "in the data.")
TIME_HELP = ("Estimated liquidation days: the longest time needed to sell each holding when "
             "trading 10% of its normal daily value (holdings sold in parallel).")


# --- Start Investing ---

def show_start_investing(outcome, index_data=None, index_name=None):
    """Inputs, recommendation and why for a new investor. Returns settings for the
    calculations section, so it starts from the recommended portfolio."""
    data = outcome.import_result.data
    symbols = sorted(data["symbol"].dropna().unique())
    _step("Your investment plan")
    left, right = st.columns(2)
    capital = left.number_input("Amount to invest (Rs.)", value=None, step=500_000.0,
                                min_value=0.0, format="%.2f", key="plan_capital",
                                placeholder="Your available capital")
    horizon = right.radio("How long can you leave the money invested?", list(decision.HORIZONS),
                          format_func=decision.HORIZONS.get, index=1, key="plan_horizon")
    preference = st.radio("Risk preference", list(decision.RISK_PREFERENCES),
                          format_func=decision.RISK_PREFERENCES.get, index=1, horizontal=True,
                          key="plan_risk")
    if not set(st.session_state.get("plan_candidates", [])) <= set(symbols):
        del st.session_state["plan_candidates"]             # new data: consider every company
    candidates = st.multiselect("Companies you're considering", symbols, default=symbols,
                                key="plan_candidates")
    max_percent = st.number_input(
        "Most to put in any one company (%)", min_value=1.0, max_value=100.0, step=5.0,
        value=decision.RISK_PRESETS[preference]["max_percent"], format="%.2f",
        key=f"plan_max_{preference}")
    if capital is None:
        st.info("Enter the amount you want to invest to see your recommendation.")
        return {"portfolio_value": None}
    plan = decision.plan_new_investment(data, candidates, capital, horizon, preference,
                                        max_percent, index_data, index_name)
    for note in plan.notes:
        st.caption(note)

    _step("Your recommendation")
    if plan.recommendation is None:
        st.error(plan.error)
        return {"portfolio_value": capital or None}
    show_recommendation(plan.recommendation, "Risk check of this mix")
    if plan.optimization is not None and plan.holdings is not None:
        _show_allocation(plan, capital)
    show_why(plan.recommendation)
    if plan.risk is not None:
        _step("Important numbers")
        _metrics([
            ("Historical average return", format_percent(plan.optimization.expected_annual_return),
             "Expected annual return from past prices (daily mean × periods per year). Not a "
             "forecast."),
            ("Volatility (a year)", format_percent(plan.risk.annualized_volatility),
             "Annualized volatility: how much the value usually moves."),
            ("Risk of large loss", format_percent(plan.risk.historical_cvar), CVAR_HELP),
            ("Estimated time to trade", _days(plan.health.assessed_exit_days), TIME_HELP),
        ])
    if plan.error and plan.recommendation.action in (decision.REVIEW_CANDIDATES,
                                                     decision.NO_DATA):
        st.caption("Try a longer horizon, a lower amount, a higher limit per company or more "
                   "companies.")
    return {"portfolio_value": capital, "holdings_text": plan.holdings_text or None,
            "optimizer": plan.settings, "universe": candidates, "exit_target": capital}


def _show_allocation(plan, capital):
    st.markdown("#### Your recommended allocation")
    days = dict(zip(plan.risk.holdings["symbol"], plan.risk.holdings["estimated_liquidation_days"]))
    rows = [{"Company": s, "Share": format_percent(w), "Amount (Rs.)": f"{w * capital:,.0f}",
             "Estimated days to buy": _days(days.get(s))} for s, w in plan.holdings if w > 5e-5]
    st.table(pd.DataFrame(rows).set_index("Company"))
    left_out = [s for s, w in plan.holdings if w <= 5e-5]
    if left_out:
        st.caption(f"Not included by the optimizer: {', '.join(left_out)}.")
    st.caption("From ExitSafe's risk-aware optimizer: it balances historical return, everyday "
               "swings and the risk of large losses within your limits.")


# --- I Already Invested ---

def show_already_invested(outcome, index_data=None, index_name=None, investments=()):
    """Current value, portfolio health, the exit question, recommendation and why for an
    existing investor. Returns settings for the calculations section."""
    data = outcome.import_result.data
    _step("Your portfolio", "One holding per line: Symbol, then the number of shares you own "
          "(or, if you prefer, the holding's current value in Rs.).")
    if investments and st.button("Fill in from My Investments", key="fill_investments"):
        text, original = positions_from_investments(investments)
        st.session_state.update({"invested_holdings": text, "invested_original": original,
                                 "holding_units": decision.SHARES})
    units = st.radio("I'm entering", list(decision.HOLDING_UNITS), horizontal=True,
                     format_func=decision.HOLDING_UNITS.get, key="holding_units")
    holdings_input = st.text_area("Your holdings", height=120, key="invested_holdings",
                                  placeholder="JKH.N0000, 1000\nCOMB.N0000, 2500")
    original = st.number_input(
        "Amount you originally invested (Rs., optional)", min_value=0.0, step=100_000.0,
        format="%.2f", key="invested_original", placeholder="For reference only",
        help="What you paid. Shown for comparison; the analysis uses the current value.",
        **({} if "invested_original" in st.session_state else {"value": None}))
    defaults = {"portfolio_value": None, "holdings_text": None, "exit_target": None}
    if not holdings_input.strip():
        st.info("Enter your holdings to see your portfolio's current value, status and "
                "recommendation.")
        return defaults
    positions, error = decision.parse_positions(holdings_input)
    if error:
        st.error(error)
        return defaults

    valuation = decision.value_holdings(data, positions, units)
    _show_valuation(valuation, units)
    if valuation.missing:
        st.warning(f"Missing current valuation data for: {', '.join(valuation.missing)}. "
                   "There is no approved price for them in the market data, so they are left "
                   "out of the current value and the analysis below (no price is assumed). "
                   "Add their price history to include them.")
    if not valuation.total:
        st.error("None of your holdings has a current price, so the portfolio can't be valued.")
        return defaults
    value = valuation.total

    overview = st.container()
    _step("Should I exit?", "Do you want to check whether you can safely exit part of this "
          "portfolio? Enter the amount you may need, or leave it empty.")
    target = st.number_input("Amount you may need to withdraw (Rs.)", min_value=0.0,
                             value=None, step=500_000.0, format="%.2f", key="invested_target",
                             placeholder="Optional")
    review = decision.review_portfolio(data, valuation.holdings_text, value, target or None,
                                       index_data, index_name)
    defaults = {"portfolio_value": value, "holdings_text": valuation.holdings_text,
                "exit_target": target or None}
    with overview:
        _metrics([
            ("Original investment", _money(original) if original else "Not entered",
             "What you paid, as entered. For reference only."),
            ("Current portfolio value" + ("" if valuation.complete else " (incomplete)"),
             _money(value), "Sum of quantity × latest approved price for each holding."),
            ("Target exit", _money(target) if target else "Not entered",
             "The amount you may need to withdraw."),
        ])
    if review.error:
        st.error(review.error)
        return defaults
    if review.exit is not None:
        _show_exit(review.exit)
    else:
        st.caption("Enter an amount above to run the Exit Safety check.")

    with overview:
        _step("Portfolio overview")
        show, icon = STATUS_STYLE[review.recommendation.status]
        show(f"Portfolio status: **{review.recommendation.status}**", icon=icon)
        st.caption("The Exit Safety Engine's checks applied to selling the whole portfolio: "
                   "time to sell, risk of large loss and a hypothetical market fall.")
        regime = review.health.market_regime
        _metrics([
            ("Risk of large loss", format_percent(review.risk.historical_cvar), CVAR_HELP),
            ("Time to sell everything", _days(review.health.assessed_exit_days), TIME_HELP),
            ("Slowest to sell", review.risk.most_illiquid_symbol or "n/a",
             "The holding with the longest estimated liquidation time."),
            ("Largest holding", format_percent(review.risk.maximum_weight),
             f"Maximum weight. HHI (concentration) {review.risk.hhi:.2f}."),
        ])
        _metrics([
            (f"If the market falls ({review.health.stress_scenario})",
             format_percent(review.health.stress_loss),
             "Hypothetical stress loss: a what-if scenario, not a forecast."),
            ("Market regime", decision.REGIME_LABELS.get(regime, "Not available"),
             "From the market index chosen above. Context only."),
        ])

    _step("Recommendation")
    show_recommendation(review.recommendation)       # the status is in the overview above
    show_why(review.recommendation)
    return defaults


def _money(value):
    return f"Rs. {value:,.2f}"


def _show_valuation(valuation, units):
    rows = []
    for r in valuation.rows:
        rows.append({
            "Company": r.symbol,
            "Shares" if units == decision.SHARES else "Value entered (Rs.)": f"{r.amount:,.2f}",
            "Latest price (Rs.)": f"{r.price:,.2f}" if r.price is not None else "—",
            "Price date": str(r.price_date.date()) if r.price is not None else "—",
            "Price source": tracking_source(r.source) if r.price is not None else "—",
            "Current value (Rs.)": (f"{r.value:,.2f}" if r.value is not None
                                    else "Missing current valuation data"),
            "Share of portfolio": (format_percent(r.value / valuation.total)
                                   if r.value is not None and valuation.total else "—"),
        })
    st.table(pd.DataFrame(rows).set_index("Company"))
    st.caption("Latest approved price: the most recent valid closing price in your market data "
               "after the data checks. Secondary AI-sourced prices are only used if you chose to "
               "include them above.")


def _show_exit(result):
    label = EXIT_LABELS[result.overall_status]
    show = {"SAFE": st.success, "CAUTION": st.warning, "AT RISK": st.error,
            "INSUFFICIENT DATA": st.info}[label]
    show(f"Exit Safety for Rs. {result.target_exit_value:,.0f}: **{label}**")
    _metrics([
        ("Estimated exit horizon", _days(result.assessed_exit_days), TIME_HELP),
        ("Liquidity coverage", format_percent(result.coverage_ratio),
         "Share of the amount with usable trading data."),
        ("Risk of large loss", format_percent(result.historical_cvar), CVAR_HELP),
    ])
    _metrics([
        (f"Stress loss ({result.stress_scenario})", format_percent(result.stress_loss),
         "Hypothetical scenario loss on the portfolio, not a forecast."),
        ("Market regime", decision.REGIME_LABELS.get(result.market_regime, "Not available"),
         "Context only; it doesn't change the status."),
    ])
    checks = [r for r in result.reasons if r.level not in ("INFO", "OK")]
    if checks:
        st.markdown("\n".join(f"- **{r.level.replace('_', ' ')}** ({r.factor.lower()}): "
                              f"{r.message}" for r in checks))
    st.caption("Market impact, spreads and transaction costs are not modelled. The full Exit "
               "Safety assessment is under See Calculations.")


def saved_investments(client):
    """The client's saved investments ("My Investments"); empty if the database is unavailable."""
    try:
        return auth._repository().investments_for(client["id"])
    except (SQLAlchemyError, ValueError):
        return []


def positions_from_investments(investments):
    """(holdings lines 'SYMBOL, shares', amount originally invested) from saved investments."""
    shares, paid = {}, 0.0
    for item in investments:
        symbol = item.symbol.strip().upper()
        shares[symbol] = shares.get(symbol, Decimal(0)) + item.quantity
        paid += float(item.quantity * item.purchase_price)
    text = "\n".join(f"{s}, {format(q.normalize(), 'f')}" for s, q in sorted(shares.items()))
    return text, paid
