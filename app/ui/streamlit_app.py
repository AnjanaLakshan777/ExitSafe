"""ExitSafe analytics test console, for manual checks (not the final dashboard).

Run with: streamlit run app/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# `streamlit run` puts this file's folder on sys.path, not the project root.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.analytics.volatility import TRADING_DAYS_PER_YEAR  # noqa: E402
from app.ui.console import (  # noqa: E402
    EXAMPLE_CSV,
    BACKTEST_SAMPLE_CSV,
    BASELINE_LABELS,
    DEFAULT_SCENARIOS,
    EXAMPLE_FILE_NAME,
    EXIT_STATUS_LABELS,
    EXIT_STRESS_SCENARIOS,
    INDEX_SAMPLE_CSV,
    KNOWN_INDEXES,
    MULTI_SYMBOL_SAMPLE_CSV,
    PASTED_FILE_NAME,
    SAMPLE_CSV,
    SAMPLE_SYMBOL,
    REGIME_UNDEFINED,
    UPLOAD_TYPES,
    alignment_summary,
    backtest_comparison_display,
    backtest_final_weights_display,
    backtest_log_display,
    canonical_column_order,
    conditional_value_at_risk,
    custom_stress_scenario,
    cvar_display,
    default_holdings_text,
    drawdown_chart_data,
    exit_plan_display,
    exit_safety,
    exit_summary_display,
    format_percent,
    import_summary,
    load_index_data,
    liquidity_display,
    liquidity_summary,
    market_regime,
    matrix_display,
    maximum_drawdown_display,
    optimization_allocation_display,
    optimization_comparison_display,
    optimization_metrics_display,
    optimize,
    parse_holdings,
    parse_sector_mapping,
    pasted_bytes,
    portfolio_holdings_display,
    portfolio_risk,
    portfolio_summary_display,
    position_display,
    position_liquidity,
    ratios_display,
    regime_current_display,
    regime_history_display,
    risk_adjusted_ratios,
    run_backtest,
    value_at_risk,
    var_display,
    run_import,
    status_level,
    stress_impact_display,
    stress_liquidity_display,
    stress_report,
    stress_summary_display,
    uses_estimated_traded_value,
    volatility_display,
)

UPLOAD_OR_PASTE = "Upload or paste data"
SAMPLE_ONE = f"Sample: one stock ({SAMPLE_SYMBOL})"
SAMPLE_THREE = "Sample: three stocks (ABC, LMN, XYZ)"
INDEX_SAMPLE = "Sample: synthetic index series (ASPI, S&P SL20 names)"
SAMPLE_BACKTEST = "Sample: synthetic backtest data (4 stocks, 321 days)"
INDEX_UPLOAD = "Upload index CSV"


def main():
    st.set_page_config(page_title="ExitSafe - Analytics Test Console")
    st.title("ExitSafe — Analytics Test Console")
    st.caption("Manual verification of market-data import, volatility and "
               "covariance/correlation analysis")

    source = st.radio("Data", [UPLOAD_OR_PASTE, SAMPLE_ONE, SAMPLE_THREE, SAMPLE_BACKTEST],
                      horizontal=True)
    use_sample = source != UPLOAD_OR_PASTE
    uploaded = st.file_uploader("Upload CSV (comma, tab, semicolon or pipe separated)",
                                type=UPLOAD_TYPES, disabled=use_sample)
    pasted = st.text_area("...or paste data, header row included (e.g. copied from a spreadsheet "
                          "or web table)", disabled=use_sample or uploaded is not None)
    st.download_button("Download a synthetic example CSV", EXAMPLE_CSV,
                       file_name=EXAMPLE_FILE_NAME, mime="text/csv")

    if source == SAMPLE_ONE:
        name, content, symbol = SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), SAMPLE_SYMBOL
    elif source == SAMPLE_THREE:
        name, content, symbol = (MULTI_SYMBOL_SAMPLE_CSV.name, MULTI_SYMBOL_SAMPLE_CSV.read_bytes(),
                                 None)
    elif source == SAMPLE_BACKTEST:
        name, content, symbol = BACKTEST_SAMPLE_CSV.name, BACKTEST_SAMPLE_CSV.read_bytes(), None
    elif uploaded is not None:
        name, content, symbol = uploaded.name, uploaded.getvalue(), None
    elif pasted_bytes(pasted):
        name, content, symbol = PASTED_FILE_NAME, pasted_bytes(pasted), None
    else:
        st.info("Upload or paste data, or choose a sample above, to begin.")
        return

    # A Symbol column in the data is used as-is. Without one, ask the user:
    # the symbol is never guessed.
    outcome = run_import(name, content, symbol)
    if outcome.needs_symbol:
        prompt = st.empty()
        symbol = st.text_input("Stock Symbol", key="symbol", placeholder="e.g. JKH.N0000",
                               help="Press Enter (or click elsewhere) after typing.")
        if not symbol.strip():
            prompt.warning("This data has no Symbol column. Enter the stock symbol to continue.")
            return
        prompt.caption("This data has no Symbol column; using the symbol entered below.")
        outcome = run_import(name, content, symbol)

    if outcome.error:
        st.error(outcome.error)
        if outcome.error_detail:
            with st.expander("Technical details"):
                st.code(outcome.error_detail)
    if outcome.import_result is None:
        return

    show_import(outcome.import_result)
    if outcome.returns is None:
        return
    show_returns(outcome.returns)
    show_volatility(outcome.volatility)
    show_covariance(outcome)
    show_drawdown(outcome)
    periods = show_ratios(outcome)
    confidence, minimum = show_var(outcome)
    show_cvar(outcome, confidence, minimum)
    participation = show_liquidity(outcome)
    holdings, value = show_portfolio_risk(outcome, confidence, minimum, periods, participation)
    optimizer_settings = show_optimization(outcome, confidence, minimum, periods, value,
                                           participation)
    index_data, index_name = show_market_regime()
    show_stress_testing(outcome, holdings, value, confidence, minimum, participation)
    show_backtesting(outcome, periods, minimum, optimizer_settings, index_data, index_name)
    show_exit_safety(outcome, holdings, value, participation, confidence, minimum, periods,
                     index_data, index_name)


def show_import(result):
    report = result.report
    st.divider()
    st.subheader("Import Summary")
    st.table(pd.DataFrame(import_summary(result), columns=["Item", "Value"]).astype(str))

    show_message = getattr(st, status_level(report.status))
    show_message(f"Validation status: {report.status}")
    for reason in report.failure_reasons:
        st.error(reason)
    for warning in report.warnings:
        st.warning(warning)
    for note in report.notes:
        st.info(note)
    if report.issue_counts:
        st.write("Issue counts")
        st.table(pd.DataFrame(report.issue_counts.items(), columns=["Issue code", "Rows"]))

    st.write("**Detected columns:**", ", ".join(report.detected_columns))
    st.write("**Normalized columns:**", ", ".join(report.normalized_columns))
    st.write("**Column mapping**")
    st.table(pd.DataFrame(report.column_mapping.items(), columns=["CSV column", "Canonical column"]))
    if report.ignored_columns:
        st.write("**Ignored columns**")
        st.table(pd.DataFrame(report.ignored_columns.items(), columns=["CSV column", "Reason"]))

    if result.data is not None:
        st.divider()
        st.subheader("Canonical Data")
        st.dataframe(result.data, column_order=canonical_column_order(result.data), hide_index=True)


def show_returns(returns):
    st.divider()
    st.subheader("Daily Returns")
    st.markdown("**Daily Return = current close / previous close − 1**, per symbol. "
                "The first row of each symbol has no previous close, so it has no return. "
                "Returns touching an INVALID row are left empty.")
    st.dataframe(returns, hide_index=True)


def show_volatility(volatility):
    st.divider()
    st.subheader("Volatility")
    for row in volatility.itertuples():
        st.markdown(f"**{row.symbol}** — Daily Volatility: **{format_percent(row.daily_volatility)}**, "
                    f"Annualized Volatility: **{format_percent(row.annualized_volatility)}** "
                    f"({row.observations} daily returns)")
    st.table(volatility_display(volatility).astype(str))

    with st.expander("How was volatility calculated?"):
        st.markdown(
            "- **Daily volatility** = sample standard deviation of daily returns "
            "(divides by *n − 1*).\n"
            f"- **Annualized volatility** = daily volatility × √{TRADING_DAYS_PER_YEAR}.\n"
            f"- Assumes **{TRADING_DAYS_PER_YEAR} trading days per year**; calendar gaps are "
            "not adjusted for.\n"
            "- At least 2 usable returns are needed; otherwise volatility is shown as n/a.\n"
            "- Percentages are rounded for display only; the calculations are not rounded.\n"
            "- Volatility describes how much returns varied in the past. On its own it does "
            "not mean the investment will lose money.")


def show_covariance(outcome):
    info = outcome.alignment
    st.divider()
    st.subheader("Covariance / Correlation")
    st.table(pd.DataFrame(alignment_summary(info), columns=["Item", "Value"]).astype(str))
    if len(info["symbols"]) == 1:
        st.info("Only one symbol: the matrices are 1 × 1 (its variance and self-correlation). "
                "Use data with several symbols to compare stocks.")
    if info["observations"] < 2:
        st.warning("Fewer than 2 common return observations: covariance and correlation "
                   "cannot be calculated and are shown as n/a.")

    st.markdown("**Daily Covariance Matrix**")
    st.dataframe(matrix_display(outcome.covariance, 8))
    st.markdown(f"**Annualized Covariance Matrix** (daily covariance × {TRADING_DAYS_PER_YEAR})")
    st.dataframe(matrix_display(outcome.annualized_covariance, 8))
    st.markdown("**Correlation Matrix**")
    st.dataframe(matrix_display(outcome.correlation, 4))

    with st.expander("What do covariance and correlation mean?"):
        st.markdown(
            "- **Covariance** shows how two stocks' returns move together. The diagonal is each "
            "stock's variance.\n"
            "- **Correlation** shows the strength and direction of that relationship on a "
            "scale from −1 to +1:\n"
            "  - **+1**: the returns move together strongly\n"
            "  - **0**: little or no linear relationship\n"
            "  - **−1**: the returns move in opposite directions strongly\n"
            "- Both use daily returns and the sample formula (divides by *n − 1*), on dates "
            "where every symbol has a return for the same period.\n"
            f"- Annualized covariance = daily covariance × {TRADING_DAYS_PER_YEAR} (not "
            f"√{TRADING_DAYS_PER_YEAR}, which is used for volatility). Correlation is not "
            "annualized.\n"
            "- n/a means there was not enough data, or a stock's returns never changed.\n"
            "- Covariance will later be used to measure portfolio risk and to optimize "
            "portfolios.")


def show_drawdown(outcome):
    st.divider()
    st.subheader("Maximum Drawdown")
    st.table(maximum_drawdown_display(outcome.maximum_drawdown).astype(str))

    series = outcome.drawdown_series
    if not series.empty:
        st.markdown("**Drawdown over time** (0% = at a running peak)")
        st.line_chart(drawdown_chart_data(series))
        st.markdown("**Drawdown series**")
        st.dataframe(series, hide_index=True)

    with st.expander("What is maximum drawdown?"):
        st.markdown(
            "- **Maximum Drawdown** shows the largest historical fall from a previous peak to a "
            "later low.\n"
            "- **Running peak** = the highest close so far; **drawdown** = close / running peak "
            "− 1.\n"
            "- Example: peak Rs. 120, trough Rs. 90 → drawdown = (90 / 120) − 1 = **−25%**.\n"
            "- The peak always comes before the trough. **Recovery date** is the first later "
            "day the close is back at or above the peak; *not recovered* means it has not "
            "happened yet in this data.\n"
            "- It helps measure how severe a past loss could have been during a decline. It is "
            "history, not a prediction.\n"
            "- INVALID rows are left out and no prices are filled in for missing days. n/a means "
            "fewer than 2 prices, or (for dates) that there was no decline at all.")


def show_ratios(outcome):
    st.divider()
    st.subheader("Sharpe / Sortino")
    left, right = st.columns(2)
    rate_percent = left.number_input("Annual Risk-Free Rate (%)", value=0.0, step=0.25,
                                     format="%.2f", min_value=-99.0, max_value=1000.0)
    periods = right.number_input("Periods per year", value=TRADING_DAYS_PER_YEAR, step=1,
                                 min_value=1)
    st.caption("0.00% is only a calculation default; enter a rate that fits your analysis. "
               "Periods per year also applies to Portfolio Risk Analysis below.")

    ratios, error = risk_adjusted_ratios(outcome.import_result.data, rate_percent, periods)
    if error:
        st.error(error)
        return periods
    st.table(ratios_display(ratios).astype(str))

    with st.expander("What do Sharpe and Sortino mean?"):
        st.markdown(
            "- **Sharpe ratio** shows return earned relative to **total** risk: "
            "annualized excess return / annualized volatility.\n"
            "- **Sortino ratio** shows return earned relative to **downside** risk: "
            "annualized excess return / downside deviation.\n"
            "- **Excess return** = daily return − daily risk-free rate, where the daily rate is "
            "(1 + annual rate)^(1 / periods) − 1. Annualized excess return = its average × "
            "periods per year.\n"
            "- **Downside deviation** = √(average of min(excess return, 0)²) × √(periods per "
            "year), averaged over all days (days above the risk-free rate count as 0).\n"
            "- **Annualized Return** is the compounded (geometric) growth rate, shown for "
            "information; the ratios use the arithmetic excess return. With few days of data "
            "it is extrapolated a long way and can look extreme.\n"
            "- Higher positive values generally indicate better risk-adjusted performance under "
            "the chosen assumptions. They describe the past and are not a buy or sell signal.\n"
            "- n/a means fewer than 2 returns, no variation (Sharpe), or no return below the "
            "risk-free rate (Sortino).")
    return periods


def show_var(outcome):
    st.divider()
    st.subheader("Value at Risk (VaR)")
    left, right = st.columns(2)
    confidence = left.number_input("Confidence Level (%)", value=95.0, step=1.0, format="%.2f",
                                   min_value=0.01, max_value=99.99)
    minimum = right.number_input("Minimum Observations", value=20, step=1, min_value=2)
    st.caption("1-day horizon, per stock. VaR is shown as a positive loss.")

    summary, error = value_at_risk(outcome.import_result.data, confidence, minimum)
    if error:
        st.error(error)
        return confidence, minimum
    for row in summary[~summary["sufficient_data"]].itertuples():
        st.warning(f"Insufficient historical observations for reliable VaR — **{row.symbol}**: "
                   f"Observations: {row.observations}, Minimum required: {row.min_observations}. "
                   "VaR: n/a")
    st.table(var_display(summary).astype(str))

    with st.expander("What does VaR mean?"):
        st.markdown(
            "- **Historical VaR** uses the observed historical return distribution: the loss at "
            "the chosen lower-tail percentile of past daily returns (linear interpolation "
            "between observations).\n"
            "- **Parametric VaR** assumes returns are approximately normally distributed: "
            "−(mean + z × standard deviation), with z from the normal distribution. Real "
            "returns often have fatter tails, so this is an assumption, not a fact.\n"
            "- Example: a **95% 1-day VaR of 3%** means that, based on the selected method and "
            "historical data/model, the estimated 1-day loss threshold is about 3% at 95% "
            "confidence. It is not the maximum possible loss.\n"
            "- VaR does **not** describe how large the loss could be once VaR is exceeded. That "
            "limitation is what the next metric, **CVaR**, addresses.\n"
            "- A small sample says little about rare events, so VaR is shown only when a stock "
            "has at least the minimum number of daily returns. INVALID rows never enter the "
            "returns.")
    return confidence, minimum


def show_cvar(outcome, confidence, minimum):
    st.divider()
    st.subheader("Conditional Value at Risk (CVaR)")
    st.caption(f"CVaR / Expected Shortfall — 1-day, per stock, at {confidence:g}% confidence "
               f"and at least {minimum} observations (the VaR inputs above).")

    summary, error = conditional_value_at_risk(outcome.import_result.data, confidence, minimum)
    if error:
        st.error(error)
        return
    for row in summary[~summary["sufficient_data"]].itertuples():
        st.warning(f"Insufficient historical observations for reliable VaR/CVaR — **{row.symbol}**: "
                   f"Observations: {row.observations}, Minimum required: {row.min_observations}. "
                   "VaR: n/a, CVaR: n/a")
    st.table(cvar_display(summary).astype(str))

    with st.expander("What does CVaR mean?"):
        st.markdown(
            "- **CVaR** (also called **Expected Shortfall**) estimates the average loss in the "
            "tail beyond the VaR confidence threshold.\n"
            "- **VaR tells us where the tail begins. CVaR tells us how severe the tail is on "
            "average.** It is not the maximum possible loss.\n"
            "- **Historical CVaR** averages the worst (100 − confidence)% of the observed daily "
            "returns. *Tail observations* shows how many that is; a fraction such as 1.5 means "
            "the worst day counts fully and the next worst counts half.\n"
            "- **Parametric CVaR** uses the same normal-distribution assumption as parametric "
            "VaR: −(mean − standard deviation × pdf(z) / α).\n"
            "- Both use exactly the same daily returns as VaR, are shown as positive losses, "
            "and are usually at least as large as VaR. They describe the past data or model, "
            "not a prediction.")


def show_liquidity(outcome):
    data = outcome.import_result.data
    st.divider()
    st.subheader("Liquidity Analysis")
    summary = liquidity_summary(data)
    st.table(liquidity_display(summary).astype(str))
    if uses_estimated_traded_value(summary):
        st.info("Where the source reports no turnover, daily traded value is **estimated** as "
                "close × volume. That is an estimate, not official turnover.")

    left, right = st.columns(2)
    value = left.number_input("Position Value (Rs.)", value=20_000_000.0, step=1_000_000.0,
                              min_value=1.0, format="%.2f")
    rate = right.number_input("Participation Rate (%)", value=10.0, step=1.0, min_value=0.01,
                              max_value=100.0, format="%.2f")
    st.caption("Estimated liquidation time — based on assumed participation rate. The same "
               "position value is assessed for each stock separately (not a portfolio).")
    positions, error = position_liquidity(data, value, rate)
    if error:
        st.error(error)
        return rate
    st.table(position_display(positions).astype(str))

    with st.expander("What do these liquidity measures mean?"):
        st.markdown(
            "- **Liquidity** describes how easily an investor can buy or sell a position without "
            "needing an unusually large share of the market's normal trading activity.\n"
            "- **Average Daily Volume**: average number of shares traded per day. **Average "
            "Daily Traded Value (ADTV)**: typical daily monetary trading activity.\n"
            "- **Traded Value Source**: *Actual turnover* when the data reports it for every "
            "day; otherwise *Estimated (close × volume)* for every day — the two are never "
            "mixed.\n"
            "- **Zero Volume Days / Rate**: days with no shares traded. They are kept as real "
            "observations (counting as 0), because they are evidence of illiquidity.\n"
            "- **Position / ADTV** shows the size of the position compared with typical daily "
            "trading activity.\n"
            "- **Estimated Liquidation Days**: an approximate number of trading days needed if "
            "the investor trades at the selected participation rate "
            "(position ÷ (ADTV × participation rate)).\n"
            "- The **10% participation rate is an assumption**, not a guaranteed execution "
            "rule. This is a simplified estimate: there is no bid/ask spread, order-book "
            "depth, market-impact or transaction-cost data yet. It is not a buy or sell "
            "recommendation.")
    return rate


def show_portfolio_risk(outcome, confidence, minimum, periods, participation):
    data = outcome.import_result.data
    st.divider()
    st.subheader("Portfolio Risk Analysis")
    st.caption("Fixed weights that you enter (no optimization). One holding per line: "
               "Symbol, Weight %. The weights must total 100% and are not adjusted "
               "automatically.")
    symbols = sorted(data["symbol"].dropna().unique())
    holdings_text = st.text_area("Holdings (Symbol, Weight %)",
                                 value=default_holdings_text(symbols), height=140)
    value = st.number_input("Portfolio Value (Rs.)", value=20_000_000.0, step=1_000_000.0,
                            min_value=1.0, format="%.2f")
    st.caption(f"Uses the inputs above: {confidence:g}% confidence, at least {minimum} "
               f"observations for VaR/CVaR, {periods} periods per year and a {participation:g}% "
               "participation rate.")

    holdings, error = parse_holdings(holdings_text)
    if error:
        st.error(error)
        return None, value
    result, error = portfolio_risk(data, holdings, value, confidence, minimum, periods,
                                   participation)
    if error:
        st.error(error)
        return None, value
    if result.observations < 2:
        st.warning(f"Fewer than 2 common return observations ({result.observations}): portfolio "
                   "statistics are n/a.")
    elif not result.sufficient_tail_data:
        st.warning("Insufficient historical observations for reliable portfolio VaR/CVaR — "
                   f"Observations: {result.observations}, Minimum required: {minimum}. "
                   "VaR: n/a, CVaR: n/a")
    st.table(portfolio_summary_display(result).astype(str))

    st.markdown("**Portfolio Holdings / Liquidity**")
    st.table(portfolio_holdings_display(result).astype(str))
    st.caption("Liquidation days are estimated for each holding separately at the "
               "participation rate; they are not added up into a portfolio figure.")

    with st.expander("What does portfolio risk mean?"):
        st.markdown(
            "- **Portfolio risk considers how the stocks behave together**, not just each stock "
            "on its own. The portfolio's daily return is the weighted sum of the holdings' "
            "returns, using only dates on which every holding has a return for the same "
            "period (no gaps filled, INVALID rows never bridged).\n"
            "- **Correlation matters**: stocks that do not move together partly offset each "
            "other, so portfolio volatility is usually lower than the weighted average of the "
            "stocks' volatilities. It is calculated as √(wᵀ Σ w × periods per year) from the "
            "covariance matrix Σ.\n"
            "- **Maximum drawdown** follows a portfolio value that starts at 1.0 and grows with "
            "the portfolio's daily returns, at constant weights.\n"
            "- **Portfolio VaR/CVaR are calculated from the portfolio's own historical return "
            "series, not by averaging individual stock VaR/CVaR values.** They are 1-day "
            "figures shown as positive losses.\n"
            "- **HHI shows portfolio concentration**: the sum of the squared weights, from 1/N "
            "for N equal holdings up to 1 for a single stock. Maximum Weight is the largest "
            "holding.\n"
            "- These figures describe the past at constant weights. There is no rebalancing, "
            "transaction cost or market impact, and they are not a buy or sell "
            "recommendation.")
    return holdings, value


def show_optimization(outcome, confidence, minimum, periods, portfolio_value, participation):
    data = outcome.import_result.data
    st.divider()
    st.subheader("Risk-Aware Portfolio Optimization")
    st.caption("Chooses long-only, fully invested weights for the selected stocks. A historical "
               "quantitative allocation model, not a guarantee of future returns.")
    symbols = sorted(data["symbol"].dropna().unique())
    universe = st.multiselect("Stocks to optimize (universe)", symbols, default=symbols)
    left, right = st.columns(2)
    min_percent = left.number_input("Minimum Weight (%)", value=0.0, step=1.0, min_value=0.0,
                                    max_value=100.0, format="%.2f")
    max_percent = right.number_input("Maximum Weight (%)", value=40.0, step=5.0, min_value=0.0,
                                     max_value=100.0, format="%.2f")
    first, second, third = st.columns(3)
    risk_aversion = first.number_input("Risk Aversion", value=1.0, step=0.5, min_value=0.0)
    cvar_weight = second.number_input("CVaR Weight", value=1.0, step=0.5, min_value=0.0)
    return_weight = third.number_input("Return Weight", value=1.0, step=0.5, min_value=0.0)
    enabled = st.checkbox("Enable Liquidity Constraint", value=False)
    limit = None
    if enabled:
        limit = st.number_input("Maximum Position / ADTV", value=10.0, step=1.0, min_value=0.01,
                                format="%.2f")
        st.caption(f"Each position is limited to {limit:g} × its average daily traded value: "
                   f"about {limit / (participation / 100):,.0f} trading days to exit at the "
                   f"{participation:g}% participation rate set in Liquidity Analysis.")
    settings = {"min_percent": min_percent, "max_percent": max_percent,
                "risk_aversion": risk_aversion, "cvar_weight": cvar_weight,
                "return_weight": return_weight, "liquidity_enabled": enabled,
                "max_position_to_adtv": limit}
    st.caption(f"Uses the inputs above: Portfolio Value Rs. {portfolio_value:,.2f}, "
               f"{confidence:g}% confidence, at least {minimum} common observations and "
               f"{periods} periods per year.")

    result, error = optimize(data, universe, min_percent, max_percent, risk_aversion,
                             cvar_weight, return_weight, confidence, minimum, periods,
                             portfolio_value, enabled, limit)
    if error:
        st.error(error)
        return settings
    if result.liquidity_unconstrained_symbols:
        st.warning("No liquidity data, so the liquidity constraint is not applied to: "
                   f"{', '.join(result.liquidity_unconstrained_symbols)}.")

    st.markdown("**Optimized Allocation**")
    st.table(optimization_allocation_display(result).astype(str))
    st.table(optimization_metrics_display(result).astype(str))

    st.markdown("**Equal Weight vs ExitSafe Optimized**")
    st.table(optimization_comparison_display(result).astype(str))
    st.caption("Both are measured on the same historical scenarios. The equal-weight portfolio "
               "is a reference only and does not have to meet the weight or liquidity limits.")

    with st.expander("How does the optimization work?"):
        st.markdown(
            "- **Optimization chooses portfolio weights that satisfy your constraints while "
            "balancing expected return, variance risk and tail risk.** It minimizes "
            "Risk Aversion × variance + CVaR Weight × CVaR − Return Weight × expected return, "
            "all in daily units. The three weights are model settings, not comparable units.\n"
            "- **Expected return** is the historical average daily return on the common dates. "
            "It is an estimate from the past, not a promised return.\n"
            "- **Variance** uses the covariance matrix (wᵀΣw), so it accounts for how the "
            "stocks move together.\n"
            "- **CVaR is included directly in the optimization objective**: it is calculated "
            "from the portfolio's own daily returns on every historical day (scenario), not "
            "from the individual stocks' CVaR values.\n"
            "- **Liquidity constraints prevent a position from becoming too large relative to "
            "normal trading activity**: position ÷ ADTV must stay at or below the limit. ADTV "
            "is reported turnover where available, otherwise an estimate (close × volume).\n"
            "- Weights are long-only (no short selling), fully invested (total 100%) and "
            "between the minimum and maximum weight. If no allocation can meet every "
            "constraint, an error explains which constraint cannot be met.\n"
            "- There are no transaction costs, market impact or rebalancing in this model, and "
            "the allocation is a quantitative result, not a buy or sell recommendation.")
    return settings


def show_market_regime():
    st.divider()
    st.subheader("Market Regime")
    st.caption("Uses a market-index series (for example ASPI or S&P SL20), separate from the "
               "stock data above.")
    source = st.radio("Index data", [INDEX_SAMPLE, INDEX_UPLOAD], horizontal=True)
    chosen = None
    if source == INDEX_SAMPLE:
        st.info("Synthetic sample: invented values for testing the section. They are not real "
                "ASPI or S&P SL20 index levels.")
        file_name, content = INDEX_SAMPLE_CSV.name, INDEX_SAMPLE_CSV.read_bytes()
    else:
        uploaded = st.file_uploader("Index CSV (Date, Index, Close or Price)", type=UPLOAD_TYPES,
                                    key="index_upload")
        chosen = st.selectbox("Index name (used only when the file has no Index column)",
                              KNOWN_INDEXES)
        if uploaded is None:
            st.info("Upload an index price file to see its market regime.")
            return None, None
        file_name, content = uploaded.name, uploaded.getvalue()

    imported, error = load_index_data(file_name, content, chosen)
    if error:
        st.error(error)
        return None, None
    if imported.invalid_rows:
        issues = ", ".join(f"{code} {count}" for code, count in imported.issue_counts.items())
        st.warning(f"{imported.invalid_rows} INVALID row(s) are excluded (never bridged): "
                   f"{issues}.")

    names = imported.index_names
    if len(names) == 1:
        index_name = names[0]
        st.markdown(f"Index: **{index_name}**")
    else:
        index_name = st.selectbox("Index", names)
    result, error = market_regime(imported.data, index_name)
    if error:
        st.error(error)
        return imported.data, index_name

    if result.current.regime == REGIME_UNDEFINED:
        st.warning(f"Warm-up: {index_name} has {result.observations} usable observation(s); a "
                   f"regime needs at least {result.required_observations}. Regime: N/A.")
    else:
        st.caption(f"The first {result.warm_up_rows} observation(s) are warm-up (N/A); regimes "
                   f"start on {result.first_classified_date.date()}.")
    st.table(regime_current_display(result).astype(str))
    st.line_chart(result.history.set_index("date")["close"].rename(f"{index_name} close"))

    st.markdown("**Historical Regime Table**")
    st.dataframe(regime_history_display(result), hide_index=True)
    t = result.thresholds
    st.caption(f"Settings (model assumptions): volatility {result.volatility_window} days vs "
               f"baseline {result.baseline_window} days; elevated above {t.elevated_volatility_ratio:g}×, "
               f"high above {t.high_volatility_ratio:g}×; trend band ±{t.neutral_band:.0%} around "
               f"the {result.trend_window}-day average; stress at a {t.stress_drawdown:.0%} drawdown "
               f"from the {result.drawdown_lookback}-day peak or close ≤ {t.strong_downtrend_ratio:g}× "
               f"the average; recovery window {t.recovery_lookback} days.")

    with st.expander("What does the market regime mean?"):
        st.markdown(
            "- **Market regime describes the current market environment based on volatility, "
            "drawdown and trend.** It describes the past data; it does not predict future "
            "prices, and it does not change any portfolio weights.\n"
            "- **Normal**: none of the conditions below. Volatility is near its usual level "
            "(the trend can still be up, down or neutral).\n"
            "- **High volatility**: recent volatility is clearly above its longer-run baseline, "
            "but the market has not fallen far enough for stress.\n"
            "- **Stress**: volatility is high **and** the index is well below its recent peak "
            "or well below its moving average.\n"
            "- **Recovery**: volatility was elevated recently and is now falling, the index is "
            "no longer in a downtrend, and it is still below its recent peak.\n"
            "- **N/A (warm-up)**: not enough history yet for every rolling window, so no "
            "regime is shown rather than a false Normal.\n"
            "- The rules are checked in the order Stress, Recovery, High volatility, Normal. All "
            "thresholds are configurable model assumptions, not market rules.")
    return imported.data, index_name


def show_stress_testing(outcome, holdings, portfolio_value, confidence, minimum, participation):
    st.divider()
    st.subheader("Stress Testing")
    st.caption("Hypothetical adverse scenarios for the holdings entered in Portfolio Risk "
               f"Analysis (Portfolio Value Rs. {portfolio_value:,.2f}). These are assumptions, "
               "not forecasts.")
    if holdings is None:
        st.info("Enter valid holdings in Portfolio Risk Analysis to run the stress tests.")
        return

    baseline = st.radio("Base return", list(BASELINE_LABELS), horizontal=True,
                        format_func=BASELINE_LABELS.get)
    sector_text = st.text_area("Sector mapping (optional, one per line: Symbol, Sector)",
                               value="", height=100)
    sector_mapping, error = parse_sector_mapping(sector_text)
    if error:
        st.error(error)
        return
    if sector_mapping is None:
        st.caption("No sector mapping entered: sector shocks are unavailable (sectors are never "
                   "guessed).")

    st.markdown("**Custom scenario** (leave a field at its default to switch it off)")
    first, second, third = st.columns(3)
    market_percent = first.number_input("Market shock (%)", value=0.0, step=1.0,
                                        min_value=-100.0, max_value=100.0, format="%.2f")
    sector_percent = second.number_input("Sector shock (%)", value=0.0, step=1.0,
                                         min_value=-100.0, max_value=100.0, format="%.2f")
    sector_name = third.text_input("Sector", value="")
    fourth, fifth = st.columns(2)
    volatility = fourth.number_input("Volatility multiplier (0 = off)", value=0.0, step=0.5,
                                     min_value=0.0, format="%.2f")
    liquidity = fifth.number_input("Liquidity multiplier (1 = off)", value=1.0, step=0.1,
                                   min_value=0.01, max_value=1.0, format="%.2f")
    custom, error = custom_stress_scenario(market_percent, sector_name, sector_percent,
                                           volatility, liquidity)
    if error:
        st.error(error)
        return
    scenarios = list(DEFAULT_SCENARIOS) + ([custom] if custom is not None else [])

    report, error = stress_report(outcome.import_result.data, holdings, portfolio_value,
                                  scenarios, sector_mapping, participation, baseline,
                                  confidence, minimum)
    if error:
        st.error(error)
        return
    st.markdown(f"Historical 1-day VaR: **{format_percent(report.historical_var)}** · "
                f"Historical 1-day CVaR: **{format_percent(report.historical_cvar)}** "
                f"({report.confidence_level:.0%} confidence, {report.historical_observations} "
                "past days). These come from past returns; the scenario losses below are "
                "hypothetical and are not a stressed VaR or CVaR.")
    st.table(stress_summary_display(report).astype(str))
    for result in report.results:
        if result.status != "OK":
            st.warning(f"{result.scenario_name}: {result.message}")
        for note in result.warnings:
            st.info(f"{result.scenario_name}: {note}")

    names = [r.scenario_name for r in report.results]
    chosen = st.selectbox("Scenario details", names,
                          index=len(names) - 1 if custom is not None else 0)
    result = report.results[names.index(chosen)]
    st.caption(result.description)
    if result.status == "OK":
        st.markdown("**Stock Impact Table**")
        st.table(stress_impact_display(result).astype(str))
        if result.liquidity_impacts is not None:
            st.markdown("**Liquidity Impact**")
            st.table(stress_liquidity_display(result).astype(str))
    with st.expander("Assumptions for this scenario"):
        st.markdown("\n".join(f"- {note}" for note in result.assumptions))

    with st.expander("What is stress testing?"):
        st.markdown(
            "- **Stress testing asks what could happen under a hypothetical adverse "
            "scenario.** It applies a chosen shock to the portfolio's holdings.\n"
            "- **These scenarios are assumptions, not forecasts.** No probability is attached, "
            "and they are not calibrated to past events.\n"
            "- **A market shock changes the assumed return environment.** It is an absolute "
            "return shock added to each holding's base return: a -10% shock turns +2% into "
            "-8%. A sector shock applies only to holdings mapped to that sector.\n"
            "- **A volatility shock** moves every holding down by the chosen number of its own "
            "daily standard deviations on the same day (no diversification).\n"
            "- **A liquidity shock changes assumed trading capacity**, not prices: ADTV is "
            "multiplied by the liquidity multiplier and the liquidation days are recalculated.\n"
            "- The portfolio return is the weighted sum of the holdings' stressed returns; "
            "each holding's contribution is weight × stressed return.\n"
            "- Historical VaR/CVaR describe past daily returns; a scenario loss is a separate, "
            "hypothetical number. Neither is a buy or sell recommendation.")


def show_backtesting(outcome, periods, minimum, optimizer_settings, index_data, index_name):
    data = outcome.import_result.data
    st.divider()
    st.subheader("Backtesting")
    st.caption("Walk-forward: train on past data only, then test on the next unseen period. "
               "ExitSafe and MeanVariance use the optimizer settings from Risk-Aware Portfolio "
               "Optimization above.")
    symbols = sorted(data["symbol"].dropna().unique())
    left, right = st.columns(2)
    capital = left.number_input("Initial Capital (Rs.)", value=1_000_000.0, step=100_000.0,
                                min_value=1.0, format="%.2f")
    risk_free = right.number_input("Risk-Free Rate (%) for the backtest", value=0.0, step=0.25,
                                   min_value=-99.0, max_value=1000.0, format="%.2f")
    first, second, third = st.columns(3)
    training = first.number_input("Training Window", value=60, step=5, min_value=2)
    test = second.number_input("Test Window", value=20, step=5, min_value=1)
    rebalance = third.number_input("Rebalance Frequency", value=20, step=5, min_value=1)
    confidence = st.number_input("Confidence Level (%) for the backtest", value=95.0, step=1.0,
                                 min_value=0.01, max_value=99.99, format="%.2f")
    selected = st.multiselect("Backtest stocks", symbols, default=symbols)
    st.caption(f"At least {minimum} common training returns per rebalance (Minimum "
               f"Observations above); {periods} periods per year. Market index: "
               + (f"{index_name} (from Market Regime)." if index_name else "none loaded."))

    key = (outcome.import_result.report.file_name, outcome.import_result.report.rows_read,
           tuple(selected), capital, risk_free, training, test, rebalance, confidence, minimum,
           periods, tuple(sorted(optimizer_settings.items())), index_name,
           None if index_data is None else len(index_data))
    if st.button("Run Backtest"):
        with st.spinner("Running the walk-forward backtest..."):
            result, error = run_backtest(data, selected, capital, training, test, rebalance,
                                         risk_free, confidence, minimum, periods,
                                         optimizer_settings, index_data, index_name)
        st.session_state["backtest"] = (key, result, error)
    stored = st.session_state.get("backtest")
    if stored is None:
        st.info("Choose the settings and click Run Backtest.")
        return
    if stored[0] != key:
        st.info("Settings or data changed: click Run Backtest to update the results.")
        return
    _, result, error = stored
    if error:
        st.error(error)
        return

    st.caption(f"Out-of-sample comparison from {result.comparison_start.date()} (value 1.0) to "
               f"{result.comparison_end.date()}: {result.observations} test days in "
               f"{len(result.windows)} walk-forward windows; {result.excluded_test_days} test "
               "day(s) excluded (no common return, never zero-filled).")
    if result.market_index_status != "AVAILABLE":
        st.info(f"Market index benchmark unavailable: {result.market_index_reason}")
    for window, date, reason in result.skipped_rebalances:
        st.warning(f"Rebalance {window} ({date.date()}) skipped: {reason}.")
    carried = result.rebalance_log[result.rebalance_log["status"] != "REBALANCED"]
    if len(carried):
        st.warning(f"{len(carried)} strategy rebalance(s) did not produce new weights; see the "
                   "Note column of the rebalance log.")

    st.markdown("**Performance Comparison (out-of-sample only)**")
    st.table(backtest_comparison_display(result).astype(str))
    st.markdown("**Equity Curve** (normalized to 1.0)")
    st.line_chart(result.equity_curve.set_index("date"))
    st.markdown("**Rebalance Log**")
    st.dataframe(backtest_log_display(result), hide_index=True)
    st.markdown("**Latest Rebalance Weights**")
    st.table(backtest_final_weights_display(result).astype(str))

    with st.expander("How does the backtest work?"):
        st.markdown(
            "- **Backtesting evaluates a strategy on historical data by repeatedly training on "
            "the past and testing on the next unseen period.**\n"
            "- **Only information available at each decision date is used.** Weights are "
            "chosen after the close of the last training day from that window's data alone, "
            "and apply from the next trading day; positions are then held without trading "
            "until the next rebalance.\n"
            "- ExitSafe uses the full optimizer (variance, CVaR, expected return, limits); "
            "MeanVariance uses the same inputs without CVaR or the liquidity limit; "
            "EqualWeight holds 1/N of each stock; MarketIndex is the index itself.\n"
            "- All metrics use only the out-of-sample test days. Days without a return for "
            "every stock are excluded, not filled.\n"
            "- There are no transaction costs, slippage, taxes or fees, and prices are not "
            "adjusted for corporate actions. A fixed list of stocks can carry survivorship "
            "bias.\n"
            "- **Past performance does not guarantee future performance.** The comparison "
            "shows numbers only; it does not say which strategy to use.")


def show_exit_safety(outcome, holdings, portfolio_value, participation, confidence, minimum,
                     periods, index_data, index_name):
    st.divider()
    st.subheader("Exit Safety Assessment")
    st.caption("Can the requested amount reasonably be exited under the current quantitative "
               "conditions? A decision-support assessment, not a guarantee of execution, price "
               "or future return.")
    if holdings is None:
        st.info("Enter valid holdings in Portfolio Risk Analysis to assess an exit.")
        return

    left, right = st.columns(2)
    target = left.number_input("Target Exit Amount (Rs.)", value=5_000_000.0, step=500_000.0,
                               min_value=0.0, format="%.2f")
    scenario = right.selectbox("Stress Scenario", EXIT_STRESS_SCENARIOS)
    liquidity_stress = st.checkbox("Apply Liquidity -50% (half the trading capacity)")
    with st.expander("Policy settings (model assumptions)"):
        first, second = st.columns(2)
        policy = {
            "max_safe_exit_days": first.number_input("Safe exit horizon (days)", value=5.0,
                                                     step=1.0, min_value=0.1),
            "max_caution_exit_days": second.number_input("Caution exit horizon (days)",
                                                         value=20.0, step=1.0, min_value=0.1),
            "cvar_caution_percent": first.number_input("CVaR caution (%)", value=5.0, step=0.5,
                                                       min_value=0.01),
            "cvar_risk_percent": second.number_input("CVaR at-risk (%)", value=10.0, step=0.5,
                                                     min_value=0.01),
            "stress_caution_percent": first.number_input("Stress loss caution (%)", value=10.0,
                                                         step=1.0, min_value=0.01),
            "stress_risk_percent": second.number_input("Stress loss at-risk (%)", value=20.0,
                                                       step=1.0, min_value=0.01),
            "coverage_minimum_percent": first.number_input("Minimum liquidity coverage (%)",
                                                           value=100.0, step=5.0,
                                                           min_value=0.01, max_value=100.0),
            "insufficient_coverage_percent": second.number_input(
                "Insufficient data below coverage (%)", value=50.0, step=5.0, min_value=0.0,
                max_value=100.0),
            "cvar_measure": st.selectbox("CVaR used by the policy", ["historical", "parametric"]),
        }
    st.caption(f"Uses Portfolio Value Rs. {portfolio_value:,.2f} and the holdings from Portfolio "
               f"Risk Analysis, a {participation:g}% participation rate (Liquidity Analysis), "
               f"{confidence:g}% confidence and the market index from Market Regime "
               f"({index_name or 'none loaded'}).")

    result, error = exit_safety(outcome.import_result.data, holdings, portfolio_value, target,
                                participation, scenario, liquidity_stress, policy, confidence,
                                minimum, periods, index_data, index_name)
    if error:
        st.error(error)
        return
    label = EXIT_STATUS_LABELS[result.overall_status]
    show_status = {"SAFE": st.success, "CAUTION": st.warning, "AT_RISK": st.error,
                   "INSUFFICIENT_DATA": st.info}[result.overall_status]
    show_status(f"Decision-support assessment: **{label}**")
    st.table(exit_summary_display(result).astype(str))

    st.markdown("**Holding Exit Plan** (proportional exit, holdings sold in parallel)")
    st.table(exit_plan_display(result).astype(str))
    st.markdown("**Why?**")
    st.markdown("\n".join(f"- **{r.level.replace('_', ' ')}** ({r.factor.lower()}): {r.message}"
                          for r in result.reasons))

    with st.expander("How is the exit assessed?"):
        st.markdown(
            "- The target is split across holdings in proportion to their weights, and the "
            "holdings are sold in parallel at the participation rate. Each holding needs "
            "planned exit ÷ (ADTV × participation rate) trading days.\n"
            "- **Estimated Exit Horizon** is the longest of those times (holdings run in "
            "parallel, so they are not added up). It is a planning estimate, not a guaranteed "
            "exit time.\n"
            "- **Liquidity Coverage** is the share of the target with usable trading data; "
            "holdings without it are listed, never ignored.\n"
            "- **Reference CVaR Loss** is target × portfolio CVaR: a 1-day reference, not the "
            "loss expected while exiting. **Hypothetical Stress Loss** comes from the chosen "
            "scenario, not a prediction.\n"
            "- The market regime is shown as context and does not change the status. Market "
            "impact, spreads and transaction costs are not modelled.\n"
            "- The thresholds are configurable model settings, not financial standards. SAFE "
            "does not guarantee execution.")


if __name__ == "__main__":
    main()
