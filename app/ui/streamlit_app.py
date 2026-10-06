"""ExitSafe - Analytics Test Console (manual verification only, not the final dashboard).

    streamlit run app/ui/streamlit_app.py

A thin presentation layer: every calculation comes from the data and analytics
layers via app.ui.console.
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
    EXAMPLE_FILE_NAME,
    MULTI_SYMBOL_SAMPLE_CSV,
    PASTED_FILE_NAME,
    SAMPLE_CSV,
    SAMPLE_SYMBOL,
    UPLOAD_TYPES,
    alignment_summary,
    canonical_column_order,
    conditional_value_at_risk,
    cvar_display,
    default_holdings_text,
    drawdown_chart_data,
    format_percent,
    import_summary,
    liquidity_display,
    liquidity_summary,
    matrix_display,
    maximum_drawdown_display,
    parse_holdings,
    pasted_bytes,
    portfolio_holdings_display,
    portfolio_risk,
    portfolio_summary_display,
    position_display,
    position_liquidity,
    ratios_display,
    risk_adjusted_ratios,
    value_at_risk,
    var_display,
    run_import,
    status_level,
    uses_estimated_traded_value,
    volatility_display,
)

UPLOAD_OR_PASTE = "Upload or paste data"
SAMPLE_ONE = f"Sample: one stock ({SAMPLE_SYMBOL})"
SAMPLE_THREE = "Sample: three stocks (ABC, LMN, XYZ)"


def main():
    st.set_page_config(page_title="ExitSafe - Analytics Test Console")
    st.title("ExitSafe — Analytics Test Console")
    st.caption("Manual verification of market-data import, volatility and "
               "covariance/correlation analysis")

    source = st.radio("Data", [UPLOAD_OR_PASTE, SAMPLE_ONE, SAMPLE_THREE], horizontal=True)
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
    show_portfolio_risk(outcome, confidence, minimum, periods, participation)


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
        return
    result, error = portfolio_risk(data, holdings, value, confidence, minimum, periods,
                                   participation)
    if error:
        st.error(error)
        return
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


if __name__ == "__main__":
    main()
