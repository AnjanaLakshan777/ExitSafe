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
    format_percent,
    import_summary,
    matrix_display,
    pasted_bytes,
    run_import,
    status_level,
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


if __name__ == "__main__":
    main()
