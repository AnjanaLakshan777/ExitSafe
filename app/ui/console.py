"""Presentation helpers for the analytics test console.

Kept free of Streamlit so they can be unit-tested. All data work is delegated to
the existing layers: CSV import -> canonical data -> validation -> analytics.
Nothing here parses CSV or calculates returns or volatility itself; it only
calls those functions and formats their results for display.
"""

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.analytics.covariance import (
    calculate_annualized_covariance_matrix,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    describe_return_alignment,
)
from app.analytics.drawdown import calculate_drawdown_series, calculate_maximum_drawdown
from app.analytics.cvar import calculate_cvar_summary
from app.analytics.liquidity import (
    ACTUAL_TURNOVER,
    ESTIMATED_TRADED_VALUE,
    calculate_liquidity_summary,
    calculate_position_liquidity,
)
from app.analytics.ratios import calculate_risk_adjusted_ratios
from app.analytics.var import calculate_var_summary
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns
from app.analytics.volatility import calculate_annualized_volatility
from app.config.paths import PROJECT_ROOT
from app.data.loaders.csv_market_loader import (
    MarketDataImportError,
    SymbolRequiredError,
    load_csv_market_data,
)

SAMPLE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_date_price_vol_change.csv"
SAMPLE_SYMBOL = "TEST.N0000"
# Synthetic 3-symbol data (ABC, LMN, XYZ) with a Symbol column, for the matrices.
MULTI_SYMBOL_SAMPLE_CSV = PROJECT_ROOT / "data" / "sample" / "sample_market_data.csv"

CANONICAL_KEY_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "change_pct",
                         "turnover", "estimated_traded_value", "validation_status",
                         "validation_warnings"]
RETURN_COLUMNS = ["date", "symbol", "close", "validation_status", CANONICAL_DAILY_RETURN]

# Invented values in the Date/Price/Open/High/Low/Vol./Change % layout (not market data).
EXAMPLE_CSV = """Date,Price,Open,High,Low,Vol.,Change %
01/16/2026,52.40,52.10,52.90,51.80,1.25M,0.58%
01/15/2026,52.10,51.60,52.30,51.40,980K,0.97%
01/14/2026,51.60,52.20,52.40,51.30,1.10M,-1.15%
01/13/2026,52.20,51.90,52.60,51.70,"1,050,000",0.58%
01/12/2026,51.90,51.50,52.00,51.20,860K,0.39%
"""
EXAMPLE_FILE_NAME = "exitsafe_synthetic_example.csv"
PASTED_FILE_NAME = "pasted_data.txt"
UPLOAD_TYPES = ["csv", "tsv", "txt"]
DELIMITER_NAMES = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}


@dataclass
class ConsoleOutcome:
    import_result: object | None = None   # MarketImportResult
    returns: pd.DataFrame | None = None
    volatility: pd.DataFrame | None = None
    alignment: dict | None = None                    # common return observations used
    covariance: pd.DataFrame | None = None           # daily
    annualized_covariance: pd.DataFrame | None = None
    correlation: pd.DataFrame | None = None
    drawdown_series: pd.DataFrame | None = None
    maximum_drawdown: pd.DataFrame | None = None
    error: str | None = None              # short, user-facing
    error_detail: str | None = None       # technical detail, shown only on request
    needs_symbol: bool = False            # data has no Symbol column: ask the user for one


def run_import(file_name, content, symbol=None):
    """Import CSV bytes with the real importer, then run returns and volatility.

    The bytes are written under their original file name in a temporary folder
    (the importer reads files so it can hash them for provenance) and removed
    afterwards.
    """
    symbol = (symbol or "").strip() or None
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / Path(file_name).name
            path.write_bytes(content)
            result = load_csv_market_data(path, symbol=symbol)
    except SymbolRequiredError as exc:
        return ConsoleOutcome(needs_symbol=True, error=friendly_import_error(exc),
                              error_detail=str(exc))
    except MarketDataImportError as exc:
        return ConsoleOutcome(error=friendly_import_error(exc), error_detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - shown to the user as a short message
        return ConsoleOutcome(error="CSV could not be imported.",
                              error_detail=f"{type(exc).__name__}: {exc}")

    if result.data is None:
        missing = ", ".join(result.validation.missing_columns)
        return ConsoleOutcome(import_result=result,
                              error=f"Required market-data columns are missing: {missing}.")
    return ConsoleOutcome(
        import_result=result,
        returns=calculate_daily_returns(result.data)[RETURN_COLUMNS],
        volatility=calculate_annualized_volatility(result.data),
        alignment=describe_return_alignment(result.data),
        covariance=calculate_covariance_matrix(result.data),
        annualized_covariance=calculate_annualized_covariance_matrix(result.data),
        correlation=calculate_correlation_matrix(result.data),
        drawdown_series=calculate_drawdown_series(result.data),
        maximum_drawdown=calculate_maximum_drawdown(result.data),
    )


def pasted_bytes(text):
    """Pasted table text as file bytes (None when nothing was pasted)."""
    text = (text or "").strip("\r\n")
    return (text + "\n").encode("utf-8") if text.strip() else None


def friendly_import_error(exc):
    message = str(exc)
    if "no symbol was given" in message:
        return "Symbol is required: this data has no Symbol column. Enter the stock symbol."
    if "conflicts with the file's Symbol column" in message:
        return "The symbol entered does not match the Symbol column in the data."
    if message.endswith("is empty"):
        return "CSV could not be imported: the file is empty."
    return "CSV could not be imported."


def import_summary(result):
    """(label, value) rows for the import summary."""
    report = result.report
    date_range = (f"{report.date_min} to {report.date_max}" if report.date_min
                  else "no valid dates")
    return [
        ("File name", report.file_name),
        ("Symbol", ", ".join(report.symbols) or "none"),
        ("Symbol taken from", report.symbol_source or "-"),
        ("Delimiter", DELIMITER_NAMES.get(report.delimiter, repr(report.delimiter))),
        ("Rows read", report.rows_read),
        ("Valid rows", report.valid_rows),
        ("Invalid rows", report.invalid_rows),
        ("Date range", date_range),
        ("Date format detected", report.date_convention),
        ("Validation status", report.status),
    ]


def status_level(status):
    """Streamlit message kind for a dataset validation status."""
    return {"PASS": "success", "WARNING": "warning"}.get(status, "error")


def canonical_column_order(data):
    """Key canonical columns first, then any others; the data itself is untouched."""
    return ([c for c in CANONICAL_KEY_COLUMNS if c in data.columns]
            + [c for c in data.columns if c not in CANONICAL_KEY_COLUMNS])


def format_percent(value, digits=2):
    """0.0124 -> '1.24%'. Display only; NaN -> 'n/a'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def volatility_display(volatility):
    """A new, display-only table with volatility shown as percentages."""
    return pd.DataFrame({
        "Symbol": volatility["symbol"],
        "Observations": volatility["observations"],
        "Daily Volatility": [_vol_text(v) for v in volatility["daily_volatility"]],
        "Annualized Volatility": [_vol_text(v) for v in volatility["annualized_volatility"]],
        "Periods per year": volatility["periods_per_year"],
    })


def _vol_text(value):
    text = format_percent(value)
    return "n/a (needs at least 2 usable returns)" if text == "n/a" else text



def alignment_summary(info):
    """(label, value) rows describing the data behind the covariance/correlation matrices."""
    start, end = info["start_date"], info["end_date"]
    per_symbol = ", ".join(f"{s}: {n}" for s, n in info["returns_per_symbol"].items())
    return [
        ("Symbols", ", ".join(info["symbols"]) or "none"),
        ("Common return observations used", info["observations"]),
        ("Date range used", f"{start.date()} to {end.date()}" if start is not None else "none"),
        ("Usable daily returns per symbol", per_symbol or "none"),
        ("Dates left out: a symbol had no usable return", info["excluded_missing"]),
        ("Dates left out: returns covered different periods", info["excluded_misaligned"]),
    ]


def matrix_display(matrix, decimals):
    """Display-only formatting of a covariance/correlation matrix (values untouched)."""
    return matrix.style.format(f"{{:.{decimals}f}}", na_rep="n/a")


def maximum_drawdown_display(summary):
    """A new, display-only table of each symbol's maximum drawdown event."""
    def day(value):
        return "n/a" if pd.isna(value) else str(pd.Timestamp(value).date())

    def price(value):
        return "n/a" if pd.isna(value) else f"{value:,.2f}"

    return pd.DataFrame({
        "Symbol": summary["symbol"],
        "Maximum Drawdown": [format_percent(v) for v in summary["maximum_drawdown"]],
        "Peak Price": [price(v) for v in summary["peak_price"]],
        "Peak Date": [day(v) for v in summary["peak_date"]],
        "Trough Price": [price(v) for v in summary["trough_price"]],
        "Trough Date": [day(v) for v in summary["trough_date"]],
        "Recovery Date": [day(v) if not pd.isna(v) else
                          ("not recovered" if not pd.isna(t) else "n/a")
                          for v, t in zip(summary["recovery_date"], summary["trough_date"])],
        "Prices used": summary["observations"],
    })


def drawdown_chart_data(series):
    """Date x symbol table of drawdowns, for a line chart (reshaped, not recalculated)."""
    return series.pivot(index="date", columns="symbol", values="drawdown")


def risk_adjusted_ratios(data, risk_free_rate_percent, periods_per_year):
    """(ratios table, error message). The rate is entered in percent (5.0 = 5%)."""
    try:
        return calculate_risk_adjusted_ratios(data, risk_free_rate=risk_free_rate_percent / 100,
                                              periods_per_year=periods_per_year), None
    except ValueError as exc:
        return None, str(exc)


def ratios_display(ratios):
    """A new, display-only table: percentages for returns/risk, 3 decimals for ratios."""
    def ratio(value):
        return "n/a" if pd.isna(value) else f"{value:.3f}"

    return pd.DataFrame({
        "Symbol": ratios["symbol"],
        "Observations": ratios["observations"],
        "Annualized Return": [format_percent(v) for v in ratios["annualized_return"]],
        "Annualized Volatility": [format_percent(v) for v in ratios["annualized_volatility"]],
        "Annualized Excess Return": [format_percent(v) for v in ratios["annualized_excess_return"]],
        "Sharpe Ratio": [ratio(v) for v in ratios["sharpe_ratio"]],
        "Downside Deviation": [format_percent(v) for v in ratios["downside_deviation"]],
        "Sortino Ratio": [ratio(v) for v in ratios["sortino_ratio"]],
    })


def value_at_risk(data, confidence_percent, min_observations):
    """(VaR summary, error message). Confidence is entered in percent (95.0 = 95%)."""
    try:
        return calculate_var_summary(data, confidence_level=confidence_percent / 100,
                                     min_observations=int(min_observations)), None
    except ValueError as exc:
        return None, str(exc)


def var_display(summary):
    """A new, display-only VaR table (losses as positive percentages)."""
    return pd.DataFrame({
        "Symbol": summary["symbol"],
        "Observations": summary["observations"],
        "Minimum required": summary["min_observations"],
        "Confidence": [f"{c * 100:g}%" for c in summary["confidence_level"]],
        "Historical VaR": [format_percent(v) for v in summary["historical_var"]],
        "Parametric VaR": [format_percent(v) for v in summary["parametric_var"]],
    })


def conditional_value_at_risk(data, confidence_percent, min_observations):
    """(CVaR summary incl. VaR, error message). Confidence is entered in percent."""
    try:
        return calculate_cvar_summary(data, confidence_level=confidence_percent / 100,
                                      min_observations=int(min_observations)), None
    except ValueError as exc:
        return None, str(exc)


def cvar_display(summary):
    """A new, display-only VaR/CVaR table (losses as positive percentages)."""
    return pd.DataFrame({
        "Symbol": summary["symbol"],
        "Observations": summary["observations"],
        "Tail observations": [f"{m:g}" for m in summary["tail_mass"]],
        "Historical VaR": [format_percent(v) for v in summary["historical_var"]],
        "Historical CVaR": [format_percent(v) for v in summary["historical_cvar"]],
        "Parametric VaR": [format_percent(v) for v in summary["parametric_var"]],
        "Parametric CVaR": [format_percent(v) for v in summary["parametric_cvar"]],
    })


TRADED_VALUE_SOURCE_LABELS = {ACTUAL_TURNOVER: "Actual turnover",
                              ESTIMATED_TRADED_VALUE: "Estimated (close × volume)"}


def _amount(value, decimals=2):
    return "n/a" if pd.isna(value) else f"{value:,.{decimals}f}"


def liquidity_summary(data):
    """Stock-level liquidity table from the analytics module (unformatted)."""
    return calculate_liquidity_summary(data)


def liquidity_display(summary):
    """A new, display-only stock-level liquidity table."""
    return pd.DataFrame({
        "Symbol": summary["symbol"],
        "Observations": summary["observations"],
        "Average Daily Volume": [_amount(v, 0) for v in summary["average_daily_volume"]],
        "Median Daily Volume": [_amount(v, 0) for v in summary["median_daily_volume"]],
        "Average Daily Traded Value (Rs.)": [_amount(v) for v in summary["average_daily_traded_value"]],
        "Traded Value Source": [TRADED_VALUE_SOURCE_LABELS.get(s, s)
                                for s in summary["traded_value_source"]],
        "Zero Volume Days": summary["zero_volume_days"],
        "Zero Volume Rate": [format_percent(v) for v in summary["zero_volume_rate"]],
    })


def position_liquidity(data, position_value, participation_percent):
    """(position liquidity table, error message). Participation is entered in percent."""
    try:
        return calculate_position_liquidity(data, float(position_value),
                                             participation_percent / 100), None
    except ValueError as exc:
        return None, str(exc)


def position_display(positions):
    """A new, display-only position-liquidity table."""
    return pd.DataFrame({
        "Symbol": positions["symbol"],
        "Position Value (Rs.)": [_amount(v) for v in positions["position_value"]],
        "Average Daily Traded Value (Rs.)": [_amount(v) for v in positions["average_daily_traded_value"]],
        "Traded Value Source": [TRADED_VALUE_SOURCE_LABELS.get(s, s)
                                for s in positions["traded_value_source"]],
        "Position / ADTV": [_amount(v, 3) for v in positions["position_to_adtv"]],
        "Daily Executable Value (Rs.)": [_amount(v) for v in positions["daily_executable_value"]],
        "Estimated Liquidation Days": [_amount(v) for v in positions["estimated_liquidation_days"]],
    })


def uses_estimated_traded_value(summary):
    """True if any symbol's traded value is estimated from close × volume."""
    return bool((summary["traded_value_source"] == ESTIMATED_TRADED_VALUE).any())
