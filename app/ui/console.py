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

from app.backtesting.backtest_engine import BacktestConfig, run_walk_forward_backtest
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
from app.analytics.portfolio_risk import calculate_portfolio_risk_summary
from app.analytics.ratios import calculate_risk_adjusted_ratios
from app.analytics.var import calculate_var_summary
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns
from app.analytics.volatility import calculate_annualized_volatility
from app.config.paths import PROJECT_ROOT
from app.data.loaders.index_series_loader import (
    IndexImportError,
    IndexNameRequiredError,
    load_index_csv,
)
from app.regime.regime_detector import UNDEFINED as REGIME_UNDEFINED
from app.regime.regime_detector import detect_market_regime
from app.stress_testing.stress_engine import (
    BASELINE_HISTORICAL_MEAN,
    BASELINE_ZERO,
    DEFAULT_SCENARIOS,
    run_stress_scenarios,
    scenario_from_components,
)
from app.stress_testing.stress_engine import LIQUIDITY_NO_DATA as LIQUIDITY_STRESS_NO_DATA
from app.stress_testing.stress_engine import LIQUIDITY_OK as LIQUIDITY_STRESS_OK
from app.stress_testing.stress_engine import LIQUIDITY_ZERO_ADTV as LIQUIDITY_STRESS_ZERO
from app.stress_testing.stress_engine import STATUS_OK as STRESS_STATUS_OK
from app.portfolio.optimizer import (
    LIQUIDITY_AT_LIMIT,
    LIQUIDITY_NO_DATA,
    LIQUIDITY_NOT_APPLIED,
    LIQUIDITY_WITHIN_LIMIT,
    optimize_portfolio,
)
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


WEIGHT_PERCENT_TOLERANCE = 1e-4      # = WEIGHT_SUM_TOLERANCE (1e-6) expressed in percent


def default_holdings_text(symbols):
    """Equal weights in percent, one "SYMBOL, weight" line each; the last line takes the
    rounding remainder so the total is exactly 100. A starting point to edit."""
    symbols = sorted(symbols)
    if not symbols:
        return ""
    share = math.floor(10000 / len(symbols)) / 100
    weights = [share] * (len(symbols) - 1) + [round(100 - share * (len(symbols) - 1), 2)]
    return "\n".join(f"{s}, {w:.2f}" for s, w in zip(symbols, weights))


def parse_holdings(text):
    """(list of (symbol, weight as a fraction), error message) from "SYMBOL, weight %" lines.

    Separators: comma, tab or spaces; a trailing % is allowed; blank lines are ignored.
    The weights must total 100% — they are never rescaled. Duplicates, negatives and
    unknown symbols are reported by the analytics layer.
    """
    pairs = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.replace(",", " ").replace("\t", " ").replace("%", " ").split()
        if len(parts) != 2:
            return None, f"Line {number}: enter one holding per line as 'SYMBOL, weight %'."
        symbol, weight_text = parts
        try:
            weight = float(weight_text.rstrip("%"))
        except ValueError:
            return None, f"Line {number}: '{weight_text}' is not a number."
        if not math.isfinite(weight):
            return None, f"Line {number}: the weight for {symbol} must be a finite number."
        pairs.append((symbol, weight / 100))
    if not pairs:
        return None, "Enter at least one holding."
    total = sum(w for _, w in pairs) * 100
    if abs(total - 100) > WEIGHT_PERCENT_TOLERANCE:
        return None, (f"Weights total {total:,.4g}%; they must total 100%. They are not "
                      "adjusted automatically.")
    return pairs, None


def portfolio_risk(data, holdings, portfolio_value, confidence_percent, min_observations,
                   periods_per_year, participation_percent):
    """(PortfolioRiskResult, error message). Percent inputs are entered as 95.0 = 95%."""
    try:
        return calculate_portfolio_risk_summary(
            data, holdings, portfolio_value=float(portfolio_value),
            confidence_level=confidence_percent / 100, min_observations=int(min_observations),
            periods_per_year=periods_per_year,
            participation_rate=participation_percent / 100), None
    except ValueError as exc:
        return None, str(exc)


def portfolio_summary_display(result):
    """A new, display-only Metric / Value table of the portfolio risk summary."""
    def day(value):
        return "n/a" if value is None or pd.isna(value) else str(pd.Timestamp(value).date())

    recovery = (day(result.recovery_date) if not pd.isna(result.recovery_date) else
                "not recovered" if not pd.isna(result.trough_date) else "n/a")
    rows = [
        ("Common observations", f"{result.observations} ({day(result.start_date)} to "
                                f"{day(result.end_date)})"),
        ("Cumulative Return", format_percent(result.cumulative_return)),
        ("Annualized Return (arithmetic)", format_percent(result.annualized_arithmetic_return)),
        ("Annualized Return (geometric)", format_percent(result.annualized_geometric_return)),
        ("Annualized Volatility", format_percent(result.annualized_volatility)),
        ("Maximum Drawdown", format_percent(result.maximum_drawdown)),
        ("Drawdown Peak / Trough / Recovery",
         f"{day(result.peak_date)} / {day(result.trough_date)} / {recovery}"),
        ("Historical VaR (1-day)", format_percent(result.historical_var)),
        ("Parametric VaR (1-day)", format_percent(result.parametric_var)),
        ("Historical CVaR (1-day)", format_percent(result.historical_cvar)),
        ("Parametric CVaR (1-day)", format_percent(result.parametric_cvar)),
        ("Maximum Weight", format_percent(result.maximum_weight)),
        ("HHI (concentration)", f"{result.hhi:.4f}"),
    ]
    if result.portfolio_value is not None:
        rows += [
            ("Most Illiquid Holding", result.most_illiquid_symbol or "n/a"),
            ("Maximum Estimated Liquidation Days",
             _amount(result.maximum_estimated_liquidation_days)),
            ("Maximum Position / ADTV", _amount(result.maximum_position_to_adtv, 3)),
        ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def portfolio_holdings_display(result):
    """A new, display-only holdings / liquidity table."""
    holdings = result.holdings
    return pd.DataFrame({
        "Symbol": holdings["symbol"],
        "Weight": [format_percent(w) for w in holdings["weight"]],
        "Position Value (Rs.)": [_amount(v) for v in holdings["position_value"]],
        "ADTV (Rs.)": [_amount(v) for v in holdings["average_daily_traded_value"]],
        "ADTV Source": [TRADED_VALUE_SOURCE_LABELS.get(s, s)
                        for s in holdings["traded_value_source"]],
        "Position / ADTV": [_amount(v, 3) for v in holdings["position_to_adtv"]],
        "Estimated Liquidation Days": [_amount(v) for v in holdings["estimated_liquidation_days"]],
    })


LIQUIDITY_STATUS_LABELS = {LIQUIDITY_AT_LIMIT: "At limit",
                           LIQUIDITY_WITHIN_LIMIT: "Within limit",
                           LIQUIDITY_NO_DATA: "No liquidity data (not constrained)",
                           LIQUIDITY_NOT_APPLIED: "Not applied"}


def optimize(data, symbols, min_weight_percent, max_weight_percent, risk_aversion, cvar_weight,
             return_weight, confidence_percent, min_observations, periods_per_year,
             portfolio_value, liquidity_enabled, max_position_to_adtv=None):
    """(OptimizationResult, error message). Percent inputs are entered as 40.0 = 40%."""
    try:
        return optimize_portfolio(
            data, list(symbols), risk_aversion=float(risk_aversion),
            cvar_weight=float(cvar_weight), return_weight=float(return_weight),
            confidence_level=confidence_percent / 100, min_observations=int(min_observations),
            periods_per_year=periods_per_year, min_weight=min_weight_percent / 100,
            max_weight=max_weight_percent / 100, portfolio_value=float(portfolio_value),
            liquidity_constraint_enabled=bool(liquidity_enabled),
            max_position_to_adtv=(float(max_position_to_adtv) if liquidity_enabled else None)), None
    except ValueError as exc:
        return None, str(exc)


def optimization_allocation_display(result):
    """A new, display-only allocation table (with liquidity columns when the constraint is on)."""
    liquidity = result.liquidity
    table = pd.DataFrame({
        "Symbol": liquidity["symbol"],
        "Weight %": [format_percent(w) for w in liquidity["weight"]],
        "Position (Rs.)": [_amount(v) for v in liquidity["position_value"]],
    })
    if result.liquidity_constraint_enabled:
        table["ADTV (Rs.)"] = [_amount(v) for v in liquidity["average_daily_traded_value"]]
        table["ADTV Source"] = [TRADED_VALUE_SOURCE_LABELS.get(s, "n/a")
                                for s in liquidity["traded_value_source"]]
        table["Position / ADTV"] = [_amount(v, 3) for v in liquidity["position_to_adtv"]]
        table["Liquidity Constraint Status"] = [LIQUIDITY_STATUS_LABELS[s] for s in
                                                liquidity["liquidity_constraint_status"]]
    return table


def optimization_metrics_display(result):
    """A new, display-only Metric / Value table of the optimized portfolio."""
    rows = [
        ("Expected Annual Return (historical, arithmetic)",
         format_percent(result.expected_annual_return)),
        ("Annualized Volatility", format_percent(result.annualized_volatility)),
        ("Historical VaR (1-day)", format_percent(result.historical_var)),
        ("Historical CVaR (1-day)", format_percent(result.historical_cvar)),
        ("HHI (concentration)", f"{result.hhi:.4f}"),
        ("Maximum Weight", format_percent(result.max_weight)),
        ("Observations", f"{result.observations} common days ({result.start_date.date()} to "
                         f"{result.end_date.date()})"),
        ("Solver Status", f"{result.solver_status} ({result.solver})"),
        ("Objective Value (daily units)", f"{result.objective_value:.6g}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def optimization_comparison_display(result):
    """Equal weight vs optimized on the same scenarios; numbers only, no verdict."""
    ew, opt = result.equal_weight_metrics, result.metrics
    rows = [("Annualized Return", ew.expected_annual_return, opt.expected_annual_return),
            ("Annualized Volatility", ew.annualized_volatility, opt.annualized_volatility),
            ("Historical VaR (1-day)", ew.historical_var, opt.historical_var),
            ("Historical CVaR (1-day)", ew.historical_cvar, opt.historical_cvar)]
    table = [(name, format_percent(a), format_percent(b)) for name, a, b in rows]
    table.append(("HHI", f"{ew.hhi:.4f}", f"{opt.hhi:.4f}"))
    return pd.DataFrame(table, columns=["Metric", "Equal Weight", "ExitSafe Optimized"])


INDEX_SAMPLE_CSV = PROJECT_ROOT / "data" / "sample" / "sample_index_data.csv"
KNOWN_INDEXES = ["ASPI", "S&P SL20"]
REGIME_LABELS = {"NORMAL": "Normal", "HIGH_VOLATILITY": "High volatility", "STRESS": "Stress",
                 "RECOVERY": "Recovery", REGIME_UNDEFINED: "N/A (warm-up)"}
TREND_LABELS = {"UPTREND": "Uptrend", "DOWNTREND": "Downtrend", "NEUTRAL": "Neutral",
                REGIME_UNDEFINED: "n/a"}


def load_index_data(file_name, content, index_name=None):
    """(IndexImportResult, error message). Bytes are read via a temporary file, as in run_import.

    ``index_name`` is used only when the file has no Index column; a file that
    names its index keeps its own names.
    """
    try:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / Path(file_name).name
            path.write_bytes(content)
            try:
                return load_index_csv(path), None
            except IndexNameRequiredError as exc:
                if not index_name:
                    return None, str(exc)
                return load_index_csv(path, index_name), None
    except IndexImportError as exc:
        return None, f"Index file could not be imported: {exc}"


def market_regime(data, index_name):
    """(MarketRegimeResult, error message) for one index with the default settings."""
    try:
        return detect_market_regime(data, index_name), None
    except ValueError as exc:
        return None, str(exc)


def regime_current_display(result):
    """A new, display-only Metric / Value table of the latest date's regime."""
    c = result.current
    rows = [
        ("Date", str(c.date.date())),
        ("Regime", REGIME_LABELS[c.regime]),
        ("Reason", c.regime_reason),
        (f"Rolling Volatility ({result.volatility_window}-day, daily)",
         format_percent(c.rolling_volatility)),
        ("Rolling Volatility (annualized, × √252, for display)",
         format_percent(c.rolling_volatility * math.sqrt(252))),
        (f"Volatility Ratio ({result.volatility_window}-day / {result.baseline_window}-day)",
         "n/a" if pd.isna(c.volatility_ratio) else f"{c.volatility_ratio:.2f}"),
        (f"Current Drawdown (from {result.drawdown_lookback}-day peak)",
         format_percent(c.current_drawdown)),
        (f"Trend State (close / {result.trend_window}-day average)",
         TREND_LABELS[c.trend_state] + ("" if pd.isna(c.trend_ratio)
                                        else f" ({c.trend_ratio:.3f})")),
        ("Index Close", _amount(c.current_close)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def regime_history_display(result):
    """A new, display-only historical regime table (date, regime, volatility, drawdown, trend)."""
    h = result.history
    return pd.DataFrame({
        "Date": h["date"].dt.date.astype(str),
        "Regime": [REGIME_LABELS[r] for r in h["regime"]],
        f"Rolling Volatility ({result.volatility_window}-day, daily)":
            [format_percent(v) for v in h["rolling_volatility"]],
        "Current Drawdown": [format_percent(v) for v in h["current_drawdown"]],
        "Trend State": [TREND_LABELS[t] for t in h["trend_state"]],
    })


BASELINE_LABELS = {BASELINE_ZERO: "Zero (scenario shock only)",
                   BASELINE_HISTORICAL_MEAN: "Historical mean daily return"}
LIQUIDITY_STATUS_TEXT = {LIQUIDITY_STRESS_OK: "OK", LIQUIDITY_STRESS_ZERO: "Zero ADTV",
                         LIQUIDITY_STRESS_NO_DATA: "No liquidity data"}


def parse_sector_mapping(text):
    """(None or {symbol: sector}, error message) from "SYMBOL, Sector" lines.

    The first comma or tab separates the symbol from the sector, so sector names
    may contain spaces ("Banking Finance & Insurance"). Blank text gives None.
    """
    mapping = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None, f"Sector line {number}: enter it as 'SYMBOL, Sector'."
        symbol, sector = parts[0].upper(), parts[1]
        if symbol in mapping and mapping[symbol].casefold() != sector.casefold():
            return None, f"{symbol} is mapped to two sectors ({mapping[symbol]}, {sector})."
        mapping[symbol] = sector
    return (mapping or None), None


def custom_stress_scenario(market_percent=0.0, sector_name="", sector_percent=0.0,
                           volatility_multiplier=0.0, liquidity_multiplier=1.0):
    """(None or a StressScenario, error message) from the custom inputs.

    A component is used only when set: market or sector shock other than 0%,
    volatility multiplier above 0, liquidity multiplier below 1.
    """
    sector_name = (sector_name or "").strip()
    if sector_percent and not sector_name:
        return None, "Enter a sector name for the custom sector shock."
    try:
        return scenario_from_components(
            "Custom scenario",
            market_shock=market_percent / 100 if market_percent else None,
            sector_name=sector_name if sector_percent else None,
            sector_shock=sector_percent / 100 if sector_percent else None,
            volatility_multiplier=volatility_multiplier if volatility_multiplier > 0 else None,
            liquidity_multiplier=liquidity_multiplier if liquidity_multiplier < 1 else None,
            description="Custom scenario built from the inputs below."), None
    except ValueError as exc:
        if "at least one shock" in str(exc):
            return None, None
        return None, str(exc)


def stress_report(data, holdings, portfolio_value, scenarios, sector_mapping,
                  participation_percent, baseline, confidence_percent, min_observations):
    """(StressReport, error message). Percent inputs are entered as 95.0 = 95%."""
    try:
        return run_stress_scenarios(
            data, holdings, scenarios, portfolio_value=float(portfolio_value),
            sector_mapping=sector_mapping, participation_rate=participation_percent / 100,
            baseline=baseline, confidence_level=confidence_percent / 100,
            min_observations=int(min_observations)), None
    except ValueError as exc:
        return None, str(exc)


def stress_summary_display(report):
    """A new, display-only scenario summary (numbers only, no labels or advice)."""
    rows = []
    for r in report.results:
        ok = r.status == STRESS_STATUS_OK
        contribution = ("n/a" if r.largest_negative_contributor is None else
                        f"{r.largest_negative_contributor} ({format_percent(r.largest_negative_contribution)})")
        rows.append({
            "Scenario": r.scenario_name,
            "Portfolio Return": format_percent(r.portfolio_return) if ok else "n/a",
            "Portfolio Loss": format_percent(r.portfolio_loss) if ok else "n/a",
            "Loss Amount (Rs.)": _amount(r.portfolio_loss_amount) if ok and r.portfolio_loss_amount is not None else "n/a",
            "Stressed Portfolio Value (Rs.)": _amount(r.stressed_portfolio_value) if ok and r.stressed_portfolio_value is not None else "n/a",
            "Largest Negative Contribution": contribution,
            "Most Exposed Holding": r.most_exposed_holding or "n/a",
            "Status": "OK" if ok else "Unavailable",
        })
    return pd.DataFrame(rows)


def stress_impact_display(result):
    """A new, display-only stock impact table for one scenario."""
    impacts = result.symbol_impacts
    table = pd.DataFrame({
        "Symbol": impacts["symbol"],
        "Weight": [format_percent(w) for w in impacts["weight"]],
        "Base Return": [format_percent(v) for v in impacts["base_return"]],
        "Stressed Return": [format_percent(v) for v in impacts["stressed_return"]],
        "Contribution": [format_percent(v) for v in impacts["stress_contribution"]],
    })
    if impacts["sector"].notna().any():
        table.insert(1, "Sector", [s or "n/a" for s in impacts["sector"]])
    return table


def stress_liquidity_display(result):
    """A new, display-only liquidity table for a liquidity scenario."""
    rows = result.liquidity_impacts
    return pd.DataFrame({
        "Symbol": rows["symbol"],
        "Base ADTV (Rs.)": [_amount(v) for v in rows["base_adtv"]],
        "Stressed ADTV (Rs.)": [_amount(v) for v in rows["stressed_adtv"]],
        "ADTV Source": [TRADED_VALUE_SOURCE_LABELS.get(s, "n/a") for s in rows["traded_value_source"]],
        "Base Liquidation Days": [_amount(v) for v in rows["base_liquidation_days"]],
        "Stressed Liquidation Days": [_amount(v) for v in rows["stressed_liquidation_days"]],
        "Status": [LIQUIDITY_STATUS_TEXT[s] for s in rows["liquidity_status"]],
    })


BACKTEST_SAMPLE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_backtest_data.csv"
PERCENT_METRICS = {"Cumulative Return", "Annualized Return (arithmetic)",
                   "Annualized Return (geometric)", "Annualized Volatility", "Maximum Drawdown",
                   "Historical VaR (1-day)", "Historical CVaR (1-day)"}


def run_backtest(data, symbols, initial_capital, training_window, test_window,
                 rebalance_frequency, risk_free_percent, confidence_percent, min_observations,
                 periods_per_year, optimizer_settings, index_data=None, index_name=None):
    """(BacktestResult, error message). Percent inputs are entered as 95.0 = 95%."""
    s = optimizer_settings
    try:
        config = BacktestConfig(
            symbols=tuple(symbols), initial_capital=float(initial_capital),
            training_window=int(training_window), test_window=int(test_window),
            rebalance_frequency=int(rebalance_frequency), periods_per_year=periods_per_year,
            confidence_level=confidence_percent / 100, min_observations=int(min_observations),
            risk_free_rate=risk_free_percent / 100, risk_aversion=float(s["risk_aversion"]),
            cvar_weight=float(s["cvar_weight"]), return_weight=float(s["return_weight"]),
            min_weight=s["min_percent"] / 100, max_weight=s["max_percent"] / 100,
            liquidity_constraint_enabled=bool(s["liquidity_enabled"]),
            max_position_to_adtv=(float(s["max_position_to_adtv"]) if s["liquidity_enabled"]
                                  else None))
        return run_walk_forward_backtest(data, config, index_data, index_name), None
    except ValueError as exc:
        return None, str(exc)


def backtest_comparison_display(result):
    """A new, display-only Metric x strategy table (numbers only, no verdict)."""
    def cell(metric, value):
        if pd.isna(value):
            return "n/a"
        if metric in PERCENT_METRICS:
            return format_percent(value)
        if metric in ("Observations", "Rebalances"):
            return str(int(value))
        return f"{value:.3f}"

    table = pd.DataFrame({"Metric": result.metrics.index})
    for label in result.metrics.columns:
        table[label] = [cell(m, v) for m, v in result.metrics[label].items()]
    return table


def _weights_text(weights):
    return ", ".join(f"{s} {w:.1%}" for s, w in weights.items()) if weights else "none"


def backtest_log_display(result):
    """A new, display-only rebalance log (one row per window and strategy)."""
    log = result.rebalance_log
    return pd.DataFrame({
        "Window": log["window"],
        "Rebalance Date": log["rebalance_date"].dt.date.astype(str),
        "Training": [f"{a.date()} to {b.date()}" for a, b in zip(log["training_start"],
                                                              log["training_end"])],
        "Test": [f"{a.date()} to {b.date()}" for a, b in zip(log["test_start"], log["test_end"])],
        "Strategy": log["strategy"],
        "Status": log["status"],
        "Weights": [_weights_text(w) for w in log["weights"]],
        "Training Obs": log["training_observations"],
        "Solver": [s if isinstance(s, str) else "n/a" for s in log["solver_status"]],
        "Note": log["reason"],
    })


def backtest_final_weights_display(result):
    """Weights chosen at each strategy's latest rebalance (Symbol x strategy)."""
    return pd.DataFrame({"Symbol": list(result.config.symbols),
                         **{label: [format_percent(w.get(s)) if w else "n/a"
                                    for s in result.config.symbols]
                            for label, w in result.final_weights.items()}})
