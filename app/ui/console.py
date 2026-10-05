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

