"""Manual smoke test of the CSV importer (not part of pytest).

    python scripts/manual_csv_import_test.py

Imports the synthetic fixture and a temporary provider-style file, prints the
result and checks the key mapping rules. Exits with 1 if a check fails.
"""

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.data.loaders.csv_market_loader import load_csv_market_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_date_price_vol_change.csv"
TEMP_CSV = """Date,Price,Open,High,Low,Vol.,Change %
12/31/2025,660.09,664.75,665,659.44,7.94M,-0.88%
12/30/2025,665.95,658.69,672.22,657.84,9.19M,1.10%
12/29/2025,658.69,658.01,660.25,654.39,8.51M,-0.69%
"""

failures = []


def check(label, condition):
    print(f"  [{'OK' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def show_import(title, path, symbol):
    print("=" * 78)
    print(title)
    print("=" * 78)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    result = load_csv_market_data(path, symbol=symbol)
    data, report = result.data, result.report

    print(f"File:                {report.file_name}")
    print(f"Detected columns:    {report.detected_columns}")
    print(f"Column mapping:      {report.column_mapping}")
    print(f"Ignored columns:     {report.ignored_columns or 'none'}")
    print(f"Normalized columns:  {report.normalized_columns}")
    print(f"Symbol:              {report.symbols} (from {report.symbol_source})")
    print(f"Rows read:           {report.rows_read}  valid: {report.valid_rows}  "
          f"invalid: {report.invalid_rows}")
    print(f"Date range:          {report.date_min} .. {report.date_max} "
          f"(convention {report.date_convention})")
    print(f"Validation status:   {report.status}")
    print(f"Warnings:            {report.warnings or 'none'}")
    print(f"Issue counts:        {report.issue_counts or 'none'}")
    print("Notes:")
    for note in report.notes:
        print(f"  - {note}")

    print("\nFirst 5 canonical rows:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(data.head(5).to_string(index=False))

    print("\nRaw 'Vol.' -> parsed volume -> estimated_traded_value (close * volume):")
    for raw_vol, row in zip(raw["Vol."], data.itertuples()):
        print(f"  {raw_vol:>10} -> {row.volume:>14,.0f}   "
              f"{row.close:>8} * {row.volume:,.0f} = {row.estimated_traded_value:,.2f}")

    print("\nChecks:")
    check("Price -> close", list(data["close"]) == [float(v) for v in raw["Price"]])
    check("Vol. -> volume (K/M/B and commas expanded)",
          all(v > 0 and v == int(v) for v in data["volume"]))
    check("Change % preserved as supplied",
          list(data["change_pct"]) == [float(v.rstrip("%")) for v in raw["Change %"]])
    check("estimated_traded_value = close * volume",
          ((data["close"] * data["volume"] - data["estimated_traded_value"]).abs() < 1e-6).all())
    check("turnover left empty (not reported by the file)", data["turnover"].isna().all())
    check("estimated_traded_value recorded as derived in provenance",
          "estimated_traded_value" in result.provenance.derived_fields)
    check("source preserved on every row", (data["source"] == result.provenance.source_name).all())
    check("provenance file hash matches the file",
          result.provenance.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest())
    print(f"\nProvenance: source={result.provenance.source_name} "
          f"type={result.provenance.source_type} file={result.provenance.original_file_name}")
    print(f"            sha256={result.provenance.file_sha256}")
    print(f"            derived_fields={result.provenance.derived_fields}\n")
    return result


def main():
    before = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    show_import("1. Synthetic fixture", FIXTURE, "TEST.N0000")

    with tempfile.TemporaryDirectory() as folder:
        temp_path = Path(folder) / "manual_example.csv"
        temp_path.write_text(TEMP_CSV, encoding="utf-8")
        show_import("2. Temporary example CSV (deleted after this run)", temp_path, "EXAMPLE.N0000")
    print(f"Temporary file removed: {not temp_path.exists()}")

    after = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    print(f"Fixture unchanged:      {before == after} ({after[:16]}...)")
    if before != after:
        failures.append("fixture modified")

    print(f"\nRESULT: {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
