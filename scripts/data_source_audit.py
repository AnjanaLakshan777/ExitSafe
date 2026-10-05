"""Data source audit: what market data do we actually have, and can it be trusted?

    python scripts/data_source_audit.py                              # every known local dataset
    python scripts/data_source_audit.py --file F.csv --source NAME   # one catalogued file
    python scripts/data_source_audit.py --file F.csv --source NAME --source-date 2026-09-30

The report is printed and written to data/processed/audit/
data_availability_report.txt, with one provenance manifest (JSON) per audited
file. Exit code is 1 if any audited dataset FAILs validation.

Nothing here downloads data; see scripts/fetch_secondary_datasets.py.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.config.paths import AUDIT_DIR, CSE_MARKET_RAW_DIR, EXTERNAL_RAW_DIR, SAMPLE_DIR  # noqa: E402
from app.data.loaders.cse_market_loader import load_raw_market_file, to_canonical_dataset  # noqa: E402
from app.data.schemas.market_schema import PRICE_COLUMNS, parse_dates, parse_numbers  # noqa: E402
from app.data.schemas.provenance import write_manifest  # noqa: E402
from app.data.source_catalog import SOURCE_CATALOG, DataSourceType  # noqa: E402
from app.data.validators.market_validator import DatasetStatus  # noqa: E402

HF_MARKET_DIR = EXTERNAL_RAW_DIR / "huggingface" / "tharu-jwd__cse-market-data"
HF_SYMBOLS_FILE = (EXTERNAL_RAW_DIR / "huggingface" / "kjhq__Sri-Lanka-Stock-Symbols-and-Metadata"
                   / "srilanka.csv")
REPORT_NAME = "data_availability_report.txt"


@dataclass
class AuditTarget:
    source_name: str
    file_path: Path
    label: str
    source_date: date | None = None
    aspi_file: Path | None = None
    sector_file: Path | None = None
    company_file: Path | None = None


@dataclass
class AuditOutcome:
    target: AuditTarget
    raw_columns: list[str]
    result: object          # CanonicalMarketData
    download: dict | None   # entry from the SOURCE.json written at download time
    availability: dict


def default_targets():
    targets = [AuditTarget("exitsafe_sample", SAMPLE_DIR / "sample_market_data.csv",
                           "ExitSafe sample")]
    hf_prices = HF_MARKET_DIR / "stock_prices.csv"
    if hf_prices.is_file():
        targets.append(AuditTarget(
            "hf_tharu_jwd_cse_market_data", hf_prices, "HF CSE dataset",
            aspi_file=HF_MARKET_DIR / "aspi.csv",
            sector_file=HF_MARKET_DIR / "sector_mapping.csv",
            company_file=HF_SYMBOLS_FILE if HF_SYMBOLS_FILE.is_file() else None,
        ))
    for path in sorted(CSE_MARKET_RAW_DIR.iterdir()) if CSE_MARKET_RAW_DIR.is_dir() else []:
        if path.suffix.lower() in (".csv", ".parquet"):
            targets.append(AuditTarget("cse_historical_official", path, f"CSE {path.name}"))
    return targets


def run_audit(target):
    download = read_download_record(target.file_path)
    raw = load_raw_market_file(
        target.file_path, target.source_name,
        retrieval_time=datetime.fromisoformat(download["retrieval_time"]) if download else None,
        source_date=target.source_date,
        source_url=download["file"]["url"] if download else None,
        source_version=download["revision"] if download else None,
    )
    result = to_canonical_dataset(raw)
    availability = {}
    if result.data is not None:
        availability = market_availability(result.data)
        availability["aspi"] = aspi_availability(target.aspi_file, result.data)
        availability["sector_mapping"] = sector_availability(target.sector_file, result.data)
        availability["company_metadata"] = company_availability(target.company_file, result.data)
    return AuditOutcome(target, [str(c) for c in raw.data.columns], result, download, availability)


def read_download_record(file_path):
    record_path = Path(file_path).parent / "SOURCE.json"
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    entry = record.get("files", {}).get(Path(file_path).name)
    return {**record, "file": entry} if entry else None


# --- availability checks -------------------------------------------------------

def market_availability(data):
    usable = data[data["validation_status"] != "INVALID"]
    volume = data["volume"]
    rows_per_symbol = usable.groupby("symbol").size()
    trading_days = usable["date"].nunique()
    return {
        "volume_positive": int((volume > 0).sum()),
        "volume_zero": int((volume == 0).sum()),
        "volume_missing": int(volume.isna().sum()),
        "volume_non_integer": int(((volume % 1).fillna(0) != 0).sum()),
        "turnover_rows": int(data["turnover"].notna().sum()),
        "trades_rows": int(data["trades"].notna().sum()),
        "rows_per_symbol_min": int(rows_per_symbol.min()) if len(rows_per_symbol) else 0,
        "rows_per_symbol_median": float(rows_per_symbol.median()) if len(rows_per_symbol) else 0,
        "rows_per_symbol_max": int(rows_per_symbol.max()) if len(rows_per_symbol) else 0,
        "symbols_on_every_trading_day": int((rows_per_symbol == trading_days).sum()),
    }


def aspi_availability(path, prices):
    if path is None or not path.is_file():
        return None
    aspi = pd.read_csv(path, dtype=str, keep_default_na=False)
    dates = parse_dates(aspi["date"])
    o, h, l, c = (parse_numbers(aspi[col]) for col in PRICE_COLUMNS)
    volume = parse_numbers(aspi["volume"]) if "volume" in aspi else pd.Series(dtype=float)
    stock_dates, aspi_dates = set(prices["date"].dropna()), set(dates.dropna())
    return {
        "file": path.name,
        "rows": len(aspi),
        "date_min": _day(dates.min()),
        "date_max": _day(dates.max()),
        "columns": list(aspi.columns),
        "volume_present": int(volume.notna().sum()),
        "ohlc_violations": int(((l > o) | (l > c) | (h < o) | (h < c)).sum()),
        "non_positive_close": int((c <= 0).sum()),
        "stock_dates_missing_in_aspi": len(stock_dates - aspi_dates),
        "aspi_dates_missing_in_stocks": len(aspi_dates - stock_dates),
    }


def sector_availability(path, prices):
    if path is None or not path.is_file():
        return None
    mapping = pd.read_csv(path, dtype=str, keep_default_na=False)
    key = "symbol" if "symbol" in mapping else mapping.columns[0]
    symbols = pd.Series(prices["symbol"].dropna().unique())
    # Mapping uses short tickers (JKH); prices use full CSE symbols (JKH.N0000).
    mapped = symbols.str.split(".").str[0].isin(set(mapping[key]))
    return {
        "file": path.name,
        "columns": list(mapping.columns),
        "rows": len(mapping),
        "sectors": int(mapping["sector"].nunique()) if "sector" in mapping else None,
        "duplicate_tickers": int(mapping[key].duplicated().sum()),
        "price_symbols_mapped": int(mapped.sum()),
        "price_symbols_total": len(symbols),
        "unmapped_examples": sorted(symbols[~mapped])[:10],
    }


def company_availability(path, prices):
    if path is None or not path.is_file():
        return None
    companies = pd.read_csv(path, dtype=str, keep_default_na=False)
    symbols = pd.Series(prices["symbol"].dropna().unique())
    return {
        "file": path.name,
        "columns": list(companies.columns),
        "rows": len(companies),
        "price_symbols_matched": int(symbols.isin(set(companies["ticker"])).sum()),
        "price_symbols_total": len(symbols),
    }


# --- report formatting ----------------------------------------------------------

def format_audit(outcome):
    source = next(s for s in SOURCE_CATALOG if s.source_name == outcome.target.source_name)
    v, p, a = outcome.result.validation, outcome.result.provenance, outcome.availability
    official = source.source_type is DataSourceType.OFFICIAL_CSE
    lines = [
        "DATA SOURCE AUDIT",
        "=================",
        f"Source:          {source.source_name}",
        f"Source type:     {source.source_type} (trust {source.trust_level}, "
        f"priority {source.source_priority}){'' if official else ' - NOT an official CSE source'}",
        f"Canonical use:   {'allowed' if source.historical_backfill_allowed else 'NOT allowed'}"
        " for historical backfill",
        f"File:            {_display_path(outcome.target.file_path)}",
        f"SHA-256:         {p.file_sha256}",
        f"Retrieved:       {_retrieval(outcome.download)}",
        f"Checksum:        {_checksum(outcome.download, p.file_sha256)}",
        f"Source date:     {p.source_date or 'not stated by source (multi-date file)'}",
        f"Rows:            {v.total_rows:,}",
        f"Symbols:         {v.symbol_count:,}",
        f"Start date:      {_day(v.date_min)}",
        f"End date:        {_day(v.date_max)}",
        f"Trading days:    {v.trading_days:,}",
        f"Columns:         {', '.join(outcome.raw_columns)}",
    ]
    if v.missing_columns:
        lines.append(f"Missing columns: {', '.join(v.missing_columns)}")
    lines += [
        f"Missing values:  {_pairs(v.missing_values)}",
        f"Duplicates:      symbol/date {v.duplicate_rows:,}, exact copies {v.exact_duplicate_rows:,}",
        f"OHLC violations: {_pairs(v.invalid_ohlc)}",
        f"Invalid rows:    {v.invalid_rows:,} ({_share(v.invalid_rows, v.total_rows)})",
        f"Issue counts:    {_pairs(v.issue_counts) or 'none'}",
    ]
    if a:
        lines += [
            f"Volume:          positive {a['volume_positive']:,}, zero {a['volume_zero']:,}, "
            f"missing {a['volume_missing']:,}, fractional {a['volume_non_integer']:,}",
            f"Turnover:        {_availability(a['turnover_rows'], v.total_rows)}",
            f"Trades:          {_availability(a['trades_rows'], v.total_rows)}",
            f"Rows per symbol: min {a['rows_per_symbol_min']}, median {a['rows_per_symbol_median']:g}, "
            f"max {a['rows_per_symbol_max']}; {a['symbols_on_every_trading_day']} symbol(s) "
            "present on every trading day",
            f"ASPI:            {_aspi(a['aspi'])}",
            f"Sector mapping:  {_sectors(a['sector_mapping'])}",
            f"Company list:    {_companies(a['company_metadata'])}",
        ]
    for title, items in (("Failure reasons", v.failure_reasons), ("Warnings", v.warnings),
                         ("Notes", v.notes)):
        if items:
            lines.append(f"{title}:")
            lines += [f"  - {item}" for item in items]
    lines.append(f"Status:          {v.status}")
    return "\n".join(lines)


def format_comparison(outcomes):
    rows = [("Metric", *[o.target.label for o in outcomes])]

    def add(name, getter):
        rows.append((name, *[str(getter(o)) for o in outcomes]))

    add("Source type", lambda o: _source(o).source_type)
    add("Trust / priority", lambda o: f"{_source(o).trust_level} / {_source(o).source_priority}")
    add("Rows", lambda o: f"{o.result.validation.total_rows:,}")
    add("Symbols", lambda o: o.result.validation.symbol_count)
    add("Start date", lambda o: _day(o.result.validation.date_min))
    add("End date", lambda o: _day(o.result.validation.date_max))
    add("Trading days", lambda o: o.result.validation.trading_days)
    add("Columns", lambda o: len(o.raw_columns))
    add("Turnover", lambda o: _yes_no(o.availability.get("turnover_rows")))
    add("Trades", lambda o: _yes_no(o.availability.get("trades_rows")))
    add("Volume missing", lambda o: o.availability.get("volume_missing", "n/a"))
    add("Volume zero", lambda o: o.availability.get("volume_zero", "n/a"))
    add("Volume fractional", lambda o: o.availability.get("volume_non_integer", "n/a"))
    add("Missing values", lambda o: sum(o.result.validation.missing_values.values()))
    add("Symbol/date duplicates", lambda o: o.result.validation.duplicate_rows)
    add("OHLC violations", lambda o: sum(o.result.validation.invalid_ohlc.values()))
    add("Invalid rows", lambda o: o.result.validation.invalid_rows)
    add("ASPI", lambda o: "yes" if o.availability.get("aspi") else "no")
    add("Sector mapping", lambda o: "yes" if o.availability.get("sector_mapping") else "no")
    add("Status", lambda o: o.result.validation.status)

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["DATASET COMPARISON", "=================="]
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def format_unavailable():
    lines = ["SOURCES NOT YET AVAILABLE LOCALLY", "================================="]
    for source in SOURCE_CATALOG:
        if not source.enabled:
            lines.append(f"- {source.source_name} ({source.source_type}): {source.description}")
    return "\n".join(lines)


def build_report(outcomes):
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sections = [f"ExitSafe data availability report - generated {generated}"]
    sections += [format_audit(o) for o in outcomes]
    if len(outcomes) > 1:
        sections.append(format_comparison(outcomes))
    sections.append(format_unavailable())
    return "\n\n".join(sections) + "\n"


# --- helpers --------------------------------------------------------------------

def _source(outcome):
    return next(s for s in SOURCE_CATALOG if s.source_name == outcome.target.source_name)


def _day(value):
    return "n/a" if value is None or pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _pairs(counts):
    return ", ".join(f"{k} {v:,}" for k, v in counts.items())


def _share(part, total):
    return f"{part / total:.2%}" if total else "n/a"


def _availability(rows, total):
    return f"available ({rows:,} rows)" if rows else "not provided by source (left null)"


def _yes_no(rows):
    return "yes" if rows else "no"


def _retrieval(download):
    if not download:
        return "unknown (no download record)"
    return f"{download['retrieval_time']} from {download['repo']} @ {download['revision'][:12]}"


def _checksum(download, sha256):
    if not download:
        return "no download record to compare against"
    return "matches download record" if download["file"]["sha256"] == sha256 else \
        "MISMATCH - file changed since download"


def _aspi(info):
    if not info:
        return "not available"
    return (f"{info['file']}: {info['rows']:,} rows {info['date_min']}..{info['date_max']}, "
            f"volume present in {info['volume_present']} rows, "
            f"OHLC violations {info['ohlc_violations']}, "
            f"stock dates missing in ASPI {info['stock_dates_missing_in_aspi']}, "
            f"ASPI dates missing in stocks {info['aspi_dates_missing_in_stocks']}")


def _sectors(info):
    if not info:
        return "not available"
    text = (f"{info['file']}: {info['rows']} tickers, {info['sectors']} sectors, "
            f"{info['price_symbols_mapped']}/{info['price_symbols_total']} price symbols mapped "
            f"(by base ticker)")
    if info["unmapped_examples"]:
        text += f"; unmapped e.g. {', '.join(info['unmapped_examples'])}"
    return text


def _companies(info):
    if not info:
        return "not available"
    return (f"{info['file']}: {info['rows']} companies, {info['price_symbols_matched']}/"
            f"{info['price_symbols_total']} price symbols matched exactly")


def _display_path(path):
    try:
        return Path(path).resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return str(path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, help="audit a single file")
    parser.add_argument("--source", help="catalog source_name of --file")
    parser.add_argument("--source-date", type=date.fromisoformat,
                        help="as-of date stated INSIDE the source payload (snapshot files only)")
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR)
    args = parser.parse_args(argv)
    if bool(args.file) != bool(args.source):
        parser.error("--file and --source must be given together")
    return args


def main(argv=None):
    args = parse_args(argv)
    targets = ([AuditTarget(args.source, args.file, args.file.name, source_date=args.source_date)]
               if args.file else default_targets())

    outcomes = [run_audit(target) for target in targets]
    report = build_report(outcomes)
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    for outcome in outcomes:
        name = f"{outcome.target.source_name}__{outcome.target.file_path.stem}.manifest.json"
        write_manifest(args.output_dir / name, outcome.result.provenance, outcome.result.validation)
    print(f"Report and manifests written to {_display_path(args.output_dir)}")

    return 1 if any(o.result.validation.status is DatasetStatus.FAIL for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
