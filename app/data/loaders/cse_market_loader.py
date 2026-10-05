"""Load locally downloaded market-data files, exactly as the source delivered them.

This loader performs no cleaning and no financial calculations. CSV cells are
read as text with pandas' NA-guessing switched off, so the in-memory frame
holds the file's literal values. Conversion to the canonical schema is a
separate, explicit step (``to_canonical_dataset``) that flags problems
instead of fixing them.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from app.data.schemas.market_schema import derived_fields_used, standardize_columns, to_canonical
from app.data.schemas.provenance import Provenance, build_provenance
from app.data.source_catalog import DataSource, get_source
from app.data.validators.market_validator import ValidationResult, validate_market_data

TEXT_SUFFIXES = (".csv", ".tsv", ".txt")
SUPPORTED_SUFFIXES = TEXT_SUFFIXES + (".parquet",)
# Candidate delimiters for text files; on a tie the earlier one wins.
DELIMITERS = (",", "\t", ";", "|")


@dataclass(frozen=True)
class RawMarketData:
    data: pd.DataFrame
    source: DataSource
    provenance: Provenance
    delimiter: str | None = None  # detected for text files; None for Parquet


@dataclass(frozen=True)
class CanonicalMarketData:
    data: pd.DataFrame | None  # None when required columns are missing
    validation: ValidationResult
    provenance: Provenance


def load_raw_market_file(file_path, source_name, retrieval_time=None, source_date=None,
                         source_url=None, source_version=None):
    """Read a local delimited-text (CSV/TSV/TXT) or Parquet file and attach provenance.

    The delimiter of a text file (comma, tab, semicolon or pipe) is detected
    from its header line, so data pasted from a spreadsheet works unchanged.

    ``source_name`` must be registered in the source catalog, so every loaded
    file is traceable to a documented source. ``source_date`` is only for a
    file whose *content* states a single as-of date; never pass a requested date.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Market data file not found: {path}")
    source = get_source(source_name)

    suffix = path.suffix.lower()
    delimiter = None
    if suffix in TEXT_SUFFIXES:
        delimiter = detect_delimiter(path)
        data = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False,
                           encoding="utf-8-sig")
    elif suffix == ".parquet":
        data = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")

    provenance = build_provenance(path, source, retrieval_time=retrieval_time,
                                  source_date=source_date, source_url=source_url,
                                  source_version=source_version)
    return RawMarketData(data=data, source=source, provenance=provenance, delimiter=delimiter)


def detect_delimiter(path):
    """The delimiter used in a text file's header line (default ",").

    Only the header is inspected: data rows can legitimately contain commas
    (e.g. an unquoted "7,130,000" in tab-separated data). Characters inside
    double quotes are ignored.
    """
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = handle.readline()
    counts = dict.fromkeys(DELIMITERS, 0)
    quoted = False
    for char in header:
        if char == '"':
            quoted = not quoted
        elif not quoted and char in counts:
            counts[char] += 1
    best = max(DELIMITERS, key=counts.get)
    return best if counts[best] else ","


def to_canonical_dataset(raw, date_format="ISO8601"):
    """Map, validate and convert a raw dataset to the canonical schema.

    If the provenance states a ``source_date``, every row must carry that date
    (the validator enforces it), so a current snapshot can never be filed
    under a different day.
    """
    standardized = standardize_columns(raw.data, raw.source.column_map)
    validation = validate_market_data(standardized, expected_date=raw.provenance.source_date,
                                      date_format=date_format)
    if validation.missing_columns:
        return CanonicalMarketData(data=None, validation=validation, provenance=raw.provenance)
    canonical = to_canonical(standardized, raw.source, validation, date_format=date_format)
    provenance = replace(raw.provenance, derived_fields=derived_fields_used(canonical))
    return CanonicalMarketData(data=canonical, validation=validation, provenance=provenance)
