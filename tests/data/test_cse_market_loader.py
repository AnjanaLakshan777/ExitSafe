import hashlib
import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from app.data.loaders.cse_market_loader import load_raw_market_file, to_canonical_dataset
from app.data.schemas.market_schema import CANONICAL_COLUMNS, standardize_columns
from app.data.schemas.provenance import Provenance, write_manifest
from app.data.source_catalog import DataSourceType, get_source
from app.data.validators.market_validator import DatasetStatus

HF = "hf_tharu_jwd_cse_market_data"   # canonical column names, no turnover/trades
SAMPLE = "exitsafe_sample"            # Title-case names + "Value Traded"

HF_CSV = (
    "symbol,date,open,high,low,close,volume\n"
    "JKH.N0000,2026-01-05,20.0,20.5,19.8,20.1,1500000.0\n"
    "JKH.N0000,2026-01-06,20.1,20.2,19.9,20.0,NA\n"
    "JKH.N0000,2026-01-07,20.0,19.0,19.5,20.0,1000\n"
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- raw loading ----------------------------------------------------------------

def test_csv_values_are_preserved_exactly(tmp_path):
    raw = load_raw_market_file(write(tmp_path, "prices.csv", HF_CSV), HF)

    assert list(raw.data.columns) == ["symbol", "date", "open", "high", "low", "close", "volume"]
    assert raw.data.loc[0, "volume"] == "1500000.0"  # not converted
    assert raw.data.loc[1, "volume"] == "NA"         # not turned into NaN
    assert raw.data.loc[2, "high"] == "19.0"         # questionable, but untouched


def test_parquet_is_supported(tmp_path):
    path = tmp_path / "prices.parquet"
    pd.DataFrame({"symbol": ["JKH.N0000"], "date": pd.to_datetime(["2026-01-05"]),
                  "open": [20.0], "high": [20.5], "low": [19.8], "close": [20.1],
                  "volume": [1500000]}).to_parquet(path)

    result = to_canonical_dataset(load_raw_market_file(path, HF))

    assert result.validation.status is DatasetStatus.PASS
    assert result.data.loc[0, "close"] == 20.1


def test_unsupported_file_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_raw_market_file(write(tmp_path, "prices.xlsx", "x"), HF)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_market_file(tmp_path / "nope.csv", HF)


def test_source_must_be_in_the_catalog(tmp_path):
    with pytest.raises(KeyError, match="Unknown data source"):
        load_raw_market_file(write(tmp_path, "prices.csv", HF_CSV), "some_random_site")


# --- provenance -----------------------------------------------------------------

def test_source_metadata_is_attached(tmp_path):
    path = write(tmp_path, "prices.csv", HF_CSV)
    retrieved = datetime(2026, 9, 30, 8, 0, tzinfo=timezone.utc)

    raw = load_raw_market_file(path, HF, retrieval_time=retrieved, source_version="abc123")
    p = raw.provenance

    assert p.source_name == HF
    assert p.source_type is DataSourceType.SECONDARY_DATASET
    assert p.original_file_name == "prices.csv"
    assert p.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert p.file_size_bytes == path.stat().st_size
    assert p.retrieval_time == retrieved
    assert p.loaded_time.tzinfo is not None
    assert p.source_date is None  # a multi-date file states no single as-of date
    assert p.source_url == get_source(HF).location
    assert p.source_version == "abc123"


def test_provenance_rejects_naive_times_and_datetime_source_dates():
    base = dict(source_name=HF, source_type=DataSourceType.SECONDARY_DATASET,
                original_file_name="f.csv", file_sha256="00", file_size_bytes=1,
                loaded_time=datetime(2026, 9, 30, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="timezone-aware"):
        Provenance(**{**base, "loaded_time": datetime(2026, 9, 30)})
    with pytest.raises(TypeError, match="source_date"):
        Provenance(**base, source_date=datetime(2026, 9, 30, tzinfo=timezone.utc))


def test_manifest_records_provenance_and_validation(tmp_path):
    result = to_canonical_dataset(load_raw_market_file(write(tmp_path, "prices.csv", HF_CSV), HF))

    manifest_path = write_manifest(tmp_path / "out" / "m.json", result.provenance, result.validation)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["provenance"]["source_type"] == "SECONDARY_DATASET"
    assert manifest["provenance"]["file_sha256"] == result.provenance.file_sha256
    assert manifest["validation"]["invalid_rows"] == 2


def test_stated_source_date_must_match_the_data(tmp_path):
    # e.g. a snapshot fetched "for" 2026-01-05 whose rows say otherwise
    path = write(tmp_path, "snapshot.csv", HF_CSV)
    result = to_canonical_dataset(load_raw_market_file(path, HF, source_date=date(2026, 1, 5)))

    assert result.validation.status is DatasetStatus.FAIL
    assert result.validation.issue_counts["DATE_MISMATCH"] == 2


# --- canonical conversion -------------------------------------------------------

def test_canonical_dataset_flags_but_keeps_invalid_rows(tmp_path):
    result = to_canonical_dataset(load_raw_market_file(write(tmp_path, "prices.csv", HF_CSV), HF))
    data = result.data

    assert list(data.columns) == CANONICAL_COLUMNS
    assert len(data) == 3  # nothing dropped
    assert list(data["validation_status"]) == ["VALID", "INVALID", "INVALID"]
    assert data.loc[1, "validation_warnings"] == "MISSING_VOLUME"
    assert "HIGH_BELOW_OPEN" in data.loc[2, "validation_warnings"]
    assert data.loc[2, "high"] == 19.0  # reported, not repaired


def test_absent_optional_values_stay_null(tmp_path):
    data = to_canonical_dataset(load_raw_market_file(write(tmp_path, "p.csv", HF_CSV), HF)).data

    assert data["turnover"].isna().all()
    assert data["trades"].isna().all()
    assert data["source_timestamp"].isna().all()  # the source gives no per-record timestamp
    assert (data["source"] == HF).all()
    assert (data["source_priority"] == get_source(HF).source_priority).all()


def test_column_map_translates_source_columns(tmp_path):
    text = ("Date,Symbol,Open,High,Low,Close,Volume,Value Traded\n"
            "2026-01-05,ABC,100,103,99,102,150000,15300000\n")
    data = to_canonical_dataset(load_raw_market_file(write(tmp_path, "s.csv", text), SAMPLE)).data

    assert data.loc[0, "turnover"] == 15_300_000
    assert data.loc[0, "date"] == pd.Timestamp("2026-01-05")


def test_missing_required_columns_produce_no_canonical_data(tmp_path):
    text = "symbol,date,close\nJKH.N0000,2026-01-05,20.1\n"
    result = to_canonical_dataset(load_raw_market_file(write(tmp_path, "p.csv", text), HF))

    assert result.data is None
    assert result.validation.status is DatasetStatus.FAIL
    assert result.validation.missing_columns == ["open", "high", "low", "volume"]


def test_column_map_cannot_overwrite_an_existing_column():
    data = pd.DataFrame({"Close": ["1"], "close": ["2"]})
    with pytest.raises(ValueError, match="overwrite"):
        standardize_columns(data, {"Close": "close"})


def test_sample_dataset_converts(tmp_path):
    from app.config.paths import SAMPLE_DIR
    result = to_canonical_dataset(load_raw_market_file(SAMPLE_DIR / "sample_market_data.csv", SAMPLE))

    assert result.validation.total_rows == 75
    assert result.validation.invalid_rows == 0
    assert result.data["turnover"].notna().all()
