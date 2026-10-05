import pytest

from app.data.schemas.market_schema import CANONICAL_COLUMNS
from app.data.source_catalog import (
    SOURCE_CATALOG,
    Coverage,
    DataSourceType,
    TrustLevel,
    get_source,
)


def test_source_names_are_unique():
    names = [s.source_name for s in SOURCE_CATALOG]
    assert len(names) == len(set(names))


def test_every_source_documents_usage_terms():
    for source in SOURCE_CATALOG:
        assert source.license_or_usage_note.strip(), source.source_name
        assert source.description.strip(), source.source_name


def test_only_cse_domains_are_marked_official_cse():
    for source in SOURCE_CATALOG:
        if source.source_type is DataSourceType.OFFICIAL_CSE:
            assert source.domain == "cse.lk", source.source_name


def test_secondary_datasets_are_not_official_or_canonical():
    secondary = [s for s in SOURCE_CATALOG if s.source_type is DataSourceType.SECONDARY_DATASET]
    assert secondary
    for source in secondary:
        assert source.trust_level in (TrustLevel.LOW, TrustLevel.UNVERIFIED)
        assert not source.historical_backfill_allowed


def test_huggingface_dataset_is_secondary():
    source = get_source("hf_tharu_jwd_cse_market_data")
    assert source.source_type is DataSourceType.SECONDARY_DATASET
    assert "NOT an official CSE source" in source.notes


def test_current_snapshots_cannot_backfill_history():
    for source in SOURCE_CATALOG:
        if source.historical_or_current is Coverage.CURRENT:
            assert not source.historical_backfill_allowed, source.source_name


def test_synthetic_data_has_lowest_priority():
    sample = get_source("exitsafe_sample")
    assert sample.trust_level is TrustLevel.SYNTHETIC
    assert sample.source_priority == max(s.source_priority for s in SOURCE_CATALOG)


def test_official_sources_outrank_secondary_ones():
    official = get_source("cse_historical_official")
    secondary = get_source("hf_tharu_jwd_cse_market_data")
    assert official.source_priority < secondary.source_priority


def test_column_maps_target_canonical_columns():
    for source in SOURCE_CATALOG:
        assert set(source.column_map.values()) <= set(CANONICAL_COLUMNS), source.source_name


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        get_source("not_a_source")
