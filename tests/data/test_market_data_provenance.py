"""Per-row source provenance and keeping AI-sourced prices out of the analytics by default."""

import pandas as pd
import pytest

from app.analytics.covariance import calculate_covariance_matrix
from app.analytics.drawdown import calculate_maximum_drawdown
from app.analytics.returns import calculate_daily_returns
from app.analytics.var import calculate_var_summary
from app.analytics.volatility import calculate_annualized_volatility
from app.data.analysis_selection import AI_EXCLUDED_NOTE, AI_SOURCED_WARNING, select_analysis_data
from app.data.loaders.csv_market_loader import load_csv_market_data
from app.data.source_catalog import (
    SOURCE_CATALOG,
    DataSourceType,
    TrustLevel,
    find_source,
    get_source,
)

CSE, GEMINI, UPLOAD = "cse_trade_summary_current", "gemini_web_search", "user_csv_upload"
HEADER = "Date,Symbol,Open,High,Low,Close,Volume,Source\n"


def rows(symbol, closes, source="", start="2026-09-01"):
    days = pd.bdate_range(start, periods=len(closes))
    return "".join(f"{d.date()},{symbol},{c},{c + 1},{c - 1},{c},1000,{source}\n"
                   for d, c in zip(days, closes))


AAA = [100 + i * 0.7 + (i % 3) for i in range(25)]
BBB = [50 + i * 0.3 - (i % 4) for i in range(25)]
OLD = rows("AAA.N0000", AAA) + rows("BBB.N0000", BBB)                 # no source given
CSE_ROW = rows("AAA.N0000", [125.0], CSE, start="2026-10-06")
GEMINI_ROW = rows("BBB.N0000", [57.0], GEMINI, start="2026-10-06")


def load(tmp_path, body, name="prices.csv", header=HEADER):
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return load_csv_market_data(path).data


def by_date(data, symbol, day):
    return data[(data["symbol"] == symbol) & (data["date"] == pd.Timestamp(day))].iloc[0]


def test_cse_row_is_primary_and_gemini_row_is_secondary(tmp_path):
    data = load(tmp_path, OLD + CSE_ROW + GEMINI_ROW)
    cse = by_date(data, "AAA.N0000", "2026-10-06")
    gemini = by_date(data, "BBB.N0000", "2026-10-06")
    assert cse["source"] == CSE and cse["source_priority"] == 2
    assert gemini["source"] == GEMINI and gemini["source_priority"] == 5
    assert gemini["source"] != CSE                                   # never relabelled as CSE
    assert get_source(gemini["source"]).is_ai_generated
    assert not get_source(cse["source"]).is_ai_generated


def test_rows_without_a_source_keep_the_file_source(tmp_path):
    data = load(tmp_path, OLD + CSE_ROW)
    old = data[data["date"] < pd.Timestamp("2026-10-06")]
    assert set(old["source"]) == {UPLOAD}                            # nothing made up
    assert set(old["source_priority"]) == {get_source(UPLOAD).source_priority}


def test_csv_without_a_source_column_imports_as_before(tmp_path):
    header = "Date,Symbol,Open,High,Low,Close,Volume\n"
    body = "".join(line.rsplit(",", 1)[0] + "\n" for line in OLD.splitlines())
    data = load(tmp_path, body, header=header)
    assert len(data) == 50 and set(data["source"]) == {UPLOAD}
    selection = select_analysis_data(data)
    assert selection.ai_sourced_rows == 0 and selection.message is None
    pd.testing.assert_frame_equal(selection.data, data)


def test_unknown_source_label_is_flagged_and_keeps_the_file_source(tmp_path):
    data = load(tmp_path, OLD + rows("AAA.N0000", [125.0], "Investing.com", start="2026-10-06"))
    row = by_date(data, "AAA.N0000", "2026-10-06")
    assert row["source"] == UPLOAD and row["validation_status"] == "WARNING"
    assert "UNKNOWN_SOURCE_LABEL" in row["validation_warnings"]
    assert len(select_analysis_data(data).data) == len(data)        # still used


def test_default_analysis_data_excludes_gemini_and_keeps_cse(tmp_path):
    data = load(tmp_path, OLD + CSE_ROW + GEMINI_ROW)
    selection = select_analysis_data(data)
    assert selection.ai_sourced_rows == 1 and not selection.allow_ai_sourced
    assert GEMINI not in set(selection.data["source"])
    assert len(selection.data) == len(data) - 1
    assert list(selection.excluded["source"]) == [GEMINI]
    assert (selection.data["source"] == CSE).sum() == 1             # CSE row still there
    assert selection.message == AI_EXCLUDED_NOTE


def test_ai_sourced_rows_can_be_included_explicitly_with_a_warning(tmp_path):
    data = load(tmp_path, OLD + CSE_ROW + GEMINI_ROW)
    selection = select_analysis_data(data, allow_ai_sourced=True)
    assert len(selection.data) == len(data) and selection.excluded.empty
    assert selection.uses_ai_sourced and selection.message == AI_SOURCED_WARNING
    assert "not official exchange data" in selection.message
    assert (selection.data["source"] == GEMINI).sum() == 1          # still marked as Gemini


def test_selection_does_not_mutate_its_input(tmp_path):
    data = load(tmp_path, OLD + CSE_ROW + GEMINI_ROW)
    before = data.copy(deep=True)
    select_analysis_data(data)
    pd.testing.assert_frame_equal(data, before)


def test_calculations_match_on_the_same_approved_data(tmp_path):
    mixed = select_analysis_data(load(tmp_path, OLD + CSE_ROW + GEMINI_ROW)).data
    approved_only = load(tmp_path, OLD + CSE_ROW, name="approved.csv")
    for calculate in (calculate_daily_returns, calculate_annualized_volatility,
                      calculate_var_summary, calculate_covariance_matrix,
                      calculate_maximum_drawdown):
        pd.testing.assert_frame_equal(calculate(mixed), calculate(approved_only))


def test_catalog_marks_gemini_as_secondary_ai_data():
    gemini, cse = get_source(GEMINI), get_source(CSE)
    assert gemini.source_type is DataSourceType.AI_GENERATED
    assert gemini.trust_level is TrustLevel.AI_GENERATED
    assert not gemini.historical_backfill_allowed and gemini.enabled
    assert cse.source_type is DataSourceType.OFFICIAL_CSE and cse.enabled
    assert cse.source_priority < gemini.source_priority
    assert gemini.source_priority < get_source("exitsafe_sample").source_priority
    assert [s.source_name for s in SOURCE_CATALOG if s.is_ai_generated] == [GEMINI]
    assert find_source("not-registered") is None


@pytest.mark.parametrize("label", [GEMINI, f"  {GEMINI}  "])
def test_source_labels_are_matched_after_trimming(tmp_path, label):
    data = load(tmp_path, OLD + rows("BBB.N0000", [57.0], label, start="2026-10-06"))
    assert by_date(data, "BBB.N0000", "2026-10-06")["source"] == GEMINI
