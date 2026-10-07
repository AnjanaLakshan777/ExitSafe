"""The price updater records each row's source, and Gemini rows stay out of analysis by default."""

import json
from datetime import date, datetime, timezone

import pytest

from app.intelligence import price_updater
from app.intelligence.collectors.price_collector import (
    CSE_SOURCE,
    GEMINI_SOURCE,
    CseSnapshot,
    PriceQuote,
)
from app.intelligence.dashboard import source_label
from app.intelligence.settings import BotSettings
from app.ui.console import ai_sourced_rows_display, run_import

NOW = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
OLD_ROWS = ("Date,Symbol,Open,High,Low,Close,Volume\r\n"
            + "".join(f"2026-09-{d:02d},AAA.N0000,{9 + d / 10},{10 + d / 10},{9 + d / 10},"
                      f"{9.5 + d / 10},50\r\n"
                      f"2026-09-{d:02d},US1,1,{2 + d / 100},1,{1.5 + d / 100},10\r\n"
                      for d in range(1, 26) if date(2026, 9, d).weekday() < 5))


def quote(symbol, source, close=10.4):
    return PriceQuote(symbol, date(2026, 10, 6), 10.0, 11.0, 9.5, close, 100, 1040.0, source)


@pytest.fixture
def tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(price_updater, "PRICE_LOG_FILE", tmp_path / "log.jsonl")
    path = tmp_path / "tracked.csv"
    path.write_bytes(b"\xef\xbb\xbf" + OLD_ROWS.encode("utf-8"))   # BOM + CRLF, like Excel
    return path


def update(path):
    snapshot = CseSnapshot(date(2026, 10, 6), {"AAA.N0000": quote("AAA.N0000", CSE_SOURCE)},
                           frozenset({"AAA.N0000"}))
    return price_updater.update_tracked_csv(
        BotSettings(tracked_csv=path, gemini_api_key="k"), cse_snapshot=lambda: snapshot,
        market_closed=lambda: True, now=NOW,
        gemini_quotes=lambda symbols, settings, today: {"US1": quote("US1", GEMINI_SOURCE, 1.9)})


def test_each_new_row_names_its_source(tracked):
    result = update(tracked)
    assert {q.symbol: q.source for q in result.added} == {"AAA.N0000": CSE_SOURCE,
                                                           "US1": GEMINI_SOURCE}
    lines = tracked.read_text(encoding="utf-8-sig").splitlines()
    assert lines[-2].endswith(",cse_trade_summary_current")
    assert lines[-1].endswith(",gemini_web_search")                 # not labelled as CSE


def test_older_file_gets_a_source_column_without_changing_old_values(tracked):
    before = tracked.read_bytes()
    update(tracked)
    after = tracked.read_bytes()
    assert after.startswith(b"\xef\xbb\xbf")                         # BOM kept
    old_lines = before[3:].decode("utf-8").split("\r\n")
    new_lines = after[3:].decode("utf-8").split("\r\n")
    assert new_lines[0] == old_lines[0] + ",Source"
    for old, new in zip(old_lines[1:-1], new_lines[1:]):
        assert new == old + ","                                      # empty source, not invented


def test_file_with_a_source_column_is_not_given_another(tracked):
    update(tracked)
    header = tracked.read_text(encoding="utf-8-sig").splitlines()[0]
    snapshot = CseSnapshot(date(2026, 10, 7), {"AAA.N0000": PriceQuote(
        "AAA.N0000", date(2026, 10, 7), 10, 11, 9.5, 10.6, 100, None, CSE_SOURCE)},
        frozenset({"AAA.N0000", "US1"}))
    price_updater.update_tracked_csv(BotSettings(tracked_csv=tracked, price_sources=("cse",)),
                                     cse_snapshot=lambda: snapshot, market_closed=lambda: True,
                                     now=NOW)
    lines = tracked.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == header and header.count("Source") == 1
    assert lines[-1].endswith(",cse_trade_summary_current")


def test_log_records_source_priority_and_analysis_use(tracked, tmp_path):
    update(tracked)
    entries = {e["symbol"]: e for e in map(json.loads, (tmp_path / "log.jsonl").read_text(
        encoding="utf-8").splitlines())}
    assert entries["AAA.N0000"]["source"] == CSE_SOURCE
    assert entries["AAA.N0000"]["source_priority"] == 2
    assert entries["AAA.N0000"]["quantitative_analysis"] == "included"
    assert entries["US1"]["source"] == GEMINI_SOURCE and entries["US1"]["source_priority"] == 5
    assert entries["US1"]["quantitative_analysis"].startswith("excluded by default")
    assert entries["US1"]["written_time"] == NOW.isoformat()


def test_updated_file_reaches_the_analytics_without_gemini_rows(tracked):
    update(tracked)
    outcome = run_import(tracked.name, tracked.read_bytes())
    data = outcome.import_result.data
    assert outcome.error is None and outcome.selection.ai_sourced_rows == 1
    assert GEMINI_SOURCE not in set(data["source"])
    assert (data["source"] == CSE_SOURCE).sum() == 1                # the CSE row is analysed
    assert set(data.loc[data["date"] < "2026-10-06", "source"]) == {"user_csv_upload"}
    assert "2026-10-06" not in set(outcome.returns.loc[outcome.returns["symbol"] == "US1",
                                                       "date"].astype(str))
    excluded = ai_sourced_rows_display(outcome.selection)
    assert list(excluded["Source"]) == [GEMINI_SOURCE] and list(excluded["Symbol"]) == ["US1"]


def test_gemini_rows_can_be_included_explicitly(tracked):
    update(tracked)
    outcome = run_import(tracked.name, tracked.read_bytes(), allow_ai_sourced=True)
    assert (outcome.import_result.data["source"] == GEMINI_SOURCE).sum() == 1
    assert outcome.selection.uses_ai_sourced
    assert "not official exchange data" in outcome.selection.message


def test_all_ai_sourced_file_leaves_nothing_to_analyse(tmp_path):
    path = tmp_path / "ai.csv"
    path.write_text("Date,Symbol,Open,High,Low,Close,Volume,Source\n"
                    "2026-10-06,US1,1,2,1,1.5,10,gemini_web_search\n", encoding="utf-8")
    outcome = run_import(path.name, path.read_bytes())
    assert outcome.returns is None and "secondary AI-sourced" in outcome.error


def test_dashboard_labels_sources():
    assert source_label(CSE_SOURCE) == "Official CSE (cse_trade_summary_current)"
    assert source_label(GEMINI_SOURCE) == "Secondary AI-sourced (gemini_web_search)"
