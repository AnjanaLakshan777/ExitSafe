"""Persistent market-data tracking: setup, the update job, provenance and client isolation.

Every network source is replaced by a fake; the database is SQLite.
"""

from datetime import date, datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.data.repositories.client_repository import ClientRepository
from app.data.repositories.tracking_repository import (MarketDataUpdateRun, MarketObservation,
                                                       NOT_YET_UPDATED, TrackingRepository)
from app.intelligence import bot
from app.intelligence import market_tracking as mt
from app.intelligence.collectors.http import CollectionError
from app.intelligence.collectors.price_collector import (CSE_SOURCE, GEMINI_SOURCE, CseSnapshot,
                                                         PriceQuote)
from app.intelligence.settings import BotSettings
from app.scheduler import update_market_data
from app.ui.console import run_import

KEY = "test-key-not-real-123"
SETTINGS = BotSettings(gemini_api_key=KEY, price_sources=("cse", "gemini"))
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)          # 17:30 in Colombo, Wednesday
SESSION = date(2026, 10, 7)
SYMBOLS = {"AAA.N0000": 100, "BBB.N0000": 50, "CCC.N0000": 20}   # CCC isn't listed on CSE


def history_csv(last="2026-10-06", symbols=SYMBOLS):
    rows = ["Date,Symbol,Open,High,Low,Close,Volume"]
    for i, day in enumerate(pd.bdate_range("2026-09-01", last)):
        for symbol, base in symbols.items():
            close = base + (i % 7) - 3
            rows.append(f"{day.date()},{symbol},{close},{close + 1},{close - 1},{close},{1000 + i}")
    return ("\n".join(rows) + "\n").encode()


def quote(symbol, close, day=SESSION, source=CSE_SOURCE, **changes):
    values = {"open": close, "high": close + 1, "low": close - 1, "close": close}
    values.update(changes)
    return PriceQuote(symbol, day, values["open"], values["high"], values["low"], values["close"],
                      1500, None, source)


def snapshot(*quotes, listed=("AAA.N0000", "BBB.N0000")):
    return lambda: CseSnapshot(SESSION, {q.symbol: q for q in quotes}, frozenset(listed))


class Gemini:
    """Fake Gemini price lookup that records which symbols it was asked for."""

    def __init__(self, quotes=None, error=None):
        self.quotes, self.error, self.calls = quotes or {}, error, []

    def __call__(self, symbols, settings, today=None):
        self.calls.append(list(symbols))
        if self.error:
            raise self.error
        return {s: q for s, q in self.quotes.items() if s in symbols}


def update(repo, cse=None, gemini=None, closed=True, now=NOW, settings=SETTINGS, **kwargs):
    cse = cse or snapshot(quote("AAA.N0000", 101.5), quote("BBB.N0000", 51.25))
    gemini = gemini if gemini is not None else Gemini(
        {"CCC.N0000": quote("CCC.N0000", 21.0, source=GEMINI_SOURCE)})
    return mt.run_tracking_update(repo, settings, cse_snapshot=cse, market_closed=lambda: closed,
                                  gemini_quotes=gemini, now=now, **kwargs)


@pytest.fixture
def engine():
    return create_engine("sqlite://", poolclass=StaticPool,
                         connect_args={"check_same_thread": False})


@pytest.fixture
def repo(engine):
    ClientRepository(engine).create_tables()
    repository = TrackingRepository(engine)
    repository.create_tables()
    return repository


@pytest.fixture
def clients(engine, repo):
    accounts = ClientRepository(engine)
    return (accounts.add_client("a@example.com", "password1").id,
            accounts.add_client("b@example.com", "password1").id)


@pytest.fixture
def tracked(repo, clients):
    return mt.start_tracking(repo, clients[0], "my_stocks.csv", history_csv(),
                             purpose=mt.START_INVESTING, settings=SETTINGS, now=NOW)


def observations(repo):
    with Session(repo.engine) as session:
        return list(session.scalars(select(MarketObservation).order_by(MarketObservation.id)))


# --- setup ---

def test_upload_creates_a_persistent_tracked_dataset(repo, clients, tracked):
    assert tracked.client_id == clients[0] and tracked.active
    assert tracked.name == "my_stocks" and tracked.purpose == mt.START_INVESTING
    assert tracked.last_status == NOT_YET_UPDATED
    assert tracked.upload.file_name == "my_stocks.csv" and tracked.upload.validation_status == "PASS"
    assert tracked.upload.date_max == date(2026, 10, 6)
    assert repo.upload_content(clients[0], tracked.id) == history_csv()


def test_symbols_come_from_the_symbol_column(tracked):
    assert [(s.symbol, s.upload_last_date) for s in tracked.symbols] == [
        (s, date(2026, 10, 6)) for s in sorted(SYMBOLS)]


def test_file_without_symbol_column_needs_an_entered_symbol(repo, clients):
    content = b"Date,Open,High,Low,Close,Volume\n" + b"".join(
        f"{d.date()},10,11,9,10,100\n".encode() for d in pd.bdate_range("2026-09-01", periods=20))
    with pytest.raises(mt.TrackingSetupError, match="no Symbol column"):
        mt.start_tracking(repo, clients[0], "prices.csv", content, settings=SETTINGS, now=NOW)
    assert repo.datasets_for(clients[0]) == []                  # nothing half-created

    dataset = mt.start_tracking(repo, clients[0], "prices.csv", content, symbol="jkh.n0000",
                                settings=SETTINGS, now=NOW)
    assert [s.symbol for s in dataset.symbols] == ["JKH.N0000"]
    assert dataset.symbol_override == "JKH.N0000"


@pytest.mark.parametrize("content, message", [
    (b"", "empty"),
    (b"Date,Symbol,Close\n2026-10-01,AAA,10\n", "columns are missing"),
    (history_csv(last="2026-10-09"), "after today"),
    (b"Date,Symbol,Open,High,Low,Close,Volume\n" + b"".join(
        f"2026-09-{d:02d},AAA,10,9,11,10,100\n".encode() for d in range(1, 21)), "validation"),
])
def test_unusable_uploads_are_refused(repo, clients, content, message):
    with pytest.raises(mt.TrackingSetupError, match=message):
        mt.start_tracking(repo, clients[0], "x.csv", content, settings=SETTINGS, now=NOW)
    assert repo.datasets_for(clients[0]) == []


def test_ambiguous_day_month_order_is_refused(repo, clients):
    content = b"Date,Symbol,Open,High,Low,Close,Volume\n" + b"".join(
        f"0{m}/0{d}/2026,AAA,10,11,9,10,100\n".encode() for m in (1, 2, 3) for d in range(1, 8))
    with pytest.raises(mt.TrackingSetupError):
        mt.start_tracking(repo, clients[0], "x.csv", content, settings=SETTINGS, now=NOW)


def test_the_same_file_is_not_tracked_twice(repo, clients, tracked):
    with pytest.raises(mt.TrackingSetupError, match="already tracked"):
        mt.start_tracking(repo, clients[0], "copy.csv", history_csv(), settings=SETTINGS, now=NOW)


def test_tracking_survives_a_restart(tmp_path, clients):
    url = f"sqlite:///{tmp_path / 'exitsafe.db'}"
    first = create_engine(url)
    ClientRepository(first).create_tables()
    client = ClientRepository(first).add_client("c@example.com", "password1").id
    TrackingRepository(first).create_tables()
    dataset = mt.start_tracking(TrackingRepository(first), client, "my_stocks.csv", history_csv(),
                                settings=SETTINGS, now=NOW)
    update(TrackingRepository(first))
    first.dispose()

    restarted = TrackingRepository(create_engine(url))           # a new process, same database
    [again] = restarted.datasets_for(client)
    assert again.id == dataset.id and again.active and again.last_status == mt.UPDATED
    assert restarted.upload_content(client, dataset.id) == history_csv()
    assert len(observations(restarted)) == 3
    restarted.engine.dispose()


# --- client isolation ---

def test_clients_cannot_see_or_change_each_others_data(repo, clients, tracked):
    a, b = clients
    assert repo.datasets_for(b) == []
    assert repo.dataset_for(b, tracked.id) is None
    assert repo.upload_content(b, tracked.id) is None
    assert repo.history_for(b, tracked.id) == []
    assert mt.analysis_input(repo, b, tracked.id) is None
    assert repo.set_active(b, tracked.id, False) is False
    assert repo.dataset_for(a, tracked.id).active


def test_a_clients_update_now_only_touches_their_own_datasets(repo, clients, tracked):
    other = mt.start_tracking(repo, clients[1], "b.csv", history_csv(symbols={"BBB.N0000": 50}),
                              settings=SETTINGS, now=NOW)
    result = update(repo, client_id=clients[1], trigger=mt.MANUAL)
    assert set(result.datasets) == {other.id}
    assert repo.dataset_for(clients[0], tracked.id).last_status == NOT_YET_UPDATED


# --- the update job ---

def test_update_stores_new_cse_rows_and_gemini_fallback(repo, clients, tracked):
    result = update(repo)
    assert result.status == mt.COMPLETED and result.rows_added == 3
    rows = {(r.symbol, r.date): r for r in observations(repo)}
    aaa = rows[("AAA.N0000", SESSION)]
    assert (aaa.close, aaa.source, aaa.source_priority) == (101.5, CSE_SOURCE, 2)
    assert aaa.validation_status == "VALID" and aaa.estimated_traded_value == 101.5 * 1500
    ccc = rows[("CCC.N0000", SESSION)]
    assert (ccc.source, ccc.source_priority) == (GEMINI_SOURCE, 5)

    outcome = result.datasets[tracked.id]
    assert outcome.status == mt.UPDATED
    assert [o.outcome for o in outcome.symbols] == [mt.ADDED, mt.ADDED, mt.ADDED_SECONDARY]
    assert outcome.message.startswith("3 of 3 companies updated.")


def test_gemini_is_only_asked_for_symbols_cse_does_not_list(repo, tracked):
    gemini = Gemini({"CCC.N0000": quote("CCC.N0000", 21.0, source=GEMINI_SOURCE)})
    update(repo, gemini=gemini)
    assert gemini.calls == [["CCC.N0000"]]


def test_rerunning_the_update_adds_nothing(repo, clients, tracked):
    update(repo)
    gemini = Gemini()
    again = update(repo, gemini=gemini)
    assert again.rows_added == 0 and len(observations(repo)) == 3
    assert again.datasets[tracked.id].status == mt.UP_TO_DATE
    assert gemini.calls == []                     # CCC already has the latest session
    with Session(repo.engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketDataUpdateRun)) == 2


def test_update_status_is_persisted(repo, clients, tracked):
    update(repo)
    dataset = TrackingRepository(repo.engine).dataset_for(clients[0], tracked.id)
    assert dataset.last_status == mt.UPDATED and dataset.last_run_id is not None
    assert dataset.last_successful_update is not None
    assert {s.symbol: (s.last_market_date, s.last_source, s.last_status) for s in dataset.symbols} \
        == {"AAA.N0000": (SESSION, CSE_SOURCE, mt.ADDED),
            "BBB.N0000": (SESSION, CSE_SOURCE, mt.ADDED),
            "CCC.N0000": (SESSION, GEMINI_SOURCE, mt.ADDED_SECONDARY)}
    [entry] = repo.history_for(clients[0], tracked.id)
    assert (entry.status, entry.rows_added, entry.trigger) == (mt.UPDATED, 3, mt.SCHEDULED)


def test_nothing_is_collected_while_the_market_is_open(repo, clients, tracked):
    def no_snapshot():
        raise AssertionError("prices must not be fetched while the session is open")
    gemini = Gemini()
    result = update(repo, cse=no_snapshot, gemini=gemini, closed=False)
    assert result.status == mt.WAITING_FOR_MARKET_CLOSE and result.retry_later
    assert observations(repo) == [] and gemini.calls == []
    dataset = repo.dataset_for(clients[0], tracked.id)
    assert dataset.last_status == mt.WAITING_FOR_MARKET_CLOSE
    assert dataset.last_successful_update is None


def test_gemini_failure_does_not_stop_cse_updates(repo, clients, tracked):
    result = update(repo, gemini=Gemini(error=CollectionError("Gemini request failed: HTTP 429")))
    assert {r.symbol for r in observations(repo)} == {"AAA.N0000", "BBB.N0000"}   # no CCC price
    outcome = result.datasets[tracked.id]
    assert outcome.status == mt.PARTIALLY_UPDATED
    assert outcome.symbols[2].outcome == mt.NOT_FOUND
    assert "secondary source (Gemini) is unavailable" in outcome.symbols[2].message
    assert any("HTTP 429" in m for m in result.messages)
    assert result.status == mt.COMPLETED_WITH_ERRORS and not result.retry_later


def test_one_invalid_symbol_does_not_stop_the_others(repo, clients, tracked):
    bad = quote("AAA.N0000", 101.0, open=90.0)                     # open below the day's low
    result = update(repo, cse=snapshot(bad, quote("BBB.N0000", 51.0)))
    assert {r.symbol for r in observations(repo)} == {"BBB.N0000", "CCC.N0000"}
    aaa = result.datasets[tracked.id].symbols[0]
    assert aaa.outcome == mt.REJECTED and "LOW_ABOVE_OPEN" in aaa.message
    assert result.datasets[tracked.id].status == mt.PARTIALLY_UPDATED


def test_future_dated_prices_are_rejected(repo, clients, tracked):
    future = quote("AAA.N0000", 101.0, day=date(2026, 10, 8))
    result = update(repo, cse=snapshot(future, quote("BBB.N0000", 51.0)))
    assert "AAA.N0000" not in {r.symbol for r in observations(repo)}
    assert "after today" in result.datasets[tracked.id].symbols[0].message


def test_stock_without_trades_gets_no_row(repo, clients, tracked):
    result = update(repo, cse=snapshot(quote("BBB.N0000", 51.0)))
    assert "AAA.N0000" not in {r.symbol for r in observations(repo)}
    assert result.datasets[tracked.id].symbols[0].outcome == mt.NO_TRADES_TODAY


def test_cse_failure_leaves_nothing_invented(repo, clients, tracked):
    def down():
        raise CollectionError("HTTP 503")
    result = update(repo, cse=down, gemini=Gemini())
    assert observations(repo) == []
    assert result.datasets[tracked.id].status == mt.UPDATE_FAILED
    assert result.retry_later


def test_missing_gemini_key_still_updates_cse_symbols(repo, clients, tracked):
    result = update(repo, settings=BotSettings(price_sources=("cse", "gemini")))
    assert {r.symbol for r in observations(repo)} == {"AAA.N0000", "BBB.N0000"}
    assert "no secondary source is set up" in result.datasets[tracked.id].symbols[2].message


def test_missed_days_are_reported_not_filled(repo, clients):
    dataset = mt.start_tracking(repo, clients[0], "old.csv", history_csv(last="2026-10-02"),
                                settings=SETTINGS, now=NOW)
    result = update(repo)
    message = result.datasets[dataset.id].symbols[0].message
    assert "Earlier weekdays without a price: 2" in message
    assert {r.date for r in observations(repo)} == {SESSION}


def test_an_official_price_replaces_a_secondary_one_for_the_same_day(repo, clients, tracked):
    def down():
        raise CollectionError("HTTP 503")
    gemini = Gemini({s: quote(s, 70.0, source=GEMINI_SOURCE) for s in SYMBOLS})
    update(repo, cse=down, gemini=gemini)                         # CSE down: Gemini for everyone
    result = update(repo)                                         # CSE back
    aaa = result.datasets[tracked.id].symbols[0]
    assert aaa.outcome == mt.ADDED and "replaces the secondary one" in aaa.message

    data = mt.analysis_input(repo, clients[0], tracked.id)
    chosen = {(r.symbol, r.source) for r in data.collected}
    assert ("AAA.N0000", CSE_SOURCE) in chosen and ("AAA.N0000", GEMINI_SOURCE) not in chosen
    assert {r.symbol for r in data.superseded} == {"AAA.N0000", "BBB.N0000"}
    added = [line for line in data.content.decode().splitlines() if "2026-10-07,AAA" in line]
    assert added == ["2026-10-07,AAA.N0000,101.5,102.5,100.5,101.5,1500,cse_trade_summary_current"]


def test_two_clients_tracking_a_symbol_share_one_stored_row(repo, clients, tracked):
    mt.start_tracking(repo, clients[1], "b.csv", history_csv(symbols={"AAA.N0000": 100}),
                      settings=SETTINGS, now=NOW)
    result = update(repo)
    assert [r.symbol for r in observations(repo)].count("AAA.N0000") == 1
    assert all(o.symbols[0].outcome == mt.ADDED for o in result.datasets.values())


def test_paused_dataset_does_not_pick_up_days_collected_for_others(repo, clients, tracked):
    repo.set_active(clients[0], tracked.id, False)
    mt.start_tracking(repo, clients[1], "b.csv", history_csv(symbols={"AAA.N0000": 100}),
                      settings=SETTINGS, now=NOW)
    update(repo)
    assert [r.symbol for r in observations(repo)] == ["AAA.N0000"]        # stored for client B
    assert mt.analysis_input(repo, clients[0], tracked.id).collected == ()
    assert len(mt.analysis_input(repo, clients[1], repo.datasets_for(clients[1])[0].id)
               .collected) == 1


def test_paused_tracking_is_not_updated(repo, clients, tracked):
    assert repo.set_active(clients[0], tracked.id, False)
    assert repo.active_tracked_symbols() == []
    result = update(repo)
    assert result.status == mt.NO_TRACKED_DATA and observations(repo) == []
    assert repo.dataset_for(clients[0], tracked.id).last_status == NOT_YET_UPDATED


def test_a_running_update_blocks_a_second_one(repo, clients, tracked):
    with repo.update_lock() as acquired:
        assert acquired
        blocked = update(repo)
    assert blocked.status == mt.ALREADY_RUNNING and observations(repo) == []
    assert update(repo).rows_added == 3


def test_an_observation_is_stored_once(repo):
    row = {"symbol": "AAA.N0000", "date": SESSION, "open": 1.0, "high": 1.0, "low": 1.0,
           "close": 1.0, "volume": 1.0, "source": CSE_SOURCE, "source_priority": 2,
           "validation_status": "VALID"}
    assert repo.add_observation(**row) is True
    assert repo.add_observation(**row) is False
    assert len(observations(repo)) == 1


def test_an_interrupted_run_is_marked_and_the_next_one_recovers(repo, clients, tracked):
    stale = repo.start_run(mt.SCHEDULED)              # a run that died before finishing
    update(repo)
    with Session(repo.engine) as session:
        assert session.get(MarketDataUpdateRun, stale).status == "INTERRUPTED"
    assert len(observations(repo)) == 3


def test_an_unexpected_error_is_recorded_without_secrets(repo, clients, tracked, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError(f"bad thing with {KEY}")
    monkeypatch.setattr(mt, "_validate", broken)
    result = update(repo)
    assert result.status == mt.FAILED and result.retry_later
    assert KEY not in " ".join(result.messages)
    with Session(repo.engine) as session:
        assert KEY not in str(session.get(MarketDataUpdateRun, result.run_id).details)


def test_no_secret_is_stored_or_shown(repo, clients, tracked):
    result = update(repo, gemini=Gemini(error=CollectionError(f"quota exceeded for {KEY}")))
    assert KEY not in " ".join(result.messages)
    with Session(repo.engine) as session:
        texts = [str(r.details) for r in session.scalars(select(MarketDataUpdateRun))]
    texts += [h.message + str(h.details) for h in repo.history_for(clients[0], tracked.id)]
    assert not any(KEY in text for text in texts)


# --- what the analytics see ---

def test_new_cse_days_reach_the_analytics_and_gemini_rows_do_not(repo, clients, tracked):
    update(repo)
    data = mt.analysis_input(repo, clients[0], tracked.id)
    outcome = run_import(data.file_name, data.content, data.symbol)

    used = outcome.import_result.data
    latest = used[used["date"] == pd.Timestamp(SESSION)]
    assert set(latest["symbol"]) == {"AAA.N0000", "BBB.N0000"}
    assert outcome.selection.ai_sourced_rows == 1                 # CCC's Gemini row, left out
    closes = used[used["symbol"] == "AAA.N0000"].sort_values("date")["close"]
    aaa = outcome.returns[outcome.returns["symbol"] == "AAA.N0000"].iloc[-1]
    assert aaa["date"] == pd.Timestamp(SESSION) and aaa["daily_return"] == pytest.approx(
        101.5 / closes.iloc[-2] - 1)

    included = run_import(data.file_name, data.content, data.symbol, allow_ai_sourced=True)
    assert pd.Timestamp(SESSION) in set(
        included.import_result.data.loc[included.import_result.data["symbol"] == "CCC.N0000",
                                        "date"])


def test_the_original_upload_is_never_rewritten(repo, clients, tracked):
    changed = quote("AAA.N0000", 555.0, day=date(2026, 10, 6))     # a day the upload already has
    result = update(repo, cse=lambda: CseSnapshot(date(2026, 10, 6), {"AAA.N0000": changed},
                                                  frozenset({"AAA.N0000", "BBB.N0000"})))
    assert result.datasets[tracked.id].symbols[0].outcome == mt.UP_TO_DATE
    data = mt.analysis_input(repo, clients[0], tracked.id)
    assert data.original_content == history_csv() == repo.upload_content(clients[0], tracked.id)
    assert "555" not in data.content.decode()


def test_a_symbol_only_file_keeps_working_after_updates(repo, clients):
    content = b"Date,Open,High,Low,Close,Volume\n" + b"".join(
        f"{d.date()},10,11,9,10,100\n".encode() for d in pd.bdate_range("2026-09-01", "2026-10-06"))
    dataset = mt.start_tracking(repo, clients[0], "jkh.csv", content, symbol="AAA.N0000",
                                settings=SETTINGS, now=NOW)
    update(repo)
    data = mt.analysis_input(repo, clients[0], dataset.id)
    outcome = run_import(data.file_name, data.content, data.symbol)
    assert outcome.error is None
    assert outcome.import_result.data["date"].max() == pd.Timestamp(SESSION)


# --- scheduling ---

def test_scheduled_update_without_database_settings_is_skipped_cleanly(monkeypatch):
    def unconfigured():
        raise ValueError("Set DB_NAME (or dbname) in .env")
    monkeypatch.setattr(mt, "TrackingRepository", unconfigured)
    result = mt.run_scheduled_update(SETTINGS)
    assert result.status == mt.DATABASE_NOT_CONFIGURED and not result.retry_later


def test_scheduled_update_uses_the_same_update_code(repo, monkeypatch):
    calls = []
    monkeypatch.setattr(mt, "run_tracking_update",
                        lambda repository, settings, **kwargs: calls.append(kwargs) or
                        mt.TrackingUpdateResult(mt.COMPLETED))
    mt.run_scheduled_update(SETTINGS, repository=repo)
    assert calls == [{"trigger": mt.SCHEDULED}]


@pytest.mark.parametrize("status, cse_failed, code", [
    (mt.COMPLETED, False, 0), (mt.NO_TRACKED_DATA, False, 0),
    (mt.WAITING_FOR_MARKET_CLOSE, False, 0), (mt.COMPLETED_WITH_ERRORS, True, 1),
    (mt.DATABASE_UNAVAILABLE, False, 1),
])
def test_command_line_job_exit_codes(monkeypatch, status, cse_failed, code):
    monkeypatch.setattr(update_market_data, "run_scheduled_update",
                        lambda: mt.TrackingUpdateResult(status, cse_failed=cse_failed))
    assert update_market_data.main() == code


def test_bot_runs_the_tracked_update_and_needs_no_local_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(bot, "run_scheduled_update",
                        lambda settings: mt.TrackingUpdateResult(mt.COMPLETED))
    assert bot.update_prices(BotSettings(tracked_csv=tmp_path / "none.csv")) is True
    monkeypatch.setattr(bot, "run_scheduled_update",
                        lambda settings: mt.TrackingUpdateResult(mt.WAITING_FOR_MARKET_CLOSE))
    assert bot.update_prices(BotSettings(tracked_csv=tmp_path / "none.csv")) is False


@pytest.mark.parametrize("now, last_success, expected", [
    (datetime(2026, 10, 7, 4, 0, tzinfo=timezone.utc), None, datetime(2026, 10, 7, 15, 0)),
    (datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc), None, datetime(2026, 10, 7, 15, 0)),
    (datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc),
     datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc), datetime(2026, 10, 8, 15, 0)),
    (datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc),
     datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc), datetime(2026, 10, 12, 15, 0)),
])
def test_next_update_time(now, last_success, expected):
    assert mt.next_update_time(now, last_success).replace(tzinfo=None) == expected
