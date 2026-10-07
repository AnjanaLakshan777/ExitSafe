"""The tracked-market-data flow in the real dashboard script, run headless (no browser)."""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

import app.ui.auth as auth
from app.data.repositories.client_repository import ClientRepository
from app.intelligence import market_tracking as mt
from app.intelligence.collectors.price_collector import COLOMBO, CSE_SOURCE, CseSnapshot, PriceQuote
from app.intelligence.settings import BotSettings
from app.ui import tracking_panel

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "app", "ui", "streamlit_app.py")
SETTINGS = BotSettings(price_sources=("cse",))
SESSION = pd.bdate_range(end=datetime.now(COLOMBO).date(), periods=1)[0].date()  # latest weekday
SYMBOLS = {"AAA.N0000": 100, "BBB.N0000": 50}


def history_csv():
    rows = ["Date,Symbol,Open,High,Low,Close,Volume"]
    last = SESSION - timedelta(days=1)
    for i, day in enumerate(pd.bdate_range(end=last, periods=70)):
        for symbol, base in SYMBOLS.items():
            close = base + (i % 7) - 3
            rows.append(f"{day.date()},{symbol},{close},{close + 1},{close - 1},{close},{1000 + i}")
    return ("\n".join(rows) + "\n").encode()


@pytest.fixture
def accounts(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    repository = ClientRepository(engine)
    repository.create_tables()
    monkeypatch.setattr(auth, "_repository", lambda: repository)
    monkeypatch.setattr(tracking_panel, "load_settings", lambda: SETTINGS)
    quotes = {s: PriceQuote(s, SESSION, c, c + 1, c - 1, c, 2000, None, CSE_SOURCE)
              for s, c in (("AAA.N0000", 104.5), ("BBB.N0000", 52.5))}
    monkeypatch.setattr(mt, "cse_market_closed", lambda: True)
    monkeypatch.setattr(mt, "collect_cse_quotes",
                        lambda: CseSnapshot(SESSION, quotes, frozenset(quotes)))
    return [repository.add_client(f"{n}@example.com", "password1", n) for n in ("ann", "ben")]


def start(client, mode):
    at = AppTest.from_file(SCRIPT, default_timeout=300)
    at.session_state["client"] = {"id": client.id, "email": client.email, "name": client.full_name}
    at.run()
    next(b for b in at.button if b.key == mode).click().run()
    return at


def test_upload_once_then_keep_using_tracked_data(accounts, monkeypatch):
    ann, ben = accounts
    at = start(ann, "mode_start")
    assert at.radio[0].value == "Upload or paste data"           # nothing tracked yet
    at.file_uploader[0].upload("my_stocks.csv", history_csv(), "text/csv")
    at.run()
    assert next(r for r in at.radio if r.key == "track_purpose").value == mt.START_INVESTING
    next(b for b in at.button if b.key == "start_tracking").click().run()
    assert not at.exception
    assert any(s.value == "Your market data is now being tracked." for s in at.success)
    assert {m.label: m.value for m in at.metric}["Companies tracked"] == "2"

    # a new session: no upload; the tracked data is offered first
    at = start(ann, "mode_invested")
    assert at.radio[0].value == "My tracked market data"
    assert not at.exception
    assert [s.value for s in at.subheader] == ["Tracked Market Data"]
    assert any(s.value == "Market data is updated automatically." for s in at.success)
    next(t for t in at.text_area if t.key == "invested_holdings").set_value(
        "AAA.N0000, 1000\nBBB.N0000, 2000").run()
    assert any(s.value.startswith("Portfolio status") for s in [*at.success, *at.warning,
                                                                *at.error, *at.info])

    calls = []
    real = mt.run_tracking_update
    monkeypatch.setattr(mt, "run_tracking_update",
                        lambda *args, **kwargs: calls.append(kwargs) or real(*args, **kwargs))
    next(b for b in at.button if b.key == "tracking_update_now").click().run()
    assert not at.exception
    assert calls == [{"client_id": ann.id, "trigger": mt.MANUAL}]
    assert any("2 of 2 companies updated." in s.value for s in at.success)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Last market date"] == str(SESSION) and metrics["Update status"] == "Updated"
    # the new CSE close values the holdings
    assert metrics["Current portfolio value"] == f"Rs. {1000 * 104.5 + 2000 * 52.5:,.2f}"
    next(t for t in at.toggle if t.key == "see_calculations").set_value(True).run()
    assert "Exit Safety Assessment" in [s.value for s in at.subheader]
    report = next(t.value for t in at.table if "Date range" in set(t.value.iloc[:, 0]))
    assert str(SESSION) in dict(zip(report.iloc[:, 0], report.iloc[:, 1]))["Date range"]
    assert any("original uploaded history only" in c.value for c in at.caption)

    next(b for b in at.button if b.key == "tracking_toggle").click().run()
    assert any("Tracking is paused" in i.value for i in at.info)

    other = start(ben, "mode_invested")
    other.radio[0].set_value("My tracked market data").run()
    assert any("aren't tracking any market data" in i.value for i in other.info)
    assert not other.metric
