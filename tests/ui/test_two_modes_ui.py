"""The two user journeys in the real dashboard script, run headless (no browser)."""

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

import app.ui.auth as auth
from app.data.repositories.client_repository import ClientRepository
from app.data.schemas.event_schema import SourceType
from app.intelligence import dashboard, threat_scan
from app.intelligence.models import RawItem
from app.intelligence.settings import BotSettings
from app.intelligence.threat_store import ThreatStore
from app.ui import decision

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "app", "ui", "streamlit_app.py")
SECTIONS = ["Import Summary", "Canonical Data", "Daily Returns", "Volatility",
            "Covariance / Correlation", "Maximum Drawdown", "Sharpe / Sortino",
            "Value at Risk (VaR)", "Conditional Value at Risk (CVaR)", "Liquidity Analysis",
            "Portfolio Risk Analysis", "Risk-Aware Portfolio Optimization", "Market Regime",
            "Stress Testing", "Backtesting", "Exit Safety Assessment"]
FOUR = "Sample: synthetic backtest data (4 stocks, 321 days)"
THREE = "Sample: three stocks (ABC, LMN, XYZ)"


@pytest.fixture
def client(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    repository = ClientRepository(engine)
    repository.create_tables()
    monkeypatch.setattr(auth, "_repository", lambda: repository)
    store = ThreatStore(tmp_path / "threats.jsonl")
    monkeypatch.setattr(dashboard, "ThreatStore", lambda: store)
    monkeypatch.setattr(threat_scan, "BOT_STATUS_FILE", tmp_path / "status.json")
    return repository.add_client("ann@example.com", "password1", "Ann"), store


def start(client, mode=None, source=None):
    at = AppTest.from_file(SCRIPT, default_timeout=300)
    at.session_state["client"] = {"id": client.id, "email": client.email, "name": "Ann"}
    at.run()
    if mode:
        next(b for b in at.button if b.key == mode).click().run()
    if source:
        at.radio[0].set_value(source).run()
    return at


def enter_shares(at, text="ABC, 100000\nXYZ, 150000\nLMN, 60000"):
    next(t for t in at.text_area if t.key == "invested_holdings").set_value(text).run()


def warnings(at):
    return [w.value for w in at.warning]


def why(at):
    return next(m.value for m in at.markdown if m.value.startswith("- "))


def decision_shown(at):
    """Everything the recommendation says: status, exit result, action and why."""
    boxes = [b.value for b in [*at.success, *at.warning, *at.error, *at.info]
             if b.value.startswith(("Suggested action", "Portfolio status", "Exit Safety"))]
    return boxes, why(at), {m.label: m.value for m in at.metric}


def test_landing_asks_what_the_user_wants_to_do(client):
    at = start(client[0])
    assert not at.exception
    assert at.title[0].value == "ExitSafe"
    assert at.caption[0].value == "Liquidity-Aware Tail-Risk Portfolio Optimizer"
    assert at.subheader[0].value == "What are you trying to do?"
    assert {b.label for b in at.button} >= {"Start Investing", "I Already Invested"}
    assert not at.metric and not at.table                     # no calculations yet


def test_start_investing_gives_a_recommendation_before_the_calculations(client):
    at = start(client[0], "mode_start", FOUR)
    assert not at.exception
    assert next(n for n in at.number_input if n.key == "plan_capital").value is None
    assert any("Enter the amount you want to invest" in i.value for i in at.info)
    next(n for n in at.number_input if n.key == "plan_capital").set_value(5_000_000.0).run()
    headers = [h.value for h in at.header]
    assert headers[:5] == ["Your market data", "Your investment plan", "Your recommendation",
                           "Why are we saying this?", "Important numbers"]
    assert any("Suggested action: **INVEST GRADUALLY**" in w for w in warnings(at))
    assert any(m.value == "#### Your recommended allocation" for m in at.markdown)
    allocation = at.table[0].value
    assert list(allocation.columns) == ["Share", "Amount (Rs.)", "Estimated days to buy"]
    assert "Estimated time to trade" in why(at)
    assert not at.subheader                                   # the 16 sections are hidden

    next(t for t in at.toggle if t.key == "see_calculations").set_value(True).run()
    assert not at.exception
    assert [s.value for s in at.subheader] == SECTIONS
    holdings = next(t for t in at.text_area if t.label == "Holdings (Symbol, Weight %)")
    assert holdings.value.splitlines()[0].startswith("ALPHA, ")   # the recommended mix
    assert next(n for n in at.number_input if n.label == "Portfolio Value (Rs.)").value == 5e6


def test_shares_are_valued_at_the_latest_price_and_missing_prices_are_flagged(client):
    at = start(client[0], "mode_invested", THREE)
    enter_shares(at, "ABC, 1000\nNOPRICE.N0000, 50")
    assert not at.exception
    table = next(t.value for t in at.table if "Current value (Rs.)" in t.value.columns)
    assert table.loc["NOPRICE.N0000", "Current value (Rs.)"] == "Missing current valuation data"
    assert any("Missing current valuation data for: NOPRICE.N0000" in w.value for w in at.warning)
    metrics = {m.label: m.value for m in at.metric}
    price = float(table.loc["ABC", "Latest price (Rs.)"].replace(",", ""))
    assert metrics["Current portfolio value (incomplete)"] == f"Rs. {1000 * price:,.2f}"
    assert metrics["Original investment"] == "Not entered"

    next(n for n in at.number_input if n.key == "invested_target").set_value(1e9).run()
    assert any("larger than the current portfolio value" in e.value for e in at.error)


def test_invalid_start_investing_input_is_explained(client):
    at = start(client[0], "mode_start", FOUR)
    next(n for n in at.number_input if n.key == "plan_capital").set_value(0.0).run()
    assert not at.exception
    assert any("more than zero" in e.value for e in at.error)


def test_already_invested_shows_status_exit_check_and_recommendation(client):
    at = start(client[0], "mode_invested", THREE)
    assert not at.metric and any("Enter your holdings" in i.value for i in at.info)
    next(r for r in at.radio if r.key == "holding_units").set_value(decision.VALUES).run()
    next(t for t in at.text_area if t.key == "invested_holdings").set_value(
        "ABC, 8000000\nXYZ, 7000000\nLMN, 5000000").run()
    assert {m.label: m.value for m in at.metric}["Current portfolio value"] == \
        "Rs. 20,000,000.00"                                     # the sum of the values entered
    assert any("Portfolio status: **AT RISK**" in e.value for e in at.error)
    assert any("Suggested action: **REVIEW YOUR PORTFOLIO**" in w for w in warnings(at))

    next(n for n in at.number_input if n.key == "invested_target").set_value(5_000_000.0).run()
    assert not at.exception
    assert any("Exit Safety for Rs. 5,000,000: **CAUTION**" in w for w in warnings(at))
    assert any("Suggested action: **CONSIDER A STAGED EXIT**" in w for w in warnings(at))
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Estimated exit horizon"] == "15.7 days"
    assert metrics["Liquidity coverage"] == "100.00%"
    headers = [h.value for h in at.header]
    assert headers.index("Portfolio overview") < headers.index("Should I exit?") \
        < headers.index("Recommendation") < headers.index("Why are we saying this?")

    next(t for t in at.toggle if t.key == "see_calculations").set_value(True).run()
    summary = next(t.value for t in at.table if list(t.value.columns) == ["Metric", "Value"]
                   and "Estimated Exit Horizon" in set(t.value["Metric"]))
    assert dict(zip(summary["Metric"], summary["Value"]))["Estimated Exit Horizon"] == \
        "15.7 trading days"                                   # same numbers underneath


def test_market_threats_are_context_only(client):
    ann, store = client
    at = start(ann, "mode_invested", THREE)
    enter_shares(at)
    next(n for n in at.number_input if n.key == "invested_target").set_value(5_000_000.0).run()
    before = decision_shown(at)

    now = datetime.now(timezone.utc)
    threat_scan.run_threat_scan(
        BotSettings(), store=store, send_email=lambda *args, **kwargs: None, now=now,
        collectors=[("fake", lambda since, errors: [RawItem(
            source_name="Wire", source_type=SourceType.NEWS,
            title="Global stock market crash as banks collapse and sovereign default looms",
            source_url="https://example.com/x", published_time=now - timedelta(hours=1),
            fetched_time=now)])])
    assert store.recent()                                     # a real stored threat

    after = start(ann, "mode_invested", THREE)
    enter_shares(after)
    next(n for n in after.number_input if n.key == "invested_target").set_value(5e6).run()
    threats = next(d.value for d in after.dataframe if "Headline" in d.value.columns)
    assert "Global stock market crash" in threats["Headline"].iloc[0]
    assert decision_shown(after) == before                    # nothing changed
