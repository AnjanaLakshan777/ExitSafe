"""Gemini threat search and price fallback, with the SDK client replaced by a fake (no network)."""

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from google.genai import errors
from streamlit.testing.v1 import AppTest

from app.data.schemas.event_schema import SourceType
from app.intelligence import dashboard, price_updater
from app.intelligence.classification.threat_keywords import assess_threat
from app.intelligence.collectors import gemini
from app.intelligence.collectors.http import CollectionError
from app.intelligence.collectors.price_collector import (
    CSE_SOURCE,
    GEMINI_SOURCE,
    CseSnapshot,
    PriceQuote,
    collect_gemini_quotes,
)
from app.intelligence.collectors.web_search_collector import collect_web_search
from app.intelligence.models import RawItem
from app.intelligence.settings import DEFAULT_GEMINI_MODEL, BotSettings, load_settings
from app.intelligence.threat_scan import run_threat_scan
from app.intelligence.threat_store import ThreatStore
from app.ui.console import run_import

KEY = "test-key-not-a-secret"
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
TODAY = date(2026, 10, 6)


class FakeClient:
    """Stands in for genai.Client: records the request and returns a canned reply or error."""

    calls = []

    def __init__(self, reply=None, error=None, cited=("https://example.com/cited",)):
        self.reply, self.error, self.cited = reply, error, cited

    def __call__(self, **kwargs):
        FakeClient.calls.append({"client": kwargs})
        return SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    def _generate(self, **kwargs):
        FakeClient.calls[-1]["request"] = kwargs
        if self.error is not None:
            raise self.error
        chunks = [SimpleNamespace(web=SimpleNamespace(uri=u)) for u in self.cited]
        return SimpleNamespace(text=self.reply, candidates=[
            SimpleNamespace(grounding_metadata=SimpleNamespace(grounding_chunks=chunks))])


@pytest.fixture
def fake_gemini(monkeypatch):
    def install(reply=None, error=None, cited=("https://example.com/cited",)):
        FakeClient.calls = []
        monkeypatch.setattr(gemini.genai, "Client", FakeClient(reply, error, cited))
        return FakeClient.calls
    return install


SETTINGS = BotSettings(gemini_api_key=KEY)


def api_error(code, status, message):
    return errors.ClientError(code, {"error": {"code": code, "status": status,
                                               "message": message}})


# Model configuration

def test_default_model_is_the_maintained_flash_alias():
    assert DEFAULT_GEMINI_MODEL == "gemini-flash-latest"
    assert load_settings({}).gemini_model == "gemini-flash-latest"
    assert load_settings({"GEMINI_MODEL": " gemini-3.8-flash "}).gemini_model == "gemini-3.8-flash"
    assert load_settings({"GEMINI_API_KEY": KEY}).gemini_api_key == KEY     # env-based key


def test_request_uses_the_configured_model_search_timeout_and_one_retry(fake_gemini):
    calls = fake_gemini(reply='{"ok": true}')
    data, cited = gemini.grounded_json("prompt", BotSettings(gemini_api_key=KEY,
                                                             gemini_model="gemini-x"))
    assert data == {"ok": True} and cited == ["https://example.com/cited"]
    client, request = calls[0]["client"], calls[0]["request"]
    assert client["api_key"] == KEY
    options = client["http_options"]
    assert options.timeout == 60_000
    assert options.retry_options.attempts == 2
    assert 429 not in options.retry_options.http_status_codes      # quota errors aren't retried
    assert request["model"] == "gemini-x"
    config = request["config"]
    assert config.temperature == 0.0 and config.tools[0].google_search is not None
    assert config.automatic_function_calling.disable is True


def test_missing_key_fails_before_any_request(fake_gemini):
    calls = fake_gemini(reply="[]")
    with pytest.raises(CollectionError, match="GEMINI_API_KEY is not set"):
        gemini.grounded_json("prompt", BotSettings())
    assert calls == []


@pytest.mark.parametrize("reply", ["no json here", None, "", "[{\"title\": \"broken\"", "```json\n{]\n```"])
def test_malformed_replies_are_rejected(fake_gemini, reply):
    fake_gemini(reply=reply)
    with pytest.raises(CollectionError, match="JSON"):
        gemini.grounded_json("prompt", SETTINGS)


@pytest.mark.parametrize("error, expected", [
    (api_error(404, "NOT_FOUND", "models/x is no longer available"), "isn't available to this API key"),
    (api_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota"), "HTTP 429 RESOURCE_EXHAUSTED"),
    (TimeoutError("read timed out"), "TimeoutError"),
])
def test_api_failures_become_short_controlled_errors(fake_gemini, error, expected):
    fake_gemini(error=error)
    with pytest.raises(CollectionError, match=expected) as caught:
        gemini.grounded_json("prompt", SETTINGS)
    assert len(str(caught.value)) < 400
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_errors_never_contain_the_api_key(fake_gemini):
    fake_gemini(error=RuntimeError(f"bad request for key={KEY}"))
    with pytest.raises(CollectionError) as caught:
        gemini.grounded_json("prompt", SETTINGS)
    assert KEY not in str(caught.value) and "[redacted]" in str(caught.value)


# Threat search

def story(**overrides):
    entry = {"title": "Stocks plunge as bank crisis spreads", "summary": "Markets fall.",
             "source_name": "Reuters", "url": "https://example.com/a", "published": "2026-10-06"}
    entry.update(overrides)
    return entry


def test_threat_search_keeps_valid_stories_as_low_credibility_search_results(fake_gemini):
    fake_gemini(reply=json.dumps([
        story(),
        story(title="Old crash story", published="2026-09-01"),            # before the window
        story(title="Tomorrow's crash", published="2026-10-07"),           # future: rejected
        story(title="Undated crash", published="soon"),                    # unreadable date
        story(title="   "),                                                 # no title
        "not an object",
        story(title="Sell-off widens", url=""),                            # falls back to cited page
    ]))
    items = collect_web_search(NOW - timedelta(hours=24), SETTINGS, now=NOW)
    assert [i.title for i in items] == ["Stocks plunge as bank crisis spreads", "Sell-off widens"]
    assert {i.source_type.value for i in items} == {"SEARCH_RESULT"}
    assert items[0].source_name == "Gemini web search (Reuters)"
    assert items[1].source_url == "https://example.com/cited"


def test_threat_search_rejects_a_non_list_reply(fake_gemini):
    fake_gemini(reply='{"title": "not a list"}')
    with pytest.raises(CollectionError, match="not a JSON list"):
        collect_web_search(NOW - timedelta(hours=24), SETTINGS, now=NOW)


def test_severity_comes_from_exitsafe_rules_not_gemini(fake_gemini, tmp_path, monkeypatch):
    monkeypatch.setattr("app.intelligence.threat_scan.BOT_STATUS_FILE", tmp_path / "s.json")
    fake_gemini(reply=json.dumps([story(), story(title="Company opens new office in Colombo")]))
    store = ThreatStore(tmp_path / "threats.jsonl")
    collectors = [("Gemini web search", lambda since, errors: collect_web_search(since, SETTINGS,
                                                                                 now=NOW))]
    result = run_threat_scan(SETTINGS, store=store, collectors=collectors, now=NOW)
    assert [r.event.title for r in result.new_threats] == ["Stocks plunge as bank crisis spreads"]
    record = result.new_threats[0]
    expected = assess_threat("Stocks plunge as bank crisis spreads", "Markets fall.")
    assert record.event.severity is expected.severity
    assert record.threat_terms == expected.matched_terms
    assert record.event.verification_status.value == "UNVERIFIED"       # never treated as fact


def test_one_failing_source_does_not_stop_the_scan(fake_gemini, tmp_path, monkeypatch):
    monkeypatch.setattr("app.intelligence.threat_scan.BOT_STATUS_FILE", tmp_path / "s.json")
    fake_gemini(error=api_error(429, "RESOURCE_EXHAUSTED", "quota"))

    def crashing(since, errors):
        raise RuntimeError("parser bug")

    def news(since, errors):
        return RawItem(source_name="Feed", source_type=SourceType.NEWS,
                       title="Markets slump on recession fears", source_url="https://example.com/n",
                       published_time=NOW - timedelta(hours=1), fetched_time=NOW)

    collectors = [("Gemini web search", lambda since, errors: collect_web_search(since, SETTINGS,
                                                                                 now=NOW)),
                  ("Broken source", crashing),
                  ("News feeds", lambda since, errors: [news(since, errors)])]
    result = run_threat_scan(SETTINGS, store=ThreatStore(tmp_path / "t.jsonl"),
                             collectors=collectors, now=NOW)
    assert [r.event.title for r in result.new_threats] == ["Markets slump on recession fears"]
    assert any("Gemini web search: Gemini request failed" in e and "429" in e
               for e in result.errors)
    assert any("Broken source: RuntimeError: parser bug" in e for e in result.errors)
    assert not any(KEY in e for e in result.errors)


# Price fallback

def price(**overrides):
    entry = {"symbol": "US1", "date": "2026-10-05", "open": 10.0, "high": 11.0, "low": 9.5,
             "close": 10.4, "volume": 1200, "turnover": None, "source_url": "https://example.com/p"}
    entry.update(overrides)
    return entry


def test_price_fallback_parses_valid_quotes_with_gemini_provenance(fake_gemini):
    fake_gemini(reply=json.dumps([price(), price(symbol="US2", source_url=None)]))
    quotes = collect_gemini_quotes(["US1", "US2"], SETTINGS, today=TODAY)
    assert {s: q.source for s, q in quotes.items()} == {"US1": GEMINI_SOURCE, "US2": GEMINI_SOURCE}
    assert quotes["US1"].close == 10.4 and quotes["US1"].day == date(2026, 10, 5)
    assert quotes["US2"].source_url == "https://example.com/cited"


@pytest.mark.parametrize("bad", [
    {"close": 12.0},                    # close above the high
    {"open": 9.0},                      # open below the low
    {"high": 9.0},                      # high below the low
    {"low": -1.0},                      # not positive
    {"volume": 12.5},                   # not a whole number of shares
    {"volume": None},                   # missing
    {"close": "n/a"},                   # not a number
    {"date": "yesterday"},              # unreadable date
    {"date": "2026-10-07"},             # future
    {"date": "2026-09-20"},             # older than 7 days
    {"symbol": "NOTASKED"},             # wasn't requested
])
def test_price_fallback_rejects_invalid_quotes(fake_gemini, bad):
    fake_gemini(reply=json.dumps([price(**bad)]))
    assert collect_gemini_quotes(["US1"], SETTINGS, today=TODAY) == {}


@pytest.fixture
def tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(price_updater, "PRICE_LOG_FILE", tmp_path / "log.jsonl")
    path = tmp_path / "tracked.csv"
    path.write_text("Date,Symbol,Open,High,Low,Close,Volume,Source\n" + "".join(
        f"2026-09-{d:02d},AAA.N0000,10,11,9.5,{10 + d / 10},100,\n"
        f"2026-09-{d:02d},US1,10,11,9.5,{10 + d / 20},1000,\n"
        for d in range(1, 26) if date(2026, 9, d).weekday() < 5), encoding="utf-8")
    return path


def cse_snapshot():
    quote = PriceQuote("AAA.N0000", TODAY, 10.0, 11.0, 9.5, 10.6, 100, 1060.0, CSE_SOURCE)
    return CseSnapshot(TODAY, {"AAA.N0000": quote}, frozenset({"AAA.N0000"}))


def update(path):
    return price_updater.update_tracked_csv(
        BotSettings(tracked_csv=path, gemini_api_key=KEY), cse_snapshot=cse_snapshot,
        market_closed=lambda: True, now=NOW)


def test_gemini_price_is_appended_with_provenance_and_left_out_of_analytics(fake_gemini, tracked,
                                                                           tmp_path):
    fake_gemini(reply=json.dumps([price(date="2026-10-06")]))
    result = update(tracked)
    assert {q.symbol: q.source for q in result.added} == {"AAA.N0000": CSE_SOURCE,
                                                           "US1": GEMINI_SOURCE}
    lines = tracked.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == "2026-10-06,US1,10,11,9.5,10.4,1200,gemini_web_search"
    log = {e["symbol"]: e for e in map(json.loads, (tmp_path / "log.jsonl").read_text(
        encoding="utf-8").splitlines())}
    assert log["US1"]["source"] == GEMINI_SOURCE and log["US1"]["source_priority"] == 5
    assert log["US1"]["quantitative_analysis"].startswith("excluded by default")

    outcome = run_import(tracked.name, tracked.read_bytes())
    analysed = outcome.import_result.data
    assert GEMINI_SOURCE not in set(analysed["source"])
    assert (analysed["source"] == CSE_SOURCE).sum() == 1
    assert list(outcome.selection.excluded["source"]) == [GEMINI_SOURCE]


def test_cse_update_continues_when_gemini_fails(fake_gemini, tracked):
    before = tracked.read_text(encoding="utf-8")
    fake_gemini(error=api_error(429, "RESOURCE_EXHAUSTED", "quota"))
    result = update(tracked)
    assert [(q.symbol, q.source) for q in result.added] == [("AAA.N0000", CSE_SOURCE)]
    assert result.skipped["US1"] == "no price found"                    # no made-up price
    assert any(e.startswith("Gemini: Gemini request failed") for e in result.errors)
    added = tracked.read_text(encoding="utf-8")[len(before):].splitlines()
    assert added == ["2026-10-06,AAA.N0000,10,11,9.5,10.6,100,cse_trade_summary_current"]


def test_malformed_gemini_price_reply_adds_nothing(fake_gemini, tracked):
    fake_gemini(reply="Sorry, I couldn't find that.")
    result = update(tracked)
    assert [q.symbol for q in result.added] == ["AAA.N0000"]
    assert any("JSON" in e for e in result.errors)


# Dashboard

def _page(which):
    from app.intelligence import dashboard as panels
    if which == "threats":
        panels.show_threat_panel()
    else:
        panels.show_tracked_csv_panel()


def _panel(which):
    return AppTest.from_function(_page, args=(which,)).run()


def test_dashboard_shows_failures_instead_of_crashing(monkeypatch, tmp_path, tracked):
    def broken(*args, **kwargs):
        raise RuntimeError("Gemini exploded")

    settings = BotSettings(gemini_api_key=KEY, tracked_csv=tracked)
    monkeypatch.setattr(dashboard, "load_settings", lambda: settings)
    monkeypatch.setattr(dashboard, "ThreatStore", lambda: ThreatStore(tmp_path / "t.jsonl"))
    monkeypatch.setattr(dashboard, "read_status", lambda: None)
    monkeypatch.setattr(dashboard, "run_threat_scan", broken)
    monkeypatch.setattr(dashboard, "update_tracked_csv", broken)

    threats = _panel("threats")
    next(b for b in threats.button if b.key == "threat_scan").click().run()
    assert not threats.exception
    assert any("The scan could not finish: RuntimeError: Gemini exploded" in e.value
               for e in threats.error)

    prices = _panel("prices")
    next(b for b in prices.button if b.key == "update_prices").click().run()
    assert not prices.exception
    assert any("The price update could not finish" in e.value for e in prices.error)
    page_text = " ".join(str(x.value) for x in [*threats.main, *prices.main] if hasattr(x, "value"))
    assert KEY not in page_text
