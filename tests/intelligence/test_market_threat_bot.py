"""Market-threat bot: threat keywords, collectors (offline), storage, alerts and price updates."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.data.schemas.event_schema import Severity, SourceType, VerificationStatus
from app.intelligence import price_updater, threat_scan
from app.intelligence.classification.threat_keywords import assess_threat
from app.intelligence.collectors import news_collector
from app.intelligence.collectors.gemini import extract_json
from app.intelligence.collectors.http import CollectionError
from app.intelligence.collectors.price_collector import (
    CSE_SOURCE,
    GEMINI_SOURCE,
    CseSnapshot,
    PriceQuote,
    parse_cse_trade_summary,
    parse_gemini_quotes,
)
from app.intelligence.models import RawItem
from app.intelligence.notifier import build_alert_email
from app.intelligence.settings import BotSettings, load_settings
from app.intelligence.sources import FeedSource, PageSource
from app.intelligence.threat_store import ThreatStore

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


def item(title, body="", source_type=SourceType.NEWS, hours_ago=1, url=None):
    return RawItem(source_name="Test Wire", source_type=source_type, title=title, body=body,
                   source_url=url or f"https://news.example/{abs(hash(title))}",
                   published_time=NOW - timedelta(hours=hours_ago), fetched_time=NOW)


# --- threat keywords ---------------------------------------------------------------------------

@pytest.mark.parametrize("title, severity", [
    ("Global stocks plunge as recession fears grow", Severity.HIGH),
    ("Stock market crash wipes $2 trillion off US shares", Severity.CRITICAL),
    ("Inflation data due on Thursday", Severity.MEDIUM),
    ("Oil prices tumble, sell-off deepens as war spreads", Severity.CRITICAL),   # 3 terms escalate
])
def test_threat_severity(title, severity):
    assert assess_threat(title).severity is severity


def test_ordinary_news_is_not_a_threat():
    assessment = assess_threat("Company opens new store in Colombo")
    assert not assessment.is_threat and assessment.matched_terms == ()


def test_word_matching_avoids_false_positives():
    assert not assess_threat("Shipping routes reopen after storm").is_threat     # not "rout"
    assert not assess_threat("Winners receive an award").is_threat               # not "war"
    assert assess_threat("Shares plunged on Monday").matched_terms == ("plunge",)


def test_terms_only_in_description_cap_at_medium():
    assessment = assess_threat("Tories unite behind new leader",
                               "Critics fear the economy could slide into war-like recession")
    assert assessment.severity is Severity.MEDIUM
    assert assessment.matched_terms == ("recession", "war")


def test_calming_headline_lowers_severity():
    assert assess_threat("S&P 500 hits record high as stocks shrug off bond slump").severity         is Severity.MEDIUM
    assert assess_threat("Stocks slump on bond fears").severity is Severity.HIGH


def test_longer_phrase_replaces_contained_term():
    assert assess_threat("US-China trade war escalates").matched_terms == ("trade war",)


# --- collectors (no network) -------------------------------------------------------------------

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Stocks plunge worldwide</title><link>https://n.example/a</link>
<pubDate>Tue, 06 Oct 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;Big &amp;amp; fast&lt;/p&gt;</description></item>
<item><title>Old story</title><link>https://n.example/b</link>
<pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
<item><title>Undated story</title><link>https://n.example/c</link></item>
</channel></rss>"""

PAGE = """<html><body>
<a href="/2026/10/06/stocks-slump">Asian stocks slump as investors flee risky assets</a>
<a href="/2026/09/01/old-story">An old story that is well outside the window here</a>
<a href="/about">About us and our long history of business journalism</a>
<a href="/2026/10/06/short">Too short</a>
</body></html>"""


def test_collect_feed_keeps_recent_dated_entries(monkeypatch):
    monkeypatch.setattr(news_collector, "fetch", lambda url: SimpleNamespace(content=RSS))
    items = news_collector.collect_feed(FeedSource("Wire", "https://n.example/rss"),
                                        since=NOW - timedelta(days=1), now=NOW)
    assert [i.title for i in items] == ["Stocks plunge worldwide"]
    assert items[0].published_time == datetime(2026, 10, 6, 10, tzinfo=timezone.utc)
    assert items[0].source_type is SourceType.NEWS


def test_collect_page_uses_dated_article_links(monkeypatch):
    calls = []
    def fake_fetch(url, check_robots=False):
        calls.append(check_robots)
        return SimpleNamespace(text=PAGE)
    monkeypatch.setattr(news_collector, "fetch", fake_fetch)
    source = PageSource("Site", "https://site.example/",
                        r"^https://site\.example/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/")
    items = news_collector.collect_page(source, since=NOW - timedelta(days=1), now=NOW)
    assert [i.title for i in items] == ["Asian stocks slump as investors flee risky assets"]
    assert items[0].source_url == "https://site.example/2026/10/06/stocks-slump"
    assert calls == [True]                      # robots.txt is always checked for pages


def test_collect_news_reports_failing_source_and_continues(monkeypatch):
    def fake_fetch(url, **kwargs):
        if "bad" in url:
            raise CollectionError("HTTP 403")
        return SimpleNamespace(content=RSS)
    monkeypatch.setattr(news_collector, "fetch", fake_fetch)
    errors = []
    items = news_collector.collect_news(
        NOW - timedelta(days=1),
        feeds=(FeedSource("Bad", "https://bad.example"), FeedSource("Good", "https://ok.example")),
        pages=(), errors=errors)
    assert len(items) == 1 and errors == ["Bad: HTTP 403"]


def test_extract_json_ignores_fences_and_prose():
    assert extract_json('Here you go:\n```json\n[{"a": 1}]\n```') == [{"a": 1}]
    with pytest.raises(CollectionError):
        extract_json("no data today")


# --- scan, store and alert ---------------------------------------------------------------------

@pytest.fixture
def status_file(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    monkeypatch.setattr(threat_scan, "BOT_STATUS_FILE", path)
    return path


EMAIL_SETTINGS = BotSettings(smtp_user="bot@example.com", smtp_password="pw",
                             alert_email_to=("me@example.com",), use_web_search=False)


def test_scan_stores_threats_and_emails_once(tmp_path, status_file):
    store = ThreatStore(tmp_path / "threats.jsonl")
    sent = []
    news = [item("Global stocks plunge as recession fears grow"),
            item("GLOBAL STOCKS PLUNGE as recession fears grow!", url="https://other/x"),  # dup
            item("Inflation data due on Thursday"),
            item("Company opens new store")]
    collectors = [("fake", lambda since, errors: news)]

    result = threat_scan.run_threat_scan(EMAIL_SETTINGS, store=store, collectors=collectors,
                                         send_email=lambda r, s: sent.append(r), now=NOW)
    assert result.items_collected == 4
    assert len(result.new_threats) == 2                       # HIGH + MEDIUM, duplicate dropped
    assert [r.event.severity for r in sent[0]] == [Severity.HIGH]   # MEDIUM is below HIGH
    record = result.new_threats[0]
    assert record.event.verification_status is VerificationStatus.REPORTED
    assert record.event.confidence == 0.6
    assert status_file.exists()

    # Reloaded from disk, the same news is neither stored nor emailed again.
    store = ThreatStore(tmp_path / "threats.jsonl")
    again = threat_scan.run_threat_scan(EMAIL_SETTINGS, store=store, collectors=collectors,
                                        send_email=lambda r, s: sent.append(r), now=NOW)
    assert again.new_threats == [] and len(sent) == 1


def test_scan_without_email_settings_notes_it(tmp_path, status_file):
    store = ThreatStore(tmp_path / "threats.jsonl")
    result = threat_scan.run_threat_scan(
        BotSettings(use_web_search=False), store=store,
        collectors=[("fake", lambda since, errors: [item("Bank run fears hit lenders")])], now=NOW)
    assert result.alerted == []
    assert any("not emailed" in note for note in result.notes)


def test_failed_email_is_retried_next_scan(tmp_path, status_file):
    store = ThreatStore(tmp_path / "threats.jsonl")
    collectors = [("fake", lambda since, errors: [item("Sovereign default feared")])]

    def broken(records, settings):
        raise OSError("SMTP down")
    first = threat_scan.run_threat_scan(EMAIL_SETTINGS, store=store, collectors=collectors,
                                        send_email=broken, now=NOW)
    assert any("SMTP down" in e for e in first.errors)
    sent = []
    threat_scan.run_threat_scan(EMAIL_SETTINGS, store=store, collectors=collectors,
                                send_email=lambda r, s: sent.append(r), now=NOW)
    assert len(sent) == 1


def test_alert_email_lists_threats_with_sources():
    record = threat_scan.threat_record(item("Stock market crash in Asia"))
    message = build_alert_email([record], "bot@example.com", ("me@example.com",))
    assert "CRITICAL" in message["Subject"]
    body = message.get_body(("plain",)).get_content()
    assert "Stock market crash in Asia" in body and "news report (unconfirmed)" in body


def test_settings_from_env():
    settings = load_settings({"GEMINI_API_KEY": " key ", "ALERT_EMAIL_TO": "a@x.com, b@x.com",
                              "SMTP_USER": "u", "SMTP_PASSWORD": "p",
                              "ALERT_MIN_SEVERITY": "critical", "PRICE_SOURCES": "cse"})
    assert settings.gemini_api_key == "key" and settings.email_configured
    assert settings.alert_email_to == ("a@x.com", "b@x.com")
    assert settings.alert_min_severity is Severity.CRITICAL
    assert settings.price_sources == ("cse",)
    with pytest.raises(ValueError):
        load_settings({"PRICE_SOURCES": "bloomberg"})


# --- prices ------------------------------------------------------------------------------------

def cse_row(symbol, volume=100, day_ms=1791276925401, **overrides):
    row = {"symbol": symbol, "open": 10.0, "high": 11.0, "low": 9.5, "price": 10.5,
           "closingPrice": 10.4, "sharevolume": volume, "turnover": 1040.0,
           "lastTradedTime": day_ms}
    row.update(overrides)
    return row


def test_parse_cse_trade_summary():
    old = 1791276925401 - 3 * 86_400_000
    snapshot = parse_cse_trade_summary({"reqTradeSummery": [
        cse_row("aaa.n0000"), cse_row("BBB.N0000", volume=0), cse_row("CCC.N0000", day_ms=old),
        cse_row("DDD.N0000", open=None)]})
    assert snapshot.session == date(2026, 10, 6)
    assert list(snapshot.quotes) == ["AAA.N0000"]
    assert snapshot.quotes["AAA.N0000"].close == 10.4               # official closing price
    assert snapshot.listed == {"AAA.N0000", "BBB.N0000", "CCC.N0000", "DDD.N0000"}


def test_parse_gemini_quotes_rejects_incomplete_future_and_stale():
    today = date(2026, 10, 6)
    good = {"symbol": "AAPL", "date": "2026-10-05", "open": 1, "high": 2, "low": 1, "close": 1.5,
            "volume": 10}
    quotes = parse_gemini_quotes([
        good,
        {**good, "symbol": "MSFT", "volume": None},                 # missing volume
        {**good, "symbol": "TSLA", "date": "2026-10-07"},           # future
        {**good, "symbol": "NVDA", "date": "2026-09-01"},           # stale
        {**good, "symbol": "NOTASKED"},
    ], ["AAPL", "MSFT", "TSLA", "NVDA"], today)
    assert list(quotes) == ["AAPL"] and quotes["AAPL"].source == GEMINI_SOURCE


def quote(symbol, day=date(2026, 10, 6), source=CSE_SOURCE):
    return PriceQuote(symbol, day, 10.0, 11.0, 9.5, 10.4, 100, 1040.0, source)


@pytest.fixture
def tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(price_updater, "PRICE_LOG_FILE", tmp_path / "log.jsonl")
    path = tmp_path / "prices.csv"
    path.write_text("Date,Symbol,Open,High,Low,Close,Volume,Value Traded,Notes\n"
                    "2026-10-05,AAA.N0000,9,10,9,9.8,50,490,x\n"
                    "2026-10-05,US1,1,2,1,1.5,10,15,\n", encoding="utf-8")
    return path


def test_update_appends_rows_in_file_layout(tracked):
    snapshot = CseSnapshot(date(2026, 10, 6), {"AAA.N0000": quote("AAA.N0000")},
                           frozenset({"AAA.N0000"}))
    asked = []
    def gemini(symbols, settings, today):
        asked.append(symbols)
        return {"US1": quote("US1", source=GEMINI_SOURCE)}
    settings = BotSettings(tracked_csv=tracked, gemini_api_key="k")

    result = price_updater.update_tracked_csv(settings, cse_snapshot=lambda: snapshot,
                                              market_closed=lambda: True, gemini_quotes=gemini,
                                              now=NOW)
    assert asked == [["US1"]]                         # Gemini only for the non-CSE symbol
    assert [q.symbol for q in result.added] == ["AAA.N0000", "US1"]
    lines = tracked.read_text(encoding="utf-8").splitlines()
    # The file had no Source column, so one is added; the new rows name their source.
    assert lines[0].endswith(",Notes,Source")
    assert lines[-2] == "2026-10-06,AAA.N0000,10,11,9.5,10.4,100,1040,,cse_trade_summary_current"
    assert lines[-1].endswith(",gemini_web_search")
    assert tracked.with_suffix(".csv.bak").exists()

    # Running again adds nothing: those dates are already in the file.
    again = price_updater.update_tracked_csv(settings, cse_snapshot=lambda: snapshot,
                                             market_closed=lambda: True, gemini_quotes=gemini,
                                             now=NOW)
    assert again.added == [] and "already in file" in again.skipped["AAA.N0000"]


def test_untraded_cse_stock_gets_no_row_and_no_gemini_lookup(tracked):
    snapshot = CseSnapshot(date(2026, 10, 6), {}, frozenset({"AAA.N0000"}))
    settings = BotSettings(tracked_csv=tracked, price_sources=("cse",))
    result = price_updater.update_tracked_csv(settings, cse_snapshot=lambda: snapshot,
                                              market_closed=lambda: True, now=NOW)
    assert result.added == []
    assert result.skipped["AAA.N0000"] == "no trades on CSE in the latest session"


def test_no_update_while_cse_session_is_open(tracked):
    before = tracked.read_text(encoding="utf-8")
    result = price_updater.update_tracked_csv(BotSettings(tracked_csv=tracked),
                                              market_closed=lambda: False, now=NOW)
    assert result.added == [] and "still open" in result.errors[0]
    assert tracked.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("values, expected", [
    (["2026-10-05", "2026-10-06"], "%Y-%m-%d"),
    (["25/09/2026", "05/10/2026"], "%d/%m/%Y"),
    (["06-Oct-2026"], "%d-%b-%Y"),
])
def test_detect_date_format(values, expected):
    assert price_updater.detect_date_format(values) == expected


def test_ambiguous_dates_are_refused():
    with pytest.raises(price_updater.TrackedCsvError, match="day/month"):
        price_updater.detect_date_format(["05/10/2026", "06/10/2026"])


def test_semicolon_csv_keeps_its_delimiter(tmp_path, monkeypatch):
    monkeypatch.setattr(price_updater, "PRICE_LOG_FILE", tmp_path / "log.jsonl")
    path = tmp_path / "p.csv"
    path.write_text("Date;Symbol;Open;High;Low;Close;Volume\n2026-10-05;AAA.N0000;9;10;9;9.8;50",
                    encoding="utf-8")                      # no trailing newline
    snapshot = CseSnapshot(date(2026, 10, 6), {"AAA.N0000": quote("AAA.N0000")},
                           frozenset({"AAA.N0000"}))
    price_updater.update_tracked_csv(BotSettings(tracked_csv=path, price_sources=("cse",)),
                                     cse_snapshot=lambda: snapshot, market_closed=lambda: True,
                                     now=NOW)
    assert path.read_text(encoding="utf-8").splitlines()[-1] == \
        "2026-10-06;AAA.N0000;10;11;9.5;10.4;100;cse_trade_summary_current"


def test_quote_validation():
    with pytest.raises(ValueError):
        replace(quote("X"), high=1.0, low=2.0)
