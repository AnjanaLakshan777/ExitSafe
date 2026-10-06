"""Daily closing prices: official CSE data first, Gemini web search only as a labelled fallback.

The CSE ``tradeSummary`` endpoint is what the cse.lk website itself calls (see
docs/CSE_DATA_DISCOVERY.md). A stock with no trades in the session gets no
quote: a missing day is left missing rather than filled with a stale price.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from app.intelligence.collectors.gemini import grounded_json
from app.intelligence.collectors.http import CollectionError, fetch

CSE_TRADE_SUMMARY_URL = "https://www.cse.lk/api/tradeSummary"
CSE_MARKET_STATUS_URL = "https://www.cse.lk/api/marketStatus"
COLOMBO = timezone(timedelta(hours=5, minutes=30))

CSE_SOURCE = "CSE tradeSummary"
GEMINI_SOURCE = "Gemini web search"
# Gemini quotes older than this are treated as stale and rejected.
MAX_GEMINI_QUOTE_AGE_DAYS = 7


@dataclass(frozen=True)
class PriceQuote:
    symbol: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: float | None
    source: str
    source_url: str | None = None

    def __post_init__(self):
        for name in ("open", "high", "low", "close"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{self.symbol}: {name} must be positive")
        if self.high < self.low:
            raise ValueError(f"{self.symbol}: high is below low")
        if self.volume < 0:
            raise ValueError(f"{self.symbol}: volume cannot be negative")


def cse_market_closed():
    """True once the CSE session has ended, so the day's figures are final."""
    status = fetch(CSE_MARKET_STATUS_URL, method="POST", check_robots=True).json().get("status", "")
    return "closed" in str(status).lower()


def _traded_day(row):
    return datetime.fromtimestamp(row["lastTradedTime"] / 1000, tz=COLOMBO).date()


@dataclass(frozen=True)
class CseSnapshot:
    session: date | None                 # latest session with trades
    quotes: dict[str, PriceQuote]        # stocks that traded in that session
    listed: frozenset[str]               # every symbol CSE returned, traded or not


def parse_cse_trade_summary(payload):
    """Snapshot of the latest session from a ``tradeSummary`` response."""
    all_rows = [r for r in (payload.get("reqTradeSummery") or []) if r.get("symbol")]
    listed = frozenset(r["symbol"].strip().upper() for r in all_rows)
    rows = [r for r in all_rows if r.get("lastTradedTime") and (r.get("sharevolume") or 0) > 0]
    if not rows:
        return CseSnapshot(None, {}, listed)
    session = max(_traded_day(r) for r in rows)
    quotes = {}
    for row in rows:
        if _traded_day(row) != session:
            continue
        try:
            quote = PriceQuote(
                symbol=row["symbol"].strip().upper(), day=session,
                open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                close=float(row.get("closingPrice") or row["price"]),
                volume=int(row["sharevolume"]),
                turnover=float(row["turnover"]) if row.get("turnover") is not None else None,
                source=CSE_SOURCE, source_url=CSE_TRADE_SUMMARY_URL)
        except (KeyError, TypeError, ValueError):
            continue      # incomplete row: skip it rather than write a partial price
        quotes[quote.symbol] = quote
    return CseSnapshot(session, quotes, listed)


def collect_cse_quotes():
    """``CseSnapshot`` of the latest CSE session."""
    response = fetch(CSE_TRADE_SUMMARY_URL, method="POST", check_robots=True)
    try:
        payload = response.json()
    except ValueError as exc:
        raise CollectionError(f"CSE tradeSummary was not valid JSON: {exc}") from exc
    return parse_cse_trade_summary(payload)


GEMINI_PRICE_PROMPT = """Find the official end-of-day trading data for the most recent completed
trading session for each of these stock symbols: {symbols}.

Reply with ONLY a JSON array (no prose), one object per symbol you found:
[{{"symbol": "...", "date": "YYYY-MM-DD", "open": 0.0, "high": 0.0, "low": 0.0,
   "close": 0.0, "volume": 0, "turnover": null, "source_url": "page the numbers came from"}}]
Copy numbers exactly as published on the page. Use null for any value you did not find.
Never estimate or calculate a value. Leave out symbols you could not find."""


def parse_gemini_quotes(data, symbols, today, cited_urls=()):
    """Validated quotes from a Gemini reply; incomplete, future or stale entries are dropped."""
    wanted = {s.upper() for s in symbols}
    quotes = {}
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol") or "").strip().upper()
        try:
            day = date.fromisoformat(str(entry.get("date"))[:10])
            if (symbol not in wanted or day > today
                    or (today - day).days > MAX_GEMINI_QUOTE_AGE_DAYS):
                continue
            quote = PriceQuote(
                symbol=symbol, day=day, open=float(entry["open"]), high=float(entry["high"]),
                low=float(entry["low"]), close=float(entry["close"]),
                volume=int(entry["volume"]),
                turnover=float(entry["turnover"]) if entry.get("turnover") is not None else None,
                source=GEMINI_SOURCE,
                source_url=entry.get("source_url") or (cited_urls[0] if cited_urls else None))
        except (KeyError, TypeError, ValueError):
            continue      # a required value was missing or not a number
        quotes[symbol] = quote
    return quotes


def collect_gemini_quotes(symbols, settings, today=None):
    """Quotes for ``symbols`` found by Gemini + Google Search. Use only when CSE has none."""
    if not symbols:
        return {}
    today = today or datetime.now(COLOMBO).date()
    data, cited = grounded_json(GEMINI_PRICE_PROMPT.format(symbols=", ".join(sorted(symbols))),
                                settings)
    return parse_gemini_quotes(data, symbols, today, cited)
