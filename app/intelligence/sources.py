"""News sources the market-threat bot reads. Edit these lists to add or remove sources.

Only sources that publish a feed, or whose robots.txt allows crawling, belong
here; the HTML collector checks robots.txt again before every page request.
"""

from dataclasses import dataclass

from app.data.schemas.event_schema import SourceType


@dataclass(frozen=True)
class FeedSource:
    name: str
    url: str
    source_type: SourceType = SourceType.NEWS


@dataclass(frozen=True)
class PageSource:
    """A listing page scraped for headline links.

    ``link_pattern`` is a regex the article URL must match. If it has named
    groups ``year``, ``month`` and ``day``, they give the publish date; links
    without a date are skipped, because an undated headline can't be placed in time.
    """

    name: str
    url: str
    link_pattern: str
    source_type: SourceType = SourceType.NEWS


RSS_FEEDS = (
    FeedSource("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    FeedSource("CNBC Markets", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    FeedSource("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    FeedSource("Financial Times Markets", "https://www.ft.com/markets?format=rss"),
    FeedSource("Investing.com Stock Market News", "https://www.investing.com/rss/news_25.rss"),
    FeedSource("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    FeedSource("The Guardian Business", "https://www.theguardian.com/business/rss"),
    FeedSource("New York Times Business",
               "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
)

_DATED_PATH = r"/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/"

HTML_PAGES = (
    PageSource("LankaBusinessOnline", "https://www.lankabusinessonline.com/",
               r"^https://www\.lankabusinessonline\.com" + _DATED_PATH),
    PageSource("IMF News", "https://www.imf.org/en/News",
               r"^https://www\.imf\.org/en/news/articles" + _DATED_PATH,
               source_type=SourceType.REGULATOR),
)
