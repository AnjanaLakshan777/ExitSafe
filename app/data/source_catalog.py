"""Catalog of the external data sources ExitSafe uses or plans to investigate.

Each entry records what a source is, how far it can be trusted and whether it
may be used to fill ExitSafe's canonical historical store. Descriptions only
state what has been verified; anything not yet checked is marked as such.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class DataSourceType(StrEnum):
    OFFICIAL_CSE = "OFFICIAL_CSE"            # published by the Colombo Stock Exchange itself
    SECONDARY_DATASET = "SECONDARY_DATASET"  # third-party compilation (research datasets, vendors)
    NEWS = "NEWS"
    REGULATOR = "REGULATOR"
    CYBERSECURITY = "CYBERSECURITY"
    OTHER = "OTHER"


class TrustLevel(StrEnum):
    HIGH = "HIGH"                # primary source, documented dates
    MEDIUM = "MEDIUM"            # primary content, but interface or dating needs care
    LOW = "LOW"                  # secondary / unofficial; research and cross-checks only
    UNVERIFIED = "UNVERIFIED"    # not yet assessed
    SYNTHETIC = "SYNTHETIC"      # generated test data, never real prices


# Lower number = preferred when sources disagree (canonical ``source_priority``).
SOURCE_PRIORITY = {
    TrustLevel.HIGH: 1,
    TrustLevel.MEDIUM: 2,
    TrustLevel.LOW: 3,
    TrustLevel.UNVERIFIED: 4,
    TrustLevel.SYNTHETIC: 9,
}


class Coverage(StrEnum):
    HISTORICAL = "HISTORICAL"
    CURRENT = "CURRENT"                  # snapshot of the current/latest session only
    HISTORICAL_AND_CURRENT = "HISTORICAL_AND_CURRENT"
    REFERENCE = "REFERENCE"              # static-ish metadata (companies, sectors)
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class DataSource:
    source_name: str
    source_type: DataSourceType
    domain: str | None
    description: str
    historical_or_current: Coverage
    expected_data_type: str
    trust_level: TrustLevel
    license_or_usage_note: str
    enabled: bool
    # May this source's rows enter the canonical historical market store?
    historical_backfill_allowed: bool
    notes: str = ""
    location: str | None = None          # URL or identifier of the dataset/page
    # Source column name -> canonical column name (see app.data.schemas.market_schema).
    column_map: dict[str, str] = field(default_factory=dict)

    @property
    def source_priority(self):
        return SOURCE_PRIORITY[self.trust_level]


SOURCE_CATALOG = (
    DataSource(
        source_name="cse_historical_official",
        source_type=DataSourceType.OFFICIAL_CSE,
        domain="cse.lk",
        description="Historical daily trade data published by the Colombo Stock Exchange.",
        historical_or_current=Coverage.HISTORICAL,
        expected_data_type="Daily OHLCV / trade summary per security",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Subject to CSE terms of use; confirm redistribution rights before sharing.",
        enabled=False,
        historical_backfill_allowed=True,
        notes=("NOT OBTAINED. Discovery (docs/CSE_DATA_DISCOVERY.md) found no public, "
               "login-free multi-year daily history on cse.lk: annual per-security statistics "
               "need a login except for 2022, and historical data is offered through CSE's "
               "'Order Publications' page. Place any officially obtained files in "
               "data/raw/cse/market/ and add a column_map once the real layout is known."),
    ),
    DataSource(
        source_name="cse_trade_summary_current",
        source_type=DataSourceType.OFFICIAL_CSE,
        domain="cse.lk",
        description="Current-session trade summary shown on the CSE website.",
        historical_or_current=Coverage.CURRENT,
        expected_data_type="Snapshot of latest prices, volume and turnover per security",
        trust_level=TrustLevel.MEDIUM,
        license_or_usage_note="Subject to CSE terms of use. Interface is undocumented and may change.",
        enabled=False,
        historical_backfill_allowed=False,
        location="https://www.cse.lk/api/tradeSummary",
        notes=("Undocumented interface behind the CSE web pages. Reflects the CURRENT session "
               "only: it must never be stored under a requested historical date. A snapshot may "
               "be kept only if its payload carries its own trade date, and that date is what "
               "gets stored (validate with expected_date). Verified fields: symbol, open, high, "
               "low, price, closingPrice (0.0 during the session), sharevolume, turnover, "
               "tradevolume, marketCap, lastTradedTime (epoch ms)."),
    ),
    DataSource(
        source_name="cse_daily_report",
        source_type=DataSourceType.OFFICIAL_CSE,
        domain="cse.lk",
        description="CSE daily market report PDFs ('DD-MM-YYYY Report'; older 'smd full') on cdn.cse.lk.",
        historical_or_current=Coverage.HISTORICAL_AND_CURRENT,
        expected_data_type="Per-security close, last traded price, high, low, turnover (no open, "
                           "no volume, no symbol codes)",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Subject to CSE terms of use.",
        enabled=False,
        historical_backfill_allowed=False,
        location="https://www.cse.lk/api/cseDailyNew",
        notes=("Report date is printed in the PDF itself. Only the latest ~5 reports are listed "
               "and URLs contain a random token, so older reports cannot be retrieved. Rows use "
               "short company names, not CSE symbols. Best used as an official cross-check."),
    ),
    DataSource(
        source_name="cse_chart_api",
        source_type=DataSourceType.OFFICIAL_CSE,
        domain="cse.lk",
        description="Per-stock daily chart data used by the CSE website (companyChartDataByStock).",
        historical_or_current=Coverage.HISTORICAL,
        expected_data_type="~1 year of daily high, low, close (p) and share volume (q) per stock",
        trust_level=TrustLevel.MEDIUM,
        license_or_usage_note="Subject to CSE terms of use. Interface is undocumented and may change.",
        enabled=False,
        historical_backfill_allowed=False,
        location="https://www.cse.lk/api/companyChartDataByStock",
        notes=("Each point carries its own timestamp (00:00 Asia/Colombo). Open is always null; "
               "turnover and trade count are absent; rolling 1-year window only. Close/high/low "
               "matched the official daily report where cross-checked."),
    ),
    DataSource(
        source_name="cse_company_directory",
        source_type=DataSourceType.OFFICIAL_CSE,
        domain="cse.lk",
        description="Listed-company profiles and sector classification on the CSE website.",
        historical_or_current=Coverage.REFERENCE,
        expected_data_type="Company name, symbol, sector",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Subject to CSE terms of use.",
        enabled=False,
        historical_backfill_allowed=False,
        notes="Not yet obtained. Sector changes over time are not captured by a single snapshot.",
    ),
    DataSource(
        source_name="cse_corporate_actions",
        source_type=DataSourceType.OFFICIAL_CSE,
        domain="cse.lk",
        description="Corporate actions (dividends, splits, rights issues, etc.) published by the CSE.",
        historical_or_current=Coverage.HISTORICAL_AND_CURRENT,
        expected_data_type="Action type, symbol, announcement / ex / record dates, ratios",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Subject to CSE terms of use.",
        enabled=False,
        historical_backfill_allowed=False,
        notes="Not yet obtained. Needed later to judge whether price series are adjusted.",
    ),
    DataSource(
        source_name="cse_announcements",
        source_type=DataSourceType.OFFICIAL_CSE,
        domain="cse.lk",
        description="Company announcements and disclosures published through the CSE.",
        historical_or_current=Coverage.HISTORICAL_AND_CURRENT,
        expected_data_type="Announcement title, company, publication time, document link",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Subject to CSE terms of use.",
        enabled=False,
        historical_backfill_allowed=False,
        notes="Not yet obtained. Feeds the event model (official disclosure source type).",
    ),
    DataSource(
        source_name="sec_sri_lanka",
        source_type=DataSourceType.REGULATOR,
        domain="sec.gov.lk",
        description="Securities and Exchange Commission of Sri Lanka publications.",
        historical_or_current=Coverage.HISTORICAL_AND_CURRENT,
        expected_data_type="Regulatory notices and enforcement actions",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Public regulator publications; check site terms before bulk use.",
        enabled=False,
        historical_backfill_allowed=False,
        notes="Not yet investigated.",
    ),
    DataSource(
        source_name="cbsl",
        source_type=DataSourceType.REGULATOR,
        domain="cbsl.gov.lk",
        description="Central Bank of Sri Lanka publications.",
        historical_or_current=Coverage.HISTORICAL_AND_CURRENT,
        expected_data_type="Policy rates, macroeconomic releases, banking directives",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Public regulator publications; check site terms before bulk use.",
        enabled=False,
        historical_backfill_allowed=False,
        notes="Not yet investigated.",
    ),
    DataSource(
        source_name="sri_lanka_cert",
        source_type=DataSourceType.CYBERSECURITY,
        domain="cert.gov.lk",
        description="Sri Lanka CERT public security advisories and alerts.",
        historical_or_current=Coverage.HISTORICAL_AND_CURRENT,
        expected_data_type="Public cybersecurity advisories",
        trust_level=TrustLevel.HIGH,
        license_or_usage_note="Public advisories only. No private, restricted or leaked data.",
        enabled=False,
        historical_backfill_allowed=False,
        notes="Not yet investigated.",
    ),
    DataSource(
        source_name="public_financial_news",
        source_type=DataSourceType.NEWS,
        domain=None,
        description="Public financial news (specific outlets not yet selected).",
        historical_or_current=Coverage.HISTORICAL_AND_CURRENT,
        expected_data_type="Articles: headline, publication time, body, URL",
        trust_level=TrustLevel.UNVERIFIED,
        license_or_usage_note="Per-outlet terms, RSS/API licences and robots.txt apply.",
        enabled=False,
        historical_backfill_allowed=False,
        notes="News is a reported claim, never a confirmed fact on its own.",
    ),
    DataSource(
        source_name="hf_tharu_jwd_cse_market_data",
        source_type=DataSourceType.SECONDARY_DATASET,
        domain="huggingface.co",
        description=("'CSE Market Data - Cyclone Ditwah Event Study' (University of Moratuwa "
                     "research dataset): daily OHLCV for CSE stocks, ASPI and a sector mapping."),
        historical_or_current=Coverage.HISTORICAL,
        expected_data_type="Daily OHLCV per stock (no turnover, no trade count)",
        trust_level=TrustLevel.LOW,
        license_or_usage_note="CC BY 4.0 - attribution to the dataset authors required.",
        enabled=True,
        historical_backfill_allowed=False,
        location="https://huggingface.co/datasets/tharu-jwd/cse-market-data",
        notes=("NOT an official CSE source. The dataset card says data came from 'the CSE's public "
               "data interface', but the collection script published with it (GitHub "
               "dehanf/Disaster-Shock-Market-Response, data/pipeline/collect.py) downloads bars "
               "from TradingView's feed (exchange CSELK) via the unofficial tvDatafeed library, "
               "and describes the CSVs as a 'reference dataset' it can replace. That script writes "
               "zero volume as blank, yet stock_prices.csv has explicit zeros and no blanks, so the "
               "exact origin of the published file is undocumented. Fractional share volumes "
               "suggest corporate-action adjustment, which is also undocumented. "
               "Research/comparison use only until cross-checked with official CSE data."),
        column_map={},  # stock_prices.csv already uses canonical column names
    ),
    DataSource(
        source_name="hf_kjhq_sri_lanka_stock_symbols",
        source_type=DataSourceType.SECONDARY_DATASET,
        domain="huggingface.co",
        description="Sri Lanka stock symbols and company metadata (name, ticker, market, sector).",
        historical_or_current=Coverage.REFERENCE,
        expected_data_type="Company name, ticker, market, sector",
        trust_level=TrustLevel.LOW,
        license_or_usage_note="CC0 1.0 (public domain dedication).",
        enabled=True,
        historical_backfill_allowed=False,
        location="https://huggingface.co/datasets/kjhq/Sri-Lanka-Stock-Symbols-and-Metadata",
        notes="Upstream source and sector taxonomy are not documented by the publisher.",
    ),
    DataSource(
        source_name="user_csv_upload",
        source_type=DataSourceType.OTHER,
        domain=None,
        description="Historical market-data CSV supplied by a user (provider not verified).",
        historical_or_current=Coverage.HISTORICAL,
        expected_data_type="Daily prices per security in any supported CSV layout "
                           "(see app/data/loaders/csv_market_loader.py)",
        trust_level=TrustLevel.UNVERIFIED,
        license_or_usage_note="The user is responsible for the provider's terms of use.",
        enabled=True,
        historical_backfill_allowed=False,
        notes=("Default source for generic CSV imports. The origin of the file is not known to "
               "ExitSafe, so it is never treated as official. When the provider is known and "
               "catalogued, import with that source_name instead."),
    ),
    DataSource(
        source_name="exitsafe_sample",
        source_type=DataSourceType.OTHER,
        domain=None,
        description="Synthetic development dataset (symbols ABC, XYZ, LMN).",
        historical_or_current=Coverage.NOT_APPLICABLE,
        expected_data_type="Daily OHLCV + value traded",
        trust_level=TrustLevel.SYNTHETIC,
        license_or_usage_note="Project-internal test fixture.",
        enabled=True,
        historical_backfill_allowed=False,
        location="data/sample/sample_market_data.csv",
        notes="Randomly generated. Not real CSE stocks or prices.",
        column_map={"Date": "date", "Symbol": "symbol", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                    "Value Traded": "turnover"},
    ),
)

_BY_NAME = {source.source_name: source for source in SOURCE_CATALOG}


def get_source(source_name):
    try:
        return _BY_NAME[source_name]
    except KeyError:
        raise KeyError(
            f"Unknown data source {source_name!r}. Register it in app/data/source_catalog.py "
            f"first. Known sources: {', '.join(_BY_NAME)}"
        ) from None
