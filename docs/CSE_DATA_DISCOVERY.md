# CSE public data discovery

**Question:** can ExitSafe get a long enough historical Colombo Stock Exchange
dataset from public, CSE-published sources, without paid CSE data products?

**Short answer:** not a multi-year one. The public CSE website provides:
- about one year of per-stock daily close / high / low / volume
- the current day's full trade summary
- only the most recent few daily, monthly and quarterly report PDFs

Multi-year daily history was not found anywhere public. Annual statistics
for every year except 2022 need a myCSE login. ExitSafe can proceed with a
one-year official window plus forward collection, but not with multi-year
official daily history.

Investigation date: 2026-09-30 (Sri Lanka), last run at 2026-09-30 09:30 UTC,
shortly after that day's close.
Evidence: `data/processed/audit/cse_source_evidence.json`. Raw files are in
`data/raw/cse/discovery/`. Both folders stay out of git, so re-create them with
the command below.
Reproduce with `python scripts/cse_source_discovery.py` (about 45 requests,
1 second apart; about 200 MB of PDFs on the first run, then cached).

## Method and rules

- Only `www.cse.lk` and `cdn.cse.lk` were accessed. `robots.txt` allows all
  paths except `/cgi-bin/`.
- **Endpoints were found by reading the CSE website's own JavaScript**, not by
  guessing. `www.cse.lk/api/...` is the interface the website itself calls from
  the browser. It is **undocumented**.
- **Every response code was recorded.** 401 and 417 answers were treated as
  "login required" and never retried with credentials. No account was created.
- **Dates come only from the source**, as a date printed in a report, a per-row
  trade date, or a per-point timestamp. A requested date was never assigned to
  a response.
- **Third-party material was used only as a lead.** The Hugging Face dataset and
  its GitHub code played no part here, and broker reports hosted on the CSE CDN
  are classed as broker content.
- **The CSE website terms of use were not reviewed.** See
  [What remains unverified](#8-what-remains-unverified).

## 1. Sources actually verified

**CONFIRMED** means the file or response was retrieved from a CSE-owned domain
and inspected.

| # | Source | Result | Access |
|---|---|---|---|
| 1 | New-format daily report PDFs ("DD-MM-YYYY Report"), listed by `api/cseDailyNew` | CONFIRMED, 6 of 6 listed reports retrieved | public |
| 2 | Old-format daily report PDFs ("smd full DD-MM-YYYY"), listed by `api/cseDaily` | CONFIRMED, 6 of 6 retrieved. This series stops at 2026-02-13 | public |
| 3 | Monthly (SMM) and quarterly (SMQ) report PDFs, listed by `api/cseMonthly` and `api/quarterlySummary` | CONFIRMED, 1 of each retrieved | public |
| 4 | Per-stock daily chart, `api/companyChartDataByStock` (period 5 = 1 year) | CONFIRMED for 7 symbols | public |
| 5 | Index charts, `api/chartData` (chartId 1 = ASPI, 40 = S&P SL20) | CONFIRMED | public |
| 6 | Current session, `api/tradeSummary`, `api/dailyMarketSummery`, `api/marketStatus` | CONFIRMED | public |
| 7 | Annual per-security statistics, `api/security_trading_statistics` | CONFIRMED for **2022 only**; 2021, 2023, 2024 and 2025 returned **HTTP 401** | 2022 public, others need login |
| 8 | Market reviews, `api/news/web?type=MR` | CONFIRMED to exist, but these are **broker-authored** PDFs, not CSE data | public |
| 9 | Historical reports for a chosen date (e.g. 2026-09-15, 2025-12-30, 2024-06-28) | **NOT FOUND**. No listing covers them, and URLs can't be derived from a date | – |
| 10 | Multi-year official daily price files | **NOT FOUND** publicly. `/publications/order-publications` exists but renders in the browser and was not inspected | unknown |
| 11 | `api/list_by_date` (annual-trading-statistics page) | HTTP 417 without login | login required |

No source here is UNCONFIRMED in the sense of "a third party claims it
exists". The earlier "CSE public data interface" claim in the Hugging Face
dataset card was not relied on.

## 2. URLs actually accessed

All URLs, status codes, byte counts and SHA-256 hashes are listed under
`requests` in the evidence file. Representative examples:

| URL | Method | HTTP |
|---|---|---|
| `https://www.cse.lk/robots.txt` | GET | 200 |
| `https://www.cse.lk/api/cseDailyNew` | GET | 200 |
| `https://www.cse.lk/api/cseDaily`, `cseWeekly`, `cseMonthly`, `quarterlySummary` | POST | 200 |
| `https://cdn.cse.lk/cmt/upload_report_file/N3IMwZcqNH6jBysL_29Sep2026110613GMT_1790679973321.pdf` | GET | 200 |
| `https://www.cse.lk/api/homeCompanyData` (form `symbol=JKH.N0000`) | POST | 200 |
| `https://www.cse.lk/api/companyChartDataByStock` (form `stockId=297, period=5`) | POST | 200 |
| `https://www.cse.lk/api/chartData` (form `chartId=1, period=5`) | POST | 200 |
| `https://www.cse.lk/api/tradeSummary`, `dailyMarketSummery`, `marketStatus` | POST | 200 |
| `https://www.cse.lk/api/security_trading_statistics` (form `year=2022`) | POST | 200 |
| same, `year=2021/2023/2024/2025` | POST | **401** |
| `https://www.cse.lk/api/list_by_date` (form `year=2025`) | POST | **417** |
| `https://www.cse.lk/api/news/web?top=false&year=2025&type=MR` | GET | 200 |

## 3. Files successfully retrieved

| File | Stated date / period (from the file itself) | Size | Pages |
|---|---|---|---|
| 6 new-format daily reports | 2026-09-22, 23, 24, 25, 28, 29 | ~27.5 MB each | 122–152 |
| 6 old-format daily reports | 2026-02-06, 09, 10, 11, 12, 13 | ~3.1 MB each | 85–116 |
| SMM August 2026 (monthly) | 08/2026 | 4.5 MB | 34 |
| SMQ 3 2025 (quarterly) | 2025:Q3 | 4.5 MB | 18 |
| 7 per-stock chart JSON responses | per-point timestamps | – | – |
| 2 index chart JSON responses | per-point timestamps | – | – |
| Annual statistics 2022 (JSON) | request year 2022; the payload has no date field | 74 KB | – |

**Date verification.** For all 12 daily reports, the date printed inside the
report equals the date in its listing title. The page headers of the parsed
section (new format) and the section heading (old format) also agree with it.
- New format: page 1 reads "Tuesday, 29 September, 2026", and every page has a
  `9/29/2026` header.
- Old format: page 1 shows `13-02-2026`, and the table heading reads "Daily
  Movements Equity on 13th February 2026".
- Two listing titles use underscores (`24_09_2026 Report`). The parser handles
  this, and the reports' own dates matched.

The 2022 annual statistics payload has **no date field**. Its year is only the
year we asked for, so it can't be attributed to 2022 by our own rule without
further confirmation.

## 4. Date coverage

| Source | Coverage observed | Frequency |
|---|---|---|
| Daily report PDFs | only the **latest 5–6 per series**: 2026-09-22 → 2026-09-29 (new format), 2026-02-06 → 2026-02-13 (old format) | daily |
| Per-stock chart | **2025-10-01 → 2026-09-30**, 241 points (236 for ACL). The window rolls forward; today's point appeared after the close | daily |
| Index charts | ~1 year, 240 points | daily |
| Monthly / quarterly reports | the latest 4 monthly and 5 quarterly reports listed | month / quarter |
| Annual statistics | 2022 only without login | year |
| tradeSummary | today's session only | – |

**No public source reaches back more than about one year at daily frequency.**

## 5. Columns

**New-format daily report, section "02. Daily Movements on Equity"** (291 rows per report):
- industry group, board, company name (**short name, e.g. "JKH", "COMMERCIAL BANK"**) and share type (N/X)
- close price, last traded price, date last traded
- high, low, turnover (Rs.)
- a column labelled "Foreign Holding". It has **negative values in 5 rows per
  report**, so its meaning is unverified.
- quantity in CDS (depository holdings, **not** shares traded)

Other sections of the same report include crossings (with base symbols),
end-of-day market capitalisation, and "Share Prices & Trends" (each traded price
level with its quantity and number of trades).

**Old-format daily report, "Daily Movements Equity"** (296–297 rows): company
name, closing price, last traded price, last traded date, high, low, foreign
holding, issued quantity, turnover, indexed market cap, quantity in CDS.

**Per-stock chart points:**
- **`t`:** timestamp, always exactly 00:00 Asia/Colombo, i.e. a trade date.
- **`h`, `l`:** high and low.
- **`p`:** close.
- **`q`:** share volume.
- **`s`:** meaning unknown.
- **`o`, `c`, `pc`, `n`:** always null.

**Index chart points:** `d` (timestamp), `v` (value), `pc` (% change). No OHLC.

**tradeSummary rows:**
- symbol, name
- open, high, low
- price, closingPrice, previousClose, change, percentageChange
- sharevolume, tradevolume, turnover, quantity
- crossingVolume, crossingTradeVol
- marketCap, marketCapPercentage
- lastTradedTime, issueDate, status, id, logoUrl

**Monthly report (SMM), "SECURITY TRADING STATISTICS"** (297 rows):
- security with full symbol parts (e.g. "KELANI TYRES N 0000")
- open, close, change, highest, lowest
- turnover, shares, trades
- 52-week highest and lowest

**Quarterly report (SMQ), "PRICE CHANGES IN THE QUARTER"** (274 rows): company
name plus 8 values. Their English labels are not in the extracted text, so the
column meanings are unverified.

**Annual statistics 2022:** symbol, name, open, high, low, close, turnover,
shareVolume, tradeVolume (323 securities).

### Mapping to the canonical schema

| Canonical | New-format daily report | Per-stock chart | tradeSummary |
|---|---|---|---|
| date | report date, only where "date last traded" equals it | `t` → date (SLT) | `lastTradedTime` → date (SLT) |
| symbol | **not available** (short name + type only) | the requested symbol | `symbol` |
| open | – | – (`o` always null) | `open` |
| high / low | High / Low | `h` / `l` | `high` / `low` |
| close | Close Price | `p` | `closingPrice` |
| volume | – ("Quantity in CDS" is holdings) | `q` | `sharevolume` |
| turnover | Turnover (Rs.) | – | `turnover` |
| trades | – | – | `tradevolume` |
| source | `cse_daily_report` | `cse_chart_api` | `cse_trade_summary_current` |
| source_priority | 1 | 2 | 2 |
| source_timestamp | – | `t` | `lastTradedTime` |

No missing value was filled in. Open and volume from the reports, and open,
turnover and trades from the chart, are left empty.

**Checking what the chart fields mean.** In 30 comparisons (5 symbols × 6 days),
where the report's short name equals the base ticker:
- the chart's `p`, `h` and `l` equalled the report's Close, High and Low **30/30**
- turnover ÷ `q` gave an average price inside the day's high–low range **30/30**

This is strong evidence that `p` is the official close and `q` is share volume.

## 6. Data-quality results (existing validator)

Our validator, `app/data/validators/market_validator.py`, was run on each source
as mapped above:

| | Daily report, latest (2026-09-29) | Daily reports, 6 days | Per-stock chart, 7 symbols | tradeSummary snapshot |
|---|---|---|---|---|
| Total rows | 277 | 1,680 | 1,682 | 282 |
| Valid / invalid | 0 / 277 | 0 / 1,680 | 0 / 1,682 | 0 / 282 |
| Stale rows excluded before validation | 14 | 66 | – | – |
| Symbol/date duplicates | 0 | 0 | 0 | 0 |
| Missing date / symbol | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Missing open / high / low / close | 277 / 0 / 0 / 0 | 1,680 / 0 / 0 / 0 | 1,682 / 0 / 1 / 0 | 0 / 0 / 0 / 0 |
| Missing volume / turnover | 277 / 0 | 1,680 / 0 | 0 / not provided | 0 / 0 |
| OHLC violations | low > close 10, high < close 11 | low > close 58, high < close 47 | low > close 1 | low > close 8, high < close 10 |
| Date range | 2026-09-29 | 2026-09-22 → 2026-09-29 | 2025-10-01 → 2026-09-30 | 2026-09-30 |
| Symbols | 277 (short names) | 291 (short names) | 7 | 282 |
| Status | FAIL | FAIL | FAIL | FAIL |

Why each one fails:
- **Missing open fails every report and chart row.** Open is a required
  canonical field and neither source provides it.
- **The official closing price can lie outside the day's high–low range.**
  Examples:
  - OFFICE EQUIPMENT on 2026-09-29: close 451.00, last trade 459.50,
    high = low = 459.50, turnover Rs 6,892.5.
  - AGST.X0000 on 2026-09-30: close 11.20, one trade of 10 shares at 11.90.

  Every such row observed was thinly traded:
  - the 105 report rows over 6 days had turnover of at most Rs 142,846
  - the 18 tradeSummary rows had at most 9 trades and Rs 86,700 turnover,
    against a median of Rs 466,188 across all securities

  In the reports, "Last Traded Price" was always inside the range. So the CSE "closing price" is not the last trade.
  Its method was not verified here. **The validator's rule that close lies
  within the day's high–low range conflicts with CSE's published closing
  price.** *(Since resolved: a close outside high–low is now the warning
  `CLOSE_OUTSIDE_HIGH_LOW`, not an error. See `DATA_SOURCES.md` section 4.
  The figures in this report were produced under the earlier rule.)*
- **tradeSummary fails every row on `DATE_MISMATCH`, by design.** The expected
  date came from `dailyMarketSummery`, which still reported **2026-09-29** while
  the market status was "Market Close (Statistics Being Finalized)" and every
  tradeSummary row was stamped **2026-09-30**. Two public endpoints disagreed
  about "today" at the same moment, which is exactly why a snapshot's date must
  be checked and never assumed.
- **Chart:** 1 row has no low (JKH on 2025-10-23). HDFC on 2026-08-28 shows the
  same close-outside-range effect (69 shares traded).

## 7. What is missing

- **Multi-year daily history.** Only ~1 year per stock is public. Nothing before
  2025-10-01 is available at daily frequency from a public CSE source.
- **Open price** in both the daily reports and the chart (only today's
  tradeSummary has it).
- **Historical daily turnover and trade count.** These appear in the daily
  reports (turnover only) for the ~5 most recent days, and in tradeSummary for
  today only.
- **CSE symbol codes in the daily reports.** The reports use short names, and
  "JKH" and "JOHN KEELLS" are different securities. No public name-to-symbol
  mapping for these short names was found.
- **Corporate-action adjustment status** of the chart series is unknown.
- **ASPI / S&P SL20 OHLC.** Only one value per day is available, and its
  timestamps are update times, not trade dates: the latest S&P SL20 point is
  stamped 09:30 SLT on the day.

## 8. What remains unverified

- **CSE terms of use** for automated access to `www.cse.lk/api` and
  `cdn.cse.lk`. Not reviewed. This must be checked, or permission sought from
  the CSE, before any scheduled collection.
- **Order Publications:** what `/publications/order-publications` sells, its
  price, and whether it includes multi-year daily data. The page renders in the
  browser and was not inspected.
- **myCSE:** whether a (free) myCSE account unlocks annual statistics for other
  years or other history.
- **Meaning of chart field `s`**, of the "Foreign Holding" column (it has
  negative values), and of the SMQ columns.
- **How the CSE computes its closing price.**
- **Whether older daily-report URLs still resolve** if their paths were known.
  They can't be derived, so this wasn't tested.
- **Whether the chart API offers periods longer than 1 year.** The website only
  uses periods 1–5, and undocumented values were not tried.

## 9. Recommended source priority

1. **Officially obtained CSE historical files** (purchase or data agreement),
   if the CSE provides them. This is the only route to multi-year official
   daily data. Not verified here.
2. **End-of-day `tradeSummary` snapshots**, collected going forward after the
   market has closed and its statistics are finalised. Each snapshot is
   accepted only if every row's `lastTradedTime` date equals the session date
   stated by the source. This has every canonical field, but only from now on.
3. **`companyChartDataByStock`** as a one-time ~1-year backfill of close, high,
   low and volume per stock. Open, turnover and trades stay empty. It needs
   about one request per listed security (~290), done politely and only after
   the terms-of-use check.
4. **Daily report PDFs** as an official cross-check of 1–3 for the dates they
   cover. They are also the source of daily turnover while they are listed.
5. **Monthly SMM tables** for monthly aggregates (research only).
6. **The secondary Hugging Face dataset** stays research-only and is never
   canonical.

## 10. Can ExitSafe realistically proceed?

**Yes, but only on a ~1-year official window, not multi-year history.**

- **What exists:** about one year of official daily close, high, low and volume
  for every listed stock is publicly retrievable, with verifiable dates, and it
  matched the official daily reports. Forward collection can add full daily
  bars (open, turnover, trades) from now on.
- **What doesn't:** multi-year official daily history is not publicly available.
  Tail-risk estimation on one year of data will be weak.
- **Liquidity analysis** can't use historical daily turnover from public
  sources. Volume × close would be an *estimate* and must be labelled as one.
- **Before building collection:**
  - confirm the CSE terms of use
  - decide whether to contact the CSE about historical data products
  - decide whether the canonical `close` is the official CSE closing price
    (which can fall outside high–low) or the last traded price, and adjust the
    validator rule to match

### Coverage decisions

| Source | Decision | Reason |
|---|---|---|
| Daily report PDFs | C. RESEARCH ONLY | Dates verified, but only ~5 recent reports are listed. No open, volume or symbol codes |
| Monthly / quarterly PDFs | C. RESEARCH ONLY | Complete per-security fields, but aggregated per month or quarter, and only the latest few are listed |
| Per-stock chart | B. USABLE WITH LIMITATIONS | ~1 year of daily bars with verified dates that match the official reports. No open, turnover or trades |
| tradeSummary snapshot | B. USABLE WITH LIMITATIONS | Every field, but today only. Forward collection only, with date checks |
| Annual statistics | C. RESEARCH ONLY | Public for 2022 only, annual aggregates, no date field in the payload |
| Index charts | C. RESEARCH ONLY | One value per day; timestamps are not reliable trade dates |
| Market reviews | D. NOT USABLE | Broker commentary, not CSE market data |

**No public CSE source qualifies as "A. SUITABLE AS CANONICAL SOURCE"** for
multi-year history.
