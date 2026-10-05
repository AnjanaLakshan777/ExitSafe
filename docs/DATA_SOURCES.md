# ExitSafe data sources

This document describes where ExitSafe's data comes from, how far each source
is trusted, and the rules every imported dataset must pass. The machine-readable
version of the source list is `app/data/source_catalog.py`; the current state of
the local data is produced by `python scripts/data_source_audit.py`.

Status as of this writing: **no multi-year official CSE daily history has been
obtained.** Public CSE sources were investigated in
[CSE_DATA_DISCOVERY.md](CSE_DATA_DISCOVERY.md). They provide recent daily report
PDFs, about one year of per-stock chart data, and a current-session snapshot, but
no login-free multi-year daily history. The only longer real dataset available
locally is a secondary research dataset (see
[Secondary datasets](#3-secondary-source-policy)).

---

## 1. Data-source categories

| Category | `DataSourceType` | Examples in the catalog | Status |
|---|---|---|---|
| Official CSE | `OFFICIAL_CSE` | daily report PDFs, per-stock chart data, current trade summary, historical daily trade data, company directory, corporate actions, announcements (all `cse.lk`) | Report PDFs, chart data and trade summary verified (see CSE_DATA_DISCOVERY.md); no multi-year history obtained |
| Secondary datasets | `SECONDARY_DATASET` | `tharu-jwd/cse-market-data` and `kjhq/Sri-Lanka-Stock-Symbols-and-Metadata` on Hugging Face | Downloaded, audited |
| Regulators | `REGULATOR` | SEC Sri Lanka (`sec.gov.lk`), Central Bank of Sri Lanka (`cbsl.gov.lk`) | Not yet investigated |
| Cybersecurity | `CYBERSECURITY` | Sri Lanka CERT public advisories (`cert.gov.lk`) | Not yet investigated |
| News | `NEWS` | public financial news; specific outlets not yet selected | Not yet investigated |
| Other | `OTHER` | `exitsafe_sample` (synthetic test data) | In repo |

Each catalog entry records: `source_name`, `source_type`, `domain`,
`description`, `historical_or_current`, `expected_data_type`, `trust_level`,
`license_or_usage_note`, `enabled`, `historical_backfill_allowed`, `notes`,
`location` and a `column_map` (source column to canonical column).

Trust levels map to the canonical `source_priority` (lower = preferred):
`HIGH` 1, `MEDIUM` 2, `LOW` 3, `UNVERIFIED` 4, `SYNTHETIC` 9.

A source is labelled `OFFICIAL_CSE` only if the Colombo Stock Exchange itself
publishes it. A dataset that says it was *derived from* CSE data is secondary.

## 2. Canonical market-data schema

Defined in `app/data/schemas/market_schema.py`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `date` | datetime64 | yes | Trading date, as stated by the source |
| `symbol` | string | yes | Security symbol as published (e.g. `JKH.N0000`) |
| `open`, `high`, `low`, `close` | float64 | yes | Prices (LKR) |
| `volume` | float64 | yes | Shares traded |
| `change_pct` | float64 | no | Change in percent **as reported by the source** (`-0.88` = −0.88%). Informational only |
| `turnover` | float64 | no | Value traded (LKR), **as reported** |
| `estimated_traded_value` | float64 | no | **Derived:** `close × volume`, only for rows with no reported turnover. Never called turnover |
| `trades` | float64 | no | Number of trades |
| `source` | string | yes | Catalog `source_name` |
| `source_priority` | int64 | yes | From the catalog trust level |
| `source_timestamp` | datetime64 (UTC) | no | Timestamp the source itself attaches to the record |
| `validation_status` | string | yes | `VALID`, `WARNING` or `INVALID` |
| `validation_warnings` | string | yes | `;`-separated issue codes |

Rules:

- A value the source does not provide stays **null**. For example,
  `source_timestamp` is never set to our download time, and trade counts are
  never invented.
- **Derived values are kept apart from reported ones.**
  - `turnover` only ever holds the source's own figure.
  - When it is absent, `estimated_traded_value = close × volume` is filled
    instead, and the provenance record lists it under `derived_fields`.
  - It is an estimate: VWAP × volume would differ, and abbreviated volumes like
    `7.94M` are rounded.
- Invalid rows are **kept and flagged**, not dropped or repaired, so the reason
  a row can't be used stays visible. Downstream code filters on
  `validation_status`.
- Import pipelines:
  - **Catalogued files with a known layout:** `load_raw_market_file` (exact raw
    values + provenance), then `to_canonical_dataset` (rename columns, validate,
    build canonical frame).
  - **Provider CSVs in any supported layout:** `load_csv_market_data`. See
    [section 2a](#2a-generic-csv-market-data-import).

The Phase 1 loader (`app/data/loaders/market_data.py`, used by
`calculate_daily_returns`) is unchanged. It still uses title-case columns and
drops invalid rows. Moving analytics onto the canonical schema is a later step.

## 2a. Generic CSV market-data import

`app/data/loaders/csv_market_loader.py` imports historical price CSVs from
different providers **without editing the file**. Example of a supported layout:

```
Date,Price,Open,High,Low,Vol.,Change %
12/31/2025,660.09,664.75,665,659.44,7.94M,-0.88%
12/30/2025,665.95,658.69,672.22,657.84,9.19M,1.10%
```

```python
from app.data.loaders.csv_market_loader import load_csv_market_data

result = load_csv_market_data("prices.csv", symbol="JKH.N0000")
result.data        # canonical frame (None if required columns are missing)
result.report      # import report: columns, mapping, counts, warnings, provenance
result.provenance  # source, file hash, load time, derived fields
```

**Column detection.** Header matching ignores case and whitespace, so
`Change %` = `change%` and `Vol.` = `vol.`.

| Canonical | Accepted headers (first listed wins if several are present) |
|---|---|
| `date` | Date |
| `symbol` | Symbol |
| `open` | Open, Opening Price |
| `high` | High |
| `low` | Low |
| `close` | **Close**, Closing Price, **Price** |
| `volume` | Volume, Share Volume, **Vol.**, Vol |
| `change_pct` | Change %, change_pct, Change |
| `turnover` | Turnover, Turnover (Rs.), Value Traded |
| `trades` | Trades, No. of Trades |

- **Price → close.** If both `Close` and `Price` are present, `Close` is used
  and `Price` is listed in the report's `ignored_columns`. Any row where the two
  differ is marked **INVALID** (`CLOSE_SOURCE_CONFLICT`); ExitSafe never picks
  one of two conflicting values silently. The same rule applies to any other
  field with two matching headers.
- **Two columns that could be the date or the symbol** stop the import with an
  error.
- **An explicit `column_map` in the source catalog** takes precedence over these
  aliases.
- **Unrecognised columns** are listed as ignored.

**Numbers.**
- `.` is always the decimal point. `,` is only accepted as a thousands
  separator in groups of three: `250,000` and `1,234,567.5` are read, while
  `1.234,5` and `1,23` are rejected as unreadable instead of being misread.
- **Volume and turnover** may use `K`, `M` or `B` (any case):
  `560K` → 560,000, `7.94M` → 7,940,000, `1.2B` → 1,200,000,000. The
  conversion uses exact decimal arithmetic. The report notes that abbreviated
  values are only as precise as the digits shown. Prices never accept
  suffixes.

**Change %.** Optional.
- `-0.88%` is stored as `change_pct = -0.88`.
- A value without a `%` sign is only accepted when the header says it holds
  percentages (contains `%` or `pct`). A bare `Change` column of plain numbers
  could be an absolute price change, so it is ignored and reported.
- **Analytical returns are always recalculated from closes**
  (`close / previous close − 1`, per symbol). `change_pct` never replaces them.
  The importer only compares the two: a gap above 0.05 percentage points marks
  the row with the warning `CHANGE_PCT_MISMATCH`.

**Symbol.** In order:
1. a `Symbol` column
2. the `symbol=` parameter (must agree with the column if both are given)
3. the file name, only with `infer_symbol_from_filename=True` and only if the
   name is exactly a symbol (e.g. `JKH.N0000.csv`; `JKH Historical Data.csv` is
   refused)

Otherwise the import stops with `MarketDataImportError`. The company is never
guessed.

**Dates.** `2025-12-31` is read as-is. For `D/M/YYYY`-style dates:
- If any first part is > 12, the whole file is day-first (`31/12/2025`).
- If any second part is > 12, it is month-first (`12/31/2025`).
- Without such evidence, a date like `01/02/2025` is **rejected as
  `AMBIGUOUS_DATE`** (row INVALID) rather than guessed. `05/05/2025` is safe
  either way and is accepted.
- `date_format="%d/%m/%Y"` (any strptime format) can be passed to resolve
  ambiguity.
- Impossible or unreadable dates are `INVALID_DATE`.

**Validation.** The shared canonical validator (section 4) is used. The
importer only adds its own row codes: `AMBIGUOUS_DATE` and `*_SOURCE_CONFLICT`
are errors; `CHANGE_PCT_MISMATCH` and `CHANGE_PCT_UNREADABLE` are warnings.

**Source.** Uploads default to the catalog entry `user_csv_upload` (trust
`UNVERIFIED`, never official, never used for backfill). Pass `source_name=` when
the file comes from a catalogued provider.

**Not supported yet:**
- delimiters other than `,`
- text dates such as `Dec 31, 2025`
- two-digit years
- files without open/high/low/volume columns: these are reported as `FAIL`
  (missing required columns), because the canonical schema requires them

## 3. Secondary-source policy

Secondary datasets are **research and comparison sources, never canonical**:

1. They are catalogued as `SECONDARY_DATASET` with trust `LOW` and
   `historical_backfill_allowed = False`.
2. They are downloaded at a **pinned revision** by
   `scripts/fetch_secondary_datasets.py`. The script writes a `SOURCE.json` with
   the URL, revision, licence, retrieval time and SHA-256 of each file.
3. They may enter the canonical history only after being cross-checked against
   official CSE data for the same symbols and dates. That check is not built yet.
4. Licence terms are followed. `tharu-jwd/cse-market-data` is CC BY 4.0, so any
   published use must credit: Jayawardana K.P.T., Wickramaratne W.P.G.A.,
   Wijesinghe D.S., Fernando W.T.D., Dinapura H.H.S., *Market Reactions to Natural
   Disasters: An Event Study of Cyclone Ditwah's Impact on the Colombo Stock
   Exchange*, University of Moratuwa (2026).

### What is known about `tharu-jwd/cse-market-data` (revision `669594502b85`)

Everything below is checked against the files and the published code.

- The dataset card says the data was collected "via a custom Python web-scraping
  pipeline querying the CSE's public data interface".
- The collection script it links to (GitHub `dehanf/Disaster-Shock-Market-Response`,
  `data/pipeline/collect.py`) actually downloads bars from **TradingView's feed**
  (exchange code `CSELK`). It uses the unofficial `tvDatafeed` library, which can
  log in with a TradingView account. That script calls the CSVs a "reference
  dataset" that its output can replace.
- That script writes zero volume as an empty value. The published
  `stock_prices.csv` has explicit zeros and no empty volumes, so it was **not**
  produced by the script exactly as published. The file's precise origin is
  undocumented.
- 3,746 rows have fractional share volumes. Shares trade in whole units, so this
  suggests the data was adjusted for splits or bonus issues. **Whether and how
  prices are adjusted is undocumented.**
- `sector_mapping.csv` has columns `symbol, sector`. The dataset card documents
  them as `ticker, sector`. The file uses short tickers (`JKH`), while prices use
  full CSE symbols (`JKH.N0000`). It has 20 sectors; the card says 19.
- `aspi.csv` has no volume at all (every value blank).

Audit results for this dataset are in [section 7](#7-known-limitations).

## 4. Validation rules

Implemented in `app/data/validators/market_validator.py`. The validator never
modifies its input.

**Required columns:** `date, symbol, open, high, low, close, volume`.
**Optional:** `turnover, trades` (validated when present) and `change_pct`
(informational).
**Importer checks:** importers can add row checks through `extra_errors` and
`extra_warnings`, so all rules still produce one validation result. If a required column is missing, the result is
`FAIL` with `missing_columns`. The validator reports this instead of raising.

**Row errors** (the row is `INVALID`):

| Code | Rule |
|---|---|
| `MISSING_DATE`, `INVALID_DATE` | date is empty or not a valid ISO date. No guessing of other formats, and impossible dates like 2026-02-30 are rejected |
| `MISSING_SYMBOL` | symbol is empty |
| `MISSING_<COL>`, `INVALID_<COL>` | required number is empty or can't be parsed. Thousands separators are accepted |
| `NON_POSITIVE_<PRICE>` | open, high, low or close is 0 or less |
| `LOW_ABOVE_OPEN` | low > open |
| `HIGH_BELOW_OPEN`, `HIGH_BELOW_LOW` | high < open / high < low |
| `NEGATIVE_VOLUME`, `NEGATIVE_TURNOVER`, `NEGATIVE_TRADES` | value < 0 |
| `DUPLICATE_SYMBOL_DATE` | several rows share a symbol and date. **All** copies are flagged; none is picked automatically |
| `DATE_MISMATCH` | the row's date differs from the as-of date the source states for the file (see section 8) |

Values treated as missing: blank, `NA`, `N/A`, `NaN`, `null`, `None`, `-`
(case-insensitive).

**Row warnings** (the row is `WARNING` but still usable):
- `CLOSE_OUTSIDE_HIGH_LOW`: close < low or close > high. This is only a warning
  because the CSE's official closing price can fall outside the day's traded
  range for thinly traded securities (see `CSE_DATA_DISCOVERY.md`), so
  `low ≤ close ≤ high` is not a universal rule.
- `ZERO_VOLUME_WITH_PRICE_RANGE`: no shares traded but high > low.
- `NON_INTEGER_VOLUME`.
- `WEEKEND_DATE`.

**Dataset status:**
- `FAIL` if any of these is true:
  - a required column is missing
  - the file has no rows
  - any row has `DATE_MISMATCH`
  - more than 5% of rows are invalid
- `WARNING` if any row is invalid or any warning was raised.
- `PASS` otherwise.

## 5. Provenance rules

Every imported file gets a `Provenance` record
(`app/data/schemas/provenance.py`):

| Field | Meaning |
|---|---|
| `source_name`, `source_type` | Catalog entry. Loading an uncatalogued source is refused |
| `original_file_name` | File name as obtained |
| `file_sha256`, `file_size_bytes` | Content fingerprint |
| `loaded_time` | When ExitSafe read the file (UTC) |
| `retrieval_time` | When the file was downloaded, if known (from `SOURCE.json`) |
| `source_date` | As-of date **stated by the source**. Null for multi-date history files |
| `source_url`, `source_version` | Where it came from, and its revision if known |

The audit script writes the provenance and the validation summary for each file
to `data/processed/audit/<source>__<file>.manifest.json`. It also reports whether
the file still matches the checksum recorded at download time.

Storage layout:

```
data/raw/cse/{market,company,corporate_actions,announcements}/   official CSE files, as obtained
data/raw/external/                                                secondary datasets (+ SOURCE.json)
data/raw/intelligence/                                            collected event items
data/processed/cse/, data/processed/intelligence/[historical/]   derived outputs
data/processed/audit/                                             audit reports and manifests
data/sample/                                                      small synthetic fixtures (in git)
```

Everything under `data/raw/` and `data/processed/` stays out of git; only the
folder structure is tracked. To reproduce the local data:

```
python scripts/fetch_secondary_datasets.py
python scripts/data_source_audit.py
```

## 6. Event-data design

The event schema lives in `app/data/schemas/event_schema.py` (`MarketEvent`),
because events are stored and replayed for backtesting. The intelligence
package produces events but doesn't define them.

Fields: `event_id`, `symbol`, `company_name`, `event_type`, `title`,
`description`, `source_name`, `source_type`, `source_url`, `event_time`,
`published_time`, `detected_time`, `severity`, `confidence`,
`verification_status`. There are also two optional fields, `affected_sector` and
`potential_market_impact`.

- **Event types:** `CYBERSECURITY_INCIDENT`, `DATA_BREACH`, `REGULATORY_ACTION`,
  `FINANCIAL_RESULT`, `MANAGEMENT_CHANGE`, `FRAUD_OR_GOVERNANCE`, `LEGAL_EVENT`,
  `CREDIT_EVENT`, `MACROECONOMIC_EVENT`, `SECTOR_EVENT`, `LIQUIDITY_EVENT`,
  `OTHER`.
- **Timestamps:** all must include a time zone, and `detected_time` can't be
  earlier than `published_time`. `event_time` may be in the future (e.g. a
  scheduled results date).
- **Confidence** is a number from 0 to 1. **Severity** is `LOW`, `MEDIUM`,
  `HIGH` or `CRITICAL`, or null if not yet assessed.
- **Verification status** is `UNVERIFIED`, `REPORTED`, `CORROBORATED`,
  `CONFIRMED`, `DENIED` or `RETRACTED`. `CONFIRMED` is only allowed for official
  source types (exchange disclosure, regulator, company). A news report or a
  threat-intelligence claim can never be stored as confirmed.
- Events are only collected from lawful public or licensed sources. No
  private-system, restricted-forum or leaked-data access. No event scraping
  exists yet.

## 7. Known limitations

Findings from the audit of the locally available data:

| | ExitSafe sample | HF `cse-market-data` |
|---|---|---|
| Nature | synthetic | secondary, research |
| Rows / symbols | 75 / 3 | 71,044 / 289 |
| Dates | 2026-01-02 to 2026-02-05 (25 days) | 2025-02-03 to 2026-02-27 (257 days) |
| Turnover / trades | turnover yes, trades no | neither |
| Missing values, duplicates, OHLC violations | none | none |
| Row warnings | 5 zero-volume rows with a price range | 67 zero-volume rows with a price range; 3,746 fractional volumes |
| ASPI | no | yes, 257 days, aligned with stock dates, no volume |
| Sector mapping | no | 286 / 289 symbols via base ticker |

Other limitations:

- **No official multi-year CSE history.** Every official entry in the catalog is
  `enabled = False`. What public CSE sources do and do not provide is recorded in
  `docs/CSE_DATA_DISCOVERY.md`.
- **About one year of history** in the secondary dataset. That is short for
  tail-risk estimates.
- **No turnover (value traded)** in the secondary dataset. The liquidity
  analysis in later phases needs turnover, which will have to come from official
  CSE data.
- **Gaps.** Only 142 of 289 symbols have a row on every trading day. Days with
  no trading usually have no row, rather than a zero-volume row.
- **Corporate-action adjustment is unknown**, and no corporate-action data has
  been obtained yet to check it.
- **Symbols are a single snapshot.** Delistings, symbol changes and sector
  reclassifications over time are not captured (survivorship bias).
- **The synthetic sample has zero-volume days with a price range.** That is
  unrealistic; it's kept because Phase 1 tests depend on the file.

## 8. Sources that must not be used for historical backfilling (unless date validation succeeds)

| Source | Why |
|---|---|
| `cse_trade_summary_current` (undocumented CSE web interface) | It returns the **current** session. A response fetched while "asking for" a past date is still today's data. It may be stored only if the payload carries its own trade date. That date is passed as `source_date`, and every row must match it (`DATE_MISMATCH` fails the import otherwise). A requested date is never written onto a snapshot. |
| Any other "latest/current" snapshot | Same rule. |
| `hf_tharu_jwd_cse_market_data` | Secondary, with undocumented origin and adjustment. Research only until cross-checked. |
| `hf_kjhq_sri_lanka_stock_symbols` | Reference snapshot with undocumented upstream. It is not a history of listings or sectors. |
| `exitsafe_sample` | Synthetic. |

The rule in code: `historical_backfill_allowed` is `True` only for
`cse_historical_official`, and a unit test ensures no `CURRENT` source ever
allows it.
