# Tracked Market Data

Upload your historical market data once. ExitSafe keeps tracking the companies in
it and adds each new trading day after the market closes, so the analysis always
runs on current data without uploading the file again.

This page explains how that works and how to run it after deployment.

## 1. What the user does

1. Log in, choose **Upload or paste data** and upload the CSV
   (e.g. `Date,Symbol,Open,High,Low,Close,Volume`).
2. Under the import summary, open **Keep this data up to date automatically**,
   pick what the data is for (*Start investing* or *I already invested*) and click
   **Start tracking**. The dashboard says *"Your market data is now being
   tracked."* and shows the companies, the last market date, the data source, the
   number of daily observations, the update status and the next expected update.
3. Later visits: choose **My tracked market data**. No upload needed. The panel
   shows how current the data is, **Update now**, **Pause tracking** /
   **Resume tracking**, and a **Data details** section (per-company status, days
   added since the upload and their source, recent updates, the original file).

Tracking is refused, with a plain message, when the file can't be read, has no
Symbol column and no symbol was entered, fails validation, has dates after
today, or has dates whose day/month order can't be told apart (new days couldn't
be written without guessing).

## 2. Where the data lives

Everything is in the PostgreSQL database that already holds client accounts.

| Table | Holds |
|---|---|
| `tracked_datasets` | One per uploaded file: owner, name, purpose, active flag, last attempted / successful update, status, message |
| `market_data_uploads` | The original file, byte for byte, with its SHA-256, row count, date range, validation summary and provenance. Never changed |
| `tracked_symbols` | The companies a dataset follows, the newest day in the upload, the newest day available, its source and the last update result |
| `market_observations` | Prices collected after each session, with the canonical fields (`date`, `symbol`, OHLC, `volume`, `turnover`, `estimated_traded_value`, `trades`, `change_pct`, `source`, `source_priority`, `source_timestamp`, `source_url`, `validation_status`, `validation_warnings`). One row per symbol, date and source, shared by every client tracking that symbol |
| `market_data_update_runs` | Every update: when, how it was started, session date, symbols processed, rows added / skipped, whether Gemini was used, errors and per-symbol detail |
| `tracked_dataset_updates` | Each dataset's own update history, shown in the dashboard |

Tracking configuration belongs to one client and is only ever read with that
client's id from the login session. Collected CSE prices are market facts, so
they are stored once and shared.

The tables are created by `python scripts/init_client_db.py`, and also by the
dashboard and the update job on start-up if they are missing. Creating them is
safe on a fresh or an existing database; existing tables are not changed.

## 3. How an update works

The scheduled job and **Update now** call the same function
(`run_tracking_update` in `app/intelligence/market_tracking.py`):

1. Only one update runs at a time (a PostgreSQL advisory lock, so this holds
   across processes and servers). A second one reports "already running".
2. Load every active symbol of every active dataset (paused datasets are left
   alone). For **Update now**, only the client's own datasets.
3. Ask the CSE whether the session has closed. While it is open nothing is
   collected and the status is *Waiting for market close*.
4. After the close, fetch the CSE trade summary once for all symbols. Each price
   is dated by its own last-trade time, never by the clock. A listed stock with
   no trades gets no row.
5. Symbols the CSE doesn't list go to Gemini web search, if a key is set and
   they don't already have the latest session.
6. Reject anything dated after today and anything the normal market-data
   validator marks INVALID.
7. Store each remaining price in its own transaction. A price already stored is
   skipped (unique symbol + date + source). A Gemini price is not stored for a
   day that already has a price.
8. Work out each dataset's result per company and save it, with the history, in
   one transaction per dataset.

Results: *Updated*, *Partially updated* (some companies failed), *Up to date*,
*No new data* (no trades), *Waiting for market close*, *Update failed*.

Running it again is harmless. If it stops half-way, what was stored stays
stored, the run is marked `INTERRUPTED` by the next run, and the next run
carries on.

## 4. Sources and provenance

- **Official CSE** (`cse_trade_summary_current`) is the primary source.
- **Gemini web search** (`gemini_web_search`) is only a fallback for symbols the
  CSE doesn't list. Its prices are stored as secondary AI-sourced data
  (priority 5), labelled as such everywhere, and **left out of the quantitative
  analysis by default**. If Gemini fails or has no quota, the CSE symbols are
  still updated and the others are reported as not updated; no price is made up.
- If both sources have the same day, the official price is used and the Gemini
  one is kept but listed as not used.
- **Missed days.** The CSE source only gives the latest session. If the update
  didn't run for a few days, those days stay empty and the update says how many
  weekdays have no price. Nothing is filled in or copied from the previous day.

## 5. What the analysis uses

The dashboard rebuilds the dataset from the original file plus the stored days
after the file's own last day, written in the file's own layout (columns,
delimiter, date format). That goes through the normal importer, validator and
`select_analysis_data`, exactly like an upload, so no formula changes and the
provenance selection still decides what reaches the analytics. Uploaded values
are never replaced.

**Backtesting** keeps using the original uploaded history only. The collected
days feed the current analysis (risk, optimization, stress testing, Exit Safety).

## 6. Running the scheduled update after deployment

The update must run outside the web app; the Streamlit process only serves pages.
Use either of these:

**A. A worker process (always on)**

```bash
python -m app.intelligence.bot --prices-only
```

It checks every `BOT_INTERVAL_MINUTES` (default 30) and updates once a day after
15:00 Sri Lanka time, retrying while the market is still open or the CSE can't be
reached. Without `--prices-only` it also scans news.

**B. A scheduled job (cron, a platform's cron job, a Windows scheduled task)**

```bash
python -m app.scheduler.update_market_data
```

One update, then exit. Suggested schedule: every 30 minutes on weekday
afternoons after the close, e.g. `*/30 9-12 * * 1-5` in UTC (14:30–18:00 Sri
Lanka time; a run before the close just waits). Exit code 0 means done (or still
waiting for the close); 1 means a source or the database failed and a later run
should retry.

Both need the same environment as the dashboard:

| Variable | Needed for |
|---|---|
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | The database (required) |
| `PRICE_SOURCES` | `cse,gemini` (default) or `cse` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Optional Gemini fallback |
| `BOT_INTERVAL_MINUTES` | How often the worker checks (option A) |

**Manual update.** The **Update now** button in the dashboard runs the same
update for the logged-in client's datasets.

**After a restart or downtime.** Nothing is lost: datasets, uploads and prices
are in the database. The next run picks up the latest session; days missed while
nothing was running stay empty, as described above. The dashboard shows when
the update service last ran and warns if it never has.
