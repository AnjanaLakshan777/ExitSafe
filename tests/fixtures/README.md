# Test fixtures

All files here are **synthetic test data**. They are not real market data, and
results computed from them are not evidence about any market or strategy.

- `synthetic_date_price_vol_change.csv`: a short single-stock file in a
  provider-style layout (Date, Price, Vol., Change %) for importer tests.
- `synthetic_backtest_data.csv`: four invented stocks (ALPHA, BRAVO, CHARLIE,
  DELTA) over 321 weekdays from 2025-01-01, generated reproducibly by
  `scripts/generate_synthetic_backtest_data.py` for walk-forward backtests.
  The stocks are driven by the synthetic ASPI series in
  `data/sample/sample_index_data.csv`, with differing returns, volatility,
  correlation and liquidity (CHARLIE has no turnover; DELTA has zero-volume
  days).
