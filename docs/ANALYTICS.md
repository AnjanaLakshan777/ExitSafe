# ExitSafe analytics

Quantitative calculations on canonical market data (see `DATA_SOURCES.md`).
All functions are plain functions: they take a DataFrame, do not modify it,
and return new results at full precision. Rounding belongs to presentation.

| # | Module | Status |
|---|---|---|
| 1 | Returns (`app/analytics/returns.py`) | implemented |
| 2 | Volatility (`app/analytics/volatility.py`) | implemented |
| 3–13 | Covariance/correlation, drawdown, Sharpe/Sortino, VaR, CVaR, liquidity, portfolio risk, optimization, regime, stress testing, backtesting | not implemented |

## 1. Daily returns

`calculate_daily_returns(data)`

**Formula:** `daily_return = close / previous close − 1`, a simple (not log)
return, computed separately for each symbol.

- **Input:** canonical columns `date`, `symbol`, `close`, plus optional
  `validation_status`. The result is sorted by symbol and date and gains a
  `daily_return` column.
- **"Previous"** means the symbol's previous record. The first record of each
  symbol has no return (NaN). Missing calendar days are not filled in.
- **INVALID rows** produce no returns: a return is kept only when **both** of its
  rows are usable (VALID or WARNING). An INVALID row is never dropped and
  bridged over, which would turn two trading days into one return. Without a
  `validation_status` column, the data is assumed to be validated already.
- **Phase 1 layout:** the older `Date`/`Symbol`/`Close` layout from
  `app/data/loaders/market_data.py` is still accepted and gets a
  `Daily Return` column.

## 2. Volatility

Volatility measures **how much returns vary**. Higher volatility means larger
historical price fluctuations. It does **not** by itself mean an investment
will lose money; it describes past variability, not direction.

```python
from app.analytics.volatility import calculate_daily_volatility, calculate_annualized_volatility

calculate_daily_volatility(data)                           # symbol, observations, daily_volatility
calculate_annualized_volatility(data)                      # ... + annualized_volatility, periods_per_year
calculate_annualized_volatility(data, periods_per_year=260)
```

**Formulas**

```
daily_volatility      = sample standard deviation of daily returns (ddof = 1)
annualized_volatility = daily_volatility × √252
```

- **Returns** come from `calculate_daily_returns`; the return formula is not
  duplicated.
- **Assumption:** 252 trading periods per year (`TRADING_DAYS_PER_YEAR`).
  Calendar gaps (holidays, missing days) are not adjusted for.
- **Output:** one row per symbol, sorted by symbol, with:
  - `observations`: the number of usable returns
  - `daily_volatility`
  - `annualized_volatility` and `periods_per_year` (annualized function only)
- **Usable returns:** only finite returns that don't touch an INVALID row.
- **Fewer than 2 usable returns:** volatility is **NaN**, never 0, and
  `observations` shows why.
- **Constant prices** give exactly 0.
- **Empty input** gives an empty result with the same columns. A missing
  `date`, `symbol` or `close` column raises `ValueError`.

**Worked example** (closes 100, 102, 101, 103):

| Step | Value |
|---|---|
| Daily returns | 102/100 − 1 = 0.02; 101/102 − 1 = −0.0098039216; 103/101 − 1 = 0.0198019802 |
| Mean | 0.0099993529 |
| Sample variance (÷ n−1 = 2) | 0.0002941371 |
| Daily volatility | √0.0002941371 = **0.0171504245** |
| Annualized volatility | 0.0171504245 × √252 = **0.2722545493** (≈ 27.2%) |
