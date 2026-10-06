# ExitSafe analytics

Quantitative calculations on canonical market data (see `DATA_SOURCES.md`).
All functions are plain functions: they take a DataFrame, do not modify it,
and return new results at full precision. Rounding belongs to presentation.

| # | Module | Status |
|---|---|---|
| 1 | Returns (`app/analytics/returns.py`) | implemented |
| 2 | Volatility (`app/analytics/volatility.py`) | implemented |
| 3 | Covariance / correlation (`app/analytics/covariance.py`) | implemented |
| 4–13 | Drawdown, Sharpe/Sortino, VaR, CVaR, liquidity, portfolio risk, optimization, regime, stress testing, backtesting | not implemented |

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

## 3. Covariance and correlation

Covariance shows **how two stocks' returns move together**. Correlation shows
**the strength and direction** of that relationship on a scale from −1 to +1:

| Correlation | Meaning |
|---|---|
| **+1** | returns move together strongly |
| **0** | little or no *linear* relationship |
| **−1** | returns move in opposite directions strongly |

**Why it matters later:** portfolio risk depends not only on each stock's
volatility but on how the stocks move together. Covariance is the input for
portfolio risk and optimization (later modules).

```python
from app.analytics.covariance import (
    calculate_return_matrix,                 # date x symbol, NaN where a symbol has no return
    calculate_aligned_return_matrix,         # common observations only
    describe_return_alignment,               # how much data was used, and why dates were left out
    calculate_covariance_matrix,             # daily
    calculate_annualized_covariance_matrix,  # daily x periods_per_year (default 252)
    calculate_correlation_matrix,
)
```

All matrices are square DataFrames with sorted symbols as rows and columns.
Returns come from `calculate_daily_returns`; the return formula is not
duplicated.

**Formulas**

```
daily covariance      Cov(X, Y) = Σ (xᵢ − x̄)(yᵢ − ȳ) / (n − 1)     (sample, ddof = 1)
annualized covariance           = daily covariance × 252
correlation (Pearson) Corr(X, Y) = Cov(X, Y) / (sd(X) × sd(Y))
```

- **The diagonal of the covariance matrix** is each stock's return variance.
- **Annualization differs from volatility.** Covariance scales by **252**, not
  √252. Volatility, a standard deviation, scales by √252 because variance
  scales by 252: annualized variance = (annualized volatility)².
- **Correlation is not annualized:** it has no time unit.
- **The 252 periods per year** can be overridden with `periods_per_year`; it
  must be a positive number.

**Missing data and alignment**

- **Common observations:** every matrix element uses the **same set of common
  observations**. A date counts only if **every** symbol has a usable daily
  return on it **and** all those returns start from the same previous date, so
  they cover the same period.
  - **Example:** if XYZ has no row on Wednesday, its Thursday return covers two
    days. That Thursday is left out rather than paired with ABC's one-day
    return.
- **INVALID rows** give no returns, and returns are never bridged across them
  (see section 1), so the dates next to an INVALID row drop out.
- **Nothing is filled in or estimated.**
- **`describe_return_alignment` reports:**
  - the symbols
  - the number of common observations
  - the date range used
  - usable returns per symbol
  - how many dates were left out because a symbol had no return
    (`excluded_missing`)
  - how many were left out because returns covered different periods
    (`excluded_misaligned`)
- **A symbol with little history limits the whole matrix,** since all symbols
  share the same dates. Check `returns_per_symbol` to see which symbol it is.
- **Volatility can differ slightly:** volatility (section 2) uses all of a
  symbol's returns, while the covariance diagonal uses only the common dates.

**Insufficient data**

- **Fewer than 2 common observations:** every matrix value is **NaN**, never 0.
- **A constant return series** (range ≤ 1e-12, which also catches
  floating-point noise such as a steady 10% growth path) has covariance 0,
  which is correct. Its correlations are **NaN**, including its diagonal,
  because correlation is undefined without variation. Other symbols keep a
  diagonal of 1.
- **Correlations are clipped to [−1, 1]** to remove floating-point overshoot.
- **Empty input** gives empty matrices.
- **Missing `date`/`symbol`/`close`** raises `ValueError`.
- **Duplicate symbol/date rows without validation** raise `ValueError`. Validated
  data marks them INVALID instead.

**Worked example** (two stocks, three daily returns):

| | Returns | Mean | Deviations |
|---|---|---|---|
| X | 1%, 2%, 3% | 2% | −0.01, 0, +0.01 |
| Y | 3%, 1%, 2% | 2% | +0.01, −0.01, 0 |

| Result | Calculation | Value |
|---|---|---|
| Cov(X, Y) | (−0.0001 + 0 + 0) / 2 | **−0.00005** |
| Var(X) = Var(Y) | (0.0001 + 0 + 0.0001) / 2 | **0.0001** |
| Corr(X, Y) | −0.00005 / √(0.0001 × 0.0001) | **−0.5** |
| Annualized Cov(X, Y) | −0.00005 × 252 | **−0.0126** |
