# ExitSafe analytics

Quantitative calculations on canonical market data (see `DATA_SOURCES.md`).
All functions are plain functions: they take a DataFrame, do not modify it,
and return new results at full precision. Rounding belongs to presentation.

| # | Module | Status |
|---|---|---|
| 1 | Returns (`app/analytics/returns.py`) | implemented |
| 2 | Volatility (`app/analytics/volatility.py`) | implemented |
| 3 | Covariance / correlation (`app/analytics/covariance.py`) | implemented |
| 4 | Maximum drawdown (`app/analytics/drawdown.py`) | implemented |
| 5 | Sharpe / Sortino ratios (`app/analytics/ratios.py`) | implemented |
| 6 | Value at Risk, stock-level 1-day (`app/analytics/var.py`) | implemented |
| 7 | CVaR / Expected Shortfall, stock-level 1-day (`app/analytics/cvar.py`) | implemented |
| 8–13 | Liquidity, portfolio risk, optimization, regime, stress testing, backtesting | not implemented |

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
- **Rows without a date** are treated like INVALID rows, so they get no return
  even in unvalidated data.
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

## 4. Maximum drawdown

**Maximum drawdown** shows the largest historical fall from a previous peak to
a later low. ExitSafe uses it to measure how severe a past loss could have been
during a decline. It describes history; it is not a prediction.

```python
from app.analytics.drawdown import calculate_drawdown_series, calculate_maximum_drawdown

calculate_drawdown_series(data)    # date, symbol, close, running_peak, drawdown
calculate_maximum_drawdown(data)   # one row per symbol: the worst event
```

**Formulas** (per symbol, in date order, from closing prices)

```
running_peak(t)  = max(close_1 ... close_t)
drawdown(t)      = close_t / running_peak(t) − 1        0 at a peak, negative below it
maximum drawdown = min over t of drawdown(t)            ≤ 0
```

- **Loss measure:** the result is ≤ 0. A worst fall of 25% is stored as
  `-0.25`, never `+0.25`; the UI shows it as "-25.00%".

**The maximum drawdown event** (`calculate_maximum_drawdown` columns)

| Column | Meaning |
|---|---|
| `observations` | number of usable prices |
| `maximum_drawdown` | the worst drawdown |
| `trough_date`, `trough_price` | when the worst drawdown happened (the earliest one if tied) |
| `peak_date`, `peak_price` | where that decline started: the **last** date at or before the trough on which the close equalled the running peak |
| `recovery_date` | first date **after** the trough on which the close is back at or above the peak price; **empty (NaT) if it hasn't recovered** by the end of the data |

- **Chronological order:** peak, trough and recovery always follow the actual
  price path. The overall maximum and minimum prices are never paired
  independently.
- **Example:** in 100 → 80 → 150 the worst fall is 100 → 80 (−20%), not
  150 → 80.

**Worked example**

| Close | 100 | 110 | 120 | 108 | 90 | 105 |
|---|---|---|---|---|---|---|
| Running peak | 100 | 110 | 120 | 120 | 120 | 120 |
| Drawdown | 0% | 0% | 0% | −10% | **−25%** | −12.5% |

- **Maximum drawdown** = (90 / 120) − 1 = **−25%**: peak 120, trough 90, not
  recovered.
- **With recovery:** 100 → 120 → 96 → 110 → 125 gives −20%, peak 120, trough
  96, recovered on the day of 125.

**Invalid data, missing dates and edge cases**

- **INVALID rows** are left out, so they are never a peak or a trough. WARNING
  rows are used. Rows without a date or symbol, or without a finite positive
  close, are also left out.
- **Nothing is filled in.** A missing day is not a drawdown event, and no price
  is interpolated. Each drawdown compares a usable close with the highest
  usable close before it.
- **Fewer than 2 usable prices:** `maximum_drawdown` is **NaN**, not 0.
- **No decline at all** (constant or always rising prices): `maximum_drawdown`
  is 0 and the peak, trough and recovery fields are empty, because there is no
  drawdown event.
- **Duplicate symbol/date rows:**
  - in validated data, all copies are INVALID and so left out
  - in unvalidated data, they raise `ValueError`
- **Empty input** gives empty tables with the usual columns.
- **Missing columns:** a missing `date`/`symbol`/`close` raises `ValueError`.

## 5. Sharpe and Sortino ratios

- **Sharpe ratio:** return earned relative to **total** risk.
- **Sortino ratio:** return earned relative to **downside** risk.

Higher positive values generally indicate better risk-adjusted performance
**under the chosen assumptions**. They describe the past, depend heavily on the
risk-free rate and the period covered, and are not buy or sell signals.

```python
from app.analytics.ratios import (
    calculate_risk_adjusted_ratios,   # everything below, one row per symbol
    calculate_sharpe_ratio,           # symbol, observations, risk_free_rate, excess return, volatility, sharpe
    calculate_sortino_ratio,          # symbol, observations, risk_free_rate, excess return, downside dev., sortino
    daily_risk_free_rate,
)

calculate_risk_adjusted_ratios(data, risk_free_rate=0.05, periods_per_year=252)
```

**Risk-free rate.** An explicit input: an annual rate as a decimal
(`0.05` = 5%).
- **The default `0.0` is only a calculation default.** Real analysis should
  supply a rate that fits the investor and period, e.g. a relevant
  government-securities yield; ExitSafe does not hard-code one.
- **Negative and zero rates are allowed.** The rate must be finite and above
  −100%.

**Formulas** (P = periods per year, default 252; r = daily simple return from
`calculate_daily_returns`)

```
daily risk-free rate      rf_d = (1 + rf)^(1/P) − 1          compounds back to rf over P periods
daily excess return       e    = r − rf_d
annualized excess return       = mean(e) × P                 arithmetic
annualized volatility          = std(r, ddof=1) × √P         the existing volatility module
Sharpe                         = annualized excess return / annualized volatility

downside deviation             = √( mean( min(e, 0)² ) ) × √P
Sortino                        = annualized excess return / downside deviation
```

**Downside-deviation convention (exact):**
- **Target:** the daily risk-free rate.
- **Which days count:** only days with a negative excess return count. They
  are squared; every other day counts as **0**.
- **The mean is over all n observations,** not just the negative ones.
- **Annualization:** the square root of that mean is annualized with √P.
- **Not the Sharpe denominator:** Sortino never reuses Sharpe's total
  volatility.

**Two different "annual returns":**
- **`annualized_excess_return`** is **arithmetic** (mean × P). It is the
  numerator of both ratios.
- **`annualized_return`** is **geometric**: (∏(1 + r))^(P / n) − 1, the
  compounded growth rate. It is shown for information only.
- **With few observations it extrapolates a long way:** 5 days of returns
  compounded to a year can show several hundred percent.

**Undefined results are NaN, never 0 or infinity:**

| Case | Result |
|---|---|
| fewer than 2 usable returns | every computed field is NaN |
| no variation in returns (daily volatility ≤ 1e-12, which also catches floating-point noise) | Sharpe is NaN |
| no excess return below the risk-free target (downside deviation ≤ 1e-12) | Sortino is NaN |

**Data handling** is the same as returns and volatility:
- INVALID rows produce no returns and are never bridged.
- WARNING rows are used.
- The first return of each symbol doesn't count.
- Each symbol is independent.
- The input is not modified.

**Assumptions:**
- Returns are independent from day to day, as in √P scaling.
- Calendar gaps are not adjusted for.
- Periods per year must be a positive finite number.

**Worked example:** daily returns +2%, −1%, +3%, −2%, +1%, P = 252.

| | rf = 0 | rf = 5% |
|---|---|---|
| daily risk-free rate | 0 | 1.05^(1/252) − 1 = 0.0001936305 |
| annualized excess return | 0.006 × 252 = 1.512 | (0.006 − 0.0001936305) × 252 = 1.463205 |
| annualized volatility | √(0.00172 / 4) × √252 = 0.329181 | 0.329181 (unchanged) |
| Sharpe | 1.512 / 0.329181 = **4.5932** | **4.4450** |
| downside deviation | √((0.01² + 0.02²) / 5) × √252 = 0.158745 | 0.160591 |
| Sortino | 1.512 / 0.158745 = **9.5247** | **9.1114** |

## 6. Value at Risk (VaR)

Stock-level, **1-day** VaR, per symbol. Two methods are implemented: historical
and parametric (Normal). Portfolio VaR, Monte Carlo VaR and CVaR are not
implemented yet.

**What it means.** At confidence C, the 1-day VaR is the loss threshold that,
**under the chosen method and data**, is expected to be exceeded on only about
(1 − C) of days.
- **Example:** a 95% 1-day VaR of 3% means "based on the selected method and
  historical data/model, the estimated 1-day loss threshold is about 3% at 95%
  confidence".
- **It is not the maximum possible loss.** It also says nothing about how large
  a loss is once the threshold is exceeded; that is what **CVaR** (next module)
  measures.

**Sign convention:** VaR is a **positive loss magnitude**. `0.032` means a 3.2%
one-day loss threshold, never `-0.032`. If even the lower tail of the sample is
a gain (e.g. only positive returns), VaR comes out negative. It is reported as
calculated, not floored at 0.

```python
from app.analytics.var import (
    calculate_var_summary,      # symbol, observations, min_observations, sufficient_data,
                                # confidence_level, historical_var, parametric_var
    calculate_historical_var,
    calculate_parametric_var,
)

calculate_var_summary(data, confidence_level=0.95, min_observations=20)
```

**Formulas** (C = confidence level, α = 1 − C, r = daily returns of one symbol)

```
historical VaR = −quantile(r, α)
parametric VaR = −(mean(r) + z_α × std(r))     z_α = scipy.stats.norm.ppf(α), negative for α < 0.5
                                               std with ddof = 1 (as in volatility)
```

- **Historical quantile method:** `numpy.quantile(method="linear")`
  (Hyndman & Fan type 7), passed explicitly and never left to a library
  default.
  - Sort the returns x₀ … xₙ₋₁ and set h = (n − 1)α.
  - The quantile is x⌊h⌋ + (h − ⌊h⌋)(x⌊h⌋₊₁ − x⌊h⌋).
- **z is computed from the normal distribution,** not hard-coded, so any
  confidence works (e.g. 93.7%).
- **Confidence** must be a finite number strictly between 0 and 1. 0.95 and
  0.99 are typical, but any value in that range is accepted.

**Assumptions and comparison**

| | Historical VaR | Parametric VaR |
|---|---|---|
| Distribution | the observed returns, as they are | **assumes** returns are approximately normal (real returns often have fatter tails) |
| Depends on | the sample; tail values come from few observations | only the mean and standard deviation of the sample |
| Constant returns | the constant (negated) | −mean, since σ = 0 |

The two methods use **exactly the same returns** for a symbol and can give
different results.

**Sample size.**
- **Minimum:** a symbol needs at least `min_observations` usable daily returns
  (**default 20**, configurable, at least 2). Otherwise both VaRs are **NaN**,
  never 0, and `observations` shows how many were available.
- **20 returns is still a small sample:** a 99% VaR from 20 returns depends on
  the one or two worst days.

**Data handling:**
- Returns come from `calculate_daily_returns`.
- INVALID rows give no returns and are never bridged.
- Rows without a date give no returns.
- WARNING rows count.
- The first return of each symbol and non-finite returns are excluded.
- Each symbol is independent, and the input is not modified.

**Limitations:**
- 1-day horizon only.
- Single stocks only: no diversification or portfolio effects.
- Prices aren't adjusted for dividends or splits.
- No check of how well the VaR would have predicted past losses (no
  backtesting).
- The parametric normal assumption can understate tail risk.

**VaR vs CVaR.** VaR is a threshold: "losses are expected to exceed X on about
5% of days". CVaR (expected shortfall) is the **average loss on those worst
days**, so it shows how bad the tail is beyond the threshold.

**Worked example** (20 synthetic daily returns: 0.02, −0.01, 0.03, −0.02, 0.01,
−0.05, 0.015, −0.01, 0.005, −0.03, 0.012, −0.008, 0.025, −0.015, 0.004,
−0.022, 0.018, −0.006, 0.009, −0.04):

| | 95% (α = 0.05) | 99% (α = 0.01) |
|---|---|---|
| Two lowest returns | −0.05, −0.04 | −0.05, −0.04 |
| h = 19α | 0.95 | 0.19 |
| Lower-tail quantile | −0.05 + 0.95 × 0.01 = −0.0405 | −0.05 + 0.19 × 0.01 = −0.0481 |
| **Historical VaR** | **4.05%** | **4.81%** |
| Mean / sample std | −0.00315 / 0.0218253 | same |
| z_α | −1.6448536 | −2.3263479 |
| **Parametric VaR** = −(−0.00315 + z × 0.0218253) | **3.9049%** | **5.3923%** |

## 7. CVaR / Expected Shortfall

**Conditional Value at Risk (CVaR)**, also called **Expected Shortfall (ES)**,
estimates the **average loss in the tail beyond the VaR confidence threshold**.

| | Answers |
|---|---|
| **VaR** | the **tail threshold**: where the worst (1 − C) of outcomes begin |
| **CVaR** | the **average severity beyond that threshold**: how bad those worst outcomes are on average |

**Why it matters for tail risk.** Two stocks can share the same VaR while one
has much heavier losses on its worst days. CVaR shows that difference.
- **It is not the maximum possible loss.**
- **It describes the data or model,** not a prediction.

```python
from app.analytics.cvar import (
    calculate_cvar_summary,      # symbol, observations, min_observations, sufficient_data,
                                 # confidence_level, tail_mass, historical_var, historical_cvar,
                                 # parametric_var, parametric_cvar
    calculate_historical_cvar,
    calculate_parametric_cvar,
)

calculate_cvar_summary(data, confidence_level=0.95, min_observations=20)
```

**Sign convention:** the same as VaR, a **positive loss magnitude**.
- **Example:** VaR = 0.032 and CVaR = 0.047 mean a 3.2% loss threshold and a
  4.7% average loss beyond it.
- **No clamping:** if the tail of the sample contains gains, CVaR is smaller,
  or even negative. It is reported as calculated.

**Historical CVaR: empirical expected shortfall with fractional tail weighting**

Each of the n daily returns carries probability 1/n. CVaR averages exactly the
worst α = 1 − C of that probability mass:

```
1. losses lᵢ = −rᵢ, sorted from largest to smallest:  d₁ ≥ d₂ ≥ … ≥ dₙ
2. tail mass  m = α × n          (rounded to 9 decimals to remove the floating-point
                                  noise of 1 − C, e.g. 1.0000000000000009 → 1)
3. k = ⌊m⌋, f = m − k
4. CVaR = (d₁ + … + d_k + f × d_{k+1}) / m
```

| Observations | Confidence | Tail mass m | Historical CVaR |
|---|---|---|---|
| 20 | 95% | 1 | the worst loss |
| 40 | 95% | 2 | the average of the worst two |
| 30 | 95% | 1.5 | (worst + 0.5 × second worst) / 1.5, **not** the plain average of the worst two |
| 20 | 99% | 0.2 | the worst loss (only part of one observation is in the tail) |

- **Not a plain cutoff mean:** this is not "the mean of the returns below
  VaR". That simple cutoff ignores the fractional boundary and depends on the
  quantile convention.
- **Consistent with VaR:** historical VaR uses the linear (type-7) quantile at
  position (n − 1)α, which never lies deeper in the tail than the mass
  boundary α·n. So **historical CVaR ≥ historical VaR** always holds.

**Parametric CVaR (Normal)** uses the same mean, standard deviation and
confidence as parametric VaR:

```
μ = mean(r),  σ = std(r, ddof = 1),  α = 1 − C,  z_α = Φ⁻¹(α)     (scipy.stats.norm.ppf)
lower-tail conditional mean   ES_return = μ − σ × φ(z_α) / α     (φ = scipy.stats.norm.pdf)
parametric CVaR               = −ES_return = −(μ − σ × φ(z_α) / α)
```

- **Always at least VaR:** under the normal model, φ(z_α)/α > −z_α, so
  parametric CVaR ≥ parametric VaR.
- **Zero variation:** with σ = 0 both equal −μ.
- **Assumption:** this inherits the normal assumption. Real return tails are
  often heavier, so the parametric figure can understate tail losses.

**Inputs, sample size and data:**
- **Confidence:** the same as VaR (shared validation), a finite number strictly
  between 0 and 1.
- **Minimum observations:** shared with VaR, **default 20**. Below it, both
  CVaRs are **NaN**, never 0. `observations` and `tail_mass` show how much
  data was used.
- **Same returns as VaR:** both modules read the sample from one helper
  (`returns_by_symbol`).
- **Data handling:**
  - INVALID rows give no returns and are never bridged.
  - Rows without a date give no returns.
  - WARNING rows count.
  - The first return of each symbol and non-finite returns are excluded.
  - Each symbol is independent, and the input is not modified.

**Limitations:**
- **Very few tail observations:** with 20–30 returns, the 95% tail is only 1–1.5
  observations, and at 99% it is a fraction of one. Historical CVaR is then
  essentially the single worst day.
- **Scope:** 1-day and stock-level only; no portfolio CVaR, Monte Carlo or
  longer horizons.
- **Unadjusted prices:** prices aren't adjusted for dividends or splits.

**Worked example:** 30 synthetic returns (the 20 from the VaR example, then
0.007, −0.012, 0.011, −0.003, 0.016, −0.019, 0.002, −0.027, 0.013, −0.035),
95% confidence.

| Step | Value |
|---|---|
| α, tail mass | 0.05, 30 × 0.05 = **1.5** |
| Worst losses (weights) | 0.05 (1.0), 0.04 (0.5) |
| **Historical CVaR** | (0.05 + 0.5 × 0.04) / 1.5 = 0.07 / 1.5 = **4.6667%** (vs plain average of worst two: 4.5%) |
| Historical VaR (for comparison) | **3.775%** |
| Mean, sample std | −0.0036667, 0.0202677 |
| z₀.₀₅, φ(z₀.₀₅) | −1.6448536, 0.1031356 |
| **Parametric CVaR** | −(−0.0036667 − 0.0202677 × 0.1031356 / 0.05) = **4.5473%** |
| Parametric VaR (for comparison) | **3.7004%** |
