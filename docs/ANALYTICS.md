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
| 8 | Liquidity, stock-level (`app/analytics/liquidity.py`) | implemented |
| 9 | Portfolio risk, fixed user weights (`app/analytics/portfolio_risk.py`) | implemented |
| 10 | Risk-aware portfolio optimization (`app/portfolio/optimizer.py`) | implemented |
| 11–13 | Market regime, stress testing, backtesting | not implemented |

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

## 8. Liquidity (stock-level)

Liquidity describes **how easily an investor can buy or sell a position without
needing an unusually large share of the market's normal trading activity**.
This module exposes raw liquidity measures for the later Exit Safety Engine. It
is not that engine, and it makes no buy or sell recommendation.

```python
from app.analytics.liquidity import calculate_liquidity_summary, calculate_position_liquidity

calculate_liquidity_summary(data)                                   # one row per symbol
calculate_position_liquidity(data, position_value=20_000_000, participation_rate=0.10)
```

**Inputs.**
- **Required:** canonical `date`, `symbol`, `close`, `volume`.
- **Optional:** `turnover`, `validation_status`.
- **Nothing is invented,** filled in or interpolated.

**Usable rows.**
- **Excluded:**
  - INVALID rows
  - rows without a date or symbol
  - rows without a finite positive close or a finite non-negative volume
- **WARNING rows count.**
- **Missing trading days don't appear,** and `observations` is the real number
  of usable rows.
- **Duplicate symbol/date rows:** in validated data all copies are INVALID; in
  unvalidated data they raise `ValueError`.

**Traded value: actual vs estimated, never mixed**

| `traded_value_source` | Used when | Daily traded value |
|---|---|---|
| `ACTUAL_TURNOVER` | **every** usable row of the symbol reports a finite, non-negative `turnover` | reported turnover |
| `ESTIMATED_TRADED_VALUE` | otherwise | **close × volume** for every usable row |

- **The estimate is not official turnover.** `estimated_traded_value` ≠
  turnover, and reported turnover is never overwritten.
- **Partial turnover:** if only some days report turnover, the whole symbol
  uses the estimate rather than mixing the two sources.
  `actual_turnover_days` shows how many days had it.

**Stock-level metrics** (`calculate_liquidity_summary`)

| Metric | Formula |
|---|---|
| Average Daily Volume (ADV) | mean(volume) |
| Median / min / max daily volume | of the usable rows |
| Average Daily Traded Value (ADTV) | mean(daily traded value from the chosen source) |
| Median daily traded value | median of the same |
| `zero_volume_days` | number of usable rows with volume = 0 |
| `zero_volume_rate` | zero_volume_days / observations |

**Zero-volume days** are **valid observations**, not errors.
- They count as 0 in ADV and ADTV.
- They are counted in the zero-volume statistics, because days with no trades
  are evidence of illiquidity.
- They are never silently dropped. A row is excluded only if it fails the
  usual validation rules.

**Position liquidity** (`calculate_position_liquidity`; the same position value
is assessed for each symbol separately, so this is not a portfolio):

```
position_to_adtv            = position_value / ADTV
daily_executable_value      = ADTV × participation_rate
estimated_liquidation_days  = position_value / daily_executable_value
```

- **`participation_rate`:** default **0.10**, must be above 0 and at most 1.
  It is an **assumption** that the investor trades no more than that share of
  typical daily traded value per day. It is not a market rule.
- **`position_value`** must be a positive, finite amount.
- **The result is an estimated liquidation time,** a simplified estimate and
  not an execution guarantee.
- **If ADTV is 0 or unavailable,** the ratio and the liquidation time are
  **NaN**, never infinity.

**No liquidity score or classification.** No weighted score or HIGH/LOW label
is produced, because any thresholds would be arbitrary at this stage. The raw
metrics are exposed instead.

**Worked example** (stock ABC, no reported turnover):

| Day | Close | Volume | Estimated traded value |
|---|---|---|---|
| 1 | 100 | 1,000,000 | 100,000,000 |
| 2 | 102 | 1,200,000 | 122,400,000 |
| 3 | 101 | 800,000 | 80,800,000 |

| Result | Calculation | Value |
|---|---|---|
| ADV | 3,000,000 / 3 | **1,000,000** |
| Estimated ADTV | 303,200,000 / 3 | **101,066,666.67** |
| Position / ADTV (position Rs. 20,000,000) | 20,000,000 / 101,066,666.67 | **0.1979** |
| Daily executable value (10%) | 101,066,666.67 × 0.10 | **10,106,666.67** |
| Estimated liquidation days | 20,000,000 / 10,106,666.67 | **1.98** |

**Limitations:**
1. **Estimated money values:** without official turnover, monetary liquidity is
   estimated from close × volume. That is not turnover, and the closing price
   is not the average trade price.
2. **No execution guarantee:** estimated liquidation time is not a promise of
   how a sale would actually go.
3. **Assumed participation:** the 10% participation rate is an assumption.
4. **No bid/ask spread data** yet.
5. **No order-book depth data** yet.
6. **No market-impact model** yet: selling more than usual may move the price.
7. **No transaction-cost or slippage model** yet.
8. **No intraday liquidity data** yet: only daily totals.
9. **Short history:** historical coverage may be short, and a few unusual days
   can move the averages a lot.

## 9. Portfolio risk (fixed weights)

Portfolio risk considers **how the stocks behave together**, not each stock on
its own. It combines the stock-level analytics above into one view for weights
that the **user supplies**. Nothing is optimized and no weights are suggested.

```python
from app.analytics.portfolio_risk import (calculate_portfolio_return_series,
                                          calculate_portfolio_risk_summary)

weights = {"ABC": 0.40, "LMN": 0.25, "XYZ": 0.35}
calculate_portfolio_return_series(data, weights)                 # date, portfolio_return
calculate_portfolio_risk_summary(data, weights, portfolio_value=20_000_000,
                                 confidence_level=0.95, min_observations=20,
                                 periods_per_year=252)           # PortfolioRiskResult
```

**Weights**
- **Accepted forms:** `{symbol: weight}`, a table with `symbol` and `weight`
  columns, or `(symbol, weight)` pairs. Weights are **fractions** (0.40 = 40%).
- **Rules:**
  - every weight must be a finite number ≥ 0 (no short selling)
  - symbols must be unique and present in the data
  - the weights must sum to 1 within **`WEIGHT_SUM_TOLERANCE` = 1e-6**
- **Errors:** a clear `ValueError` is raised for an empty portfolio, duplicate
  symbols, NaN/inf, negative weights, a sum ≠ 1 or a missing symbol.
- **Never normalized:** weights of 60 and 40, or 0.6 and 0.3, are rejected, not
  rescaled.
- **Zero weights** are allowed. The holding is listed, but it does not restrict
  the common dates.

**Return alignment** (`calculate_aligned_return_matrix`, section 3)
- **Who takes part:** only the portfolio's symbols with weight > 0.
- **Which dates count:** a date is used only if **every** such holding has a
  usable daily return on it **covering the same period** (the same previous
  date).
- **Dates that are left out:**
  - a date where any holding has no return (`excluded_missing`)
  - a date where a holding's return spans a gap (`excluded_misaligned`)
- **Data rules:**
  - INVALID rows give no return and are never bridged
  - WARNING rows count
  - missing returns are **never filled with zero**

**Formulas** (w = weights, R_i(t) = daily return of holding i, P =
`periods_per_year`, default 252, n = common observations)

| Measure | Formula |
|---|---|
| Portfolio return | R_p(t) = Σ_i w_i R_i(t) |
| Cumulative return | Π(1 + R_p) − 1 |
| Annualized arithmetic return | mean(R_p) × P |
| Annualized geometric return | (1 + cumulative)^(P / n) − 1 (labelled; extrapolates a lot on short data) |
| Daily variance | **wᵀ Σ w**, Σ = daily sample covariance matrix (ddof = 1) on the common dates |
| Daily / annualized volatility | √(wᵀ Σ w) / √(wᵀ Σ w × P) |
| Value path | V(base) = 1.0, V(t) = V(t−1) × (1 + R_p(t)) |
| Maximum drawdown | on the value path, with the same peak/trough/recovery rules as section 4 |
| Historical / parametric VaR | section 6, applied to the R_p series |
| Historical / parametric CVaR | section 7, applied to the R_p series |
| Maximum weight | max(w_i) |
| HHI | Σ w_i² (1/N for N equal weights, 1 for a single stock) |

**Why these formulas and not averages**
- **Volatility:** portfolio volatility is **not** the weighted average of stock
  volatilities. Correlation matters.
  - It equals the weighted average only when the stocks are perfectly
    correlated.
  - Otherwise it is lower.
  - Two perfectly negatively correlated stocks of equal volatility, held 50/50,
    have zero volatility.
- **VaR and CVaR** come from the portfolio's **own** return series, not from
  weighted stock VaR or CVaR.
- **Drawdown** comes from the portfolio value path, not from averaging stock
  drawdowns.
- **The base date** is the close before the first common return, so a decline
  on the first day is measured from 1.0.

**Insufficient data**
- **Fewer than 2 common observations:** every statistic is NaN.
- **Fewer than `min_observations`** (default 20): VaR and CVaR are NaN, and
  `sufficient_tail_data` is False.
- **The count is always reported** in `observations`.

**Holdings liquidity** (only when `portfolio_value` is given)
- **Position value:** position_i = portfolio_value × w_i.
- **Each holding** goes through the stock-level `calculate_position_liquidity`
  (section 8) at `participation_rate` (default 0.10).
- **Holdings table columns:**
  - `symbol`, `weight`, `position_value`
  - `average_daily_traded_value`, `traded_value_source`
  - `position_to_adtv`, `daily_executable_value`, `estimated_liquidation_days`
- **No portfolio liquidation days.** Liquidation days are **not added up**,
  because sales of different stocks can happen on the same days.
- **Only descriptive figures are reported:**
  - `most_illiquid_symbol`
  - `maximum_estimated_liquidation_days`
  - `maximum_position_to_adtv`
  - `undefined_liquidation_symbols` (holdings whose ADTV is 0 or unavailable)
- **A zero-weight holding** has 0 days to liquidate (NaN if its ADTV is
  unavailable).
- **No score or classification:** there are no concentration or liquidity
  thresholds, and no SAFE / AT RISK / BUY / SELL labels.

**Worked example** (60% A, 40% B, three common days)

| | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| A return | +2% | −1% | +3% |
| B return | −1% | +2% | 0% |
| R_p = 0.6 A + 0.4 B | **0.8%** | **0.2%** | **1.8%** |
| Value path (from 1.0) | 1.008 | 1.010016 | 1.028196 |

| Result | Calculation | Value |
|---|---|---|
| Var(A), Var(B), Cov(A, B) | sample (÷ 2) | 0.00043333, 0.00023333, −0.00026667 (correlation −0.84) |
| wᵀ Σ w | 0.36 × 0.00043333 + 0.16 × 0.00023333 + 2 × 0.24 × (−0.00026667) | **0.00006533** |
| Daily volatility | √0.00006533 | **0.808%** (= sample std of R_p) |
| Weighted average of stock volatilities | 0.6 × 2.082% + 0.4 × 1.528% | 1.860%, which is **not** the portfolio volatility |
| Annualized volatility | √(0.00006533 × 252) | **12.83%** |
| Cumulative return | 1.008 × 1.002 × 1.018 − 1 | **2.82%** |
| HHI | 0.6² + 0.4² | **0.52** |

**Limitations:**
1. **Fixed weights:** the weights are held constant over the whole period.
2. **No rebalancing model:** R_p = Σ w_i R_i implicitly restores the weights
   every day, at no cost. Weight drift between rebalances, rebalancing
   schedules and the trades they need are not modelled.
3. **No transaction costs.**
4. **No market impact:** selling a large position may move the price.
5. **Descriptive liquidity:** the liquidity summary is descriptive only. It is
   not a liquidation plan and gives no portfolio liquidation time.
6. **Short data is unstable:** with few common dates, every figure, and
   especially VaR, CVaR and the annualized returns, can change a lot.
7. **No optimization:** this module suggests or optimizes no weights (see section 10).
8. **No stress testing.**
9. **No backtesting.**

## 10. Risk-aware portfolio optimization

The optimizer **chooses** weights for a universe of stocks that the user
selects. Portfolio Risk (section 9) only measures weights the user supplies.
**This optimizer is a historical quantitative allocation model, not a guarantee
of future returns.** It produces numbers, not buy or sell instructions.

```python
from app.portfolio.optimizer import optimize_portfolio, equal_weight_portfolio

result = optimize_portfolio(data, ["ABC", "LMN", "XYZ"],
                            risk_aversion=1.0, cvar_weight=1.0, return_weight=1.0,
                            confidence_level=0.95, min_observations=20, periods_per_year=252,
                            min_weight=0.0, max_weight=0.40,
                            portfolio_value=20_000_000,
                            liquidity_constraint_enabled=True, max_position_to_adtv=10)
result.optimized_weights          # {symbol: weight}
result.equal_weight_metrics       # baseline on the same scenarios
```

**Universe**
- **Symbol rules:** every selected symbol must exist in the data, and symbols
  must be unique.
- **Never dropped silently:** a missing or duplicate symbol raises a clear
  error instead of being removed.
- **Data:** only the selected stocks' data is used, and the input is never
  modified.

**Historical scenarios**
- **Source:** the date × symbol daily-return matrix comes from
  `calculate_aligned_return_matrix` (section 3) on the universe. No return is
  recalculated.
- **What makes a scenario:** a date is a scenario only if every selected stock
  has a usable return on it covering the **same period**.
- **Data rules:**
  - INVALID rows never create bridged returns
  - WARNING rows count
  - nothing is filled with zero
- **Equal weighting:** each of the N scenarios has weight 1/N.
- **Insufficient data:** fewer than `min_observations` scenarios (default 20)
  raises `InsufficientObservationsError`. The error states the count and why
  dates were excluded.

**Expected return.** μ_i is the **arithmetic mean daily return of stock i over
the scenario dates**. The expected annual return is μ_i × `periods_per_year`.
- It is a **historical sample estimate, not a guaranteed or forecast future
  return.**
- The geometric return is not used, which keeps the objective linear and convex.

**Variance.** Σ is the daily sample covariance matrix from
`calculate_covariance_matrix` (ddof = 1), on the same dates. Portfolio variance
is **wᵀΣw**, never a weighted average of stock volatilities.

**CVaR in the optimization** (Rockafellar–Uryasev linear form for equally
weighted historical scenarios). For each scenario t with return vector r_t, at
confidence C (α = 1 − C):

```
loss_t = −r_tᵀ w
u_t ≥ loss_t − z,   u_t ≥ 0           (z: auxiliary VaR threshold, u_t: excess loss)
CVaR   = z + (1 / (α N)) Σ_t u_t
```

- **It is the portfolio's own expected shortfall.** At the optimum this equals
  the historical expected shortfall of the portfolio's own scenario returns,
  with the same fractional-tail definition as section 7.
- **No stock CVaRs are combined.** The CVaR is the portfolio's, calculated from
  its scenarios.
- **Confidence:** the default is 0.95, and any 0 < C < 1 is allowed (95% and 99%
  are tested).
- **Small samples:** with N = 25, 99% puts only a quarter of one scenario in the
  tail, so the result is just the worst day.

**Objective** (minimized; every term in **daily** return units, so they share
one scale)

```
risk_aversion × wᵀΣw  +  cvar_weight × CVaR(w)  −  return_weight × μᵀw
```

| Coefficient | Default | Effect of raising it |
|---|---|---|
| `risk_aversion` | 1.0 | penalizes variance more |
| `cvar_weight` | 1.0 | penalizes tail losses more |
| `return_weight` | 1.0 | rewards historical mean return more |

- **Rules:** the coefficients must be finite and ≥ 0, and at least one must be
  positive. They are never normalized.
- **They are model parameters, not comparable units.** Daily variance is about
  1e-4, daily CVaR about 1e-2 and the daily mean about 1e-3.
  - With all three at 1, CVaR usually dominates.
  - Variance matters only with a large `risk_aversion` (tens to hundreds).
- **Changing `cvar_weight`** moves the allocation away from stocks with bad
  tails. A higher weight can never raise the optimal portfolio's CVaR, a
  property the tests check.

**Constraints**

| Constraint | Form |
|---|---|
| Fully invested, no leverage | Σ w_i = 1 |
| Long-only, minimum weight | w_i ≥ `min_weight` (≥ 0; no short selling) |
| Maximum weight (first concentration control) | w_i ≤ `max_weight` (≤ 1) |
| Liquidity (optional) | V × w_i ≤ `max_position_to_adtv` × ADTV_i |

- **Bounds:** `0 ≤ min_weight ≤ max_weight ≤ 1`.
- **Infeasible bounds are rejected before solving:**
  - n × `min_weight` > 1
  - n × `max_weight` < 1
- **No HHI limit:** HHI = Σ w² is reported only, with no threshold.

**Liquidity constraint**
- **Turning it on:** set `liquidity_constraint_enabled=True`. It needs a
  positive `portfolio_value` V and a positive `max_position_to_adtv` k.
  - When the flag is False, no liquidity constraint is applied. The holdings
    table is still filled whenever V is given, with status `NOT_APPLIED`.
- **The constraint:** position / ADTV ≤ k for every stock. This is linear in w
  for a fixed V and ADTV.
  - It is passed to the solver as w_i ≤ k × ADTV_i / V, the same constraint
    scaled to weight units.
- **Where ADTV comes from:** the stock-level liquidity module (section 8), over
  each stock's usable rows.
  - Reported turnover is used when every usable day has it.
  - Otherwise ADTV is the **estimate** close × volume, labelled
    `ESTIMATED_TRADED_VALUE`. That estimate is never called official turnover.
- **Rules for missing or zero liquidity data:**
  - A stock with ADTV = 0 (no trading) is forced to w = 0. With a positive
    `min_weight` that is infeasible.
  - A stock with **no usable liquidity data** (no volume) is **not constrained**.
    It is reported as `NO_LIQUIDITY_DATA` and listed in
    `liquidity_unconstrained_symbols`, and the UI shows a warning.
- **Infeasible limits:**
  - If the limits leave less than 100% of total capacity, or a stock cannot
    reach `min_weight`, an `InfeasibleConstraintsError` says so before solving.
  - The fix is to raise k, lower V or add more liquid stocks.
- **Per-stock status:** `AT_LIMIT` (binding), `WITHIN_LIMIT`,
  `NO_LIQUIDITY_DATA` or `NOT_APPLIED`.
- **Link to participation:** at participation rate p, a position at the limit
  needs about k / p trading days to exit. For example, k = 10 at 10% is about
  100 days.

**Solver, status and validation**
- **Solver:** CVXPY with **CLARABEL**, an interior-point solver that ships with
  CVXPY. It is deterministic, and the same input gives the same weights.
  - Tolerances are 1e-10.
  - The objective is multiplied internally by a positive constant so its terms
    are about 1. This improves solver precision and cannot change the optimal
    weights. `objective_value` is reported in the original units.
- **Only an `optimal` status is accepted:**
  - infeasible statuses raise `InfeasibleConstraintsError`
  - unbounded, inaccurate or failed solves raise `OptimizationError`
  - no allocation is ever reported from a failed solve
- **After solving, the result is rejected unless:**
  - every weight is finite
  - every weight is within [`min_weight`, `max_weight`] and its liquidity cap
    (tolerance 1e-6)
  - the weights sum to 1 (tolerance 1e-6)
  - the objective is finite and equals the objective recomputed from the
    weights. That recomputation uses wᵀΣw, the scenario expected shortfall and
    μᵀw.
- **No rescaling:** solver noise within the tolerance (for example −1e-10) is
  clipped to the bounds. Weights are never rescaled.

**Result** (`OptimizationResult`, no raw CVXPY objects)
- **Allocation:** `optimized_weights`.
- **Metrics:** `expected_annual_return`, `daily_volatility`,
  `annualized_volatility`, `historical_var` (linear quantile, as section 6),
  `historical_cvar`, `hhi` and `max_weight`.
- **Objective:** `objective_value` and its `variance_term`, `cvar_term` and
  `return_term`.
- **Solver and data:** `solver`, `solver_status`, `observations`, the date range
  and the excluded-date counts.
- **Liquidity:** the `liquidity` table when V is given.
- **Baseline:** `equal_weight_metrics` (`equal_weight_portfolio`: 1/n each),
  measured on the **same scenarios**, so the comparison is like for like. The
  baseline does not have to meet the weight or liquidity limits.
- **No verdict:** no claim is made that either allocation is better.

**Example** (three-stock sample, 24 common days, all coefficients 1, maximum
weight 100%, no liquidity constraint)

| Metric | Equal Weight | Optimized (ABC 63.53%, LMN 5.83%, XYZ 30.64%) |
|---|---|---|
| Annualized return (arithmetic) | −133.84% | −51.95% |
| Annualized volatility | 25.25% | 20.36% |
| Historical VaR (1-day) | 3.07% | 1.68% |
| Historical CVaR (1-day) | 3.20% | 1.68% |
| HHI | 0.3333 | 0.5009 |

All returns are negative because the 24-day sample is a falling market. A short
sample like this is exactly where historical estimates are least reliable.

**Limitations:**
1. **Historical estimates:** expected returns come from the past. They are not
   forecasts, and there is no guarantee of future returns.
2. **Unstable on short history:** with little data, small changes in the data
   can change the allocation a lot.
3. **Noisy covariance:** sample covariance is noisy, especially with many
   stocks and few days, and no shrinkage is applied.
4. **Scenario-dependent CVaR:** historical CVaR depends entirely on the
   available scenarios. At 99% with few days it is essentially the single worst
   day.
5. **No transaction costs.**
6. **No market impact:** the liquidity constraint limits size relative to ADTV,
   but it is not an impact model.
7. **No turnover penalty.**
8. **No rebalancing model:** this is a single-period allocation, with no
   dynamic or multi-period optimization.
9. **No sector constraints:** there is no reliable sector data yet.
10. **No short selling.**
11. **No leverage.**
12. **No guarantee of future results.**
