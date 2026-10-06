# ExitSafe

## Liquidity-Aware Tail-Risk Portfolio Optimizer

ExitSafe is a quantitative investment analysis application built around a simple question:

> **“If I need to withdraw money from my portfolio, can I exit safely?”**

Most portfolio analysis focuses on return and risk. ExitSafe goes a step further by also looking at **liquidity and the practical difficulty of getting money out of a portfolio**, especially when market conditions become difficult.

The system combines market-data analysis, portfolio risk measurement, risk-aware optimization, stress testing, backtesting, market-regime analysis and an **Exit Safety Engine** into one workflow.

The project is being developed with the Sri Lankan equity market in mind, while keeping the data and analytics layer flexible enough to work with compatible market-data sources.

---

## Why ExitSafe?

A portfolio can look attractive on paper and still be difficult to exit when cash is urgently needed.

For example, an investor may hold Rs. 20 million across several stocks and later need Rs. 5 million. The important question is not only whether the portfolio is profitable. It is also:

- How risky is the portfolio?
- How large could a short-term loss be?
- How liquid are the individual positions?
- How long might it take to liquidate the required amount?
- What happens under a market or volatility shock?
- Does the portfolio still look reasonable when tested on historical data?

ExitSafe brings these questions together and turns the results into an investment-oriented decision.

---

## Main Features

### 1. Market Data Management

Import and prepare market data from compatible CSV sources. ExitSafe normalizes different column names into a common format and validates the data before using it in the analytics layer.

The data layer keeps information about the source, validation status and warnings, so problematic observations are not silently changed.

### 2. Stock Risk & Return Analysis

For individual stocks, ExitSafe calculates:

- Daily and annualized return
- Daily and annualized volatility
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- VaR
- CVaR / Expected Shortfall
- Liquidity measures

This gives a basic risk-and-return picture before building a portfolio.

### 3. Liquidity Risk Analysis

Liquidity is a major part of ExitSafe.

The system looks at trading volume and traded value and can estimate how long a given position may take to liquidate using an assumed market-participation rate.

It can also distinguish between **actual turnover** and **estimated traded value (`close × volume`)** when turnover data is unavailable.

### 4. Portfolio Builder

Users can define the portfolio capital, selected stocks and portfolio constraints such as:

- Minimum and maximum stock weights
- Risk settings
- Liquidity constraints
- Portfolio value
- Target withdrawal amount

### 5. Risk-Aware Portfolio Optimization

ExitSafe uses CVXPY and the CLARABEL solver to build a portfolio while considering more than expected return alone.

The optimizer considers:

- Expected return
- Portfolio variance
- Historical CVaR
- Weight constraints
- Optional liquidity constraints

The result is a recommended portfolio allocation rather than a simple highest-return ranking.

### 6. Correlation & Concentration Analysis

ExitSafe calculates correlation between assets and reports concentration measures such as HHI and maximum portfolio weight. This helps show whether a portfolio is overly dependent on a small number of holdings.

### 7. Tail-Risk Analysis

The system supports both **Historical** and **Parametric** approaches for VaR and CVaR.

These measures help answer questions such as:

> “How large could a one-day loss be in the selected tail of the return distribution?”

### 8. Market Regime Detection

The current regime model is rule-based and uses volatility, trend and drawdown information to classify market conditions as:

- `NORMAL`
- `HIGH_VOL`
- `STRESS`
- `RECOVERY`

The detected regime is shown as market context. It does not secretly change the Exit Safety status.

### 9. Stress Testing

ExitSafe can test portfolios against hypothetical situations such as:

- Market -10%, -20% and -30%
- Volatility × 1.5 and × 2.0
- Liquidity -50%
- Sector-specific shocks
- Combined shocks

These scenarios are used to understand how the portfolio could behave under adverse conditions. They are not probability forecasts.

### 10. Historical Backtesting

The backtesting module uses walk-forward testing and compares:

- ExitSafe optimized portfolio
- Equal-weight portfolio
- Mean-Variance portfolio
- Market index benchmark

The test respects the timing of decisions and keeps portfolio positions drifting naturally between rebalances instead of resetting the weights every day.

### 11. Exit Safety Engine

This is the main feature that gives ExitSafe its name.

The user can ask:

> **“I have Rs. 20M in this portfolio and need to withdraw Rs. 5M. Can I safely exit?”**

The engine checks the requested exit against portfolio liquidity, portfolio CVaR and the selected stress scenario. It then returns one of four statuses:

`SAFE` · `CAUTION` · `AT_RISK` · `INSUFFICIENT_DATA`

It also explains **why** the result was reached instead of only showing a status label.

### 12. Investment Dashboard

The Streamlit dashboard brings the results together in one place, from market-data validation and individual stock analysis through portfolio construction, backtesting and the final Exit Safety Assessment.

---

# Quantitative Calculations

ExitSafe is built as a chain of reusable calculations. The main flow is:

```text
Market Data
   ↓
Returns
   ↓
Volatility
   ↓
Covariance / Correlation
   ↓
Drawdown
   ↓
Sharpe / Sortino
   ↓
VaR
   ↓
CVaR
   ↓
Liquidity
   ↓
Portfolio Risk
   ↓
Optimization
   ↓
Market Regime
   ↓
Stress Testing
   ↓
Backtesting
   ↓
Exit Safety
```

## Return

For a stock:

```text
Daily Return = (Today's Close / Previous Close) - 1
```

Returns are calculated separately for each symbol. Invalid observations do not create a return across a missing or invalid row.

## Volatility

ExitSafe uses sample standard deviation of daily returns:

```text
Daily Volatility = Sample Standard Deviation(Returns)

Annualized Volatility = Daily Volatility × √252
```

The default annualization period is 252 trading days.

## Covariance and Correlation

The system builds return matrices using aligned common dates.

```text
Annualized Covariance = Daily Covariance × 252
```

Pearson correlation is used for the correlation matrix.

## Maximum Drawdown

```text
Drawdown = (Current Close / Running Peak) - 1
```

The system records the largest drawdown and the related peak, trough and recovery information when available.

## Sharpe Ratio

The annual risk-free rate is converted into a daily rate:

```text
Daily Risk-Free Rate = (1 + Annual Risk-Free Rate)^(1/252) - 1
```

Then:

```text
Annualized Excess Return = Mean(Daily Excess Return) × 252

Sharpe Ratio = Annualized Excess Return / Annualized Volatility
```

## Sortino Ratio

Sortino focuses on downside variation:

```text
Downside Deviation = √(Mean(min(Excess Return, 0)²)) × √252

Sortino Ratio = Annualized Excess Return / Downside Deviation
```

## VaR

ExitSafe provides two one-day VaR methods.

**Historical VaR** uses the empirical lower-tail quantile:

```text
Historical VaR = -Quantile(Returns, 1 - Confidence)
```

**Parametric VaR** uses a normal-distribution assumption:

```text
Parametric VaR = -(μ + z × σ)
```

## CVaR / Expected Shortfall

CVaR measures the average loss in the worst tail beyond the VaR boundary.

ExitSafe supports historical CVaR and parametric CVaR. The historical implementation uses the exact tail probability mass, including a fractional boundary observation where necessary.

For the parametric normal approach:

```text
CVaR = -(μ - σ × φ(z) / α)
```

where `α = 1 - confidence` and `φ(z)` is the standard normal density.

## Liquidity

Average Daily Volume:

```text
ADV = Mean(Daily Volume)
```

Average Daily Traded Value uses actual turnover when the required data is available. Otherwise ExitSafe can estimate it as:

```text
Estimated Traded Value = Close × Volume
```

For a position or planned exit:

```text
Position / ADTV = Position-to-ADTV Ratio

Daily Executable Value = ADTV × Participation Rate

Estimated Liquidation Days = Position / Daily Executable Value
```

For example, a 10% participation rate means the calculation assumes execution at 10% of normal daily traded value per day.

## Portfolio Risk

For portfolio weights `w` and asset returns `R`:

```text
Portfolio Return = Σ(wᵢ × Rᵢ)
```

Portfolio variance:

```text
Portfolio Variance = wᵀΣw
```

Annualized portfolio volatility:

```text
Annualized Volatility = √(wᵀΣw × 252)
```

Portfolio VaR and CVaR are calculated from the portfolio return series rather than simply adding or weighting individual stock VaRs.

Concentration is measured using:

```text
HHI = Σ(wᵢ²)
Maximum Weight = max(wᵢ)
```

## Portfolio Optimization

The optimization objective combines return, variance and historical CVaR:

```text
Minimize:

  Risk Aversion × Portfolio Variance
+ CVaR Weight × Portfolio CVaR
- Return Weight × Expected Portfolio Return
```

Subject to constraints such as:

```text
Σ weights = 1
weightᵢ ≥ minimum weight
weightᵢ ≤ maximum weight
```

Optional liquidity limits can also restrict a position relative to its ADTV.

## Stress Testing

Stress scenarios apply explicit hypothetical shocks to the portfolio. The current engine supports market, volatility, liquidity, sector and combined scenarios.

A liquidity shock changes the estimated execution capacity; it does not directly change asset prices.

## Backtesting

The backtest uses a walk-forward structure:

```text
Training Data
     ↓
Portfolio Decision / Rebalance
     ↓
Future Test Period
```

Only out-of-sample results are used for the performance report.

---

# How the Exit Safety Engine Works

The requested withdrawal is allocated proportionally across the current holdings.

For portfolio value `V`, target exit `E`, and stock weight `wᵢ`:

```text
Holding Valueᵢ = V × wᵢ
Planned Exitᵢ = E × wᵢ
```

The engine then checks the planned exit amount for each holding.

### Exit Horizon

The overall exit horizon is the **maximum** usable holding-level liquidation time under proportional parallel liquidation.

It is not the sum and not the average of the holding-level times.

### Liquidity Coverage

```text
Coverage Ratio = Covered Exit Value / Target Exit Value
```

Missing or unusable liquidity data is reported rather than hidden.

### Reference CVaR Loss

The portfolio's existing CVaR is reused:

```text
Reference CVaR Loss = Target Exit × Portfolio CVaR
```

This is a one-day reference amount, not a prediction of the entire liquidation-period loss.

### Reference Stress Loss

For the selected stress scenario:

```text
Reference Stress Loss = Target Exit × Scenario Loss
```

Again, this is a hypothetical reference and not a probability forecast.

### Status Rules

The current default policy is:

| Setting | Default |
|---|---:|
| Safe exit horizon | 5 days |
| Caution exit horizon | 20 days |
| CVaR caution threshold | 5% |
| CVaR at-risk threshold | 10% |
| Stress-loss caution threshold | 10% |
| Stress-loss at-risk threshold | 20% |
| Minimum liquidity coverage | 100% |
| Insufficient-data coverage threshold | 50% |
| CVaR method | Historical |

The decision is evaluated in this order:

```text
INSUFFICIENT_DATA
      ↓
AT_RISK
      ↓
CAUTION
      ↓
SAFE
```

These thresholds are model-policy settings, not universal investment rules, and can be changed through the application settings.

One important detail is the default `Market -10%` stress scenario. With a zero base return, a fully invested long-only portfolio receives a 10% stress loss regardless of its weights. Because the default caution threshold is also 10%, that scenario will make the result at least `CAUTION`. This is expected behaviour under the current policy.

---

# Architecture

ExitSafe uses a **modular monolith** architecture. It is one application, but its responsibilities are separated into clear modules so that each part can be tested and changed independently.

```text
                    ┌──────────────────────┐
                    │    Market Data       │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Ingestion / Cleaning │
                    │     / Validation     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Analytics Layer    │
                    │ Returns / Risk /     │
                    │ VaR / CVaR / Liquidity│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Portfolio Layer    │
                    │ Risk + Optimization  │
                    └──────────┬───────────┘
                               ↓
             ┌─────────────────┼─────────────────┐
             ↓                 ↓                 ↓
      Market Regime       Stress Testing     Backtesting
             └─────────────────┼─────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Exit Safety Engine  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Streamlit Dashboard │
                    └──────────────────────┘
```

### Data Layer

Handles loading, normalization, schemas, validation, provenance and source-related information.

### Analytics Layer

Contains the reusable calculations for returns, volatility, covariance, correlation, drawdown, Sharpe, Sortino, VaR, CVaR, liquidity and portfolio risk.

### Portfolio Layer

Builds and validates portfolio constraints and runs the risk-aware optimizer.

### Regime Layer

Determines the current market regime using the rule-based regime model.

### Stress Testing Layer

Runs the supported hypothetical stress scenarios and calculates portfolio impact.

### Backtesting Layer

Runs walk-forward historical strategy tests and produces out-of-sample performance results.

### Exit Engine

Combines liquidity, CVaR and stress information to produce the final Exit Safety assessment.

### UI Layer

The Streamlit interface provides the inputs, calculations, charts, tables and final investment-oriented results.

---

# Technology Stack

| Technology | Used for |
|---|---|
| Python 3.12 | Core application |
| Streamlit | Web interface |
| Pandas | Data processing |
| NumPy | Numerical calculations |
| SciPy | Statistical calculations |
| Statsmodels | Statistical analysis support |
| CVXPY | Portfolio optimization |
| CLARABEL | Optimization solver |
| Scikit-learn | Future statistical / ML modelling |
| Plotly | Interactive charts |
| Pytest | Automated testing |
| Git / GitHub | Version control |
| PostgreSQL | Planned persistent storage |
| SQLAlchemy | Planned database access |

---

# Project Structure

```text
ExitSafe/
│
├── app/
│   ├── analytics/
│   │   ├── common.py
│   │   ├── returns.py
│   │   ├── volatility.py
│   │   ├── covariance.py
│   │   ├── drawdown.py
│   │   ├── ratios.py
│   │   ├── var.py
│   │   ├── cvar.py
│   │   ├── liquidity.py
│   │   └── portfolio_risk.py
│   │
│   ├── backtesting/
│   │   └── backtest_engine.py
│   ├── config/
│   ├── data/
│   │   ├── cleaners/
│   │   ├── loaders/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   └── validators/
│   ├── exit_engine/
│   │   └── exit_safety.py
│   ├── portfolio/
│   │   └── optimizer.py
│   ├── regime/
│   │   └── regime_detector.py
│   ├── stress_testing/
│   │   └── stress_engine.py
│   └── ui/
│       ├── console.py
│       └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── database/
├── notebooks/
├── scripts/
├── docs/
├── tests/
│   ├── analytics/
│   ├── backtesting/
│   ├── exit_engine/
│   ├── fixtures/
│   └── ui/
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Data Format

ExitSafe works with normalized CSV market data. A typical file can contain:

```text
Date,Symbol,Open,High,Low,Close,Volume,Change %,Turnover,Trades
```

The importer also supports common variations such as `Price`, `Closing Price`, `Vol.` and `Share Volume`.

A few important data rules:

- `Close` is preferred over `Price` when both are provided.
- Conflicting close-price sources are treated as invalid instead of silently choosing one.
- Dates can be parsed using an explicit format when required.
- Volume values such as `7.94M` are supported.
- When turnover is missing, `close × volume` may be stored as `estimated_traded_value`.
- Estimated traded value is never presented as actual turnover.
- Validation warnings are retained so that data-quality issues remain visible.

---

# Running ExitSafe

## Requirements

- Python 3.12.x
- `pip`
- Git
- A modern web browser

## 1. Clone the project

```bash
git clone <YOUR_REPOSITORY_URL>
cd ExitSafe
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Start the application

```bash
streamlit run app/ui/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

# Running the Tests

Run the complete test suite with:

```bash
python -m pytest
```

At the current Exit Safety milestone, the suite contains **1027 passing tests** across the data layer, analytics, portfolio risk, optimization, market regimes, stress testing, backtesting, Exit Safety Engine and UI checks.

---

# Example

Suppose an investor has:

```text
Portfolio Value = Rs. 20,000,000
Target Exit     = Rs. 5,000,000
```

and the current portfolio is:

| Holding | Weight | Planned Exit |
|---|---:|---:|
| ABC | 40% | Rs. 2,000,000 |
| XYZ | 35% | Rs. 1,750,000 |
| LMN | 25% | Rs. 1,250,000 |

With a 10% participation assumption, an example assessment is:

| Metric | Result |
|---|---:|
| Liquidity coverage | 100% |
| ABC exit time | 1.2 days |
| XYZ exit time | 5.9 days |
| LMN exit time | 15.7 days |
| Exit horizon | **15.7 days** |
| Historical CVaR | 2.83% |
| Reference CVaR loss | Rs. 141,307 |
| Reference stress loss | Rs. 500,000 |
| Exit Safety | **CAUTION** |

Here the exit horizon is above the default 5-day safe level, and the default `Market -10%` scenario reaches the 10% stress-loss caution threshold.

The result is an assessment based on the configured model assumptions, not a guarantee of execution.

---

# Testing and Reliability

The project puts a strong emphasis on making the calculations reproducible and avoiding misleading results.

Examples include:

- invalid market-data rows are not silently repaired
- missing observations do not create artificial return links
- covariance and correlation use aligned return observations
- portfolio VaR and CVaR are calculated from portfolio returns
- portfolio weights are validated rather than silently normalized
- backtesting uses strict training and test timing
- portfolio positions drift between rebalances
- stress scenarios are explicitly defined as hypothetical shocks
- liquidity gaps are shown to the user
- unsupported market-impact values are never invented

---

# Current Limitations

ExitSafe is a quantitative decision-support tool, so some real-world execution effects are outside the current model.

The current version does not model:

- Bid-ask spreads
- Order-book depth
- Market impact
- Transaction costs
- Slippage
- Intraday execution
- Optimized liquidation ordering

Other limitations include the use of an assumed participation rate, possible estimation of traded value from `close × volume`, potentially short or unadjusted historical price data depending on the source, and the lack of probability estimates for stress scenarios.

A `SAFE` result means the portfolio passed the configured model checks. It does **not** guarantee that the requested amount can be executed exactly as planned in the real market.

---

# Current Status

The core ExitSafe quantitative pipeline is implemented, including:

- Market data ingestion and validation
- Return and volatility analysis
- Covariance and correlation
- Maximum drawdown
- Sharpe and Sortino ratios
- VaR and CVaR
- Liquidity analysis
- Portfolio risk analysis
- Risk-aware portfolio optimization
- Market regime detection
- Stress testing
- Walk-forward backtesting
- Exit Safety Engine
- Streamlit dashboard
- Automated testing

Future work can extend the system with stronger market-impact modelling, transaction-cost and slippage models, optimized liquidation ordering, richer intraday liquidity analysis, advanced regime models, external market/event intelligence and persistent database-backed data management.

---

# Project Idea in One Line

> **ExitSafe helps investors decide not only where to invest, but also whether they can reasonably exit when they need their money.**

---

# Disclaimer

ExitSafe is a quantitative investment research and decision-support project. Its results depend on the historical data, model assumptions, parameters and hypothetical scenarios selected by the user. It is not financial advice, and model outputs do not guarantee future returns, liquidity or execution.
