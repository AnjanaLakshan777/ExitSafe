# How ExitSafe Turns Calculations into a Recommendation

ExitSafe has two user journeys. Both follow the same order:

**Your information → Recommendation → Why? → Calculations**

The recommendation is a **rule-based interpretation layer** (`app/ui/decision.py`).
It is not a prediction model and not AI. It runs the existing engines (the
risk-aware optimizer, portfolio risk analysis and the Exit Safety Engine) with
the dashboard's default settings, then reads their statuses and numbers through
the fixed rules below. Every evidence point under **Why are we saying this?**
quotes a number those engines calculated.

World market threats, news and Gemini search results are shown as context only.
They are never inputs to the rules, so they can't change a recommendation, the
portfolio weights or an Exit Safety status.

## 1. Start Investing (new investor)

**You enter:** the amount to invest, how long the money can stay invested, a risk
preference, the companies you're considering (their price history comes from
your tracked data, an upload or a sample), and optionally the most to put in any
one company.

**What runs:**

| Input | Becomes this existing optimizer setting |
|---|---|
| Conservative | risk aversion 2, CVaR weight 2, return weight 0.5, at most 30% per company |
| Balanced | risk aversion 1, CVaR weight 1, return weight 1, at most 40% per company (the optimizer's defaults) |
| Growth | risk aversion 0.5, CVaR weight 0.5, return weight 2, at most 50% per company |
| Less than 1 year | liquidity limit on: each position at most 2 × its average daily traded value (about 20 trading days to sell at 10% participation) |
| 1 to 3 years | liquidity limit on: at most 10 × average daily traded value (about 100 days) |
| More than 3 years | liquidity limit off |

If there are too few companies for the per-company limit to add up to 100%, the
limit is raised just enough and the dashboard says so.

The optimizer's weights are the **recommended allocation**. The same portfolio
then goes through Portfolio Risk Analysis and the Exit Safety Engine (selling the
whole amount), which gives the risk check shown as GOOD, CAUTION, AT RISK or
INSUFFICIENT DATA.

**Actions** (the first rule that applies):

| Action | When |
|---|---|
| INSUFFICIENT DATA | the optimizer or the Exit Safety checks don't have enough data |
| REVIEW THE CANDIDATES | no allocation meets the limits; or the best mix lost money on average in the data; or the risk of large loss (CVaR) is at caution or worse; or the stress loss is at risk |
| INVEST GRADUALLY | buying (or later selling) the positions takes longer than the 5-day safe horizon |
| PROCEED WITH THIS ALLOCATION | none of the above |

## 2. I Already Invested (existing investor)

**You enter:** your holdings as *Symbol, number of shares* (or, if you prefer, each
holding's current value in Rs.), or fill them in from *My Investments*. Optionally
add what you originally invested (shown for reference only) and an amount you may
need to withdraw.

**Current portfolio value** is calculated, never assumed: the sum of quantity ×
latest approved price. The latest approved price is the most recent valid close in
your market data after the data checks and the provenance selection, so secondary
AI-sourced (Gemini) prices are only used if you chose to include them. A holding
with no price is shown as *missing current valuation data*, is left out of the
value and the analysis (no price is ever assumed), and a warning names it. The
original investment, the current value and the target exit are shown as three
separate amounts, and the withdrawal is checked against the current value.

**Portfolio status** is the Exit Safety Engine's own status for selling the whole
portfolio, with its default thresholds (SAFE is shown as GOOD). It covers the
time to sell, the risk of a large loss (CVaR), a hypothetical market fall
(`Market -10%`) and, as context, the market regime.

**Should I exit?** With a withdrawal amount, the Exit Safety Engine assesses that
amount: SAFE, CAUTION, AT RISK or INSUFFICIENT DATA, with the exit horizon,
liquidity coverage, CVaR, stress loss, regime and its reasons.

**Actions** (the first rule that applies):

| Action | When |
|---|---|
| INSUFFICIENT DATA | the Exit Safety checks don't have enough data |
| CONSIDER A STAGED EXIT | a withdrawal was entered and its liquidity check is at caution or worse |
| REDUCE EXPOSURE | the risk of large loss is at caution or worse, or the stress loss is at risk |
| REVIEW YOUR PORTFOLIO | selling everything takes longer than the safe horizon, or one holding is above 40% (the optimizer's default limit) |
| HOLD | none of the above |

A stress result at CAUTION on its own doesn't change the action: the default
scenario is a market-wide fall, which costs any fully invested stock portfolio
about the same.

## 3. Thresholds

The rules reuse the Exit Safety Engine's default policy: safe exit within 5
trading days (caution up to 20), CVaR caution at 5% and at risk at 10%, stress
loss caution at 10% and at risk at 20%, full liquidity coverage expected. They
are model settings, not financial standards, and can be changed in the Exit
Safety section under **See Calculations**.

## 4. See Calculations

**See Calculations** opens the full analysis: market data, returns, volatility,
covariance and correlation, drawdown, Sharpe and Sortino, VaR, CVaR, liquidity,
portfolio risk, optimization, market regime, stress testing, backtesting and the
Exit Safety assessment. Its settings start from your inputs (the recommended or
entered holdings, the amount, the optimizer settings and the withdrawal), so the
numbers match the recommendation. Changing them there doesn't change the
recommendation above.

## 5. Limits

- Recommendations are only as good as the price history behind them. Short or
  unadjusted histories can make returns look extreme.
- Market impact, spreads, transaction costs and taxes are not modelled.
- ExitSafe supports decisions; it is not financial advice and doesn't guarantee
  that a trade can be executed as planned.
