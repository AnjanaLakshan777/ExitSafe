"""Walk-forward backtesting of the ExitSafe allocation process (no look-ahead).

The backtest repeatedly TRAINS on the past and TESTS on the next unseen period.
Past performance does not guarantee future performance, and results on the
synthetic fixtures are not evidence about any real market.

Timing convention (every rebalance)
  calendar        the sorted dates on which any selected stock has a row
  decision        after the close of the last training day D = calendar[k - 1]
  training data   the ``training_window`` dates calendar[k - training_window .. k - 1]
                  (strictly before the first test day; nothing later is passed on)
  rebalance date  T = calendar[k], the first test day: positions are bought at D's
                  close, so the first out-of-sample return is close(T) / close(D) - 1
  test window     calendar[k .. k + test_window - 1]; the next rebalance is
                  ``rebalance_frequency`` dates later (test_window <= rebalance_frequency;
                  dates between a test window and the next rebalance are not invested
                  or evaluated). Only complete test windows are used.

Look-ahead prevention: weights, expected returns, covariance, CVaR, liquidity
(ADTV for the liquidity constraint) and the minimum-history check come only from
the training rows. The portfolio value used for the liquidity constraint is the
strategy's own value at D. Out-of-sample returns are realized returns of the
test dates only.

Out-of-sample returns: a test day is evaluated only if EVERY selected stock (and
the market index, when one is supplied) has a usable return covering the same
period (the common-date alignment of app.analytics.covariance). Days that fail
this are excluded, never zero-filled or bridged, and counted. Within a test
window the positions are held without trading (buy-and-hold: weights drift with
prices); the daily portfolio return is the value-weighted return of the
holdings, and value_t = value_(t-1) * (1 + return_t).

Strategies (same universe, training windows, rebalance dates and test days)
  EXITSAFE_OPTIMIZED  app.portfolio.optimizer.optimize_portfolio on the training
                      rows: risk_aversion * variance + cvar_weight * CVaR -
                      return_weight * mean, min/max weight, optional liquidity limit
  MEAN_VARIANCE       the same optimizer and inputs with cvar_weight = 0 and no
                      liquidity constraint: risk_aversion * variance -
                      return_weight * mean (a traditional mean-variance baseline)
  EQUAL_WEIGHT        1 / N for every selected stock
  MARKET_INDEX        the supplied index's returns on the same test days (no
                      weights); UNAVAILABLE when no index is supplied or it shares
                      no same-period dates
  A rebalance whose training rows hold fewer than ``min_observations`` common
  returns is skipped for every strategy; a strategy whose optimization fails is
  recorded with the reason. In both cases existing positions are kept (no
  trade); a strategy without positions is not invested, and the comparison
  starts at the first test window in which every stock strategy is invested.

Metrics (out-of-sample returns only, via the existing analytics modules on each
strategy's value path): cumulative return, annualized arithmetic return (mean x
P), annualized geometric return, annualized volatility, Sharpe and Sortino
(app.analytics.ratios, explicit risk-free rate), maximum drawdown, historical
VaR and CVaR. No transaction costs, slippage, taxes or fees are modelled.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.analytics.common import TRADING_DAYS_PER_YEAR, require_canonical_columns, validate_periods_per_year
from app.analytics.covariance import calculate_aligned_return_matrix
from app.analytics.cvar import calculate_cvar_summary
from app.analytics.drawdown import calculate_drawdown_series, calculate_maximum_drawdown
from app.analytics.liquidity import validate_position_value
from app.analytics.ratios import calculate_risk_adjusted_ratios, validate_risk_free_rate
from app.analytics.returns import CANONICAL_DAILY_RETURN, calculate_daily_returns
from app.analytics.var import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MIN_OBSERVATIONS,
    calculate_var_summary,
    validate_confidence_level,
    validate_min_observations,
)
from app.portfolio.optimizer import (
    equal_weight_portfolio,
    optimize_portfolio,
    validate_objective_weights,
    validate_symbols,
    validate_weight_bounds,
)
from app.regime.regime_detector import prepare_index_series

EXITSAFE, EQUAL_WEIGHT, MEAN_VARIANCE, MARKET_INDEX = (
    "EXITSAFE_OPTIMIZED", "EQUAL_WEIGHT", "MEAN_VARIANCE", "MARKET_INDEX")
STOCK_STRATEGIES = (EXITSAFE, EQUAL_WEIGHT, MEAN_VARIANCE)
STRATEGY_LABELS = {EXITSAFE: "ExitSafe", EQUAL_WEIGHT: "EqualWeight",
                   MEAN_VARIANCE: "MeanVariance", MARKET_INDEX: "MarketIndex"}
REBALANCED, CARRIED_FORWARD, NOT_INVESTED = "REBALANCED", "CARRIED_FORWARD", "NOT_INVESTED"
AVAILABLE, UNAVAILABLE = "AVAILABLE", "UNAVAILABLE"
METRIC_ROWS = ["Cumulative Return", "Annualized Return (arithmetic)",
               "Annualized Return (geometric)", "Annualized Volatility", "Sharpe", "Sortino",
               "Maximum Drawdown", "Historical VaR (1-day)", "Historical CVaR (1-day)",
               "Observations", "Rebalances"]
LOG_COLUMNS = ["window", "rebalance_date", "training_start", "training_end", "test_start",
               "test_end", "strategy", "status", "reason", "weights", "training_observations",
               "solver_status", "evaluated_days", "excluded_days", "parameters"]


class BacktestError(ValueError):
    """The backtest cannot produce a valid out-of-sample result."""


@dataclass(frozen=True)
class BacktestConfig:
    """Every backtest assumption in one place (defaults shown are the documented ones)."""
    symbols: tuple
    initial_capital: float = 1_000_000.0
    training_window: int = 60
    test_window: int = 20
    rebalance_frequency: int = 20
    periods_per_year: float = TRADING_DAYS_PER_YEAR
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    min_observations: int = DEFAULT_MIN_OBSERVATIONS
    risk_free_rate: float = 0.0
    risk_aversion: float = 1.0
    cvar_weight: float = 1.0
    return_weight: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    liquidity_constraint_enabled: bool = False
    max_position_to_adtv: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "symbols", tuple(validate_symbols(self.symbols)))
        validate_position_value(self.initial_capital)
        _whole(self.training_window, "training_window", 2)
        _whole(self.test_window, "test_window", 1)
        _whole(self.rebalance_frequency, "rebalance_frequency", 1)
        if self.test_window > self.rebalance_frequency:
            raise ValueError(f"test_window ({self.test_window}) cannot exceed rebalance_frequency "
                             f"({self.rebalance_frequency}): test windows would overlap")
        validate_periods_per_year(self.periods_per_year)
        validate_confidence_level(self.confidence_level)
        validate_min_observations(self.min_observations)
        validate_risk_free_rate(self.risk_free_rate)
        validate_objective_weights(self.risk_aversion, self.cvar_weight, self.return_weight)
        validate_weight_bounds(self.min_weight, self.max_weight)
        n = len(self.symbols)
        if n * self.min_weight > 1 + 1e-12 or n * self.max_weight < 1 - 1e-12:
            raise ValueError(f"Infeasible weight bounds for {n} stocks: minimum {self.min_weight:g}, "
                             f"maximum {self.max_weight:g}")
        if not isinstance(self.liquidity_constraint_enabled, bool):
            raise ValueError("liquidity_constraint_enabled must be True or False")
        if self.liquidity_constraint_enabled:
            if (isinstance(self.max_position_to_adtv, bool)
                    or not isinstance(self.max_position_to_adtv, (int, float))
                    or not math.isfinite(self.max_position_to_adtv)
                    or self.max_position_to_adtv <= 0):
                raise ValueError("max_position_to_adtv must be a positive number when the "
                                 "liquidity constraint is enabled")

    def optimizer_parameters(self, strategy):
        """The optimizer settings a strategy uses (recorded in the rebalance log)."""
        if strategy == EQUAL_WEIGHT:
            return {"weights": "1/N"}
        common = {"risk_aversion": self.risk_aversion, "return_weight": self.return_weight,
                  "min_weight": self.min_weight, "max_weight": self.max_weight,
                  "confidence_level": self.confidence_level,
                  "min_observations": self.min_observations,
                  "periods_per_year": self.periods_per_year}
        if strategy == MEAN_VARIANCE:
            return {**common, "cvar_weight": 0.0, "liquidity_constraint_enabled": False}
        return {**common, "cvar_weight": self.cvar_weight,
                "liquidity_constraint_enabled": self.liquidity_constraint_enabled,
                "max_position_to_adtv": self.max_position_to_adtv}


@dataclass(frozen=True)
class RebalanceWindow:
    window: int
    training_dates: tuple
    test_dates: tuple
    training_observations: int        # common daily returns inside the training rows
    skip_reason: str | None           # set when the training rows are insufficient


@dataclass(frozen=True)
class RebalanceResult:
    window: int
    rebalance_date: pd.Timestamp      # first test day (weights apply from here)
    training_start: pd.Timestamp
    training_end: pd.Timestamp        # decision after this close
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    strategy: str
    status: str                       # REBALANCED / CARRIED_FORWARD / NOT_INVESTED
    reason: str
    weights: dict                     # new target weights, or held (drifted) weights
    training_observations: int
    solver_status: str | None
    evaluated_days: int
    excluded_days: int
    parameters: dict


@dataclass(frozen=True)
class StrategyBacktest:
    strategy: str
    returns: pd.Series                # out-of-sample daily returns, every invested test day
    rebalances: list                  # RebalanceResult per window


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    strategies: tuple                 # labels in the comparison (MarketIndex only if available)
    returns: pd.DataFrame             # date + one column per strategy (out-of-sample)
    equity_curve: pd.DataFrame        # date + normalized value (1.0 at the comparison start)
    drawdowns: pd.DataFrame           # date + drawdown of each equity curve
    metrics: pd.DataFrame             # METRIC_ROWS x strategies
    rebalance_log: pd.DataFrame       # LOG_COLUMNS
    rebalances: list                  # RebalanceResult objects
    windows: list                     # RebalanceWindow objects
    final_weights: dict               # strategy -> weights of its latest rebalance
    final_capital: dict               # strategy -> initial_capital * final normalized value
    comparison_start: pd.Timestamp    # base date (value 1.0)
    comparison_end: pd.Timestamp
    observations: int                 # evaluated out-of-sample days
    excluded_test_days: int
    skipped_rebalances: list = field(default_factory=list)   # (window, rebalance date, reason)
    market_index_status: str = UNAVAILABLE
    market_index_reason: str = ""


def build_walk_forward_schedule(data, config):
    """The rebalance windows (complete test windows only), each with its training check."""
    universe, calendar = _universe(data, config)
    windows = []
    k, n = config.training_window, len(calendar)
    while k + config.test_window <= n:
        training = tuple(calendar[k - config.training_window:k])
        observations = len(calculate_aligned_return_matrix(_rows(universe, training)))
        reason = (None if observations >= config.min_observations else
                  f"insufficient training data: {observations} common daily return(s), "
                  f"{config.min_observations} required")
        windows.append(RebalanceWindow(len(windows) + 1, training,
                                       tuple(calendar[k:k + config.test_window]), observations,
                                       reason))
        k += config.rebalance_frequency
    if not windows:
        raise BacktestError(f"Not enough history: one training window ({config.training_window}) "
                            f"plus one test window ({config.test_window}) needs "
                            f"{config.training_window + config.test_window} observations, the "
                            f"data has {n}.")
    return windows


def strategy_weights(strategy, training_data, config, portfolio_value):
    """(weights, solver status) for one strategy from the training rows ONLY."""
    if strategy == EQUAL_WEIGHT:
        return equal_weight_portfolio(config.symbols), None
    if strategy not in (EXITSAFE, MEAN_VARIANCE):
        raise ValueError(f"Unknown strategy {strategy!r}")
    exitsafe = strategy == EXITSAFE
    result = optimize_portfolio(
        training_data, list(config.symbols), risk_aversion=config.risk_aversion,
        cvar_weight=config.cvar_weight if exitsafe else 0.0, return_weight=config.return_weight,
        confidence_level=config.confidence_level, min_observations=config.min_observations,
        periods_per_year=config.periods_per_year, min_weight=config.min_weight,
        max_weight=config.max_weight,
        portfolio_value=portfolio_value if exitsafe and config.liquidity_constraint_enabled else None,
        liquidity_constraint_enabled=exitsafe and config.liquidity_constraint_enabled,
        max_position_to_adtv=config.max_position_to_adtv if exitsafe else None)
    return result.optimized_weights, result.solver_status


def run_strategy_backtest(data, strategy, config, windows=None, evaluable=None):
    """Walk one stock strategy through the schedule. Returns a StrategyBacktest.

    ``evaluable`` (date x symbol returns of the evaluable test days) defaults to
    the common-date returns of the selected stocks.
    """
    if strategy not in STOCK_STRATEGIES:
        raise ValueError(f"Unknown stock strategy {strategy!r}")
    universe, _ = _universe(data, config)
    windows = windows if windows is not None else build_walk_forward_schedule(data, config)
    if evaluable is None:
        evaluable = calculate_aligned_return_matrix(universe).reindex(columns=list(config.symbols))
    symbols = list(config.symbols)
    holdings, returns, log = None, {}, []
    parameters = config.optimizer_parameters(strategy)
    for w in windows:
        solver, reason = None, ""
        if w.skip_reason:
            reason = w.skip_reason
        else:
            training = _rows(universe, w.training_dates)
            value = config.initial_capital * (sum(holdings.values()) if holdings else 1.0)
            try:
                weights, solver = strategy_weights(strategy, training, config, value)
                capital = sum(holdings.values()) if holdings else 1.0
                holdings = {s: capital * weights[s] for s in symbols}
            except ValueError as exc:
                reason = f"{type(exc).__name__}: {exc}"
        status = (REBALANCED if not reason else CARRIED_FORWARD if holdings else NOT_INVESTED)
        if status == CARRIED_FORWARD:
            reason += "; previous positions kept (no trade)"
        start = ({s: float(h / sum(holdings.values())) for s, h in holdings.items()}
                 if holdings else {})
        evaluated = 0
        if holdings:
            for day in w.test_dates:
                if day not in evaluable.index:
                    continue
                r = evaluable.loc[day]
                total = sum(holdings.values())
                returns[day] = float(sum(holdings[s] * r[s] for s in symbols) / total)
                holdings = {s: float(holdings[s] * (1 + r[s])) for s in symbols}
                evaluated += 1
        in_window = sum(1 for d in w.test_dates if d in evaluable.index)
        log.append(RebalanceResult(
            window=w.window, rebalance_date=w.test_dates[0], training_start=w.training_dates[0],
            training_end=w.training_dates[-1], test_start=w.test_dates[0],
            test_end=w.test_dates[-1], strategy=strategy, status=status, reason=reason,
            weights=start, training_observations=w.training_observations, solver_status=solver,
            evaluated_days=evaluated, excluded_days=len(w.test_dates) - in_window,
            parameters=parameters))
    return StrategyBacktest(strategy, pd.Series(returns, dtype="float64").sort_index(), log)


def run_walk_forward_backtest(data, config, index_data=None, index_name=None):
    """ExitSafe vs EqualWeight vs MeanVariance (vs MarketIndex) out of sample."""
    if not isinstance(config, BacktestConfig):
        raise ValueError("config must be a BacktestConfig")
    universe, calendar = _universe(data, config)
    windows = build_walk_forward_schedule(data, config)
    evaluable = calculate_aligned_return_matrix(universe).reindex(columns=list(config.symbols))

    index_returns, index_status, index_reason = None, UNAVAILABLE, "No market-index data supplied."
    if index_data is not None:
        index_returns = _index_returns(index_data, index_name, calendar)
        common = evaluable.index.intersection(index_returns.index)
        if len(common):
            evaluable, index_status, index_reason = evaluable.loc[common], AVAILABLE, ""
        else:
            index_returns = None
            index_reason = ("The market index has no same-period returns on the stock test "
                            "dates.")

    runs = {s: run_strategy_backtest(data, s, config, windows, evaluable) for s in STOCK_STRATEGIES}
    invested = [all(runs[s].rebalances[i].status != NOT_INVESTED for s in STOCK_STRATEGIES)
                for i in range(len(windows))]
    if not any(invested):
        reasons = sorted({r.reason for run in runs.values() for r in run.rebalances if r.reason})
        raise BacktestError("No completed out-of-sample period: no test window in which every "
                            "strategy holds a portfolio. " + " | ".join(reasons[:3]))
    first = invested.index(True)
    dates = [d for w in windows[first:] for d in w.test_dates if d in evaluable.index]
    if not dates:
        raise BacktestError("No completed out-of-sample period: no evaluable test day (every "
                            "test day lacks a common return for all selected stocks).")

    labels = [STRATEGY_LABELS[s] for s in STOCK_STRATEGIES]
    returns = pd.DataFrame({STRATEGY_LABELS[s]: runs[s].returns.reindex(dates)
                            for s in STOCK_STRATEGIES})
    if index_returns is not None:
        returns[STRATEGY_LABELS[MARKET_INDEX]] = index_returns.reindex(dates)
        labels.append(STRATEGY_LABELS[MARKET_INDEX])
    base_date = windows[first].training_dates[-1]
    equity = create_equity_curve(returns, base_date)
    drawdowns = _drawdowns(equity)
    rebalances = [r for s in STOCK_STRATEGIES for r in runs[s].rebalances]
    comparison_windows = {w.window for w in windows[first:]}
    counts = {STRATEGY_LABELS[s]: sum(1 for r in runs[s].rebalances
                                      if r.status == REBALANCED and r.window in comparison_windows)
              for s in STOCK_STRATEGIES}
    metrics = pd.DataFrame({label: calculate_backtest_metrics(
        returns[label], config, rebalances=counts.get(label)) for label in labels},
        index=METRIC_ROWS)
    final = {}
    for s in STOCK_STRATEGIES:
        done = [r for r in runs[s].rebalances if r.status == REBALANCED]
        final[STRATEGY_LABELS[s]] = done[-1].weights if done else {}
    return BacktestResult(
        config=config, strategies=tuple(labels), returns=returns.reset_index(names="date"),
        equity_curve=equity, drawdowns=drawdowns, metrics=metrics,
        rebalance_log=_log_frame(rebalances), rebalances=rebalances, windows=windows,
        final_weights=final,
        final_capital={label: config.initial_capital * float(equity[label].iloc[-1])
                       for label in labels},
        comparison_start=base_date, comparison_end=dates[-1], observations=len(dates),
        excluded_test_days=sum(len(w.test_dates) for w in windows[first:]) - len(dates),
        skipped_rebalances=[(w.window, w.test_dates[0], w.skip_reason) for w in windows
                            if w.skip_reason],
        market_index_status=index_status, market_index_reason=index_reason)


def create_equity_curve(returns, base_date):
    """Normalized value paths: 1.0 on ``base_date``, then value_t = value_(t-1) * (1 + r_t).

    ``returns`` is indexed by date with one column per strategy.
    """
    values = (1 + returns).cumprod()
    start = pd.DataFrame(1.0, index=pd.DatetimeIndex([base_date]), columns=returns.columns)
    curve = pd.concat([start, values])
    curve.index.name = "date"
    return curve.reset_index()


def calculate_backtest_metrics(returns, config, rebalances=None):
    """Out-of-sample metrics of one daily return series (pd.Series indexed by date).

    The series is turned into its value path (1.0 at a base row, then compounded)
    and measured with the existing analytics modules, so no formula is duplicated.
    """
    r = returns.dropna().astype("float64")
    n = len(r)
    if n == 0:
        return pd.Series(np.nan, index=METRIC_ROWS)
    path = _value_frame(r)
    ratios = calculate_risk_adjusted_ratios(path, config.risk_free_rate,
                                            config.periods_per_year).iloc[0]
    var = calculate_var_summary(path, config.confidence_level, config.min_observations).iloc[0]
    cvar = calculate_cvar_summary(path, config.confidence_level, config.min_observations).iloc[0]
    drawdown = calculate_maximum_drawdown(path).iloc[0]
    cumulative = float(path["close"].iloc[-1] - 1)
    return pd.Series({
        "Cumulative Return": cumulative,
        "Annualized Return (arithmetic)": float(r.mean() * config.periods_per_year),
        "Annualized Return (geometric)": float(ratios["annualized_return"]),
        "Annualized Volatility": float(ratios["annualized_volatility"]),
        "Sharpe": float(ratios["sharpe_ratio"]),
        "Sortino": float(ratios["sortino_ratio"]),
        "Maximum Drawdown": float(drawdown["maximum_drawdown"]),
        "Historical VaR (1-day)": float(var["historical_var"]),
        "Historical CVaR (1-day)": float(cvar["historical_cvar"]),
        "Observations": float(n),
        "Rebalances": math.nan if rebalances is None else float(rebalances),
    }, index=METRIC_ROWS)


def _value_frame(returns):
    """A one-'symbol' price-like frame of the value path (base row at value 1.0)."""
    values = np.concatenate([[1.0], np.cumprod(1 + returns.to_numpy())])
    dates = pd.bdate_range("2000-01-03", periods=len(values))   # order only; never displayed
    return pd.DataFrame({"date": dates, "symbol": "PORTFOLIO", "close": values})


def _drawdowns(equity):
    frames = []
    for label in equity.columns.drop("date"):
        series = calculate_drawdown_series(pd.DataFrame({"date": equity["date"], "symbol": label,
                                                         "close": equity[label]}))
        frames.append(series.set_index("date")["drawdown"].rename(label))
    return pd.concat(frames, axis=1).reset_index()


def _index_returns(index_data, index_name, calendar):
    """Index returns on stock calendar days whose previous index date is the previous
    stock calendar date (same period); other days are left out."""
    series, _ = prepare_index_series(index_data, index_name)
    returns = calculate_daily_returns(series.rename(columns={"index_name": "symbol"}))
    returns["previous_date"] = returns["date"].shift(1)
    usable = returns[np.isfinite(returns[CANONICAL_DAILY_RETURN])
                     & (returns["validation_status"] != "INVALID")]
    previous = dict(zip(calendar[1:], calendar[:-1]))
    same = usable[[previous.get(d) == p for d, p in zip(usable["date"], usable["previous_date"])]]
    return same.set_index("date")[CANONICAL_DAILY_RETURN]


def _universe(data, config):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Market data must be a DataFrame")
    require_canonical_columns(data, "a backtest")
    if data.empty:
        raise ValueError("Market data is empty")
    missing = sorted(set(config.symbols) - set(data["symbol"].dropna()))
    if missing:
        raise ValueError(f"Selected symbol(s) not found in the market data: {', '.join(missing)}")
    universe = data[data["symbol"].isin(config.symbols)]
    calendar = sorted(pd.to_datetime(universe["date"].dropna()).unique())
    return universe, [pd.Timestamp(d) for d in calendar]


def _rows(universe, dates):
    return universe[universe["date"].isin(dates)]


def _log_frame(rebalances):
    rows = [{"window": r.window, "rebalance_date": r.rebalance_date,
             "training_start": r.training_start, "training_end": r.training_end,
             "test_start": r.test_start, "test_end": r.test_end,
             "strategy": STRATEGY_LABELS[r.strategy], "status": r.status, "reason": r.reason,
             "weights": r.weights, "training_observations": r.training_observations,
             "solver_status": r.solver_status, "evaluated_days": r.evaluated_days,
             "excluded_days": r.excluded_days, "parameters": r.parameters} for r in rebalances]
    return (pd.DataFrame(rows, columns=LOG_COLUMNS)
            .sort_values(["window", "strategy"], kind="stable").reset_index(drop=True))


def _whole(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be a whole number >= {minimum}, got {value!r}")
