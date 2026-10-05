"""Market signals around an event: how is the stock actually trading?

This is the intelligence domain's read-only view of market data. It consumes
DataFrames produced by ``app.data.loaders.market_data.load_market_data`` and
``app.analytics.returns.calculate_daily_returns``; it never loads or cleans
market data itself.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class MarketReaction:
    symbol: str
    as_of: date
    window_days: int
    cumulative_return: float | None   # close-to-close return over the window
    value_traded_ratio: float | None  # window avg Value Traded / prior baseline avg


def measure_market_reaction(market_data, symbol, as_of, window_days=5):
    """Measure price and liquidity behaviour of ``symbol`` in the window ending ``as_of``."""
    raise NotImplementedError("Market reaction measurement is not implemented yet")
