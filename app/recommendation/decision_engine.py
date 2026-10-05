"""Combine event intelligence, market behaviour and portfolio exposure into risk signals.

This is the only place where intelligence, analytics and portfolio information
meet. It depends on the intelligence and analytics domains; neither of them
depends on it.
"""


def portfolio_exposure(symbol, holdings):
    """Share (0-1) of the portfolio's market value held in ``symbol``.

    ``holdings`` maps symbol -> current market value. Returns 0.0 for symbols
    not held or an empty portfolio.
    """
    total = sum(holdings.values())
    if total <= 0:
        return 0.0
    return holdings.get(symbol.strip().upper(), 0.0) / total


def generate_risk_signal(event, impact, market_reaction, portfolio_weight, generated_time):
    """Build a ``RiskSignal`` from an assessed event and its context.

    Planned inputs:
      event            - assessed ``MarketEvent`` (severity, confidence, status)
      impact           - ``EventImpact`` with direction and scenarios
      market_reaction  - ``MarketReaction`` (price and liquidity behaviour)
      portfolio_weight - result of ``portfolio_exposure``
    """
    raise NotImplementedError("Risk signal generation is not implemented yet")
