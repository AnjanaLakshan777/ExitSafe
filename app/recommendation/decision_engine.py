"""Combine event intelligence, market behaviour and portfolio exposure into risk signals."""


def portfolio_exposure(symbol, holdings):
    """Share of the portfolio's market value held in symbol (0 if it isn't held)."""
    total = sum(holdings.values())
    if total <= 0:
        return 0.0
    return holdings.get(symbol.strip().upper(), 0.0) / total


def generate_risk_signal(event, impact, market_reaction, portfolio_weight, generated_time):
    """Build a RiskSignal from an assessed event and its market and portfolio context."""
    raise NotImplementedError("Risk signal generation is not implemented yet")
