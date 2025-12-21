"""
Strategy layer: Market-making decision logic.

Handles quote generation, position management, and fill execution.
"""

from .market_maker import MarketMaker

__all__ = ["MarketMaker"]
