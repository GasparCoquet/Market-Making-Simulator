"""
Market-Making Simulator

A simulator for studying market microstructure, liquidity provision,
and market-making strategies with detailed PnL decomposition.
"""

from .engine.order_book import OrderBook
from .engine.simulator import MarketSimulator
from .strategy.market_maker import MarketMaker
from .analytics.pnl_tracker import PnLTracker
from .analytics.plotter import SimulationPlotter

__version__ = "0.1.0"
__all__ = ["OrderBook", "MarketMaker", "PnLTracker", "MarketSimulator", "SimulationPlotter"]
