"""
Market-Making Simulator

A simulator for studying market microstructure, liquidity provision,
and market-making strategies with detailed PnL decomposition.
"""

from .order_book import OrderBook
from .market_maker import MarketMaker
from .pnl_tracker import PnLTracker
from .simulator import MarketSimulator
from .plotter import SimulationPlotter

__version__ = "0.1.0"
__all__ = ["OrderBook", "MarketMaker", "PnLTracker", "MarketSimulator", "SimulationPlotter"]
