"""
Market-Making Simulator

A market-making simulator built on an Avellaneda-Stoikov style arrival
intensity: quoting further from the mid genuinely reduces the fill rate, and
PnL decomposes exactly into spread capture plus inventory PnL.
"""

from .engine.market_state import MarketState
from .engine.fill_model import FillModel
from .engine.simulator import MarketSimulator
from .strategy.market_maker import MarketMaker
from .analytics.pnl_tracker import PnLTracker
from .analytics.plotter import SimulationPlotter
from .risk.risk_manager import RiskManager
from .units import per_step_volatility, annualised_volatility

__version__ = "0.2.0"
__all__ = [
    "MarketState",
    "FillModel",
    "MarketMaker",
    "PnLTracker",
    "MarketSimulator",
    "SimulationPlotter",
    "RiskManager",
    "per_step_volatility",
    "annualised_volatility",
]
