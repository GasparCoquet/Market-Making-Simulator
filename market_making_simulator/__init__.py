"""
Market-Making Simulator

A market-making simulator built on an Avellaneda-Stoikov style arrival
intensity: quoting further from the mid genuinely reduces the fill rate, and
PnL decomposes exactly into spread capture plus inventory PnL.

One engine, two calibrations. `datasets.US_EQUITY` is a cash equity with a
session close and a per-share maker rebate; `datasets.CRYPTO_PERP` is a
perpetual swap that never closes, pays hourly funding, and is charged in basis
points of notional. Both are held identical in every dimensionless quantity, so
the difference between them is market structure and nothing else.
"""

from .engine.market_state import MarketState
from .engine.fill_model import FillModel
from .engine.funding import FundingModel
from .engine.simulator import MarketSimulator
from .strategy.market_maker import MarketMaker
from .analytics.pnl_tracker import PnLTracker
from .analytics.plotter import SimulationPlotter
from .risk.risk_manager import RiskManager
from .datasets import (
    CRYPTO_PERP,
    DATASET_NAMES,
    DATASETS,
    US_EQUITY,
    Dataset,
    get_dataset,
)
from .units import (
    SECONDS_PER_CALENDAR_YEAR,
    SECONDS_PER_TRADING_YEAR,
    annualised_volatility,
    per_step_volatility,
)

__version__ = "0.3.0"
__all__ = [
    "MarketState",
    "FillModel",
    "FundingModel",
    "MarketMaker",
    "PnLTracker",
    "MarketSimulator",
    "SimulationPlotter",
    "RiskManager",
    "Dataset",
    "DATASETS",
    "DATASET_NAMES",
    "US_EQUITY",
    "CRYPTO_PERP",
    "get_dataset",
    "per_step_volatility",
    "annualised_volatility",
    "SECONDS_PER_TRADING_YEAR",
    "SECONDS_PER_CALENDAR_YEAR",
]
