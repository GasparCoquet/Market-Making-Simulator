"""
Analytics layer: PnL tracking and visualization.

Handles performance metrics, decomposition, and reporting.
"""

from .pnl_tracker import PnLTracker
from .plotter import SimulationPlotter

__all__ = ["PnLTracker", "SimulationPlotter"]
