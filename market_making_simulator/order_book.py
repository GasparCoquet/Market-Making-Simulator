"""
OrderBook implementation for the market-making simulator.

Simulates a limit order book with configurable mid price, spread, and depth.
"""

from typing import Tuple, Optional
import numpy as np


class OrderBook:
    """
    Simulated limit order book.
    
    Tracks mid price, bid/ask spread, and depth at different levels.
    """
    
    def __init__(
        self,
        initial_mid: float = 100.0,
        spread: float = 0.10,
        depth_per_level: float = 100.0,
        num_levels: int = 5
    ):
        """
        Initialize the order book.
        
        Args:
            initial_mid: Initial mid price
            spread: Half-spread (distance from mid to bid or ask)
            depth_per_level: Liquidity available at each level
            num_levels: Number of price levels on each side
        """
        self.mid_price = initial_mid
        self.spread = spread
        self.depth_per_level = depth_per_level
        self.num_levels = num_levels
        
    def get_mid_price(self) -> float:
        """Get current mid price."""
        return self.mid_price
    
    def get_best_bid(self) -> float:
        """Get best bid price."""
        return self.mid_price - self.spread
    
    def get_best_ask(self) -> float:
        """Get best ask price."""
        return self.mid_price + self.spread
    
    def get_bid_ask_spread(self) -> float:
        """Get full bid-ask spread."""
        return 2 * self.spread
    
    def get_depth_at_level(self, level: int = 0) -> Tuple[float, float]:
        """
        Get available depth at a specific level.
        
        Args:
            level: Price level (0 = best bid/ask)
            
        Returns:
            Tuple of (bid_depth, ask_depth)
        """
        if level >= self.num_levels:
            return (0.0, 0.0)
        return (self.depth_per_level, self.depth_per_level)
    
    def update_mid_price(self, new_mid: float):
        """Update the mid price (simulates market movement)."""
        self.mid_price = new_mid
    
    def execute_market_buy(self, quantity: float) -> Tuple[float, float]:
        """
        Simulate a market buy order hitting the ask.
        
        Args:
            quantity: Size of the buy order
            
        Returns:
            Tuple of (execution_price, executed_quantity)
        """
        available_depth = self.depth_per_level
        executed_qty = min(quantity, available_depth)
        execution_price = self.get_best_ask()
        
        # Simulate price impact for large orders
        if quantity > available_depth:
            # Price moves up if liquidity is exhausted
            price_impact = (quantity / available_depth - 1) * self.spread * 0.5
            self.mid_price += price_impact
            
        return (execution_price, executed_qty)
    
    def execute_market_sell(self, quantity: float) -> Tuple[float, float]:
        """
        Simulate a market sell order hitting the bid.
        
        Args:
            quantity: Size of the sell order
            
        Returns:
            Tuple of (execution_price, executed_quantity)
        """
        available_depth = self.depth_per_level
        executed_qty = min(quantity, available_depth)
        execution_price = self.get_best_bid()
        
        # Simulate price impact for large orders
        if quantity > available_depth:
            # Price moves down if liquidity is exhausted
            price_impact = (quantity / available_depth - 1) * self.spread * 0.5
            self.mid_price -= price_impact
            
        return (execution_price, executed_qty)
    
    def __repr__(self) -> str:
        return (f"OrderBook(mid={self.mid_price:.2f}, "
                f"bid={self.get_best_bid():.2f}, "
                f"ask={self.get_best_ask():.2f}, "
                f"spread={self.get_bid_ask_spread():.4f})")
