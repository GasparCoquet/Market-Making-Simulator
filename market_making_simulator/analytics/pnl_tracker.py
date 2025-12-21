"""
PnLTracker implementation.

Decomposes PnL into spread capture, inventory risk, and adverse selection.
"""

from typing import Dict, List, Tuple
import numpy as np


class PnLTracker:
    """
    Tracks and decomposes PnL into different components.
    
    Components:
    - Spread capture: PnL from bid-ask spread
    - Inventory risk: PnL from holding inventory during price moves
    - Adverse selection: PnL loss from trading with informed traders
    """
    
    def __init__(self):
        """Initialize the PnL tracker."""
        self.trades: List[Dict] = []
        self.inventory_snapshots: List[Tuple[float, float, float]] = []  # (time, inventory, mid_price)
        
    def record_trade(
        self,
        timestamp: float,
        side: str,
        price: float,
        quantity: float,
        mid_price: float
    ):
        """
        Record a trade.
        
        Args:
            timestamp: Time of trade
            side: 'buy' or 'sell'
            price: Execution price
            quantity: Trade quantity
            mid_price: Mid price at time of trade
        """
        self.trades.append({
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'quantity': quantity,
            'mid_price': mid_price,
            'spread_vs_mid': price - mid_price if side == 'sell' else mid_price - price
        })
    
    def record_inventory_snapshot(
        self,
        timestamp: float,
        inventory: float,
        mid_price: float
    ):
        """
        Record inventory state at a point in time.
        
        Args:
            timestamp: Time of snapshot
            inventory: Current inventory
            mid_price: Current mid price
        """
        self.inventory_snapshots.append((timestamp, inventory, mid_price))
    
    def get_spread_capture(self) -> float:
        """
        Calculate PnL from spread capture.
        
        This is the theoretical PnL from buying at bid and selling at ask.
        
        Returns:
            Spread capture PnL
        """
        spread_pnl = sum(trade['spread_vs_mid'] * trade['quantity'] for trade in self.trades)
        return spread_pnl
    
    def get_inventory_pnl(self) -> float:
        """
        Calculate PnL from inventory risk (price moves while holding inventory).
        
        Returns:
            Inventory risk PnL
        """
        if len(self.inventory_snapshots) < 2:
            return 0.0
        
        inventory_pnl = 0.0
        for i in range(1, len(self.inventory_snapshots)):
            prev_time, prev_inv, prev_mid = self.inventory_snapshots[i-1]
            curr_time, curr_inv, curr_mid = self.inventory_snapshots[i]
            
            # PnL from price movement on previous inventory
            price_change = curr_mid - prev_mid
            inventory_pnl += prev_inv * price_change
            
        return inventory_pnl
    
    def get_adverse_selection_cost(self) -> float:
        """
        Estimate adverse selection cost.
        
        Adverse selection occurs when we trade with informed traders who know
        the price is about to move. This is estimated by looking at price moves
        after our trades.
        
        Returns:
            Estimated adverse selection cost
        """
        if len(self.trades) < 2:
            return 0.0
        
        adverse_selection = 0.0
        
        for i in range(len(self.trades) - 1):
            trade = self.trades[i]
            next_trade = self.trades[i + 1]
            
            # Look at price move after trade
            price_move = next_trade['mid_price'] - trade['mid_price']
            
            # If we bought and price went down, or sold and price went up,
            # this is adverse selection (we traded with informed traders)
            if trade['side'] == 'buy' and price_move < 0:
                adverse_selection += abs(price_move) * trade['quantity']
            elif trade['side'] == 'sell' and price_move > 0:
                adverse_selection += abs(price_move) * trade['quantity']
                
        return -adverse_selection  # Negative because it's a cost
    
    def get_pnl_decomposition(self) -> Dict[str, float]:
        """
        Get complete PnL decomposition.
        
        Returns:
            Dictionary with PnL components
        """
        spread_capture = self.get_spread_capture()
        inventory_pnl = self.get_inventory_pnl()
        adverse_selection = self.get_adverse_selection_cost()
        
        return {
            'spread_capture': spread_capture,
            'inventory_pnl': inventory_pnl,
            'adverse_selection': adverse_selection,
            'total_pnl': spread_capture + inventory_pnl + adverse_selection
        }
    
    def get_trade_count(self) -> Tuple[int, int]:
        """
        Get count of buy and sell trades.
        
        Returns:
            Tuple of (num_buys, num_sells)
        """
        num_buys = sum(1 for t in self.trades if t['side'] == 'buy')
        num_sells = sum(1 for t in self.trades if t['side'] == 'sell')
        return (num_buys, num_sells)
    
    def __repr__(self) -> str:
        decomp = self.get_pnl_decomposition()
        return (f"PnLTracker(spread={decomp['spread_capture']:.2f}, "
                f"inventory={decomp['inventory_pnl']:.2f}, "
                f"adverse_selection={decomp['adverse_selection']:.2f}, "
                f"total={decomp['total_pnl']:.2f})")
