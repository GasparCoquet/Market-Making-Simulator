"""
MarketMaker implementation.

Manages quoting, inventory, and order execution for a market-making strategy.
"""

from typing import Tuple, Optional, TYPE_CHECKING
from ..risk import RiskManager

if TYPE_CHECKING:
    from ..engine import OrderBook


class MarketMaker:
    """
    Market maker that quotes bid/ask around the mid price.
    
    Tracks inventory and adjusts quotes based on inventory risk.
    """
    
    def __init__(
        self,
        quote_spread: float = 0.05,
        quote_size: float = 10.0,
        max_inventory: float = 100.0,
        inventory_skew_factor: float = 0.01,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        Initialize the market maker.
        
        Args:
            quote_spread: Half-spread for quotes (distance from mid)
            quote_size: Size to quote on each side
            max_inventory: Maximum inventory (position limit)
            inventory_skew_factor: How much to skew quotes based on inventory
        """
        self.quote_spread = quote_spread
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.inventory_skew_factor = inventory_skew_factor
        self.risk_manager = risk_manager
        
        self.inventory = 0.0
        self.total_buy_value = 0.0
        self.total_sell_value = 0.0
        self.total_bought = 0.0
        self.total_sold = 0.0
        
    def get_inventory(self) -> float:
        """Get current inventory position."""
        return self.inventory
    
    def get_quotes(self, order_book: "OrderBook") -> Tuple[float, float, float, float]:
        """
        Generate bid/ask quotes around the mid price.
        
        Adjusts quotes based on current inventory to manage risk.
        
        Args:
            order_book: Current order book state
            
        Returns:
            Tuple of (bid_price, bid_size, ask_price, ask_size)
        """
        mid = order_book.get_mid_price()
        
        # Skew quotes based on inventory
        # Positive inventory (long) -> widen ask, tighten bid to encourage selling
        # Negative inventory (short) -> widen bid, tighten ask to encourage buying
        inventory_skew = self.inventory * self.inventory_skew_factor
        
        bid_price = mid - self.quote_spread - inventory_skew
        ask_price = mid + self.quote_spread + inventory_skew
        
        bid_size = self.quote_size if abs(self.inventory + self.quote_size) <= self.max_inventory else 0.0
        ask_size = self.quote_size if abs(self.inventory - self.quote_size) <= self.max_inventory else 0.0

        if self.risk_manager is not None:
            bid_price, bid_size, ask_price, ask_size, _ = self.risk_manager.apply(
                bid_price,
                bid_size,
                ask_price,
                ask_size,
                self.inventory,
                self.max_inventory,
                self.get_cash_pnl(),
            )
        
        return (bid_price, bid_size, ask_price, ask_size)
    
    def execute_bid_fill(self, price: float, quantity: float):
        """
        Execute when our bid gets filled (we buy).
        
        Args:
            price: Execution price
            quantity: Execution quantity
        """
        self.inventory += quantity
        self.total_buy_value += price * quantity
        self.total_bought += quantity
    
    def execute_ask_fill(self, price: float, quantity: float):
        """
        Execute when our ask gets filled (we sell).
        
        Args:
            price: Execution price
            quantity: Execution quantity
        """
        self.inventory -= quantity
        self.total_sell_value += price * quantity
        self.total_sold += quantity
    
    def get_cash_pnl(self) -> float:
        """
        Get realized cash PnL (from completed round trips).
        
        Returns:
            Cash PnL
        """
        return self.total_sell_value - self.total_buy_value
    
    def get_total_pnl(self, current_mid: float) -> float:
        """
        Get total PnL including unrealized PnL from inventory.
        
        Args:
            current_mid: Current mid price
            
        Returns:
            Total PnL
        """
        # Net cost basis is what we paid for current inventory
        net_cost = self.total_buy_value - self.total_sell_value
        # Current value of inventory at mid price
        inventory_value = self.inventory * current_mid
        # Unrealized PnL is difference between current value and cost
        unrealized_pnl = inventory_value - net_cost
        return unrealized_pnl
    
    def get_average_buy_price(self) -> Optional[float]:
        """Get average price of all buys."""
        if self.total_bought == 0:
            return None
        return self.total_buy_value / self.total_bought
    
    def get_average_sell_price(self) -> Optional[float]:
        """Get average price of all sells."""
        if self.total_sold == 0:
            return None
        return self.total_sell_value / self.total_sold
    
    def __repr__(self) -> str:
        return (f"MarketMaker(inventory={self.inventory:.2f}, "
                f"cash_pnl={self.get_cash_pnl():.2f})")
