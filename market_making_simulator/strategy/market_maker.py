"""
MarketMaker: quoting, inventory and position accounting.

Quotes are placed symmetrically around a *reservation price* that leans against
inventory, not around the mid. This is the Ho-Stoll (1981) / Avellaneda-Stoikov
(2008) rule:

    r   = mid - inventory * gamma
    bid = r - quote_spread
    ask = r + quote_spread

Both quotes shift together, so the quoted width is always exactly
2 * quote_spread and the quotes can never cross. When we are long, both quotes
move down: the ask becomes easier to lift (we sell) and the bid harder to hit
(we stop buying). That is what "lean against the wind" means. Widening the ask
when long, which is the intuitive-sounding rule, does the opposite of what is
wanted, because a wider ask is *less* likely to be lifted.
"""

from typing import Tuple, Optional, TYPE_CHECKING
from ..risk import RiskManager

if TYPE_CHECKING:
    from ..engine import MarketState


class MarketMaker:
    """Two-sided quoter with inventory-linked reservation price."""

    def __init__(
        self,
        quote_spread: float = 0.05,
        quote_size: float = 10.0,
        max_inventory: float = 100.0,
        inventory_skew_factor: float = 0.01,
        maker_rebate_per_unit: float = 0.0,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        Args:
            quote_spread: Half-width of our quote, in price units either side
                of the reservation price
            quote_size: Size quoted on each side
            max_inventory: Hard position limit; we stop quoting the side that
                would breach it
            inventory_skew_factor: gamma, the price units by which the
                reservation price shifts per unit of inventory
            maker_rebate_per_unit: Exchange rebate received per unit filled.
                Negative values represent a fee.
            risk_manager: Optional risk overlay applied to the quotes
        """
        if quote_spread < 0:
            raise ValueError("quote_spread must be non-negative")
        if quote_size < 0:
            raise ValueError("quote_size must be non-negative")
        if max_inventory < 0:
            raise ValueError("max_inventory must be non-negative")

        self.quote_spread = quote_spread
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.inventory_skew_factor = inventory_skew_factor
        self.maker_rebate_per_unit = maker_rebate_per_unit
        self.risk_manager = risk_manager

        self.inventory = 0.0
        # Trading cash only. Rebates are held separately so that the identity
        # gross_pnl == spread_capture + inventory_pnl stays exact.
        self.cash = 0.0
        self.rebates = 0.0

        self.total_buy_value = 0.0
        self.total_sell_value = 0.0
        self.total_bought = 0.0
        self.total_sold = 0.0

    def get_inventory(self) -> float:
        """Current signed position."""
        return self.inventory

    def get_reservation_price(self, mid: float) -> float:
        """Inventory-adjusted price our quotes are centred on."""
        return mid - self.inventory * self.inventory_skew_factor

    def get_quotes(self, market_state: "MarketState") -> Tuple[float, float, float, float]:
        """
        Generate quotes.

        Returns:
            (bid_price, bid_size, ask_price, ask_size). The invariant
            ask_price > bid_price holds for every inventory level whenever
            quote_spread > 0.
        """
        mid = market_state.get_mid_price()
        reservation = self.get_reservation_price(mid)

        bid_price = reservation - self.quote_spread
        ask_price = reservation + self.quote_spread

        # Buying increases inventory, selling decreases it. Stop quoting the
        # side that would breach the limit.
        bid_size = self.quote_size if self.inventory + self.quote_size <= self.max_inventory else 0.0
        ask_size = self.quote_size if self.inventory - self.quote_size >= -self.max_inventory else 0.0

        if self.risk_manager is not None:
            bid_price, bid_size, ask_price, ask_size, _ = self.risk_manager.apply(
                bid_price,
                bid_size,
                ask_price,
                ask_size,
                self.inventory,
                self.max_inventory,
            )

        return (bid_price, bid_size, ask_price, ask_size)

    def observe_risk(self, current_mid: float) -> None:
        """
        Let the risk overlay see the current mark-to-market PnL.

        Separate from `get_quotes` so that quoting stays free of side effects.
        The simulator calls this once per step, after the price move.
        """
        if self.risk_manager is not None:
            self.risk_manager.update(self.get_mark_to_market_pnl(current_mid))

    def execute_bid_fill(self, price: float, quantity: float):
        """Our bid was hit, so we bought `quantity` at `price`."""
        if quantity <= 0:
            return
        self.inventory += quantity
        self.cash -= price * quantity
        self.rebates += self.maker_rebate_per_unit * quantity
        self.total_buy_value += price * quantity
        self.total_bought += quantity

    def execute_ask_fill(self, price: float, quantity: float):
        """Our ask was lifted, so we sold `quantity` at `price`."""
        if quantity <= 0:
            return
        self.inventory -= quantity
        self.cash += price * quantity
        self.rebates += self.maker_rebate_per_unit * quantity
        self.total_sell_value += price * quantity
        self.total_sold += quantity

    def get_net_cash_flow(self) -> float:
        """
        Net cash from trading, excluding rebates.

        This is NOT a PnL. Being short 30 units of a $100 asset shows +$3000
        here while the position is worth -$3000. Use `get_gross_pnl`.
        """
        return self.cash

    def get_inventory_value(self, current_mid: float) -> float:
        """Mark-to-market value of the open position."""
        return self.inventory * current_mid

    def get_gross_pnl(self, current_mid: float) -> float:
        """
        Mark-to-market PnL before rebates and before liquidating the position.

        Equal to `spread_capture + inventory_pnl` from PnLTracker, exactly.
        """
        return self.cash + self.inventory * current_mid

    def get_mark_to_market_pnl(self, current_mid: float) -> float:
        """Gross PnL plus rebates earned. What the risk overlay sees."""
        return self.get_gross_pnl(current_mid) + self.rebates

    def get_average_buy_price(self) -> Optional[float]:
        """Volume-weighted average price of all buys."""
        if self.total_bought == 0:
            return None
        return self.total_buy_value / self.total_bought

    def get_average_sell_price(self) -> Optional[float]:
        """Volume-weighted average price of all sells."""
        if self.total_sold == 0:
            return None
        return self.total_sell_value / self.total_sold

    def __repr__(self) -> str:
        return (f"MarketMaker(inventory={self.inventory:.2f}, "
                f"cash={self.cash:.2f})")
