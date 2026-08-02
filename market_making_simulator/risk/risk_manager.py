"""
RiskManager: position throttling and a drawdown kill-switch.

The kill-switch fires on drawdown from the running peak of mark-to-market PnL.
The previous version fired on net cash flow, which meant that buying 10 units
of a $100 asset registered as a $1,000 loss and tripped the switch with zero
actual loss.
"""

from typing import Optional, Tuple


class RiskManager:
    """Inventory-based size throttling and a mark-to-market drawdown stop."""

    def __init__(
        self,
        enable_kill_switch: bool = False,
        drawdown_limit: Optional[float] = None,
        enable_size_throttle: bool = True,
        min_throttle: float = 0.2,
    ) -> None:
        """
        Args:
            enable_kill_switch: Stop quoting once the drawdown limit is hit
            drawdown_limit: Maximum tolerated fall from the peak mark-to-market
                PnL, as a positive number
            enable_size_throttle: Shrink quoted size as inventory approaches
                the position limit
            min_throttle: Floor on the throttle multiplier
        """
        if not 0.0 <= min_throttle <= 1.0:
            raise ValueError("min_throttle must be in [0, 1]")

        self.enable_kill_switch = enable_kill_switch
        self.drawdown_limit = drawdown_limit
        self.enable_size_throttle = enable_size_throttle
        self.min_throttle = min_throttle

        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.is_halted = False

    def reset(self):
        """Clear the peak, drawdown and halted state."""
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.is_halted = False

    def update(self, mark_to_market_pnl: float) -> float:
        """
        Observe the current PnL, update the running peak, and arm the stop.

        Called once per step by the simulator, after the price has moved, with
        the same mark-to-market series the summary reports. That is deliberate:
        when the risk manager watched a different series from the one printed in
        the waterfall, the summary showed a drawdown that was not the number the
        kill-switch had actually tested.

        Args:
            mark_to_market_pnl: PnL marked at mid, including rebates. Not cash
                flow. Buying 10 units at $100 is a $1,000 cash outflow and a
                zero loss, and the previous version stopped trading on it.

        Returns:
            Drawdown from the peak, as a non-negative number.
        """
        if mark_to_market_pnl > self.peak_pnl:
            self.peak_pnl = mark_to_market_pnl
        drawdown = self.peak_pnl - mark_to_market_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        if self.enable_kill_switch and self.drawdown_limit is not None:
            if drawdown >= abs(self.drawdown_limit):
                # Once halted, stay halted. A stop that un-trips on the next
                # favourable tick is not a stop.
                self.is_halted = True

        return drawdown

    def apply(
        self,
        bid_price: float,
        bid_size: float,
        ask_price: float,
        ask_size: float,
        inventory: float,
        max_inventory: float,
    ) -> Tuple[float, float, float, float, bool]:
        """
        Apply risk controls to a pair of quotes.

        Pure: reads `is_halted` but never writes it. All state changes happen in
        `update`, which the simulator calls once per step. Keeping this pure is
        what makes `MarketMaker.get_quotes` idempotent; when the two were fused,
        quoting twice at the same state double-counted into the drawdown.

        The size throttle is deliberately symmetric. It derives one scale from
        abs(inventory) / max_inventory and applies it to both sides, so at
        inventory -100 the position-reducing bid is shrunk exactly as hard as
        the position-growing ask. That looks wrong at first glance, and the
        obvious alternative is to throttle only the growing side. It is not what
        this quoter does, because direction is already handled upstream: the
        price lean skews both quotes toward the side that flattens us, which is
        what actually recruits the offsetting flow. The throttle answers a
        different question, how much gross exposure to put at risk per step near
        the limit, and near the limit the answer is less on both sides. Leaving
        the reducing side at full size would widen gross quoted exposure exactly
        where a single adverse fill hurts most, since a reducing quote is only
        reducing until the position crosses through flat.

        Returns:
            (bid_price, bid_size, ask_price, ask_size, is_active)
        """
        if self.enable_kill_switch and self.is_halted:
            return (bid_price, 0.0, ask_price, 0.0, False)

        if self.enable_size_throttle and max_inventory > 0:
            inv_ratio = min(1.0, abs(inventory) / max_inventory)
            scale = max(self.min_throttle, 1.0 - inv_ratio)
            bid_size *= scale
            ask_size *= scale

        return (bid_price, bid_size, ask_price, ask_size, True)

    def __repr__(self) -> str:
        return (f"RiskManager(kill_switch={self.enable_kill_switch}, "
                f"drawdown_limit={self.drawdown_limit}, "
                f"halted={self.is_halted})")
