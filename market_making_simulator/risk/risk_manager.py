from typing import Optional, Tuple


class RiskManager:
    """Simple risk manager for inventory limits, size throttling, and kill-switch."""

    def __init__(
        self,
        enable_kill_switch: bool = False,
        drawdown_limit: Optional[float] = None,
        enable_size_throttle: bool = True,
        min_throttle: float = 0.2,
    ) -> None:
        self.enable_kill_switch = enable_kill_switch
        self.drawdown_limit = drawdown_limit
        self.enable_size_throttle = enable_size_throttle
        self.min_throttle = min_throttle

    def apply(
        self,
        bid_price: float,
        bid_size: float,
        ask_price: float,
        ask_size: float,
        inventory: float,
        max_inventory: float,
        cash_pnl: float,
    ) -> Tuple[float, float, float, float, bool]:
        """
        Apply risk controls to quotes.

        Returns adjusted (bid_price, bid_size, ask_price, ask_size, is_active).
        """
        is_active = True

        if self.enable_kill_switch and self.drawdown_limit is not None:
            if cash_pnl <= -abs(self.drawdown_limit):
                return (bid_price, 0.0, ask_price, 0.0, False)

        if self.enable_size_throttle and max_inventory > 0:
            inv_ratio = min(1.0, abs(inventory) / max_inventory)
            scale = max(self.min_throttle, 1.0 - inv_ratio)
            bid_size *= scale
            ask_size *= scale

        return (bid_price, bid_size, ask_price, ask_size, is_active)
