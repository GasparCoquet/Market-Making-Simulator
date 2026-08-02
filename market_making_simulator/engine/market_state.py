"""
MarketState: the exogenous market the strategy quotes into.

This is deliberately NOT a limit order book. There is no queue, no per-level
depth, no price-time priority and no impact from our own trades. The market is
a mid-price process plus a reference half-spread, and our fills are governed by
the arrival-intensity model in `fill_model.py`.

That is the Avellaneda-Stoikov (2008) setup, and it is the honest description
of what this code does. See the Limitations section of the README.
"""


class MarketState:
    """
    Exogenous market: a mid price and a reference half-spread.

    The reference half-spread is the market's own quoted spread. It is used to
    charge a realistic cost when liquidating residual inventory at the end of a
    run, and as a scale for reporting. It does not constrain our own quotes.
    """

    def __init__(
        self,
        initial_mid: float = 100.0,
        reference_half_spread: float = 0.10,
    ):
        """
        Args:
            initial_mid: Starting mid price
            reference_half_spread: The market's own half-spread, used to price
                the terminal liquidation of residual inventory
        """
        if initial_mid <= 0:
            raise ValueError("initial_mid must be positive")
        if reference_half_spread < 0:
            raise ValueError("reference_half_spread must be non-negative")

        self.initial_mid = initial_mid
        self.mid_price = initial_mid
        self.reference_half_spread = reference_half_spread

    def get_mid_price(self) -> float:
        """Current mid price."""
        return self.mid_price

    def get_reference_bid(self) -> float:
        """The market's own best bid (not ours)."""
        return self.mid_price - self.reference_half_spread

    def get_reference_ask(self) -> float:
        """The market's own best ask (not ours)."""
        return self.mid_price + self.reference_half_spread

    def update_mid_price(self, new_mid: float):
        """Move the mid price."""
        self.mid_price = new_mid

    def liquidation_cost(self, inventory: float) -> float:
        """
        Cost of flattening `inventory` by crossing the market's spread.

        Always non-negative. Charged once, at the end of a run, so that a
        strategy cannot look profitable purely by carrying an open position
        marked at mid.
        """
        return abs(inventory) * self.reference_half_spread

    def __repr__(self) -> str:
        return (f"MarketState(mid={self.mid_price:.2f}, "
                f"ref_half_spread={self.reference_half_spread:.4f})")
