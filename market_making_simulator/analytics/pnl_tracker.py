"""
PnLTracker: an exact PnL decomposition, plus adverse selection as a diagnostic.

The decomposition is an identity, not three loosely related quantities that are
added up and hoped to be close:

    gross_pnl = spread_capture + inventory_pnl

where, writing m_t for the mid before the price move at step t and using signed
quantities (positive = we bought):

    spread_capture = sum over fills of  q_signed * (m_t - fill_price)
    inventory_pnl  = sum over steps of  inventory_after_fills(t) * (m_{t+1} - m_t)

Proof: let V = cash + inventory * mid. A fill of signed size q at price p when
the mid is m changes cash by -p*q and inventory by +q, so V changes by
q*(m - p), the spread term. Between fills, V changes by inventory * dm, the
inventory term. V starts at zero, so V_T is exactly the sum of the two. This is
asserted to 1e-9 in the test suite.

Adverse selection is NOT a third additive bucket. It is a *split* of the
inventory term: the part of the price move that happens right after our fills.
It is reported as a signed h-step markout,

    adverse_selection = sum over fills of  q_signed * (m_{t+h} - m_t)

which is negative when we are systematically on the wrong side of the next
move, positive when we are on the right side, and zero in expectation when the
flow is uninformed. The previous version summed only unfavourable moves, so it
was negative by construction whatever the data.
"""

from typing import Dict, List, Optional, Tuple

DEFAULT_MARKOUT_HORIZON = 5


class PnLTracker:
    """Records fills and step-level state, then decomposes PnL exactly."""

    def __init__(self, markout_horizon: int = DEFAULT_MARKOUT_HORIZON):
        """
        Args:
            markout_horizon: Number of steps ahead used to measure adverse
                selection. Trades within `markout_horizon` of the end are
                marked against the final mid.
        """
        if markout_horizon < 1:
            raise ValueError("markout_horizon must be at least 1")
        self.markout_horizon = markout_horizon

        self.trades: List[Dict] = []
        # One entry per step: (time, inventory_after_fills, mid_before, mid_after)
        self.steps: List[Tuple[float, float, float, float]] = []

    def record_trade(
        self,
        timestamp: float,
        side: str,
        price: float,
        quantity: float,
        mid_price: float,
        step_index: int,
    ):
        """
        Record a fill.

        Args:
            timestamp: Time of the fill
            side: 'buy' if our bid was hit, 'sell' if our ask was lifted
            price: Fill price
            quantity: Filled quantity, always positive
            mid_price: Mid at the moment of the fill, before the price move
            step_index: Index of the step the fill happened in, used for markout
        """
        if side not in ('buy', 'sell'):
            raise ValueError("side must be 'buy' or 'sell'")

        signed_quantity = quantity if side == 'buy' else -quantity
        self.trades.append({
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'quantity': quantity,
            'signed_quantity': signed_quantity,
            'mid_price': mid_price,
            'step_index': step_index,
            # Edge earned against the mid. Positive when we bought below the
            # mid or sold above it, negative when our skew pushed the quote
            # through the mid.
            'edge': signed_quantity * (mid_price - price),
        })

    def record_step(
        self,
        timestamp: float,
        inventory_after_fills: float,
        mid_before_move: float,
        mid_after_move: float,
    ):
        """
        Record one step.

        `inventory_after_fills` must be the position held *while* the price
        moves from `mid_before_move` to `mid_after_move`. Snapshotting before
        the fills instead is an off-by-one that breaks the identity.
        """
        self.steps.append(
            (timestamp, inventory_after_fills, mid_before_move, mid_after_move)
        )

    def get_spread_capture(self) -> float:
        """Edge earned against the mid on every fill."""
        return sum(trade['edge'] for trade in self.trades)

    def get_inventory_pnl(self) -> float:
        """PnL from price moves on the position held across each move."""
        return sum(inv * (mid_after - mid_before)
                   for _, inv, mid_before, mid_after in self.steps)

    def get_gross_pnl(self) -> float:
        """Mark-to-market PnL before rebates and liquidation."""
        return self.get_spread_capture() + self.get_inventory_pnl()

    def get_adverse_selection(self, horizon: Optional[int] = None) -> float:
        """
        Signed h-step markout on our fills, a diagnostic split of inventory PnL.

        Negative means the market moved against us right after we traded, which
        is what informed flow looks like. Zero in expectation under uninformed
        flow. Not additive with spread capture and inventory PnL.
        """
        if not self.trades or not self.steps:
            return 0.0

        h = self.markout_horizon if horizon is None else horizon
        if h < 1:
            raise ValueError("horizon must be at least 1")

        last_index = len(self.steps) - 1
        markout = 0.0
        for trade in self.trades:
            start = trade['step_index']
            # mid_before_move of step start+h is the mid exactly h steps after
            # the fill, so it is the right mark whenever that step exists. Only
            # once the horizon runs off the end do we fall back to
            # mid_after_move of the terminal step, the last observable price.
            # Testing `end == last_index` instead would mark a trade landing
            # exactly h steps before the end over h+1 steps, unlike every other
            # trade in the run.
            future_mid = (self.steps[start + h][2] if start + h <= last_index
                          else self.steps[last_index][3])
            markout += trade['signed_quantity'] * (future_mid - trade['mid_price'])
        return markout

    def get_pnl_decomposition(self) -> Dict[str, float]:
        """
        Full decomposition.

        `gross_pnl` is exactly `spread_capture + inventory_pnl`.
        `adverse_selection` is a diagnostic and is deliberately not part of
        that sum.
        """
        spread_capture = self.get_spread_capture()
        inventory_pnl = self.get_inventory_pnl()
        return {
            'spread_capture': spread_capture,
            'inventory_pnl': inventory_pnl,
            'gross_pnl': spread_capture + inventory_pnl,
            'adverse_selection': self.get_adverse_selection(),
        }

    def get_trade_count(self) -> Tuple[int, int]:
        """(number of buys, number of sells)."""
        num_buys = sum(1 for t in self.trades if t['side'] == 'buy')
        num_sells = sum(1 for t in self.trades if t['side'] == 'sell')
        return (num_buys, num_sells)

    def get_filled_volume(self) -> float:
        """Total quantity filled across both sides."""
        return sum(t['quantity'] for t in self.trades)

    def __repr__(self) -> str:
        decomp = self.get_pnl_decomposition()
        return (f"PnLTracker(spread={decomp['spread_capture']:.2f}, "
                f"inventory={decomp['inventory_pnl']:.2f}, "
                f"gross={decomp['gross_pnl']:.2f})")
