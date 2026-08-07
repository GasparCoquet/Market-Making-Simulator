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

Trades carry a `kind`. A `quote` fill is passive flow arriving at a price we
showed. A `session_close` is us crossing the market to go flat at an equity
session end, which a 24/7 book never does. Both are real trades and both are in
`spread_capture`, because the identity is about cash and inventory and does not
care why a trade happened. They are separated everywhere the distinction
changes the meaning of a number: the fill counters describe quoting activity,
and the markout measures whether *our quotes* were picked off, which a trade we
initiated ourselves cannot answer.
"""

from typing import Dict, List, Optional, Tuple

DEFAULT_MARKOUT_HORIZON = 5

# A fill that arrived at a price we quoted.
KIND_QUOTE = 'quote'
# A trade we initiated to flatten at a session close, crossing the market.
KIND_SESSION_CLOSE = 'session_close'
TRADE_KINDS = (KIND_QUOTE, KIND_SESSION_CLOSE)


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
        kind: str = KIND_QUOTE,
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
            kind: 'quote' for passive flow arriving at a price we showed, or
                'session_close' for a trade we initiated to go flat at an
                equity session end. Defaults to 'quote'.
        """
        if side not in ('buy', 'sell'):
            raise ValueError("side must be 'buy' or 'sell'")
        if kind not in TRADE_KINDS:
            raise ValueError(f"kind must be one of {TRADE_KINDS}")

        signed_quantity = quantity if side == 'buy' else -quantity
        self.trades.append({
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'quantity': quantity,
            'signed_quantity': signed_quantity,
            'mid_price': mid_price,
            'step_index': step_index,
            'kind': kind,
            # Edge earned against the mid. Positive when we bought below the
            # mid or sold above it, negative when our skew pushed the quote
            # through the mid, and always negative on a session close because
            # crossing the market means paying its half-spread.
            'edge': signed_quantity * (mid_price - price),
        })

    def record_session_close(
        self,
        timestamp: float,
        signed_quantity: float,
        price: float,
        mid_price: float,
        step_index: int,
    ):
        """
        Record the trade that flattened the book at a session close.

        A thin wrapper over `record_trade` so that callers never have to name
        the trade kind, and so the side is derived from the sign rather than
        passed in and possibly inverted.

        Args:
            timestamp: Time of the close-out
            signed_quantity: Signed quantity traded, negative when we sold a
                long position down to flat
            price: Fill price, already inclusive of the crossed half-spread
            mid_price: Mid at the moment of the close-out
            step_index: Step the markout would measure from, were session
                closes included in it. They are not.
        """
        if signed_quantity == 0.0:
            return
        self.record_trade(
            timestamp,
            'buy' if signed_quantity > 0 else 'sell',
            price,
            abs(signed_quantity),
            mid_price,
            step_index,
            kind=KIND_SESSION_CLOSE,
        )

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

    def _quoted_trades(self) -> List[Dict]:
        """Passive fills only, excluding trades we initiated ourselves."""
        return [t for t in self.trades if t['kind'] == KIND_QUOTE]

    def get_spread_capture(self) -> float:
        """
        Edge earned against the mid on every trade, quoted or not.

        Session closes are included because this is a leg of the PnL identity,
        and leaving out a real trade would break it. `get_quoted_edge` and
        `get_session_close_cost` split it when the distinction matters.
        """
        return sum(trade['edge'] for trade in self.trades)

    def get_quoted_edge(self) -> float:
        """Spread capture from passive fills only. A split of spread capture."""
        return sum(trade['edge'] for trade in self._quoted_trades())

    def get_session_close_cost(self) -> float:
        """
        What flattening at session closes cost, as a non-negative number.

        A split of spread capture, not an extra bucket: it is already inside
        `get_spread_capture`, so adding it to the waterfall would double count.
        Zero for a 24/7 book, which never has a close to flatten into.
        """
        return -sum(trade['edge'] for trade in self.trades
                    if trade['kind'] == KIND_SESSION_CLOSE)

    def get_session_close_count(self) -> int:
        """Number of session closes that actually had a position to flatten."""
        return sum(1 for t in self.trades if t['kind'] == KIND_SESSION_CLOSE)

    def get_session_close_volume(self) -> float:
        """Total quantity traded to flatten at session closes."""
        return sum(t['quantity'] for t in self.trades
                   if t['kind'] == KIND_SESSION_CLOSE)

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

        Measured over quoted fills only. Adverse selection is the question "was
        the flow that hit our quote informed", and a trade we initiated at a
        session close carries no information about that. Including it would put
        a forced, calendar-driven trade into a metric read as a property of the
        counterparties.
        """
        quoted = self._quoted_trades()
        if not quoted or not self.steps:
            return 0.0

        h = self.markout_horizon if horizon is None else horizon
        if h < 1:
            raise ValueError("horizon must be at least 1")

        last_index = len(self.steps) - 1
        markout = 0.0
        for trade in quoted:
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
        `adverse_selection`, `quoted_edge` and `session_close_cost` are
        diagnostics and are deliberately not part of that sum: the first is a
        split of the inventory term and the other two are a split of spread
        capture, so adding any of them would double count.
        """
        spread_capture = self.get_spread_capture()
        inventory_pnl = self.get_inventory_pnl()
        return {
            'spread_capture': spread_capture,
            'inventory_pnl': inventory_pnl,
            'gross_pnl': spread_capture + inventory_pnl,
            'adverse_selection': self.get_adverse_selection(),
            'quoted_edge': self.get_quoted_edge(),
            'session_close_cost': self.get_session_close_cost(),
        }

    def get_trade_count(self) -> Tuple[int, int]:
        """
        (number of quoted buys, number of quoted sells).

        Quoted fills only. This counter describes how much passive flow we
        attracted, and folding in trades we initiated ourselves would overstate
        it by whatever the calendar happened to force.
        """
        quoted = self._quoted_trades()
        num_buys = sum(1 for t in quoted if t['side'] == 'buy')
        num_sells = sum(1 for t in quoted if t['side'] == 'sell')
        return (num_buys, num_sells)

    def get_filled_volume(self) -> float:
        """Total quantity filled on quoted fills, across both sides."""
        return sum(t['quantity'] for t in self._quoted_trades())

    def __repr__(self) -> str:
        decomp = self.get_pnl_decomposition()
        return (f"PnLTracker(spread={decomp['spread_capture']:.2f}, "
                f"inventory={decomp['inventory_pnl']:.2f}, "
                f"gross={decomp['gross_pnl']:.2f})")
