"""
Tests for MarketMaker.

The central invariant here is that our ask is strictly above our bid at every
inventory level. The previous quoting rule (bid = mid - spread - inv*gamma,
ask = mid + spread + inv*gamma) crossed its own quotes whenever
|inventory| * gamma exceeded the spread, which happened on the majority of
steps in a normal run.
"""

import unittest

from market_making_simulator import MarketMaker, MarketState, RiskManager


# Inventory levels swept by the crossing tests, expressed as multiples of the
# position limit. Values beyond +/-1 matter: the quoting rule must stay sane
# even if a caller pushes the position past the limit.
INVENTORY_MULTIPLES = [
    -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1, 0.0,
    0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
]
SKEW_FACTORS = [0.0, 0.001, 0.01, 0.05, 0.2, 1.0]
QUOTE_SPREADS = [0.01, 0.05, 0.25]


class TestQuoteInvariants(unittest.TestCase):
    """Quotes must never cross, and must always be exactly 2*spread wide."""

    def setUp(self):
        self.market_state = MarketState(initial_mid=100.0, reference_half_spread=0.10)
        self.max_inventory = 100.0

    def _maker(self, quote_spread: float, skew: float) -> MarketMaker:
        return MarketMaker(
            quote_spread=quote_spread,
            quote_size=10.0,
            max_inventory=self.max_inventory,
            inventory_skew_factor=skew,
        )

    def test_ask_strictly_above_bid_at_every_inventory_level(self):
        """ask > bid for every inventory, skew and spread combination."""
        for quote_spread in QUOTE_SPREADS:
            for skew in SKEW_FACTORS:
                maker = self._maker(quote_spread, skew)
                for multiple in INVENTORY_MULTIPLES:
                    maker.inventory = multiple * self.max_inventory
                    bid, _, ask, _ = maker.get_quotes(self.market_state)
                    self.assertGreater(
                        ask, bid,
                        msg=(f"crossed quotes at inventory="
                             f"{maker.inventory}, gamma={skew}, "
                             f"spread={quote_spread}: bid={bid}, ask={ask}"),
                    )

    def test_ask_strictly_above_bid_with_risk_overlay(self):
        """The risk overlay resizes quotes, it must not reprice them into a cross."""
        for skew in SKEW_FACTORS:
            maker = MarketMaker(
                quote_spread=0.05,
                quote_size=10.0,
                max_inventory=self.max_inventory,
                inventory_skew_factor=skew,
                risk_manager=RiskManager(enable_size_throttle=True),
            )
            for multiple in INVENTORY_MULTIPLES:
                maker.inventory = multiple * self.max_inventory
                bid, _, ask, _ = maker.get_quotes(self.market_state)
                self.assertGreater(ask, bid)

    def test_quoted_width_is_exactly_twice_the_spread(self):
        """Both quotes shift together, so the width never depends on inventory."""
        for quote_spread in QUOTE_SPREADS:
            for skew in SKEW_FACTORS:
                maker = self._maker(quote_spread, skew)
                for multiple in INVENTORY_MULTIPLES:
                    maker.inventory = multiple * self.max_inventory
                    bid, _, ask, _ = maker.get_quotes(self.market_state)
                    self.assertAlmostEqual(ask - bid, 2.0 * quote_spread, places=12)

    def test_quotes_straddle_the_mid_when_flat(self):
        """With no position the reservation price is the mid."""
        maker = self._maker(0.05, 0.01)
        bid, _, ask, _ = maker.get_quotes(self.market_state)
        self.assertAlmostEqual(bid, 99.95, places=12)
        self.assertAlmostEqual(ask, 100.05, places=12)


class TestReservationPrice(unittest.TestCase):
    """Inventory must lean the quotes against the position, not widen them."""

    def setUp(self):
        self.market_state = MarketState(initial_mid=100.0, reference_half_spread=0.10)
        self.maker = MarketMaker(
            quote_spread=0.05,
            quote_size=10.0,
            max_inventory=100.0,
            inventory_skew_factor=0.01,
        )
        self.flat_bid, _, self.flat_ask, _ = self.maker.get_quotes(self.market_state)

    def test_long_inventory_pushes_both_quotes_down(self):
        """Long: the ask gets easier to lift and the bid harder to hit."""
        for inventory in [1.0, 10.0, 50.0, 100.0]:
            self.maker.inventory = inventory
            bid, _, ask, _ = self.maker.get_quotes(self.market_state)
            self.assertLess(bid, self.flat_bid)
            self.assertLess(ask, self.flat_ask)

    def test_short_inventory_pushes_both_quotes_up(self):
        """Short: the bid gets easier to hit and the ask harder to lift."""
        for inventory in [-1.0, -10.0, -50.0, -100.0]:
            self.maker.inventory = inventory
            bid, _, ask, _ = self.maker.get_quotes(self.market_state)
            self.assertGreater(bid, self.flat_bid)
            self.assertGreater(ask, self.flat_ask)

    def test_reservation_price_is_linear_in_inventory(self):
        """r = mid - inventory * gamma."""
        for inventory in [-100.0, -7.5, 0.0, 7.5, 100.0]:
            self.maker.inventory = inventory
            reservation = self.maker.get_reservation_price(100.0)
            self.assertAlmostEqual(reservation, 100.0 - inventory * 0.01, places=12)

    def test_quotes_are_centred_on_the_reservation_price(self):
        """The midpoint of our two quotes is the reservation price, not the mid."""
        for inventory in [-40.0, 0.0, 40.0]:
            self.maker.inventory = inventory
            bid, _, ask, _ = self.maker.get_quotes(self.market_state)
            reservation = self.maker.get_reservation_price(100.0)
            self.assertAlmostEqual(0.5 * (bid + ask), reservation, places=12)

    def test_zero_skew_leaves_quotes_on_the_mid(self):
        """gamma = 0 disables the lean entirely."""
        maker = MarketMaker(quote_spread=0.05, inventory_skew_factor=0.0)
        maker.inventory = 80.0
        bid, _, ask, _ = maker.get_quotes(self.market_state)
        self.assertAlmostEqual(bid, 99.95, places=12)
        self.assertAlmostEqual(ask, 100.05, places=12)


class TestPositionLimits(unittest.TestCase):
    """The side that would breach the limit stops being quoted."""

    def setUp(self):
        self.market_state = MarketState(initial_mid=100.0)
        self.maker = MarketMaker(
            quote_spread=0.05,
            quote_size=10.0,
            max_inventory=100.0,
            inventory_skew_factor=0.01,
        )

    def test_bid_withdrawn_at_the_long_limit(self):
        self.maker.inventory = 95.0
        _, bid_size, _, ask_size = self.maker.get_quotes(self.market_state)
        self.assertEqual(bid_size, 0.0)
        self.assertGreater(ask_size, 0.0)

    def test_ask_withdrawn_at_the_short_limit(self):
        self.maker.inventory = -95.0
        _, bid_size, _, ask_size = self.maker.get_quotes(self.market_state)
        self.assertEqual(ask_size, 0.0)
        self.assertGreater(bid_size, 0.0)

    def test_both_sides_quoted_when_a_full_clip_still_fits(self):
        self.maker.inventory = 50.0
        _, bid_size, _, ask_size = self.maker.get_quotes(self.market_state)
        self.assertEqual(bid_size, 10.0)
        self.assertEqual(ask_size, 10.0)


class TestPositionAccounting(unittest.TestCase):
    """Cash flow is not PnL, and rebates are not spread capture."""

    def setUp(self):
        self.maker = MarketMaker(quote_spread=0.05, quote_size=10.0)

    def test_gross_pnl_is_cash_plus_inventory_marked_at_mid(self):
        self.maker.execute_bid_fill(99.95, 10.0)
        self.assertAlmostEqual(self.maker.get_gross_pnl(100.0), 0.5, places=12)
        self.assertAlmostEqual(
            self.maker.get_gross_pnl(100.0),
            self.maker.get_net_cash_flow() + self.maker.get_inventory_value(100.0),
            places=12,
        )

    def test_net_cash_flow_is_not_a_pnl(self):
        """Short 10 at the mid: +$1,000 of cash, zero PnL."""
        self.maker.execute_ask_fill(100.0, 10.0)
        self.assertAlmostEqual(self.maker.get_net_cash_flow(), 1000.0, places=12)
        self.assertAlmostEqual(self.maker.get_gross_pnl(100.0), 0.0, places=12)

    def test_buying_at_the_mid_earns_nothing(self):
        self.maker.execute_bid_fill(100.0, 10.0)
        self.assertAlmostEqual(self.maker.get_net_cash_flow(), -1000.0, places=12)
        self.assertAlmostEqual(self.maker.get_gross_pnl(100.0), 0.0, places=12)

    def test_round_trip_captures_the_full_quoted_width(self):
        self.maker.execute_bid_fill(99.95, 10.0)
        self.maker.execute_ask_fill(100.05, 10.0)
        self.assertEqual(self.maker.get_inventory(), 0.0)
        self.assertAlmostEqual(self.maker.get_gross_pnl(100.0), 1.0, places=12)

    def test_rebates_are_held_outside_gross_pnl(self):
        """Keeping rebates out of `cash` is what keeps the PnL identity exact."""
        maker = MarketMaker(maker_rebate_per_unit=0.001)
        maker.execute_bid_fill(100.0, 10.0)
        self.assertAlmostEqual(maker.get_gross_pnl(100.0), 0.0, places=12)
        self.assertAlmostEqual(maker.rebates, 0.01, places=12)
        self.assertAlmostEqual(maker.get_mark_to_market_pnl(100.0), 0.01, places=12)

    def test_zero_and_negative_quantities_are_ignored(self):
        self.maker.execute_bid_fill(99.95, 0.0)
        self.maker.execute_ask_fill(100.05, -5.0)
        self.assertEqual(self.maker.get_inventory(), 0.0)
        self.assertEqual(self.maker.get_net_cash_flow(), 0.0)

    def test_average_prices_are_volume_weighted(self):
        self.assertIsNone(self.maker.get_average_buy_price())
        self.assertIsNone(self.maker.get_average_sell_price())
        self.maker.execute_bid_fill(99.90, 10.0)
        self.maker.execute_bid_fill(99.80, 30.0)
        self.assertAlmostEqual(self.maker.get_average_buy_price(), 99.825, places=12)


class TestConstruction(unittest.TestCase):
    """Invalid configurations are rejected up front."""

    def test_negative_spread_rejected(self):
        with self.assertRaises(ValueError):
            MarketMaker(quote_spread=-0.01)

    def test_negative_size_rejected(self):
        with self.assertRaises(ValueError):
            MarketMaker(quote_size=-1.0)

    def test_negative_max_inventory_rejected(self):
        with self.assertRaises(ValueError):
            MarketMaker(max_inventory=-10.0)


if __name__ == '__main__':
    unittest.main()
