"""
Tests for MarketState.

MarketState is an exogenous mid price plus a reference half-spread. It is not a
limit order book, and one test below pins that down: the depth and market-order
methods of the old OrderBook were never called by any simulation, so they are
gone and must not come back as an unearned claim.
"""

import unittest

from market_making_simulator import MarketState


class TestReferenceQuotes(unittest.TestCase):
    """The market's own quotes straddle the mid symmetrically."""

    def setUp(self):
        self.market_state = MarketState(initial_mid=100.0, reference_half_spread=0.10)

    def test_reference_quotes_straddle_the_mid(self):
        self.assertAlmostEqual(self.market_state.get_reference_bid(), 99.90, places=12)
        self.assertAlmostEqual(self.market_state.get_reference_ask(), 100.10, places=12)

    def test_reference_ask_above_reference_bid(self):
        self.assertGreater(
            self.market_state.get_reference_ask(),
            self.market_state.get_reference_bid(),
        )

    def test_mid_tracks_updates_and_initial_mid_does_not(self):
        """`initial_mid` is the run's anchor and must survive price moves."""
        self.market_state.update_mid_price(101.25)
        self.assertEqual(self.market_state.get_mid_price(), 101.25)
        self.assertEqual(self.market_state.initial_mid, 100.0)

    def test_zero_half_spread_collapses_the_reference_quotes(self):
        market_state = MarketState(initial_mid=50.0, reference_half_spread=0.0)
        self.assertEqual(market_state.get_reference_bid(), 50.0)
        self.assertEqual(market_state.get_reference_ask(), 50.0)


class TestLiquidationCost(unittest.TestCase):
    """Flattening a position costs half the market spread per unit, either way."""

    def setUp(self):
        self.market_state = MarketState(initial_mid=100.0, reference_half_spread=0.10)

    def test_cost_is_non_negative_for_both_signs(self):
        for inventory in [-100.0, -12.5, -1.0, 0.0, 1.0, 12.5, 100.0]:
            self.assertGreaterEqual(self.market_state.liquidation_cost(inventory), 0.0)

    def test_cost_is_symmetric_in_the_sign_of_the_position(self):
        self.assertAlmostEqual(
            self.market_state.liquidation_cost(30.0),
            self.market_state.liquidation_cost(-30.0),
            places=12,
        )

    def test_flat_position_costs_nothing(self):
        self.assertEqual(self.market_state.liquidation_cost(0.0), 0.0)

    def test_cost_scales_with_size_and_half_spread(self):
        self.assertAlmostEqual(self.market_state.liquidation_cost(30.0), 3.0, places=12)
        wide = MarketState(initial_mid=100.0, reference_half_spread=0.25)
        self.assertAlmostEqual(wide.liquidation_cost(30.0), 7.5, places=12)


class TestNotAnOrderBook(unittest.TestCase):
    """The order book abstraction is gone and must stay gone."""

    def test_no_depth_or_market_order_api(self):
        """
        These methods existed on the old OrderBook and were never called.

        Reintroducing them would imply a depth model this simulator does not
        have, which is exactly the kind of unearned claim the rewrite removed.
        """
        market_state = MarketState()
        for attribute in ('execute_market_buy', 'execute_market_sell',
                          'get_depth_at_level', 'add_order', 'bids', 'asks'):
            self.assertFalse(
                hasattr(market_state, attribute),
                msg=f"MarketState should not expose an order book API: {attribute}",
            )

    def test_order_book_class_is_not_importable(self):
        with self.assertRaises(ImportError):
            from market_making_simulator import OrderBook  # noqa: F401


class TestConstruction(unittest.TestCase):
    """Invalid configurations are rejected up front."""

    def test_non_positive_mid_rejected(self):
        with self.assertRaises(ValueError):
            MarketState(initial_mid=0.0)

    def test_negative_half_spread_rejected(self):
        with self.assertRaises(ValueError):
            MarketState(reference_half_spread=-0.01)


if __name__ == '__main__':
    unittest.main()
