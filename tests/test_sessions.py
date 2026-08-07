"""
Tests for the session close, which is the mechanic a 24/7 book does not have.

A cash equity desk does not carry an unhedged position through the overnight
gap, so at the close it crosses the market and goes flat. A perpetual swap has
no close and no gap, so it carries the position and pays funding on it instead.
This module tests the first of those two.

The close-out is deliberately implemented as an ordinary trade at a bad price
rather than as an accounting adjustment, because that is what it is. The
properties below are the ones that make the difference matter:

  * the identity must survive it, which is only automatic if the trade goes
    through the same cash and inventory path every other trade goes through
  * it must be a taker trade, so it earns no maker rebate
  * it must not pollute the counters and diagnostics that describe passive
    quoting, since we initiated it
  * it must draw no randomness, or turning it on would move the price path and
    the paired comparison against a 24/7 book would be meaningless
"""

import unittest

from market_making_simulator import MarketMaker, PnLTracker

from tests._helpers import STEP_VOLATILITY, build_simulation

SESSION = 120
# An exact number of sessions, so the run ends on a close and the terminal
# liquidation has nothing left to charge. A ragged horizon is the normal case
# and is covered by the crypto grid; here it would only blur what is tested.
STEPS = 4 * SESSION
EXPECTED_CLOSES = STEPS // SESSION


def run_with_sessions(seed, session_steps=SESSION, num_steps=STEPS, **kwargs):
    """Build a simulator with a session close and run it."""
    simulator = build_simulation(seed, **kwargs)
    simulator.session_steps = session_steps
    simulator.run(num_steps=num_steps, volatility=STEP_VOLATILITY)
    return simulator, simulator.get_summary()


class TestFlatteningHappens(unittest.TestCase):
    """The book goes flat at the close, and only at the close."""

    def setUp(self):
        self.simulator, self.summary = run_with_sessions(3)

    def test_the_book_is_flat_at_every_close(self):
        for step in self.simulator.history:
            if (step['step'] + 1) % SESSION == 0:
                self.assertAlmostEqual(step['inventory'], 0.0, delta=1e-12,
                                       msg=f"not flat at step {step['step']}")

    def test_the_book_is_not_flat_everywhere(self):
        """Otherwise the test above would pass on a strategy that never trades."""
        self.assertTrue(any(abs(step['inventory']) > 0.0
                            for step in self.simulator.history))

    def test_the_close_count_matches_the_schedule(self):
        # At most one per boundary; fewer only if the book happened to be flat
        # already, which cannot be assumed but can be bounded.
        self.assertLessEqual(self.summary['session_closes'], EXPECTED_CLOSES)
        self.assertGreater(self.summary['session_closes'], 0)
        recorded = sum(1 for step in self.simulator.history
                       if step['session_close_qty'] > 0.0)
        self.assertEqual(recorded, self.summary['session_closes'])

    def test_no_close_without_a_session_length(self):
        simulator = build_simulation(3)
        simulator.run(num_steps=STEPS, volatility=STEP_VOLATILITY)
        summary = simulator.get_summary()
        self.assertEqual(summary['session_closes'], 0)
        self.assertEqual(summary['session_close_cost'], 0.0)
        self.assertTrue(all(step['session_close_qty'] == 0.0
                            for step in simulator.history))

    def test_session_length_must_be_a_positive_whole_number(self):
        for bad in (0, -5, 3.5):
            with self.subTest(session_steps=bad):
                with self.assertRaises(ValueError):
                    build_simulation(0).__class__(
                        market_state=build_simulation(0).market_state,
                        market_maker=MarketMaker(),
                        pnl_tracker=PnLTracker(),
                        session_steps=bad,
                    )


class TestCloseOutPricing(unittest.TestCase):
    """Flattening crosses the market's own spread, and costs exactly that."""

    HALF_SPREAD = 0.10

    def setUp(self):
        self.simulator, self.summary = run_with_sessions(
            5, reference_half_spread=self.HALF_SPREAD)

    def test_cost_is_the_half_spread_on_the_flattened_quantity(self):
        self.assertAlmostEqual(
            self.summary['session_close_cost'],
            self.summary['session_close_volume'] * self.HALF_SPREAD,
            delta=1e-9,
        )

    def test_cost_is_never_negative(self):
        """Crossing the market cannot earn edge, whichever way we are."""
        self.assertGreaterEqual(self.summary['session_close_cost'], 0.0)

    def test_the_price_is_on_the_far_side_of_the_mid(self):
        for step in self.simulator.history:
            if step['session_close_qty'] == 0.0:
                continue
            # `inventory_across_move` is the position that was flattened.
            position = step['inventory_across_move']
            expected = (step['mid_price'] - self.HALF_SPREAD if position > 0
                        else step['mid_price'] + self.HALF_SPREAD)
            self.assertAlmostEqual(step['session_close_price'], expected,
                                   delta=1e-9)

    def test_a_wider_market_costs_more_to_flatten_into(self):
        cheap = run_with_sessions(5, reference_half_spread=0.02)[1]
        dear = run_with_sessions(5, reference_half_spread=0.20)[1]
        self.assertGreater(dear['session_close_cost'],
                           cheap['session_close_cost'])


class TestAccountingSurvivesTheClose(unittest.TestCase):
    """The identity and the diagnostics must both stay honest."""

    def setUp(self):
        self.simulator, self.summary = run_with_sessions(7)

    def test_the_identity_holds(self):
        self.assertAlmostEqual(
            self.summary['gross_pnl'],
            self.summary['spread_capture'] + self.summary['inventory_pnl'],
            delta=1e-9,
        )
        self.assertAlmostEqual(
            self.summary['gross_pnl'],
            self.simulator.market_maker.get_gross_pnl(
                self.simulator.market_state.get_mid_price()),
            delta=1e-9,
        )

    def test_the_waterfall_reconciles(self):
        self.assertAlmostEqual(
            self.summary['net_pnl'],
            self.summary['gross_pnl'] + self.summary['rebates']
            + self.summary['funding'] - self.summary['liquidation_cost'],
            delta=1e-9,
        )

    def test_the_close_out_is_a_split_of_spread_capture_not_an_extra_charge(self):
        """
        quoted_edge - close_out_cost == spread_capture, exactly.

        Reporting the close-out as a separate waterfall line would subtract it
        twice, which is precisely the mistake the v0.1 decomposition made with
        adverse selection.
        """
        self.assertGreater(self.summary['session_close_cost'], 0.0)
        self.assertAlmostEqual(
            self.summary['quoted_edge'] - self.summary['session_close_cost'],
            self.summary['spread_capture'],
            delta=1e-9,
        )

    def test_a_close_out_earns_no_maker_rebate(self):
        """It removes liquidity, so paying us a maker rebate would be wrong."""
        rebate = 0.002
        summary = run_with_sessions(7, maker_rebate_per_unit=rebate)[1]
        self.assertGreater(summary['session_close_volume'], 0.0)
        self.assertAlmostEqual(
            summary['rebates'], summary['filled_volume'] * rebate, delta=1e-9)

    def test_close_outs_are_not_counted_as_fills(self):
        """
        The fill counters describe passive flow we attracted. Folding in trades
        we initiated would overstate them by whatever the calendar forced, and
        would break the plotter's cross-check against the summary.
        """
        fills_in_history = sum(
            (step['bid_fill_qty'] > 0) + (step['ask_fill_qty'] > 0)
            for step in self.simulator.history
        )
        self.assertEqual(self.summary['num_trades'], fills_in_history)
        self.assertAlmostEqual(
            self.summary['filled_volume'],
            sum(step['bid_fill_qty'] + step['ask_fill_qty']
                for step in self.simulator.history),
            delta=1e-9,
        )

    def test_close_outs_are_excluded_from_the_markout(self):
        """
        Adverse selection asks whether the flow that hit our quote was
        informed. A trade we initiated at a fixed time cannot answer that, and
        including it would put a calendar artefact into a metric read as a
        property of our counterparties.
        """
        tracker = PnLTracker(markout_horizon=2)
        for index in range(5):
            tracker.record_step(float(index + 1), 0.0,
                                100.0 + index, 101.0 + index)
        tracker.record_session_close(1.0, -10.0, 99.90, 100.0, 0)
        self.assertEqual(tracker.get_adverse_selection(), 0.0)
        # And the same close-out still reaches spread capture, so it is
        # excluded from the diagnostic without vanishing from the identity.
        self.assertAlmostEqual(tracker.get_spread_capture(), -1.0, places=12)

    def test_the_final_inventory_is_flat_when_the_run_ends_on_a_close(self):
        self.assertAlmostEqual(self.summary['final_inventory'], 0.0, delta=1e-12)
        self.assertAlmostEqual(self.summary['liquidation_cost'], 0.0, delta=1e-12)


class TestTheCloseIsAnInventoryControl(unittest.TestCase):
    """
    The point of the mechanic, and the reason 24/7 is not just a longer day.

    With the inventory lean switched off, the position is a random walk into
    the limit and nothing ever brings it back. A session close resets it, for
    the price of the half-spread. A perpetual book has no such reset.
    """

    def _skewless(self, session_steps):
        simulator = build_simulation(13, inventory_skew_factor=0.0)
        simulator.session_steps = session_steps
        simulator.run(num_steps=2000, volatility=STEP_VOLATILITY)
        return simulator.get_summary()

    def test_flattening_cuts_the_inventory_a_skewless_book_carries(self):
        without = self._skewless(None)
        with_close = self._skewless(200)
        self.assertLess(
            with_close['mean_abs_inventory'],
            without['mean_abs_inventory'],
            msg="the close did not reduce the inventory carried",
        )

    def test_the_reset_is_paid_for_rather_than_free(self):
        """
        The control is bought with the half-spread, every close, on whatever
        the book happens to be holding. A version of this mechanic that reduced
        inventory at no cost would be the most valuable thing in the model and
        would be wrong.
        """
        with_close = self._skewless(200)
        self.assertGreater(with_close['session_close_cost'], 0.0)
        self.assertAlmostEqual(with_close['final_inventory'], 0.0, delta=1e-12)

    def test_the_peak_is_not_controlled_by_a_periodic_reset(self):
        """
        Only the average is. Inside one session the position is still an
        unleaned random walk and still reaches the position limit, so a reader
        cannot take the close as a substitute for the inventory lean. Stated as
        a test because the opposite is the intuitive reading of the row above.
        """
        without = self._skewless(None)
        with_close = self._skewless(200)
        self.assertAlmostEqual(with_close['peak_abs_inventory'],
                               without['peak_abs_inventory'], delta=1e-9)


class TestTheCloseDrawsNoRandomness(unittest.TestCase):
    """Turning it on must not move the price path or the arrival sequence."""

    def test_the_path_is_identical_with_and_without_closes(self):
        plain = build_simulation(29)
        closing = build_simulation(29)
        closing.session_steps = SESSION
        for simulator in (plain, closing):
            simulator.run(num_steps=STEPS, volatility=STEP_VOLATILITY)

        self.assertEqual(
            [step['mid_price'] for step in plain.history],
            [step['mid_price'] for step in closing.history],
        )
        # The fills themselves do diverge, because flattening changes the
        # inventory and therefore the quotes, but they diverge only after the
        # first close and not before it.
        for index in range(SESSION - 1):
            self.assertEqual(plain.history[index]['bid_fill_qty'],
                             closing.history[index]['bid_fill_qty'])
            self.assertEqual(plain.history[index]['ask_fill_qty'],
                             closing.history[index]['ask_fill_qty'])


class TestFlattenOnTheMaker(unittest.TestCase):
    """The maker-level primitive, checked by hand on both signs."""

    def test_selling_a_long_down_to_flat(self):
        maker = MarketMaker(maker_rebate_per_unit=0.01, maker_rebate_bps=1.0)
        maker.execute_bid_fill(99.95, 10.0)
        rebates_after_fill = maker.rebates

        signed = maker.execute_flatten(99.90)
        self.assertAlmostEqual(signed, -10.0, places=12)
        self.assertEqual(maker.inventory, 0.0)
        self.assertAlmostEqual(maker.cash, -999.5 + 999.0, places=9)
        # Taker: no rebate accrued by the flatten.
        self.assertEqual(maker.rebates, rebates_after_fill)

    def test_buying_a_short_back_to_flat(self):
        maker = MarketMaker()
        maker.execute_ask_fill(100.05, 10.0)
        signed = maker.execute_flatten(100.10)
        self.assertAlmostEqual(signed, +10.0, places=12)
        self.assertEqual(maker.inventory, 0.0)
        self.assertAlmostEqual(maker.cash, 1000.5 - 1001.0, places=9)

    def test_flattening_a_flat_book_does_nothing(self):
        maker = MarketMaker()
        self.assertEqual(maker.execute_flatten(100.0), 0.0)
        self.assertEqual(maker.cash, 0.0)
        self.assertEqual(maker.total_sold, 0.0)
        self.assertEqual(maker.total_bought, 0.0)

    def test_a_zero_quantity_close_is_not_recorded_as_a_trade(self):
        tracker = PnLTracker()
        tracker.record_session_close(1.0, 0.0, 100.0, 100.0, 0)
        self.assertEqual(tracker.trades, [])
        self.assertEqual(tracker.get_session_close_count(), 0)


if __name__ == '__main__':
    unittest.main()
