"""
Tests for PnLTracker.

The decomposition is an identity, so the test is not "do the three numbers add
up to the total we defined as their sum", which cannot fail. It is "does
spread_capture + inventory_pnl equal the mark-to-market PnL computed
independently by MarketMaker from cash and inventory", which is a real
constraint and which the previous implementation missed by $500 on a $200 run.

Adverse selection is checked separately, as a signed markout that must be zero
in expectation under uninformed flow and negative under informed flow.
"""

import unittest

import numpy as np

from market_making_simulator import PnLTracker

from tests._helpers import adverse_selection_sample, run_simulation


# Configurations used for the identity test. They cover a skew large enough to
# push our quotes through the mid (so spread capture goes negative on some
# fills), a rebate, informed flow, and a position limit tight enough to bite.
IDENTITY_CONFIGS = [
    {},
    {'quote_spread': 0.01, 'inventory_skew_factor': 0.05},
    {'quote_spread': 0.20, 'inventory_skew_factor': 0.0},
    {'maker_rebate_per_unit': 0.002, 'informed_fraction': 0.4},
    {'max_inventory': 15.0, 'inventory_skew_factor': 0.002},
    {'markout_horizon': 1, 'informed_fraction': 1.0},
]


class TestDecompositionIdentity(unittest.TestCase):
    """gross_pnl must equal the independently computed mark-to-market PnL."""

    def test_identity_holds_across_seeds_and_configurations(self):
        for config in IDENTITY_CONFIGS:
            for seed in (0, 1, 2, 3):
                simulator, _ = run_simulation(seed, num_steps=600, **config)
                final_mid = simulator.market_state.get_mid_price()
                decomposition = simulator.pnl_tracker.get_pnl_decomposition()
                self.assertAlmostEqual(
                    decomposition['gross_pnl'],
                    simulator.market_maker.get_gross_pnl(final_mid),
                    delta=1e-9,
                    msg=f"decomposition broke for config={config}, seed={seed}",
                )

    def test_gross_pnl_is_the_sum_of_its_two_components(self):
        simulator, _ = run_simulation(4, num_steps=600)
        decomposition = simulator.pnl_tracker.get_pnl_decomposition()
        self.assertAlmostEqual(
            decomposition['gross_pnl'],
            decomposition['spread_capture'] + decomposition['inventory_pnl'],
            delta=1e-12,
        )

    def test_identity_holds_with_no_fills_at_all(self):
        """A quote far enough out never trades, and the identity is 0 = 0."""
        simulator, _ = run_simulation(5, num_steps=200, quote_spread=5.0)
        final_mid = simulator.market_state.get_mid_price()
        self.assertEqual(simulator.pnl_tracker.get_trade_count(), (0, 0))
        self.assertAlmostEqual(
            simulator.pnl_tracker.get_gross_pnl(),
            simulator.market_maker.get_gross_pnl(final_mid),
            delta=1e-12,
        )


class TestSpreadCapture(unittest.TestCase):
    """Edge against the mid, signed, with no floor at zero."""

    def setUp(self):
        self.tracker = PnLTracker()

    def test_buying_below_the_mid_earns_positive_edge(self):
        self.tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0, 0)
        self.assertAlmostEqual(self.tracker.get_spread_capture(), 0.5, places=12)

    def test_selling_above_the_mid_earns_positive_edge(self):
        self.tracker.record_trade(1.0, 'sell', 100.05, 10.0, 100.0, 0)
        self.assertAlmostEqual(self.tracker.get_spread_capture(), 0.5, places=12)

    def test_a_quote_skewed_through_the_mid_earns_negative_edge(self):
        """Heavy inventory skew can push our bid above the mid. That is a cost."""
        self.tracker.record_trade(1.0, 'buy', 100.10, 10.0, 100.0, 0)
        self.assertAlmostEqual(self.tracker.get_spread_capture(), -1.0, places=12)

    def test_side_must_be_buy_or_sell(self):
        with self.assertRaises(ValueError):
            self.tracker.record_trade(1.0, 'hold', 100.0, 10.0, 100.0, 0)


class TestInventoryPnL(unittest.TestCase):
    """PnL from price moves applies to the position held across each move."""

    def setUp(self):
        self.tracker = PnLTracker()

    def test_long_through_a_rise_makes_money(self):
        self.tracker.record_step(1.0, 10.0, 100.0, 100.5)
        self.assertAlmostEqual(self.tracker.get_inventory_pnl(), 5.0, places=12)

    def test_short_through_a_rise_loses_money(self):
        self.tracker.record_step(1.0, -10.0, 100.0, 100.5)
        self.assertAlmostEqual(self.tracker.get_inventory_pnl(), -5.0, places=12)

    def test_a_flat_step_contributes_nothing(self):
        self.tracker.record_step(1.0, 0.0, 100.0, 105.0)
        self.assertAlmostEqual(self.tracker.get_inventory_pnl(), 0.0, places=12)

    def test_steps_accumulate(self):
        self.tracker.record_step(1.0, 10.0, 100.0, 101.0)
        self.tracker.record_step(2.0, -5.0, 101.0, 100.0)
        self.assertAlmostEqual(self.tracker.get_inventory_pnl(), 15.0, places=12)


class TestAdverseSelectionMechanics(unittest.TestCase):
    """The markout is signed, so it can come out positive."""

    def _rising_path(self, tracker: PnLTracker) -> None:
        """Five steps, mid rising by 1.0 each step from 100 to 105."""
        for index in range(5):
            tracker.record_step(float(index + 1), 0.0, 100.0 + index, 101.0 + index)

    def test_buying_before_a_rise_gives_a_positive_markout(self):
        tracker = PnLTracker(markout_horizon=2)
        tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0, 0)
        self._rising_path(tracker)
        self.assertAlmostEqual(tracker.get_adverse_selection(), 20.0, places=12)

    def test_selling_before_a_rise_gives_a_negative_markout(self):
        tracker = PnLTracker(markout_horizon=2)
        tracker.record_trade(1.0, 'sell', 100.05, 10.0, 100.0, 0)
        self._rising_path(tracker)
        self.assertAlmostEqual(tracker.get_adverse_selection(), -20.0, places=12)

    def test_horizon_is_clamped_at_the_end_of_the_run(self):
        """A trade near the end is marked against the final observable mid."""
        tracker = PnLTracker(markout_horizon=5)
        tracker.record_trade(4.0, 'buy', 102.95, 10.0, 103.0, 3)
        self._rising_path(tracker)
        self.assertAlmostEqual(tracker.get_adverse_selection(), 20.0, places=12)

    def test_a_trade_exactly_h_steps_from_the_end_is_marked_over_h_steps(self):
        """
        The run boundary must not stretch one trade's horizon.

        A fill in step 2 with h = 2 has step 4 available, and the mid h steps
        later is mid_before_move of step 4, which is 104. Reaching for
        mid_after_move of step 4 instead would mark this single trade over three
        steps while every other trade in the run gets two.
        """
        tracker = PnLTracker(markout_horizon=2)
        tracker.record_trade(3.0, 'buy', 101.95, 10.0, 102.0, 2)
        self._rising_path(tracker)
        markout = tracker.get_adverse_selection()
        self.assertAlmostEqual(markout, 20.0, places=12)
        # 30.0 is the h+1 answer, the off-by-one this test exists to catch.
        self.assertNotAlmostEqual(markout, 30.0, places=12)

    def test_no_trades_means_no_markout(self):
        tracker = PnLTracker()
        self._rising_path(tracker)
        self.assertEqual(tracker.get_adverse_selection(), 0.0)

    def test_markout_is_not_a_third_additive_bucket(self):
        """
        The old decomposition subtracted adverse selection as if it were a
        separate cost, double-counting part of the inventory term.
        """
        tracker = PnLTracker(markout_horizon=2)
        tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0, 0)
        self._rising_path(tracker)
        decomposition = tracker.get_pnl_decomposition()
        self.assertNotEqual(decomposition['adverse_selection'], 0.0)
        self.assertAlmostEqual(
            decomposition['gross_pnl'],
            decomposition['spread_capture'] + decomposition['inventory_pnl'],
            places=12,
        )

    def test_invalid_horizons_rejected(self):
        with self.assertRaises(ValueError):
            PnLTracker(markout_horizon=0)
        tracker = PnLTracker()
        tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0, 0)
        tracker.record_step(1.0, 10.0, 100.0, 100.5)
        with self.assertRaises(ValueError):
            tracker.get_adverse_selection(horizon=0)


class TestAdverseSelectionUnderInformedFlow(unittest.TestCase):
    """
    The metric must be zero when there is nothing to detect.

    Tolerances come from the sample standard error, not from a hardcoded
    number, so the tests state a statistical claim rather than pinning a
    particular random draw.
    """

    # One sample size for every informed_fraction, so the three settings are
    # directly comparable and the figures quoted in the README come from a
    # single stated experiment rather than two different ones.
    UNINFORMED_SEEDS = 60
    INFORMED_SEEDS = 60

    @staticmethod
    def _mean_and_standard_error(sample) -> tuple:
        values = np.array(sample, dtype=float)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))

    def test_uninformed_flow_shows_no_adverse_selection(self):
        sample, _ = adverse_selection_sample(0.0, self.UNINFORMED_SEEDS)
        mean, standard_error = self._mean_and_standard_error(sample)
        self.assertLess(
            abs(mean), 3.0 * standard_error,
            msg=f"expected zero, got {mean:.3f} with SE {standard_error:.3f}",
        )

    def test_informed_flow_shows_significant_adverse_selection(self):
        sample, _ = adverse_selection_sample(0.6, self.INFORMED_SEEDS)
        mean, standard_error = self._mean_and_standard_error(sample)
        self.assertLess(
            mean, -3.0 * standard_error,
            msg=f"expected clearly negative, got {mean:.3f} "
                f"with SE {standard_error:.3f}",
        )

    def test_adverse_selection_worsens_with_more_informed_flow(self):
        uninformed, _ = adverse_selection_sample(0.0, self.UNINFORMED_SEEDS)
        partly, _ = adverse_selection_sample(0.3, self.INFORMED_SEEDS)
        heavily, _ = adverse_selection_sample(0.6, self.INFORMED_SEEDS)
        means = [self._mean_and_standard_error(s)[0]
                 for s in (uninformed, partly, heavily)]
        for earlier, later in zip(means, means[1:]):
            self.assertLess(later, earlier)

    def test_informed_flow_costs_net_pnl(self):
        """Adverse selection is not a relabelling, it shows up in the bottom line."""
        _, uninformed_net = adverse_selection_sample(0.0, self.UNINFORMED_SEEDS)
        _, informed_net = adverse_selection_sample(0.6, self.INFORMED_SEEDS)
        uninformed_mean, uninformed_se = self._mean_and_standard_error(uninformed_net)
        informed_mean, informed_se = self._mean_and_standard_error(informed_net)
        gap_standard_error = np.sqrt(uninformed_se ** 2 + informed_se ** 2)
        self.assertLess(informed_mean, uninformed_mean - 3.0 * gap_standard_error)


class TestCounters(unittest.TestCase):
    """Trade counts and volumes."""

    def setUp(self):
        self.tracker = PnLTracker()
        self.tracker.record_trade(1.0, 'buy', 99.95, 4.0, 100.0, 0)
        self.tracker.record_trade(2.0, 'sell', 100.05, 6.0, 100.0, 1)
        self.tracker.record_trade(3.0, 'sell', 100.05, 2.5, 100.0, 2)

    def test_trade_count_splits_by_side(self):
        self.assertEqual(self.tracker.get_trade_count(), (1, 2))

    def test_filled_volume_sums_absolute_quantities(self):
        self.assertAlmostEqual(self.tracker.get_filled_volume(), 12.5, places=12)


if __name__ == '__main__':
    unittest.main()
