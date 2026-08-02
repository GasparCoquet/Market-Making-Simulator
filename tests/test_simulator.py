"""
Tests for MarketSimulator.

Covers the properties that make a run believable: the summary waterfall
reconciles, quoting wider genuinely fills less, the price process has no
built-in drift, fills are capped at our own quoted size, and two runs with the
same seed reproduce even when they are interleaved step by step.
"""

import unittest

import numpy as np

from market_making_simulator import MarketMaker, MarketState, PnLTracker, MarketSimulator

from tests._helpers import STEP_VOLATILITY, build_simulation, run_simulation


def assert_sequences_match(test_case, first, second, label: str) -> None:
    """
    Compare two step-indexed sequences and report the first divergence.

    unittest's own list diff is quadratic in the length of the sequence, which
    turns a failure on a several-hundred-step path into a multi-minute hang.
    Reporting the first differing index is both faster and more useful.
    """
    test_case.assertEqual(len(first), len(second), msg=f"{label}: different lengths")
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            test_case.fail(
                f"{label}: first divergence at step {index}, {left!r} vs {right!r}")


class TestSummaryWaterfall(unittest.TestCase):
    """Net PnL must be reachable from gross PnL by stated adjustments only."""

    def test_waterfall_reconciles_across_seeds_and_rebates(self):
        for rebate in (0.0, 0.002):
            for seed in (0, 1, 2, 3, 4):
                _, summary = run_simulation(
                    seed, num_steps=600, maker_rebate_per_unit=rebate)
                expected = (summary['gross_pnl']
                            + summary['rebates']
                            - summary['liquidation_cost'])
                self.assertAlmostEqual(summary['net_pnl'], expected, delta=1e-12)

    def test_gross_pnl_equals_spread_capture_plus_inventory_pnl(self):
        for seed in (0, 1, 2):
            _, summary = run_simulation(seed, num_steps=600)
            self.assertAlmostEqual(
                summary['gross_pnl'],
                summary['spread_capture'] + summary['inventory_pnl'],
                delta=1e-12,
            )

    def test_gross_pnl_equals_cash_flow_plus_inventory_value(self):
        simulator, summary = run_simulation(3, num_steps=600)
        self.assertAlmostEqual(
            summary['gross_pnl'],
            summary['net_cash_flow'] + summary['inventory_value'],
            delta=1e-9,
        )

    def test_liquidation_cost_is_charged_on_the_residual_position(self):
        _, summary = run_simulation(6, num_steps=600, reference_half_spread=0.10)
        self.assertAlmostEqual(
            summary['liquidation_cost'],
            abs(summary['final_inventory']) * 0.10,
            delta=1e-12,
        )
        self.assertGreaterEqual(summary['liquidation_cost'], 0.0)

    def test_max_drawdown_is_non_negative(self):
        for seed in (0, 1, 2):
            _, summary = run_simulation(seed, num_steps=600)
            self.assertGreaterEqual(summary['max_drawdown'], 0.0)

    def test_summary_is_empty_before_the_run(self):
        simulator = build_simulation(0)
        self.assertEqual(simulator.get_summary(), {})

    def test_trade_counts_agree_with_the_history(self):
        simulator, summary = run_simulation(2, num_steps=600)
        fills_in_history = sum(
            (step['bid_fill_qty'] > 0) + (step['ask_fill_qty'] > 0)
            for step in simulator.history
        )
        self.assertEqual(summary['num_trades'], fills_in_history)
        self.assertEqual(summary['num_trades'],
                         summary['num_buys'] + summary['num_sells'])


class TestFillsDependOnTheQuotedPrice(unittest.TestCase):
    """
    Widening the quote must reduce the fill count.

    Under the old coin-flip fill rule this was flat in the spread, PnL was
    linear in it, and the optimal strategy was an infinitely wide quote.
    """

    SPREAD_GRID = [0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.25]

    # Seeds 0 to 29, a contiguous block rather than a hand-picked pair, because
    # a monotonicity that only survives on chosen seeds is not a property of the
    # fill model. The path is shortened to 600 steps to pay for the extra seeds:
    # 30 x 7 x 600 runs in about 2.6s, against about 0.4s for the 2 x 7 x 1500
    # it replaces, a delta the suite's runtime budget absorbs.
    MONOTONICITY_SEEDS = range(30)
    MONOTONICITY_STEPS = 600

    def test_fill_count_is_strictly_decreasing_in_the_quoted_spread(self):
        """
        Strict on every one of the 30 seeds, not on average over them.

        600 steps is short enough to keep the suite inside its runtime budget
        and still leaves a wide margin: the smallest gap between consecutive
        spread levels over all 30 seeds and all six pairs is 40 fills, so this
        is nowhere near a tie that a different seed set would flip.
        """
        for seed in self.MONOTONICITY_SEEDS:
            counts = [
                run_simulation(seed, num_steps=self.MONOTONICITY_STEPS,
                               quote_spread=spread)[1]['num_trades']
                for spread in self.SPREAD_GRID
            ]
            for (wider, more), (narrower, fewer) in zip(
                    zip(self.SPREAD_GRID, counts), zip(self.SPREAD_GRID[1:], counts[1:])):
                self.assertLess(
                    fewer, more,
                    msg=(f"seed {seed}: quoting {narrower} filled {fewer} times, "
                         f"quoting {wider} filled {more} times"),
                )

    def test_filled_volume_is_decreasing_in_the_quoted_spread(self):
        volumes = [
            run_simulation(11, num_steps=1500, quote_spread=spread)[1]['filled_volume']
            for spread in self.SPREAD_GRID
        ]
        for earlier, later in zip(volumes, volumes[1:]):
            self.assertLess(later, earlier)

    def test_common_random_numbers_line_up_across_spreads(self):
        """
        Constant draw consumption per step means the price path is shared.

        That is what makes the comparison above a controlled experiment rather
        than a comparison of two different markets.
        """
        spreads = [0.01, 0.05, 0.25]
        paths = []
        for spread in spreads:
            simulator, _ = run_simulation(21, num_steps=300, quote_spread=spread)
            paths.append([step['mid_price'] for step in simulator.history])
        for spread, path in zip(spreads[1:], paths[1:]):
            assert_sequences_match(
                self, paths[0], path, f"price path differs at spread {spread}")


class TestFillMechanics(unittest.TestCase):
    """Fills are capped at our own clip, so partial fills happen."""

    def test_partial_fills_occur(self):
        simulator, _ = run_simulation(13, num_steps=2000, quote_size=10.0)
        fills = [
            quantity
            for step in simulator.history
            for quantity in (step['bid_fill_qty'], step['ask_fill_qty'])
            if quantity > 0.0
        ]
        self.assertGreater(len(fills), 0)
        self.assertTrue(any(quantity < 10.0 for quantity in fills))

    def test_no_fill_ever_exceeds_the_quoted_size(self):
        simulator, _ = run_simulation(13, num_steps=2000, quote_size=10.0)
        for step in simulator.history:
            self.assertLessEqual(step['bid_fill_qty'], step['bid_size'] + 1e-12)
            self.assertLessEqual(step['ask_fill_qty'], step['ask_size'] + 1e-12)

    def test_inventory_stays_within_the_position_limit(self):
        """The limit binds because we withdraw the offending side before it breaches."""
        simulator, _ = run_simulation(17, num_steps=2000, max_inventory=25.0)
        for step in simulator.history:
            self.assertLessEqual(abs(step['inventory']), 25.0 + 1e-9)

    def test_quotes_never_cross_over_a_whole_run(self):
        simulator, _ = run_simulation(19, num_steps=2000, inventory_skew_factor=0.05)
        for step in simulator.history:
            self.assertGreater(step['ask_price'], step['bid_price'])

    def test_fill_prices_match_the_quotes_that_were_shown(self):
        simulator, _ = run_simulation(23, num_steps=500)
        for step in simulator.history:
            if step['bid_fill_qty'] > 0:
                self.assertEqual(step['bid_fill_price'], step['bid_price'])
            if step['ask_fill_qty'] > 0:
                self.assertEqual(step['ask_fill_price'], step['ask_price'])


class TestPriceProcess(unittest.TestCase):
    """
    The mid must be a martingale.

    Without the Ito correction, GBM built from log returns of mean zero drifts
    up by exp(0.5*sigma^2*T), which is a free profit for anyone holding a long
    position and contaminates every comparison across volatilities.
    """

    PATHS = 2000
    STEPS = 200
    VOLATILITY = 0.02
    BASE_SEED = 1000

    def _terminal_mids(self, remove_correction: bool = False) -> np.ndarray:
        """
        Terminal mid of each path.

        Args:
            remove_correction: Cancel the -0.5*sigma^2*dt term by adding it
                back on, which reproduces the original drifting process.

        Returns:
            Array of terminal mid prices.
        """
        terminal = np.empty(self.PATHS)
        for path in range(self.PATHS):
            simulator = build_simulation(self.BASE_SEED + path)
            log_price = np.log(100.0)
            for _ in range(self.STEPS):
                log_return = simulator.draw_log_return(self.VOLATILITY, 1.0)
                if remove_correction:
                    log_return += 0.5 * self.VOLATILITY ** 2
                log_price += log_return
            terminal[path] = np.exp(log_price)
        return terminal

    def test_terminal_mid_is_unbiased(self):
        terminal = self._terminal_mids()
        standard_error = terminal.std(ddof=1) / np.sqrt(self.PATHS)
        self.assertLess(
            abs(terminal.mean() - 100.0), 3.0 * standard_error,
            msg=(f"mean terminal mid {terminal.mean():.4f} is more than 3 standard "
                 f"errors ({standard_error:.4f}) from the initial mid"),
        )

    def test_the_same_test_rejects_a_process_without_the_correction(self):
        """Confirms the test above has the power to catch the bug it guards."""
        terminal = self._terminal_mids(remove_correction=True)
        standard_error = terminal.std(ddof=1) / np.sqrt(self.PATHS)
        self.assertGreater(abs(terminal.mean() - 100.0), 3.0 * standard_error)

    def test_log_return_has_the_requested_volatility(self):
        simulator = build_simulation(99)
        returns = np.array([simulator.draw_log_return(0.02, 1.0) for _ in range(20000)])
        self.assertAlmostEqual(returns.std(ddof=1), 0.02, delta=0.02 * 0.05)

    def test_volatility_scales_the_dispersion_of_the_terminal_mid(self):
        quiet = [run_simulation(seed, num_steps=400,
                                volatility=STEP_VOLATILITY)[1]['final_mid']
                 for seed in range(20)]
        wild = [run_simulation(seed, num_steps=400,
                               volatility=10.0 * STEP_VOLATILITY)[1]['final_mid']
                for seed in range(20)]
        self.assertGreater(np.std(wild), np.std(quiet))


class TestReproducibility(unittest.TestCase):
    """Randomness is per instance, so interleaved runs still reproduce."""

    @staticmethod
    def _fingerprint(simulator: MarketSimulator) -> list:
        return [(step['mid_price'], step['bid_price'], step['ask_price'],
                 step['bid_fill_qty'], step['ask_fill_qty'], step['inventory'])
                for step in simulator.history]

    def test_same_seed_reproduces_the_run(self):
        first, _ = run_simulation(42, num_steps=300)
        second, _ = run_simulation(42, num_steps=300)
        assert_sequences_match(
            self, self._fingerprint(first), self._fingerprint(second),
            "same seed diverged")

    def test_different_seeds_give_different_runs(self):
        first, _ = run_simulation(42, num_steps=300)
        second, _ = run_simulation(43, num_steps=300)
        self.assertNotEqual(self._fingerprint(first), self._fingerprint(second))

    def test_reproducibility_survives_interleaving(self):
        """
        Step two identically seeded simulators alternately.

        With a process-global RNG the two runs would consume each other's
        draws and diverge immediately. With a per-instance generator they stay
        identical.
        """
        first = build_simulation(2024)
        second = build_simulation(2024)
        for _ in range(300):
            first.step(volatility=STEP_VOLATILITY, dt=1.0)
            second.step(volatility=STEP_VOLATILITY, dt=1.0)
        assert_sequences_match(
            self, self._fingerprint(first), self._fingerprint(second),
            "interleaved runs diverged")
        self.assertEqual(first.get_summary(), second.get_summary())

    def test_an_unrelated_generator_does_not_disturb_a_run(self):
        """Seeding or drawing from the global RNG must have no effect."""
        reference, _ = run_simulation(555, num_steps=200)
        np.random.seed(1)
        np.random.random(1000)
        disturbed, _ = run_simulation(555, num_steps=200)
        assert_sequences_match(
            self, self._fingerprint(reference), self._fingerprint(disturbed),
            "the global RNG leaked into the run")

    def test_the_seed_is_reported_in_the_summary(self):
        """A result nobody can reproduce is not a result."""
        _, summary = run_simulation(777, num_steps=100)
        self.assertEqual(summary['random_seed'], 777)


class TestStepBookkeeping(unittest.TestCase):
    """One step, checked by hand."""

    def setUp(self):
        self.simulator = build_simulation(0)

    def test_step_advances_time_and_index(self):
        self.simulator.step(volatility=STEP_VOLATILITY, dt=0.5)
        self.assertEqual(self.simulator.current_time, 0.5)
        self.assertEqual(self.simulator.step_index, 1)
        self.assertEqual(len(self.simulator.history), 1)

    def test_recorded_mid_is_the_price_after_the_move(self):
        step = self.simulator.step(volatility=STEP_VOLATILITY, dt=1.0)
        self.assertEqual(step['mid_price'], self.simulator.market_state.get_mid_price())
        self.assertEqual(step['mid_before'], 100.0)

    def test_inventory_is_snapshotted_after_fills(self):
        """
        The snapshot must be the position held across the move.

        Taking it before the fills is an off-by-one that silently breaks the
        PnL identity, which is why the recorded inventory has to match the
        market maker's position at the end of the step.
        """
        for _ in range(50):
            step = self.simulator.step(volatility=STEP_VOLATILITY, dt=1.0)
            self.assertEqual(step['inventory'], self.simulator.market_maker.get_inventory())

    def test_run_produces_one_record_per_step(self):
        history = self.simulator.run(num_steps=25, volatility=STEP_VOLATILITY)
        self.assertEqual(len(history), 25)
        self.assertEqual([step['step'] for step in history], list(range(25)))

    def test_components_can_be_wired_by_hand(self):
        """The constructor takes plain objects, with no hidden global state."""
        simulator = MarketSimulator(
            market_state=MarketState(initial_mid=50.0, reference_half_spread=0.05),
            market_maker=MarketMaker(quote_spread=0.02, quote_size=5.0),
            pnl_tracker=PnLTracker(),
            random_seed=1,
        )
        simulator.run(num_steps=50, volatility=STEP_VOLATILITY)
        self.assertEqual(len(simulator.history), 50)
        self.assertEqual(simulator.get_summary()['initial_mid'], 50.0)


if __name__ == '__main__':
    unittest.main()
