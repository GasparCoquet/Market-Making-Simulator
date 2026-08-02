"""
Tests for RiskManager.

The kill-switch must fire on mark-to-market drawdown and on nothing else. The
previous version fired on net cash flow, so buying 10 units of a $100 asset
looked like a $1,000 loss and halted a strategy that had lost nothing. The
first test below is exactly that case.

Risk state changes in `update` only. `apply` is pure and merely reads the
halted flag, which is what keeps `MarketMaker.get_quotes` idempotent. The
simulator calls `MarketMaker.observe_risk` once per step, after the price move,
and these tests call it explicitly wherever they stand in for a step.
"""

import unittest

from market_making_simulator import MarketMaker, MarketState, RiskManager

from tests._helpers import run_simulation


class TestKillSwitchIsNotCashFlow(unittest.TestCase):
    """Buying inventory moves cash without losing money."""

    def setUp(self):
        self.market_state = MarketState(initial_mid=100.0, reference_half_spread=0.10)
        self.risk_manager = RiskManager(
            enable_kill_switch=True,
            drawdown_limit=25.0,
            enable_size_throttle=False,
        )
        self.maker = MarketMaker(
            quote_spread=0.05,
            quote_size=10.0,
            max_inventory=100.0,
            inventory_skew_factor=0.0,
            risk_manager=self.risk_manager,
        )

    def test_buying_ten_units_at_the_mid_does_not_halt_trading(self):
        """Net cash flow is -$1,000, PnL is zero, so the switch must stay open."""
        self.maker.execute_bid_fill(100.0, 10.0)

        self.assertAlmostEqual(self.maker.get_net_cash_flow(), -1000.0, places=12)
        self.assertAlmostEqual(self.maker.get_mark_to_market_pnl(100.0), 0.0, places=12)

        self.maker.observe_risk(100.0)
        _, bid_size, _, ask_size = self.maker.get_quotes(self.market_state)
        self.assertFalse(self.risk_manager.is_halted)
        self.assertGreater(bid_size, 0.0)
        self.assertGreater(ask_size, 0.0)

    def test_selling_ten_units_at_the_mid_does_not_halt_trading(self):
        """The mirror case: +$1,000 of cash is not a profit either."""
        self.maker.execute_ask_fill(100.0, 10.0)
        self.maker.observe_risk(100.0)
        self.maker.get_quotes(self.market_state)
        self.assertFalse(self.risk_manager.is_halted)
        self.assertAlmostEqual(self.risk_manager.peak_pnl, 0.0, places=12)

    def test_a_large_position_at_flat_pnl_is_not_a_drawdown(self):
        """Ten clips bought at the mid: $10,000 of cash out, still no drawdown."""
        for _ in range(10):
            self.maker.execute_bid_fill(100.0, 10.0)
        self.assertAlmostEqual(self.maker.get_net_cash_flow(), -10000.0, places=12)
        self.assertEqual(self.risk_manager.update(
            self.maker.get_mark_to_market_pnl(100.0)), 0.0)
        self.assertFalse(self.risk_manager.is_halted)


class TestKillSwitchFiresOnDrawdown(unittest.TestCase):
    """A real loss must halt quoting, and it must stay halted."""

    def setUp(self):
        self.market_state = MarketState(initial_mid=100.0, reference_half_spread=0.10)
        self.risk_manager = RiskManager(
            enable_kill_switch=True,
            drawdown_limit=25.0,
            enable_size_throttle=False,
        )
        self.maker = MarketMaker(
            quote_spread=0.05,
            quote_size=10.0,
            max_inventory=1000.0,
            inventory_skew_factor=0.0,
            risk_manager=self.risk_manager,
        )

    def test_a_losing_position_halts_quoting(self):
        """Long 100 units into a $0.50 fall is a $50 loss against a $25 limit."""
        for _ in range(10):
            self.maker.execute_bid_fill(100.0, 10.0)
        self.market_state.update_mid_price(99.50)

        self.assertAlmostEqual(self.maker.get_mark_to_market_pnl(99.50), -50.0, places=12)
        self.maker.observe_risk(99.50)
        self.assertTrue(self.risk_manager.is_halted)

        _, bid_size, _, ask_size = self.maker.get_quotes(self.market_state)
        self.assertEqual(bid_size, 0.0)
        self.assertEqual(ask_size, 0.0)

    def test_the_halt_survives_a_recovery(self):
        """A stop that un-trips on the next favourable tick is not a stop."""
        for _ in range(10):
            self.maker.execute_bid_fill(100.0, 10.0)
        self.market_state.update_mid_price(99.50)
        self.maker.observe_risk(99.50)

        self.market_state.update_mid_price(100.50)
        self.maker.observe_risk(100.50)
        _, bid_size, _, ask_size = self.maker.get_quotes(self.market_state)
        self.assertTrue(self.risk_manager.is_halted)
        self.assertEqual(bid_size, 0.0)
        self.assertEqual(ask_size, 0.0)

    def test_drawdown_is_measured_from_the_peak_not_from_zero(self):
        """Giving back $30 of a $100 profit is a drawdown even though PnL is positive."""
        self.risk_manager.update(100.0)
        drawdown = self.risk_manager.update(70.0)
        self.assertAlmostEqual(drawdown, 30.0, places=12)
        self.assertAlmostEqual(self.risk_manager.max_drawdown, 30.0, places=12)

    def test_a_loss_below_the_limit_does_not_halt(self):
        self.maker.execute_bid_fill(100.0, 10.0)
        self.market_state.update_mid_price(99.00)
        self.assertAlmostEqual(self.maker.get_mark_to_market_pnl(99.00), -10.0, places=12)
        self.maker.observe_risk(99.00)
        _, bid_size, _, ask_size = self.maker.get_quotes(self.market_state)
        self.assertFalse(self.risk_manager.is_halted)
        self.assertGreater(bid_size, 0.0)
        self.assertGreater(ask_size, 0.0)

    def test_reset_clears_the_halt(self):
        self.risk_manager.update(0.0)
        self.risk_manager.is_halted = True
        self.risk_manager.reset()
        self.assertFalse(self.risk_manager.is_halted)
        self.assertEqual(self.risk_manager.peak_pnl, 0.0)
        self.assertEqual(self.risk_manager.max_drawdown, 0.0)

    def test_kill_switch_is_off_by_default(self):
        """Opt-in, so an unconfigured risk manager never silently stops a run."""
        risk_manager = RiskManager()
        for pnl in [0.0, -1e6]:
            risk_manager.update(pnl)
            _, _, _, _, is_active = risk_manager.apply(
                99.95, 10.0, 100.05, 10.0, 0.0, 100.0)
            self.assertTrue(is_active)
        self.assertFalse(risk_manager.is_halted)

    def test_quoting_is_free_of_side_effects(self):
        """
        `get_quotes` must be idempotent.

        When the drawdown bookkeeping lived inside `apply`, quoting twice at the
        same state counted the same PnL into the peak twice, so the risk state
        depended on how many times a caller happened to ask for a quote.
        """
        self.maker.execute_bid_fill(100.0, 10.0)
        self.market_state.update_mid_price(99.0)
        self.maker.observe_risk(99.0)

        before = (self.risk_manager.peak_pnl,
                  self.risk_manager.max_drawdown,
                  self.risk_manager.is_halted)
        first = self.maker.get_quotes(self.market_state)
        second = self.maker.get_quotes(self.market_state)
        after = (self.risk_manager.peak_pnl,
                 self.risk_manager.max_drawdown,
                 self.risk_manager.is_halted)

        self.assertEqual(first, second)
        self.assertEqual(before, after)


class TestSizeThrottle(unittest.TestCase):
    """Quoted size shrinks as the position approaches the limit."""

    def setUp(self):
        self.risk_manager = RiskManager(enable_size_throttle=True, min_throttle=0.2)

    def _sizes(self, inventory: float) -> tuple:
        _, bid_size, _, ask_size, _ = self.risk_manager.apply(
            99.95, 10.0, 100.05, 10.0, inventory, 100.0)
        return (bid_size, ask_size)

    def test_size_is_monotonically_decreasing_in_absolute_inventory(self):
        previous = self._sizes(0.0)[0]
        for inventory in [10.0, 30.0, 50.0, 70.0]:
            current = self._sizes(inventory)[0]
            self.assertLess(current, previous)
            previous = current

    def test_throttle_is_symmetric_in_the_sign_of_the_position(self):
        self.assertEqual(self._sizes(60.0), self._sizes(-60.0))

    def test_throttle_never_falls_below_the_floor(self):
        for inventory in [100.0, 500.0]:
            self.assertAlmostEqual(self._sizes(inventory)[0], 2.0, places=12)

    def test_throttle_does_not_move_prices(self):
        bid_price, _, ask_price, _, _ = self.risk_manager.apply(
            99.95, 10.0, 100.05, 10.0, 90.0, 100.0)
        self.assertAlmostEqual(bid_price, 99.95, places=12)
        self.assertAlmostEqual(ask_price, 100.05, places=12)

    def test_throttle_can_be_disabled(self):
        risk_manager = RiskManager(enable_size_throttle=False)
        _, bid_size, _, ask_size, _ = risk_manager.apply(
            99.95, 10.0, 100.05, 10.0, 90.0, 100.0)
        self.assertEqual((bid_size, ask_size), (10.0, 10.0))

    def test_invalid_floor_rejected(self):
        with self.assertRaises(ValueError):
            RiskManager(min_throttle=1.5)


class TestKillSwitchInSimulation(unittest.TestCase):
    """End to end: once halted, the strategy stops trading for good."""

    def test_a_tight_limit_halts_the_run_and_it_never_resumes(self):
        risk_manager = RiskManager(
            enable_kill_switch=True,
            drawdown_limit=0.5,
            enable_size_throttle=False,
        )
        simulator, _ = run_simulation(1, num_steps=1000, risk_manager=risk_manager)
        self.assertTrue(risk_manager.is_halted)

        halted_from = next(
            index for index, step in enumerate(simulator.history)
            if step['bid_size'] == 0.0 and step['ask_size'] == 0.0
        )
        for step in simulator.history[halted_from:]:
            self.assertEqual(step['bid_size'], 0.0)
            self.assertEqual(step['ask_size'], 0.0)
            self.assertFalse(step['trade_occurred'])

    def test_the_reported_drawdown_is_the_one_the_stop_tested(self):
        """
        The summary's max drawdown and the risk manager's must be the same number.

        They used to be two different series: the summary measured the
        post-move mark-to-market path while the risk manager accumulated from
        the pre-move PnL it happened to see at quoting time. A reader would
        reasonably assume the stop fired against the drawdown printed in the
        waterfall, and it did not.
        """
        risk_manager = RiskManager(
            enable_kill_switch=False,
            enable_size_throttle=False,
        )
        simulator, summary = run_simulation(
            3, num_steps=800, risk_manager=risk_manager)
        self.assertGreater(summary['max_drawdown'], 0.0)
        self.assertAlmostEqual(
            summary['max_drawdown'], risk_manager.max_drawdown, places=12)

    def test_the_summary_reports_whether_the_stop_fired(self):
        risk_manager = RiskManager(
            enable_kill_switch=True,
            drawdown_limit=0.5,
            enable_size_throttle=False,
        )
        _, halted = run_simulation(1, num_steps=1000, risk_manager=risk_manager)
        _, untouched = run_simulation(1, num_steps=1000)
        self.assertTrue(halted['risk_halted'])
        self.assertFalse(untouched['risk_halted'])

    def test_a_loose_limit_leaves_the_run_untouched(self):
        risk_manager = RiskManager(
            enable_kill_switch=True,
            drawdown_limit=1e6,
            enable_size_throttle=False,
        )
        _, guarded = run_simulation(1, num_steps=1000, risk_manager=risk_manager)
        _, unguarded = run_simulation(1, num_steps=1000)
        self.assertFalse(risk_manager.is_halted)
        self.assertEqual(guarded['num_trades'], unguarded['num_trades'])
        self.assertAlmostEqual(guarded['net_pnl'], unguarded['net_pnl'], places=12)


if __name__ == '__main__':
    unittest.main()
