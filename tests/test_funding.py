"""
Tests for the perpetual-swap funding leg.

Three things have to hold and each of them is a way the leg could be wrong
without looking wrong:

1. The sign. A positive funding rate means longs pay shorts. Getting this
   backwards turns a cost into a subsidy and the waterfall still reconciles,
   because both sides of the identity move together.
2. The cadence. A payment settles at the end of an interval of elapsed time,
   not at the step whose index is a multiple of the interval, which is an
   off-by-one that puts the first payment one step late.
3. The composition of rate and interval. Halving the rate and halving the
   interval must leave the expected total unchanged, or the two conventions the
   venues use are not interchangeable in this model and the benchmark's
   eight-hourly control row measures nothing.
"""

import unittest

from market_making_simulator import FundingModel, MarketMaker

from tests._helpers import STEP_VOLATILITY, build_simulation


class TestPaymentSchedule(unittest.TestCase):
    """Payments settle on elapsed time, not on a raw index multiple."""

    def setUp(self):
        self.model = FundingModel(rate_per_interval=1e-4, interval_steps=4)

    def test_first_payment_lands_at_the_end_of_the_first_interval(self):
        # Steps are zero-indexed, so four elapsed steps is index 3.
        due = [i for i in range(12) if self.model.is_payment_step(i)]
        self.assertEqual(due, [3, 7, 11])

    def test_step_zero_never_pays(self):
        """No time has elapsed, so there is nothing to charge carry on."""
        self.assertFalse(self.model.is_payment_step(0))

    def test_payment_count_matches_the_schedule(self):
        for num_steps in (0, 3, 4, 7, 8, 100):
            expected = sum(1 for i in range(num_steps)
                           if self.model.is_payment_step(i))
            self.assertEqual(self.model.payments_in(num_steps), expected)

    def test_interval_must_be_a_positive_whole_number(self):
        for bad in (0, -1, 2.5):
            with self.subTest(interval=bad):
                with self.assertRaises(ValueError):
                    FundingModel(interval_steps=bad)


class TestPaymentSign(unittest.TestCase):
    """Positive rate: longs pay, shorts receive."""

    def setUp(self):
        self.model = FundingModel(rate_per_interval=1e-4, interval_steps=1)

    def test_a_long_book_pays(self):
        self.assertAlmostEqual(
            self.model.payment(inventory=10.0, mark_price=100.0),
            -0.1, places=12)

    def test_a_short_book_receives(self):
        self.assertAlmostEqual(
            self.model.payment(inventory=-10.0, mark_price=100.0),
            +0.1, places=12)

    def test_a_flat_book_neither_pays_nor_receives(self):
        self.assertEqual(self.model.payment(0.0, 100.0), 0.0)

    def test_a_negative_rate_reverses_the_flow(self):
        """A perpetual below its index pays the longs."""
        model = FundingModel(rate_per_interval=-1e-4, interval_steps=1)
        self.assertGreater(model.payment(10.0, 100.0), 0.0)

    def test_payment_scales_with_notional_not_with_size(self):
        """
        The charge is on `inventory * price`, which is the whole reason a
        per-notional convention behaves differently from a per-unit one.
        """
        cheap = self.model.payment(inventory=10.0, mark_price=100.0)
        dear = self.model.payment(inventory=10.0, mark_price=1000.0)
        self.assertAlmostEqual(dear, 10.0 * cheap, places=12)


class TestRateAndIntervalCompose(unittest.TestCase):
    """The two venue conventions must be interchangeable per unit time."""

    def test_hourly_and_eight_hourly_charge_the_same_on_a_held_position(self):
        """
        A position held flat through the whole window pays the same either way.

        This is the property the benchmark's eight-hourly row is a control for.
        Held explicitly rather than through a simulation so that a failure
        points at the arithmetic rather than at a price path.
        """
        held, price, steps = 10.0, 100.0, 24
        hourly = FundingModel(rate_per_interval=1e-5, interval_steps=1)
        eight_hourly = FundingModel(rate_per_interval=8e-5, interval_steps=8)

        totals = []
        for model in (hourly, eight_hourly):
            totals.append(sum(
                model.payment(held, price)
                for step in range(steps) if model.is_payment_step(step)
            ))
        self.assertAlmostEqual(totals[0], totals[1], places=12)
        self.assertLess(totals[0], 0.0)


class TestFundingInASimulation(unittest.TestCase):
    """The leg has to reach the waterfall and the drawdown series."""

    STEPS = 400
    INTERVAL = 50

    def _run(self, rate: float):
        simulator = build_simulation(11)
        simulator.funding_model = FundingModel(
            rate_per_interval=rate, interval_steps=self.INTERVAL)
        simulator.run(num_steps=self.STEPS, volatility=STEP_VOLATILITY)
        return simulator, simulator.get_summary()

    def test_no_funding_model_means_no_funding_term(self):
        simulator = build_simulation(11)
        simulator.run(num_steps=self.STEPS, volatility=STEP_VOLATILITY)
        summary = simulator.get_summary()
        self.assertEqual(summary['funding'], 0.0)
        self.assertTrue(all(step['funding_paid'] == 0.0
                            for step in simulator.history))

    def test_payments_settle_on_schedule(self):
        simulator, _ = self._run(1e-3)
        paid_at = [step['step'] for step in simulator.history
                   if step['funding_paid'] != 0.0]
        # Every payment step in range, minus any where the book was exactly
        # flat, which cannot happen here because a fill lands on nearly every
        # interval boundary. Assert the schedule is a subset of the due steps
        # rather than equal to it, so the test cannot fail on a flat book.
        due = {i for i in range(self.STEPS)
               if (i + 1) % self.INTERVAL == 0}
        self.assertTrue(set(paid_at).issubset(due))
        self.assertGreater(len(paid_at), 0)

    def test_the_summary_total_is_the_sum_of_the_steps(self):
        simulator, summary = self._run(1e-3)
        self.assertAlmostEqual(
            summary['funding'],
            sum(step['funding_paid'] for step in simulator.history),
            delta=1e-12,
        )

    def test_funding_enters_net_pnl_and_not_gross(self):
        """
        Gross PnL is the trading leg. Funding is a cash flow beside it, exactly
        like rebates, and putting it inside `cash` would break the identity
        between the tracker's decomposition and the maker's mark to market.
        """
        without = self._run(0.0)[1]
        with_funding = self._run(1e-2)[1]
        self.assertAlmostEqual(without['gross_pnl'], with_funding['gross_pnl'],
                               delta=1e-12)
        self.assertNotAlmostEqual(without['net_pnl'], with_funding['net_pnl'],
                                  places=6)
        self.assertAlmostEqual(
            with_funding['net_pnl'],
            with_funding['gross_pnl'] + with_funding['rebates']
            + with_funding['funding'] - with_funding['liquidation_cost'],
            delta=1e-9,
        )

    def test_the_identity_survives_funding(self):
        simulator, summary = self._run(1e-2)
        self.assertAlmostEqual(
            summary['gross_pnl'],
            summary['spread_capture'] + summary['inventory_pnl'],
            delta=1e-9,
        )
        self.assertAlmostEqual(
            summary['gross_pnl'],
            simulator.market_maker.get_gross_pnl(
                simulator.market_state.get_mid_price()),
            delta=1e-9,
        )

    def test_funding_reaches_the_risk_overlay(self):
        """
        A book bleeding funding must show it in the drawdown the kill-switch
        tests, or the stop is watching a PnL series the desk does not have.
        """
        simulator = build_simulation(11)
        simulator.funding_model = FundingModel(
            rate_per_interval=1e-2, interval_steps=self.INTERVAL)
        simulator.run(num_steps=self.STEPS, volatility=STEP_VOLATILITY)
        maker = simulator.market_maker
        final_mid = simulator.market_state.get_mid_price()
        self.assertNotEqual(maker.funding, 0.0)
        self.assertAlmostEqual(
            maker.get_mark_to_market_pnl(final_mid),
            maker.get_gross_pnl(final_mid) + maker.rebates + maker.funding,
            delta=1e-12,
        )

    def test_funding_does_not_consume_randomness(self):
        """
        The whole benchmark rests on this. If the funding leg drew from the
        generator, turning it on would shift every later arrival and the paired
        differences would be comparing two different markets.
        """
        plain = build_simulation(31)
        funded = build_simulation(31)
        funded.funding_model = FundingModel(
            rate_per_interval=1e-2, interval_steps=self.INTERVAL)
        for simulator in (plain, funded):
            simulator.run(num_steps=self.STEPS, volatility=STEP_VOLATILITY)

        fingerprint = [
            [(step['mid_price'], step['bid_fill_qty'], step['ask_fill_qty'])
             for step in simulator.history]
            for simulator in (plain, funded)
        ]
        self.assertEqual(fingerprint[0], fingerprint[1])


class TestFundingAccrualOnTheMaker(unittest.TestCase):
    """The maker holds funding outside cash, like rebates."""

    def test_accrual_accumulates_and_leaves_cash_alone(self):
        maker = MarketMaker()
        maker.execute_bid_fill(99.95, 10.0)
        cash_before = maker.cash
        maker.accrue_funding(-0.5)
        maker.accrue_funding(-0.25)
        self.assertAlmostEqual(maker.funding, -0.75, places=12)
        self.assertEqual(maker.cash, cash_before)
        self.assertAlmostEqual(maker.get_gross_pnl(100.0),
                               cash_before + 10.0 * 100.0, places=9)


if __name__ == '__main__':
    unittest.main()
