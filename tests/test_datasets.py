"""
Tests for the two shipped calibrations.

The comparison between them is only worth reading if two things are true, and
both are easy to break by editing a number in `datasets.py`:

1. The equity calibration is the one the rest of the repository already ran on.
   Every absolute figure the README publishes for the equity dataset comes from
   `example.py`, which now builds from `US_EQUITY`. If a dimensionless field
   drifts, the published output silently stops being reproducible, so the
   derived absolute values are pinned here against the numbers `example.py`
   used to hardcode.
2. The two calibrations are identical in every dimensionless quantity. That is
   the whole experimental design: it is what lets a difference between the
   grids be attributed to market structure rather than to a spread someone
   picked. A field that differs and should not is asserted against here by
   comparing the two datasets field by field, so adding a new dimensionless
   field to `Dataset` without deciding which side of that line it falls on
   fails a test rather than quietly weakening the design.
"""

import unittest
from dataclasses import fields, replace

from market_making_simulator import (
    CRYPTO_PERP,
    DATASET_NAMES,
    DATASETS,
    SECONDS_PER_CALENDAR_YEAR,
    SECONDS_PER_TRADING_YEAR,
    US_EQUITY,
    Dataset,
    get_dataset,
)

# The absolute values `example.py` hardcoded before it built from a dataset.
PUBLISHED_EQUITY_CALIBRATION = {
    'initial_mid': 100.0,
    'reference_half_spread': 0.10,
    'quote_spread': 0.05,
    'quote_size': 10.0,
    'max_inventory': 100.0,
    'inventory_skew_factor': 0.01,
    'decay': 20.0,
    'mean_order_size': 10.0,
    'base_intensity': 0.8,
    'informed_fraction': 0.30,
}

# Fields allowed to differ between the two calibrations: the price scale, the
# labels, the horizon, and the four axes of market structure. Anything else
# differing means the geometry is no longer held fixed.
MAY_DIFFER = {
    'name', 'asset_class', 'unit', 'description',
    'initial_mid', 'quote_size', 'default_steps',
    'annual_volatility', 'seconds_per_year',
    'session_steps', 'funding_interval_steps', 'funding_rate_per_interval',
    'maker_rebate_per_unit', 'maker_rebate_bps',
}


class TestRegistry(unittest.TestCase):
    """Both calibrations are reachable by name."""

    def test_both_datasets_are_registered_under_their_own_name(self):
        self.assertEqual(set(DATASET_NAMES), {'us-equity', 'crypto-perp'})
        for name in DATASET_NAMES:
            self.assertEqual(DATASETS[name].name, name)
            self.assertIs(get_dataset(name), DATASETS[name])

    def test_an_unknown_name_names_the_alternatives(self):
        with self.assertRaises(KeyError) as caught:
            get_dataset('bitcoin')
        for name in DATASET_NAMES:
            self.assertIn(name, str(caught.exception))


class TestPublishedEquityCalibration(unittest.TestCase):
    """The equity dataset must be the configuration the README published."""

    def test_derived_values_match_the_hardcoded_ones(self):
        for attribute, expected in PUBLISHED_EQUITY_CALIBRATION.items():
            with self.subTest(attribute=attribute):
                self.assertAlmostEqual(
                    getattr(US_EQUITY, attribute), expected, places=12,
                    msg=f"{attribute} drifted away from the published run")

    def test_the_step_volatility_is_the_published_one(self):
        """25% annualised on one-second equity-session steps."""
        self.assertAlmostEqual(US_EQUITY.step_volatility, 1.0295e-4, delta=1e-7)

    def test_the_crossing_inventory_is_the_published_one(self):
        """Half a clip, which is the `cross |q| = 5` the benchmark reports."""
        self.assertAlmostEqual(US_EQUITY.crossing_inventory, 5.0, places=12)


class TestGeometryIsHeldFixed(unittest.TestCase):
    """The dimensionless calibration is shared, by design."""

    def test_only_the_permitted_fields_differ(self):
        differing = {
            field.name for field in fields(Dataset)
            if getattr(US_EQUITY, field.name) != getattr(CRYPTO_PERP, field.name)
        }
        self.assertTrue(
            differing.issubset(MAY_DIFFER),
            msg=f"these fields differ but should not: {sorted(differing - MAY_DIFFER)}",
        )

    def test_every_permitted_field_is_a_real_field(self):
        """Guards the allow-list itself against a rename."""
        self.assertTrue(MAY_DIFFER.issubset({f.name for f in fields(Dataset)}))

    def test_a_clip_is_the_same_notional_in_both(self):
        self.assertAlmostEqual(US_EQUITY.notional_per_clip,
                               CRYPTO_PERP.notional_per_clip, places=9)

    def test_the_quote_is_the_same_distance_in_relative_terms(self):
        for dataset in (US_EQUITY, CRYPTO_PERP):
            with self.subTest(dataset=dataset.name):
                # 5bp from the reservation price, in whatever price units.
                self.assertAlmostEqual(
                    dataset.quote_spread / dataset.initial_mid, 5e-4, places=12)
                # And the intensity falls by exactly 1/e over that distance, so
                # the two markets are equally hard to get filled in.
                self.assertAlmostEqual(
                    dataset.decay * dataset.quote_spread, 1.0, places=12)

    def test_both_cross_the_mid_at_the_same_fraction_of_a_clip(self):
        for dataset in (US_EQUITY, CRYPTO_PERP):
            with self.subTest(dataset=dataset.name):
                self.assertAlmostEqual(
                    dataset.crossing_inventory / dataset.quote_size,
                    0.5, places=12)


class TestPriceScaleInvariance(unittest.TestCase):
    """
    A calibration must survive being restated at a different price level.

    This is the property that makes a "second dataset" possible at all: an
    arrival decay stored as `k = 20` per dollar describes a 5bp quote on a $100
    share and a 0.008bp quote on a $100,000 contract, which are not the same
    market. Storing it in basis points and deriving k is what fixes that, and
    this test is what stops it being un-fixed.
    """

    def test_repricing_the_equity_leaves_every_dimensionless_quantity_alone(self):
        # Same instrument, quoted at a thousand times the price, with the lot
        # size scaled down to hold the notional per clip constant.
        rescaled = replace(US_EQUITY, initial_mid=100_000.0, quote_size=0.01)
        for ratio in ('quote_spread', 'reference_half_spread'):
            with self.subTest(ratio=ratio):
                self.assertAlmostEqual(
                    getattr(rescaled, ratio) / rescaled.initial_mid,
                    getattr(US_EQUITY, ratio) / US_EQUITY.initial_mid,
                    places=12,
                )
        self.assertAlmostEqual(rescaled.decay * rescaled.quote_spread,
                               US_EQUITY.decay * US_EQUITY.quote_spread,
                               places=12)
        self.assertAlmostEqual(
            rescaled.crossing_inventory / rescaled.quote_size,
            US_EQUITY.crossing_inventory / US_EQUITY.quote_size, places=12)
        self.assertAlmostEqual(rescaled.notional_per_clip,
                               US_EQUITY.notional_per_clip, places=6)


class TestCalendars(unittest.TestCase):
    """The 24/7 arithmetic, which is most of what "crypto" means here."""

    def test_the_two_datasets_use_different_calendars(self):
        self.assertEqual(US_EQUITY.seconds_per_year, SECONDS_PER_TRADING_YEAR)
        self.assertEqual(CRYPTO_PERP.seconds_per_year, SECONDS_PER_CALENDAR_YEAR)
        self.assertFalse(US_EQUITY.trades_around_the_clock)
        self.assertTrue(CRYPTO_PERP.trades_around_the_clock)

    def test_the_crypto_year_is_longer_by_the_session_ratio(self):
        self.assertAlmostEqual(
            SECONDS_PER_CALENDAR_YEAR / SECONDS_PER_TRADING_YEAR,
            (365 * 24) / (252 * 6.5), places=12)

    def test_a_higher_headline_volatility_can_still_move_less_per_second(self):
        """
        The point of stating the calendar. 55% annualised 24/7 is 2.2 times the
        equity's headline figure and a slightly smaller per-second sigma,
        because it is spread over 5.35 times as many seconds.
        """
        self.assertGreater(CRYPTO_PERP.annual_volatility,
                           2.0 * US_EQUITY.annual_volatility)
        self.assertLess(CRYPTO_PERP.step_volatility, US_EQUITY.step_volatility)

    def test_the_same_headline_on_the_wrong_calendar_overstates_sigma(self):
        """Reading a 24/7 asset through the equity calendar inflates it 2.31x."""
        misread = replace(CRYPTO_PERP, seconds_per_year=SECONDS_PER_TRADING_YEAR)
        self.assertAlmostEqual(
            misread.step_volatility / CRYPTO_PERP.step_volatility,
            (SECONDS_PER_CALENDAR_YEAR / SECONDS_PER_TRADING_YEAR) ** 0.5,
            places=9,
        )


class TestFeeDenominations(unittest.TestCase):
    """Per share against basis points of notional."""

    def test_the_equity_is_paid_per_share_and_the_perpetual_charged_on_notional(self):
        self.assertGreater(US_EQUITY.maker_rebate_per_unit, 0.0)
        self.assertEqual(US_EQUITY.maker_rebate_bps, 0.0)
        self.assertEqual(CRYPTO_PERP.maker_rebate_per_unit, 0.0)
        self.assertLess(CRYPTO_PERP.maker_rebate_bps, 0.0)

    def test_the_two_denominations_are_comparable_only_through_the_price(self):
        # $0.0020 a share on a $100 share is 0.2bp of notional.
        self.assertAlmostEqual(US_EQUITY.maker_fee_bps_equivalent(), 0.2,
                               places=9)
        self.assertAlmostEqual(CRYPTO_PERP.maker_fee_bps_equivalent(), -2.0,
                               places=9)
        # The same per-share rebate on a $10 share is worth ten times as much
        # in relative terms, which is exactly why the denomination matters.
        cheap = replace(US_EQUITY, initial_mid=10.0)
        self.assertAlmostEqual(cheap.maker_fee_bps_equivalent(), 2.0, places=9)

    def test_a_notional_fee_is_charged_per_fill_on_the_traded_value(self):
        maker = CRYPTO_PERP.build().market_maker
        # 0.01 contracts at $100,000 is $1,000 of notional; 2bp of that is 20c.
        self.assertAlmostEqual(maker.maker_fee(100_000.0, 0.01), -0.20,
                               places=12)


class TestBuildAndRun(unittest.TestCase):
    """The wiring, and that the structure reaches the objects."""

    def test_the_equity_build_carries_a_session_and_no_funding(self):
        simulator = US_EQUITY.build(random_seed=0)
        self.assertIsNone(simulator.funding_model)
        self.assertEqual(simulator.session_steps, US_EQUITY.session_steps)

    def test_the_crypto_build_carries_funding_and_no_session(self):
        simulator = CRYPTO_PERP.build(random_seed=0)
        self.assertIsNone(simulator.session_steps)
        self.assertIsNotNone(simulator.funding_model)
        self.assertEqual(simulator.funding_model.interval_steps,
                         CRYPTO_PERP.funding_interval_steps)

    def test_build_passes_the_derived_absolute_values_through(self):
        simulator = US_EQUITY.build(random_seed=0)
        self.assertAlmostEqual(simulator.market_maker.quote_spread,
                               US_EQUITY.quote_spread, places=12)
        self.assertAlmostEqual(simulator.fill_model.decay,
                               US_EQUITY.decay, places=12)
        self.assertAlmostEqual(simulator.market_state.reference_half_spread,
                               US_EQUITY.reference_half_spread, places=12)

    def test_the_kill_switch_is_only_armed_when_asked(self):
        self.assertFalse(
            US_EQUITY.build().market_maker.risk_manager.enable_kill_switch)
        armed = US_EQUITY.build(kill_switch_drawdown=5.0).market_maker
        self.assertTrue(armed.risk_manager.enable_kill_switch)
        self.assertEqual(armed.risk_manager.drawdown_limit, 5.0)

    def test_no_overlay_at_all_when_both_controls_are_off(self):
        maker = US_EQUITY.build(enable_size_throttle=False).market_maker
        self.assertIsNone(maker.risk_manager)

    def test_run_uses_the_dataset_volatility_and_horizon(self):
        _, summary = US_EQUITY.run(random_seed=1, num_steps=50)
        self.assertEqual(summary['total_steps'], 50)
        simulator, summary = US_EQUITY.run(random_seed=1)
        self.assertEqual(summary['total_steps'], US_EQUITY.default_steps)

    def test_both_datasets_run_and_reconcile(self):
        for dataset in (US_EQUITY, CRYPTO_PERP):
            with self.subTest(dataset=dataset.name):
                simulator, summary = dataset.run(random_seed=2, num_steps=800)
                self.assertGreater(summary['num_trades'], 0)
                self.assertAlmostEqual(
                    summary['gross_pnl'],
                    summary['spread_capture'] + summary['inventory_pnl'],
                    delta=1e-9)
                self.assertAlmostEqual(
                    summary['net_pnl'],
                    summary['gross_pnl'] + summary['rebates']
                    + summary['funding'] - summary['liquidation_cost'],
                    delta=1e-9)
                self.assertAlmostEqual(
                    summary['gross_pnl'],
                    simulator.market_maker.get_gross_pnl(
                        simulator.market_state.get_mid_price()),
                    delta=1e-6)

    def test_the_two_datasets_see_the_same_arrival_sequence(self):
        """
        Identical geometry means identical fill probabilities, so under a
        shared seed the two calibrations fill on exactly the same steps. That
        is the strongest possible statement that the difference between them is
        structural: the flow is not merely comparable, it is the same flow.
        """
        histories = [dataset.build(random_seed=99) for dataset in
                     (US_EQUITY, CRYPTO_PERP)]
        for simulator, dataset in zip(histories, (US_EQUITY, CRYPTO_PERP)):
            simulator.run(num_steps=600, volatility=dataset.step_volatility)

        def fill_pattern(simulator, dataset):
            return [(step['bid_fill_qty'] / dataset.quote_size > 0,
                     step['ask_fill_qty'] / dataset.quote_size > 0)
                    for step in simulator.history]

        self.assertEqual(fill_pattern(histories[0], US_EQUITY),
                         fill_pattern(histories[1], CRYPTO_PERP))


class TestValidation(unittest.TestCase):
    """Nonsensical calibrations are rejected at construction."""

    def test_rejected_fields(self):
        for field, value in [
            ('initial_mid', 0.0),
            ('quote_size', 0.0),
            ('decay_bps', 0.0),
            ('max_inventory_clips', 0.0),
            ('mean_order_size_clips', 0.0),
            ('default_steps', 0),
            ('quote_spread_bps', -1.0),
            ('reference_half_spread_bps', -1.0),
        ]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(US_EQUITY, **{field: value})

    def test_a_skewless_calibration_is_allowed_and_never_crosses(self):
        flat = replace(US_EQUITY, skew_bps_per_clip=0.0)
        self.assertEqual(flat.inventory_skew_factor, 0.0)
        self.assertEqual(flat.crossing_inventory, float('inf'))


class TestDescribe(unittest.TestCase):
    """The banner is read off the object, so it cannot describe another run."""

    def test_the_description_reports_the_dataset_it_was_called_on(self):
        for dataset in (US_EQUITY, CRYPTO_PERP):
            with self.subTest(dataset=dataset.name):
                text = dataset.describe()
                self.assertIn(dataset.name, text)
                self.assertIn(f"{dataset.default_steps} steps", text)

    def test_the_description_follows_an_edited_field(self):
        edited = replace(US_EQUITY, session_steps=None)
        self.assertIn('never closes', edited.describe())
        self.assertNotIn('never closes', US_EQUITY.describe())

    def test_hours_converts_steps_through_the_step_length(self):
        self.assertAlmostEqual(CRYPTO_PERP.hours(3600), 1.0, places=12)
        self.assertAlmostEqual(CRYPTO_PERP.hours(CRYPTO_PERP.default_steps),
                               24.0, places=12)


if __name__ == '__main__':
    unittest.main()
