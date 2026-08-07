"""
Tests for the volatility unit conversion.

The simulator's `volatility` argument is a per-step log-return standard
deviation. Quoting it without a time unit is how "2% volatility" ended up
meaning a $2 move per step on a $100 asset, against a 5 cent quote.
"""

import unittest

import numpy as np

from market_making_simulator import annualised_volatility, per_step_volatility
from market_making_simulator.units import (
    SECONDS_PER_CALENDAR_YEAR,
    SECONDS_PER_TRADING_YEAR,
)


class TestPerStepVolatility(unittest.TestCase):
    """Annualised in, per-step out."""

    def test_twenty_five_percent_on_one_second_steps(self):
        """The calibration the benchmarks run at."""
        self.assertAlmostEqual(per_step_volatility(0.25, 1.0), 1.03e-4, delta=1e-6)

    def test_a_one_cent_move_on_a_hundred_dollar_asset(self):
        """
        Sanity check on the regime.

        At this volatility a $100 asset moves about a cent per second, which is
        the same order as the 5 cent quote. A per-step volatility of 2% moves it
        $2 a second and no maker survives that.
        """
        self.assertAlmostEqual(100.0 * per_step_volatility(0.25, 1.0), 0.0103, delta=1e-4)

    def test_round_trip(self):
        for annual in [0.05, 0.25, 0.80]:
            for seconds in [0.5, 1.0, 60.0]:
                per_step = per_step_volatility(annual, seconds)
                self.assertAlmostEqual(
                    annualised_volatility(per_step, seconds), annual, places=12)

    def test_scales_with_the_square_root_of_the_step_length(self):
        one_second = per_step_volatility(0.25, 1.0)
        four_seconds = per_step_volatility(0.25, 4.0)
        self.assertAlmostEqual(four_seconds / one_second, 2.0, places=9)

    def test_matches_the_closed_form(self):
        steps_per_year = SECONDS_PER_TRADING_YEAR / 1.0
        self.assertAlmostEqual(
            per_step_volatility(0.25, 1.0), 0.25 / np.sqrt(steps_per_year), places=15)

    def test_zero_volatility_is_allowed(self):
        self.assertEqual(per_step_volatility(0.0, 1.0), 0.0)

    def test_trading_year_is_stated_in_seconds(self):
        """252 trading days of 6.5 hours."""
        self.assertEqual(SECONDS_PER_TRADING_YEAR, 252 * 6.5 * 3600)


class TestCalendars(unittest.TestCase):
    """
    "Annualised" means nothing until you say how many seconds are in the year.

    A cash equity trades 252 sessions of 6.5 hours. A perpetual swap trades
    every second of every day. The second year is 5.35 times longer, so the
    same headline volatility is spread over sqrt(5.35) = 2.31 times more
    standard deviations, and feeding a 24/7 asset's annualised figure through
    the equity calendar overstates its per-second sigma by exactly that factor.
    """

    def test_the_calendar_year_is_stated_in_seconds(self):
        """365 days of 24 hours, with no session to exclude."""
        self.assertEqual(SECONDS_PER_CALENDAR_YEAR, 365 * 24 * 3600)

    def test_the_trading_calendar_is_the_default(self):
        """Existing callers passed no calendar and must be unaffected."""
        self.assertEqual(
            per_step_volatility(0.25, 1.0),
            per_step_volatility(0.25, 1.0,
                                seconds_per_year=SECONDS_PER_TRADING_YEAR),
        )

    def test_the_wrong_calendar_overstates_a_24_7_asset(self):
        correct = per_step_volatility(
            0.55, 1.0, seconds_per_year=SECONDS_PER_CALENDAR_YEAR)
        misread = per_step_volatility(
            0.55, 1.0, seconds_per_year=SECONDS_PER_TRADING_YEAR)
        self.assertAlmostEqual(
            misread / correct,
            np.sqrt(SECONDS_PER_CALENDAR_YEAR / SECONDS_PER_TRADING_YEAR),
            places=12,
        )

    def test_a_higher_headline_can_be_a_smaller_per_second_move(self):
        """
        The calibration the crypto dataset uses: 55% annualised 24/7 against
        25% annualised on a session calendar. The headline is 2.2x higher and
        the per-second sigma is slightly lower.
        """
        crypto = per_step_volatility(
            0.55, 1.0, seconds_per_year=SECONDS_PER_CALENDAR_YEAR)
        equity = per_step_volatility(0.25, 1.0)
        self.assertAlmostEqual(crypto, 9.794e-5, delta=1e-8)
        self.assertAlmostEqual(equity, 1.0295e-4, delta=1e-8)
        self.assertLess(crypto, equity)

    def test_round_trip_through_a_matching_calendar(self):
        for seconds_per_year in (SECONDS_PER_TRADING_YEAR,
                                 SECONDS_PER_CALENDAR_YEAR):
            for annual in (0.10, 0.55, 1.20):
                per_step = per_step_volatility(
                    annual, 5.0, seconds_per_year=seconds_per_year)
                self.assertAlmostEqual(
                    annualised_volatility(per_step, 5.0,
                                          seconds_per_year=seconds_per_year),
                    annual, places=12)


class TestValidation(unittest.TestCase):
    """Invalid inputs are rejected up front."""

    def test_negative_annual_volatility_rejected(self):
        with self.assertRaises(ValueError):
            per_step_volatility(-0.1, 1.0)

    def test_non_positive_step_length_rejected(self):
        with self.assertRaises(ValueError):
            per_step_volatility(0.25, 0.0)
        with self.assertRaises(ValueError):
            annualised_volatility(1e-4, -1.0)

    def test_negative_per_step_volatility_rejected(self):
        with self.assertRaises(ValueError):
            annualised_volatility(-1e-4, 1.0)

    def test_non_positive_calendar_rejected(self):
        with self.assertRaises(ValueError):
            per_step_volatility(0.25, 1.0, seconds_per_year=0.0)
        with self.assertRaises(ValueError):
            annualised_volatility(1e-4, 1.0, seconds_per_year=-1.0)


if __name__ == '__main__':
    unittest.main()
