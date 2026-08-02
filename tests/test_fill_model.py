"""
Tests for FillModel.

Two properties carry the whole simulator. First, the fill probability must fall
as the quote moves away from the mid: without that, PnL is linear in the quoted
spread and the optimal quote is infinitely wide. Second, `sample_arrivals` must
consume the same number of uniforms on every branch, which is what lets common
random numbers line up across configurations.
"""

import unittest

import numpy as np

from market_making_simulator import FillModel
from market_making_simulator.engine.fill_model import DRAWS_PER_STEP


def _state_after_call(
    informed_fraction: float,
    bid_distance: float,
    ask_distance: float,
    next_log_return: float,
    seed: int = 12345,
) -> dict:
    """
    Run one `sample_arrivals` call and return the generator state afterwards.

    Two calls that consume the same number of draws leave a PCG64 generator in
    the same state, so comparing states across branches is a direct test of
    draw-count invariance.
    """
    rng = np.random.default_rng(seed)
    model = FillModel(informed_fraction=informed_fraction)
    model.sample_arrivals(
        rng,
        bid_distance=bid_distance,
        ask_distance=ask_distance,
        dt=1.0,
        next_log_return=next_log_return,
    )
    return rng.bit_generator.state


class TestIntensity(unittest.TestCase):
    """Quoting further out must fill less often."""

    def setUp(self):
        self.model = FillModel(base_intensity=0.8, decay=20.0)

    def test_intensity_is_strictly_decreasing_in_distance(self):
        distances = [-0.05, 0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0]
        intensities = [self.model.intensity(d) for d in distances]
        for earlier, later in zip(intensities, intensities[1:]):
            self.assertLess(later, earlier)

    def test_fill_probability_is_strictly_decreasing_in_distance(self):
        distances = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0]
        probabilities = [self.model.fill_probability(d) for d in distances]
        for earlier, later in zip(probabilities, probabilities[1:]):
            self.assertLess(later, earlier)

    def test_fill_probability_stays_in_the_unit_interval(self):
        """Includes quotes skewed through the mid, where the raw intensity blows up."""
        for distance in [-50.0, -1.0, -0.1, 0.0, 0.1, 1.0, 50.0]:
            probability = self.model.fill_probability(distance)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
            self.assertTrue(np.isfinite(probability))

    def test_intensity_at_the_mid_is_the_base_intensity(self):
        self.assertAlmostEqual(self.model.intensity(0.0), 0.8, places=12)

    def test_fill_probability_matches_the_closed_form(self):
        """P = 1 - exp(-A * exp(-k * delta) * dt)."""
        expected = 1.0 - np.exp(-0.8 * np.exp(-20.0 * 0.05) * 1.0)
        self.assertAlmostEqual(self.model.fill_probability(0.05, 1.0), expected, places=12)

    def test_longer_step_fills_more_often(self):
        self.assertLess(
            self.model.fill_probability(0.05, dt=0.5),
            self.model.fill_probability(0.05, dt=2.0),
        )

    def test_zero_intensity_never_fills(self):
        model = FillModel(base_intensity=0.0)
        rng = np.random.default_rng(0)
        for _ in range(200):
            bid, ask = model.sample_arrivals(rng, 0.05, 0.05, 1.0, 0.0)
            self.assertEqual(bid, 0.0)
            self.assertEqual(ask, 0.0)


class TestDrawCountInvariance(unittest.TestCase):
    """Every branch must consume exactly DRAWS_PER_STEP uniforms."""

    def test_all_branches_leave_the_generator_in_the_same_state(self):
        # Uninformed with near-certain fills, uninformed with near-zero fills,
        # informed ahead of a rise, informed ahead of a fall.
        branches = [
            (0.0, 0.0, 0.0, 0.001),
            (0.0, 5.0, 5.0, 0.001),
            (1.0, 0.0, 0.0, 0.001),
            (1.0, 0.0, 0.0, -0.001),
            (1.0, 5.0, 5.0, 0.001),
            (1.0, 5.0, 5.0, -0.001),
        ]
        states = [_state_after_call(*branch) for branch in branches]
        for state in states[1:]:
            self.assertEqual(state, states[0])

    def test_consumption_equals_five_uniforms(self):
        reference = np.random.default_rng(12345)
        reference.random(DRAWS_PER_STEP)
        self.assertEqual(
            _state_after_call(0.0, 0.05, 0.05, 0.001),
            reference.bit_generator.state,
        )

    def test_declared_draw_count_is_five(self):
        self.assertEqual(DRAWS_PER_STEP, 5)


class TestInformedFlow(unittest.TestCase):
    """Informed arrivals are one-sided and trade with the coming move."""

    def test_informed_flow_only_lifts_the_ask_before_a_rise(self):
        model = FillModel(informed_fraction=1.0)
        rng = np.random.default_rng(7)
        saw_ask_fill = False
        for _ in range(300):
            bid, ask = model.sample_arrivals(rng, 0.02, 0.02, 1.0, next_log_return=0.001)
            self.assertEqual(bid, 0.0)
            saw_ask_fill = saw_ask_fill or ask > 0.0
        self.assertTrue(saw_ask_fill)

    def test_informed_flow_only_hits_the_bid_before_a_fall(self):
        model = FillModel(informed_fraction=1.0)
        rng = np.random.default_rng(7)
        saw_bid_fill = False
        for _ in range(300):
            bid, ask = model.sample_arrivals(rng, 0.02, 0.02, 1.0, next_log_return=-0.001)
            self.assertEqual(ask, 0.0)
            saw_bid_fill = saw_bid_fill or bid > 0.0
        self.assertTrue(saw_bid_fill)

    def test_uninformed_flow_reaches_both_sides(self):
        model = FillModel(informed_fraction=0.0)
        rng = np.random.default_rng(11)
        bid_fills = 0
        ask_fills = 0
        for _ in range(400):
            bid, ask = model.sample_arrivals(rng, 0.02, 0.02, 1.0, next_log_return=0.001)
            bid_fills += bid > 0.0
            ask_fills += ask > 0.0
        self.assertGreater(bid_fills, 0)
        self.assertGreater(ask_fills, 0)

    def test_uninformed_flow_is_balanced_across_sides(self):
        """Symmetric quotes and no informed flow means no directional bias."""
        model = FillModel(informed_fraction=0.0)
        rng = np.random.default_rng(3)
        bid_fills = 0
        ask_fills = 0
        trials = 4000
        for _ in range(trials):
            bid, ask = model.sample_arrivals(rng, 0.05, 0.05, 1.0, next_log_return=0.0)
            bid_fills += bid > 0.0
            ask_fills += ask > 0.0
        probability = model.fill_probability(0.05, 1.0)
        # Binomial standard error on the difference of two independent counts.
        standard_error = np.sqrt(2.0 * trials * probability * (1.0 - probability))
        self.assertLess(abs(bid_fills - ask_fills), 3.0 * standard_error)


def _saturated_arrival_sizes(mean_order_size: float, calls: int, seed: int) -> np.ndarray:
    """
    Sample arrival sizes through the public API.

    The intensity is set high enough that both sides fill on every call, so the
    only randomness left in the returned sizes is the size draw itself. Going
    through `sample_arrivals` rather than the private inverse-CDF keeps the test
    tied to observable behaviour, so renaming an internal helper cannot break it.

    Args:
        mean_order_size: Mean of the exponential size distribution
        calls: Number of `sample_arrivals` calls, each yielding two sizes
        seed: Seed for the generator

    Returns:
        Array of 2 * `calls` arrival sizes.
    """
    model = FillModel(base_intensity=1e6, decay=0.0, mean_order_size=mean_order_size)
    rng = np.random.default_rng(seed)
    sizes = np.empty(2 * calls)
    for index in range(calls):
        bid, ask = model.sample_arrivals(rng, 0.0, 0.0, 1.0, 0.0)
        sizes[2 * index] = bid
        sizes[2 * index + 1] = ask
    return sizes


class TestOrderSizes(unittest.TestCase):
    """Sizes are exponential, so an arrival can be smaller than our clip."""

    def test_arrival_sizes_have_the_requested_mean(self):
        sizes = _saturated_arrival_sizes(10.0, calls=10000, seed=5)
        standard_error = sizes.std(ddof=1) / np.sqrt(sizes.size)
        self.assertLess(abs(sizes.mean() - 10.0), 3.0 * standard_error)

    def test_arrival_sizes_are_exponentially_distributed(self):
        """
        An exponential puts 1 - exp(-1) of its mass below its own mean.

        A uniform or a constant size would fail this, and a constant size is
        what removes partial fills from the simulation.
        """
        sizes = _saturated_arrival_sizes(10.0, calls=10000, seed=6)
        share_below_mean = float((sizes < 10.0).mean())
        expected = 1.0 - np.exp(-1.0)
        standard_error = np.sqrt(expected * (1.0 - expected) / sizes.size)
        self.assertLess(abs(share_below_mean - expected), 3.0 * standard_error)

    def test_arrival_sizes_are_finite_and_non_negative(self):
        """Covers the guard on the log at the top of the unit interval."""
        sizes = _saturated_arrival_sizes(10.0, calls=5000, seed=7)
        self.assertTrue(np.all(np.isfinite(sizes)))
        self.assertGreaterEqual(sizes.min(), 0.0)

    def test_mean_order_size_scales_the_arrivals(self):
        small = _saturated_arrival_sizes(2.0, calls=4000, seed=8)
        large = _saturated_arrival_sizes(20.0, calls=4000, seed=8)
        self.assertAlmostEqual(large.mean() / small.mean(), 10.0, delta=1e-9)

    def test_arrivals_smaller_than_a_ten_lot_are_common(self):
        """This is what produces partial fills once the size is capped at our clip."""
        model = FillModel(mean_order_size=10.0)
        rng = np.random.default_rng(9)
        small = 0
        total = 0
        for _ in range(500):
            bid, ask = model.sample_arrivals(rng, 0.02, 0.02, 1.0, 0.0)
            for size in (bid, ask):
                if size > 0.0:
                    total += 1
                    small += size < 10.0
        self.assertGreater(total, 0)
        self.assertGreater(small, 0)


class TestConstruction(unittest.TestCase):
    """Invalid configurations are rejected up front."""

    def test_negative_intensity_rejected(self):
        with self.assertRaises(ValueError):
            FillModel(base_intensity=-1.0)

    def test_negative_decay_rejected(self):
        with self.assertRaises(ValueError):
            FillModel(decay=-1.0)

    def test_non_positive_mean_size_rejected(self):
        with self.assertRaises(ValueError):
            FillModel(mean_order_size=0.0)

    def test_informed_fraction_outside_unit_interval_rejected(self):
        with self.assertRaises(ValueError):
            FillModel(informed_fraction=1.5)
        with self.assertRaises(ValueError):
            FillModel(informed_fraction=-0.1)


if __name__ == '__main__':
    unittest.main()
