"""
Tests for the RiskManager module.

Tests kill-switch, size throttling, and combined risk controls.
"""

import unittest
from market_making_simulator.risk import RiskManager


class TestRiskManager(unittest.TestCase):
    """Unit tests for RiskManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.rm_default = RiskManager()
        self.rm_kill_switch = RiskManager(enable_kill_switch=True, drawdown_limit=50.0)
        self.rm_throttle = RiskManager(enable_size_throttle=True, min_throttle=0.2)
        self.rm_combined = RiskManager(
            enable_kill_switch=True,
            drawdown_limit=50.0,
            enable_size_throttle=True,
            min_throttle=0.3,
        )

    def test_initialization(self):
        """Test RiskManager initialization."""
        self.assertEqual(self.rm_default.enable_kill_switch, False)
        self.assertIsNone(self.rm_default.drawdown_limit)
        self.assertEqual(self.rm_default.enable_size_throttle, True)
        self.assertEqual(self.rm_default.min_throttle, 0.2)

    def test_no_controls_passes_through(self):
        """Test that disabled controls pass quotes through unchanged."""
        rm = RiskManager(enable_kill_switch=False, enable_size_throttle=False)
        bid_p, bid_s, ask_p, ask_s, is_active = rm.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=0.0,
            max_inventory=100.0,
            cash_pnl=0.0,
        )
        self.assertEqual(bid_p, 99.0)
        self.assertEqual(bid_s, 10.0)
        self.assertEqual(ask_p, 101.0)
        self.assertEqual(ask_s, 10.0)
        self.assertTrue(is_active)

    def test_kill_switch_disabled_default(self):
        """Test that kill-switch is disabled by default."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_default.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=0.0,
            max_inventory=100.0,
            cash_pnl=-100.0,
        )
        self.assertTrue(is_active)

    def test_kill_switch_triggers_on_drawdown(self):
        """Test that kill-switch triggers when cash PnL breaches drawdown limit."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_kill_switch.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=0.0,
            max_inventory=100.0,
            cash_pnl=-51.0,
        )
        self.assertFalse(is_active)
        self.assertEqual(bid_s, 0.0)
        self.assertEqual(ask_s, 0.0)

    def test_kill_switch_does_not_trigger_within_limit(self):
        """Test that kill-switch doesn't trigger when PnL is above limit."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_kill_switch.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=0.0,
            max_inventory=100.0,
            cash_pnl=-25.0,
        )
        self.assertTrue(is_active)

    def test_size_throttle_zero_inventory(self):
        """Test size throttling with zero inventory (no scaling)."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_throttle.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=0.0,
            max_inventory=100.0,
            cash_pnl=0.0,
        )
        # With zero inventory, scaling factor = 1 - (0 / 100) = 1.0
        self.assertEqual(bid_s, 10.0)
        self.assertEqual(ask_s, 10.0)

    def test_size_throttle_at_max_inventory(self):
        """Test size throttling at max inventory."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_throttle.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=100.0,  # At max
            max_inventory=100.0,
            cash_pnl=0.0,
        )
        # With inventory=max, scaling factor = max(min_throttle, 1 - 1.0) = min_throttle
        expected_size = 10.0 * 0.2
        self.assertAlmostEqual(bid_s, expected_size, places=5)
        self.assertAlmostEqual(ask_s, expected_size, places=5)

    def test_size_throttle_half_max_inventory(self):
        """Test size throttling at half max inventory."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_throttle.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=50.0,  # Half max
            max_inventory=100.0,
            cash_pnl=0.0,
        )
        # scaling factor = max(0.2, 1 - 0.5) = 0.5
        expected_size = 10.0 * 0.5
        self.assertAlmostEqual(bid_s, expected_size, places=5)
        self.assertAlmostEqual(ask_s, expected_size, places=5)

    def test_size_throttle_negative_inventory(self):
        """Test size throttling with negative inventory (short position)."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_throttle.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=-75.0,  # Short 75
            max_inventory=100.0,
            cash_pnl=0.0,
        )
        # scaling factor = max(0.2, 1 - 0.75) = 0.25
        expected_size = 10.0 * 0.25
        self.assertAlmostEqual(bid_s, expected_size, places=5)
        self.assertAlmostEqual(ask_s, expected_size, places=5)

    def test_combined_kill_switch_and_throttle(self):
        """Test combined kill-switch and throttling."""
        # Kill-switch should take precedence
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_combined.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=50.0,
            max_inventory=100.0,
            cash_pnl=-51.0,
        )
        self.assertFalse(is_active)
        self.assertEqual(bid_s, 0.0)
        self.assertEqual(ask_s, 0.0)

    def test_combined_active_with_throttle(self):
        """Test combined controls when kill-switch is not triggered."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_combined.apply(
            bid_price=99.0,
            bid_size=10.0,
            ask_price=101.0,
            ask_size=10.0,
            inventory=50.0,
            max_inventory=100.0,
            cash_pnl=-25.0,
        )
        self.assertTrue(is_active)
        # Throttle factor: max(0.3, 1 - 0.5) = 0.5
        expected_size = 10.0 * 0.5
        self.assertAlmostEqual(bid_s, expected_size, places=5)
        self.assertAlmostEqual(ask_s, expected_size, places=5)

    def test_prices_unchanged(self):
        """Test that prices are never modified by risk controls."""
        bid_p, bid_s, ask_p, ask_s, is_active = self.rm_combined.apply(
            bid_price=99.5,
            bid_size=10.0,
            ask_price=100.5,
            ask_size=10.0,
            inventory=75.0,
            max_inventory=100.0,
            cash_pnl=-30.0,
        )
        self.assertEqual(bid_p, 99.5)
        self.assertEqual(ask_p, 100.5)


if __name__ == "__main__":
    unittest.main()
