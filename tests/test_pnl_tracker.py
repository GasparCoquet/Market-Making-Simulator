"""Tests for PnLTracker class."""

import unittest
from market_making_simulator import PnLTracker


class TestPnLTracker(unittest.TestCase):
    """Test cases for PnLTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = PnLTracker()
    
    def test_initialization(self):
        """Test PnL tracker initialization."""
        self.assertEqual(len(self.tracker.trades), 0)
        self.assertEqual(len(self.tracker.inventory_snapshots), 0)
    
    def test_record_trade(self):
        """Test trade recording."""
        self.tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0)
        
        self.assertEqual(len(self.tracker.trades), 1)
        trade = self.tracker.trades[0]
        self.assertEqual(trade['side'], 'buy')
        self.assertEqual(trade['price'], 99.95)
        self.assertEqual(trade['quantity'], 10.0)
        self.assertEqual(trade['mid_price'], 100.0)
        # Buy at 99.95 vs mid 100.0: spread = 0.05
        self.assertAlmostEqual(trade['spread_vs_mid'], 0.05, places=5)
    
    def test_record_inventory_snapshot(self):
        """Test inventory snapshot recording."""
        self.tracker.record_inventory_snapshot(1.0, 10.0, 100.0)
        
        self.assertEqual(len(self.tracker.inventory_snapshots), 1)
        self.assertEqual(self.tracker.inventory_snapshots[0], (1.0, 10.0, 100.0))
    
    def test_spread_capture(self):
        """Test spread capture calculation."""
        # Buy at bid side (99.95) vs mid (100.0): capture 0.05 per unit
        self.tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0)
        # Sell at ask side (100.05) vs mid (100.0): capture 0.05 per unit
        self.tracker.record_trade(2.0, 'sell', 100.05, 10.0, 100.0)
        
        # Total spread capture: (0.05 + 0.05) * 10 = 1.0
        spread_pnl = self.tracker.get_spread_capture()
        self.assertAlmostEqual(spread_pnl, 1.0, places=2)
    
    def test_inventory_pnl_favorable(self):
        """Test inventory PnL with favorable price movement."""
        # Start with inventory of 10 at mid 100
        self.tracker.record_inventory_snapshot(1.0, 10.0, 100.0)
        # Price moves up to 101 while holding inventory
        self.tracker.record_inventory_snapshot(2.0, 10.0, 101.0)
        
        # Inventory PnL: 10 * (101 - 100) = 10.0
        inventory_pnl = self.tracker.get_inventory_pnl()
        self.assertAlmostEqual(inventory_pnl, 10.0, places=2)
    
    def test_inventory_pnl_unfavorable(self):
        """Test inventory PnL with unfavorable price movement."""
        # Start with inventory of 10 at mid 100
        self.tracker.record_inventory_snapshot(1.0, 10.0, 100.0)
        # Price moves down to 99 while holding inventory
        self.tracker.record_inventory_snapshot(2.0, 10.0, 99.0)
        
        # Inventory PnL: 10 * (99 - 100) = -10.0
        inventory_pnl = self.tracker.get_inventory_pnl()
        self.assertAlmostEqual(inventory_pnl, -10.0, places=2)
    
    def test_adverse_selection_buy(self):
        """Test adverse selection when buying before price drop."""
        # Buy at 100.0
        self.tracker.record_trade(1.0, 'buy', 100.0, 10.0, 100.0)
        # Price drops to 99.0 (adverse selection - we bought before drop)
        self.tracker.record_trade(2.0, 'sell', 99.0, 10.0, 99.0)
        
        # Adverse selection cost: 10 * 1.0 = -10.0
        adverse_selection = self.tracker.get_adverse_selection_cost()
        self.assertAlmostEqual(adverse_selection, -10.0, places=2)
    
    def test_adverse_selection_sell(self):
        """Test adverse selection when selling before price rise."""
        # Sell at 100.0
        self.tracker.record_trade(1.0, 'sell', 100.0, 10.0, 100.0)
        # Price rises to 101.0 (adverse selection - we sold before rise)
        self.tracker.record_trade(2.0, 'buy', 101.0, 10.0, 101.0)
        
        # Adverse selection cost: 10 * 1.0 = -10.0
        adverse_selection = self.tracker.get_adverse_selection_cost()
        self.assertAlmostEqual(adverse_selection, -10.0, places=2)
    
    def test_pnl_decomposition(self):
        """Test complete PnL decomposition."""
        # Record some trades
        self.tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0)
        self.tracker.record_trade(2.0, 'sell', 100.05, 10.0, 100.0)
        
        # Record inventory snapshots
        self.tracker.record_inventory_snapshot(1.0, 0.0, 100.0)
        self.tracker.record_inventory_snapshot(2.0, 10.0, 100.0)
        self.tracker.record_inventory_snapshot(3.0, 0.0, 100.0)
        
        decomp = self.tracker.get_pnl_decomposition()
        
        self.assertIn('spread_capture', decomp)
        self.assertIn('inventory_pnl', decomp)
        self.assertIn('adverse_selection', decomp)
        self.assertIn('total_pnl', decomp)
        
        # Total should equal sum of components
        total_calculated = (decomp['spread_capture'] + 
                          decomp['inventory_pnl'] + 
                          decomp['adverse_selection'])
        self.assertAlmostEqual(decomp['total_pnl'], total_calculated, places=2)
    
    def test_trade_count(self):
        """Test trade count tracking."""
        self.tracker.record_trade(1.0, 'buy', 99.95, 10.0, 100.0)
        self.tracker.record_trade(2.0, 'buy', 99.90, 10.0, 100.0)
        self.tracker.record_trade(3.0, 'sell', 100.05, 10.0, 100.0)
        
        num_buys, num_sells = self.tracker.get_trade_count()
        self.assertEqual(num_buys, 2)
        self.assertEqual(num_sells, 1)


if __name__ == '__main__':
    unittest.main()
