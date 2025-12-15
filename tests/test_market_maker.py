"""Tests for MarketMaker class."""

import unittest
from market_making_simulator import OrderBook, MarketMaker


class TestMarketMaker(unittest.TestCase):
    """Test cases for MarketMaker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.order_book = OrderBook(initial_mid=100.0, spread=0.10)
        self.market_maker = MarketMaker(
            quote_spread=0.05,
            quote_size=10.0,
            max_inventory=100.0,
            inventory_skew_factor=0.01
        )
    
    def test_initialization(self):
        """Test market maker initialization."""
        self.assertEqual(self.market_maker.get_inventory(), 0.0)
        self.assertEqual(self.market_maker.get_cash_pnl(), 0.0)
    
    def test_get_quotes_zero_inventory(self):
        """Test quote generation with zero inventory."""
        bid, bid_sz, ask, ask_sz = self.market_maker.get_quotes(self.order_book)
        
        # Should quote around mid with no skew
        self.assertEqual(bid, 99.95)  # 100.0 - 0.05
        self.assertEqual(ask, 100.05)  # 100.0 + 0.05
        self.assertEqual(bid_sz, 10.0)
        self.assertEqual(ask_sz, 10.0)
    
    def test_get_quotes_positive_inventory(self):
        """Test quote generation with positive inventory (long)."""
        self.market_maker.inventory = 50.0
        bid, bid_sz, ask, ask_sz = self.market_maker.get_quotes(self.order_book)
        
        # Should skew quotes to encourage selling
        # Skew = 50 * 0.01 = 0.5
        self.assertEqual(bid, 99.45)  # 100.0 - 0.05 - 0.5
        self.assertEqual(ask, 100.55)  # 100.0 + 0.05 + 0.5
        self.assertEqual(bid_sz, 10.0)
        self.assertEqual(ask_sz, 10.0)
    
    def test_get_quotes_negative_inventory(self):
        """Test quote generation with negative inventory (short)."""
        self.market_maker.inventory = -50.0
        bid, bid_sz, ask, ask_sz = self.market_maker.get_quotes(self.order_book)
        
        # Should skew quotes to encourage buying
        # Skew = -50 * 0.01 = -0.5
        self.assertEqual(bid, 100.45)  # 100.0 - 0.05 - (-0.5)
        self.assertEqual(ask, 99.55)   # 100.0 + 0.05 + (-0.5)
        self.assertEqual(bid_sz, 10.0)
        self.assertEqual(ask_sz, 10.0)
    
    def test_get_quotes_max_inventory(self):
        """Test quote generation at max inventory."""
        self.market_maker.inventory = 95.0  # Close to max
        bid, bid_sz, ask, ask_sz = self.market_maker.get_quotes(self.order_book)
        
        # Should not quote bid (would exceed max inventory)
        self.assertEqual(bid_sz, 0.0)
        # Should still quote ask
        self.assertEqual(ask_sz, 10.0)
    
    def test_execute_bid_fill(self):
        """Test bid fill execution (we buy)."""
        self.market_maker.execute_bid_fill(99.95, 10.0)
        
        self.assertEqual(self.market_maker.get_inventory(), 10.0)
        self.assertEqual(self.market_maker.total_buy_value, 999.50)
        self.assertEqual(self.market_maker.total_bought, 10.0)
    
    def test_execute_ask_fill(self):
        """Test ask fill execution (we sell)."""
        self.market_maker.execute_ask_fill(100.05, 10.0)
        
        self.assertEqual(self.market_maker.get_inventory(), -10.0)
        self.assertEqual(self.market_maker.total_sell_value, 1000.50)
        self.assertEqual(self.market_maker.total_sold, 10.0)
    
    def test_cash_pnl_round_trip(self):
        """Test cash PnL from a round trip trade."""
        # Buy at 99.95
        self.market_maker.execute_bid_fill(99.95, 10.0)
        # Sell at 100.05
        self.market_maker.execute_ask_fill(100.05, 10.0)
        
        # Should have zero inventory
        self.assertEqual(self.market_maker.get_inventory(), 0.0)
        # Should have positive PnL from spread
        expected_pnl = (100.05 - 99.95) * 10.0
        self.assertAlmostEqual(self.market_maker.get_cash_pnl(), expected_pnl, places=2)
    
    def test_total_pnl_with_inventory(self):
        """Test total PnL including unrealized."""
        # Buy at 99.95
        self.market_maker.execute_bid_fill(99.95, 10.0)
        
        # Calculate total PnL at current mid of 100.0
        total_pnl = self.market_maker.get_total_pnl(100.0)
        
        # Cash PnL is negative (we bought)
        cash_pnl = self.market_maker.get_cash_pnl()
        self.assertEqual(cash_pnl, -999.50)
        
        # Unrealized PnL should be positive (inventory worth more)
        # Inventory value at mid: 10 * 100 = 1000
        # Cost: 999.50
        # Unrealized: 1000 - 999.50 = 0.50
        self.assertAlmostEqual(total_pnl, 0.50, places=2)
    
    def test_average_prices(self):
        """Test average buy and sell price calculations."""
        # No trades yet
        self.assertIsNone(self.market_maker.get_average_buy_price())
        self.assertIsNone(self.market_maker.get_average_sell_price())
        
        # Execute some trades
        self.market_maker.execute_bid_fill(99.90, 10.0)
        self.market_maker.execute_bid_fill(100.00, 10.0)
        self.market_maker.execute_ask_fill(100.10, 5.0)
        
        # Average buy: (99.90*10 + 100.00*10) / 20 = 99.95
        self.assertAlmostEqual(self.market_maker.get_average_buy_price(), 99.95, places=2)
        # Average sell: 100.10
        self.assertAlmostEqual(self.market_maker.get_average_sell_price(), 100.10, places=2)


if __name__ == '__main__':
    unittest.main()
