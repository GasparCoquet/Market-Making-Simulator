"""Tests for OrderBook class."""

import unittest
from market_making_simulator import OrderBook


class TestOrderBook(unittest.TestCase):
    """Test cases for OrderBook."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.order_book = OrderBook(
            initial_mid=100.0,
            spread=0.10,
            depth_per_level=100.0,
            num_levels=5
        )
    
    def test_initialization(self):
        """Test order book initialization."""
        self.assertEqual(self.order_book.get_mid_price(), 100.0)
        self.assertEqual(self.order_book.spread, 0.10)
        self.assertEqual(self.order_book.depth_per_level, 100.0)
        self.assertEqual(self.order_book.num_levels, 5)
    
    def test_best_bid_ask(self):
        """Test best bid and ask prices."""
        self.assertEqual(self.order_book.get_best_bid(), 99.90)
        self.assertEqual(self.order_book.get_best_ask(), 100.10)
    
    def test_bid_ask_spread(self):
        """Test bid-ask spread calculation."""
        self.assertEqual(self.order_book.get_bid_ask_spread(), 0.20)
    
    def test_depth_at_level(self):
        """Test depth at different levels."""
        bid_depth, ask_depth = self.order_book.get_depth_at_level(0)
        self.assertEqual(bid_depth, 100.0)
        self.assertEqual(ask_depth, 100.0)
        
        # Out of range level
        bid_depth, ask_depth = self.order_book.get_depth_at_level(10)
        self.assertEqual(bid_depth, 0.0)
        self.assertEqual(ask_depth, 0.0)
    
    def test_update_mid_price(self):
        """Test mid price update."""
        self.order_book.update_mid_price(101.0)
        self.assertEqual(self.order_book.get_mid_price(), 101.0)
        self.assertEqual(self.order_book.get_best_bid(), 100.90)
        self.assertEqual(self.order_book.get_best_ask(), 101.10)
    
    def test_execute_market_buy(self):
        """Test market buy execution."""
        initial_mid = self.order_book.get_mid_price()
        price, qty = self.order_book.execute_market_buy(50.0)
        
        self.assertEqual(price, 100.10)  # Executed at ask
        self.assertEqual(qty, 50.0)
        # Mid should not move for small orders
        self.assertEqual(self.order_book.get_mid_price(), initial_mid)
    
    def test_execute_market_sell(self):
        """Test market sell execution."""
        initial_mid = self.order_book.get_mid_price()
        price, qty = self.order_book.execute_market_sell(50.0)
        
        self.assertEqual(price, 99.90)  # Executed at bid
        self.assertEqual(qty, 50.0)
        # Mid should not move for small orders
        self.assertEqual(self.order_book.get_mid_price(), initial_mid)
    
    def test_execute_large_market_buy_impact(self):
        """Test price impact of large market buy."""
        initial_mid = self.order_book.get_mid_price()
        price, qty = self.order_book.execute_market_buy(200.0)
        
        # Should execute available quantity
        self.assertEqual(qty, 100.0)
        # Price should move up
        self.assertGreater(self.order_book.get_mid_price(), initial_mid)
    
    def test_execute_large_market_sell_impact(self):
        """Test price impact of large market sell."""
        initial_mid = self.order_book.get_mid_price()
        price, qty = self.order_book.execute_market_sell(200.0)
        
        # Should execute available quantity
        self.assertEqual(qty, 100.0)
        # Price should move down
        self.assertLess(self.order_book.get_mid_price(), initial_mid)


if __name__ == '__main__':
    unittest.main()
