"""Tests for MarketSimulator class."""

import unittest
from market_making_simulator import OrderBook, MarketMaker, PnLTracker, MarketSimulator


class TestMarketSimulator(unittest.TestCase):
    """Test cases for MarketSimulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.order_book = OrderBook(initial_mid=100.0, spread=0.10)
        self.market_maker = MarketMaker(quote_spread=0.05, quote_size=10.0)
        self.pnl_tracker = PnLTracker()
        self.simulator = MarketSimulator(
            self.order_book,
            self.market_maker,
            self.pnl_tracker,
            random_seed=42
        )
    
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.current_time, 0.0)
        self.assertEqual(len(self.simulator.history), 0)
    
    def test_simulate_price_move(self):
        """Test price movement simulation."""
        initial_mid = self.order_book.get_mid_price()
        self.simulator.simulate_price_move(volatility=0.01, dt=1.0)
        
        # Price should have moved (very unlikely to stay exactly the same)
        # But we can't predict exact value due to randomness
        new_mid = self.order_book.get_mid_price()
        self.assertIsInstance(new_mid, float)
    
    def test_simulate_order_flow(self):
        """Test order flow simulation."""
        # With arrival_rate=1.0, should always get an order
        order = self.simulator.simulate_order_flow(arrival_rate=1.0)
        self.assertIn(order, ['buy', 'sell'])
        
        # With arrival_rate=0.0, should never get an order
        order = self.simulator.simulate_order_flow(arrival_rate=0.0)
        self.assertIsNone(order)
    
    def test_step(self):
        """Test single simulation step."""
        step_data = self.simulator.step(volatility=0.01, arrival_rate=0.5, dt=1.0)
        
        # Check that step returns expected data
        self.assertIn('time', step_data)
        self.assertIn('mid_price', step_data)
        self.assertIn('inventory', step_data)
        self.assertIn('total_pnl', step_data)
        
        # Time should advance
        self.assertEqual(self.simulator.current_time, 1.0)
        # History should be recorded
        self.assertEqual(len(self.simulator.history), 1)
    
    def test_run(self):
        """Test running multiple steps."""
        num_steps = 10
        history = self.simulator.run(
            num_steps=num_steps,
            volatility=0.01,
            arrival_rate=0.5,
            dt=1.0,
            verbose=False
        )
        
        # Should have recorded all steps
        self.assertEqual(len(history), num_steps)
        self.assertEqual(len(self.simulator.history), num_steps)
        
        # Time should have advanced
        self.assertEqual(self.simulator.current_time, float(num_steps))
    
    def test_get_summary(self):
        """Test summary statistics generation."""
        # Run simulation
        self.simulator.run(num_steps=20, verbose=False)
        
        summary = self.simulator.get_summary()
        
        # Check that summary contains expected fields
        self.assertIn('total_steps', summary)
        self.assertIn('final_time', summary)
        self.assertIn('initial_mid', summary)
        self.assertIn('final_mid', summary)
        self.assertIn('final_inventory', summary)
        self.assertIn('num_trades', summary)
        self.assertIn('total_pnl', summary)
        self.assertIn('spread_capture', summary)
        self.assertIn('inventory_pnl', summary)
        self.assertIn('adverse_selection', summary)
        
        # Verify some basic properties
        self.assertEqual(summary['total_steps'], 20)
        self.assertEqual(summary['final_time'], 20.0)
    
    def test_deterministic_with_seed(self):
        """Test that simulations are deterministic with same seed."""
        # First simulation
        sim1 = MarketSimulator(
            OrderBook(initial_mid=100.0),
            MarketMaker(),
            PnLTracker(),
            random_seed=123
        )
        history1 = sim1.run(num_steps=10, verbose=False)
        
        # Second simulation with same seed
        sim2 = MarketSimulator(
            OrderBook(initial_mid=100.0),
            MarketMaker(),
            PnLTracker(),
            random_seed=123
        )
        history2 = sim2.run(num_steps=10, verbose=False)
        
        # Should produce identical results
        for i in range(10):
            self.assertAlmostEqual(
                history1[i]['mid_price'],
                history2[i]['mid_price'],
                places=5
            )
            self.assertEqual(
                history1[i]['inventory'],
                history2[i]['inventory']
            )
    
    def test_print_summary(self):
        """Test that print_summary doesn't crash."""
        # Run simulation
        self.simulator.run(num_steps=10, verbose=False)
        
        # Should not raise exception
        try:
            self.simulator.print_summary()
        except Exception as e:
            self.fail(f"print_summary raised exception: {e}")


if __name__ == '__main__':
    unittest.main()
