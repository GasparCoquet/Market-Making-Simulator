"""
MarketSimulator implementation.

Orchestrates the simulation of market-making activity over time.
"""

from typing import List, Dict, Optional
import numpy as np
from .order_book import OrderBook


class MarketSimulator:
    """
    Simulates market-making activity over time.
    
    Generates price dynamics, simulates order flow, and tracks performance.
    """
    
    def __init__(
        self,
        order_book: OrderBook,
        market_maker,  # Strategy object
        pnl_tracker,   # Analytics object
        random_seed: Optional[int] = None
    ):
        """
        Initialize the market simulator.
        
        Args:
            order_book: Order book instance
            market_maker: Market maker strategy instance
            pnl_tracker: PnL tracker instance
            random_seed: Random seed for reproducibility
        """
        self.order_book = order_book
        self.market_maker = market_maker
        self.pnl_tracker = pnl_tracker
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.current_time = 0.0
        self.history: List[Dict] = []
        
    def simulate_price_move(self, volatility: float = 0.01, dt: float = 1.0):
        """
        Simulate a price move using geometric Brownian motion.
        
        Args:
            volatility: Price volatility (standard deviation)
            dt: Time step
        """
        current_mid = self.order_book.get_mid_price()
        price_change = np.random.normal(0, volatility * np.sqrt(dt)) * current_mid
        new_mid = current_mid + price_change
        self.order_book.update_mid_price(new_mid)
        
    def simulate_order_flow(self, arrival_rate: float = 0.5) -> Optional[str]:
        """
        Simulate arrival of market orders.
        
        Args:
            arrival_rate: Probability of order arrival
            
        Returns:
            'buy' or 'sell' if order arrived, None otherwise
        """
        if np.random.random() < arrival_rate:
            return 'buy' if np.random.random() < 0.5 else 'sell'
        return None
    
    def step(
        self,
        volatility: float = 0.01,
        arrival_rate: float = 0.5,
        dt: float = 1.0
    ) -> Dict:
        """
        Execute one simulation step.
        
        Args:
            volatility: Price volatility
            arrival_rate: Order arrival rate
            dt: Time step
            
        Returns:
            Dictionary with step results
        """
        self.current_time += dt
        
        # Get current state before any changes
        initial_mid = self.order_book.get_mid_price()
        initial_inventory = self.market_maker.get_inventory()
        
        # Update market maker quotes
        bid_price, bid_size, ask_price, ask_size = self.market_maker.get_quotes(self.order_book)
        
        # Simulate incoming market orders
        order_side = self.simulate_order_flow(arrival_rate)
        trade_occurred = False
        trade_side = None
        trade_price = None
        trade_quantity = None
        
        if order_side == 'buy' and ask_size > 0:
            # Market buy hits our ask
            trade_side = 'sell'
            trade_price = ask_price
            trade_quantity = ask_size
            self.market_maker.execute_ask_fill(ask_price, ask_size)
            self.pnl_tracker.record_trade(
                self.current_time, 'sell', ask_price, ask_size, initial_mid
            )
            trade_occurred = True
            
        elif order_side == 'sell' and bid_size > 0:
            # Market sell hits our bid
            trade_side = 'buy'
            trade_price = bid_price
            trade_quantity = bid_size
            self.market_maker.execute_bid_fill(bid_price, bid_size)
            self.pnl_tracker.record_trade(
                self.current_time, 'buy', bid_price, bid_size, initial_mid
            )
            trade_occurred = True
        
        # Simulate price movement
        self.simulate_price_move(volatility, dt)
        
        # Record inventory snapshot
        current_mid = self.order_book.get_mid_price()
        current_inventory = self.market_maker.get_inventory()
        self.pnl_tracker.record_inventory_snapshot(
            self.current_time, current_inventory, current_mid
        )
        
        # Record step history
        step_data = {
            'time': self.current_time,
            'mid_price': current_mid,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'inventory': current_inventory,
            'trade_occurred': trade_occurred,
            'trade_side': trade_side,
            'trade_price': trade_price,
            'trade_quantity': trade_quantity,
            'cash_pnl': self.market_maker.get_cash_pnl(),
            'total_pnl': self.market_maker.get_total_pnl(current_mid)
        }
        self.history.append(step_data)
        
        return step_data
    
    def run(
        self,
        num_steps: int = 100,
        volatility: float = 0.01,
        arrival_rate: float = 0.5,
        dt: float = 1.0,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Run the simulation for multiple steps.
        
        Args:
            num_steps: Number of steps to simulate
            volatility: Price volatility
            arrival_rate: Order arrival rate
            dt: Time step
            verbose: Print progress
            
        Returns:
            List of step results
        """
        for i in range(num_steps):
            step_data = self.step(volatility, arrival_rate, dt)
            
            if verbose and (i % 10 == 0 or i == num_steps - 1):
                print(f"Step {i+1}/{num_steps}: "
                      f"Mid={step_data['mid_price']:.2f}, "
                      f"Inventory={step_data['inventory']:.2f}, "
                      f"PnL={step_data['total_pnl']:.2f}")
        
        return self.history
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of the simulation.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.history:
            return {}
        
        pnl_decomp = self.pnl_tracker.get_pnl_decomposition()
        num_buys, num_sells = self.pnl_tracker.get_trade_count()
        
        final_state = self.history[-1]
        
        return {
            'total_steps': len(self.history),
            'final_time': self.current_time,
            'initial_mid': self.history[0]['mid_price'],
            'final_mid': final_state['mid_price'],
            'price_change': final_state['mid_price'] - self.history[0]['mid_price'],
            'final_inventory': final_state['inventory'],
            'num_trades': num_buys + num_sells,
            'num_buys': num_buys,
            'num_sells': num_sells,
            'cash_pnl': final_state['cash_pnl'],
            'total_pnl': final_state['total_pnl'],
            'spread_capture': pnl_decomp['spread_capture'],
            'inventory_pnl': pnl_decomp['inventory_pnl'],
            'adverse_selection': pnl_decomp['adverse_selection'],
        }
    
    def print_summary(self):
        """Print a formatted summary of the simulation."""
        summary = self.get_summary()
        
        if not summary:
            print("No simulation data available.")
            return
        
        print("\n" + "="*60)
        print("MARKET-MAKING SIMULATION SUMMARY")
        print("="*60)
        print(f"\nSimulation Details:")
        print(f"  Total Steps: {summary['total_steps']}")
        print(f"  Final Time: {summary['final_time']:.1f}")
        print(f"\nMarket Statistics:")
        print(f"  Initial Mid Price: ${summary['initial_mid']:.2f}")
        print(f"  Final Mid Price: ${summary['final_mid']:.2f}")
        print(f"  Price Change: ${summary['price_change']:.2f} "
              f"({summary['price_change']/summary['initial_mid']*100:.2f}%)")
        print(f"\nTrading Activity:")
        print(f"  Total Trades: {summary['num_trades']}")
        print(f"  Buys: {summary['num_buys']}")
        print(f"  Sells: {summary['num_sells']}")
        print(f"  Final Inventory: {summary['final_inventory']:.2f}")
        print(f"\nPnL Analysis:")
        print(f"  Total PnL: ${summary['total_pnl']:.2f}")
        print(f"  Cash PnL: ${summary['cash_pnl']:.2f}")
        print(f"\nPnL Decomposition:")
        print(f"  Spread Capture: ${summary['spread_capture']:.2f}")
        print(f"  Inventory Risk: ${summary['inventory_pnl']:.2f}")
        print(f"  Adverse Selection: ${summary['adverse_selection']:.2f}")
        print("="*60 + "\n")
