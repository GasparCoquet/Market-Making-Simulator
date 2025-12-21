#!/usr/bin/env python3
"""
Benchmark scenarios for market-making simulator.

Tests different parameter configurations to understand strategy performance
under various market conditions.
"""

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
from market_making_simulator import (
    OrderBook, MarketMaker, PnLTracker, MarketSimulator
)


@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario."""
    name: str
    quote_spread: float
    inventory_skew_factor: float
    volatility: float
    arrival_rate: float
    num_steps: int = 100


class BenchmarkRunner:
    """Runs benchmark scenarios and compares results."""
    
    def __init__(self):
        """Initialize the benchmark runner."""
        self.results: List[Dict] = []
    
    def run_scenario(self, config: ScenarioConfig, seed: int = 42) -> Dict:
        """
        Run a single benchmark scenario.
        
        Args:
            config: Scenario configuration
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with scenario results
        """
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"{'='*60}")
        print(f"  Quote Spread:        {config.quote_spread}")
        print(f"  Inventory Skew:      {config.inventory_skew_factor}")
        print(f"  Volatility:          {config.volatility*100:.1f}%")
        print(f"  Arrival Rate:        {config.arrival_rate*100:.0f}%")
        
        # Setup
        order_book = OrderBook(initial_mid=100.0, spread=0.10, 
                              depth_per_level=100.0, num_levels=5)
        market_maker = MarketMaker(
            quote_spread=config.quote_spread,
            quote_size=10.0,
            max_inventory=100.0,
            inventory_skew_factor=config.inventory_skew_factor
        )
        pnl_tracker = PnLTracker()
        simulator = MarketSimulator(order_book, market_maker, pnl_tracker, 
                                   random_seed=seed)
        
        # Run
        simulator.run(
            num_steps=config.num_steps,
            volatility=config.volatility,
            arrival_rate=config.arrival_rate,
            dt=1.0,
            verbose=False
        )
        
        # Collect results
        decomp = pnl_tracker.get_pnl_decomposition()
        num_buys, num_sells = pnl_tracker.get_trade_count()
        
        result = {
            'scenario': config.name,
            'quote_spread': config.quote_spread,
            'inventory_skew': config.inventory_skew_factor,
            'volatility': config.volatility,
            'arrival_rate': config.arrival_rate,
            'total_trades': len(pnl_tracker.trades),
            'num_buys': num_buys,
            'num_sells': num_sells,
            'final_inventory': simulator.market_maker.inventory,
            'spread_capture': decomp['spread_capture'],
            'inventory_pnl': decomp['inventory_pnl'],
            'adverse_selection': decomp['adverse_selection'],
            'total_pnl': decomp['total_pnl'],
            'final_price': simulator.order_book.mid_price,
            'price_change_pct': 100 * (simulator.order_book.mid_price - 100.0) / 100.0,
        }
        
        # Print results
        print(f"\n  Results:")
        print(f"    Total PnL:           ${result['total_pnl']:>10.2f}")
        print(f"    Spread Capture:      ${result['spread_capture']:>10.2f}")
        print(f"    Inventory Risk:      ${result['inventory_pnl']:>10.2f}")
        print(f"    Adverse Selection:   ${result['adverse_selection']:>10.2f}")
        print(f"    Total Trades:        {result['total_trades']:>10.0f}")
        print(f"    Final Inventory:     {result['final_inventory']:>10.0f}")
        print(f"    Price Change:        {result['price_change_pct']:>10.2f}%")
        
        self.results.append(result)
        return result
    
    def run_all(self, scenarios: List[ScenarioConfig]):
        """
        Run all benchmark scenarios.
        
        Args:
            scenarios: List of scenario configurations
        """
        print("\n" + "="*70)
        print("MARKET-MAKING SIMULATOR BENCHMARKS")
        print("="*70)
        
        for scenario in scenarios:
            self.run_scenario(scenario)
        
        self.print_summary()
    
    def print_summary(self):
        """Print comparison table of all results."""
        if not self.results:
            print("No results to display")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        # Summary table
        summary_cols = [
            'scenario', 'quote_spread', 'inventory_skew', 'volatility',
            'total_pnl', 'spread_capture', 'adverse_selection', 'total_trades'
        ]
        
        print("\nKey Metrics:")
        print(df[summary_cols].to_string(index=False))
        
        # Best performers
        print("\n" + "-"*70)
        print("BEST PERFORMERS:")
        print(f"  Highest PnL:         {df.loc[df['total_pnl'].idxmax(), 'scenario']} "
              f"(${df['total_pnl'].max():.2f})")
        print(f"  Best Spread Capture: {df.loc[df['spread_capture'].idxmax(), 'scenario']} "
              f"(${df['spread_capture'].max():.2f})")
        print(f"  Lowest Adverse Sel.: {df.loc[df['adverse_selection'].idxmin(), 'scenario']} "
              f"(${df['adverse_selection'].min():.2f})")
        print(f"  Most Profitable:     {df.loc[df['total_pnl'].idxmax(), 'scenario']}")
        
        # Statistics
        print("\n" + "-"*70)
        print("STATISTICS:")
        print(f"  Average PnL:         ${df['total_pnl'].mean():.2f}")
        print(f"  Std Dev PnL:         ${df['total_pnl'].std():.2f}")
        print(f"  Worst Case:          ${df['total_pnl'].min():.2f}")
        print(f"  Best Case:           ${df['total_pnl'].max():.2f}")
        
        return df


def main():
    """Run benchmark suite."""
    
    # Define benchmark scenarios
    scenarios = [
        # Baseline
        ScenarioConfig(
            name="Baseline (2% vol, 50% arrival)",
            quote_spread=0.05,
            inventory_skew_factor=0.01,
            volatility=0.02,
            arrival_rate=0.5
        ),
        
        # Volatility tests
        ScenarioConfig(
            name="Low Volatility (0.5% vol)",
            quote_spread=0.05,
            inventory_skew_factor=0.01,
            volatility=0.005,
            arrival_rate=0.5
        ),
        ScenarioConfig(
            name="High Volatility (5% vol)",
            quote_spread=0.05,
            inventory_skew_factor=0.01,
            volatility=0.05,
            arrival_rate=0.5
        ),
        
        # Spread width tests
        ScenarioConfig(
            name="Tight Spread (0.02)",
            quote_spread=0.02,
            inventory_skew_factor=0.01,
            volatility=0.02,
            arrival_rate=0.5
        ),
        ScenarioConfig(
            name="Wide Spread (0.10)",
            quote_spread=0.10,
            inventory_skew_factor=0.01,
            volatility=0.02,
            arrival_rate=0.5
        ),
        
        # Inventory skewing tests
        ScenarioConfig(
            name="No Skew (skew=0.00)",
            quote_spread=0.05,
            inventory_skew_factor=0.00,
            volatility=0.02,
            arrival_rate=0.5
        ),
        ScenarioConfig(
            name="Aggressive Skew (skew=0.05)",
            quote_spread=0.05,
            inventory_skew_factor=0.05,
            volatility=0.02,
            arrival_rate=0.5
        ),
        
        # Order arrival tests
        ScenarioConfig(
            name="Thin Market (20% arrival)",
            quote_spread=0.05,
            inventory_skew_factor=0.01,
            volatility=0.02,
            arrival_rate=0.2
        ),
        ScenarioConfig(
            name="Thick Market (80% arrival)",
            quote_spread=0.05,
            inventory_skew_factor=0.01,
            volatility=0.02,
            arrival_rate=0.8
        ),
    ]
    
    # Run benchmarks
    runner = BenchmarkRunner()
    runner.run_all(scenarios)


if __name__ == "__main__":
    main()
