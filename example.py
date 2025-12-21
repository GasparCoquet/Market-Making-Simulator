#!/usr/bin/env python3
"""
Example usage of the Market-Making Simulator.

Demonstrates how to set up and run a market-making simulation with PnL analysis.
"""

from market_making_simulator import (
    OrderBook, MarketMaker, PnLTracker, MarketSimulator, SimulationPlotter
)


def main():
    """Run a basic market-making simulation."""
    
    print("Market-Making Simulator Example")
    print("=" * 60)
    
    # Initialize components
    print("\nInitializing simulation components...")
    
    # Create order book with initial parameters
    order_book = OrderBook(
        initial_mid=100.0,      # Starting mid price
        spread=0.10,            # Half-spread (10 cents)
        depth_per_level=100.0,  # 100 units at each level
        num_levels=5            # 5 price levels
    )
    print(f"  Order Book: {order_book}")
    
    # Create market maker
    market_maker = MarketMaker(
        quote_spread=0.05,          # Quote 5 cents from mid
        quote_size=10.0,            # Quote 10 units per side
        max_inventory=100.0,        # Max position of 100 units
        inventory_skew_factor=0.01  # Skew quotes by 1 cent per 100 inventory
    )
    print(f"  Market Maker: {market_maker}")
    
    # Create PnL tracker
    pnl_tracker = PnLTracker()
    print(f"  PnL Tracker: {pnl_tracker}")
    
    # Create simulator
    simulator = MarketSimulator(
        order_book=order_book,
        market_maker=market_maker,
        pnl_tracker=pnl_tracker,
        random_seed=42  # For reproducibility
    )
    
    # Run simulation
    print("\nRunning simulation...")
    print("-" * 60)
    
    simulator.run(
        num_steps=100,          # Run for 100 time steps
        volatility=0.02,        # 2% volatility
        arrival_rate=0.5,       # 50% chance of order per step
        dt=1.0,                 # 1 time unit per step
        verbose=True            # Print progress
    )
    
    # Print detailed summary
    simulator.print_summary()
    
    # Create visualizations
    print("\nGenerating plots...")
    plotter = SimulationPlotter(figsize=(14, 10))
    
    # Plot 1: Main simulation results (4-panel)
    plotter.plot_simulation(simulator.history, 
                           title="Market-Making Simulation: Price, Inventory & PnL")
    
    # Plot 2: PnL decomposition
    decomp = pnl_tracker.get_pnl_decomposition()
    plotter.plot_pnl_decomposition(
        spread_capture=decomp['spread_capture'],
        inventory_pnl=decomp['inventory_pnl'],
        adverse_selection=decomp['adverse_selection'],
        title="PnL Attribution Breakdown"
    )
    
    # Plot 3: Price with trade markers
    plotter.plot_price_with_trades(simulator.history,
                                   title="Price Movement with Buy/Sell Trades")
    
    plotter.show()
    
    # Show some example quotes at different inventory levels
    print("\nExample Quotes at Different Inventory Levels:")
    print("-" * 60)
    
    # Reset for demonstration
    test_order_book = OrderBook(initial_mid=100.0, spread=0.10)
    test_mm = MarketMaker(quote_spread=0.05, inventory_skew_factor=0.01)
    
    for inv in [0, 50, -50]:
        test_mm.inventory = inv
        bid, bid_sz, ask, ask_sz = test_mm.get_quotes(test_order_book)
        print(f"  Inventory={inv:4.0f}: Bid=${bid:.2f} x {bid_sz:.0f}  "
              f"Ask=${ask:.2f} x {ask_sz:.0f}")


if __name__ == "__main__":
    main()

