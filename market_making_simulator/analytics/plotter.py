"""
Plotting utilities for market-making simulation results.

Visualizes PnL components, inventory, price movements, and trade activity.
"""

from typing import List, Dict
import matplotlib.pyplot as plt


class SimulationPlotter:
    """Creates visualizations of market-making simulation results."""
    
    def __init__(self, figsize: tuple = (14, 10)):
        """
        Initialize the plotter.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
    
    def plot_simulation(
        self,
        history: List[Dict],
        title: str = "Market-Making Simulation Results"
    ):
        """
        Create a comprehensive multi-panel visualization.
        
        Args:
            history: Simulation history from MarketSimulator.history
            title: Title for the figure
        """
        if not history:
            print("No history to plot")
            return
        
        # Extract data from history
        times = [h['time'] for h in history]
        mid_prices = [h['mid_price'] for h in history]
        inventories = [h['inventory'] for h in history]
        cash_pnl = [h['cash_pnl'] for h in history]
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Plot 1: Price over time
        ax = axes[0, 0]
        ax.plot(times, mid_prices, 'b-', linewidth=2, label='Mid Price')
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Inventory over time
        ax = axes[0, 1]
        colors = ['green' if inv >= 0 else 'red' for inv in inventories]
        ax.bar(times, inventories, color=colors, alpha=0.6, width=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('Inventory (units)')
        ax.set_title('Inventory Over Time')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cash PnL over time (cumulative)
        ax = axes[1, 0]
        colors_pnl = ['green' if p >= 0 else 'red' for p in cash_pnl]
        ax.fill_between(times, cash_pnl, alpha=0.3, color='blue')
        ax.plot(times, cash_pnl, 'b-', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('Cash PnL ($)')
        ax.set_title('Cumulative PnL Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate stats
        final_inventory = inventories[-1] if inventories else 0
        final_pnl = cash_pnl[-1] if cash_pnl else 0
        max_inventory = max(inventories) if inventories else 0
        min_inventory = min(inventories) if inventories else 0
        final_price = mid_prices[-1] if mid_prices else 0
        initial_price = mid_prices[0] if mid_prices else 0
        price_change = final_price - initial_price
        
        stats_text = f"""
SIMULATION SUMMARY

Price:
  Initial: ${initial_price:.2f}
  Final:   ${final_price:.2f}
  Change:  ${price_change:.2f} ({100*price_change/initial_price:.2f}%)

Inventory:
  Final:   {final_inventory:.0f} units
  Max:     {max_inventory:.0f} units
  Min:     {min_inventory:.0f} units

Performance:
  Total PnL: ${final_pnl:.2f}
  Num Trades: {len(history)}
"""
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        return fig
    
    def plot_pnl_decomposition(
        self,
        spread_capture: float,
        inventory_pnl: float,
        adverse_selection: float,
        title: str = "PnL Decomposition"
    ):
        """
        Create a bar chart showing PnL component breakdown.
        
        Args:
            spread_capture: Spread capture PnL
            inventory_pnl: Inventory risk PnL
            adverse_selection: Adverse selection cost
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = ['Spread Capture', 'Inventory Risk', 'Adverse Selection']
        values = [spread_capture, inventory_pnl, adverse_selection]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val:.2f}',
                   ha='center', va='bottom' if val > 0 else 'top',
                   fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('PnL ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add total PnL
        total_pnl = sum(values)
        ax.text(0.98, 0.98, f'Total PnL: ${total_pnl:.2f}',
               transform=ax.transAxes,
               fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_price_with_trades(
        self,
        history: List[Dict],
        title: str = "Price Movement with Trades"
    ):
        """
        Plot price with trade markers.
        
        Args:
            history: Simulation history
            title: Title for the plot
        """
        if not history:
            print("No history to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = [h['time'] for h in history]
        mid_prices = [h['mid_price'] for h in history]
        
        # Plot price
        ax.plot(times, mid_prices, 'b-', linewidth=2, label='Mid Price')
        
        # Mark trades if available
        buy_times = []
        buy_prices = []
        sell_times = []
        sell_prices = []
        
        for h in history:
            if 'last_trade' in h and h['last_trade']:
                trade = h['last_trade']
                if trade['side'] == 'buy':
                    buy_times.append(h['time'])
                    buy_prices.append(h['mid_price'])
                else:
                    sell_times.append(h['time'])
                    sell_prices.append(h['mid_price'])
        
        if buy_times:
            ax.scatter(buy_times, buy_prices, color='green', marker='^', 
                      s=100, label='Buy Trades', zorder=5)
        if sell_times:
            ax.scatter(sell_times, sell_prices, color='red', marker='v', 
                      s=100, label='Sell Trades', zorder=5)
        
        ax.set_xlabel('Time (steps)', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def show(self):
        """Display all open plots."""
        plt.show()
