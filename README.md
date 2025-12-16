# Market-Making Simulator

A comprehensive Python-based market-making simulator with a synthetic limit order book. This simulator implements bid/ask quoting, inventory management, and detailed PnL decomposition (spread capture, inventory risk, adverse selection) to study liquidity provision and market microstructure.

## 🎯 Purpose

This simulator is designed to demonstrate understanding of:
- **Market microstructure** - Order book dynamics, bid-ask spreads, liquidity depth
- **Market-making strategies** - Quote placement, inventory management, risk control
- **PnL attribution** - Decomposing profits into spread capture, inventory risk, and adverse selection
- **Quantitative trading** - Understanding the challenges of providing liquidity in financial markets

## 🚀 Features

### Core Components

1. **Simulated Order Book** (`OrderBook`)
   - Configurable mid price, spread, and depth
   - Multiple price levels
   - Price impact simulation for large orders
   - Market order execution

2. **Market Maker** (`MarketMaker`)
   - Dynamic bid/ask quote generation around mid price
   - Inventory tracking with position limits
   - Inventory-based quote skewing (lean against the wind)
   - Average price tracking for buys and sells

3. **PnL Tracker** (`PnLTracker`)
   - **Spread Capture**: PnL from bid-ask spread
   - **Inventory Risk**: PnL from price moves while holding inventory
   - **Adverse Selection**: Cost of trading with informed traders
   - Trade-by-trade recording and analysis

4. **Market Simulator** (`MarketSimulator`)
   - Orchestrates complete simulation
   - Geometric Brownian motion for price dynamics
   - Configurable order arrival process
   - Comprehensive summary statistics

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/GasparCoquet/market-making-simulator.git
cd market-making-simulator

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Quick Start

Run the example script:
```bash
python example.py
```

## 📊 Understanding the Output

The simulator provides detailed PnL decomposition:

### Spread Capture
The theoretical profit from buying at the bid and selling at the ask. This represents the "edge" a market maker has.

### Inventory Risk
The PnL impact from holding inventory while prices move. Positive inventory (long) gains when prices rise but loses when prices fall.

### Adverse Selection
The cost of trading with informed traders who know the price is about to move. If we buy right before the price drops, or sell right before it rises, we suffer adverse selection.

### Example Output
```
============================================================
MARKET-MAKING SIMULATION SUMMARY
============================================================

Market Statistics:
  Initial Mid Price: $100.00
  Final Mid Price: $102.50
  Price Change: $2.50 (2.50%)

Trading Activity:
  Total Trades: 50
  Buys: 25
  Sells: 25
  Final Inventory: 0.00

PnL Analysis:
  Total PnL: $45.30

PnL Decomposition:
  Spread Capture: $50.00      (profit from bid-ask spread)
  Inventory Risk: $-10.50     (loss from price moves)
  Adverse Selection: $-5.80   (cost of informed traders)
============================================================
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Tests cover:
- Order book functionality (pricing, depth, execution)
- Market maker logic (quoting, inventory management)
- PnL decomposition accuracy
- Simulation consistency and reproducibility

## 🔧 Configuration

### OrderBook Parameters
- `initial_mid`: Starting mid price
- `spread`: Half-spread (distance from mid to bid/ask)
- `depth_per_level`: Liquidity available at each price level
- `num_levels`: Number of price levels on each side

### MarketMaker Parameters
- `quote_spread`: How far from mid to place quotes
- `quote_size`: Size to quote on each side
- `max_inventory`: Maximum position limit
- `inventory_skew_factor`: How much to skew quotes based on inventory

### Simulation Parameters
- `num_steps`: Number of time steps to simulate
- `volatility`: Price volatility (standard deviation)
- `arrival_rate`: Probability of order arrival per step
- `dt`: Time step size
- `random_seed`: Seed for reproducibility

## 📚 Key Concepts

### Inventory Skewing
When the market maker accumulates inventory, it adjusts quotes to encourage mean reversion:
- **Long inventory** → Widen ask, tighten bid (encourage selling)
- **Short inventory** → Widen bid, tighten ask (encourage buying)

### Position Limits
The market maker respects position limits to manage risk. When approaching limits:
- Stop quoting on the side that would increase position
- Continue quoting on the side that reduces position

### Price Impact
Large market orders that exceed available liquidity cause price impact, simulating real market behavior.

## 🎯 Use Cases

This simulator is valuable for:
- **Learning**: Understanding market-making mechanics and risks
- **Research**: Testing market-making strategies and parameters
- **Interviews**: Demonstrating knowledge of market microstructure
- **Analysis**: Studying the trade-offs in liquidity provision

## 📈 Why This Matters

Market-making is fundamental to modern financial markets:
- **Liquidity provision**: Market makers enable other traders to transact
- **Price discovery**: Continuous quoting helps establish fair prices
- **Risk management**: Understanding inventory risk and adverse selection is crucial
- **Profitability**: Successful market-making requires balancing many competing factors

This is a strong signal for quantitative trading, prop trading, and cryptocurrency trading roles as it demonstrates deep understanding of market microstructure.

## 🛠️ Technical Stack

- **Python 3.8+**: Core language
- **NumPy**: Numerical computations and random number generation
- **Pandas**: Data manipulation (optional, for extended analysis)

## 📝 License

See [LICENSE](LICENSE) for details.

## 📖 References

This simulator demonstrates concepts from:
- Market microstructure theory
- Optimal market-making (Avellaneda-Stoikov framework)
- Inventory risk management
- Adverse selection in financial markets
