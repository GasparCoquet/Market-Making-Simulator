"""
MarketSimulator: orchestrates one path.

Ordering inside a step matters and is the reason the old decomposition did not
reconcile. The order is:

  1. draw the price innovation for this step (informed traders see it)
  2. read our quotes at the current mid
  3. resolve arrivals and fills at the current mid
  4. move the price
  5. snapshot the position that was held *across* that move

Snapshotting before the fills instead attributes the move to the wrong
position, which is an off-by-one that silently breaks the PnL identity.

Randomness comes from a per-instance `numpy.random.Generator`, not the process
global RNG, so two simulations can be interleaved and still reproduce.
"""

from typing import List, Dict, Optional
import numpy as np

from .market_state import MarketState
from .fill_model import FillModel


class MarketSimulator:
    """Runs a market-making strategy over a simulated price path."""

    def __init__(
        self,
        market_state: MarketState,
        market_maker,
        pnl_tracker,
        fill_model: Optional[FillModel] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            market_state: Exogenous market
            market_maker: Quoting strategy
            pnl_tracker: PnL accounting
            fill_model: Arrival intensity model. Defaults to FillModel().
            random_seed: Seed for this instance's generator. Leave None for a
                non-reproducible run; the seed used is reported in the summary.
        """
        self.market_state = market_state
        self.market_maker = market_maker
        self.pnl_tracker = pnl_tracker
        self.fill_model = fill_model if fill_model is not None else FillModel()

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.current_time = 0.0
        self.step_index = 0
        self.history: List[Dict] = []
        self._pnl_path: List[float] = [0.0]

    def draw_log_return(self, volatility: float, dt: float) -> float:
        """
        One step of geometric Brownian motion, with the Ito correction.

        Without the -0.5*sigma^2*dt term the simulated asset drifts upward at
        exp(0.5*sigma^2*T), which at the 5% volatility scenario is a 13% built-in
        rise over 100 steps and contaminates any comparison across volatilities.
        """
        shock = self.rng.normal()
        return -0.5 * volatility ** 2 * dt + volatility * np.sqrt(dt) * shock

    def step(self, volatility: float = 0.01, dt: float = 1.0) -> Dict:
        """Execute one simulation step and return its record."""
        mid_before = self.market_state.get_mid_price()

        # 1. The move that is about to happen. Informed traders condition on it.
        log_return = self.draw_log_return(volatility, dt)

        # 2. Our quotes, at the current mid.
        bid_price, bid_size, ask_price, ask_size = self.market_maker.get_quotes(
            self.market_state
        )

        # 3. Arrivals, priced off the distance of each quote from the mid.
        bid_order, ask_order = self.fill_model.sample_arrivals(
            self.rng,
            bid_distance=mid_before - bid_price,
            ask_distance=ask_price - mid_before,
            dt=dt,
            next_log_return=log_return,
        )

        # A large incoming order fills us only up to our own quoted size.
        bid_fill = min(bid_order, bid_size)
        ask_fill = min(ask_order, ask_size)

        self.current_time += dt

        if bid_fill > 0:
            self.market_maker.execute_bid_fill(bid_price, bid_fill)
            self.pnl_tracker.record_trade(
                self.current_time, 'buy', bid_price, bid_fill,
                mid_before, self.step_index,
            )
        if ask_fill > 0:
            self.market_maker.execute_ask_fill(ask_price, ask_fill)
            self.pnl_tracker.record_trade(
                self.current_time, 'sell', ask_price, ask_fill,
                mid_before, self.step_index,
            )

        # 4. Move the price.
        mid_after = mid_before * np.exp(log_return)
        self.market_state.update_mid_price(mid_after)

        # 5. Snapshot the position held across the move.
        inventory_after_fills = self.market_maker.get_inventory()
        self.pnl_tracker.record_step(
            self.current_time, inventory_after_fills, mid_before, mid_after
        )

        mark_to_market = self.market_maker.get_mark_to_market_pnl(mid_after)
        self._pnl_path.append(mark_to_market)

        # The risk overlay observes the same series the summary reports, so
        # summary['max_drawdown'] is exactly the drawdown the kill-switch tested.
        # The stop therefore acts with one step of lag, which is honest: you see
        # the loss, then you stop quoting.
        self.market_maker.observe_risk(mid_after)

        step_data = {
            'step': self.step_index,
            'time': self.current_time,
            'mid_before': mid_before,
            'mid_price': mid_after,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'inventory': inventory_after_fills,
            'trade_occurred': bid_fill > 0 or ask_fill > 0,
            'bid_fill_qty': bid_fill,
            'bid_fill_price': bid_price if bid_fill > 0 else None,
            'ask_fill_qty': ask_fill,
            'ask_fill_price': ask_price if ask_fill > 0 else None,
            'net_cash_flow': self.market_maker.get_net_cash_flow(),
            'gross_pnl': self.market_maker.get_gross_pnl(mid_after),
            'mark_to_market_pnl': mark_to_market,
        }
        self.history.append(step_data)
        self.step_index += 1
        return step_data

    def run(
        self,
        num_steps: int = 100,
        volatility: float = 0.01,
        dt: float = 1.0,
        verbose: bool = False,
    ) -> List[Dict]:
        """Run `num_steps` steps and return the history."""
        for i in range(num_steps):
            step_data = self.step(volatility, dt)
            if verbose and (i % 10 == 0 or i == num_steps - 1):
                print(f"Step {i+1}/{num_steps}: "
                      f"Mid={step_data['mid_price']:.2f}, "
                      f"Inventory={step_data['inventory']:.2f}, "
                      f"PnL={step_data['mark_to_market_pnl']:.2f}")
        return self.history

    def get_max_drawdown(self) -> float:
        """Largest peak-to-trough fall in mark-to-market PnL, non-negative."""
        peak = self._pnl_path[0]
        max_dd = 0.0
        for pnl in self._pnl_path:
            peak = max(peak, pnl)
            max_dd = max(max_dd, peak - pnl)
        return max_dd

    def get_summary(self) -> Dict:
        """
        Summary statistics.

        The waterfall reconciles exactly:
            gross_pnl = spread_capture + inventory_pnl
            net_pnl   = gross_pnl + rebates - liquidation_cost
        """
        if not self.history:
            return {}

        decomp = self.pnl_tracker.get_pnl_decomposition()
        num_buys, num_sells = self.pnl_tracker.get_trade_count()
        final_mid = self.market_state.get_mid_price()
        final_inventory = self.market_maker.get_inventory()

        liquidation_cost = self.market_state.liquidation_cost(final_inventory)
        rebates = self.market_maker.rebates
        gross_pnl = decomp['gross_pnl']

        initial_mid = self.market_state.initial_mid
        price_change = final_mid - initial_mid

        return {
            'total_steps': len(self.history),
            'final_time': self.current_time,
            'random_seed': self.random_seed,

            'initial_mid': initial_mid,
            'final_mid': final_mid,
            'price_change': price_change,
            'price_change_pct': 100.0 * price_change / initial_mid,

            'num_trades': num_buys + num_sells,
            'num_buys': num_buys,
            'num_sells': num_sells,
            'filled_volume': self.pnl_tracker.get_filled_volume(),
            'fill_rate': (num_buys + num_sells) / len(self.history),
            'final_inventory': final_inventory,

            'spread_capture': decomp['spread_capture'],
            'inventory_pnl': decomp['inventory_pnl'],
            'gross_pnl': gross_pnl,
            'rebates': rebates,
            'liquidation_cost': liquidation_cost,
            'net_pnl': gross_pnl + rebates - liquidation_cost,

            'adverse_selection': decomp['adverse_selection'],
            'max_drawdown': self.get_max_drawdown(),
            # Nested getattr because the constructor duck-types market_maker, so
            # a substitute may have no risk_manager attribute at all.
            'risk_halted': bool(
                getattr(getattr(self.market_maker, 'risk_manager', None),
                        'is_halted', False)
            ),
            'net_cash_flow': self.market_maker.get_net_cash_flow(),
            'inventory_value': self.market_maker.get_inventory_value(final_mid),
        }

    def print_summary(self):
        """Print the summary as a reconciling waterfall."""
        s = self.get_summary()
        if not s:
            print("No simulation data available.")
            return

        print("\n" + "=" * 62)
        print("MARKET-MAKING SIMULATION SUMMARY")
        print("=" * 62)
        print(f"\nRun:")
        print(f"  Steps:                {s['total_steps']}")
        print(f"  Random seed:          {s['random_seed']}")
        print(f"\nMarket:")
        print(f"  Initial mid:          ${s['initial_mid']:.2f}")
        print(f"  Final mid:            ${s['final_mid']:.2f}")
        print(f"  Price change:         ${s['price_change']:.2f} "
              f"({s['price_change_pct']:.2f}%)")
        print(f"\nActivity:")
        print(f"  Fills:                {s['num_trades']} "
              f"({s['num_buys']} buys, {s['num_sells']} sells)")
        print(f"  Filled volume:        {s['filled_volume']:.1f} units")
        print(f"  Fill rate:            {s['fill_rate']:.2f} fills/step")
        print(f"  Final inventory:      {s['final_inventory']:.2f} units")
        print(f"\nPnL waterfall:")
        print(f"  Spread capture:       ${s['spread_capture']:>10.2f}")
        print(f"  Inventory PnL:        ${s['inventory_pnl']:>10.2f}")
        print(f"  {'-' * 40}")
        print(f"  Gross PnL (at mid):   ${s['gross_pnl']:>10.2f}")
        print(f"  Maker rebates:        ${s['rebates']:>10.2f}")
        print(f"  Liquidation cost:     ${-s['liquidation_cost']:>10.2f}")
        print(f"  {'-' * 40}")
        print(f"  Net PnL:              ${s['net_pnl']:>10.2f}")
        print(f"\nDiagnostics:")
        print(f"  Adverse selection:    ${s['adverse_selection']:>10.2f}  "
              f"(signed markout, part of inventory PnL)")
        print(f"  Max drawdown:         ${s['max_drawdown']:>10.2f}")
        print(f"  Net cash flow:        ${s['net_cash_flow']:>10.2f}  "
              f"(not a PnL)")
        print(f"  Inventory value:      ${s['inventory_value']:>10.2f}  "
              f"(cash flow + this = gross PnL)")
        print("=" * 62 + "\n")
