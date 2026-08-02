#!/usr/bin/env python3
"""
Example usage of the Market-Making Simulator.

Runs one path, prints the reconciling PnL waterfall, shows how the quotes lean
against inventory, and writes the figures to `plots/`.

Runs headless by default: the Agg backend is selected before pyplot is imported
unless `--show` is passed, so `python example.py` never blocks on a window and
always leaves artefacts on disk.
"""

import argparse
import os
import sys

import matplotlib

# Backend must be chosen before pyplot is imported anywhere, including via the
# package import below.
_SHOW = '--show' in sys.argv
if not _SHOW:
    matplotlib.use('Agg')

from market_making_simulator import (  # noqa: E402
    MarketState, FillModel, MarketMaker, PnLTracker, MarketSimulator,
    RiskManager, SimulationPlotter, per_step_volatility,
)
from market_making_simulator.analytics.plotter import count_fills  # noqa: E402

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

# 25% annualised on one-second steps. Stating the annualised figure and the
# step length is the only way this number means anything: the same 0.25 on
# one-minute steps is a different world.
ANNUAL_VOLATILITY = 0.25
SECONDS_PER_STEP = 1.0
NUM_STEPS = 2000
SEED = 42


def build_simulation():
    """
    Wire up one simulation.

    Returns:
        (simulator, market_maker, pnl_tracker, risk_manager, per-step sigma).
    """
    market_state = MarketState(initial_mid=100.0, reference_half_spread=0.10)

    # A share of arrivals trade with the coming move, so the adverse selection
    # diagnostic has something to measure. Set it to 0.0 and the markout goes
    # to zero in expectation.
    fill_model = FillModel(
        base_intensity=0.8,
        decay=20.0,
        mean_order_size=10.0,
        informed_fraction=0.3,
    )

    # The kill-switch fires on drawdown in mark-to-market PnL, not on cash flow.
    risk_manager = RiskManager(
        enable_kill_switch=True,
        drawdown_limit=200.0,
        enable_size_throttle=True,
        min_throttle=0.2,
    )

    market_maker = MarketMaker(
        quote_spread=0.05,
        quote_size=10.0,
        max_inventory=100.0,
        inventory_skew_factor=0.01,
        maker_rebate_per_unit=0.0,
        risk_manager=risk_manager,
    )

    pnl_tracker = PnLTracker(markout_horizon=5)

    simulator = MarketSimulator(
        market_state=market_state,
        market_maker=market_maker,
        pnl_tracker=pnl_tracker,
        fill_model=fill_model,
        random_seed=SEED,
    )

    sigma = per_step_volatility(ANNUAL_VOLATILITY, SECONDS_PER_STEP)
    return simulator, market_maker, pnl_tracker, risk_manager, sigma


def print_quote_ladder():
    """
    Show the quotes at several inventory levels.

    The point of the table is the invariant: ask > bid at every inventory, and
    both quotes move together, down when long and up when short. The previous
    version of this script printed crossed quotes here on every run.
    """
    state = MarketState(initial_mid=100.0, reference_half_spread=0.10)
    # Throttle only, no kill-switch: this table is about the quoting rule, and
    # a stop tripping on a synthetic position would just add noise.
    demo_risk = RiskManager(enable_kill_switch=False, enable_size_throttle=True)
    maker = MarketMaker(
        quote_spread=0.05,
        quote_size=10.0,
        max_inventory=100.0,
        inventory_skew_factor=0.01,
        risk_manager=demo_risk,
    )

    print(f"\nQuotes at different inventory levels "
          f"(mid = ${state.get_mid_price():.2f}, "
          f"quote_spread = ${maker.quote_spread:.2f}, "
          f"gamma = ${maker.inventory_skew_factor}/unit):")
    print("-" * 78)
    print(f"  {'Inventory':>10} {'Reservation':>12} {'Bid':>9} {'Ask':>9} "
          f"{'Ask-Bid':>9} {'Bid size':>10} {'Ask size':>10}")

    for inventory in [-100, -50, -20, 0, 20, 50, 100]:
        maker.inventory = float(inventory)
        bid, bid_size, ask, ask_size = maker.get_quotes(state)
        reservation = maker.get_reservation_price(state.get_mid_price())
        assert ask > bid, "quotes crossed"
        print(f"  {inventory:>10.0f} {reservation:>12.2f} {bid:>9.2f} "
              f"{ask:>9.2f} {ask - bid:>9.2f} {bid_size:>10.1f} "
              f"{ask_size:>10.1f}")

    print("-" * 78)
    print("  Long -> both quotes drop, so the ask is easier to lift and we sell"
          " down.")
    print("  Short -> both quotes rise, so the bid is easier to hit and we buy"
          " back.")
    print("  Ask - bid is constant at 2 x quote_spread, so the quotes never"
          " cross.")


def save_figures(simulator, summary):
    """
    Write every figure to `plots/`.

    Returns:
        Dict mapping figure name to the path written.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plotter = SimulationPlotter(figsize=(14, 10))

    figures = {
        'simulation_overview': plotter.plot_simulation(
            simulator.history,
            title="Market-Making Simulation: Price, Inventory and PnL",
        ),
        'pnl_waterfall': plotter.plot_pnl_decomposition(
            summary,
            title="PnL Waterfall (reconciles exactly)",
        ),
        'price_with_trades': plotter.plot_price_with_trades(
            simulator.history,
            title="Mid Price with Fills, Marked at the Fill Price",
        ),
        'quotes_reservation_price': plotter.plot_quotes(
            simulator.history,
            title="Quotes and Reservation Price (first 200 steps)",
            window=200,
        ),
    }

    written = {}
    for name, fig in figures.items():
        if fig is None:
            continue
        path = os.path.join(PLOTS_DIR, f"{name}.png")
        fig.savefig(path, dpi=120)
        written[name] = path
    return written


def main():
    """Run one simulation end to end and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true',
                        help="also open the figures in a window (blocks)")
    args = parser.parse_args()

    print("Market-Making Simulator Example")
    print("=" * 62)

    simulator, market_maker, pnl_tracker, risk_manager, sigma = build_simulation()

    print("\nConfiguration:")
    print(f"  Market:               {simulator.market_state}")
    print(f"  Fill model:           {simulator.fill_model}")
    # Read off the objects, so the printed configuration cannot drift away from
    # the one that was actually run.
    print(f"  Market maker:         "
          f"quote_spread={market_maker.quote_spread}, "
          f"quote_size={market_maker.quote_size:.0f}, "
          f"gamma={market_maker.inventory_skew_factor}, "
          f"max_inventory={market_maker.max_inventory:.0f}")
    print(f"  Risk manager:         {risk_manager}")
    print(f"  Markout horizon:      {pnl_tracker.markout_horizon} steps "
          f"(adverse selection is measured over this)")
    print(f"  Volatility:           {ANNUAL_VOLATILITY:.0%} annualised on "
          f"{SECONDS_PER_STEP:.0f}s steps = {sigma:.3e} per step")
    print(f"  Steps:                {NUM_STEPS}")
    print(f"  Seed:                 {SEED}")

    print("\nRunning simulation...")
    simulator.run(num_steps=NUM_STEPS, volatility=sigma, dt=1.0, verbose=False)

    simulator.print_summary()
    summary = simulator.get_summary()

    # The waterfall is an identity, so state the residual instead of asking the
    # reader to trust it.
    gross_residual = (summary['spread_capture'] + summary['inventory_pnl']
                      - summary['gross_pnl'])
    net_residual = (summary['gross_pnl'] + summary['rebates']
                    - summary['liquidation_cost'] - summary['net_pnl'])
    print("Reconciliation:")
    print(f"  spread_capture + inventory_pnl - gross_pnl        = "
          f"{gross_residual:.2e}")
    print(f"  gross_pnl + rebates - liquidation_cost - net_pnl  = "
          f"{net_residual:.2e}")

    # The invariant the rewrite exists to guarantee, checked on the real path.
    min_gap = min(h['ask_price'] - h['bid_price'] for h in simulator.history)
    crossed = sum(1 for h in simulator.history
                  if h['ask_price'] <= h['bid_price'])
    print(f"  min(ask - bid) over the run                       = "
          f"${min_gap:.4f}  ({crossed} crossed steps)")
    print(f"  risk manager halted                               = "
          f"{risk_manager.is_halted}")

    print_quote_ladder()

    print("\nGenerating plots...")
    fills = count_fills(simulator.history)

    # The v0.1 bug was a plotter that read a fill key the simulator never wrote
    # and therefore drew nothing, while this script still exited 0 and the smoke
    # job still passed. Checking the two counters against each other before
    # printing turns that silent zero into a failed run.
    assert fills['total'] == summary['num_trades'], (
        f"the plotter counted {fills['total']} fills but the simulator "
        f"reported {summary['num_trades']}: the plotter is reading the step "
        f"history with the wrong keys, so the charts are not the run")
    assert abs(fills['volume'] - summary['filled_volume']) < 1e-9, (
        f"the plotter counted {fills['volume']:.4f} units filled but the "
        f"simulator reported {summary['filled_volume']:.4f}")

    written = save_figures(simulator, summary)
    print(f"  Trade markers drawn:  {fills['total']} "
          f"({fills['buys']} buys, {fills['sells']} sells)")
    for name, path in written.items():
        print(f"  {name:<24} -> {path}")

    if args.show:
        SimulationPlotter().show()


if __name__ == "__main__":
    main()
