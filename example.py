#!/usr/bin/env python3
"""
Example usage of the Market-Making Simulator.

Runs one path of one dataset, prints the reconciling PnL waterfall, shows how
the quotes lean against inventory, and writes the figures to `plots/<dataset>/`.

Two datasets ship. `us-equity` is a $100 cash equity with a 6.5-hour session
and a per-share maker rebate. `crypto-perp` is a $100,000 perpetual swap that
never closes, settles funding hourly, and is charged in basis points of
notional. They are identical in every dimensionless quantity, so anything that
differs between the two runs is market structure rather than a spread someone
picked. See `market_making_simulator/datasets.py`.

    python example.py                          # the equity dataset, 33 minutes
    python example.py --dataset crypto-perp    # the perpetual, 8 hours
    python example.py --steps 46800            # two equity sessions

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
    DATASET_NAMES, SimulationPlotter, get_dataset,
)
from market_making_simulator.analytics.plotter import count_fills  # noqa: E402

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

DEFAULT_DATASET = 'us-equity'
SEED = 42

# The kill-switch limit is a property of our risk appetite, not of the
# instrument, so it lives here rather than in the dataset.
#
# $200 never fires on the equity run, whose whole drawdown is under $2. It does
# fire on the perpetual, and that is not a mis-set limit: a book paying 2bp of
# notional against roughly 1.4bp of gross edge per unit of volume bleeds
# monotonically, so *any* fixed drawdown limit stops it eventually and the only
# question is when. Pass `--drawdown-limit 0` to disable the stop and see the
# whole path; the benchmark grid runs unarmed for exactly that reason.
DRAWDOWN_LIMIT = 200.0

# Inventory levels for the quote ladder, in clips rather than units, so the
# same table reads correctly for a 10-share clip and a 0.01-contract clip.
LADDER_CLIPS = [-10, -5, -2, 0, 2, 5, 10]


def print_quote_ladder(dataset):
    """
    Show the quotes at several inventory levels.

    The point of the table is the invariant: ask > bid at every inventory, and
    both quotes move together, down when long and up when short. The previous
    version of this script printed crossed quotes here on every run.

    Built from the same dataset as the run, but as a fresh maker with the
    kill-switch off. Reusing the run's own maker looks tidier and is wrong: a
    run that tripped its stop leaves the overlay halted, and the ladder then
    prints a wall of zero sizes that says nothing about the quoting rule this
    table exists to show.
    """
    ladder_simulator = dataset.build(kill_switch_drawdown=None)
    maker = ladder_simulator.market_maker
    state = ladder_simulator.market_state

    print(f"\nQuotes at different inventory levels "
          f"(mid = ${dataset.initial_mid:,.2f}, "
          f"quote_spread = ${dataset.quote_spread:,.4f}, "
          f"gamma = ${dataset.inventory_skew_factor:,.4f}/{dataset.unit}):")
    print("-" * 86)
    print(f"  {'Inventory':>12} {'Reservation':>14} {'Bid':>12} {'Ask':>12} "
          f"{'Ask-Bid':>10} {'Bid size':>10} {'Ask size':>10}")

    for clips in LADDER_CLIPS:
        maker.inventory = float(clips) * dataset.quote_size
        bid, bid_size, ask, ask_size = maker.get_quotes(state)
        reservation = maker.get_reservation_price(state.get_mid_price())
        assert ask > bid, "quotes crossed"
        print(f"  {maker.inventory:>12.4g} {reservation:>14,.2f} "
              f"{bid:>12,.2f} {ask:>12,.2f} {ask - bid:>10,.4f} "
              f"{bid_size:>10.4g} {ask_size:>10.4g}")

    print("-" * 86)
    print("  Long -> both quotes drop, so the ask is easier to lift and we sell"
          " down.")
    print("  Short -> both quotes rise, so the bid is easier to hit and we buy"
          " back.")
    print("  Ask - bid is constant at 2 x quote_spread, so the quotes never"
          " cross.")


def save_figures(simulator, summary, dataset_name):
    """
    Write every figure to `plots/<dataset>/`.

    Each dataset gets its own directory so that running both does not leave one
    set of charts labelled with the other's numbers.

    Returns:
        Dict mapping figure name to the path written.
    """
    output_dir = os.path.join(PLOTS_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    plotter = SimulationPlotter(figsize=(14, 10))

    figures = {
        'simulation_overview': plotter.plot_simulation(
            simulator.history,
            title=f"Market-Making Simulation ({dataset_name}): "
                  f"Price, Inventory and PnL",
        ),
        'pnl_waterfall': plotter.plot_pnl_decomposition(
            summary,
            title=f"PnL Waterfall, {dataset_name} (reconciles exactly)",
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
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=120)
        written[name] = path
    return written


def parse_args(argv=None):
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Run one path of one dataset and write the figures.")
    parser.add_argument(
        '--dataset', choices=DATASET_NAMES, default=DEFAULT_DATASET,
        help=f"Which calibration to run (default {DEFAULT_DATASET})")
    parser.add_argument(
        '--steps', type=int, default=None,
        help="Steps to run; defaults to the dataset's own horizon")
    parser.add_argument(
        '--seed', type=int, default=SEED,
        help=f"Seed for the run (default {SEED})")
    parser.add_argument(
        '--drawdown-limit', type=float, default=DRAWDOWN_LIMIT,
        help=f"Kill-switch limit in dollars of drawdown from the peak "
             f"mark-to-market PnL (default {DRAWDOWN_LIMIT:g}); "
             f"pass 0 to disable the stop")
    parser.add_argument('--show', action='store_true',
                        help="also open the figures in a window (blocks)")
    return parser.parse_args(argv)


def main(argv=None):
    """Run one simulation end to end and write the figures."""
    args = parse_args(argv)
    if args.steps is not None and args.steps < 1:
        raise SystemExit("--steps must be at least 1")

    dataset = get_dataset(args.dataset)
    num_steps = dataset.default_steps if args.steps is None else args.steps
    drawdown_limit = args.drawdown_limit if args.drawdown_limit > 0 else None

    print("Market-Making Simulator Example")
    print("=" * 62)

    print("\nConfiguration:")
    # Read off the dataset, so the printed configuration cannot drift away from
    # the one that was actually run.
    print(dataset.describe(num_steps))
    risk_description = (
        f"kill-switch at ${drawdown_limit:,.2f} of drawdown, size throttle on"
        if drawdown_limit is not None
        else "size throttle only, no kill-switch"
    )
    print(f"  Risk manager:         {risk_description}")
    print(f"  Seed:                 {args.seed}")

    print("\nRunning simulation...")
    simulator, summary = dataset.run(
        random_seed=args.seed,
        num_steps=num_steps,
        kill_switch_drawdown=drawdown_limit,
    )
    simulator.print_summary()

    # The waterfall is an identity, so state the residual instead of asking the
    # reader to trust it.
    gross_residual = (summary['spread_capture'] + summary['inventory_pnl']
                      - summary['gross_pnl'])
    net_residual = (summary['gross_pnl'] + summary['rebates']
                    + summary['funding'] - summary['liquidation_cost']
                    - summary['net_pnl'])
    # The two splits must add back to the terms they split, or a diagnostic is
    # quietly measuring something other than a part of the waterfall.
    split_residual = (summary['quoted_edge'] - summary['session_close_cost']
                      - summary['spread_capture'])
    print("Reconciliation:")
    print(f"  spread_capture + inventory_pnl - gross_pnl          = "
          f"{gross_residual:.2e}")
    print(f"  gross + rebates + funding - liquidation - net       = "
          f"{net_residual:.2e}")
    print(f"  quoted_edge - close_out_cost - spread_capture       = "
          f"{split_residual:.2e}")

    # The invariant the rewrite exists to guarantee, checked on the real path.
    min_gap = min(h['ask_price'] - h['bid_price'] for h in simulator.history)
    crossed = sum(1 for h in simulator.history
                  if h['ask_price'] <= h['bid_price'])
    print(f"  min(ask - bid) over the run                         = "
          f"${min_gap:,.4f}  ({crossed} crossed steps)")
    print(f"  risk manager halted                                 = "
          f"{simulator.market_maker.risk_manager.is_halted}")

    # Structural facts about this dataset, checked against what the run did
    # rather than asserted from the configuration.
    funding_payments = sum(1 for h in simulator.history
                           if h['funding_paid'] != 0.0)
    print(f"  funding payments settled                            = "
          f"{funding_payments}")
    print(f"  session closes                                      = "
          f"{summary['session_closes']}")
    if simulator.market_maker.risk_manager.is_halted:
        # The step the stop bit, so a truncated run cannot be mistaken for a
        # full one. A book whose fees exceed its edge bleeds monotonically, so
        # a drawdown stop is guaranteed to fire on it eventually, and the
        # activity counters above describe only the part before that.
        halt_step = next(
            (h['step'] for h in simulator.history
             if h['bid_size'] == 0.0 and h['ask_size'] == 0.0),
            None)
        print(f"  quoting stopped at step                             = "
              f"{halt_step} of {num_steps}")

    print_quote_ladder(dataset)

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

    written = save_figures(simulator, summary, args.dataset)
    print(f"  Trade markers drawn:  {fills['total']} "
          f"({fills['buys']} buys, {fills['sells']} sells)")
    for name, path in written.items():
        print(f"  {name:<24} -> {path}")

    if args.show:
        SimulationPlotter().show()


if __name__ == "__main__":
    main()
