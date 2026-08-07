"""
Plotting utilities for market-making simulation results.

Three rules this module follows, each of them a bug that was fixed here:

1. Anything labelled "PnL" plots `mark_to_market_pnl`. Net cash flow is not a
   PnL: a short position shows positive cash and a negative mark, and the two
   can differ by thousands of dollars on the same run.
2. Trade markers are drawn at the fill price, from `bid_fill_qty` /
   `ask_fill_qty`. A single step can fill on both sides, so both are checked.
   Those keys are read with `[...]`, never with `.get(..., 0.0)`: the original
   bug was a plotter reading a key the simulator never wrote, and a default of
   zero turns that into an empty chart instead of an exception. Every record in
   `MarketSimulator.history` carries both keys, so a missing one is a bug and
   must raise.
3. The decomposition chart is a waterfall that reconciles. Adverse selection is
   a signed markout, a diagnostic split of the inventory term, so it is shown
   beside the waterfall and never summed into it.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt

# Shared colours, so the panels read as one figure.
_PRICE = '#1f77b4'
_BID = '#2ca02c'
_ASK = '#d62728'
_RESERVATION = '#ff7f0e'
_POSITIVE = '#2ca02c'
_NEGATIVE = '#d62728'
_SUBTOTAL = '#4c72b0'


def count_fills(history: List[Dict]) -> Dict[str, float]:
    """
    Count real fills in a simulation history.

    The step count is not the trade count. A run of 2000 steps with a 0.64
    fill rate has roughly 1274 fills, and reporting 2000 overstates activity by
    more than a factor of one and a half.

    Args:
        history: Simulation history from `MarketSimulator.history`

    Returns:
        Dict with `buys`, `sells`, `total` (fill counts) and `volume`
        (total quantity filled across both sides).

    Raises:
        KeyError: If a step lacks `bid_fill_qty` or `ask_fill_qty`. Reading
            these with a zero default would report an empty run as a valid one,
            which is exactly how the v0.1 plotter drew no trades at all.
    """
    buys = sum(1 for h in history if h['bid_fill_qty'] > 0)
    sells = sum(1 for h in history if h['ask_fill_qty'] > 0)
    volume = sum(h['bid_fill_qty'] + h['ask_fill_qty'] for h in history)
    return {'buys': buys, 'sells': sells, 'total': buys + sells,
            'volume': volume}


class SimulationPlotter:
    """Creates visualizations of market-making simulation results."""

    def __init__(self, figsize: tuple = (14, 10)):
        """
        Args:
            figsize: Figure size (width, height) in inches for the multi-panel
                overview. Single-panel charts size themselves.
        """
        self.figsize = figsize

    def plot_simulation(
        self,
        history: List[Dict],
        title: str = "Market-Making Simulation Results",
    ) -> Optional[plt.Figure]:
        """
        Multi-panel overview: price, inventory, mark-to-market PnL, statistics.

        Args:
            history: Simulation history from `MarketSimulator.history`
            title: Figure title

        Returns:
            The figure, or None if `history` is empty.
        """
        if not history:
            print("No history to plot")
            return None

        times = [h['time'] for h in history]
        mid_prices = [h['mid_price'] for h in history]
        inventories = [h['inventory'] for h in history]
        # Mark-to-market, not cash. See the module docstring.
        pnl = [h['mark_to_market_pnl'] for h in history]

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(times, mid_prices, color=_PRICE, linewidth=1.5, label='Mid price')
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('Price ($)')
        ax.set_title('Mid Price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        ax = axes[0, 1]
        # A bar per step is unreadable past a few hundred steps, so fill instead.
        # The outline is dropped on long runs, where it covers the fill entirely.
        ax.fill_between(times, inventories, 0, where=[i >= 0 for i in inventories],
                        color=_POSITIVE, alpha=0.6, interpolate=True, label='Long')
        ax.fill_between(times, inventories, 0, where=[i < 0 for i in inventories],
                        color=_NEGATIVE, alpha=0.6, interpolate=True, label='Short')
        if len(times) <= 500:
            ax.plot(times, inventories, color='black', linewidth=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('Inventory (units)')
        ax.set_title('Inventory Over Time')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best')

        ax = axes[1, 0]
        ax.fill_between(times, pnl, 0, alpha=0.25, color=_PRICE)
        ax.plot(times, pnl, color=_PRICE, linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('Mark-to-market PnL ($)')
        ax.set_title('Mark-to-Market PnL (before liquidation cost)')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.axis('off')

        fills = count_fills(history)
        initial_price = history[0]['mid_before']
        final_price = mid_prices[-1]
        price_change = final_price - initial_price

        stats_text = f"""
SIMULATION SUMMARY

Price:
  Initial: ${initial_price:.2f}
  Final:   ${final_price:.2f}
  Change:  ${price_change:.2f} ({100 * price_change / initial_price:.2f}%)

Inventory:
  Final:   {inventories[-1]:.1f} units
  Max:     {max(inventories):.1f} units
  Min:     {min(inventories):.1f} units

Activity ({len(history)} steps):
  Fills:   {fills['total']} ({fills['buys']} buys, {fills['sells']} sells)
  Volume:  {fills['volume']:.1f} units

Performance:
  Mark-to-market PnL: ${pnl[-1]:.2f}
"""
        ax.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        fig.tight_layout()
        return fig

    def plot_pnl_decomposition(
        self,
        summary: Dict[str, float],
        title: str = "PnL Waterfall",
    ) -> plt.Figure:
        """
        Waterfall of the PnL identity, with adverse selection alongside.

        The bars reconcile exactly:
            gross_pnl = spread_capture + inventory_pnl
            net_pnl   = gross_pnl + rebates + funding - liquidation_cost

        Adverse selection is a signed markout on our fills, a split of the
        inventory term rather than an extra bucket, so it is drawn as a separate
        diagnostic bar off the waterfall and is not added into the total.

        The funding bar is drawn only when there is funding, because an
        instrument without a perpetual leg has no such term and a permanent
        zero bar reads as a measurement rather than as an absence. The residual
        below always includes it, so a non-zero funding term can never be
        dropped from the chart and pass the reconciliation.

        Args:
            summary: Dict from `MarketSimulator.get_summary()`. Needs the keys
                spread_capture, inventory_pnl, gross_pnl, rebates,
                liquidation_cost, net_pnl and adverse_selection, and reads
                funding when present.
            title: Figure title

        Returns:
            The figure.
        """
        spread_capture = summary['spread_capture']
        inventory_pnl = summary['inventory_pnl']
        gross_pnl = summary['gross_pnl']
        rebates = summary['rebates']
        # Tolerated as absent so a summary from an older run still plots; every
        # summary this package produces carries the key.
        funding = summary.get('funding', 0.0)
        liquidation_cost = summary['liquidation_cost']
        net_pnl = summary['net_pnl']
        adverse_selection = summary['adverse_selection']

        # (label, delta, is_total). Deltas float on the running sum; totals are
        # drawn from zero so the eye can check the arithmetic.
        entries = [
            ('Spread\ncapture', spread_capture, False),
            ('Inventory\nPnL', inventory_pnl, False),
            ('Gross PnL', gross_pnl, True),
            ('Rebates', rebates, False),
        ]
        if funding != 0.0:
            entries.append(('Funding', funding, False))
        entries += [
            ('Liquidation\ncost', -liquidation_cost, False),
            ('Net PnL', net_pnl, True),
        ]

        fig, (ax, ax_diag) = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]}
        )

        running = 0.0
        for i, (label, value, is_total) in enumerate(entries):
            if is_total:
                bottom, height = 0.0, value
                colour = _SUBTOTAL
                running = value
            else:
                bottom, height = running, value
                colour = _POSITIVE if value >= 0 else _NEGATIVE
                running += value

            ax.bar(i, height, bottom=bottom, color=colour, alpha=0.8,
                   edgecolor='black', linewidth=0.6)
            top = bottom + height
            ax.text(i, top, f'${value:,.2f}', ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels([e[0] for e in entries], fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_ylabel('PnL ($)', fontsize=11)
        # Headroom so the value labels do not collide with the title.
        ax.margins(y=0.18)
        ax.grid(True, alpha=0.3, axis='y')
        fig.suptitle(title, fontsize=13, fontweight='bold')

        # Absolute values, so a positive residual on one leg cannot mask a
        # negative one on the other and report a false zero.
        residual = abs((spread_capture + inventory_pnl) - gross_pnl)
        residual += abs(
            (gross_pnl + rebates + funding - liquidation_cost) - net_pnl)
        ax.text(0.98, 0.97,
                f'Identity residual: {residual:.2e}',
                transform=ax.transAxes, fontsize=8,
                va='top', ha='right', color='dimgray')

        colour = _POSITIVE if adverse_selection >= 0 else _NEGATIVE
        ax_diag.bar(0, adverse_selection, width=0.5, color=colour, alpha=0.8,
                    edgecolor='black', linewidth=0.6)
        ax_diag.set_xlim(-0.6, 0.6)
        ax_diag.margins(y=0.18)
        ax_diag.text(0, adverse_selection, f'${adverse_selection:,.2f}',
                     ha='center', va='bottom' if adverse_selection >= 0 else 'top',
                     fontsize=9, fontweight='bold')
        ax_diag.set_xticks([0])
        ax_diag.set_xticklabels(['Adverse\nselection'], fontsize=9)
        ax_diag.axhline(y=0, color='black', linewidth=0.8)
        ax_diag.set_title('Diagnostic\n(not summed in)', fontsize=10)
        ax_diag.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        return fig

    def plot_price_with_trades(
        self,
        history: List[Dict],
        title: str = "Price Movement with Trades",
    ) -> Optional[plt.Figure]:
        """
        Mid price with a marker at the price of every fill.

        Buys are drawn at `bid_fill_price` and sells at `ask_fill_price`, not at
        the mid, so the markers sit visibly below and above the price line by
        the quoted edge, shifted by the inventory lean. A step can fill on both
        sides, so both are checked independently.

        Args:
            history: Simulation history
            title: Figure title

        Returns:
            The figure, or None if `history` is empty.
        """
        if not history:
            print("No history to plot")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        times = [h['time'] for h in history]
        # The mid the fills were priced off, which is the mid before that step's
        # move. Plotting the post-move mid instead shifts the reference by one
        # step and the marker offsets stop being exactly the quoted edge.
        mid_prices = [h['mid_before'] for h in history]
        ax.plot(times, mid_prices, color=_PRICE, linewidth=1.2,
                label='Mid price', zorder=2)

        buy_times, buy_prices, buy_sizes = [], [], []
        sell_times, sell_prices, sell_sizes = [], [], []

        for h in history:
            # Strict lookup, see the module docstring: a fill key that is absent
            # is a broken history, not a step without a fill.
            if h['bid_fill_qty'] > 0:
                buy_times.append(h['time'])
                buy_prices.append(h['bid_fill_price'])
                buy_sizes.append(h['bid_fill_qty'])
            if h['ask_fill_qty'] > 0:
                sell_times.append(h['time'])
                sell_prices.append(h['ask_fill_price'])
                sell_sizes.append(h['ask_fill_qty'])

        # Marker area scales with fill size, so partial fills are visible.
        if buy_times:
            ax.scatter(buy_times, buy_prices, color=_BID, marker='^',
                       s=[6 * q for q in buy_sizes], alpha=0.7,
                       label=f'Buys ({len(buy_times)})', zorder=5)
        if sell_times:
            ax.scatter(sell_times, sell_prices, color=_ASK, marker='v',
                       s=[6 * q for q in sell_sizes], alpha=0.7,
                       label=f'Sells ({len(sell_times)})', zorder=5)

        ax.set_xlabel('Time (steps)', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_quotes(
        self,
        history: List[Dict],
        title: str = "Quotes and Reservation Price",
        window: Optional[int] = 200,
    ) -> Optional[plt.Figure]:
        """
        Quoted bid and ask around the mid, with the reservation price.

        This is the headline mechanism. The reservation price
        r = mid - inventory * gamma leans away from the mid in the direction
        that unwinds the position, and both quotes move with it, so the width
        stays 2 * quote_spread and the quotes never cross. The lower panel plots
        r - mid against the inventory the quoting rule saw, which is the
        position after the previous step's fills, not the current one.

        Args:
            history: Simulation history
            title: Figure title
            window: Number of steps to show, taken from the start of the run.
                Quotes sit within cents of the mid, so a full multi-thousand
                step path renders as a single thick band. Pass None for all.

        Returns:
            The figure, or None if `history` is empty.
        """
        if not history:
            print("No history to plot")
            return None

        # `history['inventory']` is the position AFTER that step's fills, but
        # the quotes were computed BEFORE them, so the inventory the quoting
        # rule actually saw is the previous step's. Plotting the post-fill
        # series against the lean is an off-by-one, and on the reference run it
        # disagrees by up to $0.10, the entire quoted width.
        quoting_inventory = [0.0] + [h['inventory'] for h in history[:-1]]

        end = len(history) if window is None else min(window, len(history))
        rows = history[:end]
        quoting_inventory = quoting_inventory[:end]

        times = [h['time'] for h in rows]
        # Quotes were read at the mid *before* the move, so plot that mid to
        # keep the bid, ask and reservation price on a consistent reference.
        mids = [h['mid_before'] for h in rows]
        bids = [h['bid_price'] for h in rows]
        asks = [h['ask_price'] for h in rows]
        # r is the midpoint of our own two quotes by construction.
        reservations = [0.5 * (b + a) for b, a in zip(bids, asks)]
        lean = [r - m for r, m in zip(reservations, mids)]

        fig, (ax, ax_lean) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]},
        )
        fig.suptitle(title, fontsize=13, fontweight='bold')

        ax.fill_between(times, bids, asks, color=_PRICE, alpha=0.12,
                        label='Our quoted width')
        ax.plot(times, mids, color=_PRICE, linewidth=1.2, label='Mid')
        ax.plot(times, reservations, color=_RESERVATION, linewidth=1.2,
                linestyle='--', label='Reservation price')
        ax.plot(times, bids, color=_BID, linewidth=0.9, label='Our bid')
        ax.plot(times, asks, color=_ASK, linewidth=0.9, label='Our ask')
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=2)

        min_gap = min(a - b for a, b in zip(asks, bids))
        ax.text(0.98, 0.02, f'min(ask - bid) = ${min_gap:.4f}',
                transform=ax.transAxes, fontsize=9,
                va='bottom', ha='right', color='dimgray')

        ax_lean.plot(times, lean, color=_RESERVATION, linewidth=1.0,
                     label='Reservation price - mid')
        ax_lean.axhline(y=0, color='black', linewidth=0.6)
        ax_lean.set_ylabel('Lean ($)', fontsize=10)
        ax_lean.set_xlabel('Time (steps)', fontsize=11)
        ax_lean.grid(True, alpha=0.3)
        ax_lean.legend(loc='upper left', fontsize=9)

        ax_inv = ax_lean.twinx()
        ax_inv.plot(times, quoting_inventory, color='gray', linewidth=0.8,
                    alpha=0.7, label='Inventory when quoted')
        ax_inv.set_ylabel('Inventory (units)', fontsize=10, color='gray')
        ax_inv.tick_params(axis='y', labelcolor='gray')
        ax_inv.legend(loc='lower right', fontsize=9)

        fig.tight_layout()
        return fig

    def show(self):
        """Display all open plots. Blocks; skip it when running headless."""
        plt.show()
