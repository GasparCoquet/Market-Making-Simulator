"""
Tests for the plotting layer.

This file exists because of one specific v0.1 failure mode: the plotter read a
fill-quantity key the simulator never wrote, so every chart drew zero trade
markers while the simulation itself was fine. Nothing caught it, because a chart
with no markers still saves to disk and still exits 0.

So these tests do not check that the figures look nice. They check the two
things that were wrong and can silently go wrong again:

1. What the plotter counts equals what the simulator reports. If a key is
   renamed on one side only, `count_fills` disagrees with the summary and these
   tests fail rather than a run printing "0 buys, 0 sells".
2. The markers are actually on the figure. Counting fills correctly and then
   drawing nothing is the same bug one layer down, so the assertions read the
   scatter offsets off the returned axes rather than trusting the counter.

Everything runs against a real seeded path, never a hand-built dict, because a
fixture written by hand would carry whatever key names the plotter happens to
use and could not detect a mismatch with the simulator.
"""

import contextlib
import io
import unittest
from functools import lru_cache
from typing import Dict, List, Tuple

import matplotlib

# Must precede the pyplot import: the suite runs headless in CI and a default
# interactive backend would fail to find a display.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from market_making_simulator.analytics.plotter import (  # noqa: E402
    SimulationPlotter,
    count_fills,
)

from tests._helpers import run_simulation  # noqa: E402

SEED = 42
NUM_STEPS = 400
# Informed flow on, so the path is the one the example script actually plots
# rather than a degenerate uninformed special case.
INFORMED_FRACTION = 0.3


@lru_cache(maxsize=1)
def reference_run() -> Tuple[List[Dict], Dict]:
    """
    One seeded path shared by every test in this module.

    Cached because eight test methods read the same history and the suite has a
    runtime budget.

    Returns:
        (history, summary) for a single reproducible run.
    """
    simulator, summary = run_simulation(
        SEED, num_steps=NUM_STEPS, informed_fraction=INFORMED_FRACTION)
    return (simulator.history, summary)


def scatter_offsets_by_label(ax) -> Dict[str, List[Tuple[float, float]]]:
    """
    Read the drawn scatter points off an axes, keyed by legend label.

    Args:
        ax: A matplotlib axes holding the trade-marker scatters.

    Returns:
        Dict mapping the label prefix ('Buys' or 'Sells') to the list of
        (x, y) points actually drawn.
    """
    points: Dict[str, List[Tuple[float, float]]] = {}
    for collection in ax.collections:
        label = str(collection.get_label())
        key = label.split(' ')[0]
        offsets = [(float(x), float(y)) for x, y in collection.get_offsets()]
        points.setdefault(key, []).extend(offsets)
    return points


class PlotterTestCase(unittest.TestCase):
    """Base case that closes every figure a test opens."""

    def setUp(self):
        self.history, self.summary = reference_run()
        self.plotter = SimulationPlotter()
        self.addCleanup(plt.close, 'all')

    def assertIsFigure(self, fig, name: str) -> None:
        """Assert `fig` is a real matplotlib figure, naming the entry point."""
        self.assertIsInstance(fig, Figure, msg=f"{name} did not return a Figure")


class TestReferenceRunIsUsable(PlotterTestCase):
    """The fixture must contain fills, or every assertion below is vacuous."""

    def test_the_run_actually_traded(self):
        self.assertGreater(self.summary['num_trades'], 0)
        self.assertGreater(self.summary['filled_volume'], 0.0)


class TestCountFills(PlotterTestCase):
    """`count_fills` is what the charts and the smoke test both count with."""

    def test_total_matches_the_simulator_trade_count(self):
        # The regression: rename a fill key in the plotter and this is the
        # assertion that goes from 228 == 228 to 0 == 228.
        self.assertEqual(count_fills(self.history)['total'],
                         self.summary['num_trades'])

    def test_buys_and_sells_match_the_simulator_sides(self):
        fills = count_fills(self.history)
        self.assertEqual(fills['buys'], self.summary['num_buys'])
        self.assertEqual(fills['sells'], self.summary['num_sells'])

    def test_buys_plus_sells_equals_total(self):
        fills = count_fills(self.history)
        self.assertEqual(fills['buys'] + fills['sells'], fills['total'])

    def test_volume_matches_the_simulator_filled_volume(self):
        self.assertAlmostEqual(count_fills(self.history)['volume'],
                               self.summary['filled_volume'], delta=1e-9)

    def test_a_step_filling_both_sides_counts_twice(self):
        # Both-sided steps exist on this path, and the summary counts each fill
        # separately, so the counter must not collapse them into one trade.
        both = sum(1 for h in self.history
                   if h['bid_fill_qty'] > 0 and h['ask_fill_qty'] > 0)
        self.assertGreater(both, 0)
        distinct_steps = sum(1 for h in self.history
                             if h['bid_fill_qty'] > 0 or h['ask_fill_qty'] > 0)
        self.assertEqual(count_fills(self.history)['total'],
                         distinct_steps + both)

    def test_empty_history_counts_nothing(self):
        self.assertEqual(count_fills([]),
                         {'buys': 0, 'sells': 0, 'total': 0, 'volume': 0})

    def test_a_missing_fill_key_raises_instead_of_reporting_zero(self):
        # The defensiveness this replaces is the bug: with `.get(key, 0.0)` a
        # renamed key reports a run with no trades and every caller believes it.
        broken = [dict(h) for h in self.history]
        del broken[0]['bid_fill_qty']
        with self.assertRaises(KeyError):
            count_fills(broken)


class TestTradeMarkers(PlotterTestCase):
    """The markers must reach the figure, not just the counter."""

    def setUp(self):
        super().setUp()
        self.fig = self.plotter.plot_price_with_trades(self.history)
        self.assertIsFigure(self.fig, 'plot_price_with_trades')
        self.points = scatter_offsets_by_label(self.fig.axes[0])

    def test_markers_are_drawn_at_all(self):
        drawn = sum(len(v) for v in self.points.values())
        self.assertGreater(drawn, 0, msg="the chart drew no trade markers")

    def test_marker_count_equals_the_simulator_trade_count(self):
        drawn = sum(len(v) for v in self.points.values())
        self.assertEqual(drawn, self.summary['num_trades'])

    def test_each_side_is_drawn_separately_and_in_full(self):
        self.assertEqual(len(self.points.get('Buys', [])),
                         self.summary['num_buys'])
        self.assertEqual(len(self.points.get('Sells', [])),
                         self.summary['num_sells'])

    def test_markers_sit_at_the_fill_price_not_the_mid(self):
        # A marker drawn at the mid is the other way this chart lies: it looks
        # populated while showing none of the edge the strategy earned.
        expected_buys = sorted(
            (h['time'], h['bid_fill_price'])
            for h in self.history if h['bid_fill_qty'] > 0)
        expected_sells = sorted(
            (h['time'], h['ask_fill_price'])
            for h in self.history if h['ask_fill_qty'] > 0)
        self.assertEqual(sorted(self.points.get('Buys', [])), expected_buys)
        self.assertEqual(sorted(self.points.get('Sells', [])), expected_sells)

    def test_markers_are_drawn_at_our_own_quote(self):
        # A marker is only informative if it sits on the price we quoted, so it
        # is offset from the mid by the edge plus the inventory lean. Drawing at
        # the mid instead would put every marker on the price line.
        quotes = {h['time']: (h['bid_price'], h['ask_price'], h['mid_before'])
                  for h in self.history}
        for time, price in self.points.get('Buys', []):
            bid, _, mid = quotes[time]
            self.assertAlmostEqual(price, bid, delta=1e-9)
            self.assertNotAlmostEqual(price, mid, delta=1e-9)
        for time, price in self.points.get('Sells', []):
            _, ask, mid = quotes[time]
            self.assertAlmostEqual(price, ask, delta=1e-9)
            self.assertNotAlmostEqual(price, mid, delta=1e-9)


class TestPlottedPnLIsMarkToMarket(PlotterTestCase):
    """Anything labelled PnL plots the mark, never the cash flow."""

    def line_matching_history_length(self, ax) -> List[float]:
        """
        Return the y data of the one line on `ax` that spans the whole run.

        The axes also carries a zero reference line, which has two points, so
        length is enough to identify the series.
        """
        candidates = [list(line.get_ydata()) for line in ax.lines
                      if len(line.get_ydata()) == len(self.history)]
        self.assertEqual(len(candidates), 1,
                         msg="expected exactly one full-length series")
        return candidates[0]

    def test_the_pnl_panel_plots_mark_to_market_pnl(self):
        fig = self.plotter.plot_simulation(self.history)
        self.assertIsFigure(fig, 'plot_simulation')
        # subplots(2, 2) fills fig.axes row-major, so index 2 is axes[1, 0],
        # the PnL panel.
        plotted = self.line_matching_history_length(fig.axes[2])
        self.assertEqual(plotted, [h['mark_to_market_pnl'] for h in self.history])

    def test_the_two_series_differ_so_the_check_is_not_vacuous(self):
        # If cash flow and mark-to-market happened to coincide on this path, the
        # test above would pass on the wrong series. On a run that ends with an
        # open position they differ by the inventory value.
        mark = [h['mark_to_market_pnl'] for h in self.history]
        cash = [h['net_cash_flow'] for h in self.history]
        self.assertNotEqual(mark, cash)
        # The gap is the inventory value, which runs into the hundreds of
        # dollars on any path that carries a position at all.
        gap = max(abs(m - c) for m, c in zip(mark, cash))
        self.assertGreater(gap, 1.0)

    def test_the_pnl_axis_says_mark_to_market(self):
        fig = self.plotter.plot_simulation(self.history)
        self.assertIn('mark-to-market', fig.axes[2].get_ylabel().lower())


class TestEveryEntryPointDraws(PlotterTestCase):
    """Every public figure-producing method runs on a real history."""

    def test_plot_simulation_returns_a_figure(self):
        self.assertIsFigure(self.plotter.plot_simulation(self.history),
                            'plot_simulation')

    def test_plot_price_with_trades_returns_a_figure(self):
        self.assertIsFigure(self.plotter.plot_price_with_trades(self.history),
                            'plot_price_with_trades')

    def test_plot_quotes_returns_a_figure(self):
        self.assertIsFigure(self.plotter.plot_quotes(self.history),
                            'plot_quotes')

    def test_plot_quotes_handles_any_window(self):
        # None means the whole path, and a window longer than the run must clamp
        # rather than index past the end.
        for window in (None, 1, 50, 10 * NUM_STEPS):
            with self.subTest(window=window):
                fig = self.plotter.plot_quotes(self.history, window=window)
                self.assertIsFigure(fig, f'plot_quotes(window={window})')

    def test_plot_pnl_decomposition_returns_a_figure(self):
        self.assertIsFigure(self.plotter.plot_pnl_decomposition(self.summary),
                            'plot_pnl_decomposition')

    def test_empty_history_returns_none_rather_than_an_empty_figure(self):
        # The plotter reports the empty case on stdout, which is useful
        # interactively and is noise in a test run, so swallow it here.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.plotter.plot_simulation([]))
            self.assertIsNone(self.plotter.plot_price_with_trades([]))
            self.assertIsNone(self.plotter.plot_quotes([]))


class TestWaterfallReconcilesOnTheFigure(PlotterTestCase):
    """The residual printed on the chart is the real one."""

    def test_the_drawn_residual_is_zero_to_floating_point(self):
        fig = self.plotter.plot_pnl_decomposition(self.summary)
        residual_texts = [t.get_text() for t in fig.axes[0].texts
                          if t.get_text().startswith('Identity residual')]
        self.assertEqual(len(residual_texts), 1)
        residual = float(residual_texts[0].split(':')[1])
        self.assertLess(residual, 1e-9)


if __name__ == '__main__':
    unittest.main()
