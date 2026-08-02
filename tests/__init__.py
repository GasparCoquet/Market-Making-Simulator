"""
Tests for the market-making simulator.

The suite asserts economic invariants, not arithmetic. Each test is written so
that it fails on a specific class of modelling error: crossed quotes, a PnL
decomposition that does not reconcile, fills that ignore the quoted price, a
kill-switch that fires on cash flow, a price process with a built-in drift, or
a global RNG that breaks reproducibility when two runs are interleaved.
"""
