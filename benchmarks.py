#!/usr/bin/env python3
"""
Monte Carlo benchmark harness.

Ranking configurations on one path each is not a measurement. At the baseline
calibration a single path of 2000 steps carries a standard deviation of $6.7 of
net PnL, and $8.1 at the 60% annualised setting (measured over 200 paths, seeds
20000..20199), which is larger than several of the gaps this grid is trying to
resolve. Every figure printed below is therefore an average over N independent
paths, and no ranking is printed without the uncertainty attached to it.

Common random numbers
---------------------
Path k uses seed `base_seed + k` in every configuration. That is worth stating
precisely, because it is what makes the comparisons sharp:

  * The simulator owns one `numpy.random.Generator` seeded per run.
  * Each step draws one normal for the price innovation and then exactly
    `FillModel.DRAWS_PER_STEP` uniforms for the arrivals, on every branch.
  * Neither count depends on any parameter in the scenario grid.

So the generator stream is consumed in the same pattern whatever the
configuration, and configuration A and configuration B see the *same* price
path on path k. The difference in outcome is then attributable to the
parameters rather than to the draw. Differences against the baseline are
computed path by path and their standard error comes from the distribution of
those paired differences, which is much tighter than the standard error of
either configuration on its own.

Usage:
    python benchmarks.py
    python benchmarks.py --paths 2000 --seed 7
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_making_simulator import (
    FillModel,
    MarketMaker,
    MarketSimulator,
    MarketState,
    PnLTracker,
    RiskManager,
    per_step_volatility,
)
# Imported rather than hardcoded in the report text, so the printed caveat
# cannot drift away from the horizon the tracker actually uses.
from market_making_simulator.analytics.pnl_tracker import DEFAULT_MARKOUT_HORIZON

DEFAULT_PATHS = 500
DEFAULT_SEED = 12345
DEFAULT_STEPS = 2000

# Metrics collected per path. Order fixes the column order downstream.
_METRICS = (
    "net_pnl",
    "gross_pnl",
    "spread_capture",
    "inventory_pnl",
    "adverse_selection",
    "fills",
    "abs_final_inventory",
    "max_drawdown",
    "halted",
)


@dataclass(frozen=True)
class ScenarioConfig:
    """
    One point of the benchmark grid.

    Volatility is stored annualised and converted with
    `units.per_step_volatility` at run time, because the simulator's
    `volatility` argument is a per-step log-return sigma and quoting it
    without a time unit is how "2% volatility" came to mean a $2 move per
    second on a $100 asset.

    Frozen so it can be shipped to worker processes without any chance of a
    worker mutating shared state.
    """

    name: str
    quote_spread: float = 0.05
    inventory_skew_factor: float = 0.01
    annual_volatility: float = 0.25
    base_intensity: float = 0.8
    informed_fraction: float = 0.0
    quote_size: float = 10.0
    max_inventory: float = 100.0
    initial_mid: float = 100.0
    reference_half_spread: float = 0.10
    seconds_per_step: float = 1.0
    num_steps: int = DEFAULT_STEPS
    # None means no RiskManager is constructed at all. A number attaches one
    # with the kill-switch armed at that drawdown, in dollars of
    # mark-to-market PnL below the running peak.
    kill_switch_drawdown: Optional[float] = None


def build_scenarios(num_steps: int = DEFAULT_STEPS) -> List[ScenarioConfig]:
    """
    The benchmark grid.

    The first entry is the baseline and every paired difference is taken
    against it. Each other entry moves exactly one axis, so a difference can be
    read as the effect of that axis.

    Args:
        num_steps: Steps per path, applied to every scenario

    Returns:
        Scenario list, baseline first.
    """
    baseline = ScenarioConfig(name="baseline", num_steps=num_steps)

    scenarios = [
        baseline,
        # Volatility. 10%, 25% and 60% annualised on one-second steps are
        # 0.41, 1.03 and 2.47 basis points of per-step sigma.
        replace(baseline, name="vol 10% ann", annual_volatility=0.10),
        replace(baseline, name="vol 60% ann", annual_volatility=0.60),
        # Quote width. The whole point of the intensity model is that this is
        # a real trade-off rather than free money.
        replace(baseline, name="spread 0.02 (tight)", quote_spread=0.02),
        replace(baseline, name="spread 0.10 (wide)", quote_spread=0.10),
        # Inventory skew, gamma in the reservation price r = mid - q*gamma.
        replace(baseline, name="skew 0.00 (none)", inventory_skew_factor=0.00),
        replace(baseline, name="skew 0.05 (aggressive)", inventory_skew_factor=0.05),
        # Market thickness is the arrival intensity A, not a separate arrival
        # probability. There is no such parameter any more.
        replace(baseline, name="thin market (A=0.4)", base_intensity=0.4),
        replace(baseline, name="thick market (A=1.6)", base_intensity=1.6),
        # Informed flow. The axis that adverse selection actually responds to.
        replace(baseline, name="informed 30%", informed_fraction=0.30),
        replace(baseline, name="informed 60%", informed_fraction=0.60),
        # Risk overlay. The limit sits near the median of the drawdown the
        # RiskManager itself observes on an unarmed baseline, which is mean
        # $1.43 and median $1.41 over 200 paths of 2000 steps, so the switch
        # fires on roughly half the paths. A limit the switch can never reach
        # would leave the module as dead code dressed up as a control, and
        # would measure nothing.
        #
        # The 'max DD' column of the report is a different measurement, taken
        # after each price move rather than at quote time, and it keeps
        # accruing after a halt because the residual position is still marked
        # at a moving mid while quoting is off. That is why halted paths can
        # show a max DD above the limit that stopped them.
        replace(baseline, name="risk overlay (kill @ $1.40)", kill_switch_drawdown=1.40),
    ]
    return scenarios


def run_path(config: ScenarioConfig, seed: int) -> Dict[str, float]:
    """
    Run one path of one configuration.

    Args:
        config: Scenario parameters
        seed: Seed for this path's generator. The same seed under a different
            configuration reproduces the same price path.

    Returns:
        One value per entry of `_METRICS`.
    """
    volatility = per_step_volatility(config.annual_volatility, config.seconds_per_step)

    market_state = MarketState(
        initial_mid=config.initial_mid,
        reference_half_spread=config.reference_half_spread,
    )

    risk_manager: Optional[RiskManager] = None
    if config.kill_switch_drawdown is not None:
        risk_manager = RiskManager(
            enable_kill_switch=True,
            drawdown_limit=config.kill_switch_drawdown,
            enable_size_throttle=True,
        )

    market_maker = MarketMaker(
        quote_spread=config.quote_spread,
        quote_size=config.quote_size,
        max_inventory=config.max_inventory,
        inventory_skew_factor=config.inventory_skew_factor,
        risk_manager=risk_manager,
    )
    fill_model = FillModel(
        base_intensity=config.base_intensity,
        informed_fraction=config.informed_fraction,
    )
    simulator = MarketSimulator(
        market_state=market_state,
        market_maker=market_maker,
        pnl_tracker=PnLTracker(),
        fill_model=fill_model,
        random_seed=seed,
    )
    simulator.run(num_steps=config.num_steps, volatility=volatility, dt=1.0)

    summary = simulator.get_summary()
    return {
        "net_pnl": summary["net_pnl"],
        "gross_pnl": summary["gross_pnl"],
        "spread_capture": summary["spread_capture"],
        "inventory_pnl": summary["inventory_pnl"],
        "adverse_selection": summary["adverse_selection"],
        "fills": float(summary["num_trades"]),
        "abs_final_inventory": abs(summary["final_inventory"]),
        "max_drawdown": summary["max_drawdown"],
        "halted": 1.0 if (risk_manager is not None and risk_manager.is_halted) else 0.0,
    }


def _run_batch(task: Tuple[ScenarioConfig, Tuple[int, ...]]) -> List[Dict[str, float]]:
    """Worker entry point. Kept module level so it survives process spawn."""
    config, seeds = task
    return [run_path(config, seed) for seed in seeds]


def _chunks(seeds: Sequence[int], size: int) -> List[Tuple[int, ...]]:
    """Split the seed list into batches, to amortise process dispatch."""
    return [tuple(seeds[i:i + size]) for i in range(0, len(seeds), size)]


def run_scenario(
    config: ScenarioConfig,
    seeds: Sequence[int],
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, np.ndarray]:
    """
    Run every path of one configuration.

    Args:
        config: Scenario parameters
        seeds: Seed per path. Reused unchanged across configurations, which is
            what makes the comparison paired.
        executor: Optional process pool. Paths are independent, so this is
            embarrassingly parallel.

    Returns:
        Metric name to array of length len(seeds), in seed order.
    """
    if executor is None:
        records = [run_path(config, seed) for seed in seeds]
    else:
        batches = _chunks(seeds, max(1, len(seeds) // 32))
        records = []
        for batch_result in executor.map(_run_batch, [(config, b) for b in batches]):
            records.extend(batch_result)

    return {m: np.array([r[m] for r in records], dtype=float) for m in _METRICS}


def _standard_error(sample: np.ndarray) -> float:
    """Standard error of the mean. NaN below two observations."""
    if sample.size < 2:
        return float("nan")
    return float(sample.std(ddof=1) / np.sqrt(sample.size))


def crossing_inventory(config: ScenarioConfig) -> float:
    """
    Inventory at which our own quote reaches the mid.

    The ask is at mid - q*gamma + quote_spread, so it sits at or below the mid
    once |q| >= quote_spread / gamma, and fills there earn negative edge. With
    a quoted size of 10 per side, a threshold near or below 10 means a single
    fill is enough to push us into quoting through the mid, which shows up as
    negative spread capture rather than as price risk.

    Args:
        config: Scenario parameters

    Returns:
        The threshold, or infinity when the skew factor is zero.
    """
    if config.inventory_skew_factor <= 0:
        return float("inf")
    return config.quote_spread / config.inventory_skew_factor


def summarise(config: ScenarioConfig, metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Collapse one configuration's paths into a single summary row."""
    net = metrics["net_pnl"]
    return {
        "scenario": config.name,
        "cross_q": crossing_inventory(config),
        "net_pnl": float(net.mean()),
        "se": _standard_error(net),
        "p5": float(np.percentile(net, 5)),
        "p_loss": float((net < 0).mean()),
        "fills": float(metrics["fills"].mean()),
        "abs_inv": float(metrics["abs_final_inventory"].mean()),
        "max_dd": float(metrics["max_drawdown"].mean()),
        "adv_sel": float(metrics["adverse_selection"].mean()),
        "adv_se": _standard_error(metrics["adverse_selection"]),
        "p_halt": float(metrics["halted"].mean()),
    }


def paired_difference(
    name: str,
    metrics: Dict[str, np.ndarray],
    baseline_metrics: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Difference against the baseline, path by path.

    Pairing is the reason for common random numbers. The shared price path
    cancels in the difference, so the standard error here is typically a small
    fraction of the standard error of either configuration's own mean.

    Args:
        name: Scenario name
        metrics: This configuration's per-path metrics
        baseline_metrics: The baseline's per-path metrics, same seed order

    Returns:
        Mean paired difference in net PnL, its standard error and the paired
        t-statistic.
    """
    diff = metrics["net_pnl"] - baseline_metrics["net_pnl"]
    mean_diff = float(diff.mean())
    se_diff = _standard_error(diff)
    t_stat = mean_diff / se_diff if se_diff > 0 else float("nan")
    return {
        "scenario": name,
        "d_net_pnl": mean_diff,
        "se": se_diff,
        "t": t_stat,
        "win_rate": float((diff > 0).mean()),
    }


def run_benchmarks(
    scenarios: List[ScenarioConfig],
    num_paths: int,
    base_seed: int,
    workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the whole grid.

    Args:
        scenarios: Grid, baseline first
        num_paths: Paths per configuration
        base_seed: Path k of every configuration uses `base_seed + k`
        workers: Process count. 1 runs in this process.

    Returns:
        (summary table sorted by mean net PnL, paired difference table).
    """
    names = [c.name for c in scenarios]
    if len(set(names)) != len(names):
        # Results are keyed by name, so a duplicate would silently overwrite
        # another configuration and produce a paired difference against the
        # wrong baseline.
        raise ValueError("scenario names must be unique")

    seeds = [base_seed + k for k in range(num_paths)]
    baseline = scenarios[0]

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        per_scenario: Dict[str, Dict[str, np.ndarray]] = {}
        for config in scenarios:
            started = time.perf_counter()
            per_scenario[config.name] = run_scenario(config, seeds, executor)
            elapsed = time.perf_counter() - started
            print(f"  {config.name:<28} {num_paths} paths in {elapsed:5.1f}s")
    finally:
        if executor is not None:
            executor.shutdown()

    summary = pd.DataFrame(
        [summarise(c, per_scenario[c.name]) for c in scenarios]
    ).sort_values("net_pnl", ascending=False, ignore_index=True)

    differences = pd.DataFrame([
        paired_difference(c.name, per_scenario[c.name], per_scenario[baseline.name])
        for c in scenarios if c.name != baseline.name
    ]).sort_values("d_net_pnl", ascending=False, ignore_index=True)

    return summary, differences


def _format_summary(summary: pd.DataFrame) -> str:
    """Render the summary table with a mean +/- standard error column."""
    table = pd.DataFrame({
        "scenario": summary["scenario"],
        "cross |q|": summary["cross_q"].map("{:>6.0f}".format),
        "net PnL": summary["net_pnl"].map("{:>8.2f}".format),
        "SE": summary["se"].map("{:>6.2f}".format),
        "5th pct": summary["p5"].map("{:>8.2f}".format),
        "P(loss)": summary["p_loss"].map("{:>6.1%}".format),
        "fills": summary["fills"].map("{:>7.0f}".format),
        "|final inv|": summary["abs_inv"].map("{:>6.1f}".format),
        "max DD": summary["max_dd"].map("{:>7.2f}".format),
        "adv sel": summary["adv_sel"].map("{:>7.2f}".format),
        "adv SE": summary["adv_se"].map("{:>6.2f}".format),
        "P(halt)": summary["p_halt"].map("{:>6.1%}".format),
    })
    return table.to_string(index=False)


def _format_differences(differences: pd.DataFrame) -> str:
    """Render the paired difference table."""
    table = pd.DataFrame({
        "scenario": differences["scenario"],
        "d net PnL": differences["d_net_pnl"].map("{:>9.2f}".format),
        "SE": differences["se"].map("{:>6.2f}".format),
        "t": differences["t"].map("{:>8.1f}".format),
        "beats baseline": differences["win_rate"].map("{:>6.1%}".format),
    })
    return table.to_string(index=False)


def print_report(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    num_paths: int,
    base_seed: int,
    num_steps: int,
    wall_clock: float,
    quote_size: float,
) -> None:
    """
    Print both tables with the caveats a reader needs to interpret them.

    Args:
        quote_size: Size quoted per side in the grid, needed to say how many
            fills it takes to reach the crossing inventory.
    """
    print()
    print("=" * 108)
    print("MONTE CARLO BENCHMARK, RANKED BY MEAN NET PnL")
    print("=" * 108)
    print(
        f"{num_paths} paths per configuration, {num_steps} steps per path, "
        f"seeds {base_seed}..{base_seed + num_paths - 1}, "
        f"{wall_clock:.1f}s wall clock."
    )
    print(
        "All figures are means over those paths. SE is the standard error of "
        "the mean, not a standard deviation\nand not a confidence interval: "
        "roughly +/- 2 SE is the 95% interval. Net PnL is the mark-to-market "
        "number\nafter rebates and after charging the cost of liquidating "
        "residual inventory. P(loss) is the fraction of\npaths with net PnL "
        f"below zero. Adverse selection is a signed {DEFAULT_MARKOUT_HORIZON}"
        "-step markout, a diagnostic\nsplit of inventory PnL, so it is not a "
        "separate additive cost.\n'cross |q|' is the inventory at which the "
        "reservation price puts our own quote on the wrong side of the mid,"
        f"\nquote_spread/gamma, against a quoted size of {quote_size:g} per "
        "side. A mean ranks configurations but does not\nprice the inventory "
        "that produced it, so read the top of the table together with "
        "|final inv| and max DD.\nmax DD is measured after each price move and "
        "keeps accruing after a halt, so a halted configuration can\nshow a "
        "max DD above the limit that stopped it."
    )
    print()
    print(_format_summary(summary))

    print()
    print("=" * 108)
    print("PAIRED DIFFERENCE VERSUS BASELINE")
    print("=" * 108)
    print(
        "Every configuration ran the same seed sequence, so path k faces the "
        "same price path everywhere. The\ndifference is taken path by path, "
        "which cancels the shared price risk. t is the paired t-statistic on "
        f"{num_paths - 1}\ndegrees of freedom; |t| above about 2 is "
        "significant at the 5% level. 'beats baseline' is the fraction of "
        "paths\non which this configuration made more money than the baseline "
        "did on the same path."
    )
    print()
    print(_format_differences(differences))
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo benchmark of market-making configurations."
    )
    parser.add_argument(
        "--paths", type=int, default=DEFAULT_PATHS,
        help=f"Paths per configuration (default {DEFAULT_PATHS})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base seed; path k uses seed+k (default {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"Steps per path (default {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="Worker processes; 1 runs single process (default: cores - 2)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the grid and print the report."""
    args = parse_args(argv)
    if args.paths < 2:
        raise SystemExit("--paths must be at least 2 to report a standard error")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    scenarios = build_scenarios(num_steps=args.steps)
    print(
        f"Running {len(scenarios)} configurations x {args.paths} paths "
        f"on {args.workers} worker(s)."
    )

    started = time.perf_counter()
    summary, differences = run_benchmarks(
        scenarios, num_paths=args.paths, base_seed=args.seed, workers=args.workers
    )
    wall_clock = time.perf_counter() - started

    print_report(
        summary, differences,
        num_paths=args.paths, base_seed=args.seed,
        num_steps=args.steps, wall_clock=wall_clock,
        quote_size=scenarios[0].quote_size,
    )


if __name__ == "__main__":
    main()
