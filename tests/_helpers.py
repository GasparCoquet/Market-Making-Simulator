"""
Shared fixtures for the test suite.

Deliberately not named `test_*.py` so that unittest discovery does not collect
it. Monte Carlo results are cached, because several tests read different
statistics off the same set of paths and the suite has a runtime budget.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from market_making_simulator import (
    FillModel,
    MarketMaker,
    MarketSimulator,
    MarketState,
    PnLTracker,
    RiskManager,
    per_step_volatility,
)

# 25% annualised on one-second steps, which is 1.03e-4 per step. Every test that
# needs a price path uses this rather than a bare number, because a per-step
# volatility quoted without a time unit is how "2%" ends up meaning a $2 move on
# a $100 asset against a 5 cent quote.
STEP_VOLATILITY = per_step_volatility(0.25, seconds_per_step=1.0)


def build_simulation(
    seed: Optional[int],
    quote_spread: float = 0.05,
    quote_size: float = 10.0,
    max_inventory: float = 100.0,
    inventory_skew_factor: float = 0.01,
    maker_rebate_per_unit: float = 0.0,
    informed_fraction: float = 0.0,
    reference_half_spread: float = 0.10,
    markout_horizon: int = 5,
    risk_manager: Optional[RiskManager] = None,
) -> MarketSimulator:
    """
    Wire up a simulator with the standard components.

    Args:
        seed: Seed for the simulator's per-instance generator
        quote_spread: Half-width of our quote either side of the reservation price
        quote_size: Size quoted on each side
        max_inventory: Hard position limit
        inventory_skew_factor: gamma, the reservation price shift per unit held
        maker_rebate_per_unit: Rebate received per unit filled
        informed_fraction: Share of arrivals that trade with the coming move
        reference_half_spread: The market's own half-spread, used for liquidation
        markout_horizon: Steps ahead used for the adverse selection markout
        risk_manager: Optional risk overlay

    Returns:
        An unrun MarketSimulator.
    """
    market_state = MarketState(
        initial_mid=100.0,
        reference_half_spread=reference_half_spread,
    )
    market_maker = MarketMaker(
        quote_spread=quote_spread,
        quote_size=quote_size,
        max_inventory=max_inventory,
        inventory_skew_factor=inventory_skew_factor,
        maker_rebate_per_unit=maker_rebate_per_unit,
        risk_manager=risk_manager,
    )
    return MarketSimulator(
        market_state=market_state,
        market_maker=market_maker,
        pnl_tracker=PnLTracker(markout_horizon=markout_horizon),
        fill_model=FillModel(informed_fraction=informed_fraction),
        random_seed=seed,
    )


def run_simulation(
    seed: Optional[int],
    num_steps: int = 1000,
    volatility: Optional[float] = None,
    **kwargs,
) -> Tuple[MarketSimulator, Dict]:
    """
    Build, run and summarise one path.

    Args:
        seed: Seed for the simulator's generator
        num_steps: Number of steps to run
        volatility: Per-step log-return volatility, defaults to STEP_VOLATILITY
        **kwargs: Passed through to `build_simulation`

    Returns:
        (simulator, summary dict).
    """
    simulator = build_simulation(seed, **kwargs)
    simulator.run(
        num_steps=num_steps,
        volatility=STEP_VOLATILITY if volatility is None else volatility,
    )
    return simulator, simulator.get_summary()


@lru_cache(maxsize=None)
def adverse_selection_sample(
    informed_fraction: float,
    num_seeds: int,
    num_steps: int = 2000,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Monte Carlo sample of adverse selection and net PnL across seeds.

    Cached, because the informed-flow tests read several statistics off the same
    paths and each call costs a few seconds.

    Args:
        informed_fraction: Share of arrivals that trade with the coming move
        num_seeds: Number of independent paths
        num_steps: Steps per path

    Returns:
        (adverse selection per seed, net PnL per seed).
    """
    adverse: List[float] = []
    net: List[float] = []
    for seed in range(num_seeds):
        _, summary = run_simulation(
            seed,
            num_steps=num_steps,
            informed_fraction=informed_fraction,
        )
        adverse.append(float(summary['adverse_selection']))
        net.append(float(summary['net_pnl']))
    return (tuple(adverse), tuple(net))
