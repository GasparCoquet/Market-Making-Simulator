"""
Volatility units.

The simulator's `volatility` argument is a *per-step* log-return standard
deviation. Quoting it without a time unit is how the old benchmark ended up
labelling 0.02 as "2% volatility" when it actually meant 2% per step, which
compounds to roughly 20% over a 100-step run and 50% for the "5%" scenario.

Convert explicitly instead.
"""

import numpy as np

# 252 trading days x 6.5 hours x 3600 seconds
SECONDS_PER_TRADING_YEAR = 252 * 6.5 * 3600


def per_step_volatility(
    annual_volatility: float,
    seconds_per_step: float = 1.0,
) -> float:
    """
    Convert an annualised volatility to the per-step figure the simulator wants.

    Args:
        annual_volatility: Annualised log-return volatility, e.g. 0.25 for 25%
        seconds_per_step: Wall-clock seconds represented by one simulation step

    Returns:
        Per-step log-return standard deviation.

    Example:
        25% annualised on one-second steps is 1.03e-4 per step, so a $100 asset
        moves about one cent per step. Against a 5 cent quote that is a
        realistic regime. A "2% per step" asset moves $2 a step and no
        five-cent quote survives it.
    """
    if annual_volatility < 0:
        raise ValueError("annual_volatility must be non-negative")
    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")
    steps_per_year = SECONDS_PER_TRADING_YEAR / seconds_per_step
    return float(annual_volatility / np.sqrt(steps_per_year))


def annualised_volatility(
    per_step_vol: float,
    seconds_per_step: float = 1.0,
) -> float:
    """Inverse of `per_step_volatility`."""
    if per_step_vol < 0:
        raise ValueError("per_step_vol must be non-negative")
    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")
    steps_per_year = SECONDS_PER_TRADING_YEAR / seconds_per_step
    return float(per_step_vol * np.sqrt(steps_per_year))
