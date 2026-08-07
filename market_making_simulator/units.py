"""
Volatility units, and the calendar an annualised figure is quoted against.

The simulator's `volatility` argument is a *per-step* log-return standard
deviation. Quoting it without a time unit is how the old benchmark ended up
labelling 0.02 as "2% volatility" when it actually meant 2% per step, which
compounds to roughly 20% over a 100-step run and 50% for the "5%" scenario.

Convert explicitly instead. There is a second unit hiding behind the first:
"annualised" means nothing until you say how many seconds are in the year. A
cash equity trades 252 sessions of 6.5 hours, which is 5,896,800 seconds. A
perpetual swap trades every second of every day, which is 31,536,000, so the
crypto year is 5.35 times longer and the same headline volatility spreads over
sqrt(5.35) = 2.31 times more standard deviations.

That single ratio is most of what "24/7" means quantitatively: 55% annualised
on the crypto calendar is 9.78e-05 of per-second sigma, and 25% annualised on
the equity calendar is 1.03e-04. The crypto asset carries a headline volatility
2.2 times higher and moves *less* per second. Feeding a 24/7 asset's annualised
volatility through the equity calendar overstates its per-second sigma by 2.31x,
which is the mistake this module exists to make hard.
"""

import numpy as np

# 252 trading days x 6.5 hours x 3600 seconds. The calendar a cash equity's
# annualised volatility is quoted against: the overnight gap is realised in the
# opening print, not accumulated second by second.
SECONDS_PER_TRADING_YEAR = 252 * 6.5 * 3600

# 365 days x 24 hours x 3600 seconds. Perpetual swaps and spot crypto never
# close, so every second of the year is a trading second and there is no gap to
# exclude.
SECONDS_PER_CALENDAR_YEAR = 365 * 24 * 3600


def per_step_volatility(
    annual_volatility: float,
    seconds_per_step: float = 1.0,
    seconds_per_year: float = SECONDS_PER_TRADING_YEAR,
) -> float:
    """
    Convert an annualised volatility to the per-step figure the simulator wants.

    Args:
        annual_volatility: Annualised log-return volatility, e.g. 0.25 for 25%
        seconds_per_step: Wall-clock seconds represented by one simulation step
        seconds_per_year: Trading seconds in a year for this asset's calendar.
            Defaults to the equity session calendar, so existing callers are
            unaffected. Pass `SECONDS_PER_CALENDAR_YEAR` for a 24/7 asset.

    Returns:
        Per-step log-return standard deviation.

    Example:
        25% annualised on one-second steps is 1.03e-4 per step, so a $100 asset
        moves about one cent per step. Against a 5 cent quote that is a
        realistic regime. A "2% per step" asset moves $2 a second and no
        five-cent quote survives it.

        The same call with `seconds_per_year=SECONDS_PER_CALENDAR_YEAR` gives
        4.45e-5, because a 24/7 year holds 5.35 times as many seconds for the
        same annual dispersion to spread over.
    """
    if annual_volatility < 0:
        raise ValueError("annual_volatility must be non-negative")
    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")
    if seconds_per_year <= 0:
        raise ValueError("seconds_per_year must be positive")
    steps_per_year = seconds_per_year / seconds_per_step
    return float(annual_volatility / np.sqrt(steps_per_year))


def annualised_volatility(
    per_step_vol: float,
    seconds_per_step: float = 1.0,
    seconds_per_year: float = SECONDS_PER_TRADING_YEAR,
) -> float:
    """
    Inverse of `per_step_volatility`.

    Args:
        per_step_vol: Per-step log-return standard deviation
        seconds_per_step: Wall-clock seconds represented by one simulation step
        seconds_per_year: Trading seconds in a year for this asset's calendar.
            Must match the calendar the per-step figure was built with, or the
            round trip is off by the square root of the ratio between them.
    """
    if per_step_vol < 0:
        raise ValueError("per_step_vol must be non-negative")
    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")
    if seconds_per_year <= 0:
        raise ValueError("seconds_per_year must be positive")
    steps_per_year = seconds_per_year / seconds_per_step
    return float(per_step_vol * np.sqrt(steps_per_year))
