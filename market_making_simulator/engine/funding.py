"""
FundingModel: the perpetual-swap funding leg.

A perpetual swap has no expiry, so nothing mechanically drags its price back to
spot. Venues supply the drag with a periodic cash transfer between longs and
shorts, sized as a rate on the position's notional:

    payment = -inventory * mark_price * rate_per_interval

paid at fixed timestamps. A positive rate means the perp trades above spot and
longs pay shorts, so a long book has a negative cash flow and a short book a
positive one. The sign convention here is that the returned number is *our*
cash flow: negative when we pay.

Two conventions are in use and both are expressible here. Binance, Bybit and
OKX settle every eight hours; Hyperliquid and dYdX settle hourly at an eighth
of the rate. `funding_interval_steps` and `rate_per_interval` are separate
arguments precisely so that a change of convention is a change of two numbers
whose product per unit time is unchanged, which is a property the benchmark
grid tests rather than assumes.

What this is not
----------------
The rate is a constant. Real funding is a market price: it tracks the perp's
premium over the index, mean reverts, and blows out to tens of basis points per
interval in a squeeze. A stochastic rate is deliberately not modelled, for two
reasons. It would consume draws from the shared generator and break the common
random numbers that make the benchmark's paired comparisons sharp, and it would
introduce a funding-to-price correlation that this model has no honest way to
calibrate. The stressed-rate scenario in the benchmark grid is the substitute:
turn the constant up and read the sensitivity off the table.

Funding is a *carry* term, not a *transaction* term. It scales with
|inventory| x time, not with volume, which is why it behaves nothing like the
fee change that comes with it in a crypto calibration.
"""


class FundingModel:
    """Periodic funding payments on the carried position."""

    def __init__(
        self,
        rate_per_interval: float = 1.25e-05,
        interval_steps: int = 3600,
    ):
        """
        Args:
            rate_per_interval: Funding rate applied at each payment, as a
                fraction of position notional. The default is 1.25e-05, which
                is the ubiquitous 0.01% per eight hours settled hourly. Positive
                means longs pay shorts. Negative rates are allowed and are what
                a perp trading below spot looks like.
            interval_steps: Steps between payments. With one-second steps, 3600
                is hourly and 28800 is eight-hourly.
        """
        if int(interval_steps) != interval_steps or interval_steps < 1:
            raise ValueError("interval_steps must be a positive whole number")

        self.rate_per_interval = float(rate_per_interval)
        self.interval_steps = int(interval_steps)

    def is_payment_step(self, step_index: int) -> bool:
        """
        Whether a payment settles at the end of step `step_index`.

        Steps are zero-indexed, so the test is on `step_index + 1`: with an
        hourly interval on one-second steps the first payment lands at the end
        of step 3599, which is one hour of elapsed time, not 3600 seconds and
        one step.
        """
        return (step_index + 1) % self.interval_steps == 0

    def payment(self, inventory: float, mark_price: float) -> float:
        """
        Our cash flow at a funding timestamp.

        Args:
            inventory: Signed position at the timestamp
            mark_price: Price the notional is marked at

        Returns:
            Signed cash flow. Negative when we pay, which is what being long a
            positive-rate perp costs.
        """
        return -inventory * mark_price * self.rate_per_interval

    def payments_in(self, num_steps: int) -> int:
        """How many payments a run of `num_steps` steps settles."""
        return int(num_steps // self.interval_steps)

    def __repr__(self) -> str:
        return (f"FundingModel(rate={self.rate_per_interval:.3e}/interval, "
                f"interval={self.interval_steps} steps)")
