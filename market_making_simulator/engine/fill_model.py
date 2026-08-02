"""
FillModel: distance-dependent arrival intensity.

The single most important property of a market-making simulator is that
quoting further from the mid must reduce the fill rate. Without that, PnL is
linear in the quoted spread and the optimal strategy is an infinitely wide
quote, which makes every downstream conclusion an artefact.

Model (Avellaneda-Stoikov 2008):

    lambda(delta) = A * exp(-k * delta)
    P(fill within dt) = 1 - exp(-lambda * delta_t)

where `delta` is the distance of our quote from the mid. A quote at the mid
(delta = 0) fills with intensity A; each unit of distance multiplies the
intensity by exp(-k).

Order sizes are drawn from an exponential distribution. The fill is capped at
our own quoted size, so a large incoming order fills us partially.

Informed flow (Glosten-Milgrom 1985, in spirit): a fraction of arrivals know
the direction of the next price move and trade with it. They lift our ask
before a rise and hit our bid before a fall. With `informed_fraction = 0` the
flow is uninformed and measured adverse selection is zero in expectation; the
metric only becomes negative because informed flow was actually simulated.
"""

import numpy as np

# Exponent clamp: our quotes can be skewed through the mid at high inventory,
# which makes `delta` negative and the raw intensity explode. Clamping keeps
# the probability well defined without changing behaviour in the normal range.
_MAX_EXPONENT = 50.0

# Number of uniform draws consumed per step, regardless of which branch is
# taken. Holding this constant is what allows common random numbers to line up
# across configurations in the Monte Carlo harness.
DRAWS_PER_STEP = 5


class FillModel:
    """Arrival intensity, order sizing and informed flow."""

    def __init__(
        self,
        base_intensity: float = 0.8,
        decay: float = 20.0,
        mean_order_size: float = 10.0,
        informed_fraction: float = 0.0,
    ):
        """
        Args:
            base_intensity: A, the arrival intensity for a quote at the mid
            decay: k, how fast intensity decays with distance from the mid.
                With the defaults, a quote 5 cents from the mid fills with
                probability 1 - exp(-0.8 * exp(-1)) = 25.5% per unit of time.
            mean_order_size: Mean size of an arriving market order (exponential)
            informed_fraction: Probability that an arrival is informed
        """
        if base_intensity < 0:
            raise ValueError("base_intensity must be non-negative")
        if decay < 0:
            raise ValueError("decay must be non-negative")
        if mean_order_size <= 0:
            raise ValueError("mean_order_size must be positive")
        if not 0.0 <= informed_fraction <= 1.0:
            raise ValueError("informed_fraction must be in [0, 1]")

        self.base_intensity = base_intensity
        self.decay = decay
        self.mean_order_size = mean_order_size
        self.informed_fraction = informed_fraction

    def intensity(self, distance: float) -> float:
        """lambda(delta) = A * exp(-k * delta)."""
        exponent = np.clip(-self.decay * distance, -_MAX_EXPONENT, _MAX_EXPONENT)
        return float(self.base_intensity * np.exp(exponent))

    def fill_probability(self, distance: float, dt: float = 1.0) -> float:
        """P(at least one arrival within dt) = 1 - exp(-lambda * dt)."""
        return float(1.0 - np.exp(-self.intensity(distance) * dt))

    def _size_from_uniform(self, u: float) -> float:
        """Inverse-CDF sample of an exponential with mean `mean_order_size`."""
        # u is in [0, 1); guard the log at the boundary.
        return float(-self.mean_order_size * np.log(1.0 - min(u, 1.0 - 1e-12)))

    def sample_arrivals(
        self,
        rng: np.random.Generator,
        bid_distance: float,
        ask_distance: float,
        dt: float,
        next_log_return: float,
    ):
        """
        Draw the market orders arriving this step.

        Consumes exactly DRAWS_PER_STEP uniforms from `rng` on every path
        through the function.

        Args:
            rng: Per-instance generator
            bid_distance: mid - our bid price
            ask_distance: our ask price - mid
            dt: Time step
            next_log_return: The price move about to happen. Informed traders
                see it; uninformed traders do not.

        Returns:
            (bid_order_size, ask_order_size). `bid_order_size` is a market SELL
            hitting our bid, so we buy. `ask_order_size` is a market BUY
            lifting our ask, so we sell. Either may be 0.0.
        """
        u_informed, u_bid, u_ask, u_size_bid, u_size_ask = rng.random(DRAWS_PER_STEP)

        p_bid = self.fill_probability(bid_distance, dt)
        p_ask = self.fill_probability(ask_distance, dt)

        if u_informed < self.informed_fraction:
            # Informed arrivals are one-sided: they take the side that profits
            # from the move that is about to happen.
            if next_log_return >= 0.0:
                # Price is about to rise, so an informed trader buys from us.
                size = self._size_from_uniform(u_size_ask) if u_ask < p_ask else 0.0
                return (0.0, size)
            # Price is about to fall, so an informed trader sells to us.
            size = self._size_from_uniform(u_size_bid) if u_bid < p_bid else 0.0
            return (size, 0.0)

        bid_size = self._size_from_uniform(u_size_bid) if u_bid < p_bid else 0.0
        ask_size = self._size_from_uniform(u_size_ask) if u_ask < p_ask else 0.0
        return (bid_size, ask_size)

    def __repr__(self) -> str:
        return (f"FillModel(A={self.base_intensity}, k={self.decay}, "
                f"mean_size={self.mean_order_size}, "
                f"informed={self.informed_fraction})")
