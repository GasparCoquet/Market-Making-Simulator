"""
Named calibrations for the two asset classes this simulator covers.

A word on the name. A "dataset" here is a **calibration**, not a recording.
Nothing in this repository replays a real feed, and the README says so at
length. What a `Dataset` fixes is the set of numbers that describe an
instrument and the venue it trades on: the price level, the width of the
market, the arrival intensity, the volatility and the calendar it is annualised
against, the fee denomination, and whether the thing ever closes. Two of them
ship: a US cash equity and a crypto perpetual swap.

Why the geometry is held fixed
------------------------------
The two calibrations are deliberately identical in every dimensionless
quantity. Both quote 5bp from the reservation price into a market 10bp wide.
Both use an arrival intensity that falls by 1/e over 5bp of distance. Both lean
the reservation price by 10bp of mid per clip of inventory, so both put their
own quote through the mid at half a clip. Only the price level and the lot size
differ, and they differ together so that a clip is $1,000 of notional in both.

That is not laziness, it is the experimental design. If the crypto calibration
also quoted a different width into a different book, every difference between
the two would be a mixture of market structure and a spread I invented, and
neither could be read off the other. Holding the geometry fixed means the
remaining differences are exactly the four this package set out to model:

  * the calendar the volatility is annualised against, which is what 24/7 is
  * the funding leg, which a perpetual has and an equity does not
  * the fee denomination, per share against basis points of notional
  * the session close, which an equity has and a perpetual does not

The cost of the design is that the crypto calibration is not a real perp. A BTC
perpetual is quoted far tighter than 10bp; the top of book is often under a
basis point. A calibration matching that would need a different arrival
intensity, and then the comparison would be measuring my guess at that
intensity rather than the four axes above. The absolute PnL levels here are
meaningless in both calibrations, exactly as the README already says of the
equity one, and only paired comparisons carry information.

Fees are a tier, not a property of the instrument. The defaults here are the
common retail tiers, a $0.0020 per share rebate on an equity maker-taker venue
and a 2bp maker fee on a perpetual. Both are ordinary, and both are movable
with `dataclasses.replace`, which is how the benchmark grid sweeps them.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .analytics.pnl_tracker import DEFAULT_MARKOUT_HORIZON, PnLTracker
from .engine.fill_model import FillModel
from .engine.funding import FundingModel
from .engine.market_state import MarketState
from .engine.simulator import MarketSimulator
from .risk.risk_manager import RiskManager
from .strategy.market_maker import MarketMaker
from .units import (
    SECONDS_PER_CALENDAR_YEAR,
    SECONDS_PER_TRADING_YEAR,
    per_step_volatility,
)

BASIS_POINT = 1e-4

# One 6.5-hour US equity session, on one-second steps.
EQUITY_SESSION_STEPS = int(6.5 * 3600)
# One hour, on one-second steps. The funding cadence Hyperliquid and dYdX use.
HOURLY_STEPS = 3600
# 0.01% per eight hours, settled hourly, which is the flat rate a perpetual
# pays when the premium sits inside its clamp. This is the modal funding rate,
# not an average of the observed distribution.
HOURLY_FUNDING_RATE = 0.0001 / 8


@dataclass(frozen=True)
class Dataset:
    """
    One instrument and venue, in the units the engine wants.

    Every price-scale quantity is stored dimensionless, in basis points of the
    initial mid or in clips of the quoted size, and converted to the absolute
    price units the engine takes by the properties below. Storing them absolute
    instead is how a calibration silently becomes untransferable: an arrival
    decay of `k = 20` per dollar means a 5bp quote on a $100 share and a
    0.008bp quote on a $100,000 contract, so the same number describes two
    completely different markets.

    Frozen, so a scenario grid can `dataclasses.replace` one axis at a time and
    ship the result to worker processes without any chance of shared mutation.
    """

    name: str
    asset_class: str
    unit: str
    description: str

    # Price scale. These two set every absolute number below.
    initial_mid: float
    quote_size: float

    # Dimensionless geometry, identical across the shipped calibrations.
    quote_spread_bps: float
    reference_half_spread_bps: float
    skew_bps_per_clip: float
    decay_bps: float
    max_inventory_clips: float
    mean_order_size_clips: float
    base_intensity: float
    informed_fraction: float

    # Time. `seconds_per_year` is the calendar the annualised figure is quoted
    # against and is the whole of "24/7" as far as the arithmetic is concerned.
    annual_volatility: float
    seconds_per_step: float
    seconds_per_year: float
    default_steps: int

    # Market structure.
    session_steps: Optional[int]
    funding_interval_steps: Optional[int]
    funding_rate_per_interval: float
    maker_rebate_per_unit: float
    maker_rebate_bps: float

    def __post_init__(self):
        if self.initial_mid <= 0:
            raise ValueError("initial_mid must be positive")
        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.decay_bps <= 0:
            raise ValueError("decay_bps must be positive")
        if self.max_inventory_clips <= 0:
            raise ValueError("max_inventory_clips must be positive")
        if self.mean_order_size_clips <= 0:
            raise ValueError("mean_order_size_clips must be positive")
        if self.default_steps < 1:
            raise ValueError("default_steps must be at least 1")
        if self.quote_spread_bps < 0:
            raise ValueError("quote_spread_bps must be non-negative")
        if self.reference_half_spread_bps < 0:
            raise ValueError("reference_half_spread_bps must be non-negative")

    # -- absolute price units, derived ------------------------------------

    @property
    def quote_spread(self) -> float:
        """Half-width of our quote, in price units."""
        return self.quote_spread_bps * BASIS_POINT * self.initial_mid

    @property
    def reference_half_spread(self) -> float:
        """The market's own half-spread, in price units."""
        return self.reference_half_spread_bps * BASIS_POINT * self.initial_mid

    @property
    def inventory_skew_factor(self) -> float:
        """gamma: price units the reservation price moves per unit held."""
        return (self.skew_bps_per_clip * BASIS_POINT * self.initial_mid
                / self.quote_size)

    @property
    def decay(self) -> float:
        """
        k: the arrival intensity's decay per price unit of distance.

        Set so that intensity falls by 1/e over `decay_bps` of the mid, which is
        what makes the same number mean the same thing on a $100 share and a
        $100,000 contract.
        """
        return 1.0 / (self.decay_bps * BASIS_POINT * self.initial_mid)

    @property
    def max_inventory(self) -> float:
        """Hard position limit, in units."""
        return self.max_inventory_clips * self.quote_size

    @property
    def mean_order_size(self) -> float:
        """Mean size of an arriving market order, in units."""
        return self.mean_order_size_clips * self.quote_size

    @property
    def step_volatility(self) -> float:
        """Per-step log-return sigma, on this dataset's calendar."""
        return per_step_volatility(
            self.annual_volatility, self.seconds_per_step, self.seconds_per_year)

    @property
    def notional_per_clip(self) -> float:
        """Value of one quoted clip at the initial mid."""
        return self.initial_mid * self.quote_size

    @property
    def crossing_inventory(self) -> float:
        """
        Inventory at which the reservation price puts our quote on the mid.

        Beyond it we are quoting through the mid and paying for our own fills.
        Infinite when the skew is off.
        """
        if self.inventory_skew_factor <= 0:
            return float("inf")
        return self.quote_spread / self.inventory_skew_factor

    @property
    def trades_around_the_clock(self) -> bool:
        """Whether the market ever closes."""
        return self.session_steps is None

    def maker_fee_bps_equivalent(self) -> float:
        """
        The maker fee expressed in basis points of notional, both legs together.

        A per-share rebate and a per-notional rebate are not comparable until
        one is restated in the other's units, and restating the per-share leg
        needs the price. At the initial mid, $0.0020 a share on a $100 share is
        0.2bp; the same rebate on a $10 share would be 2bp. Positive means we
        are paid to quote.
        """
        per_unit_in_bps = (self.maker_rebate_per_unit / self.initial_mid
                           / BASIS_POINT)
        return per_unit_in_bps + self.maker_rebate_bps

    # -- construction ------------------------------------------------------

    def build(
        self,
        random_seed: Optional[int] = None,
        markout_horizon: int = DEFAULT_MARKOUT_HORIZON,
        kill_switch_drawdown: Optional[float] = None,
        enable_size_throttle: bool = True,
    ) -> MarketSimulator:
        """
        Wire up an unrun simulator for this calibration.

        Args:
            random_seed: Seed for the simulator's per-instance generator
            markout_horizon: Steps ahead used for the adverse selection markout
            kill_switch_drawdown: Attach a risk overlay with the kill-switch
                armed at this drawdown in mark-to-market PnL. None attaches an
                overlay with throttling only, or no overlay at all when
                `enable_size_throttle` is also off.
            enable_size_throttle: Shrink quoted size near the position limit

        Returns:
            An unrun `MarketSimulator`. Run it with `self.step_volatility`, or
            call `self.run` which does that for you.
        """
        risk_manager: Optional[RiskManager] = None
        if kill_switch_drawdown is not None or enable_size_throttle:
            risk_manager = RiskManager(
                enable_kill_switch=kill_switch_drawdown is not None,
                drawdown_limit=kill_switch_drawdown,
                enable_size_throttle=enable_size_throttle,
            )

        funding_model: Optional[FundingModel] = None
        if self.funding_interval_steps is not None:
            funding_model = FundingModel(
                rate_per_interval=self.funding_rate_per_interval,
                interval_steps=self.funding_interval_steps,
            )

        return MarketSimulator(
            market_state=MarketState(
                initial_mid=self.initial_mid,
                reference_half_spread=self.reference_half_spread,
            ),
            market_maker=MarketMaker(
                quote_spread=self.quote_spread,
                quote_size=self.quote_size,
                max_inventory=self.max_inventory,
                inventory_skew_factor=self.inventory_skew_factor,
                maker_rebate_per_unit=self.maker_rebate_per_unit,
                maker_rebate_bps=self.maker_rebate_bps,
                risk_manager=risk_manager,
            ),
            pnl_tracker=PnLTracker(markout_horizon=markout_horizon),
            fill_model=FillModel(
                base_intensity=self.base_intensity,
                decay=self.decay,
                mean_order_size=self.mean_order_size,
                informed_fraction=self.informed_fraction,
            ),
            random_seed=random_seed,
            funding_model=funding_model,
            session_steps=self.session_steps,
        )

    def run(
        self,
        random_seed: Optional[int] = None,
        num_steps: Optional[int] = None,
        **build_kwargs,
    ) -> Tuple[MarketSimulator, Dict]:
        """
        Build, run and summarise one path.

        Args:
            random_seed: Seed for this path
            num_steps: Steps to run, defaulting to the dataset's own horizon
            **build_kwargs: Passed through to `build`

        Returns:
            (simulator, summary dict).
        """
        simulator = self.build(random_seed=random_seed, **build_kwargs)
        simulator.run(
            num_steps=self.default_steps if num_steps is None else num_steps,
            volatility=self.step_volatility,
            dt=1.0,
        )
        return (simulator, simulator.get_summary())

    # -- reporting ---------------------------------------------------------

    def hours(self, num_steps: int) -> float:
        """Wall-clock hours a run of `num_steps` steps represents."""
        return num_steps * self.seconds_per_step / 3600.0

    def describe(self, num_steps: Optional[int] = None) -> str:
        """
        Multi-line description of the calibration, for a run banner.

        Every line is read off the object rather than written out, so a printed
        configuration cannot drift away from the one that was actually run.
        """
        steps = self.default_steps if num_steps is None else num_steps
        calendar = ("24/7" if self.seconds_per_year == SECONDS_PER_CALENDAR_YEAR
                    else "session")
        session = ("never closes"
                   if self.session_steps is None
                   else f"flatten every {self.session_steps} steps "
                        f"({self.hours(self.session_steps):.1f}h)")
        if self.funding_interval_steps is None:
            funding = "none (not a perpetual)"
        else:
            per_day = (self.funding_rate_per_interval
                       * (86400 / (self.funding_interval_steps
                                   * self.seconds_per_step)))
            funding = (f"{self.funding_rate_per_interval:.3e} every "
                       f"{self.funding_interval_steps} steps "
                       f"({self.hours(self.funding_interval_steps):.0f}h), "
                       f"{per_day / BASIS_POINT:.1f}bp/day")

        return "\n".join([
            f"  Dataset:              {self.name} ({self.asset_class})",
            f"  Instrument:           {self.description}",
            f"  Price / clip:         ${self.initial_mid:,.2f} x "
            f"{self.quote_size:g} {self.unit} = "
            f"${self.notional_per_clip:,.0f} notional",
            f"  Market width:         "
            f"{2 * self.reference_half_spread_bps:.1f}bp "
            f"(${2 * self.reference_half_spread:,.4f})",
            f"  Our quote:            "
            f"{self.quote_spread_bps:.1f}bp from reservation "
            f"(${self.quote_spread:,.4f}), crossing at "
            f"{self.crossing_inventory / self.quote_size:.1f} clips",
            f"  Fill model:           A={self.base_intensity}, "
            f"1/e over {self.decay_bps:.1f}bp (k={self.decay:g}), "
            f"informed={self.informed_fraction}",
            f"  Maker fee:            "
            f"{self.maker_rebate_per_unit:+.4f}/{self.unit} and "
            f"{self.maker_rebate_bps:+.2f}bp = "
            f"{self.maker_fee_bps_equivalent():+.2f}bp of notional",
            f"  Funding:              {funding}",
            f"  Session:              {session}",
            f"  Calendar:             {calendar}, "
            f"{self.seconds_per_year:,.0f}s/year",
            f"  Volatility:           {self.annual_volatility:.0%} annualised "
            f"= {self.step_volatility:.3e} per "
            f"{self.seconds_per_step:g}s step",
            f"  Horizon:              {steps} steps "
            f"({self.hours(steps):.1f}h)",
        ])


# The calibration the rest of this repository already ran on, restated in
# dimensionless units. `example.py` reproduces its published output from this
# object, so the two cannot drift apart.
#
# The 10bp market width is wide for a $100 large cap, where the touch is
# usually a cent or two. It is the width the fill model was chosen against and
# the README defends it as a plausible regime rather than a fitted one, so it
# is kept rather than quietly improved.
US_EQUITY = Dataset(
    name="us-equity",
    asset_class="cash equity",
    unit="share",
    description="$100 US cash equity, 6.5-hour session, maker-taker venue",
    initial_mid=100.0,
    quote_size=10.0,
    quote_spread_bps=5.0,
    reference_half_spread_bps=10.0,
    skew_bps_per_clip=10.0,
    decay_bps=5.0,
    max_inventory_clips=10.0,
    mean_order_size_clips=1.0,
    base_intensity=0.8,
    informed_fraction=0.30,
    annual_volatility=0.25,
    seconds_per_step=1.0,
    seconds_per_year=SECONDS_PER_TRADING_YEAR,
    # 2,000 steps is 33 minutes, well inside one session, and is the horizon
    # every published equity figure in the README was produced at. The session
    # close therefore does not fire on a default run; `--steps 46800` runs two
    # full sessions and is the invocation the README quotes for that mechanic.
    default_steps=2000,
    session_steps=EQUITY_SESSION_STEPS,
    funding_interval_steps=None,
    funding_rate_per_interval=0.0,
    # $0.0020 a share is an ordinary maker rebate on a US maker-taker venue.
    # Top tiers reach about $0.0030 and inverted venues charge instead.
    maker_rebate_per_unit=0.0020,
    maker_rebate_bps=0.0,
)

# The same geometry on a perpetual swap. Everything that differs from
# US_EQUITY below the price scale is one of the four axes this package models.
CRYPTO_PERP = Dataset(
    name="crypto-perp",
    asset_class="perpetual swap",
    unit="contract",
    description="$100,000 BTC perpetual, 24/7, taker-fee venue",
    initial_mid=100_000.0,
    # 0.01 contracts is $1,000 of notional, the same clip the equity quotes.
    quote_size=0.01,
    quote_spread_bps=5.0,
    reference_half_spread_bps=10.0,
    skew_bps_per_clip=10.0,
    decay_bps=5.0,
    max_inventory_clips=10.0,
    mean_order_size_clips=1.0,
    base_intensity=0.8,
    informed_fraction=0.30,
    # 55% annualised is an ordinary realised figure for BTC. On the 24/7
    # calendar it is 9.79e-05 of per-second sigma against the equity's
    # 1.03e-04, so the headline number is 2.2x higher and the per-second move
    # is slightly smaller. That is the arithmetic of 24/7, not a coincidence.
    annual_volatility=0.55,
    seconds_per_step=1.0,
    seconds_per_year=SECONDS_PER_CALENDAR_YEAR,
    # One full day, which is the natural unit for an instrument whose defining
    # property is that a day has no boundary in it. It is also the shortest
    # horizon on which all three crypto mechanics fire more than once: 24
    # hourly funding payments, 3 at the eight-hourly convention, and 3 closes
    # if an equity-style session is imposed on it for comparison.
    default_steps=24 * 3600,
    session_steps=None,
    funding_interval_steps=HOURLY_STEPS,
    funding_rate_per_interval=HOURLY_FUNDING_RATE,
    maker_rebate_per_unit=0.0,
    # A 2bp maker fee is the standard non-VIP perpetual rate on Binance, Bybit
    # and OKX. Only the top volume tiers reach zero or a rebate.
    maker_rebate_bps=-2.0,
)

DATASETS: Dict[str, Dataset] = {
    US_EQUITY.name: US_EQUITY,
    CRYPTO_PERP.name: CRYPTO_PERP,
}

DATASET_NAMES = tuple(DATASETS)


def get_dataset(name: str) -> Dataset:
    """
    Look up a shipped calibration by name.

    Args:
        name: One of `DATASET_NAMES`

    Returns:
        The dataset.

    Raises:
        KeyError: With the available names listed, because a typo here
            otherwise surfaces as a bare KeyError several frames away.
    """
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; available: {', '.join(DATASET_NAMES)}"
        ) from None


__all__ = [
    "Dataset",
    "US_EQUITY",
    "CRYPTO_PERP",
    "DATASETS",
    "DATASET_NAMES",
    "get_dataset",
]
