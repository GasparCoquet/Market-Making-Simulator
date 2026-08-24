# Market-Making Simulator

A market-making simulator where quoting further from the mid actually reduces
your fill rate, inventory skew leans the right way, and PnL decomposes into
spread capture and inventory risk as an exact identity rather than as three
numbers that are added up and hoped to agree.

Those three properties sound like table stakes. Getting any of them wrong makes
a simulator produce conclusions that are artefacts of its own bugs, which is
what happened to this one before v0.2 and is documented at the bottom.

One engine, two calibrations. The same model quotes a US cash equity and a
crypto perpetual swap, and the two are held identical in every dimensionless
quantity so that the difference between them is market structure and nothing
else, namely the calendar, the funding leg, the fee denomination and the
session close. See [Crypto](#crypto-the-same-engine-on-a-second-dataset).

## What this is, and what it is not

**It is** a continuous-time market-making model in the Avellaneda-Stoikov
family, with a mid-price diffusion, a fill intensity that decays with quote
distance, an inventory-linked reservation price, and a Monte Carlo harness that
compares configurations with paired standard errors.

**It is not a limit order book.** There is no queue, no price-time priority, no
per-level depth and no impact from our own trades. Nothing here reconstructs a
real feed. If you need queue position, this is the wrong model, and the
[Limitations](#limitations) section says so in more detail.

**A "dataset" here is a calibration, not a recording.** Nothing replays a real
feed on either asset class. What a dataset fixes is the price level, the width
of the market, the arrival intensity, the volatility and the calendar it is
annualised against, the fee denomination, and whether the instrument ever
closes. Absolute PnL levels are meaningless in both. Only the paired
comparisons carry information.

## The model

### Price

Geometric Brownian motion with the Itô correction, so the mid is a martingale
and volatility scenarios are comparable:

```
m_{t+1} = m_t * exp( -0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z ),   Z ~ N(0,1)
```

`sigma` is a **per-step** log-return volatility. Use
`units.per_step_volatility(annual_vol, seconds_per_step, seconds_per_year)` to
get it from an annualised figure. 25% annualised on one-second steps is
`1.030e-04` per step. Quoting volatility without a time unit is how the
previous version ended up labelling `0.02` as "2% volatility" when it meant 2%
*per step*, a $2 move on a $100 asset against a five-cent quote.

There is a second unit hiding behind the first, and it is the one that matters
for crypto, because "annualised" means nothing until you say how many seconds
are in the year. `seconds_per_year` defaults to the equity session calendar
(`252 * 6.5 * 3600`). Pass `SECONDS_PER_CALENDAR_YEAR` for anything that
trades around the clock.

### Fills

Arrival intensity decays exponentially in the distance of our quote from the mid:

```
lambda(delta) = A * exp(-k * delta)
P(fill within dt) = 1 - exp(-lambda(delta) * dt)
```

Order sizes are exponential with mean `mean_order_size`, and the fill is capped
at our own quoted size, so large arrivals fill us partially. Defaults are
`A = 0.8`, `k = 20.0`. A quote at the mid fills with probability 55.1% per unit
of time, one five cents out with 25.5%, one twenty cents out with 1.45%.

This is the piece that has to exist. Without it, PnL is linear in the quoted
spread and the optimal strategy is an infinitely wide quote.

### Informed flow

A fraction `informed_fraction` of arrivals see the next price move and trade
with it, lifting our ask before a rise and hitting our bid before a fall. This
is Glosten-Milgrom in spirit rather than in detail. It matters because adverse
selection is only measurable if some flow is actually informed, and at
`informed_fraction = 0` the measured value is statistically zero, as it should be.

### Quoting

Quotes sit symmetrically around a reservation price that leans against inventory:

```
r   = m - q * gamma
bid = r - quote_spread
ask = r + quote_spread
```

The quoted width is exactly `2 * quote_spread` at every inventory level, so the
quotes can never cross. When we are long, **both** quotes move down. The ask
gets easier to lift and the bid harder to hit, so the position bleeds off.

The intuitive-sounding alternative, "long inventory, widen the ask", does the
opposite of what is wanted, because a wider ask is *less* likely to be lifted.

Note the failure mode this rule has, which the benchmark measures rather than
hides. The ask reaches the mid at `|q| = quote_spread / gamma`. Past that
inventory we are quoting through the mid and paying for our own fills. At
`gamma = 0.05` with `quote_spread = 0.05` that threshold is one unit against a
ten-unit clip, which is why the aggressive-skew row loses $1,610.

### PnL

The decomposition is an identity, asserted to 1e-9 in the test suite:

```
gross_pnl = spread_capture + inventory_pnl
spread_capture = sum over fills of  q_signed * (m_t - fill_price)
inventory_pnl  = sum over steps of  inventory_after_fills(t) * (m_{t+1} - m_t)
```

Proof: let `V = cash + inventory * mid`. A fill of signed size `q` at price `p`
when the mid is `m` changes cash by `-p*q` and inventory by `+q`, so `V` moves
by `q*(m-p)`, the spread term. Between fills `V` moves by `inventory * dm`, the
inventory term. `V` starts at zero, so `V_T` is exactly the sum of the two.

The reported waterfall then adds the real cash flows that sit outside the
trading leg:

```
net_pnl = gross_pnl + rebates + funding - liquidation_cost
```

where `liquidation_cost` charges the market's half-spread on residual
inventory, so a strategy cannot look profitable by carrying an open position
marked at mid. `rebates` and `funding` are held outside `cash` precisely so
that the identity above stays exact. Folding either into the trading cash would
break the reconciliation between the tracker's decomposition and the maker's
own mark to market.

**Adverse selection is not a third additive bucket.** It is a signed `h`-step
markout, `sum of q_signed * (m_{t+h} - m_t)`, which is a *split* of the
inventory term and is reported separately. It is negative when we are
systematically on the wrong side of the next move, positive when we are on the
right side, and zero in expectation under uninformed flow.

**The close-out cost is not one either.** Flattening at a session close is an
ordinary trade at a bad price, so it is already inside `spread_capture` as
negative edge, and `quoted_edge - close_out_cost == spread_capture` exactly.
It is reported beside adverse selection for the same reason. Putting it in the
waterfall would subtract it twice, which is the mistake v0.1 made with the
markout.

### Market structure

Three things are properties of the instrument and the venue rather than of the
quoting rule, and each is off by default:

- **Funding.** Attaching a `FundingModel` gives the instrument a perpetual-swap
  funding leg. At fixed timestamps, `-inventory * mark_price * rate` settles as
  cash. Positive rate means longs pay shorts.
- **The session close.** Setting `session_steps` gives it a close, at which the
  book is flattened by crossing the market's reference half-spread.
- **Fee denomination.** `maker_rebate_per_unit` is the US equity convention,
  cents per share regardless of price. `maker_rebate_bps` is the crypto
  convention, basis points of notional. Negative means a fee. They are
  additive, and a venue normally uses one.

Neither funding nor the session close draws from the random generator, so
turning either on leaves the price path and the arrival sequence untouched and
the Monte Carlo's paired comparisons stay exact.

## Install

```bash
git clone https://github.com/GasparCoquet/Market-Making-Simulator.git
cd Market-Making-Simulator
pip install -e .
```

Requires Python 3.9 or newer. `pip install -e ".[dev]"` adds pytest.

## Run

```bash
python example.py                        # one seeded equity path, 4 plots
python example.py --dataset crypto-perp  # the same, on a perpetual swap
python benchmarks.py                     # 12 equity configs x 500 paths
python benchmarks.py --dataset crypto-perp   # 10 perp configs x 100 paths
```

`example.py` is headless by default and writes PNGs to `plots/<dataset>/`. Pass
`--show` to open them, `--steps` for a different horizon, `--drawdown-limit 0`
to disable the kill-switch. `benchmarks.py` takes `--dataset`, `--paths`,
`--steps`, `--seed` and `--workers`. The perpetual grid runs a full 24-hour day
per path, so at its defaults it takes about ten minutes on four cores against
just over one for the equity grid.

### Example output

Stdout of `python example.py` at seed 42. Cut for length: the banner, the "Run"
and "Market" sections, one `risk manager halted = False` line, the quote ladder
and the plot paths. Nothing else is altered, no number is touched, and running
the command reproduces every figure below exactly.

```
Configuration:
  Dataset:              us-equity (cash equity)
  Instrument:           $100 US cash equity, 6.5-hour session, maker-taker venue
  Price / clip:         $100.00 x 10 share = $1,000 notional
  Market width:         20.0bp ($0.2000)
  Our quote:            5.0bp from reservation ($0.0500), crossing at 0.5 clips
  Fill model:           A=0.8, 1/e over 5.0bp (k=20), informed=0.3
  Maker fee:            +0.0020/share and +0.00bp = +0.20bp of notional
  Funding:              none (not a perpetual)
  Session:              flatten every 23400 steps (6.5h)
  Calendar:             session, 5,896,800s/year
  Volatility:           25% annualised = 1.030e-04 per 1s step
  Horizon:              2000 steps (0.6h)
  Risk manager:         kill-switch at $200.00 of drawdown, size throttle on
  Seed:                 42

Activity:
  Fills:                1134 (564 buys, 570 sells)
  Filled volume:        7,158.5 units
  Fill rate:            0.57 fills/step
  Final inventory:      -5.92 units

PnL waterfall:
  Spread capture:       $    104.15
  Inventory PnL:        $     -9.08
  ----------------------------------------
  Gross PnL (at mid):   $     95.07
  Maker rebates:        $     14.32
  Funding:              $      0.00
  Liquidation cost:     $     -0.59
  ----------------------------------------
  Net PnL:              $    108.79

Diagnostics:
  Adverse selection:    $    -10.35  (signed markout, part of inventory PnL)
  Max drawdown:         $      1.94
  Net cash flow:        $    685.14  (not a PnL)
  Inventory value:      $   -590.07  (cash flow + this = gross PnL)

Reconciliation:
  spread_capture + inventory_pnl - gross_pnl          = 0.00e+00
  gross + rebates + funding - liquidation - net       = 0.00e+00
  quoted_edge - close_out_cost - spread_capture       = 0.00e+00
  min(ask - bid) over the run                         = $0.1000  (0 crossed steps)
  funding payments settled                            = 0
  session closes                                      = 0
```

Note the `net cash flow` line. It is `+$685` on a run whose PnL is `+$109`,
because the strategy ended short. Cash flow is not PnL and the summary labels it
so, which the previous version did not.

The `Funding` line is `$0.00` and the two structural counters are zero because
a cash equity has no funding leg and 2,000 steps is 33 minutes, well inside one
6.5-hour session. Both fire on the runs in the crypto section below. The maker
rebate is `+$14.32` because the equity calibration is on a maker-taker venue
paying $0.0020 a share. The benchmark grid below runs gross of fees and is
unaffected.

## Benchmark results

The equity grid. `python benchmarks.py`, 500 paths per configuration, 2000
steps per path, seeds 12345 to 12844. Every configuration runs the same seed
sequence, so differences are **paired** and cancel the shared price risk.
Pasted verbatim:

```
                   scenario cross |q|  net PnL     SE  5th pct P(loss)   fills |final inv|  max DD adv sel adv SE P(halt)
           skew 0.00 (none)       inf   307.02   1.26   261.16    0.0%     986        49.2    9.93   -0.15   0.21    0.0%
       thick market (A=1.6)         5   225.23   0.45   208.69    0.0%    1933         4.3    1.84   -0.05   0.15    0.0%
         spread 0.10 (wide)        10   218.89   0.48   201.75    0.0%     575         3.5    0.89   -0.03   0.12    0.0%
                vol 60% ann         5   110.54   0.40    95.95    0.0%    1268         4.1    2.17    0.26   0.36    0.0%
                   baseline         5   110.47   0.32    99.11    0.0%    1268         4.1    1.72    0.11   0.15    0.0%
                vol 10% ann         5   110.44   0.30    99.79    0.0%    1268         4.1    1.65    0.04   0.06    0.0%
risk overlay (kill @ $1.40)         5    83.56   1.89     7.53    1.6%     887         3.9    1.91    0.03   0.12   52.8%
               informed 30%         5    77.19   0.28    67.17    0.0%    1082         4.0    1.91  -10.03   0.15    0.0%
        thin market (A=0.4)         5    52.04   0.20    44.90    0.0%     746         3.7    1.69   -0.00   0.13    0.0%
               informed 60%         5    44.02   0.23    35.50    0.0%     897         4.3    2.19  -19.99   0.15    0.0%
        spread 0.02 (tight)         2  -143.54   0.38  -157.34  100.0%    1840         4.3  143.71   -0.01   0.15    0.0%
     skew 0.05 (aggressive)         1 -1610.87   2.03 -1682.80  100.0%    1709         3.4 1610.99   -0.13   0.13    0.0%
```

### A sanity check, not a finding

**Mean PnL is independent of volatility by construction here, and the Monte
Carlo confirms the implementation respects that.** It is tempting to present
the flat volatility rows as a result. They are not one. In this model `sigma`
enters *only* through the price path. Fill probability is a function of quote
distance alone, and quote distance depends on `quote_spread` and `gamma * q`,
never on `sigma`. So under common random numbers the entire fill sequence is
identical across volatility settings, and only inventory PnL can differ. Over
40 seeds, spread capture across the 10%, 25% and 60% settings agrees to
`1.7e-13` and the fill counts differ on **zero** paths.

What the Monte Carlo does establish is that the implementation has no
accidental volatility-to-flow channel, and that dispersion scales as it should.
Per-path standard deviation of inventory PnL is `1.01 / 2.52 / 6.05` at 10% /
25% / 60% annualised, linear in `sigma` to two figures. The paired differences
in mean are `+0.07 (SE 0.15, t = 0.4)` and `-0.03 (SE 0.07, t = -0.4)`, both
indistinguishable from zero, as they must be.

In a real market volatility very much does move mean maker PnL, through wider
spreads, thinner books and more informed flow. This model has none of those
channels, which is a limitation rather than a discovery.

### Two things that are findings

**Informed flow costs more than adverse selection measures, and most of the gap
is lost volume.** Raising `informed_fraction` from 0 to 0.3 costs `-33.61` of
mean net PnL, but only `-9.93` of that is inventory PnL, which is what the
markout captures at `-10.03 (SE 0.15)`. The other `-23.71`, about 71%, is lost
*spread capture*. Informed arrivals are one-sided by construction, so raising
the fraction removes one quote's chance to fill and drops mean fills from 1269
to 1082. At 0.6 the split is the same shape, `-66.95` net, `-19.91` inventory,
`-47.04` spread capture, 70%. The markout itself is well behaved, `0.11
(SE 0.15)` at zero informed flow and almost exactly linear thereafter, but
reading the net PnL column as "the cost of adverse selection" would be wrong by
a factor of three. The `informed_fraction` axis mixes two effects and the model
cannot separate them without a two-sided informed arrival process.

**The top row is not the recommendation.** Turning inventory skew off earns the
most on the mean and carries 49.2 units of mean absolute final inventory against
4.1 at baseline, with a max drawdown of 9.93 against 1.72. The table prints the
inventory and drawdown columns next to the mean precisely so the mean cannot be
read on its own. The two negative rows are the reservation-price rule behaving
as specified rather than a pricing bug. The ask reaches the mid at
`|q| = quote_spread / gamma`, printed as the `cross |q|` column, so at
`gamma = 0.05` that threshold is one unit against a ten-unit clip and the
strategy quotes through the mid almost permanently.

## Crypto: the same engine on a second dataset

The same model quotes a crypto perpetual swap. Nothing in the engine is
special-cased for it. A perpetual is a calibration in which the calendar has no
gaps, a funding leg settles on a clock, fees are quoted on notional, and there
is no close.

### What is held fixed, and why

The two calibrations are **identical in every dimensionless quantity**. Both
quote 5bp from the reservation price into a market 20bp wide. Both use an
arrival intensity that falls by 1/e over 5bp of distance. Both lean the
reservation price by 10bp of mid per clip, so both put their own quote through
the mid at half a clip. A clip is $1,000 of notional in both.

|                        | `us-equity`                | `crypto-perp`                  |
| ---------------------- | -------------------------- | ------------------------------ |
| instrument             | $100 cash equity           | $100,000 BTC perpetual         |
| clip                   | 10 shares = $1,000         | 0.01 contracts = $1,000        |
| our quote              | 5bp ($0.05)                | 5bp ($50)                      |
| market width           | 20bp ($0.20)               | 20bp ($200)                    |
| crosses the mid at     | 0.5 clips                  | 0.5 clips                      |
| **calendar**           | 252 x 6.5h = 5,896,800s    | 365 x 24h = 31,536,000s        |
| **volatility**         | 25% annualised             | 55% annualised                 |
| **maker fee**          | +$0.0020/share (+0.20bp)   | -2.00bp of notional            |
| **funding**            | none                       | 1.25e-05 hourly (3.0bp/day)    |
| **session**            | flatten every 6.5h         | never closes                   |
| default horizon        | 2,000 steps (33 min)       | 86,400 steps (24h)             |

Only the bold rows differ, plus the price scale and the horizon. That is the
experimental design rather than laziness. If the crypto calibration also quoted
a different width into a different book, every difference between the two would
be a mixture of market structure and an invented spread, and neither could be
read off the other.

The design is tight enough to be checked. Under a shared seed the two
calibrations fill on **exactly the same steps**, because identical geometry
means identical fill probabilities. That is asserted in the test suite. The
flow is not merely comparable across the two datasets, it is the same flow.

The cost of the design is that `crypto-perp` is not a real perp. A BTC
perpetual is quoted far tighter than 20bp. The touch is often under a basis
point. Matching that would need a different arrival intensity, and the
comparison would then be measuring a guess at that intensity rather than the
four axes below. Absolute PnL levels are meaningless here exactly as they are
in the equity grid.

### 24/7

Almost all of what "24/7" means quantitatively is one ratio. A crypto year
holds 31,536,000 trading seconds against the equity calendar's 5,896,800, so it
is 5.35 times longer and the same headline volatility spreads over
`sqrt(5.35) = 2.31` times more standard deviations.

The consequence is worth stating because it inverts the intuition:

```
us-equity     25% annualised, session calendar  ->  1.030e-04 per second
crypto-perp   55% annualised, 24/7 calendar     ->  9.794e-05 per second
```

The perpetual carries a headline volatility **2.2 times higher and moves 5%
less per second.** 25% on the equity calendar is 57.8% on the crypto one. Read
a 24/7 asset's annualised figure through the equity calendar and you overstate
its per-second sigma by 2.31x, which is why `per_step_volatility` now takes the
calendar explicitly and why both datasets carry theirs.

The second consequence is horizon. A perpetual quotes 604,800 seconds in a
calendar week against a cash equity's 117,000, a factor of **5.17**. The
mechanics that define it are also slow. Funding settles hourly and a session
would close every 6.5 hours, so nothing structural happens inside the 33-minute
window the equity examples run in. `crypto-perp` therefore defaults to a full
24-hour day, which is the shortest horizon on which every mechanic fires more
than once.

### Funding rates

A perpetual has no expiry, so nothing mechanically drags it to spot. Venues
supply the drag with a periodic transfer between longs and shorts:

```
payment = -inventory * mark_price * rate_per_interval
```

Positive rate, longs pay. The default is `1.25e-05` settled hourly, which is
the ubiquitous 0.01% per eight hours in the cadence Hyperliquid and dYdX use.
Binance, Bybit and OKX settle the same rate eight-hourly.

**Measured over 100 paths of 24 hours, funding is a rounding error, and that is
not a bug.** Switching the leg off entirely moves mean net PnL by `-0.00 (SE
0.00)`. Turning the rate up **a hundredfold**, to 3% a day, which is a squeeze,
moves it by `+0.10 (SE 0.31, t = 0.3)`, still indistinguishable from zero.

The reason is structural and is the useful thing to take away. **Funding is a
carry term, not a transaction term.** It scales with `|inventory| x time`,
while a fee scales with volume. This book carries a mean absolute inventory of
0.41 clips, which is $410 of notional, and 3bp a day on $410 is 12 cents
against a gross PnL of $3,732. It is also as often short as long, so even that
nets out. The sign of the measured mean is positive, because the quoter tends
to be long after the price has fallen and short after it has risen, but at
`t = 0.3` that is a curiosity rather than a result.

Funding becomes first-order for a book that is *persistently* one-signed, which
a two-sided quoter with an inventory lean is not. If you want it to bite, model
a desk that can only go long.

The eight-hourly row is a control rather than a scenario. The same rate per
unit of time in coarser settlements should move the mean by nothing, and it
moves it by `-0.00 (SE 0.01)`. If the rate and the interval did not compose,
the two venue conventions would not be interchangeable in this model and every
funding number above would depend on an arbitrarily chosen cadence.

Honest limits: the rate is **constant**. Real funding is a market price, it
tracks the perp's premium over the index, mean reverts, and correlates with the
move that put you in the position. A stochastic rate is deliberately absent
because it would consume draws from the shared generator and destroy the common
random numbers the paired differences rest on, and because there is no honest
way to calibrate the correlation here. The 100x row is the substitute.

### Maker rebates

This is the axis that matters, and it is a change of *denominator*.

A US equity venue pays a maker rebate **per share**, around $0.0020 to $0.0030,
independent of the price. A crypto venue charges **basis points of notional**,
around 2bp for an ordinary account, reaching zero or a small rebate only at the
top volume tiers. The same fill on a $100 share and a $100,000 contract costs
600 times as much under the second convention and the same under the first.

Over 40 seeds, with the two calibrations earning the same edge by construction:

```
us-equity   (2 sessions)   quoted edge +1.444bp of notional (SE 0.003)
                           maker fee   +0.200bp
                           net         +1.645bp

crypto-perp (24 hours)     quoted edge +1.442bp of notional (SE 0.003)
                           maker fee   -2.000bp
                           net         -0.558bp
```

The two books earn a gross edge that is **the same to within 0.002bp against a
standard error of 0.003**, as the shared geometry requires. The fee convention
alone moves the business from `+1.645bp` to `-0.558bp`. It flips the sign.

The paired Monte Carlo says the same thing at scale. Removing the 2bp fee is
worth `+5,717.40 (SE 9.15)` of mean net PnL, on a book whose entire gross PnL
is `3,731.52`. It is the largest effect in either grid, larger than turning the
inventory skew off and larger than any volatility or spread setting in the
equity table. **A 2bp maker fee is bigger than everything this strategy earns**,
which is the short version of why crypto market making is a fee-tier business
before it is a modelling one.

The caveat is the one the whole repository carries. The ratio of fee to gross
edge is set by the quoted width and the arrival intensity, and neither is
calibrated to any venue. What the model shows is not that 2bp is fatal at some
particular real spread. It is that a notional fee scales with **volume** while
spread capture scales with **edge per fill**, so the two are not comparable
across the equity and crypto conventions without first stating the notional.

### No overnight hedge

A cash equity desk does not carry an unhedged position through the overnight
gap, so at the close it crosses the market and goes flat. A perpetual has no
close and no gap. It carries the position and pays funding on it. In the model
that is `session_steps`. At the end of every such block the book is flattened
at the market's reference half-spread. It is implemented as an ordinary trade
at a bad price rather than as an accounting rule, so the PnL identity absorbs
it with no special case, and the cost shows up as negative spread capture.

The cost is easy to measure and small. Giving the perpetual an equity-style
6.5-hour close costs `$1.15` of close-out over a day, and the paired difference
in net PnL is `-1.22 (SE 1.60, t = -0.8)`. With the inventory lean switched off,
so the position random-walks instead of mean-reverting, the same three closes
cost `$20.91`.

The benefit is where the intuitive story breaks. A close is supposed to be a
free inventory control that a 24/7 book does not get. Measured on 100 paths
against an otherwise identical skewless book:

```
skew off, no close      net 2872.55   mean |inv| 6.818 clips   peak |inv| 9.199
skew off, close 6.5h    net 2889.17   mean |inv| 6.757 clips   peak |inv| 9.199
skew off, close hourly  net 3036.06   mean |inv| 6.361 clips   peak |inv| 9.199

paired vs no close, 6.5h:  d net +16.62 (SE 9.22, t = +1.8)
                           d mean |inv| -0.062 clips (SE 0.007, t = -8.2)
paired vs no close, 1h:    d net +163.51 (SE 28.88, t = +5.7)
                           d mean |inv| -0.457 clips (SE 0.021, t = -21.9)
```

At a realistic session cadence the reset is statistically real and
economically negligible, with 0.9% off the inventory carried, and a net PnL
difference that does not clear two standard errors. It takes 24 resets a day to
move net PnL significantly, and even then the inventory falls only 6.7%.

The reason is in the third column. **Peak inventory is 9.199 clips on all
three, because the position limit is what actually bounds the risk, not the
close.** The book saturates against the limit, gets flattened, and re-saturates.
Reaching for the close as the thing that controls inventory is reaching past
the control that is already binding.

**And the model cannot price the reason the flatten exists.** An equity desk
flattens into the close to avoid a gap it cannot quote through. The price
process here is geometric Brownian motion, which has no jumps, so the overnight
gap does not exist and the benefit of avoiding it cannot be measured. The
numbers above are the *cost* of flattening, measured honestly, against a
*benefit* this model is structurally unable to see. Read them as an upper bound
on the case against flattening, not as a case for carrying positions overnight.

### The perpetual grid

`python benchmarks.py --dataset crypto-perp`, 100 paths per configuration,
86,400 steps per path, seeds 12345 to 12444. Pasted verbatim:

```
                scenario   net PnL     SE P(loss)    gross      fees  funding close-out  fills mean |inv| peak |inv|   max DD
     maker rebate +0.5bp   5160.42   4.13    0.0%  3731.52   1429.35    0.001      0.00  46637       0.41       2.34     2.47
           maker fee 0bp   3731.07   3.64    0.0%  3731.52      0.00    0.001      0.00  46637       0.41       2.34     2.60
skew 0.00 + 6.5h flatten   2889.17  24.50    0.0%  4948.12  -2052.06   -0.017     20.91  36480       6.76       9.20    63.83
        skew 0.00 (none)   2872.55  24.36    0.0%  4904.76  -2025.33    0.012      0.00  36457       6.82       9.20    64.00
             informed 0%  -1524.41  11.32  100.0%  5167.00  -6691.02    0.001      0.00  54559       0.40       2.23  1525.25
   funding 100x (stress)  -1986.24  10.14  100.0%  3731.52  -5717.40    0.096      0.00  46637       0.41       2.34  1986.80
  crypto baseline (-2bp)  -1986.33  10.14  100.0%  3731.52  -5717.40    0.001      0.00  46637       0.41       2.34  1986.74
             funding off  -1986.33  10.14  100.0%  3731.52  -5717.40    0.000      0.00  46637       0.41       2.34  1986.74
        funding 8-hourly  -1986.34  10.13  100.0%  3731.52  -5717.40   -0.004      0.00  46637       0.41       2.34  1986.75
      flatten every 6.5h  -1987.55  10.01  100.0%  3728.92  -5716.10    0.001      1.15  46630       0.41       2.34  1987.98
```

```
                scenario d net PnL     SE        t beats baseline
     maker rebate +0.5bp   7146.75  11.44    624.8         100.0%
           maker fee 0bp   5717.40   9.15    624.8         100.0%
skew 0.00 + 6.5h flatten   4875.50  23.93    203.7         100.0%
        skew 0.00 (none)   4858.88  23.86    203.7         100.0%
             informed 0%    461.93   3.78    122.2         100.0%
   funding 100x (stress)      0.10   0.31      0.3          51.0%
             funding off     -0.00   0.00     -0.3          49.0%
        funding 8-hourly     -0.00   0.01     -0.5          45.0%
      flatten every 6.5h     -1.22   1.60     -0.8          50.0%
```

Two columns need reading with care. `max DD` on the fee-paying rows is
`1,986.74`, which is not price risk. A book whose fees exceed its edge bleeds
monotonically, so its drawdown is just its loss. And `informed 0%` earns *less*
than the baseline in net terms despite better markouts, because removing
informed flow raises the fill count from 46,637 to 54,559 and every extra fill
pays 2bp. That is the fee denominator again, and it is the same effect the
equity grid documents in reverse.

One consequence worth flagging: `python example.py --dataset crypto-perp` halts
after 8,678 of its 86,400 steps, because the $200 kill-switch inherited from
the equity example is reached by the fee bleed. That is the stop working
correctly, not a mis-set limit. *Any* fixed drawdown limit stops a monotonically
bleeding book eventually. Only the timing is in question. Pass
`--drawdown-limit 0` to see the whole day.

### What did not change

The equity results in this README were produced before any of the above existed
and are reproduced by the current code **exactly**, every figure in the
benchmark table to the last printed digit. The two market-structure hooks are
off by default and neither draws from the random generator, so a run without
them consumes the same stream in the same order it always did.

`example.py` now builds from `datasets.US_EQUITY` rather than from hardcoded
constants, and the test suite pins every derived absolute value against the
numbers it used to hardcode, so the published run cannot drift away from the
calibration that produced it. The two figures in the example output that did
move, the maker rebate and the drawdown, moved because the equity calibration
now carries a real venue rebate and rebates have always been inside the
mark-to-market series the drawdown is measured on.

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```

257 tests, about 16 seconds. They assert economic invariants rather than
arithmetic, so that each of the four defects above fails a test if it is
reintroduced:

- `ask > bid` at every inventory level from `-2x` to `+2x` the position limit,
  across six skew factors and three spreads
- `gross_pnl` matches `MarketMaker.get_gross_pnl` to 1e-9 across six
  configurations and four seeds, and the summary waterfall reconciles to 1e-12.
  The two sides are computed independently, one from the tracker's fill and step
  records and one from cash plus marked inventory, so the identity is a real
  constraint rather than a restatement.
- fill count is strictly decreasing in `quote_spread` under common random
  numbers, on all 30 seeds of a 7-point spread grid, 180 pairwise comparisons
  with no tie. The smallest margin is 40 fills, and strictness was separately
  confirmed at 400, 600, 800, 1000 and 1500 steps, so the shipped 600 is not a
  tuned-to-pass choice.
- the mid is a martingale. Over 2000 paths of 200 steps the mean terminal price
  is asserted within 3 standard errors of the initial price, and measures 0.5.
  A companion test re-runs the identical estimator with the Itô correction
  cancelled and asserts it *rejects*, so the martingale test cannot silently
  degrade into one that passes on anything. That case measures 6.6 standard
  errors high against the same 3 SE threshold.
- adverse selection is asserted within 3 standard errors of zero under
  uninformed flow, and measures -0.08. At `informed_fraction = 0.3` the same
  60-seed fixture gives `-10.03 (SE 0.51)`, so 19.8 standard errors below zero.
  Significance is asserted at `informed_fraction = 0.6`.
- the kill-switch does **not** fire on a $1,000 cash outflow at zero loss, does
  fire on a real drawdown, and stays halted afterwards
- `get_quotes` is idempotent, so quoting twice at the same state cannot
  double-count into the drawdown
- the drawdown the summary reports is the same number the kill-switch tested
- reproducibility survives interleaving two simulators on the same seed, which
  the old process-global RNG would fail

The crypto layer adds 94 tests on the same principle, that a property is worth
asserting where getting it wrong would still look right:

- the equity dataset's derived absolute values are pinned against the constants
  `example.py` used to hardcode, so a dimensionless field cannot drift and
  silently make the published output unreproducible
- the two datasets are compared **field by field**, and any field that differs
  and is not on an explicit allow-list fails the suite. Adding a dimensionless
  field without deciding which side of the design it falls on is a test failure
  rather than a quiet weakening of the comparison
- both datasets fill on exactly the same steps under a shared seed
- funding's sign is asserted both ways. A long book pays a positive rate and a
  short book receives it. Getting this backwards turns a cost into a subsidy
  and the waterfall still reconciles, because both sides move together
- the first funding payment lands at the end of the first *interval*, not at
  the step whose index is a multiple of it
- halving the rate and halving the interval leaves the total unchanged, which
  is what makes the two venue conventions interchangeable and is why the
  eight-hourly benchmark row is a control rather than a scenario
- neither funding nor the session close consumes randomness, checked by running
  with and without each on one seed and comparing the price path and the fill
  sequence step by step. If either drew from the generator, every paired
  difference in the crypto grid would be comparing two different markets
- a close-out earns no maker rebate, because it removes liquidity
- close-outs stay out of the fill counters and out of the markout, and stay
  *in* spread capture, with `quoted_edge - close_out_cost == spread_capture`
  asserted exactly
- the close reduces mean carried inventory and **does not** reduce the peak,
  asserted in both directions, because the opposite is the intuitive reading
- a per-notional fee is asserted to be price-dependent and a per-share rebate
  price-independent, which is the whole difference between the conventions
- the waterfall chart's residual covers funding whether or not a funding bar is
  drawn, so a chart cannot omit a real cash flow and still advertise itself as
  reconciling

## Limitations

Stated plainly, because the model is only useful if you know where it stops.

- **No order book.** No queue position, no price-time priority, no partial
  queue depletion. Fill probability is a function of quote distance and nothing
  else. Real maker PnL is dominated by queue position and this model cannot see it.
- **No market impact from our own trades.** The mid is exogenous.
- **No latency.** Quotes update instantly and are never stale, which is where a
  real market maker loses most of its money.
- **Uninformed flow is a coin flip, informed flow is a perfect oracle.** Real
  informed flow is neither, and the `informed_fraction` axis should be read as a
  comparative statics exercise, not a calibration.
- **No real data, and the fill model is not calibrated.** Every number here
  comes from a simulated path, and `A = 0.8`, `k = 20.0` were chosen so that a
  five-cent quote on a $100 asset fills at a plausible rate, not fitted to any
  venue. Absolute PnL levels are therefore meaningless. Only the paired
  comparisons between configurations, which hold the fill model fixed, carry
  information.
- **Single asset, no hedging, no borrow cost.** There is a funding leg, but no
  basis trade and no spot book to hedge a perpetual against, which is how a
  real crypto desk carries inventory.
- **Informed arrivals are one-sided.** Raising `informed_fraction` therefore
  lowers the total fill count as well as worsening the markout, so that axis
  mixes adverse selection with reduced volume and the model cannot separate
  them. See the benchmark commentary for the measured split.
- **Fees are a flat rate in either denomination**, with no tiered schedules and
  no taker fees. The equity baseline essentially never crosses the market's
  quote, 10 fills in 2000 steps, but the aggressive-skew and tight-spread rows
  cross constantly, 928 and 65 fills respectively at seed 12345. Their losses
  are therefore understated by the taker fees this model omits. The `-$1,610`
  is caused by quoting through the mid, not by the missing fees, so the
  omission makes the argument weaker rather than stronger. The same omission
  applies to every session close-out, which crosses the market by construction.
- **The funding rate is a constant.** Real funding tracks the perp's premium
  over the index, mean reverts, and blows out in a squeeze, and it correlates
  with the move that put you in the position. Modelling that would consume
  draws from the shared generator and break the common random numbers the
  paired differences rest on, and there is no honest way to calibrate the
  correlation here. The 100x stress row is the substitute.
- **No overnight gap.** Geometric Brownian motion has no jumps, so an equity
  session boundary is a point at which nothing happens. The model can price the
  *cost* of flattening into a close and is structurally unable to see the
  *benefit*, which is avoiding a gap it does not simulate. Do not read the
  crypto section's close-out numbers as an argument for carrying overnight.
- **The crypto calibration is not a real perpetual.** Its 20bp market width is
  chosen to match the equity calibration so that the two differ only in market
  structure. Real BTC perps quote inside a basis point. The fee-to-edge ratios
  in the crypto section are therefore statements about denominators, not
  calibrated claims about any venue.
- **A perpetual's price process here is the same GBM as the equity's**, with no
  index, no basis, and no mechanism connecting funding to price. Funding is
  charged on the position. Nothing charges the position for the funding.

## What changed in v0.2

The v0.1 version of this repository had four defects that invalidated its
results. They are listed in full because the git history shows them anyway, and
because a benchmark table is only worth reading if you know what was wrong with
the previous one.

1. **Fills were independent of the quoted price.** A coin flip decided whether a
   trade happened. The quote never entered the decision. PnL was therefore
   linear in `quote_spread` and every scenario reported exactly 51 trades
   whether the maker quoted one cent or fifty dollars wide. Every conclusion in
   the old benchmark table was an artefact of this. Fixed by the intensity model.
2. **The inventory skew had the wrong sign on the ask.** It widened both quotes
   when long and inverted them when short, quoting a bid above its own ask on 52
   of 100 steps of the default run. Fixed by the reservation price.
3. **The PnL decomposition did not reconcile.** Its three components summed to
   `-703.47` against a true mark-to-market PnL of `-202.30`, and the benchmark
   reported the former as "Total PnL". Fixed by making the decomposition an
   identity and by making adverse selection a diagnostic rather than a bucket.
4. **The README's example output was not output.** It did not add up
   (`50.00 - 10.50 - 5.80 = 33.70`, printed as `45.30`), and it omitted lines the
   code emits unconditionally, so it cannot have come from any run. Everything
   in this README is now pasted from a real run at a stated seed.

Also fixed: the missing Itô correction (which gave the asset a built-in
`+13.3%` drift at the high-volatility setting), single-path benchmark rankings
with no error bars (on a quantity whose per-path standard deviation was eight
times the mean), and a kill-switch that fired on cash flow rather than on
losses, so buying 10 units of a $100 asset tripped it with zero actual loss.

## References

- Avellaneda, M. and Stoikov, S. (2008). *High-frequency trading in a limit
  order book.* Quantitative Finance 8(3), 217-224. Source of the exponential
  fill intensity and the reservation price. This code implements the intensity
  and the inventory lean, and does **not** implement the closed-form optimal
  spread or the terminal-time utility optimisation.
- Ho, T. and Stoll, H. (1981). *Optimal dealer pricing under transactions and
  return uncertainty.* Journal of Financial Economics 9(1), 47-73. Source of the
  inventory-holding view of dealer quoting.
- Glosten, L. and Milgrom, P. (1985). *Bid, ask and transaction prices in a
  specialist market with heterogeneously informed traders.* Journal of Financial
  Economics 14(1), 71-100. Motivation for the informed-flow mechanism, though
  the implementation here is a simplification rather than the model.

The crypto calibration takes its conventions from venue documentation rather
than from a paper. The 0.01% per eight hours funding rate and its hourly
variant, the 2bp non-VIP maker fee, and the rebate at the top volume tiers are
the published schedules of the major perpetual venues. The $0.0020 per share
maker rebate is the ordinary US maker-taker equity tier. None of them is
calibrated to observed data here. They are the modal published numbers, chosen
so that the fee axis spans the range a real desk actually faces.

## License

See [LICENSE](LICENSE).
