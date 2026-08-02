# Market-Making Simulator

A market-making simulator where quoting further from the mid actually reduces
your fill rate, inventory skew leans the right way, and PnL decomposes into
spread capture and inventory risk as an exact identity rather than as three
numbers that are added up and hoped to agree.

Those three properties sound like table stakes. Getting any of them wrong makes
a simulator produce conclusions that are artefacts of its own bugs, which is
what happened to this one before v0.2 and is documented at the bottom.

## What this is, and what it is not

**It is** a continuous-time market-making model in the Avellaneda-Stoikov
family: a mid-price diffusion, a fill intensity that decays with quote
distance, an inventory-linked reservation price, and a Monte Carlo harness that
compares configurations with paired standard errors.

**It is not a limit order book.** There is no queue, no price-time priority, no
per-level depth and no impact from our own trades. Nothing here reconstructs a
real feed. If you need queue position, this is the wrong model, and the
[Limitations](#limitations) section says so in more detail.

## The model

### Price

Geometric Brownian motion with the Itô correction, so the mid is a martingale
and volatility scenarios are comparable:

```
m_{t+1} = m_t * exp( -0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z ),   Z ~ N(0,1)
```

`sigma` is a **per-step** log-return volatility. Use
`units.per_step_volatility(annual_vol, seconds_per_step)` to get it from an
annualised figure; 25% annualised on one-second steps is `1.030e-04` per step.
Quoting volatility without a time unit is how the previous version ended up
labelling `0.02` as "2% volatility" when it meant 2% *per step*, a $2 move on a
$100 asset against a five-cent quote.

### Fills

Arrival intensity decays exponentially in the distance of our quote from the mid:

```
lambda(delta) = A * exp(-k * delta)
P(fill within dt) = 1 - exp(-lambda(delta) * dt)
```

Order sizes are exponential with mean `mean_order_size`, and the fill is capped
at our own quoted size, so large arrivals fill us partially. Defaults are
`A = 0.8`, `k = 20.0`: a quote at the mid fills with probability 55.1% per unit
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
quotes can never cross. When we are long, **both** quotes move down: the ask
gets easier to lift and the bid harder to hit, so the position bleeds off.

The intuitive-sounding alternative, "long inventory, widen the ask", does the
opposite of what is wanted, because a wider ask is *less* likely to be lifted.

Note the failure mode this rule has, which the benchmark measures rather than
hides: the ask reaches the mid at `|q| = quote_spread / gamma`. Past that
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

The reported waterfall then adds the two real costs:

```
net_pnl = gross_pnl + rebates - liquidation_cost
```

where `liquidation_cost` charges the market's half-spread on residual
inventory, so a strategy cannot look profitable by carrying an open position
marked at mid.

**Adverse selection is not a third additive bucket.** It is a signed `h`-step
markout, `sum of q_signed * (m_{t+h} - m_t)`, which is a *split* of the
inventory term and is reported separately. It is negative when we are
systematically on the wrong side of the next move, positive when we are on the
right side, and zero in expectation under uninformed flow.

## Install

```bash
git clone https://github.com/GasparCoquet/Market-Making-Simulator.git
cd Market-Making-Simulator
pip install -e .
```

Requires Python 3.9 or newer. `pip install -e ".[dev]"` adds pytest.

## Run

```bash
python example.py        # one seeded path, prints the waterfall, writes 4 plots
python benchmarks.py     # 12 configurations x 500 Monte Carlo paths
```

`example.py` is headless by default and writes PNGs to `plots/`. Pass `--show`
to open them. `benchmarks.py` takes `--paths`, `--steps`, `--seed` and `--workers`.

### Example output

Stdout of `python example.py` at seed 42. Cut for length: the banner, the "Run"
and "Market" sections, one `risk manager halted = False` line, the quote ladder
and the plot paths. Nothing else is altered, no number is touched, and running
the command reproduces every figure below exactly.

```
Configuration:
  Market:               MarketState(mid=100.00, ref_half_spread=0.1000)
  Fill model:           FillModel(A=0.8, k=20.0, mean_size=10.0, informed=0.3)
  Market maker:         quote_spread=0.05, quote_size=10, gamma=0.01, max_inventory=100
  Risk manager:         RiskManager(kill_switch=True, drawdown_limit=200.0, halted=False)
  Markout horizon:      5 steps (adverse selection is measured over this)
  Volatility:           25% annualised on 1s steps = 1.030e-04 per step
  Steps:                2000
  Seed:                 42

Activity:
  Fills:                1134 (564 buys, 570 sells)
  Filled volume:        7158.5 units
  Fill rate:            0.57 fills/step
  Final inventory:      -5.92 units

PnL waterfall:
  Spread capture:       $    104.15
  Inventory PnL:        $     -9.08
  ----------------------------------------
  Gross PnL (at mid):   $     95.07
  Maker rebates:        $      0.00
  Liquidation cost:     $     -0.59
  ----------------------------------------
  Net PnL:              $     94.48

Diagnostics:
  Adverse selection:    $    -10.35  (signed markout, part of inventory PnL)
  Max drawdown:         $      2.02
  Net cash flow:        $    685.14  (not a PnL)
  Inventory value:      $   -590.07  (cash flow + this = gross PnL)

Reconciliation:
  spread_capture + inventory_pnl - gross_pnl        = 0.00e+00
  gross_pnl + rebates - liquidation_cost - net_pnl  = 0.00e+00
  min(ask - bid) over the run                       = $0.1000  (0 crossed steps)
```

Note the `net cash flow` line. It is `+$685` on a run whose PnL is `+$95`,
because the strategy ended short. Cash flow is not PnL and the summary labels it
so, which the previous version did not.

## Benchmark results

`python benchmarks.py`, 500 paths per configuration, 2000 steps per path, seeds
12345 to 12844. Every configuration runs the same seed sequence, so differences
are **paired** and cancel the shared price risk. Pasted verbatim:

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
enters *only* through the price path: fill probability is a function of quote
distance alone, and quote distance depends on `quote_spread` and `gamma * q`,
never on `sigma`. So under common random numbers the entire fill sequence is
identical across volatility settings, and only inventory PnL can differ. Over
40 seeds, spread capture across the 10%, 25% and 60% settings agrees to
`1.7e-13` and the fill counts differ on **zero** paths.

What the Monte Carlo does establish is that the implementation has no
accidental volatility-to-flow channel, and that dispersion scales as it should:
per-path standard deviation of inventory PnL is `1.01 / 2.52 / 6.05` at 10% /
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
*spread capture*: informed arrivals are one-sided by construction, so raising
the fraction removes one quote's chance to fill and drops mean fills from 1269
to 1082. At 0.6 the split is the same shape: `-66.95` net, `-19.91` inventory,
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
as specified rather than a pricing bug: the ask reaches the mid at
`|q| = quote_spread / gamma`, printed as the `cross |q|` column, so at
`gamma = 0.05` that threshold is one unit against a ten-unit clip and the
strategy quotes through the mid almost permanently.

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```

163 tests, about 16 seconds. They assert economic invariants rather than
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
- the mid is a martingale: over 2000 paths of 200 steps the mean terminal price
  is asserted within 3 standard errors of the initial price, and measures 0.5.
  A companion test re-runs the identical estimator with the Itô correction
  cancelled and asserts it *rejects*, so the martingale test cannot silently
  degrade into one that passes on anything. That case measures 6.6 standard
  errors high against the same 3 SE threshold.
- adverse selection is asserted within 3 standard errors of zero under
  uninformed flow, and measures -0.08. At `informed_fraction = 0.3` the same
  60-seed fixture gives `-10.03 (SE 0.51)`, so 19.8 standard errors below zero;
  significance is asserted at `informed_fraction = 0.6`.
- the kill-switch does **not** fire on a $1,000 cash outflow at zero loss, does
  fire on a real drawdown, and stays halted afterwards
- `get_quotes` is idempotent, so quoting twice at the same state cannot
  double-count into the drawdown
- the drawdown the summary reports is the same number the kill-switch tested
- reproducibility survives interleaving two simulators on the same seed, which
  the old process-global RNG would fail

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
- **Single asset, no hedging, no funding, no borrow cost.**
- **Informed arrivals are one-sided.** Raising `informed_fraction` therefore
  lowers the total fill count as well as worsening the markout, so that axis
  mixes adverse selection with reduced volume and the model cannot separate
  them. See the benchmark commentary for the measured split.
- **Fees are a flat per-unit rebate**, with no tiered schedules and no taker
  fees. The baseline essentially never crosses the market's quote, 10 fills in
  2000 steps, but the aggressive-skew and tight-spread rows cross constantly,
  928 and 65 fills respectively at seed 12345. Their losses are therefore
  understated by the taker fees this model omits. The `-$1,610` is caused by
  quoting through the mid, not by the missing fees, so the omission makes the
  argument weaker rather than stronger.

## What changed in v0.2

The v0.1 version of this repository had four defects that invalidated its
results. They are listed in full because the git history shows them anyway, and
because a benchmark table is only worth reading if you know what was wrong with
the previous one.

1. **Fills were independent of the quoted price.** A coin flip decided whether a
   trade happened; the quote never entered the decision. PnL was therefore
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

Also fixed: the missing Itô correction, which gave the asset a built-in `+13.3%`
drift at the high-volatility setting; single-path benchmark rankings with no
error bars, on a quantity whose per-path standard deviation was eight times the
mean; and a kill-switch that fired on cash flow rather than on losses, so
buying 10 units of a $100 asset tripped it with zero actual loss.

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

## License

See [LICENSE](LICENSE).
