# RS-XAUUSD-20260818-004 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What was asked and what the answer is

"Setting S1 and S2 aside, can Python find better alpha in the 30-minute data?"

No directional alpha. 26 features across 6 horizons, searched on the training period only;
the shortlist fixed there dies in validation, with two sign flips and block-bootstrap
p-values from 0.13 to 0.83. The largest information coefficient anywhere in the scan is
0.04, and the short-horizon patterns that do exist are worth less than the spread.

That is not a failure of effort. It is a measurement of what this data can support, and it
is worth having in writing, because the next idea that arrives will look better than it is.

## The thing that is forecastable

Volatility, by more than an order of magnitude, and it holds out of sample:

| | TRAIN | VALID | HOLDOUT |
|---|---|---|---|
| ATR(14) alone | 0.439 | 0.566 | 0.409 |
| ATR(14) × hour factor | **0.684** | **0.684** | **0.571** |
| Best directional feature | 0.040 | fails | — |

The hour-of-day factor spans 3.2× trough to peak — 04:00–05:00 Taipei runs at about 0.47×
a normal bar, 21:00–22:00 at about 1.51× — and its shape rank-correlates 0.924 between
TRAIN and VALID and 0.863 between TRAIN and HOLDOUT.

## Why that is worth nothing here, and it matters

The obvious use is stop placement or position sizing. Both were tested and both fail:

| Application | S1 | S2 |
|---|---|---|
| Volatility-scaled stop, best of 7 multiples | 91% of baseline | 93% |
| Volatility-targeted sizing, best of 6 exponents | 100% | 101% |

compared at equal volatility over the period, which is the alignment this project settled
on in `RS-XAUUSD-20260818-001` revision 3.

The reason is a single fact worth carrying: **R per trade has almost the same dispersion in
every volatility tercile** — S1 1.48 / 1.29 / 1.60, S2 1.66 / 1.78 / 1.65. A fixed-percentage
stop combined with fixed-fractional risk already expresses every trade in
volatility-normalised units. The forecast has nothing left to correct.

That one mechanism explains three results across two studies: this study's sizing failure,
this study's stop failure, and `RS-XAUUSD-20260818-002`'s finding that ATR at entry
separates nothing.

## What to do with this

**Do not** build a volatility filter, a volatility-scaled stop, or a volatility-targeted
sizing rule for S1 or S2. The risk framework already does that work.

**Do** note that the existing framework is better than it looks. A fixed percentage stop
looks crude next to an ATR stop; in R space it is doing the same job with fewer moving
parts and no estimation error.

**Do not** reach for the next bar-level directional feature on this dataset. Twenty-six
were tried under a protocol that could have detected an effect this data does not contain.
The bound is on the data, not on the feature list.

## Where alpha could still be

Recorded as directions, not claims — none tested here:

- **Faster timescales.** Microstructure at 1–5 minutes is the one place not yet reachable;
  the 5-minute export only starts 2026-05-04.
- **Execution rather than prediction.** Given a stable intraday volatility profile, the
  open question is not what to trade but when within the bar to fill. That needs tick data.
- **Longer horizons.** Everything here tops out at a 24-hour forward return. Weekly and
  monthly structure is a different question on a different sample size.

The honest summary is that direction at 30 minutes is efficient to the limit of what 32
months of one instrument can resolve, and the effort is better spent on the one open
prospective test (`PREREG-20260818-001`) than on another feature.
