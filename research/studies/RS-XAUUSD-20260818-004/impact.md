# RS-XAUUSD-20260818-004 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What was asked and what the answer is

"Setting S1 and S2 aside, can Python find better alpha in the 30-minute data?"

No directional feature survived validation. 26 features across 6 horizons were searched on the
training period only; five hand-entered candidates were validated, with two sign flips and
block-bootstrap p-values from 0.14 to 0.83 for the rest. The largest information coefficient in
the scan is 0.04. The other 151 cells were never validated, and the validation period holds only
about 160 independent windows, so small effects cannot be ruled out. The short-horizon patterns
are unvalidated and their spreads are below the lowest cost level (0.010%).

This study was not pre-registered; the volatility analysis was added mid-study (post-hoc).

## The thing that is forecastable

Volatility ranks far higher than any directional feature (range vs signed return — not the same yardstick), and it holds out of sample (TRAIN is in-sample):

| | TRAIN | VALID | HOLDOUT |
|---|---|---|---|
| ATR(14) alone | 0.439 | 0.566 | 0.409 |
| ATR(14) × hour factor | **0.684** | **0.684** | **0.571** |
| Best directional feature | 0.040 | fails | — |

The hour-of-day factor (median true range ÷ prior 14-bar ATR, by Taipei open hour) spans 3.28×
trough to peak — 05:00 is the trough at 0.461, 21:00 the peak at 1.513 — and its shape rank-correlates 0.924 between
TRAIN and VALID and 0.863 between TRAIN and HOLDOUT.

## What the volatility applications showed

| Application | S1 | S2 |
|---|---|---|
| Volatility-targeted sizing, best of 6 exponents (in-sample) | 100.3% (about 97.0–103.7%) | 101.4% (about 96.4–107.6%) |
| Volatility-scaled stop | **simulation invalid, do not cite** | **simulation invalid, do not cite** |

Sizing is compared on the per-trade mean/sd ratio relative to baseline; intervals are a post-hoc
bootstrap and cover 100%, so we saw no difference. The stop simulation keeps the original exit
when the new stop is wider (325/472 S1 trades at k=3.0), so the earlier 91% / 93% say nothing
about volatility-scaled stops; a full-path re-simulation is needed.

Post-hoc observation: R dispersion did not separate across forecast-volatility terciles (S1 sd
1.48 / 1.29 / 1.60, S2 1.66 / 1.78 / 1.65, intervals overlap). Candidate reasons, none tested:
fixed-percentage stops and targets put about 92% / 92% of trades on three fixed R values (94% / 93% within
0.05R), so sd tracks hit rates; or the fixed stop already acts as a normalised unit (which sits uneasily with the
15.2× spread in volatility units). Any link to `RS-XAUUSD-20260818-002` is conjecture.

## What to do with this

**Do not** treat the sizing result as evidence for a volatility rule: we saw no difference, which
is not proof of none. **Do not** cite the stop figures; the stop question is open.

**Do not** assume the next bar-level directional feature will work: five candidates were
validated and none survived, with about 160 independent windows of power. That bounds what
this protocol could detect, not what the data contain.

## Where alpha could still be

Recorded as directions, not claims — none tested here:

- **Faster timescales.** Microstructure at 1–5 minutes is the one place not yet reachable;
  the 5-minute export only starts 2026-05-04.
- **Execution rather than prediction.** Given a stable intraday volatility profile, the
  open question is not what to trade but when within the bar to fill. That needs tick data.
- **Longer horizons.** Everything here tops out at a 24-hour forward return. Weekly and
  monthly structure is a different question on a different sample size.

The honest summary is that this protocol saw no direction at 30 minutes in 32 months of one
instrument, with limited power; whether to spend effort on another feature or on the open
prospective test (`PREREG-20260818-001`) is a separate decision.
