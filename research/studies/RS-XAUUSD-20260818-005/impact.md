# RS-XAUUSD-20260818-005 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What it supports

Continuing to run both strategies at a comparable risk allocation — which is already what
happens with fixed-fractional sizing per trade. This study is a confirmation that the
current arrangement is sound, not a proposal to change it.

## What it does not support

- Dropping S2. Against S2 alone, blending lifts Sharpe from 2.33 to 3.88 and cuts the worst
  drawdown from 12.8R to 2.2R, with P(improvement) of 0.999 and 0.99 (drawdown here is month-end, on monthly-summed R; trade by trade it is 14.8R to 4.5R, post-hoc).
- Dropping S1. Against S1 alone the blend's Sharpe gain is +0.22 with a 90% interval of
  −0.75 to +0.99. That does not clear zero, so the blend is **not** shown to beat S1.
- Optimising the weight. Sharpe runs 3.71 to 3.96 across every S1 weight from 0.4 to 0.8.
  The in-sample optimum at 0.6 sits inside that plateau and carries no information.

## The number worth remembering

The two strategies' monthly returns correlate at **0.234** (95% interval about -0.12 to 0.54,
n=32, post-hoc). Two long-only strategies on the same instrument were expected to be
near-duplicates; the interval rules that out but cannot separate zero from a moderate
correlation. Everything else here rests on this one estimate.

## What the Sharpe figures are not

They are not a live expectation. Values near 3.9 do not survive real execution; these are
Strategy Tester R multiples aggregated monthly over a period in which XAUUSD rose 112%,
with no slippage beyond what the export already contains. Every claim here is a ranking
between allocations of the same backtest.

## Revision 2 — the search for an uncorrelated third strategy

The brief asked for an uncorrelated strategy type, explicitly allowing long/short. Two
candidates were built (not pre-registered; added after seeing revision 1's results) and neither
was profitable over the full sample (the shooting-star short's 22 holdout trades were slightly positive, post-hoc). The exercise produced a reusable formula.

**The data do not show the search is structurally blocked.** Regressing monthly R on gold gives
S1 a beta of 0.444 and S2 0.27; the point estimates are that 79.7% and 77.0% of their returns
are alpha rather than being long gold, but month-resampled 90% intervals are wide (about
48%-96% and 37%-100%, post-hoc). In the 8 months gold fell S1 was positive in all 8 (mean
4.06R) and S2 in 5 of 8 (mean 1.507R); n=8. (Caveat: backtest alpha, and the V3.4-to-V4.1
version history is itself a selection process.)

**Candidate 1, counter-trend fade.** Symmetric long and short, correlation 0.036 to the blend
(n=32, interval not computed). Sharpe -0.046 over 700 trades, not separable from zero. It
lowers the blend at every weight tested, using a raw-R weight that is not a risk allocation
(30% weight is about 45% of the risk; equal-risk 30% gives 3.505 vs 3.882, post-hoc). The
carried variant is the first survivor in list order; both survivors were negative in the
holdout (post-hoc). Low correlation without profit is not a diversifier.

**Candidate 2, a shooting-star short inspired by S2's hammer.** Thresholds were hand-set from
the hammer characteristics in the export's flags; the result is stricter than a mirror (the
reversed rule recovers 43 of 212 hammers). With one exit rule (hold 16 bars, 1% stop) it lost
in train (n=44, win 34.1%, mean -0.17%) and valid (n=30, win 36.7%, mean -0.28%); in the
holdout (post-hoc, n=22) it was slightly positive (win 54.5%, mean +0.115%), too few trades to
separate from zero. This says nothing about reflected patterns in general.

## The bar for any future strategy

At a 30% risk allocation, what a candidate needs to improve the blend:

| Correlation to blend | Minimum Sharpe |
|---|---|
| −0.50 | -1.19 — helps even while losing money |
| −0.25 | -0.14 — helps if it merely breaks even |
| 0.00 | **0.8** |
| +0.25 | 1.65 |
| +0.50 | 2.44 |
| +0.75 | 3.18 |

The table is a formula, not a measurement: thresholds scale in proportion to the blend's Sharpe
(about 0.21 times it at zero correlation), and this study uses the in-sample 3.88. At a blend
Sharpe of 1.5 the zero-correlation threshold would be 0.31 (post-hoc). In the formula
correlation weighs more than return, so a different instrument is one direction worth trying;
it was not tested here. Neither candidate cleared the return half of the bar, which does not
show that gold cannot hold a third strategy.

## Relationship to the alpha search

`RS-XAUUSD-20260818-004` searched the 30-minute bars for a directional edge; none of 26
features passed validation and the short-horizon reversal pattern is unvalidated. This study is
what remained after that: not a new edge, but a way of combining two that already exist. What
it supports is narrow: if only S2 runs today, adding S1 is a clear improvement.
