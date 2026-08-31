# RS-XAUUSD-20260818-005 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What it supports

Continuing to run both strategies at a comparable risk allocation — which is already what
happens with fixed-fractional sizing per trade. This study is a confirmation that the
current arrangement is sound, not a proposal to change it.

## What it does not support

- Dropping S2. Against S2 alone, blending lifts Sharpe from 2.33 to 3.88 and cuts the worst
  drawdown from 12.8R to 2.2R, with P(improvement) of 0.999 and 0.989.
- Dropping S1. Against S1 alone the blend's Sharpe gain is +0.22 with a 90% interval of
  −0.74 to +0.97. That does not clear zero, so the blend is **not** shown to beat S1.
- Optimising the weight. Sharpe runs 3.71 to 3.96 across every S1 weight from 0.4 to 0.8.
  The in-sample optimum at 0.6 sits inside that plateau and carries no information.

## The number worth remembering

The two strategies' monthly returns correlate at **0.234**. Two long-only strategies on the
same instrument were expected to be near-duplicates. They are not, and that single fact is
what makes running both worth more than running the better one twice as large.

## What the Sharpe figures are not

They are not a live expectation. Values near 3.9 do not survive real execution; these are
Strategy Tester R multiples aggregated monthly over a period in which XAUUSD rose 112%,
with no slippage beyond what the export already contains. Every claim here is a ranking
between allocations of the same backtest.

## Revision 2 — the search for an uncorrelated third strategy

The brief asked for an uncorrelated strategy type, explicitly allowing long/short. Two
candidates were built and both fail, but the exercise produced the most reusable output in
this line of work.

**The search is not structurally blocked.** Regressing monthly R on gold gives S1 a beta of
0.444 and S2 0.27; 79.7% and 77.0% of their returns are alpha rather than being long gold,
and both were positive in the 8 months gold fell (4.06R and 1.507R). Had they been mostly
beta, decorrelating would have required shorting gold. They are not. (Caveat: backtest
alpha, and the V3.4-to-V4.1 version history is itself a selection process.)

**Candidate 1, counter-trend fade.** Symmetric long and short, correlation 0.036 to the blend
— exactly the property wanted. Sharpe -0.046 over 700 trades. It lowers the blend at
every weight. Uncorrelated and unprofitable is not a diversifier.

**Candidate 2, the mirror of S2.** S2's hammer definition reverse-engineered from the
export's own flags and every characteristic reflected. Shorting the resulting shooting
stars loses in every period and every exit rule, with win rates of 13% to 39%. A working
long pattern does not become a working short by reflection.

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

Correlation is worth more than return. A mediocre strategy that is genuinely negatively
correlated beats a good one that is not — which is the argument for a **different
instrument** rather than another rule on this one. Both candidates here cleared the
correlation half of the bar and failed the return half.

## Relationship to the alpha search

`RS-XAUUSD-20260818-004` searched the 30-minute bars for a directional edge and found that
everything real is smaller than the spread. This study is what remained after that: not a
new edge, but a better use of two that already exist. It is the only positive result of the
whole search, and its mechanism — imperfect correlation — costs nothing to exploit.
