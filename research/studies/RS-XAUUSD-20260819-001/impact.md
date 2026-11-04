# RS-XAUUSD-20260819-001 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What was asked and what came back

"Find repeatable regularities in the 30-minute data; overfitting is acceptable."

Five families were mined. Four are artefacts or indistinguishable from noise, and one is a
timing regularity that holds only in part (23:00 clearly and 22:00 marginally, under both definitions of an extreme).
The artefacts are worth more than they look because each one fails in a different, reusable
way. Exploratory study, no pre-registration; revision 5 (after independent review) corrected the text below,
and figures marked *post-hoc recompute* were recalculated from the same CSV without regenerating
`results.json`.

| Family | In-sample result | What it actually was |
|---|---|---|
| Candlestick patterns | naive t up to 5.88 (overlapping 16-bar returns), every pattern profitable | the market's own drift; HAC t for the 5.88 row is 3.57 (post-hoc recompute) |
| Weekly half-hour slots | basket Sharpe 3.07 | selection; holdout −2.12 (Sharpe SE about 0.83 / 1.23 / 1.39, post-hoc) |
| Round numbers | positive in all three periods | pooled p 0.1913; $50/$25 rows not distinguishable from zero (rows overlap, tolerances differ) |
| Daily extreme timing | clustering at 09:00, 21:00–23:00 | original comparison: 9, 21, 22, 23 cluster, 4 avoids; same-quantity recompute (post-hoc): only 22 (marginal) and 23 hold |
| Multi-bar sequences | two candidates consistent in all 3 periods | not distinguishable from shuffling; neither passed an ablation or mirror that cover only part of the settings |

## Revision 2 — multi-bar sequences

Bars were discretised two ways — close-to-close move in ATR units, and where the close sat
inside its own bar — and every sequence of length 2 to 4 enumerated against 4 and 8 bar
horizons. 12 settings in all.

Family permutation p runs from 0.236 to 0.99, and in 7 of 12 settings the largest
observed |t| is below the null median. With no effect about 6 of 12 would be expected below
it, so this reads as "not distinguishable from a shuffle of the same returns", not as
"less extreme".

Two candidates passed the naive screen of holding their sign in all three periods, and each
was killed by a check that screen cannot perform:

- **uU → dU → udU → uudU** were all positive and strengthened with specificity, which is
  what a real effect looks like and an isolated cell is not. The ablation (1-2 bar
  preconditions only; uudU has 3) could not separate the pattern from a large bar alone in
  valid and holdout (k2_h4_up holdout 0.0184 vs 0.0187; train 0.0128 vs 0.0050, untested),
  the ablation-row excess sits at or below a 0.02% round trip, and the short side changes sign
  between periods. uudU's holdout +0.194% rests on n = 33.
- **MTTT and TTTL** were negative in all three periods with a coherent exhaustion story.
  If exhaustion is symmetric (an assumption), repeated closes at the bottom should
  outperform. The mirror holds in 0 of 6 settings, the top side itself is consistent in only
  1 of 6, and at that one setting the bottom side is negative in two periods of three — the
  wrong direction.

Neither check is statistical, and both were designed after the candidates were seen. Both ask
whether the claimed mechanism produces the side effects it would have to produce, which is a
screen that goes beyond sign-consistency.

## The one timing regularity, and what holds of it

Under the study's original comparison, daily highs and lows form more often than the
shuffled null at **09:00 and 21:00–23:00 Taipei**, and less often at **04:00** (04:00 high
excess −5.11, −4.85 and −5.20 percentage points of days; days = 360 / 161 / 131).

That comparison sets the real extreme (the bar with the day's highest high or lowest low)
against a null extreme taken from the cumulative close path, which are not the same quantity,
and the shuffle keeps neither the intrabar range nor the hourly volatility profile. Post-hoc
recompute with both sides defined from closes and the same rule: only **22:00 and 23:00**
cluster (11:00 appears as an avoid hour); 09:00, 21:00 and 04:00 drop out. The 22:00 pass is
marginal: its valid-period high excess is 1.09 against the 1.0 threshold, within Monte Carlo
noise (sd 0.11); over 27 shuffle seeds 22:00 passes in 16 and drops out in 11, while 23:00
clusters and 11:00 avoids in all 27 (post-hoc seed sweep). Single-period
binomial z at 09:00 is 1.0–1.6 in valid and holdout. The hours were chosen from all three
periods together, with no untouched confirmation period and no multiplicity correction, and
21:00–23:00 Taipei mixes US summer and winter time.

This is closer to the intraday volatility profile than to a rule. It says nothing about
direction.

**Where it is useful:**

- Under the original comparison a new daily extreme was set less often at 04:00 Taipei
  (03:00 and 05:00–06:00 are not consistent avoid hours), but 04:00 does not pass under the
  same-quantity recompute, so treat it as a weak reference, not a rule. No exit or entry-wait
  test was run.
- **Corrected 2026-08-23 (revision 3).** This bullet previously read "the day's low most
  often forms in the Asian morning; the day's high most often at the New York session".
  That is a directional reading of a non-directional result and the data does not support
  it. Raw formation frequency over 672 sessions puts the low in 07:00–09:00 on 30.4% of
  days and the high there on 22.3% (post-hoc recompute, 672 sessions; the earlier text said
  22.2%); in 21:00–23:00 the high leads the low by only 2.1 points. P(low before high) is
  55.4% / 58.7% / 51.9% (n = 372 / 167 / 133) on that set and 55.0% / 59.0% / 52.7% on the
  652-session set the extremes family uses; the earlier 54.9 / 56.7 / 54.9% could not be
  reproduced. About five points above a coin flip in a market that rose 112% is consistent
  with drift, which was not tested as the cause. Under the shuffled
  null (original comparison), 09:00 and 21:00–23:00 show excess for the high **and** the low
  simultaneously, and 04:00 shows a deficit for both. What the finding says is that extremes cluster at those
  hours, not which kind of extreme. See RS-XAUUSD-20260823-001.
- An exit placed to catch an extreme may be competing with more of the market at 22:00–23:00
  than at 04:00; this was not tested with any exit rule.

**Where it is not useful:** as a directional signal. It cannot tell you whether the extreme
forming at 22:00 will be a high or a low.

## The methodological result, which is the larger one

Each artefact needed a *different* null, and picking the wrong one would have produced a
confident false finding:

- **Drift.** Scoring patterns against zero on an instrument that rose 112% makes every
  pattern profitable. Score against the same period's baseline instead.
- **Arcsine geometry.** A random walk puts its extremes at the start and end of any window.
  The raw distribution of daily high and low times has a pronounced U shape whether or not
  the market has structure. Shuffle each day's own returns, and compare like with like: the
  real extreme and the null extreme must be the same quantity.
- **Selection.** Picking the best cells from a 230-cell grid produces a Sharpe above 3 that
  is worth nothing. Report the out-of-sample number beside it.

Any future pattern search on this data should state its null before it states its result.
