# RS-XAUUSD-20260819-001 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What was asked and what came back

"Find repeatable regularities in the 30-minute data; overfitting is acceptable."

Four families were mined. Three are artefacts and one is real, and the three artefacts are
worth more than they look because each one fails in a different, reusable way.

| Family | In-sample result | What it actually was |
|---|---|---|
| Candlestick patterns | t up to 5.88, every pattern profitable | the market's own drift |
| Weekly half-hour slots | basket Sharpe 3.07 | selection; holdout −2.12 |
| Round numbers | positive in all three periods | pooled p 0.17, and $50/$25 contradict |
| Daily extreme timing | clustering at 09:00, 22:00–23:00 | **survives its null** |
| Multi-bar sequences | two candidates consistent in all 3 periods | enumeration artefact |

## Revision 2 — multi-bar sequences

Bars were discretised two ways — close-to-close move in ATR units, and where the close sat
inside its own bar — and every sequence of length 2 to 4 enumerated against 4 and 8 bar
horizons. 12 settings in all.

Family permutation p runs from 0.236 to 0.99, and in **7 of 12 settings the largest
observed |t| is below the null median**. Enumerating every sequence and reporting the
strongest finds less than a shuffle of the same returns typically finds.

Two candidates passed the naive screen of holding their sign in all three periods, and each
was killed by a check that screen cannot perform:

- **uU → dU → udU → uudU** were all positive and strengthened with specificity, which is
  what a real effect looks like and an isolated cell is not. An ablation settled it: remove
  the compression precondition and the effect is unchanged, the excess sits at or below a
  0.02% round trip, and the short side changes sign between periods.
- **MTTT and TTTL** were negative in all three periods with a coherent exhaustion story.
  Exhaustion must be symmetric, so repeated closes at the bottom must outperform. The mirror
  holds in 0 of 6 settings, and at the one setting whose top side is consistent the
  bottom side is negative in two periods of three — the wrong direction.

Neither check is statistical. Both ask whether the claimed mechanism produces the side
effects it would have to produce, which is the screen that catches what sign-consistency
lets through.

## The one usable regularity

Daily highs and lows form far more often than a random walk produces at **09:00 and
22:00–23:00 Taipei**, and far less often at **04:00**. Magnitudes hold across all three
periods, not just signs — the 04:00 high excess is −4.9, −4.8 and −5.3.

This is the intraday volatility profile in a form that can be acted on. It says nothing
about direction.

**Where it is useful:**

- A new daily extreme is unlikely to be set between roughly 03:00 and 06:00 Taipei. If you
  are waiting for a better entry price in that window, it is unlikely to arrive.
- **Corrected 2026-08-23 (revision 3).** This bullet previously read "the day's low most
  often forms in the Asian morning; the day's high most often at the New York session".
  That is a directional reading of a non-directional result and the data does not support
  it. Raw formation frequency over 672 sessions puts the low in 07:00–09:00 on 30.4% of
  days and the high there on 22.2%; in 21:00–23:00 the high leads the low by only 2.1
  points. P(low before high) is 54.9% / 56.7% / 54.9% across the three periods — about five
  points above a coin flip in a market that rose 112%, which is drift. Under the shuffled
  null, 09:00 and 21:00–23:00 show excess for the high **and** the low simultaneously, and
  04:00 shows a deficit for both. What the finding says is that extremes cluster at those
  hours, not which kind of extreme. See RS-XAUUSD-20260823-001.
- An exit placed to catch an extreme is competing with the whole market at 22:00 and with
  almost nobody at 04:00.

**Where it is not useful:** as a directional signal. It cannot tell you whether the extreme
forming at 22:00 will be a high or a low.

## The methodological result, which is the larger one

Each artefact needed a *different* null, and picking the wrong one would have produced a
confident false finding:

- **Drift.** Scoring patterns against zero on an instrument that rose 112% makes every
  pattern profitable. Score against the same period's baseline instead.
- **Arcsine geometry.** A random walk puts its extremes at the start and end of any window.
  The raw distribution of daily high and low times has a pronounced U shape whether or not
  the market has structure. Shuffle each day's own returns.
- **Selection.** Picking the best cells from a 240-cell grid produces a Sharpe above 3 that
  is worth nothing. Report the out-of-sample number beside it.

Any future pattern search on this data should state its null before it states its result.
