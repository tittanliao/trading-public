# Live impact — RS-XAUUSD-20260727-001 (revision 5)

Current as of 2026-08-17. This file is a snapshot of what `請分析` does differently
today; the history lives in the private decision record.

## No live impact

This study has **no** live effect on `請分析`. Both adjustments it previously supplied
were revoked on 2026-08-17 by owner decision, which is the revocation condition the
revision-3 version of this file already recorded.

The report itself is not retracted. Its method and numbers stand, and its fail-pattern,
BB, DXY, MTF, hold-time and temporal-stability sections remain valid. What was withdrawn
is the score adjustment two of its breakdowns were used to justify.

## Revoked 1 — Macro composite verdict → S1 recommendation score

- Adopted 2026-07-28, revoked 2026-08-17.
- Was: add the Macro verdict's rank score (`STRONG BUY +1`, `WAIT 0`, `NEUTRAL −1`).
- Why revoked: the three groups are not separable at this sample size. Their 95% CIs
  overlap across roughly 50–63%; the largest observed win-rate gap is 3.98 points while
  n=82/144/211 can only resolve about 14.5; and the ranks contradict profit factor,
  since NEUTRAL scores −1 with the highest PF (1.963) and WAIT scores 0 with the lowest
  (1.837).

## Revoked 2 — 30-minute entry slot → S1 recommendation score

- Adopted 2026-07-28, revoked 2026-08-17.
- Was: add the matching `HH:00`/`HH:30` slot's rank score (−2…+2).
- Why revoked: 48 slots over 450 trades leaves a median of 9 trades per slot, and even
  the largest (n=21) can only resolve a gap of about 36.6 points. Simulating 20000 runs
  with every slot held at the baseline 56.22% produces a median best-to-worst spread of
  83.3 points, larger than the 70.0 observed; P(spread ≥ observed | no effect) = 91%.
  This carried the larger of the two weights.

## Re-adoption condition

A later registered study showing stable Macro or slot effects on sample sizes able to
resolve them, validated out of sample. Do not reinstate either adjustment on a finer
time granularity than the data can support.
