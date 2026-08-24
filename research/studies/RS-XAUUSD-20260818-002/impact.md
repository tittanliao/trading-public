# RS-XAUUSD-20260818-002 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What this closes

Two search spaces that look promising from outside and are not:

| Family | Tested | Result |
|---|---|---|
| Entry-side volume | 3 features (20-bar, 96-bar, 4-bar trend) | nothing separable |
| Entry-side volatility | 3 features (ATR %, ATR percentile, range/ATR) | nothing separable |
| Bar anatomy and location | 4 features (body, lower wick, range position, 2h return) | nothing separable |
| Cross-strategy interaction | 6 splits across both strategies | nothing separable |

Family permutation test across all ten features: p = 0.63 (S1), p = 0.60 (S2). A shuffle of
the same outcomes reproduces the largest observed spread about three times in five.

## The one conclusion worth carrying forward

`RS-XAUUSD-20260817-001` found GVZ and VIX separate nothing, and left open whether the
*proxy* was at fault — both are daily, external, and lagged relative to a 30-minute
strategy. This study answers that. ATR percentile computed from the same 30-minute series
the strategies trade does no better, and is non-monotonic in both.

Volatility regime does not separate S1 or S2 outcomes. That is now tested from two
independent directions and can be treated as settled rather than as an open question.

## Practical consequence

Do not add a volatility filter, a volume filter, or a "don't trade S2 while S1 is open"
rule. Each has been tested against its own resolution limit and against a search-wide
permutation, and none of them has anything behind it.
