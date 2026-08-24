# RS-XAUUSD-20260818-003 — impact

## Impact on 請分析

**None.** No score, threshold, gate, or output block changes.

## What this closes

The last unused data source. MGC micro gold futures have sat beside the spot series since
2024-01-02 and no study had opened the file.

| Family | Tested | Result |
|---|---|---|
| Futures-spot basis (detrended) | 3 features | nothing separable |
| Real COMEX contract volume | 2 features | nothing separable |
| Spot-to-futures volume mix | 1 feature | nothing separable |
| Futures bar range against futures ATR | 1 feature | nothing separable |

Family permutation across all seven: p = 0.57 (S1), p = 0.86 (S2).

## It also resolves an open limitation

`RS-XAUUSD-20260818-002` recorded that spot tick volume is broker activity rather than
exchange volume, and could not say whether that mattered. It does not: across 30,734
aligned bars the two rank-correlate at 0.873. Real exchange volume is close to the same
measurement and finds the same nothing. That limitation is closed rather than carried
forward.

## The trap worth keeping

Raw basis looks like a market-state variable and is a calendar. Median basis by month spans
39.7 USD across the roll cycle, so splitting trades on it splits them by month: the lowest
S1 tercile is 1.3% even-month, the other two are 82.8% and 73.2%. That split produces a
10.8pp win-rate spread with nothing behind it.

Anyone reaching for futures data again should detrend before splitting. Against its own
trailing median the cross-month spread is 0.43 USD.

## Practical consequence

Do not add a basis filter, a futures-volume filter, or a spot-versus-futures activity rule.
And do not treat "we used real exchange volume this time" as a reason to revisit the volume
question — it is the same measurement.
