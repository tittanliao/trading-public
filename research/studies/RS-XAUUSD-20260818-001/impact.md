# RS-XAUUSD-20260818-001 — impact

## Impact on 請分析

**None adopted.** No score, threshold, gate, or output block changes.

The one positive result — S2 one-bar confirmation entry — is recorded as a prospective
candidate. It is not adopted because it fails the standard this project has been applying
all along, and applying it selectively to a result that happens to be favourable would be
worse than never having tested it:

- significant in sample (p = 0.0023) but not in the held-out period on its own (p = 0.139
  on 25 confirmed trades);
- it reduces total return, so adopting it is a trade the owner has to want, not a strict
  improvement;
- both strategies are long-only across a strongly rising XAUUSD, and a filter that selects
  continuation may be reading that trend rather than the signal.

## What the owner can use today

The study answers a question that gets asked directly — "can the win rate go higher?" —
and the answer is on the record with a price attached:

> Win rate on both strategies can be raised to roughly 89% (S1) and 96% (S2) by adding a
> take-profit. Every level tested cuts total return, S1 from +88.6% to between +14% and
> +36%. A take-profit cannot turn a winner into a loser, so win rate can only go up; that
> is what makes it a bad measure of an exit change on its own.

If the owner discretionarily wants the S2 confirmation behaviour before it is confirmed,
the honest framing is: fewer trades, better each, and whether that is *more overall* depends
entirely on how you hold risk equal.

- Take about 55% of S2 signals (91 of 167 over the sample).
- Those trades won 62.6% against 47.9%, at PF 2.97 against 2.06.
- Per trade, +0.7609R against +0.5689R.

Total return relative to baseline, by how risk is aligned:

| Alignment | Filter as % of baseline | Read it? |
|---|---|---|
| Fixed risk per trade | 73.3% | one convention among four |
| Equal total risk deployed | 133.7% | implies 1.824× risk per trade |
| **Equal volatility over the period** | **115.4%** | **yes — the standard comparison** |
| Equal max drawdown | 196.3% | no; 90% interval is 65.1–266.4% |

Revision 2 quoted only the first row, which understated the case. Revision 3 quotes the
third: **a modest genuine improvement**, roughly 15% more return at the same volatility, not
the near-doubling the drawdown row suggests. S1 sits at 69.9% and is worse under every
convention, matching its failed permutation test.

Sizing instead of filtering does not work: confirmation is not known until a bar after
entry, and adding to confirmed trades at that later price fails at all six add factors in
both strategies.

## Rules this study did NOT support

Recorded so the same ground is not re-walked:

| Family | Variants tested per strategy | Improving total return |
|---|---|---|
| Take-profit overlay | 11 levels | none |
| Stop tightening | 12 levels (identity cases excluded) | none |
| Underwater time-exit | 24 rules | none for S2; 2 of 24 for S1, consistent with sweep noise |
| Trailing stop | 24 combinations | none; best costs S1 5.4% and S2 36.3% of total return |
| Breakeven move | 6 arming levels | none; costs S2 up to 59% at a 0.2% arm |
| Confirmation entry (S1) | 4 bars × 9 thresholds | fails permutation (p = 0.48) |

Trailing and breakeven were added in revision 2. Revision 1 called the exit side closed
after three families, none of which lets a winner run while still cutting a reversal — the
claim was broader than what had been tested.

## Relationship to earlier studies

`RS-XAUUSD-20260817-001` closed out the external-state family: no Macro factor, composite,
or GVZ threshold separates outcomes. This study opens the internal-path family and mostly
closes it too. Between them, the searched space now covers external regime, entry timing,
and exit structure, with one surviving candidate.

The slot statistics kept available in `scripts/research/show_entry_slot_stats.py` are
unaffected.
