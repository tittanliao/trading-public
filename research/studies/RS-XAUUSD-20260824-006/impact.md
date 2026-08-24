# Operational impact — RS-XAUUSD-20260824-006

## What changes

The %B reading on the signal surface **stays**. It survived four independent measurements,
including the test that inverted a 71% win rate two studies earlier.

It gains one required qualifier: **the bar size**.

Before:

> Bollinger %B above the upper band — 73.17% historical win rate

After:

> Bollinger %B above the upper band **on 30-minute bars** — 73.17% historical win rate.
> The hourly chart disagrees about two-thirds of the time.

## Why the qualifier is not pedantry

The same trade, the same formula, the same price feed — only the bar size differs:

| | 30-minute | hourly |
|---|---|---|
| entries called "above the upper band" | 71 | 28 |
| of 422 trades | 16.8% | 6.6% |
| in both | 22 | |
| same zone assigned | 32.23% of trades | |

A person checking "is %B above the band?" on an hourly chart and a person checking on a
30-minute chart are answering different questions two times in three. Without the
timeframe, the 73.17% is attached to a condition the reader cannot reliably evaluate.

## What does not change

No filter is introduced. `RS-XAUUSD-20260823-002` established that filtering on %B raises
the win rate and loses money, and this study reproduces that on every variant — the
tightest selection here posts an 89.29% win rate while capturing 27.60% of the available
return. %B remains a reading that informs a decision, never a gate that makes one.

## Carried forward

The instrument test passed at a 0.9998 correlation. That is a weak falsifier, and the
finding has not yet faced a strong one. A genuine instrument test for a gold signal needs
something that is not gold — the natural candidate is running the same %B construction on
a different market entirely and asking whether the effect appears there too, which is a TX
study rather than an XAUUSD one.
