"""
SMC (Smart Money Concepts) pre-computed indicator arrays for XAUUSD 30-min bars.

Call precompute(price) ONCE per dataset. It returns a dict of bool/float arrays
aligned to the price index. Signal functions then do O(1) lookups per bar.

Arrays produced:
  in_bull_fvg  — close is inside a prior bullish Fair Value Gap
  in_bear_fvg  — close is inside a prior bearish Fair Value Gap
  at_bull_ob   — close is retracing to a confirmed bullish Order Block
  at_bear_ob   — close is retracing to a confirmed bearish Order Block
  ssl_swept    — bar swept sell-side liquidity (new low, closed back above)
  bsl_swept    — bar swept buy-side liquidity (new high, closed back below)
  rsi14        — RSI(14)
  ema21        — EMA(21)
  ema50        — EMA(50)

No look-ahead: at_bull_ob / at_bear_ob entries are only marked after OB_CONFIRM_BARS
have passed (so the confirming impulse is fully visible at signal time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import indicators as ind

# ── Parameters ─────────────────────────────────────────────────────────────────
FVG_LOOKBACK    = 100   # bars to keep a FVG active after it forms   (~50 h on 30m)
OB_CONFIRM_BARS = 12    # bars after OB candle needed to confirm impulse (~6 h)
OB_MOVE_PCT     = 0.003 # minimum price move to validate an OB (0.3%)
OB_LOOKBACK     = 60    # bars to allow price to retrace to OB after confirmation
LIQ_LOOKBACK    = 30    # range for liquidity levels (prior swing high/low)


def precompute(price: pd.DataFrame) -> dict:
    """
    Scan the entire price series once and return all SMC arrays.

    Parameters
    ----------
    price : DataFrame with columns [time, open, high, low, close]

    Returns
    -------
    dict of numpy arrays, each length == len(price)
    """
    n = len(price)
    o = price["open"].to_numpy()
    h = price["high"].to_numpy()
    l = price["low"].to_numpy()
    c = price["close"].to_numpy()

    # ── Classic indicators (computed once) ────────────────────────────────────
    rsi14 = ind.rsi(c, 14)
    ema21 = ind.ema(c, 21)
    ema50 = ind.ema(c, 50)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Fair Value Gaps
    #
    # Bullish FVG at bar j:  high[j-2] < low[j]
    #   → imbalance zone = [high[j-2], low[j]]   (price moved up, left a gap)
    #   → entry signal when price retraces DOWN into this zone
    #
    # Bearish FVG at bar j:  low[j-2] > high[j]
    #   → imbalance zone = [high[j], low[j-2]]   (price moved down, left a gap)
    #   → entry signal when price retraces UP into this zone
    # ══════════════════════════════════════════════════════════════════════════
    bull_fvgs: list[tuple[int, float, float]] = []   # (j, bottom, top)
    bear_fvgs: list[tuple[int, float, float]] = []

    for j in range(2, n):
        if h[j - 2] < l[j]:
            bull_fvgs.append((j, h[j - 2], l[j]))
        if l[j - 2] > h[j]:
            bear_fvgs.append((j, h[j], l[j - 2]))

    in_bull_fvg = np.zeros(n, dtype=bool)
    for j, bot, top in bull_fvgs:
        # Earliest retrace bar: j+2 (FVG just completed)
        s = j + 2
        e = min(j + FVG_LOOKBACK, n)
        if s >= e:
            continue
        seg = c[s:e]
        in_bull_fvg[s:e] |= (seg >= bot) & (seg <= top)

    in_bear_fvg = np.zeros(n, dtype=bool)
    for j, bot, top in bear_fvgs:
        s = j + 2
        e = min(j + FVG_LOOKBACK, n)
        if s >= e:
            continue
        seg = c[s:e]
        in_bear_fvg[s:e] |= (seg >= bot) & (seg <= top)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Order Blocks
    #
    # Bullish OB: last BEARISH candle before an upward impulse ≥ OB_MOVE_PCT
    #   → OB zone = [low[j], high[j]] of that bearish candle
    #   → valid retrace = price closes inside zone AFTER the impulse is confirmed
    #
    # Bearish OB: last BULLISH candle before a downward impulse ≥ OB_MOVE_PCT
    #
    # Look-ahead safety: we only mark retrace bars starting at j + OB_CONFIRM_BARS + 1,
    # so at signal time i we have fully observed the confirming move.
    # ══════════════════════════════════════════════════════════════════════════
    at_bull_ob = np.zeros(n, dtype=bool)
    at_bear_ob = np.zeros(n, dtype=bool)

    for j in range(1, n - OB_CONFIRM_BARS - 2):
        confirm_end = j + OB_CONFIRM_BARS + 1   # exclusive upper bound for confirmation window

        # ── Bullish OB: bearish candle ────────────────────────────────────
        if c[j] < o[j]:
            ob_hi = h[j]
            ob_lo = l[j]
            future_hi = h[j + 1:confirm_end].max()
            if future_hi >= ob_hi * (1 + OB_MOVE_PCT):
                # Mark retest window (after confirmation is complete)
                s = confirm_end          # first bar where OB is fully confirmed
                e = min(j + OB_LOOKBACK, n)
                if s < e:
                    seg = c[s:e]
                    at_bull_ob[s:e] |= (seg >= ob_lo) & (seg <= ob_hi)

        # ── Bearish OB: bullish candle ────────────────────────────────────
        if c[j] > o[j]:
            ob_hi = h[j]
            ob_lo = l[j]
            future_lo = l[j + 1:confirm_end].min()
            if future_lo <= ob_lo * (1 - OB_MOVE_PCT):
                s = confirm_end
                e = min(j + OB_LOOKBACK, n)
                if s < e:
                    seg = c[s:e]
                    at_bear_ob[s:e] |= (seg >= ob_lo) & (seg <= ob_hi)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Liquidity Sweeps
    #
    # Sell-side liquidity (SSL):  price broke below the prior LIQ_LOOKBACK-bar low
    #   then closed BACK ABOVE it → smart money swept stops under lows, reversed.
    #   → Bullish signal (long entry)
    #
    # Buy-side liquidity (BSL):  price broke above the prior LIQ_LOOKBACK-bar high
    #   then closed BACK BELOW it → smart money swept highs, reversed.
    #   → Bearish signal (short entry)
    # ══════════════════════════════════════════════════════════════════════════
    ssl_swept = np.zeros(n, dtype=bool)
    bsl_swept = np.zeros(n, dtype=bool)

    for i in range(LIQ_LOOKBACK + 1, n):
        ssl_level = l[i - LIQ_LOOKBACK:i].min()   # lowest low in prior window
        bsl_level = h[i - LIQ_LOOKBACK:i].max()   # highest high in prior window

        # SSL sweep: wick below prior range low, close recovers
        if l[i] < ssl_level and c[i] > ssl_level:
            ssl_swept[i] = True

        # BSL sweep: wick above prior range high, close drops back
        if h[i] > bsl_level and c[i] < bsl_level:
            bsl_swept[i] = True

    return {
        "in_bull_fvg": in_bull_fvg,
        "in_bear_fvg": in_bear_fvg,
        "at_bull_ob":  at_bull_ob,
        "at_bear_ob":  at_bear_ob,
        "ssl_swept":   ssl_swept,
        "bsl_swept":   bsl_swept,
        "rsi14":       rsi14,
        "ema21":       ema21,
        "ema50":       ema50,
    }


def describe(smc: dict, price: pd.DataFrame) -> None:
    """Print a quick summary of detected SMC zones (for debugging)."""
    n = len(price)
    print(f"  Bars analysed : {n}")
    print(f"  Bullish FVGs  : {smc['in_bull_fvg'].sum()} bars in zone")
    print(f"  Bearish FVGs  : {smc['in_bear_fvg'].sum()} bars in zone")
    print(f"  Bullish OBs   : {smc['at_bull_ob'].sum()} bars in zone")
    print(f"  Bearish OBs   : {smc['at_bear_ob'].sum()} bars in zone")
    print(f"  SSL sweeps    : {smc['ssl_swept'].sum()} bars")
    print(f"  BSL sweeps    : {smc['bsl_swept'].sum()} bars")
