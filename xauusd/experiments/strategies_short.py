"""
20 short-side strategy signal functions for XAUUSD 30-min bars.

Each function mirrors its long-side counterpart (E01–E20) with inverted logic:
  - Trend signals: bearish crossovers / breakdowns instead of bullish
  - Oscillators: overbought exits instead of oversold
  - Patterns: shooting star / bearish engulfing instead of hammer / bullish engulfing
  - Breakouts: breakdowns below support instead of breakouts above resistance

Signal signature:
    signal(price: pd.DataFrame, i: int) -> bool
    Return True to enter a SHORT position on bar[i] close (executes at bar[i+1] open).

Groups:
    S01–S05  Trend / Momentum (bearish)
    S06–S10  Oscillator extremes (overbought)
    S11–S13  Bollinger Band (upper band / breakdown)
    S14–S16  Breakdown (below support)
    S17–S20  Pattern / Confluence (bearish)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import indicators as ind


def _arrays(price: pd.DataFrame):
    return (price["open"].to_numpy(),
            price["high"].to_numpy(),
            price["low"].to_numpy(),
            price["close"].to_numpy())


# ============================================================================
# GROUP 1 — Trend / Momentum  (S01–S05)
# ============================================================================

def S01_EMA_Cross(price: pd.DataFrame, i: int) -> bool:
    """Fast EMA(8) crosses BELOW Slow EMA(21) — bearish trend signal."""
    _, _, _, c = _arrays(price)
    fast = ind.ema(c, 8)
    slow = ind.ema(c, 21)
    return bool(fast[i] < slow[i] and fast[i-1] >= slow[i-1])


def S02_Triple_EMA(price: pd.DataFrame, i: int) -> bool:
    """
    Bearish triple EMA alignment: EMA8 < EMA21 < EMA55.
    Enter on pullback to EMA8 followed by rejection (close < open after touching EMA8).
    """
    if i < 55:
        return False
    _, _, _, c = _arrays(price)
    e8  = ind.ema(c, 8)
    e21 = ind.ema(c, 21)
    e55 = ind.ema(c, 55)
    aligned    = e8[i] < e21[i] < e55[i]
    pullback   = c[i-1] >= e8[i-1] * 0.998   # previous bar touched EMA8 from below
    rejection  = c[i] < c[i-1]               # current bar is bearish
    return bool(aligned and pullback and rejection)


def S03_MACD_Signal(price: pd.DataFrame, i: int) -> bool:
    """MACD histogram turns NEGATIVE (crosses below zero)."""
    if i < 35:
        return False
    _, _, _, c = _arrays(price)
    ml, sl, hist = ind.macd(c)
    cross_down = hist[i] < 0 and hist[i-1] >= 0
    return bool(cross_down and ml[i] < 5)   # not deep in positive territory


def S04_Supertrend(price: pd.DataFrame, i: int) -> bool:
    """Supertrend(10, 3.0) flips from bullish to BEARISH."""
    if i < 15:
        return False
    _, h, l, c = _arrays(price)
    trend = ind.supertrend(h, l, c, 10, 3.0)
    return bool(trend[i] == -1 and trend[i-1] == 1)


def S05_ADX_EMA_Rejection(price: pd.DataFrame, i: int) -> bool:
    """
    Strong downtrend (ADX > 25) + price below EMA50.
    Enter when price bounces UP to EMA20 and gets rejected (close < EMA20 after being above).
    """
    if i < 55:
        return False
    _, h, l, c = _arrays(price)
    adx_v, _, _ = ind.adx(h, l, c, 14)
    e20 = ind.ema(c, 20)
    e50 = ind.ema(c, 50)
    trending     = adx_v[i] > 25
    below_e50    = c[i] < e50[i]
    rejection    = c[i-1] > e20[i-1] and c[i] < e20[i]   # crossed back below EMA20
    return bool(trending and below_e50 and rejection)


# ============================================================================
# GROUP 2 — Oscillator Overbought  (S06–S10)
# ============================================================================

def S06_RSI_Overbought(price: pd.DataFrame, i: int) -> bool:
    """RSI(14) crosses from above 70 to below 70 (overbought exit)."""
    if i < 20:
        return False
    _, _, _, c = _arrays(price)
    r = ind.rsi(c, 14)
    return bool(r[i] < 70 and r[i-1] >= 70)


def S07_Stoch_Cross(price: pd.DataFrame, i: int) -> bool:
    """Stochastic K crosses BELOW D from the overbought zone (both > 75)."""
    if i < 20:
        return False
    _, h, l, c = _arrays(price)
    k, d = ind.stochastic(h, l, c, 14, 3, 3)
    cross_down  = k[i] < d[i] and k[i-1] >= d[i-1]
    overbought  = k[i-1] > 75
    return bool(cross_down and overbought)


def S08_CCI_Overbought(price: pd.DataFrame, i: int) -> bool:
    """CCI(20) crosses from above +100 to below (classic overbought reversal)."""
    if i < 25:
        return False
    _, h, l, c = _arrays(price)
    cci_v = ind.cci(h, l, c, 20)
    return bool(cci_v[i] < 100 and cci_v[i-1] >= 100)


def S09_Williams_R(price: pd.DataFrame, i: int) -> bool:
    """Williams %R(14) crosses from above -20 downward (overbought exit)."""
    if i < 20:
        return False
    _, h, l, c = _arrays(price)
    wr = ind.williams_r(h, l, c, 14)
    return bool(wr[i] < -20 and wr[i-1] >= -20)


def S10_RSI_Divergence(price: pd.DataFrame, i: int) -> bool:
    """
    Bearish RSI divergence proxy:
    Price makes a higher high but RSI makes a lower high over the last 5 bars.
    Suggests weakening momentum despite price strength.
    """
    if i < 20:
        return False
    _, _, _, c = _arrays(price)
    r = ind.rsi(c, 14)
    look = 5
    price_higher_high = c[i] > max(c[i-look:i])
    rsi_lower_high    = r[i]  < max(r[i-look:i])
    rsi_falling       = r[i]  < r[i-1]
    return bool(price_higher_high and rsi_lower_high and rsi_falling)


# ============================================================================
# GROUP 3 — Bollinger Band  (S11–S13)
# ============================================================================

def S11_BB_Upper_Touch(price: pd.DataFrame, i: int) -> bool:
    """
    Price touches BB upper band then closes back below it — mean reversion short.
    Confirm with RSI > 55 (not oversold) and bearish candle.
    """
    if i < 25:
        return False
    _, h, l, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    touched  = h[i] >= upper[i]    # wick touched upper band
    recovery = c[i] < upper[i]     # but closed below
    bearish  = c[i] < c[i-1]       # bearish close
    r = ind.rsi(c, 14)
    return bool(touched and recovery and bearish and r[i] > 50)


def S12_BB_Squeeze_Break(price: pd.DataFrame, i: int) -> bool:
    """
    BB squeeze followed by DOWNWARD expansion.
    Enters when BB width expands and price closes BELOW basis.
    """
    if i < 30:
        return False
    _, h, l, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    width     = upper - lower
    avg_width = ind.sma(width, 20)
    squeeze   = width[i-1] < avg_width[i-1] * 0.85
    expanding = width[i]   > width[i-1]
    below_mid = c[i] < basis[i]
    return bool(squeeze and expanding and below_mid)


def S13_BB_Basis_Rejection(price: pd.DataFrame, i: int) -> bool:
    """
    Bearish 'walking the lower band': BB basis slope negative,
    enter when price bounces up to basis and closes back below.
    """
    if i < 25:
        return False
    _, _, _, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    slope_down = basis[i] < basis[i-3]           # BB basis trending down
    bounce     = c[i-1] >= basis[i-1] * 0.999   # previous bar near/above basis
    rejection  = c[i] < basis[i]                 # current bar back below basis
    return bool(slope_down and bounce and rejection)


# ============================================================================
# GROUP 4 — Breakdown  (S14–S16)
# ============================================================================

def S14_Donchian_Break(price: pd.DataFrame, i: int) -> bool:
    """Price breaks BELOW the 20-bar Donchian low (previous 20-bar low)."""
    if i < 22:
        return False
    _, h, l, c = _arrays(price)
    dc_low = ind.donchian_low(l, 20)
    return bool(c[i] < dc_low[i] and c[i] < c[i-1])


def S15_Inside_Bar_Break(price: pd.DataFrame, i: int) -> bool:
    """
    Inside bar breakdown:
    bar[i-1] is inside bar[i-2], bar[i] breaks BELOW bar[i-2] low.
    """
    if i < 5:
        return False
    o, h, l, c = _arrays(price)
    mother_high = h[i-2]
    mother_low  = l[i-2]
    inside      = h[i-1] <= mother_high and l[i-1] >= mother_low
    breakdown   = c[i] < mother_low
    return bool(inside and breakdown)


def S16_ATR_Volatility_Break(price: pd.DataFrame, i: int) -> bool:
    """
    Bearish volatility expansion: price closes more than 1× ATR below prior close,
    with EMA21 trending down.
    """
    if i < 20:
        return False
    _, h, l, c = _arrays(price)
    atr_v = ind.atr(h, l, c, 14)
    move  = c[i-1] - c[i]          # positive when price drops
    e21   = ind.ema(c, 21)
    strong_move = move > atr_v[i] * 0.8
    trend_down  = e21[i] < e21[i-3]
    return bool(strong_move and trend_down and move > 0)


# ============================================================================
# GROUP 5 — Pattern / Confluence  (S17–S20)
# ============================================================================

def S17_AO_Saucer(price: pd.DataFrame, i: int) -> bool:
    """
    Bearish AO saucer: AO is NEGATIVE and forms a hill pattern
    (higher, higher, then lower) — momentum starting to turn back down.
    """
    if i < 40:
        return False
    _, h, l, _ = _arrays(price)
    ao = ind.awesome_oscillator(h, l)
    bearish_saucer = ao[i] < 0 and ao[i] < ao[i-1] and ao[i-1] > ao[i-2]
    return bool(bearish_saucer)


def S18_Shooting_Star(price: pd.DataFrame, i: int) -> bool:
    """
    Shooting star / hanging man candle:
    upper wick >= 2× body, lower wick <= body, bearish close,
    price is in upper 40% of 20-bar range.
    """
    if i < 22:
        return False
    o, h, l, c = _arrays(price)
    body       = abs(c[i] - o[i])
    upper_wick = h[i] - max(c[i], o[i])
    lower_wick = min(c[i], o[i]) - l[i]
    if body == 0:
        return False
    is_star    = (upper_wick >= 2 * body and lower_wick <= body and c[i] < o[i])
    range_low  = min(l[i-20:i])
    range_high = max(h[i-20:i])
    if range_high == range_low:
        return False
    in_upper   = (c[i] - range_low) / (range_high - range_low) > 0.55
    return bool(is_star and in_upper)


def S19_Bearish_Engulf(price: pd.DataFrame, i: int) -> bool:
    """
    Bearish engulfing candle:
    bar[i-1] is bullish, bar[i] is bearish and fully engulfs bar[i-1].
    Confirm RSI not oversold (> 40) and price in upper 60% of 20-bar range.
    """
    if i < 22:
        return False
    o, h, l, c = _arrays(price)
    prev_bullish = c[i-1] > o[i-1]
    curr_bearish = c[i]   < o[i]
    engulfs      = o[i] >= c[i-1] and c[i] <= o[i-1]
    r = ind.rsi(c, 14)
    range_low  = min(l[i-20:i])
    range_high = max(h[i-20:i])
    if range_high == range_low:
        return False
    in_upper = (c[i] - range_low) / (range_high - range_low) > 0.40
    return bool(prev_bullish and curr_bearish and engulfs and r[i] > 40 and in_upper)


def S20_Confluence(price: pd.DataFrame, i: int) -> bool:
    """
    Multi-factor short confluence: all three must align:
    1. BB: price touches upper band (mean-reversion short setup)
    2. RSI > 60 (overbought momentum)
    3. EMA21 slope negative (broader downtrend)
    High selectivity — only enters when multiple factors agree.
    """
    if i < 30:
        return False
    _, h, l, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    r   = ind.rsi(c, 14)
    e21 = ind.ema(c, 21)

    bb_touch   = h[i] >= upper[i] and c[i] < upper[i]
    overbought = r[i] > 60
    trend_down = e21[i] < e21[i-5]
    return bool(bb_touch and overbought and trend_down)


# ============================================================================
# Registry
# ============================================================================

STRATEGIES: dict[str, tuple] = {
    "S01_EMA_Cross":        (S01_EMA_Cross,         "Trend",      "Fast EMA(8) crosses below Slow EMA(21)"),
    "S02_Triple_EMA":       (S02_Triple_EMA,         "Trend",      "Bearish triple EMA alignment + rejection"),
    "S03_MACD_Signal":      (S03_MACD_Signal,        "Momentum",   "MACD histogram turns negative"),
    "S04_Supertrend":       (S04_Supertrend,          "Trend",      "Supertrend(10,3) flips bearish"),
    "S05_ADX_EMA_Reject":   (S05_ADX_EMA_Rejection,  "Trend",      "ADX>25 downtrend + EMA20 rejection"),
    "S06_RSI_Overbought":   (S06_RSI_Overbought,      "Oscillator", "RSI(14) crosses below 70 from overbought"),
    "S07_Stoch_Cross":      (S07_Stoch_Cross,         "Oscillator", "Stochastic K crosses below D above 75"),
    "S08_CCI_Overbought":   (S08_CCI_Overbought,      "Oscillator", "CCI(20) crosses below +100"),
    "S09_Williams_R":       (S09_Williams_R,          "Oscillator", "Williams %R(14) exits overbought zone"),
    "S10_RSI_Divergence":   (S10_RSI_Divergence,      "Oscillator", "Bearish RSI divergence proxy"),
    "S11_BB_Upper_Touch":   (S11_BB_Upper_Touch,      "BB",         "Upper band touch + bearish close recovery"),
    "S12_BB_Squeeze_Break": (S12_BB_Squeeze_Break,    "BB",         "BB squeeze followed by downward expansion"),
    "S13_BB_Basis_Reject":  (S13_BB_Basis_Rejection,  "BB",         "Rejection at BB basis in downtrend"),
    "S14_Donchian_Break":   (S14_Donchian_Break,      "Breakdown",  "20-bar Donchian channel breakdown"),
    "S15_Inside_Bar_Break": (S15_Inside_Bar_Break,    "Breakdown",  "Inside bar pattern breakdown to downside"),
    "S16_ATR_Vol_Break":    (S16_ATR_Volatility_Break,"Breakdown",  "ATR-sized move down + EMA21 declining"),
    "S17_AO_Saucer":        (S17_AO_Saucer,           "Pattern",    "Bearish AO saucer (negative AO)"),
    "S18_Shooting_Star":    (S18_Shooting_Star,        "Pattern",    "Shooting star / hanging man candle"),
    "S19_Bearish_Engulf":   (S19_Bearish_Engulf,       "Pattern",    "Bearish engulfing candle confirmation"),
    "S20_Confluence":       (S20_Confluence,           "Confluence", "BB upper touch + RSI>60 + EMA21 downtrend"),
}
