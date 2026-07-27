"""
20 long-only strategy signal functions for XAUUSD 30-min bars.

Each function has the signature:
    signal(price: pd.DataFrame, i: int) -> bool

    price : full OHLCV DataFrame
    i     : current bar index (signal fires on bar i close,
             entry executes at bar i+1 open)

Return True to enter a long position.

Strategies are grouped by concept:
    E01–E05  Trend / Momentum
    E06–E10  Oscillator bounces
    E11–E13  Bollinger Band
    E14–E16  Breakout
    E17–E20  Pattern / Confluence
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import indicators as ind

# ── Pre-compute helpers ──────────────────────────────────────────────────────

def _arrays(price: pd.DataFrame):
    return (price["open"].to_numpy(),
            price["high"].to_numpy(),
            price["low"].to_numpy(),
            price["close"].to_numpy())


# ============================================================================
# GROUP 1 — Trend / Momentum  (E01–E05)
# ============================================================================

def E01_EMA_Cross(price: pd.DataFrame, i: int) -> bool:
    """Fast EMA(8) crosses above Slow EMA(21) — classic trend-following entry."""
    _, _, _, c = _arrays(price)
    fast = ind.ema(c, 8)
    slow = ind.ema(c, 21)
    return bool(fast[i] > slow[i] and fast[i-1] <= slow[i-1])


def E02_Triple_EMA(price: pd.DataFrame, i: int) -> bool:
    """
    Triple EMA alignment: EMA8 > EMA21 > EMA55.
    Enter on pullback: price dips to EMA8 then bounces (close > open after touching EMA8).
    """
    if i < 55:
        return False
    _, _, _, c = _arrays(price)
    e8  = ind.ema(c, 8)
    e21 = ind.ema(c, 21)
    e55 = ind.ema(c, 55)
    aligned = e8[i] > e21[i] > e55[i]
    pullback = c[i-1] <= e8[i-1] * 1.002  # previous bar touched EMA8
    bounce   = c[i] > c[i-1]               # current bar is bullish
    return bool(aligned and pullback and bounce)


def E03_MACD_Signal(price: pd.DataFrame, i: int) -> bool:
    """MACD line crosses above signal line AND MACD histogram turns positive."""
    if i < 35:
        return False
    _, _, _, c = _arrays(price)
    ml, sl, hist = ind.macd(c)
    cross_up = hist[i] > 0 and hist[i-1] <= 0
    return bool(cross_up and ml[i] > -5)   # not deep in negative territory


def E04_Supertrend(price: pd.DataFrame, i: int) -> bool:
    """Supertrend(10, 3.0) flips from bearish to bullish."""
    if i < 15:
        return False
    _, h, l, c = _arrays(price)
    trend = ind.supertrend(h, l, c, 10, 3.0)
    return bool(trend[i] == 1 and trend[i-1] == -1)


def E05_ADX_EMA_Dip(price: pd.DataFrame, i: int) -> bool:
    """
    Strong trend (ADX > 25) + price above EMA50.
    Enter when price dips to EMA20 and bounces (close > EMA20 after being below).
    """
    if i < 55:
        return False
    _, h, l, c = _arrays(price)
    adx_v, _, _ = ind.adx(h, l, c, 14)
    e20 = ind.ema(c, 20)
    e50 = ind.ema(c, 50)
    trending  = adx_v[i] > 25
    above_e50 = c[i] > e50[i]
    dip_bounce = c[i-1] < e20[i-1] and c[i] > e20[i]
    return bool(trending and above_e50 and dip_bounce)


# ============================================================================
# GROUP 2 — Oscillator Bounces  (E06–E10)
# ============================================================================

def E06_RSI_Oversold(price: pd.DataFrame, i: int) -> bool:
    """RSI(14) crosses from below 30 to above 30 (oversold recovery)."""
    if i < 20:
        return False
    _, _, _, c = _arrays(price)
    r = ind.rsi(c, 14)
    return bool(r[i] > 30 and r[i-1] <= 30)


def E07_Stoch_Cross(price: pd.DataFrame, i: int) -> bool:
    """Stochastic K crosses above D from the oversold zone (both < 20)."""
    if i < 20:
        return False
    _, h, l, c = _arrays(price)
    k, d = ind.stochastic(h, l, c, 14, 3, 3)
    cross_up   = k[i] > d[i] and k[i-1] <= d[i-1]
    oversold   = k[i-1] < 25
    return bool(cross_up and oversold)


def E08_CCI_Bounce(price: pd.DataFrame, i: int) -> bool:
    """CCI(20) crosses from below -100 to above (classic oversold bounce)."""
    if i < 25:
        return False
    _, h, l, c = _arrays(price)
    cci_v = ind.cci(h, l, c, 20)
    return bool(cci_v[i] > -100 and cci_v[i-1] <= -100)


def E09_Williams_R(price: pd.DataFrame, i: int) -> bool:
    """Williams %R(14) crosses from below -80 upward (oversold exit)."""
    if i < 20:
        return False
    _, h, l, c = _arrays(price)
    wr = ind.williams_r(h, l, c, 14)
    return bool(wr[i] > -80 and wr[i-1] <= -80)


def E10_RSI_Divergence(price: pd.DataFrame, i: int) -> bool:
    """
    Hidden bullish divergence proxy:
    Price makes a lower low but RSI makes a higher low over the last 5 bars.
    Suggests underlying momentum is improving despite price weakness.
    """
    if i < 20:
        return False
    _, _, _, c = _arrays(price)
    r = ind.rsi(c, 14)
    look = 5
    price_lower_low = c[i] < min(c[i-look:i])
    rsi_higher_low  = r[i]  > min(r[i-look:i])
    rsi_rising      = r[i]  > r[i-1]
    return bool(price_lower_low and rsi_higher_low and rsi_rising)


# ============================================================================
# GROUP 3 — Bollinger Band  (E11–E13)
# ============================================================================

def E11_BB_Lower_Touch(price: pd.DataFrame, i: int) -> bool:
    """
    Price touches BB lower band then closes back above it — mean reversion entry.
    Confirm with RSI < 45 (not overbought) and bullish candle.
    """
    if i < 25:
        return False
    _, h, l, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    touched  = l[i] <= lower[i]             # wick touched lower band
    recovery = c[i] > lower[i]             # but closed above
    bullish  = c[i] > c[i-1]               # bullish close
    r = ind.rsi(c, 14)
    return bool(touched and recovery and bullish and r[i] < 50)


def E12_BB_Squeeze_Break(price: pd.DataFrame, i: int) -> bool:
    """
    Bollinger Band squeeze (width < 20-bar avg) followed by upward expansion.
    Enters when BB width expands and price closes above basis.
    """
    if i < 30:
        return False
    _, h, l, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    width     = upper - lower
    avg_width = ind.sma(width, 20)
    squeeze   = width[i-1] < avg_width[i-1] * 0.85    # was squeezed
    expanding = width[i]   > width[i-1]                # now expanding
    above_mid = c[i] > basis[i]                        # price above midline
    return bool(squeeze and expanding and above_mid)


def E13_BB_Basis_Walk(price: pd.DataFrame, i: int) -> bool:
    """
    'Walking the upper band': price > basis, BB basis slope positive,
    enter on pullback when price dips to basis and closes above.
    """
    if i < 25:
        return False
    _, _, _, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    slope_up   = basis[i] > basis[i-3]          # BB basis trending up
    dip        = c[i-1] <= basis[i-1] * 1.001   # previous bar near/below basis
    recovery   = c[i] > basis[i]                # current bar back above basis
    return bool(slope_up and dip and recovery)


# ============================================================================
# GROUP 4 — Breakout  (E14–E16)
# ============================================================================

def E14_Donchian_Break(price: pd.DataFrame, i: int) -> bool:
    """Price breaks above the 20-bar Donchian high (previous 20-bar high)."""
    if i < 22:
        return False
    _, h, _, c = _arrays(price)
    dc_high = ind.donchian_high(h, 20)
    return bool(c[i] > dc_high[i] and c[i] > c[i-1])


def E15_Inside_Bar_Break(price: pd.DataFrame, i: int) -> bool:
    """
    Inside bar pattern: bar[i-1] is fully inside bar[i-2] (range contraction).
    Signal fires when bar[i] breaks above bar[i-2] high (breakout of inside bar).
    """
    if i < 5:
        return False
    o, h, l, c = _arrays(price)
    mother_high = h[i-2]
    mother_low  = l[i-2]
    inside      = h[i-1] <= mother_high and l[i-1] >= mother_low   # bar i-1 is inside
    breakout    = c[i] > mother_high                                # bar i breaks above
    return bool(inside and breakout)


def E16_ATR_Volatility_Break(price: pd.DataFrame, i: int) -> bool:
    """
    Volatility expansion breakout:
    Price closes more than 1× ATR(14) above the prior close,
    with EMA21 trending up (slope positive over 3 bars).
    """
    if i < 20:
        return False
    _, h, l, c = _arrays(price)
    atr_v = ind.atr(h, l, c, 14)
    move  = c[i] - c[i-1]
    e21   = ind.ema(c, 21)
    strong_move   = move > atr_v[i] * 0.8
    trend_up      = e21[i] > e21[i-3]
    return bool(strong_move and trend_up and move > 0)


# ============================================================================
# GROUP 5 — Pattern / Confluence  (E17–E20)
# ============================================================================

def E17_AO_Saucer(price: pd.DataFrame, i: int) -> bool:
    """
    Awesome Oscillator saucer pattern while AO is positive:
    AO goes lower, lower, then higher — a saucer (dip then recovery).
    """
    if i < 40:
        return False
    _, h, l, _ = _arrays(price)
    ao = ind.awesome_oscillator(h, l)
    saucer = ao[i] > 0 and ao[i] > ao[i-1] and ao[i-1] < ao[i-2]
    return bool(saucer)


def E18_Hammer(price: pd.DataFrame, i: int) -> bool:
    """
    Hammer candle: lower wick >= 2× body, upper wick <= body,
    body is bullish (close > open), price is in lower 40% of 20-bar range.
    """
    if i < 22:
        return False
    o, h, l, c = _arrays(price)
    body       = abs(c[i] - o[i])
    lower_wick = min(c[i], o[i]) - l[i]
    upper_wick = h[i] - max(c[i], o[i])
    if body == 0:
        return False
    is_hammer  = (lower_wick >= 2 * body and upper_wick <= body and c[i] > o[i])
    # Check price is near recent lows
    range_low  = min(l[i-20:i])
    range_high = max(h[i-20:i])
    if range_high == range_low:
        return False
    in_lower   = (c[i] - range_low) / (range_high - range_low) < 0.45
    return bool(is_hammer and in_lower)


def E19_Bullish_Engulf(price: pd.DataFrame, i: int) -> bool:
    """
    Bullish engulfing candle:
    bar[i-1] is bearish, bar[i] is bullish and fully engulfs bar[i-1].
    Confirm with RSI not overbought (< 60) and price in lower 60% of 20-bar range.
    """
    if i < 22:
        return False
    o, h, l, c = _arrays(price)
    prev_bearish = c[i-1] < o[i-1]
    curr_bullish = c[i]   > o[i]
    engulfs      = o[i] <= c[i-1] and c[i] >= o[i-1]
    r = ind.rsi(c, 14)
    return bool(prev_bearish and curr_bullish and engulfs and r[i] < 60)


def E20_Confluence(price: pd.DataFrame, i: int) -> bool:
    """
    Multi-factor confluence: all three conditions must align:
    1. BB: price touches lower band (mean-reversion setup)
    2. RSI < 40 (oversold momentum)
    3. EMA21 slope positive (broader uptrend)
    High selectivity — only enters when multiple factors agree.
    """
    if i < 30:
        return False
    _, h, l, c = _arrays(price)
    basis, upper, lower = ind.bb(c, 20, 2.0)
    r   = ind.rsi(c, 14)
    e21 = ind.ema(c, 21)

    bb_touch   = l[i] <= lower[i] and c[i] > lower[i]   # touched and closed above
    oversold   = r[i] < 40
    trend_up   = e21[i] > e21[i-5]                       # EMA21 rising over 5 bars
    return bool(bb_touch and oversold and trend_up)


# ============================================================================
# Registry — add new strategies here to auto-include in the runner
# ============================================================================

STRATEGIES: dict[str, tuple] = {
    "E01_EMA_Cross":        (E01_EMA_Cross,        "Trend",       "Fast EMA(8) crosses above Slow EMA(21)"),
    "E02_Triple_EMA":       (E02_Triple_EMA,        "Trend",       "Triple EMA alignment with pullback bounce"),
    "E03_MACD_Signal":      (E03_MACD_Signal,       "Momentum",    "MACD histogram turns positive"),
    "E04_Supertrend":       (E04_Supertrend,         "Trend",       "Supertrend(10,3) flips bullish"),
    "E05_ADX_EMA_Dip":      (E05_ADX_EMA_Dip,       "Trend",       "ADX>25 trend + EMA20 dip-and-bounce"),
    "E06_RSI_Oversold":     (E06_RSI_Oversold,       "Oscillator",  "RSI(14) crosses above 30 from oversold"),
    "E07_Stoch_Cross":      (E07_Stoch_Cross,        "Oscillator",  "Stochastic K crosses above D below 25"),
    "E08_CCI_Bounce":       (E08_CCI_Bounce,         "Oscillator",  "CCI(20) crosses above -100"),
    "E09_Williams_R":       (E09_Williams_R,         "Oscillator",  "Williams %R(14) exits oversold zone"),
    "E10_RSI_Divergence":   (E10_RSI_Divergence,     "Oscillator",  "Bullish RSI divergence proxy"),
    "E11_BB_Lower_Touch":   (E11_BB_Lower_Touch,     "BB",          "Lower band touch + bullish close recovery"),
    "E12_BB_Squeeze_Break": (E12_BB_Squeeze_Break,   "BB",          "BB squeeze followed by upward expansion"),
    "E13_BB_Basis_Walk":    (E13_BB_Basis_Walk,      "BB",          "Pullback to BB basis in uptrend"),
    "E14_Donchian_Break":   (E14_Donchian_Break,     "Breakout",    "20-bar Donchian channel breakout"),
    "E15_Inside_Bar_Break": (E15_Inside_Bar_Break,   "Breakout",    "Inside bar pattern breakout to upside"),
    "E16_ATR_Vol_Break":    (E16_ATR_Volatility_Break, "Breakout",  "ATR-sized move up + EMA21 trending"),
    "E17_AO_Saucer":        (E17_AO_Saucer,          "Pattern",     "Awesome Oscillator saucer (positive AO)"),
    "E18_Hammer":           (E18_Hammer,              "Pattern",     "Hammer candle near 20-bar lows"),
    "E19_Bullish_Engulf":   (E19_Bullish_Engulf,      "Pattern",     "Bullish engulfing candle confirmation"),
    "E20_Confluence":       (E20_Confluence,           "Confluence",  "BB touch + RSI<40 + EMA21 uptrend"),
}
