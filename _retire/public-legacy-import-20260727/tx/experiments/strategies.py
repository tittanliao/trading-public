"""
20 Long Strategies for MTX (E01–E20).
Each function receives a DataFrame with indicators pre-calculated and
returns a boolean pd.Series (True = enter long on next bar open).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# Momentum / Oscillator
# ──────────────────────────────────────────────

def e01_rsi_oversold_bounce(df: pd.DataFrame) -> pd.Series:
    """RSI crosses up through 30 (oversold bounce)"""
    r = df["rsi"]
    return (r.shift(1) < 30) & (r >= 30) & (r < 55)


def e02_macd_crossover_below_zero(df: pd.DataFrame) -> pd.Series:
    """MACD line crosses above signal line while both < 0"""
    m, s = df["macd"], df["macd_sig"]
    cross = (m.shift(1) < s.shift(1)) & (m >= s)
    return cross & (m < 0)


def e03_macd_hist_reversal(df: pd.DataFrame) -> pd.Series:
    """MACD histogram turns up for 2 consecutive bars from below 0"""
    h = df["macd_hist"]
    return (h.shift(2) < h.shift(1)) & (h.shift(1) < h) & (h.shift(2) < 0)


def e04_stoch_oversold_cross(df: pd.DataFrame) -> pd.Series:
    """Stoch K crosses above D while both < 25"""
    k, d = df["stoch_k"], df["stoch_d"]
    cross = (k.shift(1) < d.shift(1)) & (k >= d)
    return cross & (k < 25)


def e05_williams_r_oversold(df: pd.DataFrame) -> pd.Series:
    """Williams %R crosses from below -80 to above -80"""
    w = df["willr"]
    return (w.shift(1) < -80) & (w >= -80)


def e06_cci_oversold_bounce(df: pd.DataFrame) -> pd.Series:
    """CCI crosses above -100 from below"""
    cci = df["cci"]
    return (cci.shift(1) < -100) & (cci >= -100)


def e07_rsi_50_crossover(df: pd.DataFrame) -> pd.Series:
    """RSI crosses above 50 (trend confirmation)"""
    r = df["rsi"]
    return (r.shift(1) < 50) & (r >= 50) & (df["ema21"] > df["ema55"])


# ──────────────────────────────────────────────
# Trend / EMA
# ──────────────────────────────────────────────

def e08_ema9_21_golden_cross(df: pd.DataFrame) -> pd.Series:
    """EMA9 crosses above EMA21 while price > EMA55"""
    e9, e21 = df["ema9"], df["ema21"]
    cross = (e9.shift(1) < e21.shift(1)) & (e9 >= e21)
    return cross & (df["close"] > df["ema55"])


def e09_ema55_pullback(df: pd.DataFrame) -> pd.Series:
    """Price pulls back to EMA55 zone in uptrend (EMA21 > EMA55)"""
    c = df["close"]
    atr = df["atr"]
    near_ema55 = (c - df["ema55"]).abs() < atr * 0.4
    uptrend = df["ema21"] > df["ema55"]
    rsi_ok = (df["rsi"] >= 40) & (df["rsi"] <= 65)
    return near_ema55 & uptrend & rsi_ok


def e10_supertrend_buy(df: pd.DataFrame) -> pd.Series:
    """Supertrend flips to bullish"""
    st = df["supertrend"]
    return (st.shift(1) == -1) & (st == 1)


# ──────────────────────────────────────────────
# Breakout
# ──────────────────────────────────────────────

def e11_donchian_20h_breakout(df: pd.DataFrame) -> pd.Series:
    """Close breaks above 20-bar Donchian high"""
    return (df["close"] > df["don_high20"]) & (df["rsi"] > 50)


def e12_bb_squeeze_breakout(df: pd.DataFrame) -> pd.Series:
    """BB squeeze release: width below 20th pct, then close breaks above upper"""
    width_threshold = df["bb_width"].rolling(100).quantile(0.20)
    was_squeezed = df["bb_width"].shift(1) <= width_threshold.shift(1)
    return was_squeezed & (df["close"] > df["bb_upper"])


def e13_atr_momentum_breakout(df: pd.DataFrame) -> pd.Series:
    """Close > 20-bar high + 0.3×ATR (momentum confirmed)"""
    c = df["close"]
    high20 = c.rolling(20).max().shift(1)
    return (c > high20 + df["atr"] * 0.3) & (df["rsi"] > 50)


# ──────────────────────────────────────────────
# Mean Reversion / BB
# ──────────────────────────────────────────────

def e14_bb_lower_bounce(df: pd.DataFrame) -> pd.Series:
    """Close touches BB lower band and RSI is oversold-ish"""
    return (df["bb_pct"] <= 0.05) & (df["rsi"] < 45) & (df["rsi_slope3"] > 0)


def e15_bb_pct_low_rsi_combo(df: pd.DataFrame) -> pd.Series:
    """BB %B < 0.2 and RSI crosses above 30"""
    r = df["rsi"]
    return (df["bb_pct"] < 0.20) & (r.shift(1) < 30) & (r >= 30)


# ──────────────────────────────────────────────
# Price Action / Pattern
# ──────────────────────────────────────────────

def e16_hammer_candle(df: pd.DataFrame) -> pd.Series:
    """Hammer: lower wick > 2× body, small upper wick, RSI < 55"""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    lower_wick = o.where(c >= o, c) - l  # min(o,c) - l
    upper_wick = h - o.where(c >= o, c)  # h - max(o,c)
    is_hammer = (lower_wick > body * 2) & (upper_wick < body * 0.5) & (body > 0)
    return is_hammer & (df["rsi"] < 55)


def e17_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current green bar engulfs previous red bar"""
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_red = prev_c < prev_o
    curr_green = c > o
    engulf = (o <= prev_c) & (c >= prev_o)
    return prev_red & curr_green & engulf & (df["rsi"] < 60)


def e18_rsi_divergence(df: pd.DataFrame) -> pd.Series:
    """Use TradingView-exported RSI bullish divergence signal (if present)"""
    if "bull_div" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["bull_div"].fillna(0).astype(bool)


# ──────────────────────────────────────────────
# Session-Specific
# ──────────────────────────────────────────────

def e19_day_open_range_break(df: pd.DataFrame) -> pd.Series:
    """Day session: break above the 08:45 opening bar high (first bar of day session)"""
    c = df["close"]
    is_day = df["session"] == "day"
    is_0845 = df.index.hour == 8
    first_bar_high = df["high"].where(is_day & is_0845).groupby(df.index.date).transform("first")
    first_bar_high = first_bar_high.ffill()
    return is_day & (c > first_bar_high) & (df["rsi"] > 50) & ~(is_day & is_0845)


def e20_night_session_momentum(df: pd.DataFrame) -> pd.Series:
    """Night session: RSI slope positive + price > EMA21 + MACD histogram positive"""
    is_night = df["session"] == "night"
    momentum = (df["rsi_slope3"] > 2) & (df["close"] > df["ema21"]) & (df["macd_hist"] > 0)
    return is_night & momentum & (df["rsi"] > 45) & (df["rsi"] < 65)


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

LONG_STRATEGIES: dict[str, tuple] = {
    "E01": (e01_rsi_oversold_bounce, "RSI Oversold Bounce", "Oscillator"),
    "E02": (e02_macd_crossover_below_zero, "MACD Cross Below Zero", "Momentum"),
    "E03": (e03_macd_hist_reversal, "MACD Hist Reversal", "Momentum"),
    "E04": (e04_stoch_oversold_cross, "Stoch Oversold Cross", "Oscillator"),
    "E05": (e05_williams_r_oversold, "Williams %R Oversold", "Oscillator"),
    "E06": (e06_cci_oversold_bounce, "CCI Oversold Bounce", "Oscillator"),
    "E07": (e07_rsi_50_crossover, "RSI 50 Crossover", "Momentum"),
    "E08": (e08_ema9_21_golden_cross, "EMA 9/21 Golden Cross", "Trend"),
    "E09": (e09_ema55_pullback, "EMA55 Pullback", "Trend"),
    "E10": (e10_supertrend_buy, "Supertrend Buy", "Trend"),
    "E11": (e11_donchian_20h_breakout, "Donchian 20H Break", "Breakout"),
    "E12": (e12_bb_squeeze_breakout, "BB Squeeze Break", "Breakout"),
    "E13": (e13_atr_momentum_breakout, "ATR Momentum Break", "Breakout"),
    "E14": (e14_bb_lower_bounce, "BB Lower Bounce", "MeanReversion"),
    "E15": (e15_bb_pct_low_rsi_combo, "BB Low + RSI Combo", "MeanReversion"),
    "E16": (e16_hammer_candle, "Hammer Candle", "Pattern"),
    "E17": (e17_bullish_engulfing, "Bullish Engulfing", "Pattern"),
    "E18": (e18_rsi_divergence, "RSI Divergence", "Pattern"),
    "E19": (e19_day_open_range_break, "Day Open Range Break", "Session"),
    "E20": (e20_night_session_momentum, "Night Momentum", "Session"),
}
