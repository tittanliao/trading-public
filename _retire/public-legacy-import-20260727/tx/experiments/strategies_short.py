"""
20 Short Strategies for MTX (S01–S20).
Mirror logic of long strategies but inverted direction.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def s01_rsi_overbought_drop(df: pd.DataFrame) -> pd.Series:
    """RSI crosses down through 70 (overbought reversal)"""
    r = df["rsi"]
    return (r.shift(1) >= 70) & (r < 70) & (r > 45)


def s02_macd_crossover_above_zero(df: pd.DataFrame) -> pd.Series:
    """MACD line crosses below signal while both > 0"""
    m, s = df["macd"], df["macd_sig"]
    cross = (m.shift(1) > s.shift(1)) & (m <= s)
    return cross & (m > 0)


def s03_macd_hist_reversal_short(df: pd.DataFrame) -> pd.Series:
    """MACD histogram turns down for 2 bars from above 0"""
    h = df["macd_hist"]
    return (h.shift(2) > h.shift(1)) & (h.shift(1) > h) & (h.shift(2) > 0)


def s04_stoch_overbought_cross(df: pd.DataFrame) -> pd.Series:
    """Stoch K crosses below D while both > 75"""
    k, d = df["stoch_k"], df["stoch_d"]
    cross = (k.shift(1) > d.shift(1)) & (k <= d)
    return cross & (k > 75)


def s05_williams_r_overbought(df: pd.DataFrame) -> pd.Series:
    """Williams %R crosses from above -20 to below -20"""
    w = df["willr"]
    return (w.shift(1) >= -20) & (w < -20)


def s06_cci_overbought_drop(df: pd.DataFrame) -> pd.Series:
    """CCI crosses below +100 from above"""
    cci = df["cci"]
    return (cci.shift(1) >= 100) & (cci < 100)


def s07_rsi_50_crossdown(df: pd.DataFrame) -> pd.Series:
    """RSI crosses below 50 in downtrend (EMA21 < EMA55)"""
    r = df["rsi"]
    return (r.shift(1) >= 50) & (r < 50) & (df["ema21"] < df["ema55"])


def s08_ema9_21_death_cross(df: pd.DataFrame) -> pd.Series:
    """EMA9 crosses below EMA21 while price < EMA55"""
    e9, e21 = df["ema9"], df["ema21"]
    cross = (e9.shift(1) > e21.shift(1)) & (e9 <= e21)
    return cross & (df["close"] < df["ema55"])


def s09_ema55_resistance(df: pd.DataFrame) -> pd.Series:
    """Price rallies to EMA55 zone in downtrend (EMA21 < EMA55)"""
    c = df["close"]
    atr = df["atr"]
    near_ema55 = (c - df["ema55"]).abs() < atr * 0.4
    downtrend = df["ema21"] < df["ema55"]
    rsi_ok = (df["rsi"] >= 35) & (df["rsi"] <= 60)
    return near_ema55 & downtrend & rsi_ok


def s10_supertrend_sell(df: pd.DataFrame) -> pd.Series:
    """Supertrend flips to bearish"""
    st = df["supertrend"]
    return (st.shift(1) == 1) & (st == -1)


def s11_donchian_20l_breakdown(df: pd.DataFrame) -> pd.Series:
    """Close breaks below 20-bar Donchian low"""
    return (df["close"] < df["don_low20"]) & (df["rsi"] < 50)


def s12_bb_squeeze_short(df: pd.DataFrame) -> pd.Series:
    """BB squeeze release: width below 20th pct, then close breaks below lower"""
    width_threshold = df["bb_width"].rolling(100).quantile(0.20)
    was_squeezed = df["bb_width"].shift(1) <= width_threshold.shift(1)
    return was_squeezed & (df["close"] < df["bb_lower"])


def s13_atr_momentum_breakdown(df: pd.DataFrame) -> pd.Series:
    """Close < 20-bar low - 0.3×ATR"""
    c = df["close"]
    low20 = c.rolling(20).min().shift(1)
    return (c < low20 - df["atr"] * 0.3) & (df["rsi"] < 50)


def s14_bb_upper_rejection(df: pd.DataFrame) -> pd.Series:
    """Close touches BB upper band and RSI is overbought-ish"""
    return (df["bb_pct"] >= 0.95) & (df["rsi"] > 55) & (df["rsi_slope3"] < 0)


def s15_bb_pct_high_rsi_combo(df: pd.DataFrame) -> pd.Series:
    """BB %B > 0.80 and RSI crosses below 70"""
    r = df["rsi"]
    return (df["bb_pct"] > 0.80) & (r.shift(1) >= 70) & (r < 70)


def s16_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting star: upper wick > 2× body, small lower wick, RSI > 45"""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    upper_wick = h - o.where(c >= o, c)  # h - max(o,c)
    lower_wick = o.where(c >= o, c) - l  # min(o,c) - l
    is_star = (upper_wick > body * 2) & (lower_wick < body * 0.5) & (body > 0)
    return is_star & (df["rsi"] > 45)


def s17_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current red bar engulfs previous green bar"""
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_green = prev_c > prev_o
    curr_red = c < o
    engulf = (o >= prev_c) & (c <= prev_o)
    return prev_green & curr_red & engulf & (df["rsi"] > 40)


def s18_rsi_bearish_divergence(df: pd.DataFrame) -> pd.Series:
    """TradingView-exported RSI bearish divergence signal"""
    if "bear_div" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["bear_div"].fillna(0).astype(bool)


def s19_day_open_range_short(df: pd.DataFrame) -> pd.Series:
    """Day session: break below the 08:45 opening bar low"""
    c = df["close"]
    is_day = df["session"] == "day"
    is_0845 = df.index.hour == 8
    first_bar_low = df["low"].where(is_day & is_0845).groupby(df.index.date).transform("first")
    first_bar_low = first_bar_low.ffill()
    return is_day & (c < first_bar_low) & (df["rsi"] < 50) & ~(is_day & is_0845)


def s20_night_session_short(df: pd.DataFrame) -> pd.Series:
    """Night session: RSI slope negative + price < EMA21 + MACD histogram negative"""
    is_night = df["session"] == "night"
    momentum = (df["rsi_slope3"] < -2) & (df["close"] < df["ema21"]) & (df["macd_hist"] < 0)
    return is_night & momentum & (df["rsi"] > 35) & (df["rsi"] < 55)


SHORT_STRATEGIES: dict[str, tuple] = {
    "S01": (s01_rsi_overbought_drop, "RSI Overbought Drop", "Oscillator"),
    "S02": (s02_macd_crossover_above_zero, "MACD Cross Above Zero", "Momentum"),
    "S03": (s03_macd_hist_reversal_short, "MACD Hist Reversal Short", "Momentum"),
    "S04": (s04_stoch_overbought_cross, "Stoch Overbought Cross", "Oscillator"),
    "S05": (s05_williams_r_overbought, "Williams %R Overbought", "Oscillator"),
    "S06": (s06_cci_overbought_drop, "CCI Overbought Drop", "Oscillator"),
    "S07": (s07_rsi_50_crossdown, "RSI 50 Crossdown", "Momentum"),
    "S08": (s08_ema9_21_death_cross, "EMA 9/21 Death Cross", "Trend"),
    "S09": (s09_ema55_resistance, "EMA55 Resistance", "Trend"),
    "S10": (s10_supertrend_sell, "Supertrend Sell", "Trend"),
    "S11": (s11_donchian_20l_breakdown, "Donchian 20L Break", "Breakout"),
    "S12": (s12_bb_squeeze_short, "BB Squeeze Short", "Breakout"),
    "S13": (s13_atr_momentum_breakdown, "ATR Momentum Break Short", "Breakout"),
    "S14": (s14_bb_upper_rejection, "BB Upper Rejection", "MeanReversion"),
    "S15": (s15_bb_pct_high_rsi_combo, "BB High + RSI Combo", "MeanReversion"),
    "S16": (s16_shooting_star, "Shooting Star", "Pattern"),
    "S17": (s17_bearish_engulfing, "Bearish Engulfing", "Pattern"),
    "S18": (s18_rsi_bearish_divergence, "RSI Bearish Divergence", "Pattern"),
    "S19": (s19_day_open_range_short, "Day Open Range Short", "Session"),
    "S20": (s20_night_session_short, "Night Momentum Short", "Session"),
}
