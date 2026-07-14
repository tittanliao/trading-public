import numpy as np
import pandas as pd


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]

    # EMAs
    for p in [9, 21, 55, 200]:
        df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # Bollinger Bands (20, 2)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    denom = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = (c - df["bb_lower"]) / denom
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    # ATR (14)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # Stochastic (14, 3)
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    k = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["stoch_k"] = k.rolling(3).mean()  # smoothed %K
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Donchian Channel (20) — use previous bar to avoid lookahead
    df["don_high20"] = h.shift(1).rolling(20).max()
    df["don_low20"] = l.shift(1).rolling(20).min()

    # Williams %R (14)
    df["willr"] = -100 * (high14 - c) / (high14 - low14 + 1e-9)

    # CCI (20)
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - tp_ma) / (0.015 * tp_md + 1e-9)

    # RSI slope (if RSI present in source CSV, else compute from close)
    if "rsi" not in df.columns:
        df["rsi"] = _rsi(c, 14)
        df["rsi_ma"] = df["rsi"].rolling(9).mean()
    df["rsi_slope3"] = df["rsi"].diff(3)

    # Volume ratio (skip if no volume column)
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    else:
        df["vol_ratio"] = pd.Series(1.0, index=df.index)

    # Supertrend (10, 3)
    df = _add_supertrend(df, period=10, mult=3.0)

    # Intraday VWAP (resets each calendar day, only if volume present)
    if "volume" in df.columns:
        df["vwap"] = _intraday_vwap(df)

    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _add_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    c = df["close"].values
    h = df["high"].values
    l_arr = df["low"].values
    atr = df["atr"].values
    n = len(df)

    hl2 = (h + l_arr) / 2
    basic_up = hl2 + mult * atr
    basic_lo = hl2 - mult * atr

    final_up = np.full(n, np.nan)
    final_lo = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)  # 1 = bullish, -1 = bearish

    for i in range(n):
        if np.isnan(atr[i]):
            final_up[i] = basic_up[i]
            final_lo[i] = basic_lo[i]
            continue

        if i == 0 or np.isnan(final_up[i - 1]):
            final_up[i] = basic_up[i]
            final_lo[i] = basic_lo[i]
            continue

        fu_prev = final_up[i - 1]
        fl_prev = final_lo[i - 1]
        final_up[i] = basic_up[i] if basic_up[i] < fu_prev or c[i - 1] > fu_prev else fu_prev
        final_lo[i] = basic_lo[i] if basic_lo[i] > fl_prev or c[i - 1] < fl_prev else fl_prev

        if trend[i - 1] == -1 and c[i] > final_up[i]:
            trend[i] = 1
        elif trend[i - 1] == 1 and c[i] < final_lo[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    df["supertrend"] = trend
    return df


def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["_date"] = df.index.date
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpvol"] = tp * df["volume"]
    df["_cumtpvol"] = df.groupby("_date")["_tpvol"].cumsum()
    df["_cumvol"] = df.groupby("_date")["volume"].cumsum()
    return df["_cumtpvol"] / df["_cumvol"].replace(0, np.nan)
