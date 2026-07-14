"""
Technical indicator calculations on a price DataFrame.

All functions return numpy arrays or pandas Series aligned to the input index.
No look-ahead: each value at index i uses only data up to and including bar i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series | np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(series)
    return s.ewm(span=period, adjust=False).mean().to_numpy()


def sma(series: pd.Series | np.ndarray, period: int) -> np.ndarray:
    return pd.Series(series).rolling(period).mean().to_numpy()


def stdev(series: pd.Series | np.ndarray, period: int) -> np.ndarray:
    return pd.Series(series).rolling(period).std(ddof=0).to_numpy()


def bb(close: np.ndarray, period: int = 20, mult: float = 2.0
       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (basis, upper, lower)."""
    basis = sma(close, period)
    dev   = stdev(close, period) * mult
    return basis, basis + dev, basis - dev


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(com=period - 1, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(com=period - 1, adjust=False).mean().to_numpy()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    return 100.0 - 100.0 / (1.0 + rs)


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (macd_line, signal_line, histogram)."""
    line   = ema(close, fast) - ema(close, slow)
    sig    = ema(line, signal)
    return line, sig, line - sig


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(com=period - 1, adjust=False).mean().to_numpy()


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               k_period: int = 14, smooth_k: int = 3, smooth_d: int = 3
               ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (K, D)."""
    h_max = pd.Series(high).rolling(k_period).max().to_numpy()
    l_min = pd.Series(low).rolling(k_period).min().to_numpy()
    raw_k = np.where(h_max - l_min == 0, 50.0, (close - l_min) / (h_max - l_min) * 100)
    k = sma(raw_k, smooth_k)
    d = sma(k, smooth_d)
    return k, d


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 20) -> np.ndarray:
    tp = (high + low + close) / 3.0
    tp_s  = pd.Series(tp)
    mean  = tp_s.rolling(period).mean().to_numpy()
    mad   = tp_s.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).to_numpy()
    return np.where(mad == 0, 0.0, (tp - mean) / (0.015 * mad))


def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    h_max = pd.Series(high).rolling(period).max().to_numpy()
    l_min = pd.Series(low).rolling(period).min().to_numpy()
    return np.where(h_max - l_min == 0, -50.0,
                    (h_max - close) / (h_max - l_min) * -100.0)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ADX, +DI, -DI)."""
    prev_high  = np.roll(high, 1);  prev_high[0]  = high[0]
    prev_low   = np.roll(low,  1);  prev_low[0]   = low[0]
    prev_close = np.roll(close, 1); prev_close[0] = close[0]

    up_move   = high - prev_high
    down_move = prev_low - low
    dm_plus   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr_s    = pd.Series(tr).ewm(com=period - 1, adjust=False).mean().to_numpy()
    di_plus  = pd.Series(dm_plus).ewm(com=period - 1, adjust=False).mean().to_numpy() / atr_s * 100
    di_minus = pd.Series(dm_minus).ewm(com=period - 1, adjust=False).mean().to_numpy() / atr_s * 100

    dx  = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10) * 100
    adx_ = pd.Series(dx).ewm(com=period - 1, adjust=False).mean().to_numpy()
    return adx_, di_plus, di_minus


def awesome_oscillator(high: np.ndarray, low: np.ndarray,
                        fast: int = 5, slow: int = 34) -> np.ndarray:
    hl2 = (high + low) / 2.0
    return sma(hl2, fast) - sma(hl2, slow)


def donchian_high(high: np.ndarray, period: int = 20) -> np.ndarray:
    return pd.Series(high).shift(1).rolling(period).max().to_numpy()


def donchian_low(low: np.ndarray, period: int = 20) -> np.ndarray:
    return pd.Series(low).shift(1).rolling(period).min().to_numpy()


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        volume: np.ndarray | None, period: int = 14) -> np.ndarray:
    """Money Flow Index. If volume unavailable returns RSI as proxy."""
    if volume is None:
        return rsi(close, period)
    tp   = (high + low + close) / 3.0
    rmf  = tp * volume
    prev_tp = np.roll(tp, 1); prev_tp[0] = tp[0]
    pos  = np.where(tp > prev_tp, rmf, 0.0)
    neg  = np.where(tp < prev_tp, rmf, 0.0)
    pmf  = pd.Series(pos).rolling(period).sum().to_numpy()
    nmf  = pd.Series(neg).rolling(period).sum().to_numpy()
    return 100.0 - 100.0 / (1.0 + np.where(nmf == 0, np.inf, pmf / nmf))


def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 10, mult: float = 3.0) -> np.ndarray:
    """Returns +1 (bullish) or -1 (bearish) trend direction per bar."""
    atr_v = atr(high, low, close, period)
    hl2   = (high + low) / 2.0
    upper = hl2 + mult * atr_v
    lower = hl2 - mult * atr_v

    trend = np.ones(len(close))
    final_ub = upper.copy()
    final_lb = lower.copy()

    for i in range(1, len(close)):
        final_ub[i] = upper[i] if upper[i] < final_ub[i-1] or close[i-1] > final_ub[i-1] else final_ub[i-1]
        final_lb[i] = lower[i] if lower[i] > final_lb[i-1] or close[i-1] < final_lb[i-1] else final_lb[i-1]
        if trend[i-1] == -1 and close[i] > final_ub[i]:
            trend[i] = 1
        elif trend[i-1] == 1 and close[i] < final_lb[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    return trend
