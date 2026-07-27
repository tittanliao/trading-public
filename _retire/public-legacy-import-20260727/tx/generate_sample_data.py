#!/usr/bin/env python3
"""
Generate synthetic MTX 30m OHLC data in TradingView CSV format.
Run this once to create csv/TAIFEX_MXF1!, 30.csv for testing.
Replace with real TradingView export when available.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV_DIR = Path(__file__).parent / "csv"
SEED = 42
START = "2025-01-02"
END = "2026-05-09"
BASE_PRICE = 21500  # MTX starting price


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return (100 - 100 / (1 + rs)).round(2)


def _build_timestamps(start: str, end: str) -> list:
    """Build list of 30m timestamps for day + night sessions, weekdays only."""
    timestamps = []
    dates = pd.date_range(start, end, freq="B")  # Business days

    for date in dates:
        # Day session: 08:45 – 13:15 (last open bar)
        day = pd.date_range(
            date.replace(hour=8, minute=45),
            date.replace(hour=13, minute=15),
            freq="30min",
        )
        timestamps.extend(day)

        # Night session: 15:00 – 23:30 (same calendar day)
        night_early = pd.date_range(
            date.replace(hour=15, minute=0),
            date.replace(hour=23, minute=30),
            freq="30min",
        )
        timestamps.extend(night_early)

        # Night session continued: 00:00 – 04:30 (next calendar day)
        next_date = date + pd.Timedelta(days=1)
        if next_date.weekday() < 5:
            night_late = pd.date_range(
                next_date.replace(hour=0, minute=0),
                next_date.replace(hour=4, minute=30),
                freq="30min",
            )
            timestamps.extend(night_late)

    return sorted(set(timestamps))


def _generate_ohlc(n: int, base: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate OHLC via geometric random walk with intraday patterns."""
    log_ret = rng.normal(0.00005, 0.0025, n)

    # Add some regime changes (trend + mean reversion)
    trend = np.zeros(n)
    regime = 1.0
    for i in range(n):
        if rng.random() < 0.003:
            regime = rng.choice([-1.5, -1.0, 0.5, 1.0, 1.5, 2.0])
        trend[i] = regime * 0.00008
    log_ret += trend

    closes = base * np.exp(np.cumsum(log_ret))

    # Build OHLC
    bar_vol = np.abs(rng.normal(0.003, 0.001, n)).clip(0.001, 0.012)
    opens = np.roll(closes, 1)
    opens[0] = base
    highs = np.maximum(opens, closes) * (1 + bar_vol * rng.uniform(0.3, 1.0, n))
    lows = np.minimum(opens, closes) * (1 - bar_vol * rng.uniform(0.3, 1.0, n))

    # Round to whole points (MTX trades in 1-point increments)
    return pd.DataFrame({
        "open": np.round(opens).astype(int),
        "high": np.round(highs).astype(int),
        "low": np.round(lows).astype(int),
        "close": np.round(closes).astype(int),
    })


def generate_30m(output_filename: str = "TAIFEX_MXF1!, 30.csv"):
    rng = np.random.default_rng(SEED)
    timestamps = _build_timestamps(START, END)
    n = len(timestamps)

    ohlc = _generate_ohlc(n, BASE_PRICE, rng)
    ohlc.index = timestamps

    rsi_series = _rsi(ohlc["close"])
    rsi_ma = rsi_series.rolling(9).mean().round(2)

    df = pd.DataFrame({
        "time": [t.strftime("%Y-%m-%d %H:%M:%S+08:00") for t in timestamps],
        "open": ohlc["open"].values,
        "high": ohlc["high"].values,
        "low": ohlc["low"].values,
        "close": ohlc["close"].values,
        "RSI": rsi_series.values,
        "RSI-based MA": rsi_ma.values,
        "Regular Bullish": "",
        "Regular Bullish Label": "",
        "Regular Bearish": "",
        "Regular Bearish Label": "",
    })

    out = CSV_DIR / output_filename
    CSV_DIR.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {n} bars → {out}")
    print(f"Date range : {timestamps[0]} – {timestamps[-1]}")
    print(f"Price range: {ohlc['close'].min()} – {ohlc['close'].max()}")


if __name__ == "__main__":
    generate_30m()
    print("\n✓ Sample data ready.  Now run:")
    print("  python run_experiments.py")
    print("  python run_short_experiments.py")
