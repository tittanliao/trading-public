"""
Pre-entry context analysis for immediate_loss trades.

Two analysis layers:

Layer 1 — Trade-data only (covers all historical trades):
    Examines time patterns, post-loss clustering, and entry spacing
    using only the trade CSV. No price data required.

Layer 2 — K-bar enrichment (requires overlapping price data):
    Extracts indicator state and candle shape at/before entry.
    Works automatically for any trades that fall within the loaded
    price DataFrame's time range.

To extend K-bar coverage:
    Export a longer OHLCV + indicator CSV from TradingView for the
    full backtest period and replace (or append to) FX_IDC_XAUUSD, 30.csv.
    The analysis will automatically pick up the additional rows.

K-bar features extracted per entry (when price data is available):
    rsi             — RSI(14) value at entry bar
    rsi_vs_ma       — RSI minus RSI-based MA (positive = RSI above trend, momentum up)
    rsi_slope_3     — (rsi[-1] - rsi[-4]) / 3  (3-bar RSI slope)
    prev_1_dir      — direction of bar immediately before entry (+1 bullish, -1 bearish)
    prev_3_green    — count of green (close>open) bars in last 3 bars
    prev_3_range    — avg (high-low) of last 3 bars
    momentum_3      — (close[-1] - close[-4]) / close[-4] * 100  (3-bar return)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Layer 1: Trade-data-only analysis
# ---------------------------------------------------------------------------

def add_trade_context(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Returns trades with additional context columns derived purely from
    the trade CSV (no price data needed):

        entry_hour      int   — UTC+8 hour of entry (0–23)
        entry_dow       int   — day of week (0=Mon … 6=Sun)
        entry_month     int   — calendar month (1–12)
        prev_result     str   — result of the immediately preceding trade
        trades_since_win int  — how many consecutive non-winning trades before this one
        entry_gap_bars  int   — 30-min bars between previous exit and this entry
    """
    df = trades.sort_values("entry_time").copy().reset_index(drop=True)

    df["entry_hour"] = df["entry_time"].dt.hour
    df["entry_dow"] = df["entry_time"].dt.dayofweek
    df["entry_month"] = df["entry_time"].dt.month

    df["prev_result"] = df["result"].shift(1).fillna("none")

    # Count consecutive non-wins before this trade
    tsw = []
    count = 0
    for r in df["result"]:
        tsw.append(count)
        count = count + 1 if r != "win" else 0
    df["trades_since_win"] = tsw

    # Gap from previous exit to this entry (in 30-min bars)
    prev_exit = df["exit_time"].shift(1)
    df["entry_gap_bars"] = (
        (df["entry_time"] - prev_exit).dt.total_seconds() / 1800
    ).fillna(0).astype(int)

    return df


def immediate_loss_profile(trades_with_context: pd.DataFrame) -> dict:
    """
    Compares the distribution of context features between immediate_loss
    and all other trades.

    Returns a dict of DataFrames keyed by feature name.
    """
    df = trades_with_context
    imm = df[df["fail_type"] == "immediate_loss"] if "fail_type" in df.columns else df
    rest = df[~df.index.isin(imm.index)]

    def dist(col: str, data: pd.DataFrame) -> pd.Series:
        return data[col].value_counts(normalize=True).sort_index().round(3)

    return {
        "entry_hour": pd.DataFrame({
            "immediate_loss": dist("entry_hour", imm),
            "all_trades": dist("entry_hour", df),
        }).fillna(0),
        "entry_dow": pd.DataFrame({
            "immediate_loss": dist("entry_dow", imm),
            "all_trades": dist("entry_dow", df),
        }).fillna(0).rename(
            index={0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        ),
        "prev_result": pd.DataFrame({
            "immediate_loss": dist("prev_result", imm),
            "all_trades": dist("prev_result", df),
        }).fillna(0),
        "trades_since_win": pd.DataFrame({
            "immediate_loss": dist("trades_since_win", imm),
            "all_trades": dist("trades_since_win", df),
        }).fillna(0),
    }


# ---------------------------------------------------------------------------
# Layer 2: K-bar enrichment
# ---------------------------------------------------------------------------

def _kbar_features_at(price: pd.DataFrame, entry_time: pd.Timestamp,
                      n_lookback: int = 3) -> dict | None:
    """
    Extracts indicator + candle features at and before entry_time.
    Returns None if not enough price history is available.
    """
    idx = price.index[price["time"] == entry_time]
    if idx.empty:
        # Try nearest bar within 30 min
        diff = (price["time"] - entry_time).abs()
        nearest = diff.idxmin()
        if diff[nearest].total_seconds() > 1800:
            return None
        idx = pd.Index([nearest])

    pos = idx[0]
    if pos < n_lookback:
        return None

    bar = price.loc[pos]
    prev_bars = price.loc[pos - n_lookback: pos - 1]

    feat: dict = {}

    # RSI features
    if "rsi" in price.columns:
        feat["rsi"] = bar["rsi"]
        feat["rsi_vs_ma"] = (
            bar["rsi"] - bar["rsi_ma"] if "rsi_ma" in price.columns else np.nan
        )
        if pos >= n_lookback + 1:
            oldest_rsi = price.loc[pos - n_lookback, "rsi"]
            feat["rsi_slope_3"] = (bar["rsi"] - oldest_rsi) / n_lookback
        else:
            feat["rsi_slope_3"] = np.nan
    else:
        feat["rsi"] = np.nan
        feat["rsi_vs_ma"] = np.nan
        feat["rsi_slope_3"] = np.nan

    # Previous bar direction
    pb = price.loc[pos - 1]
    feat["prev_1_dir"] = 1 if pb["close"] >= pb["open"] else -1

    # Last-n-bars candle stats
    feat["prev_3_green"] = int((prev_bars["close"] >= prev_bars["open"]).sum())
    feat["prev_3_range"] = (prev_bars["high"] - prev_bars["low"]).mean()

    # 3-bar momentum
    oldest_close = price.loc[pos - n_lookback, "close"]
    prev_close = price.loc[pos - 1, "close"]
    feat["momentum_3"] = (prev_close - oldest_close) / oldest_close * 100

    return feat


def enrich_with_kbars(classified_losses: pd.DataFrame,
                      price: pd.DataFrame,
                      n_lookback: int = 3) -> pd.DataFrame:
    """
    Adds K-bar feature columns to classified_losses for trades that
    fall within the price DataFrame's time range.

    Rows outside the price window will have NaN for all feature columns.
    Returns a new DataFrame (does not modify input).
    """
    feat_cols = [
        "rsi", "rsi_vs_ma", "rsi_slope_3",
        "prev_1_dir", "prev_3_green", "prev_3_range", "momentum_3",
    ]
    result = classified_losses.copy()
    for col in feat_cols:
        result[col] = np.nan

    for i, row in result.iterrows():
        feats = _kbar_features_at(price, row["entry_time"], n_lookback)
        if feats:
            for col, val in feats.items():
                result.at[i, col] = val

    return result


def kbar_coverage(enriched: pd.DataFrame) -> dict:
    """Reports how many trades have K-bar data vs missing."""
    total = len(enriched)
    covered = enriched["rsi"].notna().sum()
    return {
        "total_losses": total,
        "with_kbar_data": int(covered),
        "missing_kbar_data": int(total - covered),
        "coverage_pct": round(covered / total * 100, 1) if total else 0,
    }
