"""
DXY (US Dollar Index) context analysis for XAUUSD trades.

DXY and gold are typically inversely correlated — a stronger dollar
(rising DXY) tends to pressure gold prices.

This module enriches each trade with DXY regime context at entry and
produces win-rate breakdowns by DXY state.

Two data sources used:
  - DXY 1D  : covers the full trade history (2024-01-01 onwards)
  - DXY 30m : higher resolution, available from Jan 2026 onwards
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------

def _date_lookup(dxy: pd.DataFrame, dt: pd.Timestamp) -> pd.Series | None:
    """Return the DXY row whose date equals dt.date(), or the closest prior row."""
    target = pd.Timestamp(dt.date())
    mask = dxy["time"] <= target
    if not mask.any():
        return None
    return dxy.loc[mask.index[mask][-1]]


def enrich_trades_with_dxy(
    trades: pd.DataFrame,
    dxy_1d: pd.DataFrame,
    dxy_30: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adds DXY context columns to each trade row.

    1D columns (covers all trades):
        dxy_rsi_1d      — DXY RSI(14) on entry date
        dxy_rsi_ma_1d   — DXY RSI-based MA on entry date
        dxy_rsi_vs_ma   — dxy_rsi_1d - dxy_rsi_ma_1d  (positive = USD gaining momentum)
        dxy_trend_1d    — 'up' if close > 20-day MA, else 'down'
        dxy_rsi_bucket  — one of: 'oversold(<30)', 'neutral_low(30-50)',
                                   'neutral_high(50-70)', 'overbought(>70)'

    30m columns (trades from 2026-01-22 onwards only, else NaN):
        dxy_rsi_30      — DXY RSI at the nearest 30-min bar before entry
        dxy_close_30    — DXY close at that bar

    Returns a copy of trades with the new columns appended.
    """
    # Precompute 20-day SMA of DXY close on 1D data
    dxy_1d = dxy_1d.copy()
    dxy_1d["sma20"] = dxy_1d["close"].rolling(20, min_periods=1).mean()

    result = trades.copy()
    result["dxy_rsi_1d"]    = np.nan
    result["dxy_rsi_ma_1d"] = np.nan
    result["dxy_rsi_vs_ma"] = np.nan
    result["dxy_trend_1d"]  = "unknown"
    result["dxy_rsi_bucket"] = "unknown"
    result["dxy_rsi_30"]    = np.nan
    result["dxy_close_30"]  = np.nan

    # --- 1D enrichment ---
    for idx, row in result.iterrows():
        entry = row["entry_time"]
        dxy_row = _date_lookup(dxy_1d, entry)
        if dxy_row is None:
            continue

        rsi    = dxy_row.get("rsi",    np.nan)
        rsi_ma = dxy_row.get("rsi_ma", np.nan)
        sma20  = dxy_row.get("sma20",  np.nan)
        close  = dxy_row.get("close",  np.nan)

        result.at[idx, "dxy_rsi_1d"]    = rsi
        result.at[idx, "dxy_rsi_ma_1d"] = rsi_ma
        result.at[idx, "dxy_rsi_vs_ma"] = (rsi - rsi_ma) if (not np.isnan(rsi) and not np.isnan(rsi_ma)) else np.nan
        result.at[idx, "dxy_trend_1d"]  = "up" if (not np.isnan(close) and not np.isnan(sma20) and close > sma20) else "down"

        if not np.isnan(rsi):
            if rsi < 30:
                bucket = "oversold(<30)"
            elif rsi < 50:
                bucket = "neutral_low(30-50)"
            elif rsi < 70:
                bucket = "neutral_high(50-70)"
            else:
                bucket = "overbought(>70)"
            result.at[idx, "dxy_rsi_bucket"] = bucket

    # --- 30m enrichment (only where data exists) ---
    if dxy_30 is not None:
        dxy_30_sorted = dxy_30.sort_values("time").reset_index(drop=True)
        for idx, row in result.iterrows():
            entry = row["entry_time"]
            mask = dxy_30_sorted["time"] <= entry
            if not mask.any():
                continue
            bar = dxy_30_sorted.loc[mask.index[mask][-1]]
            result.at[idx, "dxy_rsi_30"]   = bar.get("rsi", np.nan)
            result.at[idx, "dxy_close_30"] = bar.get("close", np.nan)

    return result


# ---------------------------------------------------------------------------
# Analysis views
# ---------------------------------------------------------------------------

def dxy_regime_stats(enriched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns win-rate breakdowns by DXY regime.

    Keys:
        'by_bucket'  — win rate by RSI bucket (all trades with 1D data)
        'by_trend'   — win rate by DXY 1D trend direction
        'by_rsi_vs_ma' — win rate when DXY RSI is above vs below its MA
    """
    def _win_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
        rows = []
        for val, grp in df.groupby(col):
            total = len(grp)
            wins  = (grp["result"] == "win").sum()
            avg_pnl = grp["net_pnl_usd"].mean()
            rows.append({col: val, "total": total, "wins": wins,
                         "win_rate": round(wins / total, 3) if total else 0,
                         "avg_pnl_usd": round(avg_pnl, 2)})
        return pd.DataFrame(rows).set_index(col).sort_index()

    bucket_order = [
        "oversold(<30)", "neutral_low(30-50)",
        "neutral_high(50-70)", "overbought(>70)",
    ]
    by_bucket = _win_stats(
        enriched[enriched["dxy_rsi_bucket"] != "unknown"], "dxy_rsi_bucket"
    ).reindex([b for b in bucket_order if b in enriched["dxy_rsi_bucket"].values])

    by_trend = _win_stats(
        enriched[enriched["dxy_trend_1d"].isin(["up", "down"])], "dxy_trend_1d"
    )

    enriched2 = enriched.copy()
    enriched2["dxy_momentum"] = enriched2["dxy_rsi_vs_ma"].apply(
        lambda x: "RSI>MA (USD gaining)" if (not np.isnan(x) and x > 0)
        else ("RSI<MA (USD losing)" if (not np.isnan(x) and x <= 0)
              else "unknown")
    )
    by_momentum = _win_stats(
        enriched2[enriched2["dxy_momentum"] != "unknown"], "dxy_momentum"
    )

    return {
        "by_bucket": by_bucket,
        "by_trend": by_trend,
        "by_momentum": by_momentum,
    }


def dxy_correlation_stats(xauusd_1d: pd.DataFrame, dxy_1d: pd.DataFrame,
                           window: int = 30) -> pd.DataFrame:
    """
    Computes rolling window-day correlation between DXY and XAUUSD daily returns.
    Returns DataFrame with columns: time, dxy_ret, xau_ret, rolling_corr.
    """
    dxy  = dxy_1d[["time", "close"]].rename(columns={"close": "dxy_close"}).copy()
    xau  = xauusd_1d[["time", "close"]].rename(columns={"close": "xau_close"}).copy()

    dxy["dxy_ret"] = dxy["dxy_close"].pct_change()
    xau["xau_ret"] = xau["xau_close"].pct_change()

    merged = pd.merge(
        dxy[["time", "dxy_ret"]],
        xau[["time", "xau_ret"]],
        on="time", how="inner",
    ).dropna()

    merged["rolling_corr"] = (
        merged["dxy_ret"].rolling(window).corr(merged["xau_ret"])
    )
    return merged
