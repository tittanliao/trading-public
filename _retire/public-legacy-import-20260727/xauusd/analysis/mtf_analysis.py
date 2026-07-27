"""
Multi-Timeframe (MTF) alignment analysis for XAUUSD trades.

For each 30m trade entry, look up the higher-timeframe (60m, 4H, 1D) context
to test whether the entry was aligned with or against the HTF trend.

Key hypothesis:
  - A 30m long signal fired when the 4H is bearish → much higher fail rate
  - Full HTF alignment (all 3 TFs bullish) → highest win rate
  - 4H RSI overbought at entry → immediate_loss clusters here
  - Neutral 4H (no direction) → time_bleed clusters here

Columns added to trades by enrich_trades_with_htf():
    htf_60m_rsi         float  — 60m RSI at nearest bar ≤ entry
    htf_60m_rsi_ma      float  — 60m RSI-based MA
    htf_60m_rsi_state   str    — bullish | bearish | neutral | unknown
    htf_60m_rsi_bucket  str    — oversold(<30)|low(30-50)|high(50-70)|overbought(>70)
    htf_60m_vol_ratio   float  — 60m ATR / 20-bar ATR SMA (>1.3 = high vol)

    htf_4h_rsi          float  — 4H RSI at nearest bar ≤ entry
    htf_4h_rsi_ma       float  — 4H RSI-based MA
    htf_4h_rsi_state    str    — bullish | bearish | neutral | unknown
    htf_4h_rsi_bucket   str
    htf_4h_vol_ratio    float

    htf_1d_rsi          float  — 1D RSI at nearest bar ≤ entry
    htf_1d_rsi_state    str
    htf_1d_rsi_bucket   str

    htf_alignment       int    — 0-3: count of (60m, 4H, 1D) that are 'bullish'
    htf_alignment_label str    — human-readable label
    htf_conflict        bool   — 4H is bearish (counter-trend long entry)
    htf_high_vol        bool   — 4H ATR > 1.3× its 20-bar SMA
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR, its 20-bar SMA, and vol_ratio to a price DataFrame."""
    d = df.copy()
    prev_c = d["close"].shift(1)
    d["_tr"] = np.maximum(
        d["high"] - d["low"],
        np.maximum((d["high"] - prev_c).abs(), (d["low"] - prev_c).abs()),
    )
    d["atr"]       = d["_tr"].rolling(period, min_periods=1).mean()
    d["atr_sma20"] = d["atr"].rolling(20, min_periods=1).mean()
    d["vol_ratio"] = d["atr"] / d["atr_sma20"].replace(0, np.nan)
    return d.drop(columns=["_tr"])


def _rsi_state(rsi: float, rsi_ma: float, slope: float) -> str:
    """Classify RSI momentum state into bullish / bearish / neutral."""
    if pd.isna(rsi) or pd.isna(rsi_ma):
        return "unknown"
    above  = rsi > rsi_ma
    rising = (not pd.isna(slope)) and slope > 0
    if above and rising:
        return "bullish"
    if (not above) and (not rising):
        return "bearish"
    return "neutral"


def _rsi_bucket(rsi: float) -> str:
    if pd.isna(rsi):
        return "unknown"
    if rsi < 30:  return "oversold(<30)"
    if rsi < 50:  return "low(30-50)"
    if rsi < 70:  return "high(50-70)"
    return "overbought(>70)"


def _enrich_one_tf(
    trades: pd.DataFrame,
    price: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Merge the nearest HTF bar (≤ entry_time) into each trade using merge_asof.
    Adds {prefix}_rsi, {prefix}_rsi_ma, {prefix}_rsi_state,
         {prefix}_rsi_bucket, {prefix}_vol_ratio.
    """
    p = _compute_atr(price.sort_values("time").reset_index(drop=True).copy())

    if "rsi_ma" in p.columns:
        p["_slope"] = p["rsi_ma"].diff(3)

    rename = {
        "rsi":       f"{prefix}_rsi",
        "rsi_ma":    f"{prefix}_rsi_ma",
        "_slope":    f"{prefix}_slope",
        "vol_ratio": f"{prefix}_vol_ratio",
    }
    p = p.rename(columns={k: v for k, v in rename.items() if k in p.columns})
    p = p.rename(columns={"time": "entry_time"})

    want = ["entry_time"] + [v for v in rename.values() if v in p.columns]
    p = p[[c for c in want if c in p.columns]]

    t = trades.sort_values("entry_time").reset_index(drop=True).copy()
    merged = pd.merge_asof(t, p, on="entry_time", direction="backward")

    rsi_col    = f"{prefix}_rsi"
    rsi_ma_col = f"{prefix}_rsi_ma"
    slope_col  = f"{prefix}_slope"

    if rsi_col in merged.columns and rsi_ma_col in merged.columns:
        merged[f"{prefix}_rsi_state"] = merged.apply(
            lambda r: _rsi_state(r[rsi_col], r[rsi_ma_col],
                                  r.get(slope_col, np.nan)),
            axis=1,
        )
        merged[f"{prefix}_rsi_bucket"] = merged[rsi_col].apply(_rsi_bucket)

    if slope_col in merged.columns:
        merged = merged.drop(columns=[slope_col])

    return merged.sort_values("trade_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_trades_with_htf(
    trades: pd.DataFrame,
    price_60m: pd.DataFrame | None = None,
    price_4h:  pd.DataFrame | None = None,
    price_1d:  pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Enrich trades with multi-timeframe context.
    Any timeframe can be None (columns will be missing gracefully).
    Returns a copy of trades with HTF columns appended.
    """
    result = trades.copy()

    if price_60m is not None and not price_60m.empty:
        result = _enrich_one_tf(result, price_60m, "htf_60m")

    if price_4h is not None and not price_4h.empty:
        result = _enrich_one_tf(result, price_4h, "htf_4h")

    if price_1d is not None and not price_1d.empty:
        result = _enrich_one_tf(result, price_1d, "htf_1d")

    # Alignment score: count of HTFs in 'bullish' state
    state_cols = [c for c in result.columns
                  if c.startswith("htf_") and c.endswith("_rsi_state")]
    result["htf_alignment"] = result[state_cols].apply(
        lambda row: int((row == "bullish").sum()), axis=1
    )
    n_tfs = len(state_cols)
    result["htf_alignment_label"] = result["htf_alignment"].map(
        lambda x: f"{x}/{n_tfs} " + (
            "None" if x == 0 else
            "Weak" if x == 1 else
            "Moderate" if x == n_tfs - 1 and n_tfs > 1 else
            "Full"
        )
    )

    # Conflict: 4H bearish → long entry is counter-trend
    result["htf_conflict"] = (
        result["htf_4h_rsi_state"] == "bearish"
        if "htf_4h_rsi_state" in result.columns
        else False
    )

    # High-vol flag: 4H ATR > 1.3× SMA
    result["htf_high_vol"] = (
        result["htf_4h_vol_ratio"] > 1.3
        if "htf_4h_vol_ratio" in result.columns
        else False
    )

    return result


def prepare_htf_filter(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bar RSI state from a price DataFrame (with rsi + rsi_ma columns).
    Returns a 2-column DataFrame (time, rsi_state) sorted by time,
    ready to pass as htf_filter= in the backtesting engine.
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["time", "rsi_state"])
    df = price_df.sort_values("time").reset_index(drop=True).copy()
    if "rsi_ma" in df.columns:
        df["_slope"] = df["rsi_ma"].diff(3)
        df["rsi_state"] = df.apply(
            lambda r: _rsi_state(r["rsi"], r["rsi_ma"], r.get("_slope", np.nan)), axis=1
        )
    else:
        df["rsi_state"] = "unknown"
    return df[["time", "rsi_state"]].copy()


def trades_to_df(trades_list, strategy_id: str | None = None) -> pd.DataFrame:
    """
    Convert a list[Trade] (from experiments/engine.py) to a DataFrame
    compatible with enrich_trades_with_htf() and htf_stats().
    Uses pnl_pct as a proxy for net_pnl_usd.
    """
    if not trades_list:
        return pd.DataFrame()
    rows = []
    for i, t in enumerate(trades_list):
        rows.append({
            "trade_id":    i,
            "entry_time":  t.entry_time,
            "exit_time":   t.exit_time,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "pnl_pct":     t.pnl_pct,
            "net_pnl_usd": t.pnl_pct,
            "hold_bars":   t.hold_bars,
            "result":      t.result,
            "mfe_pct":     t.mfe_pct,
            "mae_pct":     t.mae_pct,
            "exit_reason": t.exit_reason,
        })
    df = pd.DataFrame(rows)
    if strategy_id:
        df["strategy_id"] = strategy_id
    return df.sort_values("entry_time").reset_index(drop=True)


def htf_stats(enriched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns a dict of summary DataFrames:
        by_alignment    — win rate by HTF alignment label
        by_4h_state     — win rate by 4H RSI state
        by_4h_bucket    — win rate by 4H RSI bucket
        by_1d_state     — win rate by 1D RSI state (if available)
        by_conflict     — win rate: conflict vs aligned
        by_vol_regime   — win rate: high-vol vs normal
        fail_by_4h      — fail type % breakdown by 4H state (losses only)
        coverage        — data coverage summary
    """
    def _win_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns or df.empty:
            return pd.DataFrame()
        rows = []
        for val, grp in df.groupby(col):
            total = len(grp)
            wins  = (grp["result"] == "win").sum()
            rows.append({
                col:         val,
                "total":     total,
                "wins":      int(wins),
                "win_rate":  round(wins / total, 3) if total else 0.0,
                "avg_pnl":   round(grp["net_pnl_usd"].mean(), 2),
            })
        return pd.DataFrame(rows).set_index(col)

    out: dict[str, pd.DataFrame] = {}

    out["by_alignment"] = _win_table(enriched, "htf_alignment_label")

    for col, key in [
        ("htf_4h_rsi_state",  "by_4h_state"),
        ("htf_4h_rsi_bucket", "by_4h_bucket"),
        ("htf_1d_rsi_state",  "by_1d_state"),
    ]:
        if col in enriched.columns:
            valid = enriched[enriched[col] != "unknown"]
            out[key] = _win_table(valid, col)

    # Conflict flag stats
    if "htf_conflict" in enriched.columns:
        labeled = enriched.copy()
        labeled["_conflict_label"] = labeled["htf_conflict"].map(
            {True: "counter-trend (4H bearish)",
             False: "aligned (4H not bearish)"}
        )
        out["by_conflict"] = _win_table(labeled, "_conflict_label")

    # High-vol flag stats
    if "htf_high_vol" in enriched.columns:
        labeled = enriched.copy()
        labeled["_vol_label"] = labeled["htf_high_vol"].map(
            {True: "high-vol (ATR>1.3×SMA)", False: "normal vol"}
        )
        out["by_vol_regime"] = _win_table(labeled, "_vol_label")

    # Fail type breakdown by 4H state (losses only, rows=4H state, cols=fail type)
    if "fail_type" in enriched.columns and "htf_4h_rsi_state" in enriched.columns:
        losses = enriched[enriched["result"] == "loss"]
        valid  = losses[losses["htf_4h_rsi_state"] != "unknown"]
        if len(valid):
            out["fail_by_4h"] = pd.crosstab(
                valid["htf_4h_rsi_state"],
                valid["fail_type"],
                margins=True,
                normalize="index",
            ).round(3)

    # Coverage report
    rows = []
    for col, label in [
        ("htf_60m_rsi", "60m"),
        ("htf_4h_rsi",  "4H"),
        ("htf_1d_rsi",  "1D"),
    ]:
        if col in enriched.columns:
            n_total  = len(enriched)
            n_with   = enriched[col].notna().sum()
            rows.append({
                "timeframe": label,
                "trades_covered": int(n_with),
                "total_trades":   n_total,
                "coverage_pct":   round(n_with / n_total * 100, 1) if n_total else 0.0,
            })
    if rows:
        out["coverage"] = pd.DataFrame(rows).set_index("timeframe")

    return out
