"""Shared fail-pattern toolkit — reference implementation of
docs/RESEARCH_DEVELOPMENT_SPEC.md section 5 (standard single-strategy report contract).

Self-contained: does not import the read-only legacy trading/xauusd/analysis/ package
at runtime, though it is ported from and stays behaviorally consistent with it. Column
schema, thresholds, and chart section taxonomy follow the committed spec exactly so a
solo-report runner and a gap-report runner can both build on this module without
re-deriving conventions.
"""
from __future__ import annotations

import base64
import io
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (docs/RESEARCH_DEVELOPMENT_SPEC.md section 5.1)
# ---------------------------------------------------------------------------

IMMEDIATE_LOSS_MFE_PCT = 0.10
FALSE_BREAKOUT_MAE_MFE_RATIO = 2.0
TIME_BLEED_MIN_BARS = 24  # 30-min bars => 12 hours

ENTRY_SLOTS_30M = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in (0, 30)]
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

BB_ZONE_ORDER = [
    "below_lower", "near_lower", "lower_mid", "near_middle",
    "upper_mid", "near_upper", "above_upper",
]

CHART_SECTIONS = [
    "performance", "fail_pattern", "timing_30m", "pre_entry",
    "kbar", "bb", "dxy", "mtf", "hold_time_streaks", "comparison",
]

_WIN, _LOSS, _NEUTRAL, _BLUE, _DXY_COLOR = "#2ecc71", "#e74c3c", "#95a5a6", "#3498db", "#e67e22"
FAIL_COLORS = {
    "immediate_loss": "#c0392b",
    "false_breakout": "#e67e22",
    "time_bleed": "#8e44ad",
    "normal_sl": "#7f8c8d",
}

# TradingView export column names changed between the V3.4 and V3.9 strategy versions.
TRADE_COLUMN_ALIASES = {
    "trade_id": ["Trade number", "Trade #"],
    "type": ["Type"],
    "datetime": ["Date and time"],
    "signal": ["Signal"],
    "price": ["Price USD"],
    "size_qty": ["Size (qty)"],
    "net_pnl_usd": ["Net PnL USD", "Net P&L USD"],
    "return_pct": ["Return %", "Net P&L %"],
    "mfe_pct": ["Favorable excursion %"],
    "mae_pct": ["Adverse excursion %"],
}


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def _pick_column(columns: list[str], canonical: str) -> str:
    for alias in TRADE_COLUMN_ALIASES[canonical]:
        if alias in columns:
            return alias
    raise KeyError(f"no column found for {canonical!r} among {columns}")


def load_trades(path: Path) -> pd.DataFrame:
    """Load a TradingView 'List of Trades' export, either V3.4 or V3.9 column schema."""
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("﻿").strip() for column in frame.columns]
    columns = list(frame.columns)
    rename = {_pick_column(columns, canonical): canonical for canonical in TRADE_COLUMN_ALIASES}
    frame = frame.rename(columns=rename)
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    entries = frame[frame["type"] == "Entry long"][
        ["trade_id", "datetime", "signal", "price", "size_qty"]
    ].rename(columns={"datetime": "entry_time", "signal": "entry_signal", "price": "entry_price"})
    exits = frame[frame["type"] == "Exit long"][
        ["trade_id", "datetime", "signal", "price", "net_pnl_usd", "return_pct", "mfe_pct", "mae_pct"]
    ].rename(columns={"datetime": "exit_time", "signal": "exit_signal", "price": "exit_price"})

    trades = entries.merge(exits, on="trade_id", how="inner")
    trades = trades[trades["exit_signal"].astype(str).str.upper() != "OPEN"].copy()

    trades["hold_bars"] = ((trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 1800).astype(int)
    trades["win"] = trades["net_pnl_usd"] > 0
    trades["result"] = np.select(
        [trades["net_pnl_usd"] > 0, trades["net_pnl_usd"] < 0], ["win", "loss"], default="breakeven"
    )
    trades["entry_hour"] = trades["entry_time"].dt.hour
    trades["entry_slot_30m"] = trades["entry_time"].dt.strftime("%H:%M")
    trades["entry_dow"] = trades["entry_time"].dt.dayofweek
    minute_of_day = trades["entry_time"].dt.hour * 60 + trades["entry_time"].dt.minute
    trades["session"] = np.select(
        [
            minute_of_day.between(7 * 60, 15 * 60 - 1),
            minute_of_day.between(15 * 60, 20 * 60 + 29),
            (minute_of_day >= 20 * 60 + 30) | (minute_of_day < 60),
        ],
        ["asia", "europe", "us"],
        default="overnight",
    )
    return trades.sort_values("entry_time").reset_index(drop=True)


def _compute_wilder_rsi(close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi_ma = rsi.rolling(period).mean()
    return rsi, rsi_ma


def load_price_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Load an OHLC(V) TradingView export. Returns (frame, rsi_provenance).

    rsi_provenance is 'native_export' if the file already had RSI/RSI-based MA columns,
    else 'locally_computed_wilder_rsi14' when this function computed them from close.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    time_col = next(c for c in raw.columns if "time" in c.lower())
    raw[time_col] = pd.to_datetime(raw[time_col], utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    raw = raw.rename(columns={time_col: "time", "RSI": "rsi", "RSI-based MA": "rsi_ma"})

    provenance = "native_export"
    if "rsi" not in raw.columns:
        provenance = "locally_computed_wilder_rsi14"
        raw["rsi"], raw["rsi_ma"] = _compute_wilder_rsi(raw["close"])

    keep = [c for c in ["time", "open", "high", "low", "close", "rsi", "rsi_ma"] if c in raw.columns]
    return raw[keep].sort_values("time").reset_index(drop=True), provenance


# ---------------------------------------------------------------------------
# 2. Metrics + stats helpers (shared shape across every breakdown)
# ---------------------------------------------------------------------------

def wilson_interval(wins: int, count: int, z: float = 1.96) -> list[float] | None:
    if count == 0:
        return None
    p = wins / count
    denom = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denom
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return [round(100 * (centre - margin / denom), 2), round(100 * (centre + margin / denom), 2)]


def stats(frame: pd.DataFrame) -> dict:
    count = len(frame)
    if count == 0:
        return {"n": 0, "wins": 0, "win_rate_pct": None, "win_rate_ci95_pct": None,
                "profit_factor": None, "net_pnl_usd": 0.0, "avg_pnl_usd": None, "avg_return_pct": None}
    wins = int((frame["net_pnl_usd"] > 0).sum())
    gross_profit = frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
    gross_loss = abs(frame.loc[frame["net_pnl_usd"] < 0, "net_pnl_usd"].sum())
    return {
        "n": count,
        "wins": wins,
        "win_rate_pct": round(100 * wins / count, 2),
        "win_rate_ci95_pct": wilson_interval(wins, count),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_usd": round(float(frame["net_pnl_usd"].sum()), 2),
        "avg_pnl_usd": round(float(frame["net_pnl_usd"].mean()), 2),
        "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
    }


def grouped_stats(frame: pd.DataFrame, column: str) -> dict:
    return {str(key): stats(group) for key, group in frame.groupby(column, observed=True)}


def entry_slot_30m_stats(frame: pd.DataFrame) -> dict:
    return {slot: stats(frame[frame["entry_slot_30m"] == slot]) for slot in ENTRY_SLOTS_30M}


def summary(trades: pd.DataFrame) -> dict:
    base = stats(trades)
    return {
        **base,
        "max_drawdown_usd": round(float(max_drawdown(trades)), 2),
        "max_consecutive_losses": int(consecutive_losses(trades).max()) if len(trades) else 0,
        "avg_hold_bars": round(float(trades["hold_bars"].mean()), 2) if len(trades) else None,
    }


def drawdown_series(trades: pd.DataFrame) -> pd.Series:
    cum = trades["net_pnl_usd"].cumsum().reset_index(drop=True)
    return cum - cum.cummax()


def max_drawdown(trades: pd.DataFrame) -> float:
    dd = drawdown_series(trades)
    return float(dd.min()) if len(dd) else 0.0


def consecutive_losses(trades: pd.DataFrame) -> pd.Series:
    streaks, count = [], 0
    for r in trades["result"]:
        if r == "loss":
            count += 1
        else:
            if count > 0:
                streaks.append(count)
            count = 0
    if count > 0:
        streaks.append(count)
    return pd.Series(streaks, dtype=int, name="streak_length")


# ---------------------------------------------------------------------------
# 3. Fail-pattern classification (spec 5.1 item 2)
# ---------------------------------------------------------------------------

def classify_fail(trades: pd.DataFrame) -> pd.DataFrame:
    losses = trades[trades["result"] == "loss"].copy()
    mae_mfe_ratio = losses["mae_pct"] / losses["mfe_pct"].replace(0, np.nan)
    conditions = [
        losses["mfe_pct"] < IMMEDIATE_LOSS_MFE_PCT,
        mae_mfe_ratio > FALSE_BREAKOUT_MAE_MFE_RATIO,
        losses["hold_bars"] >= TIME_BLEED_MIN_BARS,
    ]
    losses["fail_type"] = np.select(conditions, ["immediate_loss", "false_breakout", "time_bleed"], default="normal_sl")
    return losses.reset_index(drop=True)


def fail_type_summary(classified: pd.DataFrame) -> dict:
    total = len(classified)
    counts = classified["fail_type"].value_counts()
    return {
        name: {"count": int(count), "pct": round(100 * count / total, 1) if total else 0.0}
        for name, count in counts.items()
    }


def fail_by_session(classified: pd.DataFrame) -> dict:
    table = pd.crosstab(classified["session"], classified["fail_type"])
    return {session: {ft: int(v) for ft, v in row.items()} for session, row in table.iterrows()}


# ---------------------------------------------------------------------------
# 4. Pre-entry context for immediate_loss (spec 5.1 items 3-4)
# ---------------------------------------------------------------------------

def add_trade_context(trades: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    df = trades.sort_values("entry_time").copy().reset_index(drop=True)
    df = df.merge(classified[["trade_id", "fail_type"]], on="trade_id", how="left")
    df["prev_result"] = df["result"].shift(1).fillna("none")
    tsw, count = [], 0
    for r in df["result"]:
        tsw.append(count)
        count = count + 1 if r != "win" else 0
    df["trades_since_win"] = tsw
    return df


def _distribution_pair(imm: pd.DataFrame, all_trades: pd.DataFrame, column: str, index_labels: dict | None = None) -> dict:
    imm_dist = imm[column].value_counts(normalize=True)
    all_dist = all_trades[column].value_counts(normalize=True)
    keys = sorted(set(imm_dist.index) | set(all_dist.index))
    out = {}
    for key in keys:
        label = index_labels.get(key, str(key)) if index_labels else str(key)
        out[label] = {
            "immediate_loss": round(float(imm_dist.get(key, 0.0)), 3),
            "all_trades": round(float(all_dist.get(key, 0.0)), 3),
        }
    return out


def immediate_loss_profile(trades_ctx: pd.DataFrame) -> dict:
    imm = trades_ctx[trades_ctx["fail_type"] == "immediate_loss"]
    entry_slot = {slot: {"immediate_loss": 0.0, "all_trades": 0.0} for slot in ENTRY_SLOTS_30M}
    dist = _distribution_pair(imm, trades_ctx, "entry_slot_30m")
    entry_slot.update(dist)
    return {
        "entry_slot_30m": entry_slot,
        "entry_dow": _distribution_pair(imm, trades_ctx, "entry_dow", DOW_LABELS),
        "prev_result": _distribution_pair(imm, trades_ctx, "prev_result"),
        "trades_since_win": _distribution_pair(imm, trades_ctx, "trades_since_win"),
    }


def _kbar_features_at(price: pd.DataFrame, entry_time: pd.Timestamp, n_lookback: int = 3) -> dict | None:
    idx = price.index[price["time"] == entry_time]
    if idx.empty:
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
    feat = {
        "rsi": float(bar["rsi"]) if pd.notna(bar.get("rsi")) else None,
        "rsi_vs_ma": float(bar["rsi"] - bar["rsi_ma"]) if pd.notna(bar.get("rsi")) and pd.notna(bar.get("rsi_ma")) else None,
    }
    if pos >= n_lookback + 1 and pd.notna(bar.get("rsi")) and pd.notna(price.loc[pos - n_lookback, "rsi"]):
        feat["rsi_slope_3"] = round(float((bar["rsi"] - price.loc[pos - n_lookback, "rsi"]) / n_lookback), 3)
    else:
        feat["rsi_slope_3"] = None
    prev_bar = price.loc[pos - 1]
    feat["prev_1_dir"] = 1 if prev_bar["close"] >= prev_bar["open"] else -1
    feat["prev_3_green"] = int((prev_bars["close"] >= prev_bars["open"]).sum())
    oldest_close, prev_close = price.loc[pos - n_lookback, "close"], price.loc[pos - 1, "close"]
    feat["momentum_3"] = round(float((prev_close - oldest_close) / oldest_close * 100), 4)
    return feat


def enrich_with_kbars(classified: pd.DataFrame, price: pd.DataFrame, n_lookback: int = 3) -> pd.DataFrame:
    imm = classified[classified["fail_type"] == "immediate_loss"].copy().reset_index(drop=True)
    cols = ["rsi", "rsi_vs_ma", "rsi_slope_3", "prev_1_dir", "prev_3_green", "momentum_3"]
    for col in cols:
        imm[col] = np.nan
    for i, row in imm.iterrows():
        feats = _kbar_features_at(price, row["entry_time"], n_lookback)
        if feats:
            for col, val in feats.items():
                imm.at[i, col] = val
    return imm


def kbar_coverage(enriched: pd.DataFrame) -> dict:
    total = len(enriched)
    covered = int(enriched["rsi"].notna().sum())
    return {"total_immediate_loss": total, "with_kbar_data": covered,
            "coverage_pct": round(100 * covered / total, 1) if total else 0.0}


# ---------------------------------------------------------------------------
# 5. Bollinger Band position (spec 5.1 item 6)
# ---------------------------------------------------------------------------

def compute_bb(price: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    df = price.copy()
    df["bb_mid"] = df["close"].rolling(period, min_periods=period).mean()
    df["bb_std"] = df["close"].rolling(period, min_periods=period).std(ddof=1)
    df["bb_upper"] = df["bb_mid"] + std_mult * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - std_mult * df["bb_std"]
    band_range = df["bb_upper"] - df["bb_lower"]
    df["bb_width"] = np.where(df["bb_mid"].notna() & (df["bb_mid"] != 0), band_range / df["bb_mid"], np.nan)
    df["bb_pct_b"] = np.where(band_range != 0, (df["close"] - df["bb_lower"]) / band_range, np.nan)
    df.loc[df["bb_mid"].isna(), "bb_pct_b"] = np.nan
    return df


def bb_zone(pct_b: float | None) -> str:
    if pct_b is None or (isinstance(pct_b, float) and np.isnan(pct_b)):
        return "unknown"
    if 0.8 <= pct_b <= 1.0:
        return "near_upper"
    thresholds = [
        (float("-inf"), 0.0, "below_lower"), (0.0, 0.2, "near_lower"), (0.2, 0.4, "lower_mid"),
        (0.4, 0.6, "near_middle"), (0.6, 0.8, "upper_mid"), (1.0, float("inf"), "above_upper"),
    ]
    for lo, hi, label in thresholds:
        if lo <= pct_b < hi:
            return label
    return "unknown"


def enrich_trades_with_bb(trades: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    bb_price = compute_bb(price)
    lookup = bb_price[["time", "bb_pct_b", "bb_width"]].sort_values("time").reset_index(drop=True)
    merged = pd.merge_asof(
        trades.sort_values("entry_time"), lookup, left_on="entry_time", right_on="time", direction="backward"
    )
    merged["bb_zone"] = merged["bb_pct_b"].apply(bb_zone)
    return merged


def bb_stats(enriched: pd.DataFrame) -> dict:
    valid = enriched[enriched["bb_zone"] != "unknown"]
    return {zone: stats(valid[valid["bb_zone"] == zone]) for zone in BB_ZONE_ORDER}


# ---------------------------------------------------------------------------
# 6. DXY context (spec 5.1 item 7)
# ---------------------------------------------------------------------------

def enrich_trades_with_dxy(trades: pd.DataFrame, dxy_1d: pd.DataFrame) -> pd.DataFrame:
    dxy = dxy_1d.copy()
    dxy["sma20"] = dxy["close"].rolling(20, min_periods=1).mean()
    dxy["date"] = dxy["time"].dt.normalize()
    lookup = dxy[["date", "rsi", "rsi_ma", "close", "sma20"]].sort_values("date").reset_index(drop=True)

    result = trades.sort_values("entry_time").copy()
    result["date"] = result["entry_time"].dt.normalize()
    merged = pd.merge_asof(result, lookup, on="date", direction="backward")
    merged["dxy_rsi_1d"] = merged["rsi"]
    merged["dxy_rsi_vs_ma"] = merged["rsi"] - merged["rsi_ma"]
    merged["dxy_trend_1d"] = np.where(merged["close"] > merged["sma20"], "up", "down")
    merged["dxy_rsi_bucket"] = np.select(
        [merged["rsi"] < 30, merged["rsi"] < 50, merged["rsi"] < 70],
        ["oversold(<30)", "neutral_low(30-50)", "neutral_high(50-70)"],
        default="overbought(>70)",
    )
    merged.loc[merged["rsi"].isna(), "dxy_rsi_bucket"] = "unknown"
    merged["dxy_momentum"] = np.where(
        merged["dxy_rsi_vs_ma"].isna(), "unknown",
        np.where(merged["dxy_rsi_vs_ma"] > 0, "RSI>MA (USD gaining)", "RSI<MA (USD losing)"),
    )
    return merged.drop(columns=["date", "rsi", "rsi_ma", "close", "sma20"])


def dxy_regime_stats(enriched: pd.DataFrame) -> dict:
    valid_bucket = enriched[enriched["dxy_rsi_bucket"] != "unknown"]
    valid_momentum = enriched[enriched["dxy_momentum"] != "unknown"]
    return {
        "by_bucket": grouped_stats(valid_bucket, "dxy_rsi_bucket"),
        "by_trend": grouped_stats(enriched, "dxy_trend_1d"),
        "by_momentum": grouped_stats(valid_momentum, "dxy_momentum"),
    }


def dxy_correlation_stats(xauusd_1d: pd.DataFrame, dxy_1d: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    dxy = dxy_1d[["time", "close"]].rename(columns={"close": "dxy_close"}).copy()
    xau = xauusd_1d[["time", "close"]].rename(columns={"close": "xau_close"}).copy()
    dxy["dxy_ret"] = dxy["dxy_close"].pct_change()
    xau["xau_ret"] = xau["xau_close"].pct_change()
    merged = pd.merge(dxy[["time", "dxy_ret"]], xau[["time", "xau_ret"]], on="time", how="inner").dropna()
    merged["rolling_corr"] = merged["dxy_ret"].rolling(window).corr(merged["xau_ret"])
    return merged


# ---------------------------------------------------------------------------
# 7. Multi-timeframe alignment (spec 5.1 item 8)
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    d = df.copy()
    prev_c = d["close"].shift(1)
    tr = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - prev_c).abs(), (d["low"] - prev_c).abs()))
    d["atr"] = tr.rolling(period, min_periods=1).mean()
    d["atr_sma20"] = d["atr"].rolling(20, min_periods=1).mean()
    d["vol_ratio"] = d["atr"] / d["atr_sma20"].replace(0, np.nan)
    return d


def _rsi_state(rsi: float, rsi_ma: float, slope: float) -> str:
    if pd.isna(rsi) or pd.isna(rsi_ma):
        return "unknown"
    above, rising = rsi > rsi_ma, (not pd.isna(slope)) and slope > 0
    if above and rising:
        return "bullish"
    if (not above) and (not rising):
        return "bearish"
    return "neutral"


def _rsi_bucket(rsi: float) -> str:
    if pd.isna(rsi):
        return "unknown"
    if rsi < 30:
        return "oversold(<30)"
    if rsi < 50:
        return "low(30-50)"
    if rsi < 70:
        return "high(50-70)"
    return "overbought(>70)"


def _enrich_one_tf(trades: pd.DataFrame, price: pd.DataFrame, prefix: str) -> pd.DataFrame:
    p = _compute_atr(price.sort_values("time").reset_index(drop=True))
    p["_slope"] = p["rsi_ma"].diff(3) if "rsi_ma" in p.columns else np.nan
    p = p.rename(columns={"rsi": f"{prefix}_rsi", "rsi_ma": f"{prefix}_rsi_ma",
                           "_slope": f"{prefix}_slope", "vol_ratio": f"{prefix}_vol_ratio", "time": "entry_time"})
    keep = ["entry_time", f"{prefix}_rsi", f"{prefix}_rsi_ma", f"{prefix}_slope", f"{prefix}_vol_ratio"]
    p = p[[c for c in keep if c in p.columns]]
    merged = pd.merge_asof(trades.sort_values("entry_time"), p, on="entry_time", direction="backward")
    rsi_col, ma_col, slope_col = f"{prefix}_rsi", f"{prefix}_rsi_ma", f"{prefix}_slope"
    merged[f"{prefix}_rsi_state"] = merged.apply(
        lambda r: _rsi_state(r.get(rsi_col), r.get(ma_col), r.get(slope_col)), axis=1
    )
    merged[f"{prefix}_rsi_bucket"] = merged[rsi_col].apply(_rsi_bucket)
    return merged.drop(columns=[slope_col], errors="ignore")


def enrich_trades_with_htf(trades: pd.DataFrame, price_60m: pd.DataFrame, price_4h: pd.DataFrame, price_1d: pd.DataFrame) -> pd.DataFrame:
    result = trades.copy()
    for prefix, price in [("htf_60m", price_60m), ("htf_4h", price_4h), ("htf_1d", price_1d)]:
        result = _enrich_one_tf(result, price, prefix)
    state_cols = [c for c in result.columns if c.startswith("htf_") and c.endswith("_rsi_state")]
    result["htf_alignment"] = result[state_cols].apply(lambda row: int((row == "bullish").sum()), axis=1)
    n_tfs = len(state_cols)
    result["htf_alignment_label"] = result["htf_alignment"].map(
        lambda x: f"{x}/{n_tfs} " + ("None" if x == 0 else "Weak" if x == 1 else "Moderate" if x == n_tfs - 1 and n_tfs > 1 else "Full")
    )
    result["htf_conflict"] = result.get("htf_4h_rsi_state", pd.Series("unknown", index=result.index)) == "bearish"
    result["htf_high_vol"] = result.get("htf_4h_vol_ratio", pd.Series(np.nan, index=result.index)) > 1.3
    return result


def htf_stats(enriched: pd.DataFrame) -> dict:
    out: dict = {"by_alignment": grouped_stats(enriched, "htf_alignment_label")}
    for col, key in [("htf_4h_rsi_state", "by_4h_state"), ("htf_4h_rsi_bucket", "by_4h_bucket"), ("htf_1d_rsi_state", "by_1d_state")]:
        if col in enriched.columns:
            valid = enriched[enriched[col] != "unknown"]
            out[key] = grouped_stats(valid, col)
    labeled = enriched.copy()
    labeled["_conflict_label"] = labeled["htf_conflict"].map({True: "counter-trend (4H bearish)", False: "aligned (4H not bearish)"})
    out["by_conflict"] = grouped_stats(labeled, "_conflict_label")
    labeled["_vol_label"] = labeled["htf_high_vol"].map({True: "high-vol (ATR>1.3xSMA)", False: "normal vol"})
    out["by_vol_regime"] = grouped_stats(labeled, "_vol_label")
    coverage = {}
    for col, label in [("htf_60m_rsi", "60m"), ("htf_4h_rsi", "4H"), ("htf_1d_rsi", "1D")]:
        if col in enriched.columns:
            n_with = int(enriched[col].notna().sum())
            coverage[label] = {"trades_covered": n_with, "total_trades": len(enriched),
                                "coverage_pct": round(100 * n_with / len(enriched), 1) if len(enriched) else 0.0}
    out["coverage"] = coverage
    return out


# ---------------------------------------------------------------------------
# 8. Macro composite context (spec section 5.1 item 10, conditional).
#    Ported unchanged from scripts/research/analyze_s1_v39_context.py's
#    build_macro()/attach_macro() — frozen definition, do not re-tune here.
# ---------------------------------------------------------------------------

MACRO_MA_LENGTH = 50
MACRO_MAX_AGE = pd.Timedelta(days=4)


def load_daily_macro(path: Path) -> pd.DataFrame:
    """Daily-series loader for the Macro composite only. Deliberately does NOT do the
    UTC->Asia/Taipei conversion load_price_csv() does for intraday price/DXY data —
    this must byte-match analyze_s1_v39_context.py's load_daily(), which never
    converted timezone. Using load_price_csv() here shifts every daily bar by 8 hours,
    silently changing which macro observation each trade's backward-asof join picks."""
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("﻿").strip() for column in frame.columns]
    time_column = next(column for column in frame.columns if "time" in column.lower())
    frame[time_column] = pd.to_datetime(frame[time_column], utc=False)
    if frame[time_column].dt.tz is not None:
        frame[time_column] = frame[time_column].dt.tz_localize(None)
    return (
        frame[[time_column, "close"]]
        .rename(columns={time_column: "time"})
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )


def build_macro_composite(paths: dict[str, Path]) -> pd.DataFrame:
    """paths keys: us10y, t10yie, dxy, vix, gold — each a daily OHLC(V) CSV path."""
    series = {}
    for name, path in paths.items():
        frame = load_daily_macro(path)
        series[name] = frame.rename(columns={"close": name}).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    base = series.pop("us10y")
    for name, frame in series.items():
        base = pd.merge_asof(base.sort_values("time"), frame, on="time", direction="backward", tolerance=MACRO_MAX_AGE)
    base["real_rate"] = base["us10y"] - base["t10yie"]
    for column in ["real_rate", "us10y", "dxy", "vix", "gold"]:
        base[f"ma50_{column}"] = base[column].rolling(MACRO_MA_LENGTH, min_periods=MACRO_MA_LENGTH).mean()
    base["pt_real"] = np.where(base["real_rate"] < base["ma50_real_rate"], 2, 0)
    base["pt_10y"] = np.where(base["us10y"] < base["ma50_us10y"], 1, 0)
    base["pt_dxy"] = np.where(base["dxy"] < base["ma50_dxy"], 1, 0)
    base["pt_vix"] = np.where(base["vix"] > base["ma50_vix"], 1, 0)
    base["pt_trend"] = np.where(base["gold"] > base["ma50_gold"], 1, 0)
    required = [f"ma50_{c}" for c in ["real_rate", "us10y", "dxy", "vix", "gold"]]
    valid = base[required].notna().all(axis=1)
    base["macro_score"] = np.where(valid, base[["pt_real", "pt_10y", "pt_dxy", "pt_vix", "pt_trend"]].sum(axis=1), np.nan)
    base["macro_verdict"] = np.select(
        [base["macro_score"] <= 2, base["macro_score"].between(3, 4), base["macro_score"] >= 5],
        ["WAIT", "NEUTRAL", "STRONG BUY"], default="N/A",
    )
    return base.dropna(subset=["macro_score"]).reset_index(drop=True)


def attach_macro(trades: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    left = trades.sort_values("entry_time")
    right = macro.rename(columns={"time": "macro_time"}).sort_values("macro_time")
    joined = pd.merge_asof(
        left, right[["macro_time", "macro_score", "macro_verdict"]],
        left_on="entry_time", right_on="macro_time", direction="backward", tolerance=MACRO_MAX_AGE,
    )
    return joined


def macro_coverage(joined: pd.DataFrame) -> dict:
    matched = joined["macro_score"].notna().sum()
    total = len(joined)
    return {"matched": int(matched), "unmatched": int(total - matched),
            "pct": round(100 * matched / total, 2) if total else 0.0}


def chart_macro_verdict_winrate(macro_stats: dict, strategy_id: str, version: str) -> plt.Figure:
    order = ["WAIT", "NEUTRAL", "STRONG BUY"]
    labels = [v for v in order if macro_stats.get(v, {}).get("n")]
    wr = [macro_stats[v]["win_rate_pct"] for v in labels]
    totals = [macro_stats[v]["n"] for v in labels]
    colors = [_WIN if w >= 55 else _BLUE if w >= 45 else _LOSS for w in wr]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), wr, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    ax.set_ylabel("Win Rate %")
    ax.set_title(f"Win Rate by Macro Composite Verdict — {_label(strategy_id, version)}")
    for i, (w, n) in enumerate(zip(wr, totals)):
        ax.text(i, w + 1.5, f"{w}%\n(n={n})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Live-impact integer rank score (spec section 13.2). Applied only to
#    breakdowns a confirmed, policy-impacting study exposes to a live 請分析
#    rule — never to a purely descriptive fail-pattern report.
# ---------------------------------------------------------------------------

def _tie_collapse(groups: dict, ranked: list[str], band_score: dict[str, int]) -> None:
    by_wr: dict[float, list[str]] = {}
    for key in ranked:
        by_wr.setdefault(groups[key]["win_rate_pct"], []).append(key)
    for members in by_wr.values():
        floor = min(band_score[m] for m in members)
        for m in members:
            groups[m]["rank_score"] = floor


def compute_rank_scores(groups: dict, ordered_keys: list[str] | None = None) -> dict:
    """Section 13.2: integer rank score by win-rate ranking. Mutates and returns
    `groups` — every key gets rank_score/low_sample/rank_excluded_reason, no
    conditional branches needed downstream."""
    keys = ordered_keys if ordered_keys is not None else list(groups.keys())
    for key in keys:
        groups[key]["rank_score"] = 0
        groups[key]["low_sample"] = False
        groups[key]["rank_excluded_reason"] = None

    participating = [k for k in keys if groups[k]["n"] >= 5]
    for k in keys:
        if k not in participating:
            groups[k]["low_sample"] = True
            groups[k]["rank_excluded_reason"] = "low_sample"

    ranked = sorted(participating, key=lambda k: (-groups[k]["win_rate_pct"], -groups[k]["n"], k))
    length = len(ranked)

    if length < 2 or len({groups[k]["win_rate_pct"] for k in ranked}) == 1:
        for k in ranked:
            groups[k]["rank_score"] = 0
        return groups

    if length >= 10:
        # numpy.quantile / pandas.qcut / rank-percentile formulas all disagree with each
        # other and with this rule at the band boundaries — use array_split, not those.
        bands = np.array_split(np.arange(length), 5)
        scores = [2, 1, 0, -1, -2]
        band_score = {ranked[idx]: scores[band_idx] for band_idx, band in enumerate(bands) for idx in band}
    else:
        band_score = {ranked[i]: (1 if i == 0 else -1 if i == length - 1 else 0) for i in range(length)}

    _tie_collapse(groups, ranked, band_score)
    return groups


def mark_descriptive_only(groups: dict, keys: list[str]) -> dict:
    """For specific groups that are excluded from ranking by rule, not by sample size
    (section 13.2 rule 7 — e.g. by_session["overnight"], never `by_session`'s other
    keys). Only touches the listed keys; call after compute_rank_scores(), which
    should be given ordered_keys excluding these same keys."""
    for key in keys:
        groups[key]["rank_score"] = 0
        groups[key]["low_sample"] = False
        groups[key]["rank_excluded_reason"] = "descriptive_only"
    return groups


# ---------------------------------------------------------------------------
# 10. Charts — each returns a matplotlib Figure. Titles use only
#    {strategy_id, version, section label} — no file paths or private strings.
# ---------------------------------------------------------------------------

def _label(strategy_id: str, version: str) -> str:
    return f"{strategy_id} v{version}"


def chart_equity_curve(trades: pd.DataFrame, strategy_id: str, version: str) -> plt.Figure:
    dd = drawdown_series(trades)
    cum = trades["net_pnl_usd"].cumsum()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(trades["entry_time"], cum, linewidth=1.5, color=_BLUE)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cumulative P&L (USD)")
    ax1.set_title(f"Equity Curve — {_label(strategy_id, version)}")
    ax2.fill_between(trades["entry_time"], dd, 0, color=_LOSS, alpha=0.5)
    ax2.set_ylabel("Drawdown (USD)")
    fig.tight_layout()
    return fig


def chart_fail_type_breakdown(classified: pd.DataFrame, strategy_id: str, version: str) -> plt.Figure:
    counts = classified["fail_type"].value_counts()
    colors = [FAIL_COLORS.get(k, _NEUTRAL) for k in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, str(int(bar.get_height())), ha="center", fontsize=9)
    ax.set_title(f"Fail Pattern Breakdown — {_label(strategy_id, version)}")
    ax.set_ylabel("# Trades")
    fig.tight_layout()
    return fig


def chart_mfe_distribution(classified: pd.DataFrame, strategy_id: str, version: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    for ftype, group in classified.groupby("fail_type"):
        ax.hist(group["mfe_pct"], bins=20, alpha=0.65, label=ftype, color=FAIL_COLORS.get(ftype, _NEUTRAL))
    ax.set_title(f"MFE% Distribution of Losses — {_label(strategy_id, version)}")
    ax.set_xlabel("MFE %")
    ax.legend()
    fig.tight_layout()
    return fig


def chart_mae_vs_mfe(classified: pd.DataFrame, strategy_id: str, version: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for ftype, group in classified.groupby("fail_type"):
        ax.scatter(group["mfe_pct"], group["mae_pct"], label=ftype, color=FAIL_COLORS.get(ftype, _NEUTRAL), alpha=0.6, s=30)
    ax.set_xlabel("MFE %")
    ax.set_ylabel("MAE %")
    ax.set_title(f"MAE vs MFE for Losses — {_label(strategy_id, version)}")
    ax.legend()
    fig.tight_layout()
    return fig


def chart_entry_slot_30m_winrate(slot_stats: dict, strategy_id: str, version: str) -> plt.Figure:
    slots = ENTRY_SLOTS_30M
    values = [slot_stats[s]["win_rate_pct"] for s in slots]
    colors = [_WIN if (v is not None and v >= 50) else _LOSS if v is not None else _NEUTRAL for v in values]
    plot_values = [v if v is not None else 0 for v in values]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(range(len(slots)), plot_values, color=colors, edgecolor="white")
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(range(0, len(slots), 2))
    ax.set_xticklabels([slots[i] for i in range(0, len(slots), 2)], rotation=60, ha="right", fontsize=7)
    ax.set_title(f"Win Rate by 30-Minute Entry Slot (Asia/Taipei) — {_label(strategy_id, version)}")
    ax.set_ylabel("Win Rate %")
    fig.tight_layout()
    return fig


def chart_pre_entry_slot_30m(profile_entry_slot: dict, strategy_id: str, version: str) -> plt.Figure:
    slots = ENTRY_SLOTS_30M
    imm = [profile_entry_slot[s]["immediate_loss"] for s in slots]
    allt = [profile_entry_slot[s]["all_trades"] for s in slots]
    x = np.arange(len(slots))
    width = 0.4
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(x - width / 2, imm, width, label="immediate_loss", color=_LOSS, alpha=0.8)
    ax.bar(x + width / 2, allt, width, label="all trades", color=_BLUE, alpha=0.6)
    ax.set_xticks(range(0, len(slots), 2))
    ax.set_xticklabels([slots[i] for i in range(0, len(slots), 2)], rotation=60, ha="right", fontsize=7)
    ax.set_title(f"Entry 30-Min Slot — immediate_loss vs all trades — {_label(strategy_id, version)}")
    ax.set_ylabel("Proportion")
    ax.legend()
    fig.tight_layout()
    return fig


def chart_pre_entry_categorical(dist: dict, title_suffix: str, strategy_id: str, version: str, xlabel: str = "") -> plt.Figure:
    labels = list(dist.keys())
    imm = [dist[k]["immediate_loss"] for k in labels]
    allt = [dist[k]["all_trades"] for k in labels]
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, imm, width, label="immediate_loss", color=_LOSS, alpha=0.8)
    ax.bar(x + width / 2, allt, width, label="all trades", color=_BLUE, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"{title_suffix} — {_label(strategy_id, version)}")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Proportion")
    ax.legend()
    fig.tight_layout()
    return fig


def chart_kbar_features(enriched: pd.DataFrame, strategy_id: str, version: str) -> plt.Figure:
    df = enriched.dropna(subset=["rsi"])
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No K-bar data available for this coverage window.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"K-Bar Features at Entry — {_label(strategy_id, version)}")
        ax.axis("off")
        fig.tight_layout()
        return fig
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(df["rsi"], bins=15, color=_LOSS, edgecolor="white")
    for v, c in [(30, "green"), (50, "gray"), (70, "orange")]:
        axes[0].axvline(v, color=c, linestyle="--", linewidth=1)
    axes[0].set_title("RSI at Entry")
    axes[1].hist(df["momentum_3"], bins=15, color=_NEUTRAL, edgecolor="white")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("3-Bar Momentum % Before Entry")
    counts = df["prev_3_green"].value_counts().sort_index()
    axes[2].bar(counts.index.astype(str), counts.values, color=_BLUE, edgecolor="white")
    axes[2].set_title("Green Bars in Last 3 Before Entry")
    fig.suptitle(f"K-Bar Context at Immediate-Loss Entry ({len(df)} trades) — {_label(strategy_id, version)}")
    fig.tight_layout()
    return fig


def chart_bb_zone_winrate(bb_zone_stats: dict, strategy_id: str, version: str) -> plt.Figure:
    zones = [z for z in BB_ZONE_ORDER if bb_zone_stats.get(z, {}).get("n")]
    wr = [bb_zone_stats[z]["win_rate_pct"] for z in zones]
    totals = [bb_zone_stats[z]["n"] for z in zones]
    colors = [_WIN if w >= 60 else _BLUE if w >= 45 else _LOSS for w in wr]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(zones)), wr, color=colors)
    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels([z.replace("_", " ") for z in zones], rotation=30, ha="right", fontsize=9)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Win Rate %")
    ax.set_title(f"Win Rate by BB Zone at Entry — {_label(strategy_id, version)}")
    for i, (w, n) in enumerate(zip(wr, totals)):
        ax.text(i, w + 1.5, f"{w}%\n(n={n})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def chart_dxy_winrate(dxy_stats: dict, strategy_id: str, version: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("by_bucket", "Win Rate by DXY RSI Bucket"), ("by_trend", "Win Rate: DXY Trend"), ("by_momentum", "Win Rate: DXY RSI vs MA")]
    for ax, (key, title) in zip(axes, panels):
        df = dxy_stats.get(key, {})
        if not df:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        labels = list(df.keys())
        wr = [df[k]["win_rate_pct"] or 0 for k in labels]
        totals = [df[k]["n"] for k in labels]
        colors = [_WIN if w >= 50 else _LOSS for w in wr]
        ax.bar(range(len(labels)), wr, color=colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.axhline(50, color="gray", linewidth=1, linestyle="--")
        ax.set_ylim(0, 100)
        ax.set_title(title)
        for i, (w, n) in enumerate(zip(wr, totals)):
            ax.text(i, w + 2, f"{w}%\nn={n}", ha="center", fontsize=8)
    fig.suptitle(f"DXY Context vs Win Rate — {_label(strategy_id, version)}")
    fig.tight_layout()
    return fig


def chart_dxy_correlation(corr_df: pd.DataFrame, strategy_id: str, version: str, market: str = "XAUUSD") -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(13, 3.5))
    ax1.plot(corr_df["time"], corr_df["rolling_corr"], color=_DXY_COLOR, linewidth=1.2)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Rolling Correlation")
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_title(f"DXY × {market} 30-Day Rolling Return Correlation — {_label(strategy_id, version)}")
    fig.tight_layout()
    return fig


def chart_htf_alignment(htf_stats_out: dict, strategy_id: str, version: str) -> plt.Figure:
    df = htf_stats_out.get("by_alignment", {})
    fig, ax = plt.subplots(figsize=(8, 4))
    if not df:
        ax.text(0.5, 0.5, "No HTF data available", ha="center", transform=ax.transAxes)
        return fig
    labels = list(df.keys())
    wr = [df[k]["win_rate_pct"] or 0 for k in labels]
    totals = [df[k]["n"] for k in labels]
    bars = ax.bar(labels, wr, color=_BLUE, edgecolor="white")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    for bar, n, w in zip(bars, totals, wr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"n={n}\n{w}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Win Rate by HTF Alignment — {_label(strategy_id, version)}")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    return fig


def chart_htf_4h_state(htf_stats_out: dict, strategy_id: str, version: str) -> plt.Figure:
    df = htf_stats_out.get("by_4h_state", {})
    fig, ax = plt.subplots(figsize=(7, 4))
    if not df:
        ax.text(0.5, 0.5, "No 4H data available", ha="center", transform=ax.transAxes)
        return fig
    order = [s for s in ["bearish", "neutral", "bullish"] if s in df]
    colors = {"bearish": "#e74c3c", "neutral": "#f39c12", "bullish": "#27ae60"}
    wr = [df[s]["win_rate_pct"] or 0 for s in order]
    bars = ax.bar(order, wr, color=[colors[s] for s in order], edgecolor="white")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    for bar, s in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"n={df[s]['n']}\n{df[s]['win_rate_pct']}%", ha="center", fontsize=8)
    ax.set_title(f"Win Rate by 4H RSI State — {_label(strategy_id, version)}")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    return fig


def chart_htf_bucket_heatmap(htf_stats_out: dict, strategy_id: str, version: str) -> plt.Figure:
    df = htf_stats_out.get("by_4h_bucket", {})
    fig, ax = plt.subplots(figsize=(10, 3))
    order = ["oversold(<30)", "low(30-50)", "high(50-70)", "overbought(>70)"]
    labels = [b for b in order if b in df]
    if not labels:
        ax.text(0.5, 0.5, "No 4H bucket data available", ha="center", transform=ax.transAxes)
        return fig
    wr = np.array([df[b]["win_rate_pct"] or 0 for b in labels]) / 100
    im = ax.imshow([wr], cmap="RdYlGn", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yticks([])
    for i, b in enumerate(labels):
        ax.text(i, 0, f"{df[b]['win_rate_pct']}%\nn={df[b]['n']}", ha="center", va="center", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Win Rate")
    ax.set_title(f"Win Rate by 4H RSI Bucket — {_label(strategy_id, version)}")
    fig.tight_layout()
    return fig


def chart_hold_time_dist(trades: pd.DataFrame, strategy_id: str, version: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    wins = trades[trades["result"] == "win"]["hold_bars"]
    losses = trades[trades["result"] == "loss"]["hold_bars"]
    bins = range(0, int(trades["hold_bars"].max()) + 2, 4)
    ax.hist(wins, bins=bins, alpha=0.65, label="Win", color=_WIN)
    ax.hist(losses, bins=bins, alpha=0.65, label="Loss", color=_LOSS)
    ax.set_title(f"Hold Time Distribution (30-min bars) — {_label(strategy_id, version)}")
    ax.legend()
    fig.tight_layout()
    return fig


def chart_consecutive_losses(streaks: pd.Series, strategy_id: str, version: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    if streaks.empty:
        ax.text(0.5, 0.5, "No losing streaks", ha="center", va="center", transform=ax.transAxes)
        return fig
    max_val = int(streaks.max())
    ax.hist(streaks, bins=range(1, max_val + 2), align="left", color=_LOSS, edgecolor="white")
    ax.set_title(f"Consecutive Loss Streaks — {_label(strategy_id, version)}")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fig_to_b64(fig: plt.Figure) -> str:
    return base64.b64encode(fig_to_png_bytes(fig)).decode()


# ---------------------------------------------------------------------------
# 9. JSON-safe serialization
# ---------------------------------------------------------------------------

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj
