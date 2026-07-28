#!/usr/bin/env python3
"""Public reproduction method for the section-5 fail-pattern report contract plus the
section-13.2 live-impact rank score (docs/RESEARCH_DEVELOPMENT_SPEC.md, private repo).

Raw TradingView CSVs are intentionally not included. Supply locally authorized CSV
paths for the trade export, 30m/60m/4H/1D XAUUSD, 1D DXY, and the three Macro daily
series (US10Y, T10YIE breakeven inflation, VIX). Reproduces every published numeric
field: baseline (including drawdown/streak/hold-bar summary), trade_period,
fail-pattern breakdown, 30-minute timing, immediate-loss pre-entry profile, K-bar
coverage, BB zone, DXY regime plus rolling correlation, full MTF alignment
(by_alignment/by_4h_state/by_4h_bucket/by_1d_state/by_conflict/by_vol_regime/
coverage), hold-time/streak distribution, the Macro composite verdict, and the integer
rank_score carried by by_entry_30m/by_session/by_macro_verdict. Chart PNGs are
published as pre-verified static files and are not regenerated here; the executor
manually reviewed the chart-generation code path (no file paths in any chart
title/axis) before publishing them — see the Private decision_log.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

IMMEDIATE_LOSS_MFE_PCT = 0.10
FALSE_BREAKOUT_MAE_MFE_RATIO = 2.0
TIME_BLEED_MIN_BARS = 24
ENTRY_SLOTS_30M = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
BB_ZONE_ORDER = ["below_lower", "near_lower", "lower_mid", "near_middle", "upper_mid", "near_upper", "above_upper"]
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
MACRO_MA_LENGTH = 50
MACRO_MAX_AGE = pd.Timedelta(days=4)

TRADE_COLUMN_ALIASES = {
    "trade_id": ["Trade number", "Trade #"], "type": ["Type"], "datetime": ["Date and time"],
    "signal": ["Signal"], "price": ["Price USD"], "size_qty": ["Size (qty)"],
    "net_pnl_usd": ["Net PnL USD", "Net P&L USD"], "return_pct": ["Return %", "Net P&L %"],
    "mfe_pct": ["Favorable excursion %"], "mae_pct": ["Adverse excursion %"],
}


def _pick(columns, canonical):
    for alias in TRADE_COLUMN_ALIASES[canonical]:
        if alias in columns:
            return alias
    raise KeyError(canonical)


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [c.lstrip("﻿").strip() for c in frame.columns]
    rename = {_pick(list(frame.columns), c): c for c in TRADE_COLUMN_ALIASES}
    frame = frame.rename(columns=rename)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    entries = frame[frame["type"] == "Entry long"][["trade_id", "datetime"]].rename(columns={"datetime": "entry_time"})
    exits = frame[frame["type"] == "Exit long"][
        ["trade_id", "datetime", "signal", "net_pnl_usd", "return_pct", "mfe_pct", "mae_pct"]
    ].rename(columns={"datetime": "exit_time", "signal": "exit_signal"})
    trades = entries.merge(exits, on="trade_id", how="inner")
    trades = trades[trades["exit_signal"].astype(str).str.upper() != "OPEN"].copy()
    trades["hold_bars"] = ((trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 1800).astype(int)
    trades["win"] = trades["net_pnl_usd"] > 0
    trades["result"] = np.select([trades["net_pnl_usd"] > 0, trades["net_pnl_usd"] < 0], ["win", "loss"], default="breakeven")
    trades["entry_slot_30m"] = trades["entry_time"].dt.strftime("%H:%M")
    trades["entry_dow"] = trades["entry_time"].dt.dayofweek
    minute = trades["entry_time"].dt.hour * 60 + trades["entry_time"].dt.minute
    trades["session"] = np.select(
        [minute.between(420, 899), minute.between(900, 1229), (minute >= 1230) | (minute < 60)],
        ["asia", "europe", "us"], default="overnight",
    )
    return trades.sort_values("entry_time").reset_index(drop=True)


def _wilder_rsi(close, period=14):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi, rsi.rolling(period).mean()


def load_price(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    tcol = next(c for c in raw.columns if "time" in c.lower())
    raw[tcol] = pd.to_datetime(raw[tcol], utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    raw = raw.rename(columns={tcol: "time", "RSI": "rsi", "RSI-based MA": "rsi_ma"})
    if "rsi" not in raw.columns:
        raw["rsi"], raw["rsi_ma"] = _wilder_rsi(raw["close"])
    keep = [c for c in ["time", "open", "high", "low", "close", "rsi", "rsi_ma"] if c in raw.columns]
    return raw[keep].sort_values("time").reset_index(drop=True)


def load_daily_macro(path: Path) -> pd.DataFrame:
    """Daily-series loader for the Macro composite only. Deliberately does NOT do the
    UTC->Asia/Taipei conversion load_price() does for intraday/DXY-regime data — the
    Macro composite's 4-day backward-asof join must see the CSV's own local daily
    timestamps unshifted, matching the Private toolkit's load_daily_macro()."""
    frame = pd.read_csv(path)
    frame.columns = [c.lstrip("﻿").strip() for c in frame.columns]
    tcol = next(c for c in frame.columns if "time" in c.lower())
    frame[tcol] = pd.to_datetime(frame[tcol], utc=False)
    if frame[tcol].dt.tz is not None:
        frame[tcol] = frame[tcol].dt.tz_localize(None)
    return (frame[[tcol, "close"]].rename(columns={tcol: "time"})
            .sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True))


def wilson(wins, count, z=1.96):
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
    gp = frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
    gl = abs(frame.loc[frame["net_pnl_usd"] < 0, "net_pnl_usd"].sum())
    return {
        "n": count, "wins": wins, "win_rate_pct": round(100 * wins / count, 2), "win_rate_ci95_pct": wilson(wins, count),
        "profit_factor": round(gp / gl, 3) if gl else None, "net_pnl_usd": round(float(frame["net_pnl_usd"].sum()), 2),
        "avg_pnl_usd": round(float(frame["net_pnl_usd"].mean()), 2), "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
    }


def grouped(frame, col):
    return {str(k): stats(g) for k, g in frame.groupby(col, observed=True)}


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


def summary(trades: pd.DataFrame) -> dict:
    base = stats(trades)
    return {
        **base,
        "max_drawdown_usd": round(float(max_drawdown(trades)), 2),
        "max_consecutive_losses": int(consecutive_losses(trades).max()) if len(trades) else 0,
        "avg_hold_bars": round(float(trades["hold_bars"].mean()), 2) if len(trades) else None,
    }


def classify_fail(trades: pd.DataFrame) -> pd.DataFrame:
    losses = trades[trades["result"] == "loss"].copy()
    ratio = losses["mae_pct"] / losses["mfe_pct"].replace(0, np.nan)
    conds = [losses["mfe_pct"] < IMMEDIATE_LOSS_MFE_PCT, ratio > FALSE_BREAKOUT_MAE_MFE_RATIO, losses["hold_bars"] >= TIME_BLEED_MIN_BARS]
    losses["fail_type"] = np.select(conds, ["immediate_loss", "false_breakout", "time_bleed"], default="normal_sl")
    return losses.reset_index(drop=True)


def fail_by_session(classified: pd.DataFrame) -> dict:
    table = pd.crosstab(classified["session"], classified["fail_type"])
    return {session: {ft: int(v) for ft, v in row.items()} for session, row in table.iterrows()}


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


def _distribution_pair(imm, all_trades, column, index_labels=None):
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
    entry_slot.update(_distribution_pair(imm, trades_ctx, "entry_slot_30m"))
    return {
        "entry_slot_30m": entry_slot,
        "entry_dow": _distribution_pair(imm, trades_ctx, "entry_dow", DOW_LABELS),
        "prev_result": _distribution_pair(imm, trades_ctx, "prev_result"),
        "trades_since_win": _distribution_pair(imm, trades_ctx, "trades_since_win"),
    }


def _kbar_features_at(price: pd.DataFrame, entry_time, n_lookback=3):
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


def enrich_with_kbars(classified: pd.DataFrame, price: pd.DataFrame, n_lookback=3) -> pd.DataFrame:
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


def compute_bb(price: pd.DataFrame, period=20, std_mult=2.0) -> pd.DataFrame:
    df = price.copy()
    df["bb_mid"] = df["close"].rolling(period, min_periods=period).mean()
    df["bb_std"] = df["close"].rolling(period, min_periods=period).std(ddof=1)
    df["bb_upper"], df["bb_lower"] = df["bb_mid"] + std_mult * df["bb_std"], df["bb_mid"] - std_mult * df["bb_std"]
    rng = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = np.where(rng != 0, (df["close"] - df["bb_lower"]) / rng, np.nan)
    df.loc[df["bb_mid"].isna(), "bb_pct_b"] = np.nan
    return df


def bb_zone(pct_b):
    if pct_b is None or (isinstance(pct_b, float) and np.isnan(pct_b)):
        return "unknown"
    if 0.8 <= pct_b <= 1.0:
        return "near_upper"
    for lo, hi, label in [(-np.inf, 0.0, "below_lower"), (0.0, 0.2, "near_lower"), (0.2, 0.4, "lower_mid"),
                           (0.4, 0.6, "near_middle"), (0.6, 0.8, "upper_mid"), (1.0, np.inf, "above_upper")]:
        if lo <= pct_b < hi:
            return label
    return "unknown"


def enrich_bb(trades, price):
    bb = compute_bb(price)
    lookup = bb[["time", "bb_pct_b"]].sort_values("time").reset_index(drop=True)
    merged = pd.merge_asof(trades.sort_values("entry_time"), lookup, left_on="entry_time", right_on="time", direction="backward")
    merged["bb_zone"] = merged["bb_pct_b"].apply(bb_zone)
    return merged


def enrich_dxy(trades, dxy_1d):
    dxy = dxy_1d.copy()
    dxy["sma20"] = dxy["close"].rolling(20, min_periods=1).mean()
    dxy["date"] = dxy["time"].dt.normalize()
    lookup = dxy[["date", "rsi", "rsi_ma", "close", "sma20"]].sort_values("date").reset_index(drop=True)
    result = trades.sort_values("entry_time").copy()
    result["date"] = result["entry_time"].dt.normalize()
    merged = pd.merge_asof(result, lookup, on="date", direction="backward")
    merged["dxy_rsi_bucket"] = np.select(
        [merged["rsi"] < 30, merged["rsi"] < 50, merged["rsi"] < 70],
        ["oversold(<30)", "neutral_low(30-50)", "neutral_high(50-70)"], default="overbought(>70)",
    )
    merged.loc[merged["rsi"].isna(), "dxy_rsi_bucket"] = "unknown"
    merged["dxy_trend_1d"] = np.where(merged["close"] > merged["sma20"], "up", "down")
    dxy_rsi_vs_ma = merged["rsi"] - merged["rsi_ma"]
    merged["dxy_momentum"] = np.where(
        dxy_rsi_vs_ma.isna(), "unknown",
        np.where(dxy_rsi_vs_ma > 0, "RSI>MA (USD gaining)", "RSI<MA (USD losing)"),
    )
    return merged


def _compute_atr(df, period=14):
    d = df.copy()
    prev_c = d["close"].shift(1)
    tr = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - prev_c).abs(), (d["low"] - prev_c).abs()))
    d["atr"] = tr.rolling(period, min_periods=1).mean()
    d["atr_sma20"] = d["atr"].rolling(20, min_periods=1).mean()
    d["vol_ratio"] = d["atr"] / d["atr_sma20"].replace(0, np.nan)
    return d


def _rsi_state(rsi, rsi_ma, slope):
    if pd.isna(rsi) or pd.isna(rsi_ma):
        return "unknown"
    above, rising = rsi > rsi_ma, (not pd.isna(slope)) and slope > 0
    if above and rising:
        return "bullish"
    if (not above) and (not rising):
        return "bearish"
    return "neutral"


def _rsi_bucket(rsi):
    if pd.isna(rsi):
        return "unknown"
    if rsi < 30:
        return "oversold(<30)"
    if rsi < 50:
        return "low(30-50)"
    if rsi < 70:
        return "high(50-70)"
    return "overbought(>70)"


def _enrich_tf(trades, price, prefix):
    p = _compute_atr(price.sort_values("time").reset_index(drop=True))
    p["_slope"] = p["rsi_ma"].diff(3) if "rsi_ma" in p.columns else np.nan
    p = p.rename(columns={"rsi": f"{prefix}_rsi", "rsi_ma": f"{prefix}_rsi_ma", "_slope": f"{prefix}_slope",
                           "vol_ratio": f"{prefix}_vol_ratio", "time": "entry_time"})
    keep = ["entry_time", f"{prefix}_rsi", f"{prefix}_rsi_ma", f"{prefix}_slope", f"{prefix}_vol_ratio"]
    p = p[[c for c in keep if c in p.columns]]
    merged = pd.merge_asof(trades.sort_values("entry_time"), p, on="entry_time", direction="backward")
    rsi_col, ma_col, slope_col = f"{prefix}_rsi", f"{prefix}_rsi_ma", f"{prefix}_slope"
    merged[f"{prefix}_rsi_state"] = merged.apply(
        lambda r: _rsi_state(r.get(rsi_col), r.get(ma_col), r.get(slope_col)), axis=1
    )
    merged[f"{prefix}_rsi_bucket"] = merged[rsi_col].apply(_rsi_bucket)
    return merged.drop(columns=[slope_col], errors="ignore")


def enrich_htf(trades, price_60m, price_4h, price_1d):
    result = trades.copy()
    for prefix, price in [("htf_60m", price_60m), ("htf_4h", price_4h), ("htf_1d", price_1d)]:
        result = _enrich_tf(result, price, prefix)
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
    out = {"by_alignment": grouped(enriched, "htf_alignment_label")}
    for col, key in [("htf_4h_rsi_state", "by_4h_state"), ("htf_4h_rsi_bucket", "by_4h_bucket"), ("htf_1d_rsi_state", "by_1d_state")]:
        if col in enriched.columns:
            valid = enriched[enriched[col] != "unknown"]
            out[key] = grouped(valid, col)
    labeled = enriched.copy()
    labeled["_conflict_label"] = labeled["htf_conflict"].map({True: "counter-trend (4H bearish)", False: "aligned (4H not bearish)"})
    out["by_conflict"] = grouped(labeled, "_conflict_label")
    labeled["_vol_label"] = labeled["htf_high_vol"].map({True: "high-vol (ATR>1.3xSMA)", False: "normal vol"})
    out["by_vol_regime"] = grouped(labeled, "_vol_label")
    coverage = {}
    for col, label in [("htf_60m_rsi", "60m"), ("htf_4h_rsi", "4H"), ("htf_1d_rsi", "1D")]:
        if col in enriched.columns:
            n_with = int(enriched[col].notna().sum())
            coverage[label] = {"trades_covered": n_with, "total_trades": len(enriched),
                                "coverage_pct": round(100 * n_with / len(enriched), 1) if len(enriched) else 0.0}
    out["coverage"] = coverage
    return out


def build_macro_composite(paths: dict[str, Path]) -> pd.DataFrame:
    """paths keys: us10y, t10yie, dxy, vix, gold — each a daily OHLC(V) CSV path,
    loaded independently through load_daily_macro (not the tz-shifted load_price)."""
    series = {name: load_daily_macro(path).rename(columns={"close": name}) for name, path in paths.items()}
    base = series.pop("us10y")
    for frame in series.values():
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
    return pd.merge_asof(
        left, right[["macro_time", "macro_score", "macro_verdict"]],
        left_on="entry_time", right_on="macro_time", direction="backward", tolerance=MACRO_MAX_AGE,
    )


def macro_coverage(joined: pd.DataFrame) -> dict:
    matched = joined["macro_score"].notna().sum()
    total = len(joined)
    return {"matched": int(matched), "unmatched": int(total - matched),
            "pct": round(100 * matched / total, 2) if total else 0.0}


def dxy_correlation_stats(xauusd_1d: pd.DataFrame, dxy_1d: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    dxy = dxy_1d[["time", "close"]].rename(columns={"close": "dxy_close"}).copy()
    xau = xauusd_1d[["time", "close"]].rename(columns={"close": "xau_close"}).copy()
    dxy["dxy_ret"] = dxy["dxy_close"].pct_change()
    xau["xau_ret"] = xau["xau_close"].pct_change()
    merged = pd.merge(dxy[["time", "dxy_ret"]], xau[["time", "xau_ret"]], on="time", how="inner").dropna()
    merged["rolling_corr"] = merged["dxy_ret"].rolling(window).corr(merged["xau_ret"])
    return merged


def _tie_collapse(groups: dict, ranked: list[str], band_score: dict[str, int]) -> None:
    by_wr: dict[float, list[str]] = {}
    for key in ranked:
        by_wr.setdefault(groups[key]["win_rate_pct"], []).append(key)
    for members in by_wr.values():
        floor = min(band_score[m] for m in members)
        for m in members:
            groups[m]["rank_score"] = floor


def compute_rank_scores(groups: dict, ordered_keys: list[str] | None = None) -> dict:
    """docs/RESEARCH_DEVELOPMENT_SPEC.md section 13.2: integer rank score by win-rate
    ranking. Mutates and returns `groups`."""
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
    for key in keys:
        groups[key]["rank_score"] = 0
        groups[key]["low_sample"] = False
        groups[key]["rank_excluded_reason"] = "descriptive_only"
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--price-30m", type=Path, required=True)
    parser.add_argument("--price-60m", type=Path, required=True)
    parser.add_argument("--price-4h", type=Path, required=True)
    parser.add_argument("--price-1d", type=Path, required=True)
    parser.add_argument("--dxy-1d", type=Path, required=True)
    parser.add_argument("--macro-us10y", type=Path, required=True)
    parser.add_argument("--macro-t10yie", type=Path, required=True)
    parser.add_argument("--macro-vix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trades = load_trades(args.trades)
    price_30m, price_60m, price_4h, price_1d = (load_price(p) for p in [args.price_30m, args.price_60m, args.price_4h, args.price_1d])
    dxy_1d = load_price(args.dxy_1d)

    baseline = summary(trades)
    classified = classify_fail(trades)
    total_losses = len(classified)
    fail_by_type = {
        name: {"count": int(count), "pct": round(100 * count / total_losses, 1) if total_losses else 0.0}
        for name, count in classified["fail_type"].value_counts().items()
    }
    by_session = grouped(trades, "session")
    by_entry_30m = {slot: stats(trades[trades["entry_slot_30m"] == slot]) for slot in ENTRY_SLOTS_30M}

    trades_ctx = add_trade_context(trades, classified)
    profile = immediate_loss_profile(trades_ctx)

    kbar_enriched = enrich_with_kbars(classified, price_30m)
    coverage = kbar_coverage(kbar_enriched)

    bb_enriched = enrich_bb(trades, price_30m)
    valid_bb = bb_enriched[bb_enriched["bb_zone"] != "unknown"]
    bb_zone_stats = {z: stats(valid_bb[valid_bb["bb_zone"] == z]) for z in BB_ZONE_ORDER}

    dxy_enriched = enrich_dxy(trades, dxy_1d)
    dxy_regime = {
        "by_bucket": grouped(dxy_enriched[dxy_enriched["dxy_rsi_bucket"] != "unknown"], "dxy_rsi_bucket"),
        "by_trend": grouped(dxy_enriched, "dxy_trend_1d"),
        "by_momentum": grouped(dxy_enriched[dxy_enriched["dxy_momentum"] != "unknown"], "dxy_momentum"),
    }
    corr = dxy_correlation_stats(price_1d, dxy_1d)
    avg_30d_correlation = round(float(corr["rolling_corr"].dropna().mean()), 3)
    dxy_stats = {"regime": dxy_regime, "avg_30d_correlation": avg_30d_correlation}

    htf_enriched = enrich_htf(trades, price_60m, price_4h, price_1d)
    mtf_stats = htf_stats(htf_enriched)

    macro = build_macro_composite({
        "us10y": args.macro_us10y, "t10yie": args.macro_t10yie,
        "dxy": args.dxy_1d, "vix": args.macro_vix, "gold": args.price_1d,
    })
    macro_joined = attach_macro(trades, macro)
    macro_matched = macro_joined.dropna(subset=["macro_score"]).copy()
    by_macro_verdict = grouped(macro_matched, "macro_verdict")
    macro_matched["macro_score"] = macro_matched["macro_score"].astype(int)
    by_macro_score = grouped(macro_matched, "macro_score")

    compute_rank_scores(by_entry_30m, ordered_keys=ENTRY_SLOTS_30M)
    compute_rank_scores(by_session, ordered_keys=["asia", "europe", "us"])
    mark_descriptive_only(by_session, [k for k in ["overnight"] if k in by_session])
    compute_rank_scores(by_macro_verdict, ordered_keys=list(by_macro_verdict.keys()))

    output = {
        "baseline": baseline,
        "trade_period": {"start": str(trades["entry_time"].min()), "end": str(trades["exit_time"].max())},
        "fail_pattern": {"total_losses": total_losses, "by_type": fail_by_type, "by_session": fail_by_session(classified)},
        "by_session": by_session,
        "by_entry_30m": by_entry_30m,
        "immediate_loss_profile": profile,
        "kbar_coverage": coverage,
        "bb_zone": bb_zone_stats,
        "dxy": dxy_stats,
        "mtf": mtf_stats,
        "hold_time_streaks": {
            "avg_hold_bars": baseline["avg_hold_bars"],
            "max_consecutive_losses": baseline["max_consecutive_losses"],
            "streak_lengths": consecutive_losses(trades).tolist(),
        },
        "macro_period": {"start": str(macro["time"].min()), "end": str(macro["time"].max())},
        "macro_coverage": macro_coverage(macro_joined),
        "by_macro_verdict": by_macro_verdict,
        "by_macro_score": by_macro_score,
        "method": {
            "macro_score": "real_rate<MA50 +2; US10Y<MA50 +1; DXY<MA50 +1; VIX>MA50 +1; XAUUSD>MA50 +1",
            "macro_labels": "WAIT=0-2; NEUTRAL=3-4; STRONG BUY=5-6",
            "macro_assignment": "latest prior daily observation, maximum age 4 days",
            "scoring_method": "integer rank score, docs/RESEARCH_DEVELOPMENT_SPEC.md section 13.2",
        },
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
