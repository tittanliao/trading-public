#!/usr/bin/env python3
"""Public reproduction method for the section-5 fail-pattern report contract
(docs/RESEARCH_DEVELOPMENT_SPEC.md, private repo).

Raw TradingView CSVs are intentionally not included. Supply locally authorized CSV
paths for the trade export plus 30m/60m/4H/1D XAUUSD and 1D DXY. Reproduces every
published numeric field (baseline, fail-pattern breakdown, 30-minute timing,
immediate-loss pre-entry profile, K-bar coverage, BB zone, DXY regime, MTF alignment).
Chart PNGs are published as pre-verified static files and are not regenerated here;
the executor manually reviewed the chart-generation code path (no file paths in any
chart title/axis) before publishing them — see the Private decision_log.md.
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


def classify_fail(trades: pd.DataFrame) -> pd.DataFrame:
    losses = trades[trades["result"] == "loss"].copy()
    ratio = losses["mae_pct"] / losses["mfe_pct"].replace(0, np.nan)
    conds = [losses["mfe_pct"] < IMMEDIATE_LOSS_MFE_PCT, ratio > FALSE_BREAKOUT_MAE_MFE_RATIO, losses["hold_bars"] >= TIME_BLEED_MIN_BARS]
    losses["fail_type"] = np.select(conds, ["immediate_loss", "false_breakout", "time_bleed"], default="normal_sl")
    return losses.reset_index(drop=True)


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


def _enrich_tf(trades, price, prefix):
    p = _compute_atr(price.sort_values("time").reset_index(drop=True))
    p["_slope"] = p["rsi_ma"].diff(3) if "rsi_ma" in p.columns else np.nan
    p = p.rename(columns={"rsi": f"{prefix}_rsi", "rsi_ma": f"{prefix}_rsi_ma", "_slope": f"{prefix}_slope", "time": "entry_time"})
    keep = ["entry_time", f"{prefix}_rsi", f"{prefix}_rsi_ma", f"{prefix}_slope"]
    p = p[[c for c in keep if c in p.columns]]
    merged = pd.merge_asof(trades.sort_values("entry_time"), p, on="entry_time", direction="backward")
    merged[f"{prefix}_rsi_state"] = merged.apply(
        lambda r: _rsi_state(r.get(f"{prefix}_rsi"), r.get(f"{prefix}_rsi_ma"), r.get(f"{prefix}_slope")), axis=1
    )
    return merged.drop(columns=[f"{prefix}_slope"], errors="ignore")


def enrich_htf(trades, price_60m, price_4h, price_1d):
    result = trades.copy()
    for prefix, price in [("htf_60m", price_60m), ("htf_4h", price_4h), ("htf_1d", price_1d)]:
        result = _enrich_tf(result, price, prefix)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--price-30m", type=Path, required=True)
    parser.add_argument("--price-60m", type=Path, required=True)
    parser.add_argument("--price-4h", type=Path, required=True)
    parser.add_argument("--price-1d", type=Path, required=True)
    parser.add_argument("--dxy-1d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trades = load_trades(args.trades)
    price_30m, price_60m, price_4h, price_1d = (load_price(p) for p in [args.price_30m, args.price_60m, args.price_4h, args.price_1d])
    dxy_1d = load_price(args.dxy_1d)

    baseline = stats(trades)
    classified = classify_fail(trades)
    total_losses = len(classified)
    fail_by_type = {
        name: {"count": int(count), "pct": round(100 * count / total_losses, 1) if total_losses else 0.0}
        for name, count in classified["fail_type"].value_counts().items()
    }
    by_session = grouped(trades, "session")
    by_entry_30m = {slot: stats(trades[trades["entry_slot_30m"] == slot]) for slot in ENTRY_SLOTS_30M}

    bb_enriched = enrich_bb(trades, price_30m)
    valid_bb = bb_enriched[bb_enriched["bb_zone"] != "unknown"]
    bb_zone_stats = {z: stats(valid_bb[valid_bb["bb_zone"] == z]) for z in BB_ZONE_ORDER}

    dxy_enriched = enrich_dxy(trades, dxy_1d)
    dxy_stats = {
        "by_bucket": grouped(dxy_enriched[dxy_enriched["dxy_rsi_bucket"] != "unknown"], "dxy_rsi_bucket"),
        "by_trend": grouped(dxy_enriched, "dxy_trend_1d"),
    }

    htf_enriched = enrich_htf(trades, price_60m, price_4h, price_1d)
    mtf_stats = {}
    if "htf_4h_rsi_state" in htf_enriched.columns:
        valid = htf_enriched[htf_enriched["htf_4h_rsi_state"] != "unknown"]
        mtf_stats["by_4h_state"] = grouped(valid, "htf_4h_rsi_state")

    output = {
        "baseline": baseline,
        "fail_pattern": {"total_losses": total_losses, "by_type": fail_by_type},
        "by_session": by_session,
        "by_entry_30m": by_entry_30m,
        "bb_zone": bb_zone_stats,
        "dxy": dxy_stats,
        "mtf": mtf_stats,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
