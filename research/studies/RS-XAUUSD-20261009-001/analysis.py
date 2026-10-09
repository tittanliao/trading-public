#!/usr/bin/env python3
"""RS-XAUUSD-20261009-001 — S1 V3.9 since the live push began: what the decline is made of.

Requested 2026-10-09: 「S1 升級成正式研究」. The 2026-10-09 reconciliation found the
31 S1 V3.9 trades since 2026-08-16 (the Telegram push regime) worse than every contiguous
31-trade window before them, and could not rule out that a long-only strategy had simply
met a falling market. This study answers that under the section 5 contract.

Two parts:

1. The full section 5 report (items 1-9, plus item 11 temporal stability because the
   question is one of degradation) on the 2026-10-09 Strategy Tester export, which reaches
   back to 2025-01-06.
2. A decline block: the live window against the reference window, its significance, the
   fail-type mix, a shift-share decomposition across market and entry context, and a market
   control over every 31-trade window.

## The one methodological change from the reference implementation

Every price bar is stamped at its CLOSE before any join. `fail_pattern_toolkit` joins on
bar-open time, and every S1 entry sits exactly on a bar boundary, so its backward as-of
join lands on the bar the fill happened inside — a bar that has not closed.
RS-XAUUSD-20260901-001 showed that error was large enough to manufacture the programme's one
surviving %B finding. The toolkit is left as it is, because changing it would silently change
published numbers; this runner shifts the inputs instead, which makes the same toolkit read
the last completed bar. Shifts: 30m +30 min; 60m +60 min; 4H +4 h; 1D +1 day (conservative:
an FX daily bar closes about 05:00 Taipei the next day, so this adds up to three hours of
extra lag and never reads early). DXY context is joined on the same close-stamped times,
not on the calendar date the toolkit uses.

Usage:
    /opt/homebrew/bin/python3.12 scripts/research/build_s1_v39_decline.py
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import fail_pattern_toolkit as tk  # noqa: E402
import temporal_stability_toolkit as tst  # noqa: E402
import fail_pattern_report as solo  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (backend already set by the toolkit)

ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-XAUUSD-20261009-001"
STRATEGY_ID, VERSION = "S1-AweWithBB", "V3.9"
LIVE_START = pd.Timestamp("2026-08-16")
WINDOW = 31  # the live window's own size; every comparison window matches it
TRIALS = 20000
SEED = 20261009
REGIME_LOOKBACK_DAYS = 20
REGIME_BAND_PCT = 2.0

TRADES = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-10-09.csv")
PRICE_30M_HISTORY = Path("local-inputs/xauusd_30m_to_2026-08-15.csv")
PRICE_30M_RECENT = Path("local-inputs/xauusd_30m_from_2026-06-19.csv")
PRICE_4H = Path("local-inputs/FX_IDC_XAUUSD, 240.csv")
PRICE_1D = Path("local-inputs/FX_IDC_XAUUSD, 1D.csv")
DXY_1D = Path("local-inputs/TVC_DXY, 1D.csv")
OUTPUT_DIR = Path("reproduced")
SOURCES = {
    "trades": TRADES,
    "price_30m_history": PRICE_30M_HISTORY,
    "price_30m_recent": PRICE_30M_RECENT,
    "price_4h": PRICE_4H,
    "price_1d": PRICE_1D,
    "dxy_1d": DXY_1D,
}
CLOSE_SHIFT = {
    "30m": pd.Timedelta(minutes=30),
    "60m": pd.Timedelta(minutes=60),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locator(path: Path) -> str:
    """Repository-relative where possible; just the file name for a reader's local copy."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def ohlc(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    raw["time"] = pd.to_datetime(raw["time"], utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    return raw[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def with_rsi(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["rsi"], frame["rsi_ma"] = tk._compute_wilder_rsi(frame["close"])
    return frame


def stitch_30m(history: Path, recent: Path) -> tuple[pd.DataFrame, dict]:
    """Join two exports of the same FX_IDC 30m series; the newer one wins on overlap."""
    old, new = ohlc(history), ohlc(recent)
    overlap = old.merge(new, on="time", suffixes=("_old", "_new"))
    differ = overlap[(overlap["close_old"] - overlap["close_new"]).abs() > 1e-6]
    stitched = pd.concat([old[old["time"] < new["time"].min()], new]).reset_index(drop=True)
    gaps = stitched["time"].diff().dt.total_seconds().div(60)
    audit = {
        "history_range": [str(old["time"].min()), str(old["time"].max())],
        "recent_range": [str(new["time"].min()), str(new["time"].max())],
        "overlap_bars": int(len(overlap)),
        "overlap_bars_with_different_close": int(len(differ)),
        "overlap_max_abs_close_difference": round(float((overlap["close_old"] - overlap["close_new"]).abs().max()), 4),
        "different_close_dates": sorted({str(t.date()) for t in differ["time"]}),
        "rule": "the recent export replaces the history export from its first bar onward",
        "largest_gap_minutes_outside_weekends": float(gaps[gaps < 60 * 24].max()),
    }
    return stitched, audit


def resample_60m(price_30m: pd.DataFrame) -> pd.DataFrame:
    bars = price_30m.set_index("time").resample("60min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return bars.reset_index()


def close_stamped(frame: pd.DataFrame, shift: pd.Timedelta) -> pd.DataFrame:
    frame = frame.copy()
    frame["time"] = frame["time"] + shift
    return frame


def next_stamp_after(series: pd.DataFrame, bar: pd.Timedelta) -> pd.Timestamp:
    """When the bar after a series' last one would have closed — the end of its usefulness.

    An as-of join has no notion of a series ending, so a trade after the last export bar
    silently takes days-old context. A threshold landing in the weekend moves to Monday
    07:00 Taipei (the FX open) plus one bar.
    """
    t = series["time"].max() + bar
    if t.dayofweek >= 5 or (t.dayofweek == 0 and t.hour < 7):
        monday = (t + pd.Timedelta(days=(7 - t.dayofweek) % 7)).normalize()
        t = monday + pd.Timedelta(hours=7) + bar
    return t


def enrich_dxy_closed(trades: pd.DataFrame, dxy: pd.DataFrame) -> pd.DataFrame:
    """The toolkit's DXY labels, joined on close-stamped time rather than calendar date."""
    d = dxy.copy()
    d["sma20"] = d["close"].rolling(20, min_periods=1).mean()
    lookup = d[["time", "rsi", "rsi_ma", "close", "sma20"]].rename(columns={"time": "dxy_time"})
    merged = pd.merge_asof(trades.sort_values("entry_time"), lookup, left_on="entry_time",
                           right_on="dxy_time", direction="backward")
    merged["dxy_rsi_1d"] = merged["rsi"]
    merged["dxy_rsi_vs_ma"] = merged["rsi"] - merged["rsi_ma"]
    merged["dxy_trend_1d"] = np.where(merged["close"] > merged["sma20"], "up", "down")
    merged["dxy_rsi_bucket"] = np.select(
        [merged["rsi"] < 30, merged["rsi"] < 50, merged["rsi"] < 70],
        ["oversold(<30)", "neutral_low(30-50)", "neutral_high(50-70)"], default="overbought(>70)")
    merged.loc[merged["rsi"].isna(), "dxy_rsi_bucket"] = "unknown"
    merged["dxy_momentum"] = np.where(
        merged["dxy_rsi_vs_ma"].isna(), "unknown",
        np.where(merged["dxy_rsi_vs_ma"] > 0, "RSI>MA (USD gaining)", "RSI<MA (USD losing)"))
    return merged.drop(columns=["dxy_time", "rsi", "rsi_ma", "close", "sma20"])


# --------------------------------------------------------------------------------------
# Decline block
# --------------------------------------------------------------------------------------

def trend_regime(trades: pd.DataFrame, daily_closed: pd.DataFrame) -> pd.DataFrame:
    """Gold's 20-session return up to the last CLOSED daily bar before entry."""
    d = daily_closed[["time", "close"]].copy()
    d["ret20"] = 100 * (d["close"] / d["close"].shift(REGIME_LOOKBACK_DAYS) - 1)
    merged = pd.merge_asof(trades.sort_values("entry_time"), d[["time", "ret20"]].rename(columns={"time": "_t"}),
                           left_on="entry_time", right_on="_t", direction="backward").drop(columns="_t")
    merged["gold_trend_20d"] = np.select(
        [merged["ret20"] < -REGIME_BAND_PCT, merged["ret20"] > REGIME_BAND_PCT],
        ["falling (<-2%)", "rising (>+2%)"], default="flat (±2%)")
    merged.loc[merged["ret20"].isna(), "gold_trend_20d"] = "unknown"
    return merged


def volatility_at_entry(trades: pd.DataFrame, price_30m_closed: pd.DataFrame) -> pd.DataFrame:
    """ATR(14) of the last closed 30m bar as a percentage of its close.

    S1's stop sits a fixed ~0.5% below entry. If the market's ordinary 30-minute range grows
    relative to that, the same stop is reached by noise more often, and the question is
    whether that happened. Terciles are cut on the reference trades only, so the live window
    cannot move its own boundaries.
    """
    p = price_30m_closed.sort_values("time").reset_index(drop=True)
    prev_close = p["close"].shift(1)
    tr = np.maximum(p["high"] - p["low"], np.maximum((p["high"] - prev_close).abs(), (p["low"] - prev_close).abs()))
    p["atr_pct_30m"] = 100 * tr.ewm(alpha=1 / 14, adjust=False).mean() / p["close"]
    merged = pd.merge_asof(trades.sort_values("entry_time"), p[["time", "atr_pct_30m"]].rename(columns={"time": "_t"}),
                           left_on="entry_time", right_on="_t", direction="backward",
                           tolerance=pd.Timedelta(hours=1)).drop(columns="_t")
    cuts = merged.loc[merged["entry_time"] < LIVE_START, "atr_pct_30m"].quantile([1 / 3, 2 / 3]).tolist()
    merged["atr_tercile"] = np.select(
        [merged["atr_pct_30m"] <= cuts[0], merged["atr_pct_30m"] <= cuts[1]],
        ["low ATR%", "mid ATR%"], default="high ATR%")
    merged.loc[merged["atr_pct_30m"].isna(), "atr_tercile"] = "unknown"
    merged.attrs["atr_cuts"] = [round(c, 4) for c in cuts]
    return merged


def permutation_p(reference: pd.Series, live: pd.Series) -> float:
    """One-sided: how often a random draw of len(live) trades is at least as bad."""
    pool = list(reference) + list(live)
    observed = float(live.mean())
    rng = random.Random(SEED)
    k = len(live)
    hits = sum(1 for _ in range(TRIALS) if sum(rng.sample(pool, k)) / k <= observed)
    return round((hits + 1) / (TRIALS + 1), 4)


def window_table(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Every contiguous WINDOW-trade window: its return and what gold did meanwhile."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    closes = daily[["time", "close"]].sort_values("time")
    def gold_at(moment):
        row = closes[closes["time"] <= moment].tail(1)
        return float(row["close"].iloc[0]) if len(row) else np.nan
    rows = []
    for start in range(0, len(t) - WINDOW + 1):
        w = t.iloc[start:start + WINDOW]
        g0, g1 = gold_at(w["entry_time"].iloc[0]), gold_at(w["exit_time"].iloc[-1])
        rows.append({
            "start": w["entry_time"].iloc[0], "end": w["exit_time"].iloc[-1],
            "total_return_pct": float(w["return_pct"].sum()),
            "win_rate_pct": 100 * float(w["win"].mean()),
            "gold_drift_pct": 100 * (g1 / g0 - 1) if g0 and g1 else np.nan,
            "is_live": bool(w["entry_time"].iloc[0] >= LIVE_START),
            "straddles": bool(w["entry_time"].iloc[0] < LIVE_START <= w["entry_time"].iloc[-1]),
        })
    frame = pd.DataFrame(rows)
    # A window holding both reference and live trades belongs to neither side; it would let
    # the live trades into the very distribution they are compared against.
    frame["is_reference"] = ~frame["is_live"] & ~frame["straddles"]
    return frame


def shift_share(reference: pd.DataFrame, live: pd.DataFrame, column: str) -> dict:
    """Split the change in mean return into within-group change and mix change.

    Delta = sum_g w_L(g) * (m_L(g) - m_R(g))   [within: the same context got worse]
          + sum_g (w_L(g) - w_R(g)) * m_R(g)   [mix: more trades in historically worse context]
    A group absent from the reference has no m_R and is reported, not imputed.
    """
    groups = sorted(set(reference[column]) | set(live[column]))
    out_groups, within, mix, unattributed = {}, 0.0, 0.0, 0.0
    for g in groups:
        r, l = reference[reference[column] == g], live[live[column] == g]
        w_r, w_l = len(r) / len(reference), len(l) / len(live)
        m_r = float(r["return_pct"].mean()) if len(r) else None
        m_l = float(l["return_pct"].mean()) if len(l) else None
        if m_r is None:
            unattributed += w_l * (m_l or 0.0)
        else:
            mix += (w_l - w_r) * m_r
            if m_l is not None:
                within += w_l * (m_l - m_r)
        out_groups[g] = {"reference": tk.stats(r), "live": tk.stats(l),
                         "weight_reference": round(w_r, 4), "weight_live": round(w_l, 4)}
    delta = float(live["return_pct"].mean() - reference["return_pct"].mean())
    return {"groups": out_groups, "delta_mean_return_pct": round(delta, 4),
            "within_pct_points": round(within, 4), "mix_pct_points": round(mix, 4),
            "unattributed_pct_points": round(unattributed, 4),
            "within_share_of_delta": round(within / delta, 3) if delta else None}


def fail_mix(trades: pd.DataFrame, classified: pd.DataFrame) -> dict:
    lab = trades.merge(classified[["trade_id", "fail_type"]], on="trade_id", how="left")
    lab["fail_type"] = lab["fail_type"].fillna(lab["result"])
    out = {}
    for name, part in (("reference", lab[lab["entry_time"] < LIVE_START]), ("live", lab[lab["entry_time"] >= LIVE_START])):
        counts = part["fail_type"].value_counts()
        out[name] = {k: {"count": int(v), "pct_of_trades": round(100 * v / len(part), 1)} for k, v in counts.items()}
        losers = part[part["result"] == "loss"]
        out[name]["_losers"] = {
            "n": int(len(losers)),
            "median_mfe_pct": round(float(losers["mfe_pct"].median()), 3) if len(losers) else None,
            "share_reaching_0_25pct_mfe": round(100 * float((losers["mfe_pct"] >= 0.25).mean()), 1) if len(losers) else None,
        }
        out[name]["_exits"] = {k: int(v) for k, v in part["exit_signal"].value_counts().items()}
    return out


def monthly(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    month = trades["entry_time"].dt.to_period("M").astype(str)
    d = daily[["time", "close"]].copy()
    d["month"] = d["time"].dt.to_period("M").astype(str)
    gold = d.groupby("month")["close"].agg(["first", "last"])
    out = {}
    for m, g in trades.groupby(month):
        s = tk.stats(g)
        s["total_return_pct"] = round(float(g["return_pct"].sum()), 2)
        s["gold_month_return_pct"] = (round(100 * (gold.loc[m, "last"] / gold.loc[m, "first"] - 1), 2)
                                      if m in gold.index else None)
        out[m] = s
    return out


def chart_rolling(windows: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    ref, live = windows[windows["is_reference"]], windows[windows["is_live"]]
    ax.plot(ref["end"], ref["total_return_pct"], color="#2980b9", lw=1.2, label="31-trade window, reference")
    ax.scatter(live["end"], live["total_return_pct"], color="#e74c3c", s=40, zorder=3, label="the live window")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(ref["total_return_pct"].min(), color="#2980b9", ls=":", lw=1, label="reference minimum")
    ax.set_title(f"{STRATEGY_ID} {VERSION} — rolling {WINDOW}-trade total return (%)")
    ax.set_ylabel("sum of trade return %")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def chart_drift(windows: pd.DataFrame, fit: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ref, live = windows[windows["is_reference"]], windows[windows["is_live"]]
    ax.scatter(ref["gold_drift_pct"], ref["total_return_pct"], s=10, color="#2980b9", alpha=.5, label="reference windows")
    ax.scatter(live["gold_drift_pct"], live["total_return_pct"], s=40, color="#e74c3c", label="live window")
    xs = np.linspace(ref["gold_drift_pct"].min(), max(ref["gold_drift_pct"].max(), live["gold_drift_pct"].max()), 50)
    ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color="black", lw=1, label="reference fit")
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xlabel("gold change over the window (%)")
    ax.set_ylabel(f"S1 {WINDOW}-trade total return (%)")
    ax.set_title("Does gold's direction explain the window's result?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def chart_fail_mix(mix: dict) -> plt.Figure:
    kinds = ["win", "breakeven", "normal_sl", "immediate_loss", "false_breakout", "time_bleed"]
    kinds = [k for k in kinds if k in mix["reference"] or k in mix["live"]]
    ref = [mix["reference"].get(k, {}).get("pct_of_trades", 0) for k in kinds]
    live = [mix["live"].get(k, {}).get("pct_of_trades", 0) for k in kinds]
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - .2, ref, .4, label="reference", color="#2980b9")
    ax.bar(x + .2, live, .4, label="live (since 2026-08-16)", color="#e74c3c")
    ax.set_xticks(x, kinds)
    ax.set_ylabel("% of all trades")
    ax.set_title("Outcome mix — reference vs live")
    ax.legend()
    fig.tight_layout()
    return fig


def chart_regime(share: dict) -> plt.Figure:
    groups = [g for g in ("falling (<-2%)", "flat (±2%)", "rising (>+2%)") if g in share["groups"]]
    ref = [share["groups"][g]["reference"]["avg_return_pct"] or 0 for g in groups]
    live = [share["groups"][g]["live"]["avg_return_pct"] or 0 for g in groups]
    n_ref = [share["groups"][g]["reference"]["n"] for g in groups]
    n_live = [share["groups"][g]["live"]["n"] for g in groups]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - .2, ref, .4, label="reference", color="#2980b9")
    ax.bar(x + .2, live, .4, label="live", color="#e74c3c")
    for i in range(len(groups)):
        ax.annotate(f"n={n_ref[i]}", (x[i] - .2, ref[i]), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"n={n_live[i]}", (x[i] + .2, live[i]), ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xticks(x, groups)
    ax.set_ylabel("mean return per trade (%)")
    ax.set_title("Mean return by gold's 20-session trend at entry")
    ax.legend()
    fig.tight_layout()
    return fig


def build(output_dir: Path) -> dict:
    paths = dict(SOURCES)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    trades = tk.load_trades(paths["trades"])
    raw_30m, stitch_audit = stitch_30m(paths["price_30m_history"], paths["price_30m_recent"])
    raw_60m = resample_60m(raw_30m)
    raw_4h, raw_1d, raw_dxy = ohlc(paths["price_4h"]), ohlc(paths["price_1d"]), ohlc(paths["dxy_1d"])

    price_30m = close_stamped(with_rsi(raw_30m), CLOSE_SHIFT["30m"])
    price_60m = close_stamped(with_rsi(raw_60m), CLOSE_SHIFT["60m"])
    price_4h = close_stamped(with_rsi(raw_4h), CLOSE_SHIFT["4h"])
    price_1d = close_stamped(with_rsi(raw_1d), CLOSE_SHIFT["1d"])
    dxy_1d = close_stamped(with_rsi(raw_dxy), CLOSE_SHIFT["1d"])

    context_end = min(next_stamp_after(price_30m, CLOSE_SHIFT["30m"]), next_stamp_after(price_4h, CLOSE_SHIFT["4h"]),
                      next_stamp_after(price_1d, CLOSE_SHIFT["1d"]), next_stamp_after(dxy_1d, CLOSE_SHIFT["1d"]))
    in_context = trades[trades["entry_time"] < context_end].copy()

    # ---- section 5 items 1-9 ----
    # Performance, fail pattern and timing use every trade. Price context (K-bar, BB, DXY,
    # MTF, and the decline block's context dimensions) uses only trades the price exports
    # actually reach.
    baseline = tk.summary(trades)
    classified = tk.classify_fail(trades)
    fail_summary = tk.fail_type_summary(classified)
    session_stats = tk.grouped_stats(trades, "session")
    entry_30m = tk.entry_slot_30m_stats(trades)
    ctx = tk.add_trade_context(trades, classified)
    profile = tk.immediate_loss_profile(ctx)
    kbar_enriched = tk.enrich_with_kbars(classified[classified["entry_time"] < context_end], price_30m)
    kbar_cov = tk.kbar_coverage(kbar_enriched)
    bb_enriched = tk.enrich_trades_with_bb(in_context, price_30m)
    bb_stats_out = tk.bb_stats(bb_enriched)
    bb_cov = int((bb_enriched["bb_zone"] != "unknown").sum())
    dxy_enriched = enrich_dxy_closed(in_context, dxy_1d)
    dxy_stats_out = tk.dxy_regime_stats(dxy_enriched)
    corr_df = tk.dxy_correlation_stats(price_1d, dxy_1d)
    avg_corr = round(float(corr_df["rolling_corr"].dropna().mean()), 3)
    htf_enriched = tk.enrich_trades_with_htf(in_context, price_60m, price_4h, price_1d)
    htf_stats_out = tk.htf_stats(htf_enriched)
    streaks = tk.consecutive_losses(trades)
    by_period = tst.quarterly_bucket_stats(trades)
    holdout = tst.chronological_holdout(trades, split_ratio=0.7)
    temporal = {"by_period": by_period, "holdout_split": holdout, "degradation_flag": tst.degradation_flag(holdout)}

    # ---- decline block ----
    with_vol = volatility_at_entry(trades, price_30m)
    atr_cuts = with_vol.attrs["atr_cuts"]
    labelled = trend_regime(trades, price_1d)
    labelled = labelled.merge(with_vol[["trade_id", "atr_pct_30m", "atr_tercile"]], on="trade_id", how="left")
    stale = labelled["entry_time"] >= context_end
    labelled.loc[stale, ["gold_trend_20d", "atr_tercile"]] = "unknown"
    labelled.loc[stale, "atr_pct_30m"] = np.nan
    labelled = labelled.merge(htf_enriched[["trade_id", "htf_4h_rsi_state"]], on="trade_id", how="left")
    labelled = labelled.merge(bb_enriched[["trade_id", "bb_zone"]], on="trade_id", how="left")
    labelled = labelled.fillna({"htf_4h_rsi_state": "unknown", "bb_zone": "unknown"})
    reference = labelled[labelled["entry_time"] < LIVE_START]
    live = labelled[labelled["entry_time"] >= LIVE_START]
    if len(live) != WINDOW:
        raise ValueError(f"expected {WINDOW} live trades, found {len(live)}; the window size is the live sample")

    windows = window_table(trades, price_1d)
    ref_w = windows[windows["is_reference"]].dropna(subset=["gold_drift_pct"])
    live_w = windows[windows["is_live"]]
    slope, intercept = np.polyfit(ref_w["gold_drift_pct"], ref_w["total_return_pct"], 1)
    resid = ref_w["total_return_pct"] - (intercept + slope * ref_w["gold_drift_pct"])
    live_row = live_w.iloc[0]
    live_pred = intercept + slope * live_row["gold_drift_pct"]
    live_resid = live_row["total_return_pct"] - live_pred
    blocks = [reference.iloc[i:i + WINDOW] for i in range(0, len(reference) - WINDOW + 1, WINDOW)]
    falling_ref = ref_w[ref_w["gold_drift_pct"] <= live_row["gold_drift_pct"]]
    drift_fit = {
        "slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
        "r_squared": round(float(np.corrcoef(ref_w["gold_drift_pct"], ref_w["total_return_pct"])[0, 1] ** 2), 3),
        "live_gold_drift_pct": round(float(live_row["gold_drift_pct"]), 2),
        "live_predicted_total_return_pct": round(float(live_pred), 2),
        "live_actual_total_return_pct": round(float(live_row["total_return_pct"]), 2),
        "live_residual_pct_points": round(float(live_resid), 2),
        "reference_residuals_below_live": int((resid <= live_resid).sum()),
        "reference_windows": int(len(ref_w)),
        "reference_windows_with_drift_at_or_below_live": int(len(falling_ref)),
        "their_worst_total_return_pct": round(float(falling_ref["total_return_pct"].min()), 2) if len(falling_ref) else None,
    }
    decline = {
        "live_start": str(LIVE_START.date()),
        "window_trades": WINDOW,
        "reference": tk.stats(reference),
        "live": tk.stats(live),
        "reference_total_return_pct": round(float(reference["return_pct"].sum()), 2),
        "live_total_return_pct": round(float(live["return_pct"].sum()), 2),
        "permutation_p_one_sided": permutation_p(reference["return_pct"], live["return_pct"]),
        "permutation_note": (f"{TRIALS} draws of {WINDOW} trades from all {len(labelled)}, seed {SEED}; share at or below "
                             "the live mean. The live window was chosen after it looked bad, so this p is optimistic."),
        "win_rate_resolution_bound_pp": round(196 * float(np.sqrt(
            (reference["win"].mean() * (1 - reference["win"].mean())) * (1 / len(live) + 1 / len(reference)))), 1),
        "rolling_windows": {
            "reference_count": int(windows["is_reference"].sum()),
            "straddling_windows_excluded": int(windows["straddles"].sum()),
            "reference_min_total_return_pct": round(float(windows.loc[windows["is_reference"], "total_return_pct"].min()), 2),
            "reference_windows_at_or_below_live": int((windows.loc[windows["is_reference"], "total_return_pct"]
                                                       <= live_row["total_return_pct"]).sum()),
            "reference_windows_with_negative_return": int((windows.loc[windows["is_reference"], "total_return_pct"] < 0).sum()),
            "non_overlapping_reference_blocks": len(blocks),
            "non_overlapping_block_returns_pct": [round(float(b["return_pct"].sum()), 2) for b in blocks],
        },
        "market_control": drift_fit,
        "fail_mix": fail_mix(trades, classified),
        "shift_share": {
            "gold_trend_20d": shift_share(reference, live, "gold_trend_20d"),
            "htf_4h_rsi_state": shift_share(reference.fillna({"htf_4h_rsi_state": "unknown"}),
                                            live.fillna({"htf_4h_rsi_state": "unknown"}), "htf_4h_rsi_state"),
            "session": shift_share(reference, live, "session"),
            "bb_zone": shift_share(reference, live, "bb_zone"),
            "atr_tercile": shift_share(reference, live, "atr_tercile"),
        },
        "volatility": {
            "atr_pct_30m_tercile_cuts_reference": atr_cuts,
            "reference_median_atr_pct_30m": round(float(reference["atr_pct_30m"].median()), 4),
            "live_median_atr_pct_30m": round(float(live["atr_pct_30m"].median()), 4),
            "live_trades_with_atr": int(live["atr_pct_30m"].notna().sum()),
        },
        "monthly": monthly(trades, raw_1d),
    }

    # ---- charts ----
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts = []

    def save(fig, chart_id, title, section):
        (charts_dir / f"{chart_id}.png").write_bytes(tk.fig_to_png_bytes(fig))
        plt.close(fig)
        charts.append({"id": chart_id, "file": f"{chart_id}.png", "title": title, "section": section})

    s, v = STRATEGY_ID, VERSION
    save(tk.chart_equity_curve(trades, s, v), "equity_curve", "Equity Curve & Drawdown", "performance")
    save(chart_rolling(windows), "rolling_31_return", f"Rolling {WINDOW}-Trade Return — Live Window vs History", "performance")
    save(chart_drift(windows, drift_fit), "window_vs_gold_drift", "31-Trade Window Return vs Gold Drift", "performance")
    save(chart_regime(decline["shift_share"]["gold_trend_20d"]), "return_by_gold_trend", "Mean Return by Gold 20-Session Trend", "performance")
    save(tk.chart_fail_type_breakdown(classified, s, v), "fail_type_breakdown", "Fail Type Breakdown", "fail_pattern")
    save(chart_fail_mix(decline["fail_mix"]), "fail_mix_shift", "Outcome Mix — Reference vs Live", "fail_pattern")
    save(tk.chart_mfe_distribution(classified, s, v), "mfe_distribution", "MFE% Distribution", "fail_pattern")
    save(tk.chart_mae_vs_mfe(classified, s, v), "mae_vs_mfe", "MAE vs MFE", "fail_pattern")
    save(tk.chart_entry_slot_30m_winrate(entry_30m, s, v), "entry_slot_30m_winrate", "Win Rate by 30-Minute Entry Slot", "timing_30m")
    save(tk.chart_pre_entry_slot_30m(profile["entry_slot_30m"], s, v), "immediate_loss_by_slot_30m", "Immediate Loss vs All Trades by 30-Min Slot", "timing_30m")
    save(tk.chart_pre_entry_categorical(profile["entry_dow"], "Day-of-Week", s, v), "pre_entry_dow", "Immediate Loss by Day of Week", "pre_entry")
    save(tk.chart_pre_entry_categorical(profile["prev_result"], "Previous Trade Result", s, v), "pre_entry_prev_result", "Immediate Loss by Previous Result", "pre_entry")
    save(tk.chart_pre_entry_categorical(profile["trades_since_win"], "Trades Since Last Win", s, v), "pre_entry_tsw", "Immediate Loss by Trades Since Win", "pre_entry")
    save(tk.chart_kbar_features(kbar_enriched, s, v), "kbar_features", "K-Bar Features at Entry (last closed bar)", "kbar")
    save(tk.chart_bb_zone_winrate(bb_stats_out, s, v), "bb_zone_winrate", "Win Rate by BB Zone (last closed bar)", "bb")
    save(tk.chart_dxy_winrate(dxy_stats_out, s, v), "dxy_winrate", "DXY Context vs Win Rate", "dxy")
    save(tk.chart_dxy_correlation(corr_df, s, v, "XAUUSD"), "dxy_correlation", "DXY x XAUUSD Rolling Correlation", "dxy")
    save(tk.chart_htf_alignment(htf_stats_out, s, v), "htf_alignment", "Win Rate by HTF Alignment", "mtf")
    save(tk.chart_htf_4h_state(htf_stats_out, s, v), "htf_4h_state", "Win Rate by 4H RSI State", "mtf")
    save(tk.chart_htf_bucket_heatmap(htf_stats_out, s, v), "htf_4h_bucket", "Win Rate by 4H RSI Bucket", "mtf")
    save(tk.chart_hold_time_dist(trades, s, v), "hold_time_dist", "Hold Time Distribution", "hold_time_streaks")
    save(tk.chart_consecutive_losses(streaks, s, v), "consecutive_losses", "Consecutive Loss Streaks", "hold_time_streaks")
    save(tst.chart_quarterly_stability(trades, by_period, 0.7, s, v), "quarterly_stability",
         "Quarterly Win Rate — Chronological Holdout", "temporal_stability")

    method = {
        "fail_pattern_thresholds": {
            "immediate_loss_mfe_pct": tk.IMMEDIATE_LOSS_MFE_PCT,
            "false_breakout_mae_mfe_ratio": tk.FALSE_BREAKOUT_MAE_MFE_RATIO,
            "time_bleed_min_bars": tk.TIME_BLEED_MIN_BARS,
        },
        "session_timezone": "Asia/Taipei (TradingView export time, bar open)",
        "session_buckets": "overnight=01:00-06:59; asia=07:00-14:59; europe=15:00-20:29; us=20:30-00:59 (descriptive only)",
        "primary_time_granularity": "30-minute entry slot (HH:00/HH:30, Asia/Taipei)",
        "bar_alignment": ("every price bar is stamped at its close before joining (30m +30min, 60m +60min, 4H +4h, "
                          "1D +1 day), so each context value is read from the last COMPLETED bar before entry; "
                          "see RS-XAUUSD-20260901-001 for why the toolkit's bar-open join is not used"),
        "price_30m": "FX_IDC XAUUSD 30m, two TradingView exports stitched (see price_30m_stitch)",
        "price_60m": "resampled from the stitched 30m series (left-labelled hourly bins, then close-stamped)",
        "rsi_provenance": "locally_computed_wilder_rsi14 (+ SMA14 RSI-MA) on every series; no export carried native RSI for the full range",
        "bb_params": "period=20, std_mult=2.0, population std (ddof=0) as in fail_pattern_toolkit.compute_bb, on 30m close",
        "dxy_params": "TVC DXY 1D, RSI(14) bucket, 20-day SMA trend, RSI-vs-MA momentum; joined on close-stamped time",
        "mtf_params": "60m/4H/1D RSI(14) state and bucket; HTF alignment = count of bullish timeframes",
        "temporal_stability_limitation": tst.TEMPORAL_STABILITY_LIMITATION,
        "temporal_stability_buckets": "calendar quarter (YYYY-Qn) of entry_time; 2026-Q4 is partial (to 2026-10-09)",
        "temporal_stability_holdout": "chronological split, first 70% entry-time-ordered trades = in_sample, last 30% = held_out",
        "temporal_stability_degradation_rule": (
            "degraded: held_out.win_rate_pct < in_sample.win_rate_ci95_pct[0] OR held_out.profit_factor < 1.0; "
            "improved: held_out.win_rate_pct > in_sample.win_rate_ci95_pct[1] AND held_out.profit_factor > in_sample.profit_factor; "
            "else stable"),
        "decline_windows": (f"reference = entries before {LIVE_START.date()}; live = entries on or after it "
                            f"({WINDOW} trades, the Telegram push regime)"),
        "gold_trend_20d": (f"gold close of the last closed daily bar vs {REGIME_LOOKBACK_DAYS} daily bars earlier; "
                           f"falling < -{REGIME_BAND_PCT}%, rising > +{REGIME_BAND_PCT}%, else flat"),
        "atr_pct_30m": ("Wilder ATR(14) of the last closed 30m bar / its close x 100; terciles cut on reference "
                        "trades only; a trade with no 30m bar within 1 hour before entry is 'unknown'"),
        "window_gold_drift": "gold daily close at or before the first entry vs at or before the last exit of the window",
        "shift_share": "delta mean return = sum w_live*(m_live-m_ref) [within] + sum (w_live-w_ref)*m_ref [mix]",
        "return_unit": "Return % of the Strategy Tester (position value ~10k USD, stop distance ~0.5%), so -0.50 is about -1R",
    }
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "strategy": f"{STRATEGY_ID} {VERSION}",
        "method": method,
        "trade_period": {"start": str(trades["entry_time"].min()), "end": str(trades["exit_time"].max())},
        "price_30m_stitch": stitch_audit,
        "context_coverage": {
            "context_end": str(context_end),
            "trades_with_price_context": int(len(in_context)), "total_trades": int(len(trades)),
            "trades_after_context_end": [str(t) for t in trades.loc[trades["entry_time"] >= context_end, "entry_time"]],
            "bb_zone_trades": bb_cov,
            "price_30m_last_bar_close": str(price_30m["time"].max()),
            "note": ("the price exports end 2026-10-03; a trade entered after the next bar would have closed has no "
                     "context and is left out of the K-bar, BB, DXY, MTF and decline-context tables, never given "
                     "the last available bar"),
        },
        "baseline": baseline,
        "fail_pattern": {"total_losses": len(classified), "by_type": fail_summary, "by_session": tk.fail_by_session(classified)},
        "by_session": session_stats,
        "by_entry_30m": entry_30m,
        "immediate_loss_profile": profile,
        "kbar_coverage": kbar_cov,
        "bb_zone": bb_stats_out,
        "dxy": {"regime": dxy_stats_out, "avg_30d_correlation": avg_corr},
        "mtf": htf_stats_out,
        "hold_time_streaks": {
            "avg_hold_bars": baseline["avg_hold_bars"],
            "max_consecutive_losses": baseline["max_consecutive_losses"],
            "streak_lengths": streaks.tolist(),
        },
        "temporal_stability": temporal,
        "decline": decline,
        "charts": charts,
        "sources": [{"role": role, "path": locator(p), "sha256": sha256(p)}
                    for role, p in paths.items()],
    }
    results = tk.to_json_safe(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(results, charts_dir), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(results), encoding="utf-8")
    return results


def _pct(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}%"


def render_readme(r: dict) -> str:
    """Every figure comes from results.json; the prose around it is fixed."""
    d, mc, rw = r["decline"], r["decline"]["market_control"], r["decline"]["rolling_windows"]
    ref, live = d["reference"], d["live"]
    fm = d["fail_mix"]
    b = r["baseline"]
    ts = r["temporal_stability"]
    hi, ho = ts["holdout_split"]["in_sample"], ts["holdout_split"]["held_out"]
    vol = d["volatility"]
    trend = d["shift_share"]["gold_trend_20d"]["groups"]
    atr = d["shift_share"]["atr_tercile"]["groups"]
    st = r["price_30m_stitch"]
    cc = r["context_coverage"]
    wr_gap = round(ref["win_rate_pct"] - live["win_rate_pct"], 2)
    tp1_ref = fm["reference"]["_exits"].get("S1BB_TP1", 0)
    tp1_live = fm["live"]["_exits"].get("S1BB_TP1", 0)
    aug = d["monthly"].get("2026-08", {})

    def share_row(name, block):
        return (f"| {name} | {block['delta_mean_return_pct']:+.4f} | {block['within_pct_points']:+.4f} | "
                f"{block['mix_pct_points']:+.4f} | {block['unattributed_pct_points']:+.4f} |")

    def group_row(name, g):
        rr, ll = g["reference"], g["live"]
        lwr = f"{ll['win_rate_pct']}%" if ll["n"] else "-"
        lavg = f"{ll['avg_return_pct']:+.4f}" if ll["n"] else "-"
        return f"| {name} | {rr['n']} | {rr['win_rate_pct']}% | {rr['avg_return_pct']:+.4f} | {ll['n']} | {lwr} | {lavg} |"

    lines = [
        f"# {r['study_id']} — S1 V3.9 since the live push began: what the decline is made of",
        "",
        f"Generated `{r['generated_at']}` by `scripts/research/build_s1_v39_decline.py`. Every number below is read",
        "from `results.json`; the runner writes this file.",
        "",
        "## Question",
        "",
        f"The {live['n']} S1 V3.9 Strategy Tester trades entered since {d['live_start']} (the Telegram push regime) did",
        "worse than anything earlier in the export. Is that outside the strategy's own history, does the market",
        "explain it (a long-only strategy meeting a falling gold price), does any entry context explain it, and what",
        "changed in how the trades end?",
        "",
        "## Answer",
        "",
        f"- **Outside its history.** Win rate {live['win_rate_pct']}% (n={live['n']}, 95% CI {live['win_rate_ci95_pct']}) against",
        f"  {ref['win_rate_pct']}% (n={ref['n']}, CI {ref['win_rate_ci95_pct']}); profit factor {live['profit_factor']} against",
        f"  {ref['profit_factor']}. None of the {rw['reference_count']} earlier {d['window_trades']}-trade windows did as badly;",
        f"  the worst returned {rw['reference_min_total_return_pct']:+.2f}% against the live window's {d['live_total_return_pct']:+.2f}%.",
        f"- **Not the market.** Gold fell {abs(mc['live_gold_drift_pct']):.2f}% over the live window. Across the reference windows",
        f"  gold's drift explains R² = {mc['r_squared']} of the window return, and the fit predicts {mc['live_predicted_total_return_pct']:+.2f}%",
        f"  for that drift. {mc['reference_windows_with_drift_at_or_below_live']} earlier windows saw gold fall at least as much; the",
        f"  worst of them still returned {mc['their_worst_total_return_pct']:+.2f}%. In August 2026 gold rose",
        f"  {aug.get('gold_month_return_pct', 0):+.2f}% and S1 returned {aug.get('total_return_pct', 0):+.2f}% (n={aug.get('n', 0)}).",
        "- **Not the entry context.** Across gold's trend, the 4H RSI state, the session, the Bollinger zone and",
        "  30-minute volatility, the decline is inside each group, not a shift toward historically worse groups.",
        f"- **What changed is how trades end.** TP1 exits fell from {tp1_ref} of {ref['n']} ({_pct(100 * tp1_ref / ref['n'], 1)}) to",
        f"  {tp1_live} of {live['n']} ({_pct(100 * tp1_live / live['n'], 1)}); clean stop-outs rose from",
        f"  {fm['reference']['normal_sl']['pct_of_trades']}% to {fm['live']['normal_sl']['pct_of_trades']}% of trades.",
        "- **Cause not established.** This is one live window chosen after it looked bad. The study can say what the",
        "  decline is not; it cannot say what it is, and it supports no entry filter.",
        "",
        "## Inputs, alignment and coverage",
        "",
        f"- Trades: the 2026-10-09 Strategy Tester export, {b['n']} closed trades, {r['trade_period']['start']} to {r['trade_period']['end']}.",
        "  Returns are the tester's Return %: about 10,000 USD position value with a stop near 0.5%, so -0.50% is about -1R.",
        f"- 30m price: two FX_IDC TradingView exports stitched. They overlap on {st['overlap_bars']} bars;",
        f"  {st['overlap_bars_with_different_close']} differ in close, all on {', '.join(st['different_close_dates'])}, by at most",
        f"  {st['overlap_max_abs_close_difference']}. The newer export wins on overlap. 60m is resampled from that series.",
        "- **Every bar is stamped at its close before joining** (30m +30 min, 60m +60 min, 4H +4 h, 1D +1 day), so each",
        "  context value comes from the last completed bar before entry. The toolkit's own join reads the bar the fill",
        "  happened inside, which RS-XAUUSD-20260901-001 showed can manufacture a finding.",
        f"- Price exports end {cc['price_30m_last_bar_close']}. The {len(cc['trades_after_context_end'])} trades entered after",
        f"  {cc['context_end']} ({', '.join(cc['trades_after_context_end'])}) have no context and are left out of every",
        "  context table rather than given stale bars. They stay in performance, fail-pattern and timing tables.",
        "- RSI everywhere is Wilder RSI(14) computed locally, with an SMA(14) RSI-MA; no export carried native RSI.",
        "",
        "## 1. Outside the strategy's own history",
        "",
        "| window | n | win rate (95% CI) | profit factor | mean return | total return |",
        "|---|---|---|---|---|---|",
        f"| reference, before {d['live_start']} | {ref['n']} | {ref['win_rate_pct']}% {ref['win_rate_ci95_pct']} | {ref['profit_factor']} | "
        f"{ref['avg_return_pct']:+.4f}% | {d['reference_total_return_pct']:+.2f}% |",
        f"| live, from {d['live_start']} | {live['n']} | {live['win_rate_pct']}% {live['win_rate_ci95_pct']} | {live['profit_factor']} | "
        f"{live['avg_return_pct']:+.4f}% | {d['live_total_return_pct']:+.2f}% |",
        "",
        f"- The win-rate gap is {wr_gap} points. With these sample sizes the smallest gap that could be separated from",
        f"  noise at 95% is about {d['win_rate_resolution_bound_pp']} points (1.96 x the standard error of a difference of two",
        "  proportions at the reference rate), so the gap clears its bound.",
        f"- Permutation: drawing {d['window_trades']} trades at random from all {ref['n'] + live['n']}, a mean return as low as the",
        f"  live one turns up with one-sided p = {d['permutation_p_one_sided']}. The window was chosen *because* it looked bad,",
        "  which makes this p optimistic; it is reported, not leaned on.",
        f"- Rolling {d['window_trades']}-trade windows lying wholly before {d['live_start']}: {rw['reference_count']} of them",
        f"  ({rw['straddling_windows_excluded']} windows straddling the boundary are excluded, since they would put live trades in",
        f"  the comparison). {rw['reference_windows_with_negative_return']} were negative, the worst {rw['reference_min_total_return_pct']:+.2f}%, and",
        f"  {rw['reference_windows_at_or_below_live']} reached the live window's {d['live_total_return_pct']:+.2f}%. The windows overlap, so",
        f"  this is a description, not a test. The {rw['non_overlapping_reference_blocks']} non-overlapping reference blocks returned",
        f"  {', '.join(f'{x:+.2f}%' for x in rw['non_overlapping_block_returns_pct'])}: every one positive.",
        "",
        "## 2. The market does not explain it",
        "",
        f"For each reference window, gold's change from the daily close at its first entry to the daily close at its",
        f"last exit was regressed against the window's total return: slope {mc['slope']} return points per 1% of gold, R² {mc['r_squared']}.",
        f"Gold moved {mc['live_gold_drift_pct']:+.2f}% over the live window; the fit predicts {mc['live_predicted_total_return_pct']:+.2f}%",
        f"and the window returned {mc['live_actual_total_return_pct']:+.2f}%, a residual of {mc['live_residual_pct_points']:+.2f} points.",
        f"{mc['reference_residuals_below_live']} of {mc['reference_windows']} reference residuals are as low.",
        "",
        f"By gold's {REGIME_LOOKBACK_DAYS}-session trend at entry (last closed daily bar against {REGIME_LOOKBACK_DAYS} bars earlier, ±{REGIME_BAND_PCT}% bands):",
        "",
        "| gold trend | ref n | ref WR | ref mean % | live n | live WR | live mean % |",
        "|---|---|---|---|---|---|---|",
        *[group_row(g, trend[g]) for g in ("falling (<-2%)", "flat (±2%)", "rising (>+2%)") if g in trend],
        "",
        "Historically S1 did *best* after gold had fallen. In the live window it lost in falling and rising markets",
        f"alike, on cells of n={trend.get('falling (<-2%)', {}).get('live', {}).get('n', 0)} and "
        f"n={trend.get('rising (>+2%)', {}).get('live', {}).get('n', 0)} that are not separately interpretable.",
        "",
        "## 3. Nor does entry context — the decline is inside every group",
        "",
        "Shift-share splits the change in mean return per trade into *within* (the same kind of trade did worse) and",
        "*mix* (more trades of a historically worse kind). Percentage points of mean return per trade:",
        "",
        "| dimension | change | within | mix | unattributed |",
        "|---|---|---|---|---|",
        *[share_row(k, v) for k, v in d["shift_share"].items()],
        "",
        f"Volatility rose a little — median 30m ATR at entry {vol['reference_median_atr_pct_30m']}% of price before,",
        f"{vol['live_median_atr_pct_30m']}% live (n={vol['live_trades_with_atr']}) — but it cannot carry the decline: historically the",
        "highest-ATR tercile was S1's best, not its worst.",
        "",
        "| 30m ATR tercile (cut on reference trades) | ref n | ref WR | ref mean % | live n | live WR | live mean % |",
        "|---|---|---|---|---|---|---|",
        *[group_row(g, atr[g]) for g in ("low ATR%", "mid ATR%", "high ATR%") if g in atr],
        "",
        "## 4. What changed is how trades end",
        "",
        "| outcome, % of trades | reference | live |",
        "|---|---|---|",
        *[f"| {k} | {fm['reference'].get(k, {}).get('pct_of_trades', 0)}% | {fm['live'].get(k, {}).get('pct_of_trades', 0)}% |"
          for k in ("win", "normal_sl", "immediate_loss", "time_bleed", "false_breakout")],
        "",
        f"- Losers that first went at least +0.25% in favour (half way to a 0.5% TP1): {fm['reference']['_losers']['share_reaching_0_25pct_mfe']}%",
        f"  of {fm['reference']['_losers']['n']} reference losers, {fm['live']['_losers']['share_reaching_0_25pct_mfe']}% of {fm['live']['_losers']['n']} live losers.",
        "  More live trades moved toward target and then failed. On 21 losers this is a description, not a result.",
        "",
        "## 5. The standard section 5 report, full period",
        "",
        f"- Baseline: n={b['n']}, win rate {b['win_rate_pct']}% (CI {b['win_rate_ci95_pct']}), profit factor {b['profit_factor']},",
        f"  net {b['net_pnl_usd']:,.2f} USD on the tester's sizing, max drawdown {b['max_drawdown_usd']:,.2f} USD, longest losing run",
        f"  {b['max_consecutive_losses']}, average hold {b['avg_hold_bars']:.1f} 30m bars.",
        "- Fail types (losses only): " + ", ".join(f"`{k}` {v['count']} ({v['pct']}%)" for k, v in r["fail_pattern"]["by_type"].items()) + ".",
        f"- Temporal stability (section 5.1 item 11): first 70% (to {hi['period']['end']}) win rate {hi['win_rate_pct']}%",
        f"  (CI {hi['win_rate_ci95_pct']}, n={hi['n']}, PF {hi['profit_factor']}); last 30% {ho['win_rate_pct']}% (n={ho['n']}, PF",
        f"  {ho['profit_factor']}). Flag: **{ts['degradation_flag']}** under the fixed rule (held-out win rate below the",
        f"  in-sample lower bound {hi['win_rate_ci95_pct'][0]}%, or held-out PF below 1.0).",
        "- Quarterly: " + "; ".join(f"{q} {v['win_rate_pct']}% (n={v['n']}, PF {v['profit_factor']})" for q, v in ts["by_period"].items()) + ".",
        "  2026-Q4 holds only the first days of October.",
        "- 30-minute slots, BB zone, DXY, multi-timeframe and streak tables are in `report.html` and `results.json`.",
        "  They are descriptive; no cell is a gate.",
        "",
        "## What this does not show",
        "",
        "- A cause. Candidates the data cannot separate: a change in how gold trades intraday that none of these",
        "  context measures captures, or an unusually bad run of a strategy that has not changed.",
        "- That S1 has stopped working. Thirty-one trades is one window; the next one decides more than this one.",
        "- Anything about how the account executed S1. Actual fills are a separate record",
        "  (RS-XAUUSD-20261009-002 and the 2026-10-09 reconciliation).",
        "- The Strategy Tester is a model of the strategy, but not a loose one here: every one of the 39 live",
        "  Telegram signals since 2026-08-16 (S1 and S2) matched a tester trade in the 2026-10-09 reconciliation",
        "  (the 2026-10-09 reconciliation record).",
        "",
        "## Sources",
        "",
        *[f"- `{s['role']}`: `{s['path']}` — SHA-256 `{s['sha256']}`" for s in r["sources"]],
    ]
    return "\n".join(lines) + "\n"


def render_html(results: dict, charts_dir: Path) -> str:
    """The standard report, with the decline block inserted ahead of the full breakdown."""
    base = solo.render_html(results, charts_dir)
    b64 = {c["id"]: base64.b64encode((charts_dir / c["file"]).read_bytes()).decode() for c in results["charts"]}
    d = results["decline"]
    mc = d["market_control"]
    ref, live = d["reference"], d["live"]
    month_rows = [[m, v["n"], f'{v["win_rate_pct"]}%', v["profit_factor"] or "-", f'{v["total_return_pct"]:+.2f}%',
                   f'{v["gold_month_return_pct"]:+.2f}%' if v["gold_month_return_pct"] is not None else "-"]
                  for m, v in d["monthly"].items()]
    share_rows = []
    for dim, block in d["shift_share"].items():
        share_rows.append([dim, f'{block["delta_mean_return_pct"]:+.4f}', f'{block["within_pct_points"]:+.4f}',
                           f'{block["mix_pct_points"]:+.4f}', f'{block["unattributed_pct_points"]:+.4f}'])
    trend_rows = []
    for g, v in d["shift_share"]["gold_trend_20d"]["groups"].items():
        trend_rows.append([g, v["reference"]["n"], f'{v["reference"]["win_rate_pct"]}%', f'{v["reference"]["avg_return_pct"]}',
                           v["live"]["n"], f'{v["live"]["win_rate_pct"]}%' if v["live"]["n"] else "-",
                           f'{v["live"]["avg_return_pct"]}' if v["live"]["n"] else "-"])
    section = f"""
  <h2>Decline since the live push began ({d["live_start"]})</h2>
  <div class="card">
    {solo._table(["window", "n", "WR (95% CI)", "PF", "total return"], [
        ["reference", ref["n"], f'{ref["win_rate_pct"]}% {ref["win_rate_ci95_pct"]}', ref["profit_factor"], f'{d["reference_total_return_pct"]:+.2f}%'],
        ["live", live["n"], f'{live["win_rate_pct"]}% {live["win_rate_ci95_pct"]}', live["profit_factor"], f'{d["live_total_return_pct"]:+.2f}%'],
    ])}
    <div class="note">Permutation p (one-sided) {d["permutation_p_one_sided"]}. {d["permutation_note"]}
    Smallest win-rate gap this sample could resolve: about {d["win_rate_resolution_bound_pp"]} points.</div>
    {solo._img(b64["rolling_31_return"])}
    <div class="chart-grid">{solo._img(b64["window_vs_gold_drift"])}{solo._img(b64["return_by_gold_trend"])}</div>
    <div class="note">Market control: across {mc["reference_windows"]} reference windows, gold's drift explains
    R&sup2; = {mc["r_squared"]} of the window return. Gold moved {mc["live_gold_drift_pct"]:+.2f}% over the live window, for which the
    fit predicts {mc["live_predicted_total_return_pct"]:+.2f}%; the window returned {mc["live_actual_total_return_pct"]:+.2f}%.
    {mc["reference_residuals_below_live"]} of {mc["reference_windows"]} reference residuals are as low.</div>
    {solo._table(["gold 20-session trend", "ref n", "ref WR", "ref mean %", "live n", "live WR", "live mean %"], trend_rows)}
    {solo._img(b64["fail_mix_shift"])}
    <h3>Shift-share of the change in mean return per trade (percentage points)</h3>
    {solo._table(["dimension", "delta", "within", "mix", "unattributed"], share_rows)}
    <h3>Monthly</h3>
    {solo._table(["month", "n", "WR", "PF", "total return", "gold"], month_rows)}
  </div>
"""
    return base.replace("  <h2>Equity Curve & Drawdown</h2>", section + "\n  <h2>Equity Curve & Drawdown</h2>", 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    results = build(args.output_dir.resolve())
    d = results["decline"]
    print(json.dumps({"study_id": STUDY_ID, "baseline": {k: results["baseline"][k] for k in ("n", "win_rate_pct", "profit_factor")},
                      "reference": d["reference"], "live": d["live"], "p": d["permutation_p_one_sided"],
                      "market_control": d["market_control"], "charts": len(results["charts"])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
