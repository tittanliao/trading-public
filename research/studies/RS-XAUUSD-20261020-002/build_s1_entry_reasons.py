#!/usr/bin/env python3
"""Shared measures for RS-XAUUSD-20261020-001 and -002 — two reasons given for live trades, tested on the record.

-001: 「美盤的s1, 且超出上軌, 歷史勝率高」 — US-session S1 with the last closed 30m bar above the upper band.
-002: 「亞盤s1 通常九點開盤 九點半就開始跌了」 — gold falls from 09:30 Taipei, above all after an Asian-morning S1.

Designs are fixed in each study's decision_log.md. The runners are build_s1_us_above_band.py (-001) and
build_asia_0930_s1.py (-002); this module holds what both use and writes nothing itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import fail_pattern_toolkit as tk  # noqa: E402

TAIPEI = timezone(timedelta(hours=8))
TRADES = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv")
BARS = Path("local-inputs/xauusd-30m-full.csv")
# The live signal ledger is optional and named by environment, never by a path in this file: without it the
# descriptive live list is simply empty and every tested number is unchanged.
LEDGER = Path(os.environ.get("SIGNAL_FEEDBACK_LEDGER", "")) if os.environ.get("SIGNAL_FEEDBACK_LEDGER") else None
SEED, SHUFFLES, MIN_N = 20261020, 20000, 15


def sha(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def rate_gap(a: pd.Series, b: pd.Series) -> dict:
    """Win-rate gap a − b in points, with the 2.8σ bound on the pooled rate."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"n_a": n1, "n_b": n2}
    p1, p2 = a.mean(), b.mean()
    p = (a.sum() + b.sum()) / (n1 + n2)
    bound = 2.8 * math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return {"n_a": n1, "n_b": n2, "rate_a_pct": round(100 * p1, 2), "rate_b_pct": round(100 * p2, 2),
            "gap_pct_points": round(100 * (p1 - p2), 2), "bound_pct_points": round(100 * bound, 2),
            "resolved": bool(abs(p1 - p2) > bound), "too_small": bool(min(n1, n2) < MIN_N)}


def mean_gap(a: pd.Series, b: pd.Series, digits: int = 3) -> dict:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"n_a": n1, "n_b": n2}
    bound = 2.8 * math.sqrt(a.var(ddof=1) / n1 + b.var(ddof=1) / n2)
    return {"n_a": n1, "n_b": n2, "mean_a": round(a.mean(), digits), "mean_b": round(b.mean(), digits),
            "gap": round(a.mean() - b.mean(), digits), "bound": round(bound, digits),
            "resolved": bool(abs(a.mean() - b.mean()) > bound), "too_small": bool(min(n1, n2) < MIN_N)}


def permutation_p(flags: list[bool], wins: list[bool]) -> float:
    rng = random.Random(SEED)
    def gap(f):
        a = [w for w, x in zip(wins, f) if x]
        b = [w for w, x in zip(wins, f) if not x]
        return sum(a) / len(a) - sum(b) / len(b)
    observed = gap(flags)
    shuffled, hits = list(flags), 0
    for _ in range(SHUFFLES):
        rng.shuffle(shuffled)
        if gap(shuffled) >= observed:
            hits += 1
    return round((hits + 1) / (SHUFFLES + 1), 4)


def pf(pnl: pd.Series) -> float | None:
    loss = -pnl[pnl < 0].sum()
    return round(pnl[pnl > 0].sum() / loss, 3) if loss > 0 else None


def group_row(frame: pd.DataFrame) -> dict:
    return {"n": len(frame), "win_rate_pct": round(100 * frame["win"].mean(), 2) if len(frame) else None,
            "mean_pnl_usd": round(frame["net_pnl_usd"].mean(), 2) if len(frame) else None,
            "profit_factor": pf(frame["net_pnl_usd"]) if len(frame) else None}


# ---------------------------------------------------------------------------------------------------------------------
# -001: US session × above the upper band
# ---------------------------------------------------------------------------------------------------------------------

def live_s1() -> list[dict]:
    """S1 signals in the Telegram ledger: %B, session, decision and the strategy's exit. Descriptive only."""
    if LEDGER is None or not LEDGER.is_file():
        return []
    con = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT p.signal_ref, p.fired_at_utc, p.signal_price, p.decision, c.context_json, "
        "(SELECT order_price FROM order_fill f WHERE f.signal_id = p.signal_id AND f.fill_kind = 'exit' "
        " ORDER BY fired_at_utc LIMIT 1) AS exit_price, "
        "(SELECT json_extract(raw_payload_json, '$.order_id') FROM order_fill f WHERE f.signal_id = p.signal_id "
        " AND f.fill_kind = 'exit' ORDER BY fired_at_utc LIMIT 1) AS exit_id "
        "FROM signal_projection p LEFT JOIN signal_context c USING (signal_id) "
        "WHERE p.strategy_id = 'S1' ORDER BY p.fired_at_utc").fetchall()
    out = []
    for r in rows:
        ctx = json.loads(r["context_json"]) if r["context_json"] else {}
        at = datetime.fromisoformat(r["fired_at_utc"].replace("Z", "+00:00")).astimezone(TAIPEI)
        minute = at.hour * 60 + at.minute
        session = ("asia" if 7 * 60 <= minute < 15 * 60 else "europe" if 15 * 60 <= minute < 20 * 60 + 30
                   else "us" if minute >= 20 * 60 + 30 or minute < 60 else "overnight")
        pct_b = ctx.get("bb_pct_b")
        move = (r["exit_price"] - r["signal_price"]) if r["exit_price"] is not None else None
        out.append({"signal_ref": r["signal_ref"], "taipei": at.strftime("%Y-%m-%d %H:%M"), "session": session,
                    "pct_b": pct_b, "above_upper": bool(pct_b is not None and pct_b > 1), "decision": r["decision"],
                    "exit_id": r["exit_id"], "strategy_move_points": round(move, 2) if move is not None else None})
    return out


def study_us(trades: pd.DataFrame) -> dict:
    t = trades[trades["bb_zone"] != "unknown"].copy()
    t["above"] = t["bb_pct_b"] > 1
    t["us_above"] = (t["session"] == "us") & t["above"]
    target, rest = t[t["us_above"]], t[~t["us_above"]]
    primary = rate_gap(target["win"].astype(float), rest["win"].astype(float))
    primary["permutation_p_one_sided"] = permutation_p(t["us_above"].tolist(), t["win"].tolist())
    cells = {}
    for session in ("asia", "europe", "us", "overnight"):
        s = t[t["session"] == session]
        cells[session] = {"above": group_row(s[s["above"]]), "not_above": group_row(s[~s["above"]]),
                          "gap": rate_gap(s[s["above"]]["win"].astype(float), s[~s["above"]]["win"].astype(float))}
    live = live_s1()
    live_target = [x for x in live if x["session"] == "us" and x["above_upper"]]
    return {
        "sample": {"n": len(t), "start": str(t["entry_time"].min()), "end": str(t["entry_time"].max()),
                   "baseline": group_row(t)},
        "primary_us_above_vs_rest": primary,
        "us_above": group_row(target), "rest": group_row(rest),
        "pnl_gap_usd": mean_gap(target["net_pnl_usd"], rest["net_pnl_usd"], 2),
        "us_above_vs_us_not_above": rate_gap(t[(t["session"] == "us") & t["above"]]["win"].astype(float),
                                             t[(t["session"] == "us") & ~t["above"]]["win"].astype(float)),
        "above_vs_not_any_session": rate_gap(t[t["above"]]["win"].astype(float), t[~t["above"]]["win"].astype(float)),
        "by_session": cells,
        "live": {"signals": live, "us_above_count": len(live_target),
                 "note": "Strategy exits from the ledger; descriptive only, never tested."},
        "supported": bool(primary.get("gap_pct_points", 0) > 0 and primary.get("resolved")
                          and not primary.get("too_small")),
    }


# ---------------------------------------------------------------------------------------------------------------------
# -002: does gold fall from 09:30 Taipei?
# ---------------------------------------------------------------------------------------------------------------------

def slot_returns(price: pd.DataFrame, hours: float) -> pd.DataFrame:
    """For every bar: open at the bar, open `hours` later on the same series (exact timestamp), Taipei weekdays."""
    p = price[["time", "open"]].copy()
    later = p.rename(columns={"time": "t_end", "open": "open_end"})
    p["t_end"] = p["time"] + pd.Timedelta(hours=hours)
    out = p.merge(later, on="t_end", how="inner")
    out["ret_pts"] = out["open_end"] - out["open"]
    out["ret_pct"] = 100 * out["ret_pts"] / out["open"]
    out["slot"] = out["time"].dt.strftime("%H:%M")
    out["date"] = out["time"].dt.date
    return out[out["time"].dt.dayofweek < 5]


def describe_slot(frame: pd.DataFrame) -> dict:
    n = len(frame)
    sd = frame["ret_pct"].std(ddof=1)
    neg = (frame["ret_pts"] < 0).mean()
    return {"n": n, "mean_pts": round(frame["ret_pts"].mean(), 2), "median_pts": round(frame["ret_pts"].median(), 2),
            "mean_pct": round(frame["ret_pct"].mean(), 4), "bound_mean_pct": round(2.8 * sd / math.sqrt(n), 4),
            "share_negative_pct": round(100 * neg, 2), "bound_share_points": round(100 * 2.8 * math.sqrt(0.25 / n), 2)}


def study_asia(trades: pd.DataFrame, price: pd.DataFrame) -> dict:
    windows = {}
    for label, start, hours in (("0930_to_1130", "09:30", 2), ("0930_to_1030", "09:30", 1),
                                ("0930_to_1330", "09:30", 4), ("0900_to_0930", "09:00", 0.5)):
        r = slot_returns(price, hours)
        windows[label] = describe_slot(r[r["slot"] == start])
    main = slot_returns(price, 2)
    by_slot = main.groupby("slot")["ret_pct"].mean().sort_values()
    rank = int(list(by_slot.index).index("09:30")) + 1
    day = main[main["slot"] == "09:30"].copy()
    morning = trades[(trades["entry_time"].dt.hour * 60 + trades["entry_time"].dt.minute).between(7 * 60, 9 * 60 + 29)]
    s1_days = set(morning["entry_time"].dt.date)
    day["s1_morning"] = day["date"].isin(s1_days)
    with_s1, without = day[day["s1_morning"]], day[~day["s1_morning"]]
    others = trades.drop(morning.index)
    shorts = -with_s1["ret_pts"]
    md = describe_slot(day)
    return {
        "sample": {"days": len(day), "start": str(day["date"].min()), "end": str(day["date"].max()),
                   "s1_trades": len(trades)},
        "market": {"windows": windows, "rank_0930_among_48_slots_lowest_first": rank,
                   "slot_mean_pct_range": [round(by_slot.min(), 4), round(by_slot.max(), 4)],
                   "slot_means_pct": {k: round(v, 4) for k, v in by_slot.items()}},
        "signal": {
            "days_with_morning_s1": len(with_s1),
            "return_after_0930_with_vs_without_pct": mean_gap(with_s1["ret_pct"], without["ret_pct"], 4),
            "return_after_0930_with_s1": describe_slot(with_s1) if len(with_s1) > 1 else None,
            "morning_s1_vs_other_s1_win": rate_gap(morning["win"].astype(float), others["win"].astype(float)),
            "morning_s1": group_row(morning), "other_s1": group_row(others),
            "morning_s1_pnl_gap_usd": mean_gap(morning["net_pnl_usd"], others["net_pnl_usd"], 2),
            "short_0930_to_1130_on_s1_days": {"n": len(shorts), "mean_pts": round(shorts.mean(), 2) if len(shorts) else None,
                                              "share_profitable_pct": round(100 * (shorts > 0).mean(), 2) if len(shorts) else None,
                                              "note": "Hypothetical, no costs or spread; descriptive."}},
        "market_supported": bool(md["mean_pct"] < 0 and abs(md["mean_pct"]) > md["bound_mean_pct"]),
        "charts": [{"id": "slot_2h_returns", "file": "slot_2h_returns.png", "section": "market",
                    "title": "Mean 2-hour return from each half-hour, Taipei"}],
    }


def chart_asia(price: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = slot_returns(price, 2)
    g = r.groupby("slot")["ret_pct"]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    bound = 2.8 * sd / np.sqrt(n)
    x = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
    colors = ["#d62728" if s == "09:30" else "#7f7f7f" for s in mean.index]
    ax.bar(x, mean.values, color=colors)
    ax.errorbar(x, mean.values, yerr=bound.values, fmt="none", ecolor="#bbbbbb", lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x[::2], mean.index[::2], rotation=90, fontsize=7)
    ax.set_ylabel("mean 2-hour return (%)")
    ax.set_title("XAUUSD mean 2-hour return from each half-hour (Taipei, weekdays 2024-01 to 2026-08); 09:30 in red, "
                 "whiskers = 2.8σ bound", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def manifest(study_id: str, generated: str, directory: Path) -> None:
    (directory / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study_id": study_id, "generated_at": generated,
        "timezone_assumptions": "Trade times and 30m bars in Asia/Taipei; bars stamped at their open.",
        "sources": [
            {"role": "trades_s1", "source_system": "TradingView List of Trades export (S1 AweWithBB V3.9)",
             "locator": "local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv", **sha(TRADES), "immutable": True,
             "privacy": "private_raw_source", "note": "Same export as RS-XAUUSD-20260901-001."},
            {"role": "price_30m", "source_system": "TradingView export FX_IDC:XAUUSD 30m",
             "locator": "local-inputs/xauusd-30m-full.csv", **sha(BARS), "immutable": True,
             "privacy": "private_raw_source", "note": "2024-01-02 to 2026-08-15."},
            {"role": "live_signals", "source_system": "signal-feedback ledger (read only)",
             "locator": "signal_feedback_staging.sqlite3 signal_projection / signal_context / order_fill",
             **(sha(LEDGER) if LEDGER else {}), "immutable": False, "privacy": "private_trading_ledger",
             "note": "Hash of the live database file at run time; the live list is in RS-XAUUSD-20261020-001 "
                     "results.json. Descriptive only."}]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S1 trades, 30m bars, and the trades with %B read on the last closed bar."""
    trades = tk.load_trades(TRADES)
    price, _ = tk.load_price_csv(BARS)
    return trades, price, tk.enrich_trades_with_bb(trades, price, closed_bars=True)
