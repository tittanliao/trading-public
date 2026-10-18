#!/usr/bin/env python3
"""RS-XAUUSD-20261014-002 — S1 / S2 under Topstep mechanics: daily flat, MGC sizing, trailing loss limit.

The design is registered in the study's `decision_log.md` (committed before this runner computed any outcome);
the constants below mirror it.

    python3.12 scripts/research/build_topstep_study.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import build_atr_stop_study as atr  # noqa: E402
import build_entry_reminder_study as base  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261014-002"
OUTPUT_DIR = Path("reproduced")
INPUTS = atr.INPUTS
EXITS = atr.EXITS
TAIPEI, CHICAGO = ZoneInfo("Asia/Taipei"), ZoneInfo("America/Chicago")
FLAT, REOPEN = time(15, 10), time(17, 0)
CUTOFFS = (2, 4, 8)
OZ_PER_MGC = 10
TARGET, MLL, CONSISTENCY_DAY = 3000.0, 2000.0, 1500.0
RECENT = pd.Timestamp("2026-07-01")
BAR = pd.Timedelta(minutes=30)


def to_taipei_naive(ts: datetime) -> pd.Timestamp:
    return pd.Timestamp(ts.astimezone(TAIPEI).replace(tzinfo=None))


def session(entry: pd.Timestamp) -> tuple[bool, pd.Timestamp]:
    """(entry falls in the closed window, the next flat time in Taipei wall clock)."""
    ct = entry.tz_localize(TAIPEI).tz_convert(CHICAGO)
    closed = FLAT <= ct.time() < REOPEN or ct.weekday() == 5 or (ct.weekday() == 4 and ct.time() >= FLAT) \
        or (ct.weekday() == 6 and ct.time() < REOPEN)
    day = ct.date() if ct.time() < FLAT else ct.date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    flat = datetime.combine(day, FLAT, tzinfo=CHICAGO)
    return closed, to_taipei_naive(flat)


def replay(entry_i: int, entry: float, stop_pts: float, spec: dict, price: dict, last: int):
    """Two lots; returns (lot A R, lot B R, lot A exit index, lot B exit index)."""
    stop, tp1, tp2 = entry - stop_pts, entry + spec["tp1"] * stop_pts, entry + spec["tp2"] * stop_pts
    ra = rb = None
    ia = ib = last
    for j in range(entry_i, last + 1):
        if price["low"][j] <= stop:
            if ra is None:
                ra, ia = -1.0, j
            if rb is None:
                rb, ib = -1.0, j
            break
        if ra is None and price["high"][j] >= tp1:
            ra, ia = spec["tp1"], j
        if rb is None and price["high"][j] >= tp2:
            rb, ib = spec["tp2"], j
        if ra is not None and rb is not None:
            break
    end_r = (price["close"][last] - entry) / stop_pts
    return (ra if ra is not None else end_r), (rb if rb is not None else end_r), ia, ib


def replay_all(trades: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    index = {t: i for i, t in enumerate(price["time"])}
    arrays = {c: price[c].to_numpy() for c in ("high", "low", "close")}
    times = price["time"]
    rows = []
    for t in trades.itertuples():
        i = index.get(t.entry_time)
        if i is None:
            continue
        spec = EXITS[t.strategy]
        stop_pts = t.entry_price * spec["fixed_pct"] / 100
        closed, flat = session(t.entry_time)
        strategy_last = min(i + spec["window"], len(times) - 1)
        # last bar that ends at or before the flat time
        flat_last = int(np.searchsorted(times.to_numpy(), np.datetime64(flat - BAR), side="right")) - 1
        common = {"strategy": t.strategy, "trade_id": t.trade_id, "entry_time": t.entry_time, "stop_pts": stop_pts,
                  "closed_window": closed, "hours_to_flat": (flat - t.entry_time).total_seconds() / 3600,
                  "flat_time": flat, "tester_r": t.r}
        for name, last in (("base", strategy_last), ("flat", min(strategy_last, max(flat_last, i)))):
            ra, rb, ia, ib = replay(i, t.entry_price, stop_pts, spec, arrays, last)
            rows.append({**common, "rule": name, "ra": ra, "rb": rb, "r": (ra + rb) / 2,
                         "exit_a": times.iloc[ia], "exit_b": times.iloc[ib], "exit": times.iloc[max(ia, ib)]})
    return pd.DataFrame(rows)


def eligible(rep: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "base":
        return rep[rep["rule"] == "base"]
    frame = rep[(rep["rule"] == "flat") & ~rep["closed_window"]]
    if variant.startswith("flat_cutoff_"):
        frame = frame[frame["hours_to_flat"] >= int(variant.rsplit("_", 1)[1])]
    if variant == "s1_only_flat":
        frame = frame[frame["strategy"] == "S1"]
    return frame


def per_trade(frame: pd.DataFrame, column: str = "r") -> dict:
    r = frame[column].to_numpy(float)
    if not len(r):
        return {"n": 0}
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(r), "win_rate": round(100 * (r > 0).mean(), 1), "mean_r": round(float(r.mean()), 3),
            "sum_r": round(float(r.sum()), 1), "profit_factor": round(gains / losses, 2) if losses > 0 else None}


def account_trades(frame: pd.DataFrame, s1_contracts: int) -> pd.DataFrame:
    """One position at a time; S1 two lots × s1_contracts MGC, S2 lot A × 1 MGC."""
    out, busy = [], pd.Timestamp.min
    for t in frame.sort_values("entry_time").itertuples():
        if t.entry_time < busy:
            continue
        if t.strategy == "S1":
            pnl = s1_contracts * OZ_PER_MGC * t.stop_pts * (t.ra + t.rb)
            risk = 2 * s1_contracts * OZ_PER_MGC * t.stop_pts
            end = t.exit
        else:
            pnl = OZ_PER_MGC * t.stop_pts * t.ra
            risk = OZ_PER_MGC * t.stop_pts
            end = t.exit_a
        busy = end
        out.append({"entry_time": t.entry_time, "exit": end, "pnl": pnl, "risk": risk, "session": t.flat_time})
    return pd.DataFrame(out)


def combine(acct: pd.DataFrame, days: int, since: pd.Timestamp | None = None) -> dict:
    results = []
    starts = acct.index if since is None else acct.index[acct["entry_time"] >= since]
    last_entry = acct["entry_time"].max()
    for i in starts:
        start = acct.at[i, "entry_time"]
        if start + pd.Timedelta(days=days) > last_entry + pd.Timedelta(days=3):
            continue  # not enough data after this start to judge it
        window = acct[(acct["entry_time"] >= start) & (acct["entry_time"] < start + pd.Timedelta(days=days))]
        balance, eod_high, outcome, day_pnl, passed_on = 0.0, 0.0, "unfinished", {}, None
        for session_key, g in window.groupby("session", sort=True):
            for pnl in g["pnl"]:
                balance += pnl
                if balance <= eod_high - MLL:
                    outcome = "fail"
                    break
            if outcome == "fail":
                break
            day_pnl[session_key] = g["pnl"].sum()
            eod_high = max(eod_high, balance)
            if balance >= TARGET:
                outcome, passed_on = "pass", (session_key - start).days
                break
        results.append({"outcome": outcome, "best_day": max(day_pnl.values()) if day_pnl else 0.0,
                        "days": passed_on})
    r = pd.DataFrame(results)
    if r.empty:
        return {"starts": 0}
    passed = r[r["outcome"] == "pass"]
    return {"starts": len(r), **{k: round(100 * float((r["outcome"] == k).mean()), 1) for k in ("pass", "fail", "unfinished")},
            "median_days_to_pass": float(passed["days"].median()) if len(passed) else None,
            "passes_with_best_day_over_1500_pct": round(100 * float((passed["best_day"] > CONSISTENCY_DAY).mean()), 1) if len(passed) else None}


def main() -> int:
    price = atr.load_price()
    s1, _ = base.load_trades("S1")
    s2, _ = base.load_trades("S2")
    rep = replay_all(pd.concat([s1, s2], ignore_index=True), price)

    variants = ["base", "flat", *[f"flat_cutoff_{h}" for h in CUTOFFS], "s1_only_flat"]
    per_strategy = {}
    for v in variants:
        frame = eligible(rep, v)
        per_strategy[v] = {s: {"two_lots": per_trade(g), "lot_a_only": per_trade(g, "ra")}
                           for s, g in frame.groupby("strategy")}
    flat_rows = rep[rep["rule"] == "flat"]
    crossing = {s: {"n": len(g), "closed_window_pct": round(100 * float(g["closed_window"].mean()), 1),
                    "truncated_by_flat_pct": round(100 * float((rep[(rep["rule"] == "base") & (rep["strategy"] == s)]["exit"].to_numpy()
                                                               > g["flat_time"].to_numpy()).mean()), 1)}
                for s, g in flat_rows.groupby("strategy")}
    accounts = {}
    for v in variants:
        for contracts in (1, 2):
            acct = account_trades(eligible(rep, v), contracts)
            key = f"{v}__s1_{contracts}mgc"
            accounts[key] = {"trades": len(acct), "median_risk_usd": round(float(acct["risk"].median()), 0),
                             "total_pnl_usd": round(float(acct["pnl"].sum()), 0),
                             "combine_30d": combine(acct, 30), "combine_60d": combine(acct, 60),
                             "combine_30d_since_2026_07": combine(acct, 30, RECENT),
                             "combine_60d_since_2026_07": combine(acct, 60, RECENT)}
    results = {
        "schema_version": 1, "study_id": STUDY_ID,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {k: {"path": v.relative_to(ROOT).as_posix(), "sha256": base.sha256(v)} for k, v in INPUTS.items()},
        "price_range": [price["time"].min().isoformat(), price["time"].max().isoformat()],
        "assumed_rules": {"flat_chicago": "15:10", "reopen_chicago": "17:00", "target": TARGET, "mll_trailing_eod": MLL,
                          "consistency_best_day": CONSISTENCY_DAY, "mgc_oz": OZ_PER_MGC},
        "flat_rule_exposure": crossing,
        "per_trade": per_strategy,
        "accounts": accounts,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"exposure": crossing}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
