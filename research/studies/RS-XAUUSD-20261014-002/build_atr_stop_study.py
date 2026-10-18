#!/usr/bin/env python3
"""RS-XAUUSD-20261014-001 — ATR-based stops and weekly-balance sizing on a 50,000 USD account.

The design is registered in the study's `decision_log.md` (committed before this runner computed any outcome);
the constants below mirror it.

    python3.12 scripts/research/build_atr_stop_study.py
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import build_entry_reminder_study as base  # noqa: E402  (same inputs and loaders as RS-XAUUSD-20261013-001)

STUDY_ID = "RS-XAUUSD-20261014-001"
OUTPUT_DIR = Path("reproduced")
INPUTS = {k: base.INPUTS[k] for k in ("s1_trades", "s1_trades_earlier", "s2_trades", "s2_trades_earlier",
                                      "price_30m_history", "price_30m_recent")}
EXITS = {"S1": {"fixed_pct": 0.5, "tp1": 1.0, "tp2": 3.5, "window": 36},
         "S2": {"fixed_pct": 1.0, "tp1": 2.0, "tp2": 4.0, "window": 48}}
K_GRID = {"S1": (1.0, 1.5, 2.0, 2.5, 3.0), "S2": (2.0, 3.0, 4.0, 5.0, 6.0)}
ATR_BASES = {"atr14": 14, "atr144": 144}
START_BALANCE, RISK_PCT, LOT_STEP, OZ_PER_LOT = 50_000.0, 2.0, 0.01, 100.0
HALF = pd.Timestamp("2025-07-01")
RECENT = pd.Timestamp("2026-07-14")
BOOT, SEED = 10_000, 20261014


def load_price() -> pd.DataFrame:
    frames = [pd.read_csv(INPUTS[k], usecols=[0, 1, 2, 3, 4], encoding="utf-8-sig")
              for k in ("price_30m_history", "price_30m_recent")]
    price = pd.concat(frames, ignore_index=True)
    price["time"] = pd.to_datetime(price["time"].str[:19])
    price = price.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    tr = pd.concat([price["high"] - price["low"], (price["high"] - price["close"].shift()).abs(),
                    (price["low"] - price["close"].shift()).abs()], axis=1).max(axis=1)
    for name, n in ATR_BASES.items():
        price[name] = tr.ewm(alpha=1 / n, adjust=False).mean()
    return price


def replay(entry_i: int, entry: float, stop_pts: float, spec: dict, price: dict) -> tuple[float, int]:
    """Two equal lots, one stop; returns (trade R in units of the full risk, exit bar index)."""
    stop, tp1, tp2 = entry - stop_pts, entry + spec["tp1"] * stop_pts, entry + spec["tp2"] * stop_pts
    last = min(entry_i + spec["window"], len(price["close"]) - 1)
    lots = {"a": None, "b": None}
    for j in range(entry_i, last + 1):
        if price["low"][j] <= stop:
            for k in lots:
                if lots[k] is None:
                    lots[k] = -1.0
            return (lots["a"] + lots["b"]) / 2, j
        if lots["a"] is None and price["high"][j] >= tp1:
            lots["a"] = spec["tp1"]
        if lots["b"] is None and price["high"][j] >= tp2:
            lots["b"] = spec["tp2"]
        if lots["a"] is not None and lots["b"] is not None:
            return (lots["a"] + lots["b"]) / 2, j
    exit_r = (price["close"][last] - entry) / stop_pts
    return ((lots["a"] if lots["a"] is not None else exit_r) + (lots["b"] if lots["b"] is not None else exit_r)) / 2, last


def variants(strategy: str) -> list[tuple[str, str | None, float | None]]:
    out = [("fixed", None, None)]
    for atr in ATR_BASES:
        for k in K_GRID[strategy]:
            out.append((f"{atr}_x{k:g}", atr, k))
    return out


def replay_all(trades: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    index = {t: i for i, t in enumerate(price["time"])}
    arrays = {c: price[c].to_numpy() for c in ("high", "low", "close", *ATR_BASES)}
    times = price["time"].to_numpy()
    rows = []
    for t in trades.itertuples():
        i = index.get(t.entry_time)
        if i is None or i == 0:
            continue
        spec = EXITS[t.strategy]
        for name, atr, k in variants(t.strategy):
            stop_pts = t.entry_price * spec["fixed_pct"] / 100 if atr is None else k * arrays[atr][i - 1]
            r, j = replay(i, t.entry_price, stop_pts, spec, arrays)
            rows.append({"strategy": t.strategy, "trade_id": t.trade_id, "entry_time": t.entry_time,
                         "exit_time": pd.Timestamp(times[j]), "variant": name, "stop_pts": stop_pts,
                         "stop_pct": 100 * stop_pts / t.entry_price, "r": r, "tester_r": t.r,
                         "half": "first" if t.entry_time < HALF else "second"})
    return pd.DataFrame(rows)


def per_trade(frame: pd.DataFrame) -> dict:
    r = frame["r"].to_numpy(float)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(r), "win_rate": round(100 * (r > 0).mean(), 1) if len(r) else None,
            "mean_r": round(float(r.mean()), 3) if len(r) else None,
            "profit_factor": round(gains / losses, 2) if losses > 0 else None,
            "median_stop_pct": round(float(frame["stop_pct"].median()), 3) if len(r) else None}


def account(rep: pd.DataFrame, pick: dict[str, str], sizing: str) -> dict:
    """Both strategies in one account, one position at a time, 2% of the balance per trade."""
    rows = pd.concat([rep[(rep["strategy"] == s) & (rep["variant"] == v)] for s, v in pick.items()])
    rows = rows.sort_values("entry_time")
    balance, peak, max_dd, busy_until = START_BALANCE, START_BALANCE, 0.0, pd.Timestamp.min
    week_key, week_balance, streak, worst_streak, taken, skipped = None, START_BALANCE, 0, 0, [], 0
    weekly = {}
    pending: list[tuple[pd.Timestamp, float]] = []
    for t in rows.itertuples():
        for when, pnl in [p for p in pending if p[0] <= t.entry_time]:
            balance += pnl
        pending = [p for p in pending if p[0] > t.entry_time]
        key = (t.entry_time - pd.Timedelta(days=t.entry_time.dayofweek)).normalize()
        if key != week_key:
            week_key, week_balance = key, balance
        if t.entry_time < busy_until:
            skipped += 1
            continue
        basis = week_balance if sizing == "weekly" else balance
        # "fixed_1000" (descriptive, added at run time — decision_log): 2% of the starting 50,000, no compounding.
        risk = START_BALANCE * RISK_PCT / 100 if sizing == "fixed_1000" else basis * RISK_PCT / 100
        lots = math.floor(risk / 2 / (t.stop_pts * OZ_PER_LOT) / LOT_STEP + 1e-9) * LOT_STEP
        if lots < LOT_STEP:
            skipped += 1
            continue
        pnl = 2 * lots * OZ_PER_LOT * t.stop_pts * t.r
        pending.append((t.exit_time, pnl))
        busy_until = t.exit_time
        taken.append({"entry_time": t.entry_time, "pnl": pnl, "risk": 2 * lots * OZ_PER_LOT * t.stop_pts})
        weekly[key] = weekly.get(key, 0.0) + pnl
        streak = streak + 1 if pnl < 0 else 0
        worst_streak = max(worst_streak, streak)
        peak_candidate = balance + sum(p for _, p in pending)
        peak = max(peak, peak_candidate)
        max_dd = max(max_dd, (peak - peak_candidate) / peak)
    balance += sum(p for _, p in pending)
    taken_frame = pd.DataFrame(taken)
    recent = taken_frame[taken_frame["entry_time"] >= RECENT] if len(taken_frame) else taken_frame
    return {"final_balance": round(balance, 0), "return_pct": round(100 * (balance / START_BALANCE - 1), 1),
            "max_drawdown_pct": round(100 * max_dd, 1), "trades": len(taken), "skipped_overlap_or_size": skipped,
            "longest_losing_streak": worst_streak,
            "worst_week_usd": round(min(weekly.values()), 0) if weekly else None,
            "median_risk_usd": round(float(taken_frame["risk"].median()), 0) if len(taken_frame) else None,
            "recent_3m_pnl_usd": round(float(recent["pnl"].sum()), 0) if len(recent) else 0.0}


def paired_boot(diff: np.ndarray, rng: np.random.Generator) -> float:
    centred = diff - diff.mean()
    draws = rng.choice(centred, size=(BOOT, len(diff)), replace=True).mean(axis=1)
    return round(float((np.abs(draws) >= abs(diff.mean()) - 1e-12).mean()), 4)


def bh(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out, running, m = [0.0] * len(ps), 1.0, len(ps)
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, ps[i] * m / rank)
        out[i] = round(running, 4)
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    price = load_price()
    s1, _ = base.load_trades("S1")
    s2, _ = base.load_trades("S2")
    rep = replay_all(pd.concat([s1, s2], ignore_index=True), price)

    fixed = rep[rep["variant"] == "fixed"]
    calibration = {s: {"n": len(g), "same_sign_pct": round(100 * (np.sign(g["r"]) == np.sign(g["tester_r"])).mean(), 1),
                       "corr": round(float(np.corrcoef(g["r"], g["tester_r"])[0, 1]), 3),
                       "replay_mean_r": round(float(g["r"].mean()), 3), "tester_mean_r": round(float(g["tester_r"].mean()), 3)}
                   for s, g in fixed.groupby("strategy")}

    out, ps, names = {}, [], []
    for strategy in ("S1", "S2"):
        other = "S2" if strategy == "S1" else "S1"
        base_rows = rep[(rep["strategy"] == strategy) & (rep["variant"] == "fixed")].set_index("trade_id")
        res = {}
        for name, _atr, _k in variants(strategy):
            g = rep[(rep["strategy"] == strategy) & (rep["variant"] == name)]
            entry = {"all": per_trade(g), "halves": {h: per_trade(x) for h, x in g.groupby("half")},
                     "recent_3m": per_trade(g[g["entry_time"] >= RECENT]),
                     "account_per_trade": account(rep, {strategy: name, other: "fixed"}, "per_trade"),
                     "account_weekly": account(rep, {strategy: name, other: "fixed"}, "weekly"),
                     "account_fixed_1000": account(rep, {strategy: name, other: "fixed"}, "fixed_1000")}
            if name != "fixed":
                v = g.set_index("trade_id")["r"]
                diff = (v - base_rows["r"]).dropna().to_numpy(float)
                entry["mean_r_diff"] = round(float(diff.mean()), 3)
                entry["p"] = paired_boot(diff, rng)
                halves_better = all(entry["halves"][h]["mean_r"] > per_trade(base_rows[base_rows["half"] == h].reset_index())["mean_r"]
                                    for h in ("first", "second"))
                entry["better_both_halves"] = halves_better
                ps.append(entry["p"])
                names.append((strategy, name))
            res[name] = entry
        out[strategy] = res
    for (strategy, name), adjusted in zip(names, bh(ps)):
        e = out[strategy][name]
        e["bh_p"] = adjusted
        base_return = out[strategy]["fixed"]["account_per_trade"]["return_pct"]
        if e["p"] < 0.05 and e["better_both_halves"] and e["account_per_trade"]["return_pct"] > base_return:
            e["label"] = "可用"
        elif e["better_both_halves"]:
            e["label"] = "參考"
        else:
            e["label"] = "無證據"

    results = {
        "schema_version": 1, "study_id": STUDY_ID,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {k: {"path": v.relative_to(ROOT).as_posix(), "sha256": base.sha256(v)} for k, v in INPUTS.items()},
        "price_range": [price["time"].min().isoformat(), price["time"].max().isoformat()],
        "account": {"start_balance": START_BALANCE, "risk_pct": RISK_PCT},
        "calibration_fixed_vs_tester": calibration,
        "baseline_both_fixed": {"per_trade_sizing": account(rep, {"S1": "fixed", "S2": "fixed"}, "per_trade"),
                                "weekly_sizing": account(rep, {"S1": "fixed", "S2": "fixed"}, "weekly"),
                                "fixed_1000_sizing": account(rep, {"S1": "fixed", "S2": "fixed"}, "fixed_1000"),
                                "s2_atr14_x2_fixed_1000": account(rep, {"S1": "fixed", "S2": "atr14_x2"}, "fixed_1000"),
                                "s2_atr144_x2_fixed_1000": account(rep, {"S1": "fixed", "S2": "atr144_x2"}, "fixed_1000")},
        "variants": out,
        "recent_atr": {name: {"median_last_3m": round(float(price.loc[price["time"] >= RECENT, name].median()), 2),
                              "median_2024": round(float(price.loc[price["time"].dt.year == 2024, name].median()), 2)}
                       for name in ATR_BASES},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"calibration": calibration, "baseline": results["baseline_both_fixed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
