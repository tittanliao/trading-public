#!/usr/bin/env python3
"""RS-TX-20261105-001 — TAIEX: buy the first touch of a Fibonacci retracement of the 252-session high.

Design frozen in research/studies/RS-TX-20261105-001/decision_log.md before any outcome. For each
day, H = highest high of the previous 252 completed sessions, L = lowest low of that window on or
before H. When H/L - 1 >= threshold, layers H - f*(H-L) (f = 0.236 ... 0.786) are bought on their
first touch per setup (one setup per H date). Outcomes: race back to H vs. break of the next layer
(or L), fixed 20/60/120/250-session holds, and drawdown after entry, against any-day and
same-distance-from-high controls. Research only; nothing here feeds a live rule.

    /opt/homebrew/bin/python3.12 scripts/research/build_tx_fib_252d_pullback.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-TX-20261105-001"
TAIPEI = timezone(timedelta(hours=8))
WINDOW = 252
RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]
THRESHOLDS = [0.20, 0.30, 0.50]
PRIMARY_THR = 0.20
HOLDS = [20, 60, 120, 250]
COST = 0.003 + 2 * 0.001425  # RS-TX-20261018-001 TAIEX round trip
MATCH_BAND = 0.01
BOOT = 2000
SEED = 20261105

LIMITATIONS_ZH = [
    "這個題目是在 2026 年下跌開始之後才提出的。現在這段行情的觸發（0.236 層 2026-11-02；0.382 層 2026-11-06 跳空開盤、0.5 層同日盤中）只有已分勝負的部分進入勝率統計（0.236、0.382 對下一層都已判負），其餘仍未定、不進分母；固定持有還沒到期，不在持有統計內。研究本身沒有調參，但題目的選擇時機有事後性。",
    "前後期不穩定：0.236 層「先回前高 vs 下一層」1998–2012 為 16.7%（n=12）、2013–2026 為 57.1%（n=14）；250 日上漲比例 41.7% → 75.0%。早期樣本被 2000、2007–08、2011 空頭主導，不能當成穩定規律。",
    "隨機漫步基準不含漂移，台股長期上漲，真實基準會更高；所以「Fib 勝率 vs 基準」的比較對 Fib 已偏寬，基準是偏低的。",
    "加權指數是價格指數，不含股息（約每年 3–4%），長期持有的絕對報酬被低估；Fib 與對照組都是純價格，彼此比較是公平的。",
    "觸發之間高度重疊：同一段行情的 60–250 日持有共用同一條價格路徑；以行情段為單位的重抽樣才是誠實的區間，Wilson 偏樂觀。",
    "日線無法排出盤中高低先後；同天兩邊都碰到算負，並逐筆列出。",
    "現貨指數不能直接交易；成本採加權指數 ETF／個股慣例，台指期／小台的成本與基差不同。",
    "資料檔 TWII_1d_to_2026-11-06.csv 在 2026-11-07（週六）取得，最後一列 2026-11-06 成交量非 0，視為收盤定值；與先前 11-05 版在 7,190 個共同日期的開高低收完全相同（只有 2026-11-05 的成交量由 0 補上）。",
    "只是描述，不是因果。",
]

PRIMARY_CSV = "local-inputs/TWII_1d_to_2026-11-06.csv"
PINNED_CSV = "local-inputs/TWII_1d.csv"
MXF_CSV = "local-inputs/TAIFEX_DLY_MXF1!, 1D.csv"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    cols = ["date", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    return df[cols].sort_values("date").reset_index(drop=True)


def wilson(k: int, n: int) -> list | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def swings(df: pd.DataFrame, include_today: bool = False):
    """Per day t: H, iH, L, iL over sessions t-252..t-1 (or t-251..t when include_today)."""
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    H = np.full(n, np.nan); L = np.full(n, np.nan)
    iH = np.full(n, -1); iL = np.full(n, -1)
    for t in range(WINDOW - 1 if include_today else WINDOW, n):
        a, b = (t - WINDOW + 1, t + 1) if include_today else (t - WINDOW, t)
        seg = hi[a:b]
        j = a + int(len(seg) - 1 - np.argmax(seg[::-1]))  # latest date among ties
        lseg = lo[a:j + 1]
        k = a + int(np.argmin(lseg))
        idx = t
        H[idx], iH[idx], L[idx], iL[idx] = hi[j], j, lo[k], k
    return H, iH, L, iL


def race(df: pd.DataFrame, t: int, fill: float, target: float, stop: float, start_same_day: bool) -> dict:
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    if start_same_day and lo[t] < stop:
        return {"outcome": "loss", "days": 0, "ambiguous": False}
    for s in range(t + 1, n):
        up, dn = hi[s] >= target, lo[s] < stop
        if up and dn:
            return {"outcome": "loss", "days": s - t, "ambiguous": True}
        if up:
            return {"outcome": "win", "days": s - t, "ambiguous": False}
        if dn:
            return {"outcome": "loss", "days": s - t, "ambiguous": False}
    return {"outcome": "open", "days": n - 1 - t, "ambiguous": False}


def find_triggers(df: pd.DataFrame, sw, thr: float, mode: str) -> tuple[list[dict], dict]:
    H, iH, L, iL = sw
    op, hi, lo, cl = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    done: dict[int, set] = {}
    skipped_newhigh = 0
    stale = 0
    out = []
    for t in range(WINDOW, len(df)):
        if np.isnan(H[t]) or H[t] / L[t] - 1 < thr:
            continue
        rng = H[t] - L[t]
        trig = done.setdefault(int(iH[t]), set())
        for f in RATIOS:
            if f in trig:
                continue
            lvl = H[t] - f * rng
            hit = lo[t] <= lvl if mode == "touch" else cl[t] <= lvl
            if not hit:
                continue
            # First time since H: no earlier session after H already reached the level. Without
            # this, a setup that appears when an older, higher H leaves the window (or when L rolls
            # up) would "touch" every layer at once with price already far below them.
            j = int(iH[t])
            prior = lo[j + 1:t] if mode == "touch" else cl[j + 1:t]
            if len(prior) and prior.min() <= lvl:
                trig.add(f)
                stale += 1
                continue
            if mode == "touch" and hi[t] > H[t]:
                skipped_newhigh += 1
                trig.add(f)
                continue
            trig.add(f)
            fill = min(op[t], lvl) if mode == "touch" else cl[t]
            ni = RATIOS.index(f)
            # Rally low as seen on H's own date: lowest low of the (up to) 252 sessions ending on H.
            # Computed directly from the bars (the old iL[iH+1] lookup wrapped to the last row when
            # iH+1 < 252; the data never hit that case, so results are unchanged).
            a0 = max(0, j - WINDOW + 1)
            ep_low = a0 + int(np.argmin(lo[a0:j + 1]))
            next_lvl = H[t] - RATIOS[ni + 1] * rng if ni + 1 < len(RATIOS) else L[t]
            out.append({
                "ratio": f, "t": t, "date": dates[t], "year": int(dates[t][:4]),
                "H": float(H[t]), "H_date": dates[int(iH[t])], "L": float(L[t]), "L_date": dates[int(iL[t])],
                "rise_pct": float(100 * (H[t] / L[t] - 1)), "level": float(lvl), "fill": float(fill),
                "gap_fill": bool(mode == "touch" and op[t] < lvl),
                "next_level": float(next_lvl), "dist_from_H_pct": float(100 * (fill / H[t] - 1)),
                # Episode = the rally low as seen when H was made (window ending on H's date); stable
                # when L later rolls up, so one rally is one cluster.
                "episode": dates[ep_low],
                "L_rolled_since_H": bool(int(iL[t]) != ep_low),
            })
    return out, {"skipped_same_day_new_high": skipped_newhigh, "skipped_already_below_since_H": stale}


def enrich(df: pd.DataFrame, trigs: list[dict], mode: str) -> None:
    lo, cl = df["low"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    same = mode == "touch"
    for r in trigs:
        t = r["t"]
        r["race_next"] = race(df, t, r["fill"], r["H"], r["next_level"], same)
        r["race_L"] = race(df, t, r["fill"], r["H"], r["L"], same)
        start = t if same else t + 1
        for N in HOLDS:
            if t + N < n:
                ret = cl[t + N] / r["fill"] - 1
                mae = lo[start:t + N + 1].min() if t + N + 1 > start else r["fill"]
                r[f"ret_{N}"] = float(ret)
                r[f"pts_{N}"] = float(cl[t + N] - r["fill"])
                r[f"mae_pct_{N}"] = float(min(0.0, mae / r["fill"] - 1))
                r[f"mae_pts_{N}"] = float(min(0.0, mae - r["fill"]))
            else:
                r[f"ret_{N}"] = None
        rr = r["race_L"]
        end = t + rr["days"]
        seg = lo[start:end + 1] if end + 1 > start else np.array([r["fill"]])
        r["mae_pct_until_race_L"] = float(min(0.0, seg.min() / r["fill"] - 1))
        r["mae_pts_until_race_L"] = float(min(0.0, seg.min() - r["fill"]))


def daily_context(df: pd.DataFrame, sw) -> pd.DataFrame:
    H, iH, L, iL = sw
    cl = df["close"].to_numpy()
    ctx = pd.DataFrame({"t": np.arange(len(df)), "dd": cl / H - 1, "rise": H / L - 1})
    for N in HOLDS:
        fwd = np.full(len(df), np.nan)
        fwd[:-N] = cl[N:] / cl[:-N] - 1
        ctx[f"fwd_{N}"] = fwd
    return ctx


def summarise_ret(vals: list[float]) -> dict:
    a = np.array(vals, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "up_pct": round(100 * float((a > 0).mean()), 2),
            "up_net_pct": round(100 * float((a > COST).mean()), 2),
            "mean_pct": round(100 * float(a.mean()), 3), "median_pct": round(100 * float(np.median(a)), 3),
            "mean_net_pct": round(100 * float(a.mean() - COST), 3),
            "p10_pct": round(100 * float(np.quantile(a, 0.10)), 3), "p90_pct": round(100 * float(np.quantile(a, 0.90)), 3)}


def cluster_boot(rows: list[dict], stat, rng: np.random.Generator) -> list | None:
    eps: dict[str, list] = {}
    for r in rows:
        eps.setdefault(r["episode"], []).append(r)
    keys = list(eps)
    if len(keys) < 2:
        return None
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [x for i in pick for x in eps[keys[i]]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.quantile(vals, 0.025)), 2), round(float(np.quantile(vals, 0.975)), 2)]


def race_table(rows: list[dict], key: str, rng) -> dict:
    res = [r for r in rows if r[key]["outcome"] != "open"]
    k = sum(r[key]["outcome"] == "win" for r in res)
    n = len(res)

    def wr(s):
        s2 = [x for x in s if x[key]["outcome"] != "open"]
        return 100 * sum(x[key]["outcome"] == "win" for x in s2) / len(s2) if s2 else None
    wins = [r for r in res if r[key]["outcome"] == "win"]
    losses = [r for r in res if r[key]["outcome"] == "loss"]
    first = {}
    for r in rows:
        first.setdefault(r["episode"], r)
    fres = [r for r in first.values() if r[key]["outcome"] != "open"]
    fk = sum(r[key]["outcome"] == "win" for r in fres)
    stop_key = "next_level" if key == "race_next" else "L"
    # Driftless random-walk benchmark: P(reach H before stop) = (fill - stop) / (H - stop).
    geo = [(r["fill"] - r[stop_key]) / (r["H"] - r[stop_key]) for r in res if r["H"] > r[stop_key]]
    return {
        "random_walk_benchmark_pct": round(100 * float(np.mean(geo)), 2) if geo else None,
        "n_resolved": n, "wins": k, "losses": n - k,
        "open": sum(r[key]["outcome"] == "open" for r in rows),
        "ambiguous_counted_as_loss": sum(r[key]["ambiguous"] for r in res),
        "win_rate_pct": round(100 * k / n, 2) if n else None,
        "win_rate_wilson95_pct": wilson(k, n),
        "win_rate_episode_bootstrap95_pct": cluster_boot(rows, wr, rng),
        "median_days_to_win": float(np.median([r[key]["days"] for r in wins])) if wins else None,
        "median_days_to_loss": float(np.median([r[key]["days"] for r in losses])) if losses else None,
        "avg_win_pct": round(100 * float(np.mean([r["H"] / r["fill"] - 1 for r in wins])), 3) if wins else None,
        "avg_loss_pct": round(100 * float(np.mean([r["next_level" if key == "race_next" else "L"] / r["fill"] - 1 for r in losses])), 3) if losses else None,
        "first_per_episode": {"n_resolved": len(fres), "wins": fk,
                              "win_rate_pct": round(100 * fk / len(fres), 2) if fres else None,
                              "win_rate_wilson95_pct": wilson(fk, len(fres))},
    }


def layer_block(rows: list[dict], ctx: pd.DataFrame, thr: float, rng) -> dict:
    by_year: dict[int, int] = {}
    for r in rows:
        by_year[r["year"]] = by_year.get(r["year"], 0) + 1
    eps = {}
    for r in rows:
        eps.setdefault(r["episode"], []).append(r)
    out = {
        "n_triggers": len(rows), "n_episodes": len(eps),
        "triggers_in_multi_trigger_episodes": sum(len(v) for v in eps.values() if len(v) > 1),
        "gap_fills": sum(r.get("gap_fill", False) for r in rows),
        "triggers_after_L_rolled": sum(r.get("L_rolled_since_H", False) for r in rows),
        "by_year": dict(sorted(by_year.items())),
        "race_next_layer": race_table(rows, "race_next", rng),
        "race_L": race_table(rows, "race_L", rng),
        "holds": {}, "drawdown": {},
    }
    for N in HOLDS:
        sub = [r for r in rows if r[f"ret_{N}"] is not None]
        s = summarise_ret([r[f"ret_{N}"] for r in sub])
        s["mean_pts"] = round(float(np.mean([r[f"pts_{N}"] for r in sub])), 1) if sub else None
        s["up_episode_bootstrap95_pct"] = cluster_boot(
            sub, lambda x: 100 * float(np.mean([y[f"ret_{N}"] > 0 for y in x])) if x else None, rng)
        # C1 / C2 distance-matched controls
        c1, c2 = [], []
        c1m, c2m = [], []
        valid = ctx[~ctx[f"fwd_{N}"].isna()]
        for r in sub:
            d = r["dist_from_H_pct"] / 100
            m = valid[(valid["dd"] - d).abs() <= MATCH_BAND]
            if len(m):
                c1.append(float((m[f"fwd_{N}"] > 0).mean())); c1m.append(float(m[f"fwd_{N}"].mean()))
            m2 = m[m["rise"] >= thr]
            if len(m2):
                c2.append(float((m2[f"fwd_{N}"] > 0).mean())); c2m.append(float(m2[f"fwd_{N}"].mean()))
        s["control_same_distance"] = {"up_pct": round(100 * float(np.mean(c1)), 2) if c1 else None,
                                      "mean_pct": round(100 * float(np.mean(c1m)), 3) if c1m else None,
                                      "triggers_matched": len(c1)}
        s["control_same_distance_and_rise"] = {"up_pct": round(100 * float(np.mean(c2)), 2) if c2 else None,
                                               "mean_pct": round(100 * float(np.mean(c2m)), 3) if c2m else None,
                                               "triggers_matched": len(c2)}
        out["holds"][str(N)] = s
        if sub:
            mp = np.array([r[f"mae_pct_{N}"] for r in sub]); mt = np.array([r[f"mae_pts_{N}"] for r in sub])
            out["drawdown"][str(N)] = {"n": len(sub), "median_pct": round(100 * float(np.median(mp)), 2),
                                       "p90_worst_pct": round(100 * float(np.quantile(mp, 0.10)), 2),
                                       "worst_pct": round(100 * float(mp.min()), 2),
                                       "median_pts": round(float(np.median(mt)), 1),
                                       "worst_pts": round(float(mt.min()), 1),
                                       "share_deeper_than_10pct": round(100 * float((mp < -0.10).mean()), 2)}
    mp = np.array([r["mae_pct_until_race_L"] for r in rows]) if rows else np.array([])
    if len(mp):
        out["drawdown"]["until_race_L"] = {"n": len(rows), "median_pct": round(100 * float(np.median(mp)), 2),
                                           "worst_pct": round(100 * float(mp.min()), 2)}
    return out


def baseline(ctx: pd.DataFrame) -> dict:
    out = {}
    for N in HOLDS:
        v = ctx[f"fwd_{N}"].dropna()
        v = v[ctx.loc[v.index, "t"] >= WINDOW]
        out[str(N)] = summarise_ret(list(v))
    return out


def run_engine(df: pd.DataFrame, label: str, rng, thresholds=THRESHOLDS) -> dict:
    sw = swings(df)
    ctx = daily_context(df, sw)
    res = {"label": label, "first_eligible_date": df["date"].iloc[WINDOW].strftime("%Y-%m-%d"),
           "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"), "sessions": len(df),
           "baseline_any_day": baseline(ctx), "by_threshold": {}}
    all_trigs = {}
    for thr in thresholds:
        tb = {}
        for mode in ("touch", "close"):
            trigs, meta = find_triggers(df, sw, thr, mode)
            enrich(df, trigs, mode)
            tb[mode] = {"meta": meta, "layers": {}}
            for f in RATIOS:
                rows = [r for r in trigs if r["ratio"] == f]
                tb[mode]["layers"][str(f)] = layer_block(rows, ctx, thr, rng)
            all_trigs[(thr, mode)] = trigs
        res["by_threshold"][f"{int(thr * 100)}pct"] = tb
    return res, all_trigs, sw


def now_row(df: pd.DataFrame, trigs: list[dict]) -> dict:
    sw = swings(df, include_today=True)
    H, iH, L, iL = sw
    t = len(df) - 1
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    c = float(df["close"].iloc[-1])
    h, l = float(H[t]), float(L[t])
    levels = {str(f): round(h - f * (h - l), 1) for f in RATIOS}
    cur_setup = [r for r in trigs if r["H_date"] == dates[int(iH[t])]]
    lows_since_H = float(df["low"].iloc[int(iH[t]):].min())
    # Alternative anchor: the TWII low on 2025-12-29, a later swing low with the same H.
    od = pd.Timestamp("2025-12-29")
    ol = df.loc[df["date"] == od, "low"]
    alt_grid = None
    if len(ol):
        l2 = float(ol.iloc[0])
        lv2 = {str(f): round(h - f * (h - l2), 1) for f in RATIOS}
        alt_grid = {"anchor_note": "alternative drawing: L = the TWII low on 2025-12-29 (a later swing low), same H as the data-defined grid",
                       "L_date": "2025-12-29", "L": round(l2, 2), "H": round(h, 2), "rise_pct": round(100 * (h / l2 - 1), 2),
                       "levels": lv2, "close_minus_level_pts": {k: round(c - v, 1) for k, v in lv2.items()},
                       "retracement_of_close": round((h - c) / (h - l2), 4)}
    return {
        "alt_anchor_twii_grid": alt_grid,
        "as_of": dates[t], "close": round(c, 2),
        **({"close_provisional_note": "volume 0 on this row at download; OHLC may be revised"}
           if "volume" in df.columns and float(df["volume"].iloc[t]) == 0 else {}),
        "window": f"{dates[t - WINDOW + 1]} .. {dates[t]} (252 sessions incl. latest)",
        "H": round(h, 2), "H_date": dates[int(iH[t])], "L": round(l, 2), "L_date": dates[int(iL[t])],
        "rise_pct": round(100 * (h / l - 1), 2), "range_pts": round(h - l, 1),
        "close_vs_H_pct": round(100 * (c / h - 1), 2), "close_vs_H_pts": round(c - h, 1),
        "retracement_of_close": round((h - c) / (h - l), 4),
        "lowest_low_since_H": round(lows_since_H, 2), "retracement_of_low": round((h - lows_since_H) / (h - l), 4),
        "levels": levels,
        "close_minus_level_pts": {k: round(c - v, 1) for k, v in levels.items()},
        "close_minus_level_pct": {k: round(100 * (c / v - 1), 2) for k, v in levels.items()},
        "current_setup_touch_triggers": [{"ratio": r["ratio"], "date": r["date"], "fill": round(r["fill"], 1)} for r in cur_setup],
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_rows(trigs: list[dict]) -> list[dict]:
    keep = ["ratio", "episode", "date", "H", "H_date", "L", "L_date", "rise_pct", "level", "fill", "gap_fill", "dist_from_H_pct", "L_rolled_since_H"]
    out = []
    for r in trigs:
        d = {k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in keep}
        d["race_next"] = r["race_next"]["outcome"]; d["race_L"] = r["race_L"]["outcome"]
        for N in HOLDS:
            d[f"ret_{N}_pct"] = None if r[f"ret_{N}"] is None else round(100 * r[f"ret_{N}"], 2)
        d["mae_pct_until_race_L"] = round(100 * r["mae_pct_until_race_L"], 2)
        out.append(d)
    return out


def charts(res: dict, now: dict, df: pd.DataFrame, outdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    blue, orange, grey, aqua = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
    lay = res["by_threshold"]["20pct"]["touch"]["layers"]
    xs = [str(f) for f in RATIOS]
    out = []
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for off, key, col, lab in [(-0.18, "race_next_layer", blue, "vs. next layer down"), (0.18, "race_L", orange, "vs. the rally low L")]:
        w = [lay[x][key]["win_rate_pct"] for x in xs]
        ci_ = [race_ci(lay[x][key]) for x in xs]
        bm = [lay[x][key]["random_walk_benchmark_pct"] for x in xs]
        xi = np.arange(len(xs)) + off
        ax.bar(xi, w, width=0.34, color=col, label=lab)
        for i, (v, (c, wil)) in enumerate(zip(w, ci_)):
            if c:
                ax.plot([xi[i], xi[i]], c, color="#0b0b0b", lw=1, ls=":" if wil else "-")
            ax.plot([xi[i] - 0.17, xi[i] + 0.17], [bm[i], bm[i]], color="#0b0b0b", lw=2.2, solid_capstyle="butt")
            ax.text(xi[i] + 0.09, (v or 0) + 1.5, f"{v:.0f}%\nbase {bm[i]:.0f}%\nn={lay[xs[i]][key]['n_resolved']}", ha="left", va="bottom", fontsize=6.5, color="#0b0b0b")
    ax.plot([], [], color="#0b0b0b", lw=2.2, label="random-walk benchmark (no drift, so low)")
    ax.plot([], [], color="#0b0b0b", lw=1, ls=":", label="dotted whisker = Wilson (bootstrap collapsed)")
    ax.set_xticks(range(len(xs)), xs); ax.set_ylim(0, 100); ax.set_ylabel("back to prior high first (%)")
    ax.set_xlabel("Fibonacci layer bought on first touch"); ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("TAIEX 1998-2026, rise >= 20%: back to the 252-day high before the stop (95% episode bootstrap) vs random-walk base", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "race_win_rate.png", dpi=130); plt.close(fig)
    out.append({"id": "race_win_rate", "file": "charts/race_win_rate.png", "title": "Back-to-high win rate by layer", "section": "performance"})

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    base = res["baseline_any_day"]["250"]["up_pct"]
    fib = [lay[x]["holds"]["250"]["up_pct"] for x in xs]
    ctl = [lay[x]["holds"]["250"]["control_same_distance"]["up_pct"] for x in xs]
    xi = np.arange(len(xs))
    ax.bar(xi - 0.18, fib, width=0.34, color=blue, label="Fib first touch")
    ax.bar(xi + 0.18, ctl, width=0.34, color=aqua, label="any day at the same distance below the high")
    ax.axhline(base, color=grey, lw=1.5, ls="--", label=f"any day ({base:.0f}%)")
    for i in range(len(xs)):
        ax.text(xi[i] - 0.18, 2, f"{fib[i]:.0f}%", ha="center", fontsize=7, color="white")
        ax.text(xi[i] + 0.18, 2, f"{ctl[i]:.0f}%", ha="center", fontsize=7, color="white")
    ax.set_xticks(xi, xs); ax.set_ylim(0, 100); ax.set_ylabel("higher after 250 sessions (%)")
    ax.set_xlabel("Fibonacci layer"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("TAIEX price index: share of buys higher 250 sessions later (gross, no dividends)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "hold250_vs_controls.png", dpi=130); plt.close(fig)
    out.append({"id": "hold250_vs_controls", "file": "charts/hold250_vs_controls.png", "title": "250-session up share vs controls", "section": "performance"})

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    d = df[df["date"] >= pd.Timestamp(now["L_date"]) - pd.Timedelta(days=60)]
    ax.plot(d["date"], d["close"], color="#0b0b0b", lw=1.2)
    for f, v in now["levels"].items():
        ax.axhline(v, color=blue, lw=0.8, ls=":")
        ax.text(d["date"].iloc[0], v, f" {f}  {v:,.0f}", va="bottom", fontsize=7, color="#52514e")
    ax.axhline(now["H"], color=orange, lw=1); ax.axhline(now["L"], color=orange, lw=1)
    ax.text(d["date"].iloc[0], now["H"], f" H {now['H']:,.0f} ({now['H_date']})", va="bottom", fontsize=7)
    ax.text(d["date"].iloc[0], now["L"], f" L {now['L']:,.0f} ({now['L_date']})", va="bottom", fontsize=7)
    ax.set_title(f"TAIEX close to {now['as_of']} ({now['close']:,.0f}) and the 252-session Fibonacci grid", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "now_grid.png", dpi=130); plt.close(fig)
    out.append({"id": "now_grid", "file": "charts/now_grid.png", "title": "Current position on the grid", "section": "performance"})
    return out


def fmt(v, d=1, pct=False):
    if v is None:
        return "—"
    return f"{v:,.{d}f}{'%' if pct else ''}"


def ci(c):
    return "—" if not c else f"{c[0]:.0f}–{c[1]:.0f}%"


def race_ci(t: dict) -> tuple[list | None, bool]:
    """Episode-bootstrap interval for a race cell; Wilson when the bootstrap collapses to a point
    (all-lose resamples give 0-0%, which looks like certainty). Returns (interval, used_wilson)."""
    b = t["win_rate_episode_bootstrap95_pct"]
    if t["n_resolved"] and (not b or b[0] == b[1]):
        return t["win_rate_wilson95_pct"], True
    return b, False


def race_ci_txt(t: dict) -> str:
    c, w = race_ci(t)
    return ci(c) + ("*" if w else "")


def write_readme(payload: dict, path: Path) -> None:
    R = payload["results"]
    base = R["baseline_any_day"]
    T = R["by_threshold"]
    now = payload["now"]
    L = []
    w = L.append
    w(f"# {STUDY_ID} — 加權指數：一年高點回落到 Fib 價位買進，勝率如何\n")
    w("> 研究用途，描述性統計。不是交易規則，不影響 `請分析`。由 `scripts/research/build_tx_fib_252d_pullback.py` 產生，勿手改。\n")
    w(f"- 資料：Yahoo `^TWII` 日線（價格指數，不含股息），{payload['data_period']['start']} ～ {payload['data_period']['end']}，"
      f"{payload['coverage']['twii_sessions']:,} 個交易日；第一個可計算日 {payload['coverage']['first_eligible']}（需先有 252 日）。"
      f"新下載檔與既有 `TWII_1d.csv` 在 {payload['coverage']['refreshed_vs_pinned_shared_dates']:,} 個共同日期完全相同（最大差 {payload['coverage']['refreshed_vs_pinned_max_abs_diff']}）。")
    w(f"- 生成時間：{payload['generated_at']}。2026-11-07 以週五收盤重跑（設計、參數、規則不變，只換資料截止日）；新舊對照見 `refresh_20261106.md`。\n")
    w("## 方法（每個數字怎麼來）\n")
    w("1. **H／L**：第 t 天看「前 252 個已收盤交易日」（t−252 … t−1，不含當天）。H = 窗口內最高的日最高價；L = 窗口內、H 那天以前（含）的最低日最低價，也就是這段上升的起點。漲幅 = H ÷ L − 1。")
    w("2. **只在漲幅夠大時算**：主版本漲幅 ≥ 20%；敏感度 ≥ 30%、≥ 50%。")
    w("3. **Fib 層**：H − f ×（H − L），f = 0.236、0.382、0.5、0.618、0.786。")
    w("4. **進場（觸價版，主版本）**：H 出現之後，第一次有一天的最低價 ≤ 該層 → 以該層價格買（若當天開盤就跳空在該層之下，以開盤價成交）。同一個 H（一段上升的同一個高點）每層只算一次；H 之後若早就跌破過該層，不算。當天若同時創新高（最高價 > H）不算，0 次。"
      "**此「首次碰到」規則是在第一輪結果出來後更正的**：凍結設計的字面是「first day t with low_t ≤ level_f」，會讓 H 之後早已跌破的層在舊 H 離開窗口時一次全觸發；更正後以研究原意 first touch 為準。"
      "修正前後的規則與數字（修正前 250 日上漲比例 65.5／63.2／72.2／69.2／72.7%，修正後 58.3／50.0／58.3／50.0／33.3%）見 `decision_log.md`「Changes after results」第 1 點。")
    w("5. **進場（收盤版）**：第一次收盤 ≤ 該層，以收盤價買。")
    w("6. **勝負（a）**：買進後，哪個先發生：最高價回到 H（勝）；或最低價跌破停損（負）。停損兩種：**下一層**（0.236→0.382 … 0.786→L）、**L**。同一天兩者都碰到算負（0 次）；觸價版進場當天的最低價若已低於停損，也算負（持有 0 天）。資料結束還沒分勝負的叫「未定」，不進分母。")
    w("7. **隨機漫步基準**：如果價格完全沒有方向、像丟銅板一樣走，先碰到 H 的機率 =（進場價 − 停損）÷（H − 停損）。這只是幾何距離，任何價位都有。勝率要高過它才算 Fib 層有額外作用。**此基準不含漂移（台股長期上漲，真實基準應更高），所以這個比較對 Fib 已偏寬。**")
    w(f"8. **固定持有（b）**：第 t 天買、第 t+N 天收盤賣，N = 20／60／120／250 交易日。上漲比例 = 報酬 > 0 的比例。淨額扣來回成本 {COST*100:.3f}%（證交稅 0.3% + 手續費 0.1425%×2，沿用 RS-TX-20261018-001 加權指數慣例）。")
    w("9. **買後最大回檔（c）**：從進場那天到第 t+N 天之間最低的日最低價，比進場價低多少（點數、%）。")
    w(f"10. **對照組**：(C0) 平常日：{payload['coverage']['first_eligible']} 起每一天收盤買，持有 N 天；(C1) 同距離：收盤離自己的 252 日高點同樣遠（±1 個百分點）的所有日子；(C2) 同距離且漲幅條件也成立的日子。C1／C2 先對每一筆觸發各自找出對照日、算上漲比例，再對觸發平均。")
    w(f"11. **區間**：Wilson 95%；以及以「行情段」為單位的重抽樣 95%（{BOOT:,} 次，種子 {SEED}）。行情段 = H 形成當下看到的那個上升起點；同一段行情的多個 H、多層觸發算同一群。重抽樣區間比較誠實，因為同一段行情的觸發不是獨立的。\n")

    lay = T["20pct"]["touch"]["layers"]
    w("## 結果一：買進後先回前高，還是先跌破？（漲幅 ≥ 20%，觸價版）\n")
    w("| 層 | 觸發次數 | 行情段數 | 先回前高 vs 先破下一層 | 隨機漫步基準 | 重抽樣 95% | 先回前高 vs 先破 L | 隨機漫步基準 | 重抽樣 95% | 回前高中位天數 |")
    w("|---|---:|---:|---|---:|---|---|---:|---|---:|")
    for f in RATIOS:
        x = lay[str(f)]; a = x["race_next_layer"]; b = x["race_L"]
        w(f"| {f} | {x['n_triggers']} | {x['n_episodes']} | {fmt(a['win_rate_pct'],1,True)}（{a['wins']}/{a['n_resolved']}） | {fmt(a['random_walk_benchmark_pct'],1,True)} | {race_ci_txt(a)} | "
          f"{fmt(b['win_rate_pct'],1,True)}（{b['wins']}/{b['n_resolved']}） | {fmt(b['random_walk_benchmark_pct'],1,True)} | {race_ci_txt(b)} | {fmt(b['median_days_to_win'],0)} |")
    _open = [f"{next(f for f in RATIOS if round(f, 2) == x['ratio'])} 層（{x['date']} 觸發；" + "、".join(n for n, k in (("對下一層", "race_next"), ("對 L", "race_L")) if x[k] == "open") + "）"
             for x in payload["triggers_20pct_touch"] if x["race_next"] == "open" or x["race_L"] == "open"]
    w("\n未定：" + ("、".join(_open) if _open else "0 筆") + "，仍在進行。同天兩邊都碰到：0 筆。")
    w("區間欄標 * 的是「重抽樣退化」：該格全部都輸時，重抽樣只會得到 0–0%，看起來像確定是 0%，所以改列 Wilson 95%（例：0/3 = 0–56%、0/10 = 0–28%）。\n")
    lc = T["20pct"]["close"]["layers"]
    w("收盤版（收盤跌破才買）：")
    w("| 層 | 次數 | 先回前高 vs 下一層 | 基準 | 區間 | 先回前高 vs L | 基準 | 區間 |")
    w("|---|---:|---|---:|---|---|---:|---|")
    for f in RATIOS:
        x = lc[str(f)]; a = x["race_next_layer"]; b = x["race_L"]
        w(f"| {f} | {x['n_triggers']} | {fmt(a['win_rate_pct'],1,True)}（{a['wins']}/{a['n_resolved']}） | {fmt(a['random_walk_benchmark_pct'],1,True)} | {race_ci_txt(a)} | {fmt(b['win_rate_pct'],1,True)}（{b['wins']}/{b['n_resolved']}） | {fmt(b['random_walk_benchmark_pct'],1,True)} | {race_ci_txt(b)} |")
    w("\n收盤版有兩筆「結構必輸」：2024-08-05 收盤 19,831 已低於 0.5 層 20,196，所以 0.382 層（停損 = 0.5 層）進場當下就已在停損之下；2011-08-05 收盤 7,853 已低於 0.786 層 7,929，所以 0.618 層（停損 = 0.786 層）同樣一進場就輸。這兩筆的隨機漫步基準是負值，也被平均進去。觸價版沒有這種情況（成交價不會低於下一層或 L）。\n")

    w("\n## 結果二：固定持有 N 天，上漲比例與報酬 vs 對照組（漲幅 ≥ 20%，觸價版，未扣股息）\n")
    w("| 持有 | 平常日 C0（n） | 層 | Fib 觸發 上漲比例（n） | 扣成本後 | 重抽樣 95% | 同距離 C1 | 同距離+漲幅 C2 | Fib 平均報酬 | C0 平均 | C1 平均 | Fib 平均點數 |")
    w("|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for N in HOLDS:
        b0 = base[str(N)]
        for f in RATIOS:
            h = lay[str(f)]["holds"][str(N)]
            if not h.get("n"):
                continue
            w(f"| {N} | {fmt(b0['up_pct'],1,True)}（{b0['n']:,}） | {f} | {fmt(h['up_pct'],1,True)}（{h['n']}） | {fmt(h['up_net_pct'],1,True)} | {ci(h['up_episode_bootstrap95_pct'])} | "
              f"{fmt(h['control_same_distance']['up_pct'],1,True)} | {fmt(h['control_same_distance_and_rise']['up_pct'],1,True)} | {fmt(h['mean_pct'],2,True)} | {fmt(b0['mean_pct'],2,True)} | {fmt(h['control_same_distance']['mean_pct'],2,True)} | {fmt(h['mean_pts'],0)} |")

    w("\n## 結果三：買了之後最多再跌多少（漲幅 ≥ 20%，觸價版）\n")
    w("| 層 | 持有 | n | 中位數 | 最差 10% | 最差 | 中位點數 | 最差點數 | 跌超過 10% 的比例 |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in RATIOS:
        for N in HOLDS:
            d = lay[str(f)]["drawdown"].get(str(N))
            if d:
                w(f"| {f} | {N} | {d['n']} | {d['median_pct']}% | {d['p90_worst_pct']}% | {d['worst_pct']}% | {d['median_pts']:,.0f} | {d['worst_pts']:,.0f} | {d['share_deeper_than_10pct']}% |")
    w("\n點數是當年的指數點數（1999 年 8,000 點的 10% 是 800 點，2026 年是 4,000 多點），跨年比較請看百分比。\n")

    w("## 敏感度：漲幅門檻（觸價版；勝率欄一律改列 Wilson 95%，小樣本與 0 勝的格不用重抽樣區間）\n")
    w("| 門檻 | 層 | 次數 | 先回前高 vs 下一層 | 先回前高 vs L | 250 日上漲比例 | 同距離 C1 |")
    w("|---|---|---:|---|---|---|---:|")
    for k in ("20pct", "30pct", "50pct"):
        for f in RATIOS:
            x = T[k]["touch"]["layers"][str(f)]
            h = x["holds"]["250"]
            ra, rb = x['race_next_layer'], x['race_L']
            wl = lambda t: f"{fmt(t['win_rate_pct'],1,True)}（{t['wins']}/{t['n_resolved']}；Wilson {ci(t['win_rate_wilson95_pct'])}）" if t['n_resolved'] else "—"
            w(f"| ≥{k[:2]}% | {f} | {x['n_triggers']} | {wl(ra)} | {wl(rb)} | {fmt(h.get('up_pct'),1,True)}（{h.get('n',0)}） | {fmt(h.get('control_same_distance',{}).get('up_pct'),1,True)} |")

    w("\n## 年份分布與重複觸發（漲幅 ≥ 20%，觸價版）\n")
    for f in RATIOS:
        x = lay[str(f)]
        w(f"- {f}：{x['n_triggers']} 次，{x['n_episodes']} 段行情；年份 " + "、".join(f"{y}×{c}" if c > 1 else str(y) for y, c in x['by_year'].items())
          + f"；觸發時 L 已因窗口滾動而改變的有 {x['triggers_after_L_rolled']} 次；跳空開盤成交 {x['gap_fills']} 次。")
    eps = payload["episodes_20pct_touch"]
    multi = [e for e in eps if e["triggers"] > 1]
    w(f"\n共 {len(eps)} 段行情；其中 {len(multi)} 段觸發了不只一層（同一段行情多次觸發）：")
    for e in multi:
        w(f"- 起點 {e['episode_low_date']}：{e['triggers']} 次（層 {', '.join(str(v) for v in e['layers'])}；高點 {e['setups']} 個），第一次 {e['first']}")
    st = payload["chronological_stability_20pct_touch"]
    w("\n前後期（先回前高 vs 下一層；250 日上漲比例）：")
    w("| 層 | 1998–2012 勝率（n） | 2013–2026 勝率（n） | 1998–2012 250 日上漲 | 2013–2026 250 日上漲 |")
    w("|---|---|---|---|---|")
    for f in RATIOS:
        x = st[str(f)]
        w(f"| {f} | {fmt(x['1998-2012']['win_rate_pct'],1,True)}（{x['1998-2012']['n']}） | {fmt(x['2013-2026']['win_rate_pct'],1,True)}（{x['2013-2026']['n']}） | {fmt(x['1998-2012_up250']['up_pct'],1,True)}（{x['1998-2012_up250']['n']}） | {fmt(x['2013-2026_up250']['up_pct'],1,True)}（{x['2013-2026_up250']['n']}） |")

    cx = payload["cross_check_mxf"]
    w(f"\n## 交叉檢查：小台 MXF1!（2012–2026-08，同規則）\n")
    w(f"每日報酬相關係數 {cx['daily_log_return_corr']}（{cx['shared_dates']:,} 個共同日期）。")
    w("| 層 | MXF 次數 | MXF 先回前高 vs 下一層 | MXF vs L | 加權同期間 次數 | 加權同期間 vs 下一層 | 加權同期間 vs L |")
    w("|---|---:|---|---|---:|---|---|")
    for f in RATIOS:
        a = cx["mxf"]["by_threshold"]["20pct"]["touch"]["layers"][str(f)]
        b = cx["twii_same_span"]["by_threshold"]["20pct"]["touch"]["layers"][str(f)]
        w(f"| {f} | {a['n_triggers']} | {fmt(a['race_next_layer']['win_rate_pct'],1,True)} | {fmt(a['race_L']['win_rate_pct'],1,True)} | {b['n_triggers']} | {fmt(b['race_next_layer']['win_rate_pct'],1,True)} | {fmt(b['race_L']['win_rate_pct'],1,True)} |")
    w("\nMXF 是期貨連續月，換月跳空與正逆價差會讓 H、L 與觸發日不同；只當方向一致性的檢查，不是第二份樣本。\n")

    w("## 現在位置\n")
    w(f"- 截至 {now['as_of']} 收盤 {now['close']:,.2f}。窗口 {now['window']}。")
    w(f"- 資料算出的上升段：L {now['L']:,.2f}（{now['L_date']}）→ H {now['H']:,.2f}（{now['H_date']}），漲 {now['rise_pct']}%，區間 {now['range_pts']:,.0f} 點。")
    w(f"- 收盤離 H {now['close_vs_H_pts']:,.0f} 點（{now['close_vs_H_pct']}%），回檔比例 {now['retracement_of_close']:.3f}；H 之後最低 {now['lowest_low_since_H']:,.2f}（回檔 {now['retracement_of_low']:.3f}）。")
    w("\n| 層 | 加權（資料定義 L=2025-11-21） | 收盤 − 該層 | 加權（另一種畫法 L=2025-12-29） | 收盤 − 該層 |")
    w("|---|---:|---:|---:|---:|")
    o = now["alt_anchor_twii_grid"]
    for f in RATIOS:
        k = str(f)
        w(f"| {f} | {now['levels'][k]:,.0f} | {now['close_minus_level_pts'][k]:+,.0f}（{now['close_minus_level_pct'][k]:+.2f}%） | {o['levels'][k]:,.0f} | {o['close_minus_level_pts'][k]:+,.0f} |")
    w(f"\n- 目前這段上升已觸發的層（觸價版）：" + "、".join(f"{x['ratio']}（{x['date']}，{x['fill']:,.0f}）" for x in now["current_setup_touch_triggers"]) + "。")
    w(f"- 另一種畫法把 L 取在 2025-12-29（加權指數低點 {o['L']:,.2f}），比資料定義的 L（2025-11-21，26,396）晚、也高，因為 252 日窗口裡 2025-11-21 才是 H 之前的最低點。兩種畫法的 0.382 差 {o['levels']['0.382'] - now['levels']['0.382']:,.0f} 點。")
    w("\n## 已知限制\n")
    for x in payload["limitations_zh"]:
        w(f"- {x}")
    w(f"- 每層次數少：0.382 {lay['0.382']['n_triggers']} 次（{lay['0.382']['n_episodes']} 段行情）、0.5 {lay['0.5']['n_triggers']} 次（{lay['0.5']['n_episodes']} 段），0.618 只有 {lay['0.618']['n_triggers']} 次、0.786 只有 {lay['0.786']['n_triggers']} 次；重抽樣區間寬 40–60 個百分點，能分辨的差距大約 ±25 個百分點以上。")
    w("- 窗口滾動時 L 會變（舊低點離開 252 日窗口），各層價位跟著變；這是照「每天看過去 252 日」的定義，已逐筆標記。")
    w("\n## 來源\n")
    w("見 `source_manifest.json`（SHA-256、大小、下載方法）。設計與變更見 `decision_log.md`。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--primary", default=PRIMARY_CSV)
    ap.add_argument("--pinned", default=PINNED_CSV)
    ap.add_argument("--mxf", default=MXF_CSV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    p, q, m = ROOT / args.primary, ROOT / args.pinned, ROOT / args.mxf
    df = load(p)
    pinned = load(q)
    mm = pinned.merge(df, on="date", suffixes=("_a", "_b"))
    max_diff = float(max((mm[c + "_a"] - mm[c + "_b"]).abs().max() for c in ("open", "high", "low", "close")))
    if max_diff > 0.01 or len(mm) != len(pinned):
        print(f"refreshed file disagrees with the pinned file (max diff {max_diff}); stop", file=sys.stderr)
        return 2

    # The backtest excludes the latest (provisional) session from outcome windows only by construction:
    # holds/races need later sessions; a touch-version trigger whose entry-day low already breaks the stop is a loss (0 days);
    # a trigger on the last day that is not decided is left open (not in the denominator).
    res, trigs, _ = run_engine(df, "TWII", rng)
    mxf = load(m)
    res_mxf, trigs_mxf, _ = run_engine(mxf, "MXF1!", rng, thresholds=[PRIMARY_THR])
    # Overlap comparison: TWII on the MXF date span
    df_span = df[(df["date"] >= mxf["date"].iloc[0]) & (df["date"] <= mxf["date"].iloc[-1])].reset_index(drop=True)
    res_span, _, _ = run_engine(df_span, "TWII on MXF span", rng, thresholds=[PRIMARY_THR])
    j = df.merge(mxf, on="date", suffixes=("_twii", "_mxf"))
    rc = np.corrcoef(np.diff(np.log(j["close_twii"])), np.diff(np.log(j["close_mxf"])))[0, 1]

    now = now_row(df, trigs[(PRIMARY_THR, "touch")])
    # chronological stability (primary, touch, race vs next layer)
    stab = {}
    for f in RATIOS:
        rows = [r for r in trigs[(PRIMARY_THR, "touch")] if r["ratio"] == f and r["race_next"]["outcome"] != "open"]
        for name, lo_, hi_ in [("1998-2012", 1998, 2012), ("2013-2026", 2013, 2026)]:
            s = [r for r in rows if lo_ <= r["year"] <= hi_]
            k = sum(r["race_next"]["outcome"] == "win" for r in s)
            stab.setdefault(str(f), {})[name] = {"n": len(s), "win_rate_pct": round(100 * k / len(s), 2) if s else None}
        for N in (250,):
            for name, lo_, hi_ in [("1998-2012", 1998, 2012), ("2013-2026", 2013, 2026)]:
                s = [r for r in trigs[(PRIMARY_THR, "touch")] if r["ratio"] == f and r[f"ret_{N}"] is not None and lo_ <= r["year"] <= hi_]
                stab[str(f)][f"{name}_up250"] = {"n": len(s), "up_pct": round(100 * float(np.mean([r[f'ret_{N}'] > 0 for r in s])), 2) if s else None}
    episodes = {}
    for r in trigs[(PRIMARY_THR, "touch")]:
        e = episodes.setdefault(r["episode"], {"episode_low_date": r["episode"], "setups": set(), "layers": [], "first": r["date"]})
        e["setups"].add(r["H_date"]); e["layers"].append(r["ratio"])
    ep_list = [{**{k: v for k, v in e.items() if k != "setups"}, "setups": len(e["setups"]), "triggers": len(e["layers"])} for e in episodes.values()]
    generated = datetime.now(TAIPEI).isoformat(timespec="seconds")
    payload = {
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": generated, "strategy_version": "none (descriptive pattern test)",
        "instrument": "TAIEX (^TWII) daily, price index",
        "data_period": {"start": df["date"].iloc[0].strftime("%Y-%m-%d"), "end": df["date"].iloc[-1].strftime("%Y-%m-%d")},
        "method": {
            "window_sessions": WINDOW, "window_rule": "H = max high of sessions t-252..t-1; L = min low of that window on or before H's date",
            "setup_id": "date of H; each layer triggers at most once per setup; episode = the rally low as seen on H's date (window ending on H), so setups and layers from one rally share a cluster",
            "ratios": RATIOS, "thresholds": THRESHOLDS, "primary_threshold": PRIMARY_THR,
            "entry_touch": "first day low <= level; fill = min(open, level); skipped if that day's high > H",
            "entry_close": "first day close <= level; fill = close",
            "race": "win = high >= H first; loss = low < stop first; stop = next deeper layer (0.786 -> L) or L; same-day both = loss (ambiguous); touch: entry-day low < stop = loss",
            "holds": HOLDS, "cost_round_trip": COST, "cost_source": "RS-TX-20261018-001 TAIEX convention: 0.3% tax + 2 x 0.1425% commission",
            "controls": {"any_day": "buy at close every eligible day (t >= 252), hold N",
                         "same_distance": f"days with close/H_t - 1 within +/-{MATCH_BAND*100:.0f} pp of the trigger's fill/H - 1; mean over days, then over triggers",
                         "same_distance_and_rise": "same, and H_t/L_t - 1 >= threshold"},
            "bootstrap": {"resamples": BOOT, "seed": SEED, "cluster": "episode (rally low date at H formation)"},
            "price_index_note": "TAIEX excludes dividends; all comparisons are price-only",
        },
        "coverage": {"twii_sessions": len(df), "first_eligible": res["first_eligible_date"], "refreshed_vs_pinned_max_abs_diff": max_diff,
                     "refreshed_vs_pinned_shared_dates": int(len(mm))},
        "results": res,
        "chronological_stability_20pct_touch": stab,
        "episodes_20pct_touch": ep_list,
        "cross_check_mxf": {"mxf": res_mxf, "twii_same_span": res_span, "daily_log_return_corr": round(float(rc), 4), "shared_dates": int(len(j))},
        "now": now,
        "triggers_20pct_touch": strip_rows(trigs[(PRIMARY_THR, "touch")]),
        "triggers_20pct_close": strip_rows(trigs[(PRIMARY_THR, "close")]),
        "limitations": [
            "TAIEX price index: no dividends (about 3-4% a year), so absolute long-hold returns understate a holder; Fib vs control comparisons are like for like.",
            "Triggers overlap: holds of 60-250 sessions from different layers and setups share the same market path; episode-cluster bootstrap is the honest interval, Wilson is optimistic.",
            "Daily bars cannot order intraday high/low; same-day ambiguities are counted as losses and listed.",
            "The cash index cannot be traded directly; costs use the TAIEX ETF/stock convention. TX/MXF futures costs and basis differ.",
            "Data refreshed 2026-11-07 to the 2026-11-06 close (TWII_1d_to_2026-11-06.csv); OHLC identical to the earlier 2026-11-05 file on all 7,190 shared dates (only the 2026-11-05 volume filled in).",
            "Descriptive, not causal; no parameter was tuned, but the question was asked after the 2026 decline began.",
        ],
        "limitations_zh": LIMITATIONS_ZH,
    }
    if args.dry_run:
        print(json.dumps(payload["now"], ensure_ascii=False, indent=1))
        return 0
    outdir = Path("reproduced")
    payload["charts"] = charts(res, now, df, outdir / "charts")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_readme(payload, outdir / "README.md")
    lay = res["by_threshold"]["20pct"]["touch"]["layers"]
    b250 = res["baseline_any_day"]["250"]
    g = lambda f, k: lay[f][k]
    lc_ = res["by_threshold"]["20pct"]["close"]["layers"]
    gc = lambda f, k: lc_[f][k]
    c2 = lambda f: lay[f]["holds"]["250"]["control_same_distance_and_rise"]["up_pct"]
    headline = {
        "sessions": len(df), "data_end": payload["data_period"]["end"],
        "n_0236": g("0.236", "n_triggers"), "n_0382": g("0.382", "n_triggers"), "n_05": g("0.5", "n_triggers"),
        "n_0618": g("0.618", "n_triggers"), "n_0786": g("0.786", "n_triggers"),
        "wr_vs_L_0236_pct": g("0.236", "race_L")["win_rate_pct"], "wr_vs_L_0382_pct": g("0.382", "race_L")["win_rate_pct"],
        "wr_vs_L_05_pct": g("0.5", "race_L")["win_rate_pct"],
        "wr_vs_next_0382_pct": g("0.382", "race_next_layer")["win_rate_pct"],
        "up250_0382_pct": g("0.382", "holds")["250"]["up_pct"], "up250_any_day_pct": b250["up_pct"],
        "now_close": now["close"], "now_retracement": now["retracement_of_close"],
    }
    findings = [
        {"tone": "info", "title": "Fib win rates match plain geometry",
         "detail": f"Back to the prior high before the rally low: 0.236 {headline['wr_vs_L_0236_pct']}% (n={g('0.236','race_L')['n_resolved']}, random-walk {g('0.236','race_L')['random_walk_benchmark_pct']}%), 0.382 {headline['wr_vs_L_0382_pct']}% (n={g('0.382','race_L')['n_resolved']}, {g('0.382','race_L')['random_walk_benchmark_pct']}%), 0.5 {headline['wr_vs_L_05_pct']}% (n={g('0.5','race_L')['n_resolved']}, {g('0.5','race_L')['random_walk_benchmark_pct']}%). Every gap is inside the episode-bootstrap interval (0.618 and 0.786 have only 4 and 3 triggers: no conclusion).",
         "title_zh": "Fib 層的勝率，跟丟銅板的幾何機率差不多",
         "detail_zh": f"買進後先回前高、再跌破 L 之前：0.236 {headline['wr_vs_L_0236_pct']}%（n={g('0.236','race_L')['n_resolved']}，隨機漫步 {g('0.236','race_L')['random_walk_benchmark_pct']}%）、0.382 {headline['wr_vs_L_0382_pct']}%（n={g('0.382','race_L')['n_resolved']}，{g('0.382','race_L')['random_walk_benchmark_pct']}%）、0.5 {headline['wr_vs_L_05_pct']}%（n={g('0.5','race_L')['n_resolved']}，{g('0.5','race_L')['random_walk_benchmark_pct']}%）。差距都在重抽樣區間內（0.618、0.786 只有 4、3 筆，不下結論）。"},
        {"tone": "info", "title": f"0.382: {g('0.382','race_next_layer')['wins']} of {g('0.382','race_next_layer')['n_resolved']} got back to the high before 0.5, but plain geometry gives only about {round(g('0.382','race_next_layer')['random_walk_benchmark_pct'])}%",
         "detail": f"{g('0.382','race_next_layer')['wins']}/{g('0.382','race_next_layer')['n_resolved']} ({headline['wr_vs_next_0382_pct']}%) vs a random-walk benchmark of {g('0.382','race_next_layer')['random_walk_benchmark_pct']}% (Wilson {g('0.382','race_next_layer')['win_rate_wilson95_pct']}; episode bootstrap {g('0.382','race_next_layer')['win_rate_episode_bootstrap95_pct']}). The gap sits inside the interval and does not show 0.382 is specially weak. Close version {gc('0.382','race_next_layer')['wins']}/{gc('0.382','race_next_layer')['n_resolved']} (Wilson {gc('0.382','race_next_layer')['win_rate_wilson95_pct']}; the bootstrap collapses to 0-0): it includes 2024-08-05, whose close (19,831) was already below the 0.5 layer (20,196), a structural loss; the 0.618 close-version row has 2011-08-05 (close 7,853 below the 0.786 layer 7,929) likewise.",
         "title_zh": f"碰到 0.382 之後，{g('0.382','race_next_layer')['n_resolved']} 次只有 {g('0.382','race_next_layer')['wins']} 次沒先跌到 0.5；但單靠幾何距離，基準本來就只有約 {round(g('0.382','race_next_layer')['random_walk_benchmark_pct'])}%",
         "detail_zh": f"{g('0.382','race_next_layer')['wins']}/{g('0.382','race_next_layer')['n_resolved']}（{headline['wr_vs_next_0382_pct']}%）對隨機漫步基準 {g('0.382','race_next_layer')['random_walk_benchmark_pct']}%；Wilson {g('0.382','race_next_layer')['win_rate_wilson95_pct']}、行情段重抽樣 {g('0.382','race_next_layer')['win_rate_episode_bootstrap95_pct']}，差距在區間內，不代表 0.382 特別弱。收盤版 {gc('0.382','race_next_layer')['wins']}/{gc('0.382','race_next_layer')['n_resolved']}（重抽樣退化成 0–0，改看 Wilson {gc('0.382','race_next_layer')['win_rate_wilson95_pct']}），其中含 2024-08-05 這一筆：收盤 19,831 已低於 0.5 層 20,196，進場當下就在停損之下，結構必輸；收盤版 0.618 層也有 2011-08-05（收盤 7,853 低於 0.786 層 7,929）。"},
        {"tone": "info", "title": "A year later Fib buys trail any day, but they match the closest control: the position after a big rally, not the Fib level",
         "detail": f"Up after 250 sessions: 0.236 {g('0.236','holds')['250']['up_pct']}%, 0.382 {headline['up250_0382_pct']}%, 0.5 {g('0.5','holds')['250']['up_pct']}% vs any day {b250['up_pct']}% (n={b250['n']:,}). Closest control (same distance below the high and the rise condition holds, C2): {c2('0.236')}% / {c2('0.382')}% / {c2('0.5')}%, almost the same as Fib. So the lag behind any day comes from buying a pullback after a big rally, not from the Fib price itself; the 11-19 point gap is below the sample's resolution (about +/-25 pp). Mean return {g('0.236','holds')['250']['mean_pct']}% / {g('0.382','holds')['250']['mean_pct']}% / {g('0.5','holds')['250']['mean_pct']}% vs {b250['mean_pct']}% are point estimates with no interval.",
         "title_zh": "一年後 Fib 價位買進落後平常日，但跟最貼近的對照組差不多：差在大漲後回檔這個位置，不是 Fib 價位本身",
         "detail_zh": f"250 個交易日後上漲比例：0.236 {g('0.236','holds')['250']['up_pct']}%、0.382 {headline['up250_0382_pct']}%、0.5 {g('0.5','holds')['250']['up_pct']}%，平常日 {b250['up_pct']}%（n={b250['n']:,}）。最貼近的對照組（同距離且漲幅條件成立，C2）：{c2('0.236')}%／{c2('0.382')}%／{c2('0.5')}%，和 Fib 幾乎一樣。所以落後平常日來自大漲後回檔的位置，不是 Fib 價位本身；差距（約 11–19 個百分點）小於本樣本可分辨的範圍（約 ±25 個百分點）。平均報酬 {g('0.236','holds')['250']['mean_pct']}%／{g('0.382','holds')['250']['mean_pct']}%／{g('0.5','holds')['250']['mean_pct']}% 對 {b250['mean_pct']}%是單點估計、無區間。"},
    ]
    manifest = {
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": generated,
        "timezone_assumptions": "Daily bars on the Taiwan exchange calendar (Asia/Taipei dates). MXF1! is the TradingView TAIFEX mini-TAIEX continuous front month, daily session as exported.",
        "coverage": f"TWII {payload['data_period']['start']} .. {payload['data_period']['end']}; MXF {res_mxf['first_eligible_date']} eligible .. {res_mxf['last_date']}",
        "acquisition": {"new_data_downloaded": True, "downloaded_on": "2026-11-07 16:19 Asia/Taipei; first refresh 2026-11-05 23:32",
                        "method": "yfinance 1.4.1 in ~/.venvs/trading-research: Ticker('^TWII').history(period='max', interval='1d', auto_adjust=False); columns date,open,high,low,close,volume",
                        "authorised_by": "research request 2026-11-05 (refresh TWII to the latest close via Yahoo); rerun 2026-11-07 to the 2026-11-06 close"},
        "sources": [
            {"role": "price_1d_TAIEX", "source_system": "Yahoo Finance ^TWII daily (TWII_1d_to_2026-11-06.csv, downloaded 2026-11-07)",
             "locator": args.primary, "sha256": sha(p), "size_bytes": p.stat().st_size, "immutable": True,
             "observed": f"rows {payload['data_period']['start']} .. {payload['data_period']['end']}; OHLC identical to TWII_1d_to_2026-11-05.csv on its 7,190 dates (only the 2026-11-05 volume filled in)", "privacy": "public_market_data"},
            {"role": "price_1d_TAIEX_pinned_check", "source_system": "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18 (pinned by RS-TX-20261018-*)",
             "locator": args.pinned, "sha256": sha(q), "size_bytes": q.stat().st_size, "immutable": True,
             "observed": "rows 1997-07-02 .. 2026-10-16; pinned by other studies and never modified by this study (hence immutable); used only to verify the refreshed file", "privacy": "public_market_data"},
            {"role": "price_1d_MXF_crosscheck", "source_system": "TradingView TAIFEX_DLY:MXF1! 1D, export",
             "locator": args.mxf, "sha256": sha(m), "size_bytes": m.stat().st_size, "immutable": True,
             "observed": "rows 2012-05-09 .. 2026-08-17; continuous front month, back-adjustment unknown", "privacy": "private_raw_source"},
        ],
    }
    (outdir / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(outdir / "results.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
