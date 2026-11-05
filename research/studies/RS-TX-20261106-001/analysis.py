#!/usr/bin/env python3
"""RS-TX-20261106-001 — TAIEX: staged buying after a big rally, held 3 months / 6 months / 1 year.

Design frozen in research/studies/RS-TX-20261106-001/decision_log.md before any outcome. Reuses the
swing definition of RS-TX-20261105-001 (H = highest high of the previous 252 sessions, L = lowest low
on or before H). When H/L - 1 >= threshold, a plan buys staged units on the first touch of its levels:
primary = Fibonacci three waves (0.382 / 0.5 / 0.618 retracement, 3/9/3 units; also 1/1/1);
comparison = discounts from H (95/90/85/80%, and 90/85/80%). All filled units exit together
63 / 126 / 252 sessions after the first fill. Controls: lump sum on the first-fill day, any day, and
same distance below the high after a rally. Research only; nothing here feeds a live rule.

    /opt/homebrew/bin/python3.12 scripts/research/build_tx_staged_pullback_holds.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import study_package as pkg  # noqa: E402
from build_tx_fib_252d_pullback import load, swings, wilson  # noqa: E402  (same swing rule as RS-TX-20261105-001)

STUDY_ID = "RS-TX-20261106-001"
TAIPEI = timezone(timedelta(hours=8))
WINDOW = 252
THRESHOLDS = [0.20, 0.30, 0.50]
PRIMARY_THR = 0.20
HOLDS = [63, 126, 252]
HOLD_ZH = {63: "三個月（63 日）", 126: "半年（126 日）", 252: "一年（252 日）"}
COST = 0.003 + 2 * 0.001425  # 0.585% round trip per unit, TAIEX convention of RS-TX-20261018-001
MATCH_BAND = 0.01
BOOT = 2000
SEED = 20261106
PRIMARY_CSV = "local-inputs/TWII_1d_to_2026-11-05.csv"
EXPECTED_SHA = "980bd3fe8ab26e71749f5b571a88b31f8ce7f57137d78a70f4b2c9dc710aabbd"

PLANS = {
    "fib_339": {"kind": "fib", "levels": [0.382, 0.5, 0.618], "w": [3, 9, 3],
                "zh": "Fib 三波 3/9/3（主版本）", "en": "Fib three waves 3/9/3"},
    "fib_111": {"kind": "fib", "levels": [0.382, 0.5, 0.618], "w": [1, 1, 1],
                "zh": "Fib 三波等份 1/1/1", "en": "Fib three waves 1/1/1"},
    "disc4": {"kind": "disc", "levels": [0.95, 0.90, 0.85, 0.80], "w": [1, 1, 1, 1],
              "zh": "打折四份 95/90/85/80%", "en": "Discount 95/90/85/80"},
    "disc3": {"kind": "disc", "levels": [0.90, 0.85, 0.80], "w": [1, 1, 1],
              "zh": "打折三份 90/85/80%", "en": "Discount 90/85/80"},
}
PRIMARY = "fib_339"


def level_price(plan: dict, k: int, H: float, L: float) -> float:
    f = plan["levels"][k]
    return H - f * (H - L) if plan["kind"] == "fib" else f * H


def find_groups(df: pd.DataFrame, sw, plan: dict, thr: float) -> tuple[list[dict], dict]:
    H, iH, L, iL = sw
    op, hi, lo, cl = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    n = len(df)
    seen: set[int] = set()
    meta = {"skipped_stale_first_level": 0, "skipped_same_day_new_high": 0}
    out = []
    for t in range(WINDOW, n):
        if np.isnan(H[t]):
            continue
        s = int(iH[t])
        if s in seen:
            continue
        lvl1 = level_price(plan, 0, H[t], L[t])
        if lo[t] > lvl1:
            continue
        # The first level is touched today. From here on this setup is decided once.
        seen.add(s)
        if H[t] / L[t] - 1 < thr:
            continue  # rally too small for this threshold on the touch day (rise only shrinks later as L rolls up)
        prior = lo[s + 1:t]
        if len(prior) and prior.min() <= lvl1:
            meta["skipped_stale_first_level"] += 1  # already reached since H: not a first touch
            continue
        if hi[t] > H[t]:
            meta["skipped_same_day_new_high"] += 1
            continue
        h, l = float(H[t]), float(L[t])
        lv = [level_price(plan, k, h, l) for k in range(len(plan["levels"]))]
        fills: list[tuple[int, float] | None] = [None] * len(lv)
        # Fill levels while the setup is alive (window H still this one, no new high), up to the longest hold.
        for u in range(t, min(n, t + max(HOLDS))):
            if int(iH[u]) != s or hi[u] > h:
                break
            for k, v in enumerate(lv):
                if fills[k] is None and lo[u] <= v:
                    fills[k] = (u, float(min(op[u], v)))
            if all(f is not None for f in fills):
                break
        a0 = max(0, s - WINDOW + 1)
        ep = dates[a0 + int(np.argmin(lo[a0:s + 1]))]
        out.append({"t1": t, "date": dates[t], "year": int(dates[t][:4]), "H": h, "H_date": dates[s],
                    "L": l, "L_date": dates[int(iL[t])], "rise_pct": 100 * (h / l - 1), "levels": lv,
                    "fills": fills, "episode": ep, "first_fill_dist": fills[0][1] / h - 1})
    return out, meta


def evaluate(df: pd.DataFrame, g: dict, plan: dict, N: int) -> dict | None:
    lo, cl = df["low"].to_numpy(), df["close"].to_numpy()
    t1, e = g["t1"], g["t1"] + N
    if e >= len(df):
        return None
    w = plan["w"]
    filled = [(k, f) for k, f in enumerate(g["fills"]) if f is not None and f[0] <= e - 1]
    wsum = sum(w[k] for k, _ in filled)
    num = sum(w[k] * (cl[e] / p - 1 - COST) for k, (_, p) in filled)
    inv = num / wsum
    whole = num / sum(w)
    dd_inv, dd_whole = 0.0, 0.0
    for d in range(t1, e + 1):
        act = [(k, p) for k, (u, p) in filled if u <= d]
        if not act:
            continue
        v = sum(w[k] * (lo[d] / p - 1) for k, p in act)
        dd_inv = min(dd_inv, v / sum(w[k] for k, _ in act))
        dd_whole = min(dd_whole, v / sum(w))
    p0 = g["fills"][0][1]
    lump = cl[e] / p0 - 1 - COST
    lump_dd = min(0.0, lo[t1:e + 1].min() / p0 - 1)
    return {"ret_inv": inv, "ret_whole": whole, "dd_inv": dd_inv, "dd_whole": dd_whole,
            "units_filled": len(filled), "weight_filled_share": wsum / sum(w),
            "lump": lump, "lump_dd": lump_dd, "exit_date": df["date"].iloc[e].strftime("%Y-%m-%d")}


def context(df: pd.DataFrame, sw) -> dict:
    H, iH, L, iL = sw
    cl = df["close"].to_numpy()
    n = len(df)
    ctx = {"dd": cl / H - 1, "rise": H / L - 1, "t": np.arange(n)}
    for N in HOLDS:
        fwd = np.full(n, np.nan)
        fwd[:-N] = cl[N:] / cl[:-N] - 1 - COST
        ctx[N] = fwd
    return ctx


def c2_share(ctx: dict, dist: float, thr: float, N: int) -> float | None:
    f = ctx[N]
    m = (~np.isnan(f)) & (ctx["t"] >= WINDOW) & (np.abs(ctx["dd"] - dist) <= MATCH_BAND) & (ctx["rise"] >= thr)
    return float((f[m] > 0).mean()) if m.any() else None


def boot(rows: list[dict], stat, rng) -> list | None:
    eps: dict[str, list] = {}
    for r in rows:
        eps.setdefault(r["episode"], []).append(r)
    keys = list(eps)
    if len(keys) < 2:
        return None
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        v = stat([x for i in pick for x in eps[keys[i]]])
        if v is not None:
            vals.append(v)
    return [round(float(np.quantile(vals, 0.025)), 2), round(float(np.quantile(vals, 0.975)), 2)]


def overlap_clusters(rows: list[dict], N: int) -> list[list[dict]]:
    """Merge groups whose holds overlap (t1 within N sessions of the cluster's latest t1) or that share an episode.
    Same rule as the G0 r1 recomputation (g0_overlap.py)."""
    cl: list[list[dict]] = []
    for x in sorted(rows, key=lambda r: r["t1"]):
        if cl and (x["t1"] <= max(y["t1"] for y in cl[-1]) + N or x["episode"] in {y["episode"] for y in cl[-1]}):
            cl[-1].append(x)
        else:
            cl.append([x])
    return cl


def cluster_boot(rows: list[dict], N: int, key: str) -> tuple[int, list | None]:
    cl = overlap_clusters(rows, N)
    if len(cl) < 2:
        return len(cl), None
    rng = np.random.default_rng(SEED + 1)  # own stream: other numbers do not depend on call order
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(cl), len(cl))
        xs = [r[key] > 0 for i in pick for r in cl[i]]
        vals.append(100 * float(np.mean(xs)))
    return len(cl), [round(float(np.quantile(vals, 0.025)), 2), round(float(np.quantile(vals, 0.975)), 2)]


def pct(x, d=2):
    return None if x is None else round(100 * float(x), d)


def summarise(rows: list[dict], key: str, rng, N: int) -> dict:
    a = np.array([r[key] for r in rows], dtype=float)
    if len(a) == 0:
        return {"n": 0}
    k = int((a > 0).sum())
    b = boot(rows, lambda x: 100 * float(np.mean([y[key] > 0 for y in x])), rng)
    degenerate = k in (0, len(a))
    ncl, cb = cluster_boot(rows, N, key)
    return {"n": int(len(a)), "wins": k, "win_pct": round(100 * k / len(a), 2), "wilson95_pct": wilson(k, len(a)),
            "episode_bootstrap95_pct": b, "clusters": ncl, "cluster_bootstrap95_pct": cb, "bootstrap_degenerate": degenerate,
            "episodes": len({r["episode"] for r in rows}),
            "mean_pct": pct(a.mean(), 3), "median_pct": pct(np.median(a), 3), "worst_pct": pct(a.min(), 3),
            "best_pct": pct(a.max(), 3)}


def plan_block(df, sw, ctx, plan_key: str, thr: float, rng, anyday: dict) -> tuple[dict, list[dict]]:
    plan = PLANS[plan_key]
    groups, meta = find_groups(df, sw, plan, thr)
    block = {"meta": meta, "n_groups": len(groups), "episodes": len({g["episode"] for g in groups}),
             "by_year": {}, "holds": {}, "per_unit": {}, "open_groups": []}
    for g in groups:
        block["by_year"][str(g["year"])] = block["by_year"].get(str(g["year"]), 0) + 1
    block["by_year"] = dict(sorted(block["by_year"].items()))
    for N in HOLDS:
        rows = []
        for g in groups:
            ev = evaluate(df, g, plan, N)
            if ev is None:
                continue
            rows.append({**ev, "episode": g["episode"], "year": g["year"], "date": g["date"], "t1": g["t1"],
                         "c2": c2_share(ctx, g["first_fill_dist"], thr, N)})
        s = {"staged_invested": summarise(rows, "ret_inv", rng, N),
             "staged_whole_plan_mean_pct": pct(np.mean([r["ret_whole"] for r in rows]), 3) if rows else None,
             "staged_whole_plan_median_pct": pct(np.median([r["ret_whole"] for r in rows]), 3) if rows else None,
             "staged_whole_plan_worst_pct": pct(min(r["ret_whole"] for r in rows), 3) if rows else None,
             "dd_invested_median_pct": pct(np.median([r["dd_inv"] for r in rows]), 2) if rows else None,
             "dd_invested_worst_pct": pct(min(r["dd_inv"] for r in rows), 2) if rows else None,
             "dd_whole_worst_pct": pct(min(r["dd_whole"] for r in rows), 2) if rows else None,
             "units_filled_mean": round(float(np.mean([r["units_filled"] for r in rows])), 2) if rows else None,
             "weight_filled_share_mean_pct": pct(np.mean([r["weight_filled_share"] for r in rows]), 1) if rows else None,
             "all_units_filled_n": sum(r["units_filled"] == len(plan["w"]) for r in rows),
             "lump_sum": summarise(rows, "lump", rng, N),
             "lump_dd_worst_pct": pct(min(r["lump_dd"] for r in rows), 2) if rows else None,
             "lump_dd_median_pct": pct(np.median([r["lump_dd"] for r in rows]), 2) if rows else None,
             "staged_beats_lump_n": sum(r["ret_inv"] > r["lump"] + 1e-12 for r in rows),
             "c2_same_distance_win_pct": pct(np.mean([r["c2"] for r in rows if r["c2"] is not None]), 2) if rows else None,
             "any_day": anyday[N], "periods": {}}
        for name, a, b in [("1998-2012", 1998, 2012), ("2013-2026", 2013, 2026)]:
            sub = [r for r in rows if a <= r["year"] <= b]
            if sub:
                ks = sum(r["ret_inv"] > 0 for r in sub); kl = sum(r["lump"] > 0 for r in sub)
                s["periods"][name] = {"n": len(sub), "staged_win_pct": round(100 * ks / len(sub), 2), "staged_wilson95_pct": wilson(ks, len(sub)),
                                      "lump_win_pct": round(100 * kl / len(sub), 2),
                                      "staged_mean_pct": pct(np.mean([r["ret_inv"] for r in sub]), 3),
                                      "staged_worst_pct": pct(min(r["ret_inv"] for r in sub), 3),
                                      "any_day_win_pct": anyday[N]["periods"][name]}
            else:
                s["periods"][name] = {"n": 0}
        # Post-hoc split, a stability check only: groups whose first fill is before 2025 (drops the 2025-11..2026 rally).
        sub = [r for r in rows if r["year"] < 2025]
        if sub:
            ks = sum(r["ret_inv"] > 0 for r in sub); kl = sum(r["lump"] > 0 for r in sub)
            s["periods"]["excl_2025_2026"] = {"n": len(sub), "staged_win_pct": round(100 * ks / len(sub), 2), "staged_wilson95_pct": wilson(ks, len(sub)),
                                              "lump_win_pct": round(100 * kl / len(sub), 2), "staged_mean_pct": pct(np.mean([r["ret_inv"] for r in sub]), 3),
                                              "staged_worst_pct": pct(min(r["ret_inv"] for r in sub), 3)}
        s["rows"] = [{"date": r["date"], "exit": r["exit_date"], "episode": r["episode"], "units": r["units_filled"],
                      "ret_inv_pct": pct(r["ret_inv"]), "ret_whole_pct": pct(r["ret_whole"]), "lump_pct": pct(r["lump"]),
                      "dd_inv_pct": pct(r["dd_inv"])} for r in rows]
        block["holds"][str(N)] = s
        # per unit: each filled level held N sessions from its own fill day
        cl = df["close"].to_numpy()
        pu = {}
        for k, lvl in enumerate(plan["levels"]):
            urows = []
            for g in groups:
                f = g["fills"][k]
                if f is None or f[0] + N >= len(df):
                    continue
                urows.append({"r": cl[f[0] + N] / f[1] - 1 - COST, "episode": g["episode"], "t1": f[0],
                              "c2": c2_share(ctx, f[1] / g["H"] - 1, thr, N)})
            su = summarise(urows, "r", rng, N)
            su["c2_same_distance_win_pct"] = pct(np.mean([r["c2"] for r in urows if r["c2"] is not None]), 2) if urows else None
            pu[str(lvl)] = su
        block["per_unit"][str(N)] = pu
    for g in groups:
        if g["t1"] + min(HOLDS) >= len(df) or g["t1"] + max(HOLDS) >= len(df):
            block["open_groups"].append({"date": g["date"], "H": round(g["H"], 2), "H_date": g["H_date"],
                                         "fills": [None if f is None else {"date": df["date"].iloc[f[0]].strftime("%Y-%m-%d"), "price": round(f[1], 2)} for f in g["fills"]],
                                         "resolved_holds": [N for N in HOLDS if g["t1"] + N < len(df)]})
    block["groups"] = [{"date": g["date"], "H": round(g["H"], 2), "H_date": g["H_date"], "L": round(g["L"], 2), "L_date": g["L_date"],
                        "rise_pct": round(g["rise_pct"], 2), "episode": g["episode"],
                        "levels": [round(v, 2) for v in g["levels"]],
                        "fills": [None if f is None else {"date": df["date"].iloc[f[0]].strftime("%Y-%m-%d"), "price": round(f[1], 2)} for f in g["fills"]]}
                       for g in groups]
    return block, groups


def now_row(df: pd.DataFrame, res: dict, groups_disc4: list[dict]) -> dict:
    H, iH, L, iL = swings(df, include_today=True)
    t = len(df) - 1
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    h, l, c = float(H[t]), float(L[t]), float(df["close"].iloc[-1])
    j = int(iH[t])
    lows = df["low"].to_numpy(); opens = df["open"].to_numpy()
    out = {"as_of": dates[t], "close": round(c, 2), "close_provisional_note": "2026-11-05 row: volume 0 at download (provisional)",
           "H": round(h, 2), "H_date": dates[j], "L": round(l, 2), "L_date": dates[int(iL[t])], "rise_pct": round(100 * (h / l - 1), 2),
           "close_vs_H_pct": round(100 * (c / h - 1), 2), "lowest_low_since_H": round(float(lows[j + 1:].min()), 2), "plans": {}}
    for key, plan in PLANS.items():
        lv = []
        for k in range(len(plan["levels"])):
            v = level_price(plan, k, h, l)
            hit = [u for u in range(j + 1, t + 1) if lows[u] <= v]
            fill = None
            if hit:
                u = hit[0]
                fill = {"date": dates[u], "price": round(float(min(opens[u], v)), 2)}
            lv.append({"level": plan["levels"][k], "price": round(v, 2), "pct_of_H": round(100 * v / h, 2),
                       "close_minus_level_pct": round(100 * (c / v - 1), 2), "filled": fill})
        out["plans"][key] = lv
    # Historical context for the discount plan: groups that had filled the same number of levels within the same
    # number of sessions since the first fill (no information beyond that point).
    cur = [g for g in groups_disc4 if g["H_date"] == dates[j]]
    if cur:
        g = cur[0]
        elapsed = t - g["t1"]
        k_now = sum(1 for f in g["fills"] if f is not None and f[0] <= t)
        out["disc4_current_group"] = {"first_fill_date": g["date"], "sessions_since_first_fill": elapsed, "levels_filled": k_now}
        sim = [x for x in groups_disc4 if x is not g and sum(1 for f in x["fills"] if f is not None and f[0] <= x["t1"] + elapsed) >= k_now]
        out["disc4_current_group"]["similar_history"] = {"rule": f"past discount-4 groups with >= {k_now} levels filled within {elapsed} sessions of the first fill",
                                                         "dates": [x["date"] for x in sim]}
    return out


def charts(payload: dict, df: pd.DataFrame, outdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    blue, orange, aqua, yellow, ink, muted = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
    out = []
    B = payload["results"]["20pct"]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    series = [("Fib 3/9/3 staged", blue, lambda h: B["fib_339"]["holds"][h]["staged_invested"]),
              ("Lump sum on Fib first-fill day", orange, lambda h: B["fib_339"]["holds"][h]["lump_sum"]),
              ("Discount 95/90/85/80 staged", yellow, lambda h: B["disc4"]["holds"][h]["staged_invested"])]
    xs = [str(N) for N in HOLDS]
    wbar = 0.2
    for i, (lab, col, get) in enumerate(series):
        xi = np.arange(len(xs)) + (i - 1.5) * (wbar + 0.02)
        vals = [get(h)["win_pct"] for h in xs]
        ax.bar(xi, vals, width=wbar, color=col, label=lab)
        for x, h, v in zip(xi, xs, vals):
            c = get(h)["cluster_bootstrap95_pct"] if (get(h)["cluster_bootstrap95_pct"] and not get(h)["bootstrap_degenerate"]) else get(h)["wilson95_pct"]
            ax.plot([x, x], c, color=ink, lw=1)
            ax.text(x, 2, f"{v:.0f}%\nn={get(h)['n']}", ha="center", va="bottom", fontsize=6.5, color="white")
    xi = np.arange(len(xs)) + 1.5 * (wbar + 0.02)
    c2 = [B["fib_339"]["holds"][h]["c2_same_distance_win_pct"] for h in xs]
    ax.bar(xi, c2, width=wbar, color=aqua, label="Same distance below high after a rally (C2; Fib 3/9/3 groups only)")
    for x, v in zip(xi, c2):
        ax.text(x, 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5, color="white")
    for i, h in enumerate(xs):
        v = B["fib_339"]["holds"][h]["any_day"]["win_pct"]
        ax.plot([i - 0.5, i + 0.5], [v, v], color=ink, lw=2, ls="--")
        ax.text(i + 0.5, v + 1, f"any day {v:.0f}%", ha="right", va="bottom", fontsize=7, color=ink)
    ax.set_xticks(range(len(xs)), ["3 months (63 sessions)", "6 months (126)", "1 year (252)"])
    ax.set_ylim(0, 105); ax.set_ylabel("share of groups with net return > 0 (%)")
    ax.set_title("TAIEX 1998-2026, rise >= 20% into the 252-session high: win rate after the first fill\n(whiskers = overlap-cluster bootstrap 95%; Wilson where all-win/all-lose; C2 bar belongs to Fib 3/9/3 only)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(outdir / "win_rate_by_hold.png", dpi=130); plt.close(fig)
    out.append({"file": "charts/win_rate_by_hold.png", "section": "performance", "caption": "Win rate by hold: Fib 3/9/3 vs lump sum vs discount plan vs same-distance control vs any day"})

    now = payload["now"]
    d = df[df["date"] >= pd.Timestamp("2025-09-01")]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(d["date"], d["close"], color=ink, lw=1.4, label="TAIEX close")
    x0, x1 = pd.Timestamp(now["H_date"]), d["date"].iloc[-1]  # level lines start on the high's day, not earlier
    for lv in now["plans"]["fib_339"]:
        ax.hlines(lv["price"], x0, x1, color=blue, lw=1, ls="-")
        ax.text(x1 + pd.Timedelta(days=4), lv["price"], f"Fib {lv['level']} {lv['price']:,.0f} ({lv['pct_of_H']:.1f}% of H)", ha="left", va="center", fontsize=7, color=muted)
    for lv in now["plans"]["disc4"]:
        ax.hlines(lv["price"], x0, x1, color=yellow, lw=1, ls="--")
        ax.text(x1 + pd.Timedelta(days=4), lv["price"], f"{lv['level']*100:.0f}% of H {lv['price']:,.0f}", ha="left", va="center", fontsize=7, color=muted)
        if lv["filled"]:
            ax.plot(pd.Timestamp(lv["filled"]["date"]), lv["filled"]["price"], "o", color=yellow, ms=8, mec=ink, mew=0.6)
    ax.hlines(now["H"], x0, x1, color=muted, lw=0.8, ls=":")
    ax.text(x1 + pd.Timedelta(days=4), now["H"], f"H {now['H']:,.0f} ({now['H_date']})", ha="left", va="center", fontsize=7, color=muted)
    ax.set_xlim(d["date"].iloc[0], x1 + pd.Timedelta(days=75))
    ax.plot([], [], color=blue, lw=1, label="Fib 0.382 / 0.5 / 0.618 levels (not yet touched)")
    ax.plot([], [], color=yellow, lw=1, ls="--", label="discount levels; dot = filled")
    ax.set_title(f"TAIEX to {now['as_of']} (close {now['close']:,.0f}, provisional), H {now['H']:,.0f}, L {now['L']:,.0f} ({now['L_date']})", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "now_levels.png", dpi=130); plt.close(fig)
    out.append({"file": "charts/now_levels.png", "section": "performance", "caption": "Current H, Fib and discount levels, fills so far"})
    return out


def f_(v, d=1):
    return "—" if v is None else f"{v:,.{d}f}"


def ci_txt(s: dict) -> str:
    if not s.get("n"):
        return "—"
    b = s.get("cluster_bootstrap95_pct")
    if s["bootstrap_degenerate"] or b is None:
        w = s["wilson95_pct"]
        return f"{w[0]:.0f}–{w[1]:.0f}%*"
    return f"{b[0]:.0f}–{b[1]:.0f}%"


def write_readme(p: dict, path: Path) -> None:
    R = p["results"]
    now = p["now"]
    L = []
    a = L.append
    a(f"# {STUDY_ID} — 台股大漲後分批進場，持有三個月／半年／一年的勝率（1998–2026）\n")
    a("研究用、描述性。不是交易訊號，不影響任何正式策略或 `請分析`。資料：Yahoo `^TWII` 日線（加權指數，價格指數，不含股息），"
      f"{p['data_period']['start']} 至 {p['data_period']['end']}（最後一天為暫定值），共 {p['coverage']['sessions']:,} 個交易日；"
      f"第 253 天（{p['coverage']['first_eligible']}）起才有完整一年窗口。\n")
    a("## 一句話\n")
    b = R["20pct"]["fib_339"]["holds"]
    a(f"一年漲 20% 以上之後、第一次回檔到 0.382 開始按 3/9/3 分三波買，持有三個月／半年／一年的勝率是 "
      + "／".join(f"{b[str(N)]['staged_invested']['win_pct']:.1f}%（n={b[str(N)]['staged_invested']['n']}）" for N in HOLDS)
      + "；同一天一次買足是 " + "／".join(f"{b[str(N)]['lump_sum']['win_pct']:.1f}%" for N in HOLDS)
      + "；平常任一天買是 " + "／".join(f"{b[str(N)]['any_day']['win_pct']:.1f}%（n={b[str(N)]['any_day']['n']:,}）" for N in HOLDS)
      + "；大漲後離高點同樣距離的日子是 " + "／".join(f"{b[str(N)]['c2_same_distance_win_pct']:.1f}%" for N in HOLDS)
      + "。勝率＝扣成本後報酬 > 0 的組數比例。主版本只有 " + str(b['252']['staged_invested']['n']) + " 組（" + str(b['252']['staged_invested']['episodes']) + " 段行情；持有期重疊併群後是 "
      + "／".join(str(b[str(N)]['staged_invested']['clusters']) for N in HOLDS)
      + " 群），重疊群重抽樣區間的半寬約 "
      + "／".join(f"±{(b[str(N)]['staged_invested']['cluster_bootstrap95_pct'][1]-b[str(N)]['staged_invested']['cluster_bootstrap95_pct'][0])/2:.0f}" for N in HOLDS)
      + " 個百分點（三個月／半年／一年）——比這小的差距，這個樣本分辨不出來。"
      "分批每一組都**必然**不輸第一份那天一次買足（後面的層買在更低的價位），所以「分批比一次買足勝率高」是結構結果，不是分批有優勢的證據。\n")
    a("## 方法（每個數字怎麼來）\n")
    a("1. **高點 H 與低點 L**：第 t 天往前 252 個交易日（t−252…t−1，不含當天）的最高價 H（同值取最晚一天），以及同一窗口內、H 當天或之前的最低價 L。"
      "漲幅 = H/L − 1。主版本漲幅 ≥ 20%，另報 ≥ 30%、≥ 50%。一個 H 只算一次；H 被更高的新高取代、或滑出 252 日窗口，這個 H 就結束，沒成交的份數留現金。")
    a("2. **主版本 Fib 三波 3/9/3**：價位 = H − f×(H − L)，f = 0.382、0.5、0.618，依序買 3、9、3 份（共 15 份）。另報等份 1/1/1。"
      "價位佔 H 的比例 = 1 − f×(1 − L/H)，取決於漲幅：漲 20% 時 0.382／0.5／0.618 約是 H 的 93.6%／91.7%／89.7%，漲 50% 時 87.3%／83.3%／79.4%，漲 89.3%（現在）時 82.0%／76.4%／70.8%。")
    a("3. **對照版本 打折四份**：H 的 95%、90%、85%、80% 各 1 份；**打折三份**：90%、85%、80%。")
    a("4. **第一份成交日 t1**：當天最低價 ≤ 第一層價位，且當天漲幅條件成立、當天沒有創新高（創新高那天盤中先後不明，不算，計數）。"
      "**若 H 之後、t1 之前已經有一天碰過第一層，整組不算**（例如更高的舊 H 滑出窗口、或 L 往上滾，價格其實早就在下面）——這是 RS-TX-20261105-001 G0 審查的教訓，"
      f"主版本因此略過 {R['20pct']['fib_339']['meta']['skipped_stale_first_level']} 組、打折四份略過 {R['20pct']['disc4']['meta']['skipped_stale_first_level']} 組。")
    a("5. **價位在 t1 凍結**，之後 L 怎麼滾都不動。後面幾層各自在第一次最低價 ≤ 價位那天成交；成交價 = min(開盤, 價位)（跳空低開以開盤價成交）。同一天可以成交好幾層。"
      "只在這個 H 還有效、且出場日之前成交。")
    a(f"6. **出場**：從 t1 起算第 63／126／252 個交易日收盤，全部份數同日賣出。資料不夠長的組列為未結束、不算。成本每份來回 {COST*100:.3f}%（個股證交稅 0.3% + 手續費 0.1425%×2；指數基金（ETF）證交稅只有 0.1%，所以對 ETF 偏保守；分批每一份、一次買足、平常日、同距離一律同樣扣）。")
    a("7. **報酬**：已投入報酬 = Σ 份數×(出場收盤/成交價 − 1 − 成本) ÷ 已成交份數；含現金整體報酬 = 同一分子 ÷ 全部計畫份數（沒成交的錢報酬 0）。兩者正負號相同，所以勝率相同。"
      "**期間最大回檔** = 持有期間每天用當天最低價計算已投入部位的帳面報酬，取最低。")
    a("8. **對照組**：(a) 一次買足：t1 當天用第一份價格全部買進，同日出場；(b) 平常日：每一個有一年窗口的交易日收盤買、持有同樣天數、扣同樣成本；"
      "(c) 同距離：收盤離當時 252 日高點的距離與該組第一份成交價距離相差 ±1 個百分點以內、且漲幅條件成立的所有日子，收盤買、持有同樣天數，先對每組取平均勝率，再對各組平均（RS-TX-20261105-001 的 C2）。")
    a(f"9. **區間**：主區間是「重疊群」重抽樣：把持有期互相重疊（後一組的第一份成交日落在前一群最後一組第一份成交日 + 持有天數以內）或屬同一行情段的組併成一群，"
      f"以群為單位整群重抽 {BOOT:,} 次（seed {SEED + 1}）的 95% 區間；表中「組數（行情段／重疊群）」就是這兩個單位的個數。原本的行情段（H 當天看到的起漲低點）重抽樣"
      "仍保留在 results.json（`episode_bootstrap95_pct`）當敏感度：行情段的切法會把同一波長多頭切成好幾段（2006–2008、2025–2026 各被切成 3–5 段），單位數偏多、區間偏窄。"
      "Wilson 95% 也保留，但把重疊的組當獨立，最窄。重抽樣在全勝或全敗時退化成一點，改印 Wilson 並標 *。\n")

    a("## 結果一　主版本與對照（漲幅 ≥ 20%）\n")
    a("勝率欄：勝率（勝/組數）；區間欄：重疊群重抽樣 95% 區間（方法 9）。報酬欄為已投入資金的扣成本報酬。"
      "「平均」欄是已投入資金的平均，不能直接和平常日平均（全額投入）比；要比請看最後一欄「含現金整體報酬平均」。\n")
    a("| 持有 | 計畫 | 組數（行情段／重疊群） | 勝率 | 區間（重疊群重抽樣） | 平均 | 中位數 | 最差 | 期間最大回檔 中位／最差 | 含現金整體報酬 平均／最差 |")
    a("|---|---|---:|---:|---|---:|---:|---:|---|---|")
    for N in HOLDS:
        h = str(N)
        for key in ("fib_339", "fib_111", "disc4", "disc3"):
            s = R["20pct"][key]["holds"][h]
            si = s["staged_invested"]
            if not si.get("n"):
                continue
            a(f"| {HOLD_ZH[N]} | {PLANS[key]['zh']} | {si['n']}（{si['episodes']}／{si['clusters']}） | {si['win_pct']:.1f}%（{si['wins']}/{si['n']}） | {ci_txt(si)} | {f_(si['mean_pct'],2)}% | {f_(si['median_pct'],2)}% | {f_(si['worst_pct'],2)}% | {f_(s['dd_invested_median_pct'])}%／{f_(s['dd_invested_worst_pct'])}% | {f_(s['staged_whole_plan_mean_pct'],2)}%／{f_(s['staged_whole_plan_worst_pct'],2)}% |")
        s = R["20pct"]["fib_339"]["holds"][h]
        ls = s["lump_sum"]
        a(f"| {HOLD_ZH[N]} | 對照 (a) Fib 第一份那天一次買足 | {ls['n']} | {ls['win_pct']:.1f}%（{ls['wins']}/{ls['n']}） | {ci_txt(ls)} | {f_(ls['mean_pct'],2)}% | {f_(ls['median_pct'],2)}% | {f_(ls['worst_pct'],2)}% | {f_(s['lump_dd_median_pct'])}%／{f_(s['lump_dd_worst_pct'])}% | — |")
        ad = s["any_day"]
        a(f"| {HOLD_ZH[N]} | 對照 (b) 平常任一天買 | {ad['n']:,} 天 | {ad['win_pct']:.1f}% | Wilson {ad['wilson95_pct'][0]:.1f}–{ad['wilson95_pct'][1]:.1f}%（天數重疊，偏窄） | {f_(ad['mean_pct'],2)}% | {f_(ad['median_pct'],2)}% | {f_(ad['worst_pct'],2)}% | — | — |")
        a(f"| {HOLD_ZH[N]} | 對照 (c) 同距離（對主版本各組） | {s['staged_invested']['n']} 組的平均 | {f_(s['c2_same_distance_win_pct'])}% | — | — | — | — | — | — |")
    a("")
    a("補充（漲幅 ≥ 20%）：")
    for N in HOLDS:
        s = R["20pct"]["fib_339"]["holds"][str(N)]
        a(f"- {HOLD_ZH[N]}：主版本平均成交 {s['units_filled_mean']} 層、平均投入 {s['weight_filled_share_mean_pct']}% 的計畫資金，三層全成交 {s['all_units_filled_n']} 組；"
          f"分批報酬高於一次買足的組數 {s['staged_beats_lump_n']}/{s['staged_invested']['n']}（分批每組**必然** ≥ 一次買足：後面的層買在更低的價位，相等的組只成交了一層；代價是沒成交的錢沒投入。所以這個比例不是分批有優勢的證據）。")
    a("")

    a("## 結果二　每一份各自持有（從該份成交日起算，漲幅 ≥ 20%）\n")
    a("| 持有 | 計畫 | 層 | 筆數 | 勝率 | 區間 | 平均 | 最差 | 同距離對照 |")
    a("|---|---|---|---:|---:|---|---:|---:|---:|")
    for N in HOLDS:
        for key in ("fib_339", "disc4"):
            for lvl, su in R["20pct"][key]["per_unit"][str(N)].items():
                if not su.get("n"):
                    a(f"| {HOLD_ZH[N]} | {PLANS[key]['zh']} | {lvl} | 0 | — | — | — | — | — |")
                    continue
                a(f"| {HOLD_ZH[N]} | {PLANS[key]['zh']} | {lvl} | {su['n']} | {su['win_pct']:.1f}% | {ci_txt(su)} | {f_(su['mean_pct'],2)}% | {f_(su['worst_pct'],2)}% | {f_(su['c2_same_distance_win_pct'])}% |")
    a("")

    a("## 結果三　敏感度：漲幅 ≥ 30%、≥ 50%（主版本 Fib 3/9/3 與打折四份）\n")
    a("| 漲幅 | 計畫 | 持有 | 組數 | 勝率 | Wilson | 一次買足 | 同距離 | 平常日 | 最差 |")
    a("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    for thr in ("20pct", "30pct", "50pct"):
        for key in ("fib_339", "disc4"):
            for N in HOLDS:
                s = R[thr][key]["holds"][str(N)]
                si = s["staged_invested"]
                if not si.get("n"):
                    a(f"| ≥{thr[:2]}% | {PLANS[key]['zh']} | {HOLD_ZH[N]} | 0 | — | — | — | — | — | — |")
                    continue
                w = si["wilson95_pct"]
                a(f"| ≥{thr[:2]}% | {PLANS[key]['zh']} | {HOLD_ZH[N]} | {si['n']} | {si['win_pct']:.1f}% | {w[0]:.0f}–{w[1]:.0f}% | {s['lump_sum']['win_pct']:.1f}% | {f_(s['c2_same_distance_win_pct'])}% | {s['any_day']['win_pct']:.1f}% | {f_(si['worst_pct'],2)}% |")
    a("")

    a("## 結果四　前後期（事後切分，不能當結論；漲幅 ≥ 20%，依第一份成交年份）\n")
    a("**這是研究者看過結果後才做的事後切分，只是穩定性檢查，n 很小，不能當結論。**同期平常日勝率放在每格最後，前後期的市場環境不同（2013 年後多頭較長），要和同期平常日比。"
      "最後一欄「排除 2025–2026」把第一份成交在 2025 年以後的組拿掉（2025-11 到 2026 的那一波多頭），看結果有多依賴這一波；同樣是事後切分。\n")
    a("| 計畫 | 持有 | 1998–2012 組數／勝率（Wilson）／一次買足／平均／同期平常日 | 2013–2026 組數／勝率（Wilson）／一次買足／平均／同期平常日 | 排除 2025–2026 組數／勝率／一次買足 |")
    a("|---|---|---|---|---|")
    for key in ("fib_339", "disc4"):
        for N in HOLDS:
            ps = R["20pct"][key]["holds"][str(N)]["periods"]
            cells = []
            for nm in ("1998-2012", "2013-2026"):
                q = ps[nm]
                if not q["n"]:
                    cells.append("0")
                else:
                    cells.append(f"{q['n']}／{q['staged_win_pct']:.1f}%（{q['staged_wilson95_pct'][0]:.0f}–{q['staged_wilson95_pct'][1]:.0f}%）／{q['lump_win_pct']:.1f}%／{q['staged_mean_pct']:.2f}%／{q['any_day_win_pct']:.1f}%")
            ex = ps.get("excl_2025_2026")
            cells.append(f"{ex['n']}／{ex['staged_win_pct']:.1f}%／{ex['lump_win_pct']:.1f}%" if ex else "—")
            a(f"| {PLANS[key]['zh']} | {HOLD_ZH[N]} | {cells[0]} | {cells[1]} | {cells[2]} |")
    a("")
    a("## 結果五　年份分布（第一份成交年份，漲幅 ≥ 20%）\n")
    for key in ("fib_339", "disc4"):
        by = R["20pct"][key]["by_year"]
        a(f"- {PLANS[key]['zh']}（{R['20pct'][key]['n_groups']} 組、{R['20pct'][key]['episodes']} 段行情）：" + "、".join(f"{y} {c}" for y, c in by.items()))
    a("")
    a("主版本每一組（日期 = 第一份成交日；括號內為各層成交日／價，— 為沒成交）：\n")
    a("| 第一份 | H（日期） | L（日期） | 漲幅 | 0.382 | 0.5 | 0.618 | 三個月 | 半年 | 一年 | 一年一次買足 |")
    a("|---|---|---|---:|---|---|---|---:|---:|---:|---:|")
    rows = {N: {r["date"]: r for r in R["20pct"]["fib_339"]["holds"][str(N)]["rows"]} for N in HOLDS}
    for g in R["20pct"]["fib_339"]["groups"]:
        fl = ["—" if f is None else f"{f['date']} {f['price']:,.0f}" for f in g["fills"]]
        rr = [rows[N].get(g["date"]) for N in HOLDS]
        a(f"| {g['date']} | {g['H']:,.0f}（{g['H_date']}） | {g['L']:,.0f}（{g['L_date']}） | {g['rise_pct']:.1f}% | {fl[0]} | {fl[1]} | {fl[2]} | "
          + " | ".join("未結束" if r is None else f"{r['ret_inv_pct']:+.2f}%" for r in rr)
          + f" | {'未結束' if rr[2] is None else format(rr[2]['lump_pct'], '+.2f') + '%'} |")
    a("")

    a("## 結果六　打折四份與最貼近的對照並列（漲幅 ≥ 20%）\n")
    a("打折四份只和平常日比不夠；最貼近的對照是大漲後離高點同距離的日子（同距離），以及同一組在第一層（H 的 95%）那天一次買足。"
      "分批每組必然 ≥ 一次買足（結構結果），所以勝率高於一次買足不是優勢證據。同距離欄是對打折四份各組算的；結果一的同距離只屬於 Fib 主版本。\n")
    a("| 持有 | 打折四份勝率（n；重疊群數；重疊群區間） | 同距離 | 95% 一次買足 | 平常任一天 | 含現金平均報酬 | 排除 2025–2026 勝率（n） |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for N in HOLDS:
        s4 = R["20pct"]["disc4"]["holds"][str(N)]
        si = s4["staged_invested"]
        ex = s4["periods"].get("excl_2025_2026", {})
        a(f"| {HOLD_ZH[N]} | {si['win_pct']:.1f}%（{si['n']}；{si['clusters']} 群；{ci_txt(si)}） | {f_(s4['c2_same_distance_win_pct'])}% | {s4['lump_sum']['win_pct']:.1f}% | "
          f"{s4['any_day']['win_pct']:.1f}% | {f_(s4['staged_whole_plan_mean_pct'],2)}% | {f_(ex.get('staged_win_pct'))}%（{ex.get('n', 0)}） |")
    a("")
    a("讀法：三個月 55.9–93.6% 這種寬度的區間，平常日 61% 落在裡面，分不出來；一年 56.7% 還低於同距離 59.0%，含現金平均報酬是負的。結論用「沒有顯示優勢」，不是「較高」。\n")

    a("## 現在位置（截至 " + now["as_of"] + "，收盤暫定）\n")
    a(f"- 窗口含最新一天（描述用）：H {now['H']:,.2f}（{now['H_date']}）、L {now['L']:,.2f}（{now['L_date']}）、漲幅 {now['rise_pct']:.2f}%；"
      f"收盤 {now['close']:,.2f}，離 H {now['close_vs_H_pct']:.2f}%；H 之後最低 {now['lowest_low_since_H']:,.2f}。")
    for key in ("fib_339", "disc4"):
        a(f"- {PLANS[key]['zh']}：" + "；".join(
            f"{lv['level']} → {lv['price']:,.0f}（H 的 {lv['pct_of_H']:.1f}%）" + (f"，已成交 {lv['filled']['date']} @ {lv['filled']['price']:,.0f}" if lv["filled"] else f"，未成交，收盤還在上方 {lv['close_minus_level_pct']:.2f}%")
            for lv in now["plans"][key]))
    cg = now.get("disc4_current_group")
    if cg:
        a(f"- 打折四份這一組：第一份 {cg['first_fill_date']}，到現在 {cg['sessions_since_first_fill']} 個交易日已成交 {cg['levels_filled']} 層。"
          f"歷史上「第一份後 {cg['sessions_since_first_fill']} 個交易日內也已成交 ≥ {cg['levels_filled']} 層」的組：{len(cg['similar_history']['dates'])} 組（{'、'.join(cg['similar_history']['dates']) or '無'}）；"
          "這些組是看過現況後才挑的事後子集（n 很小），結果只是歷史次數比例，不是對這一次的預測；細節見 results.json `now.disc4_similar_outcomes`。")
        so = now.get("disc4_similar_outcomes", {})
        for N in HOLDS:
            q = so.get(str(N))
            if q and q["n"]:
                a(f"  - {HOLD_ZH[N]}：{q['wins']}/{q['n']} 組為正（{q['win_pct']:.1f}%，Wilson {q['wilson95_pct'][0]:.0f}–{q['wilson95_pct'][1]:.0f}%），平均 {q['mean_pct']:.2f}%、最差 {q['worst_pct']:.2f}%。")
    a(f"- Fib 主版本：現在這個 H 的 0.382 層還沒碰到，所以主版本**還沒開始**。若照歷史，漲幅 ≥ 50%（現在 {now['rise_pct']:.0f}%；這是依現況挑的事後子集，n 很小）時主版本三個月／半年／一年勝率為 "
      + "／".join(f"{R['50pct']['fib_339']['holds'][str(N)]['staged_invested'].get('win_pct', float('nan')):.1f}%（n={R['50pct']['fib_339']['holds'][str(N)]['staged_invested'].get('n', 0)}，Wilson {R['50pct']['fib_339']['holds'][str(N)]['staged_invested']['wilson95_pct'][0]:.0f}–{R['50pct']['fib_339']['holds'][str(N)]['staged_invested']['wilson95_pct'][1]:.0f}%）" for N in HOLDS)
      + "；漲幅 ≥ 20% 的全樣本見結果一。這是歷史的次數比例，不是對這一次的預測。\n")

    a("## 和既有研究的對照\n")
    a("| | 本研究 RS-TX-20261106-001 | RS-TX-20260815-001 | RS-TX-20260728-002 |")
    a("|---|---|---|---|")
    a("| 資料 | 加權指數 ^TWII 日線 1997–2026 | 小台連續月 MXF1! 2013–2026 | 小台連續月 MXF1! 2013–2025 |")
    a("| 入選條件 | 一年漲幅 ≥ 20%（另報 30%、50%），沒有 1 月開盤條件 | 7/1 高於 1 月開盤（沒有漲 20% 以上的條件） | 7/1 高於 1 月開盤（沒有漲 20% 以上的條件） |")
    a("| 低點錨 | 252 日窗口內、高點前的最低價 L | 當年 1 月開盤 | 當年 1 月開盤 |")
    a("| 高點錨 | 前 252 日最高價 H，一個 H 一組 | 年初到 7/1 的高點，第一層觸發前可更新 | 7/1 那天的價格 |")
    a("| 層 | H − f×(H−L)，f=0.382/0.5/0.618 | 1 月開盤 + r×(H−1月開盤)，r=0.618/0.5/0.382。回檔比例相同（其 0.618 = 從高點回 38.2%），但低點錨、高點錨都不同，**價位不同**：2026 年其第一層 41,512（2026-07-29 已觸發），本研究 0.382 是 " + f"{now['plans']['fib_339'][0]['price']:,.0f}" + "（未觸發） | H − f×(H−1月開盤)，命名同本研究，但低點錨是 1 月開盤，**價位不同** |")
    a("| 平常日對照 | 有（平常任一天、同距離、一次買足） | 無 | 無 |")
    a("| 份數 | 3/9/3 合併一組（另報 1/1/1、各份各自） | 3/9/3 合併 | 各層獨立 |")
    a("| 出場 | 第一份後 63／126／252 交易日 | 隔年 3/31 | 隔年 3/31 |")
    a("| 樣本 | 每個 H 一組，幾十組 | 一年一次，9 年有觸發 | 一年一次，各層 7–8 筆 |")
    a("| 成本 | 每份來回 0.585% | 未扣成本 | 未扣成本（以點數計） |")
    a("| 主要結果 | 見結果一 | 合併 9 年勝率 66.67%（Wilson 35.4–87.9%） | 0.382 50.0%（n=8）、0.5 62.5%（n=8）、0.618 57.1%（n=7） |")
    a("")
    a(p["comparison_note_zh"] + "\n")

    a("## 已知限制\n")
    for x in p["limitations_zh"]:
        a(f"- {x}")
    a("")
    a("## 來源\n")
    a(f"- `{PRIMARY_CSV}` sha256 `{p['coverage']['sha256']}`（RS-TX-20261105-001 已登記；本研究沒有新下載）。")
    a("- runner：`scripts/research/build_tx_staged_pullback_holds.py`；設計凍結與變更見 `decision_log.md`。")
    a("- 圖：`charts/win_rate_by_hold.png`（勝率比較）、`charts/now_levels.png`（現在的價位）。圖中文字為英文。")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


CARD_SUMMARY = ("After a 20%+ TAIEX rally, staged Fibonacci buying is positive about as often as buying on any day after 3 or 6 months (66.7% / 66.7% vs 61.0% / 65.4%) "
                "but only half the time after a year (50.0% vs 68.2% for any day), the same as other days at that distance below the high (50.3%). Staged never does worse than a lump sum "
                "by construction (later levels are bought lower), so staged beating a lump sum is not evidence of an edge; 12 overlapping groups, no edge shown.")

LIMITATIONS_ZH = [
    "這個題目是在 2026 年 10 月起的下跌中才提出；現在這一段（打折版已成交 3 層、Fib 主版本未開始）不在統計內。研究沒有調參，參數全是事先給定的預設值，但題目的選擇時機有事後性。",
    "組數只有幾十組、而且同一段大行情裡會連續出現好幾組（持有期互相重疊），獨立樣本更少。主區間是把持有期重疊的組併成一群再重抽樣（主版本 9/9/8 群、打折四份 20/15/11 群）；"
    "原本以行情段重抽樣會把同一波長多頭切成好幾段（2006–2008、2025–2026），單位數偏多、區間偏窄；Wilson 最窄。差距小於約 ±20 個百分點分不出來。",
    "分批與一次買足在同一組裡比較時，分批**必然**不輸一次買足（後面的層都買在更低的價位，相等的組只成交一層），所以分批勝率高於一次買足是結構結果、不是優勢的證據；"
    "但後面幾層常常沒成交，錢沒投入，所以要同時看「已投入報酬」與「含現金整體報酬」（打折四份一年含現金平均 −0.72%）。",
    "前後期不穩定：主版本一年 1998–2012 37.5%（n=8）對 2013–2026 75.0%（n=4），同期平常日 63.3%／73.9%；打折四份三個月 58.8%（n=17）對 94.7%（n=19），同期平常日 54.7%／67.9%。"
    "後期那 19 組裡有 6 組來自 2025-11 到 2026-06 這一波多頭（全部賺錢）；排除 2025–2026 後打折四份三個月 73.3%、半年 70.0%、一年 56.7%（各 n=30）。"
    "前後期與排除 2025–2026 都是看過結果後的事後切分，n 很小，只能當穩定性檢查，不能當結論。",
    "「現在位置」的兩個子集（漲幅 ≥ 50% 的主版本；打折四份「13 個交易日內已成交 ≥3 層」的相似 5 組）都是依現況挑的事後子集，n 很小，是歷史次數比例，不是對這一次的預測。",
    "加權指數是價格指數，不含股息（約每年 3–4%），長期持有的絕對報酬被低估；各計畫與對照組都是純價格，彼此比較公平。",
    "日線無法排出盤中先後；創新高當天碰到第一層的不算（計數）；成交價假設掛單在價位、跳空用開盤價。",
    "現貨指數不能直接交易；成本採個股證交稅 0.3% + 手續費 0.1425%×2（指數基金證交稅只有 0.1%，對 ETF 偏保守）；台指期的成本、基差、保證金與本研究不同（RS-TX-20260815-001 有期貨保證金壓力的分析）。",
    "2026-11-05 這一列下載時是暫定值（成交量 0）；「現在」的數字應等 Yahoo 補完後重跑。",
    "Fib 的 L 會隨窗口滾動（舊低點滑出），第一層的價位因此會移動；本研究在第一份成交日把三層凍結。",
    "只是描述，不是因果；不代表這一次會怎樣。",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--primary", default=PRIMARY_CSV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    p = ROOT / args.primary
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if args.primary == PRIMARY_CSV and digest != EXPECTED_SHA:
        print(f"input hash {digest} != registered {EXPECTED_SHA}; stop", file=sys.stderr)
        return 2
    df = load(p)
    sw = swings(df)
    ctx = context(df, sw)
    rng = np.random.default_rng(SEED)
    anyday = {}
    for N in HOLDS:
        f = ctx[N]
        v = f[(~np.isnan(f)) & (ctx["t"] >= WINDOW)]
        k = int((v > 0).sum())
        yrs = df["date"].dt.year.to_numpy()
        per = {}
        for name, a_, b_ in [("1998-2012", 1998, 2012), ("2013-2026", 2013, 2026)]:
            m = (~np.isnan(f)) & (ctx["t"] >= WINDOW) & (yrs >= a_) & (yrs <= b_)
            per[name] = round(100 * float((f[m] > 0).mean()), 2)
        anyday[N] = {"periods": per, "n": int(len(v)), "wins": k, "win_pct": round(100 * k / len(v), 2), "wilson95_pct": wilson(k, len(v)),
                     "mean_pct": pct(v.mean(), 3), "median_pct": pct(np.median(v), 3), "worst_pct": pct(v.min(), 3)}
    results, allgroups = {}, {}
    for thr in THRESHOLDS:
        key = f"{int(thr * 100)}pct"
        results[key] = {}
        for pk in PLANS:
            blk, groups = plan_block(df, sw, ctx, pk, thr, rng, anyday)
            results[key][pk] = blk
            allgroups[(key, pk)] = groups
    now = now_row(df, results, allgroups[("20pct", "disc4")])
    cg = now.get("disc4_current_group")
    if cg:
        sim = set(cg["similar_history"]["dates"])
        so = {}
        for N in HOLDS:
            rr = [r for r in results["20pct"]["disc4"]["holds"][str(N)]["rows"] if r["date"] in sim]
            k = sum(r["ret_inv_pct"] > 0 for r in rr)
            so[str(N)] = {"n": len(rr), "wins": k, "win_pct": round(100 * k / len(rr), 2) if rr else None,
                          "wilson95_pct": wilson(k, len(rr)), "mean_pct": round(float(np.mean([r["ret_inv_pct"] for r in rr])), 3) if rr else None,
                          "worst_pct": min((r["ret_inv_pct"] for r in rr), default=None)}
        now["disc4_similar_outcomes"] = so
    generated = datetime.now(TAIPEI).isoformat(timespec="seconds")
    B = results["20pct"]
    f252 = B["fib_339"]["holds"]["252"]
    fs = f252["staged_invested"]
    pe, pl = f252["periods"]["1998-2012"], f252["periods"]["2013-2026"]
    comp = (
        "結論是否一致：三份研究的定義不同（加權指數 vs 小台連續月、252 日滾動高低點 vs 1 月開盤錨、第一份後固定 63/126/252 日 vs 隔年 3/31、扣成本 vs 不扣），"
        "不能逐數字比較。Fib 的回檔比例相同，但低點錨與高點錨不同，價位不是同一條線"
        f"（2026 年：RS-TX-20260815-001 的第一層 41,512 已於 2026-07-29 觸發，本研究的 0.382 在 {now['plans']['fib_339'][0]['price']:,.0f}、尚未觸發）。"
        "兩份舊研究的入選條件是 7/1 高於 1 月開盤，沒有「漲 20% 以上」的條件，也沒有平常日對照，所以無法判斷它們的分批有沒有勝過平常日；本研究有平常日對照，結果是沒有顯示優勢。"
        f"本研究主版本一年勝率 {fs['win_pct']:.1f}%（{fs['wins']}/{fs['n']}，Wilson {fs['wilson95_pct'][0]:.0f}–{fs['wilson95_pct'][1]:.0f}%）；"
        f"只看 2013–2026（兩份小台研究的資料期間）是 {pl['staged_win_pct']:.1f}%（n={pl['n']}），1998–2012 是 {pe['staged_win_pct']:.1f}%（n={pe['n']}）（事後切分，n 很小）。"
        "RS-TX-20260815-001 的 3/9/3 合併勝率 66.67%（6/9 年，Wilson 35.4–87.9%），RS-TX-20260728-002 各層 50.0–62.5%（n=7–8）。"
        "三者的區間大幅重疊，沒有互相矛盾，但各自樣本都小，不能說「方向一致」或下勝負結論。"
        f"本研究多出的資訊是 2013 年以前：2000、2007–08 的崩跌讓 1998–2012 的一年勝率只有 {pe['staged_win_pct']:.1f}%，兩份小台研究的資料從 2013 年才開始，看不到這種情況；"
        "和 RS-TX-20261105-001（Fib 價位本身沒有優勢）不衝突。")
    payload = {
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": generated, "strategy_version": "none (descriptive staged-entry test)",
        "instrument": "TAIEX (^TWII) daily, price index",
        "data_period": {"start": df["date"].iloc[0].strftime("%Y-%m-%d"), "end": df["date"].iloc[-1].strftime("%Y-%m-%d")},
        "coverage": {"sessions": len(df), "first_eligible": df["date"].iloc[WINDOW].strftime("%Y-%m-%d"), "sha256": digest},
        "method": {
            "window_sessions": WINDOW, "window_rule": "H = max high of sessions t-252..t-1 (latest on ties); L = min low of that window on or before H",
            "setup": "one group per H per plan; alive while the window H is this H and no session's high exceeds it",
            "plans": PLANS, "primary_plan": PRIMARY, "thresholds": THRESHOLDS, "primary_threshold": PRIMARY_THR,
            "first_fill": "first day low <= first level with rise >= threshold and high <= H; skipped if any session between H and that day already reached the first level (stale)",
            "freeze": "all levels frozen at the first-fill day",
            "later_fills": "first day low <= level while alive and before the exit day; fill = min(open, level)",
            "exit": "close of t1 + N for all filled units", "holds": HOLDS,
            "cost_round_trip_per_unit": COST, "cost_source": "RS-TX-20261018-001 TAIEX convention: 0.3% tax + 2 x 0.1425% commission",
            "win": "invested net return > 0 (whole-plan return has the same sign)",
            "drawdown": "worst daily mark-to-low of filled units between t1 and exit (invested and whole-plan basis)",
            "controls": {"lump_sum": "all money at the first fill price on t1, same exit", "any_day": "close-to-close every day t >= 252, net of cost",
                         "same_distance_after_rally": f"days with close/H_t - 1 within +/-{MATCH_BAND*100:.0f} pp of the group's first-fill distance and rise_t >= threshold; buy at close, hold N, net; mean win share per group, then mean over groups"},
            "bootstrap": {"resamples": BOOT, "seed": SEED, "cluster": "episode = lowest low of the 252 sessions ending on H's date (sensitivity, key episode_bootstrap95_pct)",
                          "main_interval": "overlap clusters (key cluster_bootstrap95_pct, seed SEED+1): groups merged when t1 <= latest t1 in the cluster + N or the episode is shared"},
        },
        "results": results,
        "now": now,
        "comparison_note_zh": comp,
        "comparison_studies": {
            "RS-TX-20260815-001": {"anchor": "January open -> first-half high (MXF1!)", "plan": "3/9/3 combined", "exit": "following March 31", "win_rate_pct": 66.67, "n_years": 9, "wilson95_pct": [35.42, 87.94], "costs": "excluded"},
            "RS-TX-20260728-002": {"anchor": "January open -> July 1 (MXF1!)", "plan": "each level independent", "exit": "following March 31",
                                   "win_rate_pct": {"0.382": 50.0, "0.5": 62.5, "0.618": 57.14}, "n": {"0.382": 8, "0.5": 8, "0.618": 7}},
        },
        "limitations_zh": LIMITATIONS_ZH,
        "limitations": [
            "Few groups (tens) and overlapping holds inside one rally; the honest interval is the overlap-cluster bootstrap (groups with overlapping holds merged; episode bootstrap kept as a narrower sensitivity, Wilson narrowest); differences under about +/-20 pp are not resolvable.",
            "Staged never does worse than a lump sum inside one group (later levels are bought lower), so staged > lump is structural, not evidence of an edge; later levels often stay unfilled (cash); read invested and whole-plan returns together.",
            "Period splits (1998-2012 vs 2013-2026, excluding 2025-2026) and the 'now' subsets (rise >= 50%, 5 similar discount groups) are post-hoc and small-n: stability checks, not conclusions; the early/late win rates are unstable.",
            "TAIEX price index: no dividends; all comparisons are price-only.",
            "Daily bars cannot order intraday highs/lows; limit fills at the level, gap fills at the open.",
            "The cash index cannot be traded directly; costs use the stock tax 0.3% + 2 x 0.1425% commission (ETF tax is 0.1%, so conservative for an ETF).",
            "The 2026-11-05 row was provisional at download (volume 0).",
            "Descriptive, not causal; question asked during the 2026 decline.",
        ],
    }
    if args.dry_run:
        print(json.dumps(now, ensure_ascii=False, indent=1))
        return 0
    outdir = Path("reproduced")
    payload["charts"] = charts(payload, df, outdir / "charts")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_readme(payload, outdir / "README.md")
    g = lambda pk, N: B[pk]["holds"][str(N)]
    headline = {"sessions": len(df), "data_end": payload["data_period"]["end"], "n_groups_fib339": B["fib_339"]["n_groups"],
                "n_groups_disc4": B["disc4"]["n_groups"]}
    for N in HOLDS:
        headline[f"fib339_win_{N}_pct"] = g("fib_339", N)["staged_invested"]["win_pct"]
        headline[f"lump_win_{N}_pct"] = g("fib_339", N)["lump_sum"]["win_pct"]
        headline[f"anyday_win_{N}_pct"] = g("fib_339", N)["any_day"]["win_pct"]
        headline[f"c2_win_{N}_pct"] = g("fib_339", N)["c2_same_distance_win_pct"]
        headline[f"disc4_win_{N}_pct"] = g("disc4", N)["staged_invested"]["win_pct"]
    headline["fib339_worst_252_pct"] = g("fib_339", 252)["staged_invested"]["worst_pct"]
    headline["now_close"] = now["close"]
    headline["now_close_vs_H_pct"] = now["close_vs_H_pct"]
    s252 = g("fib_339", 252)
    d4 = lambda N: g("disc4", N)
    p4e, p4l = d4(63)["periods"]["1998-2012"], d4(63)["periods"]["2013-2026"]
    fib_now = now["plans"]["fib_339"]; d4_now = now["plans"]["disc4"]
    hws = [(g("fib_339", N)["staged_invested"]["cluster_bootstrap95_pct"][1] - g("fib_339", N)["staged_invested"]["cluster_bootstrap95_pct"][0]) / 2 for N in HOLDS]
    hw_lo, hw_hi = min(hws), max(hws)
    cb4 = lambda N: d4(N)["staged_invested"]["cluster_bootstrap95_pct"]
    ex4 = lambda N: d4(N)["periods"]["excl_2025_2026"]["staged_win_pct"]
    findings = [
        {"tone": "info", "title": "Fib 3/9/3 after a 20%+ rally: no better than buying any day; a year later a coin flip, same as other days at that distance",
         "detail": "Win rate (net > 0) at 63/126/252 sessions: " + " / ".join(f"{g('fib_339', N)['staged_invested']['win_pct']}% (n={g('fib_339', N)['staged_invested']['n']})" for N in HOLDS)
                   + "; any day " + " / ".join(f"{g('fib_339', N)['any_day']['win_pct']}%" for N in HOLDS)
                   + "; same distance below the high after a rally " + " / ".join(f"{g('fib_339', N)['c2_same_distance_win_pct']}%" for N in HOLDS)
                   + f". 12 groups from 10 rallies ({'/'.join(str(g('fib_339', N)['staged_invested']['clusters']) for N in HOLDS)} overlap clusters); every gap is inside the overlap-cluster bootstrap interval (half-width {hw_lo:.0f}-{hw_hi:.0f} pp).",
         "title_zh": "漲 20% 以上後照 Fib 3/9/3 分批買：沒有比平常任一天買好；一年後像丟銅板，和大漲後同距離的日子一樣",
         "detail_zh": "扣成本後賺錢的比例，三個月／半年／一年：" + "／".join(f"{g('fib_339', N)['staged_invested']['win_pct']}%（n={g('fib_339', N)['staged_invested']['n']}）" for N in HOLDS)
                      + "；平常日 " + "／".join(f"{g('fib_339', N)['any_day']['win_pct']}%" for N in HOLDS)
                      + "；大漲後離高點同距離 " + "／".join(f"{g('fib_339', N)['c2_same_distance_win_pct']}%" for N in HOLDS)
                      + f"。只有 12 組（10 段行情；持有期重疊併群後 {'／'.join(str(g('fib_339', N)['staged_invested']['clusters']) for N in HOLDS)} 群），差距都在重疊群重抽樣區間內（半寬約 {hw_lo:.0f}–{hw_hi:.0f} 個百分點）。"},
        {"tone": "info", "title": "Staging buys cheaper than a lump sum by construction, but barely cushions a crash",
         "detail": "Lump sum on the first-fill day: " + " / ".join(f"{g('fib_339', N)['lump_sum']['win_pct']}%" for N in HOLDS)
                   + f". Staged is never below the lump sum within a group (later levels are bought lower), so staged > lump is structural, not an edge. One year: staged worst {s252['staged_invested']['worst_pct']}% vs lump {s252['lump_sum']['worst_pct']}% (2007-11 group); worst drawdown {s252['dd_invested_worst_pct']}% vs {s252['lump_dd_worst_pct']}%.",
         "title_zh": "分批比一次買足買得便宜（結構必然），但遇到崩盤只少虧一點",
         "detail_zh": "第一份那天一次買足的勝率：" + "／".join(f"{g('fib_339', N)['lump_sum']['win_pct']}%" for N in HOLDS)
                      + f"。分批每組必然不輸一次買足（後面的層買得更低），所以分批勝率高於一次買足是結構結果，不是優勢證據。一年：分批最差 {s252['staged_invested']['worst_pct']}%、一次買足最差 {s252['lump_sum']['worst_pct']}%（都是 2007-11 那一組）；期間最大回檔最差 {s252['dd_invested_worst_pct']}% 對 {s252['lump_dd_worst_pct']}%。"},
        {"tone": "info", "title": "Discount plan 95/90/85/80: the win rates look higher but cannot be told apart from any day, same-distance days or a lump sum; the 3-month high comes from 2013-2026",
         "detail": "Win rate 63/126/252: " + " / ".join(f"{d4(N)['staged_invested']['win_pct']}% (n={d4(N)['staged_invested']['n']}, {d4(N)['staged_invested']['clusters']} overlap clusters, bootstrap {cb4(N)})" for N in HOLDS)
                   + f" vs any day {d4(63)['any_day']['win_pct']}% / {d4(126)['any_day']['win_pct']}% / {d4(252)['any_day']['win_pct']}% (inside the intervals: not distinguishable). Closest controls: same distance after a rally "
                   + " / ".join(f"{d4(N)['c2_same_distance_win_pct']}%" for N in HOLDS) + "; lump sum at the 95% level on the same day " + " / ".join(f"{d4(N)['lump_sum']['win_pct']}%" for N in HOLDS)
                   + ". Staged >= lump is structural, not an edge. Whole-plan mean return " + " / ".join(f"{d4(N)['staged_whole_plan_mean_pct']}%" for N in HOLDS) + " (negative after a year); the 1-year win rate is below the same-distance control. "
                   + f"3 months by period (post-hoc split): 1998-2012 {p4e['staged_win_pct']}% (n={p4e['n']}, any day {p4e['any_day_win_pct']}%), 2013-2026 {p4l['staged_win_pct']}% (n={p4l['n']}, any day {p4l['any_day_win_pct']}%); 6 of the 19 later groups come from the 2025-11..2026-06 rally (all positive). "
                   + "Excluding 2025-2026: " + " / ".join(f"{ex4(N)}%" for N in HOLDS) + " (n=30 each). No edge shown.",
         "title_zh": "對照的打折四份：勝率數字較高，但和平常日、同距離、一次買足都分不出來；三個月的高點來自 2013 年以後",
         "detail_zh": "三個月／半年／一年勝率：" + "／".join(f"{d4(N)['staged_invested']['win_pct']}%（n={d4(N)['staged_invested']['n']}，{d4(N)['staged_invested']['clusters']} 群，重疊群重抽樣 {cb4(N)[0]:.0f}–{cb4(N)[1]:.0f}%）" for N in HOLDS)
                      + f"，平常日 {d4(63)['any_day']['win_pct']}%／{d4(126)['any_day']['win_pct']}%／{d4(252)['any_day']['win_pct']}%（都在區間內，分不出來）。最貼近的對照：大漲後同距離 "
                      + "／".join(f"{d4(N)['c2_same_distance_win_pct']}%" for N in HOLDS) + "；同一組在 95% 那天一次買足 " + "／".join(f"{d4(N)['lump_sum']['win_pct']}%" for N in HOLDS)
                      + "。分批每組必然不輸一次買足，所以分批比一次買足高是結構結果，不是優勢證據；一年勝率低於同距離，含現金平均報酬 "
                      + "／".join(f"{d4(N)['staged_whole_plan_mean_pct']}%" for N in HOLDS) + "（一年是負的）。"
                      + f"三個月分前後期（事後切分，不能當結論）：1998–2012 {p4e['staged_win_pct']}%（n={p4e['n']}，同期平常日 {p4e['any_day_win_pct']}%）、2013–2026 {p4l['staged_win_pct']}%（n={p4l['n']}，同期平常日 {p4l['any_day_win_pct']}%），後者 19 組裡有 6 組來自 2025-11 到 2026-06 那一波多頭（全部賺錢）；"
                      + "排除 2025–2026 後三個月／半年／一年 " + "／".join(f"{ex4(N)}%" for N in HOLDS) + "（各 n=30）。結論：沒有顯示優勢。"},
        {"tone": "info", "title": "Now (2026-11-05, provisional close)",
         "detail": f"H {now['H']:,.0f} ({now['H_date']}), close {now['close']:,.0f} ({now['close_vs_H_pct']}%). Discount levels filled: " + ", ".join(f"{int(lv['level']*100)}% {lv['filled']['date']}" for lv in d4_now if lv["filled"]) + f"; 80% at {d4_now[3]['price']:,.0f}. Fib 0.382 at {fib_now[0]['price']:,.0f} not yet touched.",
         "title_zh": "現在（2026-11-05，收盤暫定）",
         "detail_zh": f"H {now['H']:,.0f}（{now['H_date']}），收盤 {now['close']:,.0f}（{now['close_vs_H_pct']}%）。打折版已成交：" + "、".join(f"{int(lv['level']*100)}% 於 {lv['filled']['date']}" for lv in d4_now if lv["filled"]) + f"；8 折在 {d4_now[3]['price']:,.0f}。Fib 0.382 在 {fib_now[0]['price']:,.0f}，還沒碰到。"},
    ]
    manifest = {
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": generated,
        "timezone_assumptions": "Daily bars on the Taiwan exchange calendar (Asia/Taipei dates).",
        "coverage": f"TWII {payload['data_period']['start']} .. {payload['data_period']['end']}",
        "acquisition": {"new_data_downloaded": False, "note": "reuses the file registered by RS-TX-20261105-001 (downloaded 2026-11-05 23:32 Asia/Taipei, yfinance 1.4.1)"},
        "sources": [{"role": "price_1d_TAIEX", "source_system": "Yahoo Finance ^TWII via yfinance 1.4.1, downloaded 2026-11-05",
                     "locator": args.primary, "sha256": digest, "size_bytes": p.stat().st_size, "immutable": True,
                     "observed": "rows 1997-07-02 .. 2026-11-05; last row volume 0 (provisional)", "privacy": "public_market_data"}],
    }
    (outdir / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(outdir / "results.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
