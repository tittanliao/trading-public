#!/usr/bin/env python3
"""RS-TX-20261108-001 — TAIEX after the National Financial Stabilization Fund (國安基金) enters.

Design frozen before any forward return was computed.
Events come from local-inputs/event_dates.csv (public sources and check dates per row).
Entry price = TWII close on the official entry day E; holds 5/20/60/120/250 sessions; win = return > 0.
Controls: any day, and sessions with a similar pre-entry 20-session move (same distance).
Research only; nothing here feeds a live rule.

    python3 analysis.py   # from the repository root, inputs under local-inputs/
"""

from __future__ import annotations

import argparse
import csv
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

STUDY_ID = "RS-TX-20261108-001"
TAIPEI = timezone(timedelta(hours=8))
HOLDS = [5, 20, 60, 120, 250]
PRE = 20
BANDS = [0.02, 0.03, 0.05, 0.08]
MIN_CONTROL = 50
EXCLUDE = 20
DEFAULT_CSV = "local-inputs/TWII_1d_to_2026-11-06.csv"


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def binom_tail(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def min_wins_sig(n: int, p: float, alpha: float = 0.05) -> int | None:
    for k in range(n + 1):
        if binom_tail(k, n, p) < alpha:
            return k
    return None


def pct(x: float | None, nd: int = 2) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100 * x, nd)


def path_stats(c: np.ndarray, i: int, h: int) -> dict | None:
    if i + h >= len(c):
        return None
    w = c[i : i + h + 1]
    ret = w[-1] / w[0] - 1
    mae = w.min() / w[0] - 1
    mdd = float((w / np.maximum.accumulate(w) - 1).min())
    return {"ret": float(ret), "mae": float(mae), "mdd": mdd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--events", default="local-inputs/event_dates.csv")
    ap.add_argument("--no-write-study", action="store_true")
    a = ap.parse_args()

    src = ROOT / a.csv
    df = pd.read_csv(src, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    c = df["close"].to_numpy(float)
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    pos = {d: i for i, d in enumerate(dates)}
    n_bars = len(c)

    # pre-entry 20-session move D(i) = C[i-1]/C[i-21]-1
    D = np.full(n_bars, np.nan)
    D[PRE + 1 :] = c[PRE : n_bars - 1] / c[: n_bars - PRE - 1] - 1

    events = list(csv.DictReader(open(ROOT / a.events, encoding="utf-8")))
    hist = [e for e in events if e["in_data"] == "yes"]
    for e in hist:
        for k in ("entry_official", "entry_viewer"):
            if e[k] not in pos:
                raise SystemExit(f"entry date {e[k]} not a TWII session")
    entry_idx = [pos[e["entry_official"]] for e in hist]
    excl = np.zeros(n_bars, bool)
    for i in entry_idx:
        excl[max(0, i - EXCLUDE) : i + EXCLUDE + 1] = True

    fwd = {h: np.full(n_bars, np.nan) for h in HOLDS}
    for h in HOLDS:
        fwd[h][: n_bars - h] = c[h:] / c[: n_bars - h] - 1

    def event_rows(key: str) -> list[dict]:
        rows = []
        for e in hist:
            i = pos[e[key]]
            r = {"n": int(e["n"]), "event_zh": e["event_zh"], "announce_date": e["announce_date"],
                 "entry_date": e[key], "entry_close": round(float(c[i]), 2),
                 "pre20_pct": pct(D[i]), "holds": {}}
            for h in HOLDS:
                s = path_stats(c, i, h)
                r["holds"][str(h)] = None if s is None else {
                    "exit_date": dates[i + h], "exit_close": round(float(c[i + h]), 2),
                    "ret_pct": pct(s["ret"], 3), "win": s["ret"] > 0,
                    "lowest_close_vs_entry_pct": pct(s["mae"], 3), "max_drawdown_pct": pct(s["mdd"], 3)}
            rows.append(r)
        return rows

    def summarize(rows: list[dict]) -> dict:
        out = {}
        for h in HOLDS:
            vals = [r["holds"][str(h)] for r in rows if r["holds"][str(h)] is not None]
            n = len(vals)
            k = sum(v["win"] for v in vals)
            rets = [v["ret_pct"] for v in vals]
            out[str(h)] = {"n": n, "wins": k, "win_pct": round(100 * k / n, 2) if n else None,
                           "wilson95": wilson(k, n), "mean_ret_pct": round(float(np.mean(rets)), 3) if n else None,
                           "median_ret_pct": round(float(np.median(rets)), 3) if n else None,
                           "best_ret_pct": max(rets) if n else None, "worst_ret_pct": min(rets) if n else None,
                           "worst_lowest_close_pct": min(v["lowest_close_vs_entry_pct"] for v in vals) if n else None,
                           "worst_max_drawdown_pct": min(v["max_drawdown_pct"] for v in vals) if n else None}
        return out

    primary_rows = event_rows("entry_official")
    viewer_rows = event_rows("entry_viewer")
    primary = summarize(primary_rows)
    viewer = summarize(viewer_rows)

    # controls
    anyday = {}
    for h in HOLDS:
        v = fwd[h][~np.isnan(fwd[h])]
        anyday[str(h)] = {"n": int(v.size), "win_pct": round(100 * float((v > 0).mean()), 2),
                          "mean_ret_pct": round(100 * float(v.mean()), 3),
                          "median_ret_pct": round(100 * float(np.median(v)), 3)}

    same = {str(h): {"per_event": []} for h in HOLDS}
    for e, i in zip(hist, entry_idx):
        d0 = D[i]
        for h in HOLDS:
            ok = ~np.isnan(fwd[h]) & ~np.isnan(D) & ~excl
            chosen = None
            for b in BANDS:
                m = ok & (np.abs(D - d0) <= b)
                if m.sum() >= MIN_CONTROL:
                    chosen = (b, m)
                    break
            if chosen is None:
                chosen = (BANDS[-1], ok & (np.abs(D - d0) <= BANDS[-1]))
            b, m = chosen
            v = fwd[h][m]
            same[str(h)]["per_event"].append({"n_event": int(e["n"]), "band_pp": round(100 * b, 1),
                                              "n_days": int(m.sum()),
                                              "win_pct": round(100 * float((v > 0).mean()), 2) if v.size else None,
                                              "mean_ret_pct": round(100 * float(v.mean()), 3) if v.size else None})
    for h in HOLDS:
        pe = [x for x, r in zip(same[str(h)]["per_event"], primary_rows) if r["holds"][str(h)] is not None]
        same[str(h)]["win_pct"] = round(float(np.mean([x["win_pct"] for x in pe])), 2)
        same[str(h)]["mean_ret_pct"] = round(float(np.mean([x["mean_ret_pct"] for x in pe])), 3)
        same[str(h)]["n_events_used"] = len(pe)

    tests = {}
    for h in HOLDS:
        s = primary[str(h)]
        n, k = s["n"], s["wins"]
        p_same = same[str(h)]["win_pct"] / 100
        p_any = anyday[str(h)]["win_pct"] / 100
        tests[str(h)] = {
            "p_tail_vs_same_distance": round(binom_tail(k, n, p_same), 4),
            "p_tail_vs_any_day": round(binom_tail(k, n, p_any), 4),
            "wins_needed_to_separate_from_same_distance": min_wins_sig(n, p_same),
            "wins_needed_to_separate_from_any_day": min_wins_sig(n, p_any),
            "verdict_vs_same_distance": "分得出" if binom_tail(k, n, p_same) < 0.05 else "分不出",
            "verdict_vs_any_day": "分得出" if binom_tail(k, n, p_any) < 0.05 else "分不出",
        }

    # now: event 10
    last = n_bars - 1
    hi252 = float(df["high"].iloc[max(0, last - 251) : last + 1].max())
    hi_date = dates[int(df["high"].iloc[max(0, last - 251) : last + 1].idxmax())]
    d_now = c[last] / c[last - PRE] - 1  # = D of the next session (2026-11-09)
    hist_D = sorted((r["pre20_pct"], r["n"]) for r in primary_rows)
    now = {
        "event": next(e for e in events if e["n"] == "10"),
        "data_end": dates[last], "last_close": round(float(c[last]), 2),
        "pre20_for_entry_2026_11_09_pct": pct(d_now),
        "pre20_history_sorted": hist_D,
        "high_252_intraday": round(hi252, 2), "high_252_date": hi_date,
        "last_close_vs_high_252_pct": pct(c[last] / hi252 - 1),
        "note": "Entry close on 2026-11-09 not yet known; every forward number for event 10 is pending.",
    }

    raw = src.read_bytes()
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "none (descriptive event study)",
        "method": {"entry": "TWII close on the first day of the official operating period (E)",
                   "viewer_variant": "first session strictly after the announcement date",
                   "holds_sessions": HOLDS, "win": "C[E+h]/C[E]-1 > 0 (price index, no dividends, no costs)",
                   "pre20": "C[i-1]/C[i-21]-1",
                   "same_distance": f"|D-D_event| <= band, bands {BANDS} first with >= {MIN_CONTROL} sessions; "
                                    f"exclude +/-{EXCLUDE} sessions of any entry; equal-weight mean over events",
                   "interval": "Wilson 95%", "test": "exact one-sided binomial tail P(X>=k)"},
        "coverage": {"first": dates[0], "last": dates[-1], "sessions": n_bars,
                     "source": a.csv, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
        "events": events,
        "per_event_primary": primary_rows,
        "per_event_viewer_variant": viewer_rows,
        "summary_primary": primary, "summary_viewer_variant": viewer,
        "baseline_any_day": anyday, "control_same_distance": same, "tests": tests,
        "clusters": {"note": "Events 1 (2000-03-16) and 2 (2000-10-03) are 135 sessions apart: their 250-session "
                             "windows overlap, so h=250 has 9 events but 8 independent windows.",
                     "sessions_between_1_and_2": entry_idx[1] - entry_idx[0]},
        "now": now,
        "limitations": [
            "n = 9 completed events; one event moves a win rate by 11 pp.",
            "The fund enters after sharp falls; the outcome mixes the fund's buying with ordinary rebounds after selloffs. Not causal.",
            "Price index without dividends; no costs.",
            "Event 4 date conflicts across sources (2008-09-18 vs 09-19); viewer variant covers it.",
            "Yahoo ^TWII closes; no second-vendor cross-check in this study (RS-TX-20261105-001 cross-checked the same file against MXF).",
        ],
    }
    outdir = Path("reproduced")
    (outdir / "charts").mkdir(parents=True, exist_ok=True)
    payload["charts"] = make_charts(outdir, primary, anyday, same, primary_rows, c, pos)
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps({"primary": primary, "viewer": viewer, "anyday": anyday,
                      "same": {h: {k: v for k, v in same[h].items() if k != "per_event"} for h in same},
                      "tests": tests, "now": {k: v for k, v in now.items() if k != "event"}},
                     ensure_ascii=False, indent=1))
    return 0


def make_charts(outdir: Path, primary, anyday, same, rows, c, pos) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#1f2933", "#7b8794"
    col_ev, col_same, col_any = "#c2410c", "#64748b", "#cbd5e1"
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = np.arange(len(HOLDS))
    w = 0.26
    ev = [primary[str(h)]["win_pct"] for h in HOLDS]
    sm = [same[str(h)]["win_pct"] for h in HOLDS]
    an = [anyday[str(h)]["win_pct"] for h in HOLDS]
    lo = [ev[j] - primary[str(h)]["wilson95"][0] for j, h in enumerate(HOLDS)]
    hi = [primary[str(h)]["wilson95"][1] - ev[j] for j, h in enumerate(HOLDS)]
    ax.bar(x - w, ev, w, color=col_ev, label="after fund entry (n=9)", yerr=[lo, hi], ecolor=ink, capsize=3)
    ax.bar(x, sm, w, color=col_same, label="same 20-session fall")
    ax.bar(x + w, an, w, color=col_any, label="any day")
    for j, h in enumerate(HOLDS):
        s = primary[str(h)]
        ax.text(x[j] - w, 2, f"{s['wins']}/{s['n']}", ha="center", va="bottom", fontsize=7, color="white")
    ax.axhline(50, color=muted, lw=0.8, ls="--")
    ax.set_xticks(x, [f"{h} sessions" for h in HOLDS])
    ax.set_ylim(0, 118)
    ax.set_ylabel("share of cases higher (%)")
    ax.set_title("TAIEX after the National Stabilization Fund enters (2000-2025): win rate by hold\n"
                 "(entry-day close; whiskers = Wilson 95%)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "charts/win_rate_by_hold.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for r in rows:
        i = pos[r["entry_date"]]
        seg = c[i : min(len(c), i + 251)]
        y = 100 * (seg / seg[0] - 1)
        ax.plot(np.arange(len(y)), y, lw=1.1, label=f"{r['entry_date'][:7]}")
        ax.text(len(y) - 1 + 2, y[-1], r["entry_date"][:4] + ("b" if r["n"] == 2 else ""), fontsize=6.5, va="center")
    ax.axhline(0, color=ink, lw=0.8)
    for h in HOLDS:
        ax.axvline(h, color=muted, lw=0.5, ls=":")
    ax.set_xlim(0, 275)
    ax.set_xlabel("sessions after entry-day close")
    ax.set_ylabel("TAIEX change vs entry close (%)")
    ax.set_title("Each of the 9 fund entries, first 250 sessions", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "charts/event_paths.png")
    plt.close(fig)
    return [
        {"file": "win_rate_by_hold.png", "section": "performance",
         "caption": "Win rate by hold: after fund entry vs same 20-session fall vs any day (Wilson 95% whiskers)"},
        {"file": "event_paths.png", "section": "performance",
         "caption": "Each of the 9 fund entries: TAIEX change vs entry close, first 250 sessions"},
    ]


if __name__ == "__main__":
    raise SystemExit(main())
