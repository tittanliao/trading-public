#!/usr/bin/env python3
"""RS-TX-20261018-005 — gold held throughout, a 2x Taiwan 50 from the US midterm to the presidential election.

Design fixed in research/studies/RS-TX-20261018-005/decision_log.md before any outcome. All in TWD. Data and
statistics come from the RS-TX-20261018-004 runner; the cycle calendar from RS-MULTI-20261018-001's.

    python3.12 scripts/research/build_tw_lev2_window_gold.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import build_presidential_cycle as cyc  # noqa: E402
import build_tw_barbell_stress as base  # noqa: E402
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-TX-20261018-005"
TAIPEI = timezone(timedelta(hours=8))
MIDTERMS = range(1998, 2023, 4)
INSIDE = {"lev2": 0.5, "gold": 0.5}
OUTSIDE = {"Q2": {"etf": 0.5, "gold": 0.5}, "Q3": {"tbill": 0.5, "gold": 0.5}}
FIXED = {"Q0": {"etf": 1.0}, "Q1": {"lev2": 0.5, "gold": 0.5}, "Q4": {"etf": 0.5, "gold": 0.5}}
LABELS = {"Q0": "100% Taiwan 50", "Q1": "50% 2x + 50% gold always", "Q2": "gold always; 2x in the window, 1x outside",
          "Q3": "gold always; 2x in the window, T-bills outside", "Q4": "50% Taiwan 50 + 50% gold always"}


def windows(idx: pd.DatetimeIndex) -> list[tuple]:
    close = pd.Series(1.0, index=idx)
    out = []
    for y in MIDTERMS:
        a, b = cyc.after(close, cyc.vote_day(y)), cyc.after(close, cyc.vote_day(y + 2))
        if a is not None and b is not None:
            out.append((y, a, b))
    return out


def entry_steps(r: pd.DataFrame, start: pd.Timestamp, rule: str) -> list[tuple]:
    """(day, share of the Taiwan leg in 2x from the close of that day)."""
    idx = r.index
    if rule == "E0":
        return [(start, 1.0)]
    if rule == "E1":
        steps = [(start, 1 / 6)]
        for k in range(1, 6):
            month = (start + pd.offsets.MonthBegin(k)).normalize()
            steps.append((idx[idx.searchsorted(month)], (k + 1) / 6))
        return steps
    level = (1 + r["etf"]).cumprod()
    limit = idx[idx.searchsorted(start + pd.DateOffset(months=6))]
    span = level[(level.index > start) & (level.index <= limit)]
    hit = span[span <= level[start] * 0.9]
    return [(hit.index[0] if len(hit) else limit, 1.0)]


def schedule(r: pd.DataFrame, name: str, rule: str = "E0") -> pd.Series:
    """Target weights holding over each day's return (set at the previous close)."""
    if name in FIXED:
        return pd.Series([FIXED[name]] * len(r), index=r.index)
    targets = pd.Series([OUTSIDE[name]] * len(r), index=r.index, dtype=object)
    for _, a, b in windows(r.index) + [current_window(r.index)]:
        if a is None:
            continue
        steps = entry_steps(r, a, rule) if name == "Q2" else [(a, 1.0)]
        for i, (day, share) in enumerate(steps):
            until = steps[i + 1][0] if i + 1 < len(steps) else b
            mask = (r.index > day) & (r.index <= (until if until is not None else r.index[-1]))
            w = {"lev2": 0.5 * share, "gold": 0.5}
            if share < 1:
                w[next(iter(OUTSIDE[name]))] = 0.5 * (1 - share)
            targets[mask] = [w] * int(mask.sum())
    return targets


def current_window(idx: pd.DatetimeIndex) -> tuple:
    close = pd.Series(1.0, index=idx)
    return (2026, cyc.after(close, cyc.vote_day(2026)), None)


def simulate(r: pd.DataFrame, targets: pd.Series) -> tuple[pd.Series, int]:
    holdings, current, values, prev, count = None, None, [], None, 0
    for day, row in r.iterrows():
        want = targets[day]
        if holdings is None:
            holdings, current = dict(want), want
        elif want != current or day.year != prev.year:
            holdings = {k: holdings.get(k, 0.0) for k in set(holdings) | set(want)}
            full = {k: want.get(k, 0.0) for k in holdings}
            holdings, current, count = base.rebalance(holdings, full), want, count + 1
        holdings = {k: v * (1 + row[k]) for k, v in holdings.items()}
        values.append(sum(holdings.values()))
        prev = day
    return pd.Series(values, index=r.index), count


def per_window(curves: dict, wins: list) -> dict:
    out = {}
    for y, a, b in wins:
        row = {"window": [str(a.date()), str(b.date())]}
        for name, eq in curves.items():
            part = eq[a:b]
            row[name] = {"return_pct": round(float(part.iloc[-1] / part.iloc[0] - 1) * 100, 2),
                         "max_dd_in_window_pct": round(float((part / part.cummax() - 1).min() * 100), 2)}
        out[str(y)] = row
    return out


def compare(windows_: dict, name: str, ref: str) -> dict:
    diff = np.array([w[name]["return_pct"] - w[ref]["return_pct"] for w in windows_.values()])
    bound = 2.8 * float(diff.std(ddof=1)) / math.sqrt(len(diff))
    beat = int((diff > 0).sum())
    return {"n": len(diff), "beat": beat, "mean_diff_pct_points": round(float(diff.mean()), 2),
            "resolvable_pct_points": round(bound, 2),
            "better": bool(beat >= len(diff) - 1 and diff.mean() > bound)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    r = base.returns()
    wins = windows(r.index)
    curves, table = {}, {}
    for name in ("Q0", "Q1", "Q2", "Q3", "Q4"):
        targets = schedule(r, name)
        eq, count = simulate(r, targets)
        curves[name] = eq
        in_2x = float(np.mean([t.get("lev2", 0) > 0 for t in targets]))
        table[name] = base.stats(eq) | {"crash_drawdown_pct": base.crash_dd(eq), "rebalances": count,
                                        "time_with_2x_pct": round(in_2x * 100, 1)}
    entries, entry_curves = {}, {}
    for rule in ("E0", "E1", "E2"):
        eq, _ = simulate(r, schedule(r, "Q2", rule))
        entry_curves[rule] = eq
        entries[rule] = base.stats(eq) | {"crash_drawdown_pct": base.crash_dd(eq)}
    by_window = per_window(curves, wins)
    entry_windows = per_window(entry_curves, wins)
    entry_steps_log = {rule: {str(y): [[str(d.date()), round(s, 3)] for d, s in entry_steps(r, a, rule)]
                              for y, a, _ in wins} for rule in ("E1", "E2")}
    now = current_window(r.index)
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "five fixed portfolios, three entry rules (decision log 2026-10-18)",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md", "currency": "TWD",
                   "data": "as RS-TX-20261018-004 (TAIEX + 3.0%/yr proxy; 2x = 2 x proxy - 2.0%/yr)",
                   "window": "first close after the US midterm vote day to the first close after the presidential vote day",
                   "rebalance": "first trading day of each year and at every change of target; 0.25% of value traded",
                   "labels": LABELS},
        "period": [str(r.index[0].date()), str(r.index[-1].date())],
        "portfolios": table,
        "readings": {"Q2_lower_dd_than_Q1": bool(table["Q2"]["max_drawdown_pct"] > table["Q1"]["max_drawdown_pct"]),
                     "Q2_lower_dd_than_Q0": bool(table["Q2"]["max_drawdown_pct"] > table["Q0"]["max_drawdown_pct"])},
        "windows": by_window,
        "window_comparisons": {f"{a}_vs_{b}": compare(by_window, a, b)
                               for a, b in (("Q2", "Q0"), ("Q2", "Q1"), ("Q2", "Q4"), ("Q3", "Q2"))},
        "entry_rules": entries,
        "entry_windows": entry_windows,
        "entry_vs_E0": {rule: compare(entry_windows, rule, "E0") for rule in ("E1", "E2")},
        "entry_steps": entry_steps_log,
        "current_window": {"midterm_vote_day": str(cyc.vote_day(2026).date()),
                           "presidential_vote_day": str(cyc.vote_day(2028).date()),
                           "first_close_after_midterm": str(now[1].date()) if now[1] is not None else "not yet in data"},
        "charts": [{"id": "window_gold", "file": "window_gold.png", "section": "seasonality",
                    "title": "Gold always, 2x Taiwan 50 from the midterm to the election"}],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    directory = Path("reproduced")
    chart(curves, table, wins, directory / "charts/window_gold.png")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    base.write_manifest(STUDY_ID, payload["generated_at"])
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


def chart(curves: dict, table: dict, wins: list, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, eq in curves.items():
        s = table[name]
        lw = 2.4 if name in ("Q0", "Q2") else 1.2
        a1.plot(eq.index, eq / eq.iloc[0], linewidth=lw,
                label=f"{name} {LABELS[name]}: {s['cagr_pct']}%/yr, max DD {s['max_drawdown_pct']}%")
        a2.plot(eq.index, (eq / eq.cummax() - 1) * 100, linewidth=lw)
    for _, a, b in wins:
        for ax in (a1, a2):
            ax.axvspan(a, b, color="#f2c94c", alpha=0.15)
    a1.set_yscale("log"); a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=0.3)
    a1.set_title("Gold always + 2x Taiwan 50 in the midterm-to-election window (shaded), TWD")
    a2.set_ylabel("drawdown %"); a2.grid(alpha=0.3)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=100); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
