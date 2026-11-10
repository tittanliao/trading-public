#!/usr/bin/env python3
"""When to add the extra lot during a gold decline (RS-XAUUSD-20261027-003).

Design fixed in research/studies/RS-XAUUSD-20261027-003/decision_log.md before any outcome was computed:
weekly closes since 1970; does a week that meets an add-on condition (accelerating fall, rebound start,
deep, back above the 20-week MA) have a better forward return than an ordinary week in a decline?

    python3.12 scripts/research/build_gold_addon_timing.py [--dry-run]

Revision 4 (2026-11-11) adds one post-hoc supplementary table, `post_hoc_supplementary_2008_avg_cost_path`: the
2008 plan's running average cost week by week (approved 2026-11-10 for that table only). It is not a new test and
changes no other field; it lays out the path to the week-26 average costs already in `plans`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261027-003"
WEEKLY = Path("local-inputs/XAUUSD_1wk.csv")
TAIPEI = timezone(timedelta(hours=8))

# Frozen in the decision log.
CONTEXT_DD = -0.10
HORIZONS = (13, 26, 52)
EPISODE_GAP = 13
PLAN_WEEKS, PLAN_VALUE_AT = 26, (26, 52)
# The five weeks gold's close first fell 25% below its all-time-high close (RS-XAUUSD-20261026-001, part A).
EVENT_STARTS = ("1973-11-12", "1975-09-15", "1980-03-03", "2008-10-20", "2013-04-15")
# Post-hoc supplementary table (revision 4): the 2008 episode laid out week by week. Not a design parameter.
SUPPLEMENT_START = "2008-10-20"


def r(value, digits=2):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), digits)


def rel(path: Path) -> str:
    """Repo-relative locator; the published copy runs from local-inputs/ outside the repo."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def frame() -> pd.DataFrame:
    raw = pd.read_csv(WEEKLY)
    f = pd.DataFrame({"close": raw["close"].astype(float).to_numpy()},
                     index=pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None).dt.normalize())
    c = f["close"]
    f["dd"] = c / c.cummax() - 1
    f["r4"] = c / c.shift(4) - 1
    f["r4_prev"] = f["r4"].shift(4)
    f["ma20"] = c.rolling(20).mean()
    f["C1_accelerating_fall"] = (f["r4"] <= -0.05) & (f["r4"] < f["r4_prev"])
    f["C2_rebound_start"] = (f["r4"].shift(1) <= -0.05) & (c / c.shift(1) - 1 >= 0.02)
    f["C3_deep"] = f["dd"] <= -0.20
    f["C4_back_above_ma20"] = (c > f["ma20"]) & (c.shift(1) <= f["ma20"].shift(1))
    f["context"] = f["dd"] <= CONTEXT_DD
    for h in HORIZONS:
        f[f"fwd{h}"] = (c.shift(-h) / c - 1) * 100
    return f


def episodes(mask: pd.Series) -> int:
    idx = np.nonzero(mask.to_numpy())[0]
    return int(1 + (np.diff(idx) > EPISODE_GAP).sum()) if len(idx) else 0


def compare(f: pd.DataFrame, mask: pd.Series, base_mask: pd.Series) -> dict:
    out = {"weeks": int(mask.sum()), "episodes": episodes(mask)}
    for h in HORIZONS:
        col = f"fwd{h}"
        v, b = f.loc[mask, col].dropna(), f.loc[base_mask, col].dropna()
        if len(v) < 2:
            out[f"{h}w"] = {"n": int(len(v))}
            continue
        n_eff, b_eff = max(len(v) / h, 1.0), len(b) / h
        bound = 2.8 * b.std(ddof=1) * math.sqrt(1 / n_eff + 1 / b_eff)
        eff = v.mean() - b.mean()
        out[f"{h}w"] = {"n": int(len(v)), "n_eff": r(n_eff, 1), "mean_pct": r(v.mean()), "median_pct": r(v.median()),
                        "win_rate_pct": r((v > 0).mean() * 100), "baseline_n": int(len(b)),
                        "baseline_mean_pct": r(b.mean()), "baseline_median_pct": r(b.median()),
                        "effect_pct": r(eff), "resolution_bound_pct": r(bound), "separates": bool(abs(eff) > bound)}
    return out


def buys(f: pd.DataFrame, start: pd.Timestamp, extra: str | None) -> tuple[int, list[tuple[int, int, float, float]]]:
    """The plan's weekly buys: 1 lot at each weekly close for PLAN_WEEKS weeks from `start`, plus 1 extra lot in
    weeks where `extra` is true. Returns the start row and, per week, (row, lots bought, cumulative lots,
    cumulative cost). `plan` and the post-hoc 2008 table both read this one loop."""
    c = f["close"]
    i0 = c.index.get_loc(start)
    end = min(i0 + PLAN_WEEKS, len(c))
    lots = cost = 0.0
    path = []
    for i in range(i0, end):
        n = 1 + (1 if extra and bool(f[extra].iloc[i]) else 0)
        lots += n
        cost += n * c.iloc[i]
        path.append((i, n, lots, cost))
    return i0, path


def plan(f: pd.DataFrame, start: pd.Timestamp, extra: str | None) -> dict:
    c = f["close"]
    i0, path = buys(f, start, extra)
    _, _, lots, cost = path[-1]
    out = {"lots": int(lots), "avg_cost": r(cost / lots)}
    for h in PLAN_VALUE_AT:
        j = i0 + h
        out[f"return_at_{h}w_pct"] = r((c.iloc[j] * lots / cost - 1) * 100) if j < len(c) else None
        out[f"pnl_at_{h}w_lot_equiv"] = r((c.iloc[j] * lots - cost) / c.iloc[i0]) if j < len(c) else None
    return out


def avg_cost_path(f: pd.DataFrame, start: str, published: dict) -> dict:
    """Post-hoc supplementary table (revision 4, approved 2026-11-10): the running average cost of the
    1-lot-a-week plan and of the plan plus 1 extra lot in C1 weeks, week by week, for the episode starting at
    `start`. Same buys as `plan`; the last row must equal the published week-26 average costs in `plans`."""
    c = f["close"]
    ts = pd.Timestamp(start)
    _, base = buys(f, ts, None)
    _, plus = buys(f, ts, "C1_accelerating_fall")
    rows = []
    for week, ((i, _, lots_b, cost_b), (_, n, lots_p, cost_p)) in enumerate(zip(base, plus), 1):
        avg_b, avg_p = r(cost_b / lots_b), r(cost_p / lots_p)
        rows.append({"week": week, "week_of": c.index[i].date().isoformat(), "close": r(c.iloc[i]),
                     "c1_extra_lot": n == 2, "avg_cost_base_1_per_week": avg_b, "avg_cost_plus_C1": avg_p,
                     "base_minus_plus_C1": r(avg_b - avg_p)})
    last = rows[-1]
    if (last["avg_cost_base_1_per_week"], last["avg_cost_plus_C1"]) != (
            published["base_1_per_week"]["avg_cost"], published["plus_C1"]["avg_cost"]):
        raise SystemExit("post-hoc 2008 table: last row does not match the plan's published average costs")
    avg_b, avg_p, gap = last["avg_cost_base_1_per_week"], last["avg_cost_plus_C1"], last["base_minus_plus_C1"]
    return {
        "label": (f"Post-hoc supplementary table (revision 4, added 2026-11-11). Not a new test and not a registered "
                  f"metric: it lays out, week by week, how the {ts.year} plan reached the week-26 average costs already "
                  f"in `plans` ({avg_b:,.2f} vs {avg_p:,.2f}, difference {gap:,.2f}; start {ts.date().isoformat()}). "
                  f"One episode, one path, no error bound. No conclusion, rule or parameter changed."),
        "label_zh": (f"事後補充（修訂 4，2026-11-11 加入）：不是新檢定，也不是登記指標；只是把 `plans` 裡已發布的 "
                     f"{ts.year} 那次（起點 {ts.date().isoformat()}）第 26 週均價（{avg_b:,.2f}／{avg_p:,.2f}，"
                     f"差 {gap:,.2f}）的過程逐週攤開。只有這一次、一條路徑，沒有誤差界線；結論、規則、參數都沒有改。"),
        "start": ts.date().isoformat(),
        "weeks": len(rows),
        "columns": {
            "week": "1 = the start week",
            "week_of": "Monday stamp of the weekly bar",
            "close": "that week's close (USD per ounce), the buy price",
            "c1_extra_lot": "C1 (accelerating fall) true that week, so the plus-C1 plan bought 2 lots instead of 1",
            "avg_cost_base_1_per_week": "running average cost, 1 lot every week",
            "avg_cost_plus_C1": "running average cost, 1 lot every week plus 1 extra lot in C1 weeks",
            "base_minus_plus_C1": "avg_cost_base_1_per_week − avg_cost_plus_C1, of the two rounded values (USD per ounce, a price difference)",
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    f = frame()
    conds = ["C1_accelerating_fall", "C2_rebound_start", "C3_deep", "C4_back_above_ma20"]
    everyw = pd.Series(True, index=f.index)
    tests = {c: {"vs_context_weeks": compare(f, f[c] & f["context"], f["context"]),
                 "vs_all_weeks": compare(f, f[c] & f["context"], everyw)} for c in conds}
    context = {"vs_all_weeks": compare(f, f["context"], everyw)}

    starts = list(EVENT_STARTS)
    # The 2026 decline: first week after the latest all-time high that closed 10% or more below it.
    ath_at = f["close"].idxmax()
    starts.append(f.index[(f.index > ath_at) & f["context"]][0].date().isoformat())
    plans = []
    for s in starts:
        ts = pd.Timestamp(s)
        if ts not in f.index:
            ts = f.index[f.index.get_indexer([ts], method="nearest")[0]]
        plans.append({"start": ts.date().isoformat(), "base_1_per_week": plan(f, ts, None),
                      "plus_C1": plan(f, ts, "C1_accelerating_fall"), "plus_C2": plan(f, ts, "C2_rebound_start")})
    supplement = avg_cost_path(f, SUPPLEMENT_START, next(p for p in plans if p["start"] == SUPPLEMENT_START))

    last = f.iloc[-1]
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "none — weekly gold accumulation, timing of an extra lot",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md (fixed 2026-10-27 before any outcome)",
                   "context": f"weekly close <= {CONTEXT_DD:.0%} from the all-time-high close",
                   "conditions": {"C1": "4w return <= -5% and below the previous 4w return",
                                  "C2": "previous week's 4w return <= -5% and this week +2% or more",
                                  "C3": "close <= -20% from the all-time-high close",
                                  "C4": "close above the 20-week MA after being at or below it"},
                   "horizons_weeks": HORIZONS,
                   "resolution_bound": "2.8 × SD(baseline) × √(1/n_eff + 1/N_eff), n_eff = weeks / horizon (min 1)",
                   "episodes": f"runs separated by more than {EPISODE_GAP} weeks",
                   "plans": f"{PLAN_WEEKS} weekly buys of 1 lot from the start, plus 1 extra lot in weeks where C1 (or C2) is true; no costs"},
        "coverage": {"first_week": f.index[0].date().isoformat(), "last_week": f.index[-1].date().isoformat(),
                     "weeks": int(len(f)), "context_weeks": int(f["context"].sum())},
        "context": context,
        "conditions": tests,
        "plans": plans,
        "now": {"week": f.index[-1].date().isoformat(), "close": r(last["close"]), "dd_pct": r(last["dd"] * 100),
                "r4_pct": r(last["r4"] * 100), "ma20": r(last["ma20"]),
                **{c: bool(last[c]) for c in conds}},
        "post_hoc_supplementary_2008_avg_cost_path": supplement,
        "charts": [],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
        return 0
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    directory = Path("reproduced")
    (directory / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
        "timezone_assumptions": "Weekly bars carry a Monday 00:00Z stamp; the close is that week's last close.",
        "sources": [
            {"role": "price_1w", "source_system": "repository tracked CSV", "locator": rel(WEEKLY),
             **sha(WEEKLY), "immutable": False, "privacy": "private_raw_source"},
        ]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
