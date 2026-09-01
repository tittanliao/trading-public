#!/usr/bin/env python3
"""RS-XAUUSD-20260901-005 — does the exit type matter, or just that an exit happened?

RS-XAUUSD-20260901-004 found S2 trades following an S1 stop-loss win 60.56% against 42.57%.
Its stated reading is that a stop-loss marks a drop, which is what a hammer entry is built
for. There is a competing explanation it did not rule out: S1 and S2 may simply cluster in
the same volatile stretches, in which case any S1 exit would mark the same thing and the
stop-loss is incidental.

This separates them with a placebo. A take-profit is also an S1 exit, also followed by S2
signals, and is not a drop. If the reading is right, only the stop-loss group is elevated.

Frozen before running (see decision_log.md):
  primary    S2 within the window of an S1 STOP-LOSS versus S2 within the window of an S1
             TAKE-PROFIT. Both groups are conditioned on "an S1 just exited", so the
             comparison isolates the exit type and removes the clustering explanation.
  window     48 hours. NOT chosen blind — it is the window RS-XAUUSD-20260901-004 found
             readable. 24 hours reported as sensitivity, and the dependence is disclosed.
  null       the SL/TP label shuffled across S1 exit events, holding group sizes. This is
             the null that matters: it keeps "an S1 exited" fixed and destroys only which
             kind of exit it was.
  symmetry   the mirror — S1 following an S2 stop-loss versus an S2 take-profit. If a stop
             marks a drop that a long-only dip entry wants, the mirror should also hold.
  pass       the stop-loss group beats the take-profit group AND permutation p<0.05 AND the
             direction holds out of sample
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S1_FILE = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv")
S2_FILE = Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv")
OUT = REPO / "research/studies/RS-XAUUSD-20260901-005"
WINDOWS = [24, 48]
PRIMARY_WINDOW = 48
TRIALS = 20000
SEED = 20260901
EXPORT_DATE = "2026-07-11"
SPLIT = 0.7
STOPS = {"S1": "S1BB_SL", "S2": "S2_SL"}
TARGETS = {"S1": ("S1BB_TP1", "S1BB_TP2"), "S2": ("S2_LX_TP1", "S2_LX_TP2")}


def load(path: Path) -> list[dict]:
    pairs: dict[str, dict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        pairs.setdefault(row["Trade number"], {})[row["Type"]] = row
    out = []
    for pair in pairs.values():
        entry, exit_ = pair.get("Entry long"), pair.get("Exit long")
        if not entry or not exit_ or exit_["Date and time"][:10] == EXPORT_DATE:
            continue
        out.append({
            "entry": datetime.strptime(entry["Date and time"], "%Y-%m-%d %H:%M"),
            "exit": datetime.strptime(exit_["Date and time"], "%Y-%m-%d %H:%M"),
            "exit_signal": exit_["Signal"],
            "return_pct": float(exit_["Return %"]),
        })
    return sorted(out, key=lambda t: t["entry"])


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    values = [r["return_pct"] for r in rows]
    wins = sum(1 for v in values if v > 0)
    return {"n": len(rows), "win_rate_pct": round(100 * wins / len(rows), 2),
            "mean_return_pct": round(sum(values) / len(values), 4),
            "total_return_pct": round(sum(values), 2)}


def bound(n1: int, n2: int, rate: float) -> float:
    if n1 < 2 or n2 < 2:
        return float("inf")
    p = rate / 100
    return round(100 * 2.802 * (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5, 2)


def first_following(events: list[dict], pool: list[dict], hours: int) -> list[dict]:
    span = timedelta(hours=hours)
    picked, seen = [], set()
    for event in events:
        for trade in pool:
            if event["exit"] <= trade["entry"] <= event["exit"] + span:
                if id(trade) not in seen:
                    seen.add(id(trade))
                    picked.append(trade)
                break
    return picked


def compare(source: list[dict], pool: list[dict], side: str, hours: int) -> dict:
    stops = [t for t in source if t["exit_signal"] == STOPS[side]]
    targets = [t for t in source if t["exit_signal"] in TARGETS[side]]
    after_stop = first_following(stops, pool, hours)
    after_target = first_following(targets, pool, hours)
    marked = {id(t) for t in after_stop} | {id(t) for t in after_target}
    unmarked = [t for t in pool if id(t) not in marked]
    a, b = stats(after_stop), stats(after_target)
    gap = round(a["win_rate_pct"] - b["win_rate_pct"], 2) if after_stop and after_target else None
    return {
        "window_hours": hours,
        "stop_events": len(stops), "target_events": len(targets),
        "after_stop": a, "after_target": b, "neither": stats(unmarked),
        "win_rate_gap_pct_points": gap,
        "mean_return_gap_pct": round(a["mean_return_pct"] - b["mean_return_pct"], 4)
        if after_stop and after_target else None,
        "min_detectable_pct_points": bound(
            len(after_stop), len(after_target), stats(pool)["win_rate_pct"]),
        "clears_bound": bool(gap is not None
                             and gap > bound(len(after_stop), len(after_target),
                                             stats(pool)["win_rate_pct"])),
        "_after_stop": after_stop, "_after_target": after_target,
    }


def permute(source: list[dict], pool: list[dict], side: str, hours: int,
            observed: float, rng: random.Random) -> dict:
    """Shuffle which S1 exits are stops and which are targets, holding the counts.

    This keeps 'an exit happened, and an S2 followed it' fixed and destroys only the exit
    type — which is exactly the claim under test.
    """
    events = [t for t in source
              if t["exit_signal"] == STOPS[side] or t["exit_signal"] in TARGETS[side]]
    n_stop = sum(1 for t in events if t["exit_signal"] == STOPS[side])
    hits, null = 0, []
    for _ in range(TRIALS):
        order = events[:]
        rng.shuffle(order)
        fake_stops, fake_targets = order[:n_stop], order[n_stop:]
        a = first_following(sorted(fake_stops, key=lambda t: t["exit"]), pool, hours)
        b = first_following(sorted(fake_targets, key=lambda t: t["exit"]), pool, hours)
        if not a or not b:
            continue
        gap = stats(a)["win_rate_pct"] - stats(b)["win_rate_pct"]
        null.append(gap)
        if gap >= observed:
            hits += 1
    null.sort()
    return {"trials": len(null), "p_one_sided": round(hits / len(null), 4) if null else None,
            "null_median_gap_pct_points": round(null[len(null) // 2], 2) if null else None,
            "null_95th_gap_pct_points": round(null[int(len(null) * 0.95)], 2) if null else None}


def main() -> None:
    for path in (S1_FILE, S2_FILE):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    s1, s2 = load(S1_FILE), load(S2_FILE)
    rng = random.Random(SEED)

    primary = {str(h): compare(s1, s2, "S1", h) for h in WINDOWS}
    mirror = {str(h): compare(s2, s1, "S2", h) for h in WINDOWS}

    key = str(PRIMARY_WINDOW)
    permutation = permute(s1, s2, "S1", PRIMARY_WINDOW,
                          primary[key]["win_rate_gap_pct_points"], rng)
    mirror_permutation = permute(s2, s1, "S2", PRIMARY_WINDOW,
                                 mirror[key]["win_rate_gap_pct_points"], rng)

    split_at = int(len(s2) * SPLIT)
    boundary = s2[split_at]["entry"]
    periods = {}
    for label, lo, hi in (("early_70pct", s2[0]["entry"], boundary),
                          ("recent_30pct", boundary, s2[-1]["entry"] + timedelta(days=1))):
        window = [t for t in s2 if lo <= t["entry"] < hi]
        part = compare(s1, window, "S1", PRIMARY_WINDOW)
        periods[label] = {"from": lo.strftime("%Y-%m-%d"), "to": hi.strftime("%Y-%m-%d"),
                          "after_stop": part["after_stop"], "after_target": part["after_target"],
                          "win_rate_gap_pct_points": part["win_rate_gap_pct_points"]}

    for group in (primary, mirror):
        for value in group.values():
            value.pop("_after_stop", None)
            value.pop("_after_target", None)

    results = {
        "schema_version": 1, "study_id": "RS-XAUUSD-20260901-005",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S1 AweWithBB V3.9 and S2 Hammer V3.2",
        "method": {
            "tests": "RS-XAUUSD-20260901-004's stated mechanism",
            "competing_explanation": ("S1 and S2 cluster in the same volatile stretches, so "
                                      "any S1 exit would mark the same thing"),
            "placebo": "a take-profit is also an S1 exit and is not a drop",
            "windows_hours": WINDOWS, "primary_window_hours": PRIMARY_WINDOW,
            "window_not_chosen_blind": ("48 hours is the window RS-XAUUSD-20260901-004 found "
                                        "readable; 24 is reported as sensitivity"),
            "null": "the stop/target label shuffled across exit events, holding counts",
            "permutation_trials": TRIALS, "random_seed": SEED,
            "chronological_split": SPLIT,
        },
        "baseline": {"s1": stats(s1), "s2": stats(s2)},
        "primary_s2_after_s1_exit": primary,
        "mirror_s1_after_s2_exit": mirror,
        "permutation_48h": permutation,
        "mirror_permutation_48h": mirror_permutation,
        "chronological_48h": periods,
        "verdict": {
            "stop_beats_target": bool(primary[key]["win_rate_gap_pct_points"] > 0),
            "gap_pct_points": primary[key]["win_rate_gap_pct_points"],
            "min_detectable_pct_points": primary[key]["min_detectable_pct_points"],
            "clears_bound": primary[key]["clears_bound"],
            "permutation_p": permutation["p_one_sided"],
            "direction_holds_both_periods": bool(
                all((v["win_rate_gap_pct_points"] or 0) > 0 for v in periods.values())),
            "mirror_holds": bool((mirror[key]["win_rate_gap_pct_points"] or 0) > 0),
            "passes_frozen_criteria": bool(
                primary[key]["win_rate_gap_pct_points"] > 0
                and (permutation["p_one_sided"] or 1) < 0.05
                and all((v["win_rate_gap_pct_points"] or 0) > 0 for v in periods.values())),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(results["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
