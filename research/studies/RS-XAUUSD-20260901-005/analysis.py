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

Revision note (review fix, RS-XAUUSD-20260901-005 revision 4): the frozen plan said "each S2 trade
may be claimed by one group only, first come". The first version of this runner de-duplicated
only inside each group, so 16 S2 trades (48 h) sat in both. This version claims trades with
one shared `seen` set, events taken in exit-time order. "First come" has two readings and the
plan does not say which; reading A is fixed here (an exit looks only at the first S2 inside
its window and contributes nothing if that trade is already claimed). Reading B (the exit
moves on to the next unclaimed S2) is computed as a labelled sensitivity. The pre-fix
behaviour is kept as `legacy_independent_groups` so old and new numbers can be compared.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S1_FILE = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv")
S2_FILE = Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv")
OUT = Path("reproduced")
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


def wilson(wins: int, n: int) -> list[float]:
    if n == 0:
        return [None, None]
    z, p = 1.96, wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def with_ci(rows: list[dict]) -> dict:
    out = stats(rows)
    if rows:
        wins = sum(1 for r in rows if r["return_pct"] > 0)
        out["win_rate_ci95_pct"] = wilson(wins, len(rows))
    return out


def gap_ci(a: list[dict], b: list[dict]) -> list[float] | None:
    """Wald 95% interval of the win-rate difference, in percentage points."""
    if not a or not b:
        return None
    p1 = sum(1 for r in a if r["return_pct"] > 0) / len(a)
    p2 = sum(1 for r in b if r["return_pct"] > 0) / len(b)
    se = (p1 * (1 - p1) / len(a) + p2 * (1 - p2) / len(b)) ** 0.5
    gap = p1 - p2
    return [round(100 * (gap - 1.96 * se), 1), round(100 * (gap + 1.96 * se), 1)]


def first_in_window(entries: list[datetime], event: dict, span: timedelta) -> int | None:
    """Index of the first pool trade (pool sorted by entry) inside the event's window."""
    i = bisect_left(entries, event["exit"])
    if i < len(entries) and entries[i] <= event["exit"] + span:
        return i
    return None


def claim(events: list[tuple[dict, bool]], pool: list[dict], hours: int,
          reading: str = "A") -> tuple[list[dict], list[dict]]:
    """Assign each S2 (or S1) trade to at most one group, first come by exit time.

    `events` is (exit event, is_stop) in source order; they are walked in exit-time order
    (stable). Reading A: an exit looks only at the first trade inside its window and claims
    it unless it is already claimed. Reading B: an exit takes the first unclaimed trade
    inside its window. One `seen` set is shared by the stop and the target group.
    """
    span = timedelta(hours=hours)
    entries = [t["entry"] for t in pool]
    seen: set[int] = set()
    stop_group, target_group = [], []
    for event, is_stop in sorted(events, key=lambda e: e[0]["exit"]):
        i = first_in_window(entries, event, span)
        if i is None:
            continue
        if reading == "B":
            while i < len(pool) and pool[i]["entry"] <= event["exit"] + span and i in seen:
                i += 1
            if i >= len(pool) or pool[i]["entry"] > event["exit"] + span:
                continue
        if i in seen:
            continue
        seen.add(i)
        (stop_group if is_stop else target_group).append(pool[i])
    return stop_group, target_group


def legacy_first_following(events: list[dict], pool: list[dict], hours: int) -> list[dict]:
    """The pre-fix behaviour: de-duplicate inside one group only. Kept for comparison."""
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


def labelled_events(source: list[dict], side: str) -> list[tuple[dict, bool]]:
    return [(t, t["exit_signal"] == STOPS[side]) for t in source
            if t["exit_signal"] == STOPS[side] or t["exit_signal"] in TARGETS[side]]


def near_far(source: list[dict], pool: list[dict], hours: int) -> dict:
    """True-definition split: a pool trade is near if ANY source exit lies in the
    preceding `hours` hours (exit <= entry <= exit + hours). Post-hoc, not pre-registered."""
    span = timedelta(hours=hours)
    exits = sorted(t["exit"] for t in source)
    near, far = [], []
    for trade in pool:
        j = bisect_left(exits, trade["entry"] - span)
        (near if j < len(exits) and exits[j] <= trade["entry"] else far).append(trade)
    gap = round(stats(near)["win_rate_pct"] - stats(far)["win_rate_pct"], 2) if near and far else None
    rate = stats(pool)["win_rate_pct"]
    return {"window_hours": hours, "status": "post-hoc, not pre-registered, no permutation test",
            "near_any_exit": with_ci(near), "no_exit_in_window": with_ci(far),
            "win_rate_gap_pct_points": gap, "gap_ci95_pct_points": gap_ci(near, far),
            "min_detectable_pct_points": bound(len(near), len(far), rate)}


def compare(source: list[dict], pool: list[dict], side: str, hours: int,
            reading: str = "A") -> dict:
    events = labelled_events(source, side)
    after_stop, after_target = claim(events, pool, hours, reading)
    claimed = {id(t) for t in after_stop} | {id(t) for t in after_target}
    unclaimed = [t for t in pool if id(t) not in claimed]
    rate = stats(pool)["win_rate_pct"]
    a, b = with_ci(after_stop), with_ci(after_target)
    gap = round(a["win_rate_pct"] - b["win_rate_pct"], 2) if after_stop and after_target else None
    # the pre-fix, independently de-duplicated groups, for the old/new comparison
    stops_only = [e for e, is_stop in events if is_stop]
    targets_only = [e for e, is_stop in events if not is_stop]
    old_a = legacy_first_following(stops_only, pool, hours)
    old_b = legacy_first_following(targets_only, pool, hours)
    both = {id(t) for t in old_a} & {id(t) for t in old_b}
    return {
        "window_hours": hours, "first_come_reading": reading,
        "stop_events": len(stops_only), "target_events": len(targets_only),
        "after_stop": a, "after_target": b,
        "not_claimed_by_either_group": stats(unclaimed),
        "win_rate_gap_pct_points": gap,
        "gap_ci95_pct_points": gap_ci(after_stop, after_target),
        "mean_return_gap_pct": round(a["mean_return_pct"] - b["mean_return_pct"], 4)
        if after_stop and after_target else None,
        "min_detectable_pct_points": bound(len(after_stop), len(after_target), rate),
        "clears_bound": bool(gap is not None
                             and gap > bound(len(after_stop), len(after_target), rate)),
        "legacy_independent_groups": {
            "note": ("pre-fix grouping: each group de-duplicated only inside itself, so "
                     "trades could sit in both; kept for comparison only"),
            "after_stop_n": len(old_a), "after_target_n": len(old_b),
            "trades_in_both_groups": len(both),
            "after_stop_win_rate_pct": stats(old_a).get("win_rate_pct"),
            "after_target_win_rate_pct": stats(old_b).get("win_rate_pct"),
            "win_rate_gap_pct_points": round(stats(old_a)["win_rate_pct"]
                                             - stats(old_b)["win_rate_pct"], 2)
            if old_a and old_b else None},
        "_after_stop": after_stop, "_after_target": after_target,
    }


def permute(source: list[dict], pool: list[dict], side: str, hours: int,
            observed: float, rng: random.Random, reading: str = "A") -> dict:
    """Shuffle which exits are stops and which are targets, holding the counts.

    This keeps 'an exit happened, and a trade followed it' fixed and destroys only the exit
    type — which is exactly the claim under test. The same shared-claim grouping as the
    observed statistic is applied to every shuffle.
    """
    events = [e for e, _ in labelled_events(source, side)]
    flags = [is_stop for _, is_stop in labelled_events(source, side)]
    hits, null = 0, []
    for _ in range(TRIALS):
        shuffled = flags[:]
        rng.shuffle(shuffled)
        a, b = claim(list(zip(events, shuffled)), pool, hours, reading)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output folder (default: the study folder)")
    out_dir = parser.parse_args().out
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

    # sensitivity: the other reading of "first come" (an exit moves on to the next unclaimed trade)
    reading_b = {
        "note": ("the frozen plan does not say which first-come reading was meant; reading A "
                 "is the primary one above, reading B is reported for completeness"),
        "primary_s2_after_s1_exit": {str(h): compare(s1, s2, "S1", h, "B") for h in WINDOWS},
        "mirror_s1_after_s2_exit": {str(h): compare(s2, s1, "S2", h, "B") for h in WINDOWS},
    }
    rng_b = random.Random(SEED)
    reading_b["permutation_48h"] = permute(
        s1, s2, "S1", PRIMARY_WINDOW,
        reading_b["primary_s2_after_s1_exit"][key]["win_rate_gap_pct_points"], rng_b, "B")
    reading_b["mirror_permutation_48h"] = permute(
        s2, s1, "S2", PRIMARY_WINDOW,
        reading_b["mirror_s1_after_s2_exit"][key]["win_rate_gap_pct_points"], rng_b, "B")

    split_at = int(len(s2) * SPLIT)
    boundary = s2[split_at]["entry"]
    periods = {}
    for label, lo, hi in (("early_70pct", s2[0]["entry"], boundary),
                          ("recent_30pct", boundary, s2[-1]["entry"] + timedelta(days=1))):
        window = [t for t in s2 if lo <= t["entry"] < hi]
        part = compare(s1, window, "S1", PRIMARY_WINDOW)
        part_b = compare(s1, window, "S1", PRIMARY_WINDOW, "B")
        periods[label] = {"from": lo.strftime("%Y-%m-%d"), "to": hi.strftime("%Y-%m-%d"),
                          "after_stop": part["after_stop"], "after_target": part["after_target"],
                          "win_rate_gap_pct_points": part["win_rate_gap_pct_points"],
                          "gap_ci95_pct_points": part["gap_ci95_pct_points"],
                          "min_detectable_pct_points": part["min_detectable_pct_points"],
                          "reading_b": {"after_stop": part_b["after_stop"],
                                        "after_target": part_b["after_target"],
                                        "win_rate_gap_pct_points":
                                            part_b["win_rate_gap_pct_points"],
                                        "min_detectable_pct_points":
                                            part_b["min_detectable_pct_points"]}}

    # post-hoc lead (not pre-registered): near any exit in the preceding window vs none
    near_far_lead = {
        "s2_near_any_s1_exit_48h": near_far(s1, s2, PRIMARY_WINDOW),
        "mirror_s1_near_any_s2_exit_48h": near_far(s2, s1, PRIMARY_WINDOW),
    }

    for group in (primary, mirror, reading_b["primary_s2_after_s1_exit"],
                  reading_b["mirror_s1_after_s2_exit"]):
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
            "first_come_reading": ("A: an exit looks only at the first trade inside its window "
                                   "and claims it unless already claimed; groups share one "
                                   "claim set, exits walked in exit-time order"),
            "permutation_trials": TRIALS, "random_seed": SEED,
            "chronological_split": SPLIT,
        },
        "baseline": {"s1": stats(s1), "s2": stats(s2)},
        "primary_s2_after_s1_exit": primary,
        "mirror_s1_after_s2_exit": mirror,
        "permutation_48h": permutation,
        "mirror_permutation_48h": mirror_permutation,
        "chronological_48h": periods,
        "sensitivity_reading_b": reading_b,
        "posthoc_near_vs_no_exit": near_far_lead,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(results["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
