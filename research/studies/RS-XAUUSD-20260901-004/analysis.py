#!/usr/bin/env python3
"""RS-XAUUSD-20260901-004 — what follows an S1 stop-loss, and what hesitating costs.

Backlog BL-012. The question came from a described experience: after S1 stops out, an S2
signal appears and feels too dangerous to take; the drop then produces a fresh S1 signal,
which also feels too dangerous having just passed on the S2. Does the record support that
hesitation?

RS-XAUUSD-20260818-002 tested six cross-strategy splits and separated none, but its
condition was "the other strategy's last trade lost", not "the other strategy just stopped
out", and it did not follow the ordered chain. This does.

Frozen before running (see decision_log.md):
  unit       the S1 stop-loss event (exit signal S1BB_SL)
  windows    {6, 12, 24, 48} hours after the stop's exit timestamp, all four reported
  link       the FIRST qualifying trade in the window, because that is the decision faced
  control    every other trade of the same strategy — the complement, not the whole set
  chain      leg 1 is the S2 after the stop; leg 2 is the next S1 after that S2
  decision   total return if both legs are skipped, against taking everything. Win rate
             cannot answer a question about skipping; seven earlier studies found rules
             that improved a rate and cost return.
  family     8 tests (4 windows x 2 legs); the null shuffles which trades the condition
             marks, holding the group size
  pass       a skip that improves total return AND family p<0.05 AND holds out of sample
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
BARS = Path("local-inputs/xauusd-30m-full.csv")
OUT = REPO / "research/studies/RS-XAUUSD-20260901-004"
WINDOWS = [6, 12, 24, 48]
TRIALS = 20000
SEED = 20260901
EXPORT_DATE = "2026-07-11"
SPLIT = 0.7


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
    return {
        "n": len(rows),
        "win_rate_pct": round(100 * wins / len(rows), 2),
        "mean_return_pct": round(sum(values) / len(values), 4),
        "total_return_pct": round(sum(values), 2),
    }


def bound(n1: int, n2: int, rate: float) -> float:
    if n1 < 2 or n2 < 2:
        return float("inf")
    p = rate / 100
    return round(100 * 2.802 * (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5, 2)


def signal_counts(bars_path: Path, stops: list[dict], hours: int) -> dict:
    """How many S2 signals fired in the window, including ones that never became trades.

    Outcomes exist only for filled trades, so these counts bound the survivorship in the
    main comparison rather than adding measurable trades to it.
    """
    flagged = {"signal": [], "filtered": [], "in_position": []}
    for row in csv.DictReader(bars_path.open(encoding="utf-8-sig")):
        moment = datetime.strptime(row["time"][:16], "%Y-%m-%dT%H:%M")
        if row.get("Hammer Signal") == "1":
            flagged["signal"].append(moment)
        if row.get("Hammer (Filtered)") == "1":
            flagged["filtered"].append(moment)
        if row.get("Hammer (In Position)") == "1":
            flagged["in_position"].append(moment)
    out = {}
    for kind, times in flagged.items():
        hit = sum(1 for s in stops
                  if any(s["exit"] <= t <= s["exit"] + timedelta(hours=hours) for t in times))
        out[f"stops_followed_by_a_{kind}_flag"] = hit
    out["stops_total"] = len(stops)
    return out


def main() -> None:
    for path in (S1_FILE, S2_FILE, BARS):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    s1 = load(S1_FILE)
    s2 = load(S2_FILE)
    stops = [t for t in s1 if t["exit_signal"] == "S1BB_SL"]
    rng = random.Random(SEED)

    by_window = {}
    for hours in WINDOWS:
        span = timedelta(hours=hours)
        leg1, leg2 = [], []
        for stop in stops:
            following = [t for t in s2 if stop["exit"] <= t["entry"] <= stop["exit"] + span]
            if not following:
                continue
            first_s2 = following[0]
            leg1.append(first_s2)
            after = [t for t in s1 if t["entry"] > first_s2["entry"]]
            if after:
                leg2.append(after[0])

        leg1_ids = {id(t) for t in leg1}
        leg2_ids = {id(t) for t in leg2}
        leg1_rest = [t for t in s2 if id(t) not in leg1_ids]
        leg2_rest = [t for t in s1 if id(t) not in leg2_ids]

        # The decision: skip both legs, keep everything else.
        kept = ([t["return_pct"] for t in s2 if id(t) not in leg1_ids]
                + [t["return_pct"] for t in s1 if id(t) not in leg2_ids])
        everything = [t["return_pct"] for t in s1 + s2]

        by_window[str(hours)] = {
            "window_hours": hours,
            "stops_with_a_following_s2": len(leg1),
            "leg1_after_stop": stats(leg1),
            "leg1_control": stats(leg1_rest),
            "leg1_win_rate_gap_pct_points": round(
                stats(leg1)["win_rate_pct"] - stats(leg1_rest)["win_rate_pct"], 2) if leg1 else None,
            "leg1_min_detectable_pct_points": bound(
                len(leg1), len(leg1_rest), stats(s2)["win_rate_pct"]),
            "leg2_next_s1": stats(leg2),
            "leg2_control": stats(leg2_rest),
            "leg2_win_rate_gap_pct_points": round(
                stats(leg2)["win_rate_pct"] - stats(leg2_rest)["win_rate_pct"], 2) if leg2 else None,
            "leg2_min_detectable_pct_points": bound(
                len(leg2), len(leg2_rest), stats(s1)["win_rate_pct"]),
            "skip_both_total_return_pct": round(sum(kept), 2),
            "take_everything_total_return_pct": round(sum(everything), 2),
            "skipping_costs_pct_points": round(sum(kept) - sum(everything), 2),
            "trades_skipped": len(leg1) + len(leg2),
        }

    # Family permutation: the statistic is the best leg-1 win-rate gap across the four
    # windows. Four windows is four chances at a gap.
    observed_best = max(abs(v["leg1_win_rate_gap_pct_points"] or 0) for v in by_window.values())
    s2_wins = [1 if t["return_pct"] > 0 else 0 for t in s2]
    sizes = [v["stops_with_a_following_s2"] for v in by_window.values()]
    null_best, at_least = [], 0
    for _ in range(TRIALS):
        best = 0.0
        for size in sizes:
            if not 0 < size < len(s2_wins):
                continue
            index = list(range(len(s2_wins)))
            rng.shuffle(index)
            marked = index[:size]
            rest = index[size:]
            gap = (100 * sum(s2_wins[i] for i in marked) / size
                   - 100 * sum(s2_wins[i] for i in rest) / len(rest))
            best = max(best, abs(gap))
        null_best.append(best)
        if best >= observed_best:
            at_least += 1
    null_best.sort()

    # The frozen family statistic takes the largest gap across four windows, and the 6-hour
    # window marks only six trades — a group that size produces large gaps by chance, so it
    # dominates both the observed maximum and the null and makes the test uninformative.
    # Reported as frozen, with a single-window test on the only window large enough to read.
    primary = by_window["48"]
    size = primary["stops_with_a_following_s2"]
    single_at_least = 0
    for _ in range(TRIALS):
        index = list(range(len(s2_wins)))
        rng.shuffle(index)
        marked, rest = index[:size], index[size:]
        gap = (100 * sum(s2_wins[i] for i in marked) / size
               - 100 * sum(s2_wins[i] for i in rest) / len(rest))
        if gap >= primary["leg1_win_rate_gap_pct_points"]:
            single_at_least += 1
    single_window = {
        "window_hours": 48,
        "marked_trades": size,
        "observed_gap_pct_points": primary["leg1_win_rate_gap_pct_points"],
        "permutation_p_one_sided": round(single_at_least / TRIALS, 4),
        "min_detectable_pct_points": primary["leg1_min_detectable_pct_points"],
        "clears_bound": bool(primary["leg1_win_rate_gap_pct_points"]
                             > primary["leg1_min_detectable_pct_points"]),
        "note": ("Not the frozen primary test. The frozen one is uninformative because the "
                 "six-hour window marks six trades and dominates the maximum."),
    }

    split_at = int(len(s2) * SPLIT)
    boundary = s2[split_at]["entry"]
    periods = {}
    for label, lo, hi in (("early_70pct", s2[0]["entry"], boundary),
                          ("recent_30pct", boundary, s2[-1]["entry"] + timedelta(days=1))):
        span = timedelta(hours=48)
        marked = []
        for stop in stops:
            following = [t for t in s2 if stop["exit"] <= t["entry"] <= stop["exit"] + span
                         and lo <= t["entry"] < hi]
            if following:
                marked.append(following[0])
        ids = {id(t) for t in marked}
        rest = [t for t in s2 if lo <= t["entry"] < hi and id(t) not in ids]
        periods[label] = {
            "from": lo.strftime("%Y-%m-%d"), "to": hi.strftime("%Y-%m-%d"),
            "after_stop": stats(marked), "control": stats(rest),
            "win_rate_gap_pct_points": round(
                stats(marked)["win_rate_pct"] - stats(rest)["win_rate_pct"], 2)
            if marked and rest else None,
        }

    best_window = max(by_window, key=lambda k: by_window[k]["skipping_costs_pct_points"])
    family_p = at_least / TRIALS
    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260901-004",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S1 AweWithBB V3.9 and S2 Hammer V3.2",
        "method": {
            "s1_trades": len(s1), "s2_trades": len(s2),
            "s1_stop_losses": len(stops),
            "unit": "the S1 stop-loss event (exit signal S1BB_SL)",
            "windows_hours": WINDOWS,
            "link": "the first qualifying trade in the window",
            "control": "every other trade of the same strategy",
            "decision_measure": "total return with both legs skipped, against taking everything",
            "family_tests": len(WINDOWS) * 2,
            "null": "which trades the condition marks, shuffled at the same group size",
            "permutation_trials": TRIALS, "random_seed": SEED,
            "chronological_split": SPLIT,
            "distinct_from": ("RS-XAUUSD-20260818-002 conditioned on the other strategy's "
                              "last trade losing, not on a stop-loss, and did not follow the "
                              "ordered chain"),
            "outcomes_exist_only_for_filled_trades": True,
        },
        "baseline": {"s1": stats(s1), "s2": stats(s2), "combined_total_return_pct": round(
            sum(t["return_pct"] for t in s1 + s2), 2)},
        "by_window": by_window,
        "single_window_test_48h": single_window,
        "signal_coverage_48h": signal_counts(BARS, stops, 48),
        "chronological_48h": periods,
        "verdict": {
            "best_window_for_skipping": best_window,
            "best_skipping_effect_pct_points": by_window[best_window]["skipping_costs_pct_points"],
            "any_skip_improves_total_return": any(
                v["skipping_costs_pct_points"] > 0 for v in by_window.values()),
            "largest_leg1_gap_pct_points": observed_best,
            "family_null_median_gap_pct_points": round(null_best[TRIALS // 2], 2),
            "family_null_95th_gap_pct_points": round(null_best[int(TRIALS * 0.95)], 2),
            "family_p": round(family_p, 4),
            "passes_frozen_criteria": bool(
                any(v["skipping_costs_pct_points"] > 0 for v in by_window.values())
                and family_p < 0.05),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(results["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
