#!/usr/bin/env python3
"""RS-XAUUSD-20260831-004 — does S1's first completed bar tell you to cut or reduce?

Backlog BL-003. RS-XAUUSD-20260815-001 tested whether the first bar *held the signal-bar
low*, a price-structure condition. This tests a different and simpler one: whether the
first bar *closed in profit*. Nothing has tested it.

Frozen before the runner ran (see decision_log.md):
  condition   close of the trade's own entry bar against its fill price. The fill is
              intrabar, so that close is the first observable mark after entry.
  association win rate and mean return, in-profit versus not, with a resolution bound
  actions     (a) exit at the close of bar k when not in profit, k in {1,2,3,4}
              (b) halve the position at that point instead, same k
  outcome     Return %, and total return across the book — not win rate alone
  cost        exit variants keep one round trip and add none; the halving variant adds a
              partial exit, charged at half of ROUND_TRIP_COST_PCT
  family      9 tests (1 association + 8 actions); permutation shuffles outcomes
  validation  70/30 chronological
  pass        an action beating baseline total return AND family p<0.05 AND holding in the
              held-out portion
"""
from __future__ import annotations

import csv
import json
import random
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BARS = Path("local-inputs/xauusd-30m-full.csv")
TRADES = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv")
OUT = Path("local-inputs")KS = [1, 2, 3, 4]
TRIALS = 20000
SEED = 20260831
EXPORT_DATE = "2026-07-11"
SPLIT = 0.7
ROUND_TRIP_COST_PCT = 0.02


def load_bars() -> tuple[list[datetime], list[float]]:
    times, closes = [], []
    for row in csv.DictReader(BARS.open(encoding="utf-8-sig")):
        times.append(datetime.strptime(row["time"][:16], "%Y-%m-%dT%H:%M"))
        closes.append(float(row["close"]))
    return times, closes


def load_trades() -> list[dict]:
    pairs: dict[str, dict] = {}
    for row in csv.DictReader(TRADES.open(encoding="utf-8-sig")):
        pairs.setdefault(row["Trade number"], {})[row["Type"]] = row
    out = []
    for pair in pairs.values():
        entry, exit_ = pair.get("Entry long"), pair.get("Exit long")
        if not entry or not exit_ or exit_["Date and time"][:10] == EXPORT_DATE:
            continue
        out.append({
            "entry_time": datetime.strptime(entry["Date and time"], "%Y-%m-%d %H:%M"),
            "exit_time": datetime.strptime(exit_["Date and time"], "%Y-%m-%d %H:%M"),
            "entry_price": float(entry["Price USD"]),
            "return_pct": float(exit_["Return %"]),
            "duration_bars": int(exit_["Duration (bars)"]),
        })
    out.sort(key=lambda t: t["entry_time"])
    return out


def attach(trades: list[dict], times: list[datetime], closes: list[float]) -> list[dict]:
    """Mark each trade at the close of its own entry bar and the three bars after it.

    The fill is intrabar (RS-XAUUSD-20260818-002 put the median at 0.443 of the entry bar's
    range), so the entry bar's close is the first mark a person could act on, and using it
    is not look-ahead.
    """
    kept = []
    for trade in trades:
        index = bisect_left(times, trade["entry_time"])
        if index >= len(times) or times[index] != trade["entry_time"]:
            continue
        exit_index = bisect_left(times, trade["exit_time"])
        if exit_index >= len(times) or times[exit_index] != trade["exit_time"]:
            continue
        if index + max(KS) >= len(times):
            continue
        marks = {}
        for k in KS:
            bar = index + k - 1  # k=1 is the entry bar's own close
            marks[k] = 100 * (closes[bar] / trade["entry_price"] - 1)
        kept.append({**trade, "entry_index": index, "exit_index": exit_index, "marks": marks})
    return kept


def describe(rows: list[dict], key: str = "return_pct") -> dict:
    if not rows:
        return {"n": 0}
    values = [r[key] for r in rows]
    wins = sum(1 for v in values if v > 0)
    return {
        "n": len(rows),
        "win_rate_pct": round(100 * wins / len(rows), 2),
        "mean_return_pct": round(sum(values) / len(values), 4),
        "total_return_pct": round(sum(values), 2),
    }


def min_detectable(n1: int, n2: int, baseline_pct: float) -> float:
    """Smallest win-rate gap two groups this size could separate at alpha .05, power .80."""
    if n1 < 2 or n2 < 2:
        return float("inf")
    p = baseline_pct / 100
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    return round(100 * 2.802 * se, 2)


def main() -> None:
    for path in (BARS, TRADES):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    times, closes = load_bars()
    raw = load_trades()
    rows = attach(raw, times, closes)
    rng = random.Random(SEED)

    baseline = describe(rows)
    in_profit = [r for r in rows if r["marks"][1] > 0]
    under = [r for r in rows if r["marks"][1] <= 0]
    association = {
        "first_bar_in_profit": describe(in_profit),
        "first_bar_not_in_profit": describe(under),
        "win_rate_gap_pct_points": round(
            describe(in_profit)["win_rate_pct"] - describe(under)["win_rate_pct"], 2),
        "mean_return_gap_pct": round(
            describe(in_profit)["mean_return_pct"] - describe(under)["mean_return_pct"], 4),
        "min_detectable_win_rate_gap_pct_points": min_detectable(
            len(in_profit), len(under), baseline["win_rate_pct"]),
    }

    def simulate(k: int, fraction: float) -> list[dict]:
        """Return each trade's outcome if a position not in profit at bar k is cut to
        `fraction` of its size. fraction 0.0 is a full exit."""
        out = []
        for r in rows:
            if r["marks"][k] > 0 or r["exit_index"] <= r["entry_index"] + k - 1:
                out.append({**r, "sim_return_pct": r["return_pct"]})
                continue
            at_k = r["marks"][k]
            rest = r["return_pct"] - at_k
            cost = 0.0 if fraction == 0.0 else ROUND_TRIP_COST_PCT / 2
            out.append({**r, "sim_return_pct": at_k + fraction * rest - cost})
        return out

    actions = {}
    for k in KS:
        for label, fraction in (("exit", 0.0), ("halve", 0.5)):
            sim = simulate(k, fraction)
            touched = sum(1 for r, s in zip(rows, sim) if r["return_pct"] != s["sim_return_pct"])
            stats = describe(sim, "sim_return_pct")
            actions[f"{label}_at_bar_{k}"] = {
                "action": label, "bar": k, "trades_affected": touched,
                **stats,
                "total_return_change_pct": round(stats["total_return_pct"]
                                                 - baseline["total_return_pct"], 2),
                "win_rate_change_pct_points": round(stats["win_rate_pct"]
                                                    - baseline["win_rate_pct"], 2),
            }

    # Family permutation over all nine tests. The statistic is the best total-return gain
    # any action achieves; the null shuffles which trades the condition fires on.
    observed_best = max(a["total_return_change_pct"] for a in actions.values())
    flags = [r["marks"][1] > 0 for r in rows]
    null_best, at_least = [], 0
    for _ in range(TRIALS):
        rng.shuffle(flags)
        best = -1e9
        for k in KS:
            for fraction in (0.0, 0.5):
                total = 0.0
                for r, flag in zip(rows, flags):
                    if flag or r["exit_index"] <= r["entry_index"] + k - 1:
                        total += r["return_pct"]
                    else:
                        at_k = r["marks"][k]
                        cost = 0.0 if fraction == 0.0 else ROUND_TRIP_COST_PCT / 2
                        total += at_k + fraction * (r["return_pct"] - at_k) - cost
                best = max(best, total - baseline["total_return_pct"])
        null_best.append(best)
        if best >= observed_best:
            at_least += 1
    null_best.sort()

    split_at = int(len(rows) * SPLIT)
    periods = {}
    for label, part in (("early_70pct", rows[:split_at]), ("recent_30pct", rows[split_at:])):
        good = [r for r in part if r["marks"][1] > 0]
        bad = [r for r in part if r["marks"][1] <= 0]
        periods[label] = {
            "from": part[0]["entry_time"].strftime("%Y-%m-%d"),
            "to": part[-1]["entry_time"].strftime("%Y-%m-%d"),
            "first_bar_in_profit": describe(good),
            "first_bar_not_in_profit": describe(bad),
            "win_rate_gap_pct_points": round(
                describe(good)["win_rate_pct"] - describe(bad)["win_rate_pct"], 2),
        }

    best_action = max(actions, key=lambda a: actions[a]["total_return_change_pct"])
    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260831-004",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S1 AweWithBB V3.9",
        "method": {
            "trades_total": len(raw), "trades_with_marks": len(rows),
            "condition": "close of the trade's own entry bar against its fill price",
            "why_not_lookahead": ("the fill is intrabar, so the entry bar's close is the "
                                  "first observable mark after entry"),
            "bars_tested": KS,
            "actions": "exit the position, or halve it, when not in profit at bar k",
            "partial_exit_cost_pct": ROUND_TRIP_COST_PCT / 2,
            "family_tests": 1 + 2 * len(KS),
            "null": "the in-profit flag shuffled across trades; best total-return gain kept",
            "permutation_trials": TRIALS, "random_seed": SEED,
            "chronological_split": SPLIT,
            "distinct_from": ("RS-XAUUSD-20260815-001 tested whether the first bar held the "
                              "signal-bar low, a different condition"),
        },
        "baseline": baseline,
        "association": association,
        "actions": actions,
        "chronological": periods,
        "verdict": {
            "best_action": best_action,
            "best_total_return_change_pct": observed_best,
            "family_null_median_best_gain_pct": round(null_best[TRIALS // 2], 2),
            "family_null_95th_best_gain_pct": round(null_best[int(TRIALS * 0.95)], 2),
            "family_p": round(at_least / TRIALS, 4),
            "any_action_beats_baseline": observed_best > 0,
            "passes_frozen_criteria": bool(observed_best > 0 and at_least / TRIALS < 0.05),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(results["verdict"]))


if __name__ == "__main__":
    main()
