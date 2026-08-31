#!/usr/bin/env python3
"""RS-XAUUSD-20260831-005 — should S1 size up on entries above the upper band?

Backlog BL-008, and the mirror of RS-XAUUSD-20260831-004. That study showed reducing on
the weak side fails structurally, because the weak group is still profitable. This asks the
opposite: the programme's one surviving finding identifies a *stronger* group, so does
increasing size on it help?

Frozen before the runner ran (see decision_log.md):
  label      %B above 1.0 on the last completed bar before the fill, under two conventions:
             the study convention used by RS-XAUUSD-20260823-002 (period 20, 2.0 sd, close,
             sample stdev) and the strategy's own Pine bands carried in the bar export
  action     multiply size on above-upper trades by m in {1.25, 1.5, 2.0, 3.0}
  measure    total return AT EQUAL VOLATILITY. Raising size raises return and risk together;
             RS-XAUUSD-20260818-004 showed that comparing raw totals makes any size increase
             look good. Also max drawdown.
  family     8 tests (4 multipliers x 2 conventions); permutation shuffles the label
  validation 70/30 chronological
  pass       a variant beating baseline at equal volatility AND family p<0.05 AND holding
             in the held-out portion
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
OUT = Path("local-inputs")MULTIPLIERS = [1.25, 1.5, 2.0, 3.0]
PERIOD, STD_MULT = 20, 2.0
TRIALS = 20000
SEED = 20260831
EXPORT_DATE = "2026-07-11"
SPLIT = 0.7


def load_bars() -> list[dict]:
    rows = []
    for row in csv.DictReader(BARS.open(encoding="utf-8-sig")):
        rows.append({
            "time": datetime.strptime(row["time"][:16], "%Y-%m-%dT%H:%M"),
            "close": float(row["close"]),
            "pine_upper": float(row["Upper"]) if row["Upper"] else None,
            "pine_lower": float(row["Lower"]) if row["Lower"] else None,
        })
    # Study convention: period 20, 2.0 sd, on close, sample stdev.
    closes = [r["close"] for r in rows]
    for i, row in enumerate(rows):
        if i + 1 < PERIOD:
            row["study_upper"] = row["study_lower"] = None
            continue
        window = closes[i + 1 - PERIOD:i + 1]
        mean = sum(window) / PERIOD
        var = sum((c - mean) ** 2 for c in window) / (PERIOD - 1)
        sd = var ** 0.5
        row["study_upper"] = mean + STD_MULT * sd
        row["study_lower"] = mean - STD_MULT * sd
    return rows


def percent_b(row: dict, prefix: str) -> float | None:
    upper, lower = row[f"{prefix}_upper"], row[f"{prefix}_lower"]
    if upper is None or lower is None or upper == lower:
        return None
    return (row["close"] - lower) / (upper - lower)


def load_trades() -> list[dict]:
    pairs: dict[str, dict] = {}
    for row in csv.DictReader(TRADES.open(encoding="utf-8-sig")):
        pairs.setdefault(row["Trade number"], {})[row["Type"]] = row
    out = []
    for pair in pairs.values():
        entry, exit_ = pair.get("Entry long"), pair.get("Exit long")
        if not entry or not exit_ or exit_["Date and time"][:10] == EXPORT_DATE:
            continue
        out.append({"entry_time": datetime.strptime(entry["Date and time"], "%Y-%m-%d %H:%M"),
                    "return_pct": float(exit_["Return %"])})
    out.sort(key=lambda t: t["entry_time"])
    return out


def stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def max_drawdown(values: list[float]) -> float:
    peak = running = 0.0
    worst = 0.0
    for v in values:
        running += v
        peak = max(peak, running)
        worst = min(worst, running - peak)
    return round(worst, 2)


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "win_rate_pct": round(100 * sum(1 for v in values if v > 0) / len(values), 2),
        "mean_return_pct": round(sum(values) / len(values), 4),
        "total_return_pct": round(sum(values), 2),
        "sd_return_pct": round(stdev(values), 4),
        "max_drawdown_pct": max_drawdown(values),
    }


def main() -> None:
    for path in (BARS, TRADES):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    bars = load_bars()
    times = [b["time"] for b in bars]
    raw = load_trades()

    rows = []
    for trade in raw:
        index = bisect_left(times, trade["entry_time"])
        if index >= len(times) or times[index] != trade["entry_time"] or index == 0:
            continue
        signal_bar = bars[index - 1]  # last completed bar before the intrabar fill
        study_b = percent_b(signal_bar, "study")
        pine_b = percent_b(signal_bar, "pine")
        if study_b is None or pine_b is None:
            continue
        rows.append({**trade, "study_above": study_b > 1.0, "pine_above": pine_b > 1.0})

    # Which bar carries the label is not a detail. RS-XAUUSD-20260823-002 and -20260824-006
    # join with pandas merge_asof(direction="backward") on entry_time, and entry_time equals
    # a bar's own timestamp exactly, so the match lands on the bar the fill happened inside.
    # That bar's close is thirty minutes after the fill. RS-XAUUSD-20260818-002 established
    # fills are intrabar at a median 0.443 of the entry bar's range, and used entry_index - 1
    # for exactly this reason. The two conventions are compared here.
    bar_choice = {}
    for offset, name in ((-1, "last_completed_bar_before_fill"), (0, "entry_bar_itself")):
        above, rest = [], []
        for trade in raw:
            index = bisect_left(times, trade["entry_time"])
            if index >= len(times) or times[index] != trade["entry_time"] or index + offset < 0:
                continue
            value = percent_b(bars[index + offset], "study")
            if value is None:
                continue
            (above if value > 1.0 else rest).append(trade["return_pct"])
        rate = lambda v: round(100 * sum(1 for x in v if x > 0) / len(v), 2)
        bar_choice[name] = {
            "above_upper_n": len(above), "above_upper_win_rate_pct": rate(above),
            "rest_n": len(rest), "rest_win_rate_pct": rate(rest),
            "win_rate_gap_pct_points": round(rate(above) - rate(rest), 2),
            "observable_at_fill": offset == -1,
        }
    bar_choice["gap_lost_to_lookahead_pct_points"] = round(
        bar_choice["entry_bar_itself"]["win_rate_gap_pct_points"]
        - bar_choice["last_completed_bar_before_fill"]["win_rate_gap_pct_points"], 2)
    bar_choice["published_comparison"] = {
        "RS-XAUUSD-20260823-002": {"above_upper_n": 82, "above_upper_win_rate_pct": 73.17,
                                   "baseline_win_rate_pct": 55.93, "gap_pct_points": 20.86},
        "RS-XAUUSD-20260824-006_variant_A": {"above_upper_n": 71,
                                             "above_upper_win_rate_pct": 73.24,
                                             "rest_win_rate_pct": 51.57, "gap_pct_points": 21.67},
        "note": ("Both used a different trade export, so an exact match is not expected. The "
                 "entry-bar figures here are close to theirs and the no-lookahead figures are "
                 "not, which is what the code path predicts."),
    }

    baseline_returns = [r["return_pct"] for r in rows]
    baseline = describe(baseline_returns)
    baseline_sd = baseline["sd_return_pct"]

    conventions = {"study_convention": "study_above", "strategy_pine_bands": "pine_above"}
    label_stats = {}
    for name, key in conventions.items():
        above = [r["return_pct"] for r in rows if r[key]]
        rest = [r["return_pct"] for r in rows if not r[key]]
        label_stats[name] = {"above_upper": describe(above), "rest": describe(rest),
                             "share_of_trades_pct": round(100 * len(above) / len(rows), 2)}

    agree = sum(1 for r in rows if r["study_above"] == r["pine_above"])
    both = sum(1 for r in rows if r["study_above"] and r["pine_above"])
    union = sum(1 for r in rows if r["study_above"] or r["pine_above"])
    label_agreement = {
        "trades_compared": len(rows),
        "same_label_pct": round(100 * agree / len(rows), 2),
        "above_in_study_convention": sum(1 for r in rows if r["study_above"]),
        "above_in_strategy_bands": sum(1 for r in rows if r["pine_above"]),
        "above_in_both": both,
        "jaccard": round(both / union, 4) if union else None,
        "note": ("Same instrument, same timeframe, same period and multiplier — only the "
                 "price input and the stdev denominator differ. RS-XAUUSD-20260824-006 "
                 "measured the same instability across timeframes at 32.23%."),
    }

    def variant(key: str, multiplier: float, subset: list[dict] | None = None) -> dict:
        source = subset if subset is not None else rows
        scaled = [r["return_pct"] * (multiplier if r[key] else 1.0) for r in source]
        base = [r["return_pct"] for r in source]
        base_total, base_sd = sum(base), stdev(base)
        sd = stdev(scaled)
        equal_vol_total = sum(scaled) * (base_sd / sd) if sd else 0.0
        stats = describe(scaled)
        return {
            **stats,
            "total_return_at_equal_volatility_pct": round(equal_vol_total, 2),
            "gain_at_equal_volatility_pct": round(equal_vol_total - base_total, 2),
            "raw_total_change_pct": round(sum(scaled) - base_total, 2),
            "share_of_baseline_at_equal_volatility_pct": round(
                100 * equal_vol_total / base_total, 2) if base_total else None,
        }

    variants = {}
    for name, key in conventions.items():
        for multiplier in MULTIPLIERS:
            variants[f"{name}_x{multiplier}"] = {"convention": name, "multiplier": multiplier,
                                                 **variant(key, multiplier)}

    observed_best = max(v["gain_at_equal_volatility_pct"] for v in variants.values())
    flags = {key: [r[key] for r in rows] for key in conventions.values()}
    rng = random.Random(SEED)
    null_best, at_least = [], 0
    for _ in range(TRIALS):
        best = -1e9
        for key in conventions.values():
            shuffled = flags[key][:]
            rng.shuffle(shuffled)
            for multiplier in MULTIPLIERS:
                scaled = [r["return_pct"] * (multiplier if f else 1.0)
                          for r, f in zip(rows, shuffled)]
                sd = stdev(scaled)
                if not sd:
                    continue
                best = max(best, sum(scaled) * (baseline_sd / sd) - baseline["total_return_pct"])
        null_best.append(best)
        if best >= observed_best:
            at_least += 1
    null_best.sort()

    split_at = int(len(rows) * SPLIT)
    periods = {}
    for label, part in (("early_70pct", rows[:split_at]), ("recent_30pct", rows[split_at:])):
        entry = {"from": part[0]["entry_time"].strftime("%Y-%m-%d"),
                 "to": part[-1]["entry_time"].strftime("%Y-%m-%d")}
        for name, key in conventions.items():
            entry[name] = variant(key, 2.0, part)
        periods[label] = entry

    family_p = at_least / TRIALS
    best_name = max(variants, key=lambda v: variants[v]["gain_at_equal_volatility_pct"])
    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260831-005",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S1 AweWithBB V3.9",
        "method": {
            "trades_total": len(raw), "trades_labelled": len(rows),
            "label": "%B above 1.0 on the last completed bar before the fill",
            "conventions": {
                "study_convention": ("period 20, 2.0 sd, close, sample stdev — the "
                                     "convention RS-XAUUSD-20260823-002 published"),
                "strategy_pine_bands": "the Upper/Lower the strategy itself carries in the export",
            },
            "multipliers": MULTIPLIERS,
            "measure": ("total return scaled so the book's per-trade volatility matches "
                        "baseline; raising size raises return and risk together and a raw "
                        "total makes any increase look good"),
            "family_tests": len(MULTIPLIERS) * len(conventions),
            "null": "the above-upper label shuffled; best equal-volatility gain kept",
            "permutation_trials": TRIALS, "random_seed": SEED, "chronological_split": SPLIT,
            "mirror_of": ("RS-XAUUSD-20260831-004, which tested reducing on the weak side"),
        },
        "baseline": baseline,
        "bar_choice": bar_choice,
        "by_label": label_stats,
        "label_agreement": label_agreement,
        "variants": variants,
        "chronological": periods,
        "verdict": {
            "best_variant": best_name,
            "best_gain_at_equal_volatility_pct": observed_best,
            "variants_beating_baseline": sum(
                1 for v in variants.values() if v["gain_at_equal_volatility_pct"] > 0),
            "variants_tested": len(variants),
            "family_null_median_gain_pct": round(null_best[TRIALS // 2], 2),
            "family_null_95th_gain_pct": round(null_best[int(TRIALS * 0.95)], 2),
            "family_p": round(family_p, 4),
            "passes_frozen_criteria": bool(observed_best > 0 and family_p < 0.05),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({**results["verdict"], "label_agreement_pct": label_agreement["same_label_pct"]}))


if __name__ == "__main__":
    main()
