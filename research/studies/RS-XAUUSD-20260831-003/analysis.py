#!/usr/bin/env python3
"""RS-XAUUSD-20260831-003 — is entering on a pullback better than entering into a run?

Four published studies saw this direction as a by-product and none tested it directly.
This tests it directly, on the same data those observations came from — which is the
honest limit of the exercise and is why the controls matter more than the estimate.

Frozen before the runner was executed (see decision_log.md):
  variable   pre-entry return over the last N completed 30m bars, N in {1,2,4,8,16}
  outcome    the trade's Return %
  statistic  Spearman rank correlation, per strategy per horizon
  prediction NEGATIVE in both strategies — more run-up before entry, worse outcome
  family     10 tests (5 horizons x 2 strategies); permutation shuffles outcomes
  validation 70/30 chronological; the sign must hold out of sample in both strategies
  pass       same-horizon negative sign in both AND family p<0.05 AND both holdouts hold
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
TRADES = {"S1 V3.9": Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv"),
          "S2 V3.2": Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv")}
OUT = Path("local-inputs")HORIZONS = [1, 2, 4, 8, 16]
TRIALS = 20000
SEED = 20260831
EXPORT_DATE = "2026-07-11"
SPLIT = 0.7


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def load_bars() -> tuple[list[datetime], list[float]]:
    times, closes = [], []
    for row in csv.DictReader(BARS.open(encoding="utf-8-sig")):
        times.append(datetime.strptime(row["time"][:16], "%Y-%m-%dT%H:%M"))
        closes.append(float(row["close"]))
    return times, closes


def load_trades(path: Path) -> list[dict]:
    pairs: dict[str, dict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
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


def attach(trades: list[dict], times: list[datetime], closes: list[float]) -> list[dict]:
    """Pre-entry return uses only bars strictly before the fill.

    RS-XAUUSD-20260818-002 established that fills are intrabar — the median sits at 0.443
    of the entry bar's range — so the entry bar's own close is not known at fill time and
    the window ends at entry_index - 1.
    """
    kept = []
    for trade in trades:
        index = bisect_left(times, trade["entry_time"])
        if index >= len(times) or times[index] != trade["entry_time"]:
            continue
        last = index - 1
        if last - max(HORIZONS) < 0:
            continue
        features = {}
        for horizon in HORIZONS:
            base = closes[last - horizon]
            features[horizon] = 100 * (closes[last] / base - 1) if base else None
        if any(v is None for v in features.values()):
            continue
        kept.append({**trade, "pre": features})
    return kept


def terciles(rows: list[dict], horizon: int) -> dict:
    ordered = sorted(rows, key=lambda r: r["pre"][horizon])
    size = len(ordered) // 3
    parts = {"low": ordered[:size], "mid": ordered[size:2 * size], "high": ordered[2 * size:]}
    out = {}
    for name, part in parts.items():
        rets = [r["return_pct"] for r in part]
        out[name] = {
            "n": len(part),
            "mean_return_pct": round(sum(rets) / len(rets), 4),
            "win_rate_pct": round(100 * sum(1 for r in rets if r > 0) / len(part), 2),
            "pre_entry_return_range_pct": [round(part[0]["pre"][horizon], 3),
                                           round(part[-1]["pre"][horizon], 3)],
        }
    return out


def main() -> None:
    for path in [BARS, *TRADES.values()]:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    times, closes = load_bars()
    rng = random.Random(SEED)

    strategies, observed = {}, {}
    for label, path in TRADES.items():
        raw = load_trades(path)
        rows = attach(raw, times, closes)
        split_at = int(len(rows) * SPLIT)
        early, late = rows[:split_at], rows[split_at:]
        per_horizon = {}
        for horizon in HORIZONS:
            xs = [r["pre"][horizon] for r in rows]
            ys = [r["return_pct"] for r in rows]
            rho = spearman(xs, ys)
            rho_late = spearman([r["pre"][horizon] for r in late],
                                [r["return_pct"] for r in late])
            observed[(label, horizon)] = rho
            per_horizon[str(horizon)] = {
                "bars": horizon,
                "hours": horizon / 2,
                "spearman": round(rho, 4),
                "sign_negative": rho < 0,
                "spearman_early": round(spearman([r["pre"][horizon] for r in early],
                                                 [r["return_pct"] for r in early]), 4),
                "spearman_recent": round(rho_late, 4),
                "sign_held_out_of_sample": (rho_late < 0) == (rho < 0),
                "terciles": terciles(rows, horizon),
            }
        strategies[label] = {
            "trades_total": len(raw),
            "trades_with_features": len(rows),
            "coverage_pct": round(100 * len(rows) / len(raw), 2),
            "period": {"from": rows[0]["entry_time"].strftime("%Y-%m-%d"),
                       "to": rows[-1]["entry_time"].strftime("%Y-%m-%d")},
            "split_boundary": late[0]["entry_time"].strftime("%Y-%m-%d"),
            "by_horizon": per_horizon,
            "_rows": rows,
        }

    # Family permutation over all 10 tests: shuffle outcomes within each strategy, recompute
    # every horizon, and keep the largest |rho| anywhere. Ten chances at a correlation is
    # what produced the four incidental observations in the first place.
    observed_max = max(abs(v) for v in observed.values())
    null_max = []
    at_least = 0
    for _ in range(TRIALS):
        best = 0.0
        for data in strategies.values():
            rows = data["_rows"]
            ys = [r["return_pct"] for r in rows]
            rng.shuffle(ys)
            for horizon in HORIZONS:
                xs = [r["pre"][horizon] for r in rows]
                best = max(best, abs(spearman(xs, ys)))
        null_max.append(best)
        if best >= observed_max:
            at_least += 1
    null_max.sort()

    negative_both = [h for h in HORIZONS
                     if observed[("S1 V3.9", h)] < 0 and observed[("S2 V3.2", h)] < 0]
    held_both = [h for h in negative_both
                 if strategies["S1 V3.9"]["by_horizon"][str(h)]["sign_held_out_of_sample"]
                 and strategies["S2 V3.2"]["by_horizon"][str(h)]["sign_held_out_of_sample"]]
    family_p = at_least / TRIALS

    for data in strategies.values():
        del data["_rows"]

    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260831-003",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S1 AweWithBB V3.9 and S2 Hammer V3.2",
        "method": {
            "bar_source": "30-minute FX_IDC:XAUUSD export",
            "bar_count": len(times),
            "variable": "pre-entry return over the last N completed 30m bars",
            "horizons_bars": HORIZONS,
            "feature_window_ends": "entry_index - 1; the entry bar is intrabar and unknown at fill",
            "outcome": "the trade's Return %",
            "statistic": "Spearman rank correlation",
            "prediction_frozen_before_running": "negative in both strategies",
            "family_tests": len(HORIZONS) * len(TRADES),
            "null": "outcomes shuffled within each strategy; largest |rho| across all 10 kept",
            "permutation_trials": TRIALS,
            "random_seed": SEED,
            "chronological_split": SPLIT,
            "retrospective": ("The hypothesis was generated from these same trades by four "
                              "earlier studies. This is not an out-of-sample test and the "
                              "family correction and holdout are the only real controls."),
        },
        "strategies": strategies,
        "verdict": {
            "horizons_negative_in_both": negative_both,
            "horizons_negative_and_held_out_of_sample_in_both": held_both,
            "largest_abs_spearman": round(observed_max, 4),
            "family_null_median_max_abs_spearman": round(null_max[TRIALS // 2], 4),
            "family_null_95th_max_abs_spearman": round(null_max[int(TRIALS * 0.95)], 4),
            "family_p": round(family_p, 4),
            "passes_frozen_criteria": bool(held_both) and family_p < 0.05,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(results["verdict"]))


if __name__ == "__main__":
    main()
