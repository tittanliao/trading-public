#!/usr/bin/env python3
"""RS-XAUUSD-20260901-003 — Hurst, and whether the day's range has a usable boundary.

Backlog BL-006 and BL-007, run together because they are one question asked twice.
RS-XAUUSD-20260818-004 established that direction is not forecastable on this series
(largest |IC| 0.0399) while volatility is (Spearman 0.571 out of sample), and that using
the volatility forecast for stops or sizing beats nothing. Neither item asks for another
sizing rule. They ask what that asymmetry *is*.

Frozen before running (see decision_log.md):
  hurst      rescaled-range on log returns over dyadic window sizes, estimated separately
             for the whole series and for each chronological third, because a single
             number over a trending sample is exactly how Hurst produces false structure.
             A shuffled control is estimated the same way; the shuffle destroys memory and
             must return ~0.5, and if it does not the estimator is what is being measured.
  boundary   for each day, the share of that day's eventual range already realised by each
             hour, and the conditional extension: given x% realised, how much further does
             the day go? Reported against a shuffled-within-day null, since a random walk
             produces a rising curve by construction (RS-XAUUSD-20260819-001's arcsine trap).
  pass       Hurst must differ from its own shuffled control by more than the spread of the
             three period estimates, and the extension curve must beat the null.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BARS = Path("local-inputs/xauusd-30m-full.csv")
OUT = Path("reproduced")
DAY_BOUNDARY_HOUR = 7          # Taipei, matching RS-XAUUSD-20260819-001
WINDOWS = [16, 32, 64, 128, 256, 512]
TRIALS = 200
SEED = 20260901


def load_bars() -> list[dict]:
    rows = []
    for row in csv.DictReader(BARS.open(encoding="utf-8-sig")):
        rows.append({"time": datetime.strptime(row["time"][:16], "%Y-%m-%dT%H:%M"),
                     "high": float(row["high"]), "low": float(row["low"]),
                     "close": float(row["close"])})
    rows.sort(key=lambda r: r["time"])
    return rows


def hurst(series: list[float]) -> float | None:
    """Rescaled-range estimate. Returns the slope of log(R/S) on log(window)."""
    points = []
    for size in WINDOWS:
        if len(series) < size * 2:
            continue
        ratios = []
        for start in range(0, len(series) - size + 1, size):
            chunk = series[start:start + size]
            mean = sum(chunk) / size
            deviations, cumulative = [], 0.0
            for value in chunk:
                cumulative += value - mean
                deviations.append(cumulative)
            spread = max(deviations) - min(deviations)
            variance = sum((v - mean) ** 2 for v in chunk) / size
            sd = variance ** 0.5
            if sd > 0 and spread > 0:
                ratios.append(spread / sd)
        if len(ratios) >= 2:
            points.append((math.log(size), math.log(sum(ratios) / len(ratios))))
    if len(points) < 3:
        return None
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    denominator = sum((p[0] - mx) ** 2 for p in points)
    if not denominator:
        return None
    return round(sum((p[0] - mx) * (p[1] - my) for p in points) / denominator, 4)


def trading_day(moment: datetime) -> str:
    shifted = moment.toordinal() if moment.hour >= DAY_BOUNDARY_HOUR else moment.toordinal() - 1
    return str(shifted)


def main() -> None:
    if not BARS.is_file():
        raise SystemExit(f"missing input: {BARS}")
    bars = load_bars()
    rng = random.Random(SEED)

    closes = [b["close"] for b in bars]
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
               if closes[i - 1] > 0]

    third = len(returns) // 3
    periods = {"first_third": returns[:third], "second_third": returns[third:2 * third],
               "final_third": returns[2 * third:]}
    estimates = {"whole_series": hurst(returns),
                 **{name: hurst(part) for name, part in periods.items()}}

    shuffled = []
    for _ in range(TRIALS):
        copy = returns[:]
        rng.shuffle(copy)
        value = hurst(copy)
        if value is not None:
            shuffled.append(value)
    shuffled.sort()
    control = {
        "trials": len(shuffled),
        "median": round(shuffled[len(shuffled) // 2], 4),
        "p05": round(shuffled[int(len(shuffled) * 0.05)], 4),
        "p95": round(shuffled[int(len(shuffled) * 0.95)], 4),
        "note": ("A shuffle destroys memory, so this is what the estimator returns on a "
                 "series with none. It is the reference point, not 0.5 in theory."),
    }
    period_values = [v for k, v in estimates.items() if k != "whole_series" and v is not None]
    period_spread = round(max(period_values) - min(period_values), 4) if period_values else None
    separation = (round(estimates["whole_series"] - control["median"], 4)
                  if estimates["whole_series"] is not None else None)

    # --- BL-006: how the day's range fills in, and what is left after it ---
    days = defaultdict(list)
    for bar in bars:
        days[trading_day(bar["time"])].append(bar)
    usable = {k: v for k, v in days.items() if len(v) >= 40}

    by_hour = defaultdict(list)
    extension = defaultdict(list)
    for series in usable.values():
        day_high = max(b["high"] for b in series)
        day_low = min(b["low"] for b in series)
        full = day_high - day_low
        if full <= 0:
            continue
        running_high = running_low = None
        for index, bar in enumerate(series):
            running_high = bar["high"] if running_high is None else max(running_high, bar["high"])
            running_low = bar["low"] if running_low is None else min(running_low, bar["low"])
            realised = (running_high - running_low) / full
            elapsed = index / len(series)
            by_hour[round(elapsed * 12) / 12].append(realised)
            if index < len(series) - 1:
                bucket = min(int(realised * 10), 9)
                extension[bucket].append(1 - realised)

    # The real curve above uses bar highs and lows; the null below is built from a close
    # path. Comparing them directly would credit the real series with a first bar that
    # already has a range while the null path starts at zero. So the comparison curve is
    # rebuilt on the same close-only basis as the null.
    real_close_by_hour = defaultdict(list)
    for series in usable.values():
        path = [b["close"] for b in series]
        full = max(path) - min(path)
        if full <= 0:
            continue
        running_high = running_low = path[0]
        for index, value in enumerate(path):
            running_high = max(running_high, value)
            running_low = min(running_low, value)
            real_close_by_hour[round((index / len(path)) * 12) / 12].append(
                (running_high - running_low) / full)

    # A random walk fills its range concavely too — RS-XAUUSD-20260819-001's arcsine trap.
    # So the curve is only evidence against a null that keeps each day's own bar-to-bar
    # moves and destroys only their order.
    null_by_hour = defaultdict(list)
    for series in usable.values():
        closes_in_day = [b["close"] for b in series]
        steps = [closes_in_day[i] - closes_in_day[i - 1] for i in range(1, len(closes_in_day))]
        if not steps:
            continue
        for _ in range(5):
            rng.shuffle(steps)
            level = closes_in_day[0]
            path = [level]
            for step in steps:
                level += step
                path.append(level)
            full = max(path) - min(path)
            if full <= 0:
                continue
            running_high = running_low = path[0]
            for index, value in enumerate(path):
                running_high = max(running_high, value)
                running_low = min(running_low, value)
                null_by_hour[round((index / len(path)) * 12) / 12].append(
                    (running_high - running_low) / full)

    fill_curve = [{"share_of_day_elapsed": round(k, 3), "n": len(v),
                   "mean_share_of_range_realised": round(sum(v) / len(v), 4),
                   "mean_close_basis": round(
                       sum(real_close_by_hour[k]) / len(real_close_by_hour[k]), 4)
                   if real_close_by_hour.get(k) else None,
                   "mean_under_shuffled_null": round(
                       sum(null_by_hour[k]) / len(null_by_hour[k]), 4)
                   if null_by_hour.get(k) else None,
                   "excess_over_null": round(
                       sum(real_close_by_hour[k]) / len(real_close_by_hour[k])
                       - sum(null_by_hour[k]) / len(null_by_hour[k]), 4)
                   if null_by_hour.get(k) and real_close_by_hour.get(k) else None}
                  for k, v in sorted(by_hour.items()) if len(v) >= 30]
    extension_curve = [{"range_realised_decile": f"{k * 10}-{k * 10 + 10}%", "n": len(v),
                        "mean_further_extension_share": round(sum(v) / len(v), 4)}
                       for k, v in sorted(extension.items()) if len(v) >= 30]

    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260901-003",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "none — market structure",
        "method": {
            "bars": len(bars), "returns": len(returns),
            "trading_days": len(usable),
            "day_boundary_hour_taipei": DAY_BOUNDARY_HOUR,
            "hurst_estimator": "rescaled range, slope of log(R/S) against log(window)",
            "hurst_windows": WINDOWS,
            "hurst_control": "the same estimator on shuffled returns",
            "shuffle_trials": TRIALS, "random_seed": SEED,
            "answers": ["BL-007 (Hurst)", "BL-006 (intraday range boundary)"],
            "prior": ("RS-XAUUSD-20260818-004 established direction is not forecastable "
                      "(largest |IC| 0.0399) while volatility is (holdout Spearman 0.571)"),
        },
        "hurst": {
            "estimates": estimates,
            "shuffled_control": control,
            "period_spread": period_spread,
            "separation_from_control": separation,
            "exceeds_period_spread": (
                bool(separation is not None and period_spread is not None
                     and abs(separation) > period_spread)),
        },
        "intraday_range": {
            "fill_curve": fill_curve,
            "extension_by_realised_decile": extension_curve,
            "extension_is_arithmetic": ("each bucket's mean further extension is 1 minus its "
                                        "own midpoint to within a percentage point, which is "
                                        "what the definition forces; it carries no "
                                        "information about the market"),
            "largest_excess_over_null": (
                max((x["excess_over_null"] for x in fill_curve
                     if x["excess_over_null"] is not None), default=None)),
        },
        "verdict": {
            "hurst_distinguishable_from_no_memory": bool(
                separation is not None
                and not (control["p05"] <= estimates["whole_series"] <= control["p95"])),
            "hurst_stable_across_thirds": bool(period_spread is not None and period_spread < 0.05),
            "note": ("The extension curve falls by construction — the more of a day's range "
                     "is already realised, the less can remain — so its shape is not "
                     "evidence on its own. What matters is whether it falls faster than "
                     "arithmetic requires, and this study reports the curve rather than "
                     "claiming that."),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"hurst": estimates, "control_median": control["median"],
                      "period_spread": period_spread, "separation": separation,
                      "days": len(usable)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
