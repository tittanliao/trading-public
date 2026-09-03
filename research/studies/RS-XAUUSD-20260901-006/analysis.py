#!/usr/bin/env python3
"""RS-XAUUSD-20260901-006 — is "near an S1 exit" a mechanism, or volatility clustering?

Backlog BL-013. RS-XAUUSD-20260901-005 found that S2 trades with an S1 exit in the previous
48 hours win 51.22% against 36.00% for those without — a 15.22-point gap, twice the
stop-versus-target gap it was built to measure. "Near another strategy's exit" is not
obviously a mechanism. Both strategies are more active when the market moves, and volatility
is the one quantity this series is known to forecast (RS-XAUUSD-20260818-004, holdout
Spearman 0.571), which makes it the natural confounder.

Frozen before running (see decision_log.md):
  volatility ATR(14) percentile over the trailing 240 bars, read at the last COMPLETED bar
             before the fill. Same definition as RS-XAUUSD-20260818-002's atr_percentile_240
             and the same no-lookahead convention RS-XAUUSD-20260901-001 established.
  strata     terciles of that percentile, cut on the S2 trades being compared
  primary    the within-stratum gap, weighted by stratum size, against the unstratified
             15.22. If volatility explains the effect, the weighted gap collapses.
  null       the near/not label shuffled WITHIN each stratum, which is the only null that
             tests the residual rather than the confound
  confound   S1 exit density by volatility tercile, reported so the confound is visible
             rather than assumed
  pass       the stratified gap survives at more than half the unstratified gap AND
             permutation p<0.05
"""
from __future__ import annotations

import csv
import json
import random
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BARS = Path("local-inputs/xauusd-30m-full.csv")
S1_FILE = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv")
S2_FILE = Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv")
OUT = Path("reproduced")
ATR_PERIOD, ATR_LOOKBACK = 14, 240
WINDOW_HOURS = 48
TRIALS = 20000
SEED = 20260901
EXPORT_DATE = "2026-07-11"


def load_bars() -> tuple[list[datetime], list[float]]:
    times, highs, lows, closes = [], [], [], []
    for row in csv.DictReader(BARS.open(encoding="utf-8-sig")):
        times.append(datetime.strptime(row["time"][:16], "%Y-%m-%dT%H:%M"))
        highs.append(float(row["high"]))
        lows.append(float(row["low"]))
        closes.append(float(row["close"]))
    true_range = [highs[0] - lows[0]]
    for i in range(1, len(times)):
        true_range.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                              abs(lows[i] - closes[i - 1])))
    atr, running = [], None
    for i, value in enumerate(true_range):
        running = value if running is None else (running * (ATR_PERIOD - 1) + value) / ATR_PERIOD
        atr.append(running if i >= ATR_PERIOD else None)
    percentile = []
    for i, value in enumerate(atr):
        if value is None or i < ATR_LOOKBACK:
            percentile.append(None)
            continue
        window = [v for v in atr[i - ATR_LOOKBACK:i + 1] if v is not None]
        percentile.append(sum(1 for v in window if v <= value) / len(window))
    return times, percentile


def load(path: Path) -> list[dict]:
    pairs: dict[str, dict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        pairs.setdefault(row["Trade number"], {})[row["Type"]] = row
    out = []
    for pair in pairs.values():
        entry, exit_ = pair.get("Entry long"), pair.get("Exit long")
        if not entry or not exit_ or exit_["Date and time"][:10] == EXPORT_DATE:
            continue
        out.append({"entry": datetime.strptime(entry["Date and time"], "%Y-%m-%d %H:%M"),
                    "exit": datetime.strptime(exit_["Date and time"], "%Y-%m-%d %H:%M"),
                    "return_pct": float(exit_["Return %"])})
    return sorted(out, key=lambda t: t["entry"])


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    values = [r["return_pct"] for r in rows]
    wins = sum(1 for v in values if v > 0)
    return {"n": len(rows), "win_rate_pct": round(100 * wins / len(rows), 2),
            "mean_return_pct": round(sum(values) / len(values), 4)}


def bound(n1: int, n2: int, rate: float) -> float:
    if n1 < 2 or n2 < 2:
        return float("inf")
    p = rate / 100
    return round(100 * 2.802 * (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5, 2)


def weighted_gap(strata: list[list[dict]]) -> float | None:
    """Within-stratum gap, weighted by stratum size. The confound cannot contribute."""
    total, weight = 0.0, 0
    for group in strata:
        near = [t for t in group if t["near"]]
        far = [t for t in group if not t["near"]]
        if not near or not far:
            continue
        gap = (100 * sum(1 for t in near if t["return_pct"] > 0) / len(near)
               - 100 * sum(1 for t in far if t["return_pct"] > 0) / len(far))
        total += gap * len(group)
        weight += len(group)
    return round(total / weight, 2) if weight else None


def main() -> None:
    for path in (BARS, S1_FILE, S2_FILE):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    times, percentile = load_bars()
    s1, s2 = load(S1_FILE), load(S2_FILE)
    rng = random.Random(SEED)
    span = timedelta(hours=WINDOW_HOURS)

    rows = []
    for trade in s2:
        index = bisect_left(times, trade["entry"])
        if index >= len(times) or times[index] != trade["entry"] or index == 0:
            continue
        value = percentile[index - 1]        # last completed bar before the fill
        if value is None:
            continue
        near = any(e["exit"] <= trade["entry"] <= e["exit"] + span for e in s1)
        rows.append({**trade, "atr_pct": value, "near": near})

    near_all = [t for t in rows if t["near"]]
    far_all = [t for t in rows if not t["near"]]
    unstratified = round(stats(near_all)["win_rate_pct"] - stats(far_all)["win_rate_pct"], 2)

    ordered = sorted(rows, key=lambda t: t["atr_pct"])
    size = len(ordered) // 3
    strata = [ordered[:size], ordered[size:2 * size], ordered[2 * size:]]
    labels = ["low_volatility", "mid_volatility", "high_volatility"]
    by_stratum = {}
    for label, group in zip(labels, strata):
        near = [t for t in group if t["near"]]
        far = [t for t in group if not t["near"]]
        gap = (round(stats(near)["win_rate_pct"] - stats(far)["win_rate_pct"], 2)
               if near and far else None)
        by_stratum[label] = {
            "atr_percentile_range": [round(group[0]["atr_pct"], 3),
                                     round(group[-1]["atr_pct"], 3)],
            "near_an_s1_exit": stats(near), "no_s1_exit_nearby": stats(far),
            "win_rate_gap_pct_points": gap,
            "share_near_pct": round(100 * len(near) / len(group), 2),
            "min_detectable_pct_points": bound(len(near), len(far),
                                               stats(group)["win_rate_pct"]),
        }

    stratified = weighted_gap(strata)

    # The confound made visible: if S1 exits cluster in volatile stretches, the share of S2
    # trades that are "near an exit" rises with the volatility tercile.
    confound = {label: by_stratum[label]["share_near_pct"] for label in labels}
    confound["spread_pct_points"] = round(max(confound.values()) - min(confound.values()), 2)

    # Shuffle the label WITHIN each stratum: tests the residual, not the confound.
    at_least, null = 0, []
    for _ in range(TRIALS):
        shuffled = []
        for group in strata:
            flags = [t["near"] for t in group]
            rng.shuffle(flags)
            shuffled.append([{**t, "near": f} for t, f in zip(group, flags)])
        value = weighted_gap(shuffled)
        if value is None:
            continue
        null.append(value)
        if stratified is not None and value >= stratified:
            at_least += 1
    null.sort()

    results = {
        "schema_version": 1, "study_id": "RS-XAUUSD-20260901-006",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S2 Hammer V3.2, conditioned on S1 AweWithBB V3.9 exits",
        "method": {
            "s2_trades_labelled": len(rows),
            "window_hours": WINDOW_HOURS,
            "volatility": f"ATR({ATR_PERIOD}) percentile over the trailing {ATR_LOOKBACK} bars",
            "read_at": "the last completed bar before the fill",
            "definition_source": "RS-XAUUSD-20260818-002's atr_percentile_240",
            "strata": "terciles of that percentile",
            "primary": "the size-weighted within-stratum gap against the unstratified gap",
            "null": "the near/not label shuffled within each stratum",
            "permutation_trials": TRIALS, "random_seed": SEED,
            "tests": "RS-XAUUSD-20260901-005's residual finding",
        },
        "unstratified": {
            "near_an_s1_exit": stats(near_all), "no_s1_exit_nearby": stats(far_all),
            "win_rate_gap_pct_points": unstratified,
            "min_detectable_pct_points": bound(len(near_all), len(far_all),
                                               stats(rows)["win_rate_pct"]),
        },
        "by_volatility_tercile": by_stratum,
        "confound_share_near_an_exit_by_tercile": confound,
        "stratified": {
            "weighted_within_stratum_gap_pct_points": stratified,
            "unstratified_gap_pct_points": unstratified,
            "share_of_gap_surviving_pct": round(100 * stratified / unstratified, 1)
            if unstratified else None,
            "permutation_p": round(at_least / len(null), 4) if null else None,
            "null_median_gap_pct_points": round(null[len(null) // 2], 2) if null else None,
            "null_95th_gap_pct_points": round(null[int(len(null) * 0.95)], 2) if null else None,
        },
        "verdict": {
            "gap_survives_stratification": bool(
                stratified is not None and unstratified
                and stratified > unstratified / 2),
            "explained_by_volatility": bool(
                stratified is not None and unstratified
                and stratified <= unstratified / 2),
            "passes_frozen_criteria": bool(
                stratified is not None and unstratified
                and stratified > unstratified / 2
                and (round(at_least / len(null), 4) if null else 1) < 0.05),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"unstratified": unstratified, "stratified": stratified,
                      "survives_pct": results["stratified"]["share_of_gap_surviving_pct"],
                      "p": results["stratified"]["permutation_p"],
                      "confound_spread": confound["spread_pct_points"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
