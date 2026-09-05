#!/usr/bin/env python3
"""RS-XAUUSD-20260901-002 — a scorecard for the weekly reports, and what two weeks can say.

Backlog BL-004. The weekly report is this programme's main product and has never been
scored against what happened. The backlog item said the first version's output should be
the scoring rule plus an honest bound, not a hit rate, because three editions cannot
support one. That is what this is.

Frozen before any outcome was looked at (see decision_log.md):
  direction   a week is bullish if its close is more than BAND_PCT above the prior weekly
              close, bearish if more than BAND_PCT below, range otherwise. BAND_PCT = 1.0.
  scored      the highest-probability adopted scenario's direction, and a Brier score over
              the full probability vector, so a report is not rewarded for hedging.
  levels      a numeric key level counts as in play if the week's range touched it.
  targets     the first numeric target of the top scenario counts as reached if touched.
  bound       with n editions the smallest resolvable difference from a coin flip is
              reported alongside every rate, because that is the number that decides
              whether any of this means anything yet.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PUBLIC = REPO.parent / "trading-public"
BARS = Path("local-inputs/xauusd-weekly-scoring-30m.csv")
OUT = Path("reproduced")
BAND_PCT = 1.0


def load_bars() -> list[dict]:
    rows = []
    for row in csv.DictReader(BARS.open(encoding="utf-8-sig")):
        rows.append({
            "time": datetime.fromisoformat(row["time"].replace("Z", "")).replace(tzinfo=None),
            "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"]),
        })
    rows.sort(key=lambda r: r["time"])
    return rows


def week_window(week: str) -> tuple[date, date]:
    year, number = int(week[:4]), int(week.split("-W")[1])
    start = date.fromisocalendar(year, number, 1)
    return start, start + timedelta(days=6)


def numbers(text: str) -> list[float]:
    """Every price-like number in a string, in order."""
    out = []
    for token in re.findall(r"\d[\d,]*\.?\d*", text or ""):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if 100 <= value <= 100000:  # a price, not a percentage or a count
            out.append(value)
    return out


def resolvable_rate_gap(n: int) -> float:
    """Smallest deviation from 50% that n observations could separate, alpha .05 power .80."""
    if n < 1:
        return float("inf")
    return round(100 * 2.802 * (0.25 / n) ** 0.5, 2)


def main() -> None:
    if not BARS.is_file():
        raise SystemExit(f"missing input: {BARS}")
    bars = load_bars()
    editions = sorted(p.parent.name for p in PUBLIC.glob("xauusd/weekly/*/summary.json"))

    scored, skipped = [], []
    for week in editions:
        summary = json.loads((PUBLIC / "xauusd/weekly" / week / "summary.json").read_text())
        start, end = week_window(week)
        inside = [b for b in bars if start <= b["time"].date() <= end]
        prior = [b for b in bars if b["time"].date() < start]
        if not inside or not prior or inside[-1]["time"].date() < end - timedelta(days=2):
            skipped.append({"week": week, "reason": "price data does not cover the full week",
                            "bars_available": len(inside)})
            continue

        open_ref = prior[-1]["close"]
        close = inside[-1]["close"]
        high = max(b["high"] for b in inside)
        low = min(b["low"] for b in inside)
        move = 100 * (close / open_ref - 1)
        actual = "range" if abs(move) < BAND_PCT else ("bullish" if move > 0 else "bearish")

        scenarios = summary.get("adopted_scenarios") or []
        top = max(scenarios, key=lambda s: s.get("probability", 0)) if scenarios else None
        # A tie at the top means "the highest-probability scenario" is not well defined, and
        # whichever way the tie breaks decides the hit. Recorded rather than resolved.
        best_p = top.get("probability", 0) if top else 0
        tied = [s["direction"] for s in scenarios if s.get("probability", 0) == best_p]
        weights = {s["direction"]: s.get("probability", 0) / 100 for s in scenarios}
        brier = sum((weights.get(d, 0.0) - (1.0 if d == actual else 0.0)) ** 2
                    for d in ("bullish", "range", "bearish"))

        levels = []
        for level in summary.get("key_levels") or []:
            values = numbers(level.get("value", ""))
            if not values:
                continue
            touched = any(low <= v <= high for v in values)
            levels.append({"label": level.get("label"), "values": values, "touched": touched})

        targets = numbers(top.get("targets", "")) if top else []
        first_target = targets[0] if targets else None

        scored.append({
            "week": week,
            "from": str(start), "to": str(end),
            "prior_close": round(open_ref, 2), "close": round(close, 2),
            "high": round(high, 2), "low": round(low, 2),
            "move_pct": round(move, 3),
            "actual_direction": actual,
            "forecast_top_direction": top.get("direction") if top else None,
            "forecast_top_probability": top.get("probability") if top else None,
            "direction_hit": bool(top and top.get("direction") == actual),
            "top_was_a_tie": len(tied) > 1,
            "tied_directions": tied,
            "hit_under_any_tie_break": actual in tied,
            "hit_under_every_tie_break": len(tied) == 1 and tied[0] == actual,
            "probability_vector": weights,
            "brier_score": round(brier, 4),
            "key_levels_total": len(levels),
            "key_levels_touched": sum(1 for x in levels if x["touched"]),
            "key_levels": levels,
            "first_target": first_target,
            "first_target_reached": bool(first_target and low <= first_target <= high),
        })

    n = len(scored)
    hits = sum(1 for s in scored if s["direction_hit"])
    touched = sum(s["key_levels_touched"] for s in scored)
    total_levels = sum(s["key_levels_total"] for s in scored)
    briers = [s["brier_score"] for s in scored]

    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260901-002",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "editions_found": len(editions),
            "editions_scored": n,
            "direction_band_pct": BAND_PCT,
            "direction_rule": ("close versus the prior weekly close; inside +/-1.0% is range"),
            "brier": ("squared error over the full three-way probability vector, so hedging "
                      "is not rewarded"),
            "level_rule": "a numeric key level is in play if the week's range touched it",
            "bound": ("the smallest deviation from a coin flip these many editions could "
                      "separate, at alpha 0.05 and power 0.80"),
            "frozen_before_outcomes_were_read": True,
        },
        "editions": scored,
        "not_scored": skipped,
        "totals": {
            "editions_scored": n,
            "direction_hits": hits,
            "direction_hits_if_ties_break_favourably": sum(
                1 for s in scored if s["hit_under_any_tie_break"]),
            "direction_hits_if_ties_break_unfavourably": sum(
                1 for s in scored if s["hit_under_every_tie_break"]),
            "editions_with_a_tied_top_scenario": sum(1 for s in scored if s["top_was_a_tie"]),
            "direction_hit_rate_pct": round(100 * hits / n, 2) if n else None,
            "smallest_resolvable_deviation_from_coin_flip_pct_points": resolvable_rate_gap(n),
            "mean_brier_score": round(sum(briers) / len(briers), 4) if briers else None,
            "brier_of_always_saying_range": None,
            "key_levels_total": total_levels,
            "key_levels_touched": touched,
            "key_level_touch_rate_pct": round(100 * touched / total_levels, 2) if total_levels else None,
            "first_targets_reached": sum(1 for s in scored if s["first_target_reached"]),
        },
        "verdict": {
            "resolvable": False,
            "why": ("Two editions cannot separate any hit rate from a coin flip: the bound is "
                    f"{resolvable_rate_gap(n)} percentage points against a maximum possible "
                    "deviation of 50. The scorecard is the deliverable; the rate is not."),
        },
    }
    # A baseline any forecaster must beat: name one direction every week with certainty.
    if scored:
        def brier_if_always(direction: str) -> float:
            total = 0.0
            for edition in scored:
                for option in ("bullish", "range", "bearish"):
                    predicted = 1.0 if option == direction else 0.0
                    outcome = 1.0 if option == edition["actual_direction"] else 0.0
                    total += (predicted - outcome) ** 2
            return round(total / len(scored), 4)

        results["totals"]["brier_of_always_saying_range"] = brier_if_always("range")
        results["totals"]["brier_of_always_saying_bullish"] = brier_if_always("bullish")
        results["totals"]["brier_of_always_saying_bearish"] = brier_if_always("bearish")
        results["totals"]["modal_actual_direction"] = max(
            ("bullish", "range", "bearish"),
            key=lambda d: sum(1 for s in scored if s["actual_direction"] == d))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"scored": n, "skipped": [s["week"] for s in skipped],
                      "hits": hits, "bound_pct_points": resolvable_rate_gap(n)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
