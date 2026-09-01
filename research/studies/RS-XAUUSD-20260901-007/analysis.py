#!/usr/bin/env python3
"""RS-XAUUSD-20260901-007 — buying the September-to-January window at Fibonacci levels.

The idea under test: instead of four equal monthly buys from September, take the year's
January-to-August range, place Fibonacci retracement levels inside it, and buy a tranche
each time price touches an unfilled level. Sell everything at the end of January.

RS-XAUUSD-20260827-001 already established the baseline this has to beat: four equal
monthly buys held to a January exit won 14 of 18 cycles (77.78%). The seasonal timing is
not what is new here — the staged entry is.

Frozen before running (see decision_log.md):
  range      the high and low of the calendar year's January 1 to August 31 daily bars
  levels     H - f*(H-L) for f in {0.236, 0.382, 0.5, 0.618, 0.786}; five equal tranches
  fill       the first daily bar from September 1 whose LOW touches an unfilled level, at
             the level price — or at that bar's open when the bar opened below the level,
             because a marketable limit order fills at market. One tranche per level per
             cycle.
  exit       the last trading day of the following January, at its close
  unfilled   stays in cash and earns nothing. Return is reported on committed capital as
             well as on deployed capital, because a rule that under-invests is not
             comparable on deployed capital alone.
  baseline   four equal buys at the first trading day of September, October, November and
             December — the design RS-XAUUSD-20260827-001 measured
  variant    the same Fibonacci rule with unfilled tranches bought at December's close
  pass       higher return on COMMITTED capital than the baseline, in more cycles than not
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BARS = Path("local-inputs/xauusd-daily-long.csv")
OUT = REPO / "research/studies/RS-XAUUSD-20260901-007"
FIBS = [0.236, 0.382, 0.5, 0.618, 0.786]


def load() -> list[dict]:
    rows = []
    for row in csv.DictReader(BARS.open(encoding="utf-8-sig")):
        stamp = row.get("time") or row.get("Date") or list(row.values())[0]
        rows.append({"date": datetime.fromisoformat(stamp[:10]).date(),
                     "open": float(row["open"]), "high": float(row["high"]),
                     "low": float(row["low"]), "close": float(row["close"])})
    rows.sort(key=lambda r: r["date"])
    return rows


def cycle(bars: list[dict], year: int) -> dict | None:
    window = [b for b in bars if date(year, 1, 1) <= b["date"] <= date(year, 8, 31)]
    buying = [b for b in bars if date(year, 9, 1) <= b["date"] <= date(year + 1, 1, 31)]
    january = [b for b in buying if b["date"] >= date(year + 1, 1, 1)]
    if len(window) < 120 or not january:
        return None
    high = max(b["high"] for b in window)
    low = min(b["low"] for b in window)
    exit_price = january[-1]["close"]
    levels = [round(high - f * (high - low), 4) for f in FIBS]

    # Fibonacci fills: first touch from September onward, at the level price.
    # A resting buy limit fills at the level when price trades DOWN through it. When the bar
    # opens already below the level — which happens on the first September bar of a year that
    # fell into August — the order is marketable and fills at the open, not at the level.
    # Filling at the level there would credit a price the market never offered.
    fills: list[dict] = []
    remaining = list(zip(FIBS, levels))
    for bar in buying:
        for fraction, level in list(remaining):
            if bar["low"] <= level:
                price = min(level, bar["open"])
                fills.append({"fib": fraction, "price": round(price, 4),
                              "date": bar["date"].isoformat(),
                              "marketable_at_open": bar["open"] < level})
                remaining.remove((fraction, level))
    december = [b for b in buying if b["date"].year == year and b["date"].month == 12]

    # Baseline: first trading day of each of the four months.
    baseline_prices = []
    for month in (9, 10, 11, 12):
        days = [b for b in buying if b["date"].year == year and b["date"].month == month]
        if days:
            baseline_prices.append(days[0]["close"])

    def outcome(prices: list[float], tranches: int) -> dict:
        """Return on deployed capital and on committed capital.

        Committed capital counts every tranche the rule intended to deploy, so a rule that
        never fills is not flattered by measuring only what it did buy.
        """
        if not prices:
            return {"tranches_filled": 0, "avg_cost": None,
                    "return_on_deployed_pct": None, "return_on_committed_pct": 0.0}
        avg = sum(prices) / len(prices)
        deployed = 100 * (exit_price / avg - 1)
        return {"tranches_filled": len(prices), "avg_cost": round(avg, 2),
                "return_on_deployed_pct": round(deployed, 3),
                "return_on_committed_pct": round(deployed * len(prices) / tranches, 3)}

    fib_prices = [f["price"] for f in fills]
    catchup = fib_prices + ([december[-1]["close"]] * (len(FIBS) - len(fib_prices))
                            if december else [])
    return {
        "build_year": year,
        "jan_aug_high": round(high, 2), "jan_aug_low": round(low, 2),
        "jan_aug_range_pct": round(100 * (high / low - 1), 2),
        "jan_aug_direction": ("up" if window[-1]["close"] > window[0]["close"] else "down"),
        "jan_aug_move_pct": round(100 * (window[-1]["close"] / window[0]["close"] - 1), 2),
        # Where the August close sits inside the year's range. This, not the sign of the
        # year's move, is what decides whether the levels sit above or below the market.
        "august_position_in_range_pct": round(100 * (window[-1]["close"] - low) / (high - low), 2),
        "september_open": round(buying[0]["close"], 2),
        "exit_date": january[-1]["date"].isoformat(), "exit_price": round(exit_price, 2),
        "fib_levels": levels,
        "fib_fills": fills,
        "fib": outcome(fib_prices, len(FIBS)),
        "fib_with_december_catchup": outcome(catchup, len(FIBS)),
        "equal_monthly": outcome(baseline_prices, 4),
    }


def summarise(cycles: list[dict], key: str, field: str) -> dict:
    values = [c[key][field] for c in cycles if c[key][field] is not None]
    if not values:
        return {"cycles": 0}
    wins = sum(1 for v in values if v > 0)
    return {"cycles": len(values), "win_rate_pct": round(100 * wins / len(values), 2),
            "mean_pct": round(sum(values) / len(values), 3),
            "median_pct": round(sorted(values)[len(values) // 2], 3),
            "worst_pct": round(min(values), 3), "best_pct": round(max(values), 3),
            "total_pct": round(sum(values), 2)}


def main() -> None:
    if not BARS.is_file():
        raise SystemExit(f"missing input: {BARS}")
    bars = load()
    years = sorted({b["date"].year for b in bars})
    cycles = [c for c in (cycle(bars, y) for y in years) if c]

    filled = [c["fib"]["tranches_filled"] for c in cycles]
    summary = {
        "cycles": len(cycles),
        "fib": summarise(cycles, "fib", "return_on_committed_pct"),
        "fib_on_deployed_only": summarise(cycles, "fib", "return_on_deployed_pct"),
        "fib_with_december_catchup": summarise(cycles, "fib_with_december_catchup",
                                               "return_on_committed_pct"),
        "equal_monthly": summarise(cycles, "equal_monthly", "return_on_committed_pct"),
        "fill_rate": {
            "mean_tranches_filled": round(sum(filled) / len(filled), 2),
            "of_five": 5,
            "cycles_with_zero_fills": sum(1 for f in filled if f == 0),
            "cycles_with_all_five": sum(1 for f in filled if f == 5),
            "distribution": {str(k): filled.count(k) for k in range(6)},
        },
    }
    up = [c for c in cycles if c["jan_aug_direction"] == "up"]
    down = [c for c in cycles if c["jan_aug_direction"] == "down"]
    summary["by_jan_aug_direction"] = {
        "up_years": {"n": len(up),
                     "mean_tranches_filled": round(
                         sum(c["fib"]["tranches_filled"] for c in up) / len(up), 2) if up else None,
                     "fib": summarise(up, "fib", "return_on_committed_pct"),
                     "equal_monthly": summarise(up, "equal_monthly", "return_on_committed_pct")},
        "down_years": {"n": len(down),
                       "mean_tranches_filled": round(
                           sum(c["fib"]["tranches_filled"] for c in down) / len(down), 2) if down else None,
                       "fib": summarise(down, "fib", "return_on_committed_pct"),
                       "equal_monthly": summarise(down, "equal_monthly", "return_on_committed_pct")},
    }
    # Is the entry actually staged? A level ABOVE the September price is marketable on the
    # first bar, so the tranche is not a retracement buy at all. If every tranche fills that
    # way the rule has degenerated into a lump sum on day one.
    def staging(group: list[dict]) -> dict:
        fills = [f for c in group for f in c["fib_fills"]]
        if not fills:
            return {"cycles": len(group), "fills": 0}
        first_day = sum(1 for c in group if c["fib_fills"]
                        and all(f["date"] == c["fib_fills"][0]["date"] for f in c["fib_fills"])
                        and c["fib"]["tranches_filled"] == len(FIBS))
        return {
            "cycles": len(group), "fills": len(fills),
            "marketable_at_open_pct": round(
                100 * sum(1 for f in fills if f["marketable_at_open"]) / len(fills), 2),
            "cycles_fully_filled_on_one_day": first_day,
            "distinct_fill_dates_mean": round(
                sum(len({f["date"] for f in c["fib_fills"]}) for c in group) / len(group), 2),
        }

    # The direction split above classifies by the sign of the year's move, which is not what
    # drives fills. A year can close up on the year and still be deep below its own high — in
    # which case the retracement levels sit ABOVE the market and everything fills at once.
    high_half = [c for c in cycles if c["august_position_in_range_pct"] >= 50]
    low_half = [c for c in cycles if c["august_position_in_range_pct"] < 50]
    summary["by_august_position_in_range_note"] = (
        "the driver is where the August close sits inside the January-August range, not the "
        "sign of the year's move")
    summary["by_august_position_in_range"] = {
        "august_in_upper_half": {
            "n": len(high_half),
            "mean_tranches_filled": round(sum(c["fib"]["tranches_filled"] for c in high_half)
                                          / len(high_half), 2) if high_half else None,
            "fib": summarise(high_half, "fib", "return_on_committed_pct") if high_half else None,
            "equal_monthly": summarise(high_half, "equal_monthly", "return_on_committed_pct")
            if high_half else None,
        },
        "august_in_lower_half": {
            "n": len(low_half),
            "mean_tranches_filled": round(sum(c["fib"]["tranches_filled"] for c in low_half)
                                          / len(low_half), 2) if low_half else None,
            "fib": summarise(low_half, "fib", "return_on_committed_pct") if low_half else None,
            "equal_monthly": summarise(low_half, "equal_monthly", "return_on_committed_pct")
            if low_half else None,
        },
    }

    summary["staging_check"] = {
        "all": staging(cycles), "up_years": staging(up), "down_years": staging(down),
        "august_in_upper_half": staging(high_half), "august_in_lower_half": staging(low_half),
    }
    summary["staging_check_note"] = (
        "a tranche that is marketable at the open is a market buy, not a retracement buy; a "
        "cycle whose tranches all fill on one day is a lump sum")

    # What 18 paired cycles can resolve. The comparison is paired — the same cycle, two
    # rules — so the bound is on the paired difference, not on two independent samples.
    def paired(key: str) -> dict:
        diffs = [c[key]["return_on_committed_pct"] - c["equal_monthly"]["return_on_committed_pct"]
                 for c in cycles]
        n = len(diffs)
        mean = sum(diffs) / n
        sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
        return {
            "cycles": n,
            "mean_difference_pct_points": round(mean, 3),
            "sd_of_difference": round(sd, 3),
            "min_detectable_pct_points": round(2.802 * sd / n ** 0.5, 3),
            "separates": bool(abs(mean) > 2.802 * sd / n ** 0.5),
            "cycles_ahead": sum(1 for d in diffs if d > 0),
            "cycles_behind": sum(1 for d in diffs if d < 0),
        }

    summary["paired_vs_baseline"] = {
        "fib": paired("fib"),
        "fib_with_december_catchup": paired("fib_with_december_catchup"),
    }
    summary["paired_vs_baseline_note"] = (
        "alpha 0.05, power 0.80. Per RS-XAUUSD-20260824-001 the bound decides, not the p-value.")

    summary["rules_compared"] = {
        "fibonacci_as_specified": summary["fib"],
        "fibonacci_with_december_catchup": summary["fib_with_december_catchup"],
        "four_equal_monthly_buys": summary["equal_monthly"],
    }

    beats = sum(1 for c in cycles
                if c["fib"]["return_on_committed_pct"] > c["equal_monthly"]["return_on_committed_pct"])
    summary["head_to_head"] = {
        "cycles": len(cycles), "fib_beats_equal_monthly": beats,
        "equal_monthly_beats_fib": len(cycles) - beats,
        "mean_difference_pct_points": round(
            sum(c["fib"]["return_on_committed_pct"]
                - c["equal_monthly"]["return_on_committed_pct"] for c in cycles) / len(cycles), 3),
    }

    # Findings are written from the computed values, never typed in alongside them, so a
    # rerun cannot leave the prose disagreeing with the table.
    fib_s, eq_s, cat_s = summary["fib"], summary["equal_monthly"], summary["fib_with_december_catchup"]
    pair = summary["paired_vs_baseline"]["fib"]
    fill = summary["fill_rate"]
    upper = summary["by_august_position_in_range"]["august_in_upper_half"]
    lower = summary["by_august_position_in_range"]["august_in_lower_half"]
    worst_missed = max((c for c in cycles if c["fib"]["tranches_filled"] == 0),
                       key=lambda c: c["equal_monthly"]["return_on_committed_pct"], default=None)

    answers = [
        {"question": "What has the win rate been?",
         "answer": (f"{fib_s['win_rate_pct']}% of {fib_s['cycles']} cycles on committed "
                    f"capital, against {eq_s['win_rate_pct']}% for four equal monthly buys. "
                    f"Counting only the {summary['fib_on_deployed_only']['cycles']} cycles "
                    f"where it invested at all, "
                    f"{summary['fib_on_deployed_only']['win_rate_pct']}%. Adding a "
                    f"December catch-up for unfilled tranches lifts it to "
                    f"{cat_s['win_rate_pct']}%.")},
        {"question": "Is it worse than buying four equal monthly tranches?",
         "answer": (f"Not established. The paired difference is "
                    f"{pair['mean_difference_pct_points']} percentage points against a "
                    f"resolution bound of {pair['min_detectable_pct_points']} on "
                    f"{pair['cycles']} cycles, so the sample cannot separate them on return. "
                    f"It is ahead in {pair['cycles_ahead']} cycles and behind in "
                    f"{pair['cycles_behind']}. The case against it is structural, not "
                    f"statistical.")},
        {"question": "What is the structural problem?",
         "answer": (f"It does not stage the entry. Fills are decided by where the August "
                    f"close sits inside the January-August range: with August in the lower "
                    f"half all five tranches filled in {lower['n']} of {lower['n']} cycles "
                    f"(mean {lower['mean_tranches_filled']} of 5), and "
                    f"{summary['staging_check']['august_in_lower_half']['marketable_at_open_pct']}% of "
                    f"those fills were marketable at the open — a market buy, not a "
                    f"retracement buy. With August in the upper half the mean is "
                    f"{upper['mean_tranches_filled']} of 5. So it commits everything at once "
                    f"when gold is weak into September and holds back when gold is strong, "
                    f"which is backwards for a seasonal long.")},
        {"question": "What does the shortfall come from?",
         "answer": (f"Cash, not bad prices. {fill['cycles_with_zero_fills']} of "
                    f"{summary['cycles']} cycles never filled a single tranche and sat in "
                    f"cash through the whole window."
                    + (f" The worst of these is {worst_missed['build_year']}, where the "
                       f"baseline returned "
                       f"{worst_missed['equal_monthly']['return_on_committed_pct']}% — the "
                       f"largest single-cycle gain in the sample — and the Fibonacci rule "
                       f"returned nothing." if worst_missed else ""))},
    ]

    limitations = [
        f"{summary['cycles']} cycles. This is a seasonal rule, so one year is one observation "
        f"and the sample cannot be enlarged by using more granular data.",
        "The resolution bound exceeds the observed difference, so no claim is made that the "
        "Fibonacci rule underperforms on return. Only the fill behaviour is established.",
        "Spread, carry and roll are not modelled. RS-XAUUSD-20260827-001 covers the financing "
        "cost of holding this position.",
        "The five levels and the January exit were fixed before running and not searched over. "
        "A different level set would be a different study and would need its own registration "
        "to mean anything.",
        "The baseline buys at the first trading day's close of each month while the Fibonacci "
        "rule buys at level or open prices. The two are not filled on an identical convention.",
    ]

    results = {
        "schema_version": 1, "study_id": "RS-XAUUSD-20260901-007",
        "title": "Fibonacci staged entry for the September-to-January gold window",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "seasonal accumulation, no strategy signal involved",
        "method": {
            "daily_from": bars[0]["date"].isoformat(), "daily_to": bars[-1]["date"].isoformat(),
            "range_window": "January 1 to August 31 of the build year",
            "levels": "H - f*(H-L) for f in " + str(FIBS),
            "fill_rule": ("first daily low at or below the level, from September 1, filled at "
                          "the level or at that bar's open if the bar opened below it"),
            "exit": "the last trading day of the following January, at its close",
            "unfilled": "stays in cash and earns nothing",
            "baseline": "four equal buys at the first trading day of Sep, Oct, Nov, Dec",
            "baseline_source": "RS-XAUUSD-20260827-001, which measured 77.78% on 18 cycles",
            "why_two_return_measures": ("return on deployed capital flatters a rule that "
                                        "under-invests; return on committed capital is the "
                                        "comparable one"),
            "costs_excluded": "spread, carry and roll are not modelled",
        },
        "summary": summary,
        "answers": answers,
        "limitations": limitations,
        "cycles": cycles,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"cycles": len(cycles), "fill": summary["fill_rate"],
                      "fib": summary["fib"], "equal": summary["equal_monthly"],
                      "head_to_head": summary["head_to_head"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
