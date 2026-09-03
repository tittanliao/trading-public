#!/usr/bin/env python3
"""Do S1 and S2 work better in some market states than others?

The goal this comes from is narrow and worth restating: not "find a new signal" but "make
the two signals that already exist fire at better moments". That is a different problem
from alpha discovery and it needs far less evidence, because the sample is the strategies'
own trades rather than every bar, and because it has an obvious control — the same
strategy with no state condition at all.

Sixty-three hypotheses have been closed in this programme without that control being the
default. It is the default here: every state is scored against the strategy's own
unconditional baseline, on the same trades, and the smallest win-rate gap those two groups
could separate is reported beside every comparison.

## Three scales, because which one matters is the open question

- intraday: session, how much of the day's range is already spent, hour block
- swing: Hurst regime, realised volatility, band width
- macro: trend against the 200-day mean, drawdown depth, real-yield and dollar direction

Ten state variables across two strategies is a family of comparisons, and the best cell in
a family of that size looks good by construction. A family permutation is reported per
scale, and the per-cell verdicts are meaningless without it.

## What this study cannot do

It cannot say a gate should be switched on. A state that separates outcomes on 472 trades
is a candidate for prospective testing, not a live rule — and the programme has already
produced one condition that raised a win rate while destroying return.

Usage:
    python3.12 scripts/research/build_xauusd_regime_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import fail_pattern_toolkit as tk  # noqa: E402
import regimes as rg  # noqa: E402
import screen_harness as sh  # noqa: E402
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260825-001"
TAIPEI = timezone(timedelta(hours=8))
SEED = 20260825
PERMUTATIONS = 4000
MIN_CELL = 20

BARS = Path("local-inputs/FX_IDC_XAUUSD, 30_volumn.csv")
DAILY = Path("local-inputs/XAUUSD_1d.csv")
FRED = Path("local-inputs")
STRATEGIES = {
    "S1": ("S1 AweWithBB V3.9", Path("local-inputs/v3.9")
           / "S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv"),
    "S2": ("S2 Hammer V3.2", Path("local-inputs/v3.2")
           / "S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-08-15.csv"),
}


def load_daily() -> pd.DataFrame:
    frame = pd.read_csv(DAILY)
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None).dt.normalize()
    return frame[["date", "close"]]


def load_macro() -> pd.DataFrame | None:
    frames = []
    for series in ("DFII10", "DTWEXBGS"):
        path = FRED / f"{series}.csv"
        if not path.is_file():
            continue
        one = pd.read_csv(path)
        one.columns = ["date", series]
        one["date"] = pd.to_datetime(one["date"])
        # FRED writes '.' on non-trading days; without coercion the column is object dtype
        # and every comparison silently returns False.
        one[series] = pd.to_numeric(one[series], errors="coerce")
        frames.append(one.dropna())
    if not frames:
        return None
    merged = frames[0]
    for other in frames[1:]:
        merged = merged.merge(other, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def resolvable_gap(count: int, other: int, rate: float) -> float | None:
    """Smallest win-rate gap these two groups could separate, in percentage points."""
    if count < 2 or other < 2 or not 0 < rate < 1:
        return None
    sigma = math.sqrt(rate * (1 - rate))
    return round(100 * 2.8 * sigma * math.sqrt(1 / count + 1 / other), 2)


def resolvable_return_gap(count: int, other: int, sigma: float) -> float | None:
    """Smallest difference in mean per-trade return these two groups could separate."""
    if count < 2 or other < 2 or sigma <= 0:
        return None
    return round(2.8 * sigma * math.sqrt(1 / count + 1 / other), 4)


def trades_to_resolve(gap_pct_points: float, rate: float, share: float) -> int | None:
    """How many trades of this strategy would be needed to resolve the gap it is showing.

    A null with a wide bound is not a finding of absence, and this turns that into an
    instruction. Solving the bound formula for the total sample, holding the split at the
    share the state actually fires on:

        gap = 280 x sqrt(p(1-p)) x sqrt(1/(n*s) + 1/(n*(1-s)))

    Reported for every cell whose observed gap sits inside its bound, because those are
    exactly the cells where "collect more trades" is the answer and the only open question
    is how many.
    """
    if not 0 < rate < 1 or not 0 < share < 1 or abs(gap_pct_points) < 1e-9:
        return None
    sigma = math.sqrt(rate * (1 - rate))
    needed = (2.8 * 100 * sigma / abs(gap_pct_points)) ** 2 * (1 / share + 1 / (1 - share))
    return int(math.ceil(needed))


def permutation_p(labels: np.ndarray, wins: np.ndarray, state: str,
                  stream: np.random.Generator) -> float | None:
    """Shuffle the state labels against outcomes; how often is the gap this large?"""
    inside = labels == state
    if inside.sum() < MIN_CELL or (~inside).sum() < MIN_CELL:
        return None
    observed = abs(wins[inside].mean() - wins[~inside].mean())
    shuffled = wins.copy()
    at_least = 0
    for _ in range(PERMUTATIONS):
        stream.shuffle(shuffled)
        if abs(shuffled[inside].mean() - shuffled[~inside].mean()) >= observed:
            at_least += 1
    return round(at_least / PERMUTATIONS, 4)


def measure(name: str, scale: str, labels: pd.Series, trades: pd.DataFrame,
            stream: np.random.Generator) -> dict:
    """One state variable against one strategy's trades, with its own baseline."""
    usable = labels != "unknown"
    wins = trades["win"].to_numpy(dtype=float)
    returns = trades["return_pct"].to_numpy(dtype=float)
    label_values = labels.to_numpy()

    baseline_rate = float(wins.mean() * 100)
    baseline_return = float(returns.mean())

    cells = []
    for state in sorted({v for v in label_values[usable] if isinstance(v, str)}):
        inside = (label_values == state) & usable
        outside = (label_values != state) & usable
        if inside.sum() < MIN_CELL:
            cells.append({"state": state, "n": int(inside.sum()),
                          "verdict": "underpowered",
                          "reason": f"{int(inside.sum())} trades, below the {MIN_CELL} "
                                    "needed to say anything"})
            continue
        rate = float(wins[inside].mean() * 100)
        rest = float(wins[outside].mean() * 100) if outside.sum() else float("nan")
        bound = resolvable_gap(int(inside.sum()), int(outside.sum()),
                               float(wins[usable].mean()))
        p = permutation_p(label_values[usable], wins[usable], state,
                          np.random.default_rng(abs(hash((name, state))) % (2**32)))
        gap = rate - rest

        # Win rate is not the quantity that pays. Every prior candidate in this programme
        # raised one while lowering the other, so the return gap is measured on its own
        # terms and against its own bound rather than inferred from the win rate.
        return_inside = float(returns[inside].mean())
        return_outside = float(returns[outside].mean()) if outside.sum() else float("nan")
        return_bound = resolvable_return_gap(
            int(inside.sum()), int(outside.sum()), float(returns[usable].std(ddof=1)))
        return_gap = return_inside - return_outside

        share = float(inside.sum()) / float(usable.sum())
        cells.append({
            "state": state,
            "n": int(inside.sum()),
            "win_rate_pct": round(rate, 2),
            "rest_win_rate_pct": round(rest, 2),
            "gap_pct_points": round(gap, 2),
            "smallest_resolvable_gap_pct_points": bound,
            "permutation_p": p,
            "mean_return_pct": round(return_inside, 4),
            "rest_mean_return_pct": round(return_outside, 4),
            "return_gap_pct": round(return_gap, 4),
            "smallest_resolvable_return_gap_pct": return_bound,
            "baseline_mean_return_pct": round(baseline_return, 4),
            "share_of_trades_pct": round(100 * share, 2),
            "share_of_return_captured_pct": round(
                100 * float(returns[inside].sum()) / float(returns[usable].sum()), 2)
            if returns[usable].sum() else None,
            "win_rate_and_return_agree": bool(np.sign(gap) == np.sign(return_gap)),
            "trades_needed_to_resolve": trades_to_resolve(
                gap, float(wins[usable].mean()), share),
            "verdict": ("separates" if bound is not None and abs(gap) > bound
                        and p is not None and p <= 0.05 else "no_evidence"),
        })

    measurable = [c for c in cells if c.get("permutation_p") is not None]
    return {
        "state_variable": name,
        "scale": scale,
        "trades_with_a_state": int(usable.sum()),
        "trades_total": int(len(trades)),
        "baseline_win_rate_pct": round(baseline_rate, 2),
        "baseline_mean_return_pct": round(baseline_return, 4),
        "cells": cells,
        "best_p": min((c["permutation_p"] for c in measurable), default=None),
        "cells_that_separate": sum(1 for c in cells if c.get("verdict") == "separates"),
    }


def family_permutation(results: list[dict], stream: np.random.Generator,
                       draws: int = 4000) -> dict:
    """How often does a family this size produce a best cell this strong from noise?"""
    scored = [r["best_p"] for r in results if r["best_p"] is not None]
    if not scored:
        return {"tested": 0, "family_p": None}
    observed = min(scored)
    count = sum(len([c for c in r["cells"] if c.get("permutation_p") is not None])
                for r in results)
    at_least = sum(1 for _ in range(draws) if min(stream.random(count)) <= observed)
    return {
        "cells_tested": count,
        "state_variables": len(results),
        "best_cell_p": observed,
        "family_p": round(at_least / draws, 4),
        "reading": ("the chance that a family of this many cells produces a result this "
                    "strong when no state carries information"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-cell", type=int, default=MIN_CELL)
    args = parser.parse_args()

    bars, _ = tk.load_price_csv(BARS)
    daily = load_daily()
    macro = load_macro()
    states = rg.all_states(bars, daily, macro)

    lookup = bars[["time"]].copy()
    for scale, group in states.items():
        for name, series in group.items():
            lookup[f"{scale}::{name}"] = series.to_numpy()

    by_strategy: dict[str, dict] = {}
    for key, (label, path) in STRATEGIES.items():
        trades = tk.load_trades(path).sort_values("entry_time").reset_index(drop=True)
        joined = pd.merge_asof(trades, lookup.sort_values("time"),
                               left_on="entry_time", right_on="time",
                               direction="backward")
        results = []
        for column in lookup.columns:
            if column == "time":
                continue
            scale, name = column.split("::")
            results.append(measure(name, scale, joined[column].fillna("unknown"),
                                   joined, np.random.default_rng(SEED)))
        by_scale: dict[str, dict] = {}
        for scale in states:
            subset = [r for r in results if r["scale"] == scale]
            by_scale[scale] = {
                "state_variables": subset,
                "family_permutation": family_permutation(
                    subset, np.random.default_rng(SEED + len(scale))),
            }
        by_strategy[key] = {
            "label": label,
            "trades": int(len(trades)),
            "baseline_win_rate_pct": round(float(trades["win"].mean() * 100), 2),
            "baseline_mean_return_pct": round(float(trades["return_pct"].mean()), 4),
            "first_entry": str(trades["entry_time"].min()),
            "last_entry": str(trades["entry_time"].max()),
            "by_scale": by_scale,
        }

    # Cells whose observed gap points the same way on both measures and is larger than half
    # its bound. Not findings — the shortlist of what a longer trade history would settle,
    # each with the trade count that would settle it.
    near_miss = sorted(
        ({"strategy": key, "scale": scale, "state_variable": v["state_variable"],
          "state": c["state"], "n": c["n"],
          "win_rate_pct": c["win_rate_pct"], "gap_pct_points": c["gap_pct_points"],
          "bound_pct_points": c["smallest_resolvable_gap_pct_points"],
          "return_gap_pct": c["return_gap_pct"],
          "return_bound_pct": c["smallest_resolvable_return_gap_pct"],
          "share_of_trades_pct": c["share_of_trades_pct"],
          "share_of_return_captured_pct": c["share_of_return_captured_pct"],
          "permutation_p": c["permutation_p"],
          "trades_needed_to_resolve": c["trades_needed_to_resolve"]}
         for key, data in by_strategy.items()
         for scale, block in data["by_scale"].items()
         for v in block["state_variables"]
         for c in v["cells"]
         if c.get("verdict") == "no_evidence"
         and c.get("win_rate_and_return_agree")
         and c.get("smallest_resolvable_gap_pct_points")
         and abs(c["gap_pct_points"]) > 0.5 * c["smallest_resolvable_gap_pct_points"]),
        key=lambda r: -abs(r["gap_pct_points"]))

    separating = [
        {"strategy": key, "scale": scale, "state_variable": v["state_variable"],
         "state": c["state"], "n": c["n"], "win_rate_pct": c["win_rate_pct"],
         "baseline_pct": c["rest_win_rate_pct"], "gap_pct_points": c["gap_pct_points"],
         "bound_pct_points": c["smallest_resolvable_gap_pct_points"],
         "permutation_p": c["permutation_p"],
         "share_of_trades_pct": c["share_of_trades_pct"],
         "share_of_return_captured_pct": c["share_of_return_captured_pct"]}
        for key, data in by_strategy.items()
        for scale, block in data["by_scale"].items()
        for v in block["state_variables"]
        for c in v["cells"] if c.get("verdict") == "separates"
    ]

    payload = {
        "study_id": STUDY_ID,
        "schema_version": "1.0",
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "title": "Do S1 and S2 work better in some market states than others?",
        "method": {
            "unit": "one existing strategy trade, labelled with the state at its entry",
            "control": ("every state is compared against the same strategy's trades "
                        "outside that state, never against a different sample"),
            "screens": [
                "resolution bound on the win-rate gap",
                f"label permutation, {PERMUTATIONS} draws",
                "family permutation across every cell at that scale",
            ],
            "minimum_cell": args.min_cell,
            "lookahead_guard": (
                "states use trailing windows and expanding-rank quantiles; daily and macro "
                "states are shifted one day so an intraday bar never reads its own day"
            ),
            "seed": SEED,
        },
        "coverage": {
            "bars": int(len(bars)),
            "from": str(bars["time"].iloc[0]),
            "to": str(bars["time"].iloc[-1]),
            "macro_series": [] if macro is None else [c for c in macro.columns if c != "date"],
        },
        "strategies": by_strategy,
        "cells_that_separate": separating,
        "worth_more_trades": {
            "what_this_is": (
                "Cells that point the same way on win rate and on return, and whose gap is "
                "more than half the size this sample could resolve. They are not findings. "
                "Each carries the number of trades that would settle it, which turns a wide "
                "bound from a shrug into an instruction."
            ),
            "cells": near_miss,
        },
        "limitations": [
            "A state that separates outcomes here is a candidate for prospective testing, "
            "not a live rule. This programme has already produced a condition that raised "
            "a win rate while destroying return.",
            "S2 has 157 trades. Split across three states that is roughly 50 per cell, and "
            "the resolution bound on every S2 comparison shows what that costs.",
            "Both strategies are long-only on one instrument across a period in which gold "
            "rose. A state that looks favourable may be describing the rally.",
            "Regime labels are attached at entry only. A state that changes mid-trade is "
            "not modelled.",
        ],
    }

    written = pkg.write_package(
        STUDY_ID, payload,
        market="XAUUSD",
        strategy="none — regime classification over the existing S1 and S2",
        title=payload["title"],
        question=("Not a search for a new signal: a test of whether the two signals that "
                  "already exist perform differently in identifiable market states, at "
                  "three time scales, each against the strategy's own unconditional "
                  "baseline."),
        hypothesis=("If any scale carries a usable state, its cells separate by more than "
                    "the resolution bound and survive a family correction."),
        runner="scripts/research/build_xauusd_regime_sweep.py",
        headline={
            "state_variables": len(lookup.columns) - 1,
            "cells_that_separate": len(separating),
            "cells_worth_more_trades": len(near_miss),
            "s1_trades": by_strategy["S1"]["trades"],
            "s2_trades": by_strategy["S2"]["trades"],
            "s1_baseline_win_rate_pct": by_strategy["S1"]["baseline_win_rate_pct"],
            "s2_baseline_win_rate_pct": by_strategy["S2"]["baseline_win_rate_pct"],
        },
        findings=[],
        card_summary=("Ten state variables at three time scales against S1 and S2, each "
                      "compared with the strategy's own unconditional baseline."),
        limitations=payload["limitations"],
    )

    print(json.dumps({
        "study": STUDY_ID,
        "written": written,
        "state_variables": len(lookup.columns) - 1,
        "cells_that_separate": len(separating),
        "worth_more_trades": len(near_miss),
        "family_p": {
            key: {scale: block["family_permutation"].get("family_p")
                  for scale, block in data["by_scale"].items()}
            for key, data in by_strategy.items()
        },
        "top_near_misses": near_miss[:4],
    }, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
