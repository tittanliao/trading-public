#!/usr/bin/env python3
"""RS-XAUUSD-20260901-001 — re-run the %B finding on the last COMPLETED bar.

Backlog BL-009.

The published runners join with merge_asof(direction="backward") on entry_time. Every entry
timestamp coincides exactly with a bar timestamp (472 of 472), so that match lands on the
bar the fill happened inside, whose close is thirty minutes after the fill.

The change here is one line in each join: shift the lookup's timestamps forward by one bar,
so the value carried at time T is the bar that closed at T-1. Nothing else is touched — same
inputs, same band convention, same screens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

import fail_pattern_toolkit as tk  # noqa: E402

OUT = Path("reproduced")
TRADES = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv")
BARS = Path("local-inputs/xauusd-30m-full.csv")


def lag_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    """Carry each bar's value at the NEXT bar's timestamp.

    After this, a backward asof at entry_time sees the last bar that had actually closed.
    """
    out = lookup.copy().sort_values("time").reset_index(drop=True)
    out["time"] = out["time"].shift(-1)
    return out.dropna(subset=["time"]).reset_index(drop=True)


def enrich(trades: pd.DataFrame, price: pd.DataFrame, lagged: bool) -> pd.DataFrame:
    bb = tk.compute_bb(price)
    lookup = bb[["time", "bb_pct_b", "bb_width"]].sort_values("time").reset_index(drop=True)
    if lagged:
        lookup = lag_lookup(lookup)
    merged = pd.merge_asof(
        trades.sort_values("entry_time"), lookup,
        left_on="entry_time", right_on="time", direction="backward",
    )
    merged["bb_zone"] = merged["bb_pct_b"].apply(tk.bb_zone)
    return merged


def win_rate(frame: pd.DataFrame) -> float:
    return round(100 * (frame["net_pnl_usd"] > 0).mean(), 2) if len(frame) else float("nan")


def resolvable_gap(n1: int, n2: int, rate: float) -> float:
    p = rate / 100
    if n1 < 2 or n2 < 2 or not 0 < p < 1:
        return float("inf")
    return round(100 * 2.802 * (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5, 2)


def permutation_p(frame: pd.DataFrame, observed: float, trials: int = 20000,
                  seed: int = 20260901) -> float:
    import random
    rng = random.Random(seed)
    flags = (frame["bb_zone"] == "above_upper").tolist()
    wins = (frame["net_pnl_usd"] > 0).tolist()
    n_above = sum(flags)
    if n_above == 0 or n_above == len(flags):
        return float("nan")
    hits = 0
    for _ in range(trials):
        rng.shuffle(flags)
        a = [w for w, f in zip(wins, flags) if f]
        b = [w for w, f in zip(wins, flags) if not f]
        gap = 100 * sum(a) / len(a) - 100 * sum(b) / len(b)
        if gap >= observed:
            hits += 1
    return round(hits / trials, 4)


def dose_response(frame: pd.DataFrame) -> list[dict]:
    """Win rate by %B zone, in the published zone order."""
    out = []
    for zone in tk.BB_ZONE_ORDER:
        part = frame[frame["bb_zone"] == zone]
        if len(part):
            out.append({"zone": zone, "n": int(len(part)), "win_rate_pct": win_rate(part)})
    return out


def measure(frame: pd.DataFrame) -> dict:
    valid = frame[frame["bb_zone"] != "unknown"]
    above = valid[valid["bb_zone"] == "above_upper"]
    rest = valid[valid["bb_zone"] != "above_upper"]
    gap = round(win_rate(above) - win_rate(rest), 2)
    return {
        "trades_joined": int(len(valid)),
        "above_upper_n": int(len(above)),
        "above_upper_win_rate_pct": win_rate(above),
        "rest_n": int(len(rest)),
        "rest_win_rate_pct": win_rate(rest),
        "baseline_win_rate_pct": win_rate(valid),
        "win_rate_gap_pct_points": gap,
        "smallest_resolvable_gap_pct_points": resolvable_gap(
            len(above), len(rest), win_rate(valid)),
        "clears_resolution_bound": bool(gap > resolvable_gap(len(above), len(rest), win_rate(valid))),
        "permutation_p": permutation_p(valid, gap),
        "mean_return_above_pct": round(float(above["return_pct"].mean()), 4),
        "mean_return_rest_pct": round(float(rest["return_pct"].mean()), 4),
        "share_of_trades_kept_pct": round(100 * len(above) / len(valid), 2),
        "dose_response": dose_response(valid),
    }


def main() -> None:
    trades = tk.load_trades(TRADES)
    price, _ = tk.load_price_csv(BARS)
    published = enrich(trades, price, lagged=False)
    corrected = enrich(trades, price, lagged=True)

    result = {
        "study": "BL-009 verification",
        "strategy": "S1 AweWithBB V3.9",
        "method": {
            "verifies": ["RS-XAUUSD-20260823-002", "RS-XAUUSD-20260824-006"],
            "change": ("one line in the join: the bb lookup's timestamps are shifted forward "
                       "one bar, so a backward asof at entry_time sees the last bar that had "
                       "actually closed"),
            "everything_else": "same runner, same shared toolkit, same inputs, same screens",
            "permutation_trials": 20000,
            "random_seed": 20260901,
        },
        "entry_times_on_a_bar_timestamp": int(
            trades["entry_time"].isin(set(price["time"])).sum()),
        "entry_times_total": int(len(trades)),
        "as_published_entry_bar": measure(published),
        "corrected_last_completed_bar": measure(corrected),
    }
    a, b = result["as_published_entry_bar"], result["corrected_last_completed_bar"]
    result["change"] = {
        "win_rate_gap_pct_points": round(b["win_rate_gap_pct_points"]
                                         - a["win_rate_gap_pct_points"], 2),
        "above_upper_n": b["above_upper_n"] - a["above_upper_n"],
        "above_upper_win_rate_pct": round(b["above_upper_win_rate_pct"]
                                          - a["above_upper_win_rate_pct"], 2),
        "still_clears_bound": b["clears_resolution_bound"],
        "permutation_p_published": a["permutation_p"],
        "permutation_p_corrected": b["permutation_p"],
    }
    result["schema_version"] = 1
    result["study_id"] = "RS-XAUUSD-20260901-001"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("entry_times_on_a_bar_timestamp", "entry_times_total", "change")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
