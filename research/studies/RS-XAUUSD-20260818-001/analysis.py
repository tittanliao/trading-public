#!/usr/bin/env python3
"""RS-XAUUSD-20260818-001 — trade-path structure for S1 V3.9 and S2 V3.2.

Every S1/S2 study so far has asked the same kind of question: given some state *outside*
the trade — a Macro verdict, a GVZ level, an entry slot, a CFTC regime — is this signal
better or worse? All of them came back at noise. This one asks a different question. It
reconstructs what each trade actually did, bar by bar, from the 30-minute series, and asks
whether the trade's own path carries anything the strategy is not already using.

Four families are tested, in the order a trader would reach for them:

1. `take_profit_overlay` — a full-size take-profit at +X%. This is the standard way to buy
   win rate, and the trade export's own MFE column decides it: if a trade's maximum
   favorable excursion reached X, price touched X while the trade was open, so the TP
   would have filled at or before the real exit.
2. `stop_tightening` — a tighter stop at -Y%, decided by the MAE column the same way.
3. `underwater_time_exit` — leave at bar N if the trade is not in profit there. This is
   the one with a real predictive signal behind it, and the one most likely to be
   mistaken for an edge.
4. `confirmation_entry` — do not enter at the signal; wait one bar and enter only if that
   bar closed in profit, paying the chase.

Two disciplines matter more than the numbers here.

**Conditioning on an outcome is not a rule.** Trade duration splits both strategies
enormously — S1 trades lasting over 27 bars win 72.7% against 43.0% for the shortest
quartile — and it is worth nothing, because a trade that hits its stop early is short *by
definition*. Nothing in this study conditions on a quantity that is only known once the
trade is over. State at bar N is known while the trade is still open, which is what makes
family 3 and 4 implementable and family "duration" not. Duration is computed and reported
under `circular_controls` precisely so the number is on the record as a trap.

**A predictive split is not a tradeable one.** Family 3 produces the largest effect in the
entire study — at bar 1, S1 trades in profit finish at 71.3% against 38.1% for those
underwater, a 33pp gap on 472 trades — and acting on it *loses money* at almost every
setting. The split is real; the information is already inside the strategy's own stop and
target structure. Both facts are reported together, because the first one alone reads as
an edge.

Usage:
    python3.12 -m scripts.research.build_xauusd_path_structure
    python3.12 -m scripts.research.build_xauusd_path_structure --no-charts
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-XAUUSD-20260818-001"
OUTPUT_DIR = Path("reproduced")
BARS_FILE = Path("local-inputs/FX_IDC_XAUUSD, 30_volumn.csv")
TAIPEI = timezone(timedelta(hours=8))

STRATEGIES = {
    "S1": {
        "label": "S1 AweWithBB V3.9",
        "trades": Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv"),
    },
    "S2": {
        "label": "S2 Hammer V3.2",
        "trades": Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-08-15.csv"),
    },
}

TP_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00]
STOP_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00, 1.50]
BAR_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24]
CONFIRM_BARS = [1, 2, 3, 4]
CONFIRM_THRESHOLDS = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
PERMUTATION_TRIALS = 20000
HOLDOUT_FRACTION = 0.30
RANDOM_SEED = 20260818


def stream(label: str) -> random.Random:
    """One independent stream per test, keyed by name.

    A single shared Random made every p-value depend on how much randomness the analyses
    before it happened to consume, so adding revision 3's bootstrap silently moved
    revision 2's permutation p from 0.0023 to 0.0037. Nothing about the data had changed.
    Keying the stream to the test name keeps each result stable as the study grows.
    """
    return random.Random(f"{RANDOM_SEED}:{label}")


# ---------------------------------------------------------------- loading


def load_bars() -> list[tuple[datetime, float, float, float, float]]:
    rows = []
    with BARS_FILE.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                (
                    datetime.fromisoformat(record["time"]).astimezone(TAIPEI),
                    float(record["open"]),
                    float(record["high"]),
                    float(record["low"]),
                    float(record["close"]),
                )
            )
    rows.sort(key=lambda row: row[0])
    return rows


def load_trades(path: Path) -> list[dict]:
    """Pair the export's entry and exit rows into one record per trade.

    A trade still open at export time carries the exit signal `Open` and has no realized
    outcome. It is dropped rather than counted as a flat trade.
    """
    trades: dict[int, dict] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            number = int(record["Trade number"])
            trade = trades.setdefault(number, {})
            stamp = datetime.strptime(record["Date and time"], "%Y-%m-%d %H:%M").replace(
                tzinfo=TAIPEI
            )
            if record["Type"].lower().startswith("entry"):
                trade["entry_at"] = stamp
                trade["entry_price"] = float(record["Price USD"])
                trade["entry_signal"] = record["Signal"]
            else:
                trade["exit_at"] = stamp
                trade["exit_price"] = float(record["Price USD"])
                trade["exit_signal"] = record["Signal"]
            trade["return_pct"] = float(record["Return %"])
            trade["mfe_pct"] = float(record["Favorable excursion %"])
            trade["mae_pct"] = float(record["Adverse excursion %"])
            trade["duration_bars"] = int(record["Duration (bars)"])
    complete = [
        trade
        for _, trade in sorted(trades.items())
        if {"entry_at", "exit_at"} <= set(trade) and trade.get("exit_signal") != "Open"
    ]
    complete.sort(key=lambda trade: trade["entry_at"])
    return complete


def attach_paths(trades: list[dict], bars: list[tuple]) -> list[dict]:
    """Give each trade its running state per bar from entry to exit.

    Only bars at or after the entry bar are used, so every value on a path is information
    the trade had while it was open. `close_pct` at index N is what a live decision at bar
    N would have seen.
    """
    times = [bar[0] for bar in bars]
    for trade in trades:
        start = bisect.bisect_left(times, trade["entry_at"])
        end = bisect.bisect_left(times, trade["exit_at"])
        if start >= len(bars) or end <= start:
            trade["path"] = []
            continue
        entry = trade["entry_price"]
        running_high, running_low = -math.inf, math.inf
        path = []
        for index in range(start, min(end + 1, len(bars))):
            _, _, high, low, close = bars[index]
            running_high = max(running_high, high)
            running_low = min(running_low, low)
            path.append(
                {
                    "high_pct": 100 * (high - entry) / entry,
                    "low_pct": 100 * (low - entry) / entry,
                    "close_pct": 100 * (close - entry) / entry,
                    "mfe_pct": 100 * (running_high - entry) / entry,
                    "mae_pct": 100 * (running_low - entry) / entry,
                }
            )
        trade["path"] = path
    return trades


# ---------------------------------------------------------------- statistics


def wilson(wins: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959964
    p = wins / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(100 * (centre - margin), 2), round(100 * (centre + margin), 2)]


def min_detectable_pp(n: int, baseline: float) -> float | None:
    """Smallest win-rate gap two groups of this size could resolve.

    Deliberately conservative: it assumes both groups are size n, while a subgroup is
    usually compared against a larger remainder that carries more power. It will call a
    real effect unresolvable before it calls noise an effect, which is the correct bias
    for a study whose whole purpose is refusing to over-read small samples.
    """
    if not n:
        return None
    z_alpha, z_beta = 1.959964, 0.841621
    low, high = 0.0001, 0.90
    for _ in range(90):
        mid = (low + high) / 2
        p1, p2 = baseline, min(baseline + mid, 0.999)
        pooled = (p1 + p2) / 2
        needed = (
            z_alpha * math.sqrt(2 * pooled * (1 - pooled))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2 / (p1 - p2) ** 2
        low, high = (mid, high) if needed > n else (low, mid)
    value = (low + high) / 2
    return None if value > 0.85 else round(value * 100, 1)


def metrics(returns: list[float]) -> dict:
    n = len(returns)
    if not n:
        return {"n": 0}
    wins = [value for value in returns if value > 0]
    gross_loss = -sum(value for value in returns if value <= 0)
    return {
        "n": n,
        "wins": len(wins),
        "win_rate_pct": round(100 * len(wins) / n, 2),
        "win_rate_ci95_pct": wilson(len(wins), n),
        "profit_factor": round(sum(wins) / gross_loss, 3) if gross_loss else None,
        "avg_return_pct": round(statistics.fmean(returns), 4),
        "total_return_pct": round(sum(returns), 2),
    }


def selection_permutation(pool: list[float], selected_mean: float, k: int,
                          rng: random.Random) -> dict:
    """How often does picking k at random from the same pool do this well?

    The filter's claim is that it selects better-than-average trades. The matching null is
    therefore a random selection of the same size from the same modified population — not a
    shuffle of the original returns, which would also absorb the cost of the modification
    and make a useless filter look successful.
    """
    if k < 5 or k >= len(pool):
        return {"applicable": False, "reason": "selection too small or not a subset"}
    hits = sum(
        1 for _ in range(PERMUTATION_TRIALS)
        if statistics.fmean(rng.sample(pool, k)) >= selected_mean
    )
    p_value = hits / PERMUTATION_TRIALS
    return {
        "applicable": True,
        "trials": PERMUTATION_TRIALS,
        "selected_n": k,
        "pool_n": len(pool),
        "p_selection_at_least_this_good": round(p_value, 4),
        "separable": p_value < 0.05,
    }


# ---------------------------------------------------------------- families


def take_profit_overlay(trades: list[dict], baseline: dict) -> dict:
    """A full-size take-profit at +X%, decided by each trade's realized MFE."""
    rows = {}
    for level in TP_GRID:
        simulated = [
            level if trade["mfe_pct"] >= level else trade["return_pct"] for trade in trades
        ]
        result = metrics(simulated)
        result["win_rate_delta_pp"] = round(
            result["win_rate_pct"] - baseline["win_rate_pct"], 2
        )
        result["total_return_delta_pct"] = round(
            result["total_return_pct"] - baseline["total_return_pct"], 2
        )
        rows[f"tp_{level:.2f}"] = result
    improved = [key for key, row in rows.items() if row["total_return_delta_pct"] > 0]
    return {
        "levels": rows,
        "levels_improving_total_return": improved,
        "best_win_rate_pct": max(row["win_rate_pct"] for row in rows.values()),
        "note": (
            "Win rate can only rise here: a take-profit never turns a winner into a loser. "
            "That is exactly why win rate alone cannot judge this family, and why the "
            "total return column is the one that decides it."
        ),
    }


def stop_tightening(trades: list[dict], baseline: dict) -> dict:
    """A tighter stop at -Y%, decided by each trade's realized MAE.

    Levels at or beyond the strategy's existing stop are not tightenings — they reproduce
    the current behaviour and drift back to the baseline. They are kept in the output as a
    control, because a model that did not converge there would be wrong, but they are
    flagged so a near-baseline number is never read as an improvement.
    """
    deepest = min(trade["mae_pct"] for trade in trades)
    stopped = [t for t in trades if t.get("exit_signal", "").endswith("SL")]
    existing_stop = (
        statistics.median(
            100 * (t["entry_price"] - t["exit_price"]) / t["entry_price"] for t in stopped
        )
        if len(stopped) >= 20 else None
    )
    rows = {}
    for level in STOP_GRID:
        if level > abs(deepest):
            continue
        simulated = [
            -level if trade["mae_pct"] <= -level else trade["return_pct"] for trade in trades
        ]
        result = metrics(simulated)
        result["total_return_delta_pct"] = round(
            result["total_return_pct"] - baseline["total_return_pct"], 2
        )
        result["is_a_tightening"] = (
            existing_stop is None or level < existing_stop - 0.02
        )
        rows[f"stop_{level:.2f}"] = result
    improving = [
        key for key, row in rows.items()
        if row["total_return_delta_pct"] > 0 and row["is_a_tightening"]
    ]
    return {
        "levels": rows,
        "deepest_observed_mae_pct": round(deepest, 3),
        "existing_stop_pct": round(existing_stop, 3) if existing_stop else None,
        "levels_improving_total_return": improving,
        "levels_at_or_beyond_existing_stop": [
            key for key, row in rows.items() if not row["is_a_tightening"]
        ],
        "note": (
            "Only levels inside the existing stop count. A level at or beyond it is the "
            "identity case and returns the baseline, which is a check on the model rather "
            "than a result."
        ),
    }



TRAIL_GRID = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
ARM_GRID = [0.0, 0.3, 0.5, 1.0]
BREAKEVEN_GRID = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5]


def trailing_and_breakeven(trades: list[dict], baseline: dict) -> dict:
    """Trailing stops and a stop moved to entry — the family the first revision missed.

    Revision 1 concluded the exit side was closed after testing a fixed take-profit, a
    tighter fixed stop, and a time exit. That claim was too broad for what had been tested:
    none of those three let a winner keep running while still cutting a reversal, which is
    the whole point of a trailing stop. The claim is only honest with this family in it.

    A trailing exit is decided on bar lows against the running peak, so unlike the
    take-profit family it cannot be settled from the export's MFE column alone.

    One reading note: a stop moved to entry exits at exactly 0, which the win-rate
    definition here (return > 0) counts as a non-win. That deflates the win-rate column for
    this family specifically. Total return is unaffected by the convention and is what
    decides the family.
    """
    def trailing(trade: dict, trail: float, arm: float) -> float:
        entry_ok = False
        peak = -math.inf
        for bar in trade["path"]:
            peak = max(peak, bar["high_pct"])
            if not entry_ok and peak >= arm:
                entry_ok = True
            if entry_ok:
                # The give-back is a percentage of price at the peak, not of the entry, so
                # the stop is computed on the peak level rather than subtracted from it.
                stop = ((100 + peak) * (1 - trail / 100)) - 100
                if bar["low_pct"] <= stop:
                    return stop
        return trade["return_pct"]

    def breakeven(trade: dict, arm: float) -> float:
        armed = False
        for bar in trade["path"]:
            if not armed:
                if bar["high_pct"] >= arm:
                    armed = True
                continue
            if bar["low_pct"] <= 0:
                return 0.0
        return trade["return_pct"]

    trail_rows = {}
    for trail in TRAIL_GRID:
        for arm in ARM_GRID:
            result = metrics([trailing(trade, trail, arm) for trade in trades])
            result["total_return_delta_pct"] = round(
                result["total_return_pct"] - baseline["total_return_pct"], 2
            )
            trail_rows[f"trail_{trail:.2f}_arm_{arm:.2f}"] = result

    breakeven_rows = {}
    for arm in BREAKEVEN_GRID:
        result = metrics([breakeven(trade, arm) for trade in trades])
        result["total_return_delta_pct"] = round(
            result["total_return_pct"] - baseline["total_return_pct"], 2
        )
        result["never_armed"] = result["total_return_delta_pct"] == 0.0
        breakeven_rows[f"breakeven_arm_{arm:.2f}"] = result

    improving = [
        key for rows in (trail_rows, breakeven_rows)
        for key, row in rows.items()
        if row["total_return_delta_pct"] > 0 and not row.get("never_armed")
    ]
    best = max(trail_rows.items(), key=lambda item: item[1]["total_return_delta_pct"])
    return {
        "trailing_stop": trail_rows,
        "breakeven_move": breakeven_rows,
        "variants_tested": len(trail_rows) + len(breakeven_rows),
        "variants_improving_total_return": improving,
        "best_trailing": {"setting": best[0],
                          "total_return_delta_pct": best[1]["total_return_delta_pct"]},
    }


def underwater_split(trades: list[dict], baseline: dict) -> dict:
    """Does the trade's state at bar N predict its final outcome, and can it be traded?

    Reported as two things that must be read together: the predictive split, which is
    large, and the simulated rule, which is not profitable.
    """
    splits = {}
    for bar in BAR_GRID:
        alive = [trade for trade in trades if len(trade["path"]) > bar]
        ahead = [t["return_pct"] for t in alive if t["path"][bar]["close_pct"] > 0]
        behind = [t["return_pct"] for t in alive if t["path"][bar]["close_pct"] <= 0]
        if len(ahead) < 10 or len(behind) < 10:
            continue
        up, down = metrics(ahead), metrics(behind)
        gap = up["win_rate_pct"] - down["win_rate_pct"]
        needed = min_detectable_pp(min(up["n"], down["n"]), baseline["win_rate_pct"] / 100)
        splits[f"bar_{bar}"] = {
            "still_open_n": len(alive),
            "in_profit": up,
            "underwater": down,
            "win_rate_gap_pp": round(gap, 2),
            "min_detectable_pp": needed,
            "separable": needed is not None and abs(gap) >= needed,
        }

    rules = {}
    for bar in BAR_GRID[:6]:
        for threshold in (0.0, 0.05, 0.10, 0.20):
            simulated = []
            for trade in trades:
                path = trade["path"]
                if len(path) > bar and path[bar]["close_pct"] <= -threshold:
                    simulated.append(path[bar]["close_pct"])
                else:
                    simulated.append(trade["return_pct"])
            result = metrics(simulated)
            result["total_return_delta_pct"] = round(
                result["total_return_pct"] - baseline["total_return_pct"], 2
            )
            rules[f"exit_bar{bar}_below_{threshold:.2f}"] = result

    improving = [key for key, row in rules.items() if row["total_return_delta_pct"] > 0]
    return {
        "predictive_split": splits,
        "simulated_rule": rules,
        "rules_tested": len(rules),
        "rules_improving_total_return": improving,
        "note": (
            "The split is real and large; the rule built on it is not profitable. Trades "
            "that are underwater at bar N still recover often enough, and far enough, that "
            "leaving early forfeits more than the tighter loss saves. A cell or two out of "
            f"{len(rules)} landing positive is what a sweep this wide produces on its own."
        ),
    }


def confirmation_entry(trades: list[dict], baseline: dict, rng: random.Random) -> dict:
    """Skip the signal bar; enter one bar later only if that bar closed in profit.

    Unlike the underwater rule, this never takes the weak cohort at all. It pays for that
    with a worse entry price, so the family is decided by whether the filter's gain
    survives the chase cost — which is why `delay_only` is computed: it isolates the cost
    of waiting from the benefit of selecting.
    """
    usable = [trade for trade in trades if len(trade["path"]) > max(CONFIRM_BARS)]
    if len(usable) < 40:
        return {"applicable": False, "reason": "too few trades with a long enough path"}

    def delayed_return(trade: dict, bar: int) -> float:
        price = trade["entry_price"] * (1 + trade["path"][bar]["close_pct"] / 100)
        return 100 * (trade["exit_price"] - price) / price

    original = metrics([trade["return_pct"] for trade in usable])
    by_bar = {}
    for bar in CONFIRM_BARS:
        pool = [delayed_return(trade, bar) for trade in usable]
        selected = [
            delayed_return(trade, bar)
            for trade in usable
            if trade["path"][bar]["close_pct"] > 0
        ]
        confirmed = metrics(selected)
        by_bar[f"bar_{bar}"] = {
            "delay_only": metrics(pool),
            "confirmed": confirmed,
            "avg_return_delta_pct": round(
                confirmed["avg_return_pct"] - original["avg_return_pct"], 4
            ),
            "permutation_test": selection_permutation(
                pool, confirmed["avg_return_pct"], confirmed["n"], rng
            ),
        }

    best_bar = 1
    pool = [delayed_return(trade, best_bar) for trade in usable]
    thresholds = {}
    for threshold in CONFIRM_THRESHOLDS:
        selected = [
            delayed_return(trade, best_bar)
            for trade in usable
            if trade["path"][best_bar]["close_pct"] > threshold
        ]
        if len(selected) < 15:
            continue
        result = metrics(selected)
        result["permutation_test"] = selection_permutation(
            pool, result["avg_return_pct"], result["n"], rng
        )
        thresholds[f"close_above_{threshold:+.2f}"] = result

    chase = statistics.fmean(
        trade["path"][best_bar]["close_pct"]
        for trade in usable
        if trade["path"][best_bar]["close_pct"] > 0
    )
    winner_mfe = statistics.median(
        trade["path"][-1]["mfe_pct"] for trade in usable if trade["return_pct"] > 0
    )

    by_year = {}
    for year in sorted({trade["entry_at"].year for trade in usable}):
        subset = [trade for trade in usable if trade["entry_at"].year == year]
        selected = [
            delayed_return(trade, best_bar)
            for trade in subset
            if trade["path"][best_bar]["close_pct"] > 0
        ]
        if not selected:
            continue
        by_year[str(year)] = {
            "original": metrics([trade["return_pct"] for trade in subset]),
            "confirmed": metrics(selected),
        }

    cut = int(len(usable) * (1 - HOLDOUT_FRACTION))
    holdout = {}
    for label, subset in (("in_sample", usable[:cut]), ("held_out", usable[cut:])):
        subset_pool = [delayed_return(trade, best_bar) for trade in subset]
        selected = [
            delayed_return(trade, best_bar)
            for trade in subset
            if trade["path"][best_bar]["close_pct"] > 0
        ]
        result = metrics(selected)
        holdout[label] = {
            "original": metrics([trade["return_pct"] for trade in subset]),
            "confirmed": result,
            "permutation_test": selection_permutation(
                subset_pool, result["avg_return_pct"], result["n"], rng
            ),
        }

    return {
        "applicable": True,
        "original": original,
        "by_bar": by_bar,
        "threshold_sweep_bar1": thresholds,
        "by_year": by_year,
        "chronological_holdout": holdout,
        "chase_cost": {
            "avg_chase_pct": round(chase, 3),
            "median_winner_mfe_pct": round(winner_mfe, 3),
            "chase_share_of_winner_move_pct": round(100 * chase / winner_mfe, 1),
        },
    }


def risk_adjusted(trades: list[dict]) -> dict:
    """Express the confirmation result in R, using the distance each cohort had to its stop.

    The percentage return understates the cost. Waiting a bar moves the entry up while the
    stop stays where it was, so a confirmed trade risks slightly *more* per unit, not less.
    Without this the family looks better than it is.
    """
    stopped = [trade for trade in trades if trade.get("exit_signal", "").endswith("SL")]
    if len(stopped) < 20:
        return {"applicable": False, "reason": "too few stop exits to measure risk"}
    original_risk = statistics.median(
        100 * (trade["entry_price"] - trade["exit_price"]) / trade["entry_price"]
        for trade in stopped
    )
    confirmed = [
        trade for trade in trades
        if len(trade["path"]) > 1 and trade["path"][1]["close_pct"] > 0
    ]
    confirmed_stops = [t for t in confirmed if t.get("exit_signal", "").endswith("SL")]
    if len(confirmed_stops) < 10:
        return {"applicable": False, "reason": "too few confirmed stop exits"}
    confirmed_risk = statistics.median(
        100
        * (trade["entry_price"] * (1 + trade["path"][1]["close_pct"] / 100) - trade["exit_price"])
        / (trade["entry_price"] * (1 + trade["path"][1]["close_pct"] / 100))
        for trade in confirmed_stops
    )
    original_r = [trade["return_pct"] / original_risk for trade in trades]
    confirmed_r = []
    for trade in confirmed:
        price = trade["entry_price"] * (1 + trade["path"][1]["close_pct"] / 100)
        confirmed_r.append((100 * (trade["exit_price"] - price) / price) / confirmed_risk)
    return {
        "applicable": True,
        "original_risk_pct": round(original_risk, 3),
        "confirmed_risk_pct": round(confirmed_risk, 3),
        "original_avg_r": round(statistics.fmean(original_r), 3),
        "confirmed_avg_r": round(statistics.fmean(confirmed_r), 3),
        "avg_r_improvement_pct": round(
            100 * (statistics.fmean(confirmed_r) / statistics.fmean(original_r) - 1), 1
        ),
        "original_total_r": round(sum(original_r), 1),
        "confirmed_total_r": round(sum(confirmed_r), 1),
        "confirmed_total_r_share_pct": round(100 * sum(confirmed_r) / sum(original_r), 1),
        "trade_count_share_pct": round(100 * len(confirmed_r) / len(original_r), 1),
        "note": (
            "The stop is a percentage from the original entry, not a fixed price, so a "
            "later entry sits further from it in percentage terms. Confirmed trades risk "
            "more per trade, which is why the R improvement is smaller than the raw "
            "return improvement."
        ),
    }


NOMINAL_STOP_PCT = {"S1": 0.005, "S2": 0.010}
SCALE_IN_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
BOOTSTRAP_TRIALS = 4000


def max_drawdown(returns: list[float]) -> float:
    equity = peak = drawdown = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def risk_alignment(trades: list[dict], strategy: str, bars: list[tuple],
                   rng: random.Random) -> dict:
    """How the confirmation filter compares once risk is actually held equal.

    Revision 2 reported the filter as producing 77% of the total R on 55% of the trades.
    That used one convention — fixed risk per trade — which charges the filter for taking
    fewer trades while giving it no credit for the risk it does not deploy. It is a choice,
    not a fact, and it is not the only defensible one.

    Four alignments are computed. They disagree by a factor of nearly three, so the study
    has to say which one it trusts:

    - `fixed_per_trade_risk` — the revision 2 convention.
    - `equal_total_risk_deployed` — same sum of risk over the period.
    - `equal_period_volatility` — same total variance over the period, the standard
      risk-adjusted comparison. This is the one to read.
    - `equal_max_drawdown` — the most flattering and the least trustworthy. Max drawdown is
      a single realization from one path, and the bootstrap interval on the scaling factor
      it implies is wide enough to be useless. It is computed with that interval attached
      so the number cannot be quoted without it.

    A scale-in variant is also tested: rather than skipping unconfirmed trades, enter every
    signal and add to the confirmed ones. It is the only version of this idea that could be
    decided at entry, since confirmation is not known until a bar later.
    """
    stop_pct = NOMINAL_STOP_PCT.get(strategy)
    if stop_pct is None:
        return {"applicable": False, "reason": "no nominal stop for this strategy"}

    times = [bar[0] for bar in bars]
    usable = []
    for trade in trades:
        index = bisect.bisect_left(times, trade["entry_at"])
        if index >= len(bars) or bars[index][0] != trade["entry_at"] or index + 1 >= len(bars):
            continue
        if bisect.bisect_left(times, trade["exit_at"]) <= index + 1:
            continue
        trade["bar1_close"] = bars[index + 1][4]
        usable.append(trade)
    if len(usable) < 40:
        return {"applicable": False, "reason": "too few trades survive to bar 1"}

    stopped = [t for t in usable if t.get("exit_signal", "").endswith("SL")]
    matching = sum(
        1 for t in stopped
        if abs((t["entry_price"] - t["exit_price"]) / t["entry_price"] - stop_pct) < 0.0005
    )

    baseline_r = [
        (t["exit_price"] - t["entry_price"]) / (t["entry_price"] * stop_pct) for t in usable
    ]
    confirmed = [t for t in usable if t["bar1_close"] > t["entry_price"]]
    filtered_r = [
        (t["exit_price"] - t["bar1_close"]) / (t["bar1_close"] - t["entry_price"] * (1 - stop_pct))
        for t in confirmed
    ]

    scale_in = {}
    for factor in SCALE_IN_GRID:
        values = []
        for trade in usable:
            entry = trade["entry_price"]
            stop = entry * (1 - stop_pct)
            peak_entry = trade["bar1_close"]
            exit_price = trade["exit_price"]
            if peak_entry <= entry:
                values.append((exit_price - entry) / (entry * stop_pct))
                continue
            risk = (entry - stop) + factor * (peak_entry - stop)
            values.append(((exit_price - entry) + factor * (exit_price - peak_entry)) / risk)
        result = metrics(values)
        result["total_r"] = round(sum(values), 2)
        result["avg_r"] = round(statistics.fmean(values), 4)
        result["total_r_delta"] = round(sum(values) - sum(baseline_r), 2)
        scale_in[f"add_{factor:.2f}x"] = result

    base_total, filter_total = sum(baseline_r), sum(filtered_r)
    base_sd = statistics.pstdev(baseline_r)
    filter_sd = statistics.pstdev(filtered_r)
    base_dd, filter_dd = max_drawdown(baseline_r), max_drawdown(filtered_r)

    variance_scale = math.sqrt(
        (len(baseline_r) * base_sd ** 2) / (len(filtered_r) * filter_sd ** 2)
    )
    alignments = {
        "fixed_per_trade_risk": {
            "scale": 1.0,
            "filter_total_r": round(filter_total, 2),
            "share_of_baseline_pct": round(100 * filter_total / base_total, 1),
        },
        "equal_total_risk_deployed": {
            "scale": round(len(baseline_r) / len(filtered_r), 3),
            "filter_total_r": round(filter_total * len(baseline_r) / len(filtered_r), 2),
            "share_of_baseline_pct": round(
                100 * filter_total * len(baseline_r) / len(filtered_r) / base_total, 1
            ),
        },
        "equal_period_volatility": {
            "scale": round(variance_scale, 3),
            "filter_total_r": round(filter_total * variance_scale, 2),
            "share_of_baseline_pct": round(100 * filter_total * variance_scale / base_total, 1),
            "preferred": True,
        },
    }

    bootstrap = sorted(
        max_drawdown(rng.choices(baseline_r, k=len(baseline_r)))
        / max(max_drawdown(rng.choices(filtered_r, k=len(filtered_r))), 1e-9)
        for _ in range(BOOTSTRAP_TRIALS)
    )
    low = bootstrap[int(0.05 * len(bootstrap))]
    high = bootstrap[int(0.95 * len(bootstrap))]
    alignments["equal_max_drawdown"] = {
        "scale": round(base_dd / filter_dd, 3),
        "filter_total_r": round(filter_total * base_dd / filter_dd, 2),
        "share_of_baseline_pct": round(100 * filter_total * base_dd / filter_dd / base_total, 1),
        "scale_bootstrap_90pct": [round(low, 2), round(high, 2)],
        "share_bootstrap_90pct": [
            round(100 * filter_total * low / base_total, 1),
            round(100 * filter_total * high / base_total, 1),
        ],
        "trustworthy": False,
        "why_not": (
            "Max drawdown is one realization from one ordering. The bootstrap interval on "
            "the implied scaling factor spans a factor of four, so this alignment cannot "
            "distinguish an improvement from a loss and must not be quoted alone."
        ),
    }

    return {
        "applicable": True,
        "nominal_stop_pct": stop_pct,
        "stop_exits_matching_nominal": f"{matching}/{len(stopped)}",
        "baseline": {"n": len(baseline_r), "total_r": round(base_total, 2),
                     "avg_r": round(statistics.fmean(baseline_r), 4),
                     "sd_r": round(base_sd, 3), "max_drawdown_r": round(base_dd, 2)},
        "confirmation_filter": {"n": len(filtered_r), "total_r": round(filter_total, 2),
                                "avg_r": round(statistics.fmean(filtered_r), 4),
                                "sd_r": round(filter_sd, 3),
                                "max_drawdown_r": round(filter_dd, 2)},
        "risk_alignments": alignments,
        "scale_in_instead_of_filtering": scale_in,
        "scale_in_improving_total_r": [
            key for key, row in scale_in.items() if row["total_r_delta"] > 0
        ],
    }


def circular_controls(trades: list[dict], baseline: dict) -> dict:
    """Splits that look strong and cannot be traded, kept so they are on the record.

    Duration and full-trade MAE are only known once the trade has closed. A trade that
    stopped out early is short and deep by construction. These are computed to document
    the size of the trap, not as candidates.
    """
    durations = sorted(trade["duration_bars"] for trade in trades)
    quartiles = [durations[len(durations) * i // 4] for i in (1, 2, 3)]
    buckets: dict[str, list[float]] = {}
    for trade in trades:
        bars = trade["duration_bars"]
        if bars <= quartiles[0]:
            key = f"q1_le_{quartiles[0]}"
        elif bars <= quartiles[1]:
            key = f"q2_le_{quartiles[1]}"
        elif bars <= quartiles[2]:
            key = f"q3_le_{quartiles[2]}"
        else:
            key = f"q4_gt_{quartiles[2]}"
        buckets.setdefault(key, []).append(trade["return_pct"])
    rows = {key: metrics(values) for key, values in sorted(buckets.items())}
    spread = max(row["win_rate_pct"] for row in rows.values()) - min(
        row["win_rate_pct"] for row in rows.values()
    )
    return {
        "by_duration_quartile": rows,
        "duration_win_rate_spread_pp": round(spread, 2),
        "baseline_win_rate_pct": baseline["win_rate_pct"],
        "tradeable": False,
        "why_not": (
            "Duration is an outcome. A trade that reaches its stop quickly is short "
            "because it lost, so the split reproduces the outcome it claims to predict. "
            "Nothing here can be acted on at entry or while the trade is open."
        ),
    }


def sequence_controls(trades: list[dict], baseline: dict) -> dict:
    """Order-of-arrival effects: previous outcome, streaks, and signal density."""
    base = baseline["win_rate_pct"] / 100

    def summarise(groups: dict[str, list[float]]) -> dict:
        out = {}
        for key, values in groups.items():
            if not values:
                continue
            row = metrics(values)
            needed = min_detectable_pp(row["n"], base)
            row["min_detectable_pp"] = needed
            row["separable"] = (
                needed is not None
                and abs(row["win_rate_pct"] - baseline["win_rate_pct"]) >= needed
            )
            out[key] = row
        return out

    previous: dict[str, list[float]] = {"after_win": [], "after_loss": []}
    for index in range(1, len(trades)):
        key = "after_win" if trades[index - 1]["return_pct"] > 0 else "after_loss"
        previous[key].append(trades[index]["return_pct"])

    streak: dict[str, list[float]] = {"after_two_wins": [], "after_two_losses": [], "mixed": []}
    for index in range(2, len(trades)):
        first = trades[index - 2]["return_pct"] > 0
        second = trades[index - 1]["return_pct"] > 0
        key = (
            "after_two_wins" if first and second
            else "after_two_losses" if not first and not second
            else "mixed"
        )
        streak[key].append(trades[index]["return_pct"])

    gaps = [
        (trades[i]["entry_at"] - trades[i - 1]["entry_at"]).total_seconds() / 3600
        for i in range(1, len(trades))
    ]
    median_gap = statistics.median(gaps)
    density: dict[str, list[float]] = {"clustered": [], "isolated": []}
    for index in range(1, len(trades)):
        key = "clustered" if gaps[index - 1] <= median_gap else "isolated"
        density[key].append(trades[index]["return_pct"])

    return {
        "after_previous_outcome": summarise(previous),
        "after_two_in_a_row": summarise(streak),
        "by_signal_density": summarise(density),
        "median_gap_hours": round(median_gap, 1),
    }


# ---------------------------------------------------------------- charts


def write_charts(results: dict, directory: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    charts = []

    for name, block in results["strategies"].items():
        overlay = block["take_profit_overlay"]["levels"]
        levels = [float(key.split("_")[1]) for key in overlay]
        win_rates = [row["win_rate_pct"] for row in overlay.values()]
        totals = [row["total_return_pct"] for row in overlay.values()]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(levels, win_rates, marker="o", label="win rate %")
        axis.axhline(block["baseline"]["win_rate_pct"], linestyle="--", linewidth=1,
                     label="baseline win rate")
        axis.set_xlabel("take-profit level (%)")
        axis.set_ylabel("win rate (%)")
        twin = axis.twinx()
        twin.plot(levels, totals, marker="s", color="tab:red", label="total return %")
        twin.axhline(block["baseline"]["total_return_pct"], linestyle=":", color="tab:red",
                     linewidth=1)
        twin.set_ylabel("total return (%)")
        axis.set_title(f"{name}: a take-profit buys win rate and sells return")
        lines = axis.get_lines() + twin.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="center right")
        figure.tight_layout()
        filename = f"{name.lower()}_take_profit_tradeoff.png"
        figure.savefig(directory / filename, dpi=140)
        plt.close(figure)
        charts.append({"file": filename,
                       "caption": f"{name}: win rate against total return across take-profit levels"})

        splits = block["underwater_time_exit"]["predictive_split"]
        bars = [int(key.split("_")[1]) for key in splits]
        ahead = [row["in_profit"]["win_rate_pct"] for row in splits.values()]
        behind = [row["underwater"]["win_rate_pct"] for row in splits.values()]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(bars, ahead, marker="o", label="in profit at bar N")
        axis.plot(bars, behind, marker="o", label="underwater at bar N")
        axis.axhline(block["baseline"]["win_rate_pct"], linestyle="--", linewidth=1,
                     color="grey", label="baseline")
        axis.set_xlabel("bars after entry")
        axis.set_ylabel("final win rate (%)")
        axis.set_title(f"{name}: a large split that does not pay to trade")
        axis.legend(fontsize=8)
        figure.tight_layout()
        filename = f"{name.lower()}_underwater_split.png"
        figure.savefig(directory / filename, dpi=140)
        plt.close(figure)
        charts.append({"file": filename,
                       "caption": f"{name}: final win rate by state at bar N"})

        confirm = block["confirmation_entry"]
        if confirm.get("applicable"):
            sweep = confirm["threshold_sweep_bar1"]
            thresholds = [float(key.split("_")[-1]) for key in sweep]
            averages = [row["avg_return_pct"] for row in sweep.values()]
            counts = [row["n"] for row in sweep.values()]
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.plot(thresholds, averages, marker="o", color="tab:green",
                      label="avg return per trade %")
            axis.axhline(confirm["original"]["avg_return_pct"], linestyle="--", linewidth=1,
                         label="no confirmation")
            axis.set_xlabel("required close at bar 1 (%)")
            axis.set_ylabel("avg return per trade (%)")
            twin = axis.twinx()
            twin.bar(thresholds, counts, width=0.03, alpha=0.25, color="grey")
            twin.set_ylabel("trades taken")
            axis.set_title(f"{name}: confirmation threshold is a gradient, not a spike")
            axis.legend(fontsize=8)
            figure.tight_layout()
            filename = f"{name.lower()}_confirmation_gradient.png"
            figure.savefig(directory / filename, dpi=140)
            plt.close(figure)
            charts.append({"file": filename,
                           "caption": f"{name}: per-trade return by confirmation threshold"})

    return charts


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    bars = load_bars()
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(tz=TAIPEI).isoformat(timespec="seconds"),
        "strategy": "S1 AweWithBB V3.9; S2 Hammer V3.2",
        "method": {
            "bar_source": "30-minute FX_IDC:XAUUSD export",
            "bar_count": len(bars),
            "path_reconstruction": (
                "running high/low/close from the entry bar to the exit bar; every value on "
                "a path was available while the trade was open"
            ),
            "permutation_trials": PERMUTATION_TRIALS,
            "holdout_fraction": HOLDOUT_FRACTION,
            "random_seed": RANDOM_SEED,
            "timezone": "Asia/Taipei",
        },
        "strategies": {},
    }

    for name, config in STRATEGIES.items():
        trades = attach_paths(load_trades(config["trades"]), bars)
        trades = [trade for trade in trades if trade["path"]]
        baseline = metrics([trade["return_pct"] for trade in trades])
        results["strategies"][name] = {
            "label": config["label"],
            "trades": len(trades),
            "first_entry": trades[0]["entry_at"].isoformat(),
            "last_entry": trades[-1]["entry_at"].isoformat(),
            "baseline": baseline,
            "take_profit_overlay": take_profit_overlay(trades, baseline),
            "stop_tightening": stop_tightening(trades, baseline),
            "underwater_time_exit": underwater_split(trades, baseline),
            "trailing_and_breakeven": trailing_and_breakeven(trades, baseline),
            "confirmation_entry": confirmation_entry(
                trades, baseline, stream(f"confirmation:{name}")
            ),
            "risk_adjusted_confirmation": risk_adjusted(trades),
            "risk_alignment": risk_alignment(
                trades, name, bars, stream(f"risk_alignment:{name}")
            ),
            "circular_controls": circular_controls(trades, baseline),
            "sequence_controls": sequence_controls(trades, baseline),
        }

    results["limitations"] = [
        "Every counterfactual keeps the original exit price and time. A live confirmation "
        "entry would re-anchor its stop and target to the later entry, which changes which "
        "trades stop out; that second-order effect is not modelled here.",
        "Take-profit and stop counterfactuals are decided by MFE and MAE measured on "
        "30-minute bars. Where a bar contains both the stop and the target, the order "
        "within the bar is unknown and is resolved in favour of the modelled level.",
        "The S2 confirmation result rests on 91 confirmed trades in total and 25 in the "
        "held-out period. The held-out permutation test does not reach significance on its "
        "own; only the direction is confirmed there.",
        "Both strategies are long-only over a period in which XAUUSD trended strongly "
        "upward. A confirmation filter that selects continuation may be reading that trend "
        "rather than anything about the signal.",
        "No result changes formal S1 or S2 logic, live risk, or an active entry checklist.",
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_charts:
        results["charts"] = write_charts(results, args.output / "charts")

    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "study_id": STUDY_ID,
        "output": str(args.output / "results.json"),
        "strategies": {
            name: {
                "trades": block["trades"],
                "baseline_win_rate_pct": block["baseline"]["win_rate_pct"],
                "confirmation_p": (
                    block["confirmation_entry"].get("by_bar", {})
                    .get("bar_1", {}).get("permutation_test", {})
                    .get("p_selection_at_least_this_good")
                ),
            }
            for name, block in results["strategies"].items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
