#!/usr/bin/env python3
"""RS-XAUUSD-20260819-001 — pattern mining on 30-minute gold, and what survives a null.

A request was made for repeatable regularities in the 30-minute data, with overfitting
accepted as a cost. This runs the mining broadly and reports what it finds — including three
families that produce spectacular in-sample results and are artefacts, and one that is real.

The distinction is not made by significance testing alone. Each family is compared against a
null that reproduces everything about the data *except* the effect being claimed:

- Time-slot and candlestick patterns are measured as **excess over the same period's
  baseline drift**. Without that subtraction, at a 16-bar horizon every pattern tested looks
  profitable, because the market rose and every pattern inherits the rise.
- Daily extreme timing is compared against a **random walk on the same day's own returns**,
  shuffled. A random walk places its extremes at the start and end of any window by the
  arcsine law, so the raw distribution looks structured whether or not the market is.
- The slot basket is built exactly as an overfitter would build it, and reported with its
  out-of-sample collapse, because that collapse is the most useful thing about it.

## What survives

One family. Daily highs and lows form at 09:00 and 22:00-23:00 Taipei far more often than a
random walk on the same returns produces, and at 04:00-06:00 far less often, with the same
magnitudes in all three periods. It is the intraday volatility profile expressed where it
can be acted on.

Usage:
    python3.12 -m scripts.research.build_xauusd_pattern_mining
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "RS-XAUUSD-20260819-001"
OUTPUT_DIR = Path("reproduced")
BARS_FILE = Path("local-inputs/FX_IDC_XAUUSD, 30_volumn.csv")TAIPEI = timezone(timedelta(hours=8))

TRAIN_END, VALID_END = 0.55, 0.80
COST_PCT = 0.02
DAY_BOUNDARY_HOUR = 7
NULL_SHUFFLES = 200
BLOCK_TRIALS = 4000
RANDOM_SEED = 20260819


def stream(label: str) -> random.Random:
    return random.Random(f"{RANDOM_SEED}:{label}")


def load_bars() -> list[dict]:
    rows = []
    with BARS_FILE.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            rows.append({
                "t": datetime.fromisoformat(record["time"]).astimezone(TAIPEI),
                "o": float(record["open"]), "h": float(record["high"]),
                "l": float(record["low"]), "c": float(record["close"]),
            })
    rows.sort(key=lambda bar: bar["t"])
    return rows


def splits(n: int) -> dict:
    return {"train": (0, int(n * TRAIN_END)),
            "valid": (int(n * TRAIN_END), int(n * VALID_END)),
            "holdout": (int(n * VALID_END), n)}


def forward(bars, i, h):
    if i + h >= len(bars):
        return None
    return 100 * (bars[i + h]["c"] - bars[i]["c"]) / bars[i]["c"]


def spearman(x, y):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out
    a, b = rank(x), rank(y)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return round(num / den, 3) if den else 0.0


def slot_matrix(bars, sp) -> dict:
    """Every (weekday, hour, minute) cell, and the basket an overfitter would build."""
    n = len(bars)
    rets = [None] + [100 * (bars[i]["c"] - bars[i - 1]["c"]) / bars[i - 1]["c"]
                     for i in range(1, n)]
    cells = collections.defaultdict(lambda: {"train": [], "valid": [], "holdout": []})
    for period, (a, b) in sp.items():
        for i in range(a, b):
            if rets[i] is None:
                continue
            t = bars[i]["t"]
            cells[(t.weekday(), t.hour, t.minute)][period].append(rets[i])

    scored = {}
    for key, value in cells.items():
        if len(value["train"]) < 40:
            continue
        mean = statistics.fmean(value["train"])
        sd = statistics.pstdev(value["train"])
        scored[key] = {
            "n_train": len(value["train"]),
            "train_mean_pct": round(mean, 4),
            "t": round(mean / (sd / math.sqrt(len(value["train"]))), 2) if sd else 0.0,
            "valid_mean_pct": round(statistics.fmean(value["valid"]), 4) if value["valid"] else None,
            "holdout_mean_pct": round(statistics.fmean(value["holdout"]), 4) if value["holdout"] else None,
        }
    significant = sum(1 for v in scored.values() if abs(v["t"]) > 2)
    consistent = sum(1 for v in scored.values()
                     if v["valid_mean_pct"] is not None and v["holdout_mean_pct"] is not None
                     and v["train_mean_pct"] * v["valid_mean_pct"] > 0
                     and v["train_mean_pct"] * v["holdout_mean_pct"] > 0)

    ordered = sorted(scored.items(), key=lambda kv: -abs(kv[1]["t"]))
    baskets = {}
    for size in (5, 10, 20, 40):
        chosen = {k: (1 if v["t"] > 0 else -1) for k, v in ordered[:size]}
        row = {}
        for period, (a, b) in sp.items():
            values = []
            for i in range(a, b):
                if rets[i] is None:
                    continue
                t = bars[i]["t"]
                key = (t.weekday(), t.hour, t.minute)
                if key in chosen:
                    values.append(chosen[key] * rets[i] - COST_PCT)
            if len(values) < 20:
                continue
            years = (bars[b - 1]["t"] - bars[a]["t"]).days / 365
            sd = statistics.pstdev(values)
            row[period] = {
                "trades": len(values),
                "mean_pct": round(statistics.fmean(values), 4),
                "total_pct": round(sum(values), 2),
                "annualised_sharpe": round(
                    (statistics.fmean(values) / sd) * math.sqrt(len(values) / years), 2)
                if sd and years else None,
            }
        baskets[f"top_{size}"] = row

    return {
        "cells_scored": len(scored),
        "cells_with_abs_t_over_2": significant,
        "cells_expected_by_chance": round(0.05 * len(scored)),
        "cells_sign_consistent_all_three": consistent,
        "sign_consistent_expected_by_chance": len(scored) // 4,
        "top_cells": [{"weekday": k[0], "hour": k[1], "minute": k[2], **v}
                      for k, v in ordered[:12]],
        "overfit_basket": baskets,
        "verdict": ("A noise field. The count of significant cells barely exceeds chance, "
                    "and a basket built from the strongest cells earns a train Sharpe above "
                    "3 that does not survive either later period. This is what overfitting "
                    "looks like when it is done deliberately."),
    }


def candlestick_patterns(bars, sp) -> dict:
    """Classic patterns, scored as excess over the same period's drift."""
    n = len(bars)
    o = [b["o"] for b in bars]
    h = [b["h"] for b in bars]
    low = [b["l"] for b in bars]
    c = [b["c"] for b in bars]
    span = [h[i] - low[i] for i in range(n)]

    patterns = {
        "inside_bar": lambda i: i > 0 and h[i] < h[i - 1] and low[i] > low[i - 1],
        "outside_bar": lambda i: i > 0 and h[i] > h[i - 1] and low[i] < low[i - 1],
        "narrowest_of_7": lambda i: i >= 7 and span[i] == min(span[i - 6:i + 1]),
        "widest_of_7": lambda i: i >= 7 and span[i] == max(span[i - 6:i + 1]),
        "bullish_engulfing": lambda i: (i > 0 and c[i] > o[i] and c[i - 1] < o[i - 1]
                                        and c[i] > o[i - 1] and o[i] < c[i - 1]),
        "bearish_engulfing": lambda i: (i > 0 and c[i] < o[i] and c[i - 1] > o[i - 1]
                                        and c[i] < o[i - 1] and o[i] > c[i - 1]),
        "three_up_closes": lambda i: i >= 3 and all(c[j] > c[j - 1] for j in (i, i - 1, i - 2)),
        "three_down_closes": lambda i: i >= 3 and all(c[j] < c[j - 1] for j in (i, i - 1, i - 2)),
        "doji": lambda i: span[i] > 0 and abs(c[i] - o[i]) / span[i] < 0.1,
        "near_round_100": lambda i: abs(c[i] - round(c[i] / 100) * 100) <= 2.0,
        "near_round_50": lambda i: abs(c[i] - round(c[i] / 50) * 50) <= 1.5,
        "near_round_10": lambda i: abs(c[i] - round(c[i] / 10) * 10) <= 0.5,
    }

    out = {}
    for horizon in (4, 16):
        baseline = {}
        for period, (a, b) in sp.items():
            values = [forward(bars, i, horizon) for i in range(a, b)
                      if forward(bars, i, horizon) is not None]
            baseline[period] = statistics.fmean(values)
        rows = {}
        raw_ts = []
        for name, test in patterns.items():
            per_period = {}
            for period, (a, b) in sp.items():
                per_period[period] = [forward(bars, i, horizon) for i in range(max(a, 10), b)
                                      if test(i) and forward(bars, i, horizon) is not None]
            if len(per_period["train"]) < 40:
                continue
            excess = {p: round(statistics.fmean(v) - baseline[p], 4)
                      for p, v in per_period.items() if len(v) > 10}
            sd = statistics.pstdev(per_period["train"])
            se = sd / math.sqrt(len(per_period["train"])) if sd else None
            raw_mean = statistics.fmean(per_period["train"])
            if se:
                raw_ts.append(abs(raw_mean / se))
            rows[name] = {
                "n_train": len(per_period["train"]),
                "raw_train_pct": round(raw_mean, 4),
                "raw_t_if_drift_ignored": round(raw_mean / se, 2) if se else None,
                "excess": excess,
                "excess_t_train": round(excess["train"] / se, 2) if se else None,
                "sign_consistent": all(excess["train"] * excess[p] > 0
                                       for p in ("valid", "holdout") if p in excess),
            }
        out[f"h{horizon}"] = {
            "baseline_drift_pct": {p: round(v, 4) for p, v in baseline.items()},
            "patterns": rows,
            "largest_abs_excess_t": round(max(
                (abs(r["excess_t_train"]) for r in rows.values()
                 if r["excess_t_train"] is not None), default=0), 2),
            "largest_abs_raw_t_if_drift_ignored": round(max(raw_ts, default=0), 2),
        }
    out["verdict"] = (
        "Drift, not patterns. At a 16-bar horizon the raw returns make every pattern look "
        "profitable, with t-statistics up to about 5.9; after subtracting the same period's "
        "baseline the largest excess t falls to about 1.4. The round-number family is the "
        "only one whose sign holds in all three periods, and it is tested separately.")
    return out


def round_number_test(bars, sp, rng) -> dict:
    """Block-bootstrap the one candlestick-family result whose sign held everywhere."""
    def near(i, step, tolerance):
        return abs(bars[i]["c"] - round(bars[i]["c"] / step) * step) <= tolerance

    def run(step, tolerance, horizon, span, trials=BLOCK_TRIALS):
        a, b = span
        rows = [(near(i, step, tolerance), forward(bars, i, horizon))
                for i in range(a, b) if forward(bars, i, horizon) is not None]
        close = [r for f, r in rows if f]
        away = [r for f, r in rows if not f]
        if len(close) < 100:
            return None
        observed = statistics.fmean(close) - statistics.fmean(away)
        flags = [f for f, _ in rows]
        rets = [r for _, r in rows]
        length, n = 3 * horizon, len(rows)
        blocks = max(1, n // length)
        hits = 0
        for _ in range(trials):
            index = []
            for _ in range(blocks):
                start = rng.randrange(0, max(1, n - length))
                index.extend(range(start, start + length))
            fs = [flags[j] for j in index]
            rs = [rets[j] for j in index]
            shift = rng.randrange(1, len(rs))
            rs = rs[shift:] + rs[:shift]
            g1 = [r for f, r in zip(fs, rs) if f]
            g2 = [r for f, r in zip(fs, rs) if not f]
            if g1 and g2 and abs(statistics.fmean(g1) - statistics.fmean(g2)) >= abs(observed):
                hits += 1
        return {"n_near": len(close), "difference_pct": round(observed, 4),
                "p_value": round(hits / trials, 4)}

    out = {}
    for step, tolerance in ((100, 10), (100, 5), (50, 5), (25, 3)):
        key = f"level_{step}_within_{tolerance}"
        out[key] = {p: run(step, tolerance, 16, sp[p]) for p in sp}
        out[key]["pooled"] = run(step, tolerance, 16, (0, len(bars)))
    out["verdict"] = (
        "Not supported. The 100 dollar level is positive in all three periods but never "
        "significant, pooled p is about 0.17, and the 50 and 25 dollar levels show nothing. "
        "A real order-clustering effect would appear at the finer levels too, weaker but "
        "present. Sign consistency without a mechanism is what noise looks like.")
    return out


def daily_extreme_timing(bars, sp, rng) -> dict:
    """When the day's high and low form, against a random walk on that day's own returns."""
    def day_key(t):
        return t.date() if t.hour >= DAY_BOUNDARY_HOUR else (t - timedelta(days=1)).date()

    days = collections.defaultdict(list)
    for i in range(len(bars)):
        days[day_key(bars[i]["t"])].append(i)

    def period_of(i):
        for name, (a, b) in sp.items():
            if a <= i < b:
                return name
        return None

    def profile(indices):
        real_high, real_low = collections.Counter(), collections.Counter()
        null_high, null_low = collections.Counter(), collections.Counter()
        for idx in indices:
            length = len(idx)
            hi = max(range(length), key=lambda j: bars[idx[j]]["h"])
            lo = min(range(length), key=lambda j: bars[idx[j]]["l"])
            real_high[bars[idx[hi]]["t"].hour] += 1
            real_low[bars[idx[lo]]["t"].hour] += 1
            steps = [bars[idx[j]]["c"] - bars[idx[j - 1]]["c"] for j in range(1, length)]
            for _ in range(NULL_SHUFFLES):
                rng.shuffle(steps)
                path = [0.0]
                for step in steps:
                    path.append(path[-1] + step)
                null_high[bars[idx[max(range(length), key=lambda j: path[j])]]["t"].hour] += 1
                null_low[bars[idx[min(range(length), key=lambda j: path[j])]]["t"].hour] += 1
        rh, rl = sum(real_high.values()), sum(real_low.values())
        nh, nl = sum(null_high.values()), sum(null_low.values())
        high = {h: round(100 * real_high[h] / rh - 100 * null_high[h] / nh, 2) for h in range(24)}
        low = {h: round(100 * real_low[h] / rl - 100 * null_low[h] / nl, 2) for h in range(24)}
        return high, low, len(indices)

    by_period = {}
    for name in ("train", "valid", "holdout"):
        indices = [idx for idx in days.values() if len(idx) >= 44 and period_of(idx[0]) == name]
        high, low, count = profile(indices)
        by_period[name] = {"days": count, "high_excess_pct": high, "low_excess_pct": low}

    hours = list(range(24))
    stability = {}
    for label, key in (("high", "high_excess_pct"), ("low", "low_excess_pct")):
        for other in ("valid", "holdout"):
            stability[f"{label}_train_vs_{other}"] = spearman(
                [by_period["train"][key][h] for h in hours],
                [by_period[other][key][h] for h in hours])

    consistent = []
    for h in hours:
        highs = [by_period[p]["high_excess_pct"][h] for p in ("train", "valid", "holdout")]
        lows = [by_period[p]["low_excess_pct"][h] for p in ("train", "valid", "holdout")]
        if all(v > 1.0 for v in highs) and all(v > 1.0 for v in lows):
            consistent.append({"hour": h, "direction": "extremes cluster here",
                               "high_excess": highs, "low_excess": lows})
        elif all(v < -0.8 for v in highs) and all(v < -0.8 for v in lows):
            consistent.append({"hour": h, "direction": "extremes avoid this hour",
                               "high_excess": highs, "low_excess": lows})

    return {
        "day_boundary_hour_taipei": DAY_BOUNDARY_HOUR,
        "null": "each day's own bar-to-bar changes shuffled; volatility and length preserved",
        "null_shuffles_per_day": NULL_SHUFFLES,
        "by_period": by_period,
        "shape_stability_spearman": stability,
        "hours_consistent_in_all_three_periods": consistent,
        "verdict": (
            "The one family that survives. Daily extremes form at 09:00 and 22:00-23:00 "
            "Taipei far more often than a random walk on the same returns produces, and at "
            "04:00-06:00 far less often, with comparable magnitudes in all three periods. "
            "It is the intraday volatility profile expressed in a form that can be acted on, "
            "and it says nothing about direction."),
    }


SEQUENCE_LENGTHS = [2, 3, 4]
SEQUENCE_HORIZONS = [4, 8]
SEQUENCE_MIN_SAMPLES = 50
SEQUENCE_TRIALS = 1500


def _atr_z(bars):
    """Each bar's close-to-close move in units of the trailing ATR."""
    n = len(bars)
    ranges = [bars[0]["h"] - bars[0]["l"]] + [
        max(bars[i]["h"] - bars[i]["l"], abs(bars[i]["h"] - bars[i - 1]["c"]),
            abs(bars[i]["l"] - bars[i - 1]["c"])) for i in range(1, n)]
    out = [None] * n
    for i in range(15, n):
        reference = statistics.fmean(ranges[i - 14:i])
        if reference:
            out[i] = (bars[i]["c"] - bars[i - 1]["c"]) / reference
    return out


def _encodings(bars):
    """Two encodings that carry different information.

    `move` is close-to-close size, which is what a sequence study normally uses.
    `close_position` is where the close sat inside its own bar, which says something about
    pressure within the bar that the net move does not. They are mined separately because a
    pattern absent from one and present in the other is more interesting than either alone.
    """
    z = _atr_z(bars)
    move = [None if v is None else
            ("D" if v < -0.5 else "d" if v < 0 else "u" if v < 0.5 else "U") for v in z]
    position = []
    for bar in bars:
        span = bar["h"] - bar["l"]
        if span <= 0:
            position.append(None)
            continue
        x = (bar["c"] - bar["l"]) / span
        position.append("L" if x < 1 / 3 else "M" if x < 2 / 3 else "T")
    return {"move": move, "close_position": position}, z


def _family_permutation(keys, rets, eligible, rng, trials=SEQUENCE_TRIALS):
    """Correct across the whole enumeration, not per sequence.

    Enumerating every sequence and reporting the strongest is a search over dozens of
    cells. The null shuffles the forward returns, recomputes every cell's t against the
    same fixed grouping, and keeps the largest — so the reported p is for the best result
    found anywhere in the enumeration.
    """
    def t_values(values):
        grouped = collections.defaultdict(list)
        for key, value in zip(keys, values):
            grouped[key].append(value)
        mean = statistics.fmean(values)
        out = []
        for key in eligible:
            group = grouped[key]
            sd = statistics.pstdev(group)
            out.append(abs((statistics.fmean(group) - mean) / (sd / math.sqrt(len(group))))
                       if sd else 0.0)
        return out

    observed = t_values(rets)
    shuffled = list(rets)
    maxima, counts = [], []
    for _ in range(trials):
        rng.shuffle(shuffled)
        values = t_values(shuffled)
        maxima.append(max(values))
        counts.append(sum(1 for v in values if v > 2))
    maxima.sort()
    return {
        "eligible_sequences": len(eligible),
        "observed_max_abs_t": round(max(observed), 2),
        "null_median_max_abs_t": round(statistics.median(maxima), 2),
        "null_95th_max_abs_t": round(maxima[int(0.95 * len(maxima))], 2),
        "p_max_at_least_observed": round(
            sum(1 for v in maxima if v >= max(observed)) / trials, 4),
        "observed_count_abs_t_over_2": sum(1 for v in observed if v > 2),
        "null_median_count": round(statistics.median(counts)),
        "note": ("An observed maximum below the null median means the enumeration is less "
                 "extreme than shuffling typically produces."),
    }


def sequence_mining(bars, sp, rng) -> dict:
    """Discretised multi-bar sequences under two encodings, corrected family-wide."""
    encodings, z = _encodings(bars)
    out = {"encodings": {}}

    for encoding_name, symbols in encodings.items():
        per_encoding = {}
        for length in SEQUENCE_LENGTHS:
            for horizon in SEQUENCE_HORIZONS:
                a, b = sp["train"]
                keys, rets = [], []
                for i in range(max(a, 20), b):
                    window = [symbols[j] for j in range(i - length + 1, i + 1)]
                    if any(sym is None for sym in window):
                        continue
                    value = forward(bars, i, horizon)
                    if value is None:
                        continue
                    keys.append("".join(window))
                    rets.append(value)
                grouped = collections.defaultdict(list)
                for key, value in zip(keys, rets):
                    grouped[key].append(value)
                eligible = [k for k, v in grouped.items() if len(v) >= SEQUENCE_MIN_SAMPLES]
                if len(eligible) < 4:
                    continue

                baseline = {}
                for period, (pa, pb) in sp.items():
                    values = [forward(bars, i, horizon) for i in range(pa, pb)
                              if forward(bars, i, horizon) is not None]
                    baseline[period] = statistics.fmean(values)

                consistent = []
                for key in eligible:
                    row = {}
                    for period, (pa, pb) in sp.items():
                        values = [forward(bars, i, horizon) - baseline[period]
                                  for i in range(max(pa, 20), pb)
                                  if forward(bars, i, horizon) is not None
                                  and all(symbols[j] is not None
                                          for j in range(i - length + 1, i + 1))
                                  and "".join(symbols[j] for j in
                                              range(i - length + 1, i + 1)) == key]
                        row[period] = (len(values),
                                       round(statistics.fmean(values), 4)
                                       if len(values) >= 15 else None)
                    means = [row[p][1] for p in ("train", "valid", "holdout")]
                    if all(m is not None for m in means) and (
                            all(m > 0 for m in means) or all(m < 0 for m in means)):
                        consistent.append({"sequence": key, "by_period": row})
                consistent.sort(key=lambda c: -abs(c["by_period"]["train"][1]))

                per_encoding[f"len{length}_h{horizon}"] = {
                    "family_permutation": _family_permutation(
                        keys, rets, eligible, rng),
                    "sign_consistent_sequences": len(consistent),
                    "top_consistent": consistent[:4],
                }
        out["encodings"][encoding_name] = per_encoding

    out["focused_tests"] = {
        "compression_then_thrust": _compression_thrust(bars, sp, z),
        "close_position_exhaustion": _close_exhaustion(bars, sp),
    }
    out["verdict"] = (
        "Nothing. Under both encodings the enumeration is at or below what shuffling "
        "produces, and the two candidates that survived a naive consistency screen were "
        "each killed by a check the screen does not perform: the move-encoding candidate by "
        "comparing it against the same pattern without its supposed precondition, and the "
        "close-position candidate by its mirror.")
    return out


def _compression_thrust(bars, sp, z) -> dict:
    """The move-encoding candidate, tested against the thing it must beat.

    The neighbourhood uU, dU, udU, uudU was positive in all three periods and strengthened
    with specificity, which reads as a compression-then-thrust effect. The test that matters
    is not significance but whether the compression precondition adds anything over a large
    bar on its own — and whether the short side mirrors.
    """
    def large(i, up):
        return z[i] is not None and (z[i] > 0.5 if up else z[i] < -0.5)

    def compressed(i, k, up):
        if any(z[j] is None for j in range(i - k, i + 1)) or not large(i, up):
            return False
        return all(abs(z[j]) < 0.5 for j in range(i - k, i))

    rows = {}
    for horizon in (4, 8):
        for k in (1, 2):
            for up in (True, False):
                label = f"k{k}_h{horizon}_{'up' if up else 'down'}"
                with_pre, without_pre = {}, {}
                for period, (a, b) in sp.items():
                    values = [forward(bars, i, horizon) for i in range(a, b)
                              if forward(bars, i, horizon) is not None]
                    mean = statistics.fmean(values)
                    w = [forward(bars, i, horizon) - mean for i in range(max(a, 20), b)
                         if compressed(i, k, up) and forward(bars, i, horizon) is not None]
                    o = [forward(bars, i, horizon) - mean for i in range(max(a, 20), b)
                         if large(i, up) and forward(bars, i, horizon) is not None]
                    with_pre[period] = (len(w), round(statistics.fmean(w), 4) if len(w) > 10 else None)
                    without_pre[period] = (len(o), round(statistics.fmean(o), 4) if len(o) > 10 else None)
                rows[label] = {"with_compression": with_pre, "large_bar_only": without_pre}
    return {
        "rows": rows,
        "verdict": ("Fails. The excess is 0.005% to 0.03%, at or below a 0.02% round trip; "
                    "the down side changes sign between periods; and the compression "
                    "precondition adds essentially nothing over a large bar alone, which is "
                    "what the nested neighbourhood appeared to promise."),
    }


def _close_exhaustion(bars, sp) -> dict:
    """The close-position candidate, tested by its mirror.

    MTTT and TTTL were negative in all three periods with growing magnitude, and share a
    coherent story: repeated closes at the top of the bar, then underperformance. If that is
    exhaustion, repeated closes at the bottom must outperform. The mirror is the test.
    """
    position = []
    for bar in bars:
        span = bar["h"] - bar["l"]
        position.append(None if span <= 0 else (bar["c"] - bar["l"]) / span)

    def run(i, k, top):
        if any(position[j] is None for j in range(i - k + 1, i + 1)):
            return False
        return all((position[j] >= 2 / 3) if top else (position[j] <= 1 / 3)
                   for j in range(i - k + 1, i + 1))

    rows = {}
    mirrored = []
    for horizon in (8, 16):
        for k in (2, 3, 4):
            sides = {}
            for top in (True, False):
                per_period = {}
                for period, (a, b) in sp.items():
                    values = [forward(bars, i, horizon) for i in range(a, b)
                              if forward(bars, i, horizon) is not None]
                    mean = statistics.fmean(values)
                    v = [forward(bars, i, horizon) - mean for i in range(max(a, 20), b)
                         if run(i, k, top) and forward(bars, i, horizon) is not None]
                    per_period[period] = (len(v),
                                          round(statistics.fmean(v), 4) if len(v) > 10 else None)
                sides["top" if top else "bottom"] = per_period
            top_means = [sides["top"][p][1] for p in ("train", "valid", "holdout")]
            bottom_means = [sides["bottom"][p][1] for p in ("train", "valid", "holdout")]
            top_ok = all(m is not None and m < 0 for m in top_means)
            bottom_ok = all(m is not None and m > 0 for m in bottom_means)
            rows[f"k{k}_h{horizon}"] = {**sides, "top_consistent": top_ok,
                                        "mirror_holds": top_ok and bottom_ok}
            if top_ok and bottom_ok:
                mirrored.append(f"k{k}_h{horizon}")
    return {
        "rows": rows,
        "settings_where_mirror_holds": mirrored,
        "verdict": ("Fails its mirror. One setting has a consistent top side, and its bottom "
                    "side is negative in two periods of three — the wrong direction for an "
                    "exhaustion effect, which must be symmetric. The sequences that looked "
                    "consistent were the survivors of an 81-cell enumeration."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    bars = load_bars()
    sp = splits(len(bars))
    first, last = bars[0]["c"], bars[-1]["c"]

    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(tz=TAIPEI).isoformat(timespec="seconds"),
        "strategy": "none — bar-level pattern mining, independent of S1 and S2",
        "method": {
            "bar_source": "30-minute FX_IDC:XAUUSD export",
            "bar_count": len(bars),
            "splits": {k: {"bars": v[1] - v[0],
                           "from": bars[v[0]]["t"].date().isoformat(),
                           "to": bars[v[1] - 1]["t"].date().isoformat()}
                       for k, v in sp.items()},
            "brief": ("repeatable regularities were requested and overfitting accepted; "
                      "each family is therefore reported with the null that decides it"),
            "round_trip_cost_pct": COST_PCT,
            "random_seed": RANDOM_SEED,
            "timezone": "Asia/Taipei",
        },
        "benchmark": {"buy_and_hold_return_pct": round(100 * (last - first) / first, 1)},
        "families": {
            "weekly_slot_matrix": slot_matrix(bars, sp),
            "candlestick_and_round_numbers": candlestick_patterns(bars, sp),
            "round_number_bootstrap": round_number_test(bars, sp, stream("round")),
            "daily_extreme_timing": daily_extreme_timing(bars, sp, stream("extremes")),
            "multi_bar_sequences": sequence_mining(bars, sp, stream("sequences")),
        },
        "limitations": [
            "Three of the four families are reported as artefacts rather than findings. "
            "That is the result, not a failure of the search.",
            "The surviving family is about WHEN extremes form, not about direction. It does "
            "not predict whether the next move is up or down and must not be read that way.",
            "The daily boundary is 07:00 Taipei. A different boundary moves the arcsine edge "
            "effect to different hours; the null is recomputed with the same boundary so the "
            "excess is unaffected, but raw distributions are not comparable across choices.",
            "One instrument, 32 months, one strong uptrend.",
            "No result changes formal S1 or S2 logic, live risk, or an active entry checklist.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    slots = results["families"]["weekly_slot_matrix"]
    extremes = results["families"]["daily_extreme_timing"]
    print(json.dumps({
        "study_id": STUDY_ID,
        "slot_cells_significant_vs_chance":
            f"{slots['cells_with_abs_t_over_2']}/{slots['cells_expected_by_chance']}",
        "overfit_basket_top5_sharpe": {
            p: v.get("annualised_sharpe") for p, v in slots["overfit_basket"]["top_5"].items()},
        "candlestick_t_with_and_without_drift": [
            results["families"]["candlestick_and_round_numbers"]["h16"]["largest_abs_raw_t_if_drift_ignored"],
            results["families"]["candlestick_and_round_numbers"]["h16"]["largest_abs_excess_t"]],
        "surviving_hours": [c["hour"] for c in extremes["hours_consistent_in_all_three_periods"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
