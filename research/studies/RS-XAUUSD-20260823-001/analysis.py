#!/usr/bin/env python3
"""RS-XAUUSD-20260823-001 — how the daily range accumulates across the Taipei day.

RS-XAUUSD-20260819-001 found one surviving regularity: daily extremes cluster at 09:00 and
21:00-23:00 Taipei and avoid 04:00. That result answers *when an extreme forms*. It does not
answer the question asked at the chart, which is *how much of today's move is already
behind me* — and that question is answerable without any directional claim.

This study measures the cumulative range profile at 30-minute resolution and puts it through
the screens the previous study earned:

- **Null.** Range is cumulative, so it rises through the day whether or not the market has
  session structure. The null permutes each day's own bars into the same slot positions,
  preserving that day's volatility, its bar shapes and the arithmetic of accumulation, and
  destroying only the time-of-day assignment. Excess over that null is session structure.
- **Mechanism screen.** If the profile is session participation, cumulative volume must
  track cumulative range. A range profile that does not match its own volume profile is
  telling a different story than the one being claimed.
- **Confound.** Taipei does not observe DST; the US session does. A fixed-clock profile
  therefore blurs the US block by one hour across the year. The two regimes are separated
  and compared rather than averaged silently.
- **Stationarity.** Shape stability across three periods is reported, and so is the drift in
  level — which turns out to be the finding with the shortest shelf life and the largest
  practical consequence.

Nothing here is directional and nothing here changes S1, S2 or live risk.

Usage:
    python3.12 -m scripts.research.build_xauusd_intraday_range_profile
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-XAUUSD-20260823-001"
OUTPUT_DIR = Path("reproduced-intraday-range-profile")
BARS_FILE = ROOT / "local-inputs/xauusd-30m.csv"
TAIPEI = timezone(timedelta(hours=8))

DAY_BOUNDARY_HOUR = 7
MIN_BARS_PER_DAY = 44          # a complete session is 46 bars; 44 keeps 652 of 767 day keys
TRAIN_END, VALID_END = 0.55, 0.80
NULL_SHUFFLES = 200
RANDOM_SEED = 20260823

SLOTS = [f"{(DAY_BOUNDARY_HOUR + i // 2) % 24:02d}:{'30' if i % 2 else '00'}"
         for i in range(48)]


def stream(label: str) -> random.Random:
    return random.Random(f"{RANDOM_SEED}:{label}")


# --------------------------------------------------------------------------- loading

def load_bars(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            moment = datetime.fromisoformat(record["time"]).astimezone(TAIPEI)
            rows.append({
                "t": moment,
                "o": float(record["open"]), "h": float(record["high"]),
                "l": float(record["low"]), "c": float(record["close"]),
                "v": float(record["Volume"]),
            })
    rows.sort(key=lambda bar: bar["t"])
    return rows


def us_dst(day: date) -> bool:
    """US DST: second Sunday in March to first Sunday in November."""
    march = date(day.year, 3, 1)
    start = march + timedelta(days=(6 - march.weekday()) % 7 + 7)
    november = date(day.year, 11, 1)
    end = november + timedelta(days=(6 - november.weekday()) % 7)
    return start <= day < end


def build_sessions(bars: list[dict]) -> list[dict]:
    """Group bars into 07:00-Taipei sessions and keep the near-complete ones."""
    grouped: dict[date, list[dict]] = collections.defaultdict(list)
    for bar in bars:
        moment = bar["t"]
        key = (moment.date() if moment.hour >= DAY_BOUNDARY_HOUR
               else (moment - timedelta(days=1)).date())
        grouped[key].append(bar)

    slot_index = {slot: i for i, slot in enumerate(SLOTS)}
    sessions = []
    dropped = collections.Counter()
    for key in sorted(grouped):
        day_bars = grouped[key]
        if len(day_bars) < MIN_BARS_PER_DAY:
            dropped["incomplete"] += 1
            continue
        high = max(bar["h"] for bar in day_bars)
        low = min(bar["l"] for bar in day_bars)
        if high <= low:
            dropped["zero_range"] += 1
            continue
        placed: list[list[dict]] = [[] for _ in SLOTS]
        for bar in day_bars:
            placed[slot_index[f"{bar['t'].hour:02d}:{bar['t'].minute:02d}"]].append(bar)
        sessions.append({
            "date": key, "bars": day_bars, "placed": placed,
            "high": high, "low": low, "range": high - low,
            "weekday": key.weekday(), "us_dst": us_dst(key),
            "volume": sum(bar["v"] for bar in day_bars),
        })
    return sessions, dict(dropped)


def splits(n: int) -> dict:
    return {"train": (0, int(n * TRAIN_END)),
            "valid": (int(n * TRAIN_END), int(n * VALID_END)),
            "holdout": (int(n * VALID_END), n)}


# --------------------------------------------------------------------------- helpers

def spearman(x: list[float], y: list[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = rank(list(xs)), rank(list(ys))
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def completion_path(placed: list[list[dict]], day_range: float) -> list[float]:
    """Fraction of the final daily range already traversed at the end of each slot.

    A slot with no bar (the daily maintenance break, whose position moves with US DST)
    carries the previous value forward: nothing traded, so nothing accumulated.
    """
    running_high, running_low = -math.inf, math.inf
    path = []
    for slot_bars in placed:
        for bar in slot_bars:
            running_high = max(running_high, bar["h"])
            running_low = min(running_low, bar["l"])
        path.append(0.0 if running_high == -math.inf
                    else 100.0 * (running_high - running_low) / day_range)
    return path


def volume_path(placed: list[list[dict]], day_volume: float) -> list[float]:
    running, path = 0.0, []
    for slot_bars in placed:
        running += sum(bar["v"] for bar in slot_bars)
        path.append(100.0 * running / day_volume if day_volume else 0.0)
    return path


def increments(profile: list[float]) -> list[float]:
    """Per-slot contribution. Spearman on a cumulative curve is always 1.0 — every
    cumulative profile is monotone — so every shape comparison here runs on increments."""
    return [profile[0]] + [profile[i] - profile[i - 1] for i in range(1, len(profile))]


def mean_profile(sessions, fn) -> list[float]:
    paths = [fn(s) for s in sessions]
    return [round(statistics.fmean(p[i] for p in paths), 2) for i in range(len(SLOTS))]


def median_profile(sessions, fn) -> list[float]:
    paths = [fn(s) for s in sessions]
    return [round(statistics.median(p[i] for p in paths), 2) for i in range(len(SLOTS))]


# --------------------------------------------------------------------------- families

def observed_profiles(sessions_by_period: dict) -> dict:
    out = {}
    for name, sessions in sessions_by_period.items():
        comp = lambda s: completion_path(s["placed"], s["range"])   # noqa: E731
        vol = lambda s: volume_path(s["placed"], s["volume"])       # noqa: E731
        out[name] = {
            "days": len(sessions),
            "range_completed_mean_pct": dict(zip(SLOTS, mean_profile(sessions, comp))),
            "range_completed_median_pct": dict(zip(SLOTS, median_profile(sessions, comp))),
            "volume_completed_mean_pct": dict(zip(SLOTS, mean_profile(sessions, vol))),
            "slot_coverage_days": {
                slot: sum(1 for s in sessions if s["placed"][i]) for i, slot in enumerate(SLOTS)
            },
        }
    return out


def close_completion_path(session: dict, closes: list[float]) -> list[float]:
    """Running range of a close path, per slot, as a fraction of that path's final range.

    Used only for the null test, because a shuffled-returns null produces a close path and
    not OHLC bars. The headline profile stays OHLC-based; this is the like-for-like basis
    on which observed and null can be compared at all.
    """
    running_max, running_min = closes[0], closes[0]
    per_bar = []
    for value in closes:
        running_max = max(running_max, value)
        running_min = min(running_min, value)
        per_bar.append(running_max - running_min)
    final = per_bar[-1]
    if final <= 0:
        return [0.0] * len(SLOTS)
    path, cursor, current = [], 0, 0.0
    for slot_bars in session["placed"]:
        for _ in slot_bars:
            current = per_bar[cursor]
            cursor += 1
        path.append(100.0 * current / final)
    return path


def null_comparison(sessions_by_period: dict, rng: random.Random) -> dict:
    """Observed close-path profile against a random walk on that day's own returns.

    The null shuffles each session's own bar-to-bar close changes and rebuilds the path.
    That preserves the session's realised volatility, its number of bars and the arcsine
    geometry of a random walk, and destroys only *when* in the session the large moves
    happened. Excess over it is time-of-day structure and nothing else.
    """
    out = {}
    for name, sessions in sessions_by_period.items():
        observed_paths = [close_completion_path(s, [bar["c"] for bar in s["bars"]])
                          for s in sessions]
        observed = [statistics.fmean(p[i] for p in observed_paths) for i in range(len(SLOTS))]

        shuffle_means: list[list[float]] = []
        for _ in range(NULL_SHUFFLES):
            paths = []
            for session in sessions:
                closes = [bar["c"] for bar in session["bars"]]
                steps = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
                rng.shuffle(steps)
                rebuilt = [closes[0]]
                for step in steps:
                    rebuilt.append(rebuilt[-1] + step)
                paths.append(close_completion_path(session, rebuilt))
            shuffle_means.append([statistics.fmean(p[i] for p in paths)
                                  for i in range(len(SLOTS))])

        null_mean, null_sd, zscores = [], [], []
        for i in range(len(SLOTS)):
            column = [row[i] for row in shuffle_means]
            mu = statistics.fmean(column)
            sd = statistics.pstdev(column)
            null_mean.append(mu)
            null_sd.append(sd)
            zscores.append((observed[i] - mu) / sd if sd > 1e-12 else None)

        null_max = []
        for row in shuffle_means:
            null_max.append(max(
                abs((row[i] - null_mean[i]) / null_sd[i]) if null_sd[i] > 1e-12 else 0.0
                for i in range(len(SLOTS))))
        finite = [abs(z) for z in zscores if z is not None]
        observed_max = max(finite) if finite else 0.0
        exceed = sum(1 for value in null_max if value >= observed_max)

        # Per-slot increments are the interpretable form: cumulative excess at a slot is
        # inherited from every earlier slot, while the increment is what that half hour
        # itself contributed relative to a random walk.
        obs_inc = [observed[0]] + [observed[i] - observed[i - 1] for i in range(1, len(SLOTS))]
        inc_rows = [[row[0]] + [row[i] - row[i - 1] for i in range(1, len(SLOTS))]
                    for row in shuffle_means]
        inc_mean, inc_z = [], []
        for i in range(len(SLOTS)):
            column = [row[i] for row in inc_rows]
            mu = statistics.fmean(column)
            sd = statistics.pstdev(column)
            inc_mean.append(mu)
            inc_z.append((obs_inc[i] - mu) / sd if sd > 1e-12 else None)

        out[name] = {
            "days": len(sessions),
            "basis": "close path (see close_completion_path); the headline profile is OHLC-based",
            "null": "each session's own bar-to-bar close changes shuffled, path rebuilt",
            "shuffles": NULL_SHUFFLES,
            "observed_pct": {slot: round(observed[i], 2) for i, slot in enumerate(SLOTS)},
            "null_mean_pct": {slot: round(null_mean[i], 2) for i, slot in enumerate(SLOTS)},
            "excess_pct": {slot: round(observed[i] - null_mean[i], 2)
                           for i, slot in enumerate(SLOTS)},
            "z": {slot: (round(zscores[i], 2) if zscores[i] is not None else None)
                  for i, slot in enumerate(SLOTS)},
            "family_max_abs_z_observed": round(observed_max, 2),
            "family_max_abs_z_null_median": round(statistics.median(null_max), 2),
            "family_permutation_p": round((exceed + 1) / (NULL_SHUFFLES + 1), 4),
            "slot_increment_observed_pct": {slot: round(obs_inc[i], 2)
                                            for i, slot in enumerate(SLOTS)},
            "slot_increment_excess_pct": {slot: round(obs_inc[i] - inc_mean[i], 2)
                                          for i, slot in enumerate(SLOTS)},
            "slot_increment_z": {slot: (round(inc_z[i], 2) if inc_z[i] is not None else None)
                                 for i, slot in enumerate(SLOTS)},
        }
    return out


def increment_consistency(nulls: dict) -> dict:
    """Which half hours carry more range than a random walk in ALL three periods.

    Sign consistency alone is a weak screen — RS-XAUUSD-20260819-001 found four of six
    candidates that passed it were artefacts. It is reported here with each period's z
    beside it so the magnitudes can be read, not just the signs.
    """
    periods = ["train", "valid", "holdout"]
    rows = []
    for slot in SLOTS:
        excess = [nulls[p]["slot_increment_excess_pct"][slot] for p in periods]
        zs = [nulls[p]["slot_increment_z"][slot] for p in periods]
        if any(value is None for value in excess) or all(value == 0 for value in excess):
            continue
        if all(value > 0 for value in excess) or all(value < 0 for value in excess):
            rows.append({
                "slot": slot,
                "direction": "carries more range than chance" if excess[0] > 0
                             else "carries less range than chance",
                "excess_pct": excess,
                "z": zs,
                "min_abs_z": round(min(abs(v) for v in zs if v is not None), 2),
            })
    strong = [r for r in rows if r["min_abs_z"] >= 2.0]
    return {
        "slots_sign_consistent": len(rows),
        "slots_sign_consistent_expected_by_chance": round(len(SLOTS) * 0.25, 1),
        "slots_consistent_and_abs_z_over_2_in_all_periods": len(strong),
        "detail": rows,
        "strongest": sorted(strong, key=lambda r: -r["min_abs_z"])[:8],
    }


def rejected_null_bar_permutation(sessions_by_period: dict, rng: random.Random) -> dict:
    """Kept as a record of a null that looked right and was not.

    Permuting whole bars into different slots moves each bar's absolute price level with it,
    so a bar taken from the session high can land beside one from the session low and the
    full range appears within the first hour. The null therefore front-loads range by
    construction, the observed profile sits far below it at every slot, and the resulting
    |z| of 40 to 85 measures the defect rather than the market.
    """
    out = {}
    for name, sessions in sessions_by_period.items():
        observed = [statistics.fmean(p[i] for p in
                                     [completion_path(s["placed"], s["range"]) for s in sessions])
                    for i in range(len(SLOTS))]
        means: list[list[float]] = []
        for _ in range(20):
            paths = []
            for session in sessions:
                pool = list(session["bars"])
                rng.shuffle(pool)
                shaped: list[list[dict]] = [[] for _ in SLOTS]
                cursor = 0
                for i, slot_bars in enumerate(session["placed"]):
                    shaped[i] = pool[cursor:cursor + len(slot_bars)]
                    cursor += len(slot_bars)
                paths.append(completion_path(shaped, session["range"]))
            means.append([statistics.fmean(p[i] for p in paths) for i in range(len(SLOTS))])
        null_mean = [statistics.fmean(row[i] for row in means) for i in range(len(SLOTS))]
        marks = ["07:30", "09:30", "12:30", "18:00"]
        out[name] = {
            "verdict": "rejected null — front-loads range by construction",
            "null_mean_pct_at": {m: round(null_mean[SLOTS.index(m)], 2) for m in marks},
            "observed_pct_at": {m: round(observed[SLOTS.index(m)], 2) for m in marks},
        }
    return out


def residual_and_extreme_risk(sessions_by_period: dict) -> dict:
    """What is left after a given slot, and how often a new daily extreme still arrives."""
    out = {}
    for name, sessions in sessions_by_period.items():
        rows = {}
        for i, slot in enumerate(SLOTS):
            residual, new_extreme, new_high, new_low = [], 0, 0, 0
            for session in sessions:
                path = completion_path(session["placed"], session["range"])
                residual.append(100.0 - path[i])
                later = [bar for slot_bars in session["placed"][i + 1:] for bar in slot_bars]
                if not later:
                    continue
                made_high = max((bar["h"] for bar in later), default=-math.inf) >= session["high"]
                made_low = min((bar["l"] for bar in later), default=math.inf) <= session["low"]
                new_high += int(made_high)
                new_low += int(made_low)
                new_extreme += int(made_high or made_low)
            n = len(sessions)
            rows[slot] = {
                "residual_range_mean_pct": round(statistics.fmean(residual), 2),
                "residual_range_median_pct": round(statistics.median(residual), 2),
                "new_extreme_after_pct": round(100.0 * new_extreme / n, 1),
                "new_high_after_pct": round(100.0 * new_high / n, 1),
                "new_low_after_pct": round(100.0 * new_low / n, 1),
                "n": n,
            }
        out[name] = rows
    return out


def volume_mechanism_screen(sessions_by_period: dict) -> dict:
    """If the profile is session participation, cumulative volume must track cumulative range."""
    out = {}
    for name, sessions in sessions_by_period.items():
        comp = mean_profile(sessions, lambda s: completion_path(s["placed"], s["range"]))
        vol = mean_profile(sessions, lambda s: volume_path(s["placed"], s["volume"]))
        gaps = {slot: round(comp[i] - vol[i], 2) for i, slot in enumerate(SLOTS)}
        largest = max(gaps.items(), key=lambda kv: abs(kv[1]))
        comp_inc, vol_inc = increments(comp), increments(vol)
        inc_gaps = {slot: round(comp_inc[i] - vol_inc[i], 2) for i, slot in enumerate(SLOTS)}
        inc_largest = max(inc_gaps.items(), key=lambda kv: abs(kv[1]))
        out[name] = {
            "spearman_share_vs_volume_share_per_slot": spearman(comp_inc, vol_inc),
            "largest_per_slot_share_gap_slot": inc_largest[0],
            "largest_per_slot_share_gap_pct": inc_largest[1],
            "per_slot_share_gap_pct": inc_gaps,
            "largest_gap_slot": largest[0],
            "largest_gap_pct": largest[1],
            "mean_abs_gap_pct": round(statistics.fmean(abs(v) for v in gaps.values()), 2),
            "gap_pct": gaps,
        }
    return out


def dst_confound(sessions: list[dict]) -> dict:
    groups = {"us_dst_on": [s for s in sessions if s["us_dst"]],
              "us_dst_off": [s for s in sessions if not s["us_dst"]]}
    out = {}
    for name, group in groups.items():
        if not group:
            continue
        comp = mean_profile(group, lambda s: completion_path(s["placed"], s["range"]))
        gains = [round(comp[i] - (comp[i - 1] if i else 0.0), 2) for i in range(len(SLOTS))]
        ranked = sorted(zip(SLOTS, gains), key=lambda kv: kv[1], reverse=True)
        out[name] = {
            "days": len(group),
            "range_completed_mean_pct": dict(zip(SLOTS, comp)),
            "per_slot_gain_pct": dict(zip(SLOTS, gains)),
            "busiest_slots": [slot for slot, _ in ranked[:5]],
        }
    if len(out) == 2:
        a = list(out["us_dst_on"]["range_completed_mean_pct"].values())
        b = list(out["us_dst_off"]["range_completed_mean_pct"].values())
        out["increment_spearman_on_vs_off"] = spearman(increments(a), increments(b))
        out["max_abs_level_gap_pct"] = round(max(abs(x - y) for x, y in zip(a, b)), 2)
    return out


def us_clock_alignment(sessions: list[dict]) -> dict:
    """Does the busiest half hour sit still on the Taipei clock, or on the New York clock?

    Taipei has no DST; New York does. If the range peak is driven by the 08:30 ET release
    window it must move one hour on a fixed Taipei clock and stand still on an ET clock.
    Re-aggregating the same sessions on each clock answers it directly: whichever clock
    produces the sharper peak is the one the market is actually keeping.
    """
    taipei_gain = collections.defaultdict(list)
    et_gain = collections.defaultdict(list)
    for session in sessions:
        gains = increments(completion_path(session["placed"], session["range"]))
        offset = 12 if session["us_dst"] else 13     # Taipei hours ahead of New York
        for i, slot in enumerate(SLOTS):
            taipei_gain[slot].append(gains[i])
            hour, minute = int(slot[:2]), int(slot[3:])
            et_gain[f"{(hour - offset) % 24:02d}:{minute:02d}"].append(gains[i])

    def summarise(store):
        means = {slot: round(statistics.fmean(values), 2) for slot, values in store.items()}
        ranked = sorted(means.items(), key=lambda kv: -kv[1])
        return means, ranked

    taipei_means, taipei_rank = summarise(taipei_gain)
    et_means, et_rank = summarise(et_gain)

    # The 07:00 session boundary absorbs the overnight and weekend gap, so it is the
    # largest slot on the Taipei clock by construction and it splits across two ET slots.
    # The clock question is only meaningful inside the US window, with that slot excluded.
    us_window_taipei = [f"{h:02d}:{m:02d}" for h in list(range(18, 24)) + [0, 1]
                        for m in (0, 30)]
    us_window_et = [f"{h:02d}:{m:02d}" for h in range(6, 14) for m in (0, 30)]
    taipei_us = sorted(((slot, taipei_means[slot]) for slot in us_window_taipei
                        if slot in taipei_means), key=lambda kv: -kv[1])
    et_us = sorted(((slot, et_means[slot]) for slot in us_window_et if slot in et_means),
                   key=lambda kv: -kv[1])

    dst_on = [s for s in sessions if s["us_dst"]]
    dst_off = [s for s in sessions if not s["us_dst"]]

    def slot_share(group, slot):
        index = SLOTS.index(slot)
        return round(statistics.fmean(
            increments(completion_path(g["placed"], g["range"]))[index] for g in group), 2)

    return {
        "us_window_taipei_clock_top": [{"slot": s, "mean_share_pct": v} for s, v in taipei_us[:4]],
        "us_window_new_york_clock_top": [{"slot": s, "mean_share_pct": v} for s, v in et_us[:4]],
        "us_window_peak_sharpened_by_et_alignment_pct":
            round(et_us[0][1] - taipei_us[0][1], 2),
        "et_0830_release_slot": {
            "us_dst_on": {"taipei_slot": "20:30", "mean_share_pct": slot_share(dst_on, "20:30")},
            "us_dst_off": {"taipei_slot": "21:30", "mean_share_pct": slot_share(dst_off, "21:30")},
            "same_slot_in_the_other_regime": {
                "20:30_when_dst_off": slot_share(dst_off, "20:30"),
                "21:30_when_dst_on": slot_share(dst_on, "21:30"),
            },
        },
        "taipei_clock_top_slots_all_day": [{"slot": s, "mean_share_pct": v} for s, v in taipei_rank[:5]],
        "new_york_clock_top_slots_all_day": [{"slot": s, "mean_share_pct": v} for s, v in et_rank[:5]],
        "all_day_ranking_caveat": "07:00 Taipei is the session boundary and absorbs the "
                                  "overnight/weekend gap; it splits across two ET slots, so "
                                  "the all-day ranking cannot decide the clock question.",
        "new_york_clock_mean_share_pct": et_means,
        "note": "ET 08:30 is the standard US macro release time; ET 09:30 is the NYSE open.",
    }


def weekday_profiles(sessions: list[dict]) -> dict:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    out = {}
    reference = mean_profile(sessions, lambda s: completion_path(s["placed"], s["range"]))
    for index, label in enumerate(names):
        group = [s for s in sessions if s["weekday"] == index]
        if not group:
            continue
        comp = mean_profile(group, lambda s: completion_path(s["placed"], s["range"]))
        out[label] = {
            "days": len(group),
            "completed_by_09_30_pct": comp[SLOTS.index("09:30")],
            "completed_by_15_30_pct": comp[SLOTS.index("15:30")],
            "completed_by_21_30_pct": comp[SLOTS.index("21:30")],
            "increment_spearman_vs_all_days": spearman(increments(comp), increments(reference)),
            "mean_daily_range_usd": round(statistics.fmean(s["range"] for s in group), 2),
        }
    return out


def stationarity(sessions: list[dict]) -> dict:
    """Shape is stable; level is not. This quantifies both."""
    by_quarter: dict[str, list[dict]] = collections.defaultdict(list)
    for session in sessions:
        by_quarter[f"{session['date'].year}Q{(session['date'].month - 1) // 3 + 1}"].append(session)
    marks = ["09:30", "15:30", "21:30", "23:30"]
    quarters = {}
    for label in sorted(by_quarter):
        group = by_quarter[label]
        comp = mean_profile(group, lambda s: completion_path(s["placed"], s["range"]))
        quarters[label] = {
            "days": len(group),
            "mean_daily_range_usd": round(statistics.fmean(s["range"] for s in group), 2),
            **{f"completed_by_{mark.replace(':', '')}_pct": comp[SLOTS.index(mark)]
               for mark in marks},
        }
    labels = sorted(quarters)
    series = [quarters[label]["completed_by_0930_pct"] for label in labels]
    ranges = [quarters[label]["mean_daily_range_usd"] for label in labels]
    increases = sum(1 for a, b in zip(series, series[1:]) if b > a)

    # Screen: is this a clock that moved, or a volatility regime that grew? The quarterly
    # trend confounds the two completely, because gold's daily range went from $25 to $174
    # over the same span. Sorting sessions by their own range instead of by date separates
    # them: if the Asian share rises with range *within* every period, the driver is
    # volatility, and the calendar trend is that relationship read through time.
    index_0930 = SLOTS.index("09:30")
    scored = sorted(((s["range"], completion_path(s["placed"], s["range"])[index_0930], s)
                     for s in sessions), key=lambda row: row[0])
    cut = len(scored) // 3
    tercile_rows = {}
    for label, group in (("small_range_days", scored[:cut]),
                         ("middle", scored[cut:2 * cut]),
                         ("large_range_days", scored[2 * cut:])):
        tercile_rows[label] = {
            "n": len(group),
            "median_daily_range_usd": round(statistics.median(row[0] for row in group), 2),
            "mean_completed_by_0930_pct": round(statistics.fmean(row[1] for row in group), 2),
        }
    early, late = sessions[:len(sessions) // 2], sessions[len(sessions) // 2:]

    def half_by_tercile(group):
        rows = sorted(((s["range"], completion_path(s["placed"], s["range"])[index_0930])
                       for s in group), key=lambda row: row[0])
        third = len(rows) // 3
        return {
            "small": round(statistics.fmean(r[1] for r in rows[:third]), 2),
            "large": round(statistics.fmean(r[1] for r in rows[2 * third:]), 2),
        }

    return {
        "by_quarter": quarters,
        "volatility_confound_screen": {
            "spearman_quarter_asia_share_vs_quarter_mean_range": spearman(ranges, series),
            "by_daily_range_tercile": tercile_rows,
            "first_half_of_sample": half_by_tercile(early),
            "second_half_of_sample": half_by_tercile(late),
            "reading": "the calendar trend and the range-tercile effect are the same effect "
                       "if the small/large gap holds inside both halves of the sample",
        },
        "asia_share_first_vs_last_quarter_pct": round(series[-1] - series[0], 2),
        "asia_share_quarter_over_quarter_increases": f"{increases}/{len(series) - 1}",
        "spearman_asia_share_vs_time": spearman(list(range(len(series))), series),
    }


def morning_conditioning(sessions: list[dict]) -> dict:
    """Given a quiet or busy Asian morning, what does the rest of the day do?

    Both sides are normalised by the trailing 20-session median range, so this measures
    volatility persistence rather than the level of the volatility regime.
    """
    index_0930 = SLOTS.index("09:30")
    records = []
    history: collections.deque = collections.deque(maxlen=20)
    for session in sessions:
        if len(history) == 20:
            baseline = statistics.median(history)
            if baseline > 0:
                path = completion_path(session["placed"], session["range"])
                morning_usd = session["range"] * path[index_0930] / 100.0
                records.append({
                    "morning_ratio": morning_usd / baseline,
                    "day_ratio": session["range"] / baseline,
                    "rest_ratio": (session["range"] - morning_usd) / baseline,
                })
        history.append(session["range"])
    if len(records) < 30:
        return {"n": len(records), "note": "insufficient sessions after warm-up"}
    ordered = sorted(records, key=lambda r: r["morning_ratio"])
    cut = len(ordered) // 3
    buckets = {"quiet_morning": ordered[:cut],
               "middle": ordered[cut:2 * cut],
               "busy_morning": ordered[2 * cut:]}
    out = {"n": len(records),
           "definition": "morning = 07:00-09:30 Taipei; ratios are vs the trailing 20-session median daily range"}
    for label, group in buckets.items():
        out[label] = {
            "n": len(group),
            "median_morning_ratio": round(statistics.median(r["morning_ratio"] for r in group), 3),
            "median_day_ratio": round(statistics.median(r["day_ratio"] for r in group), 3),
            "median_rest_of_day_ratio": round(statistics.median(r["rest_ratio"] for r in group), 3),
        }
    out["spearman_morning_vs_rest_of_day"] = spearman(
        [r["morning_ratio"] for r in records], [r["rest_ratio"] for r in records])
    out["spearman_morning_vs_full_day"] = spearman(
        [r["morning_ratio"] for r in records], [r["day_ratio"] for r in records])
    return out


def shape_stability(observed: dict) -> dict:
    keys = ["train", "valid", "holdout"]
    out = {}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            x = list(observed[keys[a]]["range_completed_mean_pct"].values())
            y = list(observed[keys[b]]["range_completed_mean_pct"].values())
            out[f"{keys[a]}_vs_{keys[b]}_increment_spearman"] = spearman(
                increments(x), increments(y))
            out[f"{keys[a]}_vs_{keys[b]}_max_abs_level_gap_pct"] = round(
                max(abs(i - j) for i, j in zip(x, y)), 2)
    return out


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, default=BARS_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    bars = load_bars(args.bars)
    sessions, dropped = build_sessions(bars)
    bounds = splits(len(sessions))
    by_period = {name: sessions[a:b] for name, (a, b) in bounds.items()}

    observed = observed_profiles(by_period)
    nulls = null_comparison(by_period, stream("null"))

    results = {
        "study_id": STUDY_ID,
        "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "market": "XAUUSD",
        "strategy": "none — price-series structure, independent of S1 and S2",
        "method": {
            "bars": "30-minute spot, Asia/Taipei clock",
            "day_boundary_hour_taipei": DAY_BOUNDARY_HOUR,
            "session_completeness_rule": f">= {MIN_BARS_PER_DAY} bars (a full session is 46)",
            "slots": len(SLOTS),
            "empty_slot_rule": "carry the running range forward; the maintenance break adds nothing",
            "split": {"train_end": TRAIN_END, "valid_end": VALID_END, "unit": "session index"},
            "null_shuffles": NULL_SHUFFLES,
            "seed": RANDOM_SEED,
        },
        "coverage": {
            "sessions_used": len(sessions),
            "sessions_dropped": dropped,
            "first_session": str(sessions[0]["date"]),
            "last_session": str(sessions[-1]["date"]),
            "period_days": {name: len(group) for name, group in by_period.items()},
            "mean_daily_range_usd": round(statistics.fmean(s["range"] for s in sessions), 2),
        },
        "families": {
            "observed_profile": observed,
            "vs_shuffled_returns_null": nulls,
            "increment_consistency": increment_consistency(nulls),
            "rejected_null_bar_permutation": rejected_null_bar_permutation(
                by_period, stream("rejected")),
            "residual_and_extreme_risk": residual_and_extreme_risk(by_period),
            "volume_mechanism_screen": volume_mechanism_screen(by_period),
            "dst_confound": dst_confound(sessions),
            "us_clock_alignment": us_clock_alignment(sessions),
            "weekday_profiles": weekday_profiles(sessions),
            "stationarity": stationarity(sessions),
            "morning_conditioning": morning_conditioning(sessions),
            "shape_stability": shape_stability(observed),
        },
        "limitations": [
            "Descriptive, not predictive, and never directional. The profile says how much "
            "range has accumulated, never which way price went or will go.",
            "The level of the profile is not stationary. The Asian share has risen across the "
            "sample, so any fixed threshold read off the pooled profile will be wrong in the "
            "direction of under-stating the morning.",
            "Taipei has no DST and the US session does, so the fixed-clock profile blurs the "
            "US block by one hour. The two regimes are reported separately for that reason.",
            "The daily boundary is 07:00 Taipei. A different boundary redistributes the "
            "profile; the null is recomputed with the same boundary, so the excess is "
            "unaffected, but raw levels are not comparable across boundary choices.",
            "Volume is TradingView spot tick volume, not exchange volume. "
            "RS-XAUUSD-20260818-003 established it rank-correlates 0.873 with real COMEX "
            "volume, which is adequate for a profile screen and not for a volume rule.",
            "One instrument, 652 sessions, one strong uptrend.",
            "No result changes formal S1 or S2 logic, live risk, or an entry checklist.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    residual = results["families"]["residual_and_extreme_risk"]["holdout"]
    print(json.dumps({
        "study_id": STUDY_ID,
        "sessions": len(sessions),
        "family_permutation_p": {k: v["family_permutation_p"] for k, v in nulls.items()},
        "holdout_completed_by": {
            slot: observed["holdout"]["range_completed_mean_pct"][slot]
            for slot in ("09:30", "15:30", "21:30", "23:30", "02:30")},
        "holdout_residual_after_02_30_pct": residual["02:30"]["residual_range_mean_pct"],
        "asia_share_drift_pct":
            results["families"]["stationarity"]["asia_share_first_vs_last_quarter_pct"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
