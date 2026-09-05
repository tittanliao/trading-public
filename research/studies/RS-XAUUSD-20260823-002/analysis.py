#!/usr/bin/env python3
"""RS-XAUUSD-20260823-002 — S1 V3.9's Bollinger %B at entry, put through every screen.

This candidate arrived differently from the others in the programme. An unprompted
impression held that S1 signals near the top of the band win more often; V3.9's own Pine carries
a disabled `use_bb_filter` whose tooltip records "above_upper (%B>100%) 勝率 77.8%"; and
RS-XAUUSD-20260727-001's `bb_zone` table shows a monotone gradient topping at 79.31%. Three
independent statements of the same thing is a reason to test it properly, not a reason to
believe it — RS-XAUUSD-20260819-001 found four of six sign-consistent candidates were
artefacts, and its `policy_impacts` were revoked wholesale on 2026-08-17.

## What is different here from the table that already exists

`-20260727-001` joined %B for only 165 of its 450 trades. This run joins all 472 on the
2026-08-15 export, so the headline is computed on nearly three times the evidence — and a
bucket that shrinks under more data is exactly what a small-sample artefact does.

## The screens, and what each one is for

- **Baseline, not zero.** The instrument rose over the sample; every bucket is scored as
  excess over the same period's own baseline.
- **Out-of-sample.** Train/valid/holdout by time. An ordering that only holds in-sample is
  the slot matrix again.
- **Family permutation.** Seven buckets are seven chances; outcomes are shuffled against
  the fixed bucketing and the largest |t| is kept.
- **Moving-block bootstrap.** Trades cluster in time and an iid resample treats a run of
  correlated trades as independent evidence.
- **Dose-response.** If the mechanism is "further above the band is a cleaner breakout",
  %B must predict continuously, not only at one bucket edge. A step that appears at exactly
  the boundary and nowhere else is a bucketing artefact.
- **Momentum ablation.** %B above 1 means price closed outside the band, which is itself a
  strong move. If the effect only exists in the strongest momentum tercile, it belongs to
  momentum and not to the band.
- **The tradeable question, answered in return and not in win rate.**
  RS-XAUUSD-20260818-001 established that win rate is purchasable and worth nothing alone:
  a take-profit lifted S1 to 88.6% while cutting return from +88.6% to between +14% and
  +36%. A filter that raises win rate by discarding trades must therefore be reported as
  total return, profit factor and opportunity cost, or it is being reported misleadingly.

Usage:
    python3.12 -m scripts.research.build_s1_bb_position
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import fail_pattern_toolkit as tk  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260823-002"
OUTPUT_DIR = Path("reproduced")
TRADES_FILE = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv")
BARS_FILE = Path("local-inputs/FX_IDC_XAUUSD, 30_volumn.csv")
TAIPEI = timezone(timedelta(hours=8))

TRAIN_END, VALID_END = 0.55, 0.80
PERMUTATIONS = 4000
BOOTSTRAP = 4000
BLOCK = 10
SEED = 20260823
# Round-trip cost as a percentage of notional, matching RS-XAUUSD-20260819-001.
COST_PCT = 0.02
MOMENTUM_BARS = 6

ZONE_ORDER = tk.BB_ZONE_ORDER


def rng(label: str) -> random.Random:
    return random.Random(f"{SEED}:{label}")


def block_of(frame: pd.DataFrame) -> dict:
    wins = int(frame["win"].sum())
    count = int(len(frame))
    gains = frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
    losses = -frame.loc[frame["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
    return {
        "n": count,
        "wins": wins,
        "win_rate_pct": round(100 * wins / count, 2) if count else None,
        "win_rate_ci95_pct": tk.wilson_interval(wins, count),
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
        "net_pnl_usd": round(float(frame["net_pnl_usd"].sum()), 2),
        "avg_return_pct": round(float(frame["return_pct"].mean()), 4) if count else None,
        "total_return_pct": round(float(frame["return_pct"].sum()), 2),
    }


def zone_table(frame: pd.DataFrame) -> dict:
    baseline = block_of(frame)
    out = {"baseline": baseline, "zones": {}}
    for zone in ZONE_ORDER:
        subset = frame[frame["bb_zone"] == zone]
        if subset.empty:
            continue
        row = block_of(subset)
        row["win_rate_excess_pct"] = round(
            row["win_rate_pct"] - baseline["win_rate_pct"], 2
        )
        row["avg_return_excess_pct"] = round(
            row["avg_return_pct"] - baseline["avg_return_pct"], 4
        )
        out["zones"][zone] = row
    return out


def resolvable_gap(count: int, other: int, rate: float) -> float | None:
    """Smallest win-rate difference this pair of samples could separate at 80% power."""
    if count < 2 or other < 2:
        return None
    variance = rate * (1 - rate)
    if variance <= 0:
        return None
    standard_error = math.sqrt(variance * (1 / count + 1 / other))
    return round(100 * 2.8 * standard_error, 1)


def period_split(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values("entry_time").reset_index(drop=True)
    count = len(ordered)
    bounds = {
        "train": (0, int(count * TRAIN_END)),
        "valid": (int(count * TRAIN_END), int(count * VALID_END)),
        "holdout": (int(count * VALID_END), count),
    }
    out = {}
    for name, (start, stop) in bounds.items():
        window = ordered.iloc[start:stop]
        table = zone_table(window)
        out[name] = {
            "n": len(window),
            "from": str(window["entry_time"].min()),
            "to": str(window["entry_time"].max()),
            "baseline_win_rate_pct": table["baseline"]["win_rate_pct"],
            "above_upper": table["zones"].get("above_upper"),
            "zone_win_rates_pct": {
                zone: row["win_rate_pct"] for zone, row in table["zones"].items()
            },
        }
    return out


def monotonicity(table: dict) -> dict:
    """Is the gradient ordered, and is the ordering itself unlikely by chance?"""
    present = [zone for zone in ZONE_ORDER if zone in table["zones"]]
    rates = [table["zones"][zone]["win_rate_pct"] for zone in present]
    counts = [table["zones"][zone]["n"] for zone in present]
    # Spearman against the bucket order, weighted by nothing: the question is ordering.
    ranks = list(range(len(rates)))
    mean_rank = statistics.fmean(ranks)
    mean_rate = statistics.fmean(rates)
    numerator = sum((a - mean_rank) * (b - mean_rate) for a, b in zip(ranks, rates))
    denominator = math.sqrt(
        sum((a - mean_rank) ** 2 for a in ranks) * sum((b - mean_rate) ** 2 for b in rates)
    )
    ascending = sum(1 for a, b in zip(rates, rates[1:]) if b >= a)
    return {
        "zones_in_order": present,
        "win_rates_pct": rates,
        "n_per_zone": counts,
        "spearman_vs_bucket_order": round(numerator / denominator, 3) if denominator else None,
        "ascending_steps": f"{ascending}/{len(rates) - 1}",
    }


def family_permutation(frame: pd.DataFrame, stream: random.Random) -> dict:
    """Seven buckets are seven chances. Shuffle outcomes, keep the largest |t|."""
    zones = frame["bb_zone"].to_numpy()
    wins = frame["win"].to_numpy().astype(float)
    baseline = wins.mean()

    def largest_abs_t(values: np.ndarray) -> float:
        best = 0.0
        for zone in ZONE_ORDER:
            mask = zones == zone
            count = int(mask.sum())
            if count < 5:
                continue
            rate = values[mask].mean()
            spread = math.sqrt(max(baseline * (1 - baseline), 1e-9) / count)
            best = max(best, abs((rate - baseline) / spread))
        return best

    observed = largest_abs_t(wins)
    null = []
    pool = wins.copy()
    for _ in range(PERMUTATIONS):
        stream.shuffle(pool)
        null.append(largest_abs_t(pool))
    exceed = sum(1 for value in null if value >= observed)
    return {
        "observed_max_abs_t": round(observed, 3),
        "null_median_max_abs_t": round(float(np.median(null)), 3),
        "permutations": PERMUTATIONS,
        "p_value": round((exceed + 1) / (PERMUTATIONS + 1), 4),
    }


def block_bootstrap(frame: pd.DataFrame, stream: random.Random) -> dict:
    """Above-upper versus the rest, resampled in blocks because trades cluster in time."""
    ordered = frame.sort_values("entry_time").reset_index(drop=True)
    flag = (ordered["bb_zone"] == "above_upper").to_numpy()
    wins = ordered["win"].to_numpy().astype(float)
    returns = ordered["return_pct"].to_numpy().astype(float)
    count = len(ordered)
    observed_wr = wins[flag].mean() - wins[~flag].mean()
    observed_ret = returns[flag].mean() - returns[~flag].mean()
    starts = max(count - BLOCK, 1)
    wr_draws, ret_draws = [], []
    for _ in range(BOOTSTRAP):
        index = []
        while len(index) < count:
            begin = stream.randrange(starts)
            index.extend(range(begin, min(begin + BLOCK, count)))
        index = np.array(index[:count])
        sample_flag, sample_win, sample_ret = flag[index], wins[index], returns[index]
        if sample_flag.sum() < 5 or (~sample_flag).sum() < 5:
            continue
        wr_draws.append(sample_win[sample_flag].mean() - sample_win[~sample_flag].mean())
        ret_draws.append(sample_ret[sample_flag].mean() - sample_ret[~sample_flag].mean())
    return {
        "observed_win_rate_gap_pct": round(100 * observed_wr, 2),
        "win_rate_gap_ci90_pct": [
            round(100 * float(np.percentile(wr_draws, 5)), 2),
            round(100 * float(np.percentile(wr_draws, 95)), 2),
        ],
        "win_rate_gap_p_above_zero": round(float(np.mean(np.array(wr_draws) > 0)), 4),
        "observed_avg_return_gap_pct": round(observed_ret, 4),
        "avg_return_gap_ci90_pct": [
            round(float(np.percentile(ret_draws, 5)), 4),
            round(float(np.percentile(ret_draws, 95)), 4),
        ],
        "block_size": BLOCK,
        "resamples": len(wr_draws),
    }


def dose_response(frame: pd.DataFrame) -> dict:
    """A real gradient is continuous. A step at exactly one boundary is a bucketing artefact."""
    ordered = frame.dropna(subset=["bb_pct_b"]).sort_values("bb_pct_b").reset_index(drop=True)
    edges = np.linspace(0, len(ordered), 11).astype(int)
    deciles = [ordered.iloc[a:b] for a, b in zip(edges, edges[1:])]
    rows = []
    for index, part in enumerate(deciles, start=1):
        if part.empty:
            continue
        rows.append(
            {
                "decile": index,
                "pct_b_range": [round(float(part["bb_pct_b"].min()), 3),
                                round(float(part["bb_pct_b"].max()), 3)],
                "n": int(len(part)),
                "win_rate_pct": round(100 * float(part["win"].mean()), 2),
                "avg_return_pct": round(float(part["return_pct"].mean()), 4),
            }
        )
    rates = [row["win_rate_pct"] for row in rows]
    ranks = list(range(len(rates)))
    mean_rank, mean_rate = statistics.fmean(ranks), statistics.fmean(rates)
    numerator = sum((a - mean_rank) * (b - mean_rate) for a, b in zip(ranks, rates))
    denominator = math.sqrt(
        sum((a - mean_rank) ** 2 for a in ranks) * sum((b - mean_rate) ** 2 for b in rates)
    )
    inside = ordered[ordered["bb_pct_b"] > 1.0]
    within = None
    if len(inside) >= 20:
        sorted_inside = inside.sort_values("bb_pct_b").reset_index(drop=True)
        middle = len(sorted_inside) // 2
        within = {
            "lower_half": block_of(sorted_inside.iloc[:middle]),
            "upper_half": block_of(sorted_inside.iloc[middle:]),
            "reading": "dose-response inside above_upper: does more distance keep helping?",
        }
    return {
        "by_pct_b_decile": rows,
        "spearman_win_rate_vs_pct_b_decile": round(numerator / denominator, 3)
        if denominator else None,
        "within_above_upper": within,
    }


def momentum_ablation(frame: pd.DataFrame) -> dict:
    """Closing outside the band is itself a strong move. Does the band add anything to it?"""
    subset = frame.dropna(subset=["entry_momentum_pct"])
    terciles = pd.qcut(subset["entry_momentum_pct"], 3, labels=["low", "mid", "high"])
    out = {}
    for label in ["low", "mid", "high"]:
        part = subset[terciles == label]
        above = part[part["bb_zone"] == "above_upper"]
        rest = part[part["bb_zone"] != "above_upper"]
        out[label] = {
            "momentum_range_pct": [round(float(part["entry_momentum_pct"].min()), 3),
                                   round(float(part["entry_momentum_pct"].max()), 3)],
            "above_upper": block_of(above) if len(above) else None,
            "rest": block_of(rest) if len(rest) else None,
            "win_rate_gap_pct": (
                round(100 * (above["win"].mean() - rest["win"].mean()), 2)
                if len(above) >= 5 and len(rest) >= 5 else None
            ),
        }
    gaps = [row["win_rate_gap_pct"] for row in out.values() if row["win_rate_gap_pct"] is not None]
    out["reading"] = (
        "the band survives momentum if the gap holds in every tercile; if it exists only in "
        "the high tercile the effect belongs to momentum"
    )
    out["gap_positive_in_all_terciles"] = bool(gaps) and all(gap > 0 for gap in gaps)
    return out


def threshold_sweep(frame: pd.DataFrame) -> dict:
    """The Pine's own use_bb_filter, reported in return rather than in win rate.

    V3.9 ships this filter disabled with bb_pct_b_min defaulting to 60. Enabling it discards
    trades, and discarding trades always buys win rate. The columns that decide whether it
    is worth anything are total return, profit factor and how many entries it costs.
    """
    baseline = block_of(frame)
    rows = []
    for threshold in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1]:
        kept = frame[frame["bb_pct_b"] >= threshold]
        if kept.empty:
            continue
        row = block_of(kept)
        net_after_cost = float((kept["return_pct"] - COST_PCT).sum())
        rows.append(
            {
                "bb_pct_b_min": threshold,
                "trades_kept": row["n"],
                "trades_kept_pct": round(100 * row["n"] / baseline["n"], 1),
                "win_rate_pct": row["win_rate_pct"],
                "profit_factor": row["profit_factor"],
                "total_return_pct": row["total_return_pct"],
                "total_return_pct_after_cost": round(net_after_cost, 2),
                "net_pnl_usd": row["net_pnl_usd"],
                "return_retained_pct": round(
                    100 * row["total_return_pct"] / baseline["total_return_pct"], 1
                ) if baseline["total_return_pct"] else None,
            }
        )
    best_return = max(rows, key=lambda row: row["total_return_pct"])
    best_wr = max(rows, key=lambda row: row["win_rate_pct"])
    return {
        "baseline": baseline,
        "cost_pct_round_trip": COST_PCT,
        "sweep": rows,
        "highest_total_return_at": best_return["bb_pct_b_min"],
        "highest_win_rate_at": best_wr["bb_pct_b_min"],
        "win_rate_and_return_agree": best_return["bb_pct_b_min"] == best_wr["bb_pct_b_min"],
    }


def build() -> dict:
    trades = tk.load_trades(TRADES_FILE)
    price, _ = tk.load_price_csv(BARS_FILE)
    enriched = tk.enrich_trades_with_bb(trades, price)

    # Momentum at entry: return over the prior MOMENTUM_BARS bars, joined the same way the
    # bands are, so the ablation and the headline share one alignment.
    momentum = price[["time", "close"]].copy().sort_values("time").reset_index(drop=True)
    momentum["entry_momentum_pct"] = 100 * (
        momentum["close"] / momentum["close"].shift(MOMENTUM_BARS) - 1
    )
    enriched = pd.merge_asof(
        enriched.sort_values("entry_time"),
        momentum[["time", "entry_momentum_pct"]],
        left_on="entry_time",
        right_on="time",
        direction="backward",
        suffixes=("", "_mom"),
    )

    table = zone_table(enriched)
    above = enriched[enriched["bb_zone"] == "above_upper"]
    rest = enriched[enriched["bb_zone"] != "above_upper"]
    gap_needed = resolvable_gap(len(above), len(rest), float(enriched["win"].mean()))

    return {
        "study_id": STUDY_ID,
        "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "market": "XAUUSD",
        "strategy": "S1 AweWithBB V3.9",
        "method": {
            "trades": str(TRADES_FILE.relative_to(ROOT)),
            "bars": str(BARS_FILE.relative_to(ROOT)),
            "bb_params": "period=20, std_mult=2.0, close, sample stdev (study convention)",
            "bb_note": (
                "V3.9's Pine computes its own bands on ohlc4 with a population stdev. The "
                "study convention is kept here so this result is comparable with "
                "RS-XAUUSD-20260727-001; the two differ slightly near a bucket boundary."
            ),
            "momentum_bars": MOMENTUM_BARS,
            "split": {"train_end": TRAIN_END, "valid_end": VALID_END, "unit": "trade index"},
            "permutations": PERMUTATIONS,
            "bootstrap": BOOTSTRAP,
            "block_size": BLOCK,
            "cost_pct_round_trip": COST_PCT,
            "seed": SEED,
        },
        "coverage": {
            "trades": int(len(enriched)),
            "bb_joined": int(enriched["bb_pct_b"].notna().sum()),
            "from": str(enriched["entry_time"].min()),
            "to": str(enriched["entry_time"].max()),
            "prior_study_bb_coverage": "165 of 450 trades (RS-XAUUSD-20260727-001)",
        },
        "families": {
            "zone_table": table,
            "monotonicity": monotonicity(table),
            "above_upper_vs_rest": {
                "above_upper": block_of(above),
                "rest": block_of(rest),
                "win_rate_gap_pct": round(
                    100 * float(above["win"].mean() - rest["win"].mean()), 2
                ),
                "smallest_resolvable_gap_pct": gap_needed,
                "gap_exceeds_resolution": (
                    gap_needed is not None
                    and 100 * float(above["win"].mean() - rest["win"].mean()) > gap_needed
                ),
            },
            "by_period": period_split(enriched),
            "family_permutation": family_permutation(enriched, rng("permute")),
            "block_bootstrap": block_bootstrap(enriched, rng("bootstrap")),
            "dose_response": dose_response(enriched),
            "momentum_ablation": momentum_ablation(enriched),
            "threshold_sweep": threshold_sweep(enriched),
        },
        "limitations": [
            "One strategy, one instrument, 472 trades over one strong uptrend.",
            "%B is computed from the study's close-based bands, not V3.9's ohlc4 bands; a "
            "trade sitting on a bucket boundary can fall either side of it.",
            "The filter sweep is applied to trades the strategy actually took. It cannot "
            "show what a differently-filtered strategy would have entered instead, so the "
            "retained-return column is an upper bound on the benefit, not a backtest.",
            "Entry momentum is a proxy built from the same price series, so the ablation "
            "bounds the confound rather than eliminating it.",
            "No result changes formal S1 logic, live risk, or an entry checklist without a "
            "separate adoption decision.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    results = build()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    families = results["families"]
    print(json.dumps({
        "study_id": STUDY_ID,
        "trades": results["coverage"]["trades"],
        "above_upper": families["above_upper_vs_rest"]["above_upper"],
        "gap_pct": families["above_upper_vs_rest"]["win_rate_gap_pct"],
        "needs_gap_pct": families["above_upper_vs_rest"]["smallest_resolvable_gap_pct"],
        "family_p": families["family_permutation"]["p_value"],
        "bootstrap_ci90": families["block_bootstrap"]["win_rate_gap_ci90_pct"],
        "best_return_threshold": families["threshold_sweep"]["highest_total_return_at"],
        "best_wr_threshold": families["threshold_sweep"]["highest_win_rate_at"],
    }, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
