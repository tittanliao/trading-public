#!/usr/bin/env python3
"""RS-XAUUSD-20260818-002 — entry-side volume/volatility structure and cross-strategy interaction.

Two families, both entirely negative, both worth recording because each closes a search
space that looks obviously promising from the outside.

**Volume and realized volatility at entry.** `RS-XAUUSD-20260817-001` found that GVZ and
VIX — external, daily, lagged volatility proxies — separate nothing. The natural next
thought is that the proxies were the problem, not volatility, and that a volatility measure
computed from the price series itself would do better. It does not. Ten features covering
volume level, volume trend, bar range against ATR, ATR level, ATR percentile, bar anatomy,
position in the recent range and short-horizon momentum all sit inside their own resolution
limits, and a permutation test across the whole feature set reproduces the largest observed
spread about 60% of the time.

**Cross-strategy interaction.** S1 and S2 run on the same instrument, are both long-only,
and overlap constantly. Whether one is in position, how the last one resolved, and whether
the other fired recently are all things a portfolio-level rule could use. None of them
separates anything either.

## Look-ahead

Entry fills are intrabar: the fill price sits inside the entry bar's high-low range at a
median position of 0.44 (S1) and 0.50 (S2), matching neither its open nor its close. The
entry bar's OHLC is therefore *not* known when the order fills, and every feature here is
computed from bars at index `entry_index - 1` or earlier — the last fully completed bar
before the fill. Using the entry bar itself would have been the single easiest way to
manufacture an edge in this study.

## Multiple comparisons

Testing ten features and reporting the best one is a search, and a search finds something
on unrelated data. The correction is a permutation over the *whole* set: shuffle outcomes,
recompute every feature's tercile spread against fixed group membership, take the largest,
and repeat. The reported p-value is for the largest spread found anywhere in the search,
not for the feature that happened to win.

Usage:
    python3.12 -m scripts.research.build_xauusd_entry_context
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


ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "RS-XAUUSD-20260818-002"
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

FEATURES = [
    ("vol_ratio_20", "bar volume against the prior 20-bar median"),
    ("vol_ratio_96", "bar volume against the prior 96-bar (2-day) median"),
    ("range_over_atr", "bar range against ATR(14)"),
    ("atr_pct", "ATR(14) as a percentage of price"),
    ("atr_percentile_240", "ATR(14) percentile over the trailing 240 bars"),
    ("body_ratio", "candle body as a share of its range"),
    ("lower_wick_ratio", "lower wick as a share of its range"),
    ("range_position_48", "close within the trailing 48-bar high-low range"),
    ("ret_4bar_pct", "two-hour return into the signal"),
    ("volume_trend", "last 4 bars' median volume against the prior 20"),
]

WARMUP_BARS = 250
PERMUTATION_TRIALS = 20000
RANDOM_SEED = 20260818


def load_bars() -> list[dict]:
    rows = []
    with BARS_FILE.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                {
                    "t": datetime.fromisoformat(record["time"]).astimezone(TAIPEI),
                    "o": float(record["open"]),
                    "h": float(record["high"]),
                    "l": float(record["low"]),
                    "c": float(record["close"]),
                    "v": float(record["Volume"]),
                }
            )
    rows.sort(key=lambda bar: bar["t"])
    return rows


def load_trades(path: Path) -> list[dict]:
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
            else:
                trade["exit_at"] = stamp
                trade["exit_price"] = float(record["Price USD"])
                trade["exit_signal"] = record["Signal"]
            trade["return_pct"] = float(record["Return %"])
    complete = [
        trade
        for _, trade in sorted(trades.items())
        if {"entry_at", "exit_at"} <= set(trade) and trade.get("exit_signal") != "Open"
    ]
    complete.sort(key=lambda trade: trade["entry_at"])
    return complete


def true_range(bars: list[dict], index: int) -> float:
    if index == 0:
        return bars[index]["h"] - bars[index]["l"]
    previous_close = bars[index - 1]["c"]
    return max(
        bars[index]["h"] - bars[index]["l"],
        abs(bars[index]["h"] - previous_close),
        abs(bars[index]["l"] - previous_close),
    )


def atr(bars: list[dict], index: int, length: int = 14) -> float:
    return statistics.fmean(true_range(bars, j) for j in range(index - length + 1, index + 1))


def compute_features(bars: list[dict], index: int) -> dict | None:
    """Features from bar `index`, which must be the last COMPLETED bar before the fill."""
    if index < WARMUP_BARS:
        return None
    bar = bars[index]
    span = bar["h"] - bar["l"]
    if span <= 0:
        return None
    volume_20 = statistics.median(bars[j]["v"] for j in range(index - 20, index))
    volume_96 = statistics.median(bars[j]["v"] for j in range(index - 96, index))
    if not volume_20 or not volume_96:
        return None
    current_atr = atr(bars, index)
    history = [atr(bars, k) for k in range(index - 240, index + 1)]
    window = bars[index - 47 : index + 1]
    high = max(item["h"] for item in window)
    low = min(item["l"] for item in window)
    if high <= low or not current_atr:
        return None
    return {
        "vol_ratio_20": bar["v"] / volume_20,
        "vol_ratio_96": bar["v"] / volume_96,
        "range_over_atr": span / current_atr,
        "atr_pct": 100 * current_atr / bar["c"],
        "atr_percentile_240": 100 * sum(1 for x in history if x <= current_atr) / len(history),
        "body_ratio": abs(bar["c"] - bar["o"]) / span,
        "lower_wick_ratio": (min(bar["o"], bar["c"]) - bar["l"]) / span,
        "range_position_48": 100 * (bar["c"] - low) / (high - low),
        "ret_4bar_pct": 100 * (bar["c"] - bars[index - 4]["c"]) / bars[index - 4]["c"],
        "volume_trend": statistics.median(bars[j]["v"] for j in range(index - 3, index + 1))
        / volume_20,
    }


def attach_features(trades: list[dict], bars: list[dict]) -> list[dict]:
    times = [bar["t"] for bar in bars]
    out = []
    for trade in trades:
        index = bisect.bisect_left(times, trade["entry_at"])
        if index >= len(bars) or bars[index]["t"] != trade["entry_at"]:
            continue
        features = compute_features(bars, index - 1)
        if features is None:
            continue
        trade["features"] = features
        out.append(trade)
    return out


def fill_convention(trades: list[dict], bars: list[dict]) -> dict:
    """Evidence for the look-ahead decision, recorded rather than asserted."""
    times = [bar["t"] for bar in bars]
    positions, inside, outside = [], 0, 0
    for trade in trades:
        index = bisect.bisect_left(times, trade["entry_at"])
        if index >= len(bars) or bars[index]["t"] != trade["entry_at"]:
            continue
        bar = bars[index]
        price = trade["entry_price"]
        if bar["l"] - 1e-6 <= price <= bar["h"] + 1e-6:
            inside += 1
            if bar["h"] > bar["l"]:
                positions.append((price - bar["l"]) / (bar["h"] - bar["l"]))
        else:
            outside += 1
    return {
        "fills_inside_entry_bar_range": inside,
        "fills_outside": outside,
        "median_position_in_bar": round(statistics.median(positions), 3) if positions else None,
        "conclusion": (
            "Fills are intrabar, so the entry bar's OHLC is not known at fill time. All "
            "features use bars at entry_index - 1 or earlier."
        ),
    }


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
    }


def feature_family(trades: list[dict], baseline: dict, rng: random.Random) -> dict:
    n = len(trades)
    outcomes = [1 if trade["return_pct"] > 0 else 0 for trade in trades]
    membership: dict[str, list[int]] = {}
    for name, _ in FEATURES:
        ordered = sorted(trade["features"][name] for trade in trades)
        first, second = ordered[n // 3], ordered[2 * n // 3]
        membership[name] = [
            0 if trade["features"][name] <= first
            else (1 if trade["features"][name] <= second else 2)
            for trade in trades
        ]

    rows = {}
    for name, description in FEATURES:
        groups = membership[name]
        buckets = {}
        for tercile, label in ((0, "low"), (1, "mid"), (2, "high")):
            indices = [i for i in range(n) if groups[i] == tercile]
            buckets[label] = metrics([trades[i]["return_pct"] for i in indices])
        spread = max(b["win_rate_pct"] for b in buckets.values()) - min(
            b["win_rate_pct"] for b in buckets.values()
        )
        smallest = min(b["n"] for b in buckets.values())
        needed = min_detectable_pp(smallest, baseline["win_rate_pct"] / 100)
        rows[name] = {
            "description": description,
            "terciles": buckets,
            "win_rate_spread_pp": round(spread, 2),
            "min_detectable_pp": needed,
            "separable": needed is not None and spread >= needed,
            "monotonic": (
                buckets["low"]["win_rate_pct"] <= buckets["mid"]["win_rate_pct"]
                <= buckets["high"]["win_rate_pct"]
            ) or (
                buckets["low"]["win_rate_pct"] >= buckets["mid"]["win_rate_pct"]
                >= buckets["high"]["win_rate_pct"]
            ),
        }

    observed = max(row["win_rate_spread_pp"] for row in rows.values())
    winner = max(rows, key=lambda key: rows[key]["win_rate_spread_pp"])
    shuffled = outcomes[:]
    hits = 0
    for _ in range(PERMUTATION_TRIALS):
        rng.shuffle(shuffled)
        largest = 0.0
        for name, _ in FEATURES:
            groups = membership[name]
            wins = [0, 0, 0]
            counts = [0, 0, 0]
            for i in range(n):
                wins[groups[i]] += shuffled[i]
                counts[groups[i]] += 1
            rates = [100 * wins[g] / counts[g] for g in range(3)]
            largest = max(largest, max(rates) - min(rates))
        if largest >= observed:
            hits += 1
    p_value = hits / PERMUTATION_TRIALS
    return {
        "features": rows,
        "family_permutation_test": {
            "trials": PERMUTATION_TRIALS,
            "features_searched": len(FEATURES),
            "largest_observed_spread_pp": round(observed, 2),
            "largest_spread_feature": winner,
            "p_largest_spread_at_least_observed": round(p_value, 4),
            "any_feature_separates": p_value < 0.05,
            "note": (
                "The p-value covers the largest spread found anywhere in the search, not "
                "the feature that happened to win it."
            ),
        },
    }


def cross_strategy(target: list[dict], other: list[dict], target_name: str,
                   other_name: str, baseline: dict) -> dict:
    def summarise(groups: dict[str, list[float]]) -> dict:
        out = {}
        for key, values in groups.items():
            if len(values) < 5:
                out[key] = {"n": len(values), "too_small": True}
                continue
            row = metrics(values)
            needed = min_detectable_pp(row["n"], baseline["win_rate_pct"] / 100)
            row["min_detectable_pp"] = needed
            row["separable"] = (
                needed is not None
                and abs(row["win_rate_pct"] - baseline["win_rate_pct"]) >= needed
            )
            out[key] = row
        return out

    concurrent: dict[str, list[float]] = {"other_in_position": [], "other_flat": []}
    for trade in target:
        overlapping = any(
            item["entry_at"] <= trade["entry_at"] <= item["exit_at"] for item in other
        )
        concurrent["other_in_position" if overlapping else "other_flat"].append(
            trade["return_pct"]
        )

    last_result: dict[str, list[float]] = {"after_other_win": [], "after_other_loss": [],
                                           "no_prior_other": []}
    for trade in target:
        prior = [item for item in other if item["exit_at"] < trade["entry_at"]]
        if not prior:
            last_result["no_prior_other"].append(trade["return_pct"])
            continue
        latest = max(prior, key=lambda item: item["exit_at"])
        key = "after_other_win" if latest["return_pct"] > 0 else "after_other_loss"
        last_result[key].append(trade["return_pct"])

    nearby: dict[str, list[float]] = {"other_signal_within_48h": [], "no_other_signal_48h": []}
    for trade in target:
        near = any(
            abs((item["entry_at"] - trade["entry_at"]).total_seconds()) <= 48 * 3600
            for item in other
        )
        nearby["other_signal_within_48h" if near else "no_other_signal_48h"].append(
            trade["return_pct"]
        )

    return {
        "target": target_name,
        "other": other_name,
        "by_concurrent_position": summarise(concurrent),
        "by_last_other_outcome": summarise(last_result),
        "by_nearby_other_signal": summarise(nearby),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    bars = load_bars()
    raw = {name: load_trades(config["trades"]) for name, config in STRATEGIES.items()}

    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(tz=TAIPEI).isoformat(timespec="seconds"),
        "strategy": "S1 AweWithBB V3.9; S2 Hammer V3.2",
        "method": {
            "bar_source": "30-minute FX_IDC:XAUUSD export",
            "bar_count": len(bars),
            "feature_bar": "entry_index - 1, the last fully completed bar before the fill",
            "warmup_bars": WARMUP_BARS,
            "permutation_trials": PERMUTATION_TRIALS,
            "random_seed": RANDOM_SEED,
            "timezone": "Asia/Taipei",
        },
        "strategies": {},
    }

    for name, config in STRATEGIES.items():
        trades = attach_features(list(raw[name]), bars)
        baseline = metrics([trade["return_pct"] for trade in trades])
        other_name = "S2" if name == "S1" else "S1"
        results["strategies"][name] = {
            "label": config["label"],
            "trades_with_features": len(trades),
            "trades_total": len(raw[name]),
            "baseline": baseline,
            "fill_convention": fill_convention(raw[name], bars),
            "by_entry_feature": feature_family(trades, baseline, rng),
            "cross_strategy": cross_strategy(
                raw[name], raw[other_name], name, other_name, baseline
            ),
        }

    results["limitations"] = [
        "Terciles are a coarse split. A feature whose effect lives only in a narrow tail "
        "would be diluted here, though no feature shows a monotonic gradient that would "
        "suggest one.",
        "Volume on a spot FX/CFD feed is broker tick activity, not exchange volume, so its "
        "level is not comparable across venues; only its ratio to its own recent history is "
        "used, which is the weaker but defensible reading. RESOLVED 2026-08-18 by "
        "RS-XAUUSD-20260818-003: real COMEX contract volume rank-correlates with this "
        "measure at 0.873 across 30,734 aligned bars and separates nothing either, so the "
        "proxy was adequate and this limitation does not reopen the volume question.",
        "Cross-strategy tests treat S1 and S2 as independent series. Overlapping holding "
        "periods mean the same market conditions appear in both, so these groups are not "
        "independent samples.",
        "Both strategies are long-only over a strongly rising market, so any feature "
        "correlated with the uptrend has limited room to distinguish itself.",
        "No result changes formal S1 or S2 logic, live risk, or an active entry checklist.",
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "study_id": STUDY_ID,
        "output": str(args.output / "results.json"),
        "strategies": {
            name: {
                "n": block["trades_with_features"],
                "largest_spread_pp": block["by_entry_feature"]["family_permutation_test"][
                    "largest_observed_spread_pp"],
                "family_p": block["by_entry_feature"]["family_permutation_test"][
                    "p_largest_spread_at_least_observed"],
                "any_separates": block["by_entry_feature"]["family_permutation_test"][
                    "any_feature_separates"],
            }
            for name, block in results["strategies"].items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
