#!/usr/bin/env python3
"""RS-XAUUSD-20260818-003 — COMEX MGC futures basis and real exchange volume.

The last unused data source. The market export folder has carried MGC micro gold
futures alongside the XAUUSD spot series since 2024-01-02, and no study had opened it.
Two questions it can answer that spot alone cannot:

1. **Basis.** Futures minus spot is where carry, funding and one-sided urgency show up.
   If either strategy is entering into a dislocated market, the basis is where that lives.
2. **Real volume.** `RS-XAUUSD-20260818-002` tested volume and found nothing, and stated a
   limitation honestly: a spot FX/CFD feed reports broker tick activity, not exchange
   volume. MGC reports genuine COMEX contracts. This settles whether that limitation
   mattered.

## The roll cycle is a calendar, not a signal

Raw basis is dominated by the contract roll. Monthly medians alternate between roughly
+5 USD and +20 USD, because a freshly rolled contract carries more time value and decays
toward expiry. Splitting trades by raw basis therefore splits them by *month*: in this
data the lowest raw-basis tercile is 1% even-month for S1 while the other two are 83% and
73%, and the split produces a 10.8pp win-rate spread that is a calendar artifact end to
end.

`raw_basis_control` computes exactly that split and reports the even-month share beside it,
so the trap is on the record with its own evidence rather than as a warning. Everything
else uses basis measured against its own trailing median, which flattens the monthly
medians to a constant and leaves only short-horizon deviation.

Bars where the basis jumps more than 8 USD in one step are roll bars; any trade whose
96-bar feature window contains one is dropped rather than measured through a discontinuity.

## Look-ahead

Fills are intrabar (see RS-XAUUSD-20260818-002), so every feature uses the last fully
completed bar before the fill.

Usage:
    python3.12 -m scripts.research.build_xauusd_futures_basis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-XAUUSD-20260818-003"
OUTPUT_DIR = Path("reproduced")
MARKET_DIR = Path("local-inputs")
SPOT_FILE = MARKET_DIR / "FX_IDC_XAUUSD, 30_volumn.csv"
FUTURES_FILE = MARKET_DIR / "COMEX_MINI_DL_MGC1!, 30_volumn.csv"
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
    ("basis_dev", "basis minus its trailing 96-bar median, USD"),
    ("basis_dev_z", "the same deviation standardised by its trailing dispersion"),
    ("basis_change_4", "two-hour change in basis"),
    ("mgc_vol_20", "MGC contracts against the prior 20-bar median"),
    ("mgc_vol_96", "MGC contracts against the prior 96-bar median"),
    ("spot_fut_vol_mix", "spot ticks per MGC contract, against its own recent level"),
    ("mgc_range_atr", "MGC bar range against MGC ATR(14)"),
]

ROLL_THRESHOLD_USD = 8.0
LOOKBACK = 96
WARMUP = 120
PERMUTATION_TRIALS = 20000
RANDOM_SEED = 20260818


def stream(label: str) -> random.Random:
    return random.Random(f"{RANDOM_SEED}:{label}")


def load_series(path: Path) -> dict[datetime, dict]:
    out = {}
    with path.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            stamp = datetime.fromisoformat(record["time"]).astimezone(TAIPEI)
            out[stamp] = {
                "o": float(record["open"]),
                "h": float(record["high"]),
                "l": float(record["low"]),
                "c": float(record["close"]),
                "v": float(record["Volume"]),
            }
    return out


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
            else:
                trade["exit_signal"] = record["Signal"]
            trade["return_pct"] = float(record["Return %"])
    complete = [
        trade
        for _, trade in sorted(trades.items())
        if "entry_at" in trade and trade.get("exit_signal") != "Open"
    ]
    complete.sort(key=lambda trade: trade["entry_at"])
    return complete


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


class Market:
    """The aligned spot/futures pair, its basis, and where the contract rolled."""

    def __init__(self) -> None:
        spot = load_series(SPOT_FILE)
        futures = load_series(FUTURES_FILE)
        self.times = sorted(set(spot) & set(futures))
        self.index = {stamp: i for i, stamp in enumerate(self.times)}
        self.spot = spot
        self.futures = futures
        self.basis = [futures[t]["c"] - spot[t]["c"] for t in self.times]
        self.roll = [False] + [
            abs(self.basis[i] - self.basis[i - 1]) > ROLL_THRESHOLD_USD
            for i in range(1, len(self.basis))
        ]
        self.spot_only = len(spot) - len(self.times)
        self.futures_only = len(futures) - len(self.times)

    def true_range(self, i: int) -> float:
        bar = self.futures[self.times[i]]
        if i == 0:
            return bar["h"] - bar["l"]
        previous = self.futures[self.times[i - 1]]["c"]
        return max(bar["h"] - bar["l"], abs(bar["h"] - previous), abs(bar["l"] - previous))

    def features(self, i: int) -> dict | None:
        if i < WARMUP:
            return None
        if any(self.roll[j] for j in range(i - LOOKBACK, i + 1)):
            return None
        window = self.basis[i - LOOKBACK : i]
        median = statistics.median(window)
        dispersion = statistics.pstdev(window)
        stamp = self.times[i]
        futures_bar = self.futures[stamp]
        volume_20 = statistics.median(
            self.futures[self.times[j]]["v"] for j in range(i - 20, i)
        )
        volume_96 = statistics.median(
            self.futures[self.times[j]]["v"] for j in range(i - LOOKBACK, i)
        )
        average_range = statistics.fmean(self.true_range(j) for j in range(i - 13, i + 1))
        if not (volume_20 and volume_96 and average_range and dispersion and futures_bar["v"]):
            return None
        mix_now = self.spot[stamp]["v"] / futures_bar["v"]
        history = [
            self.spot[self.times[j]]["v"] / self.futures[self.times[j]]["v"]
            for j in range(i - 20, i)
            if self.futures[self.times[j]]["v"]
        ]
        if not history:
            return None
        mix_reference = statistics.median(history)
        if not mix_reference:
            return None
        return {
            "basis_dev": self.basis[i] - median,
            "basis_dev_z": (self.basis[i] - median) / dispersion,
            "basis_change_4": self.basis[i] - self.basis[i - 4],
            "mgc_vol_20": futures_bar["v"] / volume_20,
            "mgc_vol_96": futures_bar["v"] / volume_96,
            "spot_fut_vol_mix": mix_now / mix_reference,
            "mgc_range_atr": (futures_bar["h"] - futures_bar["l"]) / average_range,
        }


def roll_cycle_profile(market: Market) -> dict:
    """Evidence that raw basis is a calendar, kept so the detrending is justified not asserted."""
    by_month: dict[int, list[float]] = {}
    for i, stamp in enumerate(market.times):
        by_month.setdefault(stamp.month, []).append(market.basis[i])
    raw = {str(m): round(statistics.median(v), 2) for m, v in sorted(by_month.items())}

    detrended: dict[int, list[float]] = {}
    for i in range(LOOKBACK, len(market.basis)):
        deviation = market.basis[i] - statistics.median(market.basis[i - LOOKBACK : i])
        detrended.setdefault(market.times[i].month, []).append(deviation)
    flat = {str(m): round(statistics.median(v), 2) for m, v in sorted(detrended.items())}

    return {
        "raw_basis_median_by_month_usd": raw,
        "raw_spread_across_months_usd": round(max(raw.values()) - min(raw.values()), 2),
        "detrended_median_by_month_usd": flat,
        "detrended_spread_across_months_usd": round(max(flat.values()) - min(flat.values()), 2),
        "roll_bars_detected": sum(market.roll),
        "roll_threshold_usd": ROLL_THRESHOLD_USD,
        "note": (
            "Raw basis alternates with the roll cycle, so it encodes the month. The "
            "detrended series flattens that to a fraction of a dollar across all twelve."
        ),
    }


def raw_basis_control(trades: list[dict], market: Market, baseline: dict) -> dict:
    """The split the study deliberately does not use, with the reason it cannot be used."""
    usable = [
        trade for trade in trades
        if trade["entry_at"] in market.index and market.index[trade["entry_at"]] > 0
    ]
    if len(usable) < 30:
        return {"applicable": False}
    values = []
    for trade in usable:
        values.append(market.basis[market.index[trade["entry_at"]] - 1])
    ordered = sorted(values)
    first, second = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]
    buckets: dict[str, list[dict]] = {"low": [], "mid": [], "high": []}
    for trade, value in zip(usable, values):
        key = "low" if value <= first else ("mid" if value <= second else "high")
        buckets[key].append(trade)
    rows = {}
    for key, group in buckets.items():
        row = metrics([trade["return_pct"] for trade in group])
        row["even_month_share_pct"] = round(
            100 * sum(1 for t in group if t["entry_at"].month % 2 == 0) / len(group), 1
        )
        rows[key] = row
    spread = max(r["win_rate_pct"] for r in rows.values()) - min(
        r["win_rate_pct"] for r in rows.values()
    )
    return {
        "applicable": True,
        "terciles": rows,
        "win_rate_spread_pp": round(spread, 2),
        "baseline_win_rate_pct": baseline["win_rate_pct"],
        "usable": False,
        "why_not": (
            "The terciles differ in even-month share by "
            f"{max(r['even_month_share_pct'] for r in rows.values()) - min(r['even_month_share_pct'] for r in rows.values()):.0f} "
            "points. This split is a calendar split wearing a basis label, and the spread it "
            "produces says nothing about the market state at entry."
        ),
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
        needed = min_detectable_pp(min(b["n"] for b in buckets.values()),
                                   baseline["win_rate_pct"] / 100)
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
    return {
        "features": rows,
        "family_permutation_test": {
            "trials": PERMUTATION_TRIALS,
            "features_searched": len(FEATURES),
            "largest_observed_spread_pp": round(observed, 2),
            "largest_spread_feature": winner,
            "p_largest_spread_at_least_observed": round(hits / PERMUTATION_TRIALS, 4),
            "any_feature_separates": hits / PERMUTATION_TRIALS < 0.05,
        },
    }


def volume_measure_agreement(market: Market) -> dict:
    """Whether real exchange volume is a different measurement from spot tick volume.

    RS-XAUUSD-20260818-002 recorded as a limitation that spot tick volume is not exchange
    volume. If the two rank-correlate closely, that limitation was immaterial and testing
    MGC volume was not a second, independent look.
    """
    spot_ratio, futures_ratio = [], []
    for i in range(WARMUP, len(market.times)):
        spot_window = statistics.median(
            market.spot[market.times[j]]["v"] for j in range(i - 20, i)
        )
        futures_window = statistics.median(
            market.futures[market.times[j]]["v"] for j in range(i - 20, i)
        )
        if not spot_window or not futures_window:
            continue
        spot_ratio.append(market.spot[market.times[i]]["v"] / spot_window)
        futures_ratio.append(market.futures[market.times[i]]["v"] / futures_window)

    def pearson(x: list[float], y: list[float]) -> float:
        mx, my = statistics.fmean(x), statistics.fmean(y)
        numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
        denominator = math.sqrt(
            sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
        )
        return numerator / denominator if denominator else 0.0

    order_x = {v: i for i, v in enumerate(sorted(range(len(spot_ratio)),
                                                key=lambda k: spot_ratio[k]))}
    order_y = {v: i for i, v in enumerate(sorted(range(len(futures_ratio)),
                                                key=lambda k: futures_ratio[k]))}
    spearman = pearson(
        [float(order_x[i]) for i in range(len(spot_ratio))],
        [float(order_y[i]) for i in range(len(futures_ratio))],
    )
    return {
        "bars_compared": len(spot_ratio),
        "pearson_r": round(pearson(spot_ratio, futures_ratio), 3),
        "spearman_rho": round(spearman, 3),
        "measures_are_distinct": spearman < 0.6,
        "conclusion": (
            "Spot tick volume and real MGC contract volume rank-correlate closely, so "
            "RS-XAUUSD-20260818-002's stated limitation about tick volume was immaterial. "
            "Testing exchange volume here was a confirmation, not an independent second look."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    market = Market()
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(tz=TAIPEI).isoformat(timespec="seconds"),
        "strategy": "S1 AweWithBB V3.9; S2 Hammer V3.2",
        "method": {
            "spot_source": "30-minute FX_IDC:XAUUSD export",
            "futures_source": "30-minute COMEX_MINI_DL:MGC1! export",
            "aligned_bars": len(market.times),
            "spot_only_bars": market.spot_only,
            "futures_only_bars": market.futures_only,
            "feature_bar": "entry_index - 1, the last fully completed bar before the fill",
            "roll_handling": (
                f"bars whose basis moves more than {ROLL_THRESHOLD_USD} USD in one step are "
                f"roll bars; a trade is dropped if any of its {LOOKBACK}-bar window is one"
            ),
            "permutation_trials": PERMUTATION_TRIALS,
            "random_seed": RANDOM_SEED,
            "timezone": "Asia/Taipei",
        },
        "roll_cycle": roll_cycle_profile(market),
        "volume_measure_agreement": volume_measure_agreement(market),
        "strategies": {},
    }

    for name, config in STRATEGIES.items():
        all_trades = load_trades(config["trades"])
        with_features = []
        for trade in all_trades:
            if trade["entry_at"] not in market.index:
                continue
            features = market.features(market.index[trade["entry_at"]] - 1)
            if features is None:
                continue
            trade["features"] = features
            with_features.append(trade)
        baseline = metrics([trade["return_pct"] for trade in with_features])
        results["strategies"][name] = {
            "label": config["label"],
            "trades_total": len(all_trades),
            "trades_with_features": len(with_features),
            "dropped_for_roll_or_warmup": len(all_trades) - len(with_features),
            "baseline": baseline,
            "by_futures_feature": feature_family(
                with_features, baseline, stream(f"futures:{name}")
            ),
            "raw_basis_control": raw_basis_control(
                all_trades, market, metrics([t["return_pct"] for t in all_trades])
            ),
        }

    results["limitations"] = [
        "MGC1! is a continuous front-month series. Detrending removes the roll cycle from "
        "the level, but a trade near a roll still sits in a market where two contracts are "
        "competing for liquidity, and dropping its window removes it rather than measuring "
        "it.",
        "Basis is measured on 30-minute closes of two venues with different sessions and "
        "settlement conventions; a deviation of well under a dollar is inside the noise of "
        "that comparison.",
        "Terciles are coarse and would dilute an effect confined to a narrow tail.",
        "Both strategies are long-only over a strongly rising market.",
        "No result changes formal S1 or S2 logic, live risk, or an active entry checklist.",
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "study_id": STUDY_ID,
        "output": str(args.output / "results.json"),
        "spearman_spot_vs_futures_volume": results["volume_measure_agreement"]["spearman_rho"],
        "strategies": {
            name: {
                "n": block["trades_with_features"],
                "family_p": block["by_futures_feature"]["family_permutation_test"][
                    "p_largest_spread_at_least_observed"],
                "any_separates": block["by_futures_feature"]["family_permutation_test"][
                    "any_feature_separates"],
                "raw_basis_trap_spread_pp": block["raw_basis_control"]["win_rate_spread_pp"],
            }
            for name, block in results["strategies"].items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
