#!/usr/bin/env python3
"""RS-XAUUSD-20260818-004 — alpha search on 30-minute bars, independent of S1 and S2.

Every earlier study asked whether an existing S1/S2 trade could be improved. This one sets
the strategies aside and asks the prior question: does the 30-minute bar series carry a
directional edge at all?

## The discipline is the study

Scanning a wide feature set and reporting the best result manufactures a finding on any
data. Three constraints are applied before anything is looked at:

- **TRAIN** (first 55%, 2024-01 to 2025-06) is the only period searched. The shortlist is
  fixed there and never revised afterwards.
- **VALID** (next 25%) tests the fixed shortlist.
- **HOLDOUT** (last 20%) is opened once, for the two results that reached it.
- Significance uses a moving-block bootstrap with a circular shift, not an iid one.
  Forward returns at horizon h overlap across h consecutive bars; iid resampling treats
  correlated observations as independent and inflates every t-statistic in sight.
- Costs are charged. A 30-minute reversal signal that ignores spread looks profitable and
  is not.

## What it finds

Direction is not forecastable here and volatility is, by more than an order of magnitude.
The largest directional IC anywhere in the scan is 0.04 and the shortlist does not survive
validation; the volatility forecast reaches 0.571 on data it has never seen.

That would suggest sizing or stop placement by forecast volatility. It does not work, and
the reason is the useful part: with a fixed-percentage stop and fixed-fractional risk, the
R distribution is already flat across volatility terciles. The risk framework has already
neutralised volatility, so a volatility forecast has nothing left to correct. That single
mechanism explains this study's negative sizing result, its negative stop result, and
RS-XAUUSD-20260818-002's negative ATR result.

Usage:
    python3.12 -m scripts.research.build_xauusd_bar_alpha
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
STUDY_ID = "RS-XAUUSD-20260818-004"
OUTPUT_DIR = Path("reproduced")
BARS_FILE = Path("local-inputs/FX_IDC_XAUUSD, 30_volumn.csv")
TAIPEI = timezone(timedelta(hours=8))

TRADES = {
    "S1": (Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv"), 0.5),
    "S2": (Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-08-15.csv"), 1.0),
}

HORIZONS = [1, 2, 4, 8, 16, 48]
COST_GRID_PCT = [0.010, 0.020, 0.040]
TRAIN_END, VALID_END = 0.55, 0.80
# Fixed on TRAIN before validation, never revised.
SHORTLIST = [("atr_pctile240", 48), ("atr_pct", 48), ("dist_sma24", 16),
             ("streak", 48), ("dist_sma24", 48)]
STOP_MULTIPLES = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
SIZING_EXPONENTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
BOOTSTRAP_TRIALS = 1500
RANDOM_SEED = 20260818


def stream(label: str) -> random.Random:
    return random.Random(f"{RANDOM_SEED}:{label}")


def load_bars() -> list[dict]:
    bars = []
    with BARS_FILE.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            bars.append({
                "t": datetime.fromisoformat(record["time"]).astimezone(TAIPEI),
                "o": float(record["open"]), "h": float(record["high"]),
                "l": float(record["low"]), "c": float(record["close"]),
                "v": float(record["Volume"]),
            })
    bars.sort(key=lambda bar: bar["t"])
    return bars


def load_trades(path: Path) -> list[dict]:
    trades: dict[int, dict] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            number = int(record["Trade number"])
            trade = trades.setdefault(number, {})
            stamp = datetime.strptime(record["Date and time"], "%Y-%m-%d %H:%M").replace(
                tzinfo=TAIPEI)
            if record["Type"].lower().startswith("entry"):
                trade["entry_at"], trade["entry_price"] = stamp, float(record["Price USD"])
            else:
                trade["exit_at"] = stamp
                trade["exit_signal"] = record["Signal"]
            trade["return_pct"] = float(record["Return %"])
    out = [t for _, t in sorted(trades.items())
           if {"entry_at", "exit_at"} <= set(t) and t.get("exit_signal") != "Open"]
    out.sort(key=lambda t: t["entry_at"])
    return out


def splits(n: int) -> dict[str, tuple[int, int]]:
    a, b = int(n * TRAIN_END), int(n * VALID_END)
    return {"train": (0, a), "valid": (a, b), "holdout": (b, n)}


def forward_returns(bars: list[dict], h: int) -> list[float | None]:
    out: list[float | None] = [None] * len(bars)
    for i in range(len(bars) - h):
        out[i] = 100 * (bars[i + h]["c"] - bars[i]["c"]) / bars[i]["c"]
    return out


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for position, index in enumerate(order):
        out[index] = float(position)
    return out


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return 0.0
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(_rank(x), _rank(y))


def true_range(bars: list[dict], i: int) -> float:
    if i == 0:
        return bars[i]["h"] - bars[i]["l"]
    previous = bars[i - 1]["c"]
    return max(bars[i]["h"] - bars[i]["l"], abs(bars[i]["h"] - previous),
               abs(bars[i]["l"] - previous))


def build_features(bars: list[dict]) -> dict[str, list[float | None]]:
    """Every feature uses bars up to and including i; execution is at bar i's close."""
    n = len(bars)
    closes = [b["c"] for b in bars]
    features: dict[str, list[float | None]] = {}

    def sma(xs, w):
        out: list[float | None] = [None] * n
        total = 0.0
        for i in range(n):
            total += xs[i]
            if i >= w:
                total -= xs[i - w]
            if i >= w - 1:
                out[i] = total / w
        return out

    ranges = [true_range(bars, i) for i in range(n)]
    atr = sma(ranges, 14)
    for w in (8, 24, 48, 96, 240):
        m = sma(closes, w)
        features[f"dist_sma{w}"] = [
            None if m[i] is None or not atr[i] else (closes[i] - m[i]) / atr[i]
            for i in range(n)]
    for w in (1, 2, 4, 8, 16, 48, 96):
        features[f"ret_{w}"] = [
            None if i < w else 100 * (closes[i] - closes[i - w]) / closes[i - w]
            for i in range(n)]
    features["atr_pct"] = [None if not atr[i] else 100 * atr[i] / closes[i] for i in range(n)]
    percentile: list[float | None] = [None] * n
    for i in range(240, n):
        window = [atr[j] for j in range(i - 240, i + 1) if atr[j]]
        if window:
            percentile[i] = 100 * sum(1 for x in window if x <= atr[i]) / len(window)
    features["atr_pctile240"] = percentile
    features["range_over_atr"] = [
        None if not atr[i] else (bars[i]["h"] - bars[i]["l"]) / atr[i] for i in range(n)]
    span = [bars[i]["h"] - bars[i]["l"] for i in range(n)]
    features["body_ratio"] = [
        None if span[i] <= 0 else (bars[i]["c"] - bars[i]["o"]) / span[i] for i in range(n)]
    features["upper_wick"] = [
        None if span[i] <= 0 else (bars[i]["h"] - max(bars[i]["o"], bars[i]["c"])) / span[i]
        for i in range(n)]
    features["lower_wick"] = [
        None if span[i] <= 0 else (min(bars[i]["o"], bars[i]["c"]) - bars[i]["l"]) / span[i]
        for i in range(n)]
    volume_ma = sma([b["v"] for b in bars], 20)
    features["vol_ratio"] = [
        None if not volume_ma[i] else bars[i]["v"] / volume_ma[i] for i in range(n)]
    features["hour"] = [float(b["t"].hour) for b in bars]
    features["dow"] = [float(b["t"].weekday()) for b in bars]
    for w in (48, 96, 240):
        column: list[float | None] = [None] * n
        for i in range(w, n):
            high = max(bars[j]["h"] for j in range(i - w + 1, i + 1))
            low = min(bars[j]["l"] for j in range(i - w + 1, i + 1))
            column[i] = 100 * (closes[i] - low) / (high - low) if high > low else None
        features[f"range_pos{w}"] = column
    streak = [0.0] * n
    for i in range(1, n):
        up = closes[i] > closes[i - 1]
        previous = streak[i - 1]
        streak[i] = (previous + 1 if previous > 0 else 1) if up else (
            previous - 1 if previous < 0 else -1)
    features["streak"] = streak
    vov: list[float | None] = [None] * n
    for i in range(96, n):
        window = [features["atr_pct"][j] for j in range(i - 96, i + 1)
                  if features["atr_pct"][j] is not None]
        if len(window) > 10 and statistics.fmean(window):
            vov[i] = statistics.pstdev(window) / statistics.fmean(window)
    features["vol_of_vol"] = vov
    return features


def ic(feature, forward, span) -> float | None:
    a, b = span
    pairs = [(feature[i], forward[i]) for i in range(a, b)
             if feature[i] is not None and forward[i] is not None]
    if len(pairs) < 500:
        return None
    return round(spearman([p[0] for p in pairs], [p[1] for p in pairs]), 4)


def block_bootstrap_p(feature, forward, span, horizon, rng) -> dict | None:
    """Circular-shift null with moving blocks sized to the horizon's own overlap.

    The forward return at horizon h shares h-1 bars with its neighbour, so a naive
    resample destroys exactly the dependence that matters. Blocks of 3h preserve it, and
    a circular shift of the outcome series breaks the feature-outcome link while leaving
    each series' own autocorrelation intact.
    """
    a, b = span
    pairs = [(feature[i], forward[i]) for i in range(a, b)
             if feature[i] is not None and forward[i] is not None]
    n = len(pairs)
    if n < 400:
        return None
    length = max(3 * horizon, 6)
    blocks = max(1, n // length)
    observed = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    hits = 0
    for _ in range(BOOTSTRAP_TRIALS):
        sample = []
        for _ in range(blocks):
            start = rng.randrange(0, max(1, n - length))
            sample.extend(pairs[start:start + length])
        xs = [p[0] for p in sample]
        ys = [p[1] for p in sample]
        shift = rng.randrange(1, len(ys))
        ys = ys[shift:] + ys[:shift]
        if abs(spearman(xs, ys)) >= abs(observed):
            hits += 1
    return {"ic": round(observed, 4), "n": n, "block_length": length,
            "p_value": round(hits / BOOTSTRAP_TRIALS, 4)}


def quantile_spread(feature, forward, span, q=5) -> dict | None:
    a, b = span
    pairs = [(feature[i], forward[i]) for i in range(a, b)
             if feature[i] is not None and forward[i] is not None]
    if len(pairs) < 500:
        return None
    n = len(pairs)
    ordered = sorted(p[0] for p in pairs)
    cuts = [ordered[int(n * k / q)] for k in range(1, q)]
    buckets: list[list[float]] = [[] for _ in range(q)]
    for value, ret in pairs:
        buckets[sum(1 for c in cuts if value > c)].append(ret)
    means = [round(statistics.fmean(x), 4) for x in buckets if x]
    spread = means[-1] - means[0]
    return {
        "quantile_means_pct": means,
        "top_minus_bottom_pct": round(spread, 4),
        "survives_cost": {f"{c}%": abs(spread) > c for c in COST_GRID_PCT},
    }


def hour_factor(bars, span) -> dict[int, float]:
    """Median ratio of a bar's true range to the trailing ATR, by hour of day."""
    a, b = span
    ranges = [true_range(bars, i) for i in range(len(bars))]
    percent = [100 * ranges[i] / bars[i]["c"] for i in range(len(bars))]
    buckets: dict[int, list[float]] = {}
    for i in range(max(a, 14), b):
        trailing = statistics.fmean(percent[i - 14:i])
        if trailing:
            buckets.setdefault(bars[i]["t"].hour, []).append(percent[i] / trailing)
    return {h: statistics.median(v) for h, v in buckets.items()}


def volatility_model(bars, sp) -> dict:
    ranges = [true_range(bars, i) for i in range(len(bars))]
    percent = [100 * ranges[i] / bars[i]["c"] for i in range(len(bars))]

    def trailing_atr(i):
        return None if i < 14 else statistics.fmean(percent[i - 14:i])

    factors = hour_factor(bars, sp["train"])

    def predict(i):
        base = trailing_atr(i)
        return None if base is None else base * factors.get(bars[i]["t"].hour, 1.0)

    accuracy = {}
    for name, fn in (("atr_only", trailing_atr), ("atr_times_hour", predict)):
        row = {}
        for period, (a, b) in sp.items():
            pairs = [(fn(i), percent[i]) for i in range(a, b) if fn(i) is not None]
            row[period] = round(spearman([p[0] for p in pairs], [p[1] for p in pairs]), 3)
        accuracy[name] = row

    stability = {}
    per_period = {}
    for period, (a, b) in sp.items():
        per_period[period] = hour_factor(bars, (a, b))
    shared = sorted(set(per_period["train"]) & set(per_period["valid"])
                    & set(per_period["holdout"]))
    for other in ("valid", "holdout"):
        stability[f"train_vs_{other}"] = round(spearman(
            [per_period["train"][h] for h in shared],
            [per_period[other][h] for h in shared]), 3)

    return {
        "forecast_accuracy_spearman": accuracy,
        "hour_factor_train": {str(h): round(v, 3) for h, v in sorted(factors.items())},
        "hour_factor_stability_spearman": stability,
        "hour_factor_by_period": {
            p: {str(h): round(per_period[p][h], 3) for h in shared} for p in per_period},
        "peak_to_trough_ratio": round(max(factors.values()) / min(factors.values()), 2),
        "note": (
            "Direction and magnitude are forecast from the same bars with the same method. "
            "The comparison is the finding: the best directional IC anywhere in this study "
            "is an order of magnitude below the volatility forecast on data it has not seen."
        ),
    }


def volatility_applications(bars, sp) -> dict:
    """Whether a working volatility forecast improves S1 or S2. It does not, and the
    diagnostic explains why: R is already flat across volatility terciles."""
    ranges = [true_range(bars, i) for i in range(len(bars))]
    percent = [100 * ranges[i] / bars[i]["c"] for i in range(len(bars))]
    factors = hour_factor(bars, sp["train"])
    times = [b["t"] for b in bars]

    def predict(i):
        if i < 14:
            return None
        return statistics.fmean(percent[i - 14:i]) * factors.get(bars[i]["t"].hour, 1.0)

    def max_drawdown(values):
        equity = peak = drawdown = 0.0
        for value in values:
            equity += value
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        return round(drawdown, 2)

    out = {}
    for name, (path, stop_pct) in TRADES.items():
        trades = []
        for trade in load_trades(path):
            i = bisect.bisect_left(times, trade["entry_at"])
            j = bisect.bisect_left(times, trade["exit_at"])
            if i >= len(bars) or bars[i]["t"] != trade["entry_at"] or j <= i or i < 1:
                continue
            forecast = predict(i - 1)
            if not forecast:
                continue
            trade["forecast_vol_pct"] = forecast
            trade["path"] = bars[i:min(j + 1, len(bars))]
            trades.append(trade)
        if len(trades) < 50:
            out[name] = {"applicable": False}
            continue

        baseline_r = [t["return_pct"] / stop_pct for t in trades]
        baseline_variance = len(baseline_r) * statistics.pvariance(baseline_r)

        ordered = sorted(t["forecast_vol_pct"] for t in trades)
        first, second = ordered[len(trades) // 3], ordered[2 * len(trades) // 3]
        terciles = {}
        for label, low, high in (("low_vol", -math.inf, first), ("mid_vol", first, second),
                                 ("high_vol", second, math.inf)):
            group = [t["return_pct"] / stop_pct for t in trades
                     if low < t["forecast_vol_pct"] <= high]
            terciles[label] = {
                "n": len(group), "mean_r": round(statistics.fmean(group), 3),
                "sd_r": round(statistics.pstdev(group), 3)}

        scaled_stop = {}
        for k in STOP_MULTIPLES:
            values = []
            for trade in trades:
                distance = k * trade["forecast_vol_pct"]
                stop_price = trade["entry_price"] * (1 - distance / 100)
                hit = None
                for bar in trade["path"]:
                    if bar["l"] <= stop_price:
                        hit = -distance
                        break
                values.append((hit if hit is not None else trade["return_pct"]) / distance)
            variance = len(values) * statistics.pvariance(values)
            scale = math.sqrt(baseline_variance / variance) if variance else 0.0
            scaled_stop[f"k={k}"] = {
                "total_r": round(sum(values), 2),
                "mean_r": round(statistics.fmean(values), 3),
                "sd_r": round(statistics.pstdev(values), 3),
                "max_drawdown_r": max_drawdown(values),
                "share_of_baseline_at_equal_volatility_pct": round(
                    100 * sum(values) * scale / sum(baseline_r), 1),
            }

        median_forecast = statistics.median(t["forecast_vol_pct"] for t in trades)
        sizing = {}
        for p in SIZING_EXPONENTS:
            values = [(t["return_pct"] / stop_pct)
                      * ((median_forecast / t["forecast_vol_pct"]) ** p) for t in trades]
            variance = len(values) * statistics.pvariance(values)
            scale = math.sqrt(baseline_variance / variance) if variance else 0.0
            sizing[f"p={p}"] = {
                "total_r": round(sum(values), 2),
                "sd_r": round(statistics.pstdev(values), 3),
                "share_of_baseline_at_equal_volatility_pct": round(
                    100 * sum(values) * scale / sum(baseline_r), 1),
            }

        ratios = [stop_pct / t["forecast_vol_pct"] for t in trades]
        out[name] = {
            "applicable": True,
            "n": len(trades),
            "fixed_stop_pct": stop_pct,
            "baseline": {"total_r": round(sum(baseline_r), 2),
                         "mean_r": round(statistics.fmean(baseline_r), 3),
                         "sd_r": round(statistics.pstdev(baseline_r), 3),
                         "max_drawdown_r": max_drawdown(baseline_r)},
            "stop_in_forecast_vol_units": {
                "median": round(statistics.median(ratios), 2),
                "min": round(min(ratios), 2), "max": round(max(ratios), 2),
                "spread_factor": round(max(ratios) / min(ratios), 1)},
            "r_by_forecast_vol_tercile": terciles,
            "volatility_scaled_stop": scaled_stop,
            "volatility_targeted_sizing": sizing,
            "best_variant_share_pct": max(
                [row["share_of_baseline_at_equal_volatility_pct"]
                 for row in scaled_stop.values()]
                + [row["share_of_baseline_at_equal_volatility_pct"]
                   for row in sizing.values()]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    bars = load_bars()
    sp = splits(len(bars))
    features = build_features(bars)
    forwards = {h: forward_returns(bars, h) for h in HORIZONS}

    scan = {}
    for name, values in features.items():
        scan[name] = {str(h): ic(values, forwards[h], sp["train"]) for h in HORIZONS}
    ranked = sorted(
        ((abs(v), v, name, h) for name, row in scan.items()
         for h, v in row.items() if v is not None), reverse=True)

    validation = {}
    for name, h in SHORTLIST:
        train_ic = ic(features[name], forwards[h], sp["train"])
        valid_ic = ic(features[name], forwards[h], sp["valid"])
        validation[f"{name}@h{h}"] = {
            "train_ic": train_ic,
            "valid_ic": valid_ic,
            "sign_held": (train_ic or 0) * (valid_ic or 0) > 0,
            "train_quantile": quantile_spread(features[name], forwards[h], sp["train"]),
            "valid_quantile": quantile_spread(features[name], forwards[h], sp["valid"]),
            "valid_block_bootstrap": block_bootstrap_p(
                features[name], forwards[h], sp["valid"], h, stream(f"bb:{name}:{h}")),
        }
    survivors = [k for k, v in validation.items()
                 if v["sign_held"] and v["valid_block_bootstrap"]
                 and v["valid_block_bootstrap"]["p_value"] < 0.05]

    first, last = bars[0]["c"], bars[-1]["c"]
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(tz=TAIPEI).isoformat(timespec="seconds"),
        "strategy": "none — bar-level, independent of S1 and S2",
        "method": {
            "bar_source": "30-minute FX_IDC:XAUUSD export",
            "bar_count": len(bars),
            "splits": {k: {"bars": v[1] - v[0],
                           "from": bars[v[0]]["t"].date().isoformat(),
                           "to": bars[v[1] - 1]["t"].date().isoformat()}
                       for k, v in sp.items()},
            "shortlist_fixed_on": "train, before any validation result was computed",
            "significance": "moving-block bootstrap with circular shift, block = 3 x horizon",
            "round_trip_costs_pct": COST_GRID_PCT,
            "bootstrap_trials": BOOTSTRAP_TRIALS,
            "random_seed": RANDOM_SEED,
            "timezone": "Asia/Taipei",
        },
        "benchmark": {
            "buy_and_hold_return_pct": round(100 * (last - first) / first, 1),
            "note": "Any long-biased rule has to beat this on a risk-adjusted basis.",
        },
        "directional_scan_train_only": {
            "features": len(features), "horizons": HORIZONS, "ic_by_feature": scan,
            "largest_abs_ic": round(ranked[0][0], 4) if ranked else None,
            "top_10": [{"feature": n, "horizon": int(h), "ic": v} for _, v, n, h in ranked[:10]],
        },
        "shortlist_validation": validation,
        "shortlist_survivors": survivors,
        "volatility_forecast": volatility_model(bars, sp),
        "volatility_applications": volatility_applications(bars, sp),
    }

    results["limitations"] = [
        "The feature set is linear and univariate. An interaction that only appears in a "
        "combination would not be found here, and searching combinations on this sample "
        "would produce false positives faster than findings.",
        "Terciles and quintiles are coarse; an effect confined to a narrow tail would be "
        "diluted.",
        "Costs are modelled as a flat round-trip percentage. Real slippage rises exactly "
        "when volatility does, which would hurt the short-horizon signals further than "
        "modelled — the direction of that error favours the conclusion drawn.",
        "HOLDOUT was opened for the volatility forecast and the session comparison. It is "
        "spent for those questions and must not be reused to rescue a directional feature.",
        "The sample is one instrument over 32 months of a strong uptrend.",
        "No result changes formal S1 or S2 logic, live risk, or an active entry checklist.",
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    vol = results["volatility_forecast"]["forecast_accuracy_spearman"]["atr_times_hour"]
    print(json.dumps({
        "study_id": STUDY_ID,
        "largest_directional_ic": results["directional_scan_train_only"]["largest_abs_ic"],
        "shortlist_survivors": survivors,
        "volatility_forecast_ic": vol,
        "best_volatility_application_pct": {
            k: v.get("best_variant_share_pct")
            for k, v in results["volatility_applications"].items()},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
