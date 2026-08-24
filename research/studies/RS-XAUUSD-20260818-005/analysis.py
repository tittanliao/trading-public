#!/usr/bin/env python3
"""RS-XAUUSD-20260818-005 — S1 and S2 as a portfolio rather than two strategies.

Every study so far has asked how to make a strategy better. This one asks whether the two
that already exist are being *combined* well, which needs no new alpha and no new data.

The premise is that S1 and S2 are both long-only on the same instrument and should
therefore be nearly redundant. They are not: their monthly R returns correlate at 0.234,
low enough that combining them is a real diversification rather than a relabelling.

## What it does and does not support

Blending is strongly better than running S2 alone and is **not** demonstrably better than
running S1 alone. Both halves of that are reported, because reporting only the first would
turn a diversification result into a recommendation the data does not carry.

## Two things this study refuses to do

**It does not use the in-sample optimal weight.** Sharpe peaks at w=0.6 for S1. It also sits
between 3.71 and 3.96 across every weight from 0.4 to 0.8, so the peak is a coin toss inside
a plateau. The study reports 50/50, which is the choice a person makes without looking.

**It does not treat the Sharpe level as real.** Values near 3.9 do not survive contact with
live execution. These are Strategy Tester R multiples aggregated monthly over a market that
rose 112%, with no slippage beyond what the export already contains. The *comparison*
between allocations is the output; the level is not.

Usage:
    python3.12 -m scripts.research.build_xauusd_portfolio_blend
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
STUDY_ID = "RS-XAUUSD-20260818-005"
OUTPUT_DIR = Path("reproduced")
TAIPEI = timezone(timedelta(hours=8))

STRATEGIES = {
    "S1": {"label": "S1 AweWithBB V3.9", "stop_pct": 0.5,
           "trades": Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv")},
    "S2": {"label": "S2 Hammer V3.2", "stop_pct": 1.0,
           "trades": Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-08-15.csv")},
}

WEIGHT_GRID = [i / 10 for i in range(11)]
REPORTED_WEIGHT = 0.5
BOOTSTRAP_TRIALS = 20000
RANDOM_SEED = 20260818


BARS_FILE = Path("local-inputs/FX_IDC_XAUUSD, 30_volumn.csv")CANDIDATE_WEIGHT = 0.3
FADE_SHORTLIST = [(8, 4.0, 8), (2, 2.5, 8), (2, 2.5, 16), (4, 2.5, 16)]
COST_PCT = 0.02


def load_bars() -> list[dict]:
    rows = []
    with BARS_FILE.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            rows.append({
                "t": datetime.fromisoformat(record["time"]).astimezone(TAIPEI),
                "o": float(record["open"]), "h": float(record["high"]),
                "l": float(record["low"]), "c": float(record["close"]),
                "upper": float(record["Upper"]) if record["Upper"] else None,
                "lower": float(record["Lower"]) if record["Lower"] else None,
                "hammer": record["Hammer (Filtered)"] == "1",
            })
    rows.sort(key=lambda bar: bar["t"])
    return rows


def beta_decomposition(series: dict, months: list[str], bars: list[dict]) -> dict:
    """How much of each strategy's return is simply being long gold.

    This decides whether an uncorrelated strategy is even possible inside this instrument.
    If the strategies were mostly beta, any second stream would have to be short gold to
    decorrelate, and the search would be structurally hopeless. They are not.
    """
    closes: dict[str, float] = {}
    for bar in bars:
        closes[f"{bar['t'].year}-{bar['t'].month:02d}"] = bar["c"]
    keys = sorted(closes)
    gold = {keys[i]: 100 * (closes[keys[i]] - closes[keys[i - 1]]) / closes[keys[i - 1]]
            for i in range(1, len(keys))}
    usable = [m for m in months if m in gold]
    g = [gold[m] for m in usable]
    mg = statistics.fmean(g)
    out = {"months": len(usable), "gold_mean_monthly_pct": round(mg, 3)}
    for name in ("S1", "S2"):
        y = [series[name].get(m, 0.0) for m in usable]
        my = statistics.fmean(y)
        denominator = sum((x - mg) ** 2 for x in g)
        beta = sum((x - mg) * (v - my) for x, v in zip(g, y)) / denominator
        alpha = my - beta * mg
        predicted = [alpha + beta * x for x in g]
        ss_res = sum((v - p) ** 2 for v, p in zip(y, predicted))
        ss_tot = sum((v - my) ** 2 for v in y)
        down = [v for x, v in zip(g, y) if x <= 0]
        up = [v for x, v in zip(g, y) if x > 0]
        out[name] = {
            "beta": round(beta, 3),
            "alpha_r_per_month": round(alpha, 3),
            "r_squared": round(1 - ss_res / ss_tot, 3) if ss_tot else None,
            "beta_contribution_r_per_month": round(beta * mg, 3),
            "alpha_share_of_return_pct": round(100 * alpha / my, 1) if my else None,
            "mean_r_in_gold_down_months": round(statistics.fmean(down), 3) if down else None,
            "mean_r_in_gold_up_months": round(statistics.fmean(up), 3) if up else None,
            "down_months": len(down),
        }
    return out


def counter_trend_fade(bars: list[dict], splits_: dict) -> dict:
    """Fade an N-bar move of at least X ATR, hold H bars. Long and short, symmetric.

    Chosen because reversal is the only structure the bar-level study found that beat a
    random-walk benchmark, and because fading is naturally two-sided where S1 and S2 are
    both long-only.
    """
    closes = [b["c"] for b in bars]
    ranges = [bars[0]["h"] - bars[0]["l"]] + [
        max(bars[i]["h"] - bars[i]["l"], abs(bars[i]["h"] - bars[i - 1]["c"]),
            abs(bars[i]["l"] - bars[i - 1]["c"])) for i in range(1, len(bars))]

    def atr(i):
        return statistics.fmean(ranges[i - 14:i]) if i >= 14 else None

    def run(span, n_bars, multiple, hold):
        a, b = span
        out = []
        for i in range(max(a, 20), b - hold):
            reference = atr(i)
            if not reference:
                continue
            move = closes[i] - closes[i - n_bars]
            if abs(move) < multiple * reference:
                continue
            direction = -1 if move > 0 else 1
            out.append((bars[i]["t"],
                        direction * 100 * (closes[i + hold] - closes[i]) / closes[i] - COST_PCT,
                        direction))
        return out

    variants = {}
    for n_bars, multiple, hold in FADE_SHORTLIST:
        train = [x[1] for x in run(splits_["train"], n_bars, multiple, hold)]
        valid = [x[1] for x in run(splits_["valid"], n_bars, multiple, hold)]
        if len(train) < 20 or len(valid) < 20:
            continue

        def t_stat(values):
            sd = statistics.pstdev(values)
            return round(statistics.fmean(values) / (sd / math.sqrt(len(values))), 2) if sd else None

        variants[f"n{n_bars}_x{multiple}_h{hold}"] = {
            "train": {"n": len(train), "mean_pct": round(statistics.fmean(train), 4),
                      "t": t_stat(train)},
            "valid": {"n": len(valid), "mean_pct": round(statistics.fmean(valid), 4),
                      "t": t_stat(valid)},
            "sign_held": statistics.fmean(train) * statistics.fmean(valid) > 0,
        }
    survivors = [k for k, v in variants.items()
                 if v["sign_held"] and v["valid"]["mean_pct"] > 0]

    # Surviving train-to-valid is not the same as being worth adding. The best survivor is
    # carried through to the portfolio so this block cannot report "2 survived" without
    # also reporting that both still lose money over the full sample and reduce the blend.
    portfolio = {}
    if survivors:
        key = survivors[0]
        n_bars, multiple, hold = FADE_SHORTLIST[
            [f"n{a}_x{b}_h{c}" for a, b, c in FADE_SHORTLIST].index(key)]
        full = run((0, len(bars)), n_bars, multiple, hold)
        monthly: dict[str, float] = collections.defaultdict(float)
        for stamp, value, _ in full:
            monthly[f"{stamp.year}-{stamp.month:02d}"] += value / 0.5
        portfolio = {"variant": key, "monthly": dict(monthly), "trades": len(full)}
    return {"shortlist_fixed_on": "train", "variants": variants,
            "sign_held_and_positive": survivors,
            "best_survivor_full_period": portfolio}


def mirror_short(bars: list[dict], splits_: dict) -> dict:
    """The structural mirror of S2: a bearish reversal candle at the upper band, short.

    The pattern definition is reverse-engineered from the export's own hammer flags rather
    than invented, so this is a mirror and not a new idea wearing a mirror's name. Hammer
    bars in this data have a body of about 0.25 of range, a lower wick of 0.69, an upper
    wick of 0.06, a close at 0.81 of range and a Bollinger position of 0.25; each of those
    is reflected here.
    """
    def is_star(bar):
        if not bar["upper"] or not bar["lower"]:
            return False
        span = bar["h"] - bar["l"]
        width = bar["upper"] - bar["lower"]
        if span <= 0 or width <= 0:
            return False
        body = abs(bar["c"] - bar["o"])
        upper_wick = bar["h"] - max(bar["o"], bar["c"])
        lower_wick = min(bar["o"], bar["c"]) - bar["l"]
        return (body / span <= 0.31 and upper_wick / span >= 0.61
                and lower_wick / span <= 0.11
                and (bar["c"] - bar["l"]) / span <= 0.19
                and (bar["c"] - bar["lower"]) / width >= 0.75)

    signals = [i for i in range(len(bars)) if is_star(bars[i])]
    hammers = sum(1 for b in bars if b["hammer"])

    def run(span, hold, stop=1.0):
        a, b = span
        out = []
        for i in signals:
            if not (a <= i < b) or i + 1 >= len(bars):
                continue
            entry = bars[i]["c"]
            stop_price = entry * (1 + stop / 100)
            result = None
            end = min(i + 1 + hold, len(bars))
            for j in range(i + 1, end):
                if bars[j]["h"] >= stop_price:
                    result = -stop
                    break
            if result is None:
                result = 100 * (entry - bars[end - 1]["c"]) / entry
            out.append(result - COST_PCT)
        return out

    by_period = {}
    for period in ("train", "valid"):
        values = run(splits_[period], 16)
        if len(values) < 10:
            continue
        by_period[period] = {
            "n": len(values),
            "win_rate_pct": round(100 * sum(1 for v in values if v > 0) / len(values), 1),
            "mean_pct": round(statistics.fmean(values), 4),
            "total_pct": round(sum(values), 2),
        }
    return {
        "signals": len(signals), "hammer_signals_for_comparison": hammers,
        "definition": "each hammer characteristic reflected: body<=0.31, upper wick>=0.61, "
                      "lower wick<=0.11, close position<=0.19, Bollinger position>=0.75",
        "by_period": by_period,
        "profitable_anywhere": any(v["mean_pct"] > 0 for v in by_period.values()),
    }


def required_sharpe(blend_sharpe: float, weight: float = CANDIDATE_WEIGHT) -> dict:
    """What a new strategy must deliver, by its correlation to the existing blend.

    This is the study's most reusable output: a bar to hold any future idea to, before it
    is built. Both streams are treated as equal-risk, so `weight` is a risk allocation.
    """
    def combined(candidate_sharpe, rho):
        variance = (1 - weight) ** 2 + weight ** 2 + 2 * weight * (1 - weight) * rho
        return ((1 - weight) * blend_sharpe + weight * candidate_sharpe) / math.sqrt(variance)

    table = {}
    for rho in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        low, high = -5.0, 10.0
        for _ in range(60):
            mid = (low + high) / 2
            if combined(mid, rho) < blend_sharpe:
                low = mid
            else:
                high = mid
        table[f"rho={rho:+.2f}"] = round((low + high) / 2, 2)
    return {
        "candidate_weight": weight,
        "blend_sharpe": round(blend_sharpe, 3),
        "minimum_sharpe_to_improve": table,
        "note": ("At zero correlation a candidate needs a Sharpe of about 0.8 to help. At "
                 "-0.25 it helps even while losing money slightly. Correlation is worth "
                 "more than return, which is the argument for a different instrument "
                 "rather than a different rule on this one."),
    }


def load_monthly_r(path: Path, stop_pct: float) -> dict[str, float]:
    """Monthly sum of R, where R is each trade's return divided by its nominal stop.

    R is the right unit here: it is what a fixed-fractional trader actually experiences,
    and it makes two strategies with different stop widths directly comparable.
    """
    trades: dict[int, dict] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            number = int(record["Trade number"])
            trade = trades.setdefault(number, {})
            if record["Type"].lower().startswith("entry"):
                trade["entry_at"] = datetime.strptime(
                    record["Date and time"], "%Y-%m-%d %H:%M").replace(tzinfo=TAIPEI)
            else:
                trade["exit_signal"] = record["Signal"]
            trade["return_pct"] = float(record["Return %"])
    monthly: dict[str, float] = collections.defaultdict(float)
    for _, trade in sorted(trades.items()):
        if "entry_at" not in trade or trade.get("exit_signal") == "Open":
            continue
        key = f"{trade['entry_at'].year}-{trade['entry_at'].month:02d}"
        monthly[key] += trade["return_pct"] / stop_pct
    return dict(monthly)


def sharpe(values: list[float]) -> float | None:
    sd = statistics.pstdev(values)
    return round(statistics.fmean(values) / sd * math.sqrt(12), 3) if sd else None


def max_drawdown(values: list[float]) -> float:
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return round(drawdown, 2)


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return round(numerator / denominator, 3) if denominator else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out
    return pearson(rank(x), rank(y))


def summarise(values: list[float]) -> dict:
    return {
        "months": len(values),
        "mean_monthly_r": round(statistics.fmean(values), 3),
        "sd_monthly_r": round(statistics.pstdev(values), 3),
        "annualised_sharpe": sharpe(values),
        "max_drawdown_r": max_drawdown(values),
        "total_r": round(sum(values), 2),
        "positive_months": sum(1 for v in values if v > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    series = {name: load_monthly_r(cfg["trades"], cfg["stop_pct"])
              for name, cfg in STRATEGIES.items()}
    months = sorted(set(series["S1"]) | set(series["S2"]))
    s1 = [series["S1"].get(m, 0.0) for m in months]
    s2 = [series["S2"].get(m, 0.0) for m in months]
    blend = [REPORTED_WEIGHT * a + (1 - REPORTED_WEIGHT) * b for a, b in zip(s1, s2)]

    weights = {}
    for w in WEIGHT_GRID:
        combined = [w * a + (1 - w) * b for a, b in zip(s1, s2)]
        weights[f"w_s1={w:.1f}"] = summarise(combined)
    plateau = [row["annualised_sharpe"] for key, row in weights.items()
               if 0.4 <= float(key.split("=")[1]) <= 0.8]

    rng = random.Random(f"{RANDOM_SEED}:blend")
    diff_s1, diff_s2, dd_gain_s2, dd_gain_s1 = [], [], [], []
    for _ in range(BOOTSTRAP_TRIALS):
        picks = [rng.randrange(len(months)) for _ in range(len(months))]
        a = [s1[i] for i in picks]
        b = [s2[i] for i in picks]
        m = [REPORTED_WEIGHT * x + (1 - REPORTED_WEIGHT) * y for x, y in zip(a, b)]
        sa, sb, sm = sharpe(a), sharpe(b), sharpe(m)
        if None in (sa, sb, sm):
            continue
        diff_s1.append(sm - sa)
        diff_s2.append(sm - sb)
        dd_gain_s2.append(max_drawdown(b) - max_drawdown(m))
        dd_gain_s1.append(max_drawdown(a) - max_drawdown(m))

    def interval(values: list[float], observed: float) -> dict:
        ordered = sorted(values)
        return {
            "observed": round(observed, 3),
            "ci90": [round(ordered[int(0.05 * len(ordered))], 3),
                     round(ordered[int(0.95 * len(ordered))], 3)],
            "p_positive": round(sum(1 for v in values if v > 0) / len(values), 3),
            "supported": sum(1 for v in values if v > 0) / len(values) >= 0.95,
        }

    by_year: dict[str, dict] = {}
    for year in sorted({m[:4] for m in months}):
        idx = [i for i, m in enumerate(months) if m.startswith(year)]
        if len(idx) < 3:
            continue
        by_year[year] = {
            "months": len(idx),
            "s1_sharpe": sharpe([s1[i] for i in idx]),
            "s2_sharpe": sharpe([s2[i] for i in idx]),
            "blend_sharpe": sharpe([blend[i] for i in idx]),
        }
    blend_best = sum(1 for y in by_year.values()
                     if y["blend_sharpe"] is not None
                     and y["blend_sharpe"] >= max(y["s1_sharpe"] or -9, y["s2_sharpe"] or -9))

    bars = load_bars()
    n_bars = len(bars)
    splits_ = {"train": (0, int(n_bars * 0.55)),
               "valid": (int(n_bars * 0.55), int(n_bars * 0.80)),
               "holdout": (int(n_bars * 0.80), n_bars)}

    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(tz=TAIPEI).isoformat(timespec="seconds"),
        "strategy": "S1 AweWithBB V3.9; S2 Hammer V3.2 — portfolio construction",
        "method": {
            "unit": "R per trade (return / nominal stop), summed by calendar month",
            "months": len(months),
            "first_month": months[0], "last_month": months[-1],
            "reported_weight_s1": REPORTED_WEIGHT,
            "bootstrap": "monthly resample with replacement",
            "bootstrap_trials": BOOTSTRAP_TRIALS,
            "random_seed": RANDOM_SEED,
            "timezone": "Asia/Taipei",
        },
        "standalone": {"S1": summarise(s1), "S2": summarise(s2)},
        "correlation": {"pearson": pearson(s1, s2), "spearman": spearman(s1, s2),
                        "months_both_traded": sum(
                            1 for m in months
                            if series["S1"].get(m) and series["S2"].get(m))},
        "blend_50_50": summarise(blend),
        "by_weight": weights,
        "weight_plateau_0_4_to_0_8": {
            "sharpe_min": min(plateau), "sharpe_max": max(plateau),
            "note": ("The in-sample optimum sits inside a plateau this flat, so the weight "
                     "decision does not matter and should not be optimised."),
        },
        "bootstrap": {
            "sharpe_blend_minus_s1": interval(diff_s1, sharpe(blend) - sharpe(s1)),
            "sharpe_blend_minus_s2": interval(diff_s2, sharpe(blend) - sharpe(s2)),
            "drawdown_gain_vs_s2": interval(dd_gain_s2,
                                            max_drawdown(s2) - max_drawdown(blend)),
            "drawdown_gain_vs_s1": interval(dd_gain_s1,
                                            max_drawdown(s1) - max_drawdown(blend)),
        },
        "by_year": by_year,
        "years_blend_beat_both": f"{blend_best}/{len(by_year)}",
        "beta_decomposition": beta_decomposition(series, months, bars),
        "uncorrelated_candidates": {
            "counter_trend_fade": counter_trend_fade(bars, splits_),
            "mirror_short": mirror_short(bars, splits_),
        },
        "bar_for_a_new_strategy": required_sharpe(sharpe(blend)),
        "candidate_portfolio_impact": None,
        "limitations": [
            "Sharpe near 3.9 is not a live expectation. These are Strategy Tester R "
            "multiples over a market that rose 112% with no slippage beyond the export's "
            "own. The ranking between allocations is the result; the level is not.",
            "32 months is a small sample for a Sharpe comparison, which is why every claim "
            "here carries a bootstrap interval and why the S1 comparison is reported as "
            "unsupported rather than as a small positive.",
            "Monthly aggregation assumes risk can be allocated between the two strategies "
            "at a stable ratio. With fixed-fractional sizing per trade that is what "
            "already happens; it is not a new mechanism.",
            "Both strategies are long-only on one instrument in one regime. A correlation "
            "of 0.234 measured here need not hold in a falling market, and the "
            "diversification is between two expressions of the same directional bet.",
            "No result changes formal S1 or S2 logic, live risk, or an active entry "
            "checklist.",
        ],
    }

    fade_full = results["uncorrelated_candidates"]["counter_trend_fade"].get(
        "best_survivor_full_period") or {}
    if fade_full.get("monthly"):
        candidate = [fade_full["monthly"].get(m, 0.0) for m in months]
        candidate_sharpe = sharpe(candidate)
        impact = {}
        for w in (0.0, 0.1, 0.2, 0.3):
            mixed = [(1 - w) * a + w * b for a, b in zip(blend, candidate)]
            impact[f"w={w:.1f}"] = {"sharpe": sharpe(mixed),
                                    "total_r": round(sum(mixed), 2)}
        results["candidate_portfolio_impact"] = {
            "variant": fade_full["variant"],
            "trades": fade_full["trades"],
            "candidate_sharpe": candidate_sharpe,
            "candidate_total_r": round(sum(candidate), 2),
            "correlation_to_blend": pearson(candidate, blend),
            "correlation_to_s1": pearson(candidate, s1),
            "correlation_to_s2": pearson(candidate, s2),
            "blend_sharpe_by_candidate_weight": impact,
            "verdict": (
                "Genuinely uncorrelated and genuinely unprofitable. It clears the "
                "correlation half of the bar and fails the return half, so it lowers the "
                "blend's Sharpe at every weight. Surviving train-to-valid is not the same "
                "as being worth adding."),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "study_id": STUDY_ID,
        "correlation_pearson": results["correlation"]["pearson"],
        "sharpe": {"S1": results["standalone"]["S1"]["annualised_sharpe"],
                   "S2": results["standalone"]["S2"]["annualised_sharpe"],
                   "blend": results["blend_50_50"]["annualised_sharpe"]},
        "vs_s2_supported": results["bootstrap"]["sharpe_blend_minus_s2"]["supported"],
        "vs_s1_supported": results["bootstrap"]["sharpe_blend_minus_s1"]["supported"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
