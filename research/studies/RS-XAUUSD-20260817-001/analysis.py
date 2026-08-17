#!/usr/bin/env python3
"""RS-XAUUSD-20260817-001 — Macro factor attribution, redundancy, and GVZ as a continuum.

Batch one of the review that followed revoking RS-XAUUSD-20260727-001's policy impacts.
That revocation showed the Macro *composite* cannot separate outcomes at this sample
size. It did not ask the next question: whether any individual factor does, whether the
composite double-counts its inputs, and whether GVZ has any breakpoint at all rather than
the assumed 13/20.

Three deliberate constraints, each a reaction to how the revoked rules failed:

1. Every group reports the smallest win-rate gap its own sample size could resolve.
   A rank score was previously assigned to buckets holding a median of 9 trades; stating
   the resolution limit next to every number makes that impossible to repeat silently.
2. Group differences are tested against a simulation in which the grouping has no effect,
   because with enough buckets a spread always appears. The revoked 30-minute-slot
   breakdown produced a 70-point spread that a no-effect null exceeds 91% of the time.
3. GVZ is swept across every candidate threshold and the whole curve is reported, never
   only its best point. Picking the best cut on 472 trades is how an overfit becomes a
   rule.

No lookahead: macro values come from the export's lookahead_off daily columns, and each
trade takes the value on its own entry bar, which already reflects the last closed daily
bar.

Usage:
    python3.12 -m scripts.research.build_xauusd_macro_attribution
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.research.lib.fail_pattern_toolkit import load_trades, stats, wilson_interval


ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "RS-XAUUSD-20260817-001"
PACKAGE = Path("reproduced-macro-attribution")
TAIPEI = timezone(timedelta(hours=8))

TRADES = {
    "S1": ROOT / "local-inputs/s1-v3.9-trades.csv",
    "S2": ROOT / "local-inputs/s2-v3.2-trades.csv",
}
EXPORT = ROOT / "local-inputs/s1s2-export-30m.csv"

# The panel's own composite, replicated so it can be tested rather than trusted.
# Real Rate is US10Y minus T10YIE and carries weight 2 while US10Y is also scored on its
# own, so up to 3 of 6 points move with a single input. Quantified in `redundancy`.
MACRO_WEIGHTS = {"real_rate": 2, "us10y": 1, "dxy": 1, "vix": 1, "gold_trend": 1}
MA_LEN = 50
SIM_TRIALS = 20000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def min_detectable_pp(n1: int, n2: int, baseline: float, alpha: float = 0.05, power: float = 0.80) -> float | None:
    """Smallest win-rate gap two groups of this size could resolve, in percentage points.

    Reported beside every comparison. Without it a 4-point gap on n=82 reads the same as
    a 4-point gap on n=2000.
    """
    if not n1 or not n2:
        return None
    z_a, z_b = 1.959964, 0.841621
    effective = 2 / (1 / n1 + 1 / n2)
    low, high = 0.0001, 0.60
    for _ in range(80):
        mid = (low + high) / 2
        p1, p2 = baseline, min(baseline + mid, 0.999)
        pooled = (p1 + p2) / 2
        need = (
            z_a * math.sqrt(2 * pooled * (1 - pooled))
            + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2 / (p1 - p2) ** 2
        if need > effective:
            low = mid
        else:
            high = mid
    return round((low + high) / 2 * 100, 1)


def spread_vs_noise(group_sizes: list[int], baseline: float, observed_spread: float, seed: int) -> dict:
    """How often a no-effect null produces a best-to-worst spread this wide.

    Splitting a fixed number of trades into more buckets widens the observed spread by
    construction. This separates a real effect from that arithmetic.
    """
    sizes = [n for n in group_sizes if n]
    if len(sizes) < 2:
        return {"applicable": False}
    rng = random.Random(seed)
    spreads = []
    for _ in range(SIM_TRIALS):
        rates = [sum(1 for _ in range(n) if rng.random() < baseline) / n for n in sizes]
        spreads.append((max(rates) - min(rates)) * 100)
    spreads.sort()
    at_least = sum(1 for s in spreads if s >= observed_spread) / SIM_TRIALS
    return {
        "applicable": True,
        "trials": SIM_TRIALS,
        "groups": len(sizes),
        "median_trades_per_group": int(np.median(sizes)),
        "observed_spread_pp": round(observed_spread, 1),
        "null_median_spread_pp": round(spreads[SIM_TRIALS // 2], 1),
        "p_spread_at_least_observed": round(at_least, 3),
        "separable": bool(at_least < 0.05),
    }


def load_export() -> pd.DataFrame:
    frame = pd.read_csv(EXPORT, encoding="utf-8-sig")
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(TAIPEI).dt.tz_localize(None)
    frame = frame.sort_values("time").reset_index(drop=True)

    # Rebuild the daily series the panel compares against. The export carries the last
    # closed daily close on every 30m bar, so one value per calendar date is the series.
    frame["date"] = frame["time"].dt.date
    daily = frame.groupby("date").last()[
        ["MACRO_US10Y", "MACRO_T10YIE", "MACRO_DXY", "MACRO_VIX", "MACRO_GVZ", "close"]
    ].copy()
    daily["real_rate"] = daily["MACRO_US10Y"] - daily["MACRO_T10YIE"]
    for column in ("MACRO_US10Y", "MACRO_DXY", "MACRO_VIX", "real_rate", "close"):
        daily[f"{column}_ma"] = daily[column].ewm(span=MA_LEN, adjust=False).mean()

    # Panel semantics: gold-bullish when yields/real-rate/dollar sit BELOW their average,
    # when volatility sits above it, and when gold itself sits above it.
    flags = pd.DataFrame(index=daily.index)
    flags["real_rate"] = daily["real_rate"] < daily["real_rate_ma"]
    flags["us10y"] = daily["MACRO_US10Y"] < daily["MACRO_US10Y_ma"]
    flags["dxy"] = daily["MACRO_DXY"] < daily["MACRO_DXY_ma"]
    flags["vix"] = daily["MACRO_VIX"] > daily["MACRO_VIX_ma"]
    flags["gold_trend"] = daily["close"] > daily["close_ma"]
    flags["score"] = sum(flags[name].astype(int) * weight for name, weight in MACRO_WEIGHTS.items())
    flags["verdict"] = np.select(
        [flags["score"] >= 5, flags["score"] <= 2], ["STRONG BUY", "WAIT"], default="NEUTRAL"
    )
    flags["gvz"] = daily["MACRO_GVZ"]
    return flags.reset_index()


def attach(trades: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Attach each trade the macro state of the PREVIOUS calendar day.

    Backward only. The entry bar's own date would carry a daily close that, intraday, has
    not happened yet.
    """
    frame = trades.copy()
    frame["date"] = frame["entry_time"].dt.date
    lookup = flags.copy()
    lookup["date"] = pd.to_datetime(lookup["date"]) + pd.Timedelta(days=1)
    lookup["date"] = lookup["date"].dt.date
    merged = frame.merge(lookup, on="date", how="left", suffixes=("", "_macro"))
    return merged[merged["score"].notna()].copy()


def group_block(frame: pd.DataFrame, column: str, baseline: float, seed: int) -> dict:
    groups = {str(key): stats(group) for key, group in frame.groupby(column, observed=True)}
    rates = [g["win_rate_pct"] for g in groups.values() if g["win_rate_pct"] is not None]
    sizes = [g["n"] for g in groups.values()]
    spread = (max(rates) - min(rates)) if len(rates) > 1 else 0.0
    ordered = sorted(groups.items(), key=lambda kv: -(kv[1]["n"] or 0))
    resolvable = None
    if len(ordered) >= 2:
        resolvable = min_detectable_pp(ordered[0][1]["n"], ordered[1][1]["n"], baseline)
    return {
        "groups": groups,
        "observed_spread_pp": round(spread, 2),
        "min_detectable_pp_two_largest_groups": resolvable,
        "noise_test": spread_vs_noise(sizes, baseline, spread, seed),
    }


def redundancy(frame: pd.DataFrame) -> dict:
    names = list(MACRO_WEIGHTS)
    agree = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            agree[f"{a}|{b}"] = round(float((frame[a] == frame[b]).mean()) * 100, 1)
    both = frame["us10y"] & frame["real_rate"]
    neither = (~frame["us10y"]) & (~frame["real_rate"])
    return {
        "pairwise_agreement_pct": agree,
        "us10y_real_rate": {
            "agreement_pct": agree["real_rate|us10y"],
            "points_moving_together": MACRO_WEIGHTS["us10y"] + MACRO_WEIGHTS["real_rate"],
            "total_points": sum(MACRO_WEIGHTS.values()),
            "share_of_score_pct": round(
                100 * (MACRO_WEIGHTS["us10y"] + MACRO_WEIGHTS["real_rate"]) / sum(MACRO_WEIGHTS.values()), 1
            ),
            "both_bullish_pct": round(float(both.mean()) * 100, 1),
            "both_bearish_pct": round(float(neither.mean()) * 100, 1),
            "note": "real_rate is US10Y minus T10YIE, so the two flags share an input by "
                    "construction. When they agree, one series drives this share of the score.",
        },
    }


def gvz_curve(frame: pd.DataFrame, baseline: float) -> dict:
    values = frame["gvz"].dropna()
    deciles = {}
    if len(values) >= 20:
        frame = frame.copy()
        frame["gvz_decile"] = pd.qcut(frame["gvz"], 10, labels=False, duplicates="drop")
        for key, group in frame.groupby("gvz_decile", observed=True):
            deciles[f"d{int(key) + 1}"] = {
                "gvz_range": [round(float(group["gvz"].min()), 2), round(float(group["gvz"].max()), 2)],
                **stats(group),
            }
    sweep = []
    for threshold in range(int(values.min()) + 1, int(values.max())):
        below, above = frame[frame["gvz"] <= threshold], frame[frame["gvz"] > threshold]
        if len(below) < 20 or len(above) < 20:
            continue
        b, a = stats(below), stats(above)
        sweep.append({
            "threshold": threshold,
            "below": {"n": b["n"], "win_rate_pct": b["win_rate_pct"], "profit_factor": b["profit_factor"]},
            "above": {"n": a["n"], "win_rate_pct": a["win_rate_pct"], "profit_factor": a["profit_factor"]},
            "win_rate_gap_pp": round((a["win_rate_pct"] or 0) - (b["win_rate_pct"] or 0), 2),
            "min_detectable_pp": min_detectable_pp(b["n"], a["n"], baseline),
        })
    best = max(sweep, key=lambda s: abs(s["win_rate_gap_pp"])) if sweep else None

    # The sweep searches many thresholds, so its best gap is biased upward by the search
    # itself. Permuting GVZ against outcomes destroys any real relationship while keeping
    # the same trade count, the same GVZ distribution and the same number of candidate
    # cuts, so the whole search is repeated under a known-null and the observed best is
    # compared against that distribution rather than against a single-comparison table.
    permutation = {"applicable": False}
    if sweep and best is not None:
        rng = np.random.default_rng(20260817)
        outcomes = (frame["net_pnl_usd"] > 0).to_numpy()
        gvz_values = frame["gvz"].to_numpy()
        thresholds = [item["threshold"] for item in sweep]
        best_gaps = []
        for _ in range(2000):
            shuffled = rng.permutation(outcomes)
            gaps = []
            for threshold in thresholds:
                mask = gvz_values > threshold
                above, below = shuffled[mask], shuffled[~mask]
                if len(above) < 20 or len(below) < 20:
                    continue
                gaps.append(abs(above.mean() - below.mean()) * 100)
            if gaps:
                best_gaps.append(max(gaps))
        if best_gaps:
            observed = abs(best["win_rate_gap_pp"])
            at_least = sum(1 for g in best_gaps if g >= observed) / len(best_gaps)
            best_gaps.sort()
            permutation = {
                "applicable": True,
                "trials": len(best_gaps),
                "observed_best_gap_pp": round(observed, 2),
                "null_median_best_gap_pp": round(best_gaps[len(best_gaps) // 2], 2),
                "null_95th_best_gap_pp": round(best_gaps[int(len(best_gaps) * 0.95)], 2),
                "p_best_gap_at_least_observed": round(at_least, 4),
                "survives_multiple_comparison": bool(at_least < 0.05),
                "note": "Accounts for having searched every threshold. A sweep on unrelated "
                        "data still produces a best gap of the null median size.",
            }
    return {
        "deciles": deciles,
        "threshold_sweep": sweep,
        "largest_gap_threshold": best,
        "permutation_test": permutation,
        "current_rule_thresholds": {"squeeze_below": 13, "extreme_above": 20},
        "interpretation_note": "The sweep is reported whole. Selecting the threshold with the "
            "largest gap is a fit to this sample, and the gap must clear "
            "min_detectable_pp before it means anything at all.",
    }


def build(name: str, trades_path: Path, flags: pd.DataFrame, seed: int) -> dict:
    trades = load_trades(trades_path)
    joined = attach(trades, flags)
    base = stats(joined)
    baseline = (base["win_rate_pct"] or 50) / 100
    factors = {}
    for factor in MACRO_WEIGHTS:
        block = group_block(joined, factor, baseline, seed)
        block["groups"] = {("bullish" if k == "True" else "bearish"): v for k, v in block["groups"].items()}
        factors[factor] = block
    return {
        "trades_total": len(trades),
        "trades_with_macro": len(joined),
        "macro_coverage_pct": round(100 * len(joined) / len(trades), 2),
        "baseline": base,
        "by_factor": factors,
        "by_verdict": group_block(joined, "verdict", baseline, seed + 1),
        "by_score": group_block(joined, "score", baseline, seed + 2),
        "redundancy": redundancy(joined),
        "gvz": gvz_curve(joined, baseline),
    }


def save_chart(chart_id: str, title: str, fig: plt.Figure) -> dict:
    charts = PACKAGE / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    path = charts / f"{chart_id}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    # `file` is the bare name: consumers join it onto the charts directory themselves.
    return {"id": chart_id, "file": f"{chart_id}.png", "title": title, "section": "macro"}


def build_charts(results: dict) -> list[dict]:
    charts = []
    for name, block in results["strategies"].items():
        # Factor gaps against the resolution limit. The bars are what was measured; the
        # line is what the sample could have detected. Every bar sitting under the line is
        # the finding.
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        factors = list(block["by_factor"])
        gaps = [block["by_factor"][f]["observed_spread_pp"] for f in factors]
        limits = [block["by_factor"][f]["min_detectable_pp_two_largest_groups"] for f in factors]
        ax.bar(factors, gaps, color="#4C78A8", label="observed win-rate gap")
        ax.plot(factors, limits, "o--", color="#E45756", label="smallest gap this sample can resolve")
        ax.set_ylabel("percentage points")
        ax.set_title(f"{name} — per-factor gap vs detection limit")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        charts.append(save_chart(f"{name.lower()}_factor_gaps", f"{name} factor gaps vs detection limit", fig))

        deciles = block["gvz"]["deciles"]
        if deciles:
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            labels = list(deciles)
            wr = [deciles[d]["win_rate_pct"] for d in labels]
            mids = [sum(deciles[d]["gvz_range"]) / 2 for d in labels]
            ax.plot(mids, wr, "o-", color="#4C78A8")
            ax.axhline(block["baseline"]["win_rate_pct"], color="#888", ls=":", label="baseline")
            for threshold in (13, 20):
                ax.axvline(threshold, color="#E45756", ls="--", alpha=0.6)
            ax.set_xlabel("GVZ (decile midpoint); dashed red = current 13 / 20 rule thresholds")
            ax.set_ylabel("win rate %")
            ax.set_title(f"{name} — win rate across the GVZ range")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            charts.append(save_chart(f"{name.lower()}_gvz_deciles", f"{name} win rate by GVZ decile", fig))

        sweep = block["gvz"]["threshold_sweep"]
        perm = block["gvz"]["permutation_test"]
        if sweep and perm.get("applicable"):
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            xs = [item["threshold"] for item in sweep]
            ys = [abs(item["win_rate_gap_pp"]) for item in sweep]
            ax.plot(xs, ys, "o-", color="#4C78A8", label="|win-rate gap| at this cut")
            ax.axhline(perm["null_median_best_gap_pp"], color="#888", ls=":",
                       label="median best gap when GVZ is shuffled")
            ax.axhline(perm["null_95th_best_gap_pp"], color="#E45756", ls="--",
                       label="95th percentile under shuffling")
            ax.set_xlabel("GVZ threshold")
            ax.set_ylabel("percentage points")
            ax.set_title(f"{name} — every threshold, against a shuffled null (p={perm['p_best_gap_at_least_observed']})")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            charts.append(save_chart(f"{name.lower()}_gvz_sweep", f"{name} GVZ threshold sweep vs null", fig))
    return charts


def report_html(results: dict) -> str:
    import base64
    parts = ["<!doctype html><meta charset='utf-8'>",
             f"<title>{STUDY_ID}</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.6}"
             "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ddd;padding:6px 8px;font-size:14px}"
             "th{background:#f4f4f4;text-align:left}img{max-width:100%;margin:1rem 0}code{background:#f4f4f4;padding:1px 4px}</style>",
             f"<h1>{STUDY_ID}</h1><p>{results['strategy']}</p>",
             f"<p>Generated {results['generated_at']}. Macro period {results['macro_period']['start']} to "
             f"{results['macro_period']['end']} ({results['macro_period']['days']} days).</p>"]
    for name, block in results["strategies"].items():
        b = block["baseline"]
        parts.append(f"<h2>{name}</h2><p>Baseline n={b['n']}, win rate {b['win_rate_pct']}%, "
                     f"PF {b['profit_factor']}. Macro coverage {block['macro_coverage_pct']}%.</p>")
        parts.append("<table><tr><th>Factor</th><th>bullish n / WR</th><th>bearish n / WR</th>"
                     "<th>gap (pp)</th><th>resolvable (pp)</th><th>p vs noise</th></tr>")
        for factor, blk in block["by_factor"].items():
            bull, bear = blk["groups"].get("bullish", {}), blk["groups"].get("bearish", {})
            parts.append(
                f"<tr><td>{factor}</td><td>{bull.get('n')} / {bull.get('win_rate_pct')}%</td>"
                f"<td>{bear.get('n')} / {bear.get('win_rate_pct')}%</td><td>{blk['observed_spread_pp']}</td>"
                f"<td>{blk['min_detectable_pp_two_largest_groups']}</td>"
                f"<td>{blk['noise_test'].get('p_spread_at_least_observed')}</td></tr>")
        parts.append("</table>")
        perm = block["gvz"]["permutation_test"]
        if perm.get("applicable"):
            best = block["gvz"]["largest_gap_threshold"]
            parts.append(
                f"<p><strong>GVZ.</strong> Largest gap is {best['win_rate_gap_pp']}pp at threshold "
                f"{best['threshold']}. Shuffling outcomes and repeating the whole threshold search gives "
                f"a best gap at least this large {perm['p_best_gap_at_least_observed']:.1%} of the time, so it "
                f"{'survives' if perm['survives_multiple_comparison'] else 'does not survive'} "
                f"correction for having searched.</p>")
    for chart in results["charts"]:
        data = base64.b64encode((PACKAGE / "charts" / chart["file"]).read_bytes()).decode()
        parts.append(f"<h3>{chart['title']}</h3><img src='data:image/png;base64,{data}'>")
    return "\n".join(parts)


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    flags = load_export()
    generated_at = datetime.now(tz=TAIPEI).isoformat(timespec="seconds")
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": generated_at,
        "strategy": "S1 AweWithBB V3.9 and S2 Hammer V3.2 — Macro factor attribution",
        "method": {
            "macro_source": "S1S2-Export-V2 lookahead_off daily columns, rebuilt to one row per date",
            "macro_assignment": "previous calendar day's macro state; backward only, never same-day",
            "composite": "panel weights replicated: real_rate x2, us10y/dxy/vix/gold_trend x1; "
                         "verdict STRONG BUY >=5, WAIT <=2, else NEUTRAL",
            "moving_average": f"EMA{MA_LEN} on the daily series",
            "win_definition": "net_pnl_usd > 0",
            "power": "every comparison reports the smallest win-rate gap its two largest "
                     "groups could resolve at alpha 0.05, power 0.80",
            "noise_test": f"{SIM_TRIALS}-run simulation holding all groups at the baseline "
                          "win rate; reports how often a no-effect null matches the observed spread",
            "gvz": "reported as deciles and as a full threshold sweep, not a single chosen cut",
        },
        "macro_period": {
            "start": str(flags["date"].min()),
            "end": str(flags["date"].max()),
            "days": len(flags),
        },
        "strategies": {name: build(name, path, flags, seed=11 * (index + 1))
                       for index, (name, path) in enumerate(TRADES.items())},
    }
    results["charts"] = build_charts(results)
    (PACKAGE / "report.html").write_text(report_html(results), encoding="utf-8")
    (PACKAGE / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "study": STUDY_ID,
        "macro_days": len(flags),
        **{f"{k}_trades": v["trades_with_macro"] for k, v in results["strategies"].items()},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
