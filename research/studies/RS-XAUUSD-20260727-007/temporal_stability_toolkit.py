"""Temporal stability / chronological holdout toolkit — reference implementation of
docs/RESEARCH_DEVELOPMENT_SPEC.md section 5.1 item 11 (owner-directed 2026-07-29).

Reuses fail_pattern_toolkit's trade loader and stats() so a bucket/holdout stat is
computed identically to every other section 5 breakdown, rather than re-deriving win
rate / profit factor / Wilson CI a second way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fail_pattern_toolkit as fp  # noqa: E402

TEMPORAL_STABILITY_LIMITATION = (
    "This is a chronological-bucket / holdout-split check on ONE static TradingView "
    "strategy-tester export, not a re-optimized walk-forward: the underlying Pine "
    "Script strategy logic is never re-run or re-tuned on any sub-window here. It can "
    "show whether the recorded win rate / profit factor is stable or concentrated in "
    "an earlier stretch of the export; it cannot prove the strategy's parameters would "
    "have been chosen the same way if only the early data had ever existed."
)


def quarterly_bucket_stats(trades: pd.DataFrame) -> dict:
    """stats() per calendar quarter of entry_time. The most recent bucket may be a
    partial quarter if the export ends mid-quarter — callers should not hide this."""
    quarters = trades["entry_time"].dt.to_period("Q").astype(str)
    return {str(q): fp.stats(group) for q, group in trades.groupby(quarters, observed=True)}


def chronological_holdout(trades: pd.DataFrame, split_ratio: float = 0.7) -> dict:
    """Sort by entry_time; first split_ratio is in_sample, remainder is held_out."""
    ordered = trades.sort_values("entry_time").reset_index(drop=True)
    split_idx = int(len(ordered) * split_ratio)
    in_sample = ordered.iloc[:split_idx]
    held_out = ordered.iloc[split_idx:]

    def _period(frame: pd.DataFrame) -> dict:
        if len(frame) == 0:
            return {"start": None, "end": None}
        return {"start": str(frame["entry_time"].min()), "end": str(frame["exit_time"].max())}

    return {
        "split_ratio": split_ratio,
        "in_sample": {"period": _period(in_sample), **fp.stats(in_sample)},
        "held_out": {"period": _period(held_out), **fp.stats(held_out)},
    }


def degradation_flag(holdout: dict) -> str:
    """Fixed, pre-defined rule — computed the same way for every study so the label is
    never chosen after looking at the number. See spec section 5.1 item 11."""
    in_sample, held_out = holdout["in_sample"], holdout["held_out"]
    if held_out["win_rate_pct"] is None or in_sample["win_rate_pct"] is None:
        return "insufficient_data"
    in_ci = in_sample["win_rate_ci95_pct"]
    held_out_pf = held_out["profit_factor"] or 0.0
    in_sample_pf = in_sample["profit_factor"] or 0.0
    if held_out["win_rate_pct"] < in_ci[0] or held_out_pf < 1.0:
        return "degraded"
    if held_out["win_rate_pct"] > in_ci[1] and held_out_pf > in_sample_pf:
        return "improved"
    return "stable"


def chart_quarterly_stability(
    trades: pd.DataFrame, by_period: dict, split_ratio: float, strategy_id: str, version: str
) -> plt.Figure:
    ordered = trades.sort_values("entry_time").reset_index(drop=True)
    split_idx = int(len(ordered) * split_ratio)
    held_out_start_period = ordered.iloc[split_idx]["entry_time"].to_period("Q")

    quarters = list(by_period.keys())
    win_rates = [by_period[q]["win_rate_pct"] or 0 for q in quarters]
    ns = [by_period[q]["n"] for q in quarters]
    total_wins = sum(by_period[q]["wins"] for q in quarters)
    total_n = sum(ns)
    overall_wr = 100 * total_wins / total_n if total_n else 0

    colors = [
        "#e67e22" if pd.Period(q, freq="Q") >= held_out_start_period else "#3498db"
        for q in quarters
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(quarters, win_rates, color=colors)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"n={n}",
                 ha="center", fontsize=7.5, color="#555")
    ax.axhline(overall_wr, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Full-period WR {overall_wr:.1f}%")
    ax.axhline(50, color="#bbb", linestyle=":", linewidth=1)
    ax.set_ylabel("Win rate %")
    ax.set_ylim(0, max(win_rates + [overall_wr]) * 1.25)
    ax.set_title(f"{strategy_id} {version} — Quarterly win rate (blue=in-sample, orange=held-out {int((1-split_ratio)*100)}%)")
    ax.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig
