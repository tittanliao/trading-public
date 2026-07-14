"""
Run all 20 long (or 20 short) strategies and produce a ranked summary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import run_backtest, summary_stats
from .indicators import add_all
from .loader import load_price
from .strategies import LONG_STRATEGIES
from .strategies_short import SHORT_STRATEGIES

DEFAULT_SL = 30
DEFAULT_TP = 60
DEFAULT_TIME_STOP = 48
LONG_CSV = "TAIFEX_DLY_MXF1!, 30.csv"
OUTPUT_LONG = Path(__file__).parent.parent / "TX-Long-Experiments"
OUTPUT_SHORT = Path(__file__).parent.parent / "TX-Short-Experiments"


def run_all_long(csv_file: str = LONG_CSV, sl_pts: int = DEFAULT_SL, tp_pts: int = DEFAULT_TP) -> list[dict]:
    df = _load_with_indicators(csv_file)
    results = []
    for code, (fn, name, group) in LONG_STRATEGIES.items():
        signals = fn(df)
        trades = run_backtest(df, signals, direction="long", sl_pts=sl_pts, tp_pts=tp_pts)
        stats = summary_stats(trades, sl_pts, tp_pts) if not trades.empty else {}
        results.append(_build_row(code, name, group, stats, trades))

    results = _rank(results)
    _save(results, trades_map=_collect_trades(df, LONG_STRATEGIES, "long", sl_pts, tp_pts),
          output_dir=OUTPUT_LONG, direction="long", sl_pts=sl_pts, tp_pts=tp_pts)
    return results


def run_all_short(csv_file: str = LONG_CSV, sl_pts: int = DEFAULT_SL, tp_pts: int = DEFAULT_TP) -> list[dict]:
    df = _load_with_indicators(csv_file)
    results = []
    for code, (fn, name, group) in SHORT_STRATEGIES.items():
        signals = fn(df)
        trades = run_backtest(df, signals, direction="short", sl_pts=sl_pts, tp_pts=tp_pts)
        stats = summary_stats(trades, sl_pts, tp_pts) if not trades.empty else {}
        results.append(_build_row(code, name, group, stats, trades))

    results = _rank(results)
    _save(results, trades_map=_collect_trades(df, SHORT_STRATEGIES, "short", sl_pts, tp_pts),
          output_dir=OUTPUT_SHORT, direction="short", sl_pts=sl_pts, tp_pts=tp_pts)
    return results


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

def _load_with_indicators(csv_file: str) -> pd.DataFrame:
    df = load_price(csv_file)
    df = df[df["session"] != "closed"]
    return add_all(df)


def _build_row(code: str, name: str, group: str, stats: dict, trades: pd.DataFrame) -> dict:
    if not stats:
        return dict(code=code, name=name, group=group, n_trades=0,
                    win_rate=0, profit_factor=0, net_pnl_pts=0,
                    net_pnl_ntd=0, max_dd_ntd=0, score=0)
    fail_dist = _fail_dist(trades)
    session_wr = _session_win_rate(trades)
    return dict(
        code=code, name=name, group=group,
        n_trades=stats["n_trades"],
        win_rate=round(stats["win_rate"], 1),
        profit_factor=round(stats["profit_factor"], 3),
        net_pnl_pts=round(stats["net_pnl_pts"], 1),
        net_pnl_ntd=round(stats["net_pnl_ntd"], 0),
        max_dd_ntd=round(stats["max_dd_ntd"], 0),
        score=0,
        fail_immediate=fail_dist.get("immediate_loss", 0),
        fail_false_break=fail_dist.get("false_breakout", 0),
        fail_time_bleed=fail_dist.get("time_bleed", 0),
        fail_normal_sl=fail_dist.get("normal_sl", 0),
        day_win_rate=session_wr.get("day", np.nan),
        night_win_rate=session_wr.get("night", np.nan),
    )


def _rank(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    if df["n_trades"].sum() == 0:
        return rows

    wr_norm = df["win_rate"] / 50
    pf_norm = df["profit_factor"].apply(lambda x: math.log(max(x, 0.01)) / math.log(3))
    max_pnl = df["net_pnl_ntd"].abs().max()
    pnl_norm = df["net_pnl_ntd"] / (max_pnl if max_pnl > 0 else 1)

    df["score"] = (wr_norm * 0.35 + pf_norm * 0.40 + pnl_norm * 0.25).round(3)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.to_dict("records")


def _collect_trades(df, registry, direction, sl_pts, tp_pts) -> dict[str, pd.DataFrame]:
    out = {}
    for code, (fn, _, _) in registry.items():
        sigs = fn(df)
        trades = run_backtest(df, sigs, direction=direction, sl_pts=sl_pts, tp_pts=tp_pts)
        out[code] = trades
    return out


def _fail_dist(trades: pd.DataFrame) -> dict:
    if trades.empty or "fail_pattern" not in trades.columns:
        return {}
    losses = trades[~trades["win"]]
    if losses.empty:
        return {}
    dist = losses["fail_pattern"].value_counts(normalize=True) * 100
    return dist.round(1).to_dict()


def _session_win_rate(trades: pd.DataFrame) -> dict:
    if trades.empty or "session" not in trades.columns:
        return {}
    out = {}
    for sess, grp in trades.groupby("session"):
        if len(grp) >= 5:
            out[sess] = round(grp["win"].mean() * 100, 1)
    return out


def _save(results: list[dict], trades_map: dict, output_dir: Path,
          direction: str, sl_pts: int, tp_pts: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pine").mkdir(exist_ok=True)

    # Save results JSON (used by report generator)
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"direction": direction, "sl_pts": sl_pts, "tp_pts": tp_pts,
                   "results": results}, f, ensure_ascii=False, indent=2, default=str)

    # Save per-strategy trade CSV
    for code, trades in trades_map.items():
        if not trades.empty:
            trades.to_csv(output_dir / f"{code}_trades.csv", index=False)

    print(f"[{direction.upper()}] Results saved → {output_dir}")
