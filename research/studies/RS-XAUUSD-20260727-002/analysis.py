#!/usr/bin/env python3
"""Public reproduction method for RS-XAUUSD-20260727-002.

Raw TradingView CSVs are intentionally not included. Supply locally authorized CSV
paths for both the V3.4 and V3.9 "List of Trades" exports plus the five daily Macro
series. This reproduces every metric at 30-minute entry-slot granularity for both
strategy versions; it does not compute a fail-pattern (fail_type/DXY/MTF/K-bar)
breakdown.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MAX_AGE = pd.Timedelta(days=4)
ENTRY_SLOTS_30M = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in (0, 30)]

# TradingView export column names changed between the V3.4 and V3.9 strategy versions.
COLUMN_ALIASES = {
    "trade_id": ["Trade number", "Trade #"],
    "type": ["Type"],
    "datetime": ["Date and time"],
    "net_pnl_usd": ["Net PnL USD", "Net P&L USD"],
    "return_pct": ["Return %", "Net P&L %"],
    "signal": ["Signal"],
}


def pick_column(columns: list[str], canonical: str) -> str:
    for alias in COLUMN_ALIASES[canonical]:
        if alias in columns:
            return alias
    raise KeyError(f"no column found for {canonical!r} among {columns}")


def daily(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("﻿").strip() for column in frame.columns]
    time_column = next(column for column in frame.columns if "time" in column.lower())
    frame[time_column] = pd.to_datetime(frame[time_column], utc=False)
    if frame[time_column].dt.tz is not None:
        frame[time_column] = frame[time_column].dt.tz_localize(None)
    return frame[[time_column, "close"]].rename(
        columns={time_column: "time", "close": name}
    ).sort_values("time")


def trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("﻿").strip() for column in frame.columns]
    columns = list(frame.columns)
    rename = {pick_column(columns, canonical): canonical for canonical in COLUMN_ALIASES}
    frame = frame.rename(columns=rename)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    entries = frame[frame["type"] == "Entry long"][["trade_id", "datetime"]].rename(
        columns={"datetime": "entry_time"}
    )
    exits = frame[frame["type"] == "Exit long"][
        ["trade_id", "datetime", "signal", "net_pnl_usd", "return_pct"]
    ].rename(columns={"datetime": "exit_time", "signal": "exit_signal"})
    result = entries.merge(exits, on="trade_id", how="inner")
    result = result[result["exit_signal"].astype(str).str.upper() != "OPEN"].copy()
    result["win"] = result["net_pnl_usd"] > 0
    result["entry_slot_30m"] = result["entry_time"].dt.strftime("%H:%M")
    minute = result["entry_time"].dt.hour * 60 + result["entry_time"].dt.minute
    result["session"] = np.select(
        [minute.between(420, 899), minute.between(900, 1229),
         (minute >= 1230) | (minute < 60)],
        ["asia", "europe", "us"],
        default="overnight",
    )
    return result.sort_values("entry_time")


def macro(paths: dict[str, Path]) -> pd.DataFrame:
    frames = {name: daily(path, name) for name, path in paths.items()}
    base = frames.pop("us10y")
    for frame in frames.values():
        base = pd.merge_asof(base, frame, on="time", direction="backward", tolerance=MAX_AGE)
    base["real_rate"] = base["us10y"] - base["t10yie"]
    for name in ["real_rate", "us10y", "dxy", "vix", "gold"]:
        base[f"ma50_{name}"] = base[name].rolling(50, min_periods=50).mean()
    valid = base[[f"ma50_{name}" for name in ["real_rate", "us10y", "dxy", "vix", "gold"]]].notna().all(axis=1)
    base = base[valid].copy()
    base["macro_score"] = (
        2 * (base["real_rate"] < base["ma50_real_rate"])
        + (base["us10y"] < base["ma50_us10y"])
        + (base["dxy"] < base["ma50_dxy"])
        + (base["vix"] > base["ma50_vix"])
        + (base["gold"] > base["ma50_gold"])
    )
    base["macro_verdict"] = np.select(
        [base["macro_score"] <= 2, base["macro_score"] <= 4],
        ["WAIT", "NEUTRAL"], default="STRONG BUY",
    )
    return base


def stats(frame: pd.DataFrame) -> dict[str, object]:
    count = len(frame)
    if count == 0:
        return {"n": 0, "wins": 0, "win_rate_pct": None, "win_rate_ci95_pct": None,
                "profit_factor": None, "net_pnl_usd": 0.0, "avg_return_pct": None}
    wins = int(frame["win"].sum())
    p, z = wins / count, 1.96
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    gross_profit = frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
    gross_loss = abs(frame.loc[frame["net_pnl_usd"] < 0, "net_pnl_usd"].sum())
    return {
        "n": count, "wins": wins, "win_rate_pct": round(100 * p, 2),
        "win_rate_ci95_pct": [
            round(100 * (centre - margin / denominator), 2),
            round(100 * (centre + margin / denominator), 2),
        ],
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_usd": round(float(frame["net_pnl_usd"].sum()), 2),
        "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
    }


def advisory_delta(group: dict[str, object], baseline: dict[str, object], max_points: float = 10.0) -> float | None:
    if not group["n"] or not baseline["avg_return_pct"]:
        return None
    weight = group["n"] / (group["n"] + 30)
    shrunk = weight * group["avg_return_pct"] + (1 - weight) * baseline["avg_return_pct"]
    relative = (shrunk - baseline["avg_return_pct"]) / abs(baseline["avg_return_pct"])
    return round(max(-max_points, min(max_points, relative * 10)), 1)


def analyze_version(trade_path: Path, macro_frame: pd.DataFrame) -> dict:
    trade_frame = trades(trade_path)
    joined = pd.merge_asof(
        trade_frame, macro_frame[["time", "macro_score", "macro_verdict"]],
        left_on="entry_time", right_on="time", direction="backward", tolerance=MAX_AGE,
    )
    matched = joined.dropna(subset=["macro_score"])
    baseline = stats(trade_frame)
    by_session = {name: stats(group) for name, group in trade_frame.groupby("session")}
    by_entry_30m = {
        slot: stats(trade_frame[trade_frame["entry_slot_30m"] == slot])
        for slot in ENTRY_SLOTS_30M
    }
    by_macro = {name: stats(group) for name, group in matched.groupby("macro_verdict")}
    for item in by_entry_30m.values():
        item["advisory_delta"] = advisory_delta(item, baseline, max_points=4.0)
    for item in by_macro.values():
        item["advisory_delta"] = advisory_delta(item, baseline)
    return {
        "baseline": baseline,
        "macro_coverage": {"matched": len(matched), "unmatched": len(trade_frame) - len(matched)},
        "by_session": by_session,
        "by_entry_30m": by_entry_30m,
        "by_macro_verdict": by_macro,
    }


def compare_entry_30m(v34: dict, v39: dict) -> dict:
    comparison = {}
    for slot in ENTRY_SLOTS_30M:
        a, b = v34["by_entry_30m"][slot], v39["by_entry_30m"][slot]
        wr_diff = (
            round(b["win_rate_pct"] - a["win_rate_pct"], 2)
            if a["win_rate_pct"] is not None and b["win_rate_pct"] is not None
            else None
        )
        comparison[slot] = {
            "v34_n": a["n"], "v34_win_rate_pct": a["win_rate_pct"], "v34_profit_factor": a["profit_factor"],
            "v39_n": b["n"], "v39_win_rate_pct": b["win_rate_pct"], "v39_profit_factor": b["profit_factor"],
            "win_rate_pct_diff_v39_minus_v34": wr_diff,
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-v34", type=Path, required=True)
    parser.add_argument("--trades-v39", type=Path, required=True)
    for name in ["us10y", "t10yie", "dxy", "vix", "gold"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    macro_frame = macro({name: getattr(args, name) for name in ["us10y", "t10yie", "dxy", "vix", "gold"]})
    versions = {
        "V3.4": analyze_version(args.trades_v34, macro_frame),
        "V3.9": analyze_version(args.trades_v39, macro_frame),
    }
    output = {
        "versions": versions,
        "comparison": {"by_entry_30m": compare_entry_30m(versions["V3.4"], versions["V3.9"])},
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
