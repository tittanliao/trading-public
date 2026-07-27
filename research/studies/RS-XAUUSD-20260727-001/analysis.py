#!/usr/bin/env python3
"""Public reproduction method for RS-XAUUSD-20260727-001.

Raw TradingView CSV is intentionally not included. Supply locally authorized CSV paths.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MAX_AGE = pd.Timedelta(days=4)


def daily(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("\ufeff").strip() for column in frame.columns]
    time_column = next(column for column in frame.columns if "time" in column.lower())
    frame[time_column] = pd.to_datetime(frame[time_column], utc=False)
    if frame[time_column].dt.tz is not None:
        frame[time_column] = frame[time_column].dt.tz_localize(None)
    return frame[[time_column, "close"]].rename(
        columns={time_column: "time", "close": name}
    ).sort_values("time")


def trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("\ufeff").strip() for column in frame.columns]
    frame["datetime"] = pd.to_datetime(frame["Date and time"])
    entries = frame[frame["Type"] == "Entry long"][
        ["Trade number", "datetime"]
    ].rename(columns={"datetime": "entry_time"})
    exits = frame[frame["Type"] == "Exit long"][
        ["Trade number", "datetime", "Signal", "Net PnL USD", "Return %"]
    ].rename(columns={
        "datetime": "exit_time", "Signal": "exit_signal",
        "Net PnL USD": "net_pnl_usd", "Return %": "return_pct",
    })
    result = entries.merge(exits, on="Trade number", how="inner")
    result = result[result["exit_signal"].astype(str).str.upper() != "OPEN"].copy()
    result["win"] = result["net_pnl_usd"] > 0
    minute = result["entry_time"].dt.hour * 60 + result["entry_time"].dt.minute
    result["session"] = np.select(
        [minute.between(420, 899), minute.between(900, 1229),
         (minute >= 1230) | (minute < 60)],
        ["Asia 07:00–14:59", "Europe 15:00–20:29", "U.S. 20:30–00:59"],
        default="Overnight 01:00–06:59",
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
    count, wins = len(frame), int(frame["win"].sum())
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
        "profit_factor": round(gross_profit / gross_loss, 3),
        "net_pnl_usd": round(float(frame["net_pnl_usd"].sum()), 2),
        "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
    }


def advisory_delta(group: dict[str, object], baseline: dict[str, object]) -> float:
    weight = group["n"] / (group["n"] + 30)
    shrunk = (
        weight * group["avg_return_pct"]
        + (1 - weight) * baseline["avg_return_pct"]
    )
    relative = (
        shrunk - baseline["avg_return_pct"]
    ) / abs(baseline["avg_return_pct"])
    return round(max(-10.0, min(10.0, relative * 10)), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=Path, required=True)
    for name in ["us10y", "t10yie", "dxy", "vix", "gold"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trade_frame = trades(args.trades)
    macro_frame = macro({name: getattr(args, name) for name in ["us10y", "t10yie", "dxy", "vix", "gold"]})
    joined = pd.merge_asof(
        trade_frame, macro_frame[["time", "macro_score", "macro_verdict"]],
        left_on="entry_time", right_on="time", direction="backward", tolerance=MAX_AGE,
    )
    matched = joined.dropna(subset=["macro_score"])
    baseline = stats(trade_frame)
    by_session = {name: stats(group) for name, group in trade_frame.groupby("session")}
    by_macro = {name: stats(group) for name, group in matched.groupby("macro_verdict")}
    for collection in [by_session, by_macro]:
        for item in collection.values():
            item["advisory_delta"] = advisory_delta(item, baseline)
    output = {
        "baseline": baseline,
        "macro_coverage": {"matched": len(matched), "unmatched": len(trade_frame) - len(matched)},
        "by_session": by_session,
        "by_macro_verdict": by_macro,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
