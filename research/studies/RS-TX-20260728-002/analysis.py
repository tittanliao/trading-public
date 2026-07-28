#!/usr/bin/env python3
"""Public reproduction method for RS-TX-20260728-002 (annual Jan-open -> Jul-1 rally,
Fibonacci pullback re-entry, exit the following March 31). One-off TX study, not a
docs/RESEARCH_DEVELOPMENT_SPEC.md spec-section contract — see the Private
decision_log.md for the full method rationale.

Raw daily OHLC CSV is intentionally not included. Supply a locally authorized daily
export (time,open,high,low,close,...). Reproduces every published field: by_level and
yearly_detail.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

FIB_LEVELS = ["0.382", "0.5", "0.618"]


def load_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    frame["time"] = pd.to_datetime(frame["time"])
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("time").reset_index(drop=True)


def price_on_or_before(df: pd.DataFrame, target: date, column: str):
    sub = df[df["time"] <= pd.Timestamp(target)]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return row["time"], float(row[column])


def first_trading_day_open(df: pd.DataFrame, year: int, month: int):
    sub = df[(df["time"].dt.year == year) & (df["time"].dt.month == month)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return row["time"], float(row["open"])


def wilson_interval(wins: int, count: int, z: float = 1.96):
    if count == 0:
        return None
    p = wins / count
    denom = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denom
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return [round(100 * (centre - margin / denom), 2), round(100 * (centre + margin / denom), 2)]


def level_stats(trades: list[dict]) -> dict:
    count = len(trades)
    if count == 0:
        return {"n": 0, "wins": 0, "win_rate_pct": None, "win_rate_ci95_pct": None,
                "profit_factor": None, "net_pnl_pts": 0.0, "avg_pnl_pts": None, "avg_return_pct": None}
    wins = sum(1 for t in trades if t["pnl_pts"] > 0)
    gross_profit = sum(t["pnl_pts"] for t in trades if t["pnl_pts"] > 0)
    gross_loss = abs(sum(t["pnl_pts"] for t in trades if t["pnl_pts"] < 0))
    return {
        "n": count, "wins": wins, "win_rate_pct": round(100 * wins / count, 2),
        "win_rate_ci95_pct": wilson_interval(wins, count),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_pts": round(sum(t["pnl_pts"] for t in trades), 2),
        "avg_pnl_pts": round(sum(t["pnl_pts"] for t in trades) / count, 2),
        "avg_return_pct": round(sum(t["return_pct"] for t in trades) / count, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = load_daily(args.daily)
    first_year = int(df["time"].dt.year.min())
    last_year = int(df["time"].dt.year.max())
    candidate_years = [y for y in range(first_year + 1, last_year) if y + 1 <= last_year]

    yearly_detail = []
    by_level_trades: dict[str, list[dict]] = {level: [] for level in FIB_LEVELS}

    for year in candidate_years:
        jan = first_trading_day_open(df, year, 1)
        jul1 = price_on_or_before(df, date(year, 7, 1), "close")
        if jan is None or jul1 is None:
            continue
        jan_date, jan_open = jan
        jul1_date, jul1_price = jul1
        pct_rise = round((jul1_price - jan_open) / jan_open * 100, 2)
        qualified = jul1_price > jan_open

        entry = {
            "year": year, "jan_open_date": str(jan_date.date()), "jan_open": round(jan_open, 2),
            "jul1_date": str(jul1_date.date()), "jul1_price": round(jul1_price, 2),
            "pct_rise": pct_rise, "qualified": qualified, "level_prices": {}, "entries": {},
        }

        exit_point = price_on_or_before(df, date(year + 1, 3, 31), "close")
        window = df[(df["time"] > jul1_date) & (df["time"] <= pd.Timestamp(date(year + 1, 3, 31)))]

        if qualified and exit_point is not None:
            exit_date, exit_price = exit_point
            rise = jul1_price - jan_open
            for level in FIB_LEVELS:
                ratio = float(level)
                level_price = round(jul1_price - ratio * rise, 2)
                entry["level_prices"][level] = level_price
                touch = window[window["low"] <= level_price]
                if touch.empty:
                    entry["entries"][level] = {"triggered": False}
                    continue
                touch_row = touch.iloc[0]
                pnl_pts = round(exit_price - level_price, 2)
                trade = {
                    "triggered": True, "entry_date": str(touch_row["time"].date()), "entry_price": level_price,
                    "exit_date": str(exit_date.date()), "exit_price": round(exit_price, 2),
                    "pnl_pts": pnl_pts, "win": pnl_pts > 0,
                }
                entry["entries"][level] = trade
                by_level_trades[level].append({"pnl_pts": pnl_pts, "return_pct": round(pnl_pts / level_price * 100, 4)})
        else:
            for level in FIB_LEVELS:
                entry["level_prices"][level] = None
                entry["entries"][level] = {"triggered": False}

        yearly_detail.append(entry)

    by_level = {level: level_stats(trades) for level, trades in by_level_trades.items()}
    output = {
        "candidate_years": {"start": candidate_years[0], "end": candidate_years[-1]},
        "by_level": by_level,
        "yearly_detail": yearly_detail,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
