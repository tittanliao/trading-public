#!/usr/bin/env python3
"""Public reproduction method for the section-14 monthly seasonality report contract
(docs/RESEARCH_DEVELOPMENT_SPEC.md, private repo).

Raw weekly OHLC CSV is intentionally not included. Supply a locally authorized weekly
export (time,open,high,low,close,...). Reproduces every published field: by_month
(calendar-month seasonality), week_in_month (week-of-month structure),
year_month_heatmap, and overall. Chart PNGs are published as pre-verified static files
and are not regenerated here.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

MONTH_NAMES = ["一月", "二月", "三月", "四月", "五月", "六月",
               "七月", "八月", "九月", "十月", "十一月", "十二月"]
WEEK_LABELS = {1: "第1週（月初）", 2: "第2週", 3: "第3週", 4: "第4週", 5: "第5週（月底）"}


def load_weekly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]

    def parse_time(value):
        text = re.sub(r"[+-]\d{2}:\d{2}$", "", str(value).strip())
        return pd.to_datetime(text)

    frame["time"] = frame["time"].apply(parse_time)
    frame = frame.sort_values("time").reset_index(drop=True)
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["year"] = frame["time"].dt.year
    frame["month"] = frame["time"].dt.month
    frame["week_of_month"] = frame.groupby(["year", "month"]).cumcount() + 1
    return frame


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["year", "month"])
    monthly = grouped.agg(
        month_open=("open", "first"), month_close=("close", "last"),
        month_high=("high", "max"), month_low=("low", "min"), n_weeks=("close", "count"),
    ).reset_index()
    monthly["chg_pts"] = monthly["month_close"] - monthly["month_open"]
    monthly["chg_pct"] = monthly["chg_pts"] / monthly["month_open"] * 100
    monthly["bullish"] = monthly["chg_pts"] > 0
    return monthly


def seasonality_by_month(monthly: pd.DataFrame) -> dict:
    result = {}
    for month in range(1, 13):
        sub = monthly[monthly["month"] == month]
        if len(sub) == 0:
            continue
        win_rate = round(float(sub["bullish"].mean() * 100), 2)
        bias = "LONG" if win_rate >= 55 else ("SHORT" if win_rate <= 45 else "NEUTRAL")
        result[str(month)] = {
            "month_name": MONTH_NAMES[month - 1], "n": int(len(sub)), "win_rate_pct": win_rate,
            "avg_chg_pts": round(float(sub["chg_pts"].mean()), 2),
            "avg_chg_pct": round(float(sub["chg_pct"].mean()), 2),
            "median_chg_pts": round(float(sub["chg_pts"].median()), 2),
            "best_chg_pts": round(float(sub["chg_pts"].max()), 2),
            "worst_chg_pts": round(float(sub["chg_pts"].min()), 2),
            "bias": bias,
        }
    return result


def week_in_month_stats(df: pd.DataFrame) -> dict:
    frame = df.copy()
    frame["week_chg"] = frame["close"] - frame["open"]
    frame["week_bull"] = frame["week_chg"] > 0
    result = {}
    for week in range(1, 6):
        sub = frame[frame["week_of_month"] == week]
        if len(sub) == 0:
            continue
        result[str(week)] = {
            "week_label": WEEK_LABELS[week], "n": int(len(sub)),
            "win_rate_pct": round(float(sub["week_bull"].mean() * 100), 2),
            "avg_chg_pts": round(float(sub["week_chg"].mean()), 2),
            "median_chg_pts": round(float(sub["week_chg"].median()), 2),
        }
    return result


def year_month_heatmap(monthly: pd.DataFrame) -> dict:
    result: dict[str, dict[str, float]] = {}
    for _, row in monthly.iterrows():
        year, month = str(int(row["year"])), str(int(row["month"]))
        result.setdefault(year, {})[month] = round(float(row["chg_pts"]), 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = load_weekly(args.weekly)
    monthly = build_monthly(df)
    by_month = seasonality_by_month(monthly)
    week_in_month = week_in_month_stats(df)
    heatmap = year_month_heatmap(monthly)
    overall = {
        "total_months": int(len(monthly)),
        "overall_win_rate_pct": round(float(monthly["bullish"].mean() * 100), 2),
        "avg_chg_pts": round(float(monthly["chg_pts"].mean()), 2),
    }
    output = {
        "instrument": "TX (MXF continuous front-month)",
        "data_period": {"start": str(df["time"].min().date()), "end": str(df["time"].max().date())},
        "by_month": by_month,
        "week_in_month": week_in_month,
        "year_month_heatmap": heatmap,
        "overall": overall,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
