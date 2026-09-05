"""Shared seasonality toolkit — reference implementation of
docs/RESEARCH_DEVELOPMENT_SPEC.md section 14 (monthly seasonality report contract).

Self-contained: no dependency on fail_pattern_toolkit.py (no shared logic between
calendar-aggregate seasonality and trade-level fail-pattern analysis — keeping them in
one file would mix unrelated concerns). Ported from the read-only legacy
trading/tx/macro_analysis.py, generalized to any instrument with a weekly OHLC CSV
(time,open,high,low,close,...).
"""
from __future__ import annotations

import base64
import io
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Chart labels are Chinese month/week names (一月, 第1週...); the default DejaVu Sans
# font has no CJK glyphs and silently renders them as boxes. Prefer a CJK-capable
# system font when available; fall back to DejaVu Sans (Latin/numeric labels still
# render fine) rather than erroring if none is installed.
plt.rcParams["font.sans-serif"] = [
    "PingFang HK", "PingFang SC", "PingFang TC", "Heiti TC", "STHeiti",
    "Microsoft JhengHei", "Noto Sans CJK TC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

MONTH_NAMES = ["一月", "二月", "三月", "四月", "五月", "六月",
               "七月", "八月", "九月", "十月", "十一月", "十二月"]
WEEK_LABELS = {1: "第1週（月初）", 2: "第2週", 3: "第3週", 4: "第4週", 5: "第5週（月底）"}

_WIN, _LOSS, _NEUTRAL = "#2ecc71", "#e74c3c", "#95a5a6"


# ---------------------------------------------------------------------------
# 1. Load and aggregate (spec section 14.1 item 1)
# ---------------------------------------------------------------------------

def load_weekly(path: Path) -> pd.DataFrame:
    """Weekly OHLC loader. Strips a trailing UTC-offset suffix on the time column
    (some weekly exports carry one), matching the legacy loader byte-for-byte."""
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
        month_open=("open", "first"),
        month_close=("close", "last"),
        month_high=("high", "max"),
        month_low=("low", "min"),
        n_weeks=("close", "count"),
    ).reset_index()
    monthly["chg_pts"] = monthly["month_close"] - monthly["month_open"]
    monthly["chg_pct"] = monthly["chg_pts"] / monthly["month_open"] * 100
    monthly["bullish"] = monthly["chg_pts"] > 0
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    return monthly


# ---------------------------------------------------------------------------
# 2. Seasonality by calendar month (spec section 14.1 item 2)
# ---------------------------------------------------------------------------

def seasonality_by_month(monthly: pd.DataFrame) -> dict:
    result = {}
    for month in range(1, 13):
        sub = monthly[monthly["month"] == month]
        if len(sub) == 0:
            continue
        win_rate = round(float(sub["bullish"].mean() * 100), 2)
        bias = "LONG" if win_rate >= 55 else ("SHORT" if win_rate <= 45 else "NEUTRAL")
        result[str(month)] = {
            "month_name": MONTH_NAMES[month - 1],
            "n": int(len(sub)),
            "win_rate_pct": win_rate,
            "avg_chg_pts": round(float(sub["chg_pts"].mean()), 2),
            "avg_chg_pct": round(float(sub["chg_pct"].mean()), 2),
            "median_chg_pts": round(float(sub["chg_pts"].median()), 2),
            "best_chg_pts": round(float(sub["chg_pts"].max()), 2),
            "worst_chg_pts": round(float(sub["chg_pts"].min()), 2),
            "bias": bias,
        }
    return result


# ---------------------------------------------------------------------------
# 3. Week-in-month structure (spec section 14.1 item 3)
# ---------------------------------------------------------------------------

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
            "week_label": WEEK_LABELS[week],
            "n": int(len(sub)),
            "win_rate_pct": round(float(sub["week_bull"].mean() * 100), 2),
            "avg_chg_pts": round(float(sub["week_chg"].mean()), 2),
            "median_chg_pts": round(float(sub["week_chg"].median()), 2),
        }
    return result


# ---------------------------------------------------------------------------
# 4. Year x month heatmap (spec section 14.1 item 4)
# ---------------------------------------------------------------------------

def year_month_heatmap(monthly: pd.DataFrame) -> dict:
    result: dict[str, dict[str, float]] = {}
    for _, row in monthly.iterrows():
        year, month = str(int(row["year"])), str(int(row["month"]))
        result.setdefault(year, {})[month] = round(float(row["chg_pts"]), 2)
    return result


def render_heatmap_html(heatmap: dict) -> str:
    """Year x month HTML table, color-coded by chg_pts. A heatmap is a dense grid
    better read as text-in-cells than a PNG image (spec section 14.1 item 4)."""

    def chg_to_color(value):
        if value is None:
            return "#2a2a2a"
        if value > 500:
            return "#0a5c2e"
        if value > 200:
            return "#1a7a4a"
        if value > 0:
            return "#27ae60"
        if value > -200:
            return "#e74c3c"
        if value > -500:
            return "#c0392b"
        return "#7b241c"

    years = sorted(heatmap.keys(), key=int)
    month_cols = [str(m) for m in range(1, 13)]
    header = "<tr><th>年份</th>" + "".join(f"<th>{m}</th>" for m in month_cols) + "</tr>"
    body = ""
    for year in years:
        row = f"<tr><td><b>{year}</b></td>"
        for month in month_cols:
            value = heatmap[year].get(month)
            if value is None:
                row += '<td style="background:#1e1e1e">—</td>'
            else:
                bg = chg_to_color(value)
                row += f'<td style="background:{bg};color:#fff;font-size:11px">{value:+.0f}</td>'
        row += "</tr>"
        body += row
    return f'<table class="heatmap-table"><thead>{header}</thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------

def overall_stats(monthly: pd.DataFrame) -> dict:
    return {
        "total_months": int(len(monthly)),
        "overall_win_rate_pct": round(float(monthly["bullish"].mean() * 100), 2),
        "avg_chg_pts": round(float(monthly["chg_pts"].mean()), 2),
    }


# ---------------------------------------------------------------------------
# Charts — section "seasonality" (spec section 14.1 items 2-3)
# ---------------------------------------------------------------------------

def _winrate_color(win_rate: float) -> str:
    return _WIN if win_rate >= 55 else (_LOSS if win_rate <= 45 else _NEUTRAL)


def chart_monthly_winrate(by_month: dict, instrument: str) -> plt.Figure:
    months = sorted(by_month.keys(), key=int)
    labels = [by_month[m]["month_name"] for m in months]
    win_rates = [by_month[m]["win_rate_pct"] for m in months]
    totals = [by_month[m]["n"] for m in months]
    colors = [_winrate_color(w) for w in win_rates]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(labels)), win_rates, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win Rate %")
    ax.set_title(f"Monthly Seasonality Win Rate — {instrument}")
    for i, (win_rate, n) in enumerate(zip(win_rates, totals)):
        ax.text(i, win_rate + 2, f"{win_rate}%\nn={n}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def chart_week_in_month_winrate(week_in_month: dict, instrument: str) -> plt.Figure:
    weeks = sorted(week_in_month.keys(), key=int)
    labels = [week_in_month[w]["week_label"] for w in weeks]
    win_rates = [week_in_month[w]["win_rate_pct"] for w in weeks]
    totals = [week_in_month[w]["n"] for w in weeks]
    colors = [_winrate_color(w) for w in win_rates]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), win_rates, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win Rate %")
    ax.set_title(f"Week-in-Month Win Rate — {instrument}")
    for i, (win_rate, n) in enumerate(zip(win_rates, totals)):
        ax.text(i, win_rate + 2, f"{win_rate}%\nn={n}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Serialization helpers (mirrors fail_pattern_toolkit.py's, kept separate per
# this module's own no-shared-logic design note)
# ---------------------------------------------------------------------------

def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fig_to_b64(fig: plt.Figure) -> str:
    return base64.b64encode(fig_to_png_bytes(fig)).decode()


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj
