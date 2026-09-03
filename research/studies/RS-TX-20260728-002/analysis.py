#!/usr/bin/env python3
"""Build RS-TX-20260728-002: annual Jan-open -> Jul-1 rally, Fibonacci pullback
re-entry, exit the following March 31. One-off TX study, not a spec-section contract
(no generic version of this exists elsewhere — see
that study's decision_log.md for the full method rationale).

Method (confirmed 2026-07-28):
  - For each candidate year Y: jan_open = Y's first trading day's open in January;
    jul1_price = the close of the trading day on or immediately before July 1 of Y.
  - Year qualifies only if jul1_price > jan_open (a real rally happened).
  - Fibonacci retracement anchors: jan_open = 0% (low anchor), jul1_price = 100% (high
    anchor). Standard retracement convention: level_price(ratio) = jul1_price -
    ratio * (jul1_price - jan_open) -- 0.382 is the shallowest level (closest to
    jul1_price, touched first as price declines), 0.618 is the deepest (closest to
    jan_open, touched last).
  - Each level is an independent potential trade: if the daily low touches
    level_price(ratio) at any point strictly after Jul 1 of year Y and on/before
    March 31 of year Y+1, that is the entry (filled at the level price, first touch
    only -- no re-entry on a later re-touch of the same level in the same window).
    Exit is always the close on/immediately before March 31 of year Y+1, regardless of
    what price does in between (no stop, no early exit).
  - win = exit_price > entry_price (long only). No `rank_score` -- this study's
    `policy_impacts` is empty (section 13.2 does not apply).

Usage:
    /opt/homebrew/bin/python3.12 scripts/research/build_tx_fib_pullback.py --study-id RS-TX-20260728-002
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import seasonality_toolkit as tk  # noqa: E402  (reuse chart PNG/JSON-safe helpers only)

DEFAULT_LEGACY = Path("../trading")
FIB_LEVELS = ["0.382", "0.5", "0.618"]
INSTRUMENT = "TX (MXF continuous front-month)"
CONTINUOUS_CONTRACT_CAVEAT = (
    f"{INSTRUMENT} is a rolling front-month continuous contract; roll adjustments "
    "landing near a Jan-1/Jul-1/Mar-31 anchor date can shift that anchor's price "
    "independent of real price movement. Known limitation, not corrected here."
)

STUDY_CONFIG = {
    "RS-TX-20260728-002": {
        "daily_csv": "tx/csv/TAIFEX_DLY_MXF1!, 1D.csv",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    frame["time"] = pd.to_datetime(frame["time"])
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("time").reset_index(drop=True)


def price_on_or_before(df: pd.DataFrame, target: date, column: str) -> tuple[pd.Timestamp, float] | None:
    """Latest trading day at or before `target`; None if none exists in range."""
    sub = df[df["time"] <= pd.Timestamp(target)]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return row["time"], float(row[column])


def first_trading_day_open(df: pd.DataFrame, year: int, month: int) -> tuple[pd.Timestamp, float] | None:
    sub = df[(df["time"].dt.year == year) & (df["time"].dt.month == month)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return row["time"], float(row["open"])


def wilson_interval(wins: int, count: int, z: float = 1.96) -> list[float] | None:
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
        "n": count,
        "wins": wins,
        "win_rate_pct": round(100 * wins / count, 2),
        "win_rate_ci95_pct": wilson_interval(wins, count),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_pts": round(sum(t["pnl_pts"] for t in trades), 2),
        "avg_pnl_pts": round(sum(t["pnl_pts"] for t in trades) / count, 2),
        "avg_return_pct": round(sum(t["return_pct"] for t in trades) / count, 4),
    }


def build(study_id: str, legacy_root: Path, output_dir: Path) -> dict:
    cfg = STUDY_CONFIG[study_id]
    legacy = legacy_root.resolve()
    daily_path = legacy / cfg["daily_csv"]
    df = load_daily(daily_path)

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
            "year": year,
            "jan_open_date": str(jan_date.date()),
            "jan_open": round(jan_open, 2),
            "jul1_date": str(jul1_date.date()),
            "jul1_price": round(jul1_price, 2),
            "pct_rise": pct_rise,
            "qualified": qualified,
            "level_prices": {},
            "entries": {},
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
                    "triggered": True,
                    "entry_date": str(touch_row["time"].date()),
                    "entry_price": level_price,
                    "exit_date": str(exit_date.date()),
                    "exit_price": round(exit_price, 2),
                    "pnl_pts": pnl_pts,
                    "win": pnl_pts > 0,
                }
                entry["entries"][level] = trade
                by_level_trades[level].append({
                    "pnl_pts": pnl_pts,
                    "return_pct": round(pnl_pts / level_price * 100, 4),
                })
        else:
            for level in FIB_LEVELS:
                entry["level_prices"][level] = None
                entry["entries"][level] = {"triggered": False}

        yearly_detail.append(entry)

    by_level = {level: level_stats(trades) for level, trades in by_level_trades.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig = chart_fib_level_winrate(by_level)
    png_bytes = tk.fig_to_png_bytes(fig)
    (charts_dir / "fib_level_winrate.png").write_bytes(png_bytes)
    b64 = base64.b64encode(png_bytes).decode()
    charts_manifest = [{
        "id": "fib_level_winrate", "file": "fib_level_winrate.png",
        "title": "Win Rate by Fibonacci Retracement Level", "section": "seasonality",
    }]

    results = {
        "schema_version": 1,
        "study_id": study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instrument": INSTRUMENT,
        "candidate_years": {"start": candidate_years[0], "end": candidate_years[-1]},
        "by_level": by_level,
        "yearly_detail": yearly_detail,
        "method": {
            "anchor_definition": "jan_open = year Y's first January trading day's open (0% anchor); "
                                  "jul1_price = close on/before July 1 of year Y (100% anchor)",
            "qualification": "year only qualifies if jul1_price > jan_open",
            "retracement_formula": "level_price(ratio) = jul1_price - ratio * (jul1_price - jan_open); "
                                    "0.382 is shallowest (closest to jul1_price), 0.618 is deepest (closest to jan_open)",
            "entry_rule": "first daily low <= level_price, strictly after Jul 1 through the following Mar 31; "
                          "filled at the level price; no re-entry on a later re-touch",
            "exit_rule": "close on/before March 31 of year Y+1, fixed regardless of price action in between",
            "continuous_contract_caveat": CONTINUOUS_CONTRACT_CAVEAT,
        },
        "charts": charts_manifest,
        "sources": [{"role": "daily", "path": str(daily_path), "sha256": sha256(daily_path)}],
    }
    results = tk.to_json_safe(results)

    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(results, b64), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(results), encoding="utf-8")

    return results


def chart_fib_level_winrate(by_level: dict):
    import matplotlib.pyplot as plt
    labels = FIB_LEVELS
    wr = [by_level[level]["win_rate_pct"] or 0 for level in labels]
    totals = [by_level[level]["n"] for level in labels]
    colors = ["#2ecc71" if w >= 55 else ("#e74c3c" if w <= 45 else "#95a5a6") for w in wr]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(labels)), wr, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win Rate %")
    ax.set_title(f"Win Rate by Fibonacci Retracement Level — {INSTRUMENT}")
    for i, (w, n) in enumerate(zip(wr, totals)):
        ax.text(i, w + 2, f"{w}%\nn={n}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9; color:#2c3e50; margin:0; padding:0; }
  .wrap { max-width: 1000px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 1.8em; margin-bottom: 4px; }
  h2 { font-size: 1.2em; border-bottom: 2px solid #3498db; padding-bottom: 4px; margin-top: 36px; color: #2980b9; }
  .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 24px; }
  .card { background: white; border-radius: 10px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-top: 20px; }
  .tbl { border-collapse: collapse; width: 100%; font-size: .85em; }
  .tbl th { background: #ecf0f1; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-top: 1px solid #ecf0f1; }
  .pos { color: #27ae60; } .neg { color: #e74c3c; }
  .note { background:#fef9e7; border-left:4px solid #f1c40f; padding:10px 16px; border-radius:4px; margin-top:12px; font-size:.88em; color:#7d6608; }
  footer { text-align:center; color:#bdc3c7; font-size:.8em; margin-top:40px; padding-top:16px; border-top:1px solid #ecf0f1; }
</style>
"""


def _level_table(by_level: dict) -> str:
    rows = ""
    for level in FIB_LEVELS:
        r = by_level[level]
        if r["n"] == 0:
            rows += f"<tr><td><b>{level}</b></td><td colspan='6'>no trades triggered</td></tr>"
            continue
        wr_class = "pos" if r["win_rate_pct"] >= 55 else ("neg" if r["win_rate_pct"] <= 45 else "")
        ci = r["win_rate_ci95_pct"]
        ci_text = f"{ci[0]}–{ci[1]}%" if ci else "—"
        rows += (
            f"<tr><td><b>{level}</b></td><td>{r['n']}</td>"
            f'<td class="{wr_class}">{r["win_rate_pct"]}%</td><td>{ci_text}</td>'
            f'<td>{r["profit_factor"] or "—"}</td><td>{r["net_pnl_pts"]:+.0f} pts</td>'
            f'<td>{r["avg_pnl_pts"]:+.0f} pts</td></tr>'
        )
    return (
        '<table class="tbl"><thead><tr><th>Fib Level</th><th>n</th><th>WR</th>'
        "<th>95% CI</th><th>PF</th><th>Net pts</th><th>Avg pts</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_html(results: dict, b64: str) -> str:
    generated = results["generated_at"][:19].replace("T", " ")
    years = results["candidate_years"]
    return f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<title>{results['instrument']} — Jan-Jul Rally + Fibonacci Pullback Re-entry</title>{CSS}</head>
<body><div class="wrap">
<h1>{results['instrument']} — 年度一月至七月漲幅 + 費波納契回撤進場</h1>
<p class="meta">日線資料 · 候選年份 {years['start']} ~ {years['end']}（共 {years['end']-years['start']+1} 年）
 · 產生時間：{generated}</p>

<div class="note">{results['method']['continuous_contract_caveat']}</div>

<h2>各費波納契水位勝率</h2>
<p class="meta">錨點：當年一月開盤 = 0%（低點），七月一號（或最近交易日）收盤 = 100%（高點）。
只有當年七月價格高於一月開盤才會產生水位；每個水位獨立進場、獨立在隔年三月底出場，分開統計。</p>
<div class="card">{_img_b64(b64)}{_level_table(results['by_level'])}</div>

<div class="note">完整逐年明細（每年是否符合資格、各水位是否觸及、進出場價與損益）請見 results.json 的 yearly_detail。</div>

<footer>{generated}</footer>
</div></body></html>"""


def _img_b64(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0;">'


def render_readme(results: dict) -> str:
    years = results["candidate_years"]
    lines = [
        f"# {results['instrument']} — Jan-Jul rally + Fibonacci pullback re-entry",
        "",
        f"- Candidate years: {years['start']}–{years['end']} "
        f"({years['end'] - years['start'] + 1} years, from daily OHLC).",
        "- Each Fibonacci level (0.382/0.5/0.618) is an independent potential trade, "
        "entered at first touch after July 1, exited the following March 31.",
    ]
    for level in FIB_LEVELS:
        r = results["by_level"][level]
        if r["n"] == 0:
            lines.append(f"- **{level}**: never triggered in this sample.")
        else:
            lines.append(
                f"- **{level}**: n={r['n']}, win rate {r['win_rate_pct']}%, "
                f"PF {r['profit_factor']}, net {r['net_pnl_pts']:+.0f} pts."
            )
    lines.append(f"- **Caveat**: {results['method']['continuous_contract_caveat']}")
    lines.append("- Full year-by-year detail is in `results.json`'s `yearly_detail`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("reproduced")
    build(args.study_id, args.legacy_root, output_dir)
    print(f"built {args.study_id} -> {output_dir}")


if __name__ == "__main__":
    main()
