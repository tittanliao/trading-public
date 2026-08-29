#!/usr/bin/env python3
"""Build a monthly seasonality report using the project's standard seasonality contract.

STUDY_CONFIG covers any instrument with a weekly OHLC CSV, not just TX — market-
agnostic by construction, matching section 14's own scope language.

Usage:
    /opt/homebrew/bin/python3.12 scripts/research/build_monthly_seasonality.py --study-id RS-TX-20260728-001

Writes results.json, report.html, README.md, and charts/*.png into the study's
research/studies/<id>/ directory. study.json/source_manifest.json/decision_log.md/
handoff.md are authored separately (they carry decision context this runner
does not have).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import seasonality_toolkit as tk  # noqa: E402

DEFAULT_LEGACY = Path("../trading")

STUDY_CONFIG = {
    "RS-TX-20260728-001": {
        "instrument": "TX (MXF continuous front-month)",
        "weekly_csv": "tx/csv/TAIFEX_DLY_MXF1!, 1W.csv",
        "continuous_contract": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9; color:#2c3e50; margin:0; padding:0; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 1.8em; margin-bottom: 4px; }
  h2 { font-size: 1.2em; border-bottom: 2px solid #3498db; padding-bottom: 4px; margin-top: 36px; color: #2980b9; }
  .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 24px; }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .kpi { background: white; border-radius: 10px; padding: 16px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.1); min-width: 140px; flex: 1; }
  .kpi-label { font-size: .8em; color: #7f8c8d; text-transform: uppercase; letter-spacing:.05em; }
  .kpi-value { font-size: 1.6em; font-weight: 700; margin-top: 2px; }
  .pos { color: #27ae60; } .neg { color: #e74c3c; }
  .card { background: white; border-radius: 10px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-top: 20px; }
  .tbl { border-collapse: collapse; width: 100%; font-size: .85em; }
  .tbl th { background: #ecf0f1; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-top: 1px solid #ecf0f1; }
  table.heatmap-table { border-collapse: collapse; font-size: 12px; }
  table.heatmap-table th { background: #ecf0f1; padding: 6px 10px; border: 1px solid #ddd; }
  table.heatmap-table td { padding: 5px 8px; border: 1px solid #eee; text-align: center; min-width: 44px; }
  .note { background:#fef9e7; border-left:4px solid #f1c40f; padding:10px 16px; border-radius:4px; margin-top:12px; font-size:.88em; color:#7d6608; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media(max-width:700px){ .chart-grid { grid-template-columns: 1fr; } }
  footer { text-align:center; color:#bdc3c7; font-size:.8em; margin-top:40px; padding-top:16px; border-top:1px solid #ecf0f1; }
</style>
"""


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0;">'


def build(study_id: str, legacy_root: Path, output_dir: Path) -> dict:
    cfg = STUDY_CONFIG[study_id]
    instrument = cfg["instrument"]
    legacy = legacy_root.resolve()
    weekly_path = legacy / cfg["weekly_csv"]

    df = tk.load_weekly(weekly_path)
    monthly = tk.build_monthly(df)
    by_month = tk.seasonality_by_month(monthly)
    week_in_month = tk.week_in_month_stats(df)
    heatmap = tk.year_month_heatmap(monthly)
    overall = tk.overall_stats(monthly)

    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts_manifest = []
    b64_by_id = {}

    def save(fig, chart_id, title, section):
        png_bytes = tk.fig_to_png_bytes(fig)
        (charts_dir / f"{chart_id}.png").write_bytes(png_bytes)
        b64_by_id[chart_id] = base64.b64encode(png_bytes).decode()
        charts_manifest.append({"id": chart_id, "file": f"{chart_id}.png", "title": title, "section": section})

    save(tk.chart_monthly_winrate(by_month, instrument), "monthly_winrate",
         "Monthly Seasonality Win Rate", "seasonality")
    save(tk.chart_week_in_month_winrate(week_in_month, instrument), "week_in_month_winrate",
         "Week-in-Month Win Rate", "seasonality")

    method = {
        "aggregation": "month_open = first weekly bar's open in that month; "
                        "month_close = last weekly bar's close in that month",
        "bias_thresholds": "win_rate_pct >= 55 -> LONG; <= 45 -> SHORT; else NEUTRAL",
    }
    if cfg.get("continuous_contract"):
        method["continuous_contract_caveat"] = (
            f"{instrument} is a rolling front-month continuous contract; roll "
            "adjustments near a month boundary can shift month_open/month_close "
            "independent of real price movement. Known limitation, not corrected here."
        )

    results = {
        "schema_version": 1,
        "study_id": study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instrument": instrument,
        "data_period": {"start": str(df["time"].min().date()), "end": str(df["time"].max().date())},
        "by_month": by_month,
        "week_in_month": week_in_month,
        "year_month_heatmap": heatmap,
        "overall": overall,
        "method": method,
        "charts": charts_manifest,
        "sources": [{"role": "weekly", "path": str(weekly_path), "sha256": sha256(weekly_path)}],
    }
    results = tk.to_json_safe(results)

    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(results, heatmap, b64_by_id), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(results), encoding="utf-8")

    return results


def _month_table(by_month: dict) -> str:
    rows = ""
    for m in sorted(by_month.keys(), key=int):
        r = by_month[m]
        wr_class = "pos" if r["win_rate_pct"] >= 55 else ("neg" if r["win_rate_pct"] <= 45 else "")
        rows += (
            f"<tr><td><b>{r['month_name']}</b></td><td>{r['n']}</td>"
            f'<td class="{wr_class}">{r["win_rate_pct"]}%</td>'
            f'<td>{r["avg_chg_pts"]:+.0f} pts</td><td>{r["avg_chg_pct"]:+.2f}%</td>'
            f'<td>{r["median_chg_pts"]:+.0f}</td><td class="pos">{r["best_chg_pts"]:+.0f}</td>'
            f'<td class="neg">{r["worst_chg_pts"]:+.0f}</td><td>{r["bias"]}</td></tr>'
        )
    return (
        '<table class="tbl"><thead><tr><th>月份</th><th>樣本數</th><th>月勝率</th>'
        "<th>平均漲跌（點）</th><th>平均漲跌%</th><th>中位數</th><th>最佳</th><th>最差</th>"
        f"<th>偏向</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _week_table(week_in_month: dict) -> str:
    rows = ""
    for w in sorted(week_in_month.keys(), key=int):
        r = week_in_month[w]
        wr_class = "pos" if r["win_rate_pct"] >= 55 else ("neg" if r["win_rate_pct"] <= 45 else "")
        rows += (
            f"<tr><td><b>{r['week_label']}</b></td><td>{r['n']}</td>"
            f'<td class="{wr_class}">{r["win_rate_pct"]}%</td>'
            f'<td>{r["avg_chg_pts"]:+.0f} pts</td><td>{r["median_chg_pts"]:+.0f}</td></tr>'
        )
    return (
        '<table class="tbl"><thead><tr><th>週次</th><th>樣本數</th><th>週勝率</th>'
        f"<th>平均漲跌（點）</th><th>中位數</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def render_html(results: dict, heatmap: dict, b64_by_id: dict) -> str:
    overall = results["overall"]
    generated = results["generated_at"][:19].replace("T", " ")
    caveat = results["method"].get("continuous_contract_caveat", "")
    return f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<title>{results['instrument']} — Monthly Seasonality Report</title>{CSS}</head>
<body><div class="wrap">
<h1>{results['instrument']} — 月度季節性報告</h1>
<p class="meta">週線資料 · {results['data_period']['start']} ~ {results['data_period']['end']}
 · 共 {overall['total_months']} 個月 · 產生時間：{generated}</p>

<div class="kpi-row">
  <div class="kpi"><div class="kpi-value">{overall['overall_win_rate_pct']}%</div>
    <div class="kpi-label">整體月勝率（月初買月底賣）</div></div>
  <div class="kpi"><div class="kpi-value {'pos' if overall['avg_chg_pts'] > 0 else 'neg'}">{overall['avg_chg_pts']:+.0f}</div>
    <div class="kpi-label">平均月漲跌（點數）</div></div>
  <div class="kpi"><div class="kpi-value">{overall['total_months']}</div>
    <div class="kpi-label">總樣本月數</div></div>
</div>

{f'<div class="note">{caveat}</div>' if caveat else ''}

<h2>季節性分析 — 每個月份的歷史偏向</h2>
<p class="meta">月勝率 ≥55% → 偏多（LONG），≤45% → 偏空（SHORT），中間 → 中性（NEUTRAL）。樣本數 = 歷史年份數。</p>
<div class="card">{_img(b64_by_id['monthly_winrate'])}{_month_table(results['by_month'])}</div>

<h2>月度熱力圖 — 歷年每月漲跌點數</h2>
<div class="card">{tk.render_heatmap_html(heatmap)}</div>

<h2>週內結構 — 每月第幾週最強 / 最弱</h2>
<p class="meta">所有月份合併，看月初、月中、月底哪一週的多頭勝率最高。</p>
<div class="card">{_img(b64_by_id['week_in_month_winrate'])}{_week_table(results['week_in_month'])}</div>

<footer>Monthly seasonality contract &nbsp;·&nbsp; {generated}</footer>
</div></body></html>"""


def render_readme(results: dict) -> str:
    overall = results["overall"]
    lines = [
        f"# {results['instrument']} — Monthly Seasonality Report",
        "",
        f"- Data: weekly OHLC, {results['data_period']['start']} to {results['data_period']['end']} "
        f"({overall['total_months']} months).",
        f"- Overall win rate (buy month-open, sell month-close): **{overall['overall_win_rate_pct']}%**, "
        f"avg {overall['avg_chg_pts']:+.0f} pts/month.",
        "- Full month-by-month seasonality, week-in-month structure, and the year x month "
        "heatmap are in `results.json` / `report.html`.",
    ]
    if "continuous_contract_caveat" in results["method"]:
        lines.append(f"- **Caveat**: {results['method']['continuous_contract_caveat']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("research/studies") / args.study_id
    build(args.study_id, args.legacy_root, output_dir)
    print(f"built {args.study_id} -> {output_dir}")


if __name__ == "__main__":
    main()
