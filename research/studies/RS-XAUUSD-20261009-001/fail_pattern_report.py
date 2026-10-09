#!/usr/bin/env python3
"""HTML rendering for the section 5 fail-pattern report, shared by new runners.

Copied verbatim from `build_s1_fail_pattern_solo.py` on 2026-10-09 rather than imported from
it: a runner is published as `analysis.py` with only `scripts/research/lib/` modules beside
it, so a study that imported the solo runner could not be reproduced from Public. The solo
runner keeps its own copy, because changing a published runner underneath its numbers is
what `study_package.py` explains this repository does not do.
"""
from __future__ import annotations

import base64
from pathlib import Path

import fail_pattern_toolkit as tk

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
  .pos { color: #27ae60; } .neg { color: #e74c3c; } .neu { color: #2980b9; }
  .card { background: white; border-radius: 10px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-top: 20px; }
  .tbl { border-collapse: collapse; width: 100%; font-size: .85em; }
  .tbl th { background: #ecf0f1; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-top: 1px solid #ecf0f1; }
  .note { background:#fef9e7; border-left:4px solid #f1c40f; padding:10px 16px; border-radius:4px; margin-top:12px; font-size:.88em; color:#7d6608; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media(max-width:700px){ .chart-grid { grid-template-columns: 1fr; } }
  footer { text-align:center; color:#bdc3c7; font-size:.8em; margin-top:40px; padding-top:16px; border-top:1px solid #ecf0f1; }
</style>
"""


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0;">'


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _stats_rows(stats_dict: dict, order: list[str] | None = None) -> list[list]:
    keys = order if order else list(stats_dict.keys())
    rows = []
    for key in keys:
        v = stats_dict.get(key)
        if not v or not v.get("n"):
            continue
        row = [key, v["n"], f'{v["win_rate_pct"]}%', v["profit_factor"] or "-", f'${v["net_pnl_usd"]:,.2f}']
        if "rank_score" in v:
            marker = " (low-sample)" if v.get("low_sample") else ""
            row.append(f'{v["rank_score"]:+d}{marker}')
        rows.append(row)
    return rows


def render_html(results: dict, charts_dir: Path) -> str:
    b64 = {c["id"]: base64_of(charts_dir / c["file"]) for c in results["charts"]}
    baseline = results["baseline"]
    strategy = results["strategy"]
    kpis = [
        ("Trades", baseline["n"], "neu"),
        ("Win Rate", f'{baseline["win_rate_pct"]}%', "pos" if baseline["win_rate_pct"] >= 50 else "neg"),
        ("Profit Factor", baseline["profit_factor"], "pos" if (baseline["profit_factor"] or 0) >= 1.5 else "neu"),
        ("Net P&L", f'${baseline["net_pnl_usd"]:,.0f}', "pos" if baseline["net_pnl_usd"] >= 0 else "neg"),
        ("Max Drawdown", f'${baseline["max_drawdown_usd"]:,.0f}', "neg"),
        ("Max Consec Loss", baseline["max_consecutive_losses"], "neu"),
        ("Avg Hold", f'{baseline["avg_hold_bars"]:.0f} bars', "neu"),
    ]
    kpi_html = '<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value {c}">{v}</div></div>' for l, v, c in kpis
    ) + "</div>"

    fail_rows = [[k, v["count"], f'{v["pct"]}%'] for k, v in results["fail_pattern"]["by_type"].items()]
    session_rows = _stats_rows(results["by_session"], ["asia", "europe", "us", "overnight"])
    entry_30m_rows = _stats_rows(results["by_entry_30m"], tk.ENTRY_SLOTS_30M)
    bb_rows = _stats_rows(results["bb_zone"], tk.BB_ZONE_ORDER)
    dxy_bucket_rows = _stats_rows(results["dxy"]["regime"]["by_bucket"])
    mtf_align_rows = _stats_rows(results["mtf"]["by_alignment"])

    has_rank_score = "rank_score" in results["by_entry_30m"].get(tk.ENTRY_SLOTS_30M[0], {})
    session_headers = ["session", "n", "WR", "PF", "Net"] + (["Score"] if has_rank_score else [])
    entry_headers = ["slot", "n", "WR", "PF", "Net"] + (["Score"] if has_rank_score else [])

    macro_html = ""
    if "by_macro_verdict" in results:
        macro_rows = _stats_rows(results["by_macro_verdict"], ["WAIT", "NEUTRAL", "STRONG BUY"])
        macro_headers = ["verdict", "n", "WR", "PF", "Net"] + (["Score"] if has_rank_score else [])
        macro_html = f"""
  <h2>Macro Composite Context</h2>
  <div class="card">
    {_img(b64.get("macro_verdict_winrate", ""))}
    {_table(macro_headers, macro_rows)}
    <div class="note">Macro coverage: {results["macro_coverage"]["matched"]}/{results["macro_coverage"]["matched"] + results["macro_coverage"]["unmatched"]} trades ({results["macro_coverage"]["pct"]}%).</div>
  </div>
"""

    temporal_html = ""
    if "temporal_stability" in results:
        ts = results["temporal_stability"]
        period_rows = _stats_rows(ts["by_period"])
        holdout = ts["holdout_split"]
        flag = ts["degradation_flag"]
        flag_class = {"stable": "neu", "improved": "pos", "degraded": "neg"}.get(flag, "neu")
        temporal_html = f"""
  <h2>Temporal Stability — Chronological Holdout</h2>
  <div class="card">
    {_img(b64.get("quarterly_stability", ""))}
    {_table(["quarter", "n", "WR", "PF", "Net"], period_rows)}
    <div class="note">
      In-sample ({holdout['split_ratio']*100:.0f}%, {holdout['in_sample']['period']['start']} &rarr; {holdout['in_sample']['period']['end']}):
      n={holdout['in_sample']['n']}, WR {holdout['in_sample']['win_rate_pct']}%, PF {holdout['in_sample']['profit_factor']}.
      Held-out ({(1-holdout['split_ratio'])*100:.0f}%, {holdout['held_out']['period']['start']} &rarr; {holdout['held_out']['period']['end']}):
      n={holdout['held_out']['n']}, WR {holdout['held_out']['win_rate_pct']}%, PF {holdout['held_out']['profit_factor']}.
      Degradation flag: <span class="{flag_class}"><b>{flag}</b></span>.
      {results['method'].get('temporal_stability_limitation', '')}
    </div>
  </div>
"""

    generated_at = results["generated_at"]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy} — Fail Pattern Report (30-min)</title>
{CSS}
</head>
<body>
<div class="wrap">
  <h1>{strategy}</h1>
  <div class="meta">Generated {generated_at} &nbsp;|&nbsp; {results["trade_period"]["start"]} → {results["trade_period"]["end"]}</div>
  {kpi_html}

  <h2>Equity Curve & Drawdown</h2>
  <div class="card">{_img(b64["equity_curve"])}</div>

  <h2>Fail Pattern Breakdown</h2>
  <div class="card">
    <div class="chart-grid">{_img(b64["fail_type_breakdown"])}{_img(b64["mfe_distribution"])}</div>
    {_img(b64["mae_vs_mfe"])}
    {_table(["fail_type", "count", "pct"], fail_rows)}
  </div>

  <h2>Session Summary {"(scored; overnight descriptive)" if has_rank_score else "(descriptive)"}</h2>
  <div class="card">{_table(session_headers, session_rows)}</div>

  <h2>30-Minute Entry-Slot Timing (primary evidence)</h2>
  <div class="card">
    {_img(b64["entry_slot_30m_winrate"])}
    {_img(b64["immediate_loss_by_slot_30m"])}
    {_table(entry_headers, entry_30m_rows)}
  </div>
  {macro_html}
  {temporal_html}
  <h2>Pre-Entry Context — Immediate Loss</h2>
  <div class="card">
    <div class="chart-grid">{_img(b64["pre_entry_dow"])}{_img(b64["pre_entry_prev_result"])}</div>
    {_img(b64["pre_entry_tsw"])}
  </div>

  <h2>K-Bar Features at Entry</h2>
  <div class="card">
    {_img(b64["kbar_features"])}
    <div class="note">K-Bar coverage: {results["kbar_coverage"]["with_kbar_data"]}/{results["kbar_coverage"]["total_immediate_loss"]} immediate_loss trades ({results["kbar_coverage"]["coverage_pct"]}%).</div>
  </div>

  <h2>Bollinger Band Position</h2>
  <div class="card">{_img(b64["bb_zone_winrate"])}{_table(["zone", "n", "WR", "PF", "Net"], bb_rows)}</div>

  <h2>DXY Context</h2>
  <div class="card">
    {_img(b64["dxy_winrate"])}
    {_img(b64["dxy_correlation"])}
    <div class="note">Avg 30-day rolling DXY-XAUUSD correlation: {results["dxy"]["avg_30d_correlation"]}</div>
    {_table(["DXY RSI bucket", "n", "WR", "PF", "Net"], dxy_bucket_rows)}
  </div>

  <h2>Multi-Timeframe Alignment</h2>
  <div class="card">
    {_img(b64["htf_alignment"])}
    {_img(b64["htf_4h_state"])}
    {_img(b64["htf_4h_bucket"])}
    {_table(["HTF alignment", "n", "WR", "PF", "Net"], mtf_align_rows)}
  </div>

  <h2>Hold Time & Streaks</h2>
  <div class="card"><div class="chart-grid">{_img(b64["hold_time_dist"])}{_img(b64["consecutive_losses"])}</div></div>

  <footer>XAUUSD Strategy Fail-Pattern Toolkit (30-min contract) &nbsp;·&nbsp; {generated_at}</footer>
</div>
</body>
</html>"""
    return html


def base64_of(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


