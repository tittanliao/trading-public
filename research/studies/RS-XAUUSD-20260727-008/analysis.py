#!/usr/bin/env python3
"""Build the S2 V1.9 vs V3.2 gap ("comparison") report per
docs/RESEARCH_DEVELOPMENT_SPEC.md section 5.

Reads the two solo studies' results.json (never recomputes the underlying fail-pattern
breakdown) and produces version-delta tables/charts only. Structurally identical to
scripts/research/build_s1_fail_pattern_gap.py; kept as a separate file rather than
generalizing that one in place, to avoid touching a script already cited by the
committed RS-XAUUSD-20260727-005 study package.

Usage:
    /opt/homebrew/bin/python3.12 scripts/research/build_s2_fail_pattern_gap.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import fail_pattern_toolkit as tk  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260727-008"
V1_STUDY = "RS-XAUUSD-20260727-006"
V2_STUDY = "RS-XAUUSD-20260727-007"
V1_LABEL, V2_LABEL = "V1.9", "V3.2"
STRATEGY_ID = "S2-Hammer"

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
  .neu { color: #2980b9; }
  .card { background: white; border-radius: 10px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-top: 20px; }
  .tbl { border-collapse: collapse; width: 100%; font-size: .85em; }
  .tbl th { background: #ecf0f1; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-top: 1px solid #ecf0f1; }
  .note { background:#fef9e7; border-left:4px solid #f1c40f; padding:10px 16px; border-radius:4px; margin-top:12px; font-size:.88em; color:#7d6608; }
  footer { text-align:center; color:#bdc3c7; font-size:.8em; margin-top:40px; padding-top:16px; border-top:1px solid #ecf0f1; }
</style>
"""

RISK_CAVEAT = (
    f"{STRATEGY_ID} evolved from a Pullback-style {V1_LABEL} signal to a "
    f"Hammer-candle-style {V2_LABEL} signal (see the .pine source files for exact rule "
    "differences); this is not a like-for-like backtest of one unchanged rule set across "
    "time. Deltas below may reflect rule changes, not only timing/regime differences."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diff_stats_table(v1: dict, v2: dict, keys: list[str]) -> dict:
    out = {}
    for key in keys:
        a, b = v1.get(key, {}), v2.get(key, {})
        wr_a, wr_b = a.get("win_rate_pct"), b.get("win_rate_pct")
        out[key] = {
            "v1_n": a.get("n", 0), "v1_win_rate_pct": wr_a, "v1_profit_factor": a.get("profit_factor"),
            "v2_n": b.get("n", 0), "v2_win_rate_pct": wr_b, "v2_profit_factor": b.get("profit_factor"),
            "win_rate_pct_diff_v2_minus_v1": round(wr_b - wr_a, 2) if (wr_a is not None and wr_b is not None) else None,
        }
    return out


def grouped_bar_chart(labels: list[str], v1_vals: list[float], v2_vals: list[float], title: str, ylabel: str) -> plt.Figure:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4))
    ax.bar(x - width / 2, v1_vals, width, label=V1_LABEL, color="#95a5a6")
    ax.bar(x + width / 2, v2_vals, width, label=V2_LABEL, color="#3498db")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def entry_slot_diff_chart(comparison: dict) -> plt.Figure:
    slots = tk.ENTRY_SLOTS_30M
    diffs = [comparison[s]["win_rate_pct_diff_v2_minus_v1"] or 0 for s in slots]
    colors = ["#27ae60" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6" for d in diffs]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(range(len(slots)), diffs, color=colors)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(range(0, len(slots), 2))
    ax.set_xticklabels([slots[i] for i in range(0, len(slots), 2)], rotation=60, ha="right", fontsize=7)
    ax.set_title(f"Win-Rate Diff by 30-Minute Slot ({V2_LABEL} − {V1_LABEL})")
    ax.set_ylabel("WR diff (pp)")
    fig.tight_layout()
    return fig


def build(output_dir: Path) -> dict:
    v1_path = Path("research/studies") / V1_STUDY / "results.json"
    v2_path = Path("research/studies") / V2_STUDY / "results.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))

    baseline_diff = {
        "v1": v1["baseline"], "v2": v2["baseline"],
        "win_rate_pct_diff": round(v2["baseline"]["win_rate_pct"] - v1["baseline"]["win_rate_pct"], 2),
        "profit_factor_diff": round(v2["baseline"]["profit_factor"] - v1["baseline"]["profit_factor"], 3),
    }

    entry_30m_diff = diff_stats_table(v1["by_entry_30m"], v2["by_entry_30m"], tk.ENTRY_SLOTS_30M)

    fail_type_share = {}
    for ft in ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]:
        a = v1["fail_pattern"]["by_type"].get(ft, {"pct": 0})
        b = v2["fail_pattern"]["by_type"].get(ft, {"pct": 0})
        fail_type_share[ft] = {"v1_pct": a["pct"], "v2_pct": b["pct"], "diff": round(b["pct"] - a["pct"], 1)}

    bb_zone_diff = diff_stats_table(v1["bb_zone"], v2["bb_zone"], tk.BB_ZONE_ORDER)
    dxy_bucket_keys = sorted(set(v1["dxy"]["regime"]["by_bucket"]) | set(v2["dxy"]["regime"]["by_bucket"]))
    dxy_bucket_diff = diff_stats_table(v1["dxy"]["regime"]["by_bucket"], v2["dxy"]["regime"]["by_bucket"], dxy_bucket_keys)
    mtf_4h_diff = diff_stats_table(v1["mtf"]["by_4h_state"], v2["mtf"]["by_4h_state"], ["bearish", "neutral", "bullish"])
    mtf_conflict_diff = diff_stats_table(
        v1["mtf"]["by_conflict"], v2["mtf"]["by_conflict"],
        ["aligned (4H not bearish)", "counter-trend (4H bearish)"],
    )

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts_manifest = []

    def save(fig, chart_id, title):
        path = charts_dir / f"{chart_id}.png"
        path.write_bytes(tk.fig_to_png_bytes(fig))
        charts_manifest.append({"id": chart_id, "file": f"{chart_id}.png", "title": title, "section": "comparison"})

    save(entry_slot_diff_chart(entry_30m_diff), "entry_slot_30m_diff", f"30-Minute Slot Win-Rate Diff ({V2_LABEL}-{V1_LABEL})")

    ft_labels = list(fail_type_share.keys())
    save(
        grouped_bar_chart(ft_labels, [fail_type_share[k]["v1_pct"] for k in ft_labels],
                           [fail_type_share[k]["v2_pct"] for k in ft_labels],
                           f"Fail-Type Share: {V1_LABEL} vs {V2_LABEL}", "% of losses"),
        "fail_type_share_comparison", "Fail-Type Share Comparison",
    )

    bb_labels = [z for z in tk.BB_ZONE_ORDER if bb_zone_diff[z]["v1_n"] or bb_zone_diff[z]["v2_n"]]
    save(
        grouped_bar_chart(bb_labels, [bb_zone_diff[z]["v1_win_rate_pct"] or 0 for z in bb_labels],
                           [bb_zone_diff[z]["v2_win_rate_pct"] or 0 for z in bb_labels],
                           f"Win Rate by BB Zone: {V1_LABEL} vs {V2_LABEL}", "Win Rate %"),
        "bb_zone_comparison", "BB Zone Win-Rate Comparison",
    )

    dxy_labels = [k for k in dxy_bucket_keys if dxy_bucket_diff[k]["v1_n"] or dxy_bucket_diff[k]["v2_n"]]
    save(
        grouped_bar_chart(dxy_labels, [dxy_bucket_diff[k]["v1_win_rate_pct"] or 0 for k in dxy_labels],
                           [dxy_bucket_diff[k]["v2_win_rate_pct"] or 0 for k in dxy_labels],
                           f"Win Rate by DXY RSI Bucket: {V1_LABEL} vs {V2_LABEL}", "Win Rate %"),
        "dxy_bucket_comparison", "DXY Bucket Win-Rate Comparison",
    )

    mtf_labels = [k for k in ["bearish", "neutral", "bullish"] if mtf_4h_diff[k]["v1_n"] or mtf_4h_diff[k]["v2_n"]]
    save(
        grouped_bar_chart(mtf_labels, [mtf_4h_diff[k]["v1_win_rate_pct"] or 0 for k in mtf_labels],
                           [mtf_4h_diff[k]["v2_win_rate_pct"] or 0 for k in mtf_labels],
                           f"Win Rate by 4H RSI State: {V1_LABEL} vs {V2_LABEL}", "Win Rate %"),
        "mtf_4h_state_comparison", "MTF 4H-State Win-Rate Comparison",
    )

    generated_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": generated_at,
        "strategy": f"{STRATEGY_ID} {V1_LABEL} vs {V2_LABEL} (gap report)",
        "version_labels": {"v1": V1_LABEL, "v2": V2_LABEL},
        "method": {
            "basis": f"reads {V1_STUDY} and {V2_STUDY} results.json; recomputes no fail-pattern classification",
            "risk_parameter_caveat": RISK_CAVEAT,
        },
        "baseline_diff": baseline_diff,
        "by_entry_30m_diff": entry_30m_diff,
        "fail_type_share_diff": fail_type_share,
        "bb_zone_diff": bb_zone_diff,
        "dxy_bucket_diff": dxy_bucket_diff,
        "mtf_4h_state_diff": mtf_4h_diff,
        "mtf_conflict_diff": mtf_conflict_diff,
        "charts": charts_manifest,
        "sources": [
            {"role": f"solo_study_{V1_STUDY}", "path": str(v1_path), "sha256": sha256(v1_path)},
            {"role": f"solo_study_{V2_STUDY}", "path": str(v2_path), "sha256": sha256(v2_path)},
        ],
    }
    results = tk.to_json_safe(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(results, charts_dir), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(results), encoding="utf-8")
    return results


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0;">'


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def base64_of(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode()


def render_html(results: dict, charts_dir: Path) -> str:
    b64 = {c["id"]: base64_of(charts_dir / c["file"]) for c in results["charts"]}
    bd = results["baseline_diff"]
    kpis = [
        (f"{V1_LABEL} WR/PF", f'{bd["v1"]["win_rate_pct"]}% / {bd["v1"]["profit_factor"]}', "neu"),
        (f"{V2_LABEL} WR/PF", f'{bd["v2"]["win_rate_pct"]}% / {bd["v2"]["profit_factor"]}', "neu"),
        (f"WR diff ({V2_LABEL}-{V1_LABEL})", f'{bd["win_rate_pct_diff"]:+.2f}pp', "neu"),
        (f"PF diff ({V2_LABEL}-{V1_LABEL})", f'{bd["profit_factor_diff"]:+.3f}', "neu"),
    ]
    kpi_html = '<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value {c}">{v}</div></div>' for l, v, c in kpis
    ) + "</div>"

    entry_rows = []
    for slot in tk.ENTRY_SLOTS_30M:
        item = results["by_entry_30m_diff"][slot]
        if not (item["v1_n"] or item["v2_n"]):
            continue
        diff = item["win_rate_pct_diff_v2_minus_v1"]
        entry_rows.append([slot, item["v1_n"], f'{item["v1_win_rate_pct"]}%', item["v2_n"],
                            f'{item["v2_win_rate_pct"]}%', f'{diff:+.2f}pp' if diff is not None else "-"])

    fail_rows = [[k, f'{v["v1_pct"]}%', f'{v["v2_pct"]}%', f'{v["diff"]:+.1f}pp'] for k, v in results["fail_type_share_diff"].items()]

    generated_at = results["generated_at"]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{STRATEGY_ID} {V1_LABEL} vs {V2_LABEL} — Gap Report</title>
{CSS}
</head>
<body>
<div class="wrap">
  <h1>{STRATEGY_ID} — {V1_LABEL} vs {V2_LABEL} Gap Report</h1>
  <div class="meta">Generated {generated_at} &nbsp;|&nbsp; reads {V1_STUDY} + {V2_STUDY}, recomputes nothing</div>
  {kpi_html}
  <div class="note">{RISK_CAVEAT}</div>

  <h2>30-Minute Entry-Slot Win-Rate Diff</h2>
  <div class="card">
    {_img(b64["entry_slot_30m_diff"])}
    {_table(["slot", f"{V1_LABEL} n", f"{V1_LABEL} WR", f"{V2_LABEL} n", f"{V2_LABEL} WR", "diff"], entry_rows)}
  </div>

  <h2>Fail-Type Share</h2>
  <div class="card">
    {_img(b64["fail_type_share_comparison"])}
    {_table(["fail_type", V1_LABEL, V2_LABEL, "diff"], fail_rows)}
  </div>

  <h2>Bollinger Band Zone</h2>
  <div class="card">{_img(b64["bb_zone_comparison"])}</div>

  <h2>DXY RSI Bucket</h2>
  <div class="card">{_img(b64["dxy_bucket_comparison"])}</div>

  <h2>MTF 4H State</h2>
  <div class="card">{_img(b64["mtf_4h_state_comparison"])}</div>

  <footer>XAUUSD Strategy Fail-Pattern Toolkit — Gap Report &nbsp;·&nbsp; {generated_at}</footer>
</div>
</body>
</html>"""
    return html


def render_readme(results: dict) -> str:
    bd = results["baseline_diff"]
    lines = [
        f"# {results['study_id']} — {STRATEGY_ID} {V1_LABEL} vs {V2_LABEL} gap report",
        "",
        f"Generated: `{results['generated_at']}`",
        "",
        "## Scope",
        "",
        f"- Reads `{V1_STUDY}` and `{V2_STUDY}` results.json; recomputes no fail-pattern classification.",
        f"- Baseline: {V1_LABEL} WR {bd['v1']['win_rate_pct']}% / PF {bd['v1']['profit_factor']}; "
        f"{V2_LABEL} WR {bd['v2']['win_rate_pct']}% / PF {bd['v2']['profit_factor']} "
        f"(diff {bd['win_rate_pct_diff']:+.2f}pp / {bd['profit_factor_diff']:+.3f}).",
        f"- **Rule-change caveat**: {RISK_CAVEAT}",
        "",
        "## Interpretation",
        "",
        "- See `report.html` for the full chart-embedded comparison and `results.json` for structured deltas.",
        "- Only the owner-confirmed active S2 version is eligible to affect any live S2 advisory score.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("research/studies") / STUDY_ID)
    args = parser.parse_args()
    results = build(args.output_dir.resolve())
    print(json.dumps({
        "study_id": STUDY_ID, "output": str(args.output_dir),
        "baseline_diff": results["baseline_diff"], "chart_count": len(results["charts"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
