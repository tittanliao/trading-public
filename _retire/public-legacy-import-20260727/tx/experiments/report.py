"""
Generate a self-contained HTML report for long or short experiment results.
All charts are embedded as base64 PNG.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────

def _colour(val, mid=50, positive_good=True):
    """Return a CSS colour based on value relative to mid."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#888"
    ratio = (val - mid) / mid
    if not positive_good:
        ratio = -ratio
    ratio = max(-1, min(1, ratio))
    if ratio > 0:
        r, g, b = int(255 - 100 * ratio), int(220 - 40 * ratio), int(200 - 100 * ratio)
    else:
        r, g, b = int(220 - 40 * abs(ratio)), int(220 - 100 * abs(ratio)), int(200 - 100 * abs(ratio))
    return f"rgb({r},{g},{b})"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────

def _chart_ranking(rows: list[dict], direction: str) -> str:
    valid = [r for r in rows if r["n_trades"] > 0]
    if not valid:
        return ""
    codes = [r["code"] for r in valid]
    wrs = [r["win_rate"] for r in valid]
    pfs = [r["profit_factor"] for r in valid]
    scores = [r["score"] for r in valid]

    colours_wr = ["#4caf50" if w >= 45 else "#ff7043" if w < 38 else "#ffa726" for w in wrs]
    colours_pf = ["#4caf50" if p >= 1.3 else "#ff7043" if p < 1.0 else "#ffa726" for p in pfs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{'Long' if direction=='long' else 'Short'} Strategy Ranking", fontsize=13, fontweight="bold")

    # Win Rate
    axes[0].barh(codes[::-1], wrs[::-1], color=colours_wr[::-1])
    axes[0].axvline(50, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Win Rate (%)")
    axes[0].set_xlim(20, 80)

    # Profit Factor
    axes[1].barh(codes[::-1], pfs[::-1], color=colours_pf[::-1])
    axes[1].axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Profit Factor")
    axes[1].set_xlim(0, max(pfs) * 1.2 + 0.1)

    # Composite Score
    clrs = ["#2196f3" if s > 0.5 else "#90caf9" for s in scores]
    axes[2].barh(codes[::-1], scores[::-1], color=clrs[::-1])
    axes[2].set_title("Composite Score")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_equity(output_dir: Path, top_codes: list[str], direction: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"Top Equity Curves — {'Long' if direction=='long' else 'Short'}", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    colours = plt.cm.tab10.colors
    for i, code in enumerate(top_codes[:5]):
        csv_path = output_dir / f"{code}_trades.csv"
        if not csv_path.exists():
            continue
        trades = pd.read_csv(csv_path)
        if trades.empty:
            continue
        equity = trades["pnl_ntd"].cumsum()
        ax.plot(equity.values, label=code, color=colours[i % 10], linewidth=1.5)

    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (NT$)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_fail_patterns(rows: list[dict]) -> str:
    valid = [r for r in rows if r["n_trades"] > 0 and "fail_immediate" in r]
    if not valid:
        return ""

    codes = [r["code"] for r in valid]
    imm = [r.get("fail_immediate", 0) or 0 for r in valid]
    fb = [r.get("fail_false_break", 0) or 0 for r in valid]
    tb = [r.get("fail_time_bleed", 0) or 0 for r in valid]
    ns = [r.get("fail_normal_sl", 0) or 0 for r in valid]

    x = np.arange(len(codes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5 * width, imm, width, label="Immediate Loss", color="#ef5350")
    ax.bar(x - 0.5 * width, fb, width, label="False Breakout", color="#ff7043")
    ax.bar(x + 0.5 * width, tb, width, label="Time Bleed", color="#ffa726")
    ax.bar(x + 1.5 * width, ns, width, label="Normal SL", color="#78909c")
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of losing trades")
    ax.set_title("Fail Pattern Distribution")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_session(rows: list[dict]) -> str:
    valid = [r for r in rows if r.get("day_win_rate") and not np.isnan(r["day_win_rate"])]
    if not valid:
        return ""

    codes = [r["code"] for r in valid]
    day_wr = [r.get("day_win_rate", 0) or 0 for r in valid]
    night_wr = [r.get("night_win_rate", 0) or 0 for r in valid]

    x = np.arange(len(codes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, day_wr, width, label="Day Session", color="#42a5f5")
    ax.bar(x + width / 2, night_wr, width, label="Night Session", color="#7e57c2")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Day vs Night Session Win Rate")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


# ──────────────────────────────────────────────
# HTML builder
# ──────────────────────────────────────────────

def _row_html(r: dict) -> str:
    n = r["n_trades"]
    if n == 0:
        return f"<tr><td>{r.get('rank','—')}</td><td>{r['code']}</td><td>{r['name']}</td><td>{r['group']}</td>" \
               f"<td colspan='7' style='color:#999;text-align:center'>No trades</td></tr>"

    wr = r["win_rate"]
    pf = r["profit_factor"]
    pnl_ntd = r["net_pnl_ntd"]
    dd = r["max_dd_ntd"]
    score = r.get("score", 0)

    wr_style = f"background:{_colour(wr, mid=45)};color:#111"
    pf_style = f"background:{_colour(pf*30, mid=35)};color:#111"
    pnl_style = f"color:{'#2e7d32' if pnl_ntd > 0 else '#c62828'};font-weight:bold"

    return (
        f"<tr>"
        f"<td style='text-align:center'>{r.get('rank','—')}</td>"
        f"<td><b>{r['code']}</b></td>"
        f"<td>{r['name']}</td>"
        f"<td><span class='tag'>{r['group']}</span></td>"
        f"<td>{n}</td>"
        f"<td style='{wr_style}'>{wr}%</td>"
        f"<td style='{pf_style}'>{pf:.3f}</td>"
        f"<td style='{pnl_style}'>NT${pnl_ntd:,.0f}</td>"
        f"<td style='color:#c62828'>NT${abs(dd):,.0f}</td>"
        f"<td>{score:.3f}</td>"
        f"</tr>"
    )


def generate(output_dir: Path, direction: str):
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {output_dir}")

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["results"]
    sl_pts = data["sl_pts"]
    tp_pts = data["tp_pts"]
    label = "Long 多單" if direction == "long" else "Short 空單"
    top_codes = [r["code"] for r in rows[:5] if r["n_trades"] > 0]

    img_ranking = _chart_ranking(rows, direction)
    img_equity = _chart_equity(output_dir, top_codes, direction)
    img_fail = _chart_fail_patterns(rows)
    img_session = _chart_session(rows)

    def img_tag(b64):
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:6px">' if b64 else ""

    rows_html = "\n".join(_row_html(r) for r in rows)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>TX {label} Experiments</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f5f5; color: #212121; }}
  .header {{ background: {'#1565c0' if direction=='long' else '#880e4f'}; color: white; padding: 24px 32px; }}
  .header h1 {{ margin: 0; font-size: 1.6em; }}
  .header p {{ margin: 4px 0 0; opacity: 0.85; font-size: 0.9em; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 16px; }}
  .section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
               box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  h2 {{ font-size: 1.1em; color: #333; margin: 0 0 16px; border-left: 4px solid {'#1565c0' if direction=='long' else '#880e4f'}; padding-left: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ background: #eee; padding: 8px 10px; text-align: left; font-weight: 600; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #fafafa; }}
  .tag {{ background: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; white-space: nowrap; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media(max-width:900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div style="padding:10px 24px 0"><a href="../../index.html" style="color:white;font-size:13px;text-decoration:none;background:rgba(255,255,255,.2);padding:4px 12px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="header">
  <h1>台指期 MTX — {label} Experiments</h1>
  <p>SL: {sl_pts} pts &nbsp;|&nbsp; TP: {tp_pts} pts &nbsp;|&nbsp; R:R 1:{tp_pts//sl_pts} &nbsp;|&nbsp; 小台每點 NT$50</p>
</div>
<div class="container">

  <div class="section">
    <h2>策略排名</h2>
    <table>
      <thead><tr>
        <th>#</th><th>代碼</th><th>策略名稱</th><th>分組</th>
        <th>交易筆</th><th>勝率</th><th>獲利因子</th><th>淨盈虧</th><th>最大回撤</th><th>綜合分數</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>排名圖表</h2>
    {img_tag(img_ranking)}
  </div>

  <div class="section">
    <h2>前 5 名 Equity Curve</h2>
    {img_tag(img_equity)}
  </div>

  <div class="grid2">
    <div class="section">
      <h2>虧損型態分佈</h2>
      {img_tag(img_fail)}
    </div>
    <div class="section">
      <h2>日盤 vs 夜盤勝率</h2>
      {img_tag(img_session)}
    </div>
  </div>

</div>
</body>
</html>
"""

    out_path = output_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Report → {out_path}")
