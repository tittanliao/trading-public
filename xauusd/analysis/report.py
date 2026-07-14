"""
HTML report generator for XAUUSD strategy analysis.

Produces a self-contained HTML file (all charts embedded as base64 PNG)
saved directly into the strategy's source folder.

Usage (called from main.py):
    from analysis import report
    report.generate(name, version, trades, classified, sess, profile, enriched, out_path)
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import metrics, fail_patterns, pre_entry, charts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;margin:8px 0;">'


def _table(df: pd.DataFrame) -> str:
    return df.to_html(classes="tbl", border=0, float_format="{:.3f}".format)


CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9; color:#2c3e50; margin:0; padding:0; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 1.8em; margin-bottom: 4px; }
  h2 { font-size: 1.2em; border-bottom: 2px solid #3498db; padding-bottom: 4px;
       margin-top: 36px; color: #2980b9; }
  .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 24px; }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .kpi { background: white; border-radius: 10px; padding: 16px 24px;
         box-shadow: 0 1px 4px rgba(0,0,0,.1); min-width: 140px; flex: 1; }
  .kpi-label { font-size: .8em; color: #7f8c8d; text-transform: uppercase; letter-spacing:.05em; }
  .kpi-value { font-size: 1.6em; font-weight: 700; margin-top: 2px; }
  .pos { color: #27ae60; }  .neg { color: #e74c3c; }  .neu { color: #2980b9; }
  .card { background: white; border-radius: 10px; padding: 20px 24px;
          box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-top: 20px; }
  .tbl { border-collapse: collapse; width: 100%; font-size: .9em; }
  .tbl th { background: #ecf0f1; padding: 8px 12px; text-align: left; }
  .tbl td { padding: 6px 12px; border-top: 1px solid #ecf0f1; }
  .tbl tr:hover td { background: #f8f9fa; }
  .note { background:#fef9e7; border-left:4px solid #f1c40f; padding:10px 16px;
          border-radius:4px; margin-top:12px; font-size:.88em; color:#7d6608; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media(max-width:700px){ .chart-grid { grid-template-columns: 1fr; } }
  footer { text-align:center; color:#bdc3c7; font-size:.8em; margin-top:40px; padding-top:16px;
           border-top:1px solid #ecf0f1; }
</style>
"""


# ---------------------------------------------------------------------------
# BB section helper
# ---------------------------------------------------------------------------

def _bb_section_html(bb_stats_out: dict | None) -> str:
    if not bb_stats_out or "by_zone" not in bb_stats_out:
        return ""

    df = bb_stats_out["by_zone"].copy()
    df = df.dropna(subset=["win_rate"])
    if df.empty:
        return ""

    import base64, io
    import matplotlib.pyplot as plt
    import numpy as np

    # Chart: win rate by BB zone
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = []
    for wr in df["win_rate"]:
        if wr >= 0.60: colors.append("#27ae60")
        elif wr >= 0.45: colors.append("#3498db")
        else: colors.append("#e74c3c")
    bars = ax.bar(range(len(df)), df["win_rate"] * 100, color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([z.replace("_", " ") for z in df.index], rotation=30, ha="right", fontsize=9)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Win Rate %")
    ax.set_title("Win Rate by BB Zone at Entry", fontsize=11)
    for i, (wr, n) in enumerate(zip(df["win_rate"], df["total"])):
        ax.text(i, wr*100 + 1.5, f"{wr:.0%}\n(n={int(n)})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    img_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;margin:8px 0">'

    # Table HTML
    rows = []
    for zone, row in df.iterrows():
        wr_cls = "color:#27ae60;font-weight:600" if row["win_rate"] >= 0.60 else ("color:#e74c3c" if row["win_rate"] < 0.40 else "")
        rows.append(f"<tr><td>{zone.replace('_',' ')}</td><td>{int(row['total'])}</td>"
                    f"<td style='{wr_cls}'>{row['win_rate']:.1%}</td>"
                    f"<td>${row['avg_pnl']:.2f}</td></tr>")
    table_html = (
        "<table style='font-size:.85em;margin-top:12px'>"
        "<thead><tr><th>BB Zone</th><th>Trades</th><th>Win Rate</th><th>Avg PnL</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    # Find best zone
    best_zone = df["win_rate"].idxmax()
    best_wr = df.loc[best_zone, "win_rate"]
    insight = f"Best zone: <strong>{best_zone.replace('_',' ')}</strong> ({best_wr:.1%} WR)"
    if best_zone in ("near_upper", "above_upper"):
        insight += " — 進場時價格接近/突破上軌，代表強趨勢延續；建議啟用 BB 位置過濾。"

    return f"""
<h3>Bollinger Band Position Analysis</h3>
<p style="color:#666;font-size:.9em">Entry-time price position relative to BB(20,2). Higher zone = closer to/above upper band.</p>
{img_tag}
{table_html}
<p style="font-size:.88em;color:#555;margin-top:8px">{insight}</p>
"""


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def generate(
    strategy_id: str,
    version: str,
    trades: pd.DataFrame,
    classified: pd.DataFrame,
    sess_stats: pd.DataFrame,
    profile: dict,
    enriched: pd.DataFrame,
    out_path: Path,
    dxy_stats: dict | None = None,
    corr_df: pd.DataFrame | None = None,
    htf_enriched: pd.DataFrame | None = None,
    htf_stats_out: dict | None = None,
    bb_stats_out: dict | None = None,
    div_stats_out: dict | None = None,
) -> None:
    """
    Renders and writes a self-contained HTML report to out_path.
    """
    summ = metrics.summary(trades)
    imm = classified[classified["fail_type"] == "immediate_loss"]
    dd = metrics.max_drawdown(trades)
    streaks = metrics.consecutive_losses(trades)

    # ---- Build KPI cards ----
    wr_cls = "pos" if summ["win_rate"] >= 0.5 else "neg"
    pf_cls = "pos" if summ["profit_factor"] >= 1.5 else ("neu" if summ["profit_factor"] >= 1 else "neg")
    pnl_cls = "pos" if summ["net_pnl"] >= 0 else "neg"
    dd_cls = "neg"

    kpis = [
        ("Trades", f"{summ['total_trades']}", "neu"),
        ("Win Rate", f"{summ['win_rate']:.1%}", wr_cls),
        ("Profit Factor", f"{summ['profit_factor']:.2f}", pf_cls),
        ("Net P&L", f"${summ['net_pnl']:,.0f}", pnl_cls),
        ("Max Drawdown", f"${dd:,.0f}", dd_cls),
        ("Max Consec Loss", f"{summ['max_consec_losses']}", "neg" if summ["max_consec_losses"] >= 5 else "neu"),
        ("Avg Hold", f"{summ['avg_hold_bars']:.0f} bars", "neu"),
    ]

    kpi_html = '<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-value {cls}">{val}</div></div>'
        for lbl, val, cls in kpis
    ) + "</div>"

    # ---- Charts ----
    eq_b64   = _fig_to_b64(charts.equity_curve(trades, strategy_id))
    ft_b64   = _fig_to_b64(charts.fail_type_breakdown(classified, strategy_id))
    mfe_b64  = _fig_to_b64(charts.mfe_distribution(classified, strategy_id))
    scat_b64 = _fig_to_b64(charts.mae_vs_mfe_scatter(classified, strategy_id))
    sess_b64 = _fig_to_b64(charts.session_heatmap(sess_stats, strategy_id))
    hr_b64   = _fig_to_b64(charts.hourly_winrate(fail_patterns.hourly_stats(trades), strategy_id))
    ht_b64   = _fig_to_b64(charts.hold_time_dist(trades, strategy_id))
    cl_b64   = _fig_to_b64(charts.consecutive_loss_hist(streaks, strategy_id))

    pre_entry_charts_html = ""
    if profile:
        ph_b64  = _fig_to_b64(charts.pre_entry_hour(profile, strategy_id))
        pd_b64  = _fig_to_b64(charts.pre_entry_dow(profile, strategy_id))
        pp_b64  = _fig_to_b64(charts.pre_entry_prev_result(profile, strategy_id))
        pt_b64  = _fig_to_b64(charts.pre_entry_tsw(profile, strategy_id))
        kb_b64  = _fig_to_b64(charts.kbar_feature_summary(enriched, strategy_id))

        cov = pre_entry.kbar_coverage(enriched)
        kb_note = (
            f'<div class="note">K-Bar coverage: {cov["with_kbar_data"]}/{cov["total_losses"]} '
            f'trades ({cov["coverage_pct"]}%). '
            f'To extend coverage, export the full OHLCV+RSI history from TradingView and '
            f'place it in <code>csv/FX_IDC_XAUUSD, 30.csv</code>.</div>'
            if cov["coverage_pct"] < 100 else ""
        )

        pre_entry_charts_html = f"""
        <h2>Pre-Entry Analysis — Immediate Loss ({len(imm)} trades)</h2>
        <div class="card">
          <div class="chart-grid">
            {_img_tag(ph_b64, "Entry Hour")}
            {_img_tag(pd_b64, "Day of Week")}
          </div>
          <div class="chart-grid">
            {_img_tag(pp_b64, "Previous Result")}
            {_img_tag(pt_b64, "Trades Since Win")}
          </div>
        </div>
        <h2>K-Bar Features at Entry</h2>
        <div class="card">
          {_img_tag(kb_b64, "K-Bar Features")}
          {kb_note}
        </div>
        """

    # ---- Fail pattern table ----
    fail_summary_html = _table(fail_patterns.fail_type_summary(classified))

    # ---- Session table ----
    sess_display = sess_stats[["total", "wins", "win_rate", "avg_pnl", "avg_mfe", "avg_mae"]].copy()
    sess_display.columns = ["Total", "Wins", "Win Rate", "Avg P&L", "Avg MFE%", "Avg MAE%"]
    sess_html = _table(sess_display)

    # ---- Fail by session table ----
    fbs_html = _table(fail_patterns.fail_by_session(classified))

    # ---- BB section ----
    bb_html = _bb_section_html(bb_stats_out)

    # ---- DXY section ----
    dxy_html = ""
    if dxy_stats:
        dxy_wr_b64 = _fig_to_b64(charts.dxy_winrate_chart(dxy_stats, strategy_id))
        rows = []
        for grp_name, df_grp in dxy_stats.items():
            if not df_grp.empty:
                rows.append(f"<b>{grp_name.replace('_', ' ').title()}</b>")
                rows.append(_table(df_grp))
        dxy_tables = "<br>".join(rows)
        corr_html = ""
        if corr_df is not None and not corr_df.empty:
            corr_b64 = _fig_to_b64(charts.dxy_correlation_chart(corr_df, strategy_id))
            avg_corr = corr_df["rolling_corr"].dropna().mean()
            corr_html = f"""
            <h2>DXY × XAUUSD Return Correlation</h2>
            <div class="card">
              <div class="note">30-day rolling correlation between DXY and XAUUSD daily returns.
                Avg: <b>{avg_corr:.3f}</b> (negative = inverse relationship, as expected for gold).</div>
              {_img_tag(corr_b64, "DXY Correlation")}
            </div>"""
        dxy_html = f"""
        <h2>DXY (US Dollar Index) Analysis</h2>
        <div class="card">
          <p style="color:#7f8c8d;font-size:.9em;">
            DXY and gold typically move inversely — a strengthening dollar pressures gold prices.
            Win rate is shown per DXY regime at each trade entry date (using 1D DXY data).
          </p>
          {_img_tag(dxy_wr_b64, "DXY Win Rate")}
          <br>{dxy_tables}
        </div>
        {corr_html}
        """

    # ---- MTF section ----
    mtf_html = ""
    if htf_stats_out and htf_enriched is not None and not htf_enriched.empty:
        align_b64   = _fig_to_b64(charts.htf_alignment_bar(htf_stats_out, strategy_id))
        state4h_b64 = _fig_to_b64(charts.htf_4h_state_bar(htf_stats_out, strategy_id))
        bucket_b64  = _fig_to_b64(charts.htf_bucket_heatmap(htf_stats_out, strategy_id))

        cov_html = ""
        if "coverage" in htf_stats_out and not htf_stats_out["coverage"].empty:
            cov_html = "<b>Data coverage:</b><br>" + _table(htf_stats_out["coverage"])

        def _tbl_or_empty(key, label):
            df = htf_stats_out.get(key)
            if df is None or df.empty:
                return ""
            return f"<b>{label}</b><br>" + _table(df) + "<br>"

        conflict_note = ""
        ct = htf_stats_out.get("by_conflict")
        if ct is not None and not ct.empty and "counter-trend (4H bearish)" in ct.index:
            ct_row = ct.loc["counter-trend (4H bearish)"]
            al_row = ct.loc["aligned (4H not bearish)"] if "aligned (4H not bearish)" in ct.index else None
            if al_row is not None:
                diff = ct_row["win_rate"] - al_row["win_rate"]
                sign = "lower" if diff < 0 else "higher"
                conflict_note = (
                    f'<div class="note">⚠️ Counter-trend entries (4H bearish) have a '
                    f'<b>{abs(diff)*100:.1f}% {sign}</b> win rate than aligned entries '
                    f'({ct_row["win_rate"]*100:.1f}% vs {al_row["win_rate"]*100:.1f}%). '
                    f'Consider filtering out entries when 4H RSI is in a bearish state.</div>'
                )

        mtf_html = f"""
        <h2>Multi-Timeframe (MTF) Analysis</h2>
        <div class="card">
          <p style="color:#7f8c8d;font-size:.9em;">
            Each 30m entry is tagged with the nearest higher-timeframe (60m, 4H, 1D) RSI context.
            <b>State definition:</b> bullish = RSI &gt; RSI-MA <em>and</em> RSI-MA slope rising;
            bearish = RSI &lt; RSI-MA <em>and</em> RSI-MA slope falling; neutral = mixed signals.
          </p>
          {cov_html}
          <br>
          {_img_tag(align_b64, "HTF Alignment")}
          {_img_tag(state4h_b64, "4H State")}
          {_img_tag(bucket_b64, "4H RSI Bucket")}
          {conflict_note}
          <br>
          {_tbl_or_empty("by_alignment",  "Win Rate by HTF Alignment Score")}
          {_tbl_or_empty("by_4h_state",   "Win Rate by 4H RSI State")}
          {_tbl_or_empty("by_4h_bucket",  "Win Rate by 4H RSI Bucket")}
          {_tbl_or_empty("by_1d_state",   "Win Rate by 1D RSI State")}
          {_tbl_or_empty("by_conflict",   "Win Rate: Counter-Trend vs Aligned")}
          {_tbl_or_empty("by_vol_regime", "Win Rate: High-Vol vs Normal")}
          {_tbl_or_empty("fail_by_4h",    "Fail Type Distribution by 4H State (% of losses)")}
        </div>
        """

    # ---- Assemble HTML ----
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{strategy_id} v{version} — Fail Pattern Report</title>
  {CSS}
</head>
<body>
<div style="padding:10px 20px 0"><a href="../../index.html" style="color:#2563eb;font-size:13px;text-decoration:none;background:#dbeafe;padding:4px 12px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="wrap">
  <h1>{strategy_id} <small style="color:#7f8c8d">v{version}</small></h1>
  <div class="meta">Generated {generated_at} &nbsp;|&nbsp;
       {summ['total_trades']} trades &nbsp;|&nbsp;
       {trades['entry_time'].min().date()} → {trades['entry_time'].max().date()}
  </div>

  {kpi_html}

  <h2>Equity Curve &amp; Drawdown</h2>
  <div class="card">{_img_tag(eq_b64, "Equity Curve")}</div>

  <h2>Fail Pattern Breakdown</h2>
  <div class="card">
    <div class="chart-grid">
      {_img_tag(ft_b64, "Fail Types")}
      {_img_tag(mfe_b64, "MFE Distribution")}
    </div>
    {_img_tag(scat_b64, "MAE vs MFE")}
    <br>
    {fail_summary_html}
  </div>

  <h2>Session &amp; Time Analysis</h2>
  <div class="card">
    {_img_tag(sess_b64, "Session Heatmap")}
    {_img_tag(hr_b64, "Hourly Win Rate")}
    <br>
    {sess_html}
    <br>
    <b>Fail type by session:</b><br>
    {fbs_html}
  </div>

  <h2>Hold Time &amp; Streak Analysis</h2>
  <div class="card">
    <div class="chart-grid">
      {_img_tag(ht_b64, "Hold Time")}
      {_img_tag(cl_b64, "Consecutive Losses")}
    </div>
  </div>

  {pre_entry_charts_html}

  {bb_html}

  {dxy_html}

  {mtf_html}

  <footer>XAUUSD Strategy Fail-Pattern Toolkit &nbsp;·&nbsp; {generated_at}</footer>
</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  HTML report → {out_path}")
