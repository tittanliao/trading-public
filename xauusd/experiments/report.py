"""
HTML report generator for the 20-strategy experiment.

Produces one self-contained HTML file with:
    - Data coverage notice
    - Strategy comparison table with scores
    - Equity curves for top-5 strategies
    - Per-strategy trade detail tables
    - Development notes
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from experiments.engine import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str, alt: str = "", style: str = "max-width:100%;margin:8px 0;") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="{style}">'


CSS = """
<style>
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5;
       color: #2c3e50; margin: 0; padding: 0; }
.wrap { max-width: 1200px; margin: 0 auto; padding: 32px 20px; }
h1 { font-size: 2em; margin-bottom: 4px; }
h2 { font-size: 1.2em; border-left: 4px solid #3498db; padding-left: 12px;
     margin-top: 36px; color: #2980b9; }
h3 { font-size: 1em; color: #555; margin: 20px 0 6px; }
.meta { color: #7f8c8d; font-size: .88em; margin-bottom: 24px; }
.notice { background: #fef9e7; border: 1px solid #f1c40f; border-radius: 8px;
          padding: 14px 18px; margin: 16px 0; font-size: .9em; color: #7d6608; }
.notice strong { display: block; margin-bottom: 4px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
         font-size: .78em; font-weight: 600; color: white; }
.badge-Trend      { background: #3498db; }
.badge-Momentum   { background: #9b59b6; }
.badge-Oscillator { background: #e67e22; }
.badge-BB         { background: #27ae60; }
.badge-Breakout   { background: #e74c3c; }
.badge-Pattern    { background: #1abc9c; }
.badge-Confluence { background: #f39c12; }
.card { background: white; border-radius: 12px; padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-top: 20px; }
table { border-collapse: collapse; width: 100%; font-size: .85em; }
th { background: #2c3e50; color: white; padding: 8px 10px; text-align: left; white-space: nowrap; }
td { padding: 7px 10px; border-bottom: 1px solid #ecf0f1; }
tr:hover td { background: #f8f9fa; }
.rank-1 td { background: #fffde7 !important; font-weight: 600; }
.rank-2 td { background: #f9fbe7 !important; }
.rank-3 td { background: #f1f8e9 !important; }
.pos { color: #27ae60; font-weight: 600; }
.neg { color: #e74c3c; font-weight: 600; }
.neu { color: #2980b9; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media(max-width: 700px) { .chart-grid { grid-template-columns: 1fr; } }
footer { text-align: center; color: #bdc3c7; font-size: .8em;
         margin-top: 48px; padding-top: 16px; border-top: 1px solid #ecf0f1; }
.toc a { display: block; padding: 3px 0; color: #2980b9; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
</style>
"""


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    "Trend": "#3498db", "Momentum": "#9b59b6", "Oscillator": "#e67e22",
    "BB": "#27ae60", "Breakout": "#e74c3c", "Pattern": "#1abc9c", "Confluence": "#f39c12",
}


def _equity_curve(trades: list[Trade], strat_id: str, price: pd.DataFrame) -> str:
    if not trades:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No trades generated", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(strat_id)
        return _b64(fig)

    # Build equity curve
    times = [price["time"].iloc[0]] + [t.exit_time for t in trades]
    pnls  = [0.0] + [t.pnl_pct for t in trades]
    cumulative = list(pd.Series(pnls).cumsum())

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, cumulative, linewidth=1.5, color="#3498db")
    ax.fill_between(times, cumulative, 0,
                    where=[c >= 0 for c in cumulative], alpha=0.15, color="#27ae60")
    ax.fill_between(times, cumulative, 0,
                    where=[c < 0 for c in cumulative], alpha=0.15, color="#e74c3c")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"{strat_id} — Cumulative P&L %", fontsize=10)
    ax.set_ylabel("Cum. P&L %")
    fig.tight_layout()
    return _b64(fig)


def _comparison_chart(scored: pd.DataFrame) -> str:
    top = scored.head(10)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    colors = [GROUP_COLORS.get(g, "#888") for g in top["group"]]

    # Score
    axes[0].barh(top.index[::-1], top["score"][::-1], color=colors[::-1])
    axes[0].set_title("Composite Score (top 10)")
    axes[0].set_xlabel("Score")

    # Win rate
    axes[1].barh(top.index[::-1], top["win_rate"][::-1], color=colors[::-1])
    axes[1].axvline(0.5, color="gray", linewidth=1, linestyle="--")
    axes[1].set_title("Win Rate")
    axes[1].set_xlabel("Win Rate")

    # Net P&L %
    pnl_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in top["net_pnl_pct"][::-1]]
    axes[2].barh(top.index[::-1], top["net_pnl_pct"][::-1], color=pnl_colors)
    axes[2].axvline(0, color="gray", linewidth=1, linestyle="--")
    axes[2].set_title("Net P&L %")
    axes[2].set_xlabel("P&L %")

    fig.tight_layout()
    return _b64(fig)


def _signal_frequency_chart(results: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [GROUP_COLORS.get(g, "#888") for g in results["group"]]
    ax.bar(results.index, results["total"], color=colors)
    ax.set_title("Trade Count per Strategy (30-min data window)")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("# Trades")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return _b64(fig)


def _top5_equity_grid(scored: pd.DataFrame, trades_map: dict, price: pd.DataFrame) -> str:
    top5 = scored.head(5).index.tolist()
    parts = []
    for sid in top5:
        parts.append(_img(_equity_curve(trades_map[sid], sid, price)))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Table renderers
# ---------------------------------------------------------------------------

def _results_table(scored: pd.DataFrame) -> str:
    rows = []
    for rank, (sid, row) in enumerate(scored.iterrows(), 1):
        cls = f"rank-{rank}" if rank <= 3 else ""
        pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] < 99 else "∞"
        wr_cls = "pos" if row["win_rate"] >= 0.5 else "neg"
        pnl_cls = "pos" if row["net_pnl_pct"] >= 0 else "neg"
        badge = f'<span class="badge badge-{row["group"]}">{row["group"]}</span>'
        rows.append(f"""
        <tr class="{cls}">
          <td>#{rank}</td>
          <td><strong>{sid}</strong></td>
          <td>{badge}</td>
          <td style="font-size:.8em;max-width:260px">{row["description"]}</td>
          <td class="{wr_cls}">{row['win_rate']:.1%}</td>
          <td>{row['total']:.0f}</td>
          <td>{pf_str}</td>
          <td class="{pnl_cls}">{row['net_pnl_pct']:.2f}%</td>
          <td>{row['avg_hold_bars']:.1f}</td>
          <td>{row['max_consec_loss']:.0f}</td>
          <td class="neu"><strong>{row['score']:.3f}</strong></td>
        </tr>""")

    return f"""
    <table>
      <thead>
        <tr>
          <th>#</th><th>Strategy</th><th>Group</th><th>Description</th>
          <th>Win Rate</th><th>Trades</th><th>Prof.Factor</th>
          <th>Net P&L%</th><th>Avg Hold</th><th>Max CL</th><th>Score</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _trade_detail_table(trades: list[Trade]) -> str:
    if not trades:
        return "<p style='color:gray'>No trades in this window.</p>"
    rows = []
    cum = 0.0
    for t in trades:
        cum += t.pnl_pct
        r_cls = "pos" if t.result == "win" else "neg"
        rows.append(f"""
        <tr>
          <td>{t.entry_time.strftime('%m-%d %H:%M')}</td>
          <td>{t.exit_time.strftime('%m-%d %H:%M') if t.exit_time else '-'}</td>
          <td>{t.entry_price:.2f}</td>
          <td>{t.exit_price:.2f}</td>
          <td>{t.exit_reason}</td>
          <td>{t.hold_bars}</td>
          <td class="{r_cls}">{t.pnl_pct:+.2f}%</td>
          <td>{cum:+.2f}%</td>
        </tr>""")
    return f"""
    <table>
      <thead><tr>
        <th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th>
        <th>Reason</th><th>Bars</th><th>P&L%</th><th>Cum%</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def _mtf_filter_table(comparison: pd.DataFrame) -> str:
    """Render a before/after MTF filter comparison table."""
    rows = []
    for sid, row in comparison.iterrows():
        delta = row["win_rate_filtered"] - row["win_rate_base"]
        delta_str = f"{'▲' if delta > 0 else '▼'} {abs(delta):.1%}"
        delta_cls = "pos" if delta > 0 else "neg"
        trade_drop = row["trades_base"] - row["trades_filtered"]
        rows.append(f"""
        <tr>
          <td><strong>{sid}</strong></td>
          <td>{row['win_rate_base']:.1%}</td>
          <td>{row['win_rate_filtered']:.1%}</td>
          <td class="{delta_cls}">{delta_str}</td>
          <td>{row['trades_base']:.0f}</td>
          <td>{row['trades_filtered']:.0f}</td>
          <td style="color:#888">-{trade_drop:.0f}</td>
          <td>{row['pf_base']:.2f}</td>
          <td>{row['pf_filtered']:.2f}</td>
        </tr>""")
    return f"""
    <table>
      <thead><tr>
        <th>Strategy</th>
        <th>WR (base)</th><th>WR (filtered)</th><th>Δ Win Rate</th>
        <th>Trades (base)</th><th>Trades (filtered)</th><th>Reduction</th>
        <th>PF (base)</th><th>PF (filtered)</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _mtf_state_table(mtf_by_state: pd.DataFrame, direction: str = "long") -> str:
    """Render win rate by 4H state for experiment trades."""
    if mtf_by_state.empty:
        return "<p style='color:gray'>No 4H data available.</p>"
    favorable = "bearish" if direction == "short" else "bullish"
    rows = []
    for state, row in mtf_by_state.iterrows():
        is_fav = str(state).startswith(favorable)
        style = " font-weight:600;background:#f1f8e9" if is_fav else ""
        rows.append(f"""
        <tr style="{style}">
          <td>{state}</td>
          <td>{row['total']}</td>
          <td>{row['wins']}</td>
          <td>{row['win_rate']:.1%}</td>
        </tr>""")
    label = "4H bearish = aligned (short)" if direction == "short" else "4H bullish = aligned (long)"
    return f"""
    <p style="font-size:.85em;color:#666">{label} — highlighted rows are favorable HTF conditions.</p>
    <table style="max-width:400px">
      <thead><tr><th>4H State</th><th>Trades</th><th>Wins</th><th>Win Rate</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _mtf_per_strategy_html(
    trades_map: dict,
    price_4h: pd.DataFrame | None,
    price_1d: pd.DataFrame | None,
    direction: str = "long",
    scored: pd.DataFrame | None = None,
) -> str:
    """Build per-strategy MTF analysis cards (4H state win rate for top-10 strategies)."""
    try:
        from analysis.mtf_analysis import enrich_trades_with_htf, htf_stats, trades_to_df
    except ImportError:
        return ""

    if price_4h is None and price_1d is None:
        return ""

    order = list(scored.index) if scored is not None else list(trades_map.keys())
    parts = []
    for sid in order[:10]:
        tlist = trades_map.get(sid, [])
        if not tlist:
            continue
        tdf = trades_to_df(tlist, strategy_id=sid)
        enriched = enrich_trades_with_htf(tdf, price_4h=price_4h, price_1d=price_1d)
        stats = htf_stats(enriched)
        state_df = stats.get("by_4h_state", pd.DataFrame())
        align_df = stats.get("by_alignment", pd.DataFrame())
        table_html = _mtf_state_table(state_df, direction=direction)
        align_html = ""
        if not align_df.empty:
            rows = []
            for lbl, row in align_df.iterrows():
                rows.append(f"<tr><td>{lbl}</td><td>{row['total']}</td>"
                            f"<td>{row['win_rate']:.1%}</td></tr>")
            align_html = (
                "<p style='font-size:.85em;color:#666;margin-top:12px'>"
                "Alignment score (0–3 TFs in favorable direction):</p>"
                "<table style='max-width:360px'>"
                "<thead><tr><th>Alignment</th><th>Trades</th><th>Win Rate</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )
        parts.append(f"""
        <div style="margin-bottom:20px;padding:14px;background:#f9f9f9;border-radius:8px">
          <strong>{sid}</strong>
          {table_html}
          {align_html}
        </div>""")

    return "".join(parts)


def generate(
    scored: pd.DataFrame,
    trades_map: dict[str, list[Trade]],
    price: pd.DataFrame,
    out_path: Path,
    mtf_comparison: pd.DataFrame | None = None,
    price_4h: pd.DataFrame | None = None,
    price_1d: pd.DataFrame | None = None,
    direction: str = "long",
) -> None:
    generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_start    = price["time"].min().strftime("%Y-%m-%d")
    data_end      = price["time"].max().strftime("%Y-%m-%d")
    data_bars     = len(price)
    best_id       = scored.index[0]
    best_row      = scored.iloc[0]

    side = direction  # "long" or "short"

    # ── Charts ──────────────────────────────────────────────────────────────
    comp_b64  = _comparison_chart(scored)
    freq_b64  = _signal_frequency_chart(scored)

    # Top-5 equity curves
    top5_ids = scored.head(5).index.tolist()

    # ── MTF sections ────────────────────────────────────────────────────────
    mtf_filter_section = ""
    if mtf_comparison is not None and not mtf_comparison.empty:
        avg_delta = (mtf_comparison["win_rate_filtered"] - mtf_comparison["win_rate_base"]).mean()
        improved  = (mtf_comparison["win_rate_filtered"] > mtf_comparison["win_rate_base"]).sum()
        avg_trade_drop_pct = (
            1 - mtf_comparison["trades_filtered"].sum() / mtf_comparison["trades_base"].sum()
        ) * 100
        filter_type = "4H not bullish (skip counter-trend shorts)" if side == "short" else "4H not bearish (skip counter-trend longs)"
        mtf_filter_section = f"""
  <h2 id="mtf-filter">MTF Filter: Before vs After ({filter_type})</h2>
  <div class="card">
    <div class="notice" style="margin-bottom:16px">
      <strong>4H RSI Filter Applied</strong>
      Entries blocked when the 4H RSI state is
      {'bullish (counter-trend for shorts)' if side == 'short' else 'bearish (counter-trend for longs)'}.
      Avg win-rate change: <strong>{avg_delta:+.1%}</strong> across all strategies &nbsp;|&nbsp;
      {improved}/{len(mtf_comparison)} strategies improved &nbsp;|&nbsp;
      Trade count reduced by ~{avg_trade_drop_pct:.0f}%
    </div>
    {_mtf_filter_table(mtf_comparison)}
  </div>"""

    mtf_analysis_section = ""
    if price_4h is not None:
        per_strat_html = _mtf_per_strategy_html(
            trades_map, price_4h, price_1d, direction=side, scored=scored
        )
        if per_strat_html:
            direction_label = "4H bearish = aligned condition for shorts" if side == "short" else "4H bullish = aligned condition for longs"
            mtf_analysis_section = f"""
  <h2 id="mtf-analysis">MTF Win-Rate Analysis (Top-10 Strategies)</h2>
  <div class="card">
    <p style="color:#666;font-size:.9em">
      Each strategy's win rate broken down by 4H RSI state and alignment score.
      <strong>{direction_label}</strong>.
    </p>
    {per_strat_html}
  </div>"""

    # Per-strategy detail sections
    detail_sections = []
    for sid, row in scored.iterrows():
        trades = trades_map[sid]
        eq_b64 = _equity_curve(trades, sid, price)
        detail_sections.append(f"""
        <div class="card" id="{sid}">
          <h3>{sid}
            <span class="badge badge-{row['group']}">{row['group']}</span>
            &nbsp;Score: <strong>{row['score']:.3f}</strong>
          </h3>
          <p style="color:#666;font-size:.88em">{row['description']}</p>
          {_img(eq_b64)}
          {_trade_detail_table(trades)}
        </div>
        """)

    # TOC
    toc_links = "".join(
        f'<a href="#{sid}">#{rank} {sid}</a>'
        for rank, sid in enumerate(scored.index, 1)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>XAUUSD 20-Strategy Experiment Report</title>
  {CSS}
</head>
<body>
<div style="padding:10px 20px 0"><a href="../../index.html" style="color:#2563eb;font-size:13px;text-decoration:none;background:#dbeafe;padding:4px 12px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="wrap">

  <h1>XAUUSD 20-Strategy Experiment</h1>
  <div class="meta">
    Generated {generated_at} &nbsp;|&nbsp;
    Data: {data_start} → {data_end} ({data_bars} × 30-min bars) &nbsp;|&nbsp;
    20 long-only strategies &nbsp;|&nbsp; SL 0.5% / TP 1.0% / Max hold 48 bars
  </div>

  <div class="notice">
    <strong>⚠ Data Coverage Notice</strong>
    This experiment runs on <strong>{data_bars} bars (~{data_bars//48} trading days)</strong>
    of 30-min OHLCV data. Results with fewer than ~20 trades should be treated as
    directional signals only, not statistically significant conclusions.
    <br><br>
    To validate these strategies properly:
    <ol style="margin:6px 0 0 16px">
      <li>Copy the corresponding Pine Script from <code>XAUUSD-Long-Experiments/pine/</code>
          into TradingView</li>
      <li>Run the backtest on the full XAUUSD 30-min history (2+ years)</li>
      <li>Export the trade CSV and drop it into this project for full analysis</li>
    </ol>
  </div>

  <!-- Winner highlight -->
  <div class="card" style="border-left: 5px solid #f1c40f">
    <h2 style="border:none;padding:0;margin:0 0 8px">
      🏆 Top Strategy: {best_id}
    </h2>
    <p>{best_row['description']}</p>
    <table style="max-width:600px">
      <tr><th>Win Rate</th><th>Trades</th><th>Profit Factor</th>
          <th>Net P&L%</th><th>Max Consec. Loss</th><th>Score</th></tr>
      <tr>
        <td class="{'pos' if best_row['win_rate']>=0.5 else 'neg'}">{best_row['win_rate']:.1%}</td>
        <td>{best_row['total']:.0f}</td>
        <td>{min(best_row['profit_factor'], 99):.2f}</td>
        <td class="{'pos' if best_row['net_pnl_pct']>=0 else 'neg'}">{best_row['net_pnl_pct']:+.2f}%</td>
        <td>{best_row['max_consec_loss']:.0f}</td>
        <td><strong>{best_row['score']:.3f}</strong></td>
      </tr>
    </table>
  </div>

  <h2>Comparison Charts</h2>
  <div class="card">
    {_img(comp_b64, "Comparison")}
    {_img(freq_b64, "Trade Frequency")}
  </div>

  <h2>Full Ranking Table</h2>
  <div class="card">
    {_results_table(scored)}
  </div>

  {mtf_filter_section}

  {mtf_analysis_section}

  <h2>Top-5 Equity Curves</h2>
  <div class="card">
    {"".join(_img(_equity_curve(trades_map[sid], sid, price)) for sid in top5_ids)}
  </div>

  <h2>Per-Strategy Detail</h2>
  <div class="card toc">
    <strong>Jump to strategy:</strong><br>
    {toc_links}
  </div>
  {"".join(detail_sections)}

  <h2>Development Notes</h2>
  <div class="card">
    <h3>Strategy Groups</h3>
    <ul>
      <li><span class="badge badge-Trend">Trend</span> E01–E05:
          Follow the prevailing direction using EMA, MACD, Supertrend, ADX</li>
      <li><span class="badge badge-Oscillator">Oscillator</span> E06–E10:
          Enter on oversold bounces using RSI, Stochastic, CCI, Williams %R</li>
      <li><span class="badge badge-BB">BB</span> E11–E13:
          Bollinger Band mean-reversion and trend-following setups</li>
      <li><span class="badge badge-Breakout">Breakout</span> E14–E16:
          Enter on confirmed price breakouts above recent structure</li>
      <li><span class="badge badge-Pattern">Pattern</span> E17–E19:
          Candle pattern recognition (Hammer, Engulfing, AO Saucer)</li>
      <li><span class="badge badge-Confluence">Confluence</span> E20:
          Multi-factor entry requiring BB + RSI + Trend alignment</li>
    </ul>

    <h3>Scoring Methodology</h3>
    <p>Composite score = Win Rate (25%) + Profit Factor capped@3 (25%) +
       log(Trade Count) (20%) + Net P&L% (20%) + Consec. Loss penalty (10%).
       All components normalised 0–1 before weighting.</p>

    <h3>Risk Parameters (fixed across all strategies)</h3>
    <ul>
      <li>Stop Loss: 0.5% below entry</li>
      <li>Take Profit: 1.0% above entry (2:1 R:R)</li>
      <li>Max hold: 48 bars (24 hours) — exit at close</li>
      <li>One trade at a time, no pyramiding</li>
    </ul>

    <h3>Pine Script Files</h3>
    <p>All 20 strategies have corresponding Pine Script v6 files in
    <code>XAUUSD-Long-Experiments/pine/</code> ready to paste into TradingView
    for full historical backtesting.</p>
  </div>

  <footer>XAUUSD 20-Strategy Experiment &nbsp;·&nbsp; {generated_at}</footer>
</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  HTML report → {out_path}")
