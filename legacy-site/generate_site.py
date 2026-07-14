#!/usr/bin/env python3
"""
generate_site.py — Trading Strategy Hub 靜態多頁生成器（20260711 拆頁重構）
==============================================================================
取代舊的 generate_index.py（單頁版，已於同日刪除；歷史見 git log）。

架構原則（詳見 DEVELOPMENT.md）：
  1. 六個頁面全部由本檔生成，生成的 HTML 檔案一律不得手改
     （每頁開頭都有 DO NOT EDIT banner）。
  2. 內容分三類，各有唯一的修改入口：
     - 動態數據區塊：改 data 檔（results.json 等）→ 重跑本檔
     - 手寫編輯內容：改 content/ 下的 fragment → 重跑本檔
     - 對話記錄：append data/logs.json → 重跑本檔（「共 N 筆」自動計算）
  3. 單頁重生成：python3 generate_site.py --page xauusd（省略 --page 則全部）

頁面清單：
  index.html    Hub 首頁（商品卡 + 最新動態）
  xauusd.html   XAUUSD 主頁（宏觀/已確認/實驗/FVG/筆記驗證/週報/H2）
  tx.html       TX 主頁（宏觀/已確認/實驗/筆記驗證/正二）
  shared.html   跨商品分析
  history.html  對話記錄（全量，計數自動）
  sitemap.html  網站地圖（fragment: content/sitemap.html）
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent
CONTENT = ROOT / "content"
LOGS_PATH = ROOT / "data" / "logs.json"

COMMODITIES = [
    {
        "id":        "xauusd",
        "name":      "XAUUSD 黃金",
        "subtitle":  "黃金/美元 · 30m · 3 策略 + 20L/20S 實驗",
        "color":     "#1e3a5f",
        "accent":    "#2563eb",
        "long_dir":  "xauusd/XAUUSD-Long-Experiments",
        "short_dir": "xauusd/XAUUSD-Short-Experiments",
        "long_pine": "xauusd/XAUUSD-Long-Experiments/pine/ALL_Long_Strategies.pine",
        "short_pine":"xauusd/XAUUSD-Short-Experiments/pine/ALL_Short_Strategies.pine",
    },
    {
        "id":        "tx",
        "name":      "TX 台指期",
        "subtitle":  "小台 MTX · 30m · 20L/20S 實驗 · 宏觀分析",
        "color":     "#1a237e",
        "accent":    "#1565c0",
        "long_dir":  "tx/TX-Long-Experiments",
        "short_dir": "tx/TX-Short-Experiments",
        "long_pine": "tx/TX-Long-Experiments/pine/ALL_Long_Strategies.pine",
        "short_pine":"tx/TX-Short-Experiments/pine/ALL_Short_Strategies.pine",
    },
]

def _load_top3(results_path: Path, n: int = 3) -> list[dict]:
    if not results_path.exists():
        return []
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["results"] if r.get("n_trades", 0) > 0][:n]


def _exp_row_xauusd(r: dict, direction: str) -> str:
    colour = "#1e3a5f" if direction == "long" else "#7c3aed"
    pnl_pct = r.get("net_pnl_pct", 0)
    pnl_col = "var(--green)" if pnl_pct >= 0 else "var(--red)"
    sign = "+" if pnl_pct >= 0 else ""
    return (
        f"<tr>"
        f"<td><b style='color:{colour}'>{r.get('rank','-')}. {r['code']}</b></td>"
        f"<td>{r['name']}</td>"
        f"<td>{r.get('n_trades', '-')}</td>"
        f"<td class='{'pos' if r['win_rate']>=45 else 'neutral'}'>{r['win_rate']}%</td>"
        f"<td>{r['profit_factor']:.3f}</td>"
        f"<td style='color:{pnl_col}'>{sign}{pnl_pct:.1f}%</td>"
        f"</tr>"
    )


def _exp_row_tx(r: dict, direction: str) -> str:
    colour = "#1565c0" if direction == "long" else "#880e4f"
    pnl = r.get("net_pnl_ntd", 0)
    pnl_col = "#2e7d32" if pnl >= 0 else "#c62828"
    return (
        f"<tr>"
        f"<td style='color:{colour};font-weight:bold'>{r.get('rank','-')}. {r['code']}</td>"
        f"<td>{r['name']}</td>"
        f"<td>{r['win_rate']}%</td>"
        f"<td>{r['profit_factor']:.3f}</td>"
        f"<td style='color:{pnl_col}'>NT${pnl:,.0f}</td>"
        f"<td>{r.get('score',0):.3f}</td>"
        f"</tr>"
    )

def _xauusd_macro_html() -> str:
    csv_path = ROOT / "xauusd/csv/20260711/FX_IDC_XAUUSD, 1W.csv"  # 20260711 重匯到最新日期
    if not csv_path.exists():
        return '<div id="xauusd-main-macro" class="main-section"><div class="tab-panel active"><p style="padding:24px;color:var(--muted)">週線 CSV 未找到</p></div></div>'

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['time'] = df['time'].apply(lambda t: pd.to_datetime(re.sub(r'[+-]\d{2}:\d{2}$', '', str(t).strip())))
    df = df.sort_values('time').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open']  = pd.to_numeric(df['open'],  errors='coerce')
    # 20260711 起匯出可能是純OHLC（無RSI欄位）；rsi_end 目前無下游消費者，缺欄位時填NaN即可
    df['RSI']   = pd.to_numeric(df['RSI'], errors='coerce') if 'RSI' in df.columns else float('nan')
    df['year']  = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['week_of_month'] = df.groupby(['year', 'month']).cumcount() + 1
    df = df[df['year'] >= 1980].copy()

    # Monthly aggregation
    grp = df.groupby(['year', 'month'])
    mon = grp.agg(m_open=('open','first'), m_close=('close','last'), rsi_end=('RSI','last')).reset_index()
    mon['chg_pct']  = (mon['m_close'] - mon['m_open']) / mon['m_open'] * 100
    mon['chg_usd']  = mon['m_close'] - mon['m_open']
    mon['bullish']  = mon['chg_pct'] > 0
    mon['date']     = pd.to_datetime(mon[['year','month']].assign(day=1))

    total_months = len(mon)
    overall_wr   = mon['bullish'].mean() * 100
    avg_pct      = mon['chg_pct'].mean()
    latest       = mon.iloc[-1]
    cur_month    = int(latest['month'])
    cur_year     = int(latest['year'])
    month_names  = ['一月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月']

    # Seasonality per month
    sea_rows = ""
    cur_note = ""
    for m in range(1, 13):
        sub = mon[mon['month'] == m]
        if len(sub) == 0:
            continue
        wr      = sub['bullish'].mean() * 100
        avg_p   = sub['chg_pct'].mean()
        avg_u   = sub['chg_usd'].mean()
        bias    = 'LONG' if wr >= 55 else ('SHORT' if wr <= 45 else 'NEUTRAL')
        bc      = 'var(--green)' if bias == 'LONG' else ('var(--red)' if bias == 'SHORT' else 'var(--muted)')
        bl      = '偏多' if bias == 'LONG' else ('偏空' if bias == 'SHORT' else '中性')
        wc      = 'var(--green)' if wr >= 55 else ('var(--red)' if wr <= 45 else 'var(--muted)')
        badge_cls = 'badge-green' if bias == 'LONG' else ('badge-red' if bias == 'SHORT' else 'badge-blue')
        sea_rows += (f"<tr><td>{month_names[m-1]}</td><td>{len(sub)}</td>"
                     f"<td style='color:{wc};font-weight:700'>{wr:.1f}%</td>"
                     f"<td>{'▲' if avg_p>=0 else '▼'} {abs(avg_p):.2f}% (${avg_u:+.0f})</td>"
                     f"<td><span class='badge {badge_cls}'>{bl}</span></td></tr>\n")
        if m == cur_month:
            cur_note = f"當月（{month_names[m-1]}）：歷史勝率 {wr:.1f}%，平均 {avg_p:+.2f}%（${avg_u:+.0f}）"

    # Week-of-month structure
    df['wk_chg'] = (df['close'] - df['open']) / df['open'] * 100
    df['wk_bull'] = df['wk_chg'] > 0
    wim = df[df['week_of_month'] <= 5].groupby('week_of_month').agg(
        n=('wk_chg','count'), wr=('wk_bull','mean'), avg=('wk_chg','mean')).reset_index()
    wim['wr'] = wim['wr'] * 100
    wlabel = {1:'第1週（月初）',2:'第2週',3:'第3週',4:'第4週',5:'第5週（月底）'}
    wim_rows = ""
    for _, r in wim.iterrows():
        wk = int(r['week_of_month'])
        wc = 'var(--green)' if r['wr'] >= 55 else ('var(--red)' if r['wr'] <= 45 else 'var(--muted)')
        wim_rows += (f"<tr><td><strong>{wlabel[wk]}</strong></td><td>{int(r['n'])}</td>"
                     f"<td style='color:{wc};font-weight:700'>{r['wr']:.1f}%</td>"
                     f"<td>{r['avg']:+.2f}%</td></tr>\n")

    # Recent 12 months
    rec = mon.sort_values('date').tail(12)
    rec_rows = ""
    for _, r in rec.iterrows():
        color = 'var(--green)' if r['bullish'] else 'var(--red)'
        sign  = '▲' if r['bullish'] else '▼'
        rec_rows += (f"<tr><td>{int(r['year'])}/{int(r['month']):02d}</td>"
                     f"<td>${r['m_open']:.0f}</td><td>${r['m_close']:.0f}</td>"
                     f"<td style='color:{color};font-weight:700'>{sign} {r['chg_pct']:+.2f}% (${r['chg_usd']:+.0f})</td></tr>\n")

    wr_color = 'var(--green)' if overall_wr >= 55 else 'var(--red)'

    # Session analysis from validation_results.json
    val_path = ROOT / "doc/validation_results.json"
    session_rows = ""
    if val_path.exists():
        with open(val_path, encoding="utf-8") as vf:
            vdata_local = json.load(vf)
        for s in vdata_local.get("xauusd", {}).get("xauusd_sessions", []):
            trend = s["趨勢K比例"]
            wc = "color:var(--red)" if trend >= 0.55 else "color:var(--green)"
            mv = s["平均波動%"]
            lk = s["漲K勝率"]
            lc = "color:var(--green)" if lk >= 0.52 else "color:var(--muted)"
            session_rows += (
                f"<tr><td><b>{s['時段']}</b></td>"
                f"<td style='{wc};font-weight:700'>{trend:.0%}</td>"
                f"<td>{mv:.3f}%</td>"
                f"<td style='{lc};font-weight:700'>{lk:.0%}</td>"
                f"<td>{s['樣本數']:,}</td>"
                f"<td><small style='color:var(--muted)'>{s['筆記判斷']}</small></td></tr>\n"
            )

    return f"""
  <!-- XAUUSD 宏觀分析 -->
  <div id="xauusd-main-macro" class="main-section active">
    <div class="subnav">
      <button class="sub-tab active" onclick="showTab('xauusd-macro','overview',this)">月度統計 &amp; 季節性</button>
      <button class="sub-tab" onclick="showTab('xauusd-macro','weekly',this)">週內結構</button>
      <button class="sub-tab" onclick="showTab('xauusd-macro','recent',this)">近 12 個月</button>
      <button class="sub-tab" onclick="showTab('xauusd-macro','session',this)">時段分析</button>
    </div>

    <div id="xauusd-macro-overview" class="tab-panel active">
      <div class="part-label"><span class="part-badge">MACRO</span>整體月度統計（1980–{int(latest['year'])}）</div>
      <div class="grid-4">
        <div class="metric-card card"><div class="metric-label">整體月勝率</div><div class="metric-val" style="color:{wr_color}">{overall_wr:.1f}%</div><div class="metric-sub">{total_months} 個月（1980–{int(latest['year'])}）</div></div>
        <div class="metric-card card"><div class="metric-label">平均月漲跌</div><div class="metric-val {'green' if avg_pct>=0 else 'red'}">{avg_pct:+.2f}%</div><div class="metric-sub">月初買、月底賣</div></div>
        <div class="metric-card card"><div class="metric-label">當月（{month_names[cur_month-1]}）</div><div class="metric-val">{cur_year}/{cur_month:02d}</div><div class="metric-sub" style="font-size:.8em">{cur_note}</div></div>
        <div class="metric-card card"><div class="metric-label">週線 K 棒數</div><div class="metric-val">{len(df)}</div><div class="metric-sub">1980–{int(latest['year'])}</div></div>
      </div>

      <div class="insight-grid">
        <div class="insight good"><strong>✅ 強勢月份</strong>一月（63.8%）、七月（56.5%）、十二月（56.5%）歷史偏多。</div>
        <div class="insight bad"><strong>❌ 弱勢月份</strong>二月（31.9%）、六月（39.1%）、三月（42.6%）歷史偏空。</div>
        <div class="insight info"><strong>📊 操作框架</strong>先確認月度季節性偏向，再用週線 RSI / BB 判斷進場時機。</div>
      </div>

      <div class="card">
        <div class="card-title">🗓 季節性偏向 — 每月歷史統計（1980–{int(latest['year'])}）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>月份</th><th>樣本</th><th>月勝率</th><th>平均漲跌</th><th>偏向</th></tr></thead>
            <tbody>{sea_rows}</tbody>
          </table>
        </div>
      </div>
      <div class="report-links">
        <a class="report-link" href="xauusd/macro_report.html">📄 完整宏觀報告（暗色主題 + 熱力圖）</a>
      </div>
    </div>

    <div id="xauusd-macro-weekly" class="tab-panel">
      <div class="part-label"><span class="part-badge">MACRO</span>週內結構 — 每月第幾週最強</div>
      <div class="card">
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>週次</th><th>樣本</th><th>週勝率</th><th>平均漲跌%</th></tr></thead>
            <tbody>{wim_rows}</tbody>
          </table>
        </div>
        <div style="margin-top:8px;font-size:.82em;color:var(--muted)">第3週（56.6%）和第5週（55.6%）勝率最高；第4週最低（48.0%）。</div>
      </div>
    </div>

    <div id="xauusd-macro-recent" class="tab-panel">
      <div class="part-label"><span class="part-badge">MACRO</span>近 12 個月回顧</div>
      <div class="card" style="max-width:620px">
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>年/月</th><th>月初開盤</th><th>月底收盤</th><th>月漲跌</th></tr></thead>
            <tbody>{rec_rows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <div id="xauusd-macro-session" class="tab-panel">
      <div class="part-label"><span class="part-badge">MACRO</span>四個時段特性（台北時間）</div>
      <div class="insight-grid">
        <div class="insight warn"><strong>⚠ 亞盤（9–10）最高波動但偏震盪</strong>趨勢K 60%，漲K勝率 58%，平均波動 0.349%（最大）— 但筆記判斷 90% 震盪，不適合追趨勢。</div>
        <div class="insight good"><strong>✅ 歐盤（20–21）為均衡時段</strong>趨勢K 57%，波動 0.236%，漲K勝率 51.5%——筆記判斷 50/50，策略適用性較廣。</div>
        <div class="insight info"><strong>📊 美盤（23–00）趨勢性強</strong>趨勢K 59%，波動 0.292%，漲K勝率 48.1%——美元加速期間波動放大，除非大趨勢否則偏震盪。</div>
        <div class="insight bad"><strong>❌ 午後（14–15）最弱時段</strong>趨勢K 46%（最低），波動 0.216%（最小）——70% 震盪，歐洲開盤前的淡水期，不建議進場。</div>
      </div>
      <div class="card">
        <div class="card-title">⏱ 四個時段統計（30m K 棒，台北時間，2026-01–04）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>時段</th><th>趨勢K比例</th><th>平均波動</th><th>漲K勝率</th><th>樣本數</th><th>筆記判斷</th></tr></thead>
            <tbody>{session_rows or "<tr><td colspan='6' style='color:#999'>請執行 validate_notes.py</td></tr>"}</tbody>
          </table>
        </div>
        <div style="margin-top:8px;font-size:.8em;color:var(--muted)">趨勢K定義：|ret| &gt; 0.15%。漲K勝率 &gt; 52% 偏多、&lt; 48% 偏空。</div>
      </div>
      <div class="report-links">
        <a class="report-link" href="xauusd/macro_report.html">📄 完整宏觀報告（熱力圖）</a>
      </div>
    </div>
  </div><!-- /xauusd-main-macro -->
"""


# ─── XAUUSD static 現有策略優化 HTML ────────────────────────────────

def _xauusd_exp_html(long_rows: str, short_rows: str, c: dict) -> str:
    long_link  = c["long_dir"] + "/report.html"
    short_link = c["short_dir"] + "/report.html"
    long_pine  = c["long_pine"]
    short_pine = c["short_pine"]
    long_exists  = (ROOT / long_link).exists()
    short_exists = (ROOT / short_link).exists()
    smc_link   = "xauusd/XAUUSD-SMC-Experiments/report.html"
    smc_exists = (ROOT / smc_link).exists()

    long_btn  = f"<a class='report-link' href='{long_link}'>完整報告 →</a>" if long_exists else ""
    short_btn = f"<a class='report-link' href='{short_link}'>完整報告 →</a>" if short_exists else ""
    smc_btn   = f"<a class='report-link' href='{smc_link}'>📄 SMC 完整報告 →</a>" if smc_exists else ""

    return f"""
  <!-- XAUUSD 實驗策略 ──────────────────────────────────── -->
  <div id="xauusd-main-exp" class="main-section">
    <div class="subnav">
      <button class="sub-tab active" onclick="showTab('xauusd-exp','overview',this)">綜合對比</button>
      <button class="sub-tab" onclick="showTab('xauusd-exp','long',this)">多單 Long (E01–E20)</button>
      <button class="sub-tab" onclick="showTab('xauusd-exp','short',this)">空單 Short (S01–S20)</button>
      <button class="sub-tab" onclick="showTab('xauusd-exp','smc',this)">🔍 SMC (M01–M20)</button>
    </div>

    <div id="xauusd-exp-overview" class="tab-panel active">
      <div class="part-label"><span class="part-badge">PART 1</span>精華重點 · Key Findings</div>
      <div class="grid-2">
        <div class="card">
          <div class="card-title">📈 多單 Top-3（E01–E20）</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>#</th><th>策略</th><th>筆數</th><th>勝率</th><th>PF</th><th>淨盈虧%</th></tr></thead>
              <tbody>{long_rows or "<tr><td colspan='6' style='color:#999'>尚無資料（請執行 run_experiments.py）</td></tr>"}</tbody>
            </table>
          </div>
          <div class="report-links">
            {long_btn}
            <a class="report-link" href="{long_pine}">ALL_Long.pine →</a>
          </div>
        </div>
        <div class="card">
          <div class="card-title">📉 空單 Top-3（S01–S20）</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>#</th><th>策略</th><th>筆數</th><th>勝率</th><th>PF</th><th>淨盈虧%</th></tr></thead>
              <tbody>{short_rows or "<tr><td colspan='6' style='color:#999'>尚無資料</td></tr>"}</tbody>
            </table>
          </div>
          <div class="report-links">
            {short_btn}
            <a class="report-link" href="{short_pine}">ALL_Short.pine →</a>
          </div>
        </div>
      </div>

      <div class="insight-grid">
        <div class="insight good"><strong>✅ HTF 4H 過濾器驗證結果</strong>空單過濾（跳過 4H bullish）：<strong>+4.1% ΔWR，16/20 改善</strong><br>多單過濾（跳過 4H bearish）：+1.6% ΔWR，11/20 改善</div>
        <div class="insight good"><strong>✅ BB 策略多空雙向有效</strong>E12 BB Squeeze（多單 #2）和 S12 BB Squeeze（空單 #3）均有效</div>
        <div class="insight warn"><strong>⚠ 此期間空單 &gt; 多單</strong>空單整體表現優於多單（S19 +12.4% vs E03 +9.0%）</div>
        <div class="insight info"><strong>📊 下一步建議</strong>在 TradingView 驗證 E03 + S19；將 HTF 4H 過濾器加入 ALL Pine 版</div>
      </div>

      <div class="part-label"><span class="part-badge">PART 2</span>分析紀錄</div>
      <div class="card"><div class="tbl-wrap">
        <table>
          <thead><tr><th>日期</th><th>重點</th><th>備忘錄</th></tr></thead>
          <tbody>
            <tr><td class="td-date">2026-05-01</td><td><span class="tag tag-analysis">分析</span> HTF 4H 過濾器加入實驗引擎</td><td class="td-memo"><strong>空單 +4.1% ΔWR（16/20 改善）</strong></td></tr>
            <tr><td class="td-date">2026-04-29</td><td><span class="tag tag-pine">Pine</span> 合併 Pine Script（下拉選單）</td><td class="td-memo">TradingView 一個腳本切換全部 20 策略</td></tr>
            <tr><td class="td-date">2026-04-27</td><td><span class="tag tag-new">新建</span> 20 多單 + 20 空單策略回測框架</td><td class="td-memo"><strong>E03 MACD Signal 最佳（PF 1.643）</strong>；S19 Bearish Engulf 空單最佳（+12.4%）</td></tr>
          </tbody>
        </table>
      </div></div>
    </div><!-- /exp-overview -->

    <div id="xauusd-exp-long" class="tab-panel">
      <div class="part-label"><span class="part-badge">PART 1</span>多單實驗精華重點</div>
      <div class="grid-4">
        <div class="metric-card card"><div class="metric-label">最佳策略</div><div class="metric-val" style="font-size:1em">E03 MACD Signal</div></div>
        <div class="metric-card card"><div class="metric-label">最佳 PF</div><div class="metric-val green">1.643</div></div>
        <div class="metric-card card"><div class="metric-label">最佳淨盈虧%</div><div class="metric-val green">+9.0%</div></div>
        <div class="metric-card card"><div class="metric-label">HTF 過濾後 ΔWR</div><div class="metric-val green">+1.6%</div><div class="metric-sub">11/20 改善</div></div>
      </div>
      <div class="report-links">
        <a class="report-link" href="{long_link}">📄 完整報告</a>
        <a class="report-link" href="{long_pine}">📋 ALL_Long_Strategies.pine</a>
      </div>
    </div>

    <div id="xauusd-exp-short" class="tab-panel">
      <div class="part-label"><span class="part-badge">PART 1</span>空單實驗精華重點</div>
      <div class="grid-4">
        <div class="metric-card card"><div class="metric-label">最佳策略</div><div class="metric-val" style="font-size:1em">S19 Bearish Engulf</div></div>
        <div class="metric-card card"><div class="metric-label">最佳 PF</div><div class="metric-val green">1.507</div></div>
        <div class="metric-card card"><div class="metric-label">最佳淨盈虧%</div><div class="metric-val green">+12.4%</div></div>
        <div class="metric-card card"><div class="metric-label">HTF 過濾後 ΔWR</div><div class="metric-val green">+4.1%</div><div class="metric-sub">16/20 改善</div></div>
      </div>
      <div class="report-links">
        <a class="report-link" href="{short_link}">📄 完整報告</a>
        <a class="report-link" href="{short_pine}">📋 ALL_Short_Strategies.pine</a>
      </div>
    </div>

    <div id="xauusd-exp-smc" class="tab-panel">
      <div class="part-label"><span class="part-badge">SMC</span>Smart Money Concepts 實驗（M01–M20）</div>
      <div class="grid-4">
        <div class="metric-card card"><div class="metric-label">最強空單</div><div class="metric-val" style="font-size:1em">M12 Bearish FVG+RSI</div></div>
        <div class="metric-card card"><div class="metric-label">M12 勝率 / PF</div><div class="metric-val green">43.0% / 1.470</div></div>
        <div class="metric-card card"><div class="metric-label">最強多單</div><div class="metric-val" style="font-size:1em">M09 FVG+OB</div></div>
        <div class="metric-card card"><div class="metric-label">M09 勝率 / PF</div><div class="metric-val" style="color:#e67e22">34.0% / 1.045</div></div>
      </div>
      <div class="insight-grid" style="margin-top:12px">
        <div class="insight good"><strong>✅ 空單 FVG 最有效</strong>M12（Bearish FVG+RSI 35-70）：158 筆，統計最可信；M17（BSL sweep+FVG）：50% WR 但僅 20 筆</div>
        <div class="insight warn"><strong>⚠ 多單 SL 太緊</strong>SL 0.5% 對 FVG 進場不合適，FVG 自然 SL 應為 0.8-1.0%；建議重跑驗證</div>
        <div class="insight info"><strong>📌 整合方式</strong>SMC 不替換 S1/S2，只做 S2 進場分級：A+（+SSL sweep+OB）/ A（+任一SMC）/ B（純錘頭）</div>
        <div class="insight info"><strong>📋 資料期限制</strong>僅 3 個月 30m 資料（2026-01-21→04-27），建議取 2 年 4H CSV 重跑確認</div>
      </div>
      <div class="report-links" style="margin-top:16px">
        {smc_btn}
      </div>
    </div>
  </div><!-- /xauusd-main-exp -->
"""


def _fvg_60m_card(res_60m, report_60m_btn) -> str:
    if not res_60m:
        return ""
    lo = res_60m.get('Long', {}).get('stats', {})
    sh = res_60m.get('Short', {}).get('stats', {})
    if not lo:
        return ""
    return f"""
      <div class="card" style="border-left:3px solid #f0c040">
        <div class="card-title">⚡ 60m 最佳化回測（2025-10 → 2026-04，7個月 3058 bars，12,288 參數組合）</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
          <div>
            <div style="font-size:.8rem;font-weight:700;color:#26a69a;margin-bottom:6px">▲ Long 多單</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
              <div class="metric-card"><div class="metric-label">WR</div><div class="metric-val green">{lo.get('win_rate',0)}%</div></div>
              <div class="metric-card"><div class="metric-label">PF</div><div class="metric-val green">{lo.get('profit_factor',0):.3f}</div></div>
              <div class="metric-card"><div class="metric-label">淨損益</div><div class="metric-val pos">+{lo.get('net_pnl_pct',0):.2f}%</div></div>
              <div class="metric-card"><div class="metric-label">筆數</div><div class="metric-val">{lo.get('trades',0)}</div></div>
            </div>
            <div style="margin-top:8px;font-size:.75em;color:var(--muted)">FVG Min 0.30% · Max 20bars · SL Fixed 2.0% · TP1 0.5R / TP2 2.0R · TB 48h</div>
          </div>
          <div>
            <div style="font-size:.8rem;font-weight:700;color:#ef5350;margin-bottom:6px">▼ Short 空單</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
              <div class="metric-card"><div class="metric-label">WR</div><div class="metric-val green">{sh.get('win_rate',0)}%</div></div>
              <div class="metric-card"><div class="metric-label">PF</div><div class="metric-val green">{sh.get('profit_factor',0):.3f}</div></div>
              <div class="metric-card"><div class="metric-label">淨損益</div><div class="metric-val pos">+{sh.get('net_pnl_pct',0):.2f}%</div></div>
              <div class="metric-card"><div class="metric-label">筆數</div><div class="metric-val">{sh.get('trades',0)}</div></div>
            </div>
            <div style="margin-top:8px;font-size:.75em;color:var(--muted)">FVG Min 0.30% · Max 10bars · SL Fixed 1.0% · TP1 0.5R / TP2 4.0R · TB 48h</div>
          </div>
        </div>
        <div class="report-links" style="margin-top:12px">{report_60m_btn}</div>
      </div>"""


def _xauusd_fvg_html() -> str:
    results_path  = ROOT / "xauusd/XAUUSD-FVG-Strategy/fvg_long_results.json"
    opt_path      = ROOT / "xauusd/XAUUSD-FVG-Strategy/optimization_results.json"
    results_60m_path = ROOT / "xauusd/XAUUSD-FVG-Strategy/fvg_60m_results.json"
    report_link   = "xauusd/XAUUSD-FVG-Strategy/report_fvg_long.html"
    report_60m_link = "xauusd/XAUUSD-FVG-Strategy/report_fvg_60m.html"
    report_exists = (ROOT / report_link).exists()
    report_btn = f"<a class='report-link' href='{report_link}'>📄 30m 多單報告 →</a>" if report_exists else ""
    report_60m_btn = f"<a class='report-link' href='{report_60m_link}'>📄 60m 多空報告 →</a>" if (ROOT / report_60m_link).exists() else ""

    if not results_path.exists():
        return """
  <div id="xauusd-main-fvg" class="main-section">
    <div class="tab-panel active"><div class="card">
      <div class="card-title">FVG 策略</div>
      <p style="color:var(--muted)">請先執行 <code>python3 xauusd/scripts/run_fvg_experiments.py</code> 生成資料。</p>
    </div></div>
  </div>"""

    with open(results_path, encoding="utf-8") as f:
        r = json.load(f)

    best_short = None
    if opt_path.exists():
        with open(opt_path, encoding="utf-8") as f:
            opt = json.load(f)
        best_short = next((x for x in opt if x.get('direction') == 'Short'), None)

    res_60m = None
    if results_60m_path.exists():
        with open(results_60m_path, encoding="utf-8") as f:
            d60 = json.load(f)
        res_60m = d60.get('results', {})

    ex = r.get('exit_breakdown', {})
    ex_html = "".join(
        f"<div style='background:var(--surface2);border-radius:6px;padding:10px 16px;text-align:center'>"
        f"<div style='font-size:1.3rem;font-weight:700;color:var(--primary)'>{v}</div>"
        f"<div style='font-size:.75rem;color:var(--muted);margin-top:2px'>{k}</div></div>"
        for k, v in ex.items()
    )

    short_card = ""
    if best_short:
        short_card = f"""
      <div class="card">
        <div class="card-title">📉 空單最佳參數（優化結果）</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px">
          <div class="metric-card"><div class="metric-label">WR</div><div class="metric-val green">{best_short['win_rate']}%</div></div>
          <div class="metric-card"><div class="metric-label">PF</div><div class="metric-val green">{best_short['profit_factor']:.3f}</div></div>
          <div class="metric-card"><div class="metric-label">淨損益</div><div class="metric-val pos">+{best_short['net_pnl_pct']:.2f}%</div></div>
          <div class="metric-card"><div class="metric-label">交易筆數</div><div class="metric-val">{best_short['trades']}</div></div>
        </div>
        <div style="margin-top:12px;font-size:.83em;color:var(--muted)">
          fvg_min={best_short['fvg_min_pct']}% · fvg_max={best_short['fvg_max_bars']}bars · SL={best_short['sl_pct']}% · TP1={best_short['tp1_r']}R · TP2={best_short['tp2_r']}R · TB={best_short['tb']}bars
        </div>
      </div>"""

    return f"""
  <!-- XAUUSD FVG 策略 ────────────────────────────────────── -->
  <div id="xauusd-main-fvg" class="main-section">
    <div class="subnav">
      <button class="sub-tab active" onclick="showTab('xauusd-fvg','overview',this)">回測總覽</button>
      <button class="sub-tab" onclick="showTab('xauusd-fvg','params',this)">參數說明</button>
    </div>

    <div id="xauusd-fvg-overview" class="tab-panel active">
      <div class="part-label"><span class="part-badge">FVG</span>Fair Value Gap 策略 V2.0 · 多 FVG 陣列追蹤</div>

      <div class="grid-4">
        <div class="metric-card card"><div class="metric-label">總交易數</div><div class="metric-val" style="color:var(--primary)">{r['trades']}</div><div class="metric-sub">30m 2026-01–04</div></div>
        <div class="metric-card card"><div class="metric-label">勝率</div><div class="metric-val {'green' if r['win_rate']>=55 else 'red'}">{r['win_rate']}%</div><div class="metric-sub">{r['wins']}W / {r['losses']}L</div></div>
        <div class="metric-card card"><div class="metric-label">獲利因子</div><div class="metric-val {'green' if r['profit_factor']>=1.5 else 'red'}">{r['profit_factor']}</div><div class="metric-sub">優化參數</div></div>
        <div class="metric-card card"><div class="metric-label">淨損益</div><div class="metric-val {'pos' if r['net_pnl_pct']>0 else 'neg'}">{r['net_pnl_pct']:+.2f}%</div><div class="metric-sub">多單 Long</div></div>
        <div class="metric-card card"><div class="metric-label">平均獲利</div><div class="metric-val pos">{r['avg_win']:+.3f}%</div></div>
        <div class="metric-card card"><div class="metric-label">平均虧損</div><div class="metric-val neg">{r['avg_loss']:+.3f}%</div></div>
        <div class="metric-card card"><div class="metric-label">最大單筆獲利</div><div class="metric-val pos">{r['max_win']:+.3f}%</div></div>
        <div class="metric-card card"><div class="metric-label">平均持倉 Bars</div><div class="metric-val">{r['avg_bars']}</div></div>
      </div>

      <div class="insight-grid">
        <div class="insight good"><strong>✅ WR 66.7%，PF 1.66</strong>42 筆多單，統計意義穩定，優於 S1/S2 的 44–53% WR。</div>
        <div class="insight good"><strong>✅ FVG 作為 S2 補充進場工具</strong>FVG 提供明確的 SL 錨點（FVG Natural）+ 流動性缺口回填邏輯，與 S2-Hammer 蜂鳥錘互補。</div>
        <div class="insight warn"><strong>⚠ 資料期間僅 3 個月</strong>2026-01-21 至 04-27，建議取更長 CSV（2年+）重跑確認穩定性。</div>
        <div class="insight info"><strong>📊 SL=1.5% Fixed 最佳</strong>FVG Natural SL 在此數據集表現不如 Fixed %。TP1=0.5R 快速鎖利後讓利潤飛。</div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">📈 多單最佳參數（Python 優化）</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div class="metric-card"><div class="metric-label">FVG 最小大小</div><div class="metric-val" style="font-size:1.1em">0.20%</div></div>
            <div class="metric-card"><div class="metric-label">FVG 最長有效</div><div class="metric-val" style="font-size:1.1em">20 bars</div></div>
            <div class="metric-card"><div class="metric-label">止損</div><div class="metric-val" style="font-size:1.1em">1.5%</div></div>
            <div class="metric-card"><div class="metric-label">TP1 / TP2</div><div class="metric-val" style="font-size:1.1em">0.5R / 2.0R</div></div>
            <div class="metric-card"><div class="metric-label">時間止損</div><div class="metric-val" style="font-size:1.1em">72 bars</div></div>
            <div class="metric-card"><div class="metric-label">SL 類型</div><div class="metric-val" style="font-size:1.1em">Fixed %</div></div>
          </div>
          <div class="report-links">{report_btn}
            <a class="report-link" href="xauusd/XAUUSD-FVG-Strategy/fvg_guide.html">📖 FVG 教學說明</a>
            <a class="report-link" href="xauusd/XAUUSD-FVG-Strategy/XAUUSD-FVG-V2.0.pine">📋 FVG V2.0 Pine</a>
          </div>
        </div>
        {short_card}
      </div>
      {_fvg_60m_card(res_60m, report_60m_btn)}

      <div class="card">
        <div class="card-title">🚪 出場類型分佈</div>
        <div style="display:flex;gap:12px;flex-wrap:wrap">{ex_html}</div>
        <div style="margin-top:10px;font-size:.82em;color:var(--muted)">SL 出場佔多數屬正常（Profit Flyer 允許在 TP2 前止損），TimeWin = 時間到且獲利、TimeLoss = 時間到但虧損。</div>
      </div>
    </div><!-- /xauusd-fvg-overview -->

    <div id="xauusd-fvg-params" class="tab-panel">
      <div class="part-label"><span class="part-badge">FVG</span>FVG V2.0 策略架構說明</div>

      <div class="insight-grid">
        <div class="insight info"><strong>📐 FVG 定義</strong>三根 K 棒型態：多頭 FVG = low[0] &gt; high[2]（缺口）；空頭 FVG = high[0] &lt; low[2]。缺口大小 &gt; fvg_min_pct 才記錄。</div>
        <div class="insight info"><strong>📦 多 FVG 陣列追蹤（V2）</strong>同時最多 max_fvg 個 FVG，最新優先。超過 fvg_max_bars 未觸及自動到期；收盤跌破底部視為無效。</div>
        <div class="insight info"><strong>🎯 進場條件</strong>價格 wick 進入 FVG 區間（low ≤ top 且 close ≥ bot），下一根開盤進場。每次只進一筆，進場後移除此 FVG。</div>
        <div class="insight info"><strong>📤 Profit Flyer 出場（S1架構）</strong>TP1 鎖利：SL 拉至 lowest(low, TB/2) 動態追蹤。TP2 後更緊追蹤 lowest(low, TB/4)。</div>
      </div>

      <div class="card">
        <div class="card-title">🔗 Pine Script 版本比較</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>版本</th><th>FVG 追蹤</th><th>視覺化</th><th>SL 類型</th><th>出場</th></tr></thead>
            <tbody>
              <tr><td>V1.0</td><td>單一 FVG（var float）</td><td>簡單 box</td><td>FVG Natural / Fixed %</td><td>Profit Flyer</td></tr>
              <tr><td>V2.0</td><td>多 FVG 陣列（最多 5 個）</td><td>多 box 自動延伸右邊界</td><td>FVG Natural / Fixed %</td><td>Profit Flyer</td></tr>
              <tr><td><b>V2.1 ★</b></td><td><b>雙陣列（Bull + Bear 同時顯示）</b></td><td><b>支撐綠 / 壓力紅，填充+邊框分離透明度</b></td><td><b>FVG Natural / Fixed %</b></td><td><b>Profit Flyer</b></td></tr>
            </tbody>
          </table>
        </div>
        <div class="report-links">
          <a class="report-link" href="xauusd/XAUUSD-FVG-Strategy/XAUUSD-FVG-V2.1.pine">📋 V2.1 Pine Script ★</a>
          <a class="report-link" href="xauusd/XAUUSD-FVG-Strategy/XAUUSD-FVG-V2.0.pine">📋 V2.0 Pine Script</a>
          <a class="report-link" href="xauusd/XAUUSD-FVG-Strategy/XAUUSD-FVG-V1.0.pine">📋 V1.0 Pine Script</a>
          <a class="report-link" href="xauusd/XAUUSD-FVG-Strategy/optimization_results_60m.json">📊 60m optimization JSON</a>
        </div>
      </div>
    </div><!-- /xauusd-fvg-params -->
  </div><!-- /xauusd-main-fvg -->
"""

def _tx_macro_html() -> str:
    csv_path = ROOT / "tx/csv/TAIFEX_DLY_MXF1!, 1W.csv"
    if not csv_path.exists():
        return '<div id="tx-main-macro" class="main-section active"><div class="tab-panel active"><p style="padding:24px;color:var(--muted)">週線 CSV 未找到</p></div></div>'

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['time']  = pd.to_datetime(df['time'].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open']  = pd.to_numeric(df['open'],  errors='coerce')
    df['year']  = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['week_of_month'] = df.groupby(['year','month']).cumcount() + 1

    grp = df.groupby(['year','month'])
    mon = grp.agg(m_open=('open','first'), m_close=('close','last')).reset_index()
    mon['chg_pts']  = mon['m_close'] - mon['m_open']
    mon['bullish']  = mon['chg_pts'] > 0
    mon['date']     = pd.to_datetime(mon[['year','month']].assign(day=1))

    total_months = len(mon)
    overall_wr   = mon['bullish'].mean() * 100
    avg_pts      = mon['chg_pts'].mean()
    latest       = mon.iloc[-1]
    cur_month    = int(latest['month'])
    cur_year     = int(latest['year'])
    month_names  = ['一月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月']

    sea_rows = ""
    cur_note = ""
    for m in range(1, 13):
        sub = mon[mon['month'] == m]
        if len(sub) == 0:
            continue
        wr     = sub['bullish'].mean() * 100
        avg_p  = sub['chg_pts'].mean()
        bias   = 'LONG' if wr >= 55 else ('SHORT' if wr <= 45 else 'NEUTRAL')
        bc_cls = 'badge-green' if bias == 'LONG' else ('badge-red' if bias == 'SHORT' else 'badge-blue')
        bl     = '偏多' if bias == 'LONG' else ('偏空' if bias == 'SHORT' else '中性')
        wc     = 'var(--green)' if wr >= 55 else ('var(--red)' if wr <= 45 else 'var(--muted)')
        sea_rows += (f"<tr><td>{month_names[m-1]}</td><td>{len(sub)}</td>"
                     f"<td style='color:{wc};font-weight:700'>{wr:.1f}%</td>"
                     f"<td>{'▲' if avg_p>=0 else '▼'} {abs(avg_p):.0f} pts</td>"
                     f"<td><span class='badge {bc_cls}'>{bl}</span></td></tr>\n")
        if m == cur_month:
            cur_note = f"當月（{month_names[m-1]}）：歷史勝率 {wr:.1f}%，平均 {avg_p:+.0f} pts"

    df['wk_chg']  = df['close'] - df['open']
    df['wk_bull'] = df['wk_chg'] > 0
    wim = df[df['week_of_month'] <= 5].groupby('week_of_month').agg(
        n=('wk_chg','count'), wr=('wk_bull','mean'), avg=('wk_chg','mean')).reset_index()
    wim['wr'] = wim['wr'] * 100
    wlabel = {1:'第1週（月初）',2:'第2週',3:'第3週',4:'第4週',5:'第5週（月底）'}
    wim_rows = ""
    for _, r in wim.iterrows():
        wk = int(r['week_of_month'])
        wc = 'var(--green)' if r['wr'] >= 55 else ('var(--red)' if r['wr'] <= 45 else 'var(--muted)')
        wim_rows += (f"<tr><td><strong>{wlabel[wk]}</strong></td><td>{int(r['n'])}</td>"
                     f"<td style='color:{wc};font-weight:700'>{r['wr']:.1f}%</td>"
                     f"<td>{r['avg']:+.0f} pts</td></tr>\n")

    rec = mon.sort_values('date').tail(12)
    rec_rows = ""
    for _, r in rec.iterrows():
        color = 'var(--green)' if r['bullish'] else 'var(--red)'
        sign  = '▲' if r['bullish'] else '▼'
        rec_rows += (f"<tr><td>{int(r['year'])}/{int(r['month']):02d}</td>"
                     f"<td>${r['m_open']:.0f}</td><td>${r['m_close']:.0f}</td>"
                     f"<td style='color:{color};font-weight:700'>{sign} {r['chg_pts']:+.0f} pts</td></tr>\n")

    wr_color = 'var(--green)' if overall_wr >= 55 else 'var(--red)'

    # Session analysis: day vs night from TX-Long-Experiments/results.json
    sess_rows = ""
    results_p = ROOT / "tx/TX-Long-Experiments/results.json"
    if results_p.exists():
        with open(results_p, encoding="utf-8") as rf:
            rdata = json.load(rf)
        all_results = [r for r in rdata.get("results", []) if r.get("n_trades", 0) > 0]
        if all_results:
            avg_day   = sum(r["day_win_rate"] for r in all_results if "day_win_rate" in r) / len(all_results)
            avg_night = sum(r["night_win_rate"] for r in all_results if "night_win_rate" in r) / len(all_results)
            confirmed = {"E07", "E09", "E12"}
            for r in all_results[:10]:
                d  = r.get("day_win_rate", 0)
                n  = r.get("night_win_rate", 0)
                dc = "color:var(--green);font-weight:700" if d >= 50 else "color:var(--muted)"
                nc = "color:var(--green);font-weight:700" if n >= 50 else "color:var(--muted)"
                star = "⭐" if r["code"] in confirmed else ""
                pref = "夜盤" if n > d + 2 else ("日盤" if d > n + 2 else "均可")
                sess_rows += (f"<tr><td><b>{r['code']}</b> {star}</td><td>{r['name']}</td>"
                              f"<td style='{dc}'>{d:.1f}%</td>"
                              f"<td style='{nc}'>{n:.1f}%</td>"
                              f"<td>{pref}</td></tr>\n")

    return f"""
    <!-- TX 宏觀分析 ───────────────────────────────────── -->
    <div id="tx-main-macro" class="main-section active">
      <div class="subnav">
        <button class="sub-tab active" onclick="showTab('tx-macro','overview',this)">月度統計 &amp; 季節性</button>
        <button class="sub-tab" onclick="showTab('tx-macro','weekly',this)">週內結構</button>
        <button class="sub-tab" onclick="showTab('tx-macro','recent',this)">近 12 個月</button>
        <button class="sub-tab" onclick="showTab('tx-macro','session',this)">時段分析</button>
      </div>

      <div id="tx-macro-overview" class="tab-panel active">
        <div class="part-label"><span class="part-badge">MACRO</span>整體月度統計（2012–{int(latest['year'])}）</div>
        <div class="grid-4">
          <div class="metric-card card"><div class="metric-label">整體月勝率</div><div class="metric-val" style="color:{wr_color}">{overall_wr:.1f}%</div><div class="metric-sub">{total_months} 個月（2012–{int(latest['year'])}）</div></div>
          <div class="metric-card card"><div class="metric-label">平均月漲跌</div><div class="metric-val {'green' if avg_pts>=0 else 'red'}">{avg_pts:+.0f} pts</div><div class="metric-sub">月初買、月底賣</div></div>
          <div class="metric-card card"><div class="metric-label">當月（{month_names[cur_month-1]}）</div><div class="metric-val">{cur_year}/{cur_month:02d}</div><div class="metric-sub" style="font-size:.8em">{cur_note}</div></div>
          <div class="metric-card card"><div class="metric-label">週線 K 棒數</div><div class="metric-val">{len(df)}</div><div class="metric-sub">2012–{int(latest['year'])}</div></div>
        </div>

        <div class="insight-grid">
          <div class="insight good"><strong>✅ 整體偏多</strong>月勝率 {overall_wr:.1f}%，月初買月底賣平均 {avg_pts:+.0f} pts — 大方向順多。</div>
          <div class="insight warn"><strong>⚠ 九月唯一偏空月</strong>歷史勝率僅 42.9%；策略測試需注意此月份空單機會較多。</div>
          <div class="insight info"><strong>📊 四月期望值最大，十二月最穩</strong>四月平均 +518 pts；十二月月勝率 78.6%（最高）。</div>
        </div>

        <div class="card">
          <div class="card-title">🗓 季節性偏向 — 每月歷史統計（2012–{int(latest['year'])}）</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>月份</th><th>樣本</th><th>月勝率</th><th>平均漲跌（pts）</th><th>偏向</th></tr></thead>
              <tbody>{sea_rows}</tbody>
            </table>
          </div>
        </div>
        <div class="report-links">
          <a class="report-link" href="tx/macro_report.html">📄 完整宏觀報告（暗色主題 + 熱力圖）</a>
        </div>
      </div>

      <div id="tx-macro-weekly" class="tab-panel">
        <div class="part-label"><span class="part-badge">MACRO</span>週內結構 — 每月第幾週最強</div>
        <div class="card">
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>週次</th><th>樣本</th><th>週勝率</th><th>平均漲跌（pts）</th></tr></thead>
              <tbody>{wim_rows}</tbody>
            </table>
          </div>
          <div style="margin-top:8px;font-size:.82em;color:var(--muted)">第1、4、5週勝率較高；第3週（月中）最弱。建議在月初或月末偏強週順月度方向進場。</div>
        </div>
      </div>

      <div id="tx-macro-recent" class="tab-panel">
        <div class="part-label"><span class="part-badge">MACRO</span>近 12 個月回顧</div>
        <div class="card" style="max-width:620px">
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>年/月</th><th>月初開盤</th><th>月底收盤</th><th>月漲跌（pts）</th></tr></thead>
              <tbody>{rec_rows}</tbody>
            </table>
          </div>
        </div>
      </div>

      <div id="tx-macro-session" class="tab-panel">
        <div class="part-label"><span class="part-badge">MACRO</span>日盤 vs 夜盤分析（台指期 MTX）</div>
        <div class="insight-grid">
          <div class="insight good"><strong>✅ 夜盤整體優於日盤</strong>確認策略（⭐E07/E09/E12）夜盤勝率平均高 6–10%；夜盤受美股直接驅動，NQ 方向更明確。</div>
          <div class="insight warn"><strong>⚠ 日盤波動受國際消息主導</strong>開盤跳空後反轉機率高，SL=120 pts 的設計下 immediate_loss 在日盤更常見。</div>
          <div class="insight info"><strong>📊 操作建議</strong>優先在夜盤開啟確認策略；日盤可縮減 Size 或加嚴過濾條件（如 NQ RSI 同向確認）。</div>
        </div>
        <div class="card">
          <div class="card-title">🌙 多單策略 日盤 vs 夜盤勝率（SL=120, TP=240）</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>代碼</th><th>策略名稱</th><th>日盤 WR</th><th>夜盤 WR</th><th>偏好時段</th></tr></thead>
              <tbody>{sess_rows or "<tr><td colspan='5' style='color:#999'>尚無資料（請執行 run_experiments.py）</td></tr>"}</tbody>
            </table>
          </div>
          <div style="margin-top:8px;font-size:.8em;color:var(--muted)">⭐ = 已確認策略（E07/E09/E12）</div>
        </div>
      </div>
    </div><!-- /tx-main-macro -->
"""


def _tx_confirmed_html() -> str:
    results_path = ROOT / "tx/TX-Long-Experiments/results.json"
    confirmed_ids = {"E07", "E09", "E12"}
    cards_html = ""
    sess_rows  = ""

    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            rdata = json.load(f)
        sl = rdata.get("sl_pts", 120)
        tp = rdata.get("tp_pts", 240)
        confirmed_list = [r for r in rdata.get("results", []) if r["code"] in confirmed_ids]
        for r in confirmed_list:
            pnl  = r.get("net_pnl_ntd", 0)
            pnl_c = "var(--green)" if pnl >= 0 else "var(--red)"
            dwr  = r.get("day_win_rate", 0)
            nwr  = r.get("night_win_rate", 0)
            pref = "夜盤" if nwr > dwr + 2 else ("日盤" if dwr > nwr + 2 else "均可")
            wc   = "var(--green)" if r["win_rate"] >= 50 else "var(--yellow)"
            cards_html += f"""
        <div class="card">
          <div class="card-title">📊 {r['code']} {r['name']} <span class="badge badge-green">✅ 已確認</span></div>
          <div class="grid-4" style="gap:10px">
            <div class="metric-card"><div class="metric-label">勝率</div><div class="metric-val" style="color:{wc}">{r['win_rate']}%</div></div>
            <div class="metric-card"><div class="metric-label">獲利因子</div><div class="metric-val green">{r['profit_factor']:.3f}</div></div>
            <div class="metric-card"><div class="metric-label">淨盈虧</div><div class="metric-val" style="color:{pnl_c}">NT${pnl:,.0f}</div></div>
            <div class="metric-card"><div class="metric-label">交易筆數</div><div class="metric-val">{r['n_trades']}</div></div>
          </div>
          <div style="margin-top:10px;font-size:.82em;color:var(--muted)">SL={sl}pts / TP={tp}pts (R:R 1:{tp//sl}) &nbsp;·&nbsp; 偏好時段：{pref}（日盤 {dwr}% / 夜盤 {nwr}%）</div>
        </div>
"""
            sess_rows += (f"<tr><td><b>{r['code']}</b></td><td>{r['name']}</td>"
                          f"<td>{r['n_trades']}</td>"
                          f"<td style='color:{wc};font-weight:700'>{r['win_rate']}%</td>"
                          f"<td>{r['profit_factor']:.3f}</td>"
                          f"<td style='color:{pnl_c};font-weight:700'>NT${pnl:,.0f}</td>"
                          f"<td style='color:{'var(--green)' if dwr>=50 else 'var(--muted)'}'>{dwr}%</td>"
                          f"<td style='color:{'var(--green)' if nwr>=50 else 'var(--muted)'}'>{nwr}%</td>"
                          f"</tr>\n")
    else:
        sl, tp = 120, 240
        cards_html = "<p style='color:var(--muted);padding:12px'>尚無資料（請執行 run_experiments.py --sl 120）</p>"

    return f"""
  <!-- TX 已確認策略 ──────────────────────────────────── -->
  <div id="tx-main-confirmed" class="main-section">
    <div class="subnav">
      <button class="sub-tab active" onclick="showTab('tx-confirmed','overview',this)">策略總覽</button>
      <button class="sub-tab" onclick="showTab('tx-confirmed','context',this)">確認依據</button>
    </div>

    <div id="tx-confirmed-overview" class="tab-panel active">
      <div class="part-label"><span class="part-badge">CONFIRMED</span>已確認策略 · SL={sl}pts / TP={tp}pts</div>
      <div class="insight-grid">
        <div class="insight good"><strong>✅ SL=120pts 是甜蜜點</strong>系統性測試 SL 30/50/60/80/100/120/150 後，SL=120 時 E09/E07/E12 均達 PF>2.0、WR>50%。</div>
        <div class="insight info"><strong>📊 確認條件</strong>PF &gt; 1.5 且 WR &gt; 48% 且淨盈虧為正，在 2025-06 至今的 MTX 30m 資料中通過。</div>
        <div class="insight warn"><strong>⚠ 當前資料期間</strong>2025-06 至今為台指大多頭，確認策略均為多單。空單策略尚無確認版本。</div>
      </div>
      <div class="grid-3">
{cards_html}
      </div>
    </div>

    <div id="tx-confirmed-context" class="tab-panel">
      <div class="part-label"><span class="part-badge">CONFIRMED</span>確認依據 · SL 敏感度分析（2026-05-13）</div>
      <div class="card">
        <div class="card-title">🧪 SL 敏感度測試結果（E09/E07/E12，R:R 固定 2:1）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>SL（pts）</th><th>TP（pts）</th><th>獲利策略數</th><th>備註</th></tr></thead>
            <tbody>
              <tr><td>30</td><td>60</td><td>1 (E12)</td><td>SL 過緊，ATR 遠大於 30pts</td></tr>
              <tr><td>60</td><td>120</td><td>3</td><td>E12 PF=1.648, WR=45.2%</td></tr>
              <tr><td>80</td><td>160</td><td>3–4</td><td>轉折點，開始改善</td></tr>
              <tr><td>100</td><td>200</td><td>5+</td><td>持續改善</td></tr>
              <tr class="rank-1"><td><b>120 ★</b></td><td><b>240</b></td><td><b>E09/E07/E12 均 PF>2.0</b></td><td><b>甜蜜點 — 新基準</b></td></tr>
              <tr><td>150</td><td>300</td><td>同上但交易筆數減少</td><td>信號更少</td></tr>
            </tbody>
          </table>
        </div>
        <div style="margin-top:10px;font-size:.85em;color:var(--text2)">
          <b>根本原因：</b>台指期 MTX 平均 ATR ≈ 50–150 pts，SL=30 只有 0.15%，雜訊就能掃出。SL=120 約等於 0.6%，才能給趨勢策略足夠的呼吸空間。
        </div>
      </div>
      <div class="card">
        <div class="card-title">📋 E09 / E07 / E12 綜合對比（SL={sl}, TP={tp}）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>代碼</th><th>策略</th><th>筆數</th><th>勝率</th><th>PF</th><th>淨盈虧</th><th>日盤 WR</th><th>夜盤 WR</th></tr></thead>
            <tbody>{sess_rows or "<tr><td colspan='8' style='color:#999'>尚無資料</td></tr>"}</tbody>
          </table>
        </div>
      </div>
      <div class="report-links">
        <a class="report-link" href="tx/TX-Long-Experiments/report.html">📄 完整多單實驗報告</a>
      </div>
    </div>
  </div><!-- /tx-main-confirmed -->
"""


def _tx_exp_html(long_rows: str, short_rows: str, c: dict) -> str:
    long_link  = c["long_dir"] + "/report.html"
    short_link = c["short_dir"] + "/report.html"
    long_pine  = c["long_pine"]
    short_pine = c["short_pine"]
    long_exists  = (ROOT / long_link).exists()
    short_exists = (ROOT / short_link).exists()

    long_btn  = f"<a href='{long_link}' class='btn btn-long'>多單報告 →</a>" if long_exists else ""
    short_btn = f"<a href='{short_link}' class='btn btn-short'>空單報告 →</a>" if short_exists else ""

    return f"""
    <!-- TX 實驗策略 ──────────────────────────────────── -->
    <div id="tx-main-exp" class="main-section">
      <div class="subnav">
        <button class="sub-tab active" onclick="showTab('tx-exp','overview',this)">綜合對比</button>
        <button class="sub-tab" onclick="showTab('tx-exp','long',this)">多單 (E01–E20)</button>
        <button class="sub-tab" onclick="showTab('tx-exp','short',this)">空單 (S01–S20)</button>
        <button class="sub-tab" onclick="showTab('tx-exp','pine',this)">Pine Script</button>
      </div>

      <div id="tx-exp-overview" class="tab-panel active">
        <div class="part-label"><span class="part-badge">PART 1</span>綜合對比 Top-3</div>
        <div class="grid-2">
          <div class="card">
            <div class="card-title">📈 多單 Top-3（E01–E20）</div>
            <div class="tbl-wrap">
              <table>
                <thead><tr><th>代碼</th><th>策略</th><th>勝率</th><th>PF</th><th>淨盈虧</th><th>分數</th></tr></thead>
                <tbody>{long_rows or "<tr><td colspan='6' style='color:#999'>尚無資料（請執行 run_experiments.py）</td></tr>"}</tbody>
              </table>
            </div>
            {long_btn}
          </div>
          <div class="card">
            <div class="card-title">📉 空單 Top-3（S01–S20）</div>
            <div class="tbl-wrap">
              <table>
                <thead><tr><th>代碼</th><th>策略</th><th>勝率</th><th>PF</th><th>淨盈虧</th><th>分數</th></tr></thead>
                <tbody>{short_rows or "<tr><td colspan='6' style='color:#999'>尚無資料</td></tr>"}</tbody>
              </table>
            </div>
            {short_btn}
          </div>
        </div>
        <div class="insight-grid">
          <div class="insight good"><strong>✅ SL=120pts 甜蜜點驗證</strong>E09/E07/E12 在 SL=120 下均達 PF>2.0、WR>50%。</div>
          <div class="insight warn"><strong>⚠ 當前期間</strong>2025-06 至今台指大多頭；空單整體虧損為預期結果。</div>
          <div class="insight info"><strong>📊 下一步</strong>加入 NQ RSI 相關性過濾；測試 4H MTF 過濾器。</div>
        </div>
      </div>

      <div id="tx-exp-long" class="tab-panel">
        <div class="part-label"><span class="part-badge">PART 1</span>多單實驗（E01–E20）</div>
        <div class="grid-4">
          <div class="metric-card card"><div class="metric-label">商品</div><div class="metric-val" style="font-size:1.1em">MTX 小台</div></div>
          <div class="metric-card card"><div class="metric-label">止損</div><div class="metric-val">120 pts</div><div class="metric-sub">NT$6,000/口</div></div>
          <div class="metric-card card"><div class="metric-label">止盈</div><div class="metric-val">240 pts</div><div class="metric-sub">R:R 1:2</div></div>
          <div class="metric-card card"><div class="metric-label">時間止損</div><div class="metric-val">48 bars</div><div class="metric-sub">24 小時</div></div>
        </div>
        <div class="report-links">
          {long_btn}
          <a class="report-link" href="{long_pine}">📋 ALL_Long_Strategies.pine</a>
        </div>
      </div>

      <div id="tx-exp-short" class="tab-panel">
        <div class="part-label"><span class="part-badge">PART 1</span>空單實驗（S01–S20）</div>
        <div class="grid-4">
          <div class="metric-card card"><div class="metric-label">商品</div><div class="metric-val" style="font-size:1.1em">MTX 小台</div></div>
          <div class="metric-card card"><div class="metric-label">止損</div><div class="metric-val">120 pts</div><div class="metric-sub">NT$6,000/口</div></div>
          <div class="metric-card card"><div class="metric-label">止盈</div><div class="metric-val">240 pts</div><div class="metric-sub">R:R 1:2</div></div>
          <div class="metric-card card"><div class="metric-label">期間特性</div><div class="metric-val" style="font-size:.9em">大多頭期</div><div class="metric-sub">空單困難為預期</div></div>
        </div>
        <div class="report-links">
          {short_btn}
          <a class="report-link" href="{short_pine}">📋 ALL_Short_Strategies.pine</a>
        </div>
      </div>

      <div id="tx-exp-pine" class="tab-panel">
        <div class="part-label"><span class="part-badge">PART 1</span>Pine Script 使用說明</div>
        <div class="card">
          <ol style="padding-left:20px;line-height:2.2;font-size:.92em">
            <li>下載 <code>ALL_Long_Strategies.pine</code> 或 <code>ALL_Short_Strategies.pine</code></li>
            <li>在 TradingView Pine Script Editor 貼上，套用到 <code>TAIFEX:MXF1!</code> 30m 圖表</li>
            <li><b>Strategy</b> 下拉選單選擇策略（E01–E20 / S01–S20）</li>
            <li><b>Enable Signals</b> 開關控制是否進場（關閉後仍顯示灰色參考箭頭）</li>
            <li><b>Stop Loss</b> 設定止損點數；<b>R:R Ratio</b> 自動計算止盈（TP = SL × R:R）</li>
            <li><b>Session</b> 可單獨開關日盤 / 夜盤</li>
          </ol>
          <div class="report-links" style="margin-top:16px">
            <a class="report-link" href="{long_pine}">📋 ALL_Long_Strategies.pine</a>
            <a class="report-link" href="{short_pine}">📋 ALL_Short_Strategies.pine</a>
          </div>
        </div>
      </div>
    </div><!-- /tx-main-exp -->
"""

def _load_shared_results() -> dict:
    p = ROOT / "shared/shared_results.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _shared_analysis_html(data: dict) -> str:
    if not data:
        return """
  <div id="commodity-shared" class="commodity-section">
    <div class="tab-panel active"><div class="card">
      <div class="card-title">跨商品分析</div>
      <p style="color:var(--muted)">請先執行 <code>python3.12 shared/run_shared_analysis.py</code> 生成資料。</p>
    </div></div>
  </div>"""

    def _rsi_table(rsi_stats: dict, unit_label: str) -> str:
        order = ['overall', 'golden_cross', 'death_cross', 'bullish_div', 'bearish_div', 'rsi_above_ma', 'rsi_below_ma']
        labels = {
            'overall': '全部樣本', 'golden_cross': 'RSI 金叉（30m）',
            'death_cross': 'RSI 死叉（30m）', 'bullish_div': '多頭背離（30m）',
            'bearish_div': '空頭背離（30m）', 'rsi_above_ma': 'RSI > MA（30m）',
            'rsi_below_ma': 'RSI < MA（30m）',
        }
        rows = []
        for state in order:
            s = rsi_stats.get(state)
            if s is None:
                rows.append(f"<tr><td>{labels[state]}</td><td colspan='3' style='color:var(--muted);font-style:italic'>無資料（CSV 未匯出訊號）</td></tr>")
                continue
            wr = s['win_rate']
            wr_cls = 'pos' if wr >= 52 else ('neg' if wr < 48 else '')
            ret = s['avg_ret']
            ret_cls = 'pos' if ret > 0 else 'neg'
            ret_str = f"{ret:+.2f}" if unit_label == '%' else f"{ret:+.0f}"
            rows.append(
                f"<tr><td>{labels[state]}</td>"
                f"<td class='{wr_cls}'>{wr:.1f}%</td>"
                f"<td class='{ret_cls}'>{ret_str}{unit_label}</td>"
                f"<td style='color:var(--muted)'>{s['count']}</td></tr>"
            )
        return "\n".join(rows)

    def _insight_card(label: str, cell: dict | None, unit_label: str, good: bool) -> str:
        if cell is None:
            return ""
        color = "good" if good else "bad"
        ret_str = f"{cell['avg_ret']:+.2f}{unit_label}" if unit_label == '%' else f"{cell['avg_ret']:+.0f}{unit_label}"
        return (
            f"<div class='insight {color}'>"
            f"<strong>{'最佳' if good else '最差'}時段 · {cell['dow']} {cell['hour']:02d}:00</strong>"
            f"勝率 {cell['win_rate']:.1f}% &nbsp;|&nbsp; 平均 {ret_str} &nbsp;|&nbsp; n={cell['count']}"
            f"</div>"
        )

    xu = data.get("xauusd", {})
    tx = data.get("tx", {})

    xu_rsi_tbl = _rsi_table(xu.get("rsi_stats", {}), xu.get("unit_label", "%"))
    tx_rsi_tbl = _rsi_table(tx.get("rsi_stats", {}), tx.get("unit_label", "pts"))

    xu_best  = _insight_card("XAUUSD", xu.get("best_wr"),  xu.get("unit_label", "%"),   True)
    xu_worst = _insight_card("XAUUSD", xu.get("worst_wr"), xu.get("unit_label", "%"),   False)
    tx_best  = _insight_card("TX",     tx.get("best_wr"),  tx.get("unit_label", "pts"), True)
    tx_worst = _insight_card("TX",     tx.get("worst_wr"), tx.get("unit_label", "pts"),  False)

    xu_wr_img  = f"<img src='data:image/png;base64,{xu['wr_heatmap_b64']}'  style='max-width:100%;border-radius:8px'>" if xu.get("wr_heatmap_b64") else ""
    xu_ret_img = f"<img src='data:image/png;base64,{xu['ret_heatmap_b64']}' style='max-width:100%;border-radius:8px'>" if xu.get("ret_heatmap_b64") else ""
    xu_rsi_img = f"<img src='data:image/png;base64,{xu['rsi_filter_b64']}' style='max-width:100%;border-radius:8px'>" if xu.get("rsi_filter_b64") else ""
    tx_wr_img  = f"<img src='data:image/png;base64,{tx['wr_heatmap_b64']}'  style='max-width:100%;border-radius:8px'>" if tx.get("wr_heatmap_b64") else ""
    tx_ret_img = f"<img src='data:image/png;base64,{tx['ret_heatmap_b64']}' style='max-width:100%;border-radius:8px'>" if tx.get("ret_heatmap_b64") else ""
    tx_rsi_img = f"<img src='data:image/png;base64,{tx['rsi_filter_b64']}' style='max-width:100%;border-radius:8px'>" if tx.get("rsi_filter_b64") else ""

    xu_n = xu.get("n_total", 0)
    xu_rsi_n = xu.get("n_rsi_overlap", 0)
    tx_n = tx.get("n_total", 0)
    tx_rsi_n = tx.get("n_rsi_overlap", 0)

    return f"""
  <!-- ══ SHARED CROSS-COMMODITY ANALYSIS ══════════════════════════ -->
  <div id="commodity-shared" class="commodity-section">
    <div class="commodity-subnav">
      <button class="nav-main-tab active" onclick="showMain('shared-main-heatmap',this)">整點熱力圖</button>
      <button class="nav-main-tab" onclick="showMain('shared-main-rsi',this)">RSI 濾鏡</button>
    </div>

    <!-- 整點熱力圖 -->
    <div id="shared-main-heatmap" class="main-section active">
      <div class="tab-panel active">
        <div class="part-label"><span class="part-badge">HEATMAP</span>整點進場 · 下一整點出場 — 勝率 &amp; 損益熱力圖</div>

        <div class="card" style="margin-bottom:8px">
          <div style="font-size:.88em;color:var(--text2);line-height:1.7">
            <strong>分析方法：</strong>每個整點（00分）進場，下一個整點出場，計算各 <em>星期幾 × 小時</em> 組合的歷史勝率與平均損益。
            「*」表示樣本 &lt; 5 筆，結果僅供參考。
            XAUUSD：<strong>{xu_n:,}</strong> 筆 60m bar（週一至週五）。
            TX MTX：<strong>{tx_n:,}</strong> 筆 60m bar（週一至週五）。
          </div>
        </div>

        <!-- XAUUSD 熱力圖 -->
        <div class="card">
          <div class="card-title">🟡 XAUUSD 黃金 — 整點進場熱力圖</div>
          <div class="insight-grid">{xu_best}{xu_worst}</div>
          <div class="grid-2" style="gap:12px;margin-top:12px">
            <div>{xu_wr_img}</div>
            <div>{xu_ret_img}</div>
          </div>
        </div>

        <!-- TX 熱力圖 -->
        <div class="card">
          <div class="card-title">🔵 TX 台指期 (MTX) — 整點進場熱力圖</div>
          <div class="insight-grid">{tx_best}{tx_worst}</div>
          <div class="grid-2" style="gap:12px;margin-top:12px">
            <div>{tx_wr_img}</div>
            <div>{tx_ret_img}</div>
          </div>
        </div>
      </div>
    </div><!-- /shared-main-heatmap -->

    <!-- RSI 濾鏡分析 -->
    <div id="shared-main-rsi" class="main-section">
      <div class="tab-panel active">
        <div class="part-label"><span class="part-badge">RSI FILTER</span>30m RSI 金叉 / 死叉 / 背離 — 整點進場過濾效果</div>

        <div class="card" style="margin-bottom:8px">
          <div style="font-size:.88em;color:var(--text2);line-height:1.7">
            <strong>分析方法：</strong>在整點進場時，查詢 30 分鐘 RSI 的狀態（金叉 / 死叉 / RSI 位於 MA 上下方 / 背離訊號），
            比較不同狀態下的歷史勝率與平均損益差異。<br>
            <strong>注意：</strong>RSI 背離（Regular Bullish / Bearish）欄位在目前匯出的 CSV 中無訊號資料，
            如需此分析請在 TradingView 匯出時確保 <em>Regular Bullish/Bearish Label</em> 欄位有值。<br>
            XAUUSD 有效 30m RSI 重疊筆數：<strong>{xu_rsi_n:,}</strong>（共 {xu_n:,} 筆）。
            TX 有效 30m RSI 重疊筆數：<strong>{tx_rsi_n:,}</strong>（共 {tx_n:,} 筆）。
          </div>
        </div>

        <!-- XAUUSD RSI Filter -->
        <div class="card">
          <div class="card-title">🟡 XAUUSD 黃金 — 30m RSI 狀態 × 勝率</div>
          <div style="margin-bottom:12px">{xu_rsi_img}</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>RSI 狀態</th><th>勝率</th><th>平均損益</th><th>樣本數</th></tr></thead>
              <tbody>{xu_rsi_tbl}</tbody>
            </table>
          </div>
        </div>

        <!-- TX RSI Filter -->
        <div class="card">
          <div class="card-title">🔵 TX 台指期 — 30m RSI 狀態 × 勝率</div>
          <div style="margin-bottom:12px">{tx_rsi_img}</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>RSI 狀態</th><th>勝率</th><th>平均損益</th><th>樣本數</th></tr></thead>
              <tbody>{tx_rsi_tbl}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div><!-- /shared-main-rsi -->

  </div><!-- /commodity-shared -->
"""


def _load_validation() -> dict:
    p = ROOT / "doc/validation_results.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _match_badge(match: bool) -> str:
    return "<span style='color:#059669;font-weight:700'>✅</span>" if match else "<span style='color:#dc2626;font-weight:700'>❌</span>"


def _xauusd_validation_html(vdata: dict) -> str:
    xu = vdata.get("xauusd", {})
    month_names = ["一","二","三","四","五","六","七","八","九","十","十一","十二"]

    # Monthly table rows
    monthly_rows = ""
    for r in xu.get("xauusd_monthly", []):
        m = int(r["month"])
        mb = _match_badge(r["match"])
        wr_col = "color:#059669" if r["win_rate"] >= 0.55 else ("color:#dc2626" if r["win_rate"] <= 0.45 else "color:#d97706")
        monthly_rows += (
            f"<tr><td>{month_names[m-1]}月</td>"
            f"<td>{r['note_view']}</td>"
            f"<td style='{wr_col};font-weight:700'>{r['data_view']} ({r['win_rate']:.0%})</td>"
            f"<td style='text-align:right'>{r['avg_ret']:.1%}</td>"
            f"<td style='text-align:center'>{mb}</td></tr>\n"
        )

    # Sessions table rows
    session_rows = ""
    for s in xu.get("xauusd_sessions", []):
        trend = s["趨勢K比例"]
        note  = s["筆記判斷"]
        # oscillation note says mostly oscillation → expect low trend%, so 趨勢K < 50% is consistent
        if "震盪" in note:
            consistent = trend < 0.55
        else:
            consistent = True  # 50/50 always partial
        mb = _match_badge(consistent)
        trend_col = "color:#dc2626" if trend >= 0.55 else "color:#059669"
        session_rows += (
            f"<tr><td>{s['時段']}</td>"
            f"<td>{note}</td>"
            f"<td style='{trend_col};font-weight:700'>{trend:.0%}</td>"
            f"<td>{s['平均波動%']:.3f}%</td>"
            f"<td>{s['樣本數']:,}</td>"
            f"<td style='text-align:center'>{mb}</td></tr>\n"
        )

    # Week-of-month rows
    wom_rows = ""
    for r in xu.get("xauusd_week_of_month", []):
        wom_rows += (
            f"<tr><td>第{int(r['week_of_month'])}週</td>"
            f"<td>{r['avg_move']:.4f}</td>"
            f"<td>{int(r['n'])}</td></tr>\n"
        )

    # Quarterly rows
    q_rows = ""
    for r in xu.get("xauusd_quarterly", []):
        wr_col = "color:#059669;font-weight:700" if r["win_rate"] >= 0.55 else "color:#d97706;font-weight:700"
        q_rows += (
            f"<tr><td>Q{int(r['quarter'])}</td>"
            f"<td style='{wr_col}'>{r['win_rate']:.0%}</td>"
            f"<td>{r['avg_ret']:.2%}</td>"
            f"<td>{int(r['n'])}</td></tr>\n"
        )

    return f"""
  <!-- XAUUSD 筆記驗證 -->
  <div id="xauusd-main-validate" class="main-section">
    <div class="tab-panel active">
      <div class="part-label"><span class="part-badge">VALIDATE</span>筆記驗證 · Notes vs Data (2014–2026)</div>

      <div class="insight-grid">
        <div class="insight info"><strong>驗證方法</strong>以 XAUUSD 日線月收益率計算勝率（WR≥55%=強、≤45%=弱、其餘=中），對照黃金秘笈/黃金短線筆記結論。</div>
        <div class="insight warn"><strong>樣本期間</strong>日線 2014–2026（月度樣本 ~12年），30m 時段資料（帶時區轉換為台北時間）。</div>
      </div>

      <div class="card">
        <div class="card-title">📅 月份強弱驗證（黃金秘笈）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>月份</th><th>筆記判斷</th><th>資料實測</th><th>平均月報酬</th><th>符合</th></tr></thead>
            <tbody>{monthly_rows}</tbody>
          </table>
        </div>
        <div style="margin-top:10px;font-size:.8em;color:var(--muted)">
          ✅ = 筆記與資料同向 ｜ ❌ = 不符（中性視為不完全符合）
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">📊 季度統計</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>季度</th><th>勝率</th><th>平均報酬</th><th>樣本數</th></tr></thead>
              <tbody>{q_rows}</tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <div class="card-title">📅 週次波動（每月第幾週）</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>週次</th><th>平均波動</th><th>樣本數</th></tr></thead>
              <tbody>{wom_rows}</tbody>
            </table>
          </div>
          <div style="margin-top:8px;font-size:.8em;color:var(--muted)">筆記：第1、3週最強</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">⏱ 四個時段特性（黃金短線 + 四個時間做單）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>時段</th><th>筆記判斷</th><th>趨勢K比例</th><th>平均波動</th><th>樣本數</th><th>符合</th></tr></thead>
            <tbody>{session_rows}</tbody>
          </table>
        </div>
        <div style="margin-top:10px;font-size:.8em;color:var(--muted)">
          趨勢K定義：abs_ret &gt; 0.15%；筆記震盪主張 → 趨勢K比例應 &lt; 55%
        </div>
      </div>

    </div>
  </div><!-- /xauusd-main-validate -->
"""


def _tx_validation_html(vdata: dict) -> str:
    tx = vdata.get("tx", {})
    month_names = ["一","二","三","四","五","六","七","八","九","十","十一","十二"]

    # Monthly rows
    monthly_rows = ""
    for r in tx.get("tx_monthly", []):
        m = int(r["month"])
        mb = _match_badge(r["match"])
        wr_col = "color:#059669" if r["win_rate"] >= 0.55 else ("color:#dc2626" if r["win_rate"] <= 0.45 else "color:#d97706")
        monthly_rows += (
            f"<tr><td>{month_names[m-1]}月</td>"
            f"<td>{r['note_view']}</td>"
            f"<td style='{wr_col};font-weight:700'>{r['data_view']} ({r['win_rate']:.0%})</td>"
            f"<td style='text-align:right'>{r['avg_ret']:.1%}</td>"
            f"<td style='text-align:center'>{mb}</td></tr>\n"
        )

    # Quarterly rows (指數密技: Q4 幾乎必漲)
    q_rows = ""
    for r in tx.get("tx_quarterly", []):
        wr_col = "color:#059669;font-weight:700" if r["win_rate"] >= 0.60 else "color:#d97706;font-weight:700"
        note_q4 = " ⭐ Q4必有多單" if int(r["quarter"]) == 4 else ""
        q_rows += (
            f"<tr><td>Q{int(r['quarter'])}{note_q4}</td>"
            f"<td style='{wr_col}'>{r['win_rate']:.0%}</td>"
            f"<td>{r['avg_ret']:.2%}</td>"
            f"<td>{int(r['n'])}</td></tr>\n"
        )

    # Week-of-month rows
    wom_rows = ""
    for r in tx.get("tx_week_of_month", []):
        wom_rows += (
            f"<tr><td>第{int(r['week_of_month'])}週</td>"
            f"<td>{r['avg_move']:.4f}</td>"
            f"<td>{int(r['n'])}</td></tr>\n"
        )

    # Election year rows
    elec_rows = ""
    for r in tx.get("tx_election", []):
        is_e = "選舉年" if r["is_election"] else "非選舉年"
        wr_col = "color:#059669;font-weight:700" if r["win_rate"] >= 0.60 else "color:#d97706;font-weight:700"
        elec_rows += (
            f"<tr><td>{is_e}</td><td>Q{int(r['q'])}</td>"
            f"<td style='{wr_col}'>{r['win_rate']:.0%}</td>"
            f"<td>{r['avg_ret']:.2%}</td>"
            f"<td>{int(r['n'])}</td></tr>\n"
        )

    return f"""
  <!-- TX 筆記驗證 -->
  <div id="tx-main-validate" class="main-section">
    <div class="tab-panel active">
      <div class="part-label"><span class="part-badge">VALIDATE</span>筆記驗證 · Notes vs Data (2012–2026)</div>

      <div class="insight-grid">
        <div class="insight info"><strong>驗證方法</strong>以 TX 日線月收益率計算勝率（WR≥55%=強、≤45%=弱），對照指數密技筆記結論。</div>
        <div class="insight warn"><strong>樣本期間</strong>日線 2012–2026（月度樣本 ~14年），選舉年：2000/2004/2008/2012/2016/2020/2024。</div>
      </div>

      <div class="card">
        <div class="card-title">📅 月份強弱驗證（指數密技）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>月份</th><th>筆記判斷</th><th>資料實測</th><th>平均月報酬</th><th>符合</th></tr></thead>
            <tbody>{monthly_rows}</tbody>
          </table>
        </div>
        <div style="margin-top:10px;font-size:.8em;color:var(--muted)">
          筆記：一月通常漲、十二月幾乎必漲、五月偏弱；資料以月勝率判定。
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">📊 季度統計（指數密技：Q4 幾乎必有多單機會）</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>季度</th><th>勝率</th><th>平均報酬</th><th>樣本數</th></tr></thead>
              <tbody>{q_rows}</tbody>
            </table>
          </div>
          <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
            各季勝率均 &gt; 60%，Q4 avg +1.7% 最高 — 支持「Q4必有多單」結論。
          </div>
        </div>
        <div class="card">
          <div class="card-title">📅 週次波動（每月第幾週）</div>
          <div class="tbl-wrap">
            <table>
              <thead><tr><th>週次</th><th>平均波動</th><th>樣本數</th></tr></thead>
              <tbody>{wom_rows}</tbody>
            </table>
          </div>
          <div style="margin-top:8px;font-size:.8em;color:var(--muted)">筆記：第1、3週最強</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">🗳 選舉年效應（指數密技：四年周期）</div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>年份類型</th><th>季度</th><th>勝率</th><th>平均報酬</th><th>樣本數</th></tr></thead>
            <tbody>{elec_rows}</tbody>
          </table>
        </div>
        <div style="margin-top:10px;font-size:.8em;color:var(--muted)">
          台灣總統大選每4年（2000/2004/2008/2012/2016/2020/2024）；選舉年 Q1/Q4 特別強。
        </div>
      </div>

    </div>
  </div><!-- /tx-main-validate -->
"""


# ─── Main generator ──────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════
# 以下為 20260711 拆頁新架構程式碼（上方為 generate_index.py 逐字移植區）
# ═════════════════════════════════════════════════════════════════════

# ─── Fragment 載入 ───────────────────────────────────────────────────
def _frag(relpath: str) -> str:
    """讀取 content/ 下的手寫 fragment（唯一合法的手寫內容存放處）"""
    return (CONTENT / relpath).read_text(encoding="utf-8")


def _xauusd_macro_with_manual() -> str:
    """動態生成的宏觀分析區塊 + 注入兩個手寫子分頁（宏觀指標解讀 / Macro 回測）"""
    macro = _xauusd_macro_html()
    btn_anchor = "時段分析</button>"
    extra_btns = (
        "\n      <button class=\"sub-tab\" onclick=\"showTab('xauusd-macro','macroindicator',this)\">📈 宏觀指標解讀</button>"
        "\n      <button class=\"sub-tab\" onclick=\"showTab('xauusd-macro','macrobacktest',this)\">🔬 Macro 回測</button>"
    )
    assert btn_anchor in macro, "macro subnav 錨點不存在，_xauusd_macro_html 結構可能已變"
    macro = macro.replace(btn_anchor, btn_anchor + extra_btns, 1)
    # 兩個手寫面板插在區塊最後一個 </div>（關閉 xauusd-main-macro）之前
    idx = macro.rstrip().rfind("</div>")
    panels = (_frag("xauusd/macro_indicator.html") + "\n"
              + _frag("xauusd/macro_backtest.html") + "\n    ")
    return macro[:idx] + panels + macro[idx:]


# ─── 對話記錄（data/logs.json 為唯一來源，計數自動）──────────────────
TAG_CFG = {
    "xauusd": {"label": "🟡 XAUUSD", "bg": "#fef3c7", "color": "#92400e", "border": "#f59e0b"},
    "tx":     {"label": "🔵 TX 台指期", "bg": "#dbeafe", "color": "#1e40af", "border": "#3b82f6"},
    "cross":  {"label": "📊 跨商品",  "bg": "#ede9fe", "color": "#5b21b6", "border": "#7c3aed"},
}


def _load_logs() -> list[dict]:
    logs = json.loads(LOGS_PATH.read_text(encoding="utf-8"))
    return sorted(logs, key=lambda e: e["date"], reverse=True)


def _log_entry_html(entry: dict) -> str:
    cfg = TAG_CFG.get(entry["tag"], TAG_CFG["cross"])
    tag = (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:12px;"
        f"background:{cfg['bg']};color:{cfg['color']};"
        f"border:1px solid {cfg['border']};font-size:.78em;font-weight:700;"
        f"margin-bottom:6px'>{cfg['label']}</span>"
    )
    items_html = "".join(f"<li>{it}</li>" for it in entry["items"])
    return (
        f"<div class='log-entry'>"
        f"<div class='log-date'>{entry['date']}</div>"
        f"<div>{tag}<div class='log-title' style='margin-top:2px'>{entry['title']}</div>"
        f"<ul class='log-items'>{items_html}</ul></div>"
        f"</div>"
    )


def _history_section() -> str:
    logs = _load_logs()
    blocks = "".join(_log_entry_html(e) for e in logs)
    legend = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:12px;"
        f"padding:3px 10px;border-radius:12px;background:{cfg['bg']};color:{cfg['color']};"
        f"border:1px solid {cfg['border']};font-size:.8em;font-weight:600'>{cfg['label']}</span>"
        for cfg in TAG_CFG.values()
    )
    return f"""
  <div id="commodity-history" class="commodity-section active">
    <div class="tab-panel active" style="max-width:1000px;margin:0 auto">
      <div class="part-label"><span class="part-badge">HISTORY</span>對話記錄 · Prompt &amp; Evolution History</div>

      <div class="card" style="margin-bottom:16px">
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:10px">
          {legend}
          <span style="color:var(--muted);font-size:.82em;margin-left:auto">共 {len(logs)} 筆記錄</span>
        </div>
        <div style="font-size:.84em;color:var(--text2)">
          每筆記錄標示商品歸屬，方便追蹤跨商品分析演進與 Prompt 歷史。
          記錄依日期由新至舊排列。新增記錄：append <code>data/logs.json</code> 後重跑 generate_site.py。
        </div>
      </div>

      <div class="card">
        {blocks}
      </div>
    </div>
  </div><!-- /commodity-history -->
"""


def _latest_logs_html(n: int = 5) -> str:
    """Hub 首頁的最新動態摘要（只列日期/tag/標題，點入 history.html 看全文）"""
    logs = _load_logs()[:n]
    rows = ""
    for e in logs:
        cfg = TAG_CFG.get(e["tag"], TAG_CFG["cross"])
        rows += (
            f"<div style='display:flex;align-items:baseline;gap:10px;padding:8px 0;"
            f"border-bottom:1px solid var(--border)'>"
            f"<span style='color:var(--muted);font-size:.82em;white-space:nowrap'>{e['date']}</span>"
            f"<span style='padding:1px 8px;border-radius:10px;background:{cfg['bg']};color:{cfg['color']};"
            f"border:1px solid {cfg['border']};font-size:.72em;font-weight:700;white-space:nowrap'>{cfg['label']}</span>"
            f"<span style='font-size:.9em'>{e['title']}</span></div>"
        )
    return rows


# ─── 頁面外殼（nav 單一來源）────────────────────────────────────────
NAV_LINKS = [
    ("xauusd.html",  "XAUUSD 黃金"),
    ("tx.html",      "TX 台指期"),
    ("shared.html",  "📊 跨商品分析"),
    ("sitemap.html", "🗺 網站地圖"),
    ("history.html", "📋 對話記錄"),
]


def _page(title: str, active_href: str, body: str) -> str:
    tabs = "\n".join(
        f'    <a class="commodity-tab{" active" if href == active_href else ""}" href="{href}">{label}</a>'
        for href, label in NAV_LINKS
    )
    return f"""<!DOCTYPE html>
<!-- ⚠️ 本檔由 generate_site.py 自動生成，禁止手改（會在下次生成時被覆蓋）。
     手寫內容請改 content/ 下的 fragment，對話記錄請改 data/logs.json，詳見 DEVELOPMENT.md -->
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="icon" type="image/svg+xml" href="favicon.svg">
<link rel="shortcut icon" href="favicon.svg">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>

<nav class="topnav">
  <div class="nav-brand"><a href="index.html" style="color:inherit;text-decoration:none">Trading Strategy Hub</a> <span>multi-commodity</span></div>
  <div class="commodity-tabs">
{tabs}
  </div>
  <div class="nav-meta">Updated {date.today().isoformat()}</div>
</nav>

{body}

<script src="assets/site.js"></script>
</body>
</html>
"""


# ─── 各頁面組裝 ──────────────────────────────────────────────────────
def build_xauusd() -> str:
    xauusd = COMMODITIES[0]
    xu_long  = _load_top3(ROOT / xauusd["long_dir"]  / "results.json")
    xu_short = _load_top3(ROOT / xauusd["short_dir"] / "results.json")
    xu_long_rows  = "\n".join(_exp_row_xauusd(r, "long")  for r in xu_long)
    xu_short_rows = "\n".join(_exp_row_xauusd(r, "short") for r in xu_short)
    vdata = _load_validation()

    body = f"""<div id="commodity-xauusd" class="commodity-section active">
  <div class="commodity-subnav">
    <button class="nav-main-tab active" onclick="showMain('xauusd-main-macro',this)">宏觀分析</button>
    <button class="nav-main-tab" onclick="showMain('xauusd-main-opt',this)">已確認策略</button>
    <button class="nav-main-tab" onclick="showMain('xauusd-main-exp',this)">實驗策略</button>
    <button class="nav-main-tab" onclick="showMain('xauusd-main-fvg',this)">🔍 FVG 策略</button>
    <button class="nav-main-tab" onclick="showMain('xauusd-main-validate',this)">筆記驗證</button>
    <button class="nav-main-tab" onclick="showMain('xauusd-main-weekly',this)">📊 週報分析</button>
    <button class="nav-main-tab" onclick="showMain('xauusd-main-h2',this)">📈 2026 H1/H2</button>
  </div>

{_xauusd_macro_with_manual()}
{_frag("xauusd/opt.html")}
{_xauusd_exp_html(xu_long_rows, xu_short_rows, xauusd)}
{_xauusd_fvg_html()}
{_xauusd_validation_html(vdata)}
{_frag("xauusd/weekly.html")}
{_frag("xauusd/h2.html")}
</div><!-- /commodity-xauusd -->"""
    return _page("XAUUSD 黃金 | Trading Strategy Hub", "xauusd.html", body)


def build_tx() -> str:
    tx = COMMODITIES[1]
    tx_long  = _load_top3(ROOT / tx["long_dir"]  / "results.json")
    tx_short = _load_top3(ROOT / tx["short_dir"] / "results.json")
    tx_long_rows  = "\n".join(_exp_row_tx(r, "long")  for r in tx_long)
    tx_short_rows = "\n".join(_exp_row_tx(r, "short") for r in tx_short)
    vdata = _load_validation()

    body = f"""<div id="commodity-tx" class="commodity-section active">
  <div class="commodity-subnav">
    <button class="nav-main-tab active" onclick="showMain('tx-main-macro',this)">宏觀分析</button>
    <button class="nav-main-tab" onclick="showMain('tx-main-confirmed',this)">已確認策略</button>
    <button class="nav-main-tab" onclick="showMain('tx-main-exp',this)">實驗策略</button>
    <button class="nav-main-tab" onclick="showMain('tx-main-validate',this)">筆記驗證</button>
    <button class="nav-main-tab" onclick="showMain('tx-main-zheng2',this)">📈 正二回測</button>
  </div>

{_tx_macro_html()}
{_tx_confirmed_html()}
{_tx_exp_html(tx_long_rows, tx_short_rows, tx)}
{_tx_validation_html(vdata)}
{_frag("tx/zheng2.html")}
</div><!-- /commodity-tx -->"""
    return _page("TX 台指期 | Trading Strategy Hub", "tx.html", body)


def build_shared() -> str:
    shared_html = _shared_analysis_html(_load_shared_results())
    # 拆頁後本頁的 commodity-section 需為 active
    shared_html = shared_html.replace(
        'id="commodity-shared" class="commodity-section"',
        'id="commodity-shared" class="commodity-section active"', 1)
    return _page("跨商品分析 | Trading Strategy Hub", "shared.html", shared_html)


def build_sitemap() -> str:
    sm = _frag("sitemap.html").replace(
        'id="commodity-sitemap" class="commodity-section"',
        'id="commodity-sitemap" class="commodity-section active"', 1)
    return _page("網站地圖 | Trading Strategy Hub", "sitemap.html", sm)


def build_history() -> str:
    return _page("對話記錄 | Trading Strategy Hub", "history.html", _history_section())


def build_hub() -> str:
    logs = _load_logs()
    body = f"""<div class="commodity-section active" style="max-width:1000px;margin:0 auto;padding:20px 16px">

  <div class="part-label"><span class="part-badge">HUB</span>多商品量化策略分析工具箱</div>

  <div class="hub-grid">
    <div class="hub-card" style="border-top:4px solid #2563eb">
      <h2>🟡 XAUUSD 黃金</h2>
      <div class="sub">黃金/美元 · 30m · S1/S2-RSI/S2-Hammer 三策略 + 20L/20S 實驗 + FVG</div>
      <div class="hub-links">
        <a class="report-link" href="xauusd.html">主頁 →</a>
        <a class="report-link" href="xauusd.html#xauusd-main-opt">已確認策略</a>
        <a class="report-link" href="xauusd.html#xauusd-main-fvg">FVG</a>
        <a class="report-link" href="xauusd.html#xauusd-main-h2">2026 H1/H2</a>
      </div>
    </div>
    <div class="hub-card" style="border-top:4px solid #1565c0">
      <h2>🔵 TX 台指期</h2>
      <div class="sub">小台 MTX · 30m · 20L/20S 實驗 · 宏觀分析 · 正二回測</div>
      <div class="hub-links">
        <a class="report-link" href="tx.html">主頁 →</a>
        <a class="report-link" href="tx.html#tx-main-confirmed">已確認策略</a>
        <a class="report-link" href="tx.html#tx-main-zheng2">正二回測</a>
      </div>
    </div>
    <div class="hub-card" style="border-top:4px solid #7c3aed">
      <h2>📊 跨商品分析</h2>
      <div class="sub">整點熱力圖 · 30m RSI 濾鏡（XAUUSD + TX 共同框架）</div>
      <div class="hub-links">
        <a class="report-link" href="shared.html">主頁 →</a>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:8px">
    <div class="card-title">📋 最新動態（共 {len(logs)} 筆記錄）</div>
    {_latest_logs_html(5)}
    <div style="margin-top:12px"><a class="report-link" href="history.html">完整對話記錄 →</a></div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="card-title">🗺 快速導覽</div>
    <div class="hub-links" style="margin-top:8px">
      <a class="report-link" href="sitemap.html">網站地圖（所有報告清單）</a>
      <a class="report-link" href="xauusd/macro_report.html">XAUUSD 宏觀完整報告</a>
      <a class="report-link" href="tx/macro_report.html">TX 宏觀完整報告</a>
    </div>
  </div>

</div>"""
    return _page("Trading Strategy Hub", "", body)


# ─── Main ────────────────────────────────────────────────────────────
PAGES = {
    "index":   ("index.html",   build_hub),
    "xauusd":  ("xauusd.html",  build_xauusd),
    "tx":      ("tx.html",      build_tx),
    "shared":  ("shared.html",  build_shared),
    "history": ("history.html", build_history),
    "sitemap": ("sitemap.html", build_sitemap),
}


def main():
    ap = argparse.ArgumentParser(description="生成 Trading Strategy Hub 靜態頁面")
    ap.add_argument("--page", choices=list(PAGES), help="只生成指定頁面（預設全部）")
    args = ap.parse_args()

    targets = [args.page] if args.page else list(PAGES)
    for key in targets:
        fname, builder = PAGES[key]
        out = ROOT / fname
        out.write_text(builder(), encoding="utf-8")
        print(f"  ✅ {fname} 已生成（{out.stat().st_size:,} bytes）")
    print("完成。提醒：生成後用 git diff --stat 檢查變動幅度是否符合預期。")


if __name__ == "__main__":
    main()
