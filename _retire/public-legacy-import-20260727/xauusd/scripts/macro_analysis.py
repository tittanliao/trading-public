"""
XAUUSD 宏觀分析 — 以週線資料為基礎
1. 月度方向統計（月初買、月底賣的勝率與平均漲跌%）
2. 季節性分析（每個月份的歷史偏向）
3. 週內結構分析（每月第 1–5 週的強弱規律）
4. 年 × 月 熱力圖
5. 輸出自含式 HTML 報告
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

WEEKLY_CSV  = Path("csv/FX_IDC_XAUUSD, 1W.csv")
OUTPUT_HTML = Path("macro_report.html")


# ── 載入週線 ──────────────────────────────────────────────────────────────
def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(WEEKLY_CSV)
    df.columns = [c.strip() for c in df.columns]

    def parse_time(t):
        t = str(t).strip()
        t = re.sub(r'[+-]\d{2}:\d{2}$', '', t)
        return pd.to_datetime(t)

    df['time'] = df['time'].apply(parse_time)
    df = df.sort_values('time').reset_index(drop=True)

    for c in ['open', 'high', 'low', 'close', 'RSI', 'RSI-based MA']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['year']  = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['week_of_month'] = df.groupby(['year', 'month']).cumcount() + 1
    return df


# ── 月度聚合 ──────────────────────────────────────────────────────────────
def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['year', 'month'])
    monthly = grp.agg(
        week_open  = ('open',  'first'),
        week_close = ('close', 'last'),
        high       = ('high',  'max'),
        low        = ('low',   'min'),
        rsi_start  = ('RSI',   'first'),
        rsi_end    = ('RSI',   'last'),
        n_weeks    = ('close', 'count'),
    ).reset_index()

    monthly['chg_pct']  = (monthly['week_close'] - monthly['week_open']) / monthly['week_open'] * 100
    monthly['chg_usd']  = monthly['week_close'] - monthly['week_open']
    monthly['bullish']  = monthly['chg_pct'] > 0
    monthly['date']     = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
    return monthly


# ── 季節性 ────────────────────────────────────────────────────────────────
def seasonality(monthly: pd.DataFrame) -> pd.DataFrame:
    month_names = ['一月','二月','三月','四月','五月','六月',
                   '七月','八月','九月','十月','十一月','十二月']
    rows = []
    for m in range(1, 13):
        sub = monthly[monthly['month'] == m]
        if len(sub) == 0:
            continue
        win_rate = sub['bullish'].mean() * 100
        avg_pct  = sub['chg_pct'].mean()
        med_pct  = sub['chg_pct'].median()
        avg_usd  = sub['chg_usd'].mean()
        best     = sub['chg_pct'].max()
        worst    = sub['chg_pct'].min()
        bias     = 'LONG' if win_rate >= 55 else ('SHORT' if win_rate <= 45 else 'NEUTRAL')
        rows.append(dict(
            month=m, month_name=month_names[m-1],
            n=len(sub), win_rate=win_rate,
            avg_pct=avg_pct, med_pct=med_pct,
            avg_usd=avg_usd, best=best, worst=worst,
            bias=bias
        ))
    return pd.DataFrame(rows)


# ── 週內結構 ──────────────────────────────────────────────────────────────
def week_in_month_stats(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2['week_chg_pct'] = (df2['close'] - df2['open']) / df2['open'] * 100
    df2['week_bull'] = df2['week_chg_pct'] > 0
    grp = df2.groupby('week_of_month').agg(
        n         = ('week_chg_pct', 'count'),
        win_rate  = ('week_bull', 'mean'),
        avg_pct   = ('week_chg_pct', 'mean'),
        med_pct   = ('week_chg_pct', 'median'),
    ).reset_index()
    grp['win_rate'] = grp['win_rate'] * 100
    return grp[grp['week_of_month'] <= 5]


# ── 近 12 個月 ────────────────────────────────────────────────────────────
def recent_12m(monthly: pd.DataFrame) -> pd.DataFrame:
    return monthly.sort_values('date').tail(12).copy()


# ── HTML 小工具 ───────────────────────────────────────────────────────────
def color_bias(bias: str):
    if bias == 'LONG':    return '#1a7a4a', '偏多'
    if bias == 'SHORT':   return '#c0392b', '偏空'
    return '#555', '中性'


def pct_bar(val: float, max_val: float = 5.0,
            pos_color='#27ae60', neg_color='#e74c3c') -> str:
    width = min(abs(val) / max_val * 100, 100)
    color = pos_color if val >= 0 else neg_color
    return (f'<div style="display:inline-block;width:{width:.1f}px;'
            f'height:10px;background:{color};vertical-align:middle;"></div>')


# ── 熱力圖（年 × 月，% 漲跌）────────────────────────────────────────────
def render_monthly_heatmap(monthly: pd.DataFrame) -> str:
    years = sorted(monthly['year'].unique())

    def chg_to_color(v: float) -> str:
        if np.isnan(v):  return '#2a2a2a'
        if v >  5:       return '#0a5c2e'
        if v >  2:       return '#1a7a4a'
        if v >  0:       return '#27ae60'
        if v > -2:       return '#e74c3c'
        if v > -5:       return '#c0392b'
        return '#7b241c'

    month_abbr = ['1','2','3','4','5','6','7','8','9','10','11','12']
    header = '<tr><th>年份</th>' + ''.join(f'<th>{m}</th>' for m in month_abbr) + '</tr>'
    body = ''
    for y in years:
        row = f'<tr><td><b>{y}</b></td>'
        for m in range(1, 13):
            sub = monthly[(monthly['year'] == y) & (monthly['month'] == m)]
            if len(sub) == 0:
                row += '<td style="background:#1e1e1e">—</td>'
            else:
                v   = sub['chg_pct'].values[0]
                usd = sub['chg_usd'].values[0]
                bg  = chg_to_color(v)
                row += (f'<td style="background:{bg};color:#fff;font-size:11px" '
                        f'title="${usd:+.0f}">{v:+.1f}%</td>')
        row += '</tr>'
        body += row

    return f"""
    <table class="heatmap-table">
      <thead>{header}</thead>
      <tbody>{body}</tbody>
    </table>
    <p style="font-size:12px;color:#888">
      顏色：深綠 &gt;5% ／ 綠 &gt;2% ／ 淺綠 &gt;0% ／ 紅 &lt;0% ／ 深紅 &lt;-5%。
      滑鼠懸停顯示 USD 漲跌。
    </p>"""


def render_seasonality_table(sea: pd.DataFrame) -> str:
    rows = ''
    for _, r in sea.iterrows():
        bias_color, bias_label = color_bias(r['bias'])
        wr_color = '#27ae60' if r['win_rate'] >= 55 else ('#e74c3c' if r['win_rate'] <= 45 else '#888')
        rows += f"""
        <tr>
          <td><b>{r['month_name']}</b></td>
          <td>{r['n']}</td>
          <td style="color:{wr_color};font-weight:bold">{r['win_rate']:.1f}%</td>
          <td>{r['avg_pct']:+.2f}% {pct_bar(r['avg_pct'])}</td>
          <td>${r['avg_usd']:+.0f}</td>
          <td>{r['med_pct']:+.2f}%</td>
          <td style="color:#27ae60">{r['best']:+.1f}%</td>
          <td style="color:#e74c3c">{r['worst']:+.1f}%</td>
          <td><span style="background:{bias_color};color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{bias_label}</span></td>
        </tr>"""
    return f"""
    <table class="data-table">
      <thead><tr>
        <th>月份</th><th>樣本數</th><th>月勝率</th><th>平均漲跌%</th>
        <th>平均漲跌$</th><th>中位數%</th><th>最佳</th><th>最差</th><th>偏向</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def render_week_in_month_table(wim: pd.DataFrame) -> str:
    week_label = {1:'第1週（月初）',2:'第2週',3:'第3週',4:'第4週',5:'第5週（月底）'}
    rows = ''
    for _, r in wim.iterrows():
        wr = r['win_rate']
        wr_color = '#27ae60' if wr >= 55 else ('#e74c3c' if wr <= 45 else '#888')
        rows += f"""
        <tr>
          <td><b>{week_label.get(int(r['week_of_month']), '?')}</b></td>
          <td>{r['n']}</td>
          <td style="color:{wr_color};font-weight:bold">{wr:.1f}%</td>
          <td>{r['avg_pct']:+.2f}% {pct_bar(r['avg_pct'], max_val=2.0)}</td>
          <td>{r['med_pct']:+.2f}%</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <thead><tr>
        <th>週次</th><th>樣本數</th><th>週勝率</th><th>平均漲跌%</th><th>中位數%</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def render_recent_table(rec: pd.DataFrame) -> str:
    rows = ''
    for _, r in rec.iterrows():
        color = '#27ae60' if r['bullish'] else '#e74c3c'
        sign  = '▲' if r['bullish'] else '▼'
        rows += f"""
        <tr>
          <td>{r['year']}/{r['month']:02d}</td>
          <td>${r['week_open']:.0f}</td>
          <td>${r['week_close']:.0f}</td>
          <td style="color:{color};font-weight:bold">{sign} {r['chg_pct']:+.2f}% (${r['chg_usd']:+.0f})</td>
          <td>{r['rsi_start']:.1f}</td>
          <td>{r['rsi_end']:.1f}</td>
        </tr>"""
    return f"""
    <table class="data-table">
      <thead><tr>
        <th>年/月</th><th>月初開盤</th><th>月底收盤</th><th>月漲跌</th>
        <th>RSI（月初）</th><th>RSI（月底）</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def render_dashboard(monthly: pd.DataFrame, sea: pd.DataFrame) -> str:
    latest = monthly.iloc[-1]
    cur_month = int(latest['month'])
    cur_year  = int(latest['year'])
    sea_row   = sea[sea['month'] == cur_month].iloc[0]
    bias_color, bias_label = color_bias(sea_row['bias'])

    rsi_state = ''
    if not pd.isna(latest['rsi_end']):
        rsi = latest['rsi_end']
        if rsi > 70:    rsi_state = f'RSI {rsi:.1f} — 超買區，留意回落風險'
        elif rsi < 30:  rsi_state = f'RSI {rsi:.1f} — 超賣區，留意反彈機會'
        else:           rsi_state = f'RSI {rsi:.1f} — 中性區間'

    return f"""
    <div class="dashboard-card">
      <h3>📍 當前月份：{cur_year} / {cur_month:02d}</h3>
      <p><b>季節性偏向（歷史 {sea_row['n']} 個樣本）：</b>
         <span style="background:{bias_color};color:#fff;padding:2px 10px;border-radius:4px;font-weight:bold">{bias_label}</span>
      </p>
      <p>月勝率：<b>{sea_row['win_rate']:.1f}%</b>　平均漲跌：<b>{sea_row['avg_pct']:+.2f}%（${sea_row['avg_usd']:+.0f}）</b>　中位數：<b>{sea_row['med_pct']:+.2f}%</b></p>
      <p>{rsi_state}</p>
      <hr style="border-color:#333">
      <p style="color:#aaa;font-size:13px">
        操作思路：先確認<b>月度方向偏{bias_label}</b>作為宏觀基準，<br>
        再用週線 RSI、BB 位置判斷當週進場時機（順月度方向操作）。
      </p>
    </div>"""


# ── HTML 報告 ─────────────────────────────────────────────────────────────
def build_html(df: pd.DataFrame, monthly: pd.DataFrame,
               sea: pd.DataFrame, wim: pd.DataFrame) -> str:
    rec          = recent_12m(monthly)
    today        = datetime.now().strftime('%Y-%m-%d')
    total_months = len(monthly)
    overall_wr   = monthly['bullish'].mean() * 100
    avg_pct      = monthly['chg_pct'].mean()

    heatmap      = render_monthly_heatmap(monthly)
    season_table = render_seasonality_table(sea)
    wim_table    = render_week_in_month_table(wim)
    recent_table = render_recent_table(rec)
    dashboard    = render_dashboard(monthly, sea)

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>XAUUSD 宏觀分析 — 月度方向 + 週內結構</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
  h1 {{ color: #f0c040; margin-bottom: 4px; }}
  h2 {{ color: #7ec8e3; margin: 28px 0 12px; border-left: 4px solid #7ec8e3; padding-left: 10px; }}
  h3 {{ color: #e0e0e0; margin-bottom: 8px; }}
  .subtitle {{ color: #888; margin-bottom: 24px; font-size: 14px; }}
  .stats-row {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 16px 24px; min-width: 160px; }}
  .stat-card .val {{ font-size: 28px; font-weight: bold; color: #f0c040; }}
  .stat-card .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .dashboard-card {{ background: #1a2a1a; border: 1px solid #2a4a2a; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; }}
  .dashboard-card p {{ margin: 6px 0; line-height: 1.6; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
  table.data-table th {{ background: #1e1e1e; color: #7ec8e3; padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; font-size: 13px; }}
  table.data-table td {{ padding: 7px 12px; border-bottom: 1px solid #222; font-size: 13px; }}
  table.data-table tr:hover {{ background: #1a1a1a; }}
  table.heatmap-table {{ border-collapse: collapse; font-size: 12px; }}
  table.heatmap-table th {{ background: #1e1e1e; color: #7ec8e3; padding: 6px 10px; border: 1px solid #333; }}
  table.heatmap-table td {{ padding: 5px 8px; border: 1px solid #1a1a1a; text-align: center; min-width: 48px; }}
  .section {{ margin-bottom: 32px; }}
  hr {{ border: none; border-top: 1px solid #333; margin: 20px 0; }}
  .note {{ color: #888; font-size: 12px; margin-top: 6px; }}
</style>
</head>
<body>
<a href="../index.html" style="display:inline-block;margin:12px 24px 0;color:#93c5fd;font-size:13px;text-decoration:none;font-weight:600">← 返回 Trading Hub</a>

<h1>XAUUSD 黃金 — 宏觀分析報告</h1>
<p class="subtitle">週線資料 · {monthly['date'].min().strftime('%Y-%m')} ~ {monthly['date'].max().strftime('%Y-%m')} · 共 {total_months} 個月 · 生成時間：{today}</p>

<div class="stats-row">
  <div class="stat-card">
    <div class="val">{overall_wr:.1f}%</div>
    <div class="lbl">整體月勝率（月初買月底賣）</div>
  </div>
  <div class="stat-card">
    <div class="val" style="color:{'#27ae60' if avg_pct>0 else '#e74c3c'}">{avg_pct:+.2f}%</div>
    <div class="lbl">平均月漲跌幅</div>
  </div>
  <div class="stat-card">
    <div class="val">{total_months}</div>
    <div class="lbl">總樣本月數</div>
  </div>
  <div class="stat-card">
    <div class="val">{len(df)}</div>
    <div class="lbl">週線 K 棒數</div>
  </div>
</div>

{dashboard}

<div class="section">
  <h2>季節性分析 — 每個月份的歷史偏向</h2>
  <p class="note">月勝率 ≥55% → 偏多，≤45% → 偏空，中間 → 中性。</p>
  {season_table}
</div>

<div class="section">
  <h2>月度熱力圖 — 歷年每月漲跌%</h2>
  {heatmap}
</div>

<div class="section">
  <h2>週內結構 — 每月第幾週最強 / 最弱</h2>
  <p class="note">所有月份合併，看月初、月中、月底哪一週的多頭勝率最高。</p>
  {wim_table}
</div>

<div class="section">
  <h2>近 12 個月回顧</h2>
  {recent_table}
</div>

</body>
</html>"""


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    print("載入週線資料...")
    df = load_weekly()
    # 過濾掉 1980 年以前的極早期資料（資料品質較差）
    df = df[df['year'] >= 1980].reset_index(drop=True)
    print(f"  {len(df)} 根週線，{df['time'].min().date()} ~ {df['time'].max().date()}")

    print("聚合月度資料...")
    monthly = build_monthly(df)
    print(f"  {len(monthly)} 個月")

    print("計算季節性...")
    sea = seasonality(monthly)

    print("計算週內結構...")
    wim = week_in_month_stats(df)

    print("生成 HTML 報告...")
    html = build_html(df, monthly, sea, wim)
    OUTPUT_HTML.write_text(html, encoding='utf-8')
    print(f"✓ 報告已輸出：{OUTPUT_HTML.resolve()}")

    print("\n=== 季節性偏向摘要 ===")
    print(f"{'月份':<6} {'勝率':>6} {'平均%':>8} {'偏向':>6}")
    print("─" * 32)
    for _, r in sea.iterrows():
        print(f"{r['month_name']:<6} {r['win_rate']:>5.1f}% {r['avg_pct']:>+7.2f}%  {r['bias']}")

    print("\n=== 週內結構 ===")
    wlabel = {1:'第1週',2:'第2週',3:'第3週',4:'第4週',5:'第5週'}
    for _, r in wim.iterrows():
        print(f"{wlabel[int(r['week_of_month'])]}: 勝率 {r['win_rate']:.1f}%  平均 {r['avg_pct']:+.2f}%")


if __name__ == '__main__':
    main()
