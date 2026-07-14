#!/usr/bin/env python3
"""
run_fvg_sensitivity.py — FVG 最佳參數鄰域敏感度測試
==============================================================================
輸出：xauusd/XAUUSD-FVG-Strategy/report_sensitivity.html

背景：run_fvg_experiments.py 對 7,560 組參數組合做網格搜尋，挑出單點最佳值
（Long: fvg_min=0.20%/fvg_max=20bars/SL=1.5%Fixed/TP1=0.5R/TP2=2.0R/TB=72，
 WR66.7% PF1.660；Short: fvg_min=0.20%/fvg_max=50bars/SL=0.8%Fixed/TP1=0.5R/
 TP2=3.0R/TB=48，WR66.7% PF2.741）。7,560 次嘗試裡選最高分，天然有「多重比較」
 風險——就算所有參數組合都只是雜訊，最佳的那組看起來也會表現特別好。

本腳本不重新網格搜尋，而是圍繞「已挑出的最佳點」做鄰域測試：
  1. One-at-a-time（OAT）：6 個參數各自單獨 ±20%，其餘固定在最佳值，
     看單一參數的擾動會不會讓績效崩潰。
  2. 聯合鄰域（Joint neighborhood）：6 個參數同時在 {-20%, 0%, +20%} 三個
     level 做全組合（3^6=729 組），檢驗「最佳點」是不是鄰域裡的孤立尖峰
     （只有那一個精確組合表現好，稍微移動就崩潰 = 過擬合訊號），還是
     鄰域裡普遍都不錯（= 真實 edge 訊號）。

判斷標準：
  - 若 ±20% 鄰域內 PF 大幅崩壞（例如跌破 1.2）→ 該參數對雜訊敏感，過擬合風險高
  - 若聯合鄰域裡有效組合（PF≥1.2）佔比過低 → 最佳點可能是孤立尖峰
  - 資料與 run_fvg_experiments.py 完全相同（xauusd/csv/ 3個月 30m），
    這裡不是換資料驗證，是同一份資料上測「參數穩定性」

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_fvg_sensitivity.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_fvg_experiments import load_price, run_backtest, CSV_30M, MIN_TRADES  # noqa: E402

OUTPUT_DIR = Path("xauusd/XAUUSD-FVG-Strategy")
OUT_HTML   = OUTPUT_DIR / "report_sensitivity.html"

# 從 optimization_results.json 抄下來的最佳點（20260705 網格搜尋結果，見 project_context.md）
BEST = {
    "Long": dict(direction="Long", fvg_min_pct=0.20, fvg_max_bars=20,
                 sl_type="Fixed %", sl_buffer=0.05, sl_pct=1.5,
                 tp1_r=0.5, tp2_r=2.0, tb=72),
    "Short": dict(direction="Short", fvg_min_pct=0.20, fvg_max_bars=50,
                  sl_type="Fixed %", sl_buffer=0.05, sl_pct=0.8,
                  tp1_r=0.5, tp2_r=3.0, tb=48),
}

# 要做敏感度測試的參數（排除 direction / sl_type，這兩個是類別型不是數值型）
NUMERIC_PARAMS = ["fvg_min_pct", "fvg_max_bars", "sl_pct", "tp1_r", "tp2_r", "tb"]
INT_PARAMS = {"fvg_max_bars", "tb"}

PF_ROBUST_THRESHOLD = 1.2  # 呼應 monthly_checklist.md 轉向訊號門檻

print("載入 30m 價格資料（與 run_fvg_experiments.py 相同資料）...")
price = load_price(CSV_30M)
print(f"  {len(price)} bars ({price['time'].iloc[0].date()} → {price['time'].iloc[-1].date()})")


def perturb(base: dict, param: str, pct: float) -> dict:
    p = dict(base)
    v = base[param] * (1 + pct)
    p[param] = int(round(v)) if param in INT_PARAMS else round(v, 3)
    return p


def run_one(params: dict) -> dict | None:
    kw = {k: v for k, v in params.items() if k != "direction"}
    return run_backtest(price, direction=params["direction"], **kw)


# ── 1. OAT（單參數 ±20%）───────────────────────────────────────────────────
def oat_sweep(direction: str) -> pd.DataFrame:
    base = BEST[direction]
    rows = []
    base_r = run_one(base)
    rows.append(dict(param="（Baseline，全部最佳值）", pct=0, **base_r,
                      value=None))
    for param in NUMERIC_PARAMS:
        for pct in (-0.2, 0.2):
            p = perturb(base, param, pct)
            r = run_one(p)
            rows.append(dict(
                param=param, pct=pct, value=p[param],
                trades=r["trades"] if r else 0,
                win_rate=r["win_rate"] if r else float("nan"),
                profit_factor=r["profit_factor"] if r else float("nan"),
                net_pnl_pct=r["net_pnl_pct"] if r else float("nan"),
            ))
    return pd.DataFrame(rows)

print("\n跑 OAT（單參數 ±20%）敏感度...")
oat_long  = oat_sweep("Long")
oat_short = oat_sweep("Short")
print(f"  Long: {len(oat_long)} 組, Short: {len(oat_short)} 組")

# ── 2. 聯合鄰域（3^6 = 729 組，全參數同時 -20%/0%/+20%）──────────────────────
def joint_neighborhood(direction: str) -> pd.DataFrame:
    base = BEST[direction]
    levels = [-0.2, 0.0, 0.2]
    rows = []
    for combo in itertools.product(levels, repeat=len(NUMERIC_PARAMS)):
        p = dict(base)
        for param, pct in zip(NUMERIC_PARAMS, combo):
            v = base[param] * (1 + pct)
            p[param] = int(round(v)) if param in INT_PARAMS else round(v, 3)
        r = run_one(p)
        if r:
            rows.append(dict(**r, is_baseline=(combo == (0.0,) * len(NUMERIC_PARAMS))))
    return pd.DataFrame(rows)

print("\n跑聯合鄰域（3^6=729 組合）...")
joint_long  = joint_neighborhood("Long")
joint_short = joint_neighborhood("Short")
print(f"  Long: {len(joint_long)}/729 組有效（≥{MIN_TRADES}筆）")
print(f"  Short: {len(joint_short)}/729 組有效（≥{MIN_TRADES}筆）")


def joint_summary(df: pd.DataFrame, direction: str) -> dict:
    base_row = df[df["is_baseline"]]
    base_pf = base_row["profit_factor"].iloc[0] if len(base_row) else float("nan")
    base_rank_pct = (df["profit_factor"] < base_pf).mean() * 100 if len(df) else float("nan")
    robust_frac = (df["profit_factor"] >= PF_ROBUST_THRESHOLD).mean() * 100 if len(df) else 0.0
    return dict(
        direction=direction, n_valid=len(df), n_total=729,
        pf_min=df["profit_factor"].min(), pf_median=df["profit_factor"].median(),
        pf_max=df["profit_factor"].max(), pf_baseline=base_pf,
        baseline_percentile=base_rank_pct, robust_frac=robust_frac,
    )

sum_long  = joint_summary(joint_long, "Long")
sum_short = joint_summary(joint_short, "Short")
print(f"\nLong  鄰域摘要: {sum_long}")
print(f"Short 鄰域摘要: {sum_short}")

# ── HTML 輸出 ────────────────────────────────────────────────────────────────
def fmt(v, f="{:.3f}"):
    return "—" if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))) else f.format(v)

def oat_table(df: pd.DataFrame, base_pf: float) -> str:
    body = ""
    for _, r in df.iterrows():
        if r["param"].startswith("（"):
            body += (f"<tr style='background:rgba(56,189,248,.12)'><td colspan='2'><strong>{r['param']}</strong></td>"
                     f"<td>{int(r['trades'])}</td><td>{r['win_rate']:.1f}%</td>"
                     f"<td><strong>{r['profit_factor']:.3f}</strong></td><td>{r['net_pnl_pct']:+.2f}%</td></tr>")
            continue
        pf = r["profit_factor"]
        collapse = (not np.isfinite(pf)) or pf < PF_ROBUST_THRESHOLD or r["trades"] < MIN_TRADES
        color = "#ef4444" if collapse else ("#22c55e" if pf >= base_pf else "#e2e8f0")
        sign = f"{r['pct']*100:+.0f}%"
        body += (f"<tr><td>{r['param']}</td><td>{sign} → {r['value']}</td>"
                 f"<td>{int(r['trades'])}</td><td>{fmt(r['win_rate'],'{:.1f}')}%</td>"
                 f"<td style='color:{color}'><strong>{fmt(pf)}</strong></td>"
                 f"<td>{fmt(r['net_pnl_pct'],'{:+.2f}')}%</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>參數</th><th>擾動</th><th>筆數</th>"
            f"<th>勝率</th><th>PF</th><th>淨盈虧%</th></tr></thead><tbody>{body}</tbody></table>")

def joint_card(s: dict) -> str:
    verdict_ok = s["robust_frac"] >= 50 and s["baseline_percentile"] <= 90
    cls = "good" if verdict_ok else "warn"
    verdict = ("最佳點附近的鄰域普遍表現不錯，不是孤立尖峰，過擬合風險較低。"
               if verdict_ok else
               "最佳點看起來像鄰域裡的孤立尖峰或鄰域整體偏弱，過擬合風險偏高，"
               "建議降低對這組精確數字的信心，優先看鄰域內普遍有效的參數範圍而非單點。")
    return f"""
<div class="card">
<div class="part">聯合鄰域摘要 — {s['direction']}</div>
<h2>729 組合中，最佳點是孤立尖峰還是普遍有效？</h2>
<table class='tbl'>
<tr><td>有效組合數（≥{MIN_TRADES}筆）</td><td>{s['n_valid']} / {s['n_total']}</td></tr>
<tr><td>鄰域 PF 範圍</td><td>{fmt(s['pf_min'])} ~ {fmt(s['pf_max'])}（中位數 {fmt(s['pf_median'])}）</td></tr>
<tr><td>最佳點（Baseline）PF</td><td><strong>{fmt(s['pf_baseline'])}</strong></td></tr>
<tr><td>Baseline 在鄰域中的百分位</td><td>贏過鄰域內 {fmt(s['baseline_percentile'],'{:.0f}')}% 的組合</td></tr>
<tr><td>鄰域內 PF≥{PF_ROBUST_THRESHOLD} 佔比（穩健度）</td><td><strong>{fmt(s['robust_frac'],'{:.0f}')}%</strong></td></tr>
</table>
<div class="{cls}"><strong>{'✅' if verdict_ok else '⚠️'}</strong> {verdict}</div>
</div>
"""

CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
:root{--green:#22c55e;--yellow:#f59e0b;--blue:#38bdf8;--muted:#94a3b8;--border:#334155}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:1.6em;margin-bottom:6px;color:#f8fafc}
h2{font-size:1.1em;color:#38bdf8;margin:8px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.part{font-size:.75em;letter-spacing:.15em;color:#f59e0b;font-weight:700;text-transform:uppercase}
.card{background:#1e293b;border:1px solid var(--border);border-radius:10px;padding:22px;margin-bottom:18px}
.tbl{width:100%;border-collapse:collapse;font-size:.85em;margin:10px 0}
.tbl th{background:#0f172a;color:var(--muted);padding:9px 12px;text-align:left;border-bottom:1px solid var(--border)}
.tbl td{padding:8px 12px;border-bottom:1px solid rgba(51,65,85,.5)}
.note{font-size:.82em;color:var(--muted);margin-top:8px}
.warn{background:rgba(245,158,11,.08);border-left:3px solid var(--yellow);padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
.good{background:rgba(34,197,94,.08);border-left:3px solid var(--green);padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
</style>
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FVG 最佳參數鄰域敏感度測試</title>{CSS}</head>
<body>
<div style="max-width:1100px;margin:0 auto 14px"><a href="../index.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="wrap">
<h1>FVG 最佳參數 <span style="color:#f59e0b">鄰域敏感度</span> 測試</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
資料與 run_fvg_experiments.py 完全相同（xauusd/csv/ 30m，{price['time'].iloc[0].date()} → {price['time'].iloc[-1].date()}）·
測的是「參數穩定性」不是「換資料驗證」</p>

<div class="warn">
<strong>為什麼要做這個測試：</strong>7,560 組參數裡選分數最高的一組，天然有多重比較問題——
就算全是雜訊，最高分那組看起來也會很好看。這裡圍繞最佳點做 ±20% 擾動，
檢查它是「鄰域普遍有效」的真實 edge，還是「只有這個精確數字才有效」的過擬合尖峰。
</div>

<!-- Long -->
<div class="card">
<div class="part">Long — OAT 單參數 ±20%</div>
<h2>最佳點：FVG Min 0.20% / Max 20bars / SL 1.5% Fixed / TP1 0.5R / TP2 2.0R / TB 72</h2>
{oat_table(oat_long, oat_long.iloc[0]["profit_factor"])}
<p class="note">紅色 = PF 跌破 {PF_ROBUST_THRESHOLD} 或筆數不足 {MIN_TRADES}（視為崩壞）；綠色 = 該方向擾動後反而更好。</p>
</div>
{joint_card(sum_long)}

<!-- Short -->
<div class="card">
<div class="part">Short — OAT 單參數 ±20%</div>
<h2>最佳點：FVG Min 0.20% / Max 50bars / SL 0.8% Fixed / TP1 0.5R / TP2 3.0R / TB 48</h2>
{oat_table(oat_short, oat_short.iloc[0]["profit_factor"])}
<p class="note">紅色 = PF 跌破 {PF_ROBUST_THRESHOLD} 或筆數不足 {MIN_TRADES}（視為崩壞）；綠色 = 該方向擾動後反而更好。</p>
</div>
{joint_card(sum_short)}

<p class="note" style="margin-top:8px">
方法：OAT（One-At-a-Time）每次只擾動一個參數 ±20%，其餘固定在最佳值；
聯合鄰域對 6 個參數同時取 {{-20%, 0%, +20%}} 三個 level 做全組合（3^6=729），
統計最佳點在整個鄰域中的相對位置與鄰域整體穩健度。與 run_fvg_experiments.py
使用同一個 run_backtest() 引擎，同一份資料，僅參數不同。
</p>

<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD FVG 最佳參數鄰域敏感度測試 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 敏感度測試報告已生成：{OUT_HTML}")
