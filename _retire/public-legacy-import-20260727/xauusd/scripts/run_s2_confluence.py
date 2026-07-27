"""
run_s2_confluence.py — S2-RSI × S2-Hammer 合流分析：合併假設檢驗
==============================================================================
輸出：xauusd/XAUUSD-Long-S2-Hammer/report_s2_confluence.html

背景（20260711）：使用者考慮把 S2-RSI 與 S2-Hammer 合併（因單獨勝率 42-44% 偏低）。
合併若有意義，前提是「兩者同時觸發」的交易品質應高於單獨觸發——本腳本用兩份
真實逐筆交易 CSV 直接檢驗這個前提。

結論先講：**前提不成立，方向完全相反。**
S2-Hammer 錘頭在 ±12h 內「沒有」S2-RSI 訊號時：WR 50.4%、PF 2.14（n=135）
S2-Hammer 錘頭與 S2-RSI 同時觸發時：      WR 30.8%、PF 1.03（n=65，打平）
且時間切分驗證兩段皆成立，OOS 段合流組 PF 僅 0.57（虧損）。

機制解釋（與既有研究吻合）：S2-RSI 觸發 = RSI 深度超賣/背離環境。
2026-07-02 的 Regime 四象限分析已發現 Z-Score < -2.5 極端超賣是陷阱區
（勝率僅 20%，極端超賣代表趨勢延伸而非反轉）。錘頭出現在 RSI 同時深度超賣
的時刻＝接刀接在瀑布中段；錘頭單獨出現才是健康回調的反轉點。

行動建議：不合併。改為 S2-Hammer 加「S2-RSI 互斥過濾器」（S2-RSI 環境成立 → S2-Hammer 跳過），
S2-RSI 考慮降級為環境 veto 濾網（待 S2 真實逐筆歸因後決定）。

注意：此為事後挖掘規則，雖通過時間切分檢驗且有獨立機制支撐，
上實盤前仍需：①確認與 V2.4/V2.3 的 Z-Score 過濾器是否同效（很可能重疊）
②Pine 實作後 TradingView 驗證。

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s2_confluence.py
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _set_cjk_font():
    for name in ("PingFang HK", "Arial Unicode MS", "STHeiti", "Heiti TC"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
_set_cjk_font()
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "xauusd"))

from analysis import loader

S2A_CSV  = ROOT / "xauusd/XAUUSD-Long-S2-RSI/S2-Hybrid-V2.0_FX_IDC_XAUUSD_2026-04-26.csv"
S2B_CSV  = ROOT / "xauusd/XAUUSD-Long-S2-Hammer/S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-04-26.csv"
OUT_HTML = ROOT / "xauusd/XAUUSD-Long-S2-Hammer/report_s2_confluence.html"

WINDOWS_H = [2, 6, 12]   # 合流判定窗口（小時）
MAIN_W_H  = 12           # 主結論採用的窗口

# ── 1. 載入 ──────────────────────────────────────────────────────────────────
print("載入 S2-RSI / S2-Hammer 真實逐筆交易...")
a = loader.load_trades(S2A_CSV)
b = loader.load_trades(S2B_CSV)
print(f"  S2-RSI: {len(a)} 筆（{a.entry_time.min().date()} → {a.entry_time.max().date()}）")
print(f"  S2-Hammer: {len(b)} 筆（{b.entry_time.min().date()} → {b.entry_time.max().date()}）")


def stat_block(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), avg=float("nan"))
    wins = df[df.result == "win"]
    gp = wins.net_pnl_usd.sum()
    gl = abs(df[df.result == "loss"].net_pnl_usd.sum())
    return dict(n=len(df), wr=len(wins) / len(df) * 100,
                pf=(gp / gl) if gl else float("inf"),
                avg=df.net_pnl_pct.mean())


def tag_confluence(base: pd.DataFrame, other: pd.DataFrame, hours: int) -> pd.Series:
    w = pd.Timedelta(hours=hours)
    return base.entry_time.apply(lambda t: ((other.entry_time - t).abs() <= w).any())


# ── 2. 各窗口合流統計 ─────────────────────────────────────────────────────────
rows = []
for h in WINDOWS_H:
    b_conf = tag_confluence(b, a, h)
    a_conf = tag_confluence(a, b, h)
    rows.append(dict(window=h, group="S2-Hammer 單獨", **stat_block(b[~b_conf])))
    rows.append(dict(window=h, group="S2-Hammer 有S2A合流", **stat_block(b[b_conf])))
    rows.append(dict(window=h, group="S2-RSI 單獨", **stat_block(a[~a_conf])))
    rows.append(dict(window=h, group="S2-RSI 有S2B合流", **stat_block(a[a_conf])))
sweep = pd.DataFrame(rows)
print("\n各窗口統計：")
print(sweep.to_string(index=False))

# ── 3. 主窗口 ±12h + 時間切分驗證 ─────────────────────────────────────────────
b = b.sort_values("entry_time").reset_index(drop=True)
b["has_a"] = tag_confluence(b, a, MAIN_W_H)
cut = b.entry_time.iloc[int(len(b) * 0.7)]

s_alone_all = stat_block(b[~b.has_a])
s_conf_all  = stat_block(b[b.has_a])
s_base_all  = stat_block(b)

is_seg, oos_seg = b[b.entry_time < cut], b[b.entry_time >= cut]
split = {
    "IS":  {"alone": stat_block(is_seg[~is_seg.has_a]),  "conf": stat_block(is_seg[is_seg.has_a])},
    "OOS": {"alone": stat_block(oos_seg[~oos_seg.has_a]), "conf": stat_block(oos_seg[oos_seg.has_a])},
}
print(f"\n時間切分點: {cut.date()}")
for seg, d in split.items():
    print(f"  {seg}: 單獨 WR {d['alone']['wr']:.1f}%/PF {d['alone']['pf']:.2f} (n={d['alone']['n']}) | "
          f"合流 WR {d['conf']['wr']:.1f}%/PF {d['conf']['pf']:.2f} (n={d['conf']['n']})")

# ── 4. 圖表 ──────────────────────────────────────────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def dark_fig(w=10, h=5, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    fig.patch.set_facecolor("#0f172a")
    for ax in (axes if n > 1 else [axes]):
        ax.set_facecolor("#1e293b"); ax.tick_params(colors="#94a3b8")
        ax.title.set_color("#e2e8f0")
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    return fig, axes

def chart_headline():
    fig, (a1, a2) = dark_fig(13, 4.8, n=2)
    groups = ["全部\n(現行)", "單獨\n(無S2-RSI)", "合流\n(有S2-RSI)"]
    wrs = [s_base_all["wr"], s_alone_all["wr"], s_conf_all["wr"]]
    pfs = [s_base_all["pf"], s_alone_all["pf"], s_conf_all["pf"]]
    colors = ["#64748b", "#22c55e", "#ef4444"]
    bars = a1.bar(groups, wrs, color=colors, width=0.55)
    a1.axhline(50, color="#475569", ls="--")
    a1.set_title(f"S2-Hammer 勝率 %（合流窗口 ±{MAIN_W_H}h）", fontsize=12)
    for bar, v in zip(bars, wrs):
        a1.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", color="#e2e8f0", fontweight="bold")
    bars2 = a2.bar(groups, pfs, color=colors, width=0.55)
    a2.axhline(1.0, color="#475569", ls="--")
    a2.set_title("S2-Hammer 獲利因子", fontsize=12)
    for bar, v in zip(bars2, pfs):
        a2.text(bar.get_x()+bar.get_width()/2, v+0.05, f"{v:.2f}", ha="center", color="#e2e8f0", fontweight="bold")
    fig.tight_layout()
    return fig_b64(fig)

def chart_split():
    fig, ax = dark_fig(11, 4.8)
    x = np.arange(2); w = 0.35
    alone = [split["IS"]["alone"]["pf"], split["OOS"]["alone"]["pf"]]
    conf  = [split["IS"]["conf"]["pf"],  split["OOS"]["conf"]["pf"]]
    ax.bar(x - w/2, alone, w, label="S2-Hammer 單獨", color="#22c55e")
    ax.bar(x + w/2, conf,  w, label="S2-Hammer 有S2A合流", color="#ef4444")
    ax.axhline(1.0, color="#475569", ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"前70% IS\n(切點{cut.date()})", "後30% OOS"])
    ax.set_title("時間切分驗證：兩段中「單獨」皆優於「合流」（PF）", fontsize=12)
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, (av, cv) in enumerate(zip(alone, conf)):
        ax.text(i-w/2, av+0.04, f"{av:.2f}", ha="center", color="#e2e8f0", fontsize=9, fontweight="bold")
        ax.text(i+w/2, cv+0.04, f"{cv:.2f}", ha="center", color="#e2e8f0", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig_b64(fig)

print("\n生成圖表...")
img_headline = chart_headline()
img_split = chart_split()

# ── 5. HTML ──────────────────────────────────────────────────────────────────
def img(b): return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'
def fmt(v, f="{:.2f}"): return "—" if (isinstance(v, float) and not np.isfinite(v)) else f.format(v)

def sweep_table():
    body = ""
    for h in WINDOWS_H:
        sub = sweep[sweep.window == h]
        for _, r in sub.iterrows():
            hl = ""
            if "S2-Hammer" in r.group:
                hl = " style='background:rgba(34,197,94,.08)'" if "單獨" in r.group else " style='background:rgba(239,68,68,.08)'"
            body += (f"<tr{hl}><td>±{h}h</td><td>{r.group}</td><td>{r.n}</td>"
                     f"<td>{fmt(r.wr,'{:.1f}')}%</td><td>{fmt(r.pf,'{:.3f}')}</td>"
                     f"<td>{fmt(r.avg,'{:+.3f}')}%</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>窗口</th><th>分組</th><th>筆數</th>"
            f"<th>勝率</th><th>PF</th><th>平均PnL%</th></tr></thead><tbody>{body}</tbody></table>")

def split_table():
    body = ""
    for seg_name, label in [("IS", f"前70%（~{cut.date()}）"), ("OOS", f"後30%（{cut.date()}~）")]:
        d = split[seg_name]
        body += (f"<tr><td rowspan='2'><strong>{label}</strong></td>"
                 f"<td style='color:#22c55e'>S2-Hammer 單獨</td><td>{d['alone']['n']}</td>"
                 f"<td>{fmt(d['alone']['wr'],'{:.1f}')}%</td><td>{fmt(d['alone']['pf'],'{:.3f}')}</td>"
                 f"<td>{fmt(d['alone']['avg'],'{:+.3f}')}%</td></tr>"
                 f"<tr><td style='color:#ef4444'>S2-Hammer 有S2A合流</td><td>{d['conf']['n']}</td>"
                 f"<td>{fmt(d['conf']['wr'],'{:.1f}')}%</td><td>{fmt(d['conf']['pf'],'{:.3f}')}</td>"
                 f"<td>{fmt(d['conf']['avg'],'{:+.3f}')}%</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>時段</th><th>分組</th><th>筆數</th>"
            f"<th>勝率</th><th>PF</th><th>平均PnL%</th></tr></thead><tbody>{body}</tbody></table>")

CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
:root{--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#38bdf8;--muted:#94a3b8;--border:#334155}
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
.bad{background:rgba(239,68,68,.08);border-left:3px solid var(--red);padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
</style>
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S2-RSI × S2-Hammer 合流分析：合併假設檢驗</title>{CSS}</head>
<body>
<div style="max-width:1100px;margin:0 auto 14px"><a href="../../xauusd.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 XAUUSD 主頁</a></div>
<div class="wrap">
<h1>S2-RSI × S2-Hammer <span style="color:#f59e0b">合流分析</span> — 合併假設檢驗</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
資料：S2-RSI V2.0（{len(a)} 筆）+ S2-Hammer V1.9（{len(b)} 筆）真實逐筆交易，2024-01 ~ 2026-04</p>

<div class="bad">
<strong>結論：不要合併。合流是反指標。</strong>
原始假設「S2-RSI（RSI）與 S2-Hammer（錘頭）同時觸發 = 更高品質訊號」被資料推翻——
S2-Hammer 錘頭在 ±{MAIN_W_H}h 內<strong>沒有</strong> S2-RSI 訊號時 WR {s_alone_all['wr']:.1f}% / PF {s_alone_all['pf']:.2f}（n={s_alone_all['n']}），
與 S2-RSI 同時觸發時僅 WR {s_conf_all['wr']:.1f}% / PF {s_conf_all['pf']:.2f}（n={s_conf_all['n']}，接近打平）。
</div>

<div class="card">
<div class="part">PART 1 — 主結果</div>
<h2>S2-Hammer 分組績效（合流窗口 ±{MAIN_W_H}h）</h2>
{img(img_headline)}
<p class="note">現行 S2-Hammer 全收：WR {s_base_all['wr']:.1f}% / PF {s_base_all['pf']:.2f}。
只要跳過「S2-RSI 也在觸發」的錘頭訊號，剩餘 {s_alone_all['n']}/{s_base_all['n']} 筆的品質顯著提升。</p>
</div>

<div class="card">
<div class="part">PART 2 — 窗口敏感度</div>
<h2>±2h / ±6h / ±12h 三種合流判定窗口</h2>
{sweep_table()}
<p class="note">綠底 = S2-Hammer 單獨、紅底 = S2-Hammer 合流。效果在 ±6h 起顯著且方向一致（±2h 樣本太少，n=21）。
S2-RSI 自身的合流/單獨差異不大（42% 上下），資訊價值集中在「它標記的環境對 S2-Hammer 有害」。</p>
</div>

<div class="card">
<div class="part">PART 3 — 時間切分驗證</div>
<h2>前 70% / 後 30% 兩段是否都成立？</h2>
{split_table()}
{img(img_split)}
<div class="good">
兩段皆成立，且 OOS 段更極端（合流組 PF {split['OOS']['conf']['pf']:.2f}，實際虧損）——
此規則不是單一時期的巧合。
</div>
</div>

<div class="card">
<div class="part">PART 4 — 機制解釋</div>
<h2>為什麼合流反而差？與既有研究的一致性</h2>
<p style="font-size:.9em;line-height:1.8">
S2-RSI 觸發 = RSI 深度超賣/背離環境。2026-07-02 的 Regime 四象限分析
（analyze_h1_regime.py，見 xauusd.html「2026 H1/H2」分頁）已獨立發現：
<strong>Z-Score &lt; -2.5 的極端超賣是陷阱區（勝率僅 20%）——極端超賣代表趨勢延伸，不是反轉。</strong><br><br>
錘頭出現在「RSI 也深度超賣」的時刻 = 接刀接在瀑布中段；
錘頭<strong>單獨</strong>出現（RSI 未達極端）才是健康回調中的反轉點。
本報告的發現與 Z-Score 陷阱區是同一機制的兩種觀測，互為佐證。
</p>
</div>

<div class="card">
<div class="part">PART 5 — 行動建議</div>
<h2>取代「合併」的三個行動</h2>
<div class="good"><strong>1. S2-Hammer 加「S2-RSI 互斥過濾器」</strong>：S2-RSI 觸發環境（近 {MAIN_W_H}h 內 RSI 深超賣/背離條件成立）→ S2-Hammer 跳過。
預期效果：WR {s_base_all['wr']:.1f}%→{s_alone_all['wr']:.1f}%、PF {s_base_all['pf']:.2f}→{s_alone_all['pf']:.2f}，保留 {s_alone_all['n']}/{s_base_all['n']} 筆交易。</div>
<div class="good"><strong>2. S2-RSI 重新定位</strong>：其資訊價值可能不在「何時進場」而在「何時不要進」——
待 S2 真實逐筆歸因（NEXT 優先事項）完成後，決定 S2-RSI 是保留為獨立策略還是降級為 S2-Hammer 的 veto 濾網。</div>
<div class="good"><strong>3. 修正優化目標</strong>：42-44% 勝率對左側反轉策略不算差（PF 1.68 靠賠率補償）。
S2 真正的病是實盤 PF 僅 1.11（vs 回測 1.68 的執行落差）與 time_bleed &gt;50%，優化方向是這兩個，不是勝率。</div>
<div class="warn">
<strong>上實盤前必做：</strong>①此為事後挖掘規則（雖通過時間切分＋有機制支撐），需先確認與
S2-RSI V2.4 / S2-Hammer V2.3 已實作的 Z-Score 過濾器（甜蜜區 -1.5~-0.5、阻擋 &lt;-2.5）是否為同一件事——
很可能大幅重疊，屆時直接啟用既有過濾器即可，不必新造輪子；
②Pine 實作後在 TradingView 用完整歷史驗證；③S2 每週最多 2 次的 H2 限制不變。
</div>
</div>

<p class="note">
方法：兩份 TradingView List of Trades 真實成交紀錄，合流判定 = 另一策略在 ±N 小時內有進場。
時間切分按 S2-Hammer 筆數 70/30。統計口徑與 xauusd/analysis/ 既有模組一致。
</p>
<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S2-RSI × S2-Hammer 合流分析 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 報告已生成：{OUT_HTML}")
