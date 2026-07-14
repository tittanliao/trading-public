"""
run_s1_v39_real_attribution.py — S1-AweWithBB V3.7 vs V3.9 真實逐筆歸因分析
==============================================================================
輸出：xauusd/XAUUSD-Long-S1-AweWithBB/report_v3.9_real.html

沿用 run_s1_v37_real_attribution.py 的方法論（V3.4 vs V3.7），這次比較：
  - V3.7（現行已確認版，FILTER: 1H MA(3) + BBW高檔(rank≥70) 皆 ON，BB Source=close）
  - V3.9（測試版，FILTER 編號化重排，同樣 1H MA + BBW高檔 ON，
    但 ⚠️ BB Source 由 close 改為 ohlc4——這是本次比較的核心變因，
    直接影響 BB 通道與 fast EMA 的計算，進而改變每一筆進場點位）
兩份 CSV 皆為 TradingView List of Trades 真實成交紀錄（非模擬）。

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s1_v39_real_attribution.py
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

from analysis import loader, fail_patterns, metrics

FOLDER   = ROOT / "xauusd/XAUUSD-Long-S1-AweWithBB"
V37_CSV  = FOLDER / "S1-Awe-V3.7_FX_IDC_XAUUSD_2026-07-05.csv"
V39_CSV  = FOLDER / "S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv"
OUT_HTML = FOLDER / "report_v3.9_real.html"

SL_PCT = 0.005   # 0.5% stop loss（用於 R 倍數換算，兩版一致）

# ── 1. 載入兩版真實逐筆交易 ───────────────────────────────────────────────────
print("載入 V3.7 / V3.9 真實逐筆交易...")

def load_and_enrich(csv_path: Path, label: str) -> pd.DataFrame:
    t = loader.load_trades(csv_path)
    t = t[t["exit_signal"] != "Open"].reset_index(drop=True)  # 排除未平倉尾單
    t["session"] = fail_patterns.tag_session(t["entry_time"])
    t["entry_hour"] = t["entry_time"].dt.hour
    t["win"] = t["result"] == "win"
    t["r_multiple"] = t["net_pnl_usd"] / (t["entry_price"] * t["size_qty"] * SL_PCT)
    t["exit_kind"] = t["exit_signal"].apply(
        lambda s: "TP2" if "TP2" in s else ("TP1" if "TP1" in s else ("SL" if "SL" in s else "other")))
    t["version"] = label
    return t

v37 = load_and_enrich(V37_CSV, "V3.7")
v39 = load_and_enrich(V39_CSV, "V3.9")
print(f"  V3.7：{len(v37)} 筆（{v37['entry_time'].min().date()} → {v37['exit_time'].max().date()}）")
print(f"  V3.9：{len(v39)} 筆（{v39['entry_time'].min().date()} → {v39['exit_time'].max().date()}）")

fail37 = fail_patterns.classify_fail(v37)
fail39 = fail_patterns.classify_fail(v39)

# ── 統計工具 ─────────────────────────────────────────────────────────────────
def stat_block(t: pd.DataFrame) -> dict:
    wins = t.loc[t["win"], "net_pnl_usd"].sum()
    loss = abs(t.loc[~t["win"], "net_pnl_usd"].sum())
    return dict(
        n=len(t), wins=int(t["win"].sum()), wr=t["win"].mean(),
        pf=(wins / loss) if loss else float("inf"),
        net=t["net_pnl_usd"].sum(), avg_r=t["r_multiple"].mean(),
        mdd=abs(metrics.max_drawdown(t)),
    )

s37 = stat_block(v37)
s39 = stat_block(v39)

def exit_breakdown(t: pd.DataFrame) -> pd.DataFrame:
    return t.groupby("exit_kind").agg(
        n=("trade_id", "count"), net=("net_pnl_usd", "sum"),
        avg_r=("r_multiple", "mean"), wr=("win", "mean"),
    ).reindex(["TP1", "TP2", "SL", "other"]).dropna(how="all")

ex37 = exit_breakdown(v37)
ex39 = exit_breakdown(v39)

# ── 共同期間對照（V3.9 資料起於 2024-01，與 V3.7 同期起點；用共同結束日避免尾端偏差）─
common_end = min(v37["exit_time"].max(), v39["exit_time"].max())
v37_common = v37[v37["exit_time"] <= common_end]
v39_common = v39[v39["exit_time"] <= common_end]
s37c, s39c = stat_block(v37_common), stat_block(v39_common)

# ── 圖表工具 ─────────────────────────────────────────────────────────────────
def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def dark_fig(w=10, h=5, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    fig.patch.set_facecolor("#0f172a")
    for ax in (axes if n > 1 else [axes]):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
    return fig, axes

def img(b):
    return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'

# ① WR / PF / Net / MDD 對比（共同期間，公平比較）
def chart_headline():
    fig, axes = dark_fig(14, 4, n=4)
    labels = [f"V3.7\n({s37c['n']})", f"V3.9\n({s39c['n']})"]
    metrics_ = [
        ("勝率 %", [s37c["wr"]*100, s39c["wr"]*100], "{:.1f}%", 0, 65, 50),
        ("獲利因子", [s37c["pf"], s39c["pf"]], "{:.3f}", 0, 2.1, 1.0),
        ("淨盈虧 $", [s37c["net"], s39c["net"]], "${:,.0f}", 0, max(s37c["net"], s39c["net"])*1.25, None),
        ("最大回撤 $", [s37c["mdd"], s39c["mdd"]], "${:,.0f}", 0, max(s37c["mdd"], s39c["mdd"])*1.3, None),
    ]
    for ax, (title, vals, fmt, lo, hi, ref) in zip(axes, metrics_):
        bars = ax.bar(labels, vals, color=["#64748b", "#38bdf8"], width=0.55)
        if ref is not None:
            ax.axhline(ref, color="#475569", ls="--")
        ax.set_title(title, fontsize=11); ax.set_ylim(lo, hi)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v + hi*0.02, fmt.format(v),
                    ha="center", color="#e2e8f0", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig_b64(fig)

# ② 出場訊號分布 + 各訊號淨利貢獻（共同期間）
def chart_exit_breakdown():
    ex37c = exit_breakdown(v37_common)
    ex39c = exit_breakdown(v39_common)
    fig, (a1, a2) = dark_fig(13, 5, n=2)
    kinds = ["TP1", "TP2", "SL"]
    x = np.arange(len(kinds)); w = 0.35
    n37 = [ex37c.loc[k, "n"] if k in ex37c.index else 0 for k in kinds]
    n39 = [ex39c.loc[k, "n"] if k in ex39c.index else 0 for k in kinds]
    p37 = [v/s37c["n"]*100 for v in n37]; p39 = [v/s39c["n"]*100 for v in n39]
    a1.bar(x - w/2, p37, w, label="V3.7", color="#64748b")
    a1.bar(x + w/2, p39, w, label="V3.9", color="#38bdf8")
    a1.set_xticks(x); a1.set_xticklabels(kinds)
    a1.set_title("出場訊號佔比 %（共同期間）", fontsize=12)
    a1.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, (a, b) in enumerate(zip(p37, p39)):
        a1.text(i-w/2, a+1, f"{a:.1f}%", ha="center", color="#e2e8f0", fontsize=8)
        a1.text(i+w/2, b+1, f"{b:.1f}%", ha="center", color="#e2e8f0", fontsize=8)

    net37 = [ex37c.loc[k, "net"] if k in ex37c.index else 0 for k in kinds]
    net39 = [ex39c.loc[k, "net"] if k in ex39c.index else 0 for k in kinds]
    a2.bar(x - w/2, net37, w, label="V3.7", color="#64748b")
    a2.bar(x + w/2, net39, w, label="V3.9", color="#38bdf8")
    a2.axhline(0, color="#475569", lw=1)
    a2.set_xticks(x); a2.set_xticklabels(kinds)
    a2.set_title("各出場類型淨利貢獻 $（共同期間）", fontsize=12)
    a2.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    fig.tight_layout()
    return fig_b64(fig)

# ③ 真實失敗模式分布對比
def chart_fail_compare():
    order = ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]
    fig, ax = dark_fig(11, 4.8)
    x = np.arange(len(order)); w = 0.35
    c37 = fail37["fail_type"].value_counts().reindex(order, fill_value=0)
    c39 = fail39["fail_type"].value_counts().reindex(order, fill_value=0)
    p37 = c37 / len(fail37) * 100
    p39 = c39 / len(fail39) * 100
    ax.bar(x - w/2, p37.values, w, label=f"V3.7（{len(fail37)} 筆虧損）", color="#64748b")
    ax.bar(x + w/2, p39.values, w, label=f"V3.9（{len(fail39)} 筆虧損）", color="#38bdf8")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
    ax.set_title("真實失敗模式佔虧損單比例 %（V3.7 vs V3.9，全樣本）", fontsize=12)
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, (a, b) in enumerate(zip(p37.values, p39.values)):
        ax.text(i-w/2, a+0.8, f"{a:.0f}%", ha="center", color="#e2e8f0", fontsize=8)
        ax.text(i+w/2, b+0.8, f"{b:.0f}%", ha="center", color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

# ④ 時段勝率對比
def chart_session_compare():
    fig, ax = dark_fig(11, 4.8)
    order = ["asia", "europe", "us"]
    g37 = v37.groupby("session")["win"].mean().reindex(order) * 100
    g39 = v39.groupby("session")["win"].mean().reindex(order) * 100
    x = np.arange(len(order)); w = 0.35
    ax.bar(x - w/2, g37.values, w, label="V3.7", color="#64748b")
    ax.bar(x + w/2, g39.values, w, label="V3.9", color="#38bdf8")
    ax.axhline(50, color="#475569", ls="--")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_title("時段勝率對比 %（全樣本）", fontsize=12)
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, (a, b) in enumerate(zip(g37.values, g39.values)):
        ax.text(i-w/2, a+1, f"{a:.1f}%", ha="center", color="#e2e8f0", fontsize=8)
        ax.text(i+w/2, b+1, f"{b:.1f}%", ha="center", color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

print("生成圖表...")
imgs = dict(
    headline = chart_headline(),
    exitbd   = chart_exit_breakdown(),
    failcmp  = chart_fail_compare(),
    sess     = chart_session_compare(),
)
print("  圖表完成")

# ── HTML 片段工具 ────────────────────────────────────────────────────────────
def pct(v): return "—" if not np.isfinite(v) else f"{v*100:.1f}%"
def num(v, f="{:.3f}"): return "∞" if not np.isfinite(v) else f.format(v)

def headline_table():
    rows = [
        ("交易筆數（共同期間）", s37c["n"], s39c["n"], ""),
        ("勝率", f"{s37c['wr']*100:.2f}%", f"{s39c['wr']*100:.2f}%", f"{(s39c['wr']-s37c['wr'])*100:+.1f} pp"),
        ("獲利因子", f"{s37c['pf']:.3f}", f"{s39c['pf']:.3f}", f"{s39c['pf']-s37c['pf']:+.3f}"),
        ("淨盈虧", f"${s37c['net']:+,.0f}", f"${s39c['net']:+,.0f}", f"${s39c['net']-s37c['net']:+,.0f}"),
        ("最大回撤", f"${s37c['mdd']:,.0f}", f"${s39c['mdd']:,.0f}", f"${s39c['mdd']-s37c['mdd']:+,.0f}"),
        ("平均 R 倍數", f"{s37c['avg_r']:.3f}R", f"{s39c['avg_r']:.3f}R", f"{s39c['avg_r']-s37c['avg_r']:+.3f}R"),
        ("回測區間（共同期間）", f"{v37_common['entry_time'].min().date()} → {common_end.date()}",
                    f"{v39_common['entry_time'].min().date()} → {common_end.date()}", ""),
        ("全樣本筆數（含 V3.9 獨有的新資料）", s37["n"], s39["n"], ""),
    ]
    body = ""
    for name, a, b, d in rows:
        body += (f"<tr><td><strong>{name}</strong></td><td>{a}</td>"
                 f"<td style='color:#38bdf8'>{b}</td><td style='color:#22c55e'>{d}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>指標</th><th>V3.7（真實）</th>"
            f"<th>V3.9（真實）</th><th>差異</th></tr></thead><tbody>{body}</tbody></table>")

def exit_table():
    ex37c = exit_breakdown(v37_common)
    ex39c = exit_breakdown(v39_common)
    body = ""
    for k in ["TP1", "TP2", "SL"]:
        a = ex37c.loc[k] if k in ex37c.index else None
        b = ex39c.loc[k] if k in ex39c.index else None
        an = int(a["n"]) if a is not None else 0
        bn = int(b["n"]) if b is not None else 0
        anet = a["net"] if a is not None else 0
        bnet = b["net"] if b is not None else 0
        ar = a["avg_r"] if a is not None else float("nan")
        br = b["avg_r"] if b is not None else float("nan")
        body += (f"<tr><td><strong>{k}</strong></td>"
                 f"<td>{an} ({an/s37c['n']*100:.1f}%)</td><td>{ar:.2f}R</td><td>${anet:+,.0f}</td>"
                 f"<td>{bn} ({bn/s39c['n']*100:.1f}%)</td><td>{br:.2f}R</td><td>${bnet:+,.0f}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>出場</th>"
            f"<th>V3.7 筆數</th><th>V3.7 均R</th><th>V3.7 淨利</th>"
            f"<th>V3.9 筆數</th><th>V3.9 均R</th><th>V3.9 淨利</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

def fail_table():
    order = ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]
    c37 = fail37["fail_type"].value_counts().reindex(order, fill_value=0)
    c39 = fail39["fail_type"].value_counts().reindex(order, fill_value=0)
    body = ""
    for k in order:
        a, b = c37[k], c39[k]
        ap, bp = a/len(fail37)*100, b/len(fail39)*100
        delta_color = "#22c55e" if bp < ap else "#ef4444"
        body += (f"<tr><td><strong>{k}</strong></td>"
                 f"<td>{a} ({ap:.0f}%)</td><td>{b} ({bp:.0f}%)</td>"
                 f"<td style='color:{delta_color}'>{bp-ap:+.1f} pp</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>失敗類型</th><th>V3.7</th><th>V3.9</th><th>變化</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

# 關鍵數字
imm37_pct = (fail37["fail_type"]=="immediate_loss").mean()*100
imm39_pct = (fail39["fail_type"]=="immediate_loss").mean()*100
sl37_pct  = ex37.loc["SL","n"]/s37["n"]*100 if "SL" in ex37.index else 0
sl39_pct  = ex39.loc["SL","n"]/s39["n"]*100 if "SL" in ex39.index else 0
time_exit_37 = (v37["exit_kind"]=="other").sum()
time_exit_39 = (v39["exit_kind"]=="other").sum()
wr_delta_common = (s39c["wr"] - s37c["wr"]) * 100
pf_delta_common = s39c["pf"] - s37c["pf"]
verdict_better = wr_delta_common > 0 and pf_delta_common > 0
verdict_worse  = wr_delta_common < 0 and pf_delta_common < 0

CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
:root{--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#38bdf8;--muted:#94a3b8;--border:#334155}
.wrap{max-width:1200px;margin:0 auto}
h1{font-size:1.7em;margin-bottom:6px;color:#f8fafc}
h2{font-size:1.15em;color:#38bdf8;margin:8px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--border)}
h3{font-size:.95em;color:#cbd5e1;margin:16px 0 8px}
.part{font-size:.75em;letter-spacing:.15em;color:#f59e0b;font-weight:700;text-transform:uppercase}
.card{background:#1e293b;border:1px solid var(--border);border-radius:10px;padding:22px;margin-bottom:18px}
.tbl{width:100%;border-collapse:collapse;font-size:.85em;margin:10px 0}
.tbl th{background:#0f172a;color:var(--muted);padding:9px 12px;text-align:left;border-bottom:1px solid var(--border)}
.tbl td{padding:8px 12px;border-bottom:1px solid rgba(51,65,85,.5)}
.tbl tr:hover td{background:rgba(255,255,255,.02)}
.note{font-size:.82em;color:var(--muted);margin-top:8px}
.callout{background:rgba(56,189,248,.08);border-left:3px solid var(--blue);padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
.warn{background:rgba(245,158,11,.08);border-left:3px solid var(--yellow);padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
.good{background:rgba(34,197,94,.08);border-left:3px solid var(--green);padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
.bad{background:rgba(239,68,68,.08);border-left:3px solid var(--red);padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
ul{margin:8px 0 8px 20px}li{margin:6px 0}
.ins-title{color:#f8fafc;font-weight:700;font-size:.98em}
</style>
"""

verdict_class = "good" if verdict_better else ("bad" if verdict_worse else "warn")
verdict_text = ("V3.9 在共同期間同時勝率與PF皆優於 V3.7" if verdict_better else
                "V3.9 在共同期間同時勝率與PF皆劣於 V3.7" if verdict_worse else
                "V3.9 與 V3.7 在共同期間表現互有優劣（勝率與PF方向不一致），需個別檢視")

HTML = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S1-AweWithBB V3.7 vs V3.9 真實逐筆歸因分析</title>{CSS}</head>
<body>
<div style="max-width:1200px;margin:0 auto 14px"><a href="../index.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="wrap">

<h1>S1-AweWithBB <span style="color:#38bdf8">V3.7 vs V3.9</span> 真實逐筆歸因分析</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
兩版皆為 TradingView 匯出的<strong>真實逐筆交易 CSV</strong>（非模擬）：
V3.7 全樣本 {s37['n']} 筆、V3.9 全樣本 {s39['n']} 筆 · 共同期間對照 {s37c['n']} vs {s39c['n']} 筆</p>
<p class="note">其他報告：
<a href="report_v3.7_real.html" style="color:var(--blue)">report_v3.7_real.html（V3.4 vs V3.7 真實歸因）</a> ·
<a href="report_v3.7.html" style="color:var(--blue)">report_v3.7.html（V3.7 confirmed）</a></p>

<div class="callout">
<strong>V3.9 相較 V3.7 的核心改動：</strong>
過濾器（1H MA 過濾 + BBW 高檔過濾）維持與 V3.7 相同設定 ON；
主要變因是 <strong>BB Source 由 close 改為 ohlc4</strong>（(O+H+L+C)/4），
直接改變 BB 通道與 fast EMA 的計算基礎，進而改變每一筆進場點位——
以下比較用「共同期間」（兩版資料重疊的時間範圍）避免尾端資料量不對等造成偏差。
</div>

<!-- ═══════════ PART 1 ═══════════ -->
<div class="card">
<div class="part">PART 1 — 真實績效對比（共同期間）</div>
<h2>V3.7 vs V3.9 頭條指標（皆為真實計算）</h2>
{headline_table()}
{img(imgs['headline'])}
<div class="{verdict_class}"><strong>{verdict_text}</strong>
勝率差 {wr_delta_common:+.1f}pp，PF差 {pf_delta_common:+.3f}。</div>
</div>

<!-- ═══════════ PART 2 ═══════════ -->
<div class="card">
<div class="part">PART 2 — 出場結構歸因</div>
<h2>BB Source 改變是否影響出場結構？</h2>
<p class="note">直接統計兩版真實交易在共同期間的出場訊號分布與各類型貢獻的淨利。</p>
{exit_table()}
{img(imgs['exitbd'])}
<p class="note">V3.7 全樣本時間出場觸發次數：{time_exit_37}；V3.9 全樣本：{time_exit_39}
（"other" = 非 TP1/TP2/SL 的出場，通常為時間止損）。</p>
</div>

<!-- ═══════════ PART 3 ═══════════ -->
<div class="card">
<div class="part">PART 3 — 真實失敗模式對比</div>
<h2>V3.9 的虧損單，體質有變化嗎？</h2>
{fail_table()}
{img(imgs['failcmp'])}
<div class="{'good' if imm39_pct < imm37_pct else 'warn'}">
<strong>immediate_loss（進場就錯）佔虧損單比例：</strong>
V3.7 {imm37_pct:.1f}% → V3.9 {imm39_pct:.1f}%（{imm39_pct-imm37_pct:+.1f} pp，全樣本）。
SL 觸發率：V3.7 {sl37_pct:.1f}% → V3.9 {sl39_pct:.1f}%（全樣本，含 V3.9 獨有的新增資料期間）。
</div>
</div>

<!-- ═══════════ PART 4 ═══════════ -->
<div class="card">
<div class="part">PART 4 — 時段勝率對比</div>
<h2>V3.9 是否仍有相同的時段弱點？</h2>
{img(imgs['sess'])}
<p class="note">此圖為全樣本（非共同期間），V3.9 因涵蓋較晚的資料範圍，時段分布可能與 V3.7 略有不同，僅供參考方向。</p>
</div>

<p class="note" style="margin-top:16px">
方法說明：V3.7、V3.9 皆為 TradingView Strategy Tester 匯出的 List of Trades 真實成交紀錄，
逐筆包含進出場時間、價格、MFE/MAE、淨盈虧。R 倍數 = 淨盈虧 ÷（進場價 × 數量 × 0.5%）。
失敗模式分類邏輯與門檻與既有 xauusd/analysis/fail_patterns.py 完全一致，確保與其他報告可比。
局限：①「共同期間」對照仍受兩份 CSV 匯出時間點不同影響，V3.9 的新增資料段（V3.7 CSV 尚未涵蓋的期間）
無法納入共同期間比較，僅列在「全樣本」欄位供參考；②本報告只隔離 BB Source 這一項變因做粗略歸因，
未做逐筆 bar-level A/B 對照（同一時間點兩種 Source 各自是否觸發），精確效果仍需 TradingView 端驗證。
</p>

<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S1-AweWithBB V3.7 vs V3.9 真實逐筆歸因分析 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 真實歸因分析報告已生成：{OUT_HTML}")
print(f"   V3.7 共同期間: WR {s37c['wr']*100:.1f}% PF {s37c['pf']:.3f} Net ${s37c['net']:+,.0f} MDD ${s37c['mdd']:,.0f} (n={s37c['n']})")
print(f"   V3.9 共同期間: WR {s39c['wr']*100:.1f}% PF {s39c['pf']:.3f} Net ${s39c['net']:+,.0f} MDD ${s39c['mdd']:,.0f} (n={s39c['n']})")
print(f"   SL%: {sl37_pct:.1f}% → {sl39_pct:.1f}% | immediate_loss%: {imm37_pct:.1f}% → {imm39_pct:.1f}%（全樣本）")
