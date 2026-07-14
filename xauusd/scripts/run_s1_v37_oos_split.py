"""
run_s1_v37_oos_split.py — S1-AweWithBB V3.7 Out-of-Sample（樣本外）測試
==============================================================================
輸出：xauusd/XAUUSD-Long-S1-AweWithBB/report_v3.7_oos.html

背景：S1 V3.7 的績效表（WR 55.1%、PF 1.743）是對 2024-01-05 ~ 2026-07-03
整段 459 筆真實成交做「全樣本」評估。全樣本評估的風險是：策略設計（V3.4→V3.7
六次迭代）在同一段資料上被反覆查看、反覆調整，即使沒有做正式參數網格搜尋，
也可能無意間把策略「擬合」到這段歷史的特定行情特徵（尤其是這段期間黃金以
大多頭為主，Long-Only 策略天然有利）。

本腳本不重新回測，改用「事後時間切分」驗證：
S1 的訊號計算全程無未來函數（ta.sma / ta.ema / request.security 皆
lookahead_off，同時間僅持一倉），代表已匯出的 459 筆真實交易可以安全地
按進場時間切成兩段來分開檢驗，不需要回到 TradingView 重新回測。

切分方式（20260710 決定，見對話紀錄）：
  取 459 筆真實交易依進場時間排序，第 70% 筆（index 321，2025-11-20 20:30）
  為切點：
    In-Sample  (IS)  = 2024-01-05 ~ 2025-11-20，321 筆
    Out-of-Sample(OOS) = 2025-11-20 ~ 2026-07-03，138 筆
  採用「按筆數切 70/30」而非「按日曆天數切」，理由：兩段的交易筆數越平均，
  統計檢定力越平衡，OOS 段也已經超過 project 自訂的 n≥30 最低樣本門檻。

判斷標準（呼應 xauusd/claude/monthly_checklist.md 轉向訊號清單）：
  - PF(OOS) < 1.2 → 觸發清單上的黃燈項目
  - WR(OOS) 與 WR(IS) 相差 > 8-10pp → edge 可能集中在 IS 那段特定行情，近期已衰退
  - 出場結構（SL 觸發率 / TP2 佔比）若 OOS 明顯劣化 → V3.7 引以為傲的「賠率改善」
    沒有延續到最近的資料

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s1_v37_oos_split.py
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

ROOT = Path(__file__).parent.parent.parent  # xauusd/scripts/ -> 兩層 .parent 到 trading/
sys.path.insert(0, str(ROOT / "xauusd"))

from analysis import loader, fail_patterns, metrics
from analysis.config import STRATEGIES

S1_CFG   = next(c for c in STRATEGIES if c["id"] == "S1-AweWithBB")
FOLDER   = S1_CFG["folder"]
V37_CSV  = FOLDER / "S1-Awe-V3.7_FX_IDC_XAUUSD_2026-07-05.csv"
OUT_HTML = FOLDER / "report_v3.7_oos.html"

SL_PCT = 0.005  # 0.5% 固定止損，用於 R 倍數換算

# 70% 切點：見上方 docstring 的推導過程（459 筆按進場時間排序第 321 筆）
SPLIT_DATE = pd.Timestamp("2025-11-20 20:30:00")

# ── 1. 載入 V3.7 真實逐筆交易，按時間切 IS / OOS ─────────────────────────────
print("載入 V3.7 真實逐筆交易...")
t = loader.load_trades(V37_CSV)
t["session"] = fail_patterns.tag_session(t["entry_time"])
t["entry_hour"] = t["entry_time"].dt.hour
t["win"] = t["result"] == "win"
t["r_multiple"] = t["net_pnl_usd"] / (t["entry_price"] * t["size_qty"] * SL_PCT)
t["exit_kind"] = t["exit_signal"].apply(
    lambda s: "TP2" if "TP2" in s else ("TP1" if "TP1" in s else ("SL" if "SL" in s else "other")))
t = t.sort_values("entry_time").reset_index(drop=True)

is_seg  = t[t["entry_time"] < SPLIT_DATE].copy()
oos_seg = t[t["entry_time"] >= SPLIT_DATE].copy()

print(f"  全樣本：{len(t)} 筆（{t['entry_time'].min().date()} → {t['exit_time'].max().date()}）")
print(f"  In-Sample　：{len(is_seg)} 筆（{is_seg['entry_time'].min().date()} → {is_seg['exit_time'].max().date()}）")
print(f"  Out-of-Sample：{len(oos_seg)} 筆（{oos_seg['entry_time'].min().date()} → {oos_seg['exit_time'].max().date()}）")

fail_is  = fail_patterns.classify_fail(is_seg)
fail_oos = fail_patterns.classify_fail(oos_seg)

# ── 統計工具（沿用 run_s1_v37_real_attribution.py 的 stat_block）───────────────
def stat_block(seg: pd.DataFrame) -> dict:
    wins = seg.loc[seg["win"], "net_pnl_usd"].sum()
    loss = abs(seg.loc[~seg["win"], "net_pnl_usd"].sum())
    return dict(
        n=len(seg), wins=int(seg["win"].sum()), wr=seg["win"].mean(),
        pf=(wins / loss) if loss else float("inf"),
        net=seg["net_pnl_usd"].sum(), avg_r=seg["r_multiple"].mean(),
        mdd=abs(metrics.max_drawdown(seg)),
    )

s_is  = stat_block(is_seg)
s_oos = stat_block(oos_seg)

def exit_breakdown(seg: pd.DataFrame) -> pd.DataFrame:
    return seg.groupby("exit_kind").agg(
        n=("trade_id", "count"), net=("net_pnl_usd", "sum"),
        avg_r=("r_multiple", "mean"), wr=("win", "mean"),
    ).reindex(["TP1", "TP2", "SL", "other"]).dropna(how="all")

ex_is  = exit_breakdown(is_seg)
ex_oos = exit_breakdown(oos_seg)

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

# ① 權益曲線 + 切分線（最直觀：能不能一眼看出 OOS 段是否走平/回落）
def chart_equity_curve():
    fig, ax = dark_fig(13, 5)
    ax.plot(t["entry_time"], t["cum_pnl_usd"], color="#38bdf8", lw=1.6)
    ax.axvline(SPLIT_DATE, color="#f59e0b", ls="--", lw=1.5)
    ax.axvspan(t["entry_time"].min(), SPLIT_DATE, color="#64748b", alpha=0.08)
    ax.axvspan(SPLIT_DATE, t["entry_time"].max(), color="#22c55e", alpha=0.08)
    ymax = t["cum_pnl_usd"].max()
    ax.text(t["entry_time"].min(), ymax*0.95, "In-Sample", color="#94a3b8", fontsize=10)
    ax.text(SPLIT_DATE, ymax*0.95, " Out-of-Sample →", color="#22c55e", fontsize=10)
    ax.set_title(f"權益曲線（累積淨利 $）— 切分點 {SPLIT_DATE.date()}", fontsize=12)
    ax.axhline(0, color="#475569", lw=1)
    fig.tight_layout()
    return fig_b64(fig)

# ② WR / PF / Net / MDD 對比
def chart_headline():
    fig, axes = dark_fig(14, 4, n=4)
    labels = [f"IS\n({s_is['n']})", f"OOS\n({s_oos['n']})"]
    metrics_ = [
        ("勝率 %", [s_is["wr"]*100, s_oos["wr"]*100], "{:.1f}%", 0, 65, 50),
        ("獲利因子", [s_is["pf"], s_oos["pf"]], "{:.3f}", 0, max(s_is["pf"], s_oos["pf"])*1.25, 1.2),
        ("淨盈虧 $", [s_is["net"], s_oos["net"]], "${:,.0f}", 0, max(s_is["net"], s_oos["net"])*1.25, None),
        ("最大回撤 $", [s_is["mdd"], s_oos["mdd"]], "${:,.0f}", 0, max(s_is["mdd"], s_oos["mdd"])*1.3, None),
    ]
    for ax, (title, vals, fmt, lo, hi, ref) in zip(axes, metrics_):
        bars = ax.bar(labels, vals, color=["#64748b", "#22c55e"], width=0.55)
        if ref is not None:
            ax.axhline(ref, color="#475569", ls="--")
        ax.set_title(title, fontsize=11); ax.set_ylim(lo, hi)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v + hi*0.02, fmt.format(v),
                    ha="center", color="#e2e8f0", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig_b64(fig)

# ③ 出場訊號分布對比
def chart_exit_breakdown():
    fig, (a1, a2) = dark_fig(13, 5, n=2)
    kinds = ["TP1", "TP2", "SL"]
    x = np.arange(len(kinds)); w = 0.35
    n_is  = [ex_is.loc[k, "n"]  if k in ex_is.index  else 0 for k in kinds]
    n_oos = [ex_oos.loc[k, "n"] if k in ex_oos.index else 0 for k in kinds]
    p_is  = [v/s_is["n"]*100  for v in n_is]
    p_oos = [v/s_oos["n"]*100 for v in n_oos]
    a1.bar(x - w/2, p_is, w, label="IS", color="#64748b")
    a1.bar(x + w/2, p_oos, w, label="OOS", color="#22c55e")
    a1.set_xticks(x); a1.set_xticklabels(kinds)
    a1.set_title("出場訊號佔比 %", fontsize=12)
    a1.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, (a, b) in enumerate(zip(p_is, p_oos)):
        a1.text(i-w/2, a+1, f"{a:.1f}%", ha="center", color="#e2e8f0", fontsize=8)
        a1.text(i+w/2, b+1, f"{b:.1f}%", ha="center", color="#e2e8f0", fontsize=8)

    net_is  = [ex_is.loc[k, "net"]  if k in ex_is.index  else 0 for k in kinds]
    net_oos = [ex_oos.loc[k, "net"] if k in ex_oos.index else 0 for k in kinds]
    a2.bar(x - w/2, net_is, w, label="IS", color="#64748b")
    a2.bar(x + w/2, net_oos, w, label="OOS", color="#22c55e")
    a2.axhline(0, color="#475569", lw=1)
    a2.set_xticks(x); a2.set_xticklabels(kinds)
    a2.set_title("各出場類型淨利貢獻 $", fontsize=12)
    a2.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    fig.tight_layout()
    return fig_b64(fig)

# ④ 失敗模式對比
def chart_fail_compare():
    order = ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]
    fig, ax = dark_fig(11, 4.8)
    x = np.arange(len(order)); w = 0.35
    c_is  = fail_is["fail_type"].value_counts().reindex(order, fill_value=0)
    c_oos = fail_oos["fail_type"].value_counts().reindex(order, fill_value=0)
    p_is  = c_is  / len(fail_is)  * 100 if len(fail_is)  else c_is*0
    p_oos = c_oos / len(fail_oos) * 100 if len(fail_oos) else c_oos*0
    ax.bar(x - w/2, p_is.values,  w, label=f"IS（{len(fail_is)} 筆虧損）",  color="#64748b")
    ax.bar(x + w/2, p_oos.values, w, label=f"OOS（{len(fail_oos)} 筆虧損）", color="#22c55e")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
    ax.set_title("真實失敗模式佔虧損單比例 %（IS vs OOS）", fontsize=12)
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, (a, b) in enumerate(zip(p_is.values, p_oos.values)):
        ax.text(i-w/2, a+0.8, f"{a:.0f}%", ha="center", color="#e2e8f0", fontsize=8)
        ax.text(i+w/2, b+0.8, f"{b:.0f}%", ha="center", color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

# ⑤ 時段勝率對比
def chart_session_compare():
    fig, ax = dark_fig(11, 4.8)
    order = ["asia", "europe", "us"]
    g_is  = is_seg.groupby("session")["win"].mean().reindex(order) * 100
    g_oos = oos_seg.groupby("session")["win"].mean().reindex(order) * 100
    x = np.arange(len(order)); w = 0.35
    ax.bar(x - w/2, g_is.values,  w, label="IS",  color="#64748b")
    ax.bar(x + w/2, g_oos.values, w, label="OOS", color="#22c55e")
    ax.axhline(50, color="#475569", ls="--")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_title("時段勝率對比 %（IS vs OOS）", fontsize=12)
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, (a, b) in enumerate(zip(g_is.values, g_oos.values)):
        ax.text(i-w/2, a+1, f"{a:.1f}%", ha="center", color="#e2e8f0", fontsize=8)
        ax.text(i+w/2, b+1, f"{b:.1f}%", ha="center", color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

print("生成圖表...")
imgs = dict(
    equity   = chart_equity_curve(),
    headline = chart_headline(),
    exitbd   = chart_exit_breakdown(),
    failcmp  = chart_fail_compare(),
    sess     = chart_session_compare(),
)
print("  圖表完成")

# ── 判斷標準（呼應 monthly_checklist.md 轉向訊號）──────────────────────────────
PF_THRESHOLD = 1.2
WR_DIFF_THRESHOLD = 8.0  # pp

pf_fail = s_oos["pf"] < PF_THRESHOLD
wr_diff = (s_is["wr"] - s_oos["wr"]) * 100
wr_fail = wr_diff > WR_DIFF_THRESHOLD

sl_is_pct  = ex_is.loc["SL","n"]/s_is["n"]*100   if "SL" in ex_is.index  else 0
sl_oos_pct = ex_oos.loc["SL","n"]/s_oos["n"]*100 if "SL" in ex_oos.index else 0
tp2_is_pct  = ex_is.loc["TP2","n"]/s_is["n"]*100   if "TP2" in ex_is.index  else 0
tp2_oos_pct = ex_oos.loc["TP2","n"]/s_oos["n"]*100 if "TP2" in ex_oos.index else 0
exit_struct_fail = (sl_oos_pct - sl_is_pct > 8) or (tp2_oos_pct < tp2_is_pct - 3)

overall_verdict = "warn" if (pf_fail or wr_fail or exit_struct_fail) else "good"

# ── HTML 片段工具 ────────────────────────────────────────────────────────────
def pct(v): return "—" if not np.isfinite(v) else f"{v*100:.1f}%"
def num(v, f="{:.3f}"): return "∞" if not np.isfinite(v) else f.format(v)

def headline_table():
    rows = [
        ("交易筆數", s_is["n"], s_oos["n"], ""),
        ("勝率", f"{s_is['wr']*100:.2f}%", f"{s_oos['wr']*100:.2f}%", f"{(s_oos['wr']-s_is['wr'])*100:+.1f} pp"),
        ("獲利因子", f"{s_is['pf']:.3f}", f"{s_oos['pf']:.3f}", f"{s_oos['pf']-s_is['pf']:+.3f}"),
        ("淨盈虧", f"${s_is['net']:+,.0f}", f"${s_oos['net']:+,.0f}", f"${s_oos['net']-s_is['net']:+,.0f}"),
        ("最大回撤", f"${s_is['mdd']:,.0f}", f"${s_oos['mdd']:,.0f}", f"${s_oos['mdd']-s_is['mdd']:+,.0f}"),
        ("平均 R 倍數", f"{s_is['avg_r']:.3f}R", f"{s_oos['avg_r']:.3f}R", f"{s_oos['avg_r']-s_is['avg_r']:+.3f}R"),
        ("區間", f"{is_seg['entry_time'].min().date()} → {is_seg['exit_time'].max().date()}",
                f"{oos_seg['entry_time'].min().date()} → {oos_seg['exit_time'].max().date()}", ""),
    ]
    body = ""
    for name, a, b, d in rows:
        body += (f"<tr><td><strong>{name}</strong></td><td>{a}</td>"
                 f"<td style='color:#22c55e'>{b}</td><td style='color:#38bdf8'>{d}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>指標</th><th>In-Sample</th>"
            f"<th>Out-of-Sample</th><th>差異</th></tr></thead><tbody>{body}</tbody></table>")

def exit_table():
    body = ""
    for k in ["TP1", "TP2", "SL"]:
        a = ex_is.loc[k] if k in ex_is.index else None
        b = ex_oos.loc[k] if k in ex_oos.index else None
        an = int(a["n"]) if a is not None else 0
        bn = int(b["n"]) if b is not None else 0
        anet = a["net"] if a is not None else 0
        bnet = b["net"] if b is not None else 0
        ar = a["avg_r"] if a is not None else float("nan")
        br = b["avg_r"] if b is not None else float("nan")
        body += (f"<tr><td><strong>{k}</strong></td>"
                 f"<td>{an} ({an/s_is['n']*100:.1f}%)</td><td>{ar:.2f}R</td><td>${anet:+,.0f}</td>"
                 f"<td>{bn} ({bn/s_oos['n']*100:.1f}%)</td><td>{br:.2f}R</td><td>${bnet:+,.0f}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>出場</th>"
            f"<th>IS 筆數</th><th>IS 均R</th><th>IS 淨利</th>"
            f"<th>OOS 筆數</th><th>OOS 均R</th><th>OOS 淨利</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

def fail_table():
    order = ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]
    c_is  = fail_is["fail_type"].value_counts().reindex(order, fill_value=0)
    c_oos = fail_oos["fail_type"].value_counts().reindex(order, fill_value=0)
    body = ""
    for k in order:
        a, b = c_is[k], c_oos[k]
        ap = a/len(fail_is)*100  if len(fail_is)  else 0
        bp = b/len(fail_oos)*100 if len(fail_oos) else 0
        delta_color = "#22c55e" if bp <= ap else "#ef4444"
        body += (f"<tr><td><strong>{k}</strong></td>"
                 f"<td>{a} ({ap:.0f}%)</td><td>{b} ({bp:.0f}%)</td>"
                 f"<td style='color:{delta_color}'>{bp-ap:+.1f} pp</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>失敗類型</th><th>IS</th><th>OOS</th><th>變化</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

verdict_html = ""
if pf_fail:
    verdict_html += (f'<div class="warn"><strong>⚠️ PF(OOS) = {s_oos["pf"]:.3f} &lt; {PF_THRESHOLD}</strong>　'
                      f'觸發 monthly_checklist.md「轉向訊號 B」門檻：OOS 段的 edge 已經薄弱，'
                      f'建議暫緩投入 Regime 過濾器參數優化（#2）與入場/出場拆解（#3），優先重新檢視 S1 訊號本體。</div>')
if wr_fail:
    verdict_html += (f'<div class="warn"><strong>⚠️ 勝率落差 {wr_diff:+.1f}pp（IS→OOS）超過 {WR_DIFF_THRESHOLD}pp 門檻</strong>　'
                      f'edge 可能集中在 IS 那段特定行情（2024初～2025末，黃金主升段），近期資料上優勢已縮小。</div>')
if exit_struct_fail:
    verdict_html += (f'<div class="warn"><strong>⚠️ 出場結構在 OOS 段劣化</strong>　'
                      f'SL 觸發率 {sl_is_pct:.1f}%→{sl_oos_pct:.1f}%，TP2 佔比 {tp2_is_pct:.1f}%→{tp2_oos_pct:.1f}%，'
                      f'V3.7 相對 V3.4 引以為傲的「賠率改善」在最近一段資料沒有完全延續。</div>')
if not (pf_fail or wr_fail or exit_struct_fail):
    verdict_html = (f'<div class="good"><strong>✅ 三項門檻皆通過</strong>　'
                     f'PF(OOS)={s_oos["pf"]:.3f} ≥ {PF_THRESHOLD}，勝率落差 {wr_diff:+.1f}pp ≤ {WR_DIFF_THRESHOLD}pp，'
                     f'出場結構未明顯劣化。V3.7 的 edge 在時間切分下暫時站得住腳，可以繼續投入 #2（Regime 參數優化）'
                     f'與 #3（入場/出場貢獻拆解）。但這仍是單一切點的一次性檢驗，不是 walk-forward，'
                     f'建議後續每累積約 100-150 筆新交易後重跑本腳本，持續盯緊 edge 是否維持。</div>')

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
ul{margin:8px 0 8px 20px}li{margin:6px 0}
.ins-title{color:#f8fafc;font-weight:700;font-size:.98em}
</style>
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S1-AweWithBB V3.7 Out-of-Sample 測試</title>{CSS}</head>
<body>
<div style="max-width:1200px;margin:0 auto 14px"><a href="../index.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="wrap">

<h1>S1-AweWithBB V3.7 <span style="color:#f59e0b">Out-of-Sample</span> 測試</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
資料來源：同一份真實逐筆交易 CSV（{s_is['n']+s_oos['n']} 筆），依進場時間切 70/30，
無需重新回測（S1 訊號計算全程無未來函數，事後時間切分在統計上有效）</p>
<p class="note">切分點：<strong>{SPLIT_DATE.date()}</strong>（按 459 筆進場時間排序的第 70% 筆）。
其他報告：
<a href="report_v3.7_real.html" style="color:var(--blue)">report_v3.7_real.html（V3.4 vs V3.7 全樣本歸因）</a></p>

<div class="callout">
<strong>為什麼要做這個測試：</strong>
V3.7 的績效表是對整段 2024-01 ~ 2026-07 做全樣本評估，而策略設計（V3.4→V3.7 六次迭代）
是在同一段資料上反覆查看、反覆調整出來的——即使沒有正式參數優化，也可能無意間
擬合到這段歷史的特定行情（尤其是這段期間黃金以大多頭為主）。
本測試把最近 30% 的資料（Out-of-Sample）當成「策略設計時沒有特別盯著看」的資料，
檢驗 edge 是否還在。
</div>

<!-- ═══════════ 判斷結果 ═══════════ -->
<div class="card">
<div class="part">判斷結果</div>
<h2>OOS 段是否通過 monthly_checklist.md 的轉向訊號門檻？</h2>
{verdict_html}
</div>

<!-- ═══════════ PART 1 ═══════════ -->
<div class="card">
<div class="part">PART 1 — 權益曲線</div>
<h2>累積淨利走勢：切分點前後有沒有明顯轉折？</h2>
{img(imgs['equity'])}
<p class="note">灰色區間 = In-Sample（{s_is['n']} 筆），綠色區間 = Out-of-Sample（{s_oos['n']} 筆）。
若 OOS 段曲線明顯走平或向下，即使頭條數字沒有觸發門檻，也是需要留意的視覺訊號。</p>
</div>

<!-- ═══════════ PART 2 ═══════════ -->
<div class="card">
<div class="part">PART 2 — 頭條指標對比</div>
<h2>IS vs OOS 核心績效</h2>
{headline_table()}
{img(imgs['headline'])}
</div>

<!-- ═══════════ PART 3 ═══════════ -->
<div class="card">
<div class="part">PART 3 — 出場結構</div>
<h2>V3.7 引以為傲的「賠率改善」延續到 OOS 了嗎？</h2>
<p class="note">V3.4→V3.7 的淨利改善主要來自 SL 觸發率下降 + TP2 佔比提升（見 report_v3.7_real.html PART 2）。
這裡檢驗這個出場結構在最近的 OOS 段是否維持。</p>
{exit_table()}
{img(imgs['exitbd'])}
</div>

<!-- ═══════════ PART 4 ═══════════ -->
<div class="card">
<div class="part">PART 4 — 失敗模式對比</div>
<h2>OOS 段的虧損單，體質有變差嗎？</h2>
{fail_table()}
{img(imgs['failcmp'])}
</div>

<!-- ═══════════ PART 5 ═══════════ -->
<div class="card">
<div class="part">PART 5 — 時段勝率對比</div>
<h2>OOS 段是否仍維持相同的時段特性？</h2>
{img(imgs['sess'])}
<p class="note">若某個時段在 IS 段表現好、OOS 段明顯轉差，可能代表該時段的優勢是特定期間的巧合，而非穩定 edge。</p>
</div>

<p class="note" style="margin-top:16px">
方法說明：In-Sample / Out-of-Sample 皆為同一份 TradingView Strategy Tester 匯出的 List of Trades
真實成交紀錄，僅依進場時間切分，未重新回測、未重新調整任何參數。
R 倍數 = 淨盈虧 ÷（進場價 × 數量 × 0.5%）。失敗模式分類邏輯與門檻與既有
xauusd/analysis/fail_patterns.py 完全一致，確保與其他報告可比。
<br>局限：這是單一切點的一次性樣本外檢驗，不是滾動式 walk-forward optimization；
且策略參數本身並未在 IS 段做正式的數值優化再拿到 OOS 驗證，所以本測試驗證的是
「策略設計整體是否對這段歷史過擬合」，而非嚴謹意義上的「參數優化的樣本外泛化能力」。
</p>

<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S1-AweWithBB V3.7 Out-of-Sample 測試 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ OOS 測試報告已生成：{OUT_HTML}")
print(f"   IS : WR {s_is['wr']*100:.1f}% PF {s_is['pf']:.3f} Net ${s_is['net']:+,.0f} MDD ${s_is['mdd']:,.0f} (n={s_is['n']})")
print(f"   OOS: WR {s_oos['wr']*100:.1f}% PF {s_oos['pf']:.3f} Net ${s_oos['net']:+,.0f} MDD ${s_oos['mdd']:,.0f} (n={s_oos['n']})")
print(f"   判斷：PF門檻{'❌ 未通過' if pf_fail else '✅ 通過'} | 勝率落差門檻{'❌ 未通過' if wr_fail else '✅ 通過'} | 出場結構{'❌ 劣化' if exit_struct_fail else '✅ 未劣化'}")
