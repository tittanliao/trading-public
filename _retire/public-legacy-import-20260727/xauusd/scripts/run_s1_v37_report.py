"""
run_s1_v37_report.py — S1-AweWithBB V3.7 完整分析報告
======================================================
輸出：xauusd/XAUUSD-Long-S1-AweWithBB/report_v3.7.html

本報告把「V3.7 TradingView 回測驗證結果」與「V3.4 交易資料的失敗模式分析」
結合起來，並用 V3.4 交易資料模擬 V3.7 新增過濾器（1H MA(3) + 1H BBW High）
會攔截掉哪些交易、對勝率／獲利因子的影響。

我們沒有 V3.7 的逐筆交易 CSV（TradingView 只給了彙總統計），因此：
  - PART 1：直接展示 V3.7 TV 回測彙總（confirmed validation）
  - PART 2：用 V3.4 逐筆交易模擬 V3.7 過濾器效果
  - PART 3：以 V3.4 資料做失敗模式分析（V3.7 過濾器設計的依據）
  - PART 4：書面洞察與建議

執行方式（在 trading/ 根目錄，20260705 移至 scripts/ 子資料夾）：
    python3.12 xauusd/scripts/run_s1_v37_report.py
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

ROOT = Path(__file__).parent.parent.parent  # 20260705 移至 scripts/ 子資料夾，多一層 .parent
sys.path.insert(0, str(ROOT / "xauusd"))

from analysis import loader, fail_patterns, metrics
from analysis.config import STRATEGIES, PRICE_CSV_60M

# ── 路徑 ────────────────────────────────────────────────────────────────────
S1_CFG    = next(c for c in STRATEGIES if c["id"] == "S1-AweWithBB")
TRADE_CSV = S1_CFG["folder"] / S1_CFG["trades_csv"]
OUT_HTML  = S1_CFG["folder"] / "report_v3.7.html"

SL_PCT = 0.005   # 0.5% stop loss（用於 R 倍數換算）

# ── V3.7 TradingView 回測彙總（confirmed validation，非本地重算）──────────────
V37 = dict(
    n=459, wins=253, wr=0.5512, pf=1.743,
    net_pnl=7577.0, net_pct=75.78, mdd=391.0, mdd_pct=3.04,
    period="2024-01-02 → 2026-07-04",
    tp2_r=3.5, time_exit_bars=36,
    filters=[
        ("1H MA(3) 趨勢過濾器", "ON",  "close > SMA(close,3) @1H 才允許進場"),
        ("1H BBW 高檔過濾器",   "ON",  "PercentRank(4σ/SMA20, 60) @1H ≥ 70% 阻擋進場"),
        ("4H BBW 低檔過濾器",   "OFF", "壓縮帶假突破防護（保留備用）"),
        ("TP2 盈虧比",          "3.5R", "由固定 R:R 改為分段 TP1(1R)+TP2(3.5R)"),
        ("時間止損",            "36 bars", "由 48 bars(24h) 縮短為 36 bars(18h)"),
    ],
)

# ── 1. 載入 V3.4 交易資料 ────────────────────────────────────────────────────
print("載入 V3.4 交易資料...")
trades = loader.load_trades(TRADE_CSV)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
trades["session"]    = fail_patterns.tag_session(trades["entry_time"])
trades["entry_hour"] = trades["entry_time"].dt.hour
trades["win"]        = trades["result"] == "win"
trades["r_multiple"] = trades["net_pnl_usd"] / (trades["entry_price"] * trades["size_qty"] * SL_PCT)

classified = fail_patterns.classify_fail(trades)
print(f"  總交易筆數：{len(trades)}，虧損：{len(classified)}")

# ── 2. 載入 1H 價格並計算 V3.7 過濾器指標 ────────────────────────────────────
print("載入 1H 價格並計算過濾器指標...")
px = loader.load_price(PRICE_CSV_60M).sort_values("time").reset_index(drop=True)
# 1H MA(3)
px["ma3"] = px["close"].rolling(3).mean()
# 1H BBW rank(60)：PercentRank( StDev(close,20)*4 / SMA(close,20), 60 )
px["bbw"] = px["close"].rolling(20).std() * 4 / px["close"].rolling(20).mean()
px["bbw_rank"] = px["bbw"].rolling(60).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
# 用「已收盤」的 1H bar 值（模擬 request.security lookahead_off）：
# bar 時戳為開盤時間，收盤時間 = 時戳 + 1H。只採用收盤時間 <= 進場時間的 bar。
px["bar_close_time"] = px["time"] + pd.Timedelta(hours=1)

merged = pd.merge_asof(
    trades.sort_values("entry_time"),
    px[["bar_close_time", "close", "ma3", "bbw_rank"]].rename(columns={"close": "close_1h"}),
    left_on="entry_time", right_on="bar_close_time",
    direction="backward",
)
merged = merged.rename(columns={"close_1h": "px_close_1h"})

# 覆蓋範圍：只有 1H CSV 涵蓋（2025-10-15 起）且指標非 NaN 的交易可模擬
cov = merged.dropna(subset=["ma3", "bbw_rank", "px_close_1h"]).copy()
cov_n = len(cov)
cov_pct = cov_n / len(trades) * 100
print(f"  1H 過濾器可模擬交易：{cov_n}/{len(trades)} ({cov_pct:.0f}%) —"
      f" 涵蓋 {cov['entry_time'].min().date()} → {cov['entry_time'].max().date()}")

# V3.7 過濾器判定（在覆蓋子集上）
cov["ma_block"]  = cov["px_close_1h"] <= cov["ma3"]        # 收在 MA3 之下 → 攔截
cov["bbw_block"] = cov["bbw_rank"]   >= 70.0               # 1H BBW rank 高檔 → 攔截
cov["v37_block"] = cov["ma_block"] | cov["bbw_block"]
cov["v37_pass"]  = ~cov["v37_block"]

# ── 統計工具 ─────────────────────────────────────────────────────────────────
def stat_block(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), net=0.0, avg_r=float("nan"))
    wins = t.loc[t["win"], "net_pnl_usd"].sum()
    loss = abs(t.loc[~t["win"], "net_pnl_usd"].sum())
    return dict(
        n=len(t), wr=t["win"].mean(),
        pf=(wins / loss) if loss else float("inf"),
        net=t["net_pnl_usd"].sum(), avg_r=t["r_multiple"].mean(),
    )

s_v34_full = stat_block(trades)             # V3.4 全樣本
s_cov_all  = stat_block(cov)                # 覆蓋子集：不過濾
s_cov_pass = stat_block(cov[cov["v37_pass"]])   # 覆蓋子集：V3.7 通過
s_blk_ma   = stat_block(cov[cov["ma_block"]])   # 被 MA 攔截
s_blk_bbw  = stat_block(cov[cov["bbw_block"]])  # 被 BBW 攔截
s_blk_all  = stat_block(cov[cov["v37_block"]])  # 任一攔截

# 被攔截交易中的失敗類型分佈
blk_losers = cov[cov["v37_block"] & ~cov["win"]].copy()
blk_losers = blk_losers.merge(
    classified[["trade_id", "fail_type"]], on="trade_id", how="left")
pass_losers = cov[cov["v37_pass"] & ~cov["win"]].copy()
pass_losers = pass_losers.merge(
    classified[["trade_id", "fail_type"]], on="trade_id", how="left")

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
    return (fig, axes) if n > 1 else (fig, axes)

def img(b):
    if not b:
        return "<p style='color:#64748b'>（此分析需要對應數據）</p>"
    return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'

# ① V3.4 vs V3.7 對比（WR / PF）
def chart_v34_vs_v37():
    fig, (a1, a2) = dark_fig(12, 4.5, n=2)
    labels = ["V3.4\n(504)", "V3.7\n(459)"]
    # WR
    wr = [s_v34_full["wr"] * 100, V37["wr"] * 100]
    b1 = a1.bar(labels, wr, color=["#64748b", "#22c55e"], width=0.55)
    a1.axhline(50, color="#475569", ls="--")
    a1.set_title("勝率 %", fontsize=12); a1.set_ylim(0, 65)
    for bar, v in zip(b1, wr):
        a1.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.1f}%",
                ha="center", color="#e2e8f0", fontsize=11, fontweight="bold")
    # PF
    pf = [s_v34_full["pf"], V37["pf"]]
    b2 = a2.bar(labels, pf, color=["#64748b", "#22c55e"], width=0.55)
    a2.axhline(1.0, color="#475569", ls="--")
    a2.set_title("獲利因子", fontsize=12); a2.set_ylim(0, 2.1)
    for bar, v in zip(b2, pf):
        a2.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.3f}",
                ha="center", color="#e2e8f0", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig_b64(fig)

# ② 過濾器模擬：分組勝率 + 獲利因子
def chart_filter_sim():
    fig, (a1, a2) = dark_fig(13, 5, n=2)
    groups = [
        ("覆蓋全樣本\n(未過濾)", s_cov_all,  "#64748b"),
        ("V3.7 通過",          s_cov_pass, "#22c55e"),
        ("被 MA 攔截",         s_blk_ma,   "#ef4444"),
        ("被 BBW 攔截",        s_blk_bbw,  "#f59e0b"),
    ]
    names  = [g[0] for g in groups]
    wr     = [g[1]["wr"] * 100 for g in groups]
    pf     = [min(g[1]["pf"], 3.0) if np.isfinite(g[1]["pf"]) else 0 for g in groups]
    ns     = [g[1]["n"] for g in groups]
    colors = [g[2] for g in groups]

    b1 = a1.bar(range(len(groups)), wr, color=colors, width=0.6)
    a1.axhline(s_cov_all["wr"] * 100, color="#38bdf8", ls="--", lw=1,
               label=f"基準 {s_cov_all['wr']*100:.1f}%")
    a1.set_xticks(range(len(groups)))
    a1.set_xticklabels(names, fontsize=9)
    a1.set_title("各群組勝率 %", fontsize=12); a1.set_ylim(0, 80)
    a1.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=8)
    for i, (v, nn) in enumerate(zip(wr, ns)):
        a1.text(i, v + 1, f"{v:.1f}%\n(n={nn})", ha="center",
                color="#e2e8f0", fontsize=9)

    b2 = a2.bar(range(len(groups)), pf, color=colors, width=0.6)
    a2.axhline(1.0, color="#475569", ls="--")
    a2.set_xticks(range(len(groups)))
    a2.set_xticklabels(names, fontsize=9)
    a2.set_title("各群組獲利因子（上限截 3.0）", fontsize=12); a2.set_ylim(0, 3.2)
    for i, g in enumerate(groups):
        pfv = g[1]["pf"]
        txt = "∞" if not np.isfinite(pfv) else f"{pfv:.2f}"
        a2.text(i, min(pfv, 3.0) + 0.05 if np.isfinite(pfv) else 0.05, txt,
                ha="center", color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    return fig_b64(fig)

# ③ 被攔截 vs 通過 的失敗類型分佈
def chart_blocked_failtypes():
    order = ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]
    cmap  = {"immediate_loss": "#ef4444", "false_breakout": "#f59e0b",
             "time_bleed": "#8b5cf6", "normal_sl": "#64748b"}
    blk_c  = blk_losers["fail_type"].value_counts().reindex(order, fill_value=0)
    pass_c = pass_losers["fail_type"].value_counts().reindex(order, fill_value=0)
    fig, ax = dark_fig(11, 4.8)
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, blk_c.values,  w, label=f"被 V3.7 攔截的虧損 (n={blk_c.sum()})",
           color="#ef4444", alpha=0.85)
    ax.bar(x + w/2, pass_c.values, w, label=f"V3.7 通過的虧損 (n={pass_c.sum()})",
           color="#38bdf8", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
    ax.set_title("失敗類型：被攔截 vs 放行（僅覆蓋子集虧損單）", fontsize=12)
    ax.set_ylabel("筆數")
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
    for i, v in enumerate(blk_c.values):
        if v: ax.text(i - w/2, v + 0.2, str(v), ha="center", color="#e2e8f0", fontsize=8)
    for i, v in enumerate(pass_c.values):
        if v: ax.text(i + w/2, v + 0.2, str(v), ha="center", color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

# ④ V3.4 失敗類型分佈 + 進場小時
def chart_v34_fail():
    fig, (a1, a2) = dark_fig(13, 5, n=2)
    cmap = {"immediate_loss": "#ef4444", "false_breakout": "#f59e0b",
            "time_bleed": "#8b5cf6", "normal_sl": "#64748b"}
    fc = classified["fail_type"].value_counts()
    a1.barh(fc.index, fc.values, color=[cmap.get(k, "#64748b") for k in fc.index])
    a1.set_title("V3.4 失敗類型分佈", fontsize=12); a1.set_xlabel("筆數")
    for i, v in enumerate(fc.values):
        a1.text(v + 0.4, i, f"{v} ({v/len(classified)*100:.0f}%)",
                va="center", color="#94a3b8", fontsize=9)
    imm = classified[classified["fail_type"] == "immediate_loss"]
    hc = imm["entry_time"].dt.hour.value_counts().sort_index()
    a2.bar(hc.index, hc.values, color="#ef4444", alpha=0.85, width=0.7)
    a2.set_title(f"Immediate Loss 進場小時（n={len(imm)}）", fontsize=12)
    a2.set_xlabel("小時 UTC+8"); a2.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig_b64(fig)

# ⑤ V3.4 時段 / 小時勝率
def chart_v34_session():
    fig, (a1, a2) = dark_fig(13, 5, n=2)
    sg = trades.groupby("session").agg(wr=("win", "mean"), n=("win", "count"))
    sc = {"asia": "#38bdf8", "europe": "#a78bfa", "us": "#f59e0b"}
    a1.bar(sg.index, sg["wr"] * 100, color=[sc.get(s, "#64748b") for s in sg.index])
    a1.axhline(50, color="#475569", ls="--")
    a1.set_title("V3.4 時段勝率", fontsize=12); a1.set_ylabel("%")
    for i, (s, row) in enumerate(sg.iterrows()):
        a1.text(i, row["wr"]*100 + 1, f"{row['wr']*100:.1f}%\n(n={row['n']})",
                ha="center", color="#e2e8f0", fontsize=9)
    hg = trades.groupby("entry_hour").agg(wr=("win", "mean"), n=("win", "count"))
    bc = ["#38bdf8" if 7 <= h <= 15 else "#a78bfa" if 16 <= h <= 21 else "#f59e0b"
          for h in hg.index]
    a2.bar(hg.index, hg["wr"] * 100, color=bc, width=0.8)
    a2.axhline(50, color="#475569", ls="--")
    a2.set_title("V3.4 小時勝率（藍=亞 紫=歐 橙=美）", fontsize=12)
    a2.set_xlabel("小時 UTC+8"); a2.set_ylabel("%"); a2.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig_b64(fig)

print("生成圖表...")
imgs = dict(
    cmp    = chart_v34_vs_v37(),
    sim    = chart_filter_sim(),
    blk    = chart_blocked_failtypes(),
    fail   = chart_v34_fail(),
    sess   = chart_v34_session(),
)
print("  圖表完成")

# ── HTML 片段工具 ────────────────────────────────────────────────────────────
def wr_cls(v): return "green" if v >= 0.55 else "yellow" if v >= 0.45 else "red"

def pct(v): return "—" if not np.isfinite(v) else f"{v*100:.1f}%"
def num(v, f="{:.3f}"): return "∞" if not np.isfinite(v) else f.format(v)

# V3.4 vs V3.7 對比表
def cmp_table():
    rows = [
        ("交易筆數",  f"{s_v34_full['n']}",              f"{V37['n']}",                 ""),
        ("勝率",      f"{s_v34_full['wr']*100:.2f}%",    f"{V37['wr']*100:.2f}%",       f"+{(V37['wr']-s_v34_full['wr'])*100:.1f} pp"),
        ("獲利因子",  f"{s_v34_full['pf']:.3f}",         f"{V37['pf']:.3f}",            f"+{V37['pf']-s_v34_full['pf']:.3f}"),
        ("淨盈虧",    f"${s_v34_full['net']:+,.0f}",     f"${V37['net_pnl']:+,.0f} (+{V37['net_pct']:.1f}%)", f"+${V37['net_pnl']-s_v34_full['net']:,.0f}"),
        ("最大回撤",  f"${abs(metrics.max_drawdown(trades)):,.0f}", f"${V37['mdd']:,.0f} ({V37['mdd_pct']:.2f}%)", "更低"),
        ("回測區間",  "2024-01 → 2026-04",               V37["period"],                 "+2 月"),
        ("時間止損",  "48 bars (24h)",                   f"{V37['time_exit_bars']} bars (18h)", "縮短"),
        ("TP 結構",   "固定 2:1",                        f"TP1 1R + TP2 {V37['tp2_r']}R", "分段"),
    ]
    body = ""
    for name, a, b, d in rows:
        body += (f"<tr><td><strong>{name}</strong></td><td>{a}</td>"
                 f"<td style='color:#22c55e'>{b}</td>"
                 f"<td style='color:#38bdf8'>{d}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>指標</th><th>V3.4（基準）</th>"
            f"<th>V3.7（確認版）</th><th>改善</th></tr></thead><tbody>{body}</tbody></table>")

def filter_cfg_table():
    body = ""
    for name, state, desc in V37["filters"]:
        badge = ("bg-green" if state == "ON" else
                 "bg-red" if state == "OFF" else "bg-blue")
        body += (f"<tr><td><strong>{name}</strong></td>"
                 f"<td><span class='badge {badge}'>{state}</span></td>"
                 f"<td style='color:#94a3b8'>{desc}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>過濾器 / 參數</th><th>狀態</th>"
            f"<th>說明</th></tr></thead><tbody>{body}</tbody></table>")

def sim_table():
    rows = [
        ("覆蓋全樣本（未套用 V3.7 過濾器）", s_cov_all,  "#94a3b8"),
        ("V3.7 通過（兩過濾器皆放行）",     s_cov_pass, "#22c55e"),
        ("被 1H MA(3) 攔截",               s_blk_ma,   "#ef4444"),
        ("被 1H BBW 高檔攔截",             s_blk_bbw,  "#f59e0b"),
        ("被任一過濾器攔截",               s_blk_all,  "#8b5cf6"),
    ]
    body = ""
    for name, s, c in rows:
        body += (f"<tr><td style='color:{c};font-weight:600'>{name}</td>"
                 f"<td>{s['n']}</td>"
                 f"<td class='{wr_cls(s['wr']) if np.isfinite(s['wr']) else ''}'>{pct(s['wr'])}</td>"
                 f"<td>{num(s['pf'])}</td>"
                 f"<td>{num(s['avg_r'], '{:.3f}R')}</td>"
                 f"<td style='color:{'#22c55e' if s['net']>=0 else '#ef4444'}'>${s['net']:+,.0f}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>群組</th><th>筆數</th><th>勝率</th>"
            f"<th>獲利因子</th><th>平均R</th><th>淨盈虧</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

def v34_fail_table():
    ft = classified.groupby("fail_type").agg(
        n=("trade_id", "count"), avg_hold=("hold_bars", "mean"),
        avg_mfe=("mfe_pct", "mean"), avg_mae=("mae_pct", "mean")).reset_index()
    cmap = {"immediate_loss": "#ef4444", "false_breakout": "#f59e0b",
            "time_bleed": "#8b5cf6", "normal_sl": "#64748b"}
    body = ""
    for _, r in ft.sort_values("n", ascending=False).iterrows():
        c = cmap.get(r["fail_type"], "#64748b")
        body += (f"<tr><td style='color:{c};font-weight:700'>{r['fail_type']}</td>"
                 f"<td>{r['n']:.0f} ({r['n']/len(classified)*100:.0f}%)</td>"
                 f"<td>{r['avg_hold']:.1f} ({r['avg_hold']*0.5:.0f}h)</td>"
                 f"<td>{r['avg_mfe']:.3f}%</td><td>{r['avg_mae']:.3f}%</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>失敗類型</th><th>筆數</th>"
            f"<th>avg 持倉</th><th>avg MFE</th><th>avg MAE</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

# 攔截效果的關鍵數字（給洞察引用）
imm_total   = (classified["fail_type"] == "immediate_loss").sum()
imm_blocked = (blk_losers["fail_type"] == "immediate_loss").sum()
imm_pass    = (pass_losers["fail_type"] == "immediate_loss").sum()
blk_win     = cov[cov["v37_block"] & cov["win"]].shape[0]
blk_loss    = cov[cov["v37_block"] & ~cov["win"]].shape[0]

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
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.metric{background:#0f172a;border:1px solid var(--border);border-radius:8px;padding:16px;text-align:center}
.metric .lbl{font-size:.75em;color:var(--muted);margin-bottom:6px}
.metric .val{font-size:1.7em;font-weight:700}
.metric .sub{font-size:.72em;color:var(--muted);margin-top:4px}
.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}.blue{color:var(--blue)}
.tbl{width:100%;border-collapse:collapse;font-size:.85em;margin:10px 0}
.tbl th{background:#0f172a;color:var(--muted);padding:9px 12px;text-align:left;border-bottom:1px solid var(--border)}
.tbl td{padding:8px 12px;border-bottom:1px solid rgba(51,65,85,.5)}
.tbl tr:hover td{background:rgba(255,255,255,.02)}
.badge{display:inline-block;padding:2px 10px;border-radius:8px;font-size:.75em;font-weight:700}
.bg-green{background:rgba(34,197,94,.18);color:var(--green)}
.bg-red{background:rgba(239,68,68,.18);color:var(--red)}
.bg-blue{background:rgba(56,189,248,.18);color:var(--blue)}
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
<title>S1-AweWithBB V3.7 完整分析報告</title>{CSS}</head>
<body>
<div style="max-width:1200px;margin:0 auto 14px"><a href="../index.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="wrap">

<h1>S1-AweWithBB <span style="color:#22c55e">V3.7</span> 完整分析報告</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
V3.7 為 TradingView 已確認回測版本 · 失敗模式分析基於 V3.4 逐筆交易（{len(trades)} 筆）</p>
<p class="note">其他報告：
<a href="report.html" style="color:var(--blue)">report.html（V3.4 原始）</a> ·
<a href="report_v2.html" style="color:var(--blue)">report_v2.html（V2 深度）</a></p>

<!-- ═══════════ PART 1 ═══════════ -->
<div class="card">
<div class="part">PART 1 — TradingView 回測驗證</div>
<h2>V3.7 已確認績效</h2>
<div class="grid4">
  <div class="metric"><div class="lbl">勝率</div><div class="val green">{V37['wr']*100:.2f}%</div><div class="sub">{V37['wins']}/{V37['n']} 筆</div></div>
  <div class="metric"><div class="lbl">獲利因子</div><div class="val green">{V37['pf']:.3f}</div><div class="sub">V3.4 為 {s_v34_full['pf']:.3f}</div></div>
  <div class="metric"><div class="lbl">淨盈虧</div><div class="val green">+${V37['net_pnl']:,.0f}</div><div class="sub">+{V37['net_pct']:.1f}%（初始 $10k）</div></div>
  <div class="metric"><div class="lbl">最大回撤</div><div class="val yellow">${V37['mdd']:,.0f}</div><div class="sub">{V37['mdd_pct']:.2f}% of 權益</div></div>
</div>
<div class="good"><strong>核心結論：</strong>V3.7 相較 V3.4 勝率 +1.9pp、獲利因子 1.525 → 1.743、
淨利提升約 24%，同時<strong>最大回撤反而下降</strong>（$494 → $391），回測區間還多涵蓋了近 2 個月。
這是「風險下降 + 報酬上升」的正向偏移，而非單純加大曝險。</div>

<h3>V3.4 → V3.7 對比</h3>
{cmp_table()}
{img(imgs['cmp'])}

<h3>V3.7 過濾器 / 參數設定</h3>
{filter_cfg_table()}
</div>

<!-- ═══════════ PART 2 ═══════════ -->
<div class="card">
<div class="part">PART 2 — 過濾器效果模擬</div>
<h2>V3.7 新過濾器會攔截 V3.4 的哪些交易？</h2>
<p class="note">方法：把 V3.4 每筆進場對應到「已收盤」的 1H K 棒（模擬 <code>request.security(lookahead_off)</code>），
計算 1H SMA(3) 與 1H BBW PercentRank(60)，再判定 V3.7 的兩個過濾器是否會攔截該筆。</p>
<div class="warn"><strong>資料涵蓋限制：</strong>1H CSV 僅涵蓋 <strong>2025-10-15 起</strong>，
因此僅 <strong>{cov_n}/{len(trades)} 筆（{cov_pct:.0f}%）</strong> V3.4 交易可被模擬
（{cov['entry_time'].min().date()} → {cov['entry_time'].max().date()}）。
以下結論屬「近半年子樣本」的方向性證據，非全樣本。</div>

<h3>各群組績效</h3>
{sim_table()}
{img(imgs['sim'])}
<div class="warn">
<strong>反直覺發現：入場過濾器在此子樣本近乎中性。</strong>
攔截 {s_blk_all['n']} 筆後，通過組勝率 <strong>{pct(s_cov_pass['wr'])}</strong>
幾乎等於未過濾的 <strong>{pct(s_cov_all['wr'])}</strong>（PF {num(s_cov_all['pf'],'{:.2f}')}→{num(s_cov_pass['pf'],'{:.2f}')}）。
拆開兩個過濾器：<strong>1H BBW 高檔</strong>攔掉 {s_blk_bbw['n']} 筆、勝率僅 {pct(s_blk_bbw['wr'])}（低於平均，方向正確）；
但 <strong>1H MA(3)</strong> 只攔掉 {s_blk_ma['n']} 筆，而這批勝率高達 {pct(s_blk_ma['wr'])}、PF {num(s_blk_ma['pf'],'{:.2f}')} ——
MA 過濾在這個子樣本其實砍掉了「好單」。兩者相抵，<strong>入場過濾的淨貢獻幾乎為零</strong>。
</div>

<h3>被攔截 vs 放行：失敗類型分佈</h3>
{img(imgs['blk'])}
<div class="warn">
更關鍵：被攔截的虧損單裡 immediate_loss <strong>只有 {imm_blocked} 筆</strong>，
放行組反而留下 <strong>{imm_pass} 筆</strong> immediate_loss（被攔截組 immediate_loss 佔比 {imm_blocked/max(blk_loss,1)*100:.0f}% ＜ 放行組 {imm_pass/max((~cov['v37_block']&~cov['win']).sum(),1)*100:.0f}%）。
換言之，這兩個入場過濾器<strong>並沒有優先濾掉 V3.4 最大痛點 immediate_loss</strong>。
被攔的多是普通 normal_sl。這強烈暗示：V3.7 的績效提升<strong>不是來自入場過濾</strong>。
</div>
</div>

<!-- ═══════════ PART 3 ═══════════ -->
<div class="card">
<div class="part">PART 3 — V3.4 失敗模式（V3.7 改善基礎）</div>
<h2>V3.7 過濾器針對的原始病灶</h2>
<p class="note">以下為 V3.4 全樣本（{len(classified)} 筆虧損）的失敗模式，是 V3.7 設計過濾器時的分析依據。</p>
<h3>失敗類型分佈 + Immediate Loss 進場時段</h3>
{img(imgs['fail'])}
{v34_fail_table()}
<h3>時段 / 小時勝率</h3>
{img(imgs['sess'])}
<div class="callout">V3.4 的 immediate_loss 集中在特定時段的進場（多為缺乏 1H 趨勢確認、或在波動已擴張的帶狀行情追高）。
V3.7 的 1H MA(3)（要求 1H 收在短均之上）＋ 1H BBW 高檔過濾（避開波動末端）正是對應解法。</div>
</div>

<!-- ═══════════ PART 4 ═══════════ -->
<div class="card">
<div class="part">PART 4 — 洞察與建議</div>
<h2>深度解讀：V3.7 的邊際優勢是結構性還是回測運氣？</h2>

<div class="good">
<p class="ins-title">1. 改善是「風險─報酬同向改善」，可信度高</p>
<p>V3.7 在勝率、獲利因子、淨利三項同時提升的情況下，最大回撤還<strong>下降 21%</strong>（$494→$391）。
若只是靠放大單筆曝險或挑到好時段，回撤通常會同步變大。回撤與報酬反向移動，較符合「濾掉了尾部虧損」的結構性改善。</p>
</div>

<div class="warn">
<p class="ins-title">2. 入場過濾器在近半年樣本近乎中性——甚至 MA(3) 砍到好單</p>
<p>用 V3.4 逐筆模擬 V3.7 的兩個入場過濾器，通過組勝率 {pct(s_cov_pass['wr'])} 幾乎等於未過濾的 {pct(s_cov_all['wr'])}，PF 僅 {num(s_cov_all['pf'],'{:.2f}')}→{num(s_cov_pass['pf'],'{:.2f}')}。
其中 1H MA(3) 攔掉的 {s_blk_ma['n']} 筆勝率高達 {pct(s_blk_ma['wr'])}（PF {num(s_blk_ma['pf'],'{:.2f}')}），等於砍掉獲利單；只有 1H BBW 高檔攔的 {s_blk_bbw['n']} 筆勝率 {pct(s_blk_bbw['wr'])} 方向正確。
而且被攔的虧損單裡 immediate_loss 只有 {imm_blocked} 筆——<strong>過濾器並沒有對準 V3.4 的主要病灶</strong>。</p>
</div>

<div class="good">
<p class="ins-title">3. 因此 V3.7 的邊際優勢幾乎確定來自「出場端」，而非入場過濾</p>
<p>兩條證據交叉指向出場結構：(a) 入場過濾模擬淨貢獻近零（見上）；(b) V3.7 交易數 459 vs V3.4 504，只少 45 筆（約 9%），
若入場過濾常態觸發，筆數應下降更多。真正改變獲利分佈的是 <strong>TP2 3.5R 分段出場</strong>（讓贏家跑更遠、提高單筆期望 R）
＋ <strong>時間止損 48→36 bars</strong>（更快斬斷 time_bleed）。這也解釋了為何「回撤下降的同時獲利上升」——是<strong>賠率（R 分佈）改善</strong>，不是勝率驅動。</p>
</div>

<div class="warn">
<p class="ins-title">4. 保留疑慮：樣本涵蓋僅 {cov_pct:.0f}%，且過濾方向未跨年驗證</p>
<p>(a) 入場過濾模擬只覆蓋 {cov_pct:.0f}% 的 V3.4 交易（1H CSV 只到 2025-10），2024 全年未納入。
(b) S1 是「右側突破 + AO 動能」策略、理論上偏好波動擴張，但 V3.7 卻在 1H BBW rank ≥ 70 時<strong>阻擋</strong>進場（避開追在波動末端）。
此判斷在近半年成立，但在單邊趨勢年（如 2024 上半年金價主升段）可能反而錯過行情，需分年檢查 BBW 過濾是否穩健。</p>
</div>

<div class="callout">
<p class="ins-title">下一步驗證實驗（單一、可執行）</p>
<p><strong>取得 V3.7 的逐筆交易 CSV，做「歸因拆解」回測：</strong>在 TradingView 上跑 4 個變體並各匯出交易清單——
(A) V3.4 基準、(B) 只加 1H MA+BBW 入場過濾（出場維持 48 bars/固定2:1）、(C) 只改出場（TP2 3.5R + 36 bars，入場不過濾）、
(D) 完整 V3.7。比較 (B) vs (C) 的 ΔPF / ΔMDD，就能判定 V3.7 的邊際優勢主要來自<strong>入場過濾</strong>還是<strong>出場結構</strong>。
若答案是出場結構，則應把同樣的 TP2/時間止損套用到 S2A / S2B 上，預期也能受益。</p>
</div>

<p class="note" style="margin-top:16px">
註：PART 1 的 V3.7 數字取自 TradingView 回測彙總（無逐筆 CSV）；PART 2–3 的計算基於本地 V3.4 逐筆交易與 1H 價格 CSV。
過濾器模擬採 lookahead_off 語意（僅用進場前已收盤的 1H bar），與 Pine 一致。
</p>
</div>

<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S1-AweWithBB V3.7 完整分析 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ V3.7 完整報告已生成：{OUT_HTML}")
print(f"   覆蓋子集：{cov_n} 筆，攔截 {s_blk_all['n']} 筆"
      f"（勝率 {pct(s_blk_all['wr'])}）vs 通過 {s_cov_pass['n']} 筆（勝率 {pct(s_cov_pass['wr'])}）")
