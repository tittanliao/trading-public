"""
run_s1_deep_analysis.py — S1-AweWithBB 深度分析報告 V2
=======================================================
輸出：xauusd/XAUUSD-Long-S1-AweWithBB/report_v2.html
V1 原始報告 (report.html) 不受影響。

執行方式（在 trading/ 根目錄，20260705 移至 scripts/ 子資料夾）：
    python3 xauusd/scripts/run_s1_deep_analysis.py
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# CJK 字型設定（Mac 優先用 PingFang HK，備用 Arial Unicode MS）
def _set_cjk_font():
    for name in ("PingFang HK", "Arial Unicode MS", "STHeiti", "Heiti TC"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
_set_cjk_font()

ROOT = Path(__file__).parent.parent.parent  # 20260705 移至 scripts/ 子資料夾，多一層 .parent
sys.path.insert(0, str(ROOT / "xauusd"))

from analysis import loader, fail_patterns
from analysis.config import STRATEGIES, PRICE_CSV, PRICE_CSV_4H, DXY_CSV_1D, XAUUSD_CSV_1D

# ── 路徑設定 ────────────────────────────────────────────────────────────────
S1_CFG   = next(c for c in STRATEGIES if c["id"] == "S1-AweWithBB")
TRADE_CSV = S1_CFG["folder"] / S1_CFG["trades_csv"]
OUT_HTML  = S1_CFG["folder"] / "report_v2.html"

SL_PCT  = 0.005   # 0.5% stop loss
TP2_R   = 3.5     # TP2 = 3.5R

# ── 共用工具 ─────────────────────────────────────────────────────────────────
def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#e2e8f0")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")
    return fig, ax

def dark_fig_multi(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("#0f172a")
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
    return fig, axes

# ── 1. 資料載入 ──────────────────────────────────────────────────────────────
print("載入交易資料...")
trades = loader.load_trades(TRADE_CSV)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
trades["session"]    = fail_patterns.tag_session(trades["entry_time"])
trades["entry_hour"] = trades["entry_time"].dt.hour
trades["entry_dow"]  = trades["entry_time"].dt.dayofweek   # 0=Mon
trades["entry_ym"]   = trades["entry_time"].dt.to_period("M")
trades["entry_q"]    = trades["entry_time"].dt.to_period("Q")
trades["win"]        = trades["result"] == "win"
# R 倍數 = 實際損益 / SL 金額（SL = 進場價 × 0.5%）
trades["r_multiple"] = trades["net_pnl_usd"] / (trades["entry_price"] * trades["size_qty"] * SL_PCT)

classified = fail_patterns.classify_fail(trades)
print(f"  總交易筆數：{len(trades)}，虧損：{len(classified)}")

print("載入 4H 價格資料...")
price_4h = loader.load_price(PRICE_CSV_4H) if PRICE_CSV_4H.exists() else None

print("載入 DXY 1D 資料...")
dxy_1d = loader.load_dxy(DXY_CSV_1D) if DXY_CSV_1D.exists() else None

# ── 2. Regime 計算（4H）────────────────────────────────────────────────────
def compute_regime(df4h: pd.DataFrame) -> pd.DataFrame:
    """在 4H OHLCV 資料上計算 EMA20 斜率 + BBW rank，並分類 Regime。"""
    df = df4h.copy().sort_values("time").reset_index(drop=True)
    df["ema20"]       = df["close"].ewm(span=20, adjust=False).mean()
    df["ema20_pre10"] = df["ema20"].shift(10)
    df["slope"]       = (df["ema20"] - df["ema20_pre10"]) / df["ema20_pre10"].replace(0, np.nan) * 100

    # BBW = 4σ / SMA20（近似 Bollinger Band Width）
    df["stdev20"] = df["close"].rolling(20).std()
    df["sma20"]   = df["close"].rolling(20).mean()
    df["bbw"]     = df["stdev20"] * 4 / df["sma20"].replace(0, np.nan)

    # BBW percentrank over 60 bars
    def prank(s, w=60):
        return s.rolling(w).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    df["bbw_rank"] = prank(df["bbw"])

    SLOPE_T = 0.15
    BBW_T   = 65.0
    def classify(row):
        s, b = row["slope"], row["bbw_rank"]
        if pd.isna(s) or pd.isna(b):
            return "UNKNOWN"
        if b >= BBW_T and s >= SLOPE_T:  return "BULL_TREND"
        if b >= BBW_T and s <= -SLOPE_T: return "BEAR_TREND"
        if b >= BBW_T:                   return "RANGE_FLAT"
        if s >= SLOPE_T:                 return "RANGE_UP"
        if s <= -SLOPE_T:                return "RANGE_DOWN"
        return "CONSOLIDATION"
    df["regime"] = df.apply(classify, axis=1)
    return df[["time","slope","bbw_rank","regime"]]

regime_df = compute_regime(price_4h) if price_4h is not None else None

# 把 Regime 合併到 trades（取進場前最近的 4H bar）
if regime_df is not None:
    trades_sorted = trades.sort_values("entry_time")
    regime_sorted = regime_df.sort_values("time")
    trades = pd.merge_asof(
        trades_sorted, regime_sorted,
        left_on="entry_time", right_on="time",
        direction="backward",
    ).drop(columns=["time"], errors="ignore")
    print(f"  Regime 合併完成，覆蓋率：{trades['regime'].notna().mean():.1%}")

# DXY RSI bucket
if dxy_1d is not None and "rsi" in dxy_1d.columns:
    dxy_1d["dxy_rsi"] = dxy_1d["rsi"]
    dxy_sorted = dxy_1d[["time","dxy_rsi"]].sort_values("time")
    trades = pd.merge_asof(
        trades.sort_values("entry_time"), dxy_sorted,
        left_on="entry_time", right_on="time",
        direction="backward",
    ).drop(columns=["time"], errors="ignore")
    def dxy_bucket(r):
        if pd.isna(r): return "N/A"
        if r < 30:  return "<30 超賣"
        if r < 50:  return "30-50"
        if r < 70:  return "50-70"
        return ">70 超買"
    trades["dxy_bucket"] = trades["dxy_rsi"].apply(dxy_bucket)
    print(f"  DXY RSI 合併完成")

# ─── 圖表生成 ──────────────────────────────────────────────────────────────

# ① Equity Curve + Drawdown
def chart_equity_drawdown():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios":[3,1]})
    fig.patch.set_facecolor("#0f172a")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    cumulative = trades["net_pnl_usd"].cumsum()
    dates      = trades["entry_time"]
    ax1.plot(dates, cumulative, color="#22c55e", lw=2)
    ax1.fill_between(dates, cumulative, 0, where=(cumulative >= 0), alpha=.15, color="#22c55e")
    ax1.fill_between(dates, cumulative, 0, where=(cumulative < 0),  alpha=.15, color="#ef4444")
    ax1.axhline(0, color="#475569", lw=0.8)
    ax1.set_title("累積盈虧曲線", color="#e2e8f0", fontsize=13)
    ax1.set_ylabel("USD", color="#94a3b8")
    # Drawdown
    peak = cumulative.cummax()
    dd   = cumulative - peak
    ax2.fill_between(dates, dd, 0, alpha=.7, color="#ef4444")
    ax2.set_ylabel("Drawdown", color="#94a3b8")
    ax2.set_xlabel("日期", color="#94a3b8")
    fig.tight_layout()
    return fig_b64(fig)

# ② Monthly Heatmap
def chart_monthly_heatmap():
    mn  = trades.groupby(trades["entry_ym"]).agg(
        wr=("win","mean"), pnl=("net_pnl_usd","sum"), n=("win","count"))
    mn  = mn.reset_index()
    mn["year"]  = mn["entry_ym"].dt.year
    mn["month"] = mn["entry_ym"].dt.month
    years  = sorted(mn["year"].unique())
    months = list(range(1, 13))
    pnl_mat = np.full((12, len(years)), np.nan)
    wr_mat  = np.full((12, len(years)), np.nan)
    for _, row in mn.iterrows():
        yi = years.index(row["year"])
        mi = row["month"] - 1
        pnl_mat[mi, yi] = row["pnl"]
        wr_mat[mi, yi]  = row["wr"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f172a")
    month_lbls = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cmap = LinearSegmentedColormap.from_list("rg", ["#ef4444","#1e293b","#22c55e"])
    norm_pnl = TwoSlopeNorm(vmin=np.nanmin(pnl_mat), vcenter=0, vmax=np.nanmax(pnl_mat))
    im1 = ax1.imshow(pnl_mat, cmap=cmap, norm=norm_pnl, aspect="auto")
    ax1.set_xticks(range(len(years))); ax1.set_xticklabels(years, color="#94a3b8")
    ax1.set_yticks(range(12)); ax1.set_yticklabels(month_lbls, color="#94a3b8")
    ax1.set_title("月度 PnL ($)", color="#e2e8f0"); ax1.set_facecolor("#0f172a")
    for i in range(12):
        for j in range(len(years)):
            v = pnl_mat[i,j]
            if not np.isnan(v):
                ax1.text(j, i, f"${v:.0f}", ha="center", va="center",
                         color="white" if abs(v) > np.nanmax(abs(pnl_mat)) * .4 else "#94a3b8", fontsize=8)
    plt.colorbar(im1, ax=ax1, fraction=.03)

    norm_wr = mcolors.Normalize(vmin=0, vmax=1)
    im2 = ax2.imshow(wr_mat, cmap=LinearSegmentedColormap.from_list("rg2",["#ef4444","#f59e0b","#22c55e"]),
                     norm=norm_wr, aspect="auto")
    ax2.set_xticks(range(len(years))); ax2.set_xticklabels(years, color="#94a3b8")
    ax2.set_yticks(range(12)); ax2.set_yticklabels(month_lbls, color="#94a3b8")
    ax2.set_title("月度勝率 (%)", color="#e2e8f0"); ax2.set_facecolor("#0f172a")
    for i in range(12):
        for j in range(len(years)):
            v = wr_mat[i,j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=8,
                         color="white")
    plt.colorbar(im2, ax=ax2, fraction=.03)
    fig.suptitle("月度績效熱力圖", color="#e2e8f0", fontsize=14)
    fig.patch.set_facecolor("#0f172a")
    fig.tight_layout()
    return fig_b64(fig)

# ③ Rolling 30-trade WR
def chart_rolling_wr():
    fig, ax = dark_fig(12, 4)
    roll_wr = trades["win"].rolling(30).mean() * 100
    ax.plot(trades["entry_time"], roll_wr, color="#38bdf8", lw=2)
    ax.axhline(50, color="#475569", lw=1, ls="--")
    ax.axhline(trades["win"].mean() * 100, color="#22c55e", lw=1, ls="--", label=f"整體 {trades['win'].mean()*100:.1f}%")
    ax.fill_between(trades["entry_time"], roll_wr, 50,
                    where=(roll_wr >= 50), alpha=.15, color="#22c55e")
    ax.fill_between(trades["entry_time"], roll_wr, 50,
                    where=(roll_wr < 50),  alpha=.15, color="#ef4444")
    ax.set_title("滾動 30 筆勝率", color="#e2e8f0")
    ax.set_ylabel("%", color="#94a3b8")
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig_b64(fig)

# ④ Hour × Weekday Heatmap
def chart_hour_dow_heatmap():
    pivot = trades.groupby(["entry_dow","entry_hour"])["win"].mean().unstack(fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    cmap = LinearSegmentedColormap.from_list("rg3", ["#ef4444","#1e293b","#22c55e"])
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot.index)], color="#94a3b8")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(h) for h in pivot.columns], color="#94a3b8", fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v - 0.5) > 0.2 else "#94a3b8")
    ax.set_title("星期 × 小時 勝率熱力圖（%）", color="#e2e8f0")
    ax.set_xlabel("進場小時（UTC+8）", color="#94a3b8")
    plt.colorbar(im, ax=ax, fraction=.02)
    fig.tight_layout()
    return fig_b64(fig)

# ⑤ R-multiple distribution
def chart_r_dist():
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    r = trades["r_multiple"].clip(-3, 6)
    bins = np.linspace(-3, 6, 40)
    wins_r   = r[trades["win"]]
    losses_r = r[~trades["win"]]
    ax1.hist(wins_r,   bins=bins, color="#22c55e", alpha=.7, label=f"獲利 (n={len(wins_r)})")
    ax1.hist(losses_r, bins=bins, color="#ef4444", alpha=.7, label=f"虧損 (n={len(losses_r)})")
    for rv, lbl, c in [(1.0,"TP1 (1R)","#f59e0b"),(3.5,"TP2 (3.5R)","#22c55e"),(-1.0,"SL (-1R)","#ef4444")]:
        ax1.axvline(rv, color=c, ls="--", lw=1.2, label=lbl)
    ax1.set_title("R 倍數分佈", color="#e2e8f0")
    ax1.set_xlabel("R 倍數", color="#94a3b8")
    ax1.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=8)

    # CDF
    sorted_r = np.sort(trades["r_multiple"].dropna())
    cdf = np.arange(1, len(sorted_r)+1) / len(sorted_r)
    ax2.plot(sorted_r, cdf, color="#38bdf8", lw=2)
    ax2.axvline(0, color="#475569", ls="--")
    ax2.axhline(1 - trades["win"].mean(), color="#ef4444", ls="--", lw=1, label="虧損比例")
    ax2.set_title("R 倍數累積分佈（CDF）", color="#e2e8f0")
    ax2.set_xlabel("R 倍數", color="#94a3b8")
    ax2.set_ylabel("累積機率", color="#94a3b8")
    ax2.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

# ⑥ Hold time analysis
def chart_hold_time():
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    w_hold = trades.loc[trades["win"],  "hold_bars"]
    l_hold = trades.loc[~trades["win"], "hold_bars"]
    ax1.hist(w_hold, bins=30, color="#22c55e", alpha=.7, label=f"獲利 (n={len(w_hold)})")
    ax1.hist(l_hold, bins=30, color="#ef4444", alpha=.7, label=f"虧損 (n={len(l_hold)})")
    ax1.set_title("持倉時間分佈（30m bars）", color="#e2e8f0")
    ax1.set_xlabel("持倉根數（1根=30分鐘）", color="#94a3b8")
    ax1.legend(facecolor="#1e293b", labelcolor="#94a3b8")

    ax2.scatter(trades.loc[trades["win"],  "hold_bars"],
                trades.loc[trades["win"],  "r_multiple"].clip(-3,6),
                color="#22c55e", alpha=.4, s=18, label="獲利")
    ax2.scatter(trades.loc[~trades["win"], "hold_bars"],
                trades.loc[~trades["win"], "r_multiple"].clip(-3,6),
                color="#ef4444", alpha=.4, s=18, label="虧損")
    ax2.axhline(0, color="#475569", lw=0.8)
    ax2.set_title("持倉時間 vs R 倍數", color="#e2e8f0")
    ax2.set_xlabel("持倉根數", color="#94a3b8")
    ax2.set_ylabel("R 倍數", color="#94a3b8")
    ax2.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig_b64(fig)

# ⑦ Fail pattern deep dive
def chart_fail_deep():
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    colors_map = {"immediate_loss":"#ef4444","false_breakout":"#f59e0b",
                  "time_bleed":"#8b5cf6","normal_sl":"#64748b"}
    fail_cnt = classified["fail_type"].value_counts()
    ax1.barh(fail_cnt.index, fail_cnt.values,
             color=[colors_map.get(k,"#64748b") for k in fail_cnt.index])
    ax1.set_title("失敗類型分佈", color="#e2e8f0")
    ax1.set_xlabel("筆數", color="#94a3b8")
    for i, v in enumerate(fail_cnt.values):
        ax1.text(v+0.3, i, f"{v} ({v/len(classified)*100:.0f}%)",
                 va="center", color="#94a3b8", fontsize=9)

    # immediate_loss by hour
    imm = classified[classified["fail_type"] == "immediate_loss"]
    if len(imm) > 0:
        h_cnt = imm["entry_time"].dt.hour.value_counts().sort_index()
        ax2.bar(h_cnt.index, h_cnt.values, color="#ef4444", alpha=.8, width=0.7)
        ax2.set_title(f"Immediate Loss 進場小時分佈（n={len(imm)}）", color="#e2e8f0")
        ax2.set_xlabel("小時（UTC+8）", color="#94a3b8")
        ax2.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig_b64(fig)

# ⑧ Regime analysis
def chart_regime():
    if "regime" not in trades.columns or trades["regime"].isna().all():
        return None
    rg = trades.groupby("regime").agg(
        n=("win","count"), wr=("win","mean"), avg_r=("r_multiple","mean"))
    rg = rg.sort_values("wr", ascending=False)
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    regime_colors = {
        "BULL_TREND":"#22c55e","BEAR_TREND":"#ef4444","RANGE_UP":"#38bdf8",
        "RANGE_DOWN":"#f59e0b","CONSOLIDATION":"#94a3b8","RANGE_FLAT":"#8b5cf6","UNKNOWN":"#475569"
    }
    colors = [regime_colors.get(r,"#64748b") for r in rg.index]
    bars = ax1.bar(range(len(rg)), rg["wr"]*100, color=colors)
    ax1.axhline(50, color="#475569", ls="--")
    ax1.set_xticks(range(len(rg))); ax1.set_xticklabels(rg.index, rotation=15, color="#94a3b8", fontsize=9)
    ax1.set_title("Regime × 勝率（4H）", color="#e2e8f0"); ax1.set_ylabel("%", color="#94a3b8")
    for bar, (_, row) in zip(bars, rg.iterrows()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{row['wr']*100:.0f}%\n(n={row['n']:.0f})", ha="center", color="#e2e8f0", fontsize=8)

    ax2.bar(range(len(rg)), rg["avg_r"], color=colors)
    ax2.axhline(0, color="#475569", lw=0.8)
    ax2.set_xticks(range(len(rg))); ax2.set_xticklabels(rg.index, rotation=15, color="#94a3b8", fontsize=9)
    ax2.set_title("Regime × 平均 R 倍數（4H）", color="#e2e8f0"); ax2.set_ylabel("avg R", color="#94a3b8")
    for i, (_, row) in enumerate(rg.iterrows()):
        ax2.text(i, row["avg_r"] + (0.02 if row["avg_r"] >= 0 else -0.07),
                 f"{row['avg_r']:.2f}R", ha="center", color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

# ⑨ Session & Hour bar charts
def chart_session_hour():
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    sess_g = trades.groupby("session").agg(wr=("win","mean"), n=("win","count"))
    sess_colors = {"asia":"#38bdf8","europe":"#a78bfa","us":"#f59e0b"}
    ax1.bar(sess_g.index, sess_g["wr"]*100,
            color=[sess_colors.get(s,"#64748b") for s in sess_g.index])
    ax1.axhline(50, color="#475569", ls="--")
    ax1.set_title("時段勝率", color="#e2e8f0"); ax1.set_ylabel("%", color="#94a3b8")
    for i, (s, row) in enumerate(sess_g.iterrows()):
        ax1.text(i, row["wr"]*100 + 1, f"{row['wr']*100:.1f}%\n(n={row['n']})",
                 ha="center", color="#e2e8f0", fontsize=9)

    hr_g = trades.groupby("entry_hour").agg(wr=("win","mean"), n=("win","count"))
    bar_colors = ["#38bdf8" if 7 <= h <= 15 else "#a78bfa" if 16 <= h <= 21 else "#f59e0b"
                  for h in hr_g.index]
    ax2.bar(hr_g.index, hr_g["wr"]*100, color=bar_colors, width=0.8)
    ax2.axhline(50, color="#475569", ls="--")
    ax2.set_title("小時勝率（藍=亞盤 紫=歐盤 橙=美盤）", color="#e2e8f0")
    ax2.set_xlabel("小時（UTC+8）", color="#94a3b8"); ax2.set_ylabel("%", color="#94a3b8")
    ax2.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig_b64(fig)

# ⑩ DXY bucket
def chart_dxy():
    if "dxy_bucket" not in trades.columns:
        return None
    dg = trades.groupby("dxy_bucket").agg(wr=("win","mean"), n=("win","count"), avg_r=("r_multiple","mean"))
    order = ["<30 超賣","30-50","50-70",">70 超買","N/A"]
    dg = dg.reindex([o for o in order if o in dg.index])
    fig, ax = dark_fig(10, 5)
    colors = ["#22c55e","#64748b","#f59e0b","#ef4444","#475569"][:len(dg)]
    bars = ax.bar(range(len(dg)), dg["wr"]*100, color=colors)
    ax.axhline(50, color="#475569", ls="--")
    ax.set_xticks(range(len(dg))); ax.set_xticklabels(dg.index, color="#94a3b8")
    ax.set_title("DXY RSI 分區勝率", color="#e2e8f0"); ax.set_ylabel("%", color="#94a3b8")
    for bar, (_, row) in zip(bars, dg.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{row['wr']*100:.1f}%\n(n={row['n']:.0f})", ha="center", color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    return fig_b64(fig)

# ── 統計計算 ──────────────────────────────────────────────────────────────
def compute_stats(t):
    pf_wins  = t.loc[t["win"], "net_pnl_usd"].sum()
    pf_loss  = abs(t.loc[~t["win"], "net_pnl_usd"].sum())
    cum = t["net_pnl_usd"].cumsum()
    peak = cum.cummax()
    mdd  = (cum - peak).min()
    return dict(
        n=len(t), wr=t["win"].mean(), pf=pf_wins/pf_loss if pf_loss else float("inf"),
        net_pnl=t["net_pnl_usd"].sum(), mdd=mdd,
        avg_hold=t["hold_bars"].mean(), avg_r=t["r_multiple"].mean(),
        max_consec=max((sum(1 for _ in g) for k, g in
                        __import__("itertools").groupby(t["win"].tolist()) if not k), default=0),
    )

stats = compute_stats(trades)

# Quarterly table
q_tbl = trades.groupby("entry_q").agg(
    n=("win","count"), wr=("win","mean"), pnl=("net_pnl_usd","sum"),
    avg_r=("r_multiple","mean")).reset_index()

# ── 生成圖表 ──────────────────────────────────────────────────────────────
print("生成圖表...")
imgs = {}
imgs["equity"]   = chart_equity_drawdown()
imgs["monthly"]  = chart_monthly_heatmap()
imgs["roll_wr"]  = chart_rolling_wr()
imgs["hour_dow"] = chart_hour_dow_heatmap()
imgs["r_dist"]   = chart_r_dist()
imgs["hold"]     = chart_hold_time()
imgs["fail"]     = chart_fail_deep()
imgs["regime"]   = chart_regime()
imgs["session"]  = chart_session_hour()
imgs["dxy"]      = chart_dxy()
print("  圖表生成完成")

# ── 季度表格 HTML ─────────────────────────────────────────────────────────
def q_table_html(q_tbl):
    rows = ""
    for _, r in q_tbl.iterrows():
        pnl_c = "#22c55e" if r["pnl"] >= 0 else "#ef4444"
        wr_c  = "#22c55e" if r["wr"] >= 0.5 else "#f59e0b" if r["wr"] >= 0.4 else "#ef4444"
        rows += f"""<tr>
          <td>{r['entry_q']}</td><td>{r['n']:.0f}</td>
          <td style="color:{wr_c}">{r['wr']*100:.1f}%</td>
          <td style="color:{pnl_c}">${r['pnl']:+.0f}</td>
          <td style="color:{'#22c55e' if r['avg_r']>=0 else '#ef4444'}">{r['avg_r']:.2f}R</td>
        </tr>"""
    return f"""<table class="tbl"><thead><tr><th>季度</th><th>筆數</th><th>勝率</th><th>PnL</th><th>平均R</th></tr></thead><tbody>{rows}</tbody></table>"""

def regime_table_html():
    if "regime" not in trades.columns:
        return ""
    rg = trades.groupby("regime").agg(
        n=("win","count"), wr=("win","mean"), avg_r=("r_multiple","mean"),
        avg_hold=("hold_bars","mean")).sort_values("wr", ascending=False)
    rows = ""
    regime_desc = {
        "BULL_TREND":"slope≥0.15% + BBW≥65%（趨勢上攻）",
        "BEAR_TREND":"slope≤-0.15% + BBW≥65%（趨勢下跌）",
        "RANGE_UP":"slope≥0.15% + BBW<65%（緩升）",
        "RANGE_DOWN":"slope≤-0.15% + BBW<65%（緩降）",
        "RANGE_FLAT":"BBW≥65% + |slope|<0.15%（橫盤高波）",
        "CONSOLIDATION":"|slope|<0.15% + BBW<65%（盤整低波）",
        "UNKNOWN":"數據不足",
    }
    for regime, row in rg.iterrows():
        wr_c  = "#22c55e" if row["wr"] >= 0.55 else "#f59e0b" if row["wr"] >= 0.45 else "#ef4444"
        r_c   = "#22c55e" if row["avg_r"] >= 0 else "#ef4444"
        rows += f"""<tr>
          <td><strong>{regime}</strong><br><span style="color:var(--muted);font-size:.8em">{regime_desc.get(regime,"")}</span></td>
          <td>{row['n']:.0f}</td>
          <td style="color:{wr_c}">{row['wr']*100:.1f}%</td>
          <td style="color:{r_c}">{row['avg_r']:.2f}R</td>
          <td>{row['avg_hold']:.1f} bars ({row['avg_hold']*0.5:.0f}h)</td>
        </tr>"""
    return f"""<table class="tbl"><thead><tr><th>Regime</th><th>筆數</th><th>勝率</th><th>avg R</th><th>avg 持倉</th></tr></thead><tbody>{rows}</tbody></table>"""

def fail_table_html():
    ft = classified.groupby("fail_type").agg(
        n=("trade_id","count"), avg_hold=("hold_bars","mean"),
        avg_mfe=("mfe_pct","mean"), avg_mae=("mae_pct","mean")).reset_index()
    colors = {"immediate_loss":"#ef4444","false_breakout":"#f59e0b","time_bleed":"#8b5cf6","normal_sl":"#64748b"}
    rows = ""
    for _, r in ft.iterrows():
        c = colors.get(r["fail_type"],"#64748b")
        rows += f"""<tr>
          <td style="color:{c};font-weight:700">{r['fail_type']}</td>
          <td>{r['n']:.0f} ({r['n']/len(classified)*100:.0f}%)</td>
          <td>{r['avg_hold']:.1f} ({r['avg_hold']*0.5:.0f}h)</td>
          <td>{r['avg_mfe']:.3f}%</td><td>{r['avg_mae']:.3f}%</td>
        </tr>"""
    return f"""<table class="tbl"><thead><tr><th>失敗類型</th><th>筆數</th><th>avg 持倉</th><th>avg MFE</th><th>avg MAE</th></tr></thead><tbody>{rows}</tbody></table>"""

# ── HTML 組裝 ─────────────────────────────────────────────────────────────
def img(key):
    b = imgs.get(key)
    if not b: return "<p style='color:#64748b'>（此分析需要對應數據）</p>"
    return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'

CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}
:root{--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#38bdf8;--muted:#94a3b8;--border:#334155}
.wrap{max-width:1200px;margin:0 auto}
h1{font-size:1.6em;margin-bottom:4px;color:#f8fafc}
h2{font-size:1.1em;color:#38bdf8;margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
h3{font-size:.95em;color:#94a3b8;margin:14px 0 8px}
.card{background:#1e293b;border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.metric{background:#0f172a;border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center}
.metric .lbl{font-size:.75em;color:var(--muted);margin-bottom:6px}
.metric .val{font-size:1.5em;font-weight:700}
.metric .sub{font-size:.75em;color:var(--muted);margin-top:4px}
.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}.blue{color:var(--blue)}
.tbl{width:100%;border-collapse:collapse;font-size:.84em;margin:10px 0}
.tbl th{background:#0f172a;color:var(--muted);padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}
.tbl td{padding:8px 12px;border-bottom:1px solid rgba(51,65,85,.5);color:#e2e8f0}
.tbl tr:hover td{background:rgba(255,255,255,.02)}
.badge{display:inline-block;padding:2px 8px;border-radius:8px;font-size:.75em;font-weight:700}
.bg-green{background:rgba(34,197,94,.15);color:var(--green)}
.bg-red{background:rgba(239,68,68,.15);color:var(--red)}
.note{font-size:.8em;color:var(--muted);margin-top:8px}
</style>
"""

def wr_color(v):
    return "green" if v >= 0.55 else "yellow" if v >= 0.45 else "red"

HTML = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<title>S1-AweWithBB V2 深度分析報告</title>{CSS}</head>
<body><div class="wrap">
<h1>S1-AweWithBB 深度分析報告 V2</h1>
<p class="note">基於 V3.4 交易 CSV · 生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · 總計 {stats['n']} 筆交易</p>
<p class="note">V1 原始報告：<a href="report.html" style="color:var(--blue)">report.html</a> &nbsp;|&nbsp; V2 深度報告（本頁）：report_v2.html</p>

<div class="card">
<h2>§0 Executive Summary</h2>
<div class="grid4">
  <div class="metric"><div class="lbl">勝率</div><div class="val {wr_color(stats['wr'])}">{stats['wr']*100:.1f}%</div><div class="sub">{stats['n']} 筆交易</div></div>
  <div class="metric"><div class="lbl">獲利因子</div><div class="val {'green' if stats['pf']>1 else 'red'}">{stats['pf']:.3f}</div><div class="sub">淨盈虧 ${stats['net_pnl']:+.0f}</div></div>
  <div class="metric"><div class="lbl">最大回撤</div><div class="val red">${stats['mdd']:.0f}</div><div class="sub">({stats['mdd']/10000*100:.2f}% of 初始資金)</div></div>
  <div class="metric"><div class="lbl">平均 R 倍數</div><div class="val {'green' if stats['avg_r']>0 else 'red'}">{stats['avg_r']:.3f}R</div><div class="sub">avg 持倉 {stats['avg_hold']:.0f} bars ({stats['avg_hold']*0.5:.0f}h)</div></div>
  <div class="metric"><div class="lbl">虧損筆數</div><div class="val red">{len(classified)}</div><div class="sub">最大連虧 {stats['max_consec']} 筆</div></div>
  <div class="metric"><div class="lbl">Immediate Loss</div><div class="val red">{len(classified[classified['fail_type']=='immediate_loss'])}</div><div class="sub">({len(classified[classified['fail_type']=='immediate_loss'])/max(len(classified),1)*100:.0f}% of 虧損)</div></div>
  <div class="metric"><div class="lbl">False Breakout</div><div class="val yellow">{len(classified[classified['fail_type']=='false_breakout'])}</div><div class="sub">({len(classified[classified['fail_type']=='false_breakout'])/max(len(classified),1)*100:.0f}% of 虧損)</div></div>
  <div class="metric"><div class="lbl">Time Bleed</div><div class="val blue">{len(classified[classified['fail_type']=='time_bleed'])}</div><div class="sub">({len(classified[classified['fail_type']=='time_bleed'])/max(len(classified),1)*100:.0f}% of 虧損)</div></div>
</div>
</div>

<div class="card">
<h2>§1 時間軸績效</h2>
<h3>累積盈虧曲線 + 回撤</h3>
{img("equity")}
<h3>月度績效熱力圖</h3>
{img("monthly")}
<h3>滾動 30 筆勝率</h3>
{img("roll_wr")}
<h3>季度績效明細</h3>
{q_table_html(q_tbl)}
</div>

<div class="card">
<h2>§2 進場時機分析</h2>
<h3>時段勝率 + 小時勝率</h3>
{img("session")}
<h3>星期 × 小時 勝率熱力圖</h3>
{img("hour_dow")}
</div>

<div class="card">
<h2>§3 R 倍數分析</h2>
{img("r_dist")}
<p class="note">TP1=1R · TP2=3.5R（V3.7 最佳值）· SL=-1R 為基準線</p>
</div>

<div class="card">
<h2>§4 持倉時間分析</h2>
{img("hold")}
</div>

<div class="card">
<h2>§5 失敗模式深度分析</h2>
{img("fail")}
{fail_table_html()}
</div>

<div class="card">
<h2>§6 Regime 環境分析（4H EMA20 斜率 × BBW Rank）</h2>
{img("regime") if imgs.get("regime") else "<p class='note'>（需要 4H 價格 CSV）</p>"}
{regime_table_html()}
<p class="note">Regime 定義：slope = (EMA20_now − EMA20_10bars_ago) / EMA20_10bars_ago × 100 · BBW rank = PercentRank(4σ/SMA20, 60) 在 4H 時框計算</p>
</div>

<div class="card">
<h2>§7 DXY 相關性分析</h2>
{img("dxy") if imgs.get("dxy") else "<p class='note'>（需要 DXY 1D CSV）</p>"}
<p class="note">DXY RSI 超賣（&lt;30）代表美元弱，對黃金通常有利。中性區（30-50）歷史上 S1 表現最差。</p>
</div>

</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ V2 深度報告已生成：{OUT_HTML}")
print(f"   V1 原始報告位置：{S1_CFG['folder'] / 'report.html'}")
