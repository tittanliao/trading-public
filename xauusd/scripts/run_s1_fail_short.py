"""
run_s1_fail_short.py — S1 虧損進場點反向做空測試
================================================
邏輯：對每一筆 S1 虧損交易，在相同進場點執行一筆模擬空單。
      分析此反向策略的勝率、R 倍數分佈，並依失敗類型、時段、小時細分。

空單規則：
  進場：S1 虧損的進場價（相同進場時間）
  SL   = entry × (1 + 0.5%)  → 1R 上方
  TP1  = entry × (1 - 0.5%)  → 1R 下方
  TP2  = entry × (1 - 1.75%) → 3.5R 下方
  時間止損：36 根 30m K 棒（≈ 18 小時）

輸出：xauusd/XAUUSD-Long-S1-AweWithBB/report_s1_fail_short.html

執行方式（在 trading/ 根目錄，20260705 移至 scripts/ 子資料夾）：
    python3 xauusd/scripts/run_s1_fail_short.py
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

from analysis import loader, fail_patterns
from analysis.config import STRATEGIES, PRICE_CSV

# ── 常數 ────────────────────────────────────────────────────────────────────
S1_CFG    = next(c for c in STRATEGIES if c["id"] == "S1-AweWithBB")
TRADE_CSV = S1_CFG["folder"] / S1_CFG["trades_csv"]
OUT_HTML  = S1_CFG["folder"] / "report_s1_fail_short.html"

SHORT_SL_PCT  = 0.005   # 0.5% above entry
SHORT_TP1_PCT = 0.005   # 0.5% below entry  (1R)
SHORT_TP2_PCT = 0.0175  # 1.75% below entry (3.5R)
TIME_LIMIT    = 36      # bars

# ── 工具函式 ─────────────────────────────────────────────────────────────────
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

# ── 資料載入 ─────────────────────────────────────────────────────────────────
print("載入 S1 交易資料...")
trades  = loader.load_trades(TRADE_CSV)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["session"]    = fail_patterns.tag_session(trades["entry_time"])
trades["entry_hour"] = trades["entry_time"].dt.hour

classified = fail_patterns.classify_fail(trades)
losses = classified.copy()
print(f"  總 S1 交易：{len(trades)} 筆，虧損（模擬空單基礎）：{len(losses)} 筆")

print("載入 30m 價格資料...")
price = loader.load_price(PRICE_CSV)
price = price.sort_values("time").reset_index(drop=True)
print(f"  30m K 棒：{len(price)} 根，時間範圍 {price['time'].min()} ~ {price['time'].max()}")

# ── 模擬空單 ─────────────────────────────────────────────────────────────────
def simulate_short(entry_time: pd.Timestamp, entry_price: float) -> dict:
    """
    從 entry_time 之後的第一根 bar 開始走帳，計算空單結果。
    回傳 dict 包含 result_tp1, result_tp2, exit_bars, exit_price, r_tp1, r_tp2。
    """
    sl_price  = entry_price * (1 + SHORT_SL_PCT)
    tp1_price = entry_price * (1 - SHORT_TP1_PCT)
    tp2_price = entry_price * (1 - SHORT_TP2_PCT)

    # 找到 entry_time 之後的 bar 索引
    idx_arr = price.index[price["time"] >= entry_time]
    if len(idx_arr) == 0:
        return dict(result_tp1="no_data", result_tp2="no_data",
                    exit_bars=0, exit_price=np.nan, r_tp1=np.nan, r_tp2=np.nan)

    start_idx = idx_arr[0]
    end_idx   = min(start_idx + TIME_LIMIT, len(price) - 1)

    hit_tp1 = hit_tp2 = hit_sl = False
    exit_bar = TIME_LIMIT
    exit_price = price.loc[end_idx, "close"] if end_idx < len(price) else entry_price

    for i in range(start_idx, end_idx + 1):
        bar = price.loc[i]
        bar_idx = i - start_idx

        # 空單：low <= TP, high >= SL
        if bar["low"] <= tp2_price and not hit_sl:
            hit_tp2 = True
            exit_bar = bar_idx
            exit_price = tp2_price
            break
        if bar["low"] <= tp1_price and not hit_sl:
            hit_tp1 = True
            exit_bar = bar_idx
            exit_price = tp1_price
            # 繼續走帳看是否 TP2 也能達到（用同一根 bar 的 high/low 範圍估計）
            # 簡化：若 tp2 在同根 bar 也觸及，算 TP2
            if bar["low"] <= tp2_price:
                hit_tp2 = True
            break
        if bar["high"] >= sl_price:
            hit_sl = True
            exit_bar = bar_idx
            exit_price = sl_price
            break

    # 時間止損（hit_sl 隱含在時間限內，否則走完 36 bars）
    r_sl = -1.0
    r_tp1_val = 1.0
    r_tp2_val = 3.5

    if hit_tp2:
        res_tp1 = "win_tp1"
        res_tp2 = "win_tp2"
        r_tp1   = r_tp1_val
        r_tp2   = r_tp2_val
    elif hit_tp1:
        res_tp1 = "win_tp1"
        res_tp2 = "loss_time" if not hit_sl else "loss_sl"
        r_tp1   = r_tp1_val
        r_tp2   = (entry_price - exit_price) / (entry_price * SHORT_TP1_PCT)
    elif hit_sl:
        res_tp1 = "loss_sl"
        res_tp2 = "loss_sl"
        r_tp1   = r_sl
        r_tp2   = r_sl
    else:
        # 時間止損
        res_tp1 = "loss_time"
        res_tp2 = "loss_time"
        r_tp1   = (entry_price - exit_price) / (entry_price * SHORT_TP1_PCT)
        r_tp2   = r_tp1

    return dict(result_tp1=res_tp1, result_tp2=res_tp2,
                exit_bars=exit_bar, exit_price=exit_price,
                r_tp1=r_tp1, r_tp2=r_tp2)

print("執行模擬空單...")
results = []
for _, row in losses.iterrows():
    sim = simulate_short(row["entry_time"], row["entry_price"])
    sim["trade_id"]   = row["trade_id"]
    sim["entry_time"] = row["entry_time"]
    sim["entry_price"]= row["entry_price"]
    sim["fail_type"]  = row["fail_type"]
    sim["session"]    = row["session"]
    sim["entry_hour"] = row["entry_hour"]
    results.append(sim)

sims = pd.DataFrame(results)
sims["win_tp1"] = sims["result_tp1"].str.startswith("win")
sims["win_tp2"] = sims["result_tp2"] == "win_tp2"

total = len(sims)
valid = sims[sims["result_tp1"] != "no_data"]
print(f"  模擬完成：{total} 筆，有效（有對應K棒）：{len(valid)} 筆")

# ── 統計摘要 ─────────────────────────────────────────────────────────────────
def summary(df, label="全部"):
    n    = len(df)
    wr1  = df["win_tp1"].mean()
    wr2  = df["win_tp2"].mean()
    avg_r1 = df["r_tp1"].mean()
    avg_r2 = df["r_tp2"].mean()
    wins1  = df["win_tp1"].sum()
    wins2  = df["win_tp2"].sum()
    return dict(label=label, n=n, wr1=wr1, wr2=wr2,
                avg_r1=avg_r1, avg_r2=avg_r2, wins1=wins1, wins2=wins2)

overall = summary(valid, "全部虧損")
print(f"\n全部虧損 → 空單 TP1 勝率：{overall['wr1']*100:.1f}%  TP2 勝率：{overall['wr2']*100:.1f}%")

# ── 圖表 ─────────────────────────────────────────────────────────────────────

def chart_overview():
    """TP1 vs TP2 勝率總覽（橫條）"""
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    cats = ["全部", "immediate_loss", "false_breakout", "time_bleed", "normal_sl"]
    wr1_vals, wr2_vals, ns = [], [], []
    for c in cats:
        if c == "全部":
            df = valid
        else:
            df = valid[valid["fail_type"] == c]
        wr1_vals.append(df["win_tp1"].mean() * 100 if len(df) else 0)
        wr2_vals.append(df["win_tp2"].mean() * 100 if len(df) else 0)
        ns.append(len(df))

    x = range(len(cats))
    w = 0.35
    colors = ["#38bdf8", "#ef4444", "#f59e0b", "#8b5cf6", "#64748b"]
    bars1 = ax1.bar([i - w/2 for i in x], wr1_vals, w, label="TP1 (0.5%)", color=colors, alpha=.9)
    bars2 = ax1.bar([i + w/2 for i in x], wr2_vals, w, label="TP2 (1.75%)", color=colors, alpha=.4)
    ax1.axhline(50, color="#475569", ls="--", lw=1)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(cats, rotation=15, ha="right", color="#94a3b8", fontsize=9)
    ax1.set_ylabel("勝率 %", color="#94a3b8"); ax1.set_title("各失敗類型 空單勝率", color="#e2e8f0")
    ax1.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    for b, v, n in zip(bars1, wr1_vals, ns):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f"{v:.0f}%\n(n={n})", ha="center", color="#e2e8f0", fontsize=8)

    # 各類型 TP2 avg R
    avg_r2_vals = []
    for c in cats:
        df = valid if c == "全部" else valid[valid["fail_type"] == c]
        avg_r2_vals.append(df["r_tp2"].mean() if len(df) else 0)
    bar_c = ["#22c55e" if v >= 0 else "#ef4444" for v in avg_r2_vals]
    ax2.bar(list(x), avg_r2_vals, color=bar_c, alpha=.85)
    ax2.axhline(0, color="#475569", lw=0.8)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(cats, rotation=15, ha="right", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("avg R", color="#94a3b8"); ax2.set_title("各失敗類型 空單 TP2 平均 R", color="#e2e8f0")
    fig.tight_layout()
    return fig_b64(fig)

def chart_session():
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    sess = ["asia", "europe", "us"]
    wr1_s, wr2_s, ns_s = [], [], []
    for s in sess:
        df = valid[valid["session"] == s]
        wr1_s.append(df["win_tp1"].mean() * 100 if len(df) else 0)
        wr2_s.append(df["win_tp2"].mean() * 100 if len(df) else 0)
        ns_s.append(len(df))
    sess_colors = {"asia":"#38bdf8","europe":"#a78bfa","us":"#f59e0b"}
    cols = [sess_colors[s] for s in sess]
    x = range(len(sess))
    w = 0.35
    ax1.bar([i - w/2 for i in x], wr1_s, w, color=cols, alpha=.9, label="TP1")
    ax1.bar([i + w/2 for i in x], wr2_s, w, color=cols, alpha=.45, label="TP2")
    ax1.axhline(50, color="#475569", ls="--")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(sess, color="#94a3b8")
    ax1.set_title("時段空單勝率", color="#e2e8f0"); ax1.set_ylabel("%", color="#94a3b8")
    ax1.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    for i, (w1, w2, n) in enumerate(zip(wr1_s, wr2_s, ns_s)):
        ax1.text(i - 0.175, w1 + 0.5, f"{w1:.0f}%\n(n={n})", ha="center", color="#e2e8f0", fontsize=9)

    # Hour bar chart (TP1 WR)
    hr_g = valid.groupby("entry_hour").agg(
        wr1=("win_tp1","mean"), wr2=("win_tp2","mean"), n=("win_tp1","count"))
    hour_colors = ["#38bdf8" if 7 <= h <= 15 else "#a78bfa" if 16 <= h <= 21 else "#f59e0b"
                   for h in hr_g.index]
    ax2.bar(hr_g.index, hr_g["wr1"] * 100, color=hour_colors, alpha=.9, width=0.8, label="TP1")
    ax2.bar(hr_g.index, hr_g["wr2"] * 100, color=hour_colors, alpha=.4, width=0.8, label="TP2")
    ax2.axhline(50, color="#475569", ls="--")
    ax2.set_title("小時空單勝率（藍=亞盤 紫=歐盤 橙=美盤）", color="#e2e8f0")
    ax2.set_xlabel("小時（UTC+8）", color="#94a3b8"); ax2.set_ylabel("%", color="#94a3b8")
    ax2.set_xticks(range(0, 24, 2)); ax2.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig_b64(fig)

def chart_r_dist():
    fig, axes = dark_fig_multi(1, 2, w=14, h=5)
    ax1, ax2 = axes
    r1 = valid["r_tp1"].clip(-2, 4)
    r2 = valid["r_tp2"].clip(-2, 5)
    bins1 = np.linspace(-2, 4, 30)
    bins2 = np.linspace(-2, 5, 30)
    ax1.hist(r1[valid["win_tp1"]],  bins=bins1, color="#22c55e", alpha=.7, label="TP1 獲利")
    ax1.hist(r1[~valid["win_tp1"]], bins=bins1, color="#ef4444", alpha=.7, label="TP1 虧損")
    ax1.axvline(0, color="#475569", lw=1); ax1.axvline(1.0, color="#22c55e", ls="--", lw=1, label="TP1")
    ax1.set_title("TP1 R 倍數分佈", color="#e2e8f0"); ax1.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    ax2.hist(r2[valid["win_tp2"]],  bins=bins2, color="#22c55e", alpha=.7, label="TP2 獲利")
    ax2.hist(r2[~valid["win_tp2"]], bins=bins2, color="#ef4444", alpha=.7, label="TP2 虧損")
    ax2.axvline(0, color="#475569", lw=1); ax2.axvline(3.5, color="#22c55e", ls="--", lw=1, label="TP2")
    ax2.set_title("TP2 R 倍數分佈", color="#e2e8f0"); ax2.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig_b64(fig)

def chart_exit_type():
    """結果分類圓餅圖"""
    fig, axes = dark_fig_multi(1, 2, w=12, h=5)
    ax1, ax2 = axes
    def pie_from(series, ax, title):
        cnts = series.value_counts()
        colors_map = {"win_tp1":"#22c55e","win_tp2":"#16a34a","loss_sl":"#ef4444",
                      "loss_time":"#f59e0b","no_data":"#64748b"}
        colors = [colors_map.get(k,"#94a3b8") for k in cnts.index]
        wedges, texts, autotexts = ax.pie(
            cnts.values, labels=cnts.index, colors=colors,
            autopct="%1.0f%%", startangle=140,
            textprops={"color":"#e2e8f0","fontsize":9},
        )
        for at in autotexts: at.set_color("#0f172a")
        ax.set_title(title, color="#e2e8f0")
        ax.set_facecolor("#0f172a")
    pie_from(valid["result_tp1"], ax1, "TP1 結果分類")
    pie_from(valid["result_tp2"], ax2, "TP2 結果分類")
    fig.patch.set_facecolor("#0f172a")
    fig.tight_layout()
    return fig_b64(fig)

def chart_hold_time():
    fig, ax = dark_fig(12, 5)
    ax.hist(valid.loc[valid["win_tp1"], "exit_bars"], bins=20, color="#22c55e",
            alpha=.7, label=f"TP1 獲利 (n={valid['win_tp1'].sum():.0f})")
    ax.hist(valid.loc[~valid["win_tp1"], "exit_bars"], bins=20, color="#ef4444",
            alpha=.7, label=f"TP1 虧損 (n={(~valid['win_tp1']).sum():.0f})")
    ax.axvline(TIME_LIMIT, color="#f59e0b", ls="--", lw=1.5, label=f"時間止損 ({TIME_LIMIT} bars)")
    ax.set_title("空單持倉時間分佈（30m bars）", color="#e2e8f0")
    ax.set_xlabel("持倉根數", color="#94a3b8"); ax.set_ylabel("筆數", color="#94a3b8")
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig_b64(fig)

def chart_fail_type_tp2():
    """失敗類型 × TP2 結果的 stacked bar"""
    fig, ax = dark_fig(12, 5)
    fail_types = ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]
    result_cats = ["win_tp2", "loss_sl", "loss_time"]
    colors_map = {"win_tp2":"#22c55e", "loss_sl":"#ef4444", "loss_time":"#f59e0b"}
    bottom = np.zeros(len(fail_types))
    for rc in result_cats:
        vals = [len(valid[(valid["fail_type"] == ft) & (valid["result_tp2"] == rc)]) /
                max(len(valid[valid["fail_type"] == ft]), 1) * 100
                for ft in fail_types]
        ax.bar(fail_types, vals, bottom=bottom, color=colors_map[rc], alpha=.85, label=rc)
        bottom += np.array(vals)
    ax.set_title("各 S1 失敗類型 → 空單 TP2 結果組成", color="#e2e8f0")
    ax.set_ylabel("%", color="#94a3b8")
    ax.set_xticklabels(fail_types, color="#94a3b8")
    ax.legend(facecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig_b64(fig)

print("生成圖表...")
imgs = {}
imgs["overview"]    = chart_overview()
imgs["session"]     = chart_session()
imgs["r_dist"]      = chart_r_dist()
imgs["exit_type"]   = chart_exit_type()
imgs["hold"]        = chart_hold_time()
imgs["fail_tp2"]    = chart_fail_type_tp2()
print("  圖表生成完成")

# ── 詳細統計表 HTML ───────────────────────────────────────────────────────────
def detail_table_by(group_col, group_vals, label):
    rows = ""
    for gv in group_vals:
        df = valid[valid[group_col] == gv] if gv != "全部" else valid
        if len(df) == 0: continue
        wr1 = df["win_tp1"].mean()
        wr2 = df["win_tp2"].mean()
        ar1 = df["r_tp1"].mean()
        ar2 = df["r_tp2"].mean()
        n   = len(df)
        col1 = "#22c55e" if wr1 >= 0.5 else "#f59e0b" if wr1 >= 0.4 else "#ef4444"
        col2 = "#22c55e" if wr2 >= 0.35 else "#f59e0b" if wr2 >= 0.25 else "#ef4444"
        rows += f"""<tr>
          <td><strong>{gv}</strong></td>
          <td>{n}</td>
          <td style="color:{col1}">{wr1*100:.1f}%</td>
          <td style="color:{col2}">{wr2*100:.1f}%</td>
          <td style="color:{'#22c55e' if ar1>=0 else '#ef4444'}">{ar1:.3f}R</td>
          <td style="color:{'#22c55e' if ar2>=0 else '#ef4444'}">{ar2:.3f}R</td>
        </tr>"""
    return f"""<p style="color:var(--muted);font-size:.85em;margin:12px 0 4px">{label}</p>
<table class="tbl"><thead><tr><th>分組</th><th>n</th><th>TP1 勝率</th><th>TP2 勝率</th><th>avg R (TP1)</th><th>avg R (TP2)</th></tr></thead>
<tbody>{rows}</tbody></table>"""

def full_detail_table():
    rows_html = detail_table_by(
        "fail_type",
        ["全部", "immediate_loss", "false_breakout", "time_bleed", "normal_sl"],
        "依 S1 失敗類型"
    )
    rows_html += detail_table_by(
        "session", ["全部", "asia", "europe", "us"],
        "依交易時段"
    )
    top_hours = valid.groupby("entry_hour")["win_tp1"].mean().sort_values(ascending=False).head(8).index.tolist()
    rows_html += detail_table_by(
        "entry_hour", ["全部"] + top_hours,
        "依進場小時（顯示 TP1 勝率最高 8 小時）"
    )
    return rows_html

# ── 關鍵洞察 ─────────────────────────────────────────────────────────────────
best_fail = valid.groupby("fail_type")["win_tp2"].mean().idxmax() if len(valid) > 0 else "N/A"
worst_fail = valid.groupby("fail_type")["win_tp2"].mean().idxmin() if len(valid) > 0 else "N/A"
best_sess  = valid.groupby("session")["win_tp1"].mean().idxmax() if len(valid) > 0 else "N/A"

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}
:root{--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#38bdf8;--muted:#94a3b8;--border:#334155}
.wrap{max-width:1200px;margin:0 auto}
h1{font-size:1.6em;margin-bottom:4px;color:#f8fafc}
h2{font-size:1.1em;color:#38bdf8;margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--border)}
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
.note{font-size:.8em;color:var(--muted);margin-top:8px}
.insight{background:rgba(56,189,248,.05);border:1px solid rgba(56,189,248,.25);border-radius:8px;padding:16px;margin:12px 0}
.insight h3{color:#38bdf8;font-size:.9em;margin-bottom:8px}
.insight p{font-size:.85em;color:#cbd5e1;line-height:1.6}
</style>"""

def img(key):
    b = imgs.get(key)
    if not b: return "<p style='color:#64748b'>（無數據）</p>"
    return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'

wr1_c = "#22c55e" if overall["wr1"] >= 0.5 else "#f59e0b" if overall["wr1"] >= 0.4 else "#ef4444"
wr2_c = "#22c55e" if overall["wr2"] >= 0.35 else "#f59e0b" if overall["wr2"] >= 0.25 else "#ef4444"

HTML = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<title>S1-Fail 反向做空測試報告</title>{CSS}</head>
<body><div class="wrap">
<h1>S1-AweWithBB — 虧損進場點反向做空測試</h1>
<p class="note">
  基礎：S1 V3.4 虧損交易（共 {len(losses)} 筆）· 生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
  30m 價格範圍 {price['time'].min().date()} ~ {price['time'].max().date()}
</p>
<p class="note">
  空單規則：SL = entry×1.005 &nbsp;|&nbsp; TP1 = entry×0.995（1R）&nbsp;|&nbsp; TP2 = entry×0.9825（3.5R）&nbsp;|&nbsp; 時間止損 = {TIME_LIMIT} bars（≈18h）
</p>

<div class="card">
<h2>§0 Executive Summary</h2>
<div class="grid4">
  <div class="metric"><div class="lbl">模擬空單總筆數</div><div class="val blue">{len(valid)}</div><div class="sub">基於 S1 虧損交易</div></div>
  <div class="metric"><div class="lbl">TP1 勝率（1R）</div><div class="val" style="color:{wr1_c}">{overall['wr1']*100:.1f}%</div><div class="sub">{overall['wins1']:.0f}/{len(valid)} 筆</div></div>
  <div class="metric"><div class="lbl">TP2 勝率（3.5R）</div><div class="val" style="color:{wr2_c}">{overall['wr2']*100:.1f}%</div><div class="sub">{overall['wins2']:.0f}/{len(valid)} 筆</div></div>
  <div class="metric"><div class="lbl">TP1 avg R</div><div class="val" style="color:{'#22c55e' if overall['avg_r1']>0 else '#ef4444'}">{overall['avg_r1']:.3f}R</div><div class="sub">TP2 avg R: {overall['avg_r2']:.3f}R</div></div>
</div>

<div class="insight">
  <h3>核心洞察</h3>
  <p>
    S1 虧損後反向做空的 TP1 勝率為 <strong>{overall['wr1']*100:.1f}%</strong>，
    {'顯示反向信號具有可操作性。' if overall['wr1'] >= 0.5 else '尚不足以直接做空 S1 進場失敗點，需要額外過濾條件。'}
    &nbsp;失敗類型中，<strong>{best_fail}</strong> 的 TP2 勝率最高——
    {'這類型進場後價格往往快速反轉，做空確實有效。' if best_fail == 'immediate_loss' else '此類型空單表現最佳，可重點觀察。'}
    &nbsp;時段方面，<strong>{best_sess}</strong> 的 TP1 勝率最高。
  </p>
</div>
</div>

<div class="card">
<h2>§1 各失敗類型 vs 時段 空單勝率</h2>
{img("overview")}
</div>

<div class="card">
<h2>§2 時段 × 小時 空單勝率</h2>
{img("session")}
</div>

<div class="card">
<h2>§3 R 倍數分佈</h2>
{img("r_dist")}
</div>

<div class="card">
<h2>§4 結果分類（TP1 / TP2）</h2>
{img("exit_type")}
<p class="note">win_tp1/tp2 = 達到目標止盈 · loss_sl = 被止損 · loss_time = 時間止損（{TIME_LIMIT} bars 未達目標）</p>
</div>

<div class="card">
<h2>§5 持倉時間分佈</h2>
{img("hold")}
</div>

<div class="card">
<h2>§6 失敗類型 × TP2 結果 Stacked Bar</h2>
{img("fail_tp2")}
<p class="note">此圖顯示哪種 S1 失敗模式最適合做空：win_tp2 比例越高，代表此類型進場後反轉幅度越大。</p>
</div>

<div class="card">
<h2>§7 詳細統計表</h2>
{full_detail_table()}
</div>

<div class="card">
<h2>§8 方法論說明</h2>
<p style="font-size:.85em;color:#cbd5e1;line-height:1.8">
  <strong>資料基礎</strong>：取 S1-AweWithBB V3.4 所有 {len(losses)} 筆虧損交易的進場時間和進場價格。<br>
  <strong>空單進場</strong>：與 S1 相同的進場時間和進場價（不另加滑點）。<br>
  <strong>走帳邏輯</strong>：每根 30m K 棒逐一檢查 low≤TP 或 high≥SL。同一根 K 棒內若 low≤TP2 則直接視為 TP2 達成。<br>
  <strong>未達數據範圍</strong>：若進場時間在 30m 資料範圍之外，標記為 no_data 並排除統計（共排除 {total - len(valid)} 筆）。<br>
  <strong>限制</strong>：實際交易中做空需借券、有點差成本；時間止損按市價，此處以 exit bar 的 close 估計。<br>
  <strong>用途</strong>：本測試用於了解 S1 信號反轉特性，<em>不代表可直接執行的空單策略</em>。
</p>
</div>

</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ S1-Fail 空單測試報告已生成：{OUT_HTML}")
