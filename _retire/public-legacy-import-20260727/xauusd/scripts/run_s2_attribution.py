"""
run_s2_attribution.py — S2-RSI / S2-Hammer 真實逐筆歸因 + OOS + 濾網重疊檢驗 + time_bleed 解剖
==============================================================================
輸出：xauusd/XAUUSD-Long-S2-Hammer/report_s2_attribution.html

S2 優化路線圖（20260711 決定）的 Step 0-2，一次完成：
  Step 0  真實逐筆歸因 + OOS 70/30（S1 V3.7 的同套方法論，S2 首次套用）
  Step 1  三濾網重疊檢驗：Z-Score（V2.4/V2.3 已實作未開）vs「S2-RSI 互斥」（合流分析發現）
          —— 用 4H EMA20/ATR14 算每筆進場當下的 Z-Score，檢驗兩者是否同一件事
  Step 2  time_bleed 解剖：拖 ≥12h 的虧損單 MFE 分佈 → scratch 規則能救回多少

資料：
  - S2-RSI  V2.0 基準 161 筆 / S2-Hammer V1.9 基準 200 筆（TradingView List of Trades，
    2024-01 ~ 2026-04-26 匯出）
  - 4H OHLC：csv/FX_IDC_XAUUSD, 240.csv（2024-05-02 起）——早於此的交易無法算 Z-Score，
    誠實標註覆蓋率，不硬湊

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s2_attribution.py
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

RSI_CSV    = ROOT / "xauusd/XAUUSD-Long-S2-RSI/S2-Hybrid-V2.0_FX_IDC_XAUUSD_2026-04-26.csv"
HAMMER_CSV = ROOT / "xauusd/XAUUSD-Long-S2-Hammer/S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-07-11.csv"  # 20260711 重匯到最新日期，200→225筆
H4_CSV     = ROOT / "xauusd/csv/20260711/FX_IDC_XAUUSD, 240.csv"  # 20260711 重匯到最新日期
OUT_HTML   = ROOT / "xauusd/XAUUSD-Long-S2-Hammer/report_s2_attribution.html"

PF_THRESHOLD = 1.2   # monthly_checklist.md 轉向門檻
WR_DIFF_THRESHOLD = 8.0
SCRATCH_MFE = 0.3    # time_bleed 解剖用：MFE 有無到過 +0.3%

# ── 1. 載入交易 ──────────────────────────────────────────────────────────────
print("載入 S2-RSI / S2-Hammer 真實逐筆交易...")

def load(csv, sl_pct):
    t = loader.load_trades(csv)
    t = t[t["exit_signal"] != "Open"].reset_index(drop=True)  # 排除未平倉尾單
    t["session"] = fail_patterns.tag_session(t["entry_time"])
    t["win"] = t["result"] == "win"
    t["r_multiple"] = t["net_pnl_usd"] / (t["entry_price"] * t["size_qty"] * sl_pct)
    t["exit_kind"] = t["exit_signal"].apply(
        lambda s: "TP2" if "TP2" in s else ("TP1" if "TP1" in s else ("SL" if "SL" in s else "other")))
    return t.sort_values("entry_time").reset_index(drop=True)

# SL：S2-RSI 0.5%、S2-Hammer 1.0%（H2 黃金參數，寬止損應對波動）
rsi = load(RSI_CSV, 0.005)
ham = load(HAMMER_CSV, 0.010)
print(f"  S2-RSI:    {len(rsi)} 筆（{rsi.entry_time.min().date()} → {rsi.entry_time.max().date()}）")
print(f"  S2-Hammer: {len(ham)} 筆（{ham.entry_time.min().date()} → {ham.entry_time.max().date()}）")

STRATS = {"S2-RSI": rsi, "S2-Hammer": ham}


def stat_block(t):
    if len(t) == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net=0, avg_r=np.nan, mdd=np.nan)
    wins = t.loc[t.win, "net_pnl_usd"].sum()
    loss = abs(t.loc[~t.win, "net_pnl_usd"].sum())
    return dict(n=len(t), wr=t.win.mean()*100,
                pf=(wins/loss) if loss else float("inf"),
                net=t.net_pnl_usd.sum(), avg_r=t.r_multiple.mean(),
                mdd=abs(metrics.max_drawdown(t)))

# ── 2. Step 0：歸因 + OOS ────────────────────────────────────────────────────
print("\nStep 0：歸因 + OOS 70/30...")
attribution = {}
for name, t in STRATS.items():
    fails = fail_patterns.classify_fail(t)
    fail_pct = fails["fail_type"].value_counts(normalize=True) * 100
    exit_bd = t.groupby("exit_kind").agg(n=("trade_id","count"), net=("net_pnl_usd","sum"),
                                          avg_r=("r_multiple","mean"))
    cut = t.entry_time.iloc[int(len(t)*0.7)]
    is_seg, oos_seg = t[t.entry_time < cut], t[t.entry_time >= cut]
    attribution[name] = dict(
        full=stat_block(t), is_=stat_block(is_seg), oos=stat_block(oos_seg), cut=cut,
        fail_pct=fail_pct.to_dict(), exit_bd=exit_bd, n_loss=len(fails))
    a = attribution[name]
    pf_ok = a["oos"]["pf"] >= PF_THRESHOLD
    wr_ok = (a["is_"]["wr"] - a["oos"]["wr"]) <= WR_DIFF_THRESHOLD
    a["pf_ok"], a["wr_ok"] = pf_ok, wr_ok
    print(f"  {name}: 全 {a['full']['wr']:.1f}%/{a['full']['pf']:.2f} | "
          f"IS {a['is_']['wr']:.1f}%/{a['is_']['pf']:.2f} (n={a['is_']['n']}) | "
          f"OOS {a['oos']['wr']:.1f}%/{a['oos']['pf']:.2f} (n={a['oos']['n']}) | "
          f"PF門檻{'✅' if pf_ok else '❌'} 勝率落差{'✅' if wr_ok else '❌'} | "
          f"time_bleed {a['fail_pct'].get('time_bleed',0):.0f}%")

# ── 3. Step 1：Z-Score at entry + 濾網重疊檢驗 ────────────────────────────────
print("\nStep 1：Z-Score 計算與濾網重疊檢驗...")
h4 = loader.load_price(H4_CSV)
c4 = h4["close"].to_numpy()
ema20 = pd.Series(c4).ewm(span=20, adjust=False).mean().to_numpy()
hi, lo = h4["high"].to_numpy(), h4["low"].to_numpy()
prev_c = np.r_[c4[0], c4[:-1]]
tr = np.maximum(hi - lo, np.maximum(abs(hi - prev_c), abs(lo - prev_c)))
atr14 = pd.Series(tr).ewm(com=13, adjust=False).mean().to_numpy()
h4z = pd.DataFrame({"time": h4["time"], "z": (c4 - ema20) / atr14})

Z_BUCKETS = [(-np.inf, -2.5, "<-2.5 陷阱區"), (-2.5, -1.5, "-2.5~-1.5"),
             (-1.5, -0.5, "-1.5~-0.5 甜蜜區"), (-0.5, 0.5, "-0.5~0.5"), (0.5, np.inf, ">0.5")]

for name, t in STRATS.items():
    m = pd.merge_asof(t[["entry_time"]].rename(columns={"entry_time":"time"}).sort_values("time"),
                      h4z.sort_values("time"), on="time", direction="backward")
    t["z_entry"] = m["z"].to_numpy()
    cov = t.z_entry.notna().mean()*100
    print(f"  {name} Z-Score 覆蓋率 {cov:.0f}%（4H CSV 自 2024-05-02 起）")

# S2-Hammer：合流 flag + Z 分佈重疊
w = pd.Timedelta(hours=12)
ham["has_rsi_sig"] = ham.entry_time.apply(lambda x: ((rsi.entry_time - x).abs() <= w).any())
zc = ham.loc[ham.has_rsi_sig & ham.z_entry.notna(), "z_entry"]
za = ham.loc[~ham.has_rsi_sig & ham.z_entry.notna(), "z_entry"]
print(f"  S2-Hammer 進場Z中位數：合流組 {zc.median():.2f} vs 單獨組 {za.median():.2f}")

zbucket_rows = {}
for name, t in STRATS.items():
    rows = []
    for lo_, hi_, label in Z_BUCKETS:
        seg = t[(t.z_entry > lo_) & (t.z_entry <= hi_)]
        rows.append(dict(bucket=label, **stat_block(seg)))
    zbucket_rows[name] = rows
    print(f"  {name} Z分桶: " + " | ".join(
        f"{r['bucket']}:{r['wr']:.0f}%({r['n']})" for r in rows if r['n']>0))

# 三種 S2-Hammer 過濾方案對照（同一批可算Z的交易上比較，公平）
ham_z = ham[ham.z_entry.notna()]
filter_compare = {
    "基準（全收，可算Z樣本）": stat_block(ham_z),
    "A. Z-Score 過濾（跳過 Z<-2.5）": stat_block(ham_z[ham_z.z_entry > -2.5]),
    "B. S2-RSI 互斥（跳過合流）": stat_block(ham_z[~ham_z.has_rsi_sig]),
    "C. 兩者皆用": stat_block(ham_z[(ham_z.z_entry > -2.5) & (~ham_z.has_rsi_sig)]),
}
print("\n  S2-Hammer 過濾方案對照:")
for k, v in filter_compare.items():
    print(f"    {k}: n={v['n']} WR={v['wr']:.1f}% PF={v['pf']:.2f}")

# 重疊度：合流組中有多少也會被 Z<-2.5 擋掉
conf_group = ham_z[ham_z.has_rsi_sig]
overlap = (conf_group.z_entry <= -2.5).mean()*100 if len(conf_group) else 0
print(f"  合流組中 Z<-2.5 佔比：{overlap:.0f}%（重疊度）")

# ── 4. Step 2：time_bleed 解剖 ───────────────────────────────────────────────
print("\nStep 2：time_bleed 解剖...")
tb_stats = {}
for name, t in STRATS.items():
    losses = t[~t.win]
    tb = losses[losses.hold_bars >= 24]  # ≥12h 的虧損單
    rescuable = (tb.mfe_pct >= SCRATCH_MFE).mean()*100 if len(tb) else 0
    # 快贏 vs 慢贏
    wins_ = t[t.win]
    tb_stats[name] = dict(
        n_loss=len(losses), n_tb=len(tb), tb_share=len(tb)/len(losses)*100 if len(losses) else 0,
        tb_mfe_med=tb.mfe_pct.median() if len(tb) else np.nan,
        rescuable_pct=rescuable,
        tb_avg_r=tb.r_multiple.mean() if len(tb) else np.nan,
        win_hold_med=wins_.hold_bars.median(), loss_hold_med=losses.hold_bars.median(),
        slow_win_n=(wins_.hold_bars >= 24).sum(), slow_win_share=(wins_.hold_bars >= 24).mean()*100,
    )
    s = tb_stats[name]
    print(f"  {name}: 虧損{s['n_loss']}筆中 {s['n_tb']}筆拖≥12h（{s['tb_share']:.0f}%），"
          f"其中 {s['rescuable_pct']:.0f}% 曾有 MFE≥{SCRATCH_MFE}%（可救回），"
          f"贏單持倉中位 {s['win_hold_med']:.0f} bars vs 虧單 {s['loss_hold_med']:.0f} bars，"
          f"慢贏(≥12h)佔贏單 {s['slow_win_share']:.0f}%")

# ── 5. 圖表 ──────────────────────────────────────────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                                     facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def dark_fig(w=10, h=5, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    fig.patch.set_facecolor("#0f172a")
    for ax in (axes if n > 1 else [axes]):
        ax.set_facecolor("#1e293b"); ax.tick_params(colors="#94a3b8"); ax.title.set_color("#e2e8f0")
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    return fig, axes

def chart_oos():
    fig, axes = dark_fig(13, 4.5, n=2)
    for ax, metric, fmtv, ref in [(axes[0], "wr", "{:.1f}%", 50), (axes[1], "pf", "{:.2f}", PF_THRESHOLD)]:
        x = np.arange(2); wd = 0.35
        is_v  = [attribution[s]["is_"][metric] for s in STRATS]
        oos_v = [attribution[s]["oos"][metric] for s in STRATS]
        ax.bar(x - wd/2, is_v, wd, label="IS 前70%", color="#64748b")
        ax.bar(x + wd/2, oos_v, wd, label="OOS 後30%", color="#22c55e")
        ax.axhline(ref, color="#475569", ls="--")
        ax.set_xticks(x); ax.set_xticklabels(list(STRATS))
        ax.set_title("勝率 %" if metric=="wr" else "獲利因子", fontsize=12)
        ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
        for i,(a,b) in enumerate(zip(is_v, oos_v)):
            ax.text(i-wd/2, a*1.01, fmtv.format(a), ha="center", color="#e2e8f0", fontsize=9)
            ax.text(i+wd/2, b*1.01, fmtv.format(b), ha="center", color="#e2e8f0", fontsize=9)
    fig.tight_layout(); return fig_b64(fig)

def chart_zbuckets():
    fig, axes = dark_fig(13, 4.8, n=2)
    for ax, name in zip(axes, STRATS):
        rows = [r for r in zbucket_rows[name] if r["n"] > 0]
        labels = [r["bucket"] for r in rows]; wrs = [r["wr"] for r in rows]; ns = [r["n"] for r in rows]
        colors = ["#ef4444" if "陷阱" in l else ("#22c55e" if "甜蜜" in l else "#64748b") for l in labels]
        bars = ax.bar(range(len(rows)), wrs, color=colors, width=0.6)
        ax.axhline(50, color="#475569", ls="--")
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_title(f"{name} — 進場 Z-Score 分桶勝率", fontsize=11)
        for bar, v, n_ in zip(bars, wrs, ns):
            ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.0f}%\nn={n_}", ha="center",
                    color="#e2e8f0", fontsize=8)
    fig.tight_layout(); return fig_b64(fig)

def chart_filters():
    fig, ax = dark_fig(11, 4.8)
    labels = list(filter_compare); pfs = [filter_compare[k]["pf"] for k in labels]
    ns = [filter_compare[k]["n"] for k in labels]
    colors = ["#64748b", "#38bdf8", "#f59e0b", "#22c55e"]
    bars = ax.bar(range(len(labels)), pfs, color=colors, width=0.55)
    ax.axhline(PF_THRESHOLD, color="#475569", ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split("（")[0] for l in labels], fontsize=9)
    ax.set_title("S2-Hammer 過濾方案對照（PF，同一批可算Z樣本）", fontsize=12)
    for bar, v, n_ in zip(bars, pfs, ns):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.04, f"{v:.2f}\nn={n_}", ha="center",
                color="#e2e8f0", fontsize=9, fontweight="bold")
    fig.tight_layout(); return fig_b64(fig)

print("\n生成圖表...")
imgs = dict(oos=chart_oos(), zb=chart_zbuckets(), flt=chart_filters())

# ── 6. HTML ──────────────────────────────────────────────────────────────────
def img(b): return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'
def fmt(v, f="{:.2f}"): return "—" if (isinstance(v,float) and not np.isfinite(v)) else f.format(v)

def attr_table():
    body = ""
    for name in STRATS:
        a = attribution[name]
        for seg, lbl in [("full","全樣本"),("is_","IS 前70%"),("oos","OOS 後30%")]:
            s = a[seg]
            hl = " style='background:rgba(56,189,248,.08)'" if seg=="oos" else ""
            body += (f"<tr{hl}><td>{name if seg=='full' else ''}</td><td>{lbl}</td>"
                     f"<td>{s['n']}</td><td>{fmt(s['wr'],'{:.1f}')}%</td><td>{fmt(s['pf'],'{:.3f}')}</td>"
                     f"<td>${s['net']:+,.0f}</td><td>{fmt(s['avg_r'],'{:+.3f}')}R</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>策略</th><th>段</th><th>筆數</th><th>勝率</th>"
            f"<th>PF</th><th>淨盈虧</th><th>平均R</th></tr></thead><tbody>{body}</tbody></table>")

def fail_table():
    order = ["immediate_loss","false_breakout","time_bleed","normal_sl"]
    body = ""
    for name in STRATS:
        fp = attribution[name]["fail_pct"]
        cells = "".join(f"<td>{fp.get(k,0):.0f}%</td>" for k in order)
        body += f"<tr><td><strong>{name}</strong>（{attribution[name]['n_loss']}筆虧損）</td>{cells}</tr>"
    return (f"<table class='tbl'><thead><tr><th>策略</th>" +
            "".join(f"<th>{k}</th>" for k in order) +
            f"</tr></thead><tbody>{body}</tbody></table>")

def filter_table():
    body = ""
    base_pf = filter_compare["基準（全收，可算Z樣本）"]["pf"]
    for k, v in filter_compare.items():
        color = "#22c55e" if np.isfinite(v["pf"]) and v["pf"] > base_pf else "#e2e8f0"
        body += (f"<tr><td>{k}</td><td>{v['n']}</td><td>{fmt(v['wr'],'{:.1f}')}%</td>"
                 f"<td style='color:{color}'><strong>{fmt(v['pf'],'{:.3f}')}</strong></td>"
                 f"<td>{fmt(v['avg_r'],'{:+.3f}')}R</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>方案</th><th>筆數</th><th>勝率</th><th>PF</th>"
            f"<th>平均R</th></tr></thead><tbody>{body}</tbody></table>")

def tb_table():
    body = ""
    for name, s in tb_stats.items():
        body += (f"<tr><td><strong>{name}</strong></td><td>{s['n_tb']}/{s['n_loss']}（{s['tb_share']:.0f}%）</td>"
                 f"<td>{fmt(s['tb_mfe_med'])}%</td><td>{s['rescuable_pct']:.0f}%</td>"
                 f"<td>{s['win_hold_med']:.0f} / {s['loss_hold_med']:.0f} bars</td>"
                 f"<td>{s['slow_win_n']}筆（{s['slow_win_share']:.0f}%）</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>策略</th><th>拖≥12h虧損佔比</th><th>其MFE中位</th>"
            f"<th>曾有MFE≥{SCRATCH_MFE}%比例</th><th>贏/虧持倉中位</th><th>慢贏(≥12h)</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

rsi_verdict_ok = attribution["S2-RSI"]["pf_ok"] and attribution["S2-RSI"]["wr_ok"]
ham_verdict_ok = attribution["S2-Hammer"]["pf_ok"] and attribution["S2-Hammer"]["wr_ok"]

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
<title>S2 真實逐筆歸因 + OOS + 濾網重疊 + time_bleed 解剖</title>{CSS}</head>
<body>
<div style="max-width:1100px;margin:0 auto 14px"><a href="../../xauusd.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 XAUUSD 主頁</a></div>
<div class="wrap">
<h1>S2-RSI / S2-Hammer <span style="color:#f59e0b">真實逐筆歸因</span>（優化路線 Step 0-2）</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
S2-RSI V2.0 基準 {len(rsi)} 筆 / S2-Hammer V1.9 基準 {len(ham)} 筆（2024-01 ~ 2026-04）·
姊妹報告：<a href="report_s2_confluence.html" style="color:var(--blue)">合流分析</a></p>

<!-- Step 0 -->
<div class="card">
<div class="part">STEP 0 — 歸因 + Out-of-Sample</div>
<h2>S2 的 edge 在樣本外還在嗎？</h2>
{attr_table()}
{img(imgs['oos'])}
<div class="{'good' if rsi_verdict_ok else 'bad'}"><strong>S2-RSI：{'✅ 通過' if rsi_verdict_ok else '❌ 未通過'}</strong>
OOS PF {fmt(attribution['S2-RSI']['oos']['pf'],'{:.3f}')}（門檻 {PF_THRESHOLD}），
勝率落差 {attribution['S2-RSI']['is_']['wr']-attribution['S2-RSI']['oos']['wr']:+.1f}pp（門檻 {WR_DIFF_THRESHOLD}pp）。</div>
<div class="{'good' if ham_verdict_ok else 'bad'}"><strong>S2-Hammer：{'✅ 通過' if ham_verdict_ok else '❌ 未通過'}</strong>
OOS PF {fmt(attribution['S2-Hammer']['oos']['pf'],'{:.3f}')}，
勝率落差 {attribution['S2-Hammer']['is_']['wr']-attribution['S2-Hammer']['oos']['wr']:+.1f}pp。</div>
<h2 style="margin-top:18px">失敗模式分佈（佔虧損單比例）</h2>
{fail_table()}
</div>

<!-- Step 1 -->
<div class="card">
<div class="part">STEP 1 — 濾網重疊檢驗</div>
<h2>Z-Score 過濾 vs S2-RSI 互斥：同一件事嗎？</h2>
{img(imgs['zb'])}
<p class="note">Z-Score = (4H close − 4H EMA20) / 4H ATR14，進場當下取值。
4H CSV 自 2024-05-02 起，早於此的交易不計（覆蓋率 S2-RSI {rsi.z_entry.notna().mean()*100:.0f}% /
S2-Hammer {ham.z_entry.notna().mean()*100:.0f}%）。</p>
<div class="warn"><strong>重疊度：S2-Hammer 合流組（與S2-RSI同時觸發）進場 Z 中位數 {zc.median():.2f}，
單獨組 {za.median():.2f}；合流組中 Z&lt;-2.5 佔 {overlap:.0f}%。</strong></div>
<h2 style="margin-top:18px">S2-Hammer 三種過濾方案正面對決</h2>
{filter_table()}
{img(imgs['flt'])}
</div>

<!-- Step 2 -->
<div class="card">
<div class="part">STEP 2 — time_bleed 解剖</div>
<h2>拖 ≥12h 的虧損單，能救回多少？</h2>
{tb_table()}
<p class="note">「曾有 MFE≥{SCRATCH_MFE}%」= 這些拖死的虧損單其實曾有浮盈到 +{SCRATCH_MFE}% 以上——
理論上「N 小時內達 +{SCRATCH_MFE}% 移保本 / 未達則 scratch」規則可以攔截。
但注意右欄：<strong>慢贏（≥12h 才獲利）的比例</strong>——若慢贏佔比高，激進的時間出場會同時殺死這些贏單，
淨效果需要 bar-level 回測驗證，本表僅為方向估計。</p>
</div>

<p class="note">
局限：①單一 70/30 切點，非 walk-forward；②Z-Score 覆蓋率受 4H CSV 起始日限制；
③time_bleed 救回估計基於整筆 MFE，非時間解析的逐 bar 模擬，精確效果需 TradingView 驗證；
④三方案對照在「可算 Z 的同一批樣本」上比較（公平），與全樣本數字略有差異。
</p>
<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S2 真實逐筆歸因 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 報告已生成：{OUT_HTML}")
