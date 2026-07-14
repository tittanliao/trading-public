"""
run_s2_hammer_v41_oos.py — S2-Hammer V4.1（Footprint D 模式）vs V3.2 差異分析 + 雙重 OOS
==============================================================================
輸出：xauusd/XAUUSD-Long-S2-Hammer/report_v41_oos.html

背景（20260712）：V4.1 預設 = V3.2 全部設定 + Footprint FILTER③ D 模式
（POC上半部 + 低檔買方吸收堆疊）。使用者 TV 實測全樣本 PF 2.05→2.2。
本報告回答三個問題：
  1. D 模式篩掉了哪些交易？它們原本是贏是輸、發生在什麼時期？
     （被篩交易的時間分佈 = footprint 歷史資料覆蓋率的直接證據）
  2. 雙重 OOS：V4.1 自己的 70/30 切分 + V1.9-OOS 相同日曆區間對照
  3. PF 提升是全期均勻，還是集中在近期（最需要救的那段）？

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s2_hammer_v41_oos.py
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

FOLDER   = ROOT / "xauusd/XAUUSD-Long-S2-Hammer"
V19_CSV  = FOLDER / "S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-07-11.csv"
V32_CSV  = FOLDER / "S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv"
V41_CSV  = FOLDER / "S2-Hammer-V4.1_FX_IDC_XAUUSD_2026-07-12.csv"
OUT_HTML = FOLDER / "report_v41_oos.html"

SL_PCT = 0.010
PF_THRESHOLD = 1.2
WR_DIFF_THRESHOLD = 8.0
V19_OOS_CUT = pd.Timestamp("2026-01-22")

print("載入 V1.9 / V3.2 / V4.1 真實逐筆交易...")

def load(csv_path: Path, label: str) -> pd.DataFrame:
    t = loader.load_trades(csv_path)
    t = t[t["exit_signal"] != "Open"].reset_index(drop=True)
    t["session"] = fail_patterns.tag_session(t["entry_time"])
    t["win"] = t["result"] == "win"
    t["r_multiple"] = t["net_pnl_usd"] / (t["entry_price"] * t["size_qty"] * SL_PCT)
    t["version"] = label
    return t.sort_values("entry_time").reset_index(drop=True)

v19 = load(V19_CSV, "V1.9")
v32 = load(V32_CSV, "V3.2")
v41 = load(V41_CSV, "V4.1")
print(f"  V1.9：{len(v19)} 筆 | V3.2：{len(v32)} 筆 | V4.1：{len(v41)} 筆")

def stat_block(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net=0, avg_r=np.nan, mdd=np.nan)
    wins = t.loc[t.win, "net_pnl_usd"].sum()
    loss = abs(t.loc[~t.win, "net_pnl_usd"].sum())
    return dict(n=len(t), wr=t.win.mean() * 100,
                pf=(wins / loss) if loss else float("inf"),
                net=t.net_pnl_usd.sum(), avg_r=t.r_multiple.mean(),
                mdd=abs(metrics.max_drawdown(t)))

s19, s32, s41 = stat_block(v19), stat_block(v32), stat_block(v41)
print(f"  全樣本：V1.9 WR{s19['wr']:.1f}%/PF{s19['pf']:.2f} | V3.2 WR{s32['wr']:.1f}%/PF{s32['pf']:.2f} | "
      f"V4.1 WR{s41['wr']:.1f}%/PF{s41['pf']:.2f}")

# ── 1. D 模式篩掉了哪些交易？（V3.2 有、V4.1 沒有的進場時間）─────────────────
print("\n1. D 模式篩選差異分析...")
# 進場時間容差配對：V4.1 的每筆找 V3.2 中 ±30 分鐘內同進場時間的交易
v41_times = set(v41.entry_time)
def in_v41(et):
    if et in v41_times:
        return True
    for dt in (pd.Timedelta(minutes=30), -pd.Timedelta(minutes=30)):
        if (et + dt) in v41_times:
            return True
    return False

v32["kept_in_v41"] = v32.entry_time.apply(in_v41)
removed = v32[~v32.kept_in_v41]
kept = v32[v32.kept_in_v41]
n_removed = len(removed)
removed_wins = int(removed.win.sum())
removed_losses = n_removed - removed_wins
removed_net = removed.net_pnl_usd.sum()
print(f"  被 D 篩掉：{n_removed} 筆（贏{removed_wins}/輸{removed_losses}），淨損益合計 ${removed_net:+,.0f}")

# 篩掉交易的時間分佈（覆蓋率證據）
removed_by_half = removed.groupby(removed.entry_time.dt.to_period("Q")).agg(
    n=("trade_id", "count"), net=("net_pnl_usd", "sum"))
print("  被篩交易的季度分佈：")
for period, row in removed_by_half.iterrows():
    print(f"    {period}: {int(row.n)} 筆, ${row.net:+,.0f}")

first_removed = removed.entry_time.min() if n_removed else None
last_removed = removed.entry_time.max() if n_removed else None

# V4.1 有而 V3.2 沒有的（理論上不該有，除非TV引擎差異/持倉互斥連鎖）
v32_times = set(v32.entry_time)
def in_v32(et):
    if et in v32_times:
        return True
    for dt in (pd.Timedelta(minutes=30), -pd.Timedelta(minutes=30)):
        if (et + dt) in v32_times:
            return True
    return False
v41["was_in_v32"] = v41.entry_time.apply(in_v32)
new_in_v41 = v41[~v41.was_in_v32]
print(f"  V4.1 新增（V3.2 沒有）：{len(new_in_v41)} 筆（過濾造成的持倉時序連鎖，正常現象）")

# ── 2. 雙重 OOS ────────────────────────────────────────────────────────────────
print("\n2. 雙重 OOS 驗證...")
def split_7030(t: pd.DataFrame):
    cut = t.entry_time.iloc[int(len(t) * 0.7)]
    return t[t.entry_time < cut], t[t.entry_time >= cut], cut

results = {}
for name, t in [("V1.9", v19), ("V3.2", v32), ("V4.1", v41)]:
    is_, oos, cut = split_7030(t)
    si, so = stat_block(is_), stat_block(oos)
    pf_ok = so["pf"] >= PF_THRESHOLD
    wr_ok = (si["wr"] - so["wr"]) <= WR_DIFF_THRESHOLD
    results[name] = dict(is_=si, oos=so, cut=cut, pf_ok=pf_ok, wr_ok=wr_ok,
                          full=stat_block(t))
    print(f"  {name} 切分點{cut.date()}：IS(n={si['n']}) WR{si['wr']:.1f}%/PF{si['pf']:.2f} | "
          f"OOS(n={so['n']}) WR{so['wr']:.1f}%/PF{so['pf']:.2f} | "
          f"PF門檻{'✅' if pf_ok else '❌'} 勝率落差{si['wr']-so['wr']:+.1f}pp{'✅' if wr_ok else '❌'}")

# 相同日曆區間（V1.9-OOS 起點之後）
window = {}
for name, t in [("V1.9", v19), ("V3.2", v32), ("V4.1", v41)]:
    seg = t[t.entry_time >= V19_OOS_CUT]
    window[name] = stat_block(seg)
    w = window[name]
    print(f"  {name} 在 {V19_OOS_CUT.date()} 之後：n={w['n']} WR{w['wr']:.1f}%/PF{w['pf']:.2f} 淨利${w['net']:+,.0f}")

# ── 3. 分期對照：改善集中在哪？────────────────────────────────────────────────
print("\n3. 分期對照（V3.2 vs V4.1，PF 提升來自哪段時期）...")
periods = [("2024", "2024-01-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"),
           ("2026H1+", "2026-01-01", "2026-12-31")]
period_rows = []
for label, a, b in periods:
    seg32 = v32[(v32.entry_time >= a) & (v32.entry_time < b)]
    seg41 = v41[(v41.entry_time >= a) & (v41.entry_time < b)]
    p32, p41 = stat_block(seg32), stat_block(seg41)
    period_rows.append((label, p32, p41))
    print(f"  {label}: V3.2 n={p32['n']} WR{p32['wr']:.1f}%/PF{p32['pf']:.2f} → "
          f"V4.1 n={p41['n']} WR{p41['wr']:.1f}%/PF{p41['pf']:.2f}")

# ── 圖表 ──────────────────────────────────────────────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def dark_fig(w=11, h=5, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    fig.patch.set_facecolor("#0f172a")
    for ax in (axes if n > 1 else [axes]):
        ax.set_facecolor("#1e293b"); ax.tick_params(colors="#94a3b8"); ax.title.set_color("#e2e8f0")
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    return fig, axes

def chart_oos_3ver():
    fig, axes = dark_fig(13, 4.5, n=2)
    names = ["V1.9", "V3.2", "V4.1"]
    for ax, metric, fmtv, ref in [(axes[0], "wr", "{:.1f}", 50), (axes[1], "pf", "{:.2f}", PF_THRESHOLD)]:
        x = np.arange(3); w = 0.35
        is_v = [results[n]["is_"][metric] for n in names]
        oos_v = [results[n]["oos"][metric] for n in names]
        ax.bar(x - w/2, is_v, w, label="IS 前70%", color="#64748b")
        ax.bar(x + w/2, oos_v, w, label="OOS 後30%", color="#22c55e")
        ax.axhline(ref, color="#475569", ls="--")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_title("勝率 %" if metric == "wr" else "獲利因子", fontsize=12)
        ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
        for i, (a, b) in enumerate(zip(is_v, oos_v)):
            ax.text(i - w/2, a * 1.02, fmtv.format(a), ha="center", color="#e2e8f0", fontsize=9)
            ax.text(i + w/2, b * 1.02, fmtv.format(b), ha="center", color="#e2e8f0", fontsize=9)
    fig.tight_layout(); return fig_b64(fig)

def chart_window():
    fig, ax = dark_fig(9, 4.8)
    names = ["V1.9", "V3.2", "V4.1"]
    pfs = [window[n]["pf"] for n in names]
    colors = ["#ef4444", "#f59e0b", "#22c55e"]
    bars = ax.bar(names, pfs, color=colors, width=0.5)
    ax.axhline(PF_THRESHOLD, color="#475569", ls="--")
    ax.axhline(1.0, color="#94a3b8", ls=":")
    ax.set_title(f"相同日曆區間（{V19_OOS_CUT.date()} 之後）PF 三版對照", fontsize=12)
    for bar, v, n in zip(bars, pfs, names):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}\nn={window[n]['n']}", ha="center",
                color="#e2e8f0", fontsize=10, fontweight="bold")
    fig.tight_layout(); return fig_b64(fig)

def chart_removed_timeline():
    fig, ax = dark_fig(11, 4.2)
    if len(removed):
        colors = ["#22c55e" if w else "#ef4444" for w in removed.win]
        ax.scatter(removed.entry_time, removed.net_pnl_usd, c=colors, s=60, zorder=3)
        ax.axhline(0, color="#94a3b8", lw=1)
    ax.set_title("被 Footprint D 模式篩掉的交易（綠=原本贏、紅=原本輸）——時間分佈即覆蓋率證據", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout(); return fig_b64(fig)

print("\n生成圖表...")
imgs = dict(oos=chart_oos_3ver(), window=chart_window(), removed=chart_removed_timeline())

# ── HTML ──────────────────────────────────────────────────────────────────────
def fmt(v, f="{:.2f}"): return "—" if (isinstance(v, float) and not np.isfinite(v)) else f.format(v)
def img(b): return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'

def full_table():
    body = ""
    for name, s in [("V1.9（原始）", s19), ("V3.2（HTF+互斥）", s32), ("V4.1（+Footprint D）", s41)]:
        body += (f"<tr><td><strong>{name}</strong></td><td>{s['n']}</td><td>{fmt(s['wr'],'{:.1f}')}%</td>"
                 f"<td>{fmt(s['pf'],'{:.3f}')}</td><td>${s['net']:+,.0f}</td><td>${s['mdd']:,.0f}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>版本</th><th>筆數</th><th>勝率</th><th>PF</th>"
            f"<th>淨利</th><th>MDD</th></tr></thead><tbody>{body}</tbody></table>")

def oos_table():
    body = ""
    for name in ["V1.9", "V3.2", "V4.1"]:
        r = results[name]
        ok = r["pf_ok"] and r["wr_ok"]
        color = "#22c55e" if ok else "#ef4444"
        body += (f"<tr><td><strong>{name}</strong></td>"
                 f"<td>{fmt(r['is_']['wr'],'{:.1f}')}%/{fmt(r['is_']['pf'])}（n={r['is_']['n']}）</td>"
                 f"<td>{fmt(r['oos']['wr'],'{:.1f}')}%/{fmt(r['oos']['pf'])}（n={r['oos']['n']}）</td>"
                 f"<td>{r['is_']['wr']-r['oos']['wr']:+.1f}pp</td>"
                 f"<td style='color:{color}'><strong>{'✅ 通過' if ok else '❌ 未通過'}</strong></td></tr>")
    return (f"<table class='tbl'><thead><tr><th>版本</th><th>IS 前70%</th><th>OOS 後30%</th>"
            f"<th>勝率落差</th><th>判定</th></tr></thead><tbody>{body}</tbody></table>")

def window_table():
    body = ""
    for name in ["V1.9", "V3.2", "V4.1"]:
        w = window[name]
        body += (f"<tr><td><strong>{name}</strong></td><td>{w['n']}</td><td>{fmt(w['wr'],'{:.1f}')}%</td>"
                 f"<td>{fmt(w['pf'])}</td><td>${w['net']:+,.0f}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>版本</th><th>筆數</th><th>勝率</th><th>PF</th><th>淨利</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

def period_table():
    body = ""
    for label, p32, p41 in period_rows:
        pf_delta = (p41["pf"] - p32["pf"]) if (np.isfinite(p41["pf"]) and np.isfinite(p32["pf"])) else np.nan
        color = "#22c55e" if np.isfinite(pf_delta) and pf_delta > 0 else "#e2e8f0"
        body += (f"<tr><td><strong>{label}</strong></td>"
                 f"<td>{p32['n']} / {fmt(p32['wr'],'{:.1f}')}% / {fmt(p32['pf'])}</td>"
                 f"<td>{p41['n']} / {fmt(p41['wr'],'{:.1f}')}% / {fmt(p41['pf'])}</td>"
                 f"<td style='color:{color}'>{fmt(pf_delta, '{:+.2f}')}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>時期</th><th>V3.2（筆/WR/PF）</th><th>V4.1（筆/WR/PF）</th>"
            f"<th>PF 變化</th></tr></thead><tbody>{body}</tbody></table>")

def removed_table():
    if not len(removed):
        return "<p class='note'>無被篩掉的交易</p>"
    body = ""
    for _, r in removed.iterrows():
        c = "#22c55e" if r.win else "#ef4444"
        body += (f"<tr><td>{r.entry_time}</td><td style='color:{c}'>{'贏' if r.win else '輸'}</td>"
                 f"<td>${r.net_pnl_usd:+,.0f}</td><td>{r.hold_bars}</td><td>{r.exit_signal}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>進場時間</th><th>原結果</th><th>原淨損益</th>"
            f"<th>持倉bars</th><th>原出場</th></tr></thead><tbody>{body}</tbody></table>")

v41_ok = results["V4.1"]["pf_ok"] and results["V4.1"]["wr_ok"]
w41 = window["V4.1"]
verdict_b_ok = w41["pf"] >= PF_THRESHOLD and w41["wr"] >= 40
overall = ("✅ 雙重驗證通過" if v41_ok and verdict_b_ok else
           "🟡 部分通過" if v41_ok or verdict_b_ok else "❌ 雙重驗證未通過")

CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
:root{--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#38bdf8;--muted:#94a3b8;--border:#334155}
.wrap{max-width:1150px;margin:0 auto}
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
<title>S2-Hammer V4.1（Footprint D）vs V3.2 差異 + OOS</title>{CSS}</head>
<body>
<div style="max-width:1150px;margin:0 auto 14px"><a href="../../xauusd.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 XAUUSD 主頁</a></div>
<div class="wrap">
<h1>S2-Hammer <span style="color:#38bdf8">V4.1（Footprint D）</span> vs V3.2 差異分析 + 雙重 OOS</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
V4.1 = V3.2 + Footprint FILTER③ D 模式（POC上半部 + 低檔買方吸收堆疊）·
姊妹報告：<a href="report_v32_oos.html" style="color:var(--blue)">V3.2 OOS 驗證</a></p>

<div class="{'good' if v41_ok and verdict_b_ok else 'warn' if v41_ok or verdict_b_ok else 'bad'}">
<strong>{overall}</strong>：V4.1 自身 70/30 切分 {'✅通過' if v41_ok else '❌未通過'}
（OOS WR {fmt(results['V4.1']['oos']['wr'],'{:.1f}')}%/PF {fmt(results['V4.1']['oos']['pf'])}，
勝率落差 {results['V4.1']['is_']['wr']-results['V4.1']['oos']['wr']:+.1f}pp）；
V1.9-OOS 相同日曆區間 {'✅達標' if verdict_b_ok else '❌未達標'}
（WR {fmt(w41['wr'],'{:.1f}')}%/PF {fmt(w41['pf'])}，n={w41['n']}）。
</div>

<!-- 全樣本 -->
<div class="card">
<div class="part">PART 1 — 三版全樣本對照</div>
<h2>V1.9 → V3.2 → V4.1 疊加效果</h2>
{full_table()}
</div>

<!-- D 篩掉了什麼 -->
<div class="card">
<div class="part">PART 2 — Footprint D 篩掉了哪些交易（覆蓋率證據）</div>
<h2>被篩掉 {n_removed} 筆：贏 {removed_wins} / 輸 {removed_losses}，淨損益合計 ${removed_net:+,.0f}</h2>
{img(imgs['removed'])}
{removed_table()}
<p class="note">被篩交易的時間範圍：{first_removed} → {last_removed}。
若被篩交易只集中在近期 = footprint 歷史資料覆蓋有限（早期K棒 na→pass，過濾器實際只作用於近期）；
若分佈全期 = footprint 歷史資料完整。V4.1 另有 {len(new_in_v41)} 筆 V3.2 沒有的新交易
（過濾改變持倉時序的連鎖效應，正常現象）。</p>
</div>

<!-- OOS -->
<div class="card">
<div class="part">PART 3 — 雙重 OOS 驗證</div>
<h2>驗證A：各自 70/30 時間切分</h2>
{oos_table()}
{img(imgs['oos'])}
<h2 style="margin-top:18px">驗證B：相同日曆區間（{V19_OOS_CUT.date()} 之後 = V1.9 原本 OOS 失敗的那段）</h2>
{window_table()}
{img(imgs['window'])}
</div>

<!-- 分期 -->
<div class="card">
<div class="part">PART 4 — 分期對照：PF 提升來自哪段時期？</div>
{period_table()}
<p class="note">若 PF 提升集中在 2026H1+（footprint 資料可用的時期），改善直接作用於最需要救的近期樣本，
含金量高於全期均勻提升。</p>
</div>

<p class="note">
局限：①V4.1 與 V3.2 的交易配對用進場時間 ±30 分鐘容差，過濾造成的持倉時序連鎖（某筆被擋→
空手→接到下一個原本持倉中錯過的訊號）會產生少量「新增」交易，屬正常現象非引擎誤差；
②70/30 為單一切點非 walk-forward；③樣本經三層過濾（HTF+互斥+FootprintD）後 n 變小，
分期子樣本的統計顯著性有限。
</p>
<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S2-Hammer V4.1 OOS 驗證 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 報告已生成：{OUT_HTML}")
