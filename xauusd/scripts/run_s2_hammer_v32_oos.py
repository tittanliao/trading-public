"""
run_s2_hammer_v32_oos.py — S2-Hammer V3.2 是否解決 V1.9 的 OOS 失敗？
==============================================================================
輸出：xauusd/XAUUSD-Long-S2-Hammer/report_v32_oos.html

背景（20260711）：V1.9 原始邏輯（無過濾器）70/30 時間切分 OOS 檢驗未通過
（後30% WR 27.9%/PF 0.92，低於損益兩平）。使用者提供 V3.2（HTF 1H RSI bearish
阻擋 + S2-RSI互斥 30m RSI上穿20，TP1/TP2=2R/4R）的 List of Trades 真實成交CSV，
本報告直接回答：過濾器是否真的解決了 OOS 問題？

雙重驗證方法：
  A. V3.2 自己的 70/30 時間切分（跟 V1.9 用同一套方法論，兩者獨立可比）
  B. V3.2 在「V1.9 OOS 那段相同日曆區間」的表現（直接對照：同一段爛時期，
     V3.2 是否真的表現更好，而不是被自己的切分點稀釋掉）

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s2_hammer_v32_oos.py
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

FOLDER    = ROOT / "xauusd/XAUUSD-Long-S2-Hammer"
V19_CSV   = FOLDER / "S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-07-11.csv"
V32_CSV   = FOLDER / "S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv"
OUT_HTML  = FOLDER / "report_v32_oos.html"

SL_PCT_V19 = 0.010  # V1.9 SL 1.0%
SL_PCT_V32 = 0.010  # V3.2 SL 1.0%（TP1/TP2 改 2R/4R，SL不變）
PF_THRESHOLD = 1.2
WR_DIFF_THRESHOLD = 8.0

print("載入 V1.9 / V3.2 真實逐筆交易...")

def load(csv_path: Path, sl_pct: float, label: str) -> pd.DataFrame:
    t = loader.load_trades(csv_path)
    t = t[t["exit_signal"] != "Open"].reset_index(drop=True)
    t["session"] = fail_patterns.tag_session(t["entry_time"])
    t["win"] = t["result"] == "win"
    t["r_multiple"] = t["net_pnl_usd"] / (t["entry_price"] * t["size_qty"] * sl_pct)
    t["exit_kind"] = t["exit_signal"].apply(
        lambda s: "TP2" if "TP2" in s else ("TP1" if "TP1" in s else ("SL" if "SL" in s else "other")))
    t["version"] = label
    return t.sort_values("entry_time").reset_index(drop=True)

v19 = load(V19_CSV, SL_PCT_V19, "V1.9")
v32 = load(V32_CSV, SL_PCT_V32, "V3.2")
print(f"  V1.9：{len(v19)} 筆（{v19.entry_time.min().date()} → {v19.entry_time.max().date()}）")
print(f"  V3.2：{len(v32)} 筆（{v32.entry_time.min().date()} → {v32.entry_time.max().date()}）")

def stat_block(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net=0, avg_r=np.nan, mdd=np.nan)
    wins = t.loc[t.win, "net_pnl_usd"].sum()
    loss = abs(t.loc[~t.win, "net_pnl_usd"].sum())
    return dict(n=len(t), wr=t.win.mean() * 100,
                pf=(wins / loss) if loss else float("inf"),
                net=t.net_pnl_usd.sum(), avg_r=t.r_multiple.mean(),
                mdd=abs(metrics.max_drawdown(t)))

# ── A. 各自 70/30 時間切分（跟 V1.9 既有方法論一致）──────────────────────────
print("\nA. 各自 70/30 時間切分 OOS...")
def split_7030(t: pd.DataFrame):
    cut = t.entry_time.iloc[int(len(t) * 0.7)]
    return t[t.entry_time < cut], t[t.entry_time >= cut], cut

v19_is, v19_oos, v19_cut = split_7030(v19)
v32_is, v32_oos, v32_cut = split_7030(v32)

s19_full, s19_is, s19_oos = stat_block(v19), stat_block(v19_is), stat_block(v19_oos)
s32_full, s32_is, s32_oos = stat_block(v32), stat_block(v32_is), stat_block(v32_oos)

v19_pf_ok = s19_oos["pf"] >= PF_THRESHOLD
v19_wr_ok = (s19_is["wr"] - s19_oos["wr"]) <= WR_DIFF_THRESHOLD
v32_pf_ok = s32_oos["pf"] >= PF_THRESHOLD
v32_wr_ok = (s32_is["wr"] - s32_oos["wr"]) <= WR_DIFF_THRESHOLD

print(f"  V1.9 切分點 {v19_cut.date()}：IS(n={s19_is['n']}) WR{s19_is['wr']:.1f}%/PF{s19_is['pf']:.2f} | "
      f"OOS(n={s19_oos['n']}) WR{s19_oos['wr']:.1f}%/PF{s19_oos['pf']:.2f} | "
      f"{'✅通過' if v19_pf_ok and v19_wr_ok else '❌未過'}")
print(f"  V3.2 切分點 {v32_cut.date()}：IS(n={s32_is['n']}) WR{s32_is['wr']:.1f}%/PF{s32_is['pf']:.2f} | "
      f"OOS(n={s32_oos['n']}) WR{s32_oos['wr']:.1f}%/PF{s32_oos['pf']:.2f} | "
      f"{'✅通過' if v32_pf_ok and v32_wr_ok else '❌未過'}")

# ── B. V3.2 在「V1.9 OOS 那段相同日曆區間」的表現（直接對照）───────────────────
print("\nB. V3.2 在 V1.9-OOS 相同日曆區間的表現...")
v32_in_v19oos_window = v32[v32.entry_time >= v19_cut]
s32_in_v19oos = stat_block(v32_in_v19oos_window)
print(f"  V1.9 OOS 起始日 {v19_cut.date()} 之後：V3.2 有 {s32_in_v19oos['n']} 筆交易，"
      f"WR{s32_in_v19oos['wr']:.1f}%/PF{s32_in_v19oos['pf']:.2f}/淨利${s32_in_v19oos['net']:+,.0f}")

# ── time_bleed 對照 ──────────────────────────────────────────────────────────
def tb_stat(t):
    losses = t[~t.win]
    tb = losses[losses.hold_bars >= 24]
    return dict(n_loss=len(losses), n_tb=len(tb),
                tb_share=len(tb) / len(losses) * 100 if len(losses) else 0)

tb19, tb32 = tb_stat(v19), tb_stat(v32)

# ── 圖表 ──────────────────────────────────────────────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def dark_fig(w=10, h=5, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    fig.patch.set_facecolor("#0f172a")
    for ax in (axes if n > 1 else [axes]):
        ax.set_facecolor("#1e293b"); ax.tick_params(colors="#94a3b8"); ax.title.set_color("#e2e8f0")
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    return fig, axes

def chart_oos_compare():
    fig, axes = dark_fig(13, 4.5, n=2)
    for ax, metric, fmtv, ref in [(axes[0], "wr", "{:.1f}%", 50), (axes[1], "pf", "{:.2f}", PF_THRESHOLD)]:
        x = np.arange(2); w = 0.35
        is_v = [s19_is[metric], s32_is[metric]]
        oos_v = [s19_oos[metric], s32_oos[metric]]
        ax.bar(x - w/2, is_v, w, label="IS 前70%", color="#64748b")
        ax.bar(x + w/2, oos_v, w, label="OOS 後30%", color="#22c55e")
        ax.axhline(ref, color="#475569", ls="--")
        ax.set_xticks(x); ax.set_xticklabels(["V1.9", "V3.2"])
        ax.set_title("勝率 %" if metric == "wr" else "獲利因子", fontsize=12)
        ax.legend(facecolor="#1e293b", labelcolor="#94a3b8", fontsize=9)
        for i, (a, b) in enumerate(zip(is_v, oos_v)):
            ax.text(i - w/2, a * 1.02, fmtv.format(a), ha="center", color="#e2e8f0", fontsize=9)
            ax.text(i + w/2, b * 1.02, fmtv.format(b), ha="center", color="#e2e8f0", fontsize=9)
    fig.tight_layout(); return fig_b64(fig)

def chart_same_window():
    fig, ax = dark_fig(9, 4.8)
    labels = [f"V1.9 OOS\n(n={s19_oos['n']})", f"V3.2 同期間\n(n={s32_in_v19oos['n']})"]
    pfs = [s19_oos["pf"], s32_in_v19oos["pf"]]
    colors = ["#ef4444", "#22c55e"]
    bars = ax.bar(labels, pfs, color=colors, width=0.5)
    ax.axhline(PF_THRESHOLD, color="#475569", ls="--")
    ax.axhline(1.0, color="#94a3b8", ls=":")
    ax.set_title(f"相同日曆區間（{v19_cut.date()} 之後）PF 對照", fontsize=12)
    for bar, v in zip(bars, pfs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}", ha="center",
                color="#e2e8f0", fontsize=11, fontweight="bold")
    fig.tight_layout(); return fig_b64(fig)

print("\n生成圖表...")
imgs = dict(oos=chart_oos_compare(), window=chart_same_window())

# ── HTML ──────────────────────────────────────────────────────────────────────
def fmt(v, f="{:.2f}"): return "—" if (isinstance(v, float) and not np.isfinite(v)) else f.format(v)
def img(b): return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'

def oos_table():
    rows = []
    for name, full, is_, oos, pf_ok, wr_ok in [
        ("V1.9（原始邏輯）", s19_full, s19_is, s19_oos, v19_pf_ok, v19_wr_ok),
        ("V3.2（過濾器ON）", s32_full, s32_is, s32_oos, v32_pf_ok, v32_wr_ok),
    ]:
        verdict = "✅ 通過" if pf_ok and wr_ok else "❌ 未通過"
        color = "#22c55e" if pf_ok and wr_ok else "#ef4444"
        rows.append(
            f"<tr><td><strong>{name}</strong></td><td>{full['n']}</td>"
            f"<td>{fmt(is_['wr'],'{:.1f}')}%/{fmt(is_['pf'],'{:.2f}')}（n={is_['n']}）</td>"
            f"<td>{fmt(oos['wr'],'{:.1f}')}%/{fmt(oos['pf'],'{:.2f}')}（n={oos['n']}）</td>"
            f"<td style='color:{color}'><strong>{verdict}</strong></td></tr>")
    return (f"<table class='tbl'><thead><tr><th>版本</th><th>全樣本筆數</th>"
            f"<th>IS 前70%（WR/PF）</th><th>OOS 後30%（WR/PF）</th><th>OOS判定</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")

verdict_b_ok = s32_in_v19oos["pf"] >= PF_THRESHOLD and s32_in_v19oos["wr"] >= 40
verdict_b_class = "good" if verdict_b_ok else "bad"
verdict_b_text = ("V3.2 在 V1.9 失敗的那段日曆區間內表現轉正" if verdict_b_ok else
                   "V3.2 在 V1.9 失敗的那段日曆區間內仍然疲弱，過濾器沒有真正解決根本問題")

overall_ok = v32_pf_ok and v32_wr_ok and verdict_b_ok
overall_class = "good" if overall_ok else ("warn" if (v32_pf_ok and v32_wr_ok) or verdict_b_ok else "bad")
overall_text = ("雙重驗證都通過，V3.2 可視為已解決 V1.9 的 OOS 失敗問題" if overall_ok else
                 "雙重驗證中至少一項未通過，V3.2 尚不能視為完全解決問題，建議謹慎看待" if
                 (v32_pf_ok and v32_wr_ok) or verdict_b_ok else
                 "雙重驗證都未通過，V3.2 的全樣本改善可能是被早期樣本稀釋掩蓋，OOS 問題並未真正解決")

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
<title>S2-Hammer V3.2 是否解決 V1.9 的 OOS 失敗？</title>{CSS}</head>
<body>
<div style="max-width:1100px;margin:0 auto 14px"><a href="../../xauusd.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 XAUUSD 主頁</a></div>
<div class="wrap">
<h1>S2-Hammer V3.2 是否解決 <span style="color:#f59e0b">V1.9 的 OOS 失敗</span>？</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
V1.9 全樣本 {s19_full['n']} 筆 / V3.2 全樣本 {s32_full['n']} 筆（皆為 TradingView 真實成交，非模擬）·
姊妹報告：<a href="report_s2_attribution.html" style="color:var(--blue)">S2 真實逐筆歸因（V1.9 原始OOS失敗發現）</a></p>

<div class="{overall_class}"><strong>結論：{overall_text}</strong></div>

<!-- A -->
<div class="card">
<div class="part">驗證 A — 各自 70/30 時間切分（跟原始 V1.9 方法論一致）</div>
<h2>V3.2 自己的樣本外表現，有沒有比 V1.9 好？</h2>
{oos_table()}
{img(imgs['oos'])}
<p class="note">V1.9 切分點：{v19_cut.date()}（IS/OOS 各自佔 70%/30% 交易筆數）；
V3.2 切分點：{v32_cut.date()}（同樣方法，但因交易數少，切分點日期會不同）。
PF門檻 {PF_THRESHOLD}，勝率落差門檻 {WR_DIFF_THRESHOLD}pp。</p>
</div>

<!-- B -->
<div class="card">
<div class="part">驗證 B — 相同日曆區間直接對照（更嚴格的檢驗）</div>
<h2>在 V1.9 失敗的那段時間裡，V3.2 表現如何？</h2>
<p class="note">驗證 A 的兩個切分點日期不同，可能讓「V3.2 樣本少所以切分點較晚」稀釋掉真正的近期表現。
這裡改用同一個日曆窗口（V1.9 OOS 起始日 {v19_cut.date()} 之後）直接比較兩版：</p>
{img(imgs['window'])}
<table class="tbl">
<thead><tr><th>版本</th><th>筆數</th><th>勝率</th><th>PF</th><th>淨利</th></tr></thead>
<tbody>
<tr><td><strong>V1.9（該區間 = 定義上的OOS）</strong></td><td>{s19_oos['n']}</td><td>{fmt(s19_oos['wr'],'{:.1f}')}%</td><td>{fmt(s19_oos['pf'])}</td><td>${s19_oos['net']:+,.0f}</td></tr>
<tr><td><strong>V3.2（同一日曆區間）</strong></td><td>{s32_in_v19oos['n']}</td><td>{fmt(s32_in_v19oos['wr'],'{:.1f}')}%</td><td>{fmt(s32_in_v19oos['pf'])}</td><td>${s32_in_v19oos['net']:+,.0f}</td></tr>
</tbody></table>
<div class="{verdict_b_class}"><strong>{verdict_b_text}</strong></div>
</div>

<!-- time_bleed -->
<div class="card">
<div class="part">附註 — time_bleed 對照</div>
<h2>過濾器有沒有連帶改善 time_bleed？</h2>
<table class="tbl">
<thead><tr><th>版本</th><th>虧損筆數</th><th>拖≥12h筆數</th><th>佔比</th></tr></thead>
<tbody>
<tr><td>V1.9</td><td>{tb19['n_loss']}</td><td>{tb19['n_tb']}</td><td>{tb19['tb_share']:.0f}%</td></tr>
<tr><td>V3.2</td><td>{tb32['n_loss']}</td><td>{tb32['n_tb']}</td><td>{tb32['tb_share']:.0f}%</td></tr>
</tbody></table>
<p class="note">V3.2 未開啟提早保本過濾器（僅 HTF RSI + S2-RSI 互斥），若 time_bleed 佔比仍高，
可考慮疊加提早保本規則做進一步驗證。</p>
</div>

<p class="note">
局限：①單一 70/30 切點（驗證A）非 walk-forward；②V3.2 交易數少（互斥過濾器篩掉合流訊號），
驗證B的樣本數可能偏小，結論隨新資料累積可能改變；③兩版SL%相同(1.0%)但TP結構不同（V1.9為2R/4R，
V3.2同為2R/4R，可比），R倍數計算基礎一致。
</p>
<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S2-Hammer V3.2 OOS 驗證 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 報告已生成：{OUT_HTML}")
print(f"   驗證A：V1.9 OOS {'✅' if v19_pf_ok and v19_wr_ok else '❌'} | V3.2 OOS {'✅' if v32_pf_ok and v32_wr_ok else '❌'}")
print(f"   驗證B：V3.2 在 V1.9-OOS 同期間 {'✅' if verdict_b_ok else '❌'}（WR{s32_in_v19oos['wr']:.1f}%/PF{s32_in_v19oos['pf']:.2f}）")
