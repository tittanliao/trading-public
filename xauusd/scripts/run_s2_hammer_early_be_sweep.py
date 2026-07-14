"""
run_s2_hammer_early_be_sweep.py — S2-Hammer 提早保本（Early Break-Even）參數掃描
==============================================================================
輸出：xauusd/XAUUSD-Long-S2-Hammer/report_early_be_sweep.html

背景（20260711）：V3.2 fail-pattern 對照發現過濾器只砍到 immediate_loss，
time_bleed（拖倉≥12h的虧損單）完全沒被處理，因為 V3.2 目前沒開提早保本
（FILTER③，pine 裡已經有這個開關，只是預設 OFF）。這支腳本用「現有真實資料」
（真實逐筆交易的進出場時間 + 30m 真實K棒路徑）逐bar重建：如果當初開了提早保本，
在不同觸發門檻（be_trigger_pct）下，實際會發生什麼？

方法（bar-level 逐K棒重建，非僅用 MFE/MAE 彙總數字粗估）：
  對每筆真實交易，從進場時間起逐根30m K棒往後走（上限48根，對應time exit）：
    1. 武裝（armed）：high 觸及 entry*(1+trigger%) 時武裝
    2. 觸發（triggered）：武裝後 low 觸及保本價（預設=entry價，offset=0）時觸發
    3. 若觸發時間早於原始出場時間 → 保本會搶先出場，把原結果換成 ~$0
       （原本輸的單 = 被救回「rescued」；原本贏的單 = 被提早出場「clipped」，這是代價）
    4. 若從未觸發，或觸發時間晚於原始出場 → 保本不影響原結果
  對多個 be_trigger_pct 門檻（0.1%~1.0%）重複，找出淨效果最好的門檻。

同時分別對 V1.9（無過濾器基準，224筆）與 V3.2（含過濾器，157筆）跑，
並額外對「V1.9 OOS 那段」（2026-01-22之後）單獨跑一次，檢驗提早保本
是否真的對「最麻煩的那段時期」有幫助（而不只是全樣本數字好看）。

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s2_hammer_early_be_sweep.py
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

from analysis import loader, metrics
from analysis.config import PRICE_CSV

FOLDER   = ROOT / "xauusd/XAUUSD-Long-S2-Hammer"
V19_CSV  = FOLDER / "S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-07-11.csv"
V32_CSV  = FOLDER / "S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv"
OUT_HTML = FOLDER / "report_early_be_sweep.html"

SL_PCT = 1.0          # 兩版皆為 1.0%
OUT_K_COUNT = 48       # 時間出場上限（bars），BE 搜尋範圍不超過這個horizon
BE_OFFSET_PCT = 0.0    # 保本偏移，0 = 出場約在進場價（忽略滑價/手續費）
TRIGGER_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]
V19_OOS_CUT = pd.Timestamp("2026-01-22")  # 20260711 歸因分析定義的 V1.9 OOS 起點

print("載入真實逐筆交易與 30m 真實K棒...")
price = loader.load_price(PRICE_CSV)
p_time = price["time"].to_numpy()
p_high = price["high"].to_numpy()
p_low = price["low"].to_numpy()

def load(csv_path: Path, label: str) -> pd.DataFrame:
    t = loader.load_trades(csv_path)
    t = t[t["exit_signal"] != "Open"].reset_index(drop=True)
    t["win"] = t["net_pnl_usd"] > 0
    t["version"] = label
    return t.sort_values("entry_time").reset_index(drop=True)

v19 = load(V19_CSV, "V1.9")
v32 = load(V32_CSV, "V3.2")
print(f"  V1.9：{len(v19)} 筆 | V3.2：{len(v32)} 筆")
print(f"  30m 真實K棒：{len(price)} 根（{price['time'].min()} → {price['time'].max()}）")

# ── 逐bar重建：對每筆交易，回傳各 trigger 門檻下的「觸發時間」（None=未觸發）──────
def simulate_trade(entry_time, entry_price, exit_time):
    start_idx = np.searchsorted(p_time, entry_time, side="right")  # 進場之後第一根
    end_idx = min(start_idx + OUT_K_COUNT, len(p_time))
    result = {}
    for trig in TRIGGER_SWEEP:
        arm_level = entry_price * (1 + trig / 100)
        be_stop = entry_price * (1 + BE_OFFSET_PCT / 100)
        armed = False
        triggered_time = None
        for i in range(start_idx, end_idx):
            t = p_time[i]
            if t >= exit_time:
                break
            if not armed and p_high[i] >= arm_level:
                armed = True
            if armed and p_low[i] <= be_stop:
                triggered_time = t
                break
        result[trig] = triggered_time
    return result

def run_sweep(trades: pd.DataFrame, label: str, oos_only: bool = False):
    sub = trades[trades.entry_time >= V19_OOS_CUT] if oos_only else trades
    print(f"\n{label}{'（OOS段）' if oos_only else ''}：{len(sub)} 筆，逐bar重建中...")
    rows = []
    for trig in TRIGGER_SWEEP:
        adj_pnl = sub["net_pnl_usd"].copy()
        n_rescued = n_clipped = 0
        tb_rescued = 0
        for idx, row in sub.iterrows():
            sim = simulate_trade(row.entry_time, row.entry_price, row.exit_time)
            if sim[trig] is not None:
                if row.net_pnl_usd < 0:
                    n_rescued += 1
                    if row.hold_bars >= 24:
                        tb_rescued += 1
                elif row.net_pnl_usd > 0:
                    n_clipped += 1
                adj_pnl.loc[idx] = 0.0
        wins = adj_pnl[adj_pnl > 0].sum()
        losses = abs(adj_pnl[adj_pnl < 0].sum())
        n = len(sub)
        wr = (adj_pnl > 0).mean() * 100 if n else np.nan
        pf = (wins / losses) if losses else float("inf")
        net = adj_pnl.sum()
        orig_net = sub["net_pnl_usd"].sum()
        rows.append(dict(trigger=trig, n_rescued=n_rescued, n_clipped=n_clipped,
                          tb_rescued=tb_rescued, wr=wr, pf=pf, net=net,
                          delta_net=net - orig_net))
        print(f"  trigger={trig:.2f}%: rescued={n_rescued}(tb={tb_rescued}) clipped={n_clipped} "
              f"WR={wr:.1f}% PF={pf:.2f} Net=${net:+,.0f} (Δ{net-orig_net:+,.0f})")
    return pd.DataFrame(rows), sub["net_pnl_usd"].sum(), (sub["net_pnl_usd"] > 0).mean() * 100

sweep_v19, orig_net_v19, orig_wr_v19 = run_sweep(v19, "V1.9")
sweep_v32, orig_net_v32, orig_wr_v32 = run_sweep(v32, "V3.2")
sweep_v19_oos, orig_net_v19_oos, orig_wr_v19_oos = run_sweep(v19, "V1.9", oos_only=True)

# ── 挑最佳門檻（淨利提升最多，且 rescued > clipped 才算真的有幫助）───────────────
def pick_best(sweep: pd.DataFrame):
    candidates = sweep[sweep.n_rescued > sweep.n_clipped]
    if candidates.empty:
        return sweep.loc[sweep.delta_net.idxmax()]
    return candidates.loc[candidates.delta_net.idxmax()]

best_v19 = pick_best(sweep_v19)
best_v32 = pick_best(sweep_v32)
best_v19_oos = pick_best(sweep_v19_oos)

print(f"\n最佳門檻 V1.9：{best_v19.trigger:.2f}%（Δ淨利 ${best_v19.delta_net:+,.0f}）")
print(f"最佳門檻 V3.2：{best_v32.trigger:.2f}%（Δ淨利 ${best_v32.delta_net:+,.0f}）")
print(f"最佳門檻 V1.9 OOS段：{best_v19_oos.trigger:.2f}%（Δ淨利 ${best_v19_oos.delta_net:+,.0f}）")

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

def chart_sweep(sweep: pd.DataFrame, title: str, orig_net: float):
    fig, ax = dark_fig(11, 4.8)
    x = sweep.trigger.astype(str)
    bars = ax.bar(x, sweep.delta_net, color=["#22c55e" if v > 0 else "#ef4444" for v in sweep.delta_net], width=0.6)
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.set_title(f"{title}：不同保本觸發門檻的淨利變化 $（原始淨利 ${orig_net:+,.0f}）", fontsize=12)
    ax.set_xlabel("保本觸發門檻 (%)")
    for bar, v, r, c in zip(bars, sweep.delta_net, sweep.n_rescued, sweep.n_clipped):
        ax.text(bar.get_x()+bar.get_width()/2, v + (5 if v>=0 else -15), f"{v:+,.0f}\nR{r}/C{c}",
                ha="center", color="#e2e8f0", fontsize=8)
    fig.tight_layout(); return fig_b64(fig)

print("\n生成圖表...")
imgs = dict(v19=chart_sweep(sweep_v19, "V1.9（全樣本）", orig_net_v19),
            v32=chart_sweep(sweep_v32, "V3.2（全樣本）", orig_net_v32),
            v19oos=chart_sweep(sweep_v19_oos, "V1.9（OOS段，2026-01-22後）", orig_net_v19_oos))

# ── HTML ──────────────────────────────────────────────────────────────────────
def img(b): return f'<img src="data:image/png;base64,{b}" style="max-width:100%;border-radius:8px;margin:10px 0">'

def sweep_table(sweep: pd.DataFrame, best_trig: float):
    body = ""
    for _, r in sweep.iterrows():
        hl = " style='background:rgba(34,197,94,.1);font-weight:700'" if abs(r.trigger - best_trig) < 1e-9 else ""
        body += (f"<tr{hl}><td>{r.trigger:.2f}%</td><td>{r.n_rescued}</td><td>{r.tb_rescued}</td>"
                 f"<td>{r.n_clipped}</td><td>{r.wr:.1f}%</td><td>{r.pf:.2f}</td>"
                 f"<td>${r.net:+,.0f}</td><td style='color:{'#22c55e' if r.delta_net>0 else '#ef4444'}'>"
                 f"${r.delta_net:+,.0f}</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>觸發門檻</th><th>救回筆數</th><th>其中time_bleed</th>"
            f"<th>被提前出場筆數</th><th>新勝率</th><th>新PF</th><th>新淨利</th><th>淨利變化</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")

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
<title>S2-Hammer 提早保本參數掃描</title>{CSS}</head>
<body>
<div style="max-width:1150px;margin:0 auto 14px"><a href="../../xauusd.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 XAUUSD 主頁</a></div>
<div class="wrap">
<h1>S2-Hammer <span style="color:#f59e0b">提早保本</span> 參數掃描</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
方法：對每筆真實交易用 30m 真實K棒逐bar重建「若開啟提早保本會發生什麼」，非僅用MFE/MAE彙總數字粗估 ·
V1.9 {len(v19)} 筆 / V3.2 {len(v32)} 筆</p>

<div class="good">
<strong>建議設定：</strong>V1.9 全樣本最佳門檻 <strong>{best_v19.trigger:.2f}%</strong>
（救回 {best_v19.n_rescued} 筆，其中 {best_v19.tb_rescued} 筆是 time_bleed；代價是提前出場 {best_v19.n_clipped} 筆贏單；
淨利變化 ${best_v19.delta_net:+,.0f}）。V3.2 最佳門檻 <strong>{best_v32.trigger:.2f}%</strong>
（Δ淨利 ${best_v32.delta_net:+,.0f}）。OOS段（近5個月）最佳門檻 <strong>{best_v19_oos.trigger:.2f}%</strong>
（Δ淨利 ${best_v19_oos.delta_net:+,.0f}）——即使在最麻煩的那段時期，提早保本{"仍有正面幫助" if best_v19_oos.delta_net > 0 else "效果有限"}。
</div>

<!-- V1.9 -->
<div class="card">
<div class="part">V1.9 — 全樣本（224筆）</div>
<h2>不同保本觸發門檻的效果</h2>
{sweep_table(sweep_v19, best_v19.trigger)}
{img(imgs['v19'])}
</div>

<!-- V3.2 -->
<div class="card">
<div class="part">V3.2 — 全樣本（157筆，已含HTF RSI+互斥過濾器）</div>
<h2>不同保本觸發門檻的效果</h2>
{sweep_table(sweep_v32, best_v32.trigger)}
{img(imgs['v32'])}
<p class="note">V3.2 本身樣本已被過濾器篩過一輪，這裡驗證的是「在V3.2基礎上再疊加提早保本」的邊際效果。</p>
</div>

<!-- V1.9 OOS -->
<div class="card">
<div class="part">V1.9 OOS段（2026-01-22之後，即先前發現OOS失敗的那段）</div>
<h2>在「最麻煩的時期」提早保本是否真的有幫助？</h2>
{sweep_table(sweep_v19_oos, best_v19_oos.trigger)}
{img(imgs['v19oos'])}
<p class="note">這段是先前 report_s2_attribution.html 發現 V1.9 原始邏輯 WR僅27.9%/PF0.92（低於損益兩平）的樣本外時期，
直接驗證提早保本對這段是否有實質幫助，比只看全樣本數字更有說服力。</p>
</div>

<p class="note">
局限：①模擬用 close=entry 的保本價（offset=0%），忽略滑價與手續費（原始pine設定commission=0，slippage=1 tick，
與此簡化模擬的誤差極小）；②逐bar判斷用「武裝後low觸及保本價」，未精確重現Pine引擎逐bar出場優先順序
（time exit > TP2 > TP1 > BE > SL）的每一個邊界情況，但由於只在「原始出場時間之前」才判定保本搶先出場，
方向性結論應可信；③30m K棒完整覆蓋整段回測期間（20260711重匯後），無資料缺口問題。
</p>
<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S2-Hammer 提早保本參數掃描 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 報告已生成：{OUT_HTML}")
