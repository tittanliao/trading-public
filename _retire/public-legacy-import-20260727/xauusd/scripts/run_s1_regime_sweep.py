"""
run_s1_regime_sweep.py — S1-AweWithBB V3.8.1 Regime 過濾器回看根數敏感度掃描
==============================================================================
輸出：xauusd/XAUUSD-Long-S1-AweWithBB/report_v3.8_regime_sweep.html

背景：V3.8.1（xauusd/XAUUSD-Long-S1-AweWithBB-V3.8.1.pine）把三組 Regime 過濾器
（斜率 / BBW 高檔 / BBW 低檔）的回看根數從 V3.7 的寫死值改成 input 可調，但
TradingView Strategy Tester 沒有辦法自動掃描一整組數值——只能手動改一次跑一次。
本腳本用 Python 重建 S1 的訊號與 H2 出場邏輯，對回看根數做網格掃描，一次跑完。

★★★ 重要方法論說明（先讀這段再看結果）★★★
1. 資料範圍限制：本機 csv/ 資料夾的 30m 檔案僅涵蓋 2026-01-21 ~ 2026-04-27
   （約 3 個月，3058 根），是目前 xauusd/csv/ 的最新匯出範圍，也是這個專案
   既有的 E01-E20 / FVG 實驗一直在用的同一份資料——本掃描沿用同一慣例，
   不是新的限制，但也代表這次掃描 ≠ 對照 report_v3.7_oos.html 用的
   2024-01 ~ 2026-07 那份 2.5 年真實成交紀錄。純粹是「同一組 3 個月資料，
   換不同 Regime 參數各跑一次」的相對比較，不是與真實逐筆歸因的絕對數字對照。
2. 出場邏輯是簡化重建，非逐筆精確複刻：Pine V3.7 的 H2 出場用 strategy.exit(stop=...)
   动态设定移動停損，實際成交序列取決於 TradingView 引擎逐 tick 撮合細節。
   本腳本用「當根K棒收盤價決定分級（SL/TP1追蹤/TP2追蹤），同根K棒低點判斷是否
   觸發停損」的簡化模型近似，未逐 tick 複刻。**但同一套簡化邏輯套用在所有被比較
   的參數組合上是完全一致的**，任何近似造成的系統性偏差會平均分攤到每一組，
   不影響「哪一組回看根數比較好」這個相對排名的公平性——這是本掃描的核心價值：
   排名可信，但單一組合的絕對 WR/PF 數字不能直接拿去對照 report_v3.7_real.html。
3. 樣本數提醒：3 個月資料下每組合的訊號筆數可能落在 10-40 筆，未必達到
   monthly_checklist.md 的 n≥30 門檻，報告會逐組標示筆數，筆數過少的組合
   結果僅供參考方向，不能當成定論。

掃描設計：
  - Sweep A（BBW 高檔回看根數，過濾器 ON，門檻固定 70%）：20/30/45/60(V3.7預設)/90/120/150
  - Sweep B（斜率回看根數，過濾器 ON 測試，門檻固定 0.15%；V3.7 預設此過濾器 OFF）：5/10/15/20/30
  - Sweep C（BBW 低檔回看根數，過濾器 ON 測試，門檻固定 30%；V3.7 預設此過濾器 OFF）：20/30/45/60/90/120
  - Baseline：V3.7 預設組合（BBW高檔=60 ON、斜率 OFF、BBW低檔 OFF），三組掃描都會標示這行方便比較

執行方式（在 trading/ 根目錄）：
    python3.12 xauusd/scripts/run_s1_regime_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "xauusd"))

from analysis import loader, config
from experiments import indicators as ind

OUT_HTML = config.STRATEGIES[0]["folder"].parent / "XAUUSD-Long-S1-AweWithBB" / "report_v3.8_regime_sweep.html"

SL_PCT   = 0.005
TP1_R    = 1.0
TP2_R    = 3.5
TIME_BARS = 36  # V3.7 值

# ── 1. 載入三個時框資料 ──────────────────────────────────────────────────────
print("載入 30m / 60m / 240m 價格資料...")
p30  = loader.load_price(config.PRICE_CSV)
p60  = loader.load_price(config.PRICE_CSV_60M)
p240 = loader.load_price(config.PRICE_CSV_4H)
print(f"  30m : {len(p30)} 根（{p30['time'].min()} → {p30['time'].max()}）")
print(f"  60m : {len(p60)} 根（{p60['time'].min()} → {p60['time'].max()}）")
print(f"  240m: {len(p240)} 根（{p240['time'].min()} → {p240['time'].max()}）")

# ── 2. 30m 核心進場指標（BB + Fast EMA + AO）───────────────────────────────────
close30, high30, low30, open30 = p30["close"].to_numpy(), p30["high"].to_numpy(), p30["low"].to_numpy(), p30["open"].to_numpy()
bb_basis, bb_upper, bb_lower = ind.bb(close30, 20, 2.0)
fast_ma = ind.ema(close30, 3)
ao = ind.awesome_oscillator(high30, low30, 5, 34)
ao_rising = np.r_[False, ao[1:] > ao[:-1]]  # math.abs(AO_State)==1 等價於「AO 較前一根上升」

crossover = np.r_[False, (fast_ma[:-1] <= bb_basis[:-1]) & (fast_ma[1:] > bb_basis[1:])]
base_long_cond = crossover & (close30 > bb_basis) & ao_rising

# ── 3. 60m：1H MA(3) 過濾器（V3.7 恆定啟用，非本次掃描對象）─────────────────────
ma1h_60 = ind.sma(p60["close"].to_numpy(), 3)
htf60 = pd.DataFrame({"time": p60["time"], "ma1h": ma1h_60})
merged_1h = pd.merge_asof(p30[["time"]].sort_values("time"), htf60.sort_values("time"), on="time", direction="backward")
filter_1h_ok = (close30 > merged_1h["ma1h"].to_numpy())

# ── 4. Regime 指標建構函式（回看根數為參數，供掃描呼叫）─────────────────────────
def bbw_rank(close_htf: np.ndarray, lookback: int) -> np.ndarray:
    bbw = ind.stdev(close_htf, 20) * 4 / ind.sma(close_htf, 20)
    return pd.Series(bbw).rolling(lookback).rank(pct=True).to_numpy() * 100

def slope_pct(close_htf: np.ndarray, lookback: int) -> np.ndarray:
    ema20 = ind.ema(close_htf, 20)
    prev = pd.Series(ema20).shift(lookback).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where((prev == 0) | np.isnan(prev), 0.0, (ema20 - prev) / prev * 100)
    return val

def merge_to_30m(htf_time: pd.Series, values: np.ndarray, colname: str) -> np.ndarray:
    htf_df = pd.DataFrame({"time": htf_time, colname: values})
    m = pd.merge_asof(p30[["time"]].sort_values("time"), htf_df.sort_values("time"), on="time", direction="backward")
    return m[colname].to_numpy()

close60_arr, close240_arr = p60["close"].to_numpy(), p240["close"].to_numpy()

def build_regime_ok(bbw_high_on: bool, bbw_high_lb: int, bbw_high_thresh: float,
                     slope_on: bool, slope_lb: int, slope_thresh: float,
                     bbw_low_on: bool, bbw_low_lb: int, bbw_low_thresh: float) -> np.ndarray:
    # 缺資料（NaN，通常是回看根數大於可用歷史）一律視為「不擋」，避免誤判成阻擋
    ok = np.ones(len(p30), dtype=bool)
    if bbw_high_on:
        r = merge_to_30m(p60["time"], bbw_rank(close60_arr, bbw_high_lb), "v")
        is_high = np.where(np.isnan(r), False, r >= bbw_high_thresh)
        ok &= ~is_high
    if slope_on:
        s = merge_to_30m(p240["time"], slope_pct(close240_arr, slope_lb), "v")
        is_down = np.where(np.isnan(s), False, s <= -slope_thresh)
        ok &= ~is_down
    if bbw_low_on:
        r2 = merge_to_30m(p240["time"], bbw_rank(close240_arr, bbw_low_lb), "v")
        is_low = np.where(np.isnan(r2), False, r2 < bbw_low_thresh)
        ok &= ~is_low
    return ok

# ── 5. 簡化版 H2 出場模擬（tier 由收盤價決定，同根K棒低點判斷觸發；見檔頭說明）───
def simulate_trades(long_cond: np.ndarray) -> list[dict]:
    n = len(p30)
    trades = []
    i = 21  # MaxUseBars = 1+max(bb_length,1) = 21
    in_trade = False
    entry_bar = entry_price = sl = tp1 = tp2 = None
    while i < n - 1:
        if not in_trade:
            if long_cond[i]:
                entry_bar = i + 1
                entry_price = open30[entry_bar]
                sl  = entry_price * (1 - SL_PCT)
                tp1 = entry_price * (1 + SL_PCT * TP1_R)
                tp2 = entry_price * (1 + SL_PCT * TP2_R)
                in_trade = True
                i = entry_bar
                continue
            i += 1
            continue

        # in_trade：從 entry_bar 起逐根檢查
        j = i
        c = close30[j]
        if c >= tp2:
            lo_window = low30[max(entry_bar, j - TIME_BARS // 4 + 1): j + 1]
            stop = max(tp2, lo_window.min())
            tier = "TP2"
        elif c >= tp1:
            lo_window = low30[max(entry_bar, j - TIME_BARS // 2 + 1): j + 1]
            stop = max(tp1, lo_window.min())
            tier = "TP1"
        else:
            stop = sl
            tier = "SL"

        exited = False
        if low30[j] <= stop:
            exit_price, exit_kind, exited = stop, tier, True
        elif (j - entry_bar) >= TIME_BARS:
            if c >= entry_price:
                lo_window = low30[max(entry_bar, j - TIME_BARS + 1): j + 1]
                exit_price = max(sl, lo_window.min())
                exit_kind = "TimeWin"
            else:
                exit_price = low30[j]
                exit_kind = "TimeLoss"
            exited = True

        if exited:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            r_mult  = (exit_price - entry_price) / (entry_price * SL_PCT)
            trades.append(dict(
                entry_time=p30["time"].iloc[entry_bar], exit_time=p30["time"].iloc[j],
                entry_price=entry_price, exit_price=exit_price, exit_kind=exit_kind,
                pnl_pct=pnl_pct, r_multiple=r_mult, hold_bars=j - entry_bar,
                result="win" if pnl_pct > 0 else "loss",
            ))
            in_trade = False
            i = j + 1
        else:
            i = j + 1
    return trades

def stat_row(trades: list[dict], label: str) -> dict:
    if not trades:
        return dict(label=label, n=0, wr=float("nan"), pf=float("nan"), net_pct=0.0, avg_r=float("nan"))
    df = pd.DataFrame(trades)
    wins = df.loc[df["result"] == "win", "pnl_pct"].sum()
    loss = abs(df.loc[df["result"] == "loss", "pnl_pct"].sum())
    return dict(
        label=label, n=len(df), wr=(df["result"] == "win").mean() * 100,
        pf=(wins / loss) if loss else float("inf"),
        net_pct=df["pnl_pct"].sum(), avg_r=df["r_multiple"].mean(),
    )

# ── 6. Baseline（V3.7 預設）───────────────────────────────────────────────────
print("計算 Baseline（V3.7 預設參數）...")
regime_base = build_regime_ok(True, 60, 70.0, False, 10, 0.15, False, 60, 30.0)
trades_base = simulate_trades(base_long_cond & filter_1h_ok & regime_base)
row_base = stat_row(trades_base, "Baseline（BBW高60 ON / 斜率OFF / BBW低OFF，= V3.7）")
print(f"  {row_base}")

# ── 7. Sweep A：BBW 高檔回看根數 ─────────────────────────────────────────────
print("Sweep A：BBW 高檔回看根數...")
rows_a = [row_base]
for lb in [20, 30, 45, 90, 120, 150]:
    regime = build_regime_ok(True, lb, 70.0, False, 10, 0.15, False, 60, 30.0)
    trades = simulate_trades(base_long_cond & filter_1h_ok & regime)
    rows_a.append(stat_row(trades, f"BBW高{lb}"))
    print(f"  lb={lb}: {rows_a[-1]}")

# ── 8. Sweep B：斜率回看根數（測試打開此過濾器）─────────────────────────────────
print("Sweep B：斜率回看根數（測試 ON）...")
rows_b = [row_base]
for lb in [5, 10, 15, 20, 30]:
    regime = build_regime_ok(True, 60, 70.0, True, lb, 0.15, False, 60, 30.0)
    trades = simulate_trades(base_long_cond & filter_1h_ok & regime)
    rows_b.append(stat_row(trades, f"+斜率{lb}"))
    print(f"  lb={lb}: {rows_b[-1]}")

# ── 9. Sweep C：BBW 低檔回看根數（測試打開此過濾器）─────────────────────────────
print("Sweep C：BBW 低檔回看根數（測試 ON）...")
rows_c = [row_base]
for lb in [20, 30, 45, 90, 120]:
    regime = build_regime_ok(True, 60, 70.0, False, 10, 0.15, True, lb, 30.0)
    trades = simulate_trades(base_long_cond & filter_1h_ok & regime)
    rows_c.append(stat_row(trades, f"+BBW低{lb}"))
    print(f"  lb={lb}: {rows_c[-1]}")

# ── 10. HTML 輸出 ────────────────────────────────────────────────────────────
def fmt(v, f="{:.2f}"):
    return "—" if (v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)))) else f.format(v)

def rows_to_table(rows: list[dict], baseline_label: str) -> str:
    body = ""
    for r in rows:
        hl = ' style="background:rgba(56,189,248,.12)"' if r["label"] == baseline_label else ""
        n_warn = ' style="color:#f59e0b"' if r["n"] < 30 else ""
        body += (f"<tr{hl}><td>{r['label']}</td><td{n_warn}>{r['n']}</td>"
                 f"<td>{fmt(r['wr'],'{:.1f}')}%</td><td>{fmt(r['pf'],'{:.3f}')}</td>"
                 f"<td>{fmt(r['net_pct'],'{:+.2f}')}%</td><td>{fmt(r['avg_r'],'{:+.3f}')}R</td></tr>")
    return (f"<table class='tbl'><thead><tr><th>參數組合</th><th>筆數</th><th>勝率</th>"
            f"<th>獲利因子</th><th>淨盈虧%</th><th>平均R</th></tr></thead><tbody>{body}</tbody></table>")

CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
:root{--green:#22c55e;--yellow:#f59e0b;--blue:#38bdf8;--muted:#94a3b8;--border:#334155}
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
</style>
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S1 V3.8.1 Regime 回看根數掃描</title>{CSS}</head>
<body>
<div style="max-width:1100px;margin:0 auto 14px"><a href="../index.html" style="color:#38bdf8;font-size:13px;text-decoration:none;background:rgba(56,189,248,.12);padding:5px 14px;border-radius:6px;font-weight:600">← 返回 Trading Hub</a></div>
<div class="wrap">
<h1>S1-AweWithBB V3.8.1 <span style="color:#f59e0b">Regime 回看根數</span> 敏感度掃描</h1>
<p class="note">生成時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
資料範圍：{p30['time'].min()} ~ {p30['time'].max()}（本機 csv/ 最新匯出，約3個月）</p>

<div class="warn">
<strong>重要：</strong>本頁數字是 Python 簡化重建的近似回測，用於「同一資料上比較不同回看根數哪個較好」，
<strong>不是</strong> report_v3.7_real.html／report_v3.7_oos.html 的真實逐筆歸因數字，兩邊不能直接對照。
資料範圍也只有約3個月（xauusd/csv/ 最新匯出），部分組合筆數（標黃色）低於 30 筆門檻，
結果僅供方向參考。若某組合明顯優於 Baseline，下一步應該回 TradingView 用真實資料再驗證，
而不是直接改實盤參數。
</div>

<div class="card">
<div class="part">Sweep A</div>
<h2>BBW 高檔回看根數（過濾器 ON，門檻固定 70%）</h2>
<p class="note">V3.7 預設 = 60（原始寫死值，無回測依據，見 V3.8.1 pine 檔頭說明）</p>
{rows_to_table(rows_a, row_base['label'])}
</div>

<div class="card">
<div class="part">Sweep B</div>
<h2>斜率回看根數（V3.7 預設此過濾器 OFF，測試打開）</h2>
<p class="note">門檻固定 0.15%；若打開後在任何回看根數下都不如 Baseline，代表這個過濾器目前不值得啟用</p>
{rows_to_table(rows_b, row_base['label'])}
</div>

<div class="card">
<div class="part">Sweep C</div>
<h2>BBW 低檔回看根數（V3.7 預設此過濾器 OFF，測試打開）</h2>
<p class="note">門檻固定 30%；同上，檢驗是否值得啟用</p>
{rows_to_table(rows_c, row_base['label'])}
</div>

<p class="note" style="margin-top:8px">
方法：30m/60m/240m 價格皆為 xauusd/csv/ 本機匯出檔（loader.load_price）。
高時框指標一律用 pd.merge_asof(..., direction="backward")取「當下已收盤」的最新高時框值，
與 Pine request.security(lookahead_off) 語意一致。出場為簡化重建的 H2 分級追蹤停損模型，
非逐筆精確複刻（詳見腳本檔頭說明）。R 倍數 = (出場價-進場價)/(進場價×0.5%)。
</p>

<footer style="text-align:center;color:#475569;font-size:.78em;margin-top:24px;padding-top:16px;border-top:1px solid var(--border)">
XAUUSD S1-AweWithBB V3.8.1 Regime 回看根數敏感度掃描 · {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</div></body></html>"""

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(HTML, encoding="utf-8")
print(f"\n✅ 掃描報告已生成：{OUT_HTML}")
