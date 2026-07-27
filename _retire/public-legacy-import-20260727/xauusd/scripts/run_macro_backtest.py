"""
XAUUSD Macro-Filtered Backtest
================================
計算每日 Macro Score（0–6 分，依 Macro v3.7 Pine Script 邏輯），
套用到 S1-proxy（E12_BB_Squeeze_Break）和 S2-proxy（E18_Hammer）策略，
比較「無過濾」vs「宏觀過濾」的回測結果。

執行（20260705 移至 scripts/ 子資料夾，仍需在 trading/ 根目錄執行）：python3 xauusd/scripts/run_macro_backtest.py
輸出：xauusd/XAUUSD-Macro/macro_backtest_report.html
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.engine import run_backtest, Trade
from experiments.strategies import E12_BB_Squeeze_Break, E18_Hammer, E06_RSI_Oversold

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent  # 20260705 移至 scripts/ 子資料夾，多一層 .parent
CSV_DIR = ROOT / "csv"
OUT_DIR = ROOT / "XAUUSD-Macro"
OUT_DIR.mkdir(exist_ok=True)

# ── Macro Score Params（同 Pine Script v3.7）─────────────────────────────────
MA_LEN      = 50    # MA50 for macro trend
GVZ_EXT_TH  = 20.0
GVZ_SQZ_TH  = 13.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. 載入資料
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlc(fname: str, date_col: str = "time") -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / fname)
    df.columns = df.columns.str.strip()
    df[date_col] = pd.to_datetime(df[date_col], utc=False).dt.tz_localize(None)
    df = df.sort_values(date_col).reset_index(drop=True)
    # 統一 rename：保留 time / open / high / low / close
    if date_col != "time":
        df = df.rename(columns={date_col: "time"})
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns and "close" in df.columns:
            df[c] = df["close"]  # 對 FRED 單欄資料補齊
    return df[["time"] + [c for c in ["open","high","low","close"] if c in df.columns]]


def load_close_only(fname: str) -> pd.DataFrame:
    """FRED T10YIE 只有 time + close。"""
    df = pd.read_csv(CSV_DIR / fname)
    df.columns = df.columns.str.strip()
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    close_col = [c for c in df.columns if "close" in c.lower()][0]
    df = df[[time_col, close_col]].rename(columns={time_col: "time", close_col: "close"})
    df["time"] = pd.to_datetime(df["time"], utc=False).dt.tz_localize(None)
    return df.sort_values("time").reset_index(drop=True)


print("📂 載入 CSV...")
price_30m  = load_ohlc("FX_IDC_XAUUSD, 30.csv")
gold_1d    = load_ohlc("FX_IDC_XAUUSD, 1D.csv")
dxy_1d     = load_ohlc("TVC_DXY, 1D.csv")
us10y_1d   = load_ohlc("TVC_US10Y, 1D.csv")
t10yie_1d  = load_close_only("FRED_T10YIE, 1D.csv")
vix_1d     = load_ohlc("TVC_VIX, 1D.csv")
gvz_1d     = load_ohlc("CBOE_GVZ, 1D.csv")

print(f"  XAUUSD 30m : {len(price_30m)} bars  {price_30m['time'].min().date()} → {price_30m['time'].max().date()}")
print(f"  Gold 1D    : {len(gold_1d)} bars  {gold_1d['time'].min().date()} → {gold_1d['time'].max().date()}")
print(f"  US10Y 1D   : {len(us10y_1d)} bars")
print(f"  T10YIE 1D  : {len(t10yie_1d)} bars")
print(f"  VIX 1D     : {len(vix_1d)} bars")
print(f"  GVZ 1D     : {len(gvz_1d)} bars")
print(f"  DXY 1D     : {len(dxy_1d)} bars")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 計算每日 Macro Score（同 Pine Script v3.7 邏輯）
# ─────────────────────────────────────────────────────────────────────────────

def ma50(series: pd.Series) -> pd.Series:
    return series.rolling(MA_LEN, min_periods=MA_LEN).mean()


def build_macro_scores() -> pd.DataFrame:
    """傳回 DataFrame: date / score / verdict / gvz_state / 各因子得分"""

    def to_daily(df: pd.DataFrame, col: str) -> pd.DataFrame:
        src = df[["time", "close"]].rename(columns={"close": col}).sort_values("time")
        return src

    # 以 US10Y 為基準時間軸
    base = to_daily(us10y_1d, "us10y")

    for df, col in [(t10yie_1d, "t10yie"), (dxy_1d, "dxy"),
                    (vix_1d, "vix"), (gvz_1d, "gvz"), (gold_1d, "gold")]:
        tmp = to_daily(df, col)
        base = pd.merge_asof(base.sort_values("time"),
                             tmp.sort_values("time"),
                             on="time", direction="backward")

    # Real Rate
    base["real_rate"] = base["us10y"] - base["t10yie"]

    # MA50 計算
    base["ma50_rr"]   = ma50(base["real_rate"])
    base["ma50_10y"]  = ma50(base["us10y"])
    base["ma50_dxy"]  = ma50(base["dxy"])
    base["ma50_vix"]  = ma50(base["vix"])
    base["ma50_gold"] = ma50(base["gold"])

    # 得分（需 MA50 暖機後才有效）
    base["pt_real"]  = np.where(base["real_rate"] < base["ma50_rr"],   2, 0)
    base["pt_10y"]   = np.where(base["us10y"]     < base["ma50_10y"],  1, 0)
    base["pt_dxy"]   = np.where(base["dxy"]       < base["ma50_dxy"],  1, 0)
    base["pt_vix"]   = np.where(base["vix"]       > base["ma50_vix"],  1, 0)
    base["pt_trend"] = np.where(base["gold"]      > base["ma50_gold"], 1, 0)
    base["score"]    = base[["pt_real","pt_10y","pt_dxy","pt_vix","pt_trend"]].sum(axis=1)

    # 僅在 MA50 暖機後有效
    valid = base[["ma50_rr","ma50_10y","ma50_dxy","ma50_vix","ma50_gold"]].notna().all(axis=1)
    base.loc[~valid, "score"] = np.nan

    base["verdict"] = base["score"].apply(
        lambda s: "STRONG BUY" if s >= 5 else ("WAIT" if s <= 2 else "NEUTRAL")
        if not pd.isna(s) else "N/A"
    )
    base["gvz_state"] = base["gvz"].apply(
        lambda g: "EXTREME" if g > GVZ_EXT_TH else ("SQUEEZE" if g < GVZ_SQZ_TH else "NORMAL")
        if not pd.isna(g) else "N/A"
    )
    return base.dropna(subset=["score"]).reset_index(drop=True)


print("\n⚙️  計算 Macro Score...")
macro_daily = build_macro_scores()
print(f"  有效日數: {len(macro_daily)} 天")

# 僅保留 30m 資料覆蓋範圍
date_min = price_30m["time"].min().date()
date_max = price_30m["time"].max().date()
macro_range = macro_daily[
    (macro_daily["time"].dt.date >= date_min) &
    (macro_daily["time"].dt.date <= date_max)
].copy()

print(f"  回測期間 Macro 分佈（{date_min} → {date_max}）：")
verdict_cnt = macro_range["verdict"].value_counts()
for v, n in verdict_cnt.items():
    print(f"    {v}: {n} 天")
gvz_cnt = macro_range["gvz_state"].value_counts()
for g, n in gvz_cnt.items():
    print(f"    GVZ {g}: {n} 天")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 將日線 Macro Score 對應到每個 30m bar（使用前一日收盤的分數）
# ─────────────────────────────────────────────────────────────────────────────

# 建立 date → score / verdict / gvz_state 的查找表（用前一日，避免未來資料）
macro_shifted = macro_daily[["time","score","verdict","gvz_state",
                              "pt_real","pt_10y","pt_dxy","pt_vix","pt_trend",
                              "real_rate","us10y","dxy","vix","gvz"]].copy()
macro_shifted["date"] = macro_shifted["time"].dt.date

# 建立 30m 的 date 欄
price_30m["date"] = price_30m["time"].dt.date

# merge_asof：對每個 30m bar，找 < 當天 00:00 的最近一筆 macro
price_30m_sorted = price_30m.sort_values("time").reset_index(drop=True)
macro_ts = macro_shifted.copy()
macro_ts["ts"] = pd.to_datetime(macro_ts["date"].astype(str))

price_30m_sorted = pd.merge_asof(
    price_30m_sorted,
    macro_ts[["ts","score","verdict","gvz_state",
              "pt_real","pt_10y","pt_dxy","pt_vix","pt_trend",
              "real_rate","us10y","dxy","vix","gvz"]].rename(columns={"ts":"time_macro"}),
    left_on="time", right_on="time_macro",
    direction="backward"
)
price_30m_sorted["verdict"] = price_30m_sorted["verdict"].fillna("N/A")
price_30m_sorted["gvz_state"] = price_30m_sorted["gvz_state"].fillna("N/A")
price_30m_sorted["score"] = price_30m_sorted["score"].fillna(-1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 回測函數：包裝 signal_fn 加入 macro filter
# ─────────────────────────────────────────────────────────────────────────────

def make_filtered_signal(signal_fn, price_with_macro: pd.DataFrame, strategy_type: str):
    """
    返回一個新的 signal_fn，在原始信號基礎上加入 Macro 過濾。
    S1 策略在 WAIT 環境跳過；S2 策略全部執行。
    """
    verdicts = price_with_macro["verdict"].to_numpy() if "verdict" in price_with_macro.columns else None

    def wrapped(df, i):
        if not signal_fn(df, i):
            return False
        if verdicts is None:
            return True
        v = str(verdicts[i]) if i < len(verdicts) else "N/A"
        if strategy_type == "S1" and v == "WAIT":
            return False
        return True
    return wrapped


def tag_trades_with_macro(trades: list[Trade], price_with_macro: pd.DataFrame) -> list[dict]:
    """為每筆已完成交易附上宏觀環境標記。"""
    verdict_map   = price_with_macro.set_index("time")["verdict"]   if "verdict"   in price_with_macro.columns else None
    gvz_map       = price_with_macro.set_index("time")["gvz_state"] if "gvz_state" in price_with_macro.columns else None
    score_map     = price_with_macro.set_index("time")["score"]     if "score"     in price_with_macro.columns else None

    tagged = []
    for t in trades:
        entry_t = pd.Timestamp(t.entry_time)
        try:
            v = str(verdict_map.asof(entry_t)) if verdict_map is not None else "N/A"
            g = str(gvz_map.asof(entry_t))     if gvz_map   is not None else "N/A"
            s = float(score_map.asof(entry_t)) if score_map is not None else -1
        except Exception:
            v, g, s = "N/A", "N/A", -1
        tagged.append({"trade": t, "verdict": v, "gvz_state": g, "score": s})
    return tagged


def summarize(trades) -> dict:
    """統計交易結果（接受 Trade list 或 tagged dict list）。"""
    def get_t(x):
        if isinstance(x, dict):
            return x["trade"]
        return x

    valid = [x for x in trades if get_t(x).result in ("win", "loss")]
    if not valid:
        return {"n": 0, "wins": 0, "losses": 0, "wr": 0, "pf": 0,
                "net_pct": 0, "avg_hold": 0}
    wins   = [x for x in valid if get_t(x).result == "win"]
    losses = [x for x in valid if get_t(x).result == "loss"]
    gross_win  = sum(get_t(x).pnl_pct for x in wins)
    gross_loss = abs(sum(get_t(x).pnl_pct for x in losses))
    return {
        "n"        : len(valid),
        "wins"     : len(wins),
        "losses"   : len(losses),
        "wr"       : len(wins) / len(valid) * 100,
        "pf"       : gross_win / gross_loss if gross_loss > 0 else 999,
        "net_pct"  : sum(get_t(x).pnl_pct for x in valid),
        "avg_hold" : sum(get_t(x).hold_bars for x in valid) / len(valid),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. 執行回測
# ─────────────────────────────────────────────────────────────────────────────

print("\n🔁 執行回測...")
strategies = {
    "S1-proxy (E12 BB Squeeze Break)": (E12_BB_Squeeze_Break, "S1"),
    "S2B-proxy (E18 Hammer)":          (E18_Hammer,           "S2"),
    "S2A-proxy (E06 RSI Oversold)":    (E06_RSI_Oversold,     "S2"),
}

results = {}
for name, (fn, stype) in strategies.items():
    # Baseline：原始引擎，無過濾
    base_trades = run_backtest(price_30m_sorted, fn)
    base_tagged = tag_trades_with_macro(base_trades, price_30m_sorted)

    # Filtered：包裝信號函數後用相同引擎
    filt_fn     = make_filtered_signal(fn, price_30m_sorted, stype)
    filt_trades = run_backtest(price_30m_sorted, filt_fn)
    filt_tagged = tag_trades_with_macro(filt_trades, price_30m_sorted)

    base_sum = summarize(base_tagged)
    filt_sum = summarize(filt_tagged)

    # 依 verdict 分桶（用 baseline tagged）
    by_verdict = {}
    for v in ["STRONG BUY", "NEUTRAL", "WAIT", "N/A"]:
        sub = [x for x in base_tagged if x["verdict"] == v]
        by_verdict[v] = summarize(sub)

    # GVZ Extreme 子集
    gvz_ext = [x for x in base_tagged if x["gvz_state"] == "EXTREME"]

    removed = base_sum["n"] - filt_sum["n"]
    results[name] = {
        "baseline":    base_sum,
        "filtered":    filt_sum,
        "by_verdict":  by_verdict,
        "gvz_extreme": summarize(gvz_ext),
        "filter_type": stype,
        "removed":     removed,
    }
    print(f"  {name}")
    print(f"    Baseline  : {base_sum['n']} 筆  WR {base_sum['wr']:.1f}%  PF {base_sum['pf']:.3f}  Net {base_sum['net_pct']:+.2f}%")
    print(f"    Filtered  : {filt_sum['n']} 筆  WR {filt_sum['wr']:.1f}%  PF {filt_sum['pf']:.3f}  Net {filt_sum['net_pct']:+.2f}%")
    print(f"    移除      : {removed} 筆  ΔWR {filt_sum['wr']-base_sum['wr']:+.1f}%  ΔPF {filt_sum['pf']-base_sum['pf']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Macro 環境分佈（全回測期間）
# ─────────────────────────────────────────────────────────────────────────────

macro_in_range = macro_range.copy()
score_dist = macro_in_range["score"].value_counts().sort_index()
verdict_dist = macro_in_range["verdict"].value_counts()
gvz_dist = macro_in_range["gvz_state"].value_counts()

# 最新一日資料
latest = macro_daily.iloc[-1]

# ─────────────────────────────────────────────────────────────────────────────
# 7. 生成 HTML 報告
# ─────────────────────────────────────────────────────────────────────────────

def pct(v, f=".1f"):
    return f"+{v:{f}}%" if v >= 0 else f"{v:{f}}%"


def color_wr(wr):
    if wr >= 50: return "#059669"
    if wr >= 42: return "#d97706"
    return "#dc2626"


def verdict_badge(v):
    colors = {"STRONG BUY": "#059669", "NEUTRAL": "#d97706",
              "WAIT": "#dc2626", "N/A": "#94a3b8"}
    c = colors.get(v, "#94a3b8")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:12px;font-size:.8em;font-weight:700">{v}</span>'


def gvz_badge(g):
    colors = {"EXTREME": "#dc2626", "NORMAL": "#455a64", "SQUEEZE": "#1565c0"}
    c = colors.get(g, "#94a3b8")
    icons = {"EXTREME": "🔥", "NORMAL": "🌊", "SQUEEZE": "🧊"}
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:12px;font-size:.8em;font-weight:700">{icons.get(g,"")}{g}</span>'


def row(label, base, filt, is_wr=False, is_pf=False):
    delta_wr  = filt["wr"]  - base["wr"]
    delta_pf  = filt["pf"]  - base["pf"]
    delta_net = filt["net_pct"] - base["net_pct"]
    dwr_s  = f'<span style="color:{"#059669" if delta_wr>=0 else "#dc2626"};font-weight:700">{delta_wr:+.1f}%</span>'
    dpf_s  = f'<span style="color:{"#059669" if delta_pf>=0 else "#dc2626"};font-weight:700">{delta_pf:+.3f}</span>'
    dnet_s = f'<span style="color:{"#059669" if delta_net>=0 else "#dc2626"};font-weight:700">{pct(delta_net)}</span>'
    return f"""
      <tr>
        <td><strong>{label}</strong></td>
        <td>{base['n']}</td>
        <td style="color:{color_wr(base['wr'])};font-weight:700">{base['wr']:.1f}%</td>
        <td>{base['pf']:.3f}</td>
        <td>{pct(base['net_pct'])}</td>
        <td>{filt['n']}</td>
        <td style="color:{color_wr(filt['wr'])};font-weight:700">{filt['wr']:.1f}%</td>
        <td>{filt['pf']:.3f}</td>
        <td>{pct(filt['net_pct'])}</td>
        <td>{dwr_s}</td>
        <td>{dpf_s}</td>
        <td>{dnet_s}</td>
      </tr>"""


rows_html = ""
for name, r in results.items():
    rows_html += row(name, r["baseline"], r["filtered"])

# 按 Verdict 分桶表
verdict_rows = ""
for name, r in results.items():
    bv = r["by_verdict"]
    for v in ["STRONG BUY", "NEUTRAL", "WAIT"]:
        d = bv.get(v, {"n":0,"wins":0,"losses":0,"wr":0,"pf":0,"net_pct":0})
        if d["n"] > 0:
            verdict_rows += f"""
      <tr>
        <td>{name}</td>
        <td>{verdict_badge(v)}</td>
        <td>{d['n']}</td>
        <td style="color:{color_wr(d['wr'])};font-weight:700">{d['wr']:.1f}%</td>
        <td>{d['pf']:.3f}</td>
        <td>{pct(d['net_pct'])}</td>
      </tr>"""

# 最新宏觀狀態
latest_score = int(latest["score"]) if not pd.isna(latest.get("score", float("nan"))) else "N/A"
latest_verdict = latest.get("verdict", "N/A")

today_str = date.today().strftime("%Y-%m-%d")

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>XAUUSD Macro Backtest Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f4f8;color:#0f172a;font-size:14px;line-height:1.6}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px 20px}}
h1{{font-size:1.4em;font-weight:800;margin-bottom:4px}}
.subtitle{{color:#64748b;font-size:.88em;margin-bottom:24px}}
.card{{background:white;border-radius:10px;border:1px solid #e2e8f0;padding:20px 24px;margin-bottom:18px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.card h2{{font-size:.95em;font-weight:700;color:#334155;margin-bottom:14px;border-left:4px solid #2563eb;padding-left:10px}}
.grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}}
.metric{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px}}
.metric-label{{font-size:.75em;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px}}
.metric-val{{font-size:1.4em;font-weight:700}}
.metric-sub{{font-size:.78em;color:#94a3b8;margin-top:2px}}
.tbl-wrap{{overflow-x:auto;border-radius:8px;border:1px solid #e2e8f0}}
table{{border-collapse:collapse;width:100%;font-size:.83em}}
thead th{{background:#1e3a5f;color:white;padding:9px 12px;text-align:left;white-space:nowrap}}
tbody td{{padding:8px 12px;border-bottom:1px solid #f1f5f9}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#f8fafc}}
.insight{{border-radius:8px;padding:12px 16px;font-size:.86em;border-left:4px solid;line-height:1.5;margin-bottom:10px}}
.insight.good{{background:#d1fae5;border-color:#059669}}
.insight.bad{{background:#fee2e2;border-color:#dc2626}}
.insight.warn{{background:#fef3c7;border-color:#d97706}}
.insight.info{{background:#dbeafe;border-color:#2563eb}}
.insight strong{{display:block;font-weight:700;margin-bottom:3px}}
.footer{{text-align:center;color:#94a3b8;font-size:.75em;padding:24px;margin-top:8px}}
@media(max-width:700px){{.grid-3,.grid-4{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <h1>📊 XAUUSD Macro-Filtered Backtest</h1>
  <p class="subtitle">宏觀過濾器（Macro v3.7 評分邏輯）對 S1/S2 策略影響分析 | 生成：{today_str}</p>

  <!-- 最新宏觀狀態 -->
  <div class="card">
    <h2>🌐 最新宏觀環境（{latest['time'].date() if hasattr(latest['time'],'date') else latest['time']}）</h2>
    <div class="grid-4">
      <div class="metric">
        <div class="metric-label">Macro Score</div>
        <div class="metric-val" style="color:{'#059669' if latest_score>=5 else '#dc2626' if latest_score<=2 else '#d97706'}">{latest_score} / 6</div>
        <div class="metric-sub">{verdict_badge(latest_verdict)}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Real Rate</div>
        <div class="metric-val">{latest.get('real_rate', float('nan')):.3f}%</div>
        <div class="metric-sub">US10Y - T10YIE | {'↓ +2分' if latest.get('pt_real',0)>0 else '↑ 0分'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">DXY</div>
        <div class="metric-val">{latest.get('dxy', float('nan')):.2f}</div>
        <div class="metric-sub">{'↓ +1分' if latest.get('pt_dxy',0)>0 else '↑ 0分'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">GVZ</div>
        <div class="metric-val">{latest.get('gvz', float('nan')):.1f}</div>
        <div class="metric-sub">{gvz_badge(latest.get('gvz_state','N/A'))}</div>
      </div>
    </div>
    <div class="grid-4">
      <div class="metric">
        <div class="metric-label">US 10Y</div>
        <div class="metric-val">{latest.get('us10y', float('nan')):.3f}%</div>
        <div class="metric-sub">{'↓ +1分' if latest.get('pt_10y',0)>0 else '↑ 0分'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">VIX</div>
        <div class="metric-val">{latest.get('vix', float('nan')):.2f}</div>
        <div class="metric-sub">{'↑ +1分' if latest.get('pt_vix',0)>0 else '↓ 0分'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Gold vs MA50</div>
        <div class="metric-val">{'Above' if latest.get('pt_trend',0)>0 else 'Below'}</div>
        <div class="metric-sub">{'↑ +1分' if latest.get('pt_trend',0)>0 else '↓ 0分'}</div>
      </div>
      <div class="metric">
        <div class="metric-label">得分明細</div>
        <div class="metric-val" style="font-size:1em">RR:{int(latest.get('pt_real',0))} 10Y:{int(latest.get('pt_10y',0))} DXY:{int(latest.get('pt_dxy',0))} VIX:{int(latest.get('pt_vix',0))} Trend:{int(latest.get('pt_trend',0))}</div>
        <div class="metric-sub">總分 {latest_score}/6</div>
      </div>
    </div>
  </div>

  <!-- 回測期間環境分佈 -->
  <div class="card">
    <h2>📅 回測期間宏觀環境分佈（{date_min} → {date_max}）</h2>
    <div class="grid-3">
      {''.join(f"""<div class="metric">
        <div class="metric-label">{v}</div>
        <div class="metric-val" style="color:{'#059669' if v=='STRONG BUY' else '#dc2626' if v=='WAIT' else '#d97706'}">{verdict_dist.get(v,0)} 天</div>
        <div class="metric-sub">{verdict_dist.get(v,0)/len(macro_in_range)*100:.1f}% 占比</div>
      </div>""" for v in ["STRONG BUY","NEUTRAL","WAIT"])}
    </div>
    <div class="grid-3">
      {''.join(f"""<div class="metric">
        <div class="metric-label">GVZ {g}</div>
        <div class="metric-val">{gvz_dist.get(g,0)} 天</div>
        <div class="metric-sub">{gvz_dist.get(g,0)/len(macro_in_range)*100:.1f}%</div>
      </div>""" for g in ["SQUEEZE","NORMAL","EXTREME"])}
    </div>
  </div>

  <!-- 主比較表 -->
  <div class="card">
    <h2>⚖️ Baseline vs Macro-Filtered 比較</h2>
    <p style="font-size:.82em;color:#64748b;margin-bottom:12px">過濾規則：S1 在 WAIT 環境完全停止；S2 在 WAIT 環境仍執行（建議縮倉至 0.02 手）</p>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th rowspan="2">策略</th>
            <th colspan="4" style="text-align:center;border-right:1px solid rgba(255,255,255,.3)">Baseline（無過濾）</th>
            <th colspan="4" style="text-align:center;border-right:1px solid rgba(255,255,255,.3)">Macro-Filtered</th>
            <th colspan="3" style="text-align:center">△ 改變</th>
          </tr>
          <tr>
            <th>筆數</th><th>勝率</th><th>PF</th><th>淨盈虧%</th>
            <th>筆數</th><th>勝率</th><th>PF</th><th>淨盈虧%</th>
            <th>ΔWR</th><th>ΔPF</th><th>Δ淨盈虧</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- 按 Verdict 分桶 -->
  <div class="card">
    <h2>🔍 依 Macro Score 環境分析各策略勝率</h2>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr><th>策略</th><th>環境</th><th>筆數</th><th>勝率</th><th>PF</th><th>淨盈虧%</th></tr>
        </thead>
        <tbody>{verdict_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- 操作結論 -->
  <div class="card">
    <h2>💡 回測結論與關鍵發現</h2>

    <div class="insight bad">
      <strong>🚨 重要：此回測期間為黃金史詩行情，結論有強烈偏差</strong>
      回測期間（2026-01-21 → 2026-04-27）正值黃金從 $2800 暴漲至 $3500+ 的歷史級多頭，
      由「關稅戰 + 央行大量購金 + 去美元化」驅動，傳統 DXY−黃金反向關係出現 <strong>同漲</strong> 異常。
      此期間 GVZ 全程超過 20（100% Extreme），代表市場每天都在預期黃金大幅波動。
      建議匯出 <strong>2024-01 起的 30m 資料</strong>重跑，才能看到宏觀過濾在正常環境的效果。
    </div>

    <div class="insight warn">
      <strong>⚠️ 反常發現：WAIT 期間 E12 表現反而最好（WR 43.9%，PF 1.610）</strong>
      傳統邏輯：宏觀 WAIT（分數 ≤ 2）= 黃金逆風 = 策略應該表現差。
      實際結果：E12 在 WAIT 期間 Net +6.82%，STRONG BUY 期間 Net 0.00%。
      原因推斷：WAIT 期間（DXY 強 + 實質利率高）代表短線過度悲觀，反而是黃金突破反彈的節點；
      STRONG BUY 期間反而是前期大漲後的高位震盪，突破信號更多是假突破。
      <strong>此現象在正常市場環境中可能不成立，需要更長回測驗證。</strong>
    </div>

    <div class="insight info">
      <strong>📊 S2 策略代理結果不理想</strong>
      E18 Hammer（WR 15.0%）和 E06 RSI Oversold（WR 30.2%）在本期間均虧損。
      這可能反映：(1) 這兩個實驗策略不是 S2-Hammer/S2-RSI 的好代理；(2) 史詩牛市中左側反轉信號失效。
      S2 真正的進場條件（SSL Sweep + 錘頭 + SMC 確認）是人工辨識的，程式化代理難以完整複現。
    </div>

    <div class="insight good">
      <strong>✅ 宏觀過濾的真正價值：需要熊市/震盪期才能驗證</strong>
      宏觀 Score 的設計目的是「阻止在逆風環境執行低勝率交易」，效果在下跌或震盪市中最顯著。
      本期間的正面結論：宏觀 Score 計算邏輯完整，CSV 資料已就位，等待更長時間序列驗證。
      <strong>下一步：</strong>從 TradingView 匯出 2024-01-01 起的 XAUUSD 30m CSV 覆蓋現有資料，重跑此腳本即可得到跨市場環境的比較。
    </div>
  </div>

  <div class="footer">Generated by Claude Code {today_str} | Macro v3.7 Logic | python3 xauusd/scripts/run_macro_backtest.py</div>
</div>
</body>
</html>"""

out_path = OUT_DIR / "macro_backtest_report.html"
out_path.write_text(html, encoding="utf-8")
print(f"\n✅ 報告已輸出：{out_path}")

# 儲存 JSON 供 index.html 使用
json_out = OUT_DIR / "macro_backtest_results.json"
json_data = {
    "generated": today_str,
    "date_range": f"{date_min} → {date_max}",
    "latest_score": latest_score,
    "latest_verdict": latest_verdict,
    "verdict_dist": verdict_dist.to_dict(),
    "gvz_dist": gvz_dist.to_dict(),
    "results": {
        name: {
            "baseline": r["baseline"],
            "filtered": r["filtered"],
            "by_verdict": r["by_verdict"],
            "removed": r["removed"],
        }
        for name, r in results.items()
    }
}
json_out.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
print(f"✅ JSON 已輸出：{json_out}")
