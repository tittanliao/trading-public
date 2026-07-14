"""
XAUUSD Real Strategy Macro-Filtered Backtest
=============================================
用真實策略交易 CSV（S1-AweWithBB / S2-RSI / S2-Hammer）搭配
每日 Macro Score（v3.7 邏輯），分析宏觀過濾對各策略真實績效的影響。

執行（20260705 移至 scripts/ 子資料夾，仍需在 trading/ 根目錄執行）：python3.12 xauusd/scripts/run_real_strategy_macro_backtest.py
輸出：xauusd/XAUUSD-Macro/real_macro_backtest_report.html
      xauusd/XAUUSD-Macro/real_macro_backtest_results.json
"""
from __future__ import annotations

import os, sys, json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent  # 20260705 移至 scripts/ 子資料夾，多一層 .parent
CSV_DIR = ROOT / "csv"
OUT_DIR = ROOT / "XAUUSD-Macro"
OUT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
MA_LEN     = 50
GVZ_EXT_TH = 20.0
GVZ_SQZ_TH = 13.0

STRATEGIES = {
    "S1-AweWithBB": {
        "csv": ROOT / "XAUUSD-Long-S1-AweWithBB" / "S1-Awe-V3.4_FX_IDC_XAUUSD_2026-04-26.csv",
        "type": "S1",
        "version": "V3.4",
        "win_signal": ["TP"],   # 含 TP 的 exit signal = 勝
        "loss_signal": ["SL"],  # 含 SL = 敗
    },
    "S2-RSI": {
        "csv": ROOT / "XAUUSD-Long-S2-RSI" / "S2-Hybrid-V2.0_FX_IDC_XAUUSD_2026-04-26.csv",
        "type": "S2",
        "version": "V2.0",
        "win_signal": ["TP"],
        "loss_signal": ["SL"],
    },
    "S2-Hammer": {
        "csv": ROOT / "XAUUSD-Long-S2-Hammer" / "S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-04-26.csv",
        "type": "S2",
        "version": "V1.9",
        "win_signal": ["TP"],
        "loss_signal": ["SL"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. 載入 Macro CSV
# ─────────────────────────────────────────────────────────────────────────────

def _load(fname: str, close_only=False) -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / fname)
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    t_col = [c for c in df.columns if "time" in c.lower()][0]
    df[t_col] = pd.to_datetime(df[t_col], utc=False).dt.tz_localize(None)
    df = df.rename(columns={t_col: "time"}).sort_values("time").reset_index(drop=True)
    if close_only:
        return df[["time", "close"]]
    return df[["time", "close"]]  # macro 只需要日收盤


print("📂 載入宏觀 CSV...")
us10y_1d  = _load("TVC_US10Y, 1D.csv")
t10yie_1d = _load("FRED_T10YIE, 1D.csv")
dxy_1d    = _load("TVC_DXY, 1D.csv")
vix_1d    = _load("TVC_VIX, 1D.csv")
gvz_1d    = _load("CBOE_GVZ, 1D.csv")
gold_1d   = _load("FX_IDC_XAUUSD, 1D.csv")

for label, df in [("US10Y", us10y_1d), ("T10YIE", t10yie_1d), ("DXY", dxy_1d),
                   ("VIX", vix_1d), ("GVZ", gvz_1d), ("Gold 1D", gold_1d)]:
    print(f"  {label}: {len(df)} bars  {df['time'].min().date()} → {df['time'].max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 計算每日 Macro Score（同 Pine Script v3.7）
# ─────────────────────────────────────────────────────────────────────────────

def ma50(s: pd.Series) -> pd.Series:
    return s.rolling(MA_LEN, min_periods=MA_LEN).mean()


def build_macro_scores() -> pd.DataFrame:
    base = us10y_1d.rename(columns={"close": "us10y"})
    for src, col in [(t10yie_1d, "t10yie"), (dxy_1d, "dxy"),
                     (vix_1d, "vix"), (gvz_1d, "gvz"), (gold_1d, "gold")]:
        tmp = src.rename(columns={"close": col})
        base = pd.merge_asof(base.sort_values("time"),
                             tmp.sort_values("time"),
                             on="time", direction="backward")

    base["real_rate"]  = base["us10y"] - base["t10yie"]
    base["ma50_rr"]    = ma50(base["real_rate"])
    base["ma50_10y"]   = ma50(base["us10y"])
    base["ma50_dxy"]   = ma50(base["dxy"])
    base["ma50_vix"]   = ma50(base["vix"])
    base["ma50_gold"]  = ma50(base["gold"])

    base["pt_real"]  = np.where(base["real_rate"] < base["ma50_rr"],   2, 0)
    base["pt_10y"]   = np.where(base["us10y"]     < base["ma50_10y"],  1, 0)
    base["pt_dxy"]   = np.where(base["dxy"]       < base["ma50_dxy"],  1, 0)
    base["pt_vix"]   = np.where(base["vix"]       > base["ma50_vix"],  1, 0)
    base["pt_trend"] = np.where(base["gold"]      > base["ma50_gold"], 1, 0)
    base["score"]    = base[["pt_real","pt_10y","pt_dxy","pt_vix","pt_trend"]].sum(axis=1)

    valid = base[["ma50_rr","ma50_10y","ma50_dxy","ma50_vix","ma50_gold"]].notna().all(axis=1)
    base.loc[~valid, "score"] = np.nan

    base["verdict"] = base["score"].apply(
        lambda s: ("STRONG BUY" if s >= 5 else "WAIT" if s <= 2 else "NEUTRAL")
        if not pd.isna(s) else "N/A"
    )
    base["gvz_state"] = base["gvz"].apply(
        lambda g: ("EXTREME" if g > GVZ_EXT_TH else "SQUEEZE" if g < GVZ_SQZ_TH else "NORMAL")
        if not pd.isna(g) else "N/A"
    )
    return base.dropna(subset=["score"]).reset_index(drop=True)


print("\n⚙️  計算 Macro Score...")
macro_daily = build_macro_scores()
print(f"  有效日數: {len(macro_daily)} 天  "
      f"({macro_daily['time'].min().date()} → {macro_daily['time'].max().date()})")

# 建立 asof 查找用的 Series（indexed by timestamp）
macro_lookup = macro_daily.set_index("time")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 載入策略交易 CSV → 每筆交易配上 Macro 資訊
# ─────────────────────────────────────────────────────────────────────────────

def _is_win(signal: str, win_kws: list[str]) -> bool | None:
    """None = 跳過（Open / 無法分類）"""
    s = str(signal).upper()
    if any(k.upper() in s for k in win_kws):
        return True
    if "SL" in s:
        return False
    return None


def load_trades(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["csv"])
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["Date and time"])

    entries = df[df["Type"] == "Entry long"][["Trade #","datetime","Price USD"]].rename(
        columns={"datetime": "entry_time", "Price USD": "entry_price"})
    exits   = df[df["Type"] == "Exit long"][["Trade #","datetime","Price USD","Net P&L %","Signal"]].rename(
        columns={"datetime": "exit_time", "Price USD": "exit_price",
                 "Net P&L %": "net_pct", "Signal": "exit_signal"})

    merged = entries.merge(exits, on="Trade #", how="inner")

    # 勝負分類
    merged["outcome"] = merged["exit_signal"].apply(
        lambda s: _is_win(s, cfg["win_signal"])
    )
    merged = merged[merged["outcome"].notna()].copy()
    merged["result"] = merged["outcome"].map({True: "win", False: "loss"})

    # 持倉 bars（30 分鐘 = 0.5 小時）
    merged["hold_bars"] = (merged["exit_time"] - merged["entry_time"]).dt.total_seconds() / 1800

    # 對應 Macro 分數（用進場日前一天收盤的 Macro）
    entry_ts = merged["entry_time"].values
    verdicts, gvz_states, scores = [], [], []
    pt_reals, pt_10ys, pt_dxys, pt_vixs, pt_trends = [], [], [], [], []

    for et in entry_ts:
        ts = pd.Timestamp(et)
        try:
            row = macro_lookup.asof(ts - pd.Timedelta(hours=1))  # 取進場前最近的宏觀日
            verdicts.append(row["verdict"] if not pd.isna(row.get("verdict","")) else "N/A")
            gvz_states.append(row["gvz_state"] if not pd.isna(row.get("gvz_state","")) else "N/A")
            scores.append(float(row["score"]) if not pd.isna(row.get("score", np.nan)) else -1)
            pt_reals.append(int(row.get("pt_real", 0)))
            pt_10ys.append(int(row.get("pt_10y", 0)))
            pt_dxys.append(int(row.get("pt_dxy", 0)))
            pt_vixs.append(int(row.get("pt_vix", 0)))
            pt_trends.append(int(row.get("pt_trend", 0)))
        except Exception:
            verdicts.append("N/A"); gvz_states.append("N/A"); scores.append(-1)
            pt_reals.append(0); pt_10ys.append(0); pt_dxys.append(0)
            pt_vixs.append(0); pt_trends.append(0)

    merged["verdict"]   = verdicts
    merged["gvz_state"] = gvz_states
    merged["score"]     = scores
    merged["pt_real"]   = pt_reals
    merged["pt_10y"]    = pt_10ys
    merged["pt_dxy"]    = pt_dxys
    merged["pt_vix"]    = pt_vixs
    merged["pt_trend"]  = pt_trends
    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 統計函數
# ─────────────────────────────────────────────────────────────────────────────

VERDICTS = ["STRONG BUY", "NEUTRAL", "WAIT"]
GVZ_STATES = ["SQUEEZE", "NORMAL", "EXTREME"]

def stats(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0, "wins": 0, "losses": 0, "wr": 0.0, "pf": 0.0, "net_pct": 0.0, "avg_hold": 0.0}
    wins   = df[df["result"] == "win"]
    losses = df[df["result"] == "loss"]
    gw = wins["net_pct"].sum()
    gl = abs(losses["net_pct"].sum())
    return {
        "n":        len(df),
        "wins":     len(wins),
        "losses":   len(losses),
        "wr":       len(wins) / len(df) * 100,
        "pf":       gw / gl if gl > 0 else 999.0,
        "net_pct":  df["net_pct"].sum(),
        "avg_hold": df["hold_bars"].mean(),
    }


def analyze(df: pd.DataFrame, stype: str) -> dict:
    baseline = stats(df)

    # S1: 過濾掉 WAIT；S2: 全部執行（縮倉由風控決定）
    if stype == "S1":
        filtered_df = df[df["verdict"] != "WAIT"]
    else:
        filtered_df = df

    filtered = stats(filtered_df)

    by_verdict = {v: stats(df[df["verdict"] == v]) for v in VERDICTS}
    by_gvz     = {g: stats(df[df["gvz_state"] == g]) for g in GVZ_STATES}

    # 按 score bucket（0-1 / 2 / 3-4 / 5-6）
    score_buckets = {
        "0-1 (WAIT深度)":  stats(df[df["score"].between(0, 1)]),
        "2 (WAIT邊界)":    stats(df[df["score"] == 2]),
        "3-4 (NEUTRAL)":   stats(df[df["score"].between(3, 4)]),
        "5-6 (STRONG BUY)": stats(df[df["score"].between(5, 6)]),
    }

    return {
        "baseline": baseline,
        "filtered": filtered,
        "by_verdict": by_verdict,
        "by_gvz": by_gvz,
        "by_score_bucket": score_buckets,
        "removed": baseline["n"] - filtered["n"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. 執行
# ─────────────────────────────────────────────────────────────────────────────

print("\n🔁 分析各策略...")
all_results = {}
all_trades  = {}

for name, cfg in STRATEGIES.items():
    trades_df = load_trades(cfg)
    result    = analyze(trades_df, cfg["type"])
    all_results[name] = result
    all_trades[name]  = trades_df
    b = result["baseline"]
    f = result["filtered"]
    print(f"\n  {name} ({cfg['version']})")
    print(f"    Baseline : {b['n']:3d} 筆  WR {b['wr']:.1f}%  PF {b['pf']:.3f}  Net {b['net_pct']:+.2f}%")
    print(f"    Filtered : {f['n']:3d} 筆  WR {f['wr']:.1f}%  PF {f['pf']:.3f}  Net {f['net_pct']:+.2f}%")
    print(f"    移除     : {result['removed']} 筆 (WAIT 期 S1 過濾)" if cfg["type"]=="S1" else
          f"    移除     : {result['removed']} 筆 (S2 不過濾)")
    print(f"    By Verdict:")
    for v in VERDICTS:
        bv = result["by_verdict"][v]
        if bv["n"] > 0:
            print(f"      {v:12s}: {bv['n']:3d} 筆  WR {bv['wr']:.1f}%  PF {bv['pf']:.3f}  Net {bv['net_pct']:+.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 儲存 JSON
# ─────────────────────────────────────────────────────────────────────────────

def to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_serializable(i) for i in obj]
    return obj


json_out = {
    "generated": str(date.today()),
    "date_range": f"{macro_daily['time'].min().date()} → {macro_daily['time'].max().date()}",
    "results": to_serializable(all_results),
}
with open(OUT_DIR / "real_macro_backtest_results.json", "w", encoding="utf-8") as f:
    json.dump(json_out, f, ensure_ascii=False, indent=2)
print(f"\n✅ JSON: {OUT_DIR / 'real_macro_backtest_results.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. HTML 報告
# ─────────────────────────────────────────────────────────────────────────────

def badge(v: str) -> str:
    colors = {
        "WAIT": ("#ef4444","white"), "NEUTRAL": ("#f59e0b","white"),
        "STRONG BUY": ("#22c55e","white"),
        "EXTREME": ("#ef4444","white"), "NORMAL": ("#3b82f6","white"),
        "SQUEEZE": ("#8b5cf6","white"),
    }
    bg, fg = colors.get(v, ("#6b7280","white"))
    return f"<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-size:.8em;font-weight:700'>{v}</span>"


def verdict_row(v: str, bv: dict) -> str:
    if bv["n"] == 0:
        return f"<tr><td>{badge(v)}</td><td colspan='5' style='color:#6b7280;font-style:italic'>無交易</td></tr>"
    wr_color = "#22c55e" if bv["wr"] >= 50 else ("#f59e0b" if bv["wr"] >= 40 else "#ef4444")
    pf_color = "#22c55e" if bv["pf"] >= 1.3 else ("#f59e0b" if bv["pf"] >= 1.0 else "#ef4444")
    return (f"<tr><td>{badge(v)}</td><td>{bv['n']}</td>"
            f"<td style='color:{wr_color};font-weight:700'>{bv['wr']:.1f}%</td>"
            f"<td style='color:{pf_color};font-weight:700'>{bv['pf']:.3f}</td>"
            f"<td style='color:{'#22c55e' if bv['net_pct']>0 else '#ef4444'};font-weight:700'>"
            f"{bv['net_pct']:+.2f}%</td>"
            f"<td>{bv['avg_hold']:.1f} bars</td></tr>")


def strategy_section(name: str, r: dict, cfg: dict) -> str:
    b  = r["baseline"]
    f  = r["filtered"]
    bv = r["by_verdict"]
    bg = r["by_gvz"]
    bs = r["by_score_bucket"]
    stype = cfg["type"]

    # Baseline vs Filtered 對比
    def stat_td(s: dict, ref: dict | None = None) -> str:
        wr_c = "#22c55e" if s["wr"] >= 50 else ("#f59e0b" if s["wr"] >= 40 else "#ef4444")
        pf_c = "#22c55e" if s["pf"] >= 1.3 else ("#f59e0b" if s["pf"] >= 1.0 else "#ef4444")
        nl_c = "#22c55e" if s["net_pct"] > 0 else "#ef4444"
        cells = (f"<td>{s['n']}</td>"
                 f"<td style='color:{wr_c};font-weight:700'>{s['wr']:.1f}%</td>"
                 f"<td style='color:{pf_c};font-weight:700'>{s['pf']:.3f}</td>"
                 f"<td style='color:{nl_c};font-weight:700'>{s['net_pct']:+.2f}%</td>"
                 f"<td>{s['avg_hold']:.1f} bars</td>")
        if ref:
            dwr = s["wr"] - ref["wr"]
            dpf = s["pf"] - ref["pf"]
            dn  = s["net_pct"] - ref["net_pct"]
            cells += (f"<td style='color:{'#22c55e' if dwr>0 else '#ef4444'}'>{dwr:+.1f}%</td>"
                      f"<td style='color:{'#22c55e' if dpf>0 else '#ef4444'}'>{dpf:+.3f}</td>"
                      f"<td style='color:{'#22c55e' if dn>0 else '#ef4444'}'>{dn:+.2f}%</td>")
        return cells

    filter_note = (f"排除 WAIT 期間（{r['removed']} 筆），僅保留 NEUTRAL + STRONG BUY"
                   if stype == "S1" else "S2 策略不套用 WAIT 過濾（全數執行）")

    # Score bucket 欄
    bs_rows = ""
    for label, s in bs.items():
        if s["n"] == 0:
            bs_rows += f"<tr><td>{label}</td><td colspan='5' style='color:#6b7280'>無交易</td></tr>"
            continue
        wr_c = "#22c55e" if s["wr"] >= 50 else ("#f59e0b" if s["wr"] >= 40 else "#ef4444")
        pf_c = "#22c55e" if s["pf"] >= 1.3 else ("#f59e0b" if s["pf"] >= 1.0 else "#ef4444")
        bs_rows += (f"<tr><td>{label}</td><td>{s['n']}</td>"
                    f"<td style='color:{wr_c};font-weight:700'>{s['wr']:.1f}%</td>"
                    f"<td style='color:{pf_c};font-weight:700'>{s['pf']:.3f}</td>"
                    f"<td style='color:{'#22c55e' if s['net_pct']>0 else '#ef4444'};font-weight:700'>"
                    f"{s['net_pct']:+.2f}%</td>"
                    f"<td>{s['avg_hold']:.1f}</td></tr>")

    return f"""
    <div class="card">
      <h2 style="color:var(--primary)">{name} <span style="font-weight:400;font-size:.85em;color:#6b7280">({cfg["version"]})</span></h2>

      <h3 class="section-title">Baseline vs 過濾後</h3>
      <p style="font-size:.85em;color:#6b7280;margin:0 0 8px">{filter_note}</p>
      <table class="data-table">
        <thead><tr><th>條件</th><th>交易筆</th><th>勝率</th><th>獲利因子</th>
          <th>淨盈虧%</th><th>平均持倉</th><th>ΔWR</th><th>ΔPF</th><th>Δ淨盈虧</th></tr></thead>
        <tbody>
          <tr><td><strong>Baseline（無過濾）</strong></td>{stat_td(b)}<td>—</td><td>—</td><td>—</td></tr>
          <tr><td><strong>Filtered（宏觀過濾）</strong></td>{stat_td(f, b)}</tr>
        </tbody>
      </table>

      <h3 class="section-title" style="margin-top:20px">按宏觀 Verdict 分組</h3>
      <table class="data-table">
        <thead><tr><th>Verdict</th><th>交易筆</th><th>勝率</th><th>獲利因子</th><th>淨盈虧%</th><th>平均持倉</th></tr></thead>
        <tbody>
          {"".join(verdict_row(v, bv[v]) for v in VERDICTS)}
        </tbody>
      </table>

      <h3 class="section-title" style="margin-top:20px">按 GVZ 狀態分組</h3>
      <table class="data-table">
        <thead><tr><th>GVZ</th><th>交易筆</th><th>勝率</th><th>獲利因子</th><th>淨盈虧%</th><th>平均持倉</th></tr></thead>
        <tbody>
          {"".join(verdict_row(g, bg[g]) for g in GVZ_STATES)}
        </tbody>
      </table>

      <h3 class="section-title" style="margin-top:20px">按 Score 分桶（0–6）</h3>
      <table class="data-table">
        <thead><tr><th>Score 區間</th><th>交易筆</th><th>勝率</th><th>獲利因子</th><th>淨盈虧%</th><th>平均持倉 bars</th></tr></thead>
        <tbody>{bs_rows}</tbody>
      </table>
    </div>"""


sections = "\n".join(
    strategy_section(name, all_results[name], STRATEGIES[name])
    for name in STRATEGIES
)

# 整體宏觀環境分佈（用最寬的日期範圍，也就是所有策略交易所覆蓋的範圍）
trade_date_min = min(df["entry_time"].min() for df in all_trades.values())
trade_date_max = max(df["entry_time"].max() for df in all_trades.values())
macro_in_range = macro_daily[
    (macro_daily["time"] >= trade_date_min) &
    (macro_daily["time"] <= trade_date_max)
]
vd = macro_in_range["verdict"].value_counts()
gd = macro_in_range["gvz_state"].value_counts()

def dist_badge(counts: pd.Series, keys: list, colors: dict) -> str:
    total = counts.sum()
    parts = []
    for k in keys:
        n = counts.get(k, 0)
        bg, fg = colors.get(k, ("#6b7280","white"))
        pct = n / total * 100 if total > 0 else 0
        parts.append(f"<span style='background:{bg};color:{fg};padding:3px 10px;border-radius:12px;"
                     f"font-size:.82em;font-weight:700'>{k}: {n}天 ({pct:.0f}%)</span>")
    return " ".join(parts)

VERDICT_COLORS = {"WAIT": ("#ef4444","white"), "NEUTRAL": ("#f59e0b","white"), "STRONG BUY": ("#22c55e","white")}
GVZ_COLORS     = {"SQUEEZE": ("#8b5cf6","white"), "NORMAL": ("#3b82f6","white"), "EXTREME": ("#ef4444","white")}

# ── 關鍵洞察產生器 ────────────────────────────────────────────────────────────
def key_insights() -> str:
    lines = []
    for name, r in all_results.items():
        bv = r["by_verdict"]
        stype = STRATEGIES[name]["type"]

        # 找勝率最高的 verdict
        best_v = max((v for v in VERDICTS if bv[v]["n"] >= 5),
                     key=lambda v: bv[v]["wr"], default=None)
        worst_v = min((v for v in VERDICTS if bv[v]["n"] >= 5),
                      key=lambda v: bv[v]["wr"], default=None)

        if best_v and worst_v and best_v != worst_v:
            best  = bv[best_v]
            worst = bv[worst_v]
            delta = best["wr"] - worst["wr"]
            lines.append(
                f"<li><strong>{name}</strong>：最佳宏觀環境 = {badge(best_v)} "
                f"WR {best['wr']:.1f}% / PF {best['pf']:.3f}；"
                f"最差 = {badge(worst_v)} WR {worst['wr']:.1f}% / PF {worst['pf']:.3f}。"
                f"最佳 vs 最差 ΔWR = <strong>{delta:+.1f}%</strong></li>"
            )

        # S1 filter impact
        if stype == "S1" and r["removed"] > 0:
            b, f = r["baseline"], r["filtered"]
            dwr = f["wr"] - b["wr"]
            dpf = f["pf"] - b["pf"]
            dn  = f["net_pct"] - b["net_pct"]
            color = "#22c55e" if dwr > 0 else "#ef4444"
            lines.append(
                f"<li><strong>{name} WAIT 過濾效果</strong>：移除 {r['removed']} 筆後 "
                f"ΔWR = <span style='color:{color};font-weight:700'>{dwr:+.1f}%</span>，"
                f"ΔPF = <span style='color:{color};font-weight:700'>{dpf:+.3f}</span>，"
                f"Δ淨盈虧 = <span style='color:{color};font-weight:700'>{dn:+.2f}%</span>"
                f"{'（過濾有效）' if dwr > 0 else '（過濾反效果，此段宏觀逆向）'}</li>"
            )
    return "<ul style='margin:0;padding-left:18px;line-height:1.8'>" + "\n".join(lines) + "</ul>"


html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>XAUUSD Real Strategy × Macro Backtest</title>
<style>
  :root {{
    --bg:#0f172a; --card:#1e293b; --border:#334155;
    --text:#e2e8f0; --text2:#94a3b8; --primary:#60a5fa;
    --red:#ef4444; --green:#22c55e; --yellow:#f59e0b;
  }}
  body {{ margin:0; padding:20px; font-family:'Segoe UI',system-ui,sans-serif;
         background:var(--bg); color:var(--text); }}
  h1   {{ color:var(--primary); margin:0 0 4px; }}
  h2   {{ color:var(--text); margin:0 0 12px; }}
  h3.section-title {{ color:var(--text2); font-size:.85em; text-transform:uppercase;
                      letter-spacing:.06em; margin:16px 0 6px; border-bottom:1px solid var(--border);
                      padding-bottom:4px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
           padding:20px; margin-bottom:20px; }}
  .meta  {{ color:var(--text2); font-size:.85em; margin-bottom:16px; }}
  .data-table {{ width:100%; border-collapse:collapse; font-size:.87em; }}
  .data-table th {{ background:#0f172a; color:var(--text2); font-weight:600;
                    padding:8px 10px; text-align:left; border-bottom:2px solid var(--border); }}
  .data-table td {{ padding:7px 10px; border-bottom:1px solid var(--border); }}
  .data-table tr:hover td {{ background:rgba(255,255,255,.04); }}
  .insight {{ padding:10px 14px; border-radius:8px; margin:8px 0; font-size:.88em; }}
  .insight.warn {{ background:rgba(245,158,11,.12); border-left:3px solid var(--yellow); }}
  .insight.good {{ background:rgba(34,197,94,.10); border-left:3px solid var(--green); }}
  .insight.bad  {{ background:rgba(239,68,68,.10); border-left:3px solid var(--red); }}
  .insight.info {{ background:rgba(96,165,250,.10); border-left:3px solid var(--primary); }}
  .dist {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }}
</style>
</head>
<body>
<h1>XAUUSD 真實策略 × Macro Score 回測</h1>
<div class="meta">
  生成時間：{date.today()} ｜
  策略資料涵蓋：{trade_date_min.date()} → {trade_date_max.date()} ｜
  宏觀資料：2012 起（US10Y / T10YIE / DXY / VIX / GVZ）
</div>

<div class="card">
  <h2>宏觀環境分佈（策略交易期間）</h2>
  <p style="color:var(--text2);font-size:.85em;margin:0 0 8px">
    涵蓋期間：{trade_date_min.date()} → {trade_date_max.date()}，共 {len(macro_in_range)} 個有效日
  </p>
  <div class="dist">{dist_badge(vd, VERDICTS, VERDICT_COLORS)}</div>
  <div class="dist">{dist_badge(gd, GVZ_STATES, GVZ_COLORS)}</div>
  <div class="insight info" style="margin-top:10px">
    <strong>ℹ️ 說明</strong>
    Macro Score = Real Rate vs MA50（×2 pt）+ US10Y（+1）+ DXY（+1）+ VIX（+1）+ Gold vs MA50（+1），總分 0–6。
    WAIT ≤ 2 / NEUTRAL 3–4 / STRONG BUY ≥ 5。S1 策略套用 WAIT 過濾，S2 策略全部執行（縮倉由風控決定）。
  </div>
</div>

<div class="card">
  <h2>🔑 關鍵洞察</h2>
  {key_insights()}
</div>

{sections}

<div class="card">
  <h2>💡 操作結論</h2>
  <div class="insight warn">
    <strong>⚠️ 樣本覆蓋說明</strong>
    本回測使用真實策略交易（TradingView Pine Script 回測匯出），非程式化信號重建。
    因此策略本身的訊號邏輯（錘頭型態、RSI 背離、AweWithBB 動能）已準確反映。
    宏觀分數使用每日宏觀資料（日線收盤前向映射），不引入未來資料。
  </div>
  <div class="insight info">
    <strong>📌 S1（AweWithBB）操作原則</strong>
    宏觀 WAIT 期交易表現較差時，WAIT 過濾可提升勝率，建議嚴格執行「WAIT 不做 S1」。
    若 WAIT 期表現反而好（如 2026 Q1 牛市），代表宏觀邏輯被強勢趨勢覆蓋，仍應謹慎縮倉觀察。
  </div>
  <div class="insight info">
    <strong>📌 S2（S2-RSI / S2-Hammer）操作原則</strong>
    S2 屬左側逆勢，宏觀 WAIT 時建議縮至 0.02 手（風控決定），不完全停止。
    若 STRONG BUY 期 S2 勝率明顯高於 WAIT 期，代表宏觀支撐確實提升 S2 的成功率。
  </div>
  <div class="insight good">
    <strong>✅ 資料已就位，可隨時更新</strong>
    更新 xauusd/csv/ 下的 30m XAUUSD CSV 後，重跑此腳本即可得到最新回測結果。
    建議每季更新一次並對比宏觀過濾效果變化。
  </div>
</div>

</body>
</html>
"""

html_path = OUT_DIR / "real_macro_backtest_report.html"
html_path.write_text(html, encoding="utf-8")
print(f"✅ HTML : {html_path}")
