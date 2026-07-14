#!/usr/bin/env python3
"""
驗證交易筆記的規則是否與 CSV 歷史資料一致。
分析項目：
  XAUUSD — 黃金秘笈：月份強弱、季節規律
  XAUUSD — 黃金短線：四個時段特性（趨勢 vs 震盪）
  TX     — 指數密技：月份強弱、季度規律
"""
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ─── Loader ──────────────────────────────────────────────────────────
def load_csv(path: Path, tz: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if tz:
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    else:
        df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


# ════════════════════════════════════════════════════════════════════
# XAUUSD 分析
# ════════════════════════════════════════════════════════════════════
def analyze_xauusd():
    results = {}

    # ── 1. 月份強弱（日線 2014–2026）─────────────────────────────
    daily = load_csv(ROOT / "xauusd/csv/FX_IDC_XAUUSD, 1D.csv", tz=False)
    daily["ret"] = daily["close"].pct_change()
    daily["month"] = daily.index.month
    daily["year"]  = daily.index.year

    # 以月收益率（月初→月末）統計
    monthly = (daily["close"]
               .resample("ME").last()
               .pct_change()
               .dropna())
    monthly_df = monthly.to_frame("ret")
    monthly_df["month"] = monthly_df.index.month
    monthly_df["year"]  = monthly_df.index.year

    month_stats = (monthly_df.groupby("month")["ret"]
                   .agg(n="count", win_rate=lambda x: (x > 0).mean(),
                        avg_ret="mean", median_ret="median")
                   .round(4))

    # 筆記強勢月：6,8,1,12,3,10,7；弱勢月：5,2,9,11,4
    note_strong = [6, 8, 1, 12, 3, 10, 7]
    note_weak   = [5, 2, 9, 11, 4]
    month_stats["note_view"] = month_stats.index.map(
        lambda m: "強" if m in note_strong else ("弱" if m in note_weak else "?"))
    month_stats["data_view"] = month_stats["win_rate"].map(
        lambda w: "強" if w >= 0.55 else ("弱" if w <= 0.45 else "中"))
    month_stats["match"] = month_stats["note_view"] == month_stats["data_view"]
    results["xauusd_monthly"] = month_stats.reset_index()

    # ── 2. 季度統計 ───────────────────────────────────────────────
    monthly_df["quarter"] = ((monthly_df["month"] - 1) // 3) + 1
    q_stats = (monthly_df.groupby("quarter")["ret"]
               .agg(n="count", win_rate=lambda x: (x > 0).mean(), avg_ret="mean")
               .round(4))
    results["xauusd_quarterly"] = q_stats.reset_index()

    # ── 3. 四個時段特性（30m 日內）────────────────────────────────
    # 早 09:00-10:00, 下午 14:30-15:30, 晚 20:30-21:30, 半夜 23:30-00:30
    m30 = load_csv(ROOT / "xauusd/csv/FX_IDC_XAUUSD, 30.csv", tz=True)
    m30["hour"] = m30.index.hour
    m30["ret"]  = (m30["close"] - m30["open"]) / m30["open"] * 100
    m30["abs_ret"] = m30["ret"].abs()

    sessions = {
        "早上 9-10":    (m30["hour"].isin([9])),
        "下午 14-15":   (m30["hour"].isin([14])),
        "晚上 20-21":   (m30["hour"].isin([20])),
        "半夜 23-00":   (m30["hour"].isin([23])),
    }
    session_stats = []
    for name, mask in sessions.items():
        sub = m30[mask]
        # 趨勢判定：若每根絕對漲幅 > 0.15% 視為趨勢 K 棒
        trend_pct = (sub["abs_ret"] > 0.15).mean()
        session_stats.append({
            "時段": name,
            "樣本數": len(sub),
            "平均波動%": round(sub["abs_ret"].mean(), 3),
            "趨勢K比例": round(trend_pct, 3),
            "漲K勝率": round((sub["ret"] > 0).mean(), 3),
            "筆記判斷": {
                "早上 9-10": "90%震盪",
                "下午 14-15": "70%震盪",
                "晚上 20-21": "50/50",
                "半夜 23-00": "偏震盪(除非>20$)",
            }[name],
        })
    results["xauusd_sessions"] = session_stats

    # ── 4. 週次強弱（黃金秘笈：第1、3週最強）────────────────────
    # 以每月1號起算第幾週
    daily2 = daily.copy()
    daily2["month"] = daily2.index.month
    daily2["year"]  = daily2.index.year
    daily2["dom"]   = daily2.index.day
    daily2["week_of_month"] = daily2["dom"].apply(lambda d: min((d - 1) // 7 + 1, 4))
    daily2["abs_ret"] = daily2["ret"].abs()
    wom_stats = (daily2.groupby("week_of_month")["abs_ret"]
                 .agg(n="count", avg_move="mean")
                 .round(4))
    results["xauusd_week_of_month"] = wom_stats.reset_index()

    return results


# ════════════════════════════════════════════════════════════════════
# TX 分析
# ════════════════════════════════════════════════════════════════════
def analyze_tx():
    results = {}

    # ── 1. 月份強弱（日線 2012–2026）─────────────────────────────
    daily = load_csv(ROOT / "tx/csv/TAIFEX_DLY_MXF1!, 1D.csv", tz=False)
    daily["ret"] = daily["close"].pct_change()
    daily["month"] = daily.index.month
    daily["year"]  = daily.index.year

    monthly = (daily["close"]
               .resample("ME").last()
               .pct_change()
               .dropna())
    monthly_df = monthly.to_frame("ret")
    monthly_df["month"] = monthly_df.index.month
    monthly_df["year"]  = monthly_df.index.year

    month_stats = (monthly_df.groupby("month")["ret"]
                   .agg(n="count", win_rate=lambda x: (x > 0).mean(),
                        avg_ret="mean", median_ret="median")
                   .round(4))

    # 指數密技：五月偏弱；一月通常上漲；十二月幾乎必漲
    note_strong = [1, 12, 4, 11]    # 一月、十二月、四月、十一月
    note_weak   = [5, 8, 9]         # 五月弱、八/九月偏弱
    month_stats["note_view"] = month_stats.index.map(
        lambda m: "強" if m in note_strong else ("弱" if m in note_weak else "中"))
    month_stats["data_view"] = month_stats["win_rate"].map(
        lambda w: "強" if w >= 0.55 else ("弱" if w <= 0.45 else "中"))
    month_stats["match"] = month_stats["note_view"] == month_stats["data_view"]
    results["tx_monthly"] = month_stats.reset_index()

    # ── 2. 季度統計（指數密技：Q4 幾乎必有多單機會）──────────────
    monthly_df["quarter"] = ((monthly_df["month"] - 1) // 3) + 1
    q_stats = (monthly_df.groupby("quarter")["ret"]
               .agg(n="count", win_rate=lambda x: (x > 0).mean(), avg_ret="mean",
                    pos_years=lambda x: (x > 0).sum())
               .round(4))

    # 找每季最大漲跌年份（for Q4 analysis）
    results["tx_quarterly"] = q_stats.reset_index()

    # ── 3. 週次強弱（指數密技：第1、3週最強）───────────────────
    daily2 = daily.copy()
    daily2["dom"] = daily2.index.day
    daily2["week_of_month"] = daily2["dom"].apply(lambda d: min((d - 1) // 7 + 1, 4))
    daily2["abs_ret"] = daily2["ret"].abs()
    wom_stats = (daily2.groupby("week_of_month")["abs_ret"]
                 .agg(n="count", avg_move="mean")
                 .round(4))
    results["tx_week_of_month"] = wom_stats.reset_index()

    # ── 4. TX 時段分析（日盤 vs 夜盤，30m 資料）─────────────────
    m30 = load_csv(ROOT / "tx/csv/TAIFEX_DLY_MXF1!, 30.csv", tz=True)
    m30["hour"] = m30.index.hour
    m30["ret"]  = (m30["close"] - m30["open"]) / m30["open"] * 100
    m30["abs_ret"] = m30["ret"].abs()

    # 日盤 08:45-13:45, 夜盤 15:00-05:00
    def session_label(h):
        if 9 <= h <= 13:
            return "日盤"
        elif h == 15 or (16 <= h <= 23) or (0 <= h <= 4):
            return "夜盤"
        else:
            return "空窗"
    m30["session"] = m30["hour"].map(session_label)
    sess_stats = (m30[m30["session"] != "空窗"]
                  .groupby("session")["abs_ret"]
                  .agg(n="count", avg_move="mean", trend_pct=lambda x: (x > 0.2).mean())
                  .round(4))
    results["tx_sessions"] = sess_stats.reset_index()

    # ── 5. 選舉年效應（指數密技：四年選舉周期）──────────────────
    # 台灣總統大選：每4年一次（2000,2004,2008,2012,2016,2020,2024）
    election_years = {2000, 2004, 2008, 2012, 2016, 2020, 2024}
    monthly_df2 = monthly_df.copy()
    monthly_df2["is_election"] = monthly_df2["year"].isin(election_years)
    monthly_df2["q"] = ((monthly_df2["month"] - 1) // 3) + 1
    elec_stats = (monthly_df2.groupby(["is_election", "q"])["ret"]
                  .agg(win_rate=lambda x: (x > 0).mean(), avg_ret="mean", n="count")
                  .round(4))
    results["tx_election"] = elec_stats.reset_index()

    return results


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("分析 XAUUSD...")
    xu = analyze_xauusd()
    print("分析 TX...")
    tx = analyze_tx()
    all_results = {"xauusd": xu, "tx": tx}

    out = ROOT / "doc/validation_results.json"
    # Convert DataFrames to dicts
    def to_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, list):
            return obj
        return obj
    serialized = {}
    for k, v in all_results.items():
        serialized[k] = {k2: to_serializable(v2) for k2, v2 in v.items()}

    with open(out, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2, default=str)
    print(f"結果寫入 {out}")

    # Quick summary
    print("\n=== XAUUSD 月份驗證（筆記 vs 資料）===")
    df = xu["xauusd_monthly"]
    month_names = ["一","二","三","四","五","六","七","八","九","十","十一","十二"]
    for _, r in df.iterrows():
        m = int(r["month"])
        match = "✅" if r["match"] else "❌"
        print(f"  {month_names[m-1]:>2}月  筆記:{r['note_view']}  資料:{r['data_view']}(WR={r['win_rate']:.0%})  {match}")

    print("\n=== TX 月份驗證 ===")
    df2 = tx["tx_monthly"]
    for _, r in df2.iterrows():
        m = int(r["month"])
        match = "✅" if r["match"] else "❌"
        print(f"  {month_names[m-1]:>2}月  筆記:{r['note_view']}  資料:{r['data_view']}(WR={r['win_rate']:.0%})  {match}")

    print("\n=== XAUUSD 四個時段 ===")
    for s in xu["xauusd_sessions"]:
        print(f"  {s['時段']}  趨勢K={s['趨勢K比例']:.0%}  筆記:{s['筆記判斷']}")

    print("\n=== TX 季度統計 ===")
    for _, r in tx["tx_quarterly"].iterrows():
        print(f"  Q{int(r['quarter'])}  WR={r['win_rate']:.0%}  avg={r['avg_ret']:.1%}  n={int(r['n'])}")
