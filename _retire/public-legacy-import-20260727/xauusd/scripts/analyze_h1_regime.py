"""
analyze_h1_regime.py
--------------------
用 2026 H1 真實交易紀錄 × 4H OHLC，驗證 Regime 四象限對 S1/S2 勝率的影響。

Regime 四象限：
  方向 (slope): 4H EMA20 在過去 10 根的斜率（%）
  波動 (bbw_rank): 目前 BBW 在過去 60 根 4H 的百分位

輸出：
  1. regime_trades.csv  — 每筆交易加上 Regime 欄位
  2. 終端機勝率表        — Regime × 策略 勝率
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 路徑 ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # 20260705 移至 scripts/ 子資料夾，多一層 .parent
TRADE_CSV = ROOT / "csv" / "XAUUSD_2026H1.csv"
OHLC_4H   = ROOT / "csv" / "FX_IDC_XAUUSD, 240.csv"
OUT_CSV   = ROOT / "csv" / "XAUUSD_2026H1_regime.csv"

# ── 1. 讀取 4H 價格資料 ───────────────────────────────────────────
def load_4h():
    df = pd.read_csv(OHLC_4H)
    df["time"] = pd.to_datetime(df["time"], utc=False).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)

    # EMA20
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    # BB(20, 2)
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bbw"]      = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR(14)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1))
        )
    )
    df["atr14"] = df["tr"].rolling(14).mean()

    # EMA20 斜率：(ema_now - ema_10_bars_ago) / ema_10_bars_ago × 100
    df["ema20_slope"] = (df["ema20"] - df["ema20"].shift(10)) / df["ema20"].shift(10) * 100

    # BBW 百分位（60 根滾動）
    df["bbw_rank"] = df["bbw"].rolling(60).rank(pct=True) * 100

    # Price Z-Score = (close - ema20) / atr14
    df["z_score"] = (df["close"] - df["ema20"]) / df["atr14"]

    return df[["time", "close", "ema20", "ema20_slope", "bbw", "bbw_rank", "atr14", "z_score"]]


# ── 2. 讀取交易紀錄 ───────────────────────────────────────────────
def parse_tw_date(s):
    """把「2026年1月6日 20:30 (GMT+8)」解析成 datetime。"""
    import re
    m = re.match(r"(\d+)年(\d+)月(\d+)日\s+(\d+):(\d+)", str(s))
    if not m:
        return pd.NaT
    y, mo, d, h, mi = [int(x) for x in m.groups()]
    return pd.Timestamp(y, mo, d, h, mi)

def load_trades():
    df = pd.read_csv(TRADE_CSV)
    df.columns = df.columns.str.strip()
    df["entry_time"] = df["Trade Date"].apply(parse_tw_date)
    df["net_r"]      = pd.to_numeric(df["Net R"], errors="coerce")
    df["win"]        = pd.to_numeric(df["Win"],   errors="coerce").fillna(0).astype(int)

    # 策略分類（簡化成 4 類）
    def classify(row):
        tid = str(row.get("Trade ID", ""))
        if "FOMO" in tid or "FOMO" in str(row.get("Error Tag", "")):
            return "FOMO"
        if "S1-Fail" in tid or "S1-Reverse" in tid or "S1-FailPattern" in tid:
            return "S1-Fail"
        if "S2-Fail" in tid or "S2-FailPattern" in tid:
            return "S2-Fail"
        if "S2" in tid:
            return "S2"
        if "S1" in tid:
            return "S1"
        return "Other"

    df["strategy"] = df.apply(classify, axis=1)
    return df[["entry_time", "net_r", "win", "strategy", "Net R", "Risk", "Session", "Remark"]]


# ── 3. merge_asof：找每筆交易進場時最近的 4H bar ─────────────────
def attach_regime(trades, ohlc):
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    merged = pd.merge_asof(
        trades,
        ohlc,
        left_on="entry_time",
        right_on="time",
        direction="backward"
    )
    return merged


# ── 4. Regime 四象限標籤 ─────────────────────────────────────────
SLOPE_UP   =  0.15   # EMA slope % 上升門檻
SLOPE_DOWN = -0.15   # EMA slope % 下降門檻
BBW_HIGH   =  65     # BBW 百分位：趨勢性行情門檻

def label_regime(row):
    slope = row["ema20_slope"]
    rank  = row["bbw_rank"]
    if pd.isna(slope) or pd.isna(rank):
        return "UNKNOWN"
    if rank >= BBW_HIGH:
        if slope >= SLOPE_UP:
            return "BULL_TREND"
        elif slope <= SLOPE_DOWN:
            return "BEAR_TREND"
        else:
            return "HIGH_VOL_RANGE"
    else:
        if slope >= SLOPE_UP:
            return "RANGE_UP"
        elif slope <= SLOPE_DOWN:
            return "RANGE_DOWN"
        else:
            return "CONSOLIDATION"


# ── 5. 輸出報告 ───────────────────────────────────────────────────
def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def run():
    print("載入 4H 資料...")
    ohlc   = load_4h()
    print(f"  {len(ohlc)} 根 4H bars，{ohlc['time'].min()} → {ohlc['time'].max()}")

    print("載入交易紀錄...")
    trades = load_trades()
    print(f"  {len(trades)} 筆交易，{trades['entry_time'].min()} → {trades['entry_time'].max()}")

    print("合併 Regime 指標...")
    df = attach_regime(trades, ohlc)
    df["regime"] = df.apply(label_regime, axis=1)

    # 排除 FOMO / Other / 無效時間
    valid = df[df["strategy"].isin(["S1", "S2", "S1-Fail", "S2-Fail"])].copy()
    valid = valid[valid["entry_time"].notna() & valid["net_r"].notna()]

    # ── 報告 1：Regime 分佈 ──────────────────────────────────────
    print_section("Regime 分佈（S1/S2 有效筆數）")
    regime_counts = valid.groupby("regime").size().rename("筆數")
    print(regime_counts.to_string())

    # ── 報告 2：Regime × 策略 勝率 ──────────────────────────────
    print_section("Regime × 策略 勝率")

    REGIME_ORDER = ["BULL_TREND", "BEAR_TREND", "HIGH_VOL_RANGE",
                    "RANGE_UP", "RANGE_DOWN", "CONSOLIDATION", "UNKNOWN"]
    STRAT_ORDER  = ["S1", "S1-Fail", "S2", "S2-Fail"]

    rows = []
    for regime in REGIME_ORDER:
        sub = valid[valid["regime"] == regime]
        if len(sub) == 0:
            continue
        row = {"Regime": regime, "總筆": len(sub)}
        for strat in STRAT_ORDER:
            s = sub[sub["strategy"] == strat]
            if len(s) == 0:
                row[strat] = "-"
            else:
                wr = s["win"].mean() * 100
                row[strat] = f"{wr:.0f}% (n={len(s)})"
        rows.append(row)

    rpt = pd.DataFrame(rows).set_index("Regime")
    print(rpt.to_string())

    # ── 報告 3：Regime × 策略 平均 R ────────────────────────────
    print_section("Regime × 策略 平均 Net R")
    rows2 = []
    for regime in REGIME_ORDER:
        sub = valid[valid["regime"] == regime]
        if len(sub) == 0:
            continue
        row = {"Regime": regime}
        for strat in STRAT_ORDER:
            s = sub[sub["strategy"] == strat]
            if len(s) == 0:
                row[strat] = "-"
            else:
                avg_r = s["net_r"].mean()
                row[strat] = f"{avg_r:+.2f}R"
        rows2.append(row)

    rpt2 = pd.DataFrame(rows2).set_index("Regime")
    print(rpt2.to_string())

    # ── 報告 4：Z-Score 分組 × S2 勝率 ─────────────────────────
    print_section("Z-Score 分組 × S2 勝率（均值回歸驗證）")
    s2 = valid[valid["strategy"] == "S2"].copy()
    s2["z_bucket"] = pd.cut(
        s2["z_score"],
        bins=[-np.inf, -2.5, -1.5, -0.5, 0.5, np.inf],
        labels=["< -2.5", "-2.5 ~ -1.5", "-1.5 ~ -0.5", "-0.5 ~ +0.5", "> +0.5"]
    )
    z_tbl = s2.groupby("z_bucket", observed=True).agg(
        筆數=("win","count"),
        勝率=("win", lambda x: f"{x.mean()*100:.0f}%"),
        avg_R=("net_r","mean")
    )
    z_tbl["avg_R"] = z_tbl["avg_R"].map("{:+.2f}R".format)
    print(z_tbl.to_string())

    # ── 報告 5：BBW_rank 分組 × S1 勝率（Momentum 驗證）────────
    print_section("BBW Rank 分組 × S1 勝率（Momentum 驗證）")
    s1 = valid[valid["strategy"] == "S1"].copy()
    s1["bbw_bucket"] = pd.cut(
        s1["bbw_rank"],
        bins=[0, 30, 50, 70, 100],
        labels=["0-30% 收縮", "30-50%", "50-70%", "70-100% 擴張"]
    )
    b_tbl = s1.groupby("bbw_bucket", observed=True).agg(
        筆數=("win","count"),
        勝率=("win", lambda x: f"{x.mean()*100:.0f}%"),
        avg_R=("net_r","mean")
    )
    b_tbl["avg_R"] = b_tbl["avg_R"].map("{:+.2f}R".format)
    print(b_tbl.to_string())

    # ── 儲存 CSV ────────────────────────────────────────────────
    out_cols = ["entry_time", "strategy", "net_r", "win", "regime",
                "ema20_slope", "bbw_rank", "z_score", "Session", "Remark"]
    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 輸出：{OUT_CSV}")


if __name__ == "__main__":
    run()
