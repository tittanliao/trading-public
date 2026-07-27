"""
analyze_ma_bounce.py  v2
台指日線 × 六種信號 收盤跌破 → 買入正二績效分析 + 年度熱力圖

信號類型：
  MA5   = MA5 週線跌破
  MA20  = MA20 月線跌破
  MA60  = MA60 季線跌破
  MA120 = MA120 半年線跌破
  BB_MID  = BB 中軌（Basis=MA20）跌破
  BB_LOW  = BB 下軌（Lower=Basis-2σ）跌破

輸出：
  tx/ma_bounce_signals.csv   — 每筆信號詳細資料
  tx/zheng2_heatmap_ret.png  — 年度正二報酬熱力圖
  tx/zheng2_heatmap_mdd.png  — 年度 MDD 熱力圖
  tx/zheng2_results.json     — 所有統計（供 index.html 引用）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = ['Heiti TC', 'Arial Unicode MS', 'sans-serif']
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import base64, json
from io import BytesIO
from pathlib import Path
from datetime import timedelta

ROOT      = Path(__file__).parent.parent
DAILY_CSV = ROOT / "tx" / "csv" / "TAIFEX_DLY_MXF1!, 1D.csv"
OUT_CSV   = ROOT / "tx" / "ma_bounce_signals.csv"
OUT_JSON  = ROOT / "tx" / "zheng2_results.json"
OUT_RET   = ROOT / "tx" / "zheng2_heatmap_ret.png"
OUT_MDD   = ROOT / "tx" / "zheng2_heatmap_mdd.png"

INVEST = 1_000_000

HOLD_DAYS = {"1M": 30, "3M": 90, "6M": 180, "12M": 365}
HOLD_LABEL = "12M"   # 熱力圖預設持有期

# ── 1. 載入資料 ───────────────────────────────────────────────────
def load_daily():
    df = pd.read_csv(DAILY_CSV)
    df.columns = df.columns.str.strip()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["daily_ret"] = df["close"].pct_change()
    for p in [5, 20, 60, 120]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()
    # BB 用 CSV 預計算值；若缺值則自行計算
    if df["Basis"].isna().all():
        df["Basis"] = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["Upper"] = df["Basis"] + 2 * std20
        df["Lower"] = df["Basis"] - 2 * std20
    return df


# ── 2. 信號定義 ───────────────────────────────────────────────────
SIGNAL_DEFS = {
    "MA5 週線":    lambda df: _crossdown(df, "close", "ma5"),
    "MA20 月線":   lambda df: _crossdown(df, "close", "ma20"),
    "MA60 季線":   lambda df: _crossdown(df, "close", "ma60"),
    "MA120 半年線": lambda df: _crossdown(df, "close", "ma120"),
    "BB中軌":      lambda df: _crossdown(df, "close", "Basis"),
    "BB下軌":      lambda df: _crossdown(df, "close", "Lower"),
}

def _crossdown(df, price_col, ref_col):
    prev_above = df[price_col].shift(1) >= df[ref_col].shift(1)
    curr_below  = df[price_col] < df[ref_col]
    valid       = df[ref_col].notna() & df[ref_col].shift(1).notna()
    return df.index[prev_above & curr_below & valid].tolist()


# ── 3. 單次績效計算 ───────────────────────────────────────────────
def calc_perf(df, signal_idx, hold_days):
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry_date  = df.loc[entry_idx, "time"]
    entry_price = df.loc[entry_idx, "close"]
    exit_date   = entry_date + timedelta(days=hold_days)

    period = df[(df["time"] >= entry_date) & (df["time"] <= exit_date)].copy()
    if len(period) < 2:
        return None

    tx_ret    = (period.iloc[-1]["close"] - entry_price) / entry_price
    rets      = period["daily_ret"].fillna(0).values
    z2_equity = np.cumprod(1 + rets * 2)
    z2_ret    = float(z2_equity[-1] - 1)
    eq        = np.insert(z2_equity, 0, 1.0)
    mdd       = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())

    return {
        "signal_date":  df.loc[signal_idx, "time"].date(),
        "signal_year":  df.loc[signal_idx, "time"].year,
        "entry_date":   entry_date.date(),
        "entry_price":  float(entry_price),
        "tx_ret":       float(tx_ret),
        "z2_ret":       float(z2_ret),
        "mdd":          float(mdd),
        "profit_twd":   float(INVEST * z2_ret),
    }


# ── 4. 熱力圖生成 ─────────────────────────────────────────────────
def make_heatmap(matrix_df, title, fmt_fn, cmap, norm, out_path, label_suffix=""):
    years   = matrix_df.columns.tolist()
    signals = matrix_df.index.tolist()
    data    = matrix_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(12, len(years) * 0.7), len(signals) * 0.9 + 1.2))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(years)));   ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(signals))); ax.set_yticklabels(signals, fontsize=10)

    for i in range(len(signals)):
        for j in range(len(years)):
            v = data[i, j]
            if np.isnan(v):
                txt, color = "—", "#999"
            else:
                txt   = fmt_fn(v)
                color = "white" if abs(v) > 0.25 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"{title}  （持有 {HOLD_LABEL}，正二 proxy）", fontsize=13, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(label_suffix, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()

    buf = BytesIO()
    fig2, ax2 = plt.subplots(figsize=(max(12, len(years) * 0.7), len(signals) * 0.9 + 1.2))
    im2 = ax2.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax2.set_xticks(range(len(years)));   ax2.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax2.set_yticks(range(len(signals))); ax2.set_yticklabels(signals, fontsize=10)
    for i in range(len(signals)):
        for j in range(len(years)):
            v = data[i, j]
            if np.isnan(v):
                txt, color = "—", "#999"
            else:
                txt   = fmt_fn(v)
                color = "white" if abs(v) > 0.25 else "black"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    ax2.set_title(f"{title}  （持有 {HOLD_LABEL}，正二 proxy）", fontsize=13, pad=12)
    plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02).set_label(label_suffix, fontsize=9)
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ── 5. 主程式 ─────────────────────────────────────────────────────
def run():
    df = load_daily()
    years = sorted(df["time"].dt.year.unique())
    print(f"台指日線：{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}  ({len(df)} bars)")

    all_rows   = []
    summary    = {}           # signal_name → hold_label → stats
    year_ret   = {}           # signal_name → {year: avg_z2_ret}
    year_mdd   = {}           # signal_name → {year: avg_mdd}

    for sig_name, sig_fn in SIGNAL_DEFS.items():
        indices = sig_fn(df)
        summary[sig_name] = {}
        year_ret[sig_name] = {}
        year_mdd[sig_name] = {}

        print(f"\n{'='*62}")
        print(f"  {sig_name}  ｜  信號次數：{len(indices)}")
        print(f"{'='*62}")
        print(f"  {'持有':^5}  {'次數':^4}  {'TX勝率':^7}  {'TX均報':^8}  {'正二均報':^9}  {'avg MDD':^8}  {'100萬均獲利':^12}  {'最差單次':^10}")
        print("  " + "─" * 65)

        for hp_label, hp_days in HOLD_DAYS.items():
            rows = []
            for idx in indices:
                r = calc_perf(df, idx, hp_days)
                if r:
                    r["signal"] = sig_name
                    r["hold"]   = hp_label
                    rows.append(r)
                    all_rows.append(r)

            if not rows:
                print(f"  {hp_label:<5}  —")
                continue

            n      = len(rows)
            tx_wr  = np.mean([r["tx_ret"] > 0 for r in rows])
            tx_avg = np.mean([r["tx_ret"] for r in rows])
            z2_avg = np.mean([r["z2_ret"] for r in rows])
            mdd_avg= np.mean([r["mdd"] for r in rows])
            pr_avg = np.mean([r["profit_twd"] for r in rows])
            worst  = min(r["profit_twd"] for r in rows)

            print(f"  {hp_label:<5}  {n:>4}  {tx_wr*100:>6.0f}%  {tx_avg*100:>+7.1f}%  {z2_avg*100:>+9.1f}%  {mdd_avg*100:>+7.1f}%  {pr_avg:>+12,.0f}  {worst:>+10,.0f}")

            summary[sig_name][hp_label] = {
                "n": n, "tx_wr": round(tx_wr, 3),
                "tx_avg": round(tx_avg, 4), "z2_avg": round(z2_avg, 4),
                "mdd_avg": round(mdd_avg, 4), "pr_avg": round(pr_avg, 0),
                "worst": round(worst, 0),
            }

            # 年度細分（用於熱力圖）
            if hp_label == HOLD_LABEL:
                for yr in years:
                    yr_rows = [r for r in rows if r["signal_year"] == yr]
                    if yr_rows:
                        year_ret[sig_name][yr] = float(np.mean([r["z2_ret"] for r in yr_rows]))
                        year_mdd[sig_name][yr] = float(np.mean([r["mdd"]    for r in yr_rows]))

    # ── 熱力圖 ──────────────────────────────────────────────────
    sig_order = list(SIGNAL_DEFS.keys())
    yr_cols   = [y for y in years if y <= 2025]   # 2026 資料不完整

    ret_matrix = pd.DataFrame(index=sig_order, columns=yr_cols, dtype=float)
    mdd_matrix = pd.DataFrame(index=sig_order, columns=yr_cols, dtype=float)
    for s in sig_order:
        for y in yr_cols:
            ret_matrix.loc[s, y] = year_ret[s].get(y, np.nan)
            mdd_matrix.loc[s, y] = year_mdd[s].get(y, np.nan)

    # 報酬熱力圖：紅→白→綠
    cmap_ret = LinearSegmentedColormap.from_list("ret", ["#dc2626","#fef9c3","#16a34a"])
    norm_ret = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=1.2)
    b64_ret = make_heatmap(ret_matrix, "年度平均正二報酬",
                           lambda v: f"{v*100:+.0f}%",
                           cmap_ret, norm_ret, OUT_RET, "正二報酬")

    # MDD 熱力圖：綠（小回撤）→ 紅（大回撤）
    cmap_mdd = LinearSegmentedColormap.from_list("mdd", ["#16a34a","#fef9c3","#dc2626"])
    norm_mdd = TwoSlopeNorm(vmin=-0.6, vcenter=-0.2, vmax=0.0)
    b64_mdd = make_heatmap(mdd_matrix, "年度平均 MDD（持有期最大回撤）",
                           lambda v: f"{v*100:.0f}%",
                           cmap_mdd, norm_mdd, OUT_MDD, "MDD")

    # ── JSON 輸出 ────────────────────────────────────────────────
    out = {
        "summary": summary,
        "heatmap_ret_b64": b64_ret,
        "heatmap_mdd_b64": b64_mdd,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── CSV ─────────────────────────────────────────────────────
    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ 熱力圖：{OUT_RET} / {OUT_MDD}")
    print(f"✅ JSON  ：{OUT_JSON}")
    print(f"✅ CSV   ：{OUT_CSV}")
    return b64_ret, b64_mdd, summary


if __name__ == "__main__":
    run()
