"""
run_lab10_optimizer.py — Lab10：10 個高勝率候選策略（順勢×5 + 逆勢×5）參數最佳化
==============================================================================
輸出：xauusd/XAUUSD-Long-Lab/report_lab10.html + lab10_results.json

用途（20260712 與使用者確立）：用 2.5 年完整匯出資料（含 footprint）設計 10 個
獨立的做多策略，Python 掃參數後把最優參數寫進 Pine 下拉選單腳本
（XAUUSD-Long-Lab10-V1.pine），作為 S1/S2 的優化燃料——
表現好的進場邏輯/濾網之後可拆進主策略。

策略清單（T=順勢 Trend、R=逆勢 Reversal）：
  T1 週VWAP回踩續漲    T2 突破+Delta確認     T3 HTF動能疊加
  T4 BBW擴張突破       T5 POC階梯上移(FP核心)
  R1 RSI超賣+Delta改善  R2 恐慌爆量長下影     R3 VWAP深度回歸
  R4 Trapped Sellers(FP核心)  R5 DXY弱勢逢低接

出場一律用 Profit Flyer（與 S1/S2 相同結構）：
  順勢用 S1 檔位（SL0.5%/TP1 1R/TP2 3.5R/36bars）
  逆勢用 S2 檔位（SL1.0%/TP1 2R/TP2 4R/48bars）

排名邏輯：先過門檻（n≥門檻、min(IS,OOS) PF≥1.2），再按勝率排——
使用者要「高勝率」，但 PF/OOS 門檻防止勝率幻覺（高WR小賺大賠）。

執行方式（在 trading/ 根目錄）：
  python3.12 xauusd/scripts/run_lab10_optimizer.py --export "xauusd/csv/XAUUSD-S1S2-Export/FX_IDC_XAUUSD, 30 (7).csv"
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "xauusd"))
sys.path.insert(0, str(ROOT / "xauusd/scripts"))

import run_s1s2_filter_optimizer as base  # 重用：load_export/htf_indicator/rsi/atr/引擎等

OUT_DIR = ROOT / "xauusd/XAUUSD-Long-Lab"
OUT_HTML = OUT_DIR / "report_lab10.html"
OUT_JSON = OUT_DIR / "lab10_results.json"

MIN_TRADES = 40          # 一般策略樣本門檻
MIN_TRADES_FP = 15       # footprint 核心策略（資料僅 2026/4 起）放寬
MIN_SCORE_PF = 1.2       # min(IS PF, OOS PF) 下限
IS_RATIO = 0.7

EXIT_TREND = dict(sl=0.005, tp1=1.0, tp2=3.5, out=36)
EXIT_REV   = dict(sl=0.010, tp1=2.0, tp2=4.0, out=48)


def consec(cond: pd.Series, k: int) -> pd.Series:
    return cond.fillna(False).astype(int).rolling(k).min() == 1


# ══════════════════════════════════════════════════════════════════════════════
# 10 個策略定義：每個 = dict(id, side, desc, grid, fn(df, ctx, p)->Series, exit, fp_core)
# ══════════════════════════════════════════════════════════════════════════════

def s_t1(df, ctx, p):
    """週VWAP回踩：價格在週VWAP上方，回踩觸及VWAP帶後收紅收回上方（順勢續漲）"""
    w = ctx["wvwap"]
    touch = df["low"] <= w * (1 + p["tol"] / 100)
    return (df["close"] > w) & touch & (df["close"] > df["open"])

def s_t2(df, ctx, p):
    """突破+Delta確認：收盤突破N根高點；可選footprint delta佔比確認（na放行）"""
    hh = df["high"].rolling(p["n"]).max().shift(1)
    sig = df["close"] > hh
    if p["dmin"] > 0 and ctx["has_fp"]:
        dr = ctx["FP_DeltaRatio"]
        sig = sig & (dr >= p["dmin"]).where(ctx["fp_ok_mask"], True).fillna(True)
    return sig

def s_t3(df, ctx, p):
    """HTF動能疊加：RSI上穿門檻 且 收盤在4H MA上方（動能點火+結構順勢）"""
    ma = base.htf_indicator(df, 240, lambda h: h["close"].rolling(p["malen"]).mean())
    return base.crossover(ctx["rsi14"], pd.Series(p["lvl"], index=df.index)) & (df["close"] > ma)

def s_t4(df, ctx, p):
    """BBW擴張突破：60m BBW rank 上穿門檻（波動點火）+ 紅K收在20SMA上"""
    rank = base.htf_indicator(df, 60, lambda h: base.percentrank(
        h["close"].rolling(20).std(ddof=0) * 4 / h["close"].rolling(20).mean(), 60))
    ignite = (rank >= p["thr"]) & (rank.shift(1) < p["thr"])
    basis = ctx["ohlc4"].rolling(20).mean()
    return ignite & (df["close"] > df["open"]) & (df["close"] > basis)

def s_t5(df, ctx, p):
    """POC階梯上移（FP核心）：POC位置連續k根 ≥ 門檻% 且紅K——高價連續被接受"""
    if not ctx["has_fp"]:
        return pd.Series(False, index=df.index)
    cond = (ctx["FP_POCpos"] >= p["poc"]) & ctx["fp_ok_mask"]
    return consec(cond, p["k"]) & (df["close"] > df["open"])

def s_r1(df, ctx, p):
    """RSI超賣+Delta改善：RSI<門檻的紅K；可選footprint delta相對改善（na放行）"""
    sig = (ctx["rsi14"] < p["th"]) & (df["close"] > df["open"])
    if p["imp"] > 0 and ctx["has_fp"]:
        dr = ctx["FP_DeltaRatio"]
        prior = dr.where(ctx["fp_ok_mask"]).rolling(5, min_periods=2).mean().shift(1)
        sig = sig & ((dr - prior) >= p["imp"]).where(ctx["fp_ok_mask"], True).fillna(True)
    return sig

def s_r2(df, ctx, p):
    """恐慌爆量長下影：下影線佔比≥ws 且 量≥m×均量（selling climax 輕量版，
    比正式錘頭定義寬鬆以擴大樣本）"""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_btm = df[["open", "close"]].min(axis=1)
    wick = (body_btm - df["low"]) / rng
    return (wick >= p["ws"]) & (ctx["vol"] >= ctx["vol_sma20"] * p["m"]).fillna(False)

def s_r3(df, ctx, p):
    """VWAP深度回歸：收盤深低於週VWAP達 d×ATR 且收紅（超跌回歸均值）"""
    depth = (ctx["wvwap"] - df["close"]) / ctx["atr14"]
    return (depth >= p["d"]) & (df["close"] > df["open"])

def s_r4(df, ctx, p):
    """Trapped Sellers（FP核心）：低檔出現賣方失衡 但 delta相對前5根改善
    且收在K棒上半——恐慌賣單被吃光、賣方被困"""
    if not ctx["has_fp"]:
        return pd.Series(False, index=df.index)
    simb = ctx["FP_SellImbLow50"] > 0.5
    dr = ctx["FP_DeltaRatio"]
    prior = dr.where(ctx["fp_ok_mask"]).rolling(5, min_periods=2).mean().shift(1)
    upper_half = df["close"] > (df["high"] + df["low"]) / 2
    return simb & ((dr - prior) >= p["imp"]) & upper_half & ctx["fp_ok_mask"]

def s_r5(df, ctx, p):
    """DXY弱勢逢低接：DXY RSI<50 且 自48根高點回檔≥p% 且 反轉紅K收高"""
    hh = df["high"].rolling(48).max().shift(1)
    pull = df["close"] <= hh * (1 - p["pb"] / 100)
    rev = (df["close"] > df["open"]) & (df["close"] > df["close"].shift(1))
    return (ctx["dxy_rsi"] < 50) & pull & rev

STRATS = [
    dict(id="T1", side="順勢", name="週VWAP回踩續漲", fn=s_t1, exit=EXIT_TREND, fp_core=False,
         grid=dict(tol=[0.0, 0.1, 0.2])),
    dict(id="T2", side="順勢", name="突破+Delta確認", fn=s_t2, exit=EXIT_TREND, fp_core=False,
         grid=dict(n=[24, 48, 96], dmin=[0, 10, 20])),
    dict(id="T3", side="順勢", name="HTF動能疊加", fn=s_t3, exit=EXIT_TREND, fp_core=False,
         grid=dict(lvl=[50, 55], malen=[10, 20])),
    dict(id="T4", side="順勢", name="BBW擴張突破", fn=s_t4, exit=EXIT_TREND, fp_core=False,
         grid=dict(thr=[70, 80])),
    dict(id="T5", side="順勢", name="POC階梯上移(FP)", fn=s_t5, exit=EXIT_TREND, fp_core=True,
         grid=dict(poc=[55, 60, 65], k=[2, 3])),
    dict(id="R1", side="逆勢", name="RSI超賣+Delta改善", fn=s_r1, exit=EXIT_REV, fp_core=False,
         grid=dict(th=[25, 30], imp=[0, 5])),
    dict(id="R2", side="逆勢", name="恐慌爆量長下影", fn=s_r2, exit=EXIT_REV, fp_core=False,
         grid=dict(ws=[0.5, 0.6], m=[1.5, 2.0])),
    dict(id="R3", side="逆勢", name="VWAP深度回歸", fn=s_r3, exit=EXIT_REV, fp_core=False,
         grid=dict(d=[2.0, 3.0])),
    dict(id="R4", side="逆勢", name="Trapped Sellers(FP)", fn=s_r4, exit=EXIT_REV, fp_core=True,
         grid=dict(imp=[5.0, 10.0])),
    dict(id="R5", side="逆勢", name="DXY弱勢逢低接", fn=s_r5, exit=EXIT_REV, fp_core=False,
         grid=dict(pb=[1.0, 1.5, 2.0])),
]


def optimize_strategy(df, ctx, spec, cut_time):
    keys = list(spec["grid"].keys())
    rows = []
    for vals in itertools.product(*(spec["grid"][k] for k in keys)):
        p = dict(zip(keys, vals))
        sig = spec["fn"](df, ctx, p).fillna(False)
        tr = base.run_engine(df, sig, spec["exit"]["sl"], spec["exit"]["tp1"],
                              spec["exit"]["tp2"], spec["exit"]["out"])
        full = base.stat_block(tr)
        min_n = MIN_TRADES_FP if spec["fp_core"] else MIN_TRADES
        if full["n"] == 0:
            continue
        tr["entry_time"] = pd.to_datetime(tr["entry_time"])
        is_ = base.stat_block(tr[tr.entry_time < cut_time])
        oos = base.stat_block(tr[tr.entry_time >= cut_time])
        score = min(is_["pf"] if np.isfinite(is_["pf"]) else 0,
                    oos["pf"] if np.isfinite(oos["pf"]) else 0)
        rows.append(dict(params=p, n=full["n"], wr=full["wr"], pf=full["pf"],
                         net_r=full["net_r"], is_pf=is_["pf"], oos_n=oos["n"],
                         oos_pf=oos["pf"], oos_wr=oos["wr"], score=score,
                         qualified=(full["n"] >= min_n and score >= MIN_SCORE_PF)))
    # 先取合格者按 WR 排；若無合格者取 score 最高者標註未達標
    qual = [r for r in rows if r["qualified"]]
    pool = qual if qual else rows
    pool.sort(key=lambda r: (r["wr"], r["score"]), reverse=True)
    return pool[0] if pool else None, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print(f"載入：{args.export}")
    df = base.load_export(Path(args.export))
    print(f"  {len(df)} 根 30m（{df['time'].min()} → {df['time'].max()}）")
    ctx = base.build_common(df)
    print(f"  Footprint 欄位：{'✅' if ctx['has_fp'] else '❌（T5/R4 將無訊號）'}")
    cut_time = df["time"].iloc[int(len(df) * IS_RATIO)]
    print(f"  IS/OOS 切分點：{cut_time}")

    results = {}
    for spec in STRATS:
        best, allrows = optimize_strategy(df, ctx, spec, cut_time)
        results[spec["id"]] = dict(side=spec["side"], name=spec["name"],
                                    fp_core=spec["fp_core"], exit=spec["exit"],
                                    best=best, all=allrows)
        if best:
            q = "✅" if best["qualified"] else "⚠️未達門檻"
            print(f"  {spec['id']}（{spec['side']}）{spec['name']}: 最佳 {best['params']} "
                  f"n={best['n']} WR={best['wr']:.1f}% PF={best['pf']:.2f} "
                  f"OOS={best['oos_n']}/PF{base.fmt(best['oos_pf'])} {q}")
        else:
            print(f"  {spec['id']}（{spec['side']}）{spec['name']}: 無任何訊號")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
    print(f"✅ JSON：{OUT_JSON}")

    # ── HTML 報表 ──
    ranked = sorted([(k, v) for k, v in results.items() if v["best"]],
                    key=lambda kv: kv[1]["best"]["wr"], reverse=True)
    body = ""
    for sid, v in ranked:
        b = v["best"]
        q = "✅" if b["qualified"] else "⚠️"
        fpnote = "（FP核心，資料僅2026/4起）" if v["fp_core"] else ""
        body += (f"<tr><td><strong>{sid}</strong> {v['side']}</td><td>{v['name']}{fpnote}</td>"
                 f"<td style='font-size:.8em'>{b['params']}</td><td>{b['n']}</td>"
                 f"<td><strong>{b['wr']:.1f}%</strong></td><td>{base.fmt(b['pf'])}</td>"
                 f"<td>{b['net_r']:.0f}R</td><td>{b['oos_n']}/{base.fmt(b['oos_pf'])}</td>"
                 f"<td>{q}</td></tr>")
    css = """<style>*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
.wrap{max-width:1200px;margin:0 auto}h1{font-size:1.5em;margin-bottom:6px;color:#f8fafc}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:22px;margin-bottom:18px}
.tbl{width:100%;border-collapse:collapse;font-size:.85em;margin:10px 0}
.tbl th{background:#0f172a;color:#94a3b8;padding:8px 10px;text-align:left;border-bottom:1px solid #334155}
.tbl td{padding:7px 10px;border-bottom:1px solid rgba(51,65,85,.5)}
.note{font-size:.82em;color:#94a3b8;margin-top:8px}
.warn{background:rgba(245,158,11,.08);border-left:3px solid #f59e0b;padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}</style>"""
    html = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8">
<title>Lab10 — 10策略參數最佳化</title>{css}</head><body><div class="wrap">
<h1>Lab10 — 10 個高勝率候選策略（S1/S2 優化燃料）</h1>
<p class="note">生成：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
資料 {len(df)} 根 30m（{df['time'].min().date()} → {df['time'].max().date()}）·
出場一律 Profit Flyer（順勢=S1檔位 0.5%/1R/3.5R/36；逆勢=S2檔位 1%/2R/4R/48）·
按勝率排名，門檻：n≥{MIN_TRADES}（FP核心≥{MIN_TRADES_FP}）且 min(IS,OOS) PF≥{MIN_SCORE_PF}</p>
<div class="card">
<table class="tbl"><thead><tr><th>策略</th><th>邏輯</th><th>最佳參數</th><th>筆數</th>
<th>勝率</th><th>PF</th><th>淨利R</th><th>OOS n/PF</th><th>門檻</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="note">對應 Pine：<code>XAUUSD-Long-Lab10-V1.pine</code>（下拉選單，最佳參數已設為預設值）。
引擎重播已驗證與 TV 吻合率 ~92%（見 report_s1s2_optimizer.html），絕對數字仍以 TV 回測為準。</p>
</div>
<div class="warn"><strong>用途聲明：</strong>Lab10 是 S1/S2 的「優化燃料」——表現好的進場邏輯（例如
footprint 吸收、VWAP 結構、DXY 擇時）之後拆成主策略的 filter 或觸發器，並非要直接上線 10 個新策略。
高勝率但低頻的策略（如 R2 恐慌爆量）適合當獨立 alert。</p></div>
</div></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ 報告：{OUT_HTML}")


if __name__ == "__main__":
    main()
