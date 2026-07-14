"""
run_lab10_short_optimizer.py — Lab10-Short：10 個空單候選策略（順勢空×5 + 逆勢空×5）
==============================================================================
輸出：xauusd/XAUUSD-Short-Lab/report_lab10_short.html + lab10_short_results.json

與 run_lab10_optimizer.py（多單版）相同概念：2.5 年匯出資料 → Python 參數網格
→ 最佳參數寫進 Pine 下拉選單（XAUUSD-Short-Lab10-V1.pine）。

⚠️ 空單特別注意：
  1. 2024-2026 黃金整體大多頭，空單天生逆風——結果解讀時 OOS 段（2025/10 起，
     含 2026H1 修正行情）比全樣本更有參考價值
  2. 匯出指標 V1 只算了「低區」失衡堆疊（為多單設計），空單 footprint 條件
     僅能用 DeltaRatio / POC 位置（頂部出貨 = POC高檔+delta負）；
     上緣失衡鏡像（頂部買方失衡=trapped buyers）需匯出指標 V2 才能做
  3. 空單引擎為多單引擎的鏡像（多單引擎已驗證 92% 吻合率；空單無 TV 逐筆可對照，
     邏輯鏡像自同一套 Profit Flyer 結構）

執行方式（在 trading/ 根目錄）：
  python3.12 xauusd/scripts/run_lab10_short_optimizer.py --export "xauusd/csv/XAUUSD-S1S2-Export/FX_IDC_XAUUSD, 30 (7).csv"
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

import run_s1s2_filter_optimizer as base

OUT_DIR = ROOT / "xauusd/XAUUSD-Short-Lab"
OUT_HTML = OUT_DIR / "report_lab10_short.html"
OUT_JSON = OUT_DIR / "lab10_short_results.json"

MIN_TRADES = 40
MIN_TRADES_FP = 15
MIN_SCORE_PF = 1.2
IS_RATIO = 0.7

EXIT_TREND = dict(sl=0.005, tp1=1.0, tp2=3.5, out=36)   # 順勢空＝S1 檔位鏡像
EXIT_REV   = dict(sl=0.010, tp1=2.0, tp2=4.0, out=48)   # 逆勢空＝S2 檔位鏡像


def consec(cond: pd.Series, k: int) -> pd.Series:
    return cond.fillna(False).astype(int).rolling(k).min() == 1


# ══════════════════════════════════════════════════════════════════════════════
# 空單引擎（多單 Profit Flyer 鏡像；訊號bar收盤→次bar開盤進場）
# ══════════════════════════════════════════════════════════════════════════════

def run_engine_short(df: pd.DataFrame, signal: pd.Series, sl_pct: float,
                     tp1_r: float, tp2_r: float, out_k: int) -> pd.DataFrame:
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    t = df["time"].to_numpy()
    sig = signal.fillna(False).to_numpy()
    high_q = pd.Series(h).rolling(max(out_k // 4, 1)).max().to_numpy()
    high_h = pd.Series(h).rolling(max(out_k // 2, 1)).max().to_numpy()
    high_f = pd.Series(h).rolling(out_k).max().to_numpy()

    trades = []
    in_pos = False
    entry_px = entry_i = 0
    stop = np.nan
    pending_entry = False

    for i in range(1, len(df)):
        if pending_entry and not in_pos:
            in_pos = True
            entry_px = o[i]
            entry_i = i
            stop = np.nan
            pending_entry = False
        if in_pos and not np.isnan(stop) and h[i] >= stop:
            fill = max(o[i], stop)
            trades.append(dict(entry_time=t[entry_i], exit_time=t[i],
                               entry_px=entry_px, exit_px=fill,
                               pnl_r=(entry_px / fill - 1) * (fill / entry_px) / sl_pct
                                     if False else (entry_px - fill) / entry_px / sl_pct,
                               bars=i - entry_i))
            in_pos = False
            stop = np.nan
        if in_pos:
            sl = entry_px * (1 + sl_pct)
            pt1 = entry_px * (1 - sl_pct * tp1_r)
            pt2 = entry_px * (1 - sl_pct * tp2_r)
            if c[i] <= pt2:
                stop = min(pt2, high_q[i])
            elif c[i] <= pt1:
                stop = min(pt1, high_h[i])
            else:
                stop = sl
            if i - entry_i >= out_k:
                stop = min(sl, high_f[i]) if c[i] <= entry_px else h[i]
        if not in_pos and sig[i]:
            pending_entry = True
    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# 10 個空單策略定義
# ══════════════════════════════════════════════════════════════════════════════

def s_ts1(df, ctx, p):
    """TS1 BBW擴張破位（順勢空）：60m BBW rank 上穿門檻（波動點火）+ 黑K收在20SMA下
    ——多單版 T4（Lab10順勢最佳）的鏡像"""
    rank = base.htf_indicator(df, 60, lambda h: base.percentrank(
        h["close"].rolling(20).std(ddof=0) * 4 / h["close"].rolling(20).mean(), 60))
    ignite = (rank >= p["thr"]) & (rank.shift(1) < p["thr"])
    basis = ctx["ohlc4"].rolling(20).mean()
    return ignite & (df["close"] < df["open"]) & (df["close"] < basis)

def s_ts2(df, ctx, p):
    """TS2 破底+Delta確認（順勢空）：收盤跌破N根低點；可選 delta ≤ -門檻 確認（na放行）"""
    ll = df["low"].rolling(p["n"]).min().shift(1)
    sig = df["close"] < ll
    if p["dmin"] > 0 and ctx["has_fp"]:
        dr = ctx["FP_DeltaRatio"]
        sig = sig & (dr <= -p["dmin"]).where(ctx["fp_ok_mask"], True).fillna(True)
    return sig

def s_ts3(df, ctx, p):
    """TS3 HTF動能空（順勢空）：RSI 下穿門檻 且 收盤在4H MA下方"""
    ma = base.htf_indicator(df, 240, lambda h: h["close"].rolling(p["malen"]).mean())
    cross_dn = (ctx["rsi14"] < p["lvl"]) & (ctx["rsi14"].shift(1) >= p["lvl"])
    return cross_dn & (df["close"] < ma)

def s_ts4(df, ctx, p):
    """TS4 週VWAP失守（順勢空）：前一根收在週VWAP上、本根黑K收破VWAP（機構成本線失守）"""
    w = ctx["wvwap"]
    return (df["close"].shift(1) > w.shift(1)) & (df["close"] < w) & (df["close"] < df["open"])

def s_ts5(df, ctx, p):
    """TS5 DXY強勢順空（順勢空）：DXY日線RSI ≥ 門檻（美元強）+ 收盤在4H MA20下 + 黑K"""
    ma = base.htf_indicator(df, 240, lambda h: h["close"].rolling(20).mean())
    return (ctx["dxy_rsi"] >= p["dxy"]) & (df["close"] < ma) & (df["close"] < df["open"])

def s_rs1(df, ctx, p):
    """RS1 買方衰竭長上影（逆勢空）：上影線佔比≥ws 且 量≥m×均量（buying climax）
    ——多單版 R2（Lab10逆勢最佳）的鏡像"""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_top = df[["open", "close"]].max(axis=1)
    wick = (df["high"] - body_top) / rng
    return (wick >= p["ws"]) & (ctx["vol"] >= ctx["vol_sma20"] * p["m"]).fillna(False)

def s_rs2(df, ctx, p):
    """RS2 RSI超買+Delta轉弱（逆勢空）：RSI>門檻的黑K；可選 delta 相對前5根惡化（na放行）"""
    sig = (ctx["rsi14"] > p["th"]) & (df["close"] < df["open"])
    if p["det"] > 0 and ctx["has_fp"]:
        dr = ctx["FP_DeltaRatio"]
        prior = dr.where(ctx["fp_ok_mask"]).rolling(5, min_periods=2).mean().shift(1)
        sig = sig & ((prior - dr) >= p["det"]).where(ctx["fp_ok_mask"], True).fillna(True)
    return sig

def s_rs3(df, ctx, p):
    """RS3 VWAP乖離過熱（逆勢空）：收盤高於週VWAP達 d×ATR 且收黑（過度延伸回歸）"""
    ext = (df["close"] - ctx["wvwap"]) / ctx["atr14"]
    return (ext >= p["d"]) & (df["close"] < df["open"])

def s_rs4(df, ctx, p):
    """RS4 頂部出貨（逆勢空，FP）：POC 在K棒高檔（≥門檻%）+ delta為負（高檔被賣壓接手=
    distribution）+ 收在下半。註：上緣買方失衡（trapped buyers）需匯出指標V2"""
    if not ctx["has_fp"]:
        return pd.Series(False, index=df.index)
    poc = ctx["FP_POCpos"]; dr = ctx["FP_DeltaRatio"]
    lower_half = df["close"] < (df["high"] + df["low"]) / 2
    return (poc >= p["poc"]) & (dr <= -p["dneg"]) & lower_half & ctx["fp_ok_mask"]

def s_rs5(df, ctx, p):
    """RS5 假突破反轉（逆勢空）：盤中創48根新高但收盤跌回前高之下 且 收黑（bull trap）"""
    hh = df["high"].rolling(48).max().shift(1)
    return (df["high"] > hh) & (df["close"] < hh * (1 - p["fb"] / 100)) & (df["close"] < df["open"])

STRATS = [
    dict(id="TS1", side="順勢空", name="BBW擴張破位", fn=s_ts1, exit=EXIT_TREND, fp_core=False,
         grid=dict(thr=[70, 80])),
    dict(id="TS2", side="順勢空", name="破底+Delta確認", fn=s_ts2, exit=EXIT_TREND, fp_core=False,
         grid=dict(n=[24, 48, 96], dmin=[0, 10, 20])),
    dict(id="TS3", side="順勢空", name="HTF動能空", fn=s_ts3, exit=EXIT_TREND, fp_core=False,
         grid=dict(lvl=[45, 50], malen=[10, 20])),
    dict(id="TS4", side="順勢空", name="週VWAP失守", fn=s_ts4, exit=EXIT_TREND, fp_core=False,
         grid=dict(_=[0])),
    dict(id="TS5", side="順勢空", name="DXY強勢順空", fn=s_ts5, exit=EXIT_TREND, fp_core=False,
         grid=dict(dxy=[55, 60, 65])),
    dict(id="RS1", side="逆勢空", name="買方衰竭長上影", fn=s_rs1, exit=EXIT_REV, fp_core=False,
         grid=dict(ws=[0.5, 0.6], m=[1.5, 2.0])),
    dict(id="RS2", side="逆勢空", name="RSI超買+Delta轉弱", fn=s_rs2, exit=EXIT_REV, fp_core=False,
         grid=dict(th=[70, 75], det=[0, 5])),
    dict(id="RS3", side="逆勢空", name="VWAP乖離過熱", fn=s_rs3, exit=EXIT_REV, fp_core=False,
         grid=dict(d=[2.0, 3.0])),
    dict(id="RS4", side="逆勢空", name="頂部出貨(FP)", fn=s_rs4, exit=EXIT_REV, fp_core=True,
         grid=dict(poc=[65, 70], dneg=[5.0, 10.0])),
    dict(id="RS5", side="逆勢空", name="假突破反轉", fn=s_rs5, exit=EXIT_REV, fp_core=False,
         grid=dict(fb=[0.0, 0.1, 0.2])),
]


def optimize_strategy(df, ctx, spec, cut_time):
    keys = list(spec["grid"].keys())
    rows = []
    for vals in itertools.product(*(spec["grid"][k] for k in keys)):
        p = dict(zip(keys, vals))
        sig = spec["fn"](df, ctx, p).fillna(False)
        tr = run_engine_short(df, sig, spec["exit"]["sl"], spec["exit"]["tp1"],
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
    print(f"  Footprint 欄位：{'✅' if ctx['has_fp'] else '❌（RS4 將無訊號）'}")
    cut_time = df["time"].iloc[int(len(df) * IS_RATIO)]
    print(f"  IS/OOS 切分點：{cut_time}（⚠️ 全期大多頭，空單重點看 OOS 段）")

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

    ranked = sorted([(k, v) for k, v in results.items() if v["best"]],
                    key=lambda kv: kv[1]["best"]["wr"], reverse=True)
    body = ""
    for sid, v in ranked:
        b = v["best"]
        q = "✅" if b["qualified"] else "⚠️"
        fpnote = "（FP，資料僅2026/4起）" if v["fp_core"] else ""
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
<title>Lab10-Short — 10空單策略參數最佳化</title>{css}</head><body><div class="wrap">
<h1>Lab10-Short — 10 個空單候選策略（第一版）</h1>
<p class="note">生成：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
資料 {len(df)} 根 30m（{df['time'].min().date()} → {df['time'].max().date()}）·
空單引擎=已驗證多單引擎之鏡像 · 出場 Profit Flyer 鏡像（順勢空=S1檔位/逆勢空=S2檔位）·
門檻：n≥{MIN_TRADES}（FP≥{MIN_TRADES_FP}）且 min(IS,OOS) PF≥{MIN_SCORE_PF}，按勝率排名</p>
<div class="warn"><strong>⚠️ 逆風警告：</strong>2024-2026 黃金為大多頭，空單全樣本數字天生受壓——
OOS 段（2025/10 起，含 2026H1 修正）比全樣本更有參考價值；
另匯出指標 V1 缺上緣失衡欄位，trapped-buyers 類條件需匯出指標 V2。</div>
<div class="card">
<table class="tbl"><thead><tr><th>策略</th><th>邏輯</th><th>最佳參數</th><th>筆數</th>
<th>勝率</th><th>PF</th><th>淨利R</th><th>OOS n/PF</th><th>門檻</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="note">對應 Pine：<code>XAUUSD-Short-Lab10-V1.pine</code>（下拉選單，最佳參數已設為預設值）。</p>
</div>
</div></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ 報告：{OUT_HTML}")


if __name__ == "__main__":
    main()
