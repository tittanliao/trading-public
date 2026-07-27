"""
run_bull_short_lab.py — 多頭市場下的空單：出場結構 × Regime Gate 掃描
==============================================================================
輸出：xauusd/XAUUSD-Short-Lab/report_bull_short.html + bull_short_results.json

背景（20260712）：Lab10-Short 全樣本 0/10 過門檻。本實驗檢驗核心假設：
  「空單在多頭下輸，不是進場錯，而是出場結構照抄多單錯了」
多頭下的下跌是急促短命的修正——空單應該短打（小TP、數小時內強制離場），
而不是用多單的 Profit Flyer 檔位（TP2 4R、持倉上限24h）硬扛上升 drift。

三維掃描：
  觸發（5個 Lab10-Short 最有望的進場，參數固定用先前最佳值）
    × 出場結構（SL/TP1/TP2/時間出場 短打網格，含原版48bars對照）
    × Regime Gate（無 / 4H MA下方 / DXY強 / 週VWAP下方 / 亞盤時段 / MA+DXY組合）

執行方式（在 trading/ 根目錄）：
  python3.12 xauusd/scripts/run_bull_short_lab.py --export "xauusd/csv/XAUUSD-S1S2-Export/FX_IDC_XAUUSD, 30 (7).csv"
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
from run_lab10_short_optimizer import run_engine_short

OUT_DIR = ROOT / "xauusd/XAUUSD-Short-Lab"
OUT_HTML = OUT_DIR / "report_bull_short.html"
OUT_JSON = OUT_DIR / "bull_short_results.json"

MIN_TRADES = 40
MIN_TRADES_FP = 15
IS_RATIO = 0.7

# 出場網格：短打檔位 + 一組原版對照（1%/2R/4R/48）
EXIT_GRID = [dict(sl=sl, tp1=t1, tp2=t2, out=o)
             for sl in (0.005, 0.008)
             for t1 in (1.0, 1.5)
             for t2 in (2.0, 3.0)
             for o in (8, 16, 48)]
EXIT_GRID.append(dict(sl=0.010, tp1=2.0, tp2=4.0, out=48))  # 原版對照


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    args = ap.parse_args()

    print(f"載入：{args.export}")
    df = base.load_export(Path(args.export))
    ctx = base.build_common(df)
    print(f"  {len(df)} 根 30m；Footprint {'✅' if ctx['has_fp'] else '❌'}")
    cut_time = df["time"].iloc[int(len(df) * IS_RATIO)]

    # ── 觸發（參數 = Lab10-Short 各自最佳）────────────────────────────────────
    rsi14 = ctx["rsi14"]
    ma240_10 = base.htf_indicator(df, 240, lambda h: h["close"].rolling(10).mean())
    ma240_20 = base.htf_indicator(df, 240, lambda h: h["close"].rolling(20).mean())
    is_black = df["close"] < df["open"]
    dr = ctx["FP_DeltaRatio"] if ctx["has_fp"] else pd.Series(np.nan, index=df.index)
    fp_ok = ctx["fp_ok_mask"] if ctx["has_fp"] else pd.Series(False, index=df.index)

    ll96 = df["low"].rolling(96).min().shift(1)
    ts2 = (df["close"] < ll96) & (dr <= -20).where(fp_ok, True).fillna(True)

    ts3 = (rsi14 < 45) & (rsi14.shift(1) >= 45) & (df["close"] < ma240_10)

    ts5 = (ctx["dxy_rsi"] >= 60) & (df["close"] < ma240_20) & is_black

    if ctx["has_fp"]:
        poc = ctx["FP_POCpos"]
        rs4 = (poc >= 65) & (dr <= -5) & (df["close"] < (df["high"] + df["low"]) / 2) & fp_ok
    else:
        rs4 = pd.Series(False, index=df.index)

    hh48 = df["high"].rolling(48).max().shift(1)
    rs5 = (df["high"] > hh48) & (df["close"] < hh48 * (1 - 0.002)) & is_black

    TRIGGERS = {
        "TS2破底+Δ-20": (ts2, False),
        "TS3動能空45": (ts3, False),
        "TS5 DXY60順空": (ts5, False),
        "RS4頂部出貨FP": (rs4, True),
        "RS5假突破0.2": (rs5, False),
    }

    # ── Regime Gates ────────────────────────────────────────────────────────────
    hour = df["time"].dt.hour
    GATES = {
        "無gate": pd.Series(True, index=df.index),
        "4H MA20下方": df["close"] < ma240_20,
        "DXY RSI≥55": ctx["dxy_rsi"] >= 55,
        "週VWAP下方": df["close"] < ctx["wvwap"],
        "亞盤(07-16)": (hour >= 7) & (hour < 16),
        "MA下方+DXY≥55": (df["close"] < ma240_20) & (ctx["dxy_rsi"] >= 55),
    }

    # ── 掃描 ────────────────────────────────────────────────────────────────────
    rows = []
    total = len(TRIGGERS) * len(GATES) * len(EXIT_GRID)
    print(f"  掃描 {total} 個組合（觸發{len(TRIGGERS)} × gate{len(GATES)} × 出場{len(EXIT_GRID)}）...")
    for (tname, (tsig, fp_core)), (gname, gate) in itertools.product(TRIGGERS.items(), GATES.items()):
        sig = (tsig & gate).fillna(False)
        if sig.sum() == 0:
            continue
        for ex in EXIT_GRID:
            tr = run_engine_short(df, sig, ex["sl"], ex["tp1"], ex["tp2"], ex["out"])
            full = base.stat_block(tr)
            if full["n"] == 0:
                continue
            tr["entry_time"] = pd.to_datetime(tr["entry_time"])
            is_ = base.stat_block(tr[tr.entry_time < cut_time])
            oos = base.stat_block(tr[tr.entry_time >= cut_time])
            score = min(is_["pf"] if np.isfinite(is_["pf"]) else 0,
                        oos["pf"] if np.isfinite(oos["pf"]) else 0)
            if fp_core:
                # FP觸發僅近期有資料 → IS 無樣本，score 改用 OOS PF（標註）
                score = oos["pf"] if np.isfinite(oos["pf"]) else 0
            min_n = MIN_TRADES_FP if fp_core else MIN_TRADES
            rows.append(dict(trigger=tname, gate=gname,
                             exit=f"SL{ex['sl']*100:.1f}%/TP{ex['tp1']}/{ex['tp2']}R/{ex['out']}bars",
                             out_k=ex["out"], n=full["n"], wr=full["wr"], pf=full["pf"],
                             net_r=full["net_r"], is_pf=is_["pf"], oos_n=oos["n"],
                             oos_pf=oos["pf"], score=score, fp_core=fp_core,
                             qualified=(full["n"] >= min_n and score >= 1.2)))
    print(f"  完成：{len(rows)} 個有訊號的組合")

    qual = [r for r in rows if r["qualified"]]
    qual.sort(key=lambda r: (r["score"], r["wr"]), reverse=True)
    print(f"  ✅ 過門檻（n達標且 score PF≥1.2）：{len(qual)} 個")
    for r in qual[:12]:
        print(f"    {r['trigger']} + {r['gate']} + {r['exit']}: n={r['n']} WR={r['wr']:.1f}% "
              f"PF={r['pf']:.2f} score={r['score']:.2f}")

    # 核心假設檢驗：短打 vs 原版出場（同觸發同gate下，out8/16 vs out48 的平均PF）
    dfres = pd.DataFrame(rows)
    hyp = dfres[~dfres.fp_core].groupby(dfres.out_k.map(lambda o: "短打(≤16bars)" if o <= 16 else "長抱(48bars)")).agg(
        mean_pf=("pf", "mean"), median_pf=("pf", "median"), n_combos=("pf", "count"))
    print("\n  核心假設檢驗（非FP觸發，全組合平均）：")
    print(hyp.to_string())

    OUT_JSON.write_text(json.dumps(dict(qualified=qual[:60], hypothesis=hyp.reset_index().to_dict("records")),
                                    ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # ── HTML ─────────────────────────────────────────────────────────────────────
    def tbl(rs):
        body = ""
        for i, r in enumerate(rs[:30]):
            hl = " style='background:rgba(34,197,94,.08)'" if i < 3 else ""
            fpn = "（FP:score=OOS）" if r["fp_core"] else ""
            body += (f"<tr{hl}><td>{i+1}</td><td>{r['trigger']}{fpn}</td><td>{r['gate']}</td>"
                     f"<td>{r['exit']}</td><td>{r['n']}</td><td><strong>{r['wr']:.1f}%</strong></td>"
                     f"<td>{base.fmt(r['pf'])}</td><td>{r['net_r']:.0f}R</td>"
                     f"<td>{r['oos_n']}/{base.fmt(r['oos_pf'])}</td><td><strong>{base.fmt(r['score'])}</strong></td></tr>")
        return (f"<table class='tbl'><thead><tr><th>#</th><th>觸發</th><th>Gate</th><th>出場</th>"
                f"<th>n</th><th>WR</th><th>PF</th><th>淨R</th><th>OOS n/PF</th><th>score</th></tr></thead>"
                f"<tbody>{body}</tbody></table>")
    hyp_rows = "".join(f"<tr><td>{i}</td><td>{r['mean_pf']:.3f}</td><td>{r['median_pf']:.3f}</td><td>{int(r['n_combos'])}</td></tr>"
                        for i, r in hyp.iterrows())
    css = """<style>*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
.wrap{max-width:1250px;margin:0 auto}h1{font-size:1.5em;margin-bottom:6px;color:#f8fafc}
h2{font-size:1.05em;color:#38bdf8;margin:8px 0 12px;padding-bottom:6px;border-bottom:1px solid #334155}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:22px;margin-bottom:18px}
.tbl{width:100%;border-collapse:collapse;font-size:.83em;margin:10px 0}
.tbl th{background:#0f172a;color:#94a3b8;padding:8px 10px;text-align:left;border-bottom:1px solid #334155}
.tbl td{padding:7px 10px;border-bottom:1px solid rgba(51,65,85,.5)}
.note{font-size:.82em;color:#94a3b8;margin-top:8px}</style>"""
    html = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8">
<title>多頭下的空單 — 出場×Gate掃描</title>{css}</head><body><div class="wrap">
<h1>多頭市場下的空單：出場結構 × Regime Gate 掃描</h1>
<p class="note">生成：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ·
假設檢驗：「空單輸在出場照抄多單，不是進場」· 觸發參數固定（Lab10-Short 最佳），
掃 出場短打網格 × 6 種 regime gate · 門檻 n≥{MIN_TRADES}（FP≥{MIN_TRADES_FP}）且 score PF≥1.2</p>
<div class="card"><h2>核心假設檢驗：短打 vs 長抱（非FP觸發全組合平均）</h2>
<table class="tbl"><thead><tr><th>出場類型</th><th>平均PF</th><th>中位PF</th><th>組合數</th></tr></thead>
<tbody>{hyp_rows}</tbody></table></div>
<div class="card"><h2>過門檻組合排名（共 {len(qual)} 個）</h2>{tbl(qual)}
<p class="note">FP 觸發（RS4）因資料僅 2026/4 起、IS 無樣本，score 直接用 OOS PF（無 IS 對照，可信度打折）。</p></div>
</div></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ 報告：{OUT_HTML}")


if __name__ == "__main__":
    main()
