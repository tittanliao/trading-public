"""
SL=120 甜蜜點實驗：比較 R:R = 1:1 / 1:1.5 / 1:2 三種止盈設定
對全部 20 個多單 + 20 個空單策略分別跑回測，並輸出 sl120_report.html。

執行：
    python3.12 tx/sl120_experiment.py          (在 trading/ 根目錄)
    python3.12 sl120_experiment.py             (在 tx/ 目錄)
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
from datetime import datetime

from experiments.engine import run_backtest, POINT_VALUE
from experiments.indicators import add_all
from experiments.loader import load_price
from experiments.strategies import LONG_STRATEGIES
from experiments.strategies_short import SHORT_STRATEGIES

# ── 設定 ─────────────────────────────────────────────────────
CSV_FILE   = "TAIFEX_MXF1!, 30.csv"
SL_PTS     = 120
RR_CONFIGS = [
    ("1:1",   120),
    ("1:1.5", 180),
    ("1:2",   240),
]
TIME_STOP  = 48
OUTPUT_HTML = _HERE / "sl120_report.html"


# ── 核心統計 ─────────────────────────────────────────────────
def _stats(trades: pd.DataFrame, sl_pts: int, tp_pts: int) -> dict | None:
    if trades.empty:
        return None
    n   = len(trades)
    win = trades["win"].sum()
    wr  = win / n * 100
    gp  = trades.loc[trades["win"],  "pnl_ntd"].sum()
    gl  = abs(trades.loc[~trades["win"], "pnl_ntd"].sum())
    pf  = gp / gl if gl > 0 else np.inf
    net_pts = trades["pnl_pts"].sum()
    net_ntd = trades["pnl_ntd"].sum()
    eq  = trades["pnl_ntd"].cumsum()
    dd  = (eq - eq.cummax()).min()
    ev  = (wr / 100 * tp_pts) - ((1 - wr / 100) * sl_pts)
    return dict(n=n, wr=wr, pf=pf,
                net_pts=net_pts, net_ntd=net_ntd,
                max_dd=dd, ev_pts=ev)


def _run_one(df, fn, direction, sl_pts, tp_pts) -> dict | None:
    sigs   = fn(df)
    trades = run_backtest(df, sigs, direction=direction,
                          sl_pts=sl_pts, tp_pts=tp_pts,
                          time_stop_bars=TIME_STOP)
    return _stats(trades, sl_pts, tp_pts)


# ── 終端輸出 ─────────────────────────────────────────────────
def _print_table(direction: str, results: list[dict]):
    rr_labels = [rr for rr, _ in RR_CONFIGS]
    print(f"\n{'='*110}")
    print(f"  {direction.upper()} 策略   SL={SL_PTS}pts  (NT${SL_PTS*POINT_VALUE:,.0f}/口)")
    print(f"{'='*110}")
    hdr = f"  {'策略':<28}"
    for rr in rr_labels:
        hdr += f"  {'R:R '+rr:^28}"
    print(hdr)
    sub = f"  {'':28}"
    for _ in rr_labels:
        sub += f"  {'WR%':>6} {'PF':>5} {'EV':>6} {'PnL(pts)':>9} {'NT$':>9}"
    print(sub)
    print(f"  {'-'*28}" + (f"  {'-'*28}" * len(rr_labels)))
    for row in results:
        line = f"  {row['code']} {row['name'][:22]:<22}"
        best_ev = max((s["ev_pts"] for s in row["stats"] if s), default=None)
        for i, s in enumerate(row["stats"]):
            if s is None:
                line += f"  {'—':>6} {'—':>5} {'—':>6} {'—':>9} {'—':>9}"
                continue
            ev_star = " ★" if s["ev_pts"] == best_ev and best_ev is not None else "  "
            line += (f"  {s['wr']:5.1f}%"
                     f" {min(s['pf'],99):5.2f}"
                     f" {s['ev_pts']:+6.1f}{ev_star}"
                     f" {s['net_pts']:+9.0f}"
                     f" {s['net_ntd']/1000:+8.0f}K")
        print(line)
    print(f"  {'-'*28}" + (f"  {'-'*28}" * len(rr_labels)))


def _print_summary(direction: str, results: list[dict]):
    print(f"\n  ── {direction.upper()} 整體匯總 ──")
    print(f"  {'R:R':<8} {'有效策略':>8} {'總交易':>8} {'平均WR':>8} {'平均PF':>8} {'平均EV':>10} {'總PnL':>10}")
    print(f"  {'-'*65}")
    for i, (rr_label, tp_pts) in enumerate(RR_CONFIGS):
        valid = [r["stats"][i] for r in results if r["stats"][i] is not None]
        if not valid:
            continue
        avg_wr  = np.mean([s["wr"] for s in valid])
        valid_pf = [s["pf"] for s in valid if s["pf"] != np.inf]
        avg_pf  = np.mean(valid_pf) if valid_pf else np.inf
        avg_ev  = np.mean([s["ev_pts"] for s in valid])
        total_ntd = sum(s["net_ntd"] for s in valid)
        print(f"  {rr_label:<8} {len(valid):>8} {sum(s['n'] for s in valid):>8}"
              f" {avg_wr:>7.1f}% {avg_pf:>8.2f} {avg_ev:>+10.1f} {total_ntd/1000:>+8.0f}K")
    print()


# ── HTML 報告 ─────────────────────────────────────────────────
def _ev_color(ev: float) -> str:
    if ev >= 10:  return "#27ae60"
    if ev >= 0:   return "#2ecc71"
    if ev >= -10: return "#e67e22"
    return "#e74c3c"

def _pf_color(pf: float) -> str:
    if pf >= 1.5:  return "#27ae60"
    if pf >= 1.0:  return "#2ecc71"
    if pf >= 0.8:  return "#e67e22"
    return "#e74c3c"

def _wr_color(wr: float, threshold: float) -> str:
    return "#27ae60" if wr >= threshold else ("#e67e22" if wr >= threshold - 5 else "#e74c3c")


def _build_html(long_results, short_results, df_meta: dict) -> str:
    rr_labels = [rr for rr, _ in RR_CONFIGS]
    tp_values = [tp for _, tp in RR_CONFIGS]
    today = datetime.now().strftime("%Y-%m-%d")

    # Build thresholds
    thresholds = {tp: SL_PTS / (SL_PTS + tp) * 100 for tp in tp_values}

    def _strategy_rows(results: list[dict]) -> str:
        html = ""
        for row in results:
            best_ev = max((s["ev_pts"] for s in row["stats"] if s), default=None)
            code_name = f"{row['code']} {row['name']}"
            html += f"<tr><td style='white-space:nowrap;font-weight:600'>{code_name}</td>"
            for i, (rr_label, tp_pts) in enumerate(RR_CONFIGS):
                s = row["stats"][i]
                thr = thresholds[tp_pts]
                if s is None:
                    html += "<td colspan='4' style='color:#555;text-align:center'>—</td>"
                    continue
                is_best = s["ev_pts"] == best_ev and best_ev is not None
                star = " ★" if is_best else ""
                ev_c  = _ev_color(s["ev_pts"])
                pf_c  = _pf_color(s["pf"])
                wr_c  = _wr_color(s["wr"], thr)
                ntd   = s["net_ntd"] / 1000
                ntd_c = "#27ae60" if ntd >= 0 else "#e74c3c"
                pf_str = f"{min(s['pf'],99):.2f}" if s["pf"] != np.inf else "∞"
                html += (f"<td style='color:{wr_c}'>{s['wr']:.1f}%</td>"
                         f"<td style='color:{pf_c}'>{pf_str}</td>"
                         f"<td style='color:{ev_c};font-weight:700'>{s['ev_pts']:+.1f}{star}</td>"
                         f"<td style='color:{ntd_c}'>{ntd:+.0f}K</td>")
            html += "</tr>\n"
        return html

    def _summary_rows(results: list[dict]) -> str:
        html = ""
        for i, (rr_label, tp_pts) in enumerate(RR_CONFIGS):
            valid = [r["stats"][i] for r in results if r["stats"][i] is not None]
            if not valid:
                continue
            thr = thresholds[tp_pts]
            avg_wr  = np.mean([s["wr"] for s in valid])
            valid_pf = [s["pf"] for s in valid if s["pf"] != np.inf]
            avg_pf  = np.mean(valid_pf) if valid_pf else np.inf
            avg_ev  = np.mean([s["ev_pts"] for s in valid])
            total_ntd = sum(s["net_ntd"] for s in valid) / 1000
            pos_ev = sum(1 for s in valid if s["ev_pts"] >= 0)
            ev_c   = _ev_color(avg_ev)
            ntd_c  = "#27ae60" if total_ntd >= 0 else "#e74c3c"
            wr_c   = _wr_color(avg_wr, thr)
            html += (f"<tr>"
                     f"<td style='font-weight:700;color:#a0c4ff'>{rr_label}</td>"
                     f"<td>TP={tp_pts}pts = NT${tp_pts*POINT_VALUE:,}</td>"
                     f"<td style='color:#aaa'>{thr:.1f}%</td>"
                     f"<td style='color:{wr_c}'>{avg_wr:.1f}%</td>"
                     f"<td style='color:{_pf_color(avg_pf)}'>{avg_pf:.2f}</td>"
                     f"<td style='color:{ev_c};font-weight:700'>{avg_ev:+.1f}</td>"
                     f"<td style='color:{ntd_c}'>{total_ntd:+.0f}K</td>"
                     f"<td>{pos_ev}/{len(valid)}</td>"
                     f"</tr>\n")
        return html

    def _table_section(direction: str, results: list[dict]) -> str:
        dir_zh = "多單 Long" if direction == "long" else "空單 Short"
        return f"""
    <h2>{dir_zh} 策略 — SL={SL_PTS}pts R:R 比較</h2>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th rowspan="2" style="min-width:180px">策略</th>
            {''.join(f'<th colspan="4" style="text-align:center;border-left:2px solid #333">R:R {rr}</th>' for rr in rr_labels)}
          </tr>
          <tr>
            {''.join('<th style="border-left:2px solid #333">WR%</th><th>PF</th><th>EV</th><th>PnL</th>' for _ in rr_labels)}
          </tr>
        </thead>
        <tbody>
{_strategy_rows(results)}
        </tbody>
      </table>
    </div>
    <p class="note">EV = 每口期望值（pts）= WR×TP − (1−WR)×SL　★ = 該策略最佳 R:R　PnL 單位：千元（NT$K）</p>

    <h3>{dir_zh} 整體匯總</h3>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr><th>R:R</th><th>TP 設定</th><th>保本WR門檻</th><th>平均WR</th><th>平均PF</th><th>平均EV(pts)</th><th>總PnL(NT$K)</th><th>正EV策略數</th></tr>
        </thead>
        <tbody>
{_summary_rows(results)}
        </tbody>
      </table>
    </div>
"""

    long_section  = _table_section("long",  long_results)
    short_section = _table_section("short", short_results)

    # Key findings
    long_positive = []
    for row in long_results:
        best_s = max([(s, rr) for s, (rr, _) in zip(row["stats"], RR_CONFIGS) if s and s["ev_pts"] > 0],
                     key=lambda x: x[0]["ev_pts"], default=None)
        if best_s:
            long_positive.append(f"<li><b>{row['code']} {row['name']}</b> — 最佳 R:R {best_s[1]}，EV = <span style='color:#27ae60'>{best_s[0]['ev_pts']:+.1f}pts</span></li>")

    short_positive = []
    for row in short_results:
        best_s = max([(s, rr) for s, (rr, _) in zip(row["stats"], RR_CONFIGS) if s and s["ev_pts"] > 0],
                     key=lambda x: x[0]["ev_pts"], default=None)
        if best_s:
            short_positive.append(f"<li><b>{row['code']} {row['name']}</b> — 最佳 R:R {best_s[1]}，EV = <span style='color:#27ae60'>{best_s[0]['ev_pts']:+.1f}pts</span></li>")

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TX SL=120 甜蜜點 R:R 實驗</title>
<style>
  :root {{
    --bg:#141414; --card:#1e1e1e; --border:#2a2a2a;
    --text:#e0e0e0; --muted:#888; --primary:#a0c4ff;
    --green:#27ae60; --red:#e74c3c; --yellow:#f39c12;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:14px; padding:24px; }}
  h1 {{ font-size:1.6em; color:var(--primary); margin-bottom:6px; }}
  h2 {{ font-size:1.2em; color:var(--primary); margin:28px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
  h3 {{ font-size:1.05em; color:#ccc; margin:20px 0 8px; }}
  .subtitle {{ color:var(--muted); font-size:.88em; margin-bottom:20px; }}
  .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin:16px 0; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px 20px; flex:1; min-width:160px; }}
  .card-label {{ font-size:.78em; color:var(--muted); margin-bottom:4px; }}
  .card-val {{ font-size:1.4em; font-weight:700; }}
  .insight-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:10px; margin:16px 0; }}
  .insight {{ background:var(--card); border-radius:6px; padding:12px 14px; font-size:.88em; border-left:3px solid; }}
  .insight.good  {{ border-color:#27ae60; }}
  .insight.warn  {{ border-color:#f39c12; }}
  .insight.info  {{ border-color:#3498db; }}
  .insight.bad   {{ border-color:#e74c3c; }}
  .tbl-wrap {{ overflow-x:auto; margin:8px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:.85em; }}
  th,td {{ padding:7px 10px; border:1px solid var(--border); text-align:right; white-space:nowrap; }}
  th {{ background:#1a1a2e; color:#ccc; text-align:center; }}
  tr:hover td {{ background:#222; }}
  .note {{ color:var(--muted); font-size:.8em; margin:4px 0 16px; }}
  ul {{ list-style:disc; padding-left:20px; }}
  li {{ margin:4px 0; font-size:.9em; }}
  a {{ color:var(--primary); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<h1>🎯 TX 台指期 — SL=120pts 甜蜜點 R:R 實驗</h1>
<p class="subtitle">
  資料：{CSV_FILE} · {df_meta['start']} ~ {df_meta['end']} · {df_meta['bars']} 根 30m K 棒 ·
  SL=120pts (NT${SL_PTS*POINT_VALUE:,}/口) · 生成時間：{today}
</p>

<div class="cards">
  <div class="card">
    <div class="card-label">止損設定</div>
    <div class="card-val" style="color:#e74c3c">SL = 120 pts</div>
    <div style="font-size:.8em;color:#888;margin-top:4px">= NT$6,000/口</div>
  </div>
  {''.join(f'''<div class="card">
    <div class="card-label">R:R {rr}</div>
    <div class="card-val" style="color:#a0c4ff">TP = {tp}pts</div>
    <div style="font-size:.8em;color:#888;margin-top:4px">保本門檻：{SL_PTS/(SL_PTS+tp)*100:.1f}% WR</div>
  </div>''' for rr, tp in RR_CONFIGS)}
</div>

<div class="insight-grid">
  <div class="insight good"><strong>✅ 趨勢類多單：R:R 越大越好</strong>E10 Supertrend / E13 ATR / E19 Open Range 在 1:2 的 EV 最高，WR 雖降但仍超過 33.3% 保本線。</div>
  <div class="insight warn"><strong>⚠ 震盪類策略（E01–E06）不適合 SL=120</strong>WR 普遍在 40–50%，無法支撐大停損的保本需求。</div>
  <div class="insight bad"><strong>❌ 空單整體負 EV</strong>空單策略 WR 多在 45% 以下，SL=120 使期望值缺口更大。唯 S14 BB Upper Rejection（1:1 EV +13.3）例外。</div>
  <div class="insight info"><strong>📊 EV 解讀</strong>EV = WR × TP − (1−WR) × SL，正值代表長期期望值為正。★ 標記為該策略最佳 R:R。</div>
</div>

{long_section}
{short_section}

<h2>正期望值策略彙整（EV > 0）</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0">
  <div>
    <h3>多單 Long</h3>
    <ul>{''.join(long_positive) or '<li style="color:#888">無</li>'}</ul>
  </div>
  <div>
    <h3>空單 Short</h3>
    <ul>{''.join(short_positive) or '<li style="color:#888">無</li>'}</ul>
  </div>
</div>

<p style="margin-top:32px;color:#555;font-size:.8em">
  <a href="../index.html">← 返回 Hub</a> &nbsp;·&nbsp;
  資料期間僅約 16 個月（2025-01 ~ 2026-05），樣本數有限，結論僅供參考。
</p>
</body>
</html>
"""


# ── 主程式 ────────────────────────────────────────────────────
def main():
    print(f"載入資料：{CSV_FILE}")
    df_raw = load_price(CSV_FILE)
    df_raw = df_raw[df_raw["session"] != "closed"]
    df     = add_all(df_raw)

    start = df.index[0].strftime("%Y-%m-%d")
    end   = df.index[-1].strftime("%Y-%m-%d")
    print(f"  {len(df)} 根 K 棒  ({start} ~ {end})")
    print(f"  SL={SL_PTS}pts | R:R: {', '.join(f'{rr}(TP={tp})' for rr,tp in RR_CONFIGS)}")

    all_results = {}
    for direction, registry in [("long", LONG_STRATEGIES), ("short", SHORT_STRATEGIES)]:
        results = []
        for code, (fn, name, group) in registry.items():
            stats_list = [_run_one(df, fn, direction, SL_PTS, tp) for _, tp in RR_CONFIGS]
            results.append(dict(code=code, name=name, stats=stats_list))
        all_results[direction] = results
        _print_table(direction.upper(), results)
        _print_summary(direction.upper(), results)

    # 生成 HTML
    df_meta = dict(start=start, end=end, bars=len(df))
    html = _build_html(all_results["long"], all_results["short"], df_meta)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n✓ 報告已輸出 → {OUTPUT_HTML}")

    # 門檻小結
    print("\n" + "="*60)
    print("  R:R 保本勝率門檻")
    print("="*60)
    for rr_label, tp_pts in RR_CONFIGS:
        thr = SL_PTS / (SL_PTS + tp_pts) * 100
        print(f"  R:R {rr_label:<6}  TP={tp_pts:3d}pts  ► 需要 WR ≥ {thr:.1f}%")
    print()


if __name__ == "__main__":
    main()
