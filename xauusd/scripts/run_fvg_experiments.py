#!/usr/bin/env python3
"""
FVG (Fair Value Gap) 策略參數最佳化
對應 Pine Script: XAUUSD-FVG-V1.0.pine

用法（20260705 移至 scripts/ 子資料夾，仍需在 trading/ 根目錄執行）:
    python3.12 xauusd/scripts/run_fvg_experiments.py

輸出:
    xauusd/XAUUSD-FVG-Strategy/optimization_results.json
"""
import pandas as pd
import numpy as np
import json
import itertools
import time
from pathlib import Path

# ── 路徑 ─────────────────────────────────────────────────────────────────────
CSV_30M    = Path("xauusd/csv/FX_IDC_XAUUSD, 30.csv")
OUTPUT_DIR = Path("xauusd/XAUUSD-FVG-Strategy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 參數網格 ─────────────────────────────────────────────────────────────────
MIN_TRADES   = 20                         # 交易筆數下限（避免過擬合）
DIRECTIONS   = ['Long', 'Short']
FVG_MIN_PCT  = [0.03, 0.05, 0.10, 0.20]  # 最小 FVG 大小 (%)
FVG_MAX_BARS = [20, 50, 100]              # FVG 最長有效 K 棒數
TP1_R_LIST   = [0.5, 1.0, 1.5, 2.0]     # TP1 R 倍數
TP2_R_LIST   = [2.0, 3.0, 4.0, 5.0]     # TP2 R 倍數
TB_LIST      = [24, 48, 72]              # 時間止損 K 棒數

# SL 設定（Natural 和 Fixed 分開列舉）
SL_CONFIGS = [
    {'sl_type': 'FVG Natural', 'sl_buffer': 0.00, 'sl_pct': 0.8},
    {'sl_type': 'FVG Natural', 'sl_buffer': 0.05, 'sl_pct': 0.8},
    {'sl_type': 'FVG Natural', 'sl_buffer': 0.10, 'sl_pct': 0.8},
    {'sl_type': 'Fixed %',     'sl_buffer': 0.05, 'sl_pct': 0.5},
    {'sl_type': 'Fixed %',     'sl_buffer': 0.05, 'sl_pct': 0.8},
    {'sl_type': 'Fixed %',     'sl_buffer': 0.05, 'sl_pct': 1.0},
    {'sl_type': 'Fixed %',     'sl_buffer': 0.05, 'sl_pct': 1.5},
]

# ── 資料載入 ─────────────────────────────────────────────────────────────────
def load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df['time'] = (pd.to_datetime(df['time'], utc=True)
                  .dt.tz_convert('Asia/Taipei')
                  .dt.tz_localize(None))
    return df.sort_values('time').reset_index(drop=True)

# ── 單次回測 ─────────────────────────────────────────────────────────────────
def run_backtest(price: pd.DataFrame,
                 direction:    str,
                 fvg_min_pct:  float,
                 fvg_max_bars: int,
                 sl_type:      str,
                 sl_pct:       float,
                 sl_buffer:    float,
                 tp1_r:        float,
                 tp2_r:        float,
                 tb:           int) -> dict | None:
    """
    FVG 回測引擎，邏輯與 XAUUSD-FVG-V1.0.pine 一致：
    - 進場：信號 K 棒收盤後，下一根開盤進場
    - FVG Natural SL：FVG 邊界外加緩衝
    - Profit Flyer：達 TP1/TP2 後拖曳止損
    - 時間止損：超過 tb K 棒後強制平倉
    """
    h, l, c, o = (price['high'].values, price['low'].values,
                  price['close'].values, price['open'].values)
    n       = len(price)
    is_long = (direction == 'Long')
    tb4     = max(1, tb // 4)
    tb2     = max(1, tb // 2)

    # FVG 狀態
    act_top = act_bot = act_bar = None

    # 交易狀態
    in_trade      = False
    pending_entry = False
    locked_bot = locked_top = None
    entry_price = sl_price = tp1_price = tp2_price = None
    entry_bar_idx = None
    tp1_hit       = False
    trades        = []

    for i in range(2, n):

        # ── 1. 上一根信號 → 本根開盤進場 ────────────────────────────────────
        if pending_entry:
            ep = o[i]
            if sl_type == 'FVG Natural':
                sl = (locked_bot * (1 - sl_buffer / 100) if is_long
                      else locked_top * (1 + sl_buffer / 100))
            else:
                sl = ep * (1 - sl_pct / 100) if is_long else ep * (1 + sl_pct / 100)

            r = abs(ep - sl)
            if r < 1e-8:            # SL 距離為零，略過此單
                pending_entry = False
                continue

            entry_price   = ep
            sl_price      = sl
            tp1_price     = ep + r * tp1_r if is_long else ep - r * tp1_r
            tp2_price     = ep + r * tp2_r if is_long else ep - r * tp2_r
            entry_bar_idx = i
            in_trade      = True
            tp1_hit       = False
            pending_entry = False

        # ── 2. 出場檢查 ──────────────────────────────────────────────────────
        if in_trade:
            bars_in    = i - entry_bar_idx
            exit_price = None
            exit_type  = None

            if is_long:
                if l[i] <= sl_price:                    # SL 觸發
                    exit_price, exit_type = sl_price, 'SL'
                else:
                    if c[i] >= tp2_price:               # 拖曳至 TP2
                        sl_price = max(tp2_price, float(np.min(l[max(0, i - tb4):i + 1])))
                        tp1_hit  = True
                    elif c[i] >= tp1_price and not tp1_hit:
                        sl_price = max(tp1_price, float(np.min(l[max(0, i - tb2):i + 1])))
                        tp1_hit  = True
                    if bars_in >= tb:                   # 時間止損
                        if c[i] >= entry_price:
                            exit_price, exit_type = sl_price, 'TimeWin'
                        else:
                            exit_price, exit_type = c[i], 'TimeLoss'
            else:
                if h[i] >= sl_price:                    # SL 觸發
                    exit_price, exit_type = sl_price, 'SL'
                else:
                    if c[i] <= tp2_price:               # 拖曳至 TP2
                        sl_price = min(tp2_price, float(np.max(h[max(0, i - tb4):i + 1])))
                        tp1_hit  = True
                    elif c[i] <= tp1_price and not tp1_hit:
                        sl_price = min(tp1_price, float(np.max(h[max(0, i - tb2):i + 1])))
                        tp1_hit  = True
                    if bars_in >= tb:
                        if c[i] <= entry_price:
                            exit_price, exit_type = sl_price, 'TimeWin'
                        else:
                            exit_price, exit_type = c[i], 'TimeLoss'

            if exit_price is not None:
                pnl = ((exit_price - entry_price) / entry_price * 100 if is_long
                       else (entry_price - exit_price) / entry_price * 100)
                trades.append({'pnl_pct': pnl, 'exit_type': exit_type})
                in_trade = False
                tp1_hit  = False

        # ── 3. FVG 偵測 ──────────────────────────────────────────────────────
        if is_long and l[i] > h[i - 2]:
            sz = (l[i] - h[i - 2]) / c[i] * 100
            if sz >= fvg_min_pct:
                act_top, act_bot, act_bar = l[i], h[i - 2], i
        elif not is_long and h[i] < l[i - 2]:
            sz = (l[i - 2] - h[i]) / c[i] * 100
            if sz >= fvg_min_pct:
                act_top, act_bot, act_bar = l[i - 2], h[i], i

        # FVG 到期 / 失效
        if act_bar is not None and (i - act_bar) > fvg_max_bars:
            act_top = act_bot = act_bar = None
        if act_top is not None:
            if is_long and c[i] < act_bot:
                act_top = act_bot = act_bar = None
            elif not is_long and c[i] > act_top:
                act_top = act_bot = act_bar = None

        # ── 4. 進場信號 ──────────────────────────────────────────────────────
        if (not in_trade and not pending_entry
                and act_top is not None and act_bar is not None
                and i > act_bar):
            sig = ((l[i] <= act_top and c[i] >= act_bot) if is_long
                   else (h[i] >= act_bot and c[i] <= act_top))
            if sig:
                locked_bot, locked_top       = act_bot, act_top
                act_top = act_bot = act_bar  = None   # 消耗 FVG
                pending_entry                = True

    # ── 統計 ─────────────────────────────────────────────────────────────────
    if len(trades) < MIN_TRADES:
        return None

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t['pnl_pct'] > 0]
    loss = df_t[df_t['pnl_pct'] <= 0]
    gp   = wins['pnl_pct'].sum() if len(wins) else 0.0
    gl   = abs(loss['pnl_pct'].sum()) if len(loss) else 1e-9
    pf   = gp / gl
    wr   = len(wins) / len(df_t) * 100
    net  = df_t['pnl_pct'].sum()

    # 複合評分（PF 佔主導，WR 和 Net 補充）
    score = (pf - 1) * 30 + wr * 0.4 + net * 0.3

    return {
        'trades':        len(df_t),
        'win_rate':      round(wr, 1),
        'profit_factor': round(pf, 3),
        'net_pnl_pct':   round(net, 2),
        'avg_win_pct':   round(wins['pnl_pct'].mean(), 3) if len(wins) else 0.0,
        'avg_loss_pct':  round(loss['pnl_pct'].mean(), 3) if len(loss) else 0.0,
        'score':         round(score, 3),
    }

# ── 建立參數集合 ─────────────────────────────────────────────────────────────
def build_param_sets() -> list[dict]:
    sets = []
    for direction, fvg_min, fvg_max, sl_cfg, tp1, tp2, tb in itertools.product(
        DIRECTIONS, FVG_MIN_PCT, FVG_MAX_BARS, SL_CONFIGS, TP1_R_LIST, TP2_R_LIST, TB_LIST
    ):
        if tp2 <= tp1:          # TP2 必須大於 TP1
            continue
        sets.append({
            'direction':    direction,
            'fvg_min_pct':  fvg_min,
            'fvg_max_bars': fvg_max,
            **sl_cfg,
            'tp1_r': tp1,
            'tp2_r': tp2,
            'tb':    tb,
        })
    return sets

# ── 輸出表格 ─────────────────────────────────────────────────────────────────
def print_top(df: pd.DataFrame, direction: str, n: int = 10):
    sub = df[df['direction'] == direction].nlargest(n, 'score').reset_index(drop=True)
    if sub.empty:
        print(f"  [{direction}] 無有效結果")
        return
    bar = '=' * 95
    print(f"\n{bar}")
    print(f"  TOP {n} — {direction}")
    print(bar)
    hdr = (f"{'#':>3}  {'SL型':>8}  {'SL%':>4}  {'緩衝':>4}  {'FVG%':>5}  "
           f"{'FVGbar':>6}  {'TP1R':>4}  {'TP2R':>4}  {'TB':>4}  "
           f"{'筆':>5}  {'WR%':>6}  {'PF':>6}  {'Net%':>7}  Score")
    print(hdr)
    print('-' * 95)
    for i, r in sub.iterrows():
        sl_lbl = 'Natural' if r['sl_type'] == 'FVG Natural' else 'Fixed'
        print(f"{i+1:>3}  {sl_lbl:>8}  {r['sl_pct']:>4.1f}  {r['sl_buffer']:>4.2f}  "
              f"{r['fvg_min_pct']:>5.2f}  {int(r['fvg_max_bars']):>6}  "
              f"{r['tp1_r']:>4.1f}  {r['tp2_r']:>4.1f}  {int(r['tb']):>4}  "
              f"{int(r['trades']):>5}  {r['win_rate']:>6.1f}  {r['profit_factor']:>6.3f}  "
              f"{r['net_pnl_pct']:>+7.2f}  {r['score']:.2f}")

def print_best(df: pd.DataFrame, direction: str):
    sub = df[df['direction'] == direction]
    if sub.empty:
        return
    b = sub.loc[sub['score'].idxmax()]
    sl_desc = (f"FVG Natural（緩衝 {b['sl_buffer']}%）"
               if b['sl_type'] == 'FVG Natural'
               else f"Fixed {b['sl_pct']}%")
    print(f"\n【{direction} 最佳建議】 Score: {b['score']:.2f}")
    print(f"  FVG Min {b['fvg_min_pct']}%  |  FVG Max {int(b['fvg_max_bars'])} bars")
    print(f"  SL: {sl_desc}")
    print(f"  TP1: {b['tp1_r']}R  |  TP2: {b['tp2_r']}R  |  時間止損: {int(b['tb'])} bars")
    print(f"  結果: {int(b['trades'])} 筆  WR {b['win_rate']}%  "
          f"PF {b['profit_factor']:.3f}  Net {b['net_pnl_pct']:+.2f}%")

# ── 主程式 ───────────────────────────────────────────────────────────────────
def main():
    print("FVG 策略參數最佳化")
    print("=" * 50)

    # 載入資料
    if not CSV_30M.exists():
        print(f"[ERROR] 找不到 CSV: {CSV_30M}")
        return
    print(f"載入 CSV: {CSV_30M}")
    price = load_price(CSV_30M)
    print(f"  {len(price)} bars  ({price['time'].iloc[0].date()} → {price['time'].iloc[-1].date()})")

    # 建立參數集合
    param_sets = build_param_sets()
    print(f"\n參數組合數: {len(param_sets)}")
    print(f"最少交易筆數門檻: {MIN_TRADES}\n")

    # 格線搜索
    results = []
    t0 = time.time()
    for k, params in enumerate(param_sets):
        if k > 0 and k % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / k * (len(param_sets) - k)
            print(f"  {k:>5}/{len(param_sets)}  ({elapsed:.0f}s 已用, ~{eta:.0f}s 剩餘)")
        r = run_backtest(price, **params)
        if r:
            results.append({**params, **r})

    elapsed = time.time() - t0
    print(f"\n完成！有效結果 {len(results)} 組  （耗時 {elapsed:.1f}s）")

    if not results:
        print("無有效結果，請降低 MIN_TRADES 或擴大參數範圍。")
        return

    df = pd.DataFrame(results)
    df = df[df['profit_factor'] >= 1.0]   # 只保留正期望

    # 存檔
    out_path = OUTPUT_DIR / "optimization_results.json"
    df.to_json(out_path, orient='records', indent=2, force_ascii=False)
    print(f"結果已存至: {out_path}")

    # 排名輸出
    for direction in ['Long', 'Short']:
        print_top(df, direction, n=10)

    print(f"\n{'='*60}")
    print("  建議參數（直接貼到 Pine Script）")
    print(f"{'='*60}")
    for direction in ['Long', 'Short']:
        print_best(df, direction)

    # 額外：SL 類型分析
    print(f"\n{'='*60}")
    print("  SL 類型比較（有效結果中位數）")
    print(f"{'='*60}")
    for direction in ['Long', 'Short']:
        sub = df[df['direction'] == direction]
        if sub.empty:
            continue
        grp = sub.groupby('sl_type')[['profit_factor', 'win_rate', 'net_pnl_pct']].median()
        print(f"\n[{direction}]")
        print(grp.to_string())

if __name__ == '__main__':
    main()
