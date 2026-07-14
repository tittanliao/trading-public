"""
XAUUSD Strategy Signal Scanner
Translates Pine Script logic for S1-AweWithBB-V3.4, S2-Hybrid-V2.0, S2-Pullback-V1.9
Outputs current signal status JSON for weekly report integration

Usage:
    python xauusd/signal_scanner.py
    python xauusd/signal_scanner.py --tf 30  # timeframe in minutes (default: 30)
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_DIR = Path(__file__).parent.parent / ".." / ".." / ".." / \
          "googledrive/XAUUSD/weekly report/csv"
# Also try relative paths
ALT_CSV_DIRS = [
    Path("/Users/tittan/googledrive/XAUUSD/weekly report/csv"),
    Path(__file__).parent / "../../googledrive/XAUUSD/weekly report/csv",
]

OUTPUT_JSON = Path(__file__).parent / "signal_status.json"

# S1 parameters (V3.4 defaults)
S1_BB_LEN      = 20
S1_BB_MULT     = 2.0
S1_FAST_EMA    = 3
S1_AO_FAST     = 5
S1_AO_SLOW     = 34
S1_MA1H_LEN    = 3
S1_SL_PCT      = 0.5
S1_TP1_RATIO   = 1.0
S1_TP2_RATIO   = 3.5

# S2-Hybrid parameters (V2.0 defaults)
S2H_RSI_LEN    = 14
S2H_RSI_LIMIT  = 30
S2H_SL_PCT     = 1.0
S2H_TP1_RATIO  = 2.0
S2H_TP2_RATIO  = 4.0

# S2-Pullback parameters (V1.9 defaults)
S2P_TAIL_RATIO = 2.0
S2P_UPPER_RATIO= 0.5
S2P_ATR_MULT   = 0.3
S2P_ATR_LEN    = 14
S2P_SL_PCT     = 1.0

# ── Helpers ────────────────────────────────────────────────────────────────────

def find_csv_dir():
    for d in [CSV_DIR] + ALT_CSV_DIRS:
        if d.exists():
            return d
    raise FileNotFoundError(f"CSV directory not found. Tried: {[str(d) for d in [CSV_DIR]+ALT_CSV_DIRS]}")

def load_csv(csv_dir, prefix, tf_min):
    """Load XAUUSD or DXY CSV for given timeframe."""
    tf_map = {1: "1", 5: "5", 15: "15", 30: "30", 60: "60", 240: "240", 1440: "1D", 10080: "1W"}
    tf_str = tf_map.get(tf_min, str(tf_min))

    candidates = list(csv_dir.glob(f"{prefix}*{tf_str}*.csv")) + \
                 list(csv_dir.glob(f"{prefix}*, {tf_str}.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for prefix={prefix} tf={tf_str} in {csv_dir}")

    path = sorted(candidates)[-1]
    df = pd.read_csv(path, header=0)

    # Standardise column names
    cols = df.columns.tolist()
    rename = {}
    for i, c in enumerate(cols):
        lc = str(c).lower().strip()
        if i == 0 or 'time' in lc:
            rename[c] = 'time'
        elif lc in ('open', 'o'):
            rename[c] = 'open'
        elif lc in ('high', 'h'):
            rename[c] = 'high'
        elif lc in ('low', 'l'):
            rename[c] = 'low'
        elif lc in ('close', 'c'):
            rename[c] = 'close'
    df = df.rename(columns=rename)

    # Parse time
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    return df, str(path.name)

# ── Indicator functions ────────────────────────────────────────────────────────

def calc_bb(close, length=20, mult=2.0):
    basis = close.rolling(length).mean()
    std   = close.rolling(length).std(ddof=0)
    return basis, basis + mult*std, basis - mult*std

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_rsi(close, length=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(length).mean()
    loss  = (-delta.clip(upper=0)).rolling(length).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_ao(high, low, fast=5, slow=34):
    hl2     = (high + low) / 2
    sma_fast = hl2.rolling(fast).mean()
    sma_slow = hl2.rolling(slow).mean()
    return sma_fast - sma_slow

def calc_atr(high, low, close, length=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def ao_state(ao):
    """1=positive&rising, 2=positive&falling, -1=negative&rising, -2=negative&falling"""
    rising = ao > ao.shift(1)
    pos    = ao >= 0
    state  = np.where(pos & rising, 1,
             np.where(pos & ~rising, 2,
             np.where(~pos & rising, -1, -2)))
    return pd.Series(state, index=ao.index)

# ── S1 Scanner ────────────────────────────────────────────────────────────────

def scan_s1(df30, df60=None):
    """Scan S1-AweWithBB-V3.4 on 30m data with optional 1H MA filter."""
    c = df30['close'].copy()
    h = df30['high'].copy()
    l = df30['low'].copy()

    basis, upper, lower = calc_bb(c, S1_BB_LEN, S1_BB_MULT)
    fast_ma = calc_ema(c, S1_FAST_EMA)
    ao      = calc_ao(h, l, S1_AO_FAST, S1_AO_SLOW)
    ao_st   = ao_state(ao)

    crossover = (fast_ma > basis) & (fast_ma.shift(1) <= basis.shift(1))

    # 1H MA filter: resample 60m from 30m data if no 60m df provided
    if df60 is not None:
        ma1h_series = calc_ema(df60['close'], S1_MA1H_LEN)
        # Build lookup: time → ma1h value, then map to 30m bars via merge_asof
        df60_ma = df60[['time']].copy()
        df60_ma['ma1h'] = ma1h_series.values
        df30_tmp = df30[['time']].copy()
        merged = pd.merge_asof(
            df30_tmp.sort_values('time'),
            df60_ma.sort_values('time'),
            on='time', direction='backward'
        )
        ma1h_mapped = merged['ma1h'].values
        filter_ok = pd.Series(c.values > ma1h_mapped, index=c.index).fillna(True)
    else:
        # Use 30m resampled as fallback
        ma1h_mapped = calc_ema(c, S1_MA1H_LEN * 2)  # approximate
        filter_ok = pd.Series(True, index=c.index)

    signal = crossover & (c > basis) & (ao_st.abs() == 1) & filter_ok

    # Current bar (last row)
    last = df30.iloc[-1]
    i    = len(df30) - 1

    bb_pct_b = (c.iloc[i] - lower.iloc[i]) / (upper.iloc[i] - lower.iloc[i]) if (upper.iloc[i] - lower.iloc[i]) > 0 else np.nan

    # Find last signal
    sig_idx = signal[signal].index.tolist()
    last_sig = None
    if sig_idx:
        li = sig_idx[-1]
        row = df30.iloc[li]
        ep  = row['close']
        sl  = ep * (1 - S1_SL_PCT / 100)
        tp1 = ep * (1 + S1_SL_PCT / 100 * S1_TP1_RATIO)
        tp2 = ep * (1 + S1_SL_PCT / 100 * S1_TP2_RATIO)
        last_sig = {
            "time": str(row['time']),
            "entry": round(float(ep), 2),
            "sl": round(float(sl), 2),
            "tp1": round(float(tp1), 2),
            "tp2": round(float(tp2), 2),
            "bars_ago": i - li,
        }

    return {
        "strategy": "S1-AweWithBB-V3.4",
        "timeframe": "30m",
        "current_bar": {
            "time": str(last['time']),
            "close": round(float(last['close']), 2),
            "bb_basis": round(float(basis.iloc[i]), 2),
            "bb_upper": round(float(upper.iloc[i]), 2),
            "bb_lower": round(float(lower.iloc[i]), 2),
            "bb_pct_b": round(float(bb_pct_b), 3),
            "fast_ema3": round(float(fast_ma.iloc[i]), 2),
            "ao_val": round(float(ao.iloc[i]), 3),
            "ao_state": int(ao_st.iloc[i]),
            "ao_rising": bool(ao_st.iloc[i] in [1, -1]),
            "close_above_basis": bool(c.iloc[i] > basis.iloc[i]),
            "fast_ema_above_basis": bool(fast_ma.iloc[i] > basis.iloc[i]),
            "crossover_now": bool(signal.iloc[i]),
        },
        "signal_now": bool(signal.iloc[i]),
        "distance_to_trigger": {
            "fast_ema_vs_basis": round(float(fast_ma.iloc[i] - basis.iloc[i]), 2),
            "note": "positive = fast EMA above BB basis (S1 basis condition met)" if fast_ma.iloc[i] > basis.iloc[i] else f"fast EMA 需再漲 {abs(fast_ma.iloc[i] - basis.iloc[i]):.1f} 點才站上 BB basis"
        },
        "last_signal": last_sig,
        "recent_signals_count_30d": int(signal.iloc[-1440:].sum()),
    }

# ── S2-Hybrid Scanner ─────────────────────────────────────────────────────────

def scan_s2_hybrid(df30):
    c   = df30['close'].copy()
    rsi = calc_rsi(c, S2H_RSI_LEN)

    # Mode A: RSI crossover above 30
    crossover_30 = (rsi > S2H_RSI_LIMIT) & (rsi.shift(1) <= S2H_RSI_LIMIT)

    i    = len(df30) - 1
    last = df30.iloc[-1]

    # Find last crossover signal
    sig_idx = crossover_30[crossover_30].index.tolist()
    last_rev = None
    if sig_idx:
        li  = sig_idx[-1]
        row = df30.iloc[li]
        ep  = row['close']
        sl  = ep * (1 - S2H_SL_PCT / 100)
        tp1 = ep * (1 + S2H_SL_PCT / 100 * S2H_TP1_RATIO)
        tp2 = ep * (1 + S2H_SL_PCT / 100 * S2H_TP2_RATIO)
        last_rev = {
            "time": str(row['time']),
            "price": round(float(ep), 2),
            "rsi_at_signal": round(float(rsi.iloc[li]), 2),
            "sl": round(float(sl), 2),
            "tp1": round(float(tp1), 2),
            "tp2": round(float(tp2), 2),
            "bars_ago": i - li,
        }

    # Mode B: Bullish Divergence (simplified scan - look at last 60 bars)
    # Find recent RSI pivot lows < 30
    lookback = min(120, i)
    rsi_slice = rsi.iloc[i-lookback:i+1]
    price_slice = c.iloc[i-lookback:i+1]
    div_detected = False
    div_info = None
    for j in range(5, len(rsi_slice)-2):
        r = rsi_slice.iloc[j]
        if r < 30:
            for k in range(j+5, min(j+60, len(rsi_slice))):
                r2 = rsi_slice.iloc[k]
                p2 = price_slice.iloc[k]
                p1 = price_slice.iloc[j]
                if r2 < 30 and p2 < p1 and r2 > r:
                    div_detected = True
                    div_info = {
                        "low1_time": str(df30['time'].iloc[i-lookback+j]),
                        "low1_price": round(float(p1), 2),
                        "low1_rsi": round(float(r), 2),
                        "low2_time": str(df30['time'].iloc[i-lookback+k]),
                        "low2_price": round(float(p2), 2),
                        "low2_rsi": round(float(r2), 2),
                    }

    rsi_now = float(rsi.iloc[i])
    return {
        "strategy": "S2-Hybrid-V2.0",
        "timeframe": "30m",
        "current_bar": {
            "time": str(last['time']),
            "close": round(float(last['close']), 2),
            "rsi14": round(rsi_now, 2),
            "rsi_above_30": rsi_now > 30,
            "rsi_below_30": rsi_now < 30,
            "crossover_now": bool(crossover_30.iloc[i]),
        },
        "signal_now": bool(crossover_30.iloc[i]),
        "mode_a_last_signal": last_rev,
        "mode_b_divergence": {
            "detected_in_lookback": div_detected,
            "detail": div_info,
        },
        "recent_signals_count_30d": int(crossover_30.iloc[-1440:].sum()),
        "note": f"RSI 現在 {rsi_now:.1f}，{'已在 30 以上，可能即將再次觸發（需跌回 30 再彈）' if rsi_now > 30 else f'低於 30，距觸發還需上穿 30（差 {30-rsi_now:.1f}）'}",
    }

# ── S2-Pullback Scanner ───────────────────────────────────────────────────────

def scan_s2_pullback(df30):
    o = df30['open'].copy()
    h = df30['high'].copy()
    l = df30['low'].copy()
    c = df30['close'].copy()

    atr = calc_atr(h, l, c, S2P_ATR_LEN)

    body_top  = pd.concat([c, o], axis=1).max(axis=1)
    body_btm  = pd.concat([c, o], axis=1).min(axis=1)
    body_size = body_top - body_btm
    lower_shd = body_btm - l
    upper_shd = h - body_top
    total_rng = h - l

    is_hammer = (
        (lower_shd > body_size * S2P_TAIL_RATIO) &
        (upper_shd < body_size * S2P_UPPER_RATIO) &
        (total_rng > atr * S2P_ATR_MULT)
    )

    i    = len(df30) - 1
    last = df30.iloc[-1]
    ep   = last['close']
    sl   = ep * (1 - S2P_SL_PCT / 100)
    tp1  = ep * (1 + S2P_SL_PCT / 100 * S2H_TP1_RATIO)
    tp2  = ep * (1 + S2P_SL_PCT / 100 * S2H_TP2_RATIO)

    # Last hammer
    sig_idx = is_hammer[is_hammer].index.tolist()
    last_hammer = None
    if sig_idx:
        li  = sig_idx[-1]
        row = df30.iloc[li]
        le  = row['close']
        last_hammer = {
            "time": str(row['time']),
            "price": round(float(le), 2),
            "lower_shadow": round(float(lower_shd.iloc[li]), 2),
            "body_size": round(float(body_size.iloc[li]), 2),
            "ratio": round(float(lower_shd.iloc[li] / body_size.iloc[li]) if body_size.iloc[li] > 0 else 0, 2),
            "atr_at_signal": round(float(atr.iloc[li]), 2),
            "bars_ago": i - li,
            "sl": round(float(le * (1 - S2P_SL_PCT/100)), 2),
            "tp1": round(float(le * (1 + S2P_SL_PCT/100 * S2H_TP1_RATIO)), 2),
            "tp2": round(float(le * (1 + S2P_SL_PCT/100 * S2H_TP2_RATIO)), 2),
        }

    cur_lower = float(lower_shd.iloc[i])
    cur_body  = float(body_size.iloc[i]) if float(body_size.iloc[i]) > 0 else 0.001
    cur_atr   = float(atr.iloc[i])

    return {
        "strategy": "S2-Pullback-V1.9",
        "timeframe": "30m",
        "current_bar": {
            "time": str(last['time']),
            "close": round(float(ep), 2),
            "lower_shadow": round(cur_lower, 2),
            "body_size": round(cur_body, 2),
            "tail_ratio": round(cur_lower / cur_body, 2),
            "upper_shadow": round(float(upper_shd.iloc[i]), 2),
            "total_range": round(float(total_rng.iloc[i]), 2),
            "atr14": round(cur_atr, 2),
            "is_hammer": bool(is_hammer.iloc[i]),
            "hammer_conditions": {
                "tail_ok": bool(lower_shd.iloc[i] > body_size.iloc[i] * S2P_TAIL_RATIO),
                "upper_ok": bool(upper_shd.iloc[i] < body_size.iloc[i] * S2P_UPPER_RATIO),
                "range_ok": bool(total_rng.iloc[i] > atr.iloc[i] * S2P_ATR_MULT),
            }
        },
        "signal_now": bool(is_hammer.iloc[i]),
        "last_hammer": last_hammer,
        "recent_signals_count_30d": int(is_hammer.iloc[-1440:].sum()),
        "if_entry_now": {
            "entry": round(float(ep), 2),
            "sl": round(float(sl), 2),
            "tp1": round(float(tp1), 2),
            "tp2": round(float(tp2), 2),
        }
    }

# ── Summary ────────────────────────────────────────────────────────────────────

def build_summary(s1, s2h, s2p):
    c = s1['current_bar']['close']
    basis = s1['current_bar']['bb_basis']
    upper = s1['current_bar']['bb_upper']
    lower = s1['current_bar']['bb_lower']
    pct_b = s1['current_bar']['bb_pct_b']
    ao    = s1['current_bar']['ao_val']
    rsi   = s2h['current_bar']['rsi14']

    lines = [f"── 訊號狀態摘要 {s1['current_bar']['time'][:16]} ──"]
    lines.append(f"現價: {c:.2f}  │  BB %B: {pct_b:.2f}  │  RSI(14): {rsi:.1f}  │  AO: {ao:.2f}")
    lines.append(f"BB Basis: {basis:.2f}  │  Upper: {upper:.2f}  │  Lower: {lower:.2f}")
    lines.append("")

    # S1
    fema = s1['current_bar']['fast_ema3']
    dist = s1['distance_to_trigger']['fast_ema_vs_basis']
    s1_fire = "🔴 未觸發"
    if s1['signal_now']:
        s1_fire = "✅ 觸發！"
    elif fema > basis:
        ao_ok = s1['current_bar']['ao_rising']
        s1_fire = f"⚡ Fast EMA 站上 Basis，AO {'↑ 上升中' if ao_ok else '↓ 未上升'}，等 crossover 確認"
    else:
        s1_fire = f"⏳ Fast EMA ({fema:.2f}) 需突破 Basis ({basis:.2f})，差 {abs(dist):.1f} 點"
    lines.append(f"[S1 AweWithBB]  {s1_fire}")
    if s1['last_signal']:
        ls = s1['last_signal']
        lines.append(f"  上次訊號: {ls['time'][:16]} @ {ls['entry']}  ({ls['bars_ago']} bars ago)")

    # S2-Hybrid
    rsi_note = s2h['note']
    s2h_fire = "✅ 觸發！" if s2h['signal_now'] else "🔴 未觸發"
    lines.append(f"[S2 Hybrid]     {s2h_fire}  {rsi_note}")
    if s2h['mode_a_last_signal']:
        ls = s2h['mode_a_last_signal']
        lines.append(f"  上次訊號: {ls['time'][:16]} @ {ls['price']}  RSI={ls['rsi_at_signal']}  ({ls['bars_ago']} bars ago)")

    # S2-Pullback
    s2p_fire = "✅ 觸發！" if s2p['signal_now'] else "🔴 未觸發"
    cur = s2p['current_bar']
    lines.append(f"[S2 Pullback]   {s2p_fire}  下影比={cur['tail_ratio']:.2f}（需>{S2P_TAIL_RATIO}）  ATR={cur['atr14']:.2f}")
    if s2p['last_hammer']:
        lh = s2p['last_hammer']
        lines.append(f"  上次錘頭: {lh['time'][:16]} @ {lh['price']}  下影比={lh['ratio']:.2f}  ({lh['bars_ago']} bars ago)")

    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    tf = 30
    if "--tf" in sys.argv:
        idx = sys.argv.index("--tf")
        tf  = int(sys.argv[idx+1])

    print(f"📡 XAUUSD Signal Scanner (timeframe: {tf}m)")
    csv_dir = find_csv_dir()
    print(f"📂 CSV dir: {csv_dir}")

    df30, fn30 = load_csv(csv_dir, "FX_IDC_XAUUSD", tf)
    print(f"  Loaded 30m: {fn30} ({len(df30)} bars, last: {df30['time'].iloc[-1]})")

    try:
        df60, fn60 = load_csv(csv_dir, "FX_IDC_XAUUSD", 60)
        print(f"  Loaded 60m: {fn60} ({len(df60)} bars)")
    except FileNotFoundError:
        df60 = None
        print("  No 60m data, skipping 1H MA filter")

    s1  = scan_s1(df30, df60)
    s2h = scan_s2_hybrid(df30)
    s2p = scan_s2_pullback(df30)

    summary = build_summary(s1, s2h, s2p)
    print("\n" + summary + "\n")

    result = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "s1": s1,
        "s2_hybrid": s2h,
        "s2_pullback": s2p,
    }

    out = OUTPUT_JSON
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"✅ Saved to: {out}")
    return result

if __name__ == "__main__":
    main()
