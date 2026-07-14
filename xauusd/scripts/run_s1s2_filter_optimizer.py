"""
run_s1s2_filter_optimizer.py — S1/S2 全 filter 參數 Python 最佳化引擎
==============================================================================
工作流（20260712 與使用者確立）：
  1. TradingView 30m FX_IDC:XAUUSD 圖掛上 XAUUSD-S1S2-Export-V1.pine 指標
  2. Export chart data → CSV（含 OHLC + FP_* footprint 欄位 + TickVol + CHK_* 校驗欄位）
  3. 本腳本讀取匯出檔，用 Python 重播 S1/S2 進出場邏輯，網格掃描所有 filter 參數組合

設計：
  - 大部分指標（HTF MA/RSI/BBW/Z-Score/VWAP/DXY）由 Python 從 30m OHLC 重採樣計算
    →參數完全自由，不受匯出時的設定限制
  - Footprint 值只能來自匯出欄位（無法從 OHLC 重算）；imbalance門檻/ticks_per_row
    在匯出時已固定，Python 只能掃「用這些值的條件門檻」（poc位置%、堆疊N等）
  - CHK_* 欄位用來驗證 Python 重算 vs TradingView 的一致性（對齊/lookahead檢查），
    對上了才能信任最佳化結果
  - 引擎驗證：用相同設定重播，跟真實 TV 逐筆 CSV（V4.1/V3.9）比對進場時間與 PF

執行方式（在 trading/ 根目錄）：
  python3.12 xauusd/scripts/run_s1s2_filter_optimizer.py --export "xauusd/csv/exports/你的匯出.csv" --strategy both --mode all
  --mode checksum   只做校驗比對
  --mode engine     校驗 + 引擎 vs 真實逐筆CSV驗證
  --mode optimize   校驗 + 引擎驗證 + 網格最佳化（預設 all = 全部）
輸出：xauusd/report_s1s2_optimizer.html + xauusd/s1s2_optimizer_results.json
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

from analysis import loader

DXY_1D_CSV = ROOT / "xauusd/csv/20260711/TVC_DXY, 1D.csv"
S2_TRADES_CSV = ROOT / "xauusd/XAUUSD-Long-S2-Hammer/S2-Hammer-V4.1_FX_IDC_XAUUSD_2026-07-12.csv"
S1_TRADES_CSV = ROOT / "xauusd/XAUUSD-Long-S1-AweWithBB/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv"
OUT_HTML = ROOT / "xauusd/report_s1s2_optimizer.html"
OUT_JSON = ROOT / "xauusd/s1s2_optimizer_results.json"

MIN_TRADES = 20          # 組合樣本數下限，低於此不進排名
IS_RATIO = 0.7           # 匯出窗口內 70/30 IS/OOS 切分
TOP_N = 25               # 報表列出前 N 名

# ══════════════════════════════════════════════════════════════════════════════
# 1. 基礎指標函式（對齊 Pine 語義）
# ══════════════════════════════════════════════════════════════════════════════

def rma(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    up = rma(d.clip(lower=0), length)
    dn = rma(-d.clip(upper=0), length)
    return 100 - 100 / (1 + up / dn)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, length)

def percentrank(s: pd.Series, length: int) -> pd.Series:
    # Pine ta.percentrank：目前值在過去 length 根（不含自己）中的百分位
    def _pr(win):
        cur = win[-1]
        past = win[:-1]
        return (past <= cur).sum() / len(past) * 100
    return s.rolling(length + 1).apply(_pr, raw=True)

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


# ══════════════════════════════════════════════════════════════════════════════
# 2. 匯出檔載入 + HTF 重採樣（模擬 request.security lookahead_off 語義）
# ══════════════════════════════════════════════════════════════════════════════

def load_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    tcol = "time" if "time" in df.columns else df.columns[0]
    df["time"] = (pd.to_datetime(df[tcol], utc=True)
                  .dt.tz_convert("Asia/Taipei").dt.tz_localize(None))
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise SystemExit(f"匯出檔缺少必要欄位 {c}；實際欄位：{list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # volume 欄位彈性偵測
    vol_col = next((c for c in ["TickVol", "Volume", "volume"] if c in df.columns), None)
    df["vol"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else np.nan
    df = df.sort_values("time").reset_index(drop=True)
    df["close_time"] = df["time"] + pd.Timedelta(minutes=30)
    return df

def resample_htf(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = df.set_index("time").resample(f"{minutes}min", label="left", closed="left")
    h = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "vol": g["vol"].sum(),
    }).dropna(subset=["close"]).reset_index()
    h["close_time"] = h["time"] + pd.Timedelta(minutes=minutes)
    return h

def align_htf(df30: pd.DataFrame, htf: pd.DataFrame, col: str) -> pd.Series:
    """把 HTF 序列對齊回 30m：30m bar 看得到的是「已收盤」的最後一根 HTF 值
    （close_time(HTF) <= close_time(30m)），等同 request.security lookahead_off。"""
    m = pd.merge_asof(
        df30[["close_time"]], htf[["close_time", col]].dropna(),
        on="close_time", direction="backward")
    return m[col].set_axis(df30.index)

def htf_indicator(df30: pd.DataFrame, minutes: int, fn) -> pd.Series:
    """在指定 HTF 上計算 fn(htf_df)->Series，再對齊回 30m。minutes=1440 代表日線。"""
    htf = resample_htf(df30, minutes)
    htf["_v"] = fn(htf)
    return align_htf(df30, htf, "_v")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 校驗（Python 重算 vs TradingView CHK_* 欄位）
# ══════════════════════════════════════════════════════════════════════════════

def run_checksums(df: pd.DataFrame) -> list[dict]:
    checks = []
    def add(name, py_series):
        if name not in df.columns:
            checks.append(dict(name=name, status="欄位不存在（略過）", corr=np.nan, mad=np.nan))
            return
        tv = pd.to_numeric(df[name], errors="coerce")
        both = pd.DataFrame({"tv": tv, "py": py_series}).dropna()
        if len(both) < 50:
            checks.append(dict(name=name, status=f"重疊樣本不足（{len(both)}）", corr=np.nan, mad=np.nan))
            return
        corr = both.tv.corr(both.py)
        mad = (both.tv - both.py).abs().median()
        status = "✅" if corr > 0.995 else ("🟡" if corr > 0.97 else "❌")
        checks.append(dict(name=name, status=status, corr=corr, mad=mad, n=len(both)))

    add("CHK_MA60", htf_indicator(df, 60, lambda h: h["close"].rolling(3).mean()))
    add("CHK_RSI14", rsi(df["close"], 14))
    add("CHK_RSI60", htf_indicator(df, 60, lambda h: rsi(h["close"], 14)))
    add("CHK_RSIMA60", htf_indicator(df, 60, lambda h: rsi(h["close"], 14).rolling(14).mean()))
    add("CHK_BBWrank60", htf_indicator(df, 60, lambda h: percentrank(
        h["close"].rolling(20).std(ddof=0) * 4 / h["close"].rolling(20).mean(), 60)))
    add("CHK_Slope240", htf_indicator(df, 240, lambda h: (
        lambda e: (e - e.shift(10)) / e.shift(10) * 100)(h["close"].ewm(span=20, adjust=False).mean())))
    add("CHK_Z240", (df["close"] - htf_indicator(df, 240, lambda h: h["close"].ewm(span=20, adjust=False).mean()))
        / htf_indicator(df, 240, lambda h: atr(h, 14)))
    # Weekly VWAP（週一重置累計）
    wk = df["time"].dt.isocalendar()
    week_key = wk["year"].astype(str) + "-" + wk["week"].astype(str)
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    pv = (hlc3 * df["vol"]).groupby(week_key).cumsum()
    vv = df["vol"].groupby(week_key).cumsum()
    add("CHK_WVWAP", pv / vv)

    for c in checks:
        corr_s = f"{c['corr']:.4f}" if np.isfinite(c.get("corr", np.nan)) else "—"
        mad_s = f"{c['mad']:.4g}" if np.isfinite(c.get("mad", np.nan)) else "—"
        print(f"  {c['name']:16s} {c['status']}  corr={corr_s}  中位絕對差={mad_s}")
    return checks


# ══════════════════════════════════════════════════════════════════════════════
# 4. 交易引擎（複製 Pine Profit Flyer 出場結構，訊號bar收盤→次bar開盤進場）
# ══════════════════════════════════════════════════════════════════════════════

def run_engine(df: pd.DataFrame, signal: pd.Series, sl_pct: float,
               tp1_r: float, tp2_r: float, out_k: int) -> pd.DataFrame:
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    t = df["time"].to_numpy()
    sig = signal.fillna(False).to_numpy()
    low_q = pd.Series(l).rolling(max(out_k // 4, 1)).min().to_numpy()
    low_h = pd.Series(l).rolling(max(out_k // 2, 1)).min().to_numpy()
    low_f = pd.Series(l).rolling(out_k).min().to_numpy()

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
            stop = np.nan  # 進場bar內無停損保護（與Pine一致）
            pending_entry = False
        if in_pos and not np.isnan(stop) and l[i] <= stop:
            fill = min(o[i], stop)
            trades.append(dict(entry_time=t[entry_i], exit_time=t[i],
                               entry_px=entry_px, exit_px=fill,
                               pnl_r=(fill / entry_px - 1) / sl_pct,
                               bars=i - entry_i))
            in_pos = False
            stop = np.nan
        if in_pos:
            sl = entry_px * (1 - sl_pct)
            pt1 = entry_px * (1 + sl_pct * tp1_r)
            pt2 = entry_px * (1 + sl_pct * tp2_r)
            if c[i] >= pt2:
                stop = max(pt2, low_q[i])
            elif c[i] >= pt1:
                stop = max(pt1, low_h[i])
            else:
                stop = sl
            if i - entry_i >= out_k:
                stop = max(sl, low_f[i]) if c[i] >= entry_px else l[i]
        if not in_pos and sig[i]:
            pending_entry = True
    return pd.DataFrame(trades)

def stat_block(tr: pd.DataFrame) -> dict:
    if len(tr) == 0:
        return dict(n=0, wr=np.nan, pf=np.nan, net_r=0.0)
    wins = tr.loc[tr.pnl_r > 0, "pnl_r"].sum()
    loss = abs(tr.loc[tr.pnl_r < 0, "pnl_r"].sum())
    return dict(n=len(tr), wr=(tr.pnl_r > 0).mean() * 100,
                pf=(wins / loss) if loss else float("inf"),
                net_r=tr.pnl_r.sum())


# ══════════════════════════════════════════════════════════════════════════════
# 5. 訊號與 Filter 庫
# ══════════════════════════════════════════════════════════════════════════════

def build_common(df: pd.DataFrame) -> dict:
    """預先計算共用序列（各 filter 從這裡取用，避免重複計算）"""
    ctx = {}
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    hl2 = (df["high"] + df["low"]) / 2
    ctx["ohlc4"] = ohlc4
    ctx["rsi14"] = rsi(df["close"], 14)
    ctx["atr14"] = atr(df, 14)
    # S1 核心
    basis = ohlc4.rolling(20).mean()
    fast = ohlc4.ewm(span=3, adjust=False).mean()
    ao = hl2.rolling(5).mean() - hl2.rolling(34).mean()
    ctx["s1_core"] = crossover(fast, basis) & (df["close"] > basis) & (ao > ao.shift(1))
    dev = ohlc4.rolling(20).std(ddof=0) * 2
    ctx["bb_pctb"] = (df["close"] - (basis - dev)) / (2 * dev) * 100
    # S2 核心（錘頭）
    body_top = df[["open", "close"]].max(axis=1)
    body_btm = df[["open", "close"]].min(axis=1)
    body = body_top - body_btm
    ctx["s2_core"] = ((body_btm - df["low"] > body * 2.0)
                      & (df["high"] - body_top < body * 0.5)
                      & (df["high"] - df["low"] > ctx["atr14"] * 0.3))
    # DXY（日線，從既有 CSV，模擬 lookahead_off：D 收盤後隔日生效）
    if DXY_1D_CSV.exists():
        dxy = loader.load_price(DXY_1D_CSV)
        dxy["close_time"] = dxy["time"] + pd.Timedelta(days=1)
        dxy["dxy_rsi"] = rsi(dxy["close"], 14)
        dxy["dxy_ema20"] = dxy["close"].ewm(span=20, adjust=False).mean()
        dxy["dxy_weak"] = (dxy["close"] < dxy["dxy_ema20"]).astype(float)
        m = pd.merge_asof(df[["close_time"]],
                          dxy[["close_time", "dxy_rsi", "dxy_weak"]].dropna(),
                          on="close_time", direction="backward")
        ctx["dxy_rsi"] = m["dxy_rsi"].set_axis(df.index)
        ctx["dxy_weak"] = m["dxy_weak"].set_axis(df.index) > 0.5
    else:
        ctx["dxy_rsi"] = pd.Series(np.nan, index=df.index)
        ctx["dxy_weak"] = pd.Series(False, index=df.index)
    # Volume
    ctx["vol"] = df["vol"]
    ctx["vol_sma20"] = df["vol"].rolling(20).mean()
    # Weekly VWAP
    wk = df["time"].dt.isocalendar()
    week_key = wk["year"].astype(str) + "-" + wk["week"].astype(str)
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    ctx["wvwap"] = (hlc3 * df["vol"]).groupby(week_key).cumsum() / df["vol"].groupby(week_key).cumsum()
    # Footprint（匯出欄位；沒有就整組跳過）
    ctx["has_fp"] = "FP_ok" in df.columns
    if ctx["has_fp"]:
        for col in ["FP_ok", "FP_DeltaRatio", "FP_POCpos", "FP_VALpos",
                    "FP_BuyStackFull", "FP_BuyStackLow33", "FP_BuyStackLow50",
                    "FP_SellImbLow33", "FP_SellImbLow50"]:
            ctx[col] = pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        ctx["fp_ok_mask"] = ctx["FP_ok"] > 0.5
    return ctx

# ── Filter 定義：每個回傳 (名稱, bool Series[允許進場]) ──────────────────────
def f_htf_ma(df, ctx, minutes, length):
    ma = htf_indicator(df, minutes, lambda h: h["close"].rolling(length).mean())
    return f"HTF_MA({minutes}m,{length})", df["close"] > ma

def f_bbw_high(df, ctx, minutes, rank_len, thresh):
    r = htf_indicator(df, minutes, lambda h: percentrank(
        h["close"].rolling(20).std(ddof=0) * 4 / h["close"].rolling(20).mean(), rank_len))
    return f"BBWHigh({minutes}m,rank{rank_len}<{thresh})", ~(r >= thresh)

def f_htf_rsi_bear(df, ctx, minutes, length, ma_len):
    r = htf_indicator(df, minutes, lambda h: rsi(h["close"], length))
    rm = htf_indicator(df, minutes, lambda h: rsi(h["close"], length).rolling(ma_len).mean())
    rmp = htf_indicator(df, minutes, lambda h: rsi(h["close"], length).rolling(ma_len).mean().shift(1))
    bearish = (r < rm) & (rm < rmp)
    return f"HTFRSIbear({minutes}m)", ~bearish

def f_mutex(df, ctx, limit, bars):
    cross = crossover(ctx["rsi14"], pd.Series(limit, index=df.index))
    since = (~cross).groupby(cross.cumsum()).cumcount()
    in_env = (cross.cumsum() > 0) & (since <= bars)
    return f"Mutex(RSI14×{limit},{bars}bars)", ~in_env

def f_regime_consol(df, ctx, minutes=240, slope_th=0.15, bbw_th=65.0):
    slope = htf_indicator(df, minutes, lambda h: (
        lambda e: (e - e.shift(10)) / e.shift(10) * 100)(h["close"].ewm(span=20, adjust=False).mean()))
    rank = htf_indicator(df, minutes, lambda h: percentrank(
        h["close"].rolling(20).std(ddof=0) * 4 / h["close"].rolling(20).mean(), 60))
    consol = (rank < bbw_th) & (slope.abs() < slope_th)
    return f"Regime(!CONSOL@{minutes}m)", ~consol

def f_dxy_band(df, ctx, lo=30.0, hi=50.0):
    dead = (ctx["dxy_rsi"] >= lo) & (ctx["dxy_rsi"] < hi)
    return f"DXYband(block{lo:.0f}-{hi:.0f})", ~dead

def f_dxy_weak(df, ctx):
    return "DXYweak(close<EMA20 D)", ctx["dxy_weak"]

def f_vol_mult(df, ctx, mult, label):
    ok = ctx["vol"] >= ctx["vol_sma20"] * mult
    return f"{label}(≥{mult}×均量)", ok.fillna(True)  # 無volume資料時放行

def f_wvwap(df, ctx):
    return "WeeklyVWAP(close>)", (df["close"] > ctx["wvwap"]).fillna(True)

def f_fp(df, ctx, mode, **kw):
    """footprint 條件；na（無資料）一律放行（與 pine fp_na_pass=true 一致）"""
    ok_mask = ctx["fp_ok_mask"]
    poc = ctx["FP_POCpos"]; dr = ctx["FP_DeltaRatio"]
    if mode == "A":
        cond = poc >= kw.get("poc", 55)
        name = f"FP_A(POC≥{kw.get('poc', 55)})"
    elif mode == "C":
        col = f"FP_BuyStackLow{kw.get('zone', 50)}"
        cond = ctx[col] >= kw.get("n", 2)
        name = f"FP_C(stack{kw.get('zone',50)}≥{kw.get('n',2)})"
    elif mode == "D":
        col = f"FP_BuyStackLow{kw.get('zone', 50)}"
        cond = (poc >= kw.get("poc", 55)) & (ctx[col] >= kw.get("n", 2))
        name = f"FP_D(POC≥{kw.get('poc',55)}+stack{kw.get('zone',50)}≥{kw.get('n',2)})"
    elif mode == "G":
        col = f"FP_BuyStackLow{kw.get('zone', 50)}"
        clim = ctx["vol"] >= ctx["vol_sma20"] * kw.get("climax", 2.0)
        cond = (poc >= kw.get("poc", 55)) & (ctx[col] >= kw.get("n", 2)) & clim.fillna(False)
        name = f"FP_G(D+climax{kw.get('climax',2.0)})"
    elif mode == "H":
        simb = ctx[f"FP_SellImbLow{kw.get('zone', 50)}"] > 0.5
        dr_avg = dr.where(ok_mask).rolling(5, min_periods=2).mean().shift(1)
        cond = simb & ((dr - dr_avg) >= kw.get("improve", 5.0))
        name = "FP_H(trapped)"
    elif mode == "S1A":
        cond = (dr > 0) & (dr >= kw.get("minratio", 10.0))
        name = f"FP_S1A(Δ≥{kw.get('minratio', 10.0)}%)"
    elif mode == "S1F":
        block = (df["close"] > df["open"]) & (dr < 0)
        cond = ~block
        name = "FP_S1F(背離阻擋)"
    else:
        raise ValueError(mode)
    return name, cond.where(ok_mask, True).fillna(True)  # 無fp資料→放行


# ══════════════════════════════════════════════════════════════════════════════
# 6. 引擎驗證（vs 真實 TV 逐筆 CSV）
# ══════════════════════════════════════════════════════════════════════════════

def validate_engine(df, ctx, strategy: str) -> dict:
    if strategy == "s2":
        csv_path, label = S2_TRADES_CSV, "S2 V4.1（HTF60+Mutex20+FP_D）"
        _, f1 = f_htf_rsi_bear(df, ctx, 60, 14, 14)
        _, f2 = f_mutex(df, ctx, 20, 24)
        if ctx["has_fp"]:
            _, f3 = f_fp(df, ctx, "D", poc=55, zone=50, n=2)
        else:
            f3 = pd.Series(True, index=df.index)
        sig = ctx["s2_core"] & f1 & f2 & f3
        tr = run_engine(df, sig, 0.010, 2.0, 4.0, 48)
    else:
        csv_path, label = S1_TRADES_CSV, "S1 V3.9（HTF MA60/3 + BBWHigh60<70）"
        _, f1 = f_htf_ma(df, ctx, 60, 3)
        _, f2 = f_bbw_high(df, ctx, 60, 60, 70)
        sig = ctx["s1_core"] & f1 & f2
        tr = run_engine(df, sig, 0.005, 1.0, 3.5, 36)

    result = dict(label=label, py_n=len(tr))
    if not csv_path.exists():
        result["status"] = "真實CSV不存在，略過比對"
        return result
    tv = loader.load_trades(csv_path)
    tv = tv[tv["exit_signal"] != "Open"]
    lo, hi = df["time"].min(), df["time"].max()
    tv_w = tv[(tv.entry_time >= lo) & (tv.entry_time <= hi)]
    if len(tr):
        py_times = pd.to_datetime(tr.entry_time)
        matched = sum(any(abs((et - pt).total_seconds()) <= 1800 for pt in py_times)
                      for et in tv_w.entry_time)
    else:
        matched = 0
    ps = stat_block(tr)
    result.update(tv_n=len(tv_w), matched=matched,
                  match_rate=matched / len(tv_w) * 100 if len(tv_w) else np.nan,
                  py_pf=ps["pf"], py_wr=ps["wr"])
    tvw_wins = tv_w[tv_w.net_pnl_usd > 0].net_pnl_usd.sum()
    tvw_loss = abs(tv_w[tv_w.net_pnl_usd < 0].net_pnl_usd.sum())
    result["tv_pf"] = tvw_wins / tvw_loss if tvw_loss else float("inf")
    print(f"  {label}")
    print(f"    引擎重播 {result['py_n']} 筆 / TV 同窗口 {result['tv_n']} 筆 / "
          f"進場時間吻合 {matched}（{result.get('match_rate', 0):.0f}%）")
    print(f"    PF：Python {result['py_pf']:.2f} vs TV {result['tv_pf']:.2f}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 7. 網格最佳化
# ══════════════════════════════════════════════════════════════════════════════

def grid_s1(df, ctx):
    """S1 各 filter 的候選值；None = 不啟用該 filter"""
    opts = {
        "htf_ma": [None, (60, 3), (120, 3), (240, 3)],
        "bbw_high": [None, (60, 60, 70), (60, 60, 80), (240, 60, 70)],
        "wvwap": [False, True],
        "vol_surge": [None, 1.5, 2.0],
        "dxy_weak": [False, True],
    }
    if ctx["has_fp"]:
        opts["fp"] = [None, ("S1A", dict(minratio=10)), ("S1A", dict(minratio=20)), ("S1F", {})]
    else:
        opts["fp"] = [None]
    return opts, dict(core="s1_core", sl=0.005, tp1=1.0, tp2=3.5, out=36)

def grid_s2(df, ctx):
    opts = {
        "htf_rsi": [None, (60, 14, 14), (240, 14, 14)],
        "mutex": [None, (20, 24), (30, 24)],
        "regime": [False, True],
        "dxy_band": [False, True],
        "vol_climax": [None, 2.0],
    }
    if ctx["has_fp"]:
        opts["fp"] = [None,
                      ("A", dict(poc=55)),
                      ("C", dict(zone=50, n=2)),
                      ("D", dict(poc=50, zone=50, n=2)),
                      ("D", dict(poc=55, zone=50, n=2)),
                      ("D", dict(poc=60, zone=50, n=2)),
                      ("D", dict(poc=55, zone=33, n=2)),
                      ("D", dict(poc=55, zone=50, n=3)),
                      ("G", dict(poc=55, zone=50, n=2, climax=2.0)),
                      ("H", dict(zone=50, improve=5.0))]
    else:
        opts["fp"] = [None]
    return opts, dict(core="s2_core", sl=0.010, tp1=2.0, tp2=4.0, out=48)

def build_mask(df, ctx, strategy, combo) -> tuple[list[str], pd.Series]:
    names, mask = [], pd.Series(True, index=df.index)
    def apply(name, s):
        nonlocal mask
        names.append(name)
        mask = mask & s.fillna(False)
    if strategy == "s1":
        if combo["htf_ma"]:
            apply(*f_htf_ma(df, ctx, *combo["htf_ma"]))
        if combo["bbw_high"]:
            apply(*f_bbw_high(df, ctx, *combo["bbw_high"]))
        if combo["wvwap"]:
            apply(*f_wvwap(df, ctx))
        if combo["vol_surge"]:
            apply(*f_vol_mult(df, ctx, combo["vol_surge"], "VolSurge"))
        if combo["dxy_weak"]:
            apply(*f_dxy_weak(df, ctx))
        if combo["fp"]:
            apply(*f_fp(df, ctx, combo["fp"][0], **combo["fp"][1]))
    else:
        if combo["htf_rsi"]:
            apply(*f_htf_rsi_bear(df, ctx, *combo["htf_rsi"]))
        if combo["mutex"]:
            apply(*f_mutex(df, ctx, *combo["mutex"]))
        if combo["regime"]:
            apply(*f_regime_consol(df, ctx))
        if combo["dxy_band"]:
            apply(*f_dxy_band(df, ctx))
        if combo["vol_climax"]:
            apply(*f_vol_mult(df, ctx, combo["vol_climax"], "VolClimax"))
        if combo["fp"]:
            apply(*f_fp(df, ctx, combo["fp"][0], **combo["fp"][1]))
    return names, mask

# HTF/filters 需要暖機；另 filter 計算做 memoize 加速
_memo = {}
def memo_mask(df, ctx, strategy, key, combo):
    if key not in _memo:
        _memo[key] = build_mask(df, ctx, strategy, combo)
    return _memo[key]

def optimize(df, ctx, strategy: str) -> list[dict]:
    opts, eng = (grid_s1 if strategy == "s1" else grid_s2)(df, ctx)
    keys = list(opts.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(opts[k] for k in keys))]
    print(f"  {strategy.upper()}：{len(combos)} 個組合掃描中...")
    core = ctx[eng["core"]]
    cut_i = int(len(df) * IS_RATIO)
    cut_time = df["time"].iloc[cut_i]
    rows = []
    for combo in combos:
        key = json.dumps({k: str(v) for k, v in combo.items()}, sort_keys=True)
        names, mask = memo_mask(df, ctx, strategy, key, combo)
        sig = core & mask
        tr = run_engine(df, sig, eng["sl"], eng["tp1"], eng["tp2"], eng["out"])
        full = stat_block(tr)
        if full["n"] < MIN_TRADES:
            continue
        tr["entry_time"] = pd.to_datetime(tr["entry_time"])
        is_ = stat_block(tr[tr.entry_time < cut_time])
        oos = stat_block(tr[tr.entry_time >= cut_time])
        score = min(is_["pf"] if np.isfinite(is_["pf"]) else 0,
                    oos["pf"] if np.isfinite(oos["pf"]) else 0)
        rows.append(dict(filters=" + ".join(names) if names else "（無filter，純核心訊號）",
                         n=full["n"], wr=full["wr"], pf=full["pf"], net_r=full["net_r"],
                         is_n=is_["n"], is_pf=is_["pf"], oos_n=oos["n"],
                         oos_pf=oos["pf"], oos_wr=oos["wr"], score=score))
    rows.sort(key=lambda r: r["score"], reverse=True)
    print(f"  完成：{len(rows)} 個組合達到樣本數門檻（≥{MIN_TRADES}筆）")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 8. 主流程 + 報表
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v, f="{:.2f}"):
    return "—" if (isinstance(v, float) and not np.isfinite(v)) else f.format(v)

def html_report(export_path, df, checks, engine_results, opt_results):
    def check_rows():
        return "".join(
            f"<tr><td>{c['name']}</td><td>{c['status']}</td>"
            f"<td>{fmt(c.get('corr', np.nan), '{:.4f}')}</td>"
            f"<td>{fmt(c.get('mad', np.nan), '{:.4g}')}</td></tr>" for c in checks)
    def eng_rows():
        out = ""
        for r in engine_results:
            out += (f"<tr><td>{r['label']}</td><td>{r.get('py_n','—')}</td><td>{r.get('tv_n','—')}</td>"
                    f"<td>{fmt(r.get('match_rate', np.nan), '{:.0f}')}%</td>"
                    f"<td>{fmt(r.get('py_pf', np.nan))} / {fmt(r.get('tv_pf', np.nan))}</td></tr>")
        return out
    def opt_table(rows):
        body = ""
        for i, r in enumerate(rows[:TOP_N]):
            hl = " style='background:rgba(34,197,94,.08)'" if i < 3 else ""
            body += (f"<tr{hl}><td>{i+1}</td><td style='font-size:.8em'>{r['filters']}</td>"
                     f"<td>{r['n']}</td><td>{fmt(r['wr'], '{:.1f}')}%</td><td>{fmt(r['pf'])}</td>"
                     f"<td>{r['is_n']}/{fmt(r['is_pf'])}</td>"
                     f"<td>{r['oos_n']}/{fmt(r['oos_pf'])}</td><td><strong>{fmt(r['score'])}</strong></td></tr>")
        return (f"<table class='tbl'><thead><tr><th>#</th><th>Filter 組合</th><th>筆數</th><th>勝率</th>"
                f"<th>PF</th><th>IS n/PF</th><th>OOS n/PF</th><th>score=min(IS,OOS) PF</th></tr></thead>"
                f"<tbody>{body}</tbody></table>")

    css = """<style>*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.5}
.wrap{max-width:1250px;margin:0 auto}h1{font-size:1.5em;margin-bottom:6px;color:#f8fafc}
h2{font-size:1.05em;color:#38bdf8;margin:8px 0 12px;padding-bottom:6px;border-bottom:1px solid #334155}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:22px;margin-bottom:18px}
.tbl{width:100%;border-collapse:collapse;font-size:.85em;margin:10px 0}
.tbl th{background:#0f172a;color:#94a3b8;padding:8px 10px;text-align:left;border-bottom:1px solid #334155}
.tbl td{padding:7px 10px;border-bottom:1px solid rgba(51,65,85,.5)}
.note{font-size:.82em;color:#94a3b8;margin-top:8px}
.warn{background:rgba(245,158,11,.08);border-left:3px solid #f59e0b;padding:12px 16px;border-radius:4px;margin:12px 0;font-size:.9em}
</style>"""
    parts = [f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8">
<title>S1/S2 Filter 最佳化</title>{css}</head><body><div class="wrap">
<h1>S1/S2 Filter Python 最佳化報告</h1>
<p class="note">生成：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · 資料：{export_path}
（{df['time'].min()} → {df['time'].max()}，{len(df)} 根 30m）· IS/OOS 切分 {int(IS_RATIO*100)}/{int((1-IS_RATIO)*100)}</p>
<div class="card"><h2>① 校驗（Python 重算 vs TradingView）</h2>
<table class="tbl"><thead><tr><th>欄位</th><th>狀態</th><th>相關係數</th><th>中位絕對差</th></tr></thead>
<tbody>{check_rows()}</tbody></table>
<p class="note">✅ corr>0.995（可信）🟡 0.97-0.995（可用但注意）❌ <0.97（該指標的重算有對齊問題，勿信任相關組合）</p></div>
<div class="card"><h2>② 引擎驗證（Python 重播 vs TV 真實逐筆）</h2>
<table class="tbl"><thead><tr><th>設定</th><th>Python筆數</th><th>TV筆數</th><th>進場吻合率</th><th>PF（Py/TV）</th></tr></thead>
<tbody>{eng_rows()}</tbody></table>
<p class="note">吻合率 ≥85% 且 PF 差距 <0.15 → 引擎可信；否則排名只能看相對順序、不能看絕對值</p></div>"""]
    for strat, rows in opt_results.items():
        parts.append(f"<div class='card'><h2>③ {strat.upper()} Filter 組合排名（前{TOP_N}，共{len(rows)}個達門檻）</h2>{opt_table(rows)}</div>")
    parts.append("""<div class="warn"><strong>過擬合警告：</strong>score 用 min(IS PF, OOS PF) 抑制單邊過擬合，
但排名前幾名之間的差異可能只是雜訊——關注「哪些 filter 反覆出現在前段班」而非單一最佳組合；
最終仍需回 TradingView 用完整歷史驗證（Python 窗口受匯出深度限制，通常短於 TV 完整回測）。</div>
</div></body></html>""")
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print(f"\n✅ 報告：{OUT_HTML}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="TradingView 匯出 CSV 路徑")
    ap.add_argument("--strategy", default="both", choices=["s1", "s2", "both"])
    ap.add_argument("--mode", default="all", choices=["checksum", "engine", "optimize", "all"])
    args = ap.parse_args()

    print(f"載入匯出檔：{args.export}")
    df = load_export(Path(args.export))
    print(f"  {len(df)} 根 30m（{df['time'].min()} → {df['time'].max()}）")
    fp_cols = [c for c in df.columns if c.startswith("FP_")]
    print(f"  Footprint 欄位：{len(fp_cols)} 個 {'✅' if fp_cols else '❌（fp相關組合將跳過）'}")

    ctx = build_common(df)

    print("\n① 校驗（Python 重算 vs TV CHK_* 欄位）...")
    checks = run_checksums(df)

    engine_results = []
    if args.mode in ("engine", "optimize", "all"):
        print("\n② 引擎驗證（vs TV 真實逐筆 CSV）...")
        if args.strategy in ("s2", "both"):
            engine_results.append(validate_engine(df, ctx, "s2"))
        if args.strategy in ("s1", "both"):
            engine_results.append(validate_engine(df, ctx, "s1"))

    opt_results = {}
    if args.mode in ("optimize", "all"):
        print("\n③ 網格最佳化...")
        if args.strategy in ("s1", "both"):
            opt_results["s1"] = optimize(df, ctx, "s1")
        if args.strategy in ("s2", "both"):
            opt_results["s2"] = optimize(df, ctx, "s2")
        OUT_JSON.write_text(json.dumps(
            {k: v[:100] for k, v in opt_results.items()}, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print(f"✅ 結果 JSON：{OUT_JSON}")

    html_report(args.export, df, checks, engine_results, opt_results)


if __name__ == "__main__":
    main()
