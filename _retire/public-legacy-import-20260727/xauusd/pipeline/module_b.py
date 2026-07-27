"""
Module B：手動匯出 CSV 驗證防呆（Validation Gate）
════════════════════════════════════════════════════
流程：
  1. 確認 TV_EXPORT_PATH 存在（不存在 → 立即終止，輸出 SOP 提示）
  2. 解析 TV Export CSV：彈性時間格式 → 統一轉 UTC+8
  3. 與 Module A 取得的 tvDatafeed 熱資料取 Datetime 交集
  4. 嚴格比對 OHLC（容忍差 1e-4）：任何違規 → 拋出 Exception 終止
  5. 寬容比對 Volume（僅記錄 Log，不中斷）

設計原則：Fail-fast。資料品質是後續一切分析的基礎，
寧可中止管線、也不允許錯誤資料進入冷庫。

單獨執行：不建議（需傳入 hot_xauusd DataFrame），供 run_pipeline.py 呼叫
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import TV_EXPORT_PATH, OHLC_TOLERANCE, TIMEZONE, VALIDATION_FAIL_IF_MISSING

logger = logging.getLogger(__name__)

OHLC_COLS = ["Open", "High", "Low", "Close"]


# ── TV Export CSV 解析 ─────────────────────────────────────────────────────────

def _parse_tv_export(path: Path) -> pd.DataFrame:
    """
    載入 TradingView 手動匯出 CSV，回傳帶 UTC+8 DatetimeIndex 的 DataFrame。

    TV Export 常見格式（依匯出選項不同）：
      欄位：time, open, high, low, close, Volume[, RSI, RSI-based MA, ...]
      時間：2026-01-15T10:30:00+08:00  或  2026-01-15 10:30:00+08:00

    解析策略：
      - 找出時間欄（time / Time / datetime / Datetime）
      - pd.to_datetime 彈性解析（容忍各種 ISO 格式）
      - 有時區資訊 → tz_convert；無時區 → tz_localize(UTC+8)
    """
    df = pd.read_csv(path, low_memory=False)
    # 去除欄位名稱前後空白
    df.columns = [c.strip() for c in df.columns]

    # 尋找時間欄
    time_col = next(
        (c for c in df.columns if c.lower() in ("time", "datetime")), None
    )
    if time_col is None:
        raise ValueError(
            f"TV Export CSV 找不到時間欄，現有欄位：{list(df.columns)}\n"
            f"請確認匯出時有選擇 ISO 時間格式。"
        )

    # 彈性時間解析
    raw_dt = df[time_col].astype(str).str.strip()
    try:
        dt = pd.to_datetime(raw_dt, utc=False)
    except Exception as exc:
        raise ValueError(f"時間欄解析失敗：{exc}\n  範例值：{raw_dt.iloc[0]}") from exc

    if dt.dt.tz is None:
        # 無時區資訊 → 假設匯出時已選 UTC+8
        dt = dt.dt.tz_localize(TIMEZONE)
        logger.debug("TV Export 時間無時區標記，已假設 %s", TIMEZONE)
    else:
        dt = dt.dt.tz_convert(TIMEZONE)

    df["Datetime"] = dt
    df = df.set_index("Datetime").sort_index()

    # 欄位名稱標準化（TV Export 輸出小寫）
    col_map = {"open": "Open", "high": "High",
               "low": "Low",   "close": "Close", "volume": "Volume"}
    df = df.rename(columns={c: col_map[c.lower()] for c in df.columns
                             if c.lower() in col_map})

    # 只保留 OHLCV
    avail = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[avail]


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run_module_b(hot_xauusd: pd.DataFrame) -> None:
    """
    執行模組 B 驗證防呆。
    驗證通過 → 靜默返回；驗證失敗 → 拋出 Exception（由 run_pipeline.py 攔截）。

    Parameters
    ----------
    hot_xauusd : Module A 回傳的 XAUUSD DataFrame
    """
    logger.info("=" * 60)
    logger.info("【模組 B】資料驗證防呆（Validation Gate）")
    logger.info("=" * 60)

    # ── Step 1：確認 TV Export 檔案存在 ───────────────────────────────────────
    if not TV_EXPORT_PATH.exists():
        sop = (
            "\n"
            "  ── TradingView 手動匯出 SOP ──────────────────────────────\n"
            "  1. 開啟 TradingView，切換至 OANDA:XAUUSD 30m 圖表\n"
            "  2. 右上角選單 → 「匯出圖表資料 (Export chart data)」\n"
            "  3. 時間格式選擇 ISO 時間（UTC+8 Asia/Taipei）\n"
            f"  4. 將檔案儲存至：{TV_EXPORT_PATH}\n"
            "  ─────────────────────────────────────────────────────────"
        )
        msg = f"TV Export CSV 不存在：{TV_EXPORT_PATH}{sop}"

        if VALIDATION_FAIL_IF_MISSING:
            raise FileNotFoundError(msg)
        else:
            logger.warning("⚠  %s\n  VALIDATION_FAIL_IF_MISSING=False，跳過驗證繼續執行。", msg)
            return

    # ── Step 2：解析 TV Export CSV ────────────────────────────────────────────
    logger.info("載入 TV Export：%s", TV_EXPORT_PATH.name)
    tv_df = _parse_tv_export(TV_EXPORT_PATH)
    logger.info("  TV Export：%d 列  %s → %s",
                len(tv_df), tv_df.index[0], tv_df.index[-1])
    logger.info("  Hot Data ：%d 列  %s → %s",
                len(hot_xauusd), hot_xauusd.index[0], hot_xauusd.index[-1])

    # ── Step 3：Datetime 對齊（取交集） ──────────────────────────────────────
    common_idx = hot_xauusd.index.intersection(tv_df.index)
    if len(common_idx) == 0:
        raise ValueError(
            "tvDatafeed 資料與 TV Export 無共同 Datetime（0 筆交集）。\n"
            "可能原因：時區不符、商品不同、或兩份資料時間範圍完全不重疊。"
        )

    hot_aligned = hot_xauusd.loc[common_idx, [c for c in OHLC_COLS + ["Volume"]
                                               if c in hot_xauusd.columns]]
    tv_aligned  = tv_df.loc[common_idx,      [c for c in OHLC_COLS + ["Volume"]
                                               if c in tv_df.columns]]
    logger.info("  共同 Datetime：%d 筆（%s → %s）",
                len(common_idx), common_idx[0], common_idx[-1])

    # ── Step 4：OHLC 嚴格比對（Fail-fast） ───────────────────────────────────
    avail_ohlc = [c for c in OHLC_COLS if c in hot_aligned.columns and c in tv_aligned.columns]
    diff_ohlc  = (hot_aligned[avail_ohlc].astype(float)
                  - tv_aligned[avail_ohlc].astype(float)).abs()
    max_diff_per_row = diff_ohlc.max(axis=1)
    violations = max_diff_per_row[max_diff_per_row > OHLC_TOLERANCE]

    if not violations.empty:
        logger.error("❌ OHLC 驗證失敗：%d 筆超出容忍差 %.0e", len(violations), OHLC_TOLERANCE)
        logger.error("   %-26s  %-6s  %-6s  %-6s  %-6s  |  %-6s  %-6s  %-6s  %-6s  |  MaxΔ",
                     "Datetime", "O(tv)", "H(tv)", "L(tv)", "C(tv)",
                     "O(ex)", "H(ex)", "L(ex)", "C(ex)")
        for dt_idx in violations.index[:10]:   # 最多印 10 筆
            hr = hot_aligned.loc[dt_idx, avail_ohlc].values
            tr = tv_aligned.loc[dt_idx, avail_ohlc].values
            logger.error("   %s  %8.4f %8.4f %8.4f %8.4f  |  %8.4f %8.4f %8.4f %8.4f  |  %.6f",
                         dt_idx, *hr, *tr, violations[dt_idx])
        if len(violations) > 10:
            logger.error("   ... 及其他 %d 筆（總計 %d 筆違規）",
                         len(violations) - 10, len(violations))
        raise ValueError(
            f"模組 B 驗證失敗：{len(violations)} 筆 OHLC 超出容忍差 {OHLC_TOLERANCE}。\n"
            "請確認 tvDatafeed 抓取的商品（OANDA:XAUUSD）與 TV Export 圖表設定一致。"
        )

    global_max = diff_ohlc.max().max()
    logger.info("  ✅ OHLC 比對通過（%d 筆，全域最大差值 %.2e）", len(common_idx), global_max)

    # ── Step 5：Volume 寬容比對（僅 Log） ─────────────────────────────────────
    if "Volume" in hot_aligned.columns and "Volume" in tv_aligned.columns:
        vol_diff = (hot_aligned["Volume"].astype(float)
                    - tv_aligned["Volume"].astype(float)).abs()
        vol_mismatch_n = int((vol_diff > 0).sum())
        if vol_mismatch_n > 0:
            logger.warning("  ⚠  Volume 差異：%d/%d 筆（CFD 特性，不影響驗證）",
                           vol_mismatch_n, len(common_idx))
            logger.info("     Volume 最大差值：%.4f", vol_diff.max())
        else:
            logger.info("  ✅ Volume 完全一致")

    logger.info("【模組 B】驗證通過\n")
