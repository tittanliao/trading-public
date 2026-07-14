"""
Module C：冷熱資料庫融合（UPSERT into Cold DB）
════════════════════════════════════════════════
流程：
  1. 確認冷庫存在（不存在 → 終止並提示初始化 SOP）
  2. 讀取冷庫 XAUUSD_M30 資料表
  3. 取 Module A 的 XAUUSD 熱資料（只保留 OHLCV，不含指標）
  4. pd.concat 垂直合併，Datetime 去重 keep='last'（熱資料優先）
  5. 覆寫回 SQLite
  6. 輸出稽核 Log（筆數變化、日期範圍）

Schema（與冷庫保持一致）：
  Datetime TEXT PRIMARY KEY   → ISO 8601 + 08:00（e.g. 2024-01-15 10:30:00+08:00）
  Open     REAL
  High     REAL
  Low      REAL
  Close    REAL
  Volume   REAL

單獨執行：不建議（需傳入 hot_xauusd DataFrame），供 run_pipeline.py 呼叫
"""
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import DB_PATH, DB_TABLE, TIMEZONE

logger = logging.getLogger(__name__)

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
ISO_FMT    = "%Y-%m-%d %H:%M:%S+08:00"


# ── Datetime ↔ ISO 字串轉換 ────────────────────────────────────────────────────

def _to_iso(idx: pd.DatetimeIndex) -> pd.Index:
    """DatetimeIndex（UTC+8）→ ISO 8601 字串 Series（帶 +08:00）"""
    return pd.Index(idx.strftime(ISO_FMT), name="Datetime")


def _from_iso(series: pd.Series) -> pd.DatetimeIndex:
    """ISO 8601 字串 → DatetimeIndex（Asia/Taipei）"""
    dt = pd.to_datetime(series, utc=False)
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(TIMEZONE)
    else:
        dt = dt.dt.tz_convert(TIMEZONE)
    dt.name = "Datetime"
    return dt


# ── SQLite 讀寫 ────────────────────────────────────────────────────────────────

def read_cold_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    讀取冷庫資料表，回傳帶 UTC+8 DatetimeIndex 的 DataFrame。
    欄位：Open / High / Low / Close / Volume（全 REAL）
    """
    df = pd.read_sql_query(
        f"SELECT Datetime, Open, High, Low, Close, Volume FROM {DB_TABLE} ORDER BY Datetime",
        conn,
    )
    df.index = _from_iso(df["Datetime"])
    df = df.drop(columns=["Datetime"])
    return df


def write_cold_db(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """
    覆寫冷庫資料表（replace 模式）。
    Datetime 存為 ISO 8601 字串，欄位型別與初始 Schema 一致。
    """
    df_out = df.copy()
    df_out.index = _to_iso(df_out.index)
    df_out = df_out[[c for c in OHLCV_COLS if c in df_out.columns]]

    df_out.to_sql(
        DB_TABLE,
        conn,
        if_exists="replace",
        index=True,
        index_label="Datetime",
        dtype={
            "Datetime": "TEXT",
            "Open": "REAL", "High": "REAL",
            "Low": "REAL",  "Close": "REAL", "Volume": "REAL",
        },
    )


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run_module_c(hot_xauusd: pd.DataFrame) -> None:
    """
    執行模組 C：UPSERT 熱資料進冷庫。

    Parameters
    ----------
    hot_xauusd : Module A 回傳的 XAUUSD DataFrame（含指標亦可；只取 OHLCV 入庫）
    """
    logger.info("=" * 60)
    logger.info("【模組 C】冷熱資料庫融合（UPSERT）")
    logger.info("=" * 60)

    # ── Step 1：確認冷庫存在 ──────────────────────────────────────────────────
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"冷庫不存在：{DB_PATH}\n\n"
            "  ── 初始化 SOP ────────────────────────────────────────────\n"
            "  1. 使用 TickStory（Dukascopy 源）匯出 XAUUSD M30 歷史 CSV\n"
            "  2. 執行：python init_cold_db.py <csv路徑>\n"
            "     → 自動建立 XAUUSD_M30_Cold.db 並匯入 10 年歷史資料\n"
            "  ─────────────────────────────────────────────────────────"
        )

    # ── Step 2：讀取冷庫 ──────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    try:
        cold_df    = read_cold_db(conn)
        cnt_before = len(cold_df)
        logger.info("  冷庫現有：%d 根  （%s → %s）",
                    cnt_before, cold_df.index[0].date(), cold_df.index[-1].date())

        # ── Step 3：準備熱資料（只保留 OHLCV） ───────────────────────────────
        hot_ohlcv = hot_xauusd[[c for c in OHLCV_COLS if c in hot_xauusd.columns]].copy()
        logger.info("  熱資料：%d 根  （%s → %s）",
                    len(hot_ohlcv), hot_ohlcv.index[0].date(), hot_ohlcv.index[-1].date())

        # 型別強制對齊（避免 concat 後型別混用）
        for col in OHLCV_COLS:
            if col in hot_ohlcv.columns:
                hot_ohlcv[col] = hot_ohlcv[col].astype("float64")
            if col in cold_df.columns:
                cold_df[col] = cold_df[col].astype("float64")

        # ── Step 4：合併去重（熱資料優先，keep='last'） ───────────────────────
        overlap_n = int(cold_df.index.isin(hot_ohlcv.index).sum())
        merged    = (pd.concat([cold_df, hot_ohlcv])
                       .pipe(lambda df: df[~df.index.duplicated(keep="last")])
                       .sort_index())
        cnt_after = len(merged)
        new_n     = cnt_after - cnt_before

        # ── Step 5：覆寫 DB ───────────────────────────────────────────────────
        write_cold_db(conn, merged)
        conn.commit()

        # ── Step 6：稽核 Log ──────────────────────────────────────────────────
        logger.info("  ─────────────────────────────────────────────────")
        logger.info("  ✅ UPSERT 完成")
        logger.info("     融合前    ：%7d 根", cnt_before)
        logger.info("     融合後    ：%7d 根  （增加 %+d）", cnt_after, new_n)
        logger.info("     重疊覆蓋  ：%7d 根  （熱資料優先）", overlap_n)
        logger.info("     純新增    ：%7d 根", new_n)
        logger.info("     DB 時間範圍：%s → %s",
                    merged.index[0].date(), merged.index[-1].date())

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info("【模組 C】完成\n")
