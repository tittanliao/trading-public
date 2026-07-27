"""
init_cold_db.py — 冷庫初始化（一次性執行）
════════════════════════════════════════════
用途：將 TickStory（Dukascopy 源）匯出的歷史 CSV 匯入 SQLite，建立冷庫基底。

TickStory 匯出 CSV 常見格式（自動偵測）：
  格式 A（最常見）：Datetime, Open, High, Low, Close, Volume
    2014-01-02 02:00:00,1202.75,1203.25,1202.75,1202.75,0
  格式 B（帶毫秒）：
    2014-01-02 02:00:00.000,1202.75,...

執行：
  python init_cold_db.py <tickstory_csv_path>
  python init_cold_db.py /path/to/XAUUSD_M30_2014_2024.csv

成功後產出：./XAUUSD_M30_Cold.db（與 init_cold_db.py 同目錄）
"""
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import DB_PATH, DB_TABLE, TIMEZONE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

ISO_FMT = "%Y-%m-%d %H:%M:%S+08:00"


def load_tickstory_csv(csv_path: Path) -> pd.DataFrame:
    """
    讀取 TickStory 匯出 CSV，回傳標準化 DataFrame。
    TickStory 時間通常無時區標記，假設已是 Asia/Taipei（交易者設定本地時區匯出）。
    """
    logger.info("讀取 TickStory CSV：%s", csv_path.name)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # 偵測時間欄
    time_col = next(
        (c for c in df.columns if c.lower() in ("datetime", "time", "date")), None
    )
    if time_col is None:
        raise ValueError(f"找不到時間欄，現有欄位：{list(df.columns)}")

    # 解析時間（TickStory 輸出通常無時區，預設為 Asia/Taipei）
    dt = pd.to_datetime(df[time_col].astype(str).str.strip(), utc=False)
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(TIMEZONE)
    else:
        dt = dt.dt.tz_convert(TIMEZONE)
    dt.name = "Datetime"

    # 標準化欄位名稱
    col_map = {"open": "Open", "high": "High",
               "low": "Low",   "close": "Close", "volume": "Volume"}
    df = df.rename(columns={c: col_map[c.lower()] for c in df.columns
                             if c.lower() in col_map})
    df.index = dt

    avail = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[avail].sort_index()

    logger.info("  讀取完成：%d 根（%s → %s）",
                len(df), df.index[0].date(), df.index[-1].date())
    return df


def init_db(csv_path: Path) -> None:
    """建立 SQLite 冷庫並匯入 TickStory 歷史資料。"""
    if DB_PATH.exists():
        ans = input(f"冷庫已存在（{DB_PATH}），覆寫將遺失所有資料。繼續？[y/N] ").strip().lower()
        if ans != "y":
            logger.info("取消操作。")
            sys.exit(0)

    df = load_tickstory_csv(csv_path)

    # 去重（TickStory 資料有時有重複 bar）
    dup_n = df.index.duplicated().sum()
    if dup_n > 0:
        logger.warning("發現 %d 筆重複 Datetime，保留最後一筆", dup_n)
        df = df[~df.index.duplicated(keep="last")]

    # 轉換 Datetime → ISO 8601 字串
    df_out = df.copy()
    df_out.index = pd.Index(df_out.index.strftime(ISO_FMT), name="Datetime")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype("float64")

    # 寫入 SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        df_out.to_sql(
            DB_TABLE, conn,
            if_exists="replace",
            index=True,
            index_label="Datetime",
            dtype={
                "Datetime": "TEXT",
                "Open": "REAL", "High": "REAL",
                "Low": "REAL",  "Close": "REAL", "Volume": "REAL",
            },
        )
        # Datetime 設主鍵索引（加速後續 UPSERT 查詢）
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_dt ON {DB_TABLE}(Datetime)")
        conn.commit()
        logger.info("✅ 冷庫建立完成：%s", DB_PATH)
        logger.info("   資料筆數：%d 根  （%s → %s）",
                    len(df_out), df.index[0].date(), df.index[-1].date())
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python init_cold_db.py <tickstory_csv_path>")
        print("範例：python init_cold_db.py ~/Downloads/XAUUSD_M30_2014_2024.csv")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"錯誤：找不到檔案 {csv_file}")
        sys.exit(1)

    init_db(csv_file)
