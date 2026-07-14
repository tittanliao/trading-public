"""
Module A：熱資料獲取與週報產出
═══════════════════════════════
流程：
  1. 建立 tvDatafeed 連線（支援帳號登入 / 匿名 / 自動重試）
  2. 逐商品抓取 M30 原始 OHLCV（XAUUSD / DXY / MGC1!）
  3. Datetime 統一轉 UTC+8（Asia/Taipei）
  4. 篩選「今天往回 3 個月」
  5. XAUUSD：計算 BB(20,2) / EMA50 / EMA200 / Awesome Oscillator
  6. 輸出三份獨立 CSV 至 output/

單獨執行：python module_a.py
"""
import logging
import re
import sys
import time
from pathlib import Path
from dateutil.relativedelta import relativedelta

import pandas as pd
import pandas_ta as ta
import requests
from tvDatafeed import TvDatafeed, Interval

# 允許直接執行（python module_a.py）時找到 public-safe settings
sys.path.insert(0, str(Path(__file__).parent))
from settings import (
    TV_USERNAME, TV_PASSWORD, TV_AUTH_TOKEN,
    TV_SESSIONID, TV_SESSIONID_SIGN, TV_DEVICE_T,
    SYMBOLS, N_BARS, LOOKBACK_MONTHS,
    BB_LENGTH, BB_STD, EMA_FAST, EMA_SLOW,
    OUTPUT_DIR, TIMEZONE,
)

logger = logging.getLogger(__name__)


# ── tvDatafeed 連線 ────────────────────────────────────────────────────────────

def _fetch_token_via_sessionid() -> str | None:
    """用 sessionid cookie 向 TradingView 取得 JWT auth_token。
    Google 登入帳號沒有獨立的 TV 密碼，必須用此方式取得 token。
    """
    if not TV_SESSIONID:
        return None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.tradingview.com/',
        }
        cookies = {
            'sessionid':      TV_SESSIONID,
            'sessionid_sign': TV_SESSIONID_SIGN,
            'device_t':       TV_DEVICE_T,
        }
        r = requests.get('https://www.tradingview.com/', headers=headers,
                         cookies=cookies, timeout=15)
        m = re.search(r'"auth_token"\s*:\s*"([^"]+)"', r.text)
        if m:
            token = m.group(1)
            logger.info("tvDatafeed：sessionid 取得 auth_token 成功（長度 %d）", len(token))
            return token
        logger.warning("tvDatafeed：sessionid 有效但找不到 auth_token，可能已過期")
    except Exception as exc:
        logger.warning("tvDatafeed：sessionid 取得 token 失敗：%s", exc)
    return None


def get_tv_client(max_retries: int = 3) -> TvDatafeed:
    """
    建立 tvDatafeed 連線，認證優先順序：
      1. TV_SESSIONID   → 動態取得 auth_token（Google 登入帳號）
      2. TV_AUTH_TOKEN  → 直接注入靜態 token
      3. TV_USERNAME/PASSWORD → 帳號密碼登入
      4. 匿名模式
    """
    for attempt in range(1, max_retries + 1):
        try:
            tv = TvDatafeed()   # 先建立匿名連線

            if TV_SESSIONID:
                token = _fetch_token_via_sessionid()
                if token:
                    tv.token = token
                    logger.info("tvDatafeed：sessionid 認證完成")
                else:
                    logger.warning("sessionid 認證失敗，降級為匿名模式")
            elif TV_AUTH_TOKEN:
                tv.token = TV_AUTH_TOKEN
                logger.info("tvDatafeed：使用靜態 auth_token 認證")
            elif TV_USERNAME and TV_PASSWORD:
                tv2 = TvDatafeed(TV_USERNAME, TV_PASSWORD)
                if tv2.token != "unauthorized_user_token":
                    tv = tv2
                    logger.info("tvDatafeed：帳號密碼登入成功（%s）", TV_USERNAME)
                else:
                    logger.warning("帳號密碼登入失敗，降級為匿名模式")
            else:
                logger.warning("tvDatafeed：匿名模式（部分商品或歷史深度可能受限）")

            return tv
        except Exception as exc:
            logger.error("連線失敗（第 %d/%d 次）：%s", attempt, max_retries, exc)
            if attempt < max_retries:
                wait = 5 * attempt
                logger.info("等待 %d 秒後重試...", wait)
                time.sleep(wait)
    raise ConnectionError("tvDatafeed 連線失敗，已達最大重試次數")


def fetch_symbol(tv: TvDatafeed, symbol: str, exchange: str,
                 n_bars: int, max_retries: int = 3) -> pd.DataFrame:
    """抓取單一商品 M30 資料，失敗時重試。"""
    for attempt in range(1, max_retries + 1):
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_30_minute,
                n_bars=n_bars,
            )
            if df is None or df.empty:
                raise ValueError("API 回傳空資料")
            logger.info("  %-8s (%s)：取得 %d bars", symbol, exchange, len(df))
            return df
        except Exception as exc:
            logger.error("  %s 抓取失敗（第 %d/%d 次）：%s", symbol, attempt, max_retries, exc)
            if attempt < max_retries:
                wait = 10 * attempt
                logger.info("  等待 %d 秒後重試...", wait)
                time.sleep(wait)
    raise RuntimeError(f"{symbol} 資料抓取失敗（已達最大重試次數）")


# ── Datetime 處理 ──────────────────────────────────────────────────────────────

def to_taipei(df: pd.DataFrame) -> pd.DataFrame:
    """DatetimeIndex 統一轉換為 Asia/Taipei（UTC+8）。"""
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(TIMEZONE)
    df.index.name = "Datetime"
    return df


def filter_lookback(df: pd.DataFrame) -> pd.DataFrame:
    """保留今天往回 LOOKBACK_MONTHS 個月的資料。"""
    cutoff = pd.Timestamp.now(tz=TIMEZONE) - relativedelta(months=LOOKBACK_MONTHS)
    filtered = df[df.index >= cutoff]
    logger.debug("  篩選後：%d 根（cutoff: %s）", len(filtered), cutoff.date())
    return filtered


# ── 技術指標（XAUUSD only） ────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算並附加 BB / EMA / AO。
    pandas-ta 直接接受 Series，與 df 欄位大小寫無關。
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Bollinger Bands (20, 2.0)
    bb = ta.bbands(close, length=BB_LENGTH, std=BB_STD)
    # pandas-ta 實際輸出欄位名帶 ddof 後綴，例如 BBL_20_2.0_2.0
    # 用 startswith 比對前綴，確保版本相容
    prefix_map = {
        "BBL": "BB_Lower",
        "BBM": "BB_Mid",
        "BBU": "BB_Upper",
        "BBB": "BB_Bandwidth",
        "BBP": "BB_Pct",
    }
    rename_bb = {col: prefix_map[col[:3]] for col in bb.columns if col[:3] in prefix_map}
    bb = bb.rename(columns=rename_bb)

    # EMA
    ema_fast = ta.ema(close, length=EMA_FAST).rename(f"EMA_{EMA_FAST}")
    ema_slow = ta.ema(close, length=EMA_SLOW).rename(f"EMA_{EMA_SLOW}")

    # Awesome Oscillator（34-EMA(median) - 5-EMA(median)，TradingView 預設）
    ao = ta.ao(high, low).rename("AO")

    return pd.concat([df, bb, ema_fast, ema_slow, ao], axis=1)


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run_module_a() -> dict[str, pd.DataFrame]:
    """
    執行模組 A，回傳 {'XAUUSD': df, 'DXY': df, 'MGC1!': df}。
    各 df 的 index 為 Asia/Taipei DatetimeIndex，近 3 個月資料。
    """
    logger.info("=" * 60)
    logger.info("【模組 A】熱資料獲取")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tv = get_tv_client()
    results: dict[str, pd.DataFrame] = {}

    for symbol, cfg in SYMBOLS.items():
        logger.info("▶ %s (%s)", symbol, cfg["exchange"])
        is_optional = cfg.get("optional", False)

        # 抓取原始資料
        try:
            df = fetch_symbol(tv, symbol, cfg["exchange"], N_BARS)
        except RuntimeError as exc:
            if is_optional:
                logger.warning("  ⚠  %s 抓取失敗（optional，跳過）：%s", symbol, exc)
                continue
            raise

        # tvDatafeed v2 可能附帶 symbol 欄，統一標準化欄位名
        col_map = {
            "open": "Open", "high": "High",
            "low": "Low",   "close": "Close", "volume": "Volume",
        }
        df = df.rename(columns={c: col_map[c.lower()] for c in df.columns
                                 if c.lower() in col_map})
        # 只保留 OHLCV
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"]
                  if c in df.columns]]

        # Datetime 對齊 + 時間篩選
        df = to_taipei(df)
        df = filter_lookback(df)

        # XAUUSD：加指標
        if cfg["with_indicators"]:
            df = add_indicators(df)
            logger.info("  指標計算完成：BB(%d,%.1f) / EMA%d / EMA%d / AO",
                        BB_LENGTH, BB_STD, EMA_FAST, EMA_SLOW)

        # 輸出 CSV
        csv_path = OUTPUT_DIR / cfg["output_csv"]
        df.to_csv(csv_path)
        logger.info("  → %s（%d 列，%s → %s）",
                    csv_path.name, len(df), df.index[0].date(), df.index[-1].date())

        results[symbol] = df

    logger.info("【模組 A】完成\n")
    return results


# ── 單獨執行入口 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_module_a()
