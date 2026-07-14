"""
run_pipeline.py — XAUUSD M30 自動化資料管線 主入口
═════════════════════════════════════════════════════
執行順序：Module A → Module B → Module C

  A：tvDatafeed 抓取熱資料，輸出三份週報 CSV
  B：與 TV 手動匯出 CSV 比對驗證（OHLC 嚴格防呆）
  C：驗證通過後 UPSERT 進 SQLite 冷庫

用法
----
  全流程（最常用）：
    python run_pipeline.py

  只產週報 CSV，不寫 DB（例如每日更新）：
    python run_pipeline.py --skip-db

  跳過 B 驗證直接 UPSERT（緊急情況，需明確加 --force）：
    python run_pipeline.py --skip-val --force

  指定 Log 等級（DEBUG 可看詳細差值）：
    python run_pipeline.py --log-level DEBUG
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 確保同目錄下的模組可以被找到
sys.path.insert(0, str(Path(__file__).parent))

from module_a import run_module_a
from module_b import run_module_b
from module_c import run_module_c


# ── Logging 設定 ───────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Log 輸出：%s", log_file)


# ── CLI 引數 ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XAUUSD M30 自動化資料管線（A→B→C）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="跳過 Module C（不寫入 SQLite 冷庫，只產週報 CSV）",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="跳過 Module B 驗證（需搭配 --force）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="與 --skip-val 搭配，確認有意跳過驗證防呆",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log 詳細等級（預設 INFO）",
    )
    return parser.parse_args()


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # --skip-val 需要 --force 明確確認
    if args.skip_val and not args.force:
        print("錯誤：跳過驗證（--skip-val）需明確加上 --force 確認。")
        print("      跳過驗證意味著未經比對就寫入冷庫，請謹慎使用。")
        sys.exit(1)

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  XAUUSD M30 自動化資料管線 V3                       ║")
    logger.info("║  冷熱分離架構 | 嚴格 OHLC 驗證 | SQLite UPSERT      ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info("啟動時間：%s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if args.skip_db:
        logger.warning("⚠  --skip-db：本次不寫入冷庫")
    if args.skip_val:
        logger.warning("⚠  --skip-val --force：本次跳過 OHLC 驗證防呆")

    try:
        # ── Module A：熱資料獲取 ───────────────────────────────────────────────
        results    = run_module_a()
        hot_xauusd = results["XAUUSD"]

        # ── Module B：驗證防呆 ─────────────────────────────────────────────────
        if not args.skip_val:
            run_module_b(hot_xauusd)
        else:
            logger.warning("【模組 B】已跳過（--skip-val --force）\n")

        # ── Module C：UPSERT 冷庫 ──────────────────────────────────────────────
        if not args.skip_db:
            run_module_c(hot_xauusd)
        else:
            logger.info("【模組 C】已跳過（--skip-db）\n")

        # ── 完成 ───────────────────────────────────────────────────────────────
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  ✅ 管線完成                                         ║")
        logger.info("╚══════════════════════════════════════════════════════╝")

    except FileNotFoundError as exc:
        logger.error("❌ 找不到必要檔案：%s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("❌ 資料驗證失敗：%s", exc)
        sys.exit(1)
    except ConnectionError as exc:
        logger.error("❌ 網路連線問題：%s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("❌ 執行階段錯誤：%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("❌ 未預期錯誤：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
