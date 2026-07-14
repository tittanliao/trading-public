"""Public-safe runtime settings for the XAUUSD data pipeline.

Credentials are read only from environment variables.  No secret values belong
in this repository.  Paths may be redirected to Private with environment
variables when running the pipeline locally.
"""
from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


TV_USERNAME = os.environ.get("TV_USERNAME", "")
TV_PASSWORD = os.environ.get("TV_PASSWORD", "")
TV_AUTH_TOKEN = os.environ.get("TV_AUTH_TOKEN", "")
TV_SESSIONID = os.environ.get("TV_SESSIONID", "")
TV_SESSIONID_SIGN = os.environ.get("TV_SESSIONID_SIGN", "")
TV_DEVICE_T = os.environ.get("TV_DEVICE_T", "")

SYMBOLS = {
    "XAUUSD": {"exchange": "OANDA", "output_csv": "XAUUSD_M30.csv", "with_indicators": True},
    "DXY": {"exchange": "TVC", "output_csv": "DXY_M30.csv", "with_indicators": False},
    "MGC1!": {"exchange": "COMEX", "output_csv": "MGC1_M30.csv", "with_indicators": False, "optional": True},
}
N_BARS = int(os.environ.get("XAUUSD_N_BARS", "10000"))
LOOKBACK_MONTHS = 3
BB_LENGTH = 20
BB_STD = 2.0
EMA_FAST = 50
EMA_SLOW = 200
TIMEZONE = "Asia/Taipei"

OUTPUT_DIR = _path("XAUUSD_OUTPUT_DIR", HERE / "output")
TV_EXPORT_PATH = _path("XAUUSD_TV_EXPORT_PATH", HERE / "input" / "XAUUSD_TV_export.csv")
DB_PATH = _path("XAUUSD_DB_PATH", HERE / "XAUUSD_M30_Cold.db")
DB_TABLE = "XAUUSD_M30"
OHLC_TOLERANCE = float(os.environ.get("XAUUSD_OHLC_TOLERANCE", "0.0001"))
VALIDATION_FAIL_IF_MISSING = os.environ.get("XAUUSD_VALIDATION_FAIL_IF_MISSING", "true").lower() not in {"0", "false", "no"}
