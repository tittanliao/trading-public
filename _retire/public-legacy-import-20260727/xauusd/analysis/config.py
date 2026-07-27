"""
Central configuration for all XAUUSD strategies.

To add a new strategy: append a dict to STRATEGIES following the same keys.
The analysis pipeline will automatically pick it up.
"""
from pathlib import Path

ROOT    = Path(__file__).parent.parent
CSV_DIR = ROOT / "csv" / "20260711"  # 20260711 重匯到最新日期；舊版存於 csv/20260705/
                                       # ⚠️ 新匯出無 RSI/RSI-based MA/背離欄位（純OHLC），
                                       # loader._parse_price_df 會自動用 close 本地補算 RSI(14)+SMA(14)

STRATEGIES = [
    {
        "id": "S1-AweWithBB",       # Right-side breakout: BB + AO momentum
        "version": "3.4",
        "folder": ROOT / "XAUUSD-Long-S1-AweWithBB",
        "trades_csv": "S1-Awe-V3.4_FX_IDC_XAUUSD_2026-04-26.csv",
    },
    {
        "id": "S2-RSI",            # Left-side reversion: indicator-triggered (RSI crossover / divergence)
        "version": "2.0",
        "folder": ROOT / "XAUUSD-Long-S2-RSI",
        "trades_csv": "S2-Hybrid-V2.0_FX_IDC_XAUUSD_2026-04-26.csv",
    },
    {
        "id": "S2-Hammer",         # Left-side reversion: price-action triggered (hammer candle)
        "version": "1.9",          # 現行確認版基準。V3.2測試版報告另存 report_v3.2.html（20260711
        "folder": ROOT / "XAUUSD-Long-S2-Hammer",  # 手動切config.py+重跑main.py產生，不隨此設定自動更新）
        "trades_csv": "S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-07-11.csv",
    },
]

PRICE_CSV     = CSV_DIR / "FX_IDC_XAUUSD, 30.csv"
PRICE_CSV_60M = CSV_DIR / "FX_IDC_XAUUSD, 60.csv"
PRICE_CSV_4H  = CSV_DIR / "FX_IDC_XAUUSD, 240.csv"
DXY_CSV_30    = CSV_DIR / "TVC_DXY, 30.csv"
DXY_CSV_1D    = CSV_DIR / "TVC_DXY, 1D.csv"
XAUUSD_CSV_1D = CSV_DIR / "FX_IDC_XAUUSD, 1D.csv"

# --- Fail pattern classification thresholds ---
# A loss where MFE% never exceeded this value is "immediate_loss" (entry was wrong instantly)
IMMEDIATE_LOSS_MFE_PCT = 0.10

# A losing trade that held >= this many 30-min bars before stopping out = "time_bleed"
TIME_BLEED_MIN_BARS = 24  # 12 hours

# A loss where MFE was positive but MAE/MFE ratio is high = "false_breakout"
# i.e. it moved in our favour but then reversed fully to SL
FALSE_BREAKOUT_MAE_MFE_RATIO = 2.0
