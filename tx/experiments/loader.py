import pandas as pd
from pathlib import Path

CSV_DIR = Path(__file__).parent.parent / "csv"


def load_price(filename: str) -> pd.DataFrame:
    """Load TradingView CSV export (MTX or any instrument).

    Expected columns (whitespace-stripped):
        time, open, high, low, close, RSI, RSI-based MA,
        Regular Bullish, Regular Bearish  [optional]
    """
    path = CSV_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}\n請先把 TradingView 匯出的 CSV 放到 csv/ 資料夾")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalise column names
    rename = {
        "RSI": "rsi",
        "RSI-based MA": "rsi_ma",
        "Regular Bullish": "bull_div",
        "Regular Bullish Label": "bull_div_label",
        "Regular Bearish": "bear_div",
        "Regular Bearish Label": "bear_div_label",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Parse time — TradingView exports with +08:00 or as naive
    df["time"] = pd.to_datetime(df["time"], utc=False)
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)

    df = df.set_index("time").sort_index()

    # Ensure numeric OHLC
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["rsi", "rsi_ma", "bull_div", "bear_div"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["session"] = df.index.map(_session_label)
    return df


def _session_label(dt) -> str:
    t = dt.hour * 60 + dt.minute
    if 8 * 60 + 45 <= t <= 13 * 60 + 45:
        return "day"
    if t >= 15 * 60 or t < 5 * 60:
        return "night"
    return "closed"
