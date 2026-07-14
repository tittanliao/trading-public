"""
Loads and normalizes CSV files exported from TradingView.

TradingView backtest CSV format (each trade = 2 rows: Entry + Exit):
    Trade #, Type, Date and time, Signal, Price USD, Size (qty),
    Size (value), Net P&L USD, Net P&L %, Favorable excursion USD,
    Favorable excursion %, Adverse excursion USD, Adverse excursion %,
    Cumulative P&L USD, Cumulative P&L %

load_trades() merges entry/exit pairs into one row per trade.
load_price()  reads the 30-min OHLCV candle file.
"""
from pathlib import Path

import pandas as pd


def load_trades(csv_path: Path) -> pd.DataFrame:
    """
    Returns one row per completed trade with both entry and exit fields.

    Output columns:
        trade_id        int    — TradingView trade number
        entry_time      datetime64[ns]
        exit_time       datetime64[ns]
        entry_signal    str    — e.g. S1_LE, S2_Rev
        exit_signal     str    — e.g. S1_SL, S1_LX_TP1
        entry_price     float
        exit_price      float
        size_qty        float
        size_value      float
        net_pnl_usd     float
        net_pnl_pct     float
        mfe_usd         float  — Maximum Favorable Excursion
        mfe_pct         float
        mae_usd         float  — Maximum Adverse Excursion
        mae_pct         float
        cum_pnl_usd     float
        cum_pnl_pct     float
        hold_bars       int    — number of 30-min bars held
        result          str    — "win" | "loss" | "breakeven"
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()

    # Newer TradingView exports use different header names for the same fields
    # (e.g. "Trade number" instead of "Trade #", "Net PnL USD" instead of
    # "Net P&L USD", "Return %" instead of "Net P&L %"). Normalize here so
    # both formats work.
    raw = raw.rename(columns={
        "Trade number": "Trade #",
        "Net PnL USD": "Net P&L USD",
        "Return %": "Net P&L %",
        "Cumulative PnL USD": "Cumulative P&L USD",
        "Cumulative PnL %": "Cumulative P&L %",
    })

    entries = raw[raw["Type"] == "Entry long"].rename(columns={
        "Date and time": "entry_time",
        "Signal": "entry_signal",
        "Price USD": "entry_price",
    })[["Trade #", "entry_time", "entry_signal", "entry_price"]]

    exits = raw[raw["Type"] == "Exit long"].rename(columns={
        "Date and time": "exit_time",
        "Signal": "exit_signal",
        "Price USD": "exit_price",
    })

    trades = entries.merge(
        exits[[
            "Trade #", "exit_time", "exit_signal", "exit_price",
            "Size (qty)", "Size (value)",
            "Net P&L USD", "Net P&L %",
            "Favorable excursion USD", "Favorable excursion %",
            "Adverse excursion USD", "Adverse excursion %",
            "Cumulative P&L USD", "Cumulative P&L %",
        ]],
        on="Trade #",
        how="inner",
    ).rename(columns={
        "Trade #": "trade_id",
        "Size (qty)": "size_qty",
        "Size (value)": "size_value",
        "Net P&L USD": "net_pnl_usd",
        "Net P&L %": "net_pnl_pct",
        "Favorable excursion USD": "mfe_usd",
        "Favorable excursion %": "mfe_pct",
        "Adverse excursion USD": "mae_usd",
        "Adverse excursion %": "mae_pct",
        "Cumulative P&L USD": "cum_pnl_usd",
        "Cumulative P&L %": "cum_pnl_pct",
    })

    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    # 30-min bars held (1 bar = 1800 seconds)
    trades["hold_bars"] = (
        (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 1800
    ).astype(int)

    trades["result"] = trades["net_pnl_usd"].apply(
        lambda x: "win" if x > 0 else ("loss" if x < 0 else "breakeven")
    )

    return trades.sort_values("entry_time").reset_index(drop=True)


def _compute_rsi_locally(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute RSI(14) + RSI-based MA(14) from close when the TradingView export
    is plain OHLC (no indicator overlay columns) — e.g. 20260711 XAUUSD/DXY exports.

    Matches XAUUSD-Macro-v3.7.pine's indicator definition: Wilder's RSI(14) on
    close (RMA smoothing) + SMA(14) of RSI as the "RSI-based MA". Uses
    ewm(alpha=1/14, adjust=False) as a vectorized approximation of RMA — the
    seed differs slightly from TradingView's SMA-seeded warm-up, but the
    transient decays within ~50 bars, well before MaxUseBars-gated trades begin.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - 100 / (1 + rs)
    df["rsi_ma"] = df["rsi"].rolling(14).mean()
    return df


def _parse_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw TradingView price/indicator CSV into a clean DataFrame."""
    df["time"] = (
        pd.to_datetime(df["time"], utc=True)
        .dt.tz_convert("Asia/Taipei")
        .dt.tz_localize(None)
    )
    rename_map = {
        "RSI": "rsi",
        "RSI-based MA": "rsi_ma",
        "Regular Bullish": "bull_div",
        "Regular Bearish": "bear_div",
    }
    df = df.rename(columns=rename_map)

    if "rsi" not in df.columns:
        # Export has no indicator overlay (plain OHLC) — compute locally so
        # divergence/MTF/DXY-RSI analysis downstream still has real values
        # instead of silently degrading to NaN/"unknown".
        print("  [loader] RSI/RSI-MA missing from CSV — computed locally (Wilder RSI14 + SMA14)")
        df = _compute_rsi_locally(df)

    keep = ["time", "open", "high", "low", "close"]
    for col in ["rsi", "rsi_ma", "bull_div", "bear_div"]:
        if col in df.columns:
            keep.append(col)

    return df[keep].sort_values("time").reset_index(drop=True)


def load_price(csv_path: Path) -> pd.DataFrame:
    """
    Reads a 30-min (or any timeframe) OHLCV file exported from TradingView.

    Normalised columns: time, open, high, low, close, rsi, rsi_ma,
                        bull_div, bear_div  (indicator columns present when exported).
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    return _parse_price_df(raw)


def load_dxy(csv_path: Path) -> pd.DataFrame:
    """
    Reads a DXY CSV exported from TradingView (any timeframe).
    Returns same column layout as load_price().
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    return _parse_price_df(raw)
