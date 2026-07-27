"""
Backtest engine for MTX (小台指期).
SL/TP are in points (固定點數). 1 point = NT$50 for MTX.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

POINT_VALUE = 50  # NT$ per point, MTX 小台


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    direction: str = "long",
    sl_pts: int = 30,
    tp_pts: int = 60,
    time_stop_bars: int = 48,
) -> pd.DataFrame:
    """Run a single-instrument bar-by-bar backtest.

    Signals are taken on bar[i]; entry is bar[i+1] open.
    No overlapping trades — next signal is ignored while in a trade.

    Returns a DataFrame of trades.
    """
    assert direction in ("long", "short"), "direction must be 'long' or 'short'"

    ohlc = df[["open", "high", "low", "close"]].values
    sess = df["session"].values if "session" in df.columns else np.full(len(df), "unknown")
    idx = df.index
    sig = signals.reindex(df.index).fillna(False).values

    trades: list[dict] = []
    in_trade = False

    for i in range(len(df) - 1):
        if in_trade or not sig[i]:
            continue

        entry_price = ohlc[i + 1, 0]  # next bar open
        entry_time = idx[i + 1]
        entry_session = sess[i + 1]

        if direction == "long":
            sl = entry_price - sl_pts
            tp = entry_price + tp_pts
        else:
            sl = entry_price + sl_pts
            tp = entry_price - tp_pts

        max_fav_pts = 0.0
        max_adv_pts = 0.0
        exit_price = None
        exit_reason = None
        exit_time = None
        hold_bars = 0

        end_idx = min(i + 1 + time_stop_bars, len(df) - 1)
        for j in range(i + 1, end_idx + 1):
            bar_h = ohlc[j, 1]
            bar_l = ohlc[j, 2]
            bar_c = ohlc[j, 3]
            hold_bars = j - (i + 1)

            if direction == "long":
                max_fav_pts = max(max_fav_pts, bar_h - entry_price)
                max_adv_pts = max(max_adv_pts, entry_price - bar_l)
                if bar_l <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    exit_time = idx[j]
                    break
                if bar_h >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    exit_time = idx[j]
                    break
            else:
                max_fav_pts = max(max_fav_pts, entry_price - bar_l)
                max_adv_pts = max(max_adv_pts, bar_h - entry_price)
                if bar_h >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                    exit_time = idx[j]
                    break
                if bar_l <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                    exit_time = idx[j]
                    break

        if exit_price is None:
            exit_price = ohlc[end_idx, 3]
            exit_reason = "time"
            exit_time = idx[end_idx]

        pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        pnl_ntd = pnl_pts * POINT_VALUE

        fail_pattern = _classify_fail(pnl_pts, max_fav_pts, max_adv_pts, exit_reason, sl_pts)

        trades.append(
            dict(
                entry_time=entry_time,
                exit_time=exit_time,
                session=entry_session,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                exit_reason=exit_reason,
                hold_bars=hold_bars,
                pnl_pts=pnl_pts,
                pnl_ntd=pnl_ntd,
                mfe_pts=max_fav_pts,
                mae_pts=max_adv_pts,
                win=pnl_pts > 0,
                fail_pattern=fail_pattern,
            )
        )
        in_trade = False  # no overlapping trades (can be changed to True for realism)

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)


def _classify_fail(pnl_pts, mfe_pts, mae_pts, exit_reason, sl_pts) -> str:
    if pnl_pts > 0:
        return "win"
    immediate_threshold = sl_pts * 0.15  # ~4.5 pts for 30pt SL
    if mfe_pts < immediate_threshold:
        return "immediate_loss"
    if mfe_pts > 0 and mae_pts / (mfe_pts + 1e-9) > 2.0:
        return "false_breakout"
    if exit_reason == "time":
        return "time_bleed"
    return "normal_sl"


def summary_stats(trades: pd.DataFrame, sl_pts: int, tp_pts: int) -> dict:
    if trades.empty:
        return {}

    n = len(trades)
    wins = trades["win"].sum()
    win_rate = wins / n * 100

    gross_profit = trades.loc[trades["win"], "pnl_ntd"].sum()
    gross_loss = abs(trades.loc[~trades["win"], "pnl_ntd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    net_pnl_pts = trades["pnl_pts"].sum()
    net_pnl_ntd = trades["pnl_ntd"].sum()

    # Max drawdown (equity curve in NT$)
    equity = trades["pnl_ntd"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd_ntd = dd.min()

    return dict(
        n_trades=n,
        win_rate=win_rate,
        profit_factor=profit_factor,
        net_pnl_pts=net_pnl_pts,
        net_pnl_ntd=net_pnl_ntd,
        max_dd_ntd=max_dd_ntd,
        sl_pts=sl_pts,
        tp_pts=tp_pts,
    )
