"""
Fail pattern analysis for losing trades.

Classification logic (applied in priority order):
    immediate_loss  — MFE% < IMMEDIATE_LOSS_MFE_PCT
                      Entry direction was wrong from the start; price never moved in our favour.
    false_breakout  — MFE% >= threshold but MAE/MFE ratio > FALSE_BREAKOUT_MAE_MFE_RATIO
                      Price moved in our favour briefly, then fully reversed to SL.
    time_bleed      — hold_bars >= TIME_BLEED_MIN_BARS
                      Trade dragged on for a long time before stopping out.
    normal_sl       — all other losses (clean stop-loss, no special pattern)

Session tags use entry_time assumed to be UTC+8 (Asia/Taipei):
    asia           07:00 – 15:59
    europe         16:00 – 21:59
    us             22:00 – 06:59 (next day)
"""
import pandas as pd
import numpy as np

from analysis.config import (
    IMMEDIATE_LOSS_MFE_PCT,
    TIME_BLEED_MIN_BARS,
    FALSE_BREAKOUT_MAE_MFE_RATIO,
)


# ---------------------------------------------------------------------------
# Session tagging
# ---------------------------------------------------------------------------

def tag_session(entry_time: pd.Series) -> pd.Series:
    """Maps entry hour (UTC+8) to a session label."""
    hour = entry_time.dt.hour
    conditions = [
        (hour >= 7) & (hour <= 15),
        (hour >= 16) & (hour <= 21),
    ]
    choices = ["asia", "europe"]
    return pd.Series(
        np.select(conditions, choices, default="us"),
        index=entry_time.index,
        name="session",
    )


# ---------------------------------------------------------------------------
# Fail classification
# ---------------------------------------------------------------------------

def classify_fail(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to losing trades and adds 'fail_type' and 'session' columns.

    Returns a new DataFrame (subset of trades) — does not modify the input.
    """
    losses = trades[trades["result"] == "loss"].copy()

    mae_mfe_ratio = losses["mae_pct"] / losses["mfe_pct"].replace(0, np.nan)

    conditions = [
        losses["mfe_pct"] < IMMEDIATE_LOSS_MFE_PCT,
        mae_mfe_ratio > FALSE_BREAKOUT_MAE_MFE_RATIO,
        losses["hold_bars"] >= TIME_BLEED_MIN_BARS,
    ]
    choices = ["immediate_loss", "false_breakout", "time_bleed"]

    losses["fail_type"] = np.select(conditions, choices, default="normal_sl")
    losses["session"] = tag_session(losses["entry_time"])
    return losses.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregated views
# ---------------------------------------------------------------------------

def fail_type_summary(classified_losses: pd.DataFrame) -> pd.DataFrame:
    """
    Returns count and percentage breakdown of each fail_type.
    """
    total = len(classified_losses)
    counts = classified_losses["fail_type"].value_counts()
    return pd.DataFrame({
        "count": counts,
        "pct": (counts / total * 100).round(1),
    })


def session_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate and avg P&L broken down by trading session.
    Accepts the full trades DataFrame (wins + losses).
    """
    df = trades.copy()
    df["session"] = tag_session(df["entry_time"])
    return (
        df.groupby("session")
        .agg(
            total=("trade_id", "count"),
            wins=("result", lambda x: (x == "win").sum()),
            avg_pnl=("net_pnl_usd", "mean"),
            avg_mfe=("mfe_pct", "mean"),
            avg_mae=("mae_pct", "mean"),
        )
        .assign(win_rate=lambda d: (d["wins"] / d["total"]).round(3))
        .sort_values("win_rate", ascending=False)
    )


def hourly_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate, trade count, and avg P&L grouped by entry hour (UTC+8, 0–23).
    """
    df = trades.copy()
    df["entry_hour"] = df["entry_time"].dt.hour
    return (
        df.groupby("entry_hour")
        .agg(
            total=("trade_id", "count"),
            wins=("result", lambda x: (x == "win").sum()),
            avg_pnl=("net_pnl_usd", "mean"),
            avg_mfe=("mfe_pct", "mean"),
            avg_mae=("mae_pct", "mean"),
        )
        .assign(win_rate=lambda d: (d["wins"] / d["total"]).round(3))
    )


def fail_by_session(classified_losses: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tab of fail_type × session — how many of each pattern occur per session.
    """
    return pd.crosstab(
        classified_losses["session"],
        classified_losses["fail_type"],
        margins=True,
    )
