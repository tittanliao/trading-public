"""
Performance metrics for a trades DataFrame (output of loader.load_trades).

All public functions are pure — they do not modify the input DataFrame.
"""
import pandas as pd
import numpy as np


def summary(trades: pd.DataFrame) -> dict:
    """
    Returns a dict of scalar performance metrics for a strategy.

    Keys: total_trades, wins, losses, win_rate, gross_profit, gross_loss,
          profit_factor, net_pnl, avg_win, avg_loss, avg_mfe_pct_loss,
          avg_mae_pct_loss, avg_hold_bars, max_consec_losses
    """
    wins = trades[trades["result"] == "win"]
    losses = trades[trades["result"] == "loss"]
    total = len(trades)

    gross_profit = wins["net_pnl_usd"].sum()
    gross_loss = abs(losses["net_pnl_usd"].sum())

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / total if total else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
        "net_pnl": trades["net_pnl_usd"].sum(),
        "avg_win": wins["net_pnl_usd"].mean() if len(wins) else 0.0,
        "avg_loss": losses["net_pnl_usd"].mean() if len(losses) else 0.0,
        "avg_mfe_pct_loss": losses["mfe_pct"].mean() if len(losses) else 0.0,
        "avg_mae_pct_loss": losses["mae_pct"].mean() if len(losses) else 0.0,
        "avg_hold_bars": trades["hold_bars"].mean(),
        "max_consec_losses": consecutive_losses(trades).max() if len(trades) else 0,
    }


def drawdown_series(trades: pd.DataFrame) -> pd.Series:
    """Running drawdown from peak equity (always <= 0)."""
    cum = trades["cum_pnl_usd"].reset_index(drop=True)
    peak = cum.cummax()
    return cum - peak


def max_drawdown(trades: pd.DataFrame) -> float:
    return drawdown_series(trades).min()


def consecutive_losses(trades: pd.DataFrame) -> pd.Series:
    """
    Returns a Series where each value is the length of a consecutive-loss streak.
    Example: WLLLLWLLW → [4, 2]
    """
    streaks, count = [], 0
    for r in trades["result"]:
        if r == "loss":
            count += 1
        else:
            if count > 0:
                streaks.append(count)
            count = 0
    if count > 0:
        streaks.append(count)
    return pd.Series(streaks, dtype=int, name="streak_length")


def compare_strategies(strategy_summaries: dict[str, dict]) -> pd.DataFrame:
    """
    Accepts {strategy_id: summary_dict, ...} and returns a comparison DataFrame.
    Useful for side-by-side display in a notebook.
    """
    return pd.DataFrame(strategy_summaries).T.round(4)
