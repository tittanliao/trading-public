"""
Runs all 20 strategies on the price data and returns results as a DataFrame.
"""
from __future__ import annotations

import pandas as pd

from experiments.engine import run_backtest, run_backtest_short, summary, Trade
from experiments.strategies import STRATEGIES
from experiments.strategies_short import STRATEGIES as SHORT_STRATEGIES


def run_all(
    price: pd.DataFrame,
    htf_filter: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[Trade]]]:
    """
    Returns:
        results_df  — one row per strategy with all summary metrics
        trades_map  — {strategy_id: [Trade, ...]}
    """
    rows = []
    trades_map: dict[str, list[Trade]] = {}

    for strat_id, (fn, group, description) in STRATEGIES.items():
        trades = run_backtest(price, fn, htf_filter=htf_filter)
        summ   = summary(trades)
        trades_map[strat_id] = trades

        rows.append({
            "id":           strat_id,
            "group":        group,
            "description":  description,
            **summ,
        })

    df = pd.DataFrame(rows).set_index("id")
    return df, trades_map


def run_all_short(
    price: pd.DataFrame,
    htf_filter: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[Trade]]]:
    """
    Same as run_all() but for the 20 short strategies.
    htf_filter blocks entries when 4H is bullish (counter-trend short).
    """
    rows = []
    trades_map: dict[str, list[Trade]] = {}

    for strat_id, (fn, group, description) in SHORT_STRATEGIES.items():
        trades = run_backtest_short(price, fn, htf_filter=htf_filter)
        summ   = summary(trades)
        trades_map[strat_id] = trades

        rows.append({
            "id":           strat_id,
            "group":        group,
            "description":  description,
            **summ,
        })

    df = pd.DataFrame(rows).set_index("id")
    return df, trades_map


def score(results: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score for ranking strategies:
        - Win rate      (25%)
        - Profit factor (25%, capped at 3.0 to avoid inf domination)
        - Trade count   (20%, log-scaled — more trades = more reliable signal)
        - Net P&L %     (20%)
        - Max consec loss penalty (10%, inverted)

    Returns results with a 'score' column, sorted descending.
    """
    r = results.copy()

    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    import numpy as np
    pf_capped = r["profit_factor"].clip(upper=3.0)
    trade_log = np.log1p(r["total"])

    r["score"] = (
        _norm(r["win_rate"])     * 0.25 +
        _norm(pf_capped)         * 0.25 +
        _norm(trade_log)         * 0.20 +
        _norm(r["net_pnl_pct"])  * 0.20 +
        _norm(-r["max_consec_loss"]) * 0.10
    )

    return r.sort_values("score", ascending=False)
