"""
Programmatic RSI divergence detection for XAUUSD analysis.

Bullish divergence (potential long reversal):
  - Price makes a lower low (swing low 2 < swing low 1)
  - RSI makes a higher low at the same two swing points

Bearish divergence (potential short reversal / long failure reason):
  - Price makes a higher high (swing high 2 > swing high 1)
  - RSI makes a lower high at the same two swing points

The existing CSV already carries `rsi` and `rsi_ma` columns exported from
TradingView, so no recomputation of RSI is needed.

Public API
----------
find_swing_lows(price, lookback=5)            → pd.DataFrame
find_swing_highs(price, lookback=5)           → pd.DataFrame
detect_bull_divergence(price, swing_df,
                       max_bars_between=40)   → pd.Series (bool, indexed by time)
detect_bear_divergence(price, swing_df,
                       max_bars_between=40)   → pd.Series (bool, indexed by time)
enrich_trades_with_divergence(trades, price,
                              lookback_bars=3) → pd.DataFrame
divergence_stats(enriched)                    → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Swing low / high detection
# ---------------------------------------------------------------------------

def find_swing_lows(price: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Identifies swing-low bars in *price*.

    A bar at integer position i is a swing low when:
        price.low[i] <= all values of price.low[i-lookback : i+lookback+1]

    Parameters
    ----------
    price    : DataFrame with columns ``time``, ``low``, ``rsi``.
               Must be sorted by time (ascending) with a default RangeIndex.
    lookback : Number of bars on each side to compare against.

    Returns
    -------
    DataFrame with columns ``time``, ``low``, ``rsi``, ``bar_idx``
    (integer position in *price*), sorted by time.
    """
    lows = price["low"].to_numpy()
    n = len(lows)

    is_swing = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] <= window.min():
            is_swing[i] = True

    idx = np.where(is_swing)[0]
    result = price.iloc[idx][["time", "low", "rsi"]].copy()
    result["bar_idx"] = idx
    return result.reset_index(drop=True)


def find_swing_highs(price: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Identifies swing-high bars in *price*.

    A bar at integer position i is a swing high when:
        price.high[i] >= all values of price.high[i-lookback : i+lookback+1]

    Returns
    -------
    DataFrame with columns ``time``, ``high``, ``rsi``, ``bar_idx``.
    """
    highs = price["high"].to_numpy()
    n = len(highs)

    is_swing = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window = highs[i - lookback : i + lookback + 1]
        if highs[i] >= window.max():
            is_swing[i] = True

    idx = np.where(is_swing)[0]
    result = price.iloc[idx][["time", "high", "rsi"]].copy()
    result["bar_idx"] = idx
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Divergence detection
# ---------------------------------------------------------------------------

def detect_bull_divergence(
    price: pd.DataFrame,
    swing_df: pd.DataFrame,
    max_bars_between: int = 40,
) -> pd.Series:
    """
    Detects bullish RSI divergence between consecutive swing lows.

    For each consecutive pair (L1, L2) where L2 comes after L1:
      - If bars between them > max_bars_between: skip (too far apart)
      - Bullish divergence fires at L2 when:
          price_low(L2) < price_low(L1)   AND   rsi(L2) > rsi(L1)

    Parameters
    ----------
    price            : Full OHLCV+RSI DataFrame (used for its ``time`` index).
    swing_df         : Output of ``find_swing_lows()``.
    max_bars_between : Maximum bar distance between L1 and L2 to consider.

    Returns
    -------
    Boolean pd.Series indexed by *time* values from *price*.
    True at times where a bullish divergence signal fires.
    """
    signal_times: set[pd.Timestamp] = set()

    swings = swing_df.sort_values("bar_idx").reset_index(drop=True)
    if len(swings) < 2:
        return pd.Series(False, index=price["time"])

    for i in range(len(swings) - 1):
        l1 = swings.iloc[i]
        l2 = swings.iloc[i + 1]

        bars_between = int(l2["bar_idx"]) - int(l1["bar_idx"])
        if bars_between > max_bars_between:
            continue

        price_diverge = l2["low"] < l1["low"]
        rsi_diverge   = l2["rsi"] > l1["rsi"]

        if price_diverge and rsi_diverge:
            signal_times.add(l2["time"])

    result = pd.Series(
        price["time"].isin(signal_times).values,
        index=price["time"],
        name="bull_div_signal",
    )
    return result


def detect_bear_divergence(
    price: pd.DataFrame,
    swing_df: pd.DataFrame,
    max_bars_between: int = 40,
) -> pd.Series:
    """
    Detects bearish RSI divergence between consecutive swing highs.

    For each consecutive pair (H1, H2) where H2 comes after H1:
      - If bars between them > max_bars_between: skip
      - Bearish divergence fires at H2 when:
          price_high(H2) > price_high(H1)   AND   rsi(H2) < rsi(H1)

    Parameters
    ----------
    price            : Full OHLCV+RSI DataFrame.
    swing_df         : Output of ``find_swing_highs()``.
    max_bars_between : Maximum bar distance between H1 and H2 to consider.

    Returns
    -------
    Boolean pd.Series indexed by *time* values from *price*.
    True at times where a bearish divergence signal fires.
    """
    signal_times: set[pd.Timestamp] = set()

    swings = swing_df.sort_values("bar_idx").reset_index(drop=True)
    if len(swings) < 2:
        return pd.Series(False, index=price["time"])

    for i in range(len(swings) - 1):
        h1 = swings.iloc[i]
        h2 = swings.iloc[i + 1]

        bars_between = int(h2["bar_idx"]) - int(h1["bar_idx"])
        if bars_between > max_bars_between:
            continue

        price_diverge = h2["high"] > h1["high"]
        rsi_diverge   = h2["rsi"] < h1["rsi"]

        if price_diverge and rsi_diverge:
            signal_times.add(h2["time"])

    result = pd.Series(
        price["time"].isin(signal_times).values,
        index=price["time"],
        name="bear_div_signal",
    )
    return result


# ---------------------------------------------------------------------------
# 3. Trade enrichment
# ---------------------------------------------------------------------------

def enrich_trades_with_divergence(
    trades: pd.DataFrame,
    price: pd.DataFrame,
    lookback_bars: int = 3,
    swing_lookback: int = 5,
    max_bars_between: int = 40,
) -> pd.DataFrame:
    """
    Adds pre-entry divergence context columns to each trade.

    For each trade, the function checks whether a bullish or bearish
    divergence signal fired within *lookback_bars* bars **before** entry.
    "Before entry" means the half-open window
    ``(entry_time - lookback_bars*30min, entry_time]``
    (the entry bar itself is included because divergence on the entry bar
    can still influence the decision).

    Uses ``merge_asof`` / binary-search for O(n log n) performance — no
    row-by-row Python loops over trades.

    Parameters
    ----------
    trades        : DataFrame with at least ``entry_time`` and ``trade_id``.
    price         : 30-min OHLCV+RSI DataFrame (output of ``loader.load_price``).
    lookback_bars : How many bars back from entry to look for divergence signals.
    swing_lookback: ``lookback`` passed to ``find_swing_lows/highs``.
    max_bars_between: Passed to ``detect_bull/bear_divergence``.

    Columns added
    -------------
    pre_bull_div        bool — any bullish divergence signal in look-back window
    pre_bear_div        bool — any bearish divergence signal in look-back window
    bars_since_bull_div int  — bars since most recent bullish divergence; -1 if none
    """
    # --- compute divergence signals on the price series ---
    sw_lows  = find_swing_lows(price, lookback=swing_lookback)
    sw_highs = find_swing_highs(price, lookback=swing_lookback)

    bull_sig = detect_bull_divergence(price, sw_lows,  max_bars_between)
    bear_sig = detect_bear_divergence(price, sw_highs, max_bars_between)

    # Build a compact events DataFrame (only bars where a signal exists)
    price_times = price["time"].reset_index(drop=True)

    bull_events = (
        price_times[bull_sig.values].reset_index(drop=True).rename("signal_time")
        .to_frame()
    )
    bull_events.index = bull_events["signal_time"]  # time-indexed for merge_asof

    bear_events = (
        price_times[bear_sig.values].reset_index(drop=True).rename("signal_time")
        .to_frame()
    )
    bear_events.index = bear_events["signal_time"]

    # Map every price-bar time to its integer position (for bar-distance math)
    time_to_bar: dict[pd.Timestamp, int] = {
        t: i for i, t in enumerate(price["time"])
    }

    # --- vectorised look-up with merge_asof ---
    window_seconds = lookback_bars * 30 * 60  # lookback in seconds

    # _enrich_one_div expects a RangeIndex so that positional alignment works
    trades_sorted = trades.sort_values("entry_time").reset_index(drop=True)

    def _enrich_one_div(
        trades_df: pd.DataFrame,
        events_df: pd.DataFrame,
        col_present: str,
        col_bars: str,
    ) -> tuple[pd.Series, pd.Series]:
        """
        For each trade, find the most recent signal at or before entry_time,
        then decide whether it falls within the look-back window.

        trades_df must have a clean RangeIndex (reset before calling).

        Returns two Series aligned to trades_df.index:
            col_present — bool, signal present in window
            col_bars    — int, bars since signal (-1 = none in window)
        """
        if events_df.empty:
            present = pd.Series(False, index=trades_df.index, name=col_present)
            bars    = pd.Series(-1,    index=trades_df.index, name=col_bars)
            return present, bars

        # merge_asof requires both sides sorted; reset index first to avoid
        # ambiguity when index name == column name ("signal_time")
        events_sorted = events_df.reset_index(drop=True).sort_values("signal_time").reset_index(drop=True)

        # trades_df must be sorted by entry_time for merge_asof
        merged = pd.merge_asof(
            trades_df[["entry_time"]].reset_index(drop=True),  # positional 0..n-1
            events_sorted.rename(columns={"signal_time": "last_signal"}),
            left_on="entry_time",
            right_on="last_signal",
            direction="backward",
        )
        # merged is aligned row-for-row with trades_df (same length, same order)

        window_td = pd.Timedelta(seconds=window_seconds)

        in_window = (
            merged["last_signal"].notna()
            & ((merged["entry_time"] - merged["last_signal"]) <= window_td)
        )

        # Bars since signal: use integer bar positions from time_to_bar map
        entry_bar_idx = merged["entry_time"].map(time_to_bar)
        sig_bar_idx   = merged["last_signal"].map(time_to_bar)
        bars_since = (entry_bar_idx - sig_bar_idx).where(in_window, other=-1).fillna(-1).astype(int)

        return (
            pd.Series(in_window.values, index=trades_df.index, name=col_present),
            pd.Series(bars_since.values, index=trades_df.index, name=col_bars),
        )

    pre_bull, bars_bull = _enrich_one_div(
        trades_sorted, bull_events,
        col_present="pre_bull_div",
        col_bars="bars_since_bull_div",
    )
    pre_bear, _ = _enrich_one_div(
        trades_sorted, bear_events,
        col_present="pre_bear_div",
        col_bars="_bear_tmp",
    )

    # trades_sorted has a RangeIndex 0..n-1 aligned with trades sorted by entry_time.
    # Attach the original index values so we can map back to trades' original index.
    orig_index = trades.sort_values("entry_time").index
    pre_bull_mapped    = pd.Series(pre_bull.values,    index=orig_index, name="pre_bull_div")
    pre_bear_mapped    = pd.Series(pre_bear.values,    index=orig_index, name="pre_bear_div")
    bars_bull_mapped   = pd.Series(bars_bull.values,   index=orig_index, name="bars_since_bull_div")

    result = trades.copy()
    result["pre_bull_div"]        = pre_bull_mapped.reindex(result.index).fillna(False).astype(bool)
    result["pre_bear_div"]        = pre_bear_mapped.reindex(result.index).fillna(False).astype(bool)
    result["bars_since_bull_div"] = bars_bull_mapped.reindex(result.index).fillna(-1).astype(int)

    return result


# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------

def divergence_stats(enriched: pd.DataFrame) -> dict:
    """
    Computes win-rate breakdowns by pre-entry divergence state.

    Parameters
    ----------
    enriched : Output of ``enrich_trades_with_divergence()``.
               Must contain ``result`` (``"win"`` | ``"loss"`` | ``"breakeven"``),
               ``pre_bull_div``, and ``pre_bear_div``.

    Returns
    -------
    dict with keys:
        ``"by_bull_div"`` — DataFrame comparing trades with vs without bullish div
        ``"by_bear_div"`` — DataFrame comparing trades with vs without bearish div

    Each DataFrame has index values ``True`` / ``False`` and columns:
        total, wins, win_rate, avg_pnl_usd (if ``net_pnl_usd`` present)
    """

    def _group_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
        rows = []
        for flag in [True, False]:
            grp = df[df[col] == flag]
            total = len(grp)
            if total == 0:
                rows.append({col: flag, "total": 0, "wins": 0,
                             "win_rate": float("nan"), "avg_pnl_usd": float("nan")})
                continue
            wins = (grp["result"] == "win").sum()
            avg_pnl = (
                grp["net_pnl_usd"].mean()
                if "net_pnl_usd" in grp.columns
                else float("nan")
            )
            rows.append({
                col: flag,
                "total": int(total),
                "wins": int(wins),
                "win_rate": round(wins / total, 3),
                "avg_pnl_usd": round(avg_pnl, 2) if not np.isnan(avg_pnl) else float("nan"),
            })
        return pd.DataFrame(rows).set_index(col)

    return {
        "by_bull_div": _group_stats(enriched, "pre_bull_div"),
        "by_bear_div": _group_stats(enriched, "pre_bear_div"),
    }
