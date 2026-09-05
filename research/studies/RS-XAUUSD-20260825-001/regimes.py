#!/usr/bin/env python3
"""Regime states at three time scales, defined once so studies can share them.

The question these exist to answer is not "what will the market do" but "is this a moment
when a strategy that already exists tends to work". That is a much smaller question than
alpha discovery: the sample is the strategy's own trades, not every bar, and it has an
obvious control — the same strategy with no regime condition at all.

Three scales are provided because it is not known in advance which one carries anything,
and they demand different evidence:

- **intraday** — session, position within the day's range. Changes several times a day, so
  a trade set of a few hundred spreads thinly across states.
- **swing** (days to weeks) — Hurst, realised volatility, band width. This is the scale the
  handbook's Hurst state machine proposed, tested here rather than assumed.
- **macro** (months to quarters) — trend relative to the 200-day mean, drawdown from the
  52-week high, the direction of real yields and the dollar. Few independent episodes even
  across years, which the resolution bound will show.

## Every state is knowable at the moment it is applied

Each definition uses only trailing windows, and the quantile cuts use an expanding rank
with a warm-up rather than a full-sample quantile. A full-sample quantile decides in 2024
what "high volatility" means using 2026 data — the same class of error that made a
dollar-weakness condition look like a 71% win rate until its comparison group was matched
to its own era.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


WARMUP = 250


def expanding_bucket(series: pd.Series, labels: tuple[str, ...],
                     warmup: int = WARMUP) -> pd.Series:
    """Cut a series into equal-width quantile buckets using only prior history."""
    rank = series.expanding(min_periods=warmup).rank(pct=True)
    edges = np.linspace(0, 1, len(labels) + 1)
    out = pd.Series("unknown", index=series.index, dtype=object)
    for i, label in enumerate(labels):
        lower, upper = edges[i], edges[i + 1]
        mask = (rank > lower) & (rank <= upper) if i else (rank >= 0) & (rank <= upper)
        out[mask.fillna(False)] = label
    return out


def hurst(values: np.ndarray, lags: tuple[int, ...] = (2, 4, 8, 16, 32)) -> float:
    """Hurst exponent from how the spread of k-step moves scales with k.

    The standard deviation of k-step differences grows like k**H. Regressing log(sd) on
    log(k) recovers H: 0.5 is a random walk, above 0.5 trends persist, below 0.5 they
    revert. Implemented directly because the `hurst` package is not a dependency here and
    this estimator is four lines.

    Returns NaN rather than a number when the window is too short or degenerate — a Hurst
    of exactly 0.5 from a failed fit would be indistinguishable from a real random walk.
    """
    values = np.asarray(values, dtype=float)
    if values.size < max(lags) * 4 or not np.all(np.isfinite(values)):
        return float("nan")
    spreads = []
    for lag in lags:
        diff = values[lag:] - values[:-lag]
        if diff.size < 8:
            return float("nan")
        spread = diff.std(ddof=1)
        if spread <= 0:
            return float("nan")
        spreads.append(spread)
    slope = np.polyfit(np.log(lags), np.log(spreads), 1)[0]
    return float(slope)


def intraday_states(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """Session and position within the day. `bars` needs `time`, `high`, `low`, `close`."""
    time = bars["time"]
    hour = time.dt.hour + time.dt.minute / 60.0

    session = pd.Series("asia", index=bars.index, dtype=object)
    session[(hour >= 15) & (hour < 21)] = "london"
    session[(hour >= 21) | (hour < 7)] = "new_york"

    # The session day starts at 07:00 Taipei, the boundary every study here uses.
    day = (time - pd.Timedelta(hours=7)).dt.date
    grouped = bars.groupby(day)
    running_high = grouped["high"].cummax()
    running_low = grouped["low"].cummin()
    done = running_high - running_low

    # Compared against the median completed day, computed on prior days only.
    daily_range = grouped["high"].max() - grouped["low"].min()
    typical = daily_range.shift(1).expanding(min_periods=20).median()
    reference = pd.Series(day).map(typical).to_numpy()
    completion = pd.Series(done.to_numpy() / reference, index=bars.index)

    return {
        "session": session,
        "range_completion": expanding_bucket(
            completion, ("early", "middle", "late")),
        "hour_block": pd.cut(hour, [-0.01, 7, 12, 17, 21, 24.01],
                             labels=["overnight", "asia_am", "asia_pm", "london", "ny"]
                             ).astype(object),
    }


def swing_states(bars: pd.DataFrame, window: int = 240) -> dict[str, pd.Series]:
    """Days-to-weeks state. `window` is in bars — 240 x 30min is about five sessions."""
    close = bars["close"]
    returns = close.pct_change()

    h = close.rolling(window).apply(lambda w: hurst(w), raw=True)
    realised = returns.rolling(20).std(ddof=1)

    hurst_state = pd.Series("unknown", index=bars.index, dtype=object)
    hurst_state[h > 0.55] = "trending"
    hurst_state[(h >= 0.45) & (h <= 0.55)] = "random_walk"
    hurst_state[h < 0.45] = "mean_reverting"

    # Bollinger band width is deliberately NOT a separate state. Width is
    # 2 x rolling_sd x close / rolling_mean, and close/rolling_mean sits so close to 1 that
    # it assigned 99.70% of bars to the same bucket as realised volatility on this data.
    # Two names for one variable inflate a family correction's denominator and make a
    # coincidence look like two independent hits.
    return {
        "hurst_regime": hurst_state,
        "hurst_tercile": expanding_bucket(h, ("hurst_low", "hurst_mid", "hurst_high")),
        "realised_vol": expanding_bucket(realised, ("vol_low", "vol_mid", "vol_high")),
    }


def macro_states(bars: pd.DataFrame, daily: pd.DataFrame,
                 macro: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Months-to-quarters state, joined from daily bars onto the intraday index.

    `daily` needs `date` and `close`; `macro` may carry `date`, `DFII10`, `DTWEXBGS`.
    The join is backward-looking: a bar at 10:00 sees the previous completed daily close,
    never its own day's.
    """
    frame = daily.sort_values("date").reset_index(drop=True).copy()
    frame["ma200"] = frame["close"].rolling(200).mean()
    frame["high52"] = frame["close"].rolling(252).max()
    frame["drawdown"] = frame["close"] / frame["high52"] - 1
    frame["trend"] = np.where(frame["close"].isna() | frame["ma200"].isna(), "unknown",
                              np.where(frame["close"] > frame["ma200"],
                                       "above_200d", "below_200d"))
    frame["drawdown_state"] = expanding_bucket(
        -frame["drawdown"], ("shallow", "moderate", "deep"))

    if macro is not None:
        merged = frame.merge(macro, on="date", how="left")
        for column, name in (("DFII10", "real_yield"), ("DTWEXBGS", "dollar")):
            if column in merged:
                change = merged[column] - merged[column].shift(20)
                merged[f"{name}_direction"] = np.where(
                    change.isna(), "unknown",
                    np.where(change < 0, f"{name}_falling", f"{name}_rising"))
        frame = merged

    # Shift by one day so an intraday bar never reads the daily bar it belongs to.
    columns = [c for c in ("trend", "drawdown_state", "real_yield_direction",
                           "dollar_direction") if c in frame]
    lookup = frame[["date"] + columns].copy()
    for column in columns:
        lookup[column] = lookup[column].shift(1)
    lookup["date"] = pd.to_datetime(lookup["date"])

    joined = pd.merge_asof(
        bars[["time"]].sort_values("time"),
        lookup.sort_values("date"),
        left_on="time", right_on="date", direction="backward",
    )
    return {column: joined[column].fillna("unknown").reset_index(drop=True)
            for column in columns}


def all_states(bars: pd.DataFrame, daily: pd.DataFrame,
               macro: pd.DataFrame | None = None) -> dict[str, dict[str, pd.Series]]:
    """Every state variable, grouped by the scale it belongs to."""
    return {
        "intraday": intraday_states(bars),
        "swing": swing_states(bars),
        "macro": macro_states(bars, daily, macro),
    }
