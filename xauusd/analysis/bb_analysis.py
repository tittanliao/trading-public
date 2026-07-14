"""
Bollinger Band position analysis for XAUUSD trades.

Bollinger Bands (BB) measure price volatility and position relative to a
rolling mean.  This module:
  1. Computes BB columns on a price DataFrame  (compute_bb)
  2. Classifies %B into named zones            (bb_zone)
  3. Enriches a trades table with BB context   (enrich_trades_with_bb)
  4. Produces win-rate breakdowns by zone      (bb_stats)

Classic %B formula:
    %B = (close - lower) / (upper - lower)

%B = 0  → price is on the lower band
%B = 1  → price is on the upper band
%B = 0.5 → price is at the middle band (SMA)
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Compute Bollinger Band columns
# ---------------------------------------------------------------------------

def compute_bb(
    price: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Add Bollinger Band columns to *price* and return a copy.

    Parameters
    ----------
    price : DataFrame
        Must contain a ``close`` column.  The DataFrame should be sorted
        chronologically (ascending time) before calling this function.
    period : int
        Rolling window length for the SMA and standard deviation (default 20).
    std_mult : float
        Number of standard deviations for the bands (default 2.0).

    Returns
    -------
    DataFrame
        Original columns plus:
        - ``bb_mid``   : rolling SMA of close
        - ``bb_std``   : rolling population std of close (ddof=1)
        - ``bb_upper`` : bb_mid + std_mult × bb_std
        - ``bb_lower`` : bb_mid − std_mult × bb_std
        - ``bb_width`` : (bb_upper − bb_lower) / bb_mid  (normalised width)
        - ``bb_pct_b`` : (close − bb_lower) / (bb_upper − bb_lower)
    """
    df = price.copy()

    df["bb_mid"]   = df["close"].rolling(period, min_periods=period).mean()
    df["bb_std"]   = df["close"].rolling(period, min_periods=period).std(ddof=1)
    df["bb_upper"] = df["bb_mid"] + std_mult * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - std_mult * df["bb_std"]

    band_range = df["bb_upper"] - df["bb_lower"]

    # Normalised width — avoids division-by-zero by masking zero-range rows
    df["bb_width"] = np.where(
        df["bb_mid"].notna() & (df["bb_mid"] != 0),
        band_range / df["bb_mid"],
        np.nan,
    )

    # %B — NaN when band range is zero (flat / insufficient data)
    df["bb_pct_b"] = np.where(
        band_range != 0,
        (df["close"] - df["bb_lower"]) / band_range,
        np.nan,
    )
    # Keep NaN where bands themselves are NaN
    df.loc[df["bb_mid"].isna(), "bb_pct_b"] = np.nan

    return df


# ---------------------------------------------------------------------------
# 2. Classify %B into named zones
# ---------------------------------------------------------------------------

# Zone boundaries in ascending order (right-exclusive except the last closed bound)
_ZONE_THRESHOLDS: list[tuple[float, float, str]] = [
    (float("-inf"), 0.0,  "below_lower"),
    (0.0,           0.2,  "near_lower"),
    (0.2,           0.4,  "lower_mid"),
    (0.4,           0.6,  "near_middle"),
    (0.6,           0.8,  "upper_mid"),
    (0.8,           1.0,  "near_upper"),   # inclusive upper bound handled below
    (1.0,           float("inf"), "above_upper"),
]

# Canonical display order (used by bb_stats for consistent table ordering)
ZONE_ORDER: list[str] = [lo for *_, lo in (
    ("below_lower",), ("near_lower",), ("lower_mid",), ("near_middle",),
    ("upper_mid",), ("near_upper",), ("above_upper",)
)]
ZONE_ORDER = [
    "below_lower", "near_lower", "lower_mid", "near_middle",
    "upper_mid", "near_upper", "above_upper",
]


def bb_zone(pct_b: float | None) -> str:
    """
    Classify a %B value into a named Bollinger Band zone.

    Parameters
    ----------
    pct_b : float or None / NaN
        The %B value to classify.

    Returns
    -------
    str
        One of: ``"below_lower"``, ``"near_lower"``, ``"lower_mid"``,
        ``"near_middle"``, ``"upper_mid"``, ``"near_upper"``,
        ``"above_upper"``, or ``"unknown"`` (when pct_b is NaN/None).
    """
    if pct_b is None or (isinstance(pct_b, float) and np.isnan(pct_b)):
        return "unknown"

    # near_upper includes exactly 1.0 (price touches upper band)
    if 0.8 <= pct_b <= 1.0:
        return "near_upper"

    for lo, hi, label in _ZONE_THRESHOLDS:
        if lo <= pct_b < hi:
            return label

    # Fallback — should not be reached for finite float values
    return "unknown"


# ---------------------------------------------------------------------------
# 3. Enrich trades table with BB context at entry
# ---------------------------------------------------------------------------

def enrich_trades_with_bb(
    trades: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add Bollinger Band context columns to each trade row.

    The function finds the nearest *prior* price bar for each trade's
    ``entry_time`` using ``pd.merge_asof`` (O(n log n)) and attaches:
      - ``bb_pct_b``  — %B at entry
      - ``bb_zone``   — named zone (see :func:`bb_zone`)
      - ``bb_mid``    — middle band value
      - ``bb_upper``  — upper band value
      - ``bb_lower``  — lower band value
      - ``bb_width``  — normalised band width

    Parameters
    ----------
    trades : DataFrame
        Must contain ``entry_time`` (datetime-like), ``trade_id``,
        ``result``, and ``net_pnl_usd`` columns.
    price : DataFrame
        Price data.  Must contain a ``time`` column (datetime-like) and a
        ``close`` column.  BB columns will be computed via
        :func:`compute_bb` if not already present; existing BB columns are
        reused as-is.

    Returns
    -------
    DataFrame
        Copy of *trades* with the six BB columns appended.
    """
    # Compute BB columns if not already present
    bb_cols = {"bb_pct_b", "bb_mid", "bb_upper", "bb_lower", "bb_width"}
    if not bb_cols.issubset(price.columns):
        price = compute_bb(price)

    # Prepare price lookup table — must be sorted by time for merge_asof
    lookup = (
        price[["time", "bb_pct_b", "bb_mid", "bb_upper", "bb_lower", "bb_width"]]
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Ensure datetime types are compatible
    lookup["time"] = pd.to_datetime(lookup["time"])

    trades_out = trades.copy()
    trades_out["entry_time"] = pd.to_datetime(trades_out["entry_time"])

    trades_sorted = trades_out.sort_values("entry_time").reset_index()
    original_index_col = "index"  # preserves original index for re-sorting later

    merged = pd.merge_asof(
        trades_sorted,
        lookup,
        left_on="entry_time",
        right_on="time",
        direction="backward",
    )

    # Restore original row order
    merged = merged.sort_values(original_index_col).set_index(original_index_col)
    merged.index.name = None

    # Classify %B into zone strings
    merged["bb_zone"] = merged["bb_pct_b"].apply(bb_zone)

    # Drop the auxiliary 'time' column added by merge_asof (from price side)
    if "time_y" in merged.columns:
        merged = merged.drop(columns=["time_y"])
    if "time_x" in merged.columns:
        merged = merged.rename(columns={"time_x": "time"})

    return merged


# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------

def bb_stats(enriched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute win-rate breakdowns by Bollinger Band zone and band-width quartile.

    Parameters
    ----------
    enriched : DataFrame
        Output of :func:`enrich_trades_with_bb`.  Must contain
        ``bb_zone``, ``bb_pct_b``, ``bb_width``, ``result``, and
        ``net_pnl_usd`` columns.

    Returns
    -------
    dict with two keys:

    ``"by_zone"`` : DataFrame
        Indexed by ``bb_zone`` in logical order.  Columns:
        - ``total``       — number of trades in this zone
        - ``wins``        — number of winning trades
        - ``win_rate``    — wins / total (0–1, rounded to 3 dp)
        - ``avg_pnl``     — mean net_pnl_usd (rounded to 2 dp)
        - ``avg_pct_b``   — mean %B within the zone (rounded to 3 dp)

    ``"bb_width_quantiles"`` : DataFrame
        Win rate by BB-width quartile.  Columns:
        - ``quartile``    — Q1 (tight) … Q4 (wide / volatile)
        - ``width_min``   — minimum bb_width in quartile
        - ``width_max``   — maximum bb_width in quartile
        - ``total``       — number of trades
        - ``wins``        — winning trades
        - ``win_rate``    — rounded to 3 dp
        - ``avg_pnl``     — rounded to 2 dp
    """
    # ---- 4a. By zone ---------------------------------------------------------
    valid = enriched[enriched["bb_zone"] != "unknown"].copy()

    zone_rows: list[dict] = []
    for zone in ZONE_ORDER:
        grp = valid[valid["bb_zone"] == zone]
        total = len(grp)
        if total == 0:
            zone_rows.append({
                "bb_zone": zone,
                "total": 0, "wins": 0,
                "win_rate": np.nan, "avg_pnl": np.nan, "avg_pct_b": np.nan,
            })
            continue
        wins = int((grp["result"] == "win").sum())
        zone_rows.append({
            "bb_zone":   zone,
            "total":     total,
            "wins":      wins,
            "win_rate":  round(wins / total, 3),
            "avg_pnl":   round(grp["net_pnl_usd"].mean(), 2),
            "avg_pct_b": round(grp["bb_pct_b"].mean(), 3),
        })

    by_zone = pd.DataFrame(zone_rows).set_index("bb_zone")

    # ---- 4b. By BB-width quartile -------------------------------------------
    width_valid = enriched.dropna(subset=["bb_width"]).copy()

    if len(width_valid) >= 4:
        try:
            width_valid["_q"] = pd.qcut(
                width_valid["bb_width"],
                q=4,
                labels=["Q1 (tight)", "Q2", "Q3", "Q4 (wide)"],
                duplicates="drop",
            )
        except ValueError:
            # duplicates="drop" collapsed below 4 edges (low-coverage sample) —
            # fall back to auto interval labels instead of crashing.
            width_valid["_q"] = pd.qcut(width_valid["bb_width"], q=4, duplicates="drop")
        quartile_rows: list[dict] = []
        for label, grp in width_valid.groupby("_q", observed=True):
            total = len(grp)
            wins  = int((grp["result"] == "win").sum())
            quartile_rows.append({
                "quartile":  str(label),
                "width_min": round(grp["bb_width"].min(), 4),
                "width_max": round(grp["bb_width"].max(), 4),
                "total":     total,
                "wins":      wins,
                "win_rate":  round(wins / total, 3) if total else np.nan,
                "avg_pnl":   round(grp["net_pnl_usd"].mean(), 2),
            })
        bb_width_quantiles = pd.DataFrame(quartile_rows).set_index("quartile")
    else:
        # Not enough data for quartile split
        bb_width_quantiles = pd.DataFrame(
            columns=["width_min", "width_max", "total", "wins", "win_rate", "avg_pnl"]
        )
        bb_width_quantiles.index.name = "quartile"

    return {
        "by_zone":            by_zone,
        "bb_width_quantiles": bb_width_quantiles,
    }
