"""
SMC strategy signal functions — XAUUSD 30-min bars.

Usage:
    smc = smc_indicators.precompute(price)
    STRATEGIES_LONG, STRATEGIES_SHORT = make_strategies(smc)

All signal functions have the engine-compatible signature:
    fn(price: pd.DataFrame, i: int) -> bool

Strategy groups:
    M01–M03  FVG-based long entries
    M04–M05  Order Block long entries
    M06–M08  Liquidity sweep long entries
    M09–M10  Confluence long entries
    M11–M13  FVG-based short entries
    M14–M15  Order Block short entries
    M16–M18  Liquidity sweep short entries
    M19–M20  Confluence short entries
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_strategies(smc: dict) -> tuple[dict, dict]:
    """
    Build strategy registries closed over pre-computed SMC arrays.

    Returns
    -------
    (STRATEGIES_LONG, STRATEGIES_SHORT)
    Each dict maps strategy_id → (signal_fn, group, description)
    """
    in_bull  = smc["in_bull_fvg"]
    in_bear  = smc["in_bear_fvg"]
    bull_ob  = smc["at_bull_ob"]
    bear_ob  = smc["at_bear_ob"]
    ssl      = smc["ssl_swept"]
    bsl      = smc["bsl_swept"]
    rsi      = smc["rsi14"]
    ema21    = smc["ema21"]
    ema50    = smc["ema50"]

    # ══════════════════════════════════════════════════════════════════════════
    # LONG strategies (M01–M10)
    # ══════════════════════════════════════════════════════════════════════════

    def M01_FVG_Long(price, i):
        """Price retraces into a bullish FVG zone — pure imbalance fill."""
        return bool(in_bull[i])

    def M02_FVG_Long_RSI(price, i):
        """Bullish FVG + RSI 30–65 (not oversold panic, not overbought)."""
        return bool(in_bull[i] and 30 < rsi[i] < 65)

    def M03_FVG_Long_Trend(price, i):
        """Bullish FVG + price above EMA50 (retrace within uptrend)."""
        c = price["close"].to_numpy()
        return bool(in_bull[i] and c[i] > ema50[i])

    def M04_OB_Long(price, i):
        """Price retraces to a confirmed bullish Order Block — pure OB hold."""
        return bool(bull_ob[i])

    def M05_OB_Long_RSI(price, i):
        """Bullish OB + RSI < 55 (not overbought entering OB zone)."""
        return bool(bull_ob[i] and rsi[i] < 55)

    def M06_SSL_Sweep_Long(price, i):
        """Sell-side liquidity sweep reversal: swept lows, closed back above."""
        return bool(ssl[i])

    def M07_SSL_FVG_Long(price, i):
        """SSL sweep coincides with bullish FVG — sweep + imbalance confluence."""
        return bool(ssl[i] and in_bull[i])

    def M08_SSL_OB_Long(price, i):
        """SSL sweep at a bullish OB — sweep + institutional zone confluence."""
        return bool(ssl[i] and bull_ob[i])

    def M09_FVG_OB_Long(price, i):
        """FVG inside an OB — double confluence long (no sweep required)."""
        return bool(in_bull[i] and bull_ob[i])

    def M10_Triple_Long(price, i):
        """FVG + OB + SSL sweep — all three confluences aligned."""
        return bool(in_bull[i] and bull_ob[i] and ssl[i])

    # ══════════════════════════════════════════════════════════════════════════
    # SHORT strategies (M11–M20)
    # ══════════════════════════════════════════════════════════════════════════

    def M11_FVG_Short(price, i):
        """Price retraces into a bearish FVG zone — pure imbalance fill short."""
        return bool(in_bear[i])

    def M12_FVG_Short_RSI(price, i):
        """Bearish FVG + RSI 35–70 (not oversold panic, not extreme overbought)."""
        return bool(in_bear[i] and 35 < rsi[i] < 70)

    def M13_FVG_Short_Trend(price, i):
        """Bearish FVG + price below EMA50 (rally into FVG within downtrend)."""
        c = price["close"].to_numpy()
        return bool(in_bear[i] and c[i] < ema50[i])

    def M14_OB_Short(price, i):
        """Price retraces to a confirmed bearish Order Block — pure OB rejection."""
        return bool(bear_ob[i])

    def M15_OB_Short_RSI(price, i):
        """Bearish OB + RSI > 50 (momentum not oversold entering OB)."""
        return bool(bear_ob[i] and rsi[i] > 50)

    def M16_BSL_Sweep_Short(price, i):
        """Buy-side liquidity sweep reversal: swept highs, closed back below."""
        return bool(bsl[i])

    def M17_BSL_FVG_Short(price, i):
        """BSL sweep coincides with bearish FVG — sweep + imbalance confluence."""
        return bool(bsl[i] and in_bear[i])

    def M18_BSL_OB_Short(price, i):
        """BSL sweep at a bearish OB — sweep + institutional zone confluence."""
        return bool(bsl[i] and bear_ob[i])

    def M19_FVG_OB_Short(price, i):
        """Bearish FVG inside a bearish OB — double confluence short."""
        return bool(in_bear[i] and bear_ob[i])

    def M20_Triple_Short(price, i):
        """Bearish FVG + OB + BSL sweep — all three confluences aligned."""
        return bool(in_bear[i] and bear_ob[i] and bsl[i])

    # ── Registries ────────────────────────────────────────────────────────────
    # Format: strategy_id → (signal_fn, group_label, description)

    STRATEGIES_LONG = {
        "M01": (M01_FVG_Long,      "FVG",    "Bullish FVG fill"),
        "M02": (M02_FVG_Long_RSI,  "FVG",    "Bullish FVG + RSI 30-65"),
        "M03": (M03_FVG_Long_Trend,"FVG",    "Bullish FVG + above EMA50"),
        "M04": (M04_OB_Long,       "OB",     "Bullish OB hold"),
        "M05": (M05_OB_Long_RSI,   "OB",     "Bullish OB + RSI < 55"),
        "M06": (M06_SSL_Sweep_Long,"Sweep",  "SSL sweep reversal"),
        "M07": (M07_SSL_FVG_Long,  "Sweep",  "SSL sweep + FVG"),
        "M08": (M08_SSL_OB_Long,   "Sweep",  "SSL sweep + OB"),
        "M09": (M09_FVG_OB_Long,   "Conf",   "FVG + OB confluence"),
        "M10": (M10_Triple_Long,   "Conf",   "FVG + OB + SSL sweep (triple)"),
    }

    STRATEGIES_SHORT = {
        "M11": (M11_FVG_Short,       "FVG",   "Bearish FVG fill"),
        "M12": (M12_FVG_Short_RSI,   "FVG",   "Bearish FVG + RSI 35-70"),
        "M13": (M13_FVG_Short_Trend, "FVG",   "Bearish FVG + below EMA50"),
        "M14": (M14_OB_Short,        "OB",    "Bearish OB rejection"),
        "M15": (M15_OB_Short_RSI,    "OB",    "Bearish OB + RSI > 50"),
        "M16": (M16_BSL_Sweep_Short, "Sweep", "BSL sweep reversal"),
        "M17": (M17_BSL_FVG_Short,   "Sweep", "BSL sweep + FVG"),
        "M18": (M18_BSL_OB_Short,    "Sweep", "BSL sweep + OB"),
        "M19": (M19_FVG_OB_Short,    "Conf",  "Bearish FVG + OB confluence"),
        "M20": (M20_Triple_Short,    "Conf",  "Bearish FVG + OB + BSL sweep (triple)"),
    }

    return STRATEGIES_LONG, STRATEGIES_SHORT
