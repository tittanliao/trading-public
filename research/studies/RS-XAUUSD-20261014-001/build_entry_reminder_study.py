"""Loaders published for a sibling study; the rest of this module stays Private."""
from __future__ import annotations

import hashlib

import io

import json

import math

import re

import sys

from datetime import datetime

from pathlib import Path

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

import fail_pattern_toolkit as tk  # noqa: E402

EXPORTS = Path("local-inputs")

INPUTS = {
    "s1_trades": EXPORTS / "s1-awewithbb/v3.9/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-10-09.csv",
    "s1_trades_earlier": EXPORTS / "s1-awewithbb/v3.9/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-08-15.csv",
    "s2_trades": EXPORTS / "s2-hammer/v3.2/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-10-09.csv",
    "s2_trades_earlier": EXPORTS / "s2-hammer/v3.2/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-08-15.csv",
    "price_30m_history": Path("local-inputs/FX_IDC_XAUUSD, 30.csv"),
    "price_30m_recent": Path("local-inputs/FX_IDC_XAUUSD, 30.csv"),
    "price_1d": Path("local-inputs/FX_IDC_XAUUSD, 1D.csv"),
    "gvz_1d": Path("local-inputs/GVZ_1d.csv"),
}

STRATEGIES = {
    "S1": {"stop_pct": 0.5, "window": 36, "stop_signal": "S1BB_SL"},
    "S2": {"stop_pct": 1.0, "window": 48, "stop_signal": "S2_SL"},
}

HALF = pd.Timestamp("2025-07-01")

def _read_export(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw[raw["Date and time"] != "Open"]
    return tk.load_trades(io.StringIO(raw.to_csv(index=False)))

def load_trades(strategy: str) -> tuple[pd.DataFrame, dict]:
    new = _read_export(INPUTS[f"{strategy.lower()}_trades"])
    old = _read_export(INPUTS[f"{strategy.lower()}_trades_earlier"])
    first = new["entry_time"].min()
    overlap_end = old["entry_time"].max()
    a = set(old.loc[old["entry_time"] >= first, "entry_time"])
    b = set(new.loc[new["entry_time"] <= overlap_end, "entry_time"])
    trades = pd.concat([old[old["entry_time"] < first], new], ignore_index=True)
    trades["strategy"] = strategy
    trades["r"] = trades["return_pct"] / STRATEGIES[strategy]["stop_pct"]
    trades["half"] = np.where(trades["entry_time"] < HALF, "first", "second")
    check = {"earlier_export_entries_used": int((old["entry_time"] < first).sum()),
             "overlap_shared": len(a & b), "overlap_only_earlier": len(a - b), "overlap_only_newer": len(b - a)}
    return trades.sort_values("entry_time").reset_index(drop=True), check

