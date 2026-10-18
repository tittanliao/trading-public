#!/usr/bin/env python3
"""RS-TX-20261018-002 — a 2x Taiwan 50 product held from the midterm year to the election, at full and half capital.

Design fixed in research/studies/RS-TX-20261018-002/decision_log.md before any outcome. Synthetic 2x (2 × daily
return − 2.0% a year) on TAIEX from 2001 and 0050 from 2014, validated against the real 00631L from 2016.

    python3.12 scripts/research/build_tw_lev2_cycle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import build_presidential_cycle as cyc  # noqa: E402
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-TX-20261018-002"
TAIPEI = timezone(timedelta(hours=8))
DRAG = 0.02                      # 1.0% fund fee + 1.0% financing, a year
COST = 0.001 + 2 * 0.001425      # ETF round trip
ELECTIONS = range(2000, 2025, 4)
LEV_FILE = cyc.DATA / "00631L_TW_1d.csv"


def series(name: str) -> tuple[pd.Series, pd.Series]:
    """(1x close, 2x close) for one base."""
    if name == "TAIEX":
        one = cyc.load("TAIEX")
        one = one[one.index >= "2001-01-01"]
    elif name == "0050":
        one = cyc.load("0050")
    else:                        # real 00631L against 0050 as its 1x
        two = pd.read_csv(LEV_FILE, index_col=0, parse_dates=True)["close"]
        two = two[two.index >= "2016-01-04"]
        one = cyc.load("0050").reindex(two.index).dropna()
        return one, two.reindex(one.index)
    daily = 2 * one.pct_change().fillna(0.0) - DRAG / 252
    daily.iloc[0] = 0.0
    return one, one.iloc[0] * (1 + daily).cumprod()


def windows(close: pd.Series, kind: str) -> list[tuple]:
    first, last = close.index[0], close.index[-1]
    spans = []
    for e in ELECTIONS:
        y2, y4 = e + 2, e + 4
        if kind == "y2_start_to_election":
            a = cyc.year_first(close, y2)
            if a is None or a < first:      # R3 review 2026-10-18: was Timestamp(y2,1,1) < first, which dropped 2014
                continue
        else:
            day = cyc.vote_day(y2)
            if day < first:
                continue
            a = cyc.after(close, day)
            if a is None:
                continue
        b = cyc.after(close, cyc.vote_day(y4)) if cyc.vote_day(y4) >= first else None
        spans.append((a, b if b is not None else last, b is not None))
    return spans


def curve(price: pd.Series, spans: list[tuple], weight: float) -> pd.Series:
    r = price.pct_change().fillna(0.0)
    w = pd.Series(0.0, index=price.index)
    for a, b, _ in spans:
        w[(price.index > a) & (price.index <= b)] = weight
    eq = (1 + r * w).cumprod()
    for a, b, done in spans:
        eq[eq.index >= a] *= (1 - weight * COST / 2)
        if done:
            eq[eq.index >= b] *= (1 - weight * COST / 2)
    return eq


def stats(eq: pd.Series, spans: list[tuple], index: pd.DatetimeIndex) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    dd = float((eq / eq.cummax() - 1).min())
    inside = [float((eq[a:b] / eq[a:b].cummax() - 1).min()) for a, b, _ in spans] if spans else [dd]
    held = sum(((index > a) & (index <= b)).sum() for a, b, _ in spans) if spans else len(index) - 1
    cagr = (1 + total) ** (1 / years) - 1
    return {"total_return_pct": round(total * 100, 2), "cagr_pct": round(cagr * 100, 2),
            "max_drawdown_pct": round(dd * 100, 2), "worst_in_window_drawdown_pct": round(min(inside) * 100, 2),
            "time_invested_pct": round(float(held / (len(index) - 1) * 100), 1),
            "cagr_over_max_dd": round(cagr * 100 / -dd / 100, 3) if dd < 0 else None}


def analyse(name: str) -> dict:
    one, two = series(name)
    out = {"coverage": [str(one.index[0].date()), str(one.index[-1].date())], "windows": {}}
    always = [(one.index[0], one.index[-1], False)]
    out["buy_hold_1x"] = stats(curve(one, always, 1.0), [], one.index)
    out["buy_hold_2x"] = stats(curve(two, always, 1.0), [], one.index)
    for kind in ("y2_start_to_election", "midterm_to_election"):
        spans = windows(one, kind)
        block = {"strategies": {}, "cycles": {}}
        curves = {"lev2_full": curve(two, spans, 1.0), "lev2_half": curve(two, spans, 0.5),
                  "etf1_full": curve(one, spans, 1.0)}
        for s, eq in curves.items():
            block["strategies"][s] = stats(eq, spans, one.index)
        for a, b, done in spans:
            row = {"window": [str(a.date()), str(b.date())], "complete": done}
            for s, eq in curves.items():
                row[s] = round(float(eq[b] / eq[a] - 1) * 100, 2)
                row[s + "_dd"] = round(float((eq[a:b] / eq[a:b].cummax() - 1).min()) * 100, 2)
            block["cycles"][str(a.year - 2 + (0 if kind == "y2_start_to_election" else 0))] = row
        out["windows"][kind] = block
    return out


def validation() -> dict:
    real = pd.read_csv(LEV_FILE, index_col=0, parse_dates=True)["close"]
    real = real[real.index >= "2016-01-04"]
    base = cyc.load("0050").reindex(real.index).dropna()
    real = real.reindex(base.index)
    synth = (1 + (2 * base.pct_change().fillna(0.0) - DRAG / 252)).cumprod()
    years = (real.index[-1] - real.index[0]).days / 365.25
    cagr = lambda s: ((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100  # noqa: E731
    return {"period": [str(real.index[0].date()), str(real.index[-1].date())],
            "real_00631L_cagr_pct": round(float(cagr(real)), 2), "synthetic_cagr_pct": round(float(cagr(synth)), 2),
            "daily_return_correlation": round(float(real.pct_change().corr(synth.pct_change())), 4),
            "real_max_dd_pct": round(float((real / real.cummax() - 1).min() * 100), 2),
            "synthetic_max_dd_pct": round(float((synth / synth.cummax() - 1).min() * 100), 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "2x Taiwan 50 from the midterm year to the election (decision log 2026-10-18)",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md", "synthetic_2x": "2 x daily return - 2.0%/252",
                   "round_trip_cost": COST, "cash": "0%", "half": "50% in 2x, 50% cash, inside the window"},
        "validation_vs_00631L": validation(),
        "series": {n: analyse(n) for n in ("TAIEX", "0050", "00631L")},
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    srcs = []
    for n, f in (("TAIEX", "TWII_1d.csv"), ("0050", "0050_TW_1d.csv"), ("00631L", "00631L_TW_1d.csv")):
        data = (cyc.DATA / f).read_bytes()
        srcs.append({"role": f"price_1d_{n}", "source_system": "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18",
                     "locator": f"data/stocks/market/yahoo/{f}", "sha256": hashlib.sha256(data).hexdigest(),
                     "size_bytes": len(data), "immutable": False, "privacy": "public_market_data"})
    (Path("reproduced") / "source_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
         "timezone_assumptions": "Daily closes on the Taiwan exchange calendar.", "sources": srcs},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(Path("reproduced") / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
