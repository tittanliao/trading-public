#!/usr/bin/env python3
"""RS-TX-20261018-001 — presidential-cycle in/out strategies on TAIEX, TSMC and 0050 against buy-and-hold.

Design fixed in research/studies/RS-TX-20261018-001/decision_log.md before any outcome. Six fixed strategies, no
search. Cycle calendar and data loading are shared with RS-MULTI-20261018-001's runner. Descriptive research only.

    python3.12 scripts/research/build_tw_presidential_strategy.py
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

STUDY_ID = "RS-TX-20261018-001"
TAIPEI = timezone(timedelta(hours=8))
COST = {"TAIEX": 0.003 + 2 * 0.001425, "2330": 0.003 + 2 * 0.001425, "0050": 0.001 + 2 * 0.001425}
START = {"TAIEX": "2001-01-01", "2330": "2001-01-01", "0050": "2014-01-02"}
ELECTIONS = range(1996, 2025, 4)
STRATEGIES = ("buy_hold", "recalled_timing", "midterm_to_y3_end", "midterm_to_election", "skip_midterm_year",
              "out_march_y3")


def in_intervals(close: pd.Series, name: str) -> list[tuple]:
    """(entry, exit) closes; the position is held from the entry close to the exit close."""
    first, last = close.index[0], close.index[-1]
    if name == "buy_hold":
        return [(first, last)]
    spans, outs = [], []

    def after(day):          # dates before the series start are not trades, not its first close
        return cyc.after(close, day) if day >= first else None

    def on_or_after(day):
        return cyc.on_or_after(close, day) if day >= first else None

    for e in ELECTIONS:
        y1, y2, y3, y4 = e + 1, e + 2, e + 3, e + 4
        if y4 < first.year:          # a cycle wholly before the data
            continue
        mid = after(cyc.vote_day(y2))
        if name == "recalled_timing":
            spans.append((pd.Timestamp(y1, 1, 1), mid))
        elif name == "midterm_to_y3_end":
            spans.append((mid, cyc.year_last(close, y3)))
        elif name == "midterm_to_election":
            spans.append((mid, after(cyc.vote_day(y4))))
        elif name == "skip_midterm_year":
            outs.append((cyc.year_last(close, y1), mid))
        elif name == "out_march_y3":
            outs.append((on_or_after(pd.Timestamp(y3, 3, 19)), cyc.year_first(close, y4)))
    if outs:
        cursor = first
        for a, b in outs:
            if a is None:                 # this out-window starts before the data or after it
                if b is not None and b > cursor:
                    cursor = b if b < last else cursor
                continue
            if a <= cursor:
                cursor = max(cursor, b) if b is not None else last
                continue
            spans.append((cursor, min(a, last)))
            cursor = b if b is not None else last
            if cursor >= last:
                break
        if cursor < last:
            spans.append((cursor, last))
    clipped = []
    for a, b in spans:
        if a is None:
            continue
        b = last if b is None else min(b, last)
        a = cyc.on_or_after(close, max(a, first))
        if a is not None and b > a:
            clipped.append((a, b))
    return clipped


def equity(close: pd.Series, spans: list[tuple], cost: float) -> pd.Series:
    returns = close.pct_change().fillna(0.0)
    position = pd.Series(0.0, index=close.index)
    for a, b in spans:
        position[(close.index > a) & (close.index <= b)] = 1.0
    curve = (1 + returns * position).cumprod()
    for _, b in spans:
        curve[curve.index >= b] *= (1 - cost)
    return curve


def stats(curve: pd.Series, spans: list[tuple], close: pd.Series) -> dict:
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = float(curve.iloc[-1] / curve.iloc[0] - 1)
    drawdown = float((curve / curve.cummax() - 1).min())
    held = sum(((close.index > a) & (close.index <= b)).sum() for a, b in spans)
    return {
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2),
        "max_drawdown_pct": round(drawdown * 100, 2),
        "time_invested_pct": round(held / (len(close) - 1) * 100, 1),
        "round_trips": len(spans),
    }


def per_cycle(close: pd.Series, curves: dict) -> dict:
    out = {}
    for e in ELECTIONS:
        a = cyc.year_first(close, e + 1)
        b = cyc.year_first(close, e + 5) or close.index[-1]
        if a is None or a < close.index[0] or b <= a:
            continue
        if e + 1 < close.index[0].year or (a == close.index[0] and a.year < 2014 and False):
            continue
        row = {"window": [str(a.date()), str(b.date())], "complete": cyc.year_first(close, e + 5) is not None}
        for name, curve in curves.items():
            row[name] = round(float(curve[b] / curve[a] - 1) * 100, 2)
        out[str(e)] = row
    return out


def compare(cycles: dict, name: str) -> dict:
    done = [c for c in cycles.values() if c["complete"]]
    diff = np.array([c[name] - c["buy_hold"] for c in done])
    if len(diff) == 0:
        return {}
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    return {
        "complete_cycles": len(diff),
        "beat_buy_hold": int((diff > 0).sum()),
        "mean_excess_pct_points": round(float(diff.mean()), 2),
        "resolvable_pct_points": round(2.8 * sd / math.sqrt(len(diff)), 2) if len(diff) > 1 else None,
    }


def analyse(name: str) -> dict:
    close = cyc.load(name if name != "TAIEX" else "TAIEX")
    close = close[close.index >= START[name]]
    curves, table, spans_all = {}, {}, {}
    for strategy in STRATEGIES:
        spans = in_intervals(close, strategy)
        curve = equity(close, spans, COST[name])
        curves[strategy] = curve
        spans_all[strategy] = [[str(a.date()), str(b.date())] for a, b in spans]
        table[strategy] = stats(curve, spans, close)
    cycles = per_cycle(close, curves)
    return {
        "coverage": [str(close.index[0].date()), str(close.index[-1].date())],
        "round_trip_cost": COST[name],
        "strategies": table,
        "vs_buy_hold_by_cycle": {s: compare(cycles, s) for s in STRATEGIES if s != "buy_hold"},
        "cycles": cycles,
        "spans": spans_all,
        "_curves": curves,
    }


def chart(results: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 4.2 * len(results)))
    for ax, (name, res) in zip(axes, results.items()):
        for strategy, curve in res["_curves"].items():
            ax.plot(curve.index, curve / curve.iloc[0], linewidth=2.2 if strategy == "buy_hold" else 1.3,
                    label=f"{strategy} ({res['strategies'][strategy]['cagr_pct']}%/yr)")
        ax.set_yscale("log")
        ax.set_title(f"{name}: presidential-cycle in/out strategies vs buy-and-hold (cash 0%, costs included)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = {name: analyse(name) for name in ("TAIEX", "2330", "0050")}
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "six fixed presidential-cycle in/out rules (decision log 2026-10-18)",
        "method": {
            "design": f"research/studies/{STUDY_ID}/decision_log.md",
            "trades": "at the close; position held from the entry close to the exit close",
            "costs": "round trip: securities tax 0.3% (0.1% for 0050) + commission 0.1425% each side",
            "cash": "0%",
            "dividends": "TAIEX price index excludes dividends (about 3-4% a year); 2330 and 0050 adjusted",
            "per_cycle": "first close of Y1 to first close of the next Y1",
        },
        "series": {n: {k: v for k, v in r.items() if k != "_curves"} for n, r in results.items()},
        "charts": [{"id": "equity_curves", "file": "equity_curves.png", "section": "seasonality",
                    "title": "Presidential-cycle in/out strategies against buy-and-hold"}],
    }
    if args.dry_run:
        for n, r in payload["series"].items():
            print("==", n, r["coverage"])
            for s, v in r["strategies"].items():
                print(f"  {s:22s}", v, r["vs_buy_hold_by_cycle"].get(s, ""))
            for e, c in r["cycles"].items():
                print("   ", e, c)
        return 0
    directory = Path("reproduced")
    chart(results, directory / "charts/equity_curves.png")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    sources = []
    for name, spec in (("TAIEX", "TWII_1d.csv"), ("2330", "2330_TW_1d.csv"), ("0050", "0050_TW_1d.csv")):
        path = cyc.DATA / spec
        data = path.read_bytes()
        sources.append({"role": f"price_1d_{name}", "source_system": "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18",
                        "locator": f"data/stocks/market/yahoo/{spec}", "sha256": hashlib.sha256(data).hexdigest(),
                        "size_bytes": len(data), "immutable": False, "privacy": "public_market_data"})
    (directory / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
        "timezone_assumptions": "Daily closes on the Taiwan exchange calendar.", "sources": sources},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
