#!/usr/bin/env python3
"""RS-MULTI-20261018-001 — the US presidential four-year cycle on the S&P 500, TAIEX, 0050 and TSMC.

Design fixed in research/studies/RS-MULTI-20261018-001/decision_log.md before any data was fetched. Tests the
commonly recalled timing (buy at the start of the post-election year, sell after the midterm), the commonly cited
midterm-to-pre-election rally, and the cost of being out from March of the pre-election year. Descriptive research.

    python3.12 scripts/research/build_presidential_cycle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-MULTI-20261018-001"
DATA = Path("local-inputs")
TAIPEI = timezone(timedelta(hours=8))
SERIES = {
    "SP500": {"file": "GSPC_1d.csv", "label": "S&P 500 (price)", "start": None},
    "TAIEX": {"file": "TWII_1d.csv", "label": "TAIEX (price)", "start": None},
    "0050": {"file": "0050_TW_1d.csv", "label": "0050 (adjusted, from 2014)", "start": "2014-01-02"},
    "2330": {"file": "2330_TW_1d.csv", "label": "TSMC 2330 (adjusted)", "start": None},
}
ELECTIONS = range(1952, 2025, 4)


def load(name: str) -> pd.Series:
    spec = SERIES[name]
    frame = pd.read_csv(DATA / spec["file"], index_col=0, parse_dates=True)
    close = frame["close"].dropna()
    return close[close.index >= spec["start"]] if spec["start"] else close


def vote_day(year: int) -> pd.Timestamp:
    first = date(year, 11, 1)
    monday = first + timedelta(days=(0 - first.weekday()) % 7)
    return pd.Timestamp(monday + timedelta(days=1))


def after(close: pd.Series, day: pd.Timestamp):
    i = close.index.searchsorted(day, side="right")
    return close.index[i] if i < len(close) else None


def on_or_after(close: pd.Series, day: pd.Timestamp):
    i = close.index.searchsorted(day, side="left")
    return close.index[i] if i < len(close) else None


def year_first(close: pd.Series, year: int):
    block = close[(close.index.year == year)]
    return block.index[0] if len(block) else None


def year_last(close: pd.Series, year: int):
    block = close[(close.index.year == year)]
    return block.index[-1] if len(block) else None


def ret(close: pd.Series, a, b):
    if a is None or b is None or b <= a or a < close.index[0]:
        return None
    if a == close.index[0] and close.index[0] > pd.Timestamp(a.year, 1, 10):
        return None
    return round(float(close[b] / close[a] - 1) * 100, 3)


def windows(close: pd.Series, e: int) -> dict:
    y1, y2, y3, y4 = e + 1, e + 2, e + 3, e + 4
    start_ok = lambda d: d is not None and d >= close.index[0] and not (  # noqa: E731
        d == close.index[0] and close.index[0] > pd.Timestamp(d.year, 1, 10))
    out = {}
    for k, y in enumerate((y1, y2, y3, y4), 1):
        a, b = year_last(close, y - 1), year_last(close, y)
        full = a is not None and b is not None and b.year == y and b >= pd.Timestamp(y, 12, 20)
        out[f"Y{k}"] = ret(close, a, b) if full else None
    mid, elect = vote_day(y2), vote_day(y4)
    a = year_first(close, y1)
    out["recalled_timing"] = ret(close, a, after(close, mid)) if start_ok(a) else None
    m = after(close, mid)
    end3 = year_last(close, y3)
    out["midterm_to_y3_end"] = ret(close, m, end3) if end3 is not None and end3 >= pd.Timestamp(y3, 12, 20) else None
    out["midterm_to_election"] = ret(close, m, after(close, elect))
    out_day = on_or_after(close, pd.Timestamp(y3, 3, 19))
    out["out_from_mar_y3_to_y4_start"] = ret(close, out_day, year_first(close, y4))
    out["out_from_mar_y3_to_election"] = ret(close, out_day, after(close, elect))
    out["out_from_mar_y3_to_next_y1"] = ret(close, out_day, year_first(close, y4 + 1))
    return out


def describe(values: list[float]) -> dict | None:
    s = pd.Series([v for v in values if v is not None], dtype=float)
    if s.empty:
        return None
    sd = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    return {
        "n": int(len(s)),
        "win_rate_pct": round(float((s > 0).mean() * 100), 2),
        "mean_pct": round(float(s.mean()), 3),
        "median_pct": round(float(s.median()), 3),
        "worst_pct": round(float(s.min()), 3),
        "best_pct": round(float(s.max()), 3),
        "mean_resolvable_vs_zero": round(2.8 * sd / math.sqrt(len(s)), 3) if len(s) > 1 else None,
    }


def placebo(close: pd.Series, named_offset: int, months: int) -> dict:
    """Same holding length started at the first close of every month of the 48-month cycle."""
    by_offset, pools = {}, {}
    for offset in range(48):
        values = []
        for e in ELECTIONS:
            start_month = pd.Timestamp(e + 1, 1, 1) + pd.DateOffset(months=offset)
            a = on_or_after(close, start_month)
            if a is None or a.month != start_month.month or start_month < close.index[0]:
                continue
            b = on_or_after(close, start_month + pd.DateOffset(months=months))
            values.append(ret(close, a, b))
        by_offset[offset] = describe(values)
        pools[offset] = [v for v in values if v is not None]
    valid = {k: v for k, v in by_offset.items() if v}
    named = valid.get(named_offset)
    others = [v["mean_pct"] for k, v in valid.items() if k != named_offset]
    rank = sorted(valid, key=lambda k: valid[k]["mean_pct"], reverse=True).index(named_offset) + 1 if named else None
    gap = bound = None
    if named:
        mine = np.array(pools[named_offset])
        rest = np.array([v for k, vals in pools.items() if k != named_offset for v in vals])
        gap = round(float(mine.mean() - rest.mean()), 3)
        # Overlapping windows: the pooled others are not independent, so this bound is optimistic.
        bound = round(float(2.8 * math.sqrt(mine.var(ddof=1) / len(mine) + rest.var(ddof=1) / len(rest))), 3)
    return {
        "months_held": months, "named_offset": named_offset,
        "named_minus_others_mean_pct_points": gap, "resolvable_pct_points": bound,
        "named": named, "rank_by_mean_of": [rank, len(valid)],
        "other_offsets_mean_of_means": round(float(np.mean(others)), 3) if others else None,
        "by_offset_mean": {str(k): v["mean_pct"] for k, v in valid.items()},
        "by_offset_win": {str(k): v["win_rate_pct"] for k, v in valid.items()},
    }


def analyse(name: str) -> dict:
    close = load(name)
    cycles = {e: windows(close, e) for e in ELECTIONS}
    keys = next(iter(cycles.values())).keys()
    summary = {k: describe([c[k] for c in cycles.values()]) for k in keys}
    return {
        "label": SERIES[name]["label"],
        "coverage": [str(close.index[0].date()), str(close.index[-1].date())],
        "summary": summary,
        "placebo_recalled_timing": placebo(close, 0, 22),
        "placebo_midterm_to_y3_end": placebo(close, 22, 14),
        "current_cycle_2024": cycles[2024],
        "cycles": {str(e): c for e, c in cycles.items() if any(v is not None for v in c.values())},
    }


def sha(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "none — calendar windows of the US presidential cycle",
        "method": {
            "design": f"research/studies/{STUDY_ID}/decision_log.md",
            "cycle": "E = US presidential election year 1952-2024; Y1..Y4 = E+1..E+4",
            "vote_day": "first Tuesday after the first Monday of November; 'after' = first close strictly after",
            "recalled_timing": "first close of Y1 -> first close after the Y2 midterm",
            "midterm_windows": "first close after the Y2 midterm -> last close of Y3 / first close after the Y4 election",
            "out_from_march_y3": "first close on or after 19 March Y3 -> first close of Y4 / after the Y4 election / of the next Y1; positive = what being out cost",
            "placebo": "same holding length (22 and 14 months) from the first close of each of the 48 cycle months",
            "bound": "mean resolves only beyond 2.8 x SD / sqrt(n)",
            "cash": "0%; dividends excluded for the S&P 500 and TAIEX price indices",
        },
        "series": {name: analyse(name) for name in SERIES},
    }
    if args.dry_run:
        slim = json.loads(json.dumps(payload))
        for v in slim["series"].values():
            v.pop("cycles"); v["placebo_recalled_timing"].pop("by_offset_win"); v["placebo_midterm_to_y3_end"].pop("by_offset_win")
        print(json.dumps(slim, ensure_ascii=False, indent=1))
        return 0
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
        "timezone_assumptions": "Daily closes on each exchange's own calendar date.",
        "sources": [{"role": f"price_1d_{name}", "source_system": "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18",
                     "locator": f"data/stocks/market/yahoo/{spec['file']}", **sha(DATA / spec["file"]),
                     "immutable": False, "privacy": "public_market_data"} for name, spec in SERIES.items()],
    }
    (Path("reproduced") / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(Path("reproduced") / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
