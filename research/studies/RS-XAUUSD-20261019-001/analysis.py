#!/usr/bin/env python3
"""RS-XAUUSD-20261019-001 — does gold dip and then rise about 100 dollars after a US midterm election?

Design fixed in research/studies/RS-XAUUSD-20261019-001/decision_log.md before the registered run. Weekly XAUUSD
closes from 1974; midterms and presidential elections against every week and against the same calendar position in
other years. Descriptive research only.

    python3.12 scripts/research/build_gold_election_path.py
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
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261019-001"
TAIPEI = timezone(timedelta(hours=8))
WEEKLY = Path("local-inputs/XAUUSD_1wk.csv")
DAILY = Path("local-inputs/XAUUSD_1d.csv")
FIRST_YEAR = 1974
MIDTERMS = list(range(1974, 2023, 4))
PRESIDENTIAL = list(range(1976, 2025, 4))
HORIZONS = (4, 8, 13, 26)
DIP_WEEKS = 8
RISE_WEEKS = 26
PRICE_NOW = 4168.17           # XAUUSD_1d close 2026-10-16, the last close before this design
S24 = round(100 / PRICE_NOW, 4)


def sha(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def load_weekly(path: Path = WEEKLY) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["time"])
    stamp = frame["time"].dt.tz_localize(None) + pd.Timedelta(days=4)     # Monday stamp, Friday close
    s = pd.Series(frame["close"].values, index=stamp)
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_daily(path: Path = DAILY) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["time"])
    s = pd.Series(frame["close"].values, index=frame["time"].dt.tz_localize(None))
    return s[~s.index.duplicated(keep="last")].sort_index()


def vote_day(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year, 11, 2)                    # the Tuesday after the first Monday: 2..8 November
    while d.weekday() != 1:
        d += pd.Timedelta(days=1)
    return d


def measure(values: np.ndarray, i: int, dip_n: int = DIP_WEEKS, rise_n: int = RISE_WEEKS,
            horizons: tuple = HORIZONS) -> dict | None:
    """Path measures from the close at position i. None if the series ends too soon."""
    if i + rise_n >= len(values):
        return None
    start = values[i]
    first = values[i + 1:i + 1 + dip_n]
    k = int(first.argmin())
    low = float(first[k])
    high = float(values[i + 1 + k:i + 1 + rise_n].max())
    dipped = low < start
    rise_pct = high / low - 1
    out = {f"r{h}_pct": round((values[i + h] / start - 1) * 100, 3) for h in horizons}
    out.update({
        "start": round(float(start), 3), "dip_pct": round((low / start - 1) * 100, 3), "dip_at": k + 1,
        "dipped": bool(dipped), "rise_from_low_usd": round(high - low, 2), "rise_from_low_pct": round(rise_pct * 100, 3),
        "s100": bool(dipped and high - low >= 100), "s10": bool(dipped and rise_pct >= 0.10),
        "s24": bool(dipped and rise_pct >= S24),
    })
    return out


def event(s: pd.Series, year: int) -> dict | None:
    v = vote_day(year)
    before = s[s.index < v]
    if before.empty:
        return None
    i = len(before) - 1
    m = measure(s.values, i)
    return None if m is None else {"t0": str(before.index[-1].date()), "vote_day": str(v.date()), **m}


def summary(rows: list[dict]) -> dict:
    r = pd.DataFrame(rows)
    out = {"n": len(r)}
    for h in HORIZONS:
        out[f"r{h}_mean_pct"] = round(float(r[f"r{h}_pct"].mean()), 3)
    out["r26_positive"] = int((r["r26_pct"] > 0).sum())
    out["dipped"] = int(r["dipped"].sum())
    out["dip_median_pct"] = round(float(r["dip_pct"].median()), 3)
    for key in ("s100", "s10", "s24"):
        out[f"{key}_rate"] = round(float(r[key].mean()), 4)
    return out


def against(group: dict, base_rows: list[dict], label: str) -> dict:
    """Group minus baseline, with the house bound computed from the baseline's spread and the group's n."""
    base = pd.DataFrame(base_rows)
    n = group["n"]
    out = {"baseline": label, "baseline_n": len(base)}
    for key in ("r26", "r13"):
        mean, sd = float(base[f"{key}_pct"].mean()), float(base[f"{key}_pct"].std(ddof=1))
        diff = group[f"{key}_mean_pct"] - mean
        bound = 2.8 * sd / math.sqrt(n)
        out[key] = {"baseline_mean_pct": round(mean, 3), "diff_pct_points": round(diff, 3),
                    "bound_pct_points": round(bound, 3), "resolved": bool(diff > bound)}
    for key in ("s10", "s24", "s100"):
        p = float(base[key].mean())
        diff = group[f"{key}_rate"] - p
        bound = 2.8 * math.sqrt(p * (1 - p) / n)
        out[key] = {"baseline_rate": round(p, 4), "diff": round(diff, 4), "bound": round(bound, 4),
                    "resolved": bool(diff > bound)}
    return out


def daily_check(d: pd.Series) -> dict:
    """The midterms the daily series covers, with trading-day horizons (8 weeks = 40 days, 26 weeks = 130)."""
    out = {}
    for y in MIDTERMS:
        v = vote_day(y)
        before = d[d.index < v]
        if before.empty or before.index[-1] < d.index[0] + pd.Timedelta(days=1):
            continue
        m = measure(d.values, len(before) - 1, dip_n=40, rise_n=130, horizons=(20, 40, 65, 130))
        if m is not None:
            out[str(y)] = {"t0": str(before.index[-1].date()), **m}
    return out


def chart(s: pd.Series, mid: dict, cal: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 7))
    values = s.values
    paths = []
    def path(row: dict) -> np.ndarray:
        i = int(np.searchsorted(s.index, pd.Timestamp(row["t0"])))
        return values[i:i + RISE_WEEKS + 1] / values[i] * 100 - 100

    for y, row in mid.items():
        paths.append(path(row))
        ax.plot(range(RISE_WEEKS + 1), paths[-1], linewidth=1.0, alpha=0.6, label=y)
    base = np.array([values[i:i + RISE_WEEKS + 1] / values[i] * 100 - 100
                     for i in range(int(np.searchsorted(s.index, pd.Timestamp(FIRST_YEAR, 1, 1))),
                                    len(values) - RISE_WEEKS - 1)])
    q = np.percentile(base, [25, 75], axis=0)
    ax.fill_between(range(RISE_WEEKS + 1), q[0], q[1], color="#999999", alpha=0.25, label="any week: middle half")
    ax.plot(range(RISE_WEEKS + 1), base.mean(axis=0), color="#555555", linewidth=2, label="any week: mean")
    ax.plot(range(RISE_WEEKS + 1), np.mean([path(r) for r in cal.values()], axis=0), color="#1f5fbf", linewidth=2.2,
            linestyle="--", label="same November week, non-election years: mean")
    ax.plot(range(RISE_WEEKS + 1), np.mean(paths, axis=0), color="#c0392b", linewidth=2.6, label="midterm mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(DIP_WEEKS, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("weeks after the last close before the midterm vote"); ax.set_ylabel("% from that close")
    ax.set_title("Gold after 13 US midterms (1974-2022) against every week since 1974")
    ax.legend(fontsize=7, ncol=4, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=100); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    s = load_weekly()
    values = s.values
    start_i = int(np.searchsorted(s.index, pd.Timestamp(FIRST_YEAR, 1, 1)))
    b_all = [m for i in range(start_i, len(values)) if (m := measure(values, i)) is not None]

    mid = {str(y): e for y in MIDTERMS if (e := event(s, y)) is not None}
    pres = {str(y): e for y in PRESIDENTIAL if (e := event(s, y)) is not None}
    other_years = [y for y in range(FIRST_YEAR, 2026) if y not in MIDTERMS and y not in PRESIDENTIAL]
    b_cal = {str(y): e for y in other_years if (e := event(s, y)) is not None}
    sens_1970 = event(s, 1970)

    mid_sum, pres_sum = summary(list(mid.values())), summary(list(pres.values()))
    halves = {"1974-1994": summary([v for k, v in mid.items() if int(k) <= 1994]),
              "1998-2022": summary([v for k, v in mid.items() if int(k) >= 1998])}
    comparisons = {
        "midterm_vs_all_weeks": against(mid_sum, b_all, "every weekly start 1974 onward"),
        "midterm_vs_same_calendar": against(mid_sum, list(b_cal.values()), "same November position, non-election years"),
        "presidential_vs_all_weeks": against(pres_sum, b_all, "every weekly start 1974 onward"),
        "presidential_vs_same_calendar": against(pres_sum, list(b_cal.values()),
                                                 "same November position, non-election years"),
    }
    supported = all(comparisons[c][k]["resolved"] for c in ("midterm_vs_all_weeks", "midterm_vs_same_calendar")
                    for k in ("s10", "r26"))
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "none — a market saying about gold after US midterm elections",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md",
                   "data": "XAUUSD weekly closes, Monday stamp moved to Friday; daily closes for the 2010-2022 check",
                   "t0": "last weekly close before the vote day (Tuesday after the first Monday of November)",
                   "dip": f"lowest close in weeks 1-{DIP_WEEKS}", "rise": f"highest close from that low through week {RISE_WEEKS}",
                   "saying": {"s100": "dipped, then +100 dollars from the low", "s10": "dipped, then +10% from the low",
                              "s24": f"dipped, then +{S24 * 100:.2f}% from the low (100 dollars at {PRICE_NOW})"},
                   "bounds": "means 2.8 x SD / sqrt(n); rates 2.8 x sqrt(p(1-p)/n); SD and p from the baseline",
                   "verdict_rule": "supported only if S10 and R26 beat both baselines by more than the bound"},
        "period": [str(s.index[start_i].date()), str(s.index[-1].date())],
        "midterm": {"summary": mid_sum, "events": mid, "halves": halves},
        "presidential": {"summary": pres_sum, "events": pres},
        "same_calendar_other_years": {"summary": summary(list(b_cal.values())), "events": b_cal},
        "all_weeks": summary(b_all),
        "comparisons": comparisons,
        "verdict": {"supported": supported,
                    "reading": "supported" if supported else "not distinguishable from ordinary weeks"},
        "sensitivity_1970": sens_1970,
        "daily_check": daily_check(load_daily()),
        "current_window": {"vote_day": str(vote_day(2026).date()), "vote_day_taipei": "2026-11-04",
                           "t0": "2026-10-30 close (not yet in data)",
                           "s24_threshold_pct": round(S24 * 100, 3)},
        "charts": [{"id": "midterm_paths", "file": "midterm_paths.png", "section": "seasonality",
                    "title": "Gold after 13 US midterms against every week since 1974"}],
    }
    if args.dry_run:
        slim = {k: v for k, v in payload.items() if k not in ("same_calendar_other_years",)}
        print(json.dumps(slim, ensure_ascii=False, indent=1))
        return 0
    directory = Path("reproduced")
    chart(s, mid, b_cal, directory / "charts/midterm_paths.png")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (directory / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
        "timezone_assumptions": "Weekly bars stamped Monday 00:00Z, moved to their Friday close; daily bars 00:00Z, "
                                "treated as calendar dates. Vote days are US calendar dates.",
        "sources": [
            {"role": "price_1w", "source_system": "repository tracked CSV",
             "locator": "XAUUSD_1wk.csv", **sha(WEEKLY),
             "immutable": False, "privacy": "private_raw_source", "note": "1970 onward; 1974 onward used."},
            {"role": "price_1d", "source_system": "repository tracked CSV",
             "locator": "XAUUSD_1d.csv", **sha(DAILY),
             "immutable": False, "privacy": "private_raw_source", "note": "2008-03-26 onward; daily check only."}]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
