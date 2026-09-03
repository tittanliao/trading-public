#!/usr/bin/env python3
"""Month-open to month-close returns for XAUUSD and TX, as a year x month heatmap.

"Buy on the first trading day, sell on the last" — one number per instrument-month, laid
out as years down and calendar months across.

RS-TX-20260728-001 already answers this for TX under the section 14 monthly seasonality
contract, but from WEEKLY bars, where a month's open is the open of its first weekly bar
rather than of its first trading day. Weekly bars straddle month boundaries, so that open
can belong to the previous month. This runner uses DAILY bars wherever they exist, which
makes the month boundary exact, and reports the weekly-derived series alongside for XAUUSD
so the long history is still available and the approximation is measurable rather than
assumed.

  XAUUSD  daily 2008-2026 (exact) + weekly 1980-2026 (approximate boundaries, long history)
  TX      daily 2012-2026 (exact) + weekly 2012-2026 (approximate boundaries)

Returns are in PERCENT, not points. Gold runs 300 to 4500 and TX 7400 to 42000 across
these samples, so a points heatmap would be a picture of the price level rather than of
seasonality.

An incomplete month is excluded: a month counts only when the series continues into a
later month, so the last month of every file is dropped rather than half-measured.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("reproduced")

# XAUUSD points at the TRACKED weekly-workflow output, not at a staged copy, so a
# production data refresh flows into this study with no manual step. TX has no such
# pipeline — its only source is the read-only legacy repo — so it is staged into
# local-inputs and `stage_tx()` below says exactly how.
SERIES = {
    "xauusd_daily": {"path": Path("local-inputs/XAUUSD_1d.csv"), "label": "XAUUSD 1d — weekly-workflow output", "instrument": "XAUUSD",
                     "grain": "daily", "boundary": "exact", "start_year": None,
                     "refresh": "the weekly production refresh"},
    "xauusd_weekly": {"path": Path("local-inputs/XAUUSD_1wk.csv"), "label": "XAUUSD 1wk — weekly-workflow output", "instrument": "XAUUSD",
                      "grain": "weekly", "boundary": "approximate", "start_year": 1980,
                      "refresh": "the weekly production refresh"},
    "tx_daily": {"path": Path("local-inputs/tx-mxf-daily.csv"), "label": "TX MXF1! 1D — staged export", "instrument": "TX (MXF1! front-month)",
                 "grain": "daily", "boundary": "exact", "start_year": None,
                 "refresh": "stage from a fresh TAIFEX_DLY_MXF1! 1D export"},
    "tx_weekly": {"path": Path("local-inputs/tx-mxf-weekly.csv"), "label": "TX MXF1! 1W — staged export", "instrument": "TX (MXF1! front-month)",
                  "grain": "weekly", "boundary": "approximate", "start_year": None,
                  "refresh": "stage from a fresh TAIFEX_DLY_MXF1! 1W export"},
}
LEGACY_TX = {
    Path("local-inputs/tx-mxf-daily.csv"): Path("local-inputs/TAIFEX_DLY_MXF1!, 1D.csv"),
    Path("local-inputs/tx-mxf-weekly.csv"): Path("local-inputs/TAIFEX_DLY_MXF1!, 1W.csv"),
}


def stage_tx() -> None:
    """Copy the TX exports out of the read-only legacy repo if they are not staged yet.

    Never writes to the legacy repo, and never overwrites a staged file — a fresher TX
    export dropped into local-inputs by hand must win over the 2026-05 legacy one.
    """
    for dest, src in LEGACY_TX.items():
        if dest.is_file() or not src.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        print(f"staged {dest.name} from the read-only legacy repo")
# Gold before 1980 is a close-only reconstruction in this export: every weekly bar has
# high == low. Month returns computed from it would be an artefact of the reconstruction,
# so the weekly series starts where real OHLC starts. Same cut RS-XAUUSD-20260827-001 made.


def load(path: Path, start_year: int | None) -> list[dict]:
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        stamp = (row.get("time") or "").strip()
        if not stamp:
            continue
        try:
            day = datetime.fromisoformat(stamp[:10]).date()
            values = {k: float(row[k]) for k in ("open", "high", "low", "close")}
        except (ValueError, TypeError, KeyError):
            continue          # blank or malformed row in a legacy export
        if start_year and day.year < start_year:
            continue
        rows.append({"date": day, **values})
    rows.sort(key=lambda r: r["date"])
    return rows


def monthly(bars: list[dict]) -> list[dict]:
    """One record per completed month: first bar's open to last bar's close."""
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for bar in bars:
        buckets[(bar["date"].year, bar["date"].month)].append(bar)
    keys = sorted(buckets)
    out = []
    for year, month in keys[:-1]:          # the final month is never known to be complete
        group = buckets[(year, month)]
        first, last = group[0], group[-1]
        change = last["close"] - first["open"]
        out.append({
            "year": year, "month": month, "bars": len(group),
            "first_date": first["date"].isoformat(), "last_date": last["date"].isoformat(),
            "month_open": round(first["open"], 2), "month_close": round(last["close"], 2),
            "change_pts": round(change, 2),
            "change_pct": round(100 * change / first["open"], 3),
            "up": change > 0,
        })
    return out


def by_month(records: list[dict]) -> dict:
    result = {}
    for month in range(1, 13):
        sub = [r for r in records if r["month"] == month]
        if not sub:
            continue
        values = [r["change_pct"] for r in sub]
        wins = sum(1 for r in sub if r["up"])
        ordered = sorted(values)
        median = (ordered[len(ordered) // 2] if len(ordered) % 2
                  else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2)
        result[str(month)] = {
            "n": len(sub),
            "win_rate_pct": round(100 * wins / len(sub), 2),
            "mean_pct": round(sum(values) / len(values), 3),
            "median_pct": round(median, 3),
            "best_pct": round(max(values), 3),
            "worst_pct": round(min(values), 3),
            "total_pct": round(sum(values), 2),
        }
    return result


def resolution_bound(records: list[dict], month: int) -> float | None:
    """Smallest mean difference this month's sample could separate from the rest.

    alpha 0.05, power 0.80. A monthly seasonal claim gets one observation per year, so
    this is usually larger than the effect and that is the point of printing it.
    """
    a = [r["change_pct"] for r in records if r["month"] == month]
    b = [r["change_pct"] for r in records if r["month"] != month]
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = a + b
    mean = sum(pooled) / len(pooled)
    sd = (sum((x - mean) ** 2 for x in pooled) / (len(pooled) - 1)) ** 0.5
    return round(2.802 * sd * (1 / len(a) + 1 / len(b)) ** 0.5, 3)


def build(key: str, cfg: dict) -> dict:
    path = cfg["path"]
    if not path.is_file():
        raise SystemExit(f"missing input: {path}\n  refresh with: {cfg['refresh']}")
    bars = load(path, cfg["start_year"])
    records = monthly(bars)
    months = by_month(records)
    for month, stats in months.items():
        stats["min_detectable_pct"] = resolution_bound(records, int(month))
        stats["separates"] = bool(
            stats["min_detectable_pct"] is not None
            and abs(stats["mean_pct"]) > stats["min_detectable_pct"])
    grid: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        grid[str(r["year"])][str(r["month"])] = r["change_pct"]
    values = [r["change_pct"] for r in records]
    return {
        "instrument": cfg["instrument"], "grain": cfg["grain"],
        "month_boundary": cfg["boundary"],
        # The reader-facing label, not the path. The Public exporter reduces tracked input
        # paths to basenames but has no rule for staged ones, so emitting a path here would
        # publish gold as a basename and TX as a staging path — one field, two conventions.
        "source": cfg["label"],
        "bars": len(bars),
        "from": bars[0]["date"].isoformat(), "to": bars[-1]["date"].isoformat(),
        "months_measured": len(records),
        "first_month": f"{records[0]['year']}-{records[0]['month']:02d}",
        "last_month": f"{records[-1]['year']}-{records[-1]['month']:02d}",
        "overall": {
            "win_rate_pct": round(100 * sum(1 for r in records if r["up"]) / len(records), 2),
            "mean_pct": round(sum(values) / len(values), 3),
            "best_pct": round(max(values), 3), "worst_pct": round(min(values), 3),
        },
        "by_month": months,
        "year_month_grid_pct": dict(grid),
        "records": records,
    }


def main() -> None:
    stage_tx()
    built = {key: build(key, cfg) for key, cfg in SERIES.items()}

    # Does the weekly month boundary actually distort the picture? Compare the two grains
    # on the overlap only — anything else compares different years as well as different
    # boundaries.
    agreement = {}
    for instrument, daily_key, weekly_key in (("XAUUSD", "xauusd_daily", "xauusd_weekly"),
                                              ("TX", "tx_daily", "tx_weekly")):
        d, w = built[daily_key], built[weekly_key]
        lo = max(d["first_month"], w["first_month"])
        hi = min(d["last_month"], w["last_month"])
        rows = []
        for month in range(1, 13):
            dm = [r["change_pct"] for r in d["records"]
                  if r["month"] == month and lo <= f"{r['year']}-{r['month']:02d}" <= hi]
            wm = [r["change_pct"] for r in w["records"]
                  if r["month"] == month and lo <= f"{r['year']}-{r['month']:02d}" <= hi]
            if not dm or not wm:
                continue
            rows.append({
                "month": month, "n": len(dm),
                "daily_win_rate_pct": round(100 * sum(1 for v in dm if v > 0) / len(dm), 2),
                "weekly_win_rate_pct": round(100 * sum(1 for v in wm if v > 0) / len(wm), 2),
                "daily_mean_pct": round(sum(dm) / len(dm), 3),
                "weekly_mean_pct": round(sum(wm) / len(wm), 3),
            })
        gaps = [abs(r["daily_win_rate_pct"] - r["weekly_win_rate_pct"]) for r in rows]
        agreement[instrument] = {
            "overlap_from": lo, "overlap_to": hi,
            "mean_absolute_win_rate_gap_pct_points": round(sum(gaps) / len(gaps), 2) if gaps else None,
            "max_absolute_win_rate_gap_pct_points": round(max(gaps), 2) if gaps else None,
            "by_month": rows,
        }

    results = {
        "schema_version": 1, "study_id": "RS-MULTI-20260901-001",
        "title": "Buy the month open, sell the month close — XAUUSD and TX",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "rule": "buy at the first bar's open of each calendar month, sell at the last bar's close",
            "unit": "percent of the month's open price",
            "incomplete_months": "excluded; the final month of each series is dropped",
            "why_daily": ("a weekly bar straddles the month boundary, so a weekly-derived "
                          "month open can belong to the previous month"),
            "xauusd_pre_1980_excluded": ("that part of the export is a close-only "
                                         "reconstruction — every weekly bar has high == low"),
            "costs_excluded": "spread, commission, carry and futures roll are not modelled",
            "prior_work": "RS-TX-20260728-001 measured TX under the section 14 weekly contract",
        },
        "series": {k: {kk: vv for kk, vv in v.items() if kk != "records"}
                   for k, v in built.items()},
        "input_freshness": {
            k: {"source": SERIES[k]["label"], "last_bar": v["to"],
                "days_stale_at_generation": (
                    datetime.now(timezone.utc).date()
                    - datetime.fromisoformat(v["to"]).date()).days,
                "refresh": SERIES[k]["refresh"]}
            for k, v in built.items()},
        "grain_agreement": agreement,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    for key, value in built.items():
        print(f"{key:14s} {value['first_month']}..{value['last_month']}  "
              f"{value['months_measured']:>4d} months  win {value['overall']['win_rate_pct']}%  "
              f"mean {value['overall']['mean_pct']}%")
    for instrument, value in agreement.items():
        print(f"{instrument} grain agreement: mean |gap| "
              f"{value['mean_absolute_win_rate_gap_pct_points']} pts, max "
              f"{value['max_absolute_win_rate_gap_pct_points']} pts "
              f"({value['overlap_from']}..{value['overlap_to']})")


if __name__ == "__main__":
    main()
