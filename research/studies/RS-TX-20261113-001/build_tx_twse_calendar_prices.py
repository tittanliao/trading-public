#!/usr/bin/env python3
"""RS-TX-20261113-001 revision 2 — put the Yahoo ^TWII daily file on the TWSE trading calendar.

Yahoo ^TWII has no Saturday rows; TWSE traded on Saturdays until 2001, so before 2001 two Yahoo rows
in a row are not always two consecutive sessions. This script:
  - keeps every Yahoo row whose date is a TWSE session (OHLCV unchanged, source=yahoo);
  - inserts each TWSE session inside the Yahoo date range that Yahoo lacks (close = TWSE FMTQIK close,
    open/high/low/volume empty, source=twse);
  - drops each Yahoo row whose date is not a TWSE session (source rows listed in the report);
  - compares closes on shared dates; where they differ by more than --tol points the exchange close
    replaces the Yahoo close (source=yahoo_close_from_twse) and the date is listed in the report.
The runner reads only date and close. Writes the merged CSV and a JSON report; reads nothing else.

Usage: build_tx_twse_calendar_prices.py --yahoo CSV --twse CSV --out CSV --report JSON [--tol 0.5]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yahoo", required=True)
    ap.add_argument("--twse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args()

    y = pd.read_csv(a.yahoo, dtype={"date": str})
    t = pd.read_csv(a.twse, dtype={"date": str})
    first, last = y["date"].min(), y["date"].max()
    t = t[(t["date"] >= first) & (t["date"] <= last)].copy()
    ys, ts = set(y["date"]), set(t["date"])

    inserted = sorted(ts - ys)
    dropped = sorted(ys - ts)
    shared = y[y["date"].isin(ts)].merge(t[["date", "taiex_close"]], on="date")
    shared["diff"] = shared["close"] - shared["taiex_close"]
    big = shared[shared["diff"].abs() > a.tol]

    keep = y[y["date"].isin(ts)].copy()
    keep["source"] = "yahoo"
    fix = keep["date"].isin(set(big["date"]))
    tmap = dict(zip(t["date"], t["taiex_close"]))
    keep.loc[fix, "close"] = keep.loc[fix, "date"].map(tmap)
    keep.loc[fix, "source"] = "yahoo_close_from_twse"
    ins = t[t["date"].isin(inserted)][["date", "taiex_close"]].rename(columns={"taiex_close": "close"})
    ins["source"] = "twse"
    m = pd.concat([keep, ins], ignore_index=True).sort_values("date").reset_index(drop=True)
    m = m[["date", "open", "high", "low", "close", "volume", "source"]]
    m.to_csv(a.out, index=False)

    by_year = {}
    for s in inserted:
        by_year[s[:4]] = by_year.get(s[:4], 0) + 1
    wd = pd.to_datetime(pd.Series(inserted)).dt.dayofweek.value_counts().to_dict() if inserted else {}
    rep = {
        "yahoo_rows": len(y), "twse_sessions_in_range": len(t), "merged_rows": len(m),
        "range": [first, last],
        "inserted_twse_only": inserted, "inserted_by_year": by_year,
        "inserted_by_weekday": {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][k]: int(v) for k, v in sorted(wd.items())},
        "dropped_yahoo_only": dropped,
        "shared": len(shared),
        "close_diff_abs_max": round(float(shared["diff"].abs().max()), 4),
        "close_diff_over_tol": [{"date": r.date, "yahoo": round(r.close, 2), "twse": r.taiex_close,
                                 "diff": round(r.diff, 2)} for r in big.itertuples()],
        "tol_points": a.tol,
    }
    with open(a.report, "w") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(f"yahoo {len(y)} twse {len(t)} merged {len(m)}; inserted {len(inserted)} dropped {len(dropped)}; "
          f"close diffs > {a.tol}: {len(big)} (max {rep['close_diff_abs_max']})")


if __name__ == "__main__":
    main()
