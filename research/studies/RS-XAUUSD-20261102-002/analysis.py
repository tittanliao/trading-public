#!/usr/bin/env python3
"""RS-XAUUSD-20261102-002 — what does gold do around a US CPI release?

Design fixed in research/studies/the study record before the run.
Descriptive research only: no trading rule, no live impact.

    python3.12 scripts/research/build_xauusd_cpi_reaction.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
try:  # the Public copy runs without the package helper; it writes reproduced/results.json itself
    import study_package as pkg  # noqa: E402
except ImportError:
    pkg = None

STUDY_ID = "RS-XAUUSD-20261102-002"
OUT = Path("reproduced")
TAIPEI_TZ = timezone(timedelta(hours=8))
NY = ZoneInfo("America/New_York")
TPE = ZoneInfo("Asia/Taipei")
DAILY = Path("local-inputs/XAUUSD_1d.csv")
EXPORTS = Path("local-inputs")
M30_FILES = ("FX_IDC_XAUUSD, 30.csv", "FX_IDC_XAUUSD, 30_to_2026-10-03.csv")
M15_FILE = "FX_IDC_XAUUSD, 15_to_2026-10-31.csv"
DATES = Path("local-inputs/cpi_release_dates.csv")
NFP_DATES = Path("local-inputs/nfp_release_dates.csv")   # excluded from every baseline
DAILY_START = pd.Timestamp("2008-03-26")
ERAS = (("2008–2015", 2008, 2015), ("2016–2023", 2016, 2023), ("2024–2026", 2024, 2026))
DRIFT_DAYS = 5
BARS_1H, BARS_4H = 2, 8


def sha(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]


def mean_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference in means with a 95% resolution bound (1.96 x pooled standard error)."""
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = a.mean() - b.mean()
    return {"diff": round(float(d), 4), "bound_95": round(1.96 * se, 4),
            "resolved": bool(abs(d) > 1.96 * se)}


def prop_diff(a: dict, b: dict) -> dict:
    """Difference of two shares in percentage points with a 95% bound (normal approximation)."""
    pa, pb = a["k"] / a["n"], b["k"] / b["n"]
    se = math.sqrt(pa * (1 - pa) / a["n"] + pb * (1 - pb) / b["n"])
    return {"diff_pts": round((pa - pb) * 100, 1), "bound_95_pts": round(196 * se, 1),
            "resolved": bool(abs(pa - pb) > 1.96 * se)}


def share(mask: np.ndarray) -> dict:
    k, n = int(mask.sum()), int(len(mask))
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "ci95_pct": wilson(k, n)}


def load_dates() -> list[pd.Timestamp]:
    frame = pd.read_csv(DATES)
    return [pd.Timestamp(d) for d in frame["date_et"]]


def load_nfp() -> set:
    return {pd.Timestamp(d).normalize() for d in pd.read_csv(NFP_DATES)["date_et"]}


# ---------------------------------------------------------------- daily layer
def load_daily() -> pd.DataFrame:
    f = pd.read_csv(DAILY, parse_dates=["time"])
    f.index = f["time"].dt.tz_localize(None).dt.normalize()
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f[["open", "high", "low", "close"]].astype(float)


def daily_layer(dates: list[pd.Timestamp]) -> dict:
    d = load_daily()
    d = d[d.index >= DAILY_START - pd.Timedelta(days=10)]
    close = d["close"]
    ret = close.pct_change() * 100
    rng = (d["high"] - d["low"]) / close.shift(1) * 100
    drift = (close.shift(-DRIFT_DAYS) / close - 1) * 100
    prev_ret = ret.shift(1)
    frame = pd.DataFrame({"ret": ret, "rng": rng, "drift": drift, "prev_ret": prev_ret,
                          "prev_rng": rng.shift(1), "close": close})
    frame = frame[frame.index >= DAILY_START].dropna(subset=["ret", "rng"])
    ev_set = {x.normalize() for x in dates}
    frame["nfp"] = frame.index.isin(ev_set)            # "nfp" = the event column (CPI here; name kept from the NFP runner)
    other = frame.index.isin(load_nfp())                # NFP days leave the baseline
    frame = frame[~(other & ~frame["nfp"])]
    frame["fri"] = frame.index.weekday.isin([1, 2, 3])  # CPI lands Tue-Thu; "fri" = Tue-Thu baseline here
    missing = sorted(str(x.date()) for x in ev_set
                     if DAILY_START <= x <= frame.index.max() and x not in frame.index)
    ev, base, fri = frame[frame.nfp], frame[~frame.nfp], frame[~frame.nfp & frame.fri]

    def block(e: pd.DataFrame, b: pd.DataFrame, f: pd.DataFrame) -> dict:
        ea, ba, fa = e.ret.abs().values, b.ret.abs().values, f.ret.abs().values
        return {
            "n_event": int(len(e)), "n_base": int(len(b)), "n_fri": int(len(f)),
            "abs_ret_mean": {"event": round(ea.mean(), 3), "base": round(ba.mean(), 3),
                             "fri": round(fa.mean(), 3), "vs_base": mean_diff(ea, ba),
                             "vs_fri": mean_diff(ea, fa)},
            "abs_ret_median": {"event": round(float(np.median(ea)), 3),
                               "base": round(float(np.median(ba)), 3)},
            "range_mean": {"event": round(e.rng.mean(), 3), "base": round(b.rng.mean(), 3),
                           "vs_base": mean_diff(e.rng.values, b.rng.values)},
            "ge1": {"event": share(ea >= 1), "base": share(ba >= 1)},
            "ge2": {"event": share(ea >= 2), "base": share(ba >= 2), "diff": prop_diff(share(ea >= 2), share(ba >= 2))},
            "up": {"event": share(e.ret.values > 0), "base": share(b.ret.values > 0),
                   "diff": prop_diff(share(e.ret.values > 0), share(b.ret.values > 0))},
            "ret_mean": {"event": round(e.ret.mean(), 3), "base": round(b.ret.mean(), 3),
                         "vs_base": mean_diff(e.ret.values, b.ret.values)},
        }

    out = block(ev, base, fri)
    evd, bd = ev.dropna(subset=["drift"]), base.dropna(subset=["drift"])
    out["drift_5d"] = {"event_mean": round(evd.drift.mean(), 3), "base_mean": round(bd.drift.mean(), 3),
                       "event_up": share(evd.drift.values > 0), "base_up": share(bd.drift.values > 0),
                       "vs_base": mean_diff(evd.drift.values, bd.drift.values)}
    evp, bp = ev.dropna(subset=["prev_rng"]), base.dropna(subset=["prev_rng"])
    out["day_before"] = {"abs_ret_mean": round(evp.prev_ret.abs().mean(), 3),
                         "base_abs_ret_mean": round(bp.ret.abs().mean(), 3),
                         "range_mean": round(evp.prev_rng.mean(), 3), "base_range_mean": round(bp.rng.mean(), 3),
                         "vs_base_abs": mean_diff(evp.prev_ret.abs().values, bp.ret.abs().values)}
    out["eras"] = {}
    for name, a, b in ERAS:
        e = ev[(ev.index.year >= a) & (ev.index.year <= b)]
        bb = base[(base.index.year >= a) & (base.index.year <= b)]
        ff = fri[(fri.index.year >= a) & (fri.index.year <= b)]
        if len(e) >= 5:
            out["eras"][name] = block(e, bb, ff)
    big = ev.reindex(ev.ret.abs().sort_values(ascending=False).index).head(8)
    out["largest"] = [{"date": str(i.date()), "ret_pct": round(r.ret, 2), "range_pct": round(r.rng, 2),
                       "close": round(r.close, 2)} for i, r in big.iterrows()]
    out["events"] = [{"date": str(i.date()), "ret_pct": round(r.ret, 3), "range_pct": round(r.rng, 3),
                      "drift5_pct": None if pd.isna(r.drift) else round(r.drift, 3)} for i, r in ev.iterrows()]
    out["coverage"] = {"first": str(frame.index.min().date()), "last": str(frame.index.max().date()),
                       "release_dates_in_span_missing_from_series": missing}
    return out


# ------------------------------------------------------------- intraday layer
def load_30m() -> pd.DataFrame:
    parts = []
    for name in M30_FILES:
        f = pd.read_csv(EXPORTS / name, usecols=["time", "open", "high", "low", "close"])
        parts.append(f)
    m15 = pd.read_csv(EXPORTS / M15_FILE, usecols=["time", "open", "high", "low", "close"])
    m15["time"] = pd.to_datetime(m15["time"], utc=True)
    m15 = m15.set_index("time").sort_index()
    r = m15.resample("30min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    r = r.reset_index()
    parts.append(r)
    allf = []
    for f in parts:
        f = f.copy()
        f["time"] = pd.to_datetime(f["time"], utc=True)
        allf.append(f)
    m = pd.concat(allf).drop_duplicates("time", keep="first").set_index("time").sort_index()
    return m.astype(float)


def bar_metrics(m: pd.DataFrame, start_utc: pd.Timestamp) -> dict | None:
    if start_utc not in m.index:
        return None
    i = m.index.get_loc(start_utc)
    if i + BARS_4H > len(m):
        return None
    win = m.iloc[i:i + BARS_4H]
    # require the 4 hours to be contiguous (no session gap inside)
    if (win.index[-1] - win.index[0]) != pd.Timedelta(minutes=30 * (BARS_4H - 1)):
        return None
    o = win["open"].iloc[0]
    b0 = win.iloc[0]
    r30 = (b0.close / o - 1) * 100
    r1h = (win["close"].iloc[BARS_1H - 1] / o - 1) * 100
    r4h = (win["close"].iloc[-1] / o - 1) * 100
    hi4, lo4 = win["high"].max(), win["low"].min()
    after = (win["close"].iloc[-1] / b0.close - 1) * 100      # post-hoc: bars 2..8 only
    return {"r30": r30, "rng30": (b0.high - b0.low) / o * 100, "r1h": r1h, "r4h": r4h, "after": after,
            "reverse_after": bool(np.sign(r30) != 0 and np.sign(after) != 0 and np.sign(after) != np.sign(r30)),
            "rng4h": (hi4 - lo4) / o * 100,
            "whipsaw": bool(np.sign(r30) != 0 and np.sign(r4h) != 0 and np.sign(r30) != np.sign(r4h)),
            "open": o, "points30": b0.high - b0.low}


def release_utc(day: pd.Timestamp) -> pd.Timestamp:
    local = datetime(day.year, day.month, day.day, 8, 30, tzinfo=NY)
    return pd.Timestamp(local).tz_convert("UTC")


def intraday_layer(dates: list[pd.Timestamp]) -> dict:
    m = load_30m()
    first, last = m.index.min(), m.index.max()
    ev_days = [d for d in dates if first <= release_utc(d) <= last]
    ev, skipped = [], []
    for d in ev_days:
        x = bar_metrics(m, release_utc(d))
        if x is None:
            skipped.append(str(d.date()))
            continue
        x["date"] = str(d.date())
        x["taipei"] = release_utc(d).tz_convert(TPE).strftime("%H:%M")
        ev.append(x)
    ev_set = {d.normalize() for d in dates}
    nfp_set = load_nfp()
    base = []
    d = first.tz_convert(NY).normalize().tz_localize(None)
    end = last.tz_convert(NY).tz_localize(None)
    while d <= end:
        if d.weekday() < 5 and d not in ev_set and d not in nfp_set:
            x = bar_metrics(m, release_utc(d))
            if x is not None:
                base.append(x)
        d += pd.Timedelta(days=1)
    E, B = pd.DataFrame(ev), pd.DataFrame(base)
    med_b = float(B.rng30.median())
    out = {
        "n_event": len(E), "n_base_fridays": len(B), "skipped_release_dates": skipped,
        "coverage": {"first": str(first.tz_convert(TPE)), "last": str(last.tz_convert(TPE))},
        "rng30": {"event_mean": round(E.rng30.mean(), 3), "event_median": round(E.rng30.median(), 3),
                  "base_mean": round(B.rng30.mean(), 3), "base_median": round(med_b, 3),
                  "multiple_of_base_median": round(float(E.rng30.median()) / med_b, 2),
                  "vs_base": mean_diff(E.rng30.values, B.rng30.values)},
        "points30_median": {"event": round(float(E.points30.median()), 2), "base": round(float(B.points30.median()), 2)},
        "abs_r1h": {"event_mean": round(E.r1h.abs().mean(), 3), "base_mean": round(B.r1h.abs().mean(), 3),
                    "vs_base": mean_diff(E.r1h.abs().values, B.r1h.abs().values)},
        "rng4h": {"event_mean": round(E.rng4h.mean(), 3), "base_mean": round(B.rng4h.mean(), 3),
                  "vs_base": mean_diff(E.rng4h.values, B.rng4h.values)},
        "whipsaw": {"event": share(E.whipsaw.values), "base": share(B.whipsaw.values),
                    "diff": prop_diff(share(E.whipsaw.values), share(B.whipsaw.values))},
        "posthoc_reverse_after_first_bar": {
            "note": "pre-registered secondary (carried over from the NFP study): sign of close(bar 8) / close(bar 1) "
                    "against the first-bar sign, so the first bar no longer counts toward its own continuation",
            "event": share(E.reverse_after.values), "base": share(B.reverse_after.values),
            "diff": prop_diff(share(E.reverse_after.values), share(B.reverse_after.values)),
            "event_after_mean_abs": round(E["after"].abs().mean(), 3), "base_after_mean_abs": round(B["after"].abs().mean(), 3)},
        "first_bar_up": {"event": share(E.r30.values > 0)},
        "r4h_up": {"event": share(E.r4h.values > 0)},
        "events": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                    if k in ("date", "taipei", "r30", "rng30", "r1h", "r4h", "rng4h", "whipsaw", "points30", "after", "reverse_after")}
                   for r in ev],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    dates = load_dates()
    daily = daily_layer(dates)
    intra = intraday_layer(dates)
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI_TZ).isoformat(timespec="seconds"),
        "method": {
            "release_time": "08:30 America/New_York, converted with zoneinfo",
            "daily": "XAUUSD_1d close-to-close; release day D vs non-release days (NFP days excluded) and non-release Tue-Thu; "
                     f"drift close(D+{DRIFT_DAYS})/close(D); bound = 1.96 x pooled SE",
            "intraday": "FX_IDC:XAUUSD 30m chart exports (Oct 2026 resampled from 15m); release bar opens 08:30 ET; "
                        "1h = close of 2nd bar, 4h = close of 8th bar, both vs release-bar open; baseline = same ET "
                        "bar on non-release weekdays (NFP days excluded); whipsaw = first-bar sign differs from 4h sign",
            "eras": [e[0] for e in ERAS],
        },
        "daily": daily, "intraday": intra,
        "limitations": [
            "No CPI consensus or surprise data: 'hot vs cool print' is not tested.",
            "Daily series is the repository XAUUSD_1d, not TradingView; levels differ slightly.",
            f"Intraday sample is {intra['n_event']} releases (2024-01 onward); shares carry wide intervals.",
            "Descriptive only; no trading rule is implied.",
        ],
    }
    if a.dry:
        print(json.dumps({"daily": {k: v for k, v in daily.items() if k not in ("events", "eras")},
                          "intraday": {k: v for k, v in intra.items() if k != "events"}},
                         ensure_ascii=False, indent=1, default=str))
        return
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    def loc(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return path.name

    inputs = [{"role": "price_1d", "source_system": "repository tracked CSV", "locator": loc(DAILY),
               **sha(DAILY), "immutable": False, "privacy": "private_raw_source", "note": "2008-03-26 onward used"}]
    inputs += [{"role": f"price_30m_{k}", "source_system": "TradingView FX_IDC:XAUUSD chart export",
                "locator": loc(EXPORTS / n), **sha(EXPORTS / n), "immutable": True, "privacy": "private_raw_source"}
               for k, n in enumerate(M30_FILES, 1)]
    inputs += [{"role": "price_15m", "source_system": "TradingView FX_IDC:XAUUSD chart export",
                "locator": loc(EXPORTS / M15_FILE), **sha(EXPORTS / M15_FILE), "immutable": True,
                "privacy": "private_raw_source", "note": "resampled to 30m to extend past 2026-10-03"},
               {"role": "cpi_release_dates", "source_system": "BLS CPI archived news releases (public)",
                "locator": loc(DATES), **sha(DATES), "immutable": True, "privacy": "public", "observed": "2026-11-02",
                "note": "https://www.bls.gov/bls/news-release/cpi.htm; dates from cpi_MMDDYYYY links"}]
    manifest = {"schema_version": 1, "study_id": STUDY_ID, "inputs": inputs}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
