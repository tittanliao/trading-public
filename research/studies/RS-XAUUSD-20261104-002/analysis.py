#!/usr/bin/env python3
"""RS-XAUUSD-20261104-002 — gold around FOMC rate decisions (event study, FX_IDC 30m).

Implements the pre-registration, Design amendment 1 and Design amendment 2 in
the research record. Descriptive research only: no trading
rule, no live impact.

Two stages, run in this order:

    ~/.venvs/trading-research/bin/python scripts/research/build_xauusd_fomc_event.py --checks-only
        writes precheck.json: coverage, DST, overlap and event/control-day counts. It decides which
        days are usable from bar existence and zero-range flags only; no return, range or direction
        is computed.
    ~/.venvs/trading-research/bin/python scripts/research/build_xauusd_fomc_event.py
        refuses to run unless the pre-run checks pass, then writes results.json and README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-XAUUSD-20261104-002"
OUT = Path("reproduced")
OUT.mkdir(parents=True, exist_ok=True)
EXPORTS = Path("local-inputs")
MACRO = Path("local-inputs")
NY = ZoneInfo("America/New_York")
TPE = ZoneInfo("Asia/Taipei")
TAIPEI_TZ = timezone(timedelta(hours=8))
F30_1, F30_2 = "FX_IDC_XAUUSD, 30.csv", "FX_IDC_XAUUSD, 30_to_2026-10-03.csv"
F15 = "FX_IDC_XAUUSD, 15_to_2026-10-31.csv"
MGC30 = "COMEX_MINI_DL_MGC1!, 30.csv"
SLOTS = ["14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]
HOLM_ALPHA = 0.05
N_PRIMARY = 4
FREEZE = "2026-11-04"
SAMPLE_END = pd.Timestamp("2026-10-28")
OFFSETS_L = list(range(-10, -1))
OFFSETS_R = list(range(2, 11))
MIN_TOTAL, MIN_SIDE, MIN_GROUP = 8, 4, 8


# ----------------------------------------------------------------- helpers
def sha(path: Path) -> dict:
    b = path.read_bytes()
    return {"sha256": hashlib.sha256(b).hexdigest(), "size_bytes": len(b)}


def rel(path: Path) -> str:
    return str(path)


def tcrit(alpha_two_sided: float, df: float) -> float:
    return float(stats.t.ppf(1 - alpha_two_sided / 2, df))


def thanksgiving_friday(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=11, day=1)
    thursdays = [d + pd.Timedelta(days=i) for i in range(30) if (d + pd.Timedelta(days=i)).weekday() == 3
                 and (d + pd.Timedelta(days=i)).month == 11]
    return thursdays[3] + pd.Timedelta(days=1)


def holiday_adjacent(d: pd.Timestamp) -> bool:
    return (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day == 1) or d == thanksgiving_friday(d.year)


# ----------------------------------------------------------------- bars
def read_export(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    f["label"] = f["time"].astype(str)
    f["utc"] = pd.to_datetime(f["time"], utc=True)
    return f.drop(columns="time")


def synth30(m15: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """30m from 15m: valid only when both 15m bars exist (amendment A3). Returns (bars, dropped, dropped_on_weekdays)."""
    x = m15.set_index("utc").sort_index()
    g = x.resample("30min", label="left", closed="left")
    agg = g.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    cnt = g["open"].count()
    ok = cnt == 2
    dropped_idx = agg.index[(cnt == 1)]
    wk = [t.astimezone(NY).weekday() < 5 for t in dropped_idx]
    out = agg[ok].reset_index()
    out["label"] = out["utc"].dt.tz_convert(TPE).dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    return out, int((cnt == 1).sum()), int(sum(wk))


def build_union() -> tuple[pd.DataFrame, dict]:
    a, b, c = read_export(EXPORTS / F30_1), read_export(EXPORTS / F30_2), read_export(EXPORTS / F15)
    syn, dropped, dropped_wk = synth30(c)
    a["src"], b["src"], syn["src"] = 1, 2, 4
    allb = pd.concat([a, b, syn], ignore_index=True)
    dup_within = int(a["utc"].duplicated().sum() + b["utc"].duplicated().sum() + c["utc"].duplicated().sum())
    u = allb.drop_duplicates("utc", keep="first").sort_values("utc").reset_index(drop=True)
    info = {"rows": {"30": len(a), "30_to_2026-10-03": len(b), "15m": len(c), "synthesized_30m": len(syn)},
            "duplicates_within_files": dup_within, "union_bars": len(u),
            "duplicates_after_union": int(u["utc"].duplicated().sum()),
            "synth_dropped_one_15m_bar": dropped, "synth_dropped_on_weekdays": dropped_wk,
            "first_bar_utc": str(u["utc"].iloc[0]), "last_bar_utc": str(u["utc"].iloc[-1]),
            "bars_by_source": {str(k): int(v) for k, v in u["src"].value_counts().sort_index().items()}}
    return u, info


def long_table(bars: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    ny = x["utc"].dt.tz_convert(NY)
    x["d"] = ny.dt.tz_localize(None).dt.normalize()
    x["hm"] = ny.dt.strftime("%H:%M")
    x["wd"] = ny.dt.weekday
    return x


def day_wide(lt: pd.DataFrame) -> pd.DataFrame:
    """One row per NY date with the six 14:00-16:30 ET bars as columns o1..c6 (NaN if absent)."""
    s = lt[lt["hm"].isin(SLOTS) & (lt["wd"] < 5)]
    p = s.pivot(index="d", columns="hm", values=["open", "high", "low", "close"])
    cols = {}
    for k, hm in enumerate(SLOTS, 1):
        for fld, ch in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")):
            if (fld, hm) in p.columns:
                cols[f"{ch}{k}"] = p[(fld, hm)]
            else:
                cols[f"{ch}{k}"] = pd.Series(np.nan, index=p.index)
    w = pd.DataFrame(cols)
    full = w.notna().all(axis=1)
    flat = pd.Series(False, index=w.index)
    for k in range(1, 7):
        flat |= (w[f"h{k}"] <= w[f"l{k}"])
    w["usable"] = full & ~flat
    w["has_all6"] = full
    w["flat_any"] = flat & full
    return w


def day_metrics(w: pd.DataFrame) -> pd.DataFrame:
    """Range and return measures for usable days (percent of price, signed returns in percent)."""
    u = w[w["usable"]].copy()
    m = pd.DataFrame(index=u.index)
    m["R30"] = (u.h1 - u.l1) / u.o1 * 100
    m["pts30"] = u.h1 - u.l1
    m["R60"] = (np.maximum(u.h1, u.h2) - np.minimum(u.l1, u.l2)) / u.o1 * 100
    m["r60"] = (u.c2 / u.o1 - 1) * 100
    m["Rs"] = m["R30"]
    m["Rp"] = (u.h2 - u.l2) / u.o2 * 100
    m["d"] = np.log(m["Rp"]) - np.log(m["Rs"])
    m["R3"] = (u.h3 - u.l3) / u.o3 * 100
    m["d3"] = np.log(m["R3"]) - np.log(m["Rs"])
    hs = u[[f"h{k}" for k in range(1, 7)]].max(axis=1)
    ls = u[[f"l{k}" for k in range(1, 7)]].min(axis=1)
    m["R180"] = (hs - ls) / u.o1 * 100
    m["r_b1"] = (u.c1 / u.o1 - 1) * 100
    m["r_b2"] = (u.c2 / u.o2 - 1) * 100
    m["lnR30"], m["lnR60"], m["lnR180"] = np.log(m["R30"]), np.log(m["R60"]), np.log(m["R180"])
    return m


# ----------------------------------------------------------------- calendars
def load_calendars() -> dict:
    fm = pd.read_csv(MACRO / "fomc_meetings.csv", dtype=str, keep_default_na=False)
    fm["date"] = pd.to_datetime(fm["statement_date_et"])
    all_fomc = set(fm["date"])
    minutes = {pd.Timestamp(x) for x in fm["minutes_release_date"] if x}
    beige = {pd.Timestamp(x) for x in pd.read_csv(MACRO / "fed_beige_book_dates.csv")["release_date_et"]}
    cpi = {pd.Timestamp(x) for x in pd.read_csv(MACRO / "cpi_release_dates.csv")["date_et"]}
    nfp = {pd.Timestamp(x) for x in pd.read_csv(MACRO / "nfp_release_dates.csv")["date_et"]}
    ev = fm[(fm["scheduled"] == "yes") & (fm["statement_time_et"] == "14:00") & (fm["presconf"] == "yes")
            & (fm["date"] >= "2024-01-01")].copy()
    return {"fm": fm, "events": ev.reset_index(drop=True), "fomc_days": all_fomc, "minutes": minutes,
            "beige": beige, "cpi": cpi, "nfp": nfp}


def excluded_control(d: pd.Timestamp, cal: dict) -> str | None:
    if holiday_adjacent(d):
        return "holiday_adjacent"
    if d in cal["fomc_days"]:
        return "fomc_day"
    if d in cal["minutes"]:
        return "minutes_day"
    if d in cal["beige"]:
        return "beige_book_day"
    if d in cal["cpi"]:
        return "cpi_day"
    if d in cal["nfp"]:
        return "nfp_day"
    return None


def weekday_index() -> pd.DatetimeIndex:
    return pd.bdate_range("2023-10-02", "2027-02-26")


def build_events(w: pd.DataFrame, cal: dict, first_bar_date: pd.Timestamp, last_bar_date: pd.Timestamp) -> list[dict]:
    wk = weekday_index()
    pos = {d: i for i, d in enumerate(wk)}
    out = []
    for _, r in cal["events"].iterrows():
        d = r["date"]
        rec = {"date": d, "sep": r["sep"] == "yes", "decision": r["decision"], "presconf_start": r["presconf_start_et"],
               "same_day_release": [x for x, s in (("CPI", cal["cpi"]), ("NFP", cal["nfp"])) if d in s]}
        if d > SAMPLE_END:
            rec["status"] = "out_of_range"
            out.append(rec)
            continue
        i = pos[d]
        first_needed = wk[i - 10]
        rec["start_rule_ok"] = bool(first_needed >= first_bar_date)
        rec["usable_event_bars"] = bool(d in w.index and w.loc[d, "usable"])
        rec["event_bars_detail"] = ("ok" if rec["usable_event_bars"] else
                                    "missing_bars" if (d not in w.index or not w.loc[d, "has_all6"]) else "zero_range_bar")
        ctl = {"L": [], "R": []}
        why: dict[str, int] = {}
        for side, offs in (("L", OFFSETS_L), ("R", OFFSETS_R)):
            for o in offs:
                c = wk[i + o]
                reason = excluded_control(c, cal)
                if reason is None and not (c in w.index and w.loc[c, "usable"]):
                    reason = "no_6_bars_or_flat" if c <= last_bar_date else "after_data_end"
                if reason:
                    why[reason] = why.get(reason, 0) + 1
                else:
                    ctl[side].append(c)
        rec["control_L"], rec["control_R"] = ctl["L"], ctl["R"]
        rec["control_dropped_reasons"] = why
        nL, nR = len(ctl["L"]), len(ctl["R"])
        ok = rec["start_rule_ok"] and rec["usable_event_bars"] and (nL + nR) >= MIN_TOTAL and nL >= MIN_SIDE and nR >= MIN_SIDE
        if ok:
            rec["status"] = "judged"
        elif not rec["start_rule_ok"]:
            rec["status"] = "excluded: before the start rule"
        elif not rec["usable_event_bars"]:
            rec["status"] = f"excluded: event bars unusable ({rec['event_bars_detail']})"
        else:
            rec["status"] = f"excluded: control days insufficient (left {nL}, right {nR}, total {nL + nR}; need >= {MIN_TOTAL} total and >= {MIN_SIDE} per side)"
        out.append(rec)
    return out


# ----------------------------------------------------------------- statistics
def ttest_mean(x: np.ndarray) -> dict:
    n = len(x)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    t = m / se
    p = float(2 * stats.t.sf(abs(t), n - 1))
    return {"n": n, "mean": m, "sd": sd, "se": se, "t": t, "p": p}


def holm(ps: list[float], alpha: float = HOLM_ALPHA) -> dict:
    """Step-down: j-th smallest tested at alpha/(m-j+1); stop at the first failure."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    reject = [False] * m
    level = [None] * m
    stop_level = None
    stopped = False
    for j, i in enumerate(order, 1):
        a = alpha / (m - j + 1)
        level[i] = a
        if not stopped and ps[i] <= a:
            reject[i] = True
        else:
            if not stopped:
                stop_level = a
            stopped = True
            level[i] = stop_level
    return {"reject": reject, "level": level}


def primary_block(vals: dict[str, np.ndarray], multiples: set[str]) -> dict:
    names = list(vals)
    tt = {k: ttest_mean(v) for k, v in vals.items()}
    h = holm([tt[k]["p"] for k in names])
    out = {}
    for idx, k in enumerate(names):
        r = tt[k]
        n = r["n"]
        lvl = h["level"][idx]
        b_holm = tcrit(lvl, n - 1) * r["se"]
        b_first = tcrit(0.05 / N_PRIMARY, n - 1) * r["se"]
        o = {**r,
             "holm_level_applied": lvl, "holm_rejects": bool(h["reject"][idx]), "cleared": bool(h["reject"][idx]),
             "t_crit_at_holm_level": tcrit(lvl, n - 1), "bound_at_holm_level": b_holm,
             "t_crit_first_step": tcrit(0.05 / N_PRIMARY, n - 1), "bound_first_step": b_first}
        if k in multiples:
            M = r["mean"]
            o["multiple"] = math.exp(M)
            o["multiple_interval_holm"] = [math.exp(M - b_holm), math.exp(M + b_holm)]
            o["multiple_interval_first_step"] = [math.exp(M - b_first), math.exp(M + b_first)]
            o["not_distinguishable_range_first_step"] = [math.exp(-b_first), math.exp(b_first)]
        out[k] = o
    return out


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    d = float(a.mean() - b.mean())
    t = d / se
    p = float(2 * stats.t.sf(abs(t), df))
    crit = tcrit(0.05, df) if min(na, nb) < 30 else 1.96
    return {"n_a": na, "n_b": nb, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": d, "se": se, "df": float(df),
            "p": p, "bound_95": crit * se, "resolved_uncorrected": bool(abs(d) > crit * se)}


def one_sample_secondary(x: np.ndarray, multiple: bool) -> dict:
    r = ttest_mean(x)
    n = r["n"]
    crit = tcrit(0.05, n - 1) if n < 30 else 1.96
    o = {"n": n, "mean": r["mean"], "se": r["se"], "p": r["p"], "bound_95": crit * r["se"],
         "resolved_uncorrected": bool(abs(r["mean"]) > crit * r["se"])}
    if multiple:
        o["multiple"] = math.exp(r["mean"])
        o["multiple_interval"] = [math.exp(r["mean"] - crit * r["se"]), math.exp(r["mean"] + crit * r["se"])]
    return o


def two_prop(k1: int, n1: int, k2: int, n2: int) -> dict:
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pool = (k1 + k2) / (n1 + n2)
    sep = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / sep if sep > 0 else 0.0
    return {"k_event": k1, "n_event": n1, "pct_event": round(100 * p1, 1), "k_control": k2, "n_control": n2,
            "pct_control": round(100 * p2, 1), "diff_pts": round(100 * (p1 - p2), 1), "bound_95_pts": round(196 * se, 1),
            "resolved_uncorrected": bool(abs(p1 - p2) > 1.96 * se), "p": float(2 * stats.norm.sf(abs(z)))}


def bh(ps: dict[str, float], q: float = 0.10) -> dict[str, bool]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    kmax = 0
    for k, (_, p) in enumerate(items, 1):
        if p <= k / m * q:
            kmax = k
    return {name: (k <= kmax) for k, (name, _) in enumerate(items, 1)}


# ----------------------------------------------------------------- event frame
def event_frame(events: list[dict], m: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Per judged event: own values and the mean over its control days, for every measure."""
    rows = []
    for e in events:
        if e["status"] != "judged":
            continue
        ctl = e["control_L"] + e["control_R"]
        row = {"date": e["date"], "sep": e["sep"], "decision": e["decision"], "n_ctl": len(ctl),
               "n_L": len(e["control_L"]), "n_R": len(e["control_R"]),
               "same_day_release": "|".join(e["same_day_release"])}
        for col in ("lnR30", "lnR60", "r60", "d", "lnR180", "d3", "r_b1", "r_b2", "R30", "pts30", "Rp", "Rs", "R60"):
            row[col] = float(m.loc[e["date"], col])
            if col in ("lnR30", "lnR60", "r60", "d", "lnR180", "d3", "r_b1", "r_b2"):
                row[f"ctl_{col}"] = float(m.loc[ctl, col].mean())
        row["ctl_median_R30"] = float(m.loc[ctl, "R30"].median())
        row["rev"] = bool(np.sign(row["r_b1"]) * np.sign(row["r_b2"]) < 0)
        row["ctl_rev_share"] = float((np.sign(m.loc[ctl, "r_b1"]) * np.sign(m.loc[ctl, "r_b2"]) < 0).mean())
        rows.append(row)
    f = pd.DataFrame(rows)
    if len(f):
        f["y30"] = f["lnR30"] - f["ctl_lnR30"]
        f["y60"] = f["lnR60"] - f["ctl_lnR60"]
        f["z"] = f["r60"] - f["ctl_r60"]
        f["q"] = f["d"] - f["ctl_d"]
        f["y180"] = f["lnR180"] - f["ctl_lnR180"]
        f["q3"] = f["d3"] - f["ctl_d3"]
        f["s8a"] = f["r_b1"] - f["ctl_r_b1"]
        f["s8b"] = f["r_b2"] - f["ctl_r_b2"]
    return f


def primaries_from(f: pd.DataFrame) -> dict:
    return primary_block({"P1a": f["y30"].values, "P1b": f["y60"].values, "P2": f["z"].values, "P3": f["q"].values},
                         {"P1a", "P1b", "P3"})


# ----------------------------------------------------------------- checks
def overlap_check(a: pd.DataFrame, b: pd.DataFrame, lo: pd.Timestamp | None = None, hi: pd.Timestamp | None = None) -> dict:
    j = a.merge(b, on="utc", suffixes=("_a", "_b"))
    if lo is not None:
        j = j[(j["utc"] >= lo) & (j["utc"] <= hi)]
    if not len(j):
        return {"bars_compared": 0}

    def within(x, y, tol):
        return (np.abs(x - y) / np.abs(y) * 100) <= tol

    same = (within(j.open_a, j.open_b, 0.01) & within(j.close_a, j.close_b, 0.01)
            & within(j.high_a, j.high_b, 0.02) & within(j.low_a, j.low_b, 0.02))
    bad = j.loc[~same, "utc"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d").value_counts().sort_index()
    return {"bars_compared": int(len(j)), "bars_matching": int(same.sum()), "match_rate_pct": round(100 * same.mean(), 3),
            "first_overlap_utc": str(j["utc"].min()), "last_overlap_utc": str(j["utc"].max()),
            "differing_days_et": {k: int(v) for k, v in bad.head(40).items()}, "differing_day_count": int(len(bad))}


def run_checks(events: list[dict], w: pd.DataFrame, info: dict, lt: pd.DataFrame, cal: dict, mgc: pd.DataFrame | None) -> dict:
    a, b, c = read_export(EXPORTS / F30_1), read_export(EXPORTS / F30_2), read_export(EXPORTS / F15)
    syn, _, _ = synth30(c)
    # C3: synthesized 15m vs native 30m (both files), on the overlap
    nat = pd.concat([a, b]).drop_duplicates("utc", keep="first")
    c3 = overlap_check(syn[["utc", "open", "high", "low", "close"]], nat[["utc", "open", "high", "low", "close"]])
    c3["passes_ge_99pct"] = bool(c3.get("match_rate_pct", 0) >= 99)
    # C4: file 1 vs file 2 on 2026-06-19 -> 2026-08-14 (Taipei)
    lo, hi = pd.Timestamp("2026-06-18T20:00:00Z"), pd.Timestamp("2026-08-14T20:00:00Z")
    c4 = overlap_check(a[["utc", "open", "high", "low", "close"]], b[["utc", "open", "high", "low", "close"]], lo, hi)
    c4["passes_ge_99pct"] = bool(c4.get("match_rate_pct", 0) >= 99)
    # C2: DST / Taipei label of bar 1
    c2_rows = []
    lbl = lt.set_index(["d", "hm"])["label"]
    for e in events:
        if e["status"] == "out_of_range":
            continue
        d = e["date"]
        dt = datetime(d.year, d.month, d.day, 14, 0, tzinfo=NY)
        expect = "02:00" if dt.dst() != timedelta(0) else "03:00"
        raw = lbl.get((d, "14:00"))
        got = None if raw is None else raw[11:16]
        near = []
        for n in range(-7, 8):
            t0 = datetime(d.year, d.month, d.day, 12, tzinfo=NY) + timedelta(days=n)
            t1 = t0 + timedelta(days=1)
            if t0.dst() != t1.dst():
                near.append(str((t0 + timedelta(days=1)).date()))
        c2_rows.append({"date": str(d.date()), "ny_dst": dt.dst() != timedelta(0), "expected_taipei_label_hour": expect,
                        "label_in_file": got, "match": (got == expect) if got else None,
                        "within_7_days_of_dst_change": bool(near)})
    c2_ok = all(r["match"] is not False for r in c2_rows) and info["duplicates_after_union"] == 0
    # C1
    in_sample = [e for e in events if e["status"] != "out_of_range"]
    judged = [e for e in in_sample if e["status"] == "judged"]
    need = math.ceil(0.9 * len(in_sample))
    usable_ev = [e for e in in_sample if e.get("usable_event_bars")]
    c1 = {"in_sample_events": len(in_sample), "events_with_usable_bars": len(usable_ev), "judged_events": len(judged),
          "threshold_ceil_0.9n": need, "passes": bool(len(judged) >= need),
          "excluded": [{"date": str(e["date"].date()), "reason": e["status"]} for e in in_sample if e["status"] != "judged"]}
    # C7 flat bars over all weekdays in 14:00-17:00 ET
    s = lt[lt["hm"].isin(SLOTS) & (lt["wd"] < 5)]
    c7 = {"bars_14_to_17_ET_weekdays": int(len(s)), "zero_range_bars": int((s.high <= s.low).sum())}
    c8 = None
    if mgc is not None:
        c8 = {"bars": int(len(mgc)), "duplicates": int(mgc["utc"].duplicated().sum()),
              "first_bar_utc": str(mgc["utc"].min()), "last_bar_utc": str(mgc["utc"].max())}
    fm = cal["fm"]
    c5 = {"events": len(cal["events"]), "statement_time_all_14_00": bool((cal["events"]["statement_time_et"] == "14:00").all()),
          "presconf_page_exists_all": bool((cal["events"]["presconf"] == "yes").all()),
          "presconf_start_explicit_14_30": int((cal["events"]["presconf_start_et"] == "14:30").sum()),
          "presconf_start_assumed_14_30_counted": int((cal["events"]["presconf_start_et"] == "14:30 assumed").sum()),
          "excluded_meetings_listed": [{"date": r["statement_date_et"], "why": r["note"] or r["scheduled"]}
                                       for _, r in fm.iterrows() if r["scheduled"] != "yes"]}
    c6 = {"meetings_checked": int((fm["fred_check"] != "n/a").sum()), "disagree": [r["statement_date_et"] for _, r in fm.iterrows()
                                                                                    if r["fred_check"] not in ("agree", "n/a")]}
    c6["passes"] = len(c6["disagree"]) == 0
    all_pass = bool(c1["passes"] and c2_ok and c3["passes_ge_99pct"] and c6["passes"])
    return {"C1_coverage": c1, "C2_timezone_dst": {"passes": c2_ok, "events": c2_rows, "duplicates_after_union": info["duplicates_after_union"]},
            "C3_bar_open_convention_15m_vs_30m": c3, "C4_file_overlap": c4, "C5_event_times": c5, "C6_calendar_vs_FRED": c6,
            "C7_flat_bars": c7, "C8_mgc_source": c8, "all_blocking_checks_pass": all_pass}


# ----------------------------------------------------------------- main analysis
def analyse(events: list[dict], m: pd.DataFrame, w: pd.DataFrame, lt: pd.DataFrame, cal: dict, mgc_pack: dict | None) -> dict:
    f = event_frame(events, m, w)
    n = len(f)
    res: dict = {"n_events": n}
    prim = primaries_from(f)
    res["primaries"] = prim
    # P3 three numbers
    R = math.exp(float(f["d"].mean()))
    C = math.exp(float(f["ctl_d"].mean()))
    res["P3_three_numbers"] = {"R_fomc_press_over_statement_geomean": R, "C_control_press_over_statement_geomean": C,
                               "X_R_over_C": R / C, "X_check_exp_mean_q": math.exp(float(f["q"].mean()))}
    res["descriptive"] = {
        "median_event_ratio_P1a": float(np.median(np.exp(f["y30"]))), "median_event_ratio_P1b": float(np.median(np.exp(f["y60"]))),
        "median_event_ratio_P3": float(np.median(np.exp(f["q"]))),
        "mean_points_first30_events": float(f["pts30"].mean()), "median_points_first30_events": float(f["pts30"].median()),
        "median_R30_pct_events": float(f["R30"].median()), "median_R60_pct_events": float(f["R60"].median()),
        "mean_abs_r60_events_pct": float(f["r60"].abs().mean()), "mean_r60_events_pct": float(f["r60"].mean()),
        "share_up_r60_events": float((f["r60"] > 0).mean()),
        "median_control_R30_pct_pooled_events": float(f["ctl_median_R30"].median())}
    # Wilcoxon
    wil = {}
    for k, col in (("P1a", "y30"), ("P1b", "y60"), ("P2", "z"), ("P3", "q")):
        wr = stats.wilcoxon(f[col].values)
        wil[k] = {"statistic": float(wr.statistic), "p": float(wr.pvalue),
                  "agrees_with_t_verdict_at_0.05": bool((wr.pvalue < 0.05) == (prim[k]["p"] < 0.05))}
    res["sensitivity_wilcoxon"] = wil
    # leave-one-event-out
    loo = {k: {"verdict_full": prim[k]["cleared"], "flips": []} for k in prim}
    for i in range(n):
        g = f.drop(index=f.index[i])
        pb = primaries_from(g)
        for k in prim:
            if pb[k]["cleared"] != prim[k]["cleared"]:
                loo[k]["flips"].append(str(f.loc[f.index[i], "date"].date()))
    for k in loo:
        loo[k]["n_flips"] = len(loo[k]["flips"])
    res["sensitivity_leave_one_out"] = loo
    # without same-day CPI/NFP
    g = f[f["same_day_release"] == ""]
    if len(g) >= 3:
        pb = primaries_from(g)
        res["sensitivity_without_same_day_release"] = {"n": len(g), "excluded": [str(d.date()) for d in f.loc[f["same_day_release"] != "", "date"]],
                                                       **{k: {"mean": pb[k]["mean"], "multiple": pb[k].get("multiple"), "cleared": pb[k]["cleared"],
                                                              "bound_first_step": pb[k]["bound_first_step"]} for k in pb}}
    # global baseline, CPI-study formulas (all usable weekdays after the same exclusions, no +-10 window)
    first_d, last_d = w.index.min(), w.index.max()
    base_days = [d for d in m.index if excluded_control(d, cal) is None and first_d <= d <= last_d]
    bm = m.loc[base_days]
    res["sensitivity_global_baseline_CPI_formulas"] = {
        "baseline_days": len(base_days), "events_n": n,
        "R30_points_median_events": float(f["pts30"].median()), "R30_points_median_baseline": float(bm["pts30"].median()),
        "ratio_points_median": float(f["pts30"].median() / bm["pts30"].median()),
        "R30_pct_median_events": float(f["R30"].median()), "R30_pct_median_baseline": float(bm["R30"].median()),
        "ratio_pct_median": float(f["R30"].median() / bm["R30"].median()),
        "mean_abs_r60_events_pct": float(f["r60"].abs().mean()), "mean_abs_r60_baseline_pct": float(bm["r60"].abs().mean()),
        "ratio_mean_abs_r60": float(f["r60"].abs().mean() / bm["r60"].abs().mean()),
        "cpi_study_reference": {"points_median_ratio": 2.18, "pct_median_ratio": 2.51, "abs_move_1h_ratio": 2.22,
                                "source": "RS-XAUUSD-20261102-002 README (n=33, different denominator: 08:30 ET bars)"},
        "note": "descriptive; may be said as 'same formula as the CPI study', never as a test (amendment A9)"}
    # Post-hoc descriptive additions (written after the first results were seen; every one is labelled
    # 事後補算 in the README and never changes a verdict)
    k_up, n_up = int((f["r60"] > 0).sum()), n
    wil = stats.binomtest(k_up, n_up).proportion_ci(method="wilson")
    ev_start = cal["events"].set_index("date")["presconf_start_et"]
    f_assumed = f["date"].map(lambda d: ev_start[d] == "14:30 assumed").astype(bool)

    def sub_x(g: pd.DataFrame) -> dict:
        q = g["q"].values
        nn = len(q)
        se = q.std(ddof=1) / math.sqrt(nn)
        tc = float(stats.t.ppf(1 - 0.025 / 2, nn - 1))    # the Holm level P3 was judged at (0.025)
        return {"n": nn, "X": math.exp(float(q.mean())), "interval_holm_level_0.025": [math.exp(float(q.mean()) - tc * se), math.exp(float(q.mean()) + tc * se)],
                "p": float(stats.ttest_1samp(q, 0).pvalue)}
    wt = stats.ttest_ind(f.loc[f_assumed, "q"], f.loc[~f_assumed, "q"], equal_var=False)
    b0830 = lt[(lt["hm"] == "08:30") & (lt["wd"] < 5)].set_index("d")
    b0830 = b0830[b0830["high"] > b0830["low"]].copy()
    b0830["R30"] = (b0830["high"] - b0830["low"]) / b0830["open"] * 100
    ord0830 = [d for d in b0830.index if first_d <= d <= last_d and excluded_control(d, cal) is None]
    cpi0830 = [d for d in sorted(cal["cpi"]) if d in b0830.index and first_d <= d <= last_d]
    res["post_hoc_descriptive"] = {
        "label": "事後補算: written after the results were seen; descriptive; no verdict depends on it",
        "r60_up_share": {"k": k_up, "n": n_up, "share": k_up / n_up, "wilson95": [float(wil.low), float(wil.high)],
                         "p_vs_half_binomial": float(stats.binomtest(k_up, n_up).pvalue)},
        "P3_by_press_conference_start": {
            "start_stated_on_fed_calendar": sub_x(f[~f_assumed]), "start_assumed_14_30": sub_x(f[f_assumed]),
            "difference_welch_p": float(wt.pvalue),
            "note": "the 13 pre-2025-09 start times were re-checked against federalreserve.gov pages on 2026-11-04 (none states a start time); "
                    "the assumed group is the 2024-01 to 2025-07 meetings, the stated group 2025-09 to 2026-09"},
        "denominators_for_CPI_comparison": {
            "ordinary_weekday_1400_ET_R30_pct_median": float(m.loc[base_days, "R30"].median()), "ordinary_1400_days": len(base_days),
            "ordinary_weekday_0830_ET_R30_pct_median": float(b0830.loc[ord0830, "R30"].median()), "ordinary_0830_days": len(ord0830),
            "FOMC_bar1_R30_pct_median": float(f["R30"].median()), "FOMC_n": n,
            "CPI_0830_bar_R30_pct_median": float(b0830.loc[cpi0830, "R30"].median()), "CPI_n": len(cpi0830),
            "ratio_1400_over_0830_ordinary": float(m.loc[base_days, "R30"].median() / b0830.loc[ord0830, "R30"].median()),
            "ratio_FOMC_over_CPI_actual_median": float(f["R30"].median() / b0830.loc[cpi0830, "R30"].median()),
            "note": "ordinary weekdays here use this study's exclusions; the CPI study reports 0.305% for its own 666 other weekdays"}}
    # MGC
    if mgc_pack is not None:
        fm_ = event_frame(mgc_pack["events"], mgc_pack["m"], mgc_pack["w"])
        if len(fm_) >= 5:
            pm = primaries_from(fm_)
            res["sensitivity_mgc"] = {"n": len(fm_), "dates": [str(d.date()) for d in fm_["date"]],
                                      "agreement_with_primary": {k: (pm[k]["cleared"] == prim[k]["cleared"]) for k in pm},
                                      **{k: {"mean": pm[k]["mean"], "multiple": pm[k].get("multiple"), "cleared": pm[k]["cleared"],
                                             "bound_first_step": pm[k]["bound_first_step"], "p": pm[k]["p"]} for k in pm}}
        # same events on FX_IDC for apples-to-apples
        same = f[f["date"].isin(fm_["date"])]
        if len(same) >= 5:
            ps = primaries_from(same)
            res["sensitivity_mgc"]["fx_idc_same_events"] = {"n": len(same), **{k: {"mean": ps[k]["mean"], "multiple": ps[k].get("multiple"),
                                                                                 "cleared": ps[k]["cleared"]} for k in ps}}
    # ------------------------------------------------------------- secondary
    sec: dict = {}
    planned = []

    def add(name: str, label: str, o: dict, kind: str = "mean") -> None:
        o["label"] = label
        sec[name] = o
        planned.append(name)

    add("S1", "3-hour range multiple (bars 1-6)", one_sample_secondary(f["y180"].values, True))
    add("S2", "bar 3 vs bar 1 (P3 form)", one_sample_secondary(f["q3"].values, True))
    # S3 day before
    wk = weekday_index()
    pos = {d: i for i, d in enumerate(wk)}
    yb = []
    s3_dropped = []
    bar1 = lt[(lt["hm"] == "14:00") & (lt["wd"] < 5)].set_index("d")
    for e in events:
        if e["status"] != "judged":
            continue
        pd_ = wk[pos[e["date"]] - 1]
        if pd_ in bar1.index and bar1.loc[pd_, "high"] > bar1.loc[pd_, "low"]:
            r30 = (bar1.loc[pd_, "high"] - bar1.loc[pd_, "low"]) / bar1.loc[pd_, "open"] * 100
            ctl = e["control_L"] + e["control_R"]
            yb.append(math.log(r30) - float(m.loc[ctl, "lnR30"].mean()))
        else:
            s3_dropped.append(str(e["date"].date()))
    o = one_sample_secondary(np.array(yb), True)
    o["dropped_events"] = s3_dropped
    add("S3", "day before (t-1) R30 at 14:00 ET vs control", o)
    # S4 SEP vs non-SEP
    sep, non = f[f["sep"]], f[~f["sep"]]
    for nm, col, lab in (("S4a", "y30", "SEP minus non-SEP on y (P1a)"), ("S4b", "z", "SEP minus non-SEP on z (P2)")):
        if len(sep) >= MIN_GROUP and len(non) >= MIN_GROUP:
            add(nm, lab, welch(sep[col].values, non[col].values))
        else:
            add(nm, lab, {"judged": False, "reason": "sample too small, not judged", "n_a": len(sep), "n_b": len(non)})
    # S5 decision
    hold = f[f["decision"] == "hold"]
    for nm, g, lab in (("S5a", f[f["decision"] == "cut"], "cut minus hold on z"), ("S5b", f[f["decision"] == "hike"], "hike minus hold on z")):
        if len(g) >= MIN_GROUP and len(hold) >= MIN_GROUP:
            add(nm, lab, welch(g["z"].values, hold["z"].values))
        else:
            add(nm, lab, {"judged": False, "reason": "sample too small, not judged", "n_a": len(g), "n_b": len(hold),
                          "mean_z_a": (float(g["z"].mean()) if len(g) else None), "mean_z_hold": float(hold["z"].mean())})
    # S6 dot direction
    dots = pd.read_csv(MACRO / "fomc_sep_dots.csv", dtype={"direction": str}, keep_default_na=False)
    dots["date"] = pd.to_datetime(dots["statement_date_et"])
    fz = f.merge(dots[["date", "direction"]], on="date", how="left")
    dov, haw = fz[fz["direction"] == "lower"], fz[fz["direction"] == "higher"]
    unch = fz[fz["direction"] == "unchanged"]
    if len(dov) >= MIN_GROUP and len(haw) >= MIN_GROUP:
        add("S6", "dovish (lower dot) minus hawkish (higher dot) on z", welch(dov["z"].values, haw["z"].values))
    else:
        add("S6", "dovish minus hawkish on z", {"judged": False, "reason": "sample too small, not judged", "n_dovish": len(dov), "n_hawkish": len(haw),
                                               "n_unchanged": len(unch), "mean_z_dovish": float(dov["z"].mean()) if len(dov) else None,
                                               "mean_z_hawkish": float(haw["z"].mean()) if len(haw) else None})
    # S7 descriptive
    cons = pd.read_csv(MACRO / "fomc_consensus.csv", keep_default_na=False)
    res["S7_surprise_descriptive"] = {"classes": {k: int(v) for k, v in cons["class"].value_counts().items()},
                                      "note": "no pre-meeting preview article was searched or coded in this run: every row is `not_searched` (not `unknown`, which would mean "
                                              "searched and not found) and none is filled from memory; the study cannot say how gold moves on a surprise (amendment A7/A9)"}
    # S8
    add("S8a", "bar 1 signed return minus control (pp)", one_sample_secondary(f["s8a"].values, False))
    add("S8b", "bar 2 signed return minus control (pp)", one_sample_secondary(f["s8b"].values, False))
    ctl_days = sorted({d for e in events if e["status"] == "judged" for d in e["control_L"] + e["control_R"]})
    cm = m.loc[ctl_days]
    ctl_rev = int((np.sign(cm["r_b1"]) * np.sign(cm["r_b2"]) < 0).sum())
    add("S8c", "reversal share (bar 2 sign != bar 1 sign): FOMC vs unique control days", two_prop(int(f["rev"].sum()), n, ctl_rev, len(cm)))
    # S9 FOMC vs CPI
    cpi_idx = []
    b0830 = lt[(lt["hm"] == "08:30") & (lt["wd"] < 5)].set_index("d")
    for d in sorted(cal["cpi"]):
        if d in b0830.index and first_d <= d <= last_d and b0830.loc[d, "high"] > b0830.loc[d, "low"]:
            cpi_idx.append(math.log((b0830.loc[d, "high"] - b0830.loc[d, "low"]) / b0830.loc[d, "open"] * 100))
    wv = welch(f["lnR30"].values, np.array(cpi_idx))
    wv["multiple_fomc_over_cpi"] = math.exp(wv["diff"])
    wv["note"] = "different denominators: FOMC 14:00 ET bar vs ordinary 14:00 ET bars; CPI 08:30 ET bar vs ordinary 08:30 ET bars"
    add("S9", "ln R30: FOMC bar 1 vs CPI 08:30 ET bar", wv)
    res["secondary"] = sec
    res["event_table"] = [
        {"date": str(r["date"].date()), "sep": bool(r["sep"]), "decision": r["decision"], "same_day_release": r["same_day_release"],
         "n_control": int(r["n_ctl"]), "R30_pct": round(r["R30"], 4), "pts30": round(r["pts30"], 2),
         "multiple_30m": round(math.exp(r["y30"]), 3), "multiple_1h": round(math.exp(r["y60"]), 3),
         "r60_pct": round(r["r60"], 4), "z_pp": round(r["z"], 4), "ratio_press_over_statement": round(math.exp(r["d"]), 3),
         "control_ratio_press_over_statement": round(math.exp(r["ctl_d"]), 3), "q_multiple": round(math.exp(r["q"]), 3)}
        for _, r in f.iterrows()]
    return res, f, planned


def load_mgc() -> pd.DataFrame:
    return read_export(EXPORTS / MGC30)


def pipeline_for(bars: pd.DataFrame, cal: dict, metrics: bool = True) -> dict:
    lt = long_table(bars)
    w = day_wide(lt)
    m = day_metrics(w) if metrics else None   # --checks-only never computes a range or return
    first_date = lt["d"].min()
    last_date = lt["d"].max()
    events = build_events(w, cal, first_date, last_date)
    return {"lt": lt, "w": w, "m": m, "events": events, "first": first_date, "last": last_date}


def fmt_events(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        out.append({"date": str(e["date"].date()), "sep": e["sep"], "decision": e["decision"],
                    "same_day_release": e["same_day_release"], "status": e["status"],
                    "control_left": len(e.get("control_L", [])), "control_right": len(e.get("control_R", [])),
                    "control_dropped_reasons": e.get("control_dropped_reasons", {}), "presconf_start": e["presconf_start"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks-only", action="store_true")
    a = ap.parse_args()
    cal = load_calendars()
    bars, info = build_union()
    P = pipeline_for(bars, cal, metrics=not a.checks_only)
    mgc = load_mgc()
    checks = run_checks(P["events"], P["w"], info, P["lt"], cal, mgc)
    pre = {"schema_version": 1, "study_id": STUDY_ID, "stage": "pre-run checks (no return, range or direction computed)",
           "generated_at": datetime.now(TAIPEI_TZ).isoformat(timespec="seconds"), "data_freeze": FREEZE,
           "bar_union": info, "checks": checks, "events": fmt_events(P["events"])}
    (OUT / "precheck.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"all_blocking_checks_pass": checks["all_blocking_checks_pass"],
                      "C1": {k: checks["C1_coverage"][k] for k in ("in_sample_events", "judged_events", "threshold_ceil_0.9n", "passes")},
                      "C2": checks["C2_timezone_dst"]["passes"], "C3": checks["C3_bar_open_convention_15m_vs_30m"],
                      "C4": {k: v for k, v in checks["C4_file_overlap"].items() if k != "differing_days_et"},
                      "C6": checks["C6_calendar_vs_FRED"], "C7": checks["C7_flat_bars"], "C8": checks["C8_mgc_source"]},
                     ensure_ascii=False, indent=1, default=str))
    if a.checks_only:
        return
    if not checks["all_blocking_checks_pass"]:
        sys.exit("pre-run checks failed: stop and report (see precheck.json)")
    mp = pipeline_for(mgc, cal)
    mgc_pack = {"events": mp["events"], "m": mp["m"], "w": mp["w"]}
    res, f, planned = analyse(P["events"], P["m"], P["w"], P["lt"], cal, mgc_pack)
    # BH over judged secondaries
    pvals = {k: v["p"] for k, v in res["secondary"].items() if "p" in v}
    passed = bh(pvals) if pvals else {}
    for k, v in res["secondary"].items():
        v["bh_q010_pass"] = passed.get(k) if k in passed else None
    cleared_unc = sum(1 for v in res["secondary"].values() if v.get("resolved_uncorrected"))
    res["secondary_summary"] = {"planned_family_size": 16, "planned_tests_in_this_run": sorted(res["secondary"]),
                                "judged": len(pvals), "cleared_uncorrected": cleared_unc,
                                "expected_by_chance_about_5pct": round(0.05 * len(pvals), 2),
                                "bh_q010_passing": sorted(k for k, v in passed.items() if v),
                                "not_run": ["S10a", "S10b", "S10c", "S10d (daily layer 2008-2026)"],
                                "s10_status": "formally dropped (decision 2026-11-04, taken after the intraday results were seen); no S10 number is reported",
                                "daily_2013_sensitivity_status": "formally dropped (same decision); not run",
                                "not_judged_or_not_run": 7,
                                "bh_robustness_unjudged_as_p1": {
                                    f"m={mm}": sorted(k for k, v in bh({**pvals, **{f"unjudged_{j}": 1.0 for j in range(mm - len(pvals))}}).items() if v and not k.startswith("unjudged_"))
                                    for mm in (13, 16)}}
    inputs = {"fx_idc_30m_1": sha(EXPORTS / F30_1) | {"locator": rel(EXPORTS / F30_1)},
              "fx_idc_30m_2": sha(EXPORTS / F30_2) | {"locator": rel(EXPORTS / F30_2)},
              "fx_idc_15m": sha(EXPORTS / F15) | {"locator": rel(EXPORTS / F15)},
              "mgc_30m": sha(EXPORTS / MGC30) | {"locator": rel(EXPORTS / MGC30)}}
    for nm in ("fomc_meetings.csv", "fed_beige_book_dates.csv", "fomc_sep_dots.csv", "fomc_consensus.csv", "cpi_release_dates.csv", "nfp_release_dates.csv"):
        inputs[nm] = sha(MACRO / nm) | {"locator": rel(MACRO / nm)}
    payload = {
        "schema_version": 1, "study_id": STUDY_ID, "revision": 4, "status": "computed",
        "generated_at": datetime.now(TAIPEI_TZ).isoformat(timespec="seconds"), "data_freeze": FREEZE,
        "method_parameters": {
            "bars": "FX_IDC:XAUUSD 30m bars opening 14:00..16:30 America/New_York; bar 1 statement, bar 2 first press-conference bar",
            "control_offsets_weekdays": "-10..-2 and +2..+10 (Mon-Fri counting; holidays occupy a slot)",
            "min_control": f">= {MIN_TOTAL} total and >= {MIN_SIDE} per side", "holm": "4 primaries, family alpha 0.05, two-sided t, df n-1",
            "secondary_bound": "t(0.975, n-1) when n < 30 (Welch df for two-group tests) else 1.96, times SE; uncorrected",
            "sample": "scheduled meetings 2024-01 .. 2026-10-28 (amendment 2: existing data, no 2019 rule)"},
        "inputs": inputs, "coverage": {"bar_union": info, "events": fmt_events(P["events"]), "n_judged": res["n_events"]},
        "baseline": "each event's own control days: ordinary weekdays at offsets -10..-2 and +2..+10 (Monday-Friday counting; the days t-1 and t+1 are not used) at the same ET bars, after the exclusions; 11-16 control days per judged meeting", "prechecks": "precheck.json",
        "results": res, "limitations": [
            "Descriptive event study; no trading rule is implied.",
            "Small n: 21 judged meetings, 2024-2026 only. C1 is met exactly (21 of 23): one fewer usable meeting would have stopped the study under the registered rule. S5, S6 splits are below the 8-event rule and are not judged.",
            "S7 surprise: no preview article was searched; every row of the consensus file is `not_searched`. The study cannot say how gold moves on a surprise.",
            "Secondary family: 16 registered, 9 judged, 2 pass BH (S1, S3). S5a, S5b, S6 are below the 8-meeting rule (registered). S10a-d (daily layer 2008-2026) and the FX_IDC daily sensitivity from 2013 were not run and are formally dropped (decision of 2026-11-04, taken after the intraday results were seen); no S10 number is reported. Counting the 7 unjudged tests as p = 1 (m = 16), S1 and S3 still pass.",
            "S1 (3-hour range, bars 1-6) contains the two bars of P1b, so it is not independent evidence.",
            "13 of 23 meetings (12 of the 21 judged): the 14:30 ET press-conference start is assumed (the Fed monthly calendar states it explicitly only from 2025-09-17). On 2026-11-04 the Fed's press-conference page, monthly calendar page, statement release and transcript PDF of each of the 13 were opened: none states a start time, so none is confirmed and none contradicts 14:30. Post-hoc: P3 X = 1.52 on the 12 assumed meetings and 1.15 on the 9 with a stated start (not distinguishable at n = 9); the two groups do not differ detectably (Welch p = 0.12).",
            "Other 14:00-17:00 ET events (Fed speakers etc.) are not excluded from control days.",
            "Dollar-point figures depend on the gold price level (2024-2026); compare the percent figures.",
            "FX_IDC is a spot composite; MGC is the COMEX source sensitivity."]}
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_report(payload)
    print("results.json, README.md and study.json written; n =", res["n_events"])


# ----------------------------------------------------------------- README and study.json
def x2(v: float) -> str:
    return f"{v:.2f}"


def write_report(payload: dict) -> None:
    r = payload["results"]
    P = r["primaries"]
    n = r["n_events"]
    cov = payload["coverage"]["events"]
    inn = [e for e in cov if e["status"] != "out_of_range"]
    excl = [e for e in inn if e["status"] != "judged"]
    t3 = r["P3_three_numbers"]
    g = r["sensitivity_global_baseline_CPI_formulas"]
    d = r["descriptive"]
    sec = r["secondary"]
    ss = r["secondary_summary"]
    a, b, c, p3 = P["P1a"], P["P1b"], P["P2"], P["P3"]

    def itv(o: dict, key: str = "multiple_interval_holm") -> str:
        lo, hi = o[key]
        return f"{x2(lo)}–{x2(hi)}"

    def verdict(o: dict) -> str:
        return "分得出" if o["cleared"] else "分不出"

    fragile = {k: r["sensitivity_leave_one_out"][k]["n_flips"] for k in P}
    mgc = r.get("sensitivity_mgc", {})
    ph = r["post_hoc_descriptive"]
    pb = ph["P3_by_press_conference_start"]
    dn = ph["denominators_for_CPI_comparison"]
    excl_txt = "、".join(e["date"] for e in excl)
    claims = {
        "P1": (f"{n} 次 FOMC（排程會議 {len(inn)} 次，{len(excl)} 次因對照日不足排除：{excl_txt}），公布後 30 分鐘黃金高低差一般是前後兩週平常日同一時段的 "
               f"{x2(a['multiple'])} 倍（界線 {itv(a)} 倍；{verdict(a)}）；1 小時 {x2(b['multiple'])} 倍（界線 {itv(b)} 倍；{verdict(b)}）。"
               f"倍數是幾何平均；中位數是 {x2(d['median_event_ratio_P1a'])} 倍與 {x2(d['median_event_ratio_P1b'])} 倍。"
               f"CPI 研究的 2.2 倍用的是 08:30 的平常日，分母不同，不能直接比大小。"),
        "CPI_same_formula": (f"同 CPI 研究算法（全部平常日當分母）：第一根 30 分鐘高低差中位數 {g['R30_points_median_events']:.1f} 點對 {g['R30_points_median_baseline']:.2f} 點"
                             f"（{x2(g['ratio_points_median'])} 倍）；佔價格百分比的中位數 {g['R30_pct_median_events']:.3f}% 對 {g['R30_pct_median_baseline']:.3f}%（{x2(g['ratio_pct_median'])} 倍）；"
                             f"1 小時絕對漲跌平均 {g['mean_abs_r60_events_pct']:.3f}% 對 {g['mean_abs_r60_baseline_pct']:.3f}%（{x2(g['ratio_mean_abs_r60'])} 倍）。"
                             f"CPI 研究對應數字是 2.18、2.51、2.22 倍。這是描述，不是檢定。"
                             f"倍數不能直接比大小，因為分母不同：平常日下午兩點那根的中位數 {dn['ordinary_weekday_1400_ET_R30_pct_median']:.3f}%，約是平常日早上八點半那根 {dn['ordinary_weekday_0830_ET_R30_pct_median']:.3f}% 的一半（比值 {dn['ratio_1400_over_0830_ordinary']:.2f}）。"
                             f"實際幅度的中位數：FOMC 公布那根 {dn['FOMC_bar1_R30_pct_median']:.2f}%（{dn['FOMC_n']} 次），CPI 公布那根 {dn['CPI_0830_bar_R30_pct_median']:.2f}%（{dn['CPI_n']} 次）；S9 比較兩者，分不出誰大（未校正）。"),
        "P2": (f"公布後 1 小時的方向：平均與平常日相差 {c['mean']:+.2f} 個百分點（負值＝比平常日更偏跌），界線 ±{c['bound_at_holm_level']:.2f}（Holm 水準）／±{c['bound_first_step']:.2f}（首步水準），{n} 次；"
               f"{verdict(c)}。{n} 次中 {ph['r60_up_share']['k']} 次 1 小時收漲（95% 區間約 {ph['r60_up_share']['wilson95'][0]*100:.0f}–{ph['r60_up_share']['wilson95'][1]*100:.0f}%，分不出偏向；事後補算，平均 {d['mean_r60_events_pct']:+.3f}%）。這是平均，不是每次。"),
        "P3": (f"檢定的是 X：「記者會那根（14:30–15:00）÷ 公布那根（14:00–14:30）」這個比值，FOMC 當天比平常日放大 {x2(t3['X_R_over_C'])} 倍（界線 {itv(p3)}；{verdict(p3)}）。"
               f"三個數字一起看：FOMC 當天這個比值 R = {x2(t3['R_fomc_press_over_statement_geomean'])} 倍，平常日 C = {x2(t3['C_control_press_over_statement_geomean'])} 倍，X = R ÷ C = {x2(t3['X_R_over_C'])}。"
               + ("R 只是描述，本身沒有檢定；R 大於 1，可加一句「記者會那半小時比較大」，不當結論。" if t3["R_fomc_press_over_statement_geomean"] > 1 and p3["cleared"] else "R 本身沒有檢定。")),
        "P3_post_hoc": (f"事後補算（不改判定）：記者會開始時間有官方行事曆標示的 {pb['start_stated_on_fed_calendar']['n']} 次，X = {x2(pb['start_stated_on_fed_calendar']['X'])}（界線 {x2(pb['start_stated_on_fed_calendar']['interval_holm_level_0.025'][0])}–{x2(pb['start_stated_on_fed_calendar']['interval_holm_level_0.025'][1])}，分不出）；"
                        f"依慣例推定 14:30 的 {pb['start_assumed_14_30']['n']} 次，X = {x2(pb['start_assumed_14_30']['X'])}（界線 {x2(pb['start_assumed_14_30']['interval_holm_level_0.025'][0])}–{x2(pb['start_assumed_14_30']['interval_holm_level_0.025'][1])}）；兩組彼此分不出差別（p = {pb['difference_welch_p']:.2f}）。"
                        f"推定的那 {pb['start_assumed_14_30']['n']} 次已對聯準會官方頁面再查一次，頁面都沒有寫開始時間，所以沒有任何一次被確認，也沒有任何一次與 14:30 矛盾。"),
        "about_12_10": f"過去 {n} 次的平均是上面的數字；這不是對 12/10 這一次的預測。",
        "surprise": "2024 年以來的樣本裡沒有讀會前預期，這份研究回答不了「意外時黃金怎麼動」。",
    }
    bhs = ", ".join(f"{k} ({sec[k]['label']}: {x2(sec[k]['multiple'])}x, bound {sec[k]['multiple_interval'][0]:.2f}-{sec[k]['multiple_interval'][1]:.2f}, n={sec[k]['n']})" for k in ss["bh_q010_passing"])
    payload["claims_zh"] = claims
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        f"# {STUDY_ID} — gold around FOMC rate decisions (results, computed {payload['generated_at'][:10]}, data freeze {FREEZE})",
        "",
        "Status: `progress` — results written; Pre-registration and the two design amendments (both made before any result; the second fixes the sample to the existing data) govern. "
        "Pre-run checks C1–C8 passed before this run (`precheck.json`). Descriptive event study; not trading advice; no `policy_impacts`.",
        "",
        "## What was measured",
        "",
        f"- **Events.** {len(inn)} scheduled FOMC meetings from 2024-01-31 to 2026-10-28 (Fed calendar, `fomc_meetings.csv`; statement 14:00 ET at all {len(inn)}, press conference page present at all {len(inn)}; "
        f"the 14:30 ET start is stated on the Fed calendar for 10 and assumed for 13 per check C5; the 13 were re-checked on 2026-11-04 against federalreserve.gov pages and none states a start time, see Limitations). The 2025-08-22 notation vote is not an event. **{n} judged**, {len(excl)} excluded by the pre-registered control-day rule: "
        + "; ".join(f"{e['date']} (left {e['control_left']}, right {e['control_right']} control days; need ≥ 8 total and ≥ 4 per side)" for e in excl)
        + ". C1: usable events ≥ ceil(0.9 × 23) = 21, met exactly.",
        "- **Bars.** FX_IDC:XAUUSD 30m TradingView exports (union of `FX_IDC_XAUUSD, 30.csv`, `30_to_2026-10-03.csv`, and 30m synthesized from the 15m export only where both 15m bars exist; 33,484 bars, 2024-01-01T23:00Z to 2026-10-30T20:30Z). "
        "Bar 1 = 14:00–14:30 ET (statement), bar 2 = 14:30–15:00 ET; converted with `zoneinfo` (Taipei label 02:00 in US daylight time, 03:00 in standard time; C2 passed).",
        "- **Control days.** For each meeting, the same ET bars on the ordinary Monday–Friday days at offsets −10..−2 and +2..+10, excluding days without six contiguous non-flat bars, 12-24 to 01-01, the Friday after Thanksgiving, other FOMC statement days, FOMC minutes, Beige Book, CPI and NFP days. "
        f"{min(e['n_control'] for e in r['event_table'])}–{max(e['n_control'] for e in r['event_table'])} control days per judged meeting.",
        "- **Measures.** R30 = (high − low) / open × 100 of bar 1; R60 = range of bars 1–2 over bar 1 open; r60 = (close of bar 2 / open of bar 1 − 1) × 100; press/statement ratio = range% of bar 2 ÷ range% of bar 1. "
        "Per meeting, each log measure minus the mean of the same log measure over its own control days; unit = meeting; t-test with n − 1 degrees of freedom.",
        "- **Verdict rule.** Holm step-down over the 4 primaries, family α = 0.05, two-sided (levels 0.0125 / 0.0167 / 0.025 / 0.05). The printed bound uses the Holm level applied; the conservative first-step bound t(0.99375, n−1) × SE is printed as well.",
        "",
        "## Primary results (n = %d meetings; each meeting against its own control days)" % n,
        "",
        "| primary | estimate | bound (Holm level / first step) | interval (Holm level) | p | verdict |",
        "|---|---|---|---|---|---|",
        f"| P1a 30-min range | {x2(a['multiple'])}× (mean log {a['mean']:.3f}, sd {a['sd']:.3f}) | ±{a['bound_at_holm_level']:.3f} / ±{a['bound_first_step']:.3f} log | {itv(a)}× | {a['p']:.2g} | {verdict(a)} |",
        f"| P1b 1-hour range | {x2(b['multiple'])}× (mean log {b['mean']:.3f}, sd {b['sd']:.3f}) | ±{b['bound_at_holm_level']:.3f} / ±{b['bound_first_step']:.3f} log | {itv(b)}× | {b['p']:.2g} | {verdict(b)} |",
        f"| P2 1-hour direction (pp, minus control) | {c['mean']:+.3f} pp (sd {c['sd']:.3f}) | ±{c['bound_at_holm_level']:.3f} / ±{c['bound_first_step']:.3f} pp | {c['mean']-c['bound_at_holm_level']:+.2f} to {c['mean']+c['bound_at_holm_level']:+.2f} pp | {c['p']:.2g} | {verdict(c)} |",
        f"| P3 (tested quantity X) press bar ÷ statement bar, relative to control | {x2(p3['multiple'])}× (mean log {p3['mean']:.3f}, sd {p3['sd']:.3f}) | ±{p3['bound_at_holm_level']:.3f} / ±{p3['bound_first_step']:.3f} log | {itv(p3)}× | {p3['p']:.2g} | {verdict(p3)} |",
        "",
        f"- **P3 three numbers.** The tested quantity is X = R ÷ C = {x2(t3['X_R_over_C'])} (interval {itv(p3)}, p = {p3['p']:.2g}). R (FOMC days, geometric mean of press-bar range ÷ statement-bar range) = {x2(t3['R_fomc_press_over_statement_geomean'])} and C (same ratio on control days) = {x2(t3['C_control_press_over_statement_geomean'])} are descriptive and are not tested on their own; state all three together. Multiples are geometric means; the median per-meeting values are P1a {x2(d['median_event_ratio_P1a'])}×, P1b {x2(d['median_event_ratio_P1b'])}×, P3 {x2(d['median_event_ratio_P3'])}×.",
        f"- **分不出的範圍 at n = {n} (first-step level, around 1).** P1a {x2(a['not_distinguishable_range_first_step'][0])}–{x2(a['not_distinguishable_range_first_step'][1])}×; P1b {x2(b['not_distinguishable_range_first_step'][0])}–{x2(b['not_distinguishable_range_first_step'][1])}×; P2 ±{c['bound_first_step']:.2f} pp; P3 {x2(p3['not_distinguishable_range_first_step'][0])}–{x2(p3['not_distinguishable_range_first_step'][1])}×. The pre-registration's n = 23 guesses were P1 0.71–1.41×, P2 ±0.28 pp, P3 0.67–1.49× (sd assumptions 0.6 / 0.5 / 0.7, which were guesses); the observed sd were {a['sd']:.2f} / {c['sd']:.2f} / {p3['sd']:.2f}, so P2 resolves less finely than guessed.",
        f"- **P3 by press-conference start (事後補算, does not change the verdict).** Meetings with the 14:30 ET start stated on the Fed calendar (2025-09 to 2026-09): n = {pb['start_stated_on_fed_calendar']['n']}, X = {x2(pb['start_stated_on_fed_calendar']['X'])} (interval {x2(pb['start_stated_on_fed_calendar']['interval_holm_level_0.025'][0])}–{x2(pb['start_stated_on_fed_calendar']['interval_holm_level_0.025'][1])} at the Holm 0.025 level, p = {pb['start_stated_on_fed_calendar']['p']:.2f}; not distinguishable, and n = 9 has little power). Meetings with the start assumed 14:30 (2024-01 to 2025-07): n = {pb['start_assumed_14_30']['n']}, X = {x2(pb['start_assumed_14_30']['X'])} (interval {x2(pb['start_assumed_14_30']['interval_holm_level_0.025'][0])}–{x2(pb['start_assumed_14_30']['interval_holm_level_0.025'][1])}, p = {pb['start_assumed_14_30']['p']:.3f}). The two groups do not differ detectably (Welch p = {pb['difference_welch_p']:.2f}). The P3 verdict therefore rests mainly on the assumed-time meetings; the assumption could not be confirmed (see Limitations).",
        f"- **Descriptive.** First-30-minute range on the {n} meetings: median {d['median_points_first30_events']:.1f} points (mean {d['mean_points_first30_events']:.1f}), median {d['median_R30_pct_events']:.3f}% of price, against a median {d['median_control_R30_pct_pooled_events']:.3f}% on the control days (point figures depend on the gold price level over 2024-2026; compare the % figures). 1-hour move: mean {d['mean_r60_events_pct']:+.3f}%, mean absolute {d['mean_abs_r60_events_pct']:.3f}%; closed up in {ph['r60_up_share']['k']} of {n} meetings (Wilson 95% interval {ph['r60_up_share']['wilson95'][0]*100:.0f}–{ph['r60_up_share']['wilson95'][1]*100:.0f}%, p = {ph['r60_up_share']['p_vs_half_binomial']:.2f} against one half; 事後補算; no lean either way can be told).",
        "",
        "## Comparison with the CPI study (same formula, descriptive only)",
        "",
        f"All usable weekdays in coverage as denominator ({g['baseline_days']} days, same exclusions, no ±10 window), the CPI study's formulas: first-30-minute range median {g['R30_points_median_events']:.1f} points vs {g['R30_points_median_baseline']:.2f} = **{x2(g['ratio_points_median'])}×** (CPI study 2.18×); median range in % of price {g['R30_pct_median_events']:.3f}% vs {g['R30_pct_median_baseline']:.3f}% = **{x2(g['ratio_pct_median'])}×** (CPI 2.51×); mean absolute 1-hour move {g['mean_abs_r60_events_pct']:.3f}% vs {g['mean_abs_r60_baseline_pct']:.3f}% = **{x2(g['ratio_mean_abs_r60'])}×** (CPI 2.22×). The CPI study compares 08:30 ET bars with ordinary 08:30 ET bars (n = 33); this one compares 14:00 ET bars. **The multiples are not comparable in size because the denominators differ (事後補算):** the median first-30-minute range on ordinary weekdays is {dn['ordinary_weekday_1400_ET_R30_pct_median']:.3f}% at 14:00 ET (n = {dn['ordinary_1400_days']}) against {dn['ordinary_weekday_0830_ET_R30_pct_median']:.3f}% at 08:30 ET (n = {dn['ordinary_0830_days']}; the CPI study reports 0.305% for its own 666 days), i.e. 14:00 is about half as active (ratio {dn['ratio_1400_over_0830_ordinary']:.2f}). The actual median ranges are {dn['FOMC_bar1_R30_pct_median']:.3f}% for the FOMC bar ({dn['FOMC_n']} meetings) and {dn['CPI_0830_bar_R30_pct_median']:.3f}% for the CPI bar ({dn['CPI_n']} releases); S9 compares them and cannot tell which is larger. Nothing here supports saying one release moves gold more than the other.",
        "",
        "## Sensitivities (never change a verdict)",
        "",
        f"- **Wilcoxon signed-rank:** P1a p = {r['sensitivity_wilcoxon']['P1a']['p']:.2g}, P1b p = {r['sensitivity_wilcoxon']['P1b']['p']:.2g}, P2 p = {r['sensitivity_wilcoxon']['P2']['p']:.2g}, P3 p = {r['sensitivity_wilcoxon']['P3']['p']:.2g}; agrees with the t-test side of 0.05 for all four: {all(v['agrees_with_t_verdict_at_0.05'] for v in r['sensitivity_wilcoxon'].values())}.",
        f"- **Leave-one-meeting-out (full Holm recomputed):** flips P1a {fragile['P1a']}, P1b {fragile['P1b']}, P2 {fragile['P2']}, P3 {fragile['P3']} of {n} removals.",
        f"- **Without the same-day CPI/NFP meeting** ({', '.join(r['sensitivity_without_same_day_release']['excluded'])}; n = {r['sensitivity_without_same_day_release']['n']}): P1a {x2(r['sensitivity_without_same_day_release']['P1a']['multiple'])}×, P1b {x2(r['sensitivity_without_same_day_release']['P1b']['multiple'])}×, P2 {r['sensitivity_without_same_day_release']['P2']['mean']:+.2f} pp, P3 {x2(r['sensitivity_without_same_day_release']['P3']['multiple'])}×; verdicts unchanged: {all(r['sensitivity_without_same_day_release'][k]['cleared'] == P[k]['cleared'] for k in P)}.",
        (f"- **COMEX MGC1! 30m** (n = {mgc['n']} meetings through 2026-07-29): P1a {x2(mgc['P1a']['multiple'])}×, P1b {x2(mgc['P1b']['multiple'])}×, P2 {mgc['P2']['mean']:+.2f} pp, P3 {x2(mgc['P3']['multiple'])}×; verdicts agree with FX_IDC: {all(mgc['agreement_with_primary'].values())}. On the same {mgc['fx_idc_same_events']['n']} meetings FX_IDC gives (事後補算, descriptive) P1a {x2(mgc['fx_idc_same_events']['P1a']['multiple'])}×, P1b {x2(mgc['fx_idc_same_events']['P1b']['multiple'])}×, P3 {x2(mgc['fx_idc_same_events']['P3']['multiple'])}×." if mgc else "- MGC: not computed."),
        "",
        "## Secondary analyses (uncorrected; labelled 次要、未校正)",
        "",
        f"Planned family 16; run here {len(ss['planned_tests_in_this_run'])} ({', '.join(ss['planned_tests_in_this_run'])}); judged {ss['judged']}; cleared uncorrected {ss['cleared_uncorrected']} (about {ss['expected_by_chance_about_5pct']} expected by chance); BH q = 0.10 passing: {', '.join(ss['bh_q010_passing']) or 'none'}. **Say it as: 16 registered, {ss['judged']} actually judged, {len(ss['bh_q010_passing'])} pass.** **Departure from the registration:** S10a–d (daily layer 2008–2026) were not run and, together with the FX_IDC daily sensitivity from 2013, are formally dropped (decision of 2026-11-04, taken after the intraday results were seen; no S10 number is reported or may be used). BH is therefore computed over the {ss['judged']} judged tests. Robustness: counting the unjudged or unrun tests as p = 1 (m = 13 or m = 16), the tests passing BH are still {', '.join(ss['bh_robustness_unjudged_as_p1']['m=16'])}. S1 (bars 1–6) contains the two bars of P1b and is not independent evidence. S5a (cut − hold: 5 vs 15 meetings), S5b (hike − hold: 1 vs 15) and S6 (dovish dots 2, hawkish 3, unchanged 5) are below the 8-meeting rule: 「樣本不足，不判定」.",
        "",
        "| test | what | estimate | bound (95%) | p | cleared (uncorrected) | BH q=0.10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ("S1", "S2", "S3", "S4a", "S4b", "S8a", "S8b", "S8c", "S9"):
        v = sec[k]
        if "multiple" in v and "diff" not in v:
            est = f"{x2(v['multiple'])}× (n={v['n']})"
            bd = f"{v['multiple_interval'][0]:.2f}–{v['multiple_interval'][1]:.2f}×"
        elif k == "S8c":
            est = f"{v['pct_event']}% (n={v['n_event']}) vs {v['pct_control']}% (n={v['n_control']})"
            bd = f"±{v['bound_95_pts']} pts"
        elif "diff" in v:
            est = f"diff {v['diff']:+.3f} ({v['n_a']} vs {v['n_b']})"
            bd = f"±{v['bound_95']:.3f}"
        else:
            est = f"{v['mean']:+.3f} pp (n={v['n']})"
            bd = f"±{v['bound_95']:.3f}"
        lines.append(f"| {k} | {v['label']} | {est} | {bd} | {v['p']:.2g} | {'yes' if v['resolved_uncorrected'] else 'no'} | {'yes' if v['bh_q010_pass'] else 'no'} |")
    lines += [
        "",
        f"- S9: FOMC bar 1 is {x2(sec['S9']['multiple_fomc_over_cpi'])}× the CPI 08:30 ET bar in geometric-mean range (n = {sec['S9']['n_a']} vs {sec['S9']['n_b']}; not resolved); denominators differ (see above).",
        f"- S5a mean z: cut {sec['S5a']['mean_z_a']:+.2f} pp (n = {sec['S5a']['n_a']}), hold {sec['S5a']['mean_z_hold']:+.2f} pp (n = {sec['S5a']['n_b']}); S5b hike {sec['S5b']['mean_z_a']:+.2f} pp (n = {sec['S5b']['n_a']}). Descriptive only.",
        "- S7 surprise vs pre-meeting consensus: no preview article was searched or coded; all 23 rows of `fomc_consensus.csv` are `not_searched` (a status of `unknown` would wrongly mean searched and not found). Not judged, descriptive only.",
        "",
        "## 可以說 / 不可以說（照說法對照表）",
        "",
        f"- **P1.** {claims['P1']}",
        f"- **同 CPI 算法.** {claims['CPI_same_formula']}",
        f"- **P2.** {claims['P2']}",
        f"- **P3.** {claims['P3']}",
        f"- **P3 事後補算.** {claims['P3_post_hoc']}",
        f"- **12/10.** {claims['about_12_10']}",
        f"- **意外決議.** {claims['surprise']}",
        f"- **脆弱度.** 逐次拿掉一次會議，四項結論都沒有翻轉；Wilcoxon、去掉同日 CPI 的 2024-06-12、換 COMEX 資料，結論都一致，不需加「靠少數幾次會議」。" if sum(fragile.values()) == 0 else f"- **脆弱度.** 有翻轉：{fragile}；已清結論要加「這個結果靠少數幾次會議」。",
        f"- **次要（未校正、可能是巧合；登記 16 項、實際判定 {ss['judged']} 項、{len(ss['bh_q010_passing'])} 項過 BH）.** {bhs}。S1 含 P1b 的兩根，不是獨立證據。",
        "- 不可以說：「FOMC 一定大波動」「FOMC 比 CPI 猛」「比 CPI 大 X 倍」「記者會那根是公布那根的 1.35 倍」「1 小時收漲 43%（沒有附 9／21 與區間）」「漲跌各半」「FOMC 後黃金會漲／跌」「可以照做」「鮑爾講話比決議重要」「記者會不重要」「明天會大波動 X 倍」。",
        "",
        "## Limitations",
        "",
        f"- {n} meetings from 2024 to 2026 only; nothing is said about 2019–2023, including the 2022 hiking cycle. One hike and five cuts are in the sample.",
        "- The 2024-12-18 and 2026-10-28 meetings are excluded because the pre-registered control windows (holiday-adjacent days; end of the data file) leave too few control days. 事後補算 (read-only recomputation, not part of the registered analysis): putting both back with only the 8-day total minimum gives P1a 3.46×, P1b 4.21×, P2 −0.15 pp, P3 1.36×, same verdicts.",
        ] + [f"- {t}" for t in payload["limitations"][1:]] + [
        "- Not fully blind: the CPI study's multiples and the common view that FOMC days are volatile were known before the run (see the research record).",
        "",
        "## Provenance",
        "",
        "- Runner `scripts/research/build_xauusd_fomc_event.py` (checks first with `--checks-only`, then the full run); calendar collector `scripts/research/collect_fomc_calendar.py`. Inputs and hashes: `reproduction_inputs.json`. Fed pages: `fetch_manifest.csv` (url, sha256).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # study.json headline and findings
    sj = {}  # Public form: the study manifest is not read or rewritten
    sj["headline"] = {"meetings_judged": n, "meetings_in_sample": len(inn), "p1a_multiple": round(a["multiple"], 2), "p1a_interval_holm": [round(x, 2) for x in a["multiple_interval_holm"]],
                      "p1b_multiple": round(b["multiple"], 2), "p1b_interval_holm": [round(x, 2) for x in b["multiple_interval_holm"]],
                      "p2_mean_pp": round(c["mean"], 3), "p2_bound_holm_pp": round(c["bound_at_holm_level"], 3), "p3_R": round(t3["R_fomc_press_over_statement_geomean"], 2),
                      "p3_C": round(t3["C_control_press_over_statement_geomean"], 2), "p3_X": round(t3["X_R_over_C"], 2), "p3_interval_holm": [round(x, 2) for x in p3["multiple_interval_holm"]],
                      "p1a_cleared": a["cleared"], "p1b_cleared": b["cleared"], "p2_cleared": c["cleared"], "p3_cleared": p3["cleared"]}
    sj["findings"] = [
        {"title": "The first 30 minutes and first hour after the statement are several times a normal day", "tone": "info",
         "detail": f"{n} scheduled FOMC meetings, 2024-01 to 2026-10: first-30-minute range {x2(a['multiple'])}x the same ET bar on the nearest ordinary weekdays (interval {itv(a)}), first-hour range {x2(b['multiple'])}x ({itv(b)}); both clear Holm. Local-baseline geometric means; not comparable with the CPI study's 2.2x, which uses 08:30 ET bars. Same CPI-study formula on all weekdays: {x2(g['ratio_points_median'])}x / {x2(g['ratio_pct_median'])}x / {x2(g['ratio_mean_abs_r60'])}x (CPI 2.18 / 2.51 / 2.22).",
         "title_zh": "公布後 30 分鐘與 1 小時的波動是平常日的好幾倍", "detail_zh": claims["P1"] + claims["CPI_same_formula"]},
        {"title": "The 1-hour direction is not distinguishable from normal days", "tone": "info",
         "detail": f"Mean 1-hour move minus control {c['mean']:+.3f} pp, bound +/-{c['bound_at_holm_level']:.2f} (Holm level) / +/-{c['bound_first_step']:.2f} (first step), n = {n}: not distinguishable. The 1-hour move closed up in {ph['r60_up_share']['k']} of {n} meetings (Wilson 95% interval {ph['r60_up_share']['wilson95'][0]*100:.0f}-{ph['r60_up_share']['wilson95'][1]*100:.0f}%; post-hoc): no lean either way can be told. The sample has {sum(1 for e in r['event_table'] if e['decision']=='hold')} holds, {sum(1 for e in r['event_table'] if e['decision']=='cut')} cuts and 1 hike; the decision and dot-plot splits are below the 8-meeting rule.",
         "title_zh": "公布後 1 小時的方向分不出偏向", "detail_zh": claims["P2"]},
        {"title": f"The press-bar / statement-bar ratio is higher on FOMC days than on ordinary days (X {x2(t3['X_R_over_C'])}x; R {x2(t3['R_fomc_press_over_statement_geomean'])}x, C {x2(t3['C_control_press_over_statement_geomean'])}x)", "tone": "info",
         "detail": f"The tested quantity is X = R / C = {x2(t3['X_R_over_C'])}x (interval {itv(p3)}); it clears Holm at n = {n}. R = {x2(t3['R_fomc_press_over_statement_geomean'])}x is the press bar (14:30-15:00 ET) range over the statement bar range on FOMC days and C = {x2(t3['C_control_press_over_statement_geomean'])}x the same ratio on control days; R and C are descriptive and not tested alone. Post-hoc: X = {x2(pb['start_stated_on_fed_calendar']['X'])}x on the {pb['start_stated_on_fed_calendar']['n']} meetings whose press-conference start is stated on the Fed calendar (not distinguishable) and {x2(pb['start_assumed_14_30']['X'])}x on the {pb['start_assumed_14_30']['n']} with an assumed 14:30 start; the two do not differ detectably. Bar 3 vs bar 1 (secondary S2) {x2(sec['S2']['multiple'])}x, not distinguishable.",
         "title_zh": f"FOMC 當天「記者會那半小時 ÷ 公布那半小時」的比值，比平常日高（相對 {x2(t3['X_R_over_C'])} 倍；R {x2(t3['R_fomc_press_over_statement_geomean'])}、C {x2(t3['C_control_press_over_statement_geomean'])}）", "detail_zh": claims["P3"] + claims["P3_post_hoc"]},
    ]
    sj["revision"] = 4
    sj["title"] = "Gold around FOMC rate decisions (FX_IDC 30m, existing data 2024-2026): first 30m and 1h range multiple vs the same ET bars on nearby ordinary weekdays, 1h direction, and press-conference bar vs statement bar (pre-registered, amended twice before any result)"
    sj["runner"] = "scripts/research/build_xauusd_fomc_event.py"
    sj["limitations"] = payload["limitations"] + ["Not fully blind: the CPI study's intraday multiples (RS-XAUUSD-20261102-002) and the general view that FOMC days are volatile were known before the run.",
                                                  "The dot-plot split compares the median dot with the previous published SEP (December: next year-end), not with market expectations."]
    sj["stage"] = "results computed (pre-run checks passed)"
    (OUT / "study.json").write_text(json.dumps(sj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
