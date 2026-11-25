#!/usr/bin/env python3
"""RS-TX-20261124-001 — TAIEX big up days on below-normal trading value: what came next.

Definitions (chosen AFTER 3%/4%/5% x 0.8/1.0 had been looked at on the same
data; disclosed on the earlier study's page):
  ret[i]   = close[i] / close[i-1] - 1, consecutive TWSE sessions (FMTQIK TAIEX close).
  ratio[i] = value_twd[i] / mean(value_twd[i-20 .. i-1]) (the 20 sessions before, day itself
             excluded; FMTQIK 成交金額, whole market).
  Primary  : ret >= 4% and ratio < 0.8.
  Contrasts: ret >= 4% and ratio >= 1.0; ret >= 4% (all).
  Sensitivity: ret >= 3/4/5% x ratio < 0.7/0.8/0.9 (plus >= 1.0 and all per ret threshold).
  Forward  : close[i+h] / close[i] - 1, h = 5, 20, 60 sessions.
  dd20     : min(close[i+1 .. i+20]) / close[i] - 1 (lowest close of the next 20 sessions
             against the event-day close; positive means it never closed below).
  Episodes : events less than or equal to 20 sessions after the previous event join its episode.
  Baseline : every session with a defined ratio and a full forward window.
No event is dropped; the last session in the data (the day that prompted the study) has no
forward window and is reported separately, not in any statistic.

Usage: build_tx_low_volume_surge.py [--base CSV] [--update CSV] [--mi-index JSON] [--out-dir DIR]
Writes only results.json and README.md under --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

STUDY_ID = "RS-TX-20261124-001"
HORIZONS = [5, 20, 60]
LOOKBACK = 20
DD_WINDOW = 20
EPISODE_GAP = 20
PRIMARY = (4.0, 0.8)
HIGH_RATIO = 1.0
RET_GRID = [3.0, 4.0, 5.0]
RATIO_GRID = [0.7, 0.8, 0.9]
LIMIT_CHANGE = "2015-06-01"
BOOT = 10_000
SEED = 20261124
DEFAULT_DIR = f"research/studies/{STUDY_ID}"
WEEKDAY_ZH = "一二三四五六日"


# ---------------------------------------------------------------- data

def read_fmtqik(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_series(base: str, update: str) -> tuple[list[dict], dict]:
    a, b = read_fmtqik(base), read_fmtqik(update)
    a_by = {r["date"]: r for r in a}
    overlap = [r["date"] for r in b if r["date"] in a_by]
    mismatched = [d for d in overlap if a_by[d] != next(r for r in b if r["date"] == d)]
    if mismatched:
        raise SystemExit(f"update file differs from base on {mismatched}")
    last_base = max(a_by)
    appended = [r for r in b if r["date"] > last_base]
    rows = sorted(a + appended, key=lambda r: r["date"])
    dates = [r["date"] for r in rows]
    if len(set(dates)) != len(dates):
        raise SystemExit("duplicate dates")
    bad = []
    for i in range(1, len(rows)):
        d = float(rows[i]["taiex_close"]) - float(rows[i - 1]["taiex_close"])
        if abs(d - float(rows[i]["change_points"])) > 0.02:
            bad.append(rows[i]["date"])
    if bad:
        raise SystemExit(f"close change != published change in points on {bad[:10]}")
    info = {"base_sessions": len(a), "update_sessions": len(b), "overlap_sessions": len(overlap),
            "overlap_identical": True, "appended_sessions": len(appended),
            "appended_first": appended[0]["date"] if appended else None,
            "appended_last": appended[-1]["date"] if appended else None,
            "close_change_checks": len(rows) - 1, "close_change_mismatches": 0}
    return rows, info


# ---------------------------------------------------------------- stats

def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (cen - half), 2), round(100 * (cen + half), 2)]


EXACT_MAX_N = 10
_EXACT_CACHE: dict[tuple, tuple[list[float], float]] = {}


def exact_median_boot(x: np.ndarray) -> tuple[list[float], float]:
    """Exact bootstrap distribution of the median: every multiset of n draws with replacement,
    weighted by its multinomial probability n! / (prod c_k!) / n**n. Returns the 2.5% and 97.5%
    quantiles (smallest median whose cumulative probability reaches the level) and P(median <= 0).
    The median of an even-size resample is the mean of its two middle values, as numpy's."""
    xs = tuple(sorted(float(v) for v in x))
    if xs in _EXACT_CACHE:
        return _EXACT_CACHE[xs]
    n = len(xs)
    lf = [math.lgamma(k + 1) for k in range(n + 1)]
    lnorm = lf[n] - n * math.log(n)
    ranks = (n // 2 - 1, n // 2) if n % 2 == 0 else (n // 2, n // 2)
    dist: dict[float, float] = {}
    counts = [0] * n

    def emit() -> None:
        p = math.exp(lnorm - sum(lf[c] for c in counts))
        vals, cum, k = [], 0, 0
        for r in ranks:
            while cum + counts[k] <= r:
                cum += counts[k]
                k += 1
            vals.append(xs[k])
        m = (vals[0] + vals[1]) / 2
        dist[m] = dist.get(m, 0.0) + p

    def rec(pos: int, remaining: int) -> None:
        if pos == n - 1:
            counts[pos] = remaining
            emit()
            return
        for c in range(remaining + 1):
            counts[pos] = c
            rec(pos + 1, remaining - c)
        counts[pos] = 0

    rec(0, n)
    cdf, out, levels = 0.0, [], [0.025, 0.975]
    for m in sorted(dist):
        cdf += dist[m]
        while levels and cdf >= levels[0] - 1e-12:
            out.append(round(m, 2))
            levels.pop(0)
    p_le0 = sum(p for m, p in dist.items() if m <= 0)
    _EXACT_CACHE[xs] = (out, round(p_le0, 4))
    return _EXACT_CACHE[xs]


def boot_median(x: np.ndarray, rng: np.random.Generator) -> list[float] | None:
    if len(x) < 2:
        return None
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    if len(x) <= EXACT_MAX_N:
        # exact enumeration; the random draws above are still made and discarded so that
        # every later Monte Carlo interval keeps the value it had in revision 1 r1
        return exact_median_boot(x)[0]
    med = np.median(x[idx], axis=1)
    return [round(float(np.percentile(med, 2.5)), 2), round(float(np.percentile(med, 97.5)), 2)]


def boot_median_diff(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> list[float] | None:
    if len(x) < 2 or len(y) < 2:
        return None
    mx = np.median(x[rng.integers(0, len(x), size=(BOOT, len(x)))], axis=1)
    my = np.median(y[rng.integers(0, len(y), size=(BOOT, len(y)))], axis=1)
    d = mx - my
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]


def r2(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 2)


class Data:
    def __init__(self, rows: list[dict]):
        self.dates = [r["date"] for r in rows]
        self.close = np.array([float(r["taiex_close"]) for r in rows])
        self.value = np.array([float(r["value_twd"]) for r in rows])
        self.shares = np.array([float(r["volume_shares"]) for r in rows])
        self.trades = np.array([float(r["trades"]) for r in rows])
        self.points = np.array([float(r["change_points"]) for r in rows])
        n = len(rows)
        self.last = n - 1
        self.ret = np.full(n, np.nan)
        self.ret[1:] = self.close[1:] / self.close[:-1] - 1
        self.ratio = np.full(n, np.nan)
        self.shares_ratio = np.full(n, np.nan)
        self.trades_ratio = np.full(n, np.nan)
        self.avg20_value = np.full(n, np.nan)
        for i in range(LOOKBACK, n):
            self.avg20_value[i] = self.value[i - LOOKBACK:i].mean()
            self.ratio[i] = self.value[i] / self.avg20_value[i]
            self.shares_ratio[i] = self.shares[i] / self.shares[i - LOOKBACK:i].mean()
            self.trades_ratio[i] = self.trades[i] / self.trades[i - LOOKBACK:i].mean()
        self.weekday = [date.fromisoformat(d).weekday() for d in self.dates]
        # longer denominators and a weekday-only denominator (review r1, suggestions 4 and 5)
        self.ratio60 = np.full(n, np.nan)
        self.ratio120 = np.full(n, np.nan)
        for i in range(60, n):
            self.ratio60[i] = self.value[i] / self.value[i - 60:i].mean()
        for i in range(120, n):
            self.ratio120[i] = self.value[i] / self.value[i - 120:i].mean()
        self.ratio_wd = np.full(n, np.nan)
        wd: list[int] = []
        for i in range(n):
            if len(wd) >= LOOKBACK:
                self.ratio_wd[i] = self.value[i] / self.value[wd[-LOOKBACK:]].mean()
            if self.weekday[i] != 5:
                wd.append(i)

    def fwd(self, i: int, h: int) -> float | None:
        return (self.close[i + h] / self.close[i] - 1) * 100 if i + h <= self.last else None

    def dd(self, i: int) -> tuple[float, str] | None:
        if i + DD_WINDOW > self.last:
            return None
        seg = self.close[i + 1:i + DD_WINDOW + 1]
        j = int(np.argmin(seg))
        return (seg[j] / self.close[i] - 1) * 100, self.dates[i + 1 + j]

    def select(self, r_min: float, cond) -> list[int]:
        """Historical events: every session before the last with a defined ratio."""
        return [i for i in range(LOOKBACK, self.last)
                if self.ret[i] >= r_min / 100 and cond(self.ratio[i])]


def episodes(idx: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for i in idx:
        if out and i - out[-1][-1] <= EPISODE_GAP:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def group_stats(D: Data, idx: list[int], rng: np.random.Generator, label: str) -> dict:
    out = {"label": label, "n": len(idx), "episodes": len(episodes(idx)), "by_horizon": {}}
    for h in HORIZONS:
        valid = [i for i in idx if i + h <= D.last]
        r = np.array([D.fwd(i, h) for i in valid])
        if len(r) == 0:
            out["by_horizon"][str(h)] = {"n": 0}
            continue
        k = int((r > 0).sum())
        lo, hi = int(np.argmin(r)), int(np.argmax(r))
        out["by_horizon"][str(h)] = {
            "n": len(r), "up": k, "down": len(r) - k, "up_pct": round(100 * k / len(r), 2),
            "up_wilson95": wilson(k, len(r)),
            "median_pct": r2(np.median(r)), "median_boot95": boot_median(r, rng),
            "median_boot_method": ("exact enumeration" if 2 <= len(r) <= EXACT_MAX_N else
                                   (f"monte carlo {BOOT:,} resamples, seed {SEED}" if len(r) >= 2 else None)),
            "median_le0_prob_exact": exact_median_boot(r)[1] if 2 <= len(r) <= EXACT_MAX_N else None,
            "mean_pct": r2(r.mean()),
            "min_pct": r2(r[lo]), "min_date": D.dates[valid[lo]],
            "max_pct": r2(r[hi]), "max_date": D.dates[valid[hi]],
        }
    dd = [D.dd(i) for i in idx]
    ddv = np.array([x[0] for x in dd if x is not None])
    if len(ddv):
        out["dd20"] = {"n": len(ddv), "median_pct": r2(np.median(ddv)), "worst_pct": r2(ddv.min()),
                       "closed_below_event_close": int((ddv < 0).sum()),
                       "at_least_5pct_below": int((ddv <= -5).sum()),
                       "at_least_10pct_below": int((ddv <= -10).sum())}
    else:
        out["dd20"] = {"n": 0}
    return out


def baseline(D: Data) -> dict:
    out = {}
    for h in HORIZONS:
        valid = [i for i in range(LOOKBACK, D.last - h + 1)]
        r = (D.close[np.array(valid) + h] / D.close[np.array(valid)] - 1) * 100
        k = int((r > 0).sum())
        out[str(h)] = {"n": len(r), "up": k, "up_pct": round(100 * k / len(r), 2),
                       "median_pct": r2(np.median(r)), "first": D.dates[valid[0]], "last": D.dates[valid[-1]]}
    valid = list(range(LOOKBACK, D.last - DD_WINDOW + 1))
    ddv = np.array([D.dd(i)[0] for i in valid])
    out["dd20"] = {"n": len(ddv), "median_pct": r2(np.median(ddv)),
                   "at_least_10pct_below_pct": round(100 * float((ddv <= -10).mean()), 2)}
    return out


def baseline_by_regime(D: Data) -> dict:
    """Every session's forward return, split by whether the start session is before 2015-06-01."""
    out = {}
    for name, keep in (("7%", lambda d: d < LIMIT_CHANGE), ("10%", lambda d: d >= LIMIT_CHANGE)):
        out[name] = {}
        for h in HORIZONS:
            valid = np.array([i for i in range(LOOKBACK, D.last - h + 1) if keep(D.dates[i])])
            r = (D.close[valid + h] / D.close[valid] - 1) * 100
            k = int((r > 0).sum())
            out[name][str(h)] = {"n": len(r), "up": k, "up_pct": round(100 * k / len(r), 2),
                                 "median_pct": r2(np.median(r)), "first": D.dates[valid[0]], "last": D.dates[valid[-1]]}
    return out


def saturday_robustness(D: Data) -> dict:
    """Review r1 suggestion 5: the 20-session denominator without Saturday sessions. Lists every
    up-3%-or-more day (the widest sensitivity row) whose side of 0.7 / 0.8 / 0.9 / 1.0 changes."""
    sat = [i for i in range(len(D.dates)) if D.weekday[i] == 5]
    through_2000 = [i for i in sat if D.dates[i] <= "2000-12-31"]
    later = [i for i in sat if D.dates[i] > "2000-12-31"]
    wk_2000 = [i for i in range(LOOKBACK, len(D.dates)) if D.weekday[i] != 5 and D.dates[i] <= "2000-12-31"]
    sat_r = [D.ratio[i] for i in through_2000 if not math.isnan(D.ratio[i])]
    changes = []
    for i in D.select(3.0, lambda q: True):
        a, b = D.ratio[i], D.ratio_wd[i]
        for t in (0.7, 0.8, 0.9, 1.0):
            if (a < t) != (b < t):
                changes.append({"date": D.dates[i], "ret_pct": r2(100 * D.ret[i]), "threshold": t,
                                "value_ratio": round(float(a), 3), "value_ratio_weekday_only": round(float(b), 3)})
    return {
        "saturday_sessions_through_2000": len(through_2000),
        "saturday_sessions_after_2000": len(later),
        "saturday_sessions_after_2000_years": sorted({D.dates[i][:4] for i in later}),
        "median_value_ratio_saturdays_through_2000": r2(float(np.median(sat_r))) if sat_r else None,
        "median_value_ratio_weekdays_through_2000": r2(float(np.median([D.ratio[i] for i in wk_2000]))),
        "alternative": "denominator = mean value of the 20 most recent non-Saturday sessions before the day",
        "up3_days_changing_side": changes,
    }


def event_rows(D: Data, idx: list[int], big_all: list[int] | None = None) -> list[dict]:
    eps = episodes(idx)
    ep_of = {i: k + 1 for k, e in enumerate(eps) for i in e}
    rows = []
    for i in idx:
        dd = D.dd(i)
        near = None
        if big_all is not None:
            near = [{"date": D.dates[j], "sessions_from_event": j - i, "ret_pct": r2(100 * D.ret[j]),
                     "value_ratio": round(float(D.ratio[j]), 3)}
                    for j in big_all if j != i and abs(j - i) <= EPISODE_GAP]
        rows.append({
            "date": D.dates[i], "weekday_zh": WEEKDAY_ZH[D.weekday[i]],
            "ret_pct": r2(100 * D.ret[i]), "points": r2(D.points[i]), "close": r2(D.close[i]),
            "value_twd": int(D.value[i]), "avg20_value_twd": int(round(D.avg20_value[i])),
            "value_ratio": round(float(D.ratio[i]), 3),
            "shares_ratio": round(float(D.shares_ratio[i]), 3),
            "trades_ratio": round(float(D.trades_ratio[i]), 3),
            **{f"fwd{h}_pct": r2(D.fwd(i, h)) for h in HORIZONS},
            "dd20_pct": r2(dd[0]) if dd else None, "dd20_low_date": dd[1] if dd else None,
            "limit_regime": "7%" if D.dates[i] < LIMIT_CHANGE else "10%",
            "saturdays_in_ratio_window": int(sum(1 for j in range(i - LOOKBACK, i) if D.weekday[j] == 5)),
            "value_ratio_60": None if math.isnan(D.ratio60[i]) else round(float(D.ratio60[i]), 3),
            "value_ratio_120": None if math.isnan(D.ratio120[i]) else round(float(D.ratio120[i]), 3),
            "value_ratio_weekday_only": None if math.isnan(D.ratio_wd[i]) else round(float(D.ratio_wd[i]), 3),
            "episode": ep_of[i],
            **({"other_big_up_within_20": near} if near is not None else {}),
        })
    return rows


def clusters(D: Data, idx: list[int]) -> list[list[dict]]:
    return [[{"date": D.dates[i], "ret_pct": r2(100 * D.ret[i]), "value_ratio": round(float(D.ratio[i]), 3)} for i in e]
            for e in episodes(idx) if len(e) > 1]


def mi_index_facts(path: str) -> dict:
    j = json.loads(Path(path).read_text())
    if j.get("stat") != "OK" or j.get("date") != "20261124":
        raise SystemExit("MI_INDEX file is not the 2026-11-24 OK response")
    out: dict = {"date": "2026-11-24"}

    def n(s: str) -> int:
        return int(s.replace(",", ""))

    for t in j["tables"]:
        if t.get("title") == "漲跌證券數合計":
            if t["fields"] != ["類型", "整體市場", "股票"]:
                raise SystemExit(f"unexpected fields {t['fields']}")
            for row in t["data"]:
                name, stock = row[0], row[2]
                if "(" in stock:
                    a, b = stock.rstrip(")").split("(")
                    out[name] = {"count": n(a), "limit": n(b)}
                else:
                    out[name] = {"count": n(stock)}
        if t.get("title", "").endswith("大盤統計資訊"):
            for row in t["data"]:
                if row[0] in ("1.一般股票", "4.ETF"):
                    out[f"value_{row[0].split('.')[1]}"] = n(row[1])
                    out[f"shares_{row[0].split('.')[1]}"] = n(row[2])
    st = out
    return {
        "date": "2026-11-24",
        "stocks_up": st["上漲(漲停)"]["count"], "stocks_limit_up": st["上漲(漲停)"]["limit"],
        "stocks_down": st["下跌(跌停)"]["count"], "stocks_limit_down": st["下跌(跌停)"]["limit"],
        "stocks_unchanged": st["持平"]["count"], "stocks_no_trade": st["未成交"]["count"],
        "stocks_no_comparable_price": st["無比價"]["count"],
        "value_common_stocks_twd": st.get("value_一般股票"), "value_etf_twd": st.get("value_ETF"),
        "shares_common_stocks": st.get("shares_一般股票"), "shares_etf": st.get("shares_ETF"),
        "source_table": "漲跌證券數合計, column 股票 (not 整體市場, which includes warrants and other securities)",
    }


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/stocks/market/twse/TAIEX_FMTQIK_1997-06-02_to_2026-11-13.csv")
    ap.add_argument("--update", default=f"{DEFAULT_DIR}/fmtqik_2026-11.csv")
    ap.add_argument("--mi-index", default=f"{DEFAULT_DIR}/raw/mi_index_ms_20261124.json")
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    a = ap.parse_args()

    rows, merge = load_series(a.base, a.update)
    D = Data(rows)
    rng = np.random.default_rng(SEED)
    R, Q = PRIMARY

    prim = D.select(R, lambda q: q < Q)
    high = D.select(R, lambda q: q >= HIGH_RATIO)
    allr = D.select(R, lambda q: True)
    groups = {
        "primary": group_stats(D, prim, rng, f"ret >= {R}% and value ratio < {Q}"),
        "high_volume": group_stats(D, high, rng, f"ret >= {R}% and value ratio >= {HIGH_RATIO}"),
        "all_big_up": group_stats(D, allr, rng, f"ret >= {R}% (any value ratio)"),
    }
    groups["primary"]["episode_first_only"] = group_stats(
        D, [e[0] for e in episodes(prim)], rng, "primary, first event of each episode only")
    diffs = {}
    for h in HORIZONS:
        x = np.array([v for v in (D.fwd(i, h) for i in prim) if v is not None])
        y = np.array([v for v in (D.fwd(i, h) for i in high) if v is not None])
        eff = r2(np.median(x) - np.median(y)) if len(x) and len(y) else None
        bci = boot_median_diff(x, y, rng)
        half = r2((bci[1] - bci[0]) / 2) if bci else None
        diffs[str(h)] = {
            "primary_median_minus_high_volume_median_pct": eff,
            "boot95": bci,
            "smallest_resolvable_pct": half,
            "effect_to_bound": round(abs(eff) / half, 2) if eff is not None and half else None,
            "note": "percentile bootstrap, each group resampled on its own; an interval containing 0 means the two medians cannot be separated with these n; smallest_resolvable_pct = half-width of that interval",
        }
    share = {}
    base_all = baseline(D)
    for h in HORIZONS:
        b = groups["primary"]["by_horizon"][str(h)]
        w = b["up_wilson95"]
        eff = round(b["up_pct"] - base_all[str(h)]["up_pct"], 2)
        half = round((w[1] - w[0]) / 2, 2)
        share[str(h)] = {"primary_up_pct": b["up_pct"], "baseline_up_pct": base_all[str(h)]["up_pct"],
                         "effect_pp": eff, "smallest_resolvable_pp": half,
                         "effect_to_bound": round(abs(eff) / half, 2),
                         "baseline_inside_wilson": w[0] <= base_all[str(h)]["up_pct"] <= w[1]}

    sens = []
    for rr in RET_GRID:
        cells = [(f"< {q}", lambda v, q=q: v < q) for q in RATIO_GRID]
        cells += [(f">= {HIGH_RATIO}", lambda v: v >= HIGH_RATIO), ("all", lambda v: True)]
        for name, cond in cells:
            idx = D.select(rr, cond)
            g = group_stats(D, idx, rng, f"ret >= {rr}% and ratio {name}")
            row = {"ret_min_pct": rr, "ratio": name, "n": g["n"], "episodes": g["episodes"]}
            for h in HORIZONS:
                b = g["by_horizon"][str(h)]
                row[f"n{h}"] = b["n"]
                row[f"up{h}"] = b.get("up")
                row[f"median{h}_pct"] = b.get("median_pct")
            row["up20_wilson95"] = g["by_horizon"]["20"].get("up_wilson95")
            row["dd20_median_pct"] = g["dd20"].get("median_pct")
            row["dd20_at_least_10pct_below"] = g["dd20"].get("at_least_10pct_below")
            sens.append(row)

    # the day that prompted the study
    L = D.last
    big = [i for i in range(1, L + 1)]
    order = sorted(big, key=lambda i: -D.ret[i])
    top = [{"rank": k + 1, "date": D.dates[i], "ret_pct": r2(100 * D.ret[i]), "points": r2(D.points[i]),
            "value_ratio": None if math.isnan(D.ratio[i]) else round(float(D.ratio[i]), 3)} for k, i in enumerate(order[:5])]
    hist4 = [i for i in allr]
    lower_ratio_hist = [D.dates[i] for i in hist4 if D.ratio[i] <= D.ratio[L]]
    now = {
        "date": D.dates[L], "weekday_zh": WEEKDAY_ZH[D.weekday[L]], "ret_pct": r2(100 * D.ret[L]),
        "ret_pct_4dp": round(100 * float(D.ret[L]), 4),
        "points": r2(D.points[L]), "close": r2(D.close[L]), "prev_close": r2(D.close[L - 1]),
        "value_twd": int(D.value[L]), "avg20_value_twd": int(round(D.avg20_value[L])),
        "avg20_window": [D.dates[L - LOOKBACK], D.dates[L - 1]],
        "value_ratio": round(float(D.ratio[L]), 4), "shares_ratio": round(float(D.shares_ratio[L]), 4),
        "trades_ratio": round(float(D.trades_ratio[L]), 4),
        "qualifies_primary": bool(D.ret[L] >= R / 100 and D.ratio[L] < Q),
        "return_rank_since": D.dates[1], "return_rank": order.index(L) + 1, "top5_returns": top,
        "big_up_days_with_lower_or_equal_ratio": lower_ratio_hist,
        "big_up_days_ratio_min": {"date": D.dates[min(hist4, key=lambda i: D.ratio[i])],
                                  "value_ratio": round(float(min(D.ratio[i] for i in hist4)), 3)},
        "forward_available": False,
        "mi_index": mi_index_facts(a.mi_index),
        "volume_shares": int(D.shares[L]),
        "value_ratio_60": round(float(D.ratio60[L]), 4),
        "value_ratio_120": round(float(D.ratio120[L]), 4),
        "value_ratio_weekday_only": round(float(D.ratio_wd[L]), 4),
    }
    prior_low = [j for j in range(L) if D.value[j] <= D.value[L]]
    now["lowest_value_since"] = ({"date": D.dates[prior_low[-1]], "value_twd": int(D.value[prior_low[-1]]),
                                  "note": "the most recent earlier session with value at or below this day's"}
                                 if prior_low else None)
    mi = now["mi_index"]
    now["shares_split"] = {
        "etf_pct": round(100 * mi["shares_etf"] / now["volume_shares"], 2),
        "common_stocks_pct": round(100 * mi["shares_common_stocks"] / now["volume_shares"], 2),
        "etf_value_pct": round(100 * mi["value_etf_twd"] / now["value_twd"], 2),
        "common_stocks_value_pct": round(100 * mi["value_common_stocks_twd"] / now["value_twd"], 2),
        "note": "shares and value of MI_INDEX 大盤統計資訊 rows 1.一般股票 and 4.ETF over FMTQIK totals of the same day",
    }

    results = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "strategy_version": None,
        "method": {
            "ret": "close[i] / close[i-1] - 1, consecutive TWSE sessions, FMTQIK TAIEX close",
            "value_ratio": f"value_twd[i] / mean(value_twd[i-{LOOKBACK} .. i-1]); FMTQIK 成交金額 (whole market, TWD); day itself excluded; arithmetic mean",
            "primary": f"ret >= {R}% and value_ratio < {Q}",
            "contrasts": [f"ret >= {R}% and value_ratio >= {HIGH_RATIO}", f"ret >= {R}% (all)"],
            "sensitivity": {"ret_min_pct": RET_GRID, "ratio_below": RATIO_GRID, "also": [f">= {HIGH_RATIO}", "all"]},
            "forward_return": "close[i+h] / close[i] - 1 with h sessions (rows of the file, Saturdays included)",
            "horizons": HORIZONS,
            "dd20": f"min(close[i+1 .. i+{DD_WINDOW}]) / close[i] - 1 (closing prices only; intraday lows not in the source)",
            "episodes": f"an event within {EPISODE_GAP} sessions after the previous event joins its episode",
            "intervals": f"Wilson 95% for the up share; percentile bootstrap ({BOOT:,} resamples of events, seed {SEED}) for medians and for the median difference",
            "baseline": "every session from the 21st row on with a full forward window",
            "historical_events": "every qualifying session before the last session in the data; none dropped, no de-clustering",
            "definition_chosen_after_looking": "yes: the work-order author tried ret 3/4/5% x ratio 0.8/1.0 on this data before fixing ret>=4%, ratio<0.8; hence the full sensitivity grid",
        },
        "coverage": {"first": D.dates[0], "last": D.dates[L], "sessions": len(D.dates),
                     "first_with_ratio": D.dates[LOOKBACK], "merge": merge},
        "baseline": baseline(D),
        "groups": groups,
        "median_difference_primary_vs_high_volume": diffs,
        "up_share_primary_vs_baseline": share,
        "events_primary": event_rows(D, prim, allr),
        "events_high_volume": event_rows(D, high),
        "events_all_big_up_count_by_regime": {
            "7%": sum(1 for i in allr if D.dates[i] < LIMIT_CHANGE), "10%": sum(1 for i in allr if D.dates[i] >= LIMIT_CHANGE)},
        "primary_by_regime": {
            "7%": group_stats(D, [i for i in prim if D.dates[i] < LIMIT_CHANGE], rng, "primary, before 2015-06-01 (7% limit)"),
            "10%": group_stats(D, [i for i in prim if D.dates[i] >= LIMIT_CHANGE], rng, "primary, from 2015-06-01 (10% limit)"),
            "note": "descriptive split, n=4 or so per side; not a conclusion",
        },
        # review r1 must-fix 1: the contrasts and the baseline split the same way (placed after
        # primary_by_regime so every earlier random draw is unchanged)
        "high_volume_by_regime": {
            "7%": group_stats(D, [i for i in high if D.dates[i] < LIMIT_CHANGE], rng, "high volume, before 2015-06-01 (7% limit)"),
            "10%": group_stats(D, [i for i in high if D.dates[i] >= LIMIT_CHANGE], rng, "high volume, from 2015-06-01 (10% limit)"),
        },
        "all_big_up_by_regime": {
            "7%": group_stats(D, [i for i in allr if D.dates[i] < LIMIT_CHANGE], rng, "all up-4% days, before 2015-06-01 (7% limit)"),
            "10%": group_stats(D, [i for i in allr if D.dates[i] >= LIMIT_CHANGE], rng, "all up-4% days, from 2015-06-01 (10% limit)"),
        },
        "baseline_by_regime": baseline_by_regime(D),
        "saturday_robustness": saturday_robustness(D),
        "clusters_all_big_up": clusters(D, allr),
        "sensitivity": sens,
        "now": now,
        "limitations": LIMITATIONS,
        "inference_not_tested": INFERENCE,
    }
    results["work_order_recheck"] = recheck(results)
    results["limitations"] = list(LIMITATIONS)
    results["limitations"][1] = cluster_sentence(results)
    results["limitations"][2] = regime_sentence(results)
    results["limitations"][3] = saturday_sentence(results)
    results["limitations"].append(denominator_sentence(results))
    results["limitations"].append(
        f"n ≤ {EXACT_MAX_N} 的中位數區間用精確列舉（不靠亂數）；n > {EXACT_MAX_N} 的仍是 {BOOT:,} 次抽樣，換種子時邊界會移動"
        "（審查用別的種子、20,000 次：漲 ≥4% 全部組 60 日上界 +10.07，本檔 +8.78）。量小減量大的中位數差也是抽樣區間，審查用 100 個種子都包含 0。")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text(readme(results))
    g = groups["primary"]["by_horizon"]["20"]
    print(f"primary n={len(prim)} 20d up {g.get('up')}/{g.get('n')} median {g.get('median_pct')}; "
          f"high n={len(high)}; all n={len(allr)}; wrote {out}")


def recheck(r: dict) -> list[dict]:
    """The work order's exploration figures (2026-11-24 19:00), each against this run."""
    now, mi = r["now"], r["now"]["mi_index"]
    p, h = r["groups"]["primary"], r["groups"]["high_volume"]
    p20 = p["by_horizon"]["20"]
    dd = sorted(e["dd20_pct"] for e in r["events_primary"] if e["dd20_pct"] is not None)
    below5 = [x for x in dd if x <= -5]
    top1 = now["top5_returns"][0]
    out = [
        {"claim": "11/24 +3,310.46 點（+8.99%），收 40,122.35",
         "recomputed": f"{now['points']:+,.2f} 點（{now['ret_pct']:+.2f}%），收 {now['close']:,.2f}",
         "match": now["points"] == 3310.46 and now["ret_pct"] == 8.99 and now["close"] == 40122.35},
        {"claim": "成交金額 432,559,877,529 元，前 20 日平均的 0.44 倍",
         "recomputed": f"{now['value_twd']:,} 元，{now['value_ratio']:.4f} 倍",
         "match": now["value_twd"] == 432559877529 and round(now["value_ratio"], 2) == 0.44},
        {"claim": "1997 年以來單日漲幅第二，第一是 2025-04-10 +9.25%",
         "recomputed": f"第 {now['return_rank']}；第一 {top1['date']} {top1['ret_pct']:+.2f}%",
         "match": now["return_rank"] == 2 and top1["date"] == "2025-04-10" and top1["ret_pct"] == 9.25},
        {"claim": "MI_INDEX 股票上漲 1,053（漲停 335），下跌 26（跌停 4）",
         "recomputed": f"上漲 {mi['stocks_up']:,}（漲停 {mi['stocks_limit_up']}），下跌 {mi['stocks_down']}（跌停 {mi['stocks_limit_down']}）",
         "match": (mi["stocks_up"], mi["stocks_limit_up"], mi["stocks_down"], mi["stocks_limit_down"]) == (1053, 335, 26, 4)},
        {"claim": "≥4% 且量 <0.8 倍今天以前 8 次；20 日後 6 漲 2 跌，中位數 +4.1%",
         "recomputed": f"{p['n']} 次；20 日後 {p20['up']} 漲 {p20['down']} 跌，中位數 {p20['median_pct']:+.2f}%",
         "match": p["n"] == 8 and p20["up"] == 6 and p20["down"] == 2 and round(p20["median_pct"], 1) == 4.1},
        {"claim": "其中 3 次在 20 天內曾比大漲當天收盤低 10–13%",
         "recomputed": (f"以收盤價算，20 日內最低比大漲當天低 5% 以上的有 {len(below5)} 次："
                        + "、".join(f"{x:.2f}%" for x in below5)
                        + f"；低 10% 以上只有 {sum(1 for x in dd if x <= -10)} 次"),
         "match": len(below5) == 3 and all(-13 <= x <= -10 for x in below5)},
        {"claim": "量 ≥1.0 倍 39 次，中位數 +1.4%",
         "recomputed": f"{h['n']} 次，20 日中位數 {h['by_horizon']['20']['median_pct']:+.2f}%",
         "match": h["n"] == 39 and round(h["by_horizon"]["20"]["median_pct"], 1) == 1.4},
    ]
    return out


def cluster_sentence(r: dict) -> str:
    p = r["groups"]["primary"]
    a = r["groups"]["all_big_up"]
    parts = [f"事件在時間上群聚：漲 ≥ {PRIMARY[0]:.0f}% 的 {a['n']} 天只分成 {a['episodes']} 段行情（相隔 ≤ {EPISODE_GAP} 個交易日算同一段），"
             "同一段的 20、60 日窗口幾乎重疊，獨立樣本比次數少。"]
    if p["episodes"] == p["n"]:
        parts.append(f"主定義的 {p['n']} 次彼此都相隔超過 {EPISODE_GAP} 個交易日（{p['episodes']} 段），")
    else:
        parts.append(f"主定義 {p['n']} 次分在 {p['episodes']} 段，")
    nb = [(e["date"], x) for e in r["events_primary"] for x in e.get("other_big_up_within_20") or []]
    if nb:
        parts.append("但有些主定義事件旁邊 20 個交易日內還有別的大漲日、走勢共用："
                     + "；".join(f"{d} 和 {x['date']}（相差 {x['sessions_from_event']:+d} 個交易日，{x['ret_pct']:+.2f}%，量比 {x['value_ratio']:.2f}）" for d, x in nb)
                     + "。")
    return "".join(parts)


def regime_text(r: dict) -> str:
    """Review r1 must-fix 1: the contrast for the 10% / 7% split of the primary definition."""
    def c(g: dict, h: str = "20") -> str:
        b = g["by_horizon"][h]
        return f"{b.get('up', 0)}/{b.get('n', 0)}"

    def m(g: dict, h: str = "20") -> str:
        return pct(g["by_horizon"][h].get("median_pct"))

    a10, a7 = r["all_big_up_by_regime"]["10%"], r["all_big_up_by_regime"]["7%"]
    h10, p10 = r["high_volume_by_regime"]["10%"], r["primary_by_regime"]["10%"]
    b10 = a10["by_horizon"]["20"]
    same = b10.get("n") and b10["up"] == b10["n"]
    return (f"同一時期的對照：10% 制所有漲 ≥4% 的日子，20 日後 {c(a10)} 上漲（中位數 {m(a10)}），量大的 {c(h10)}；"
            f"7% 制所有漲 ≥4% 的日子 {c(a7)}（{m(a7)}）。主定義 10% 制 {c(p10)}"
            + ("，同時期的大漲日也都上漲，分不出是量小的關係。" if same else "；同時期的大漲日並非全部上漲，見上表。"))


def regime_sentence(r: dict) -> str:
    return ("2015-06-01 起漲跌幅限制從 7% 改成 10%：之前一天最多漲 7%、漲停的意義和現在不同，前後的大漲日不能完全等同；每筆事件標了它屬於哪一制。"
            + regime_text(r))


def saturday_sentence(r: dict) -> str:
    s = r["saturday_robustness"]
    ch = s["up3_days_changing_side"]
    return (f"週六交易日：2000 年以前的週六盤 {s['saturday_sessions_through_2000']} 天，和之後的補班週六 {s['saturday_sessions_after_2000']} 天"
            f"（{s['saturday_sessions_after_2000_years'][0]}–{s['saturday_sessions_after_2000_years'][-1]}）都算交易日。"
            f"2000 年以前週六的量比中位數 {s['median_value_ratio_saturdays_through_2000']:.2f}、同期平日 {s['median_value_ratio_weekdays_through_2000']:.2f}，"
            "週六只比平日略低，對 20 日分母的影響很小；h 個交易日在日曆上也比較短。每筆事件列出量比窗口裡有幾個週六。"
            "改用只算平日的分母（前 20 個非週六交易日）重算："
            + ("主定義和兩個對照組（漲 ≥4%，0.8 和 1.0 兩條線）沒有任何一天換組。"
               if not any(x["ret_pct"] >= PRIMARY[0] and x["threshold"] in (PRIMARY[1], HIGH_RATIO) for x in ch)
               else "主定義或對照組有日子換組，見 saturday_robustness。")
            + ("敏感度表的其他格也全部不變。" if not ch else
               "敏感度表裡會跨線的有：" + "、".join(f"{x['date']}（+{x['ret_pct']:.2f}%，{x['value_ratio']:.3f} → {x['value_ratio_weekday_only']:.3f}，跨 {x['threshold']}）" for x in ch)
               + "；受影響的格（漲幅門檻 × 線）：" + "、".join(sorted({f"{rr:.0f}% × {x['threshold']}" for x in ch for rr in RET_GRID if x['ret_pct'] >= rr}))
               + "，數字沒有重算進敏感度表。"))


def denominator_sentence(r: dict) -> str:
    now = r["now"]
    ev = "；".join(f"{e['date']} {e['value_ratio']:.2f}／{e['value_ratio_60']:.2f}／{e['value_ratio_120']:.2f}" for e in r["events_primary"])
    lv = now["lowest_value_since"]
    return ("量比的分母只用前 20 個交易日：前 20 天如果剛好是恐慌大量，「量小」只是相對那段大量。主定義 8 次對前 20／60／120 日平均的倍數："
            + ev + "。其中 2020-03-24 和 2021-05-18 對前 60 日接近 1 倍、對前 120 日超過 1 倍。"
            f"{now['date']} 對前 20／60／120 日是 {now['value_ratio']:.3f}／{now['value_ratio_60']:.3f}／{now['value_ratio_120']:.3f} 倍；"
            + (f"成交金額 {now['value_twd']:,} 元是 {lv['date']}（{lv['value_twd']:,} 元）以來最低。" if lv else "成交金額是資料內最低。"))


LIMITATIONS = [
    "樣本很少：主定義歷史上只有個位數次，每個比例的 Wilson 95% 區間寬達數十個百分點；中位數的 bootstrap 區間也很寬。",
    "事件在時間上群聚（例如 2020-03 的幾天、2008 年底），同一段行情的 20、60 日窗口互相重疊，獨立樣本比次數更少；表中另列「段數」。",
    "2015-06-01 起漲跌幅限制從 7% 改成 10%：之前一天最多漲 7%、漲停的意義和現在不同，前後的大漲日不能完全等同；每筆事件標了它屬於哪一制。",
    "2001 年以前有週六交易（半日盤，成交金額較小）：20 日平均成交金額裡有週六時，平均會被拉低、量比會偏高；h 個交易日也比 h 個平日短。每筆事件列出量比窗口裡有幾個週六。",
    "先看資料、後選定義：定義確定之前，已經有人在同一份資料上試過漲幅 3%、4%、5% 和量 0.8、1.0 倍，所以主定義不是事先登記的；敏感度表就是為了讓讀者看到換門檻後結果怎麼變。",
    "只是描述，不是預測：不推因果、不代表之後會怎麼走；也沒有扣交易成本，加權指數是價格指數不含股息。",
    "成交金額是全市場（含 ETF、權證等，FMTQIK 的定義），不是只算股票；ETF 占比逐年變大，量比的意義前後可能不同。",
    "20 日內最低用收盤價，盤中更低的價格沒有算進去。",
]

INFERENCE = (
    "推論（未驗證）：2026-11-24 股票漲停 335 檔、上漲 1,053 檔（MI_INDEX 漲跌證券數合計，股票欄），"
    "量小可能有一部分是很多股票鎖在漲停、想買的人買不到，成交因此變少。這只是推論；本研究沒有逐一查歷史事件當天的漲停家數，"
    "所以不能說歷史上的量小大漲日也是這個原因，也不能用來解釋歷史結果。"
)


# ---------------------------------------------------------------- README

def pct(v, sign=True) -> str:
    return "–" if v is None else (f"{v:+.2f}%" if sign else f"{v:.2f}%")


def ci(v) -> str:
    return "–" if not v else f"{v[0]:+.2f} 到 {v[1]:+.2f}"


def group_table(g: dict, base: dict) -> str:
    s = ["| 之後 | 有資料的次數 n | 上漲次數／n（Wilson 95%） | 中位數報酬（bootstrap 95%） | 最差（日期） | 最好（日期） | 平常日：上漲比例／中位數 |",
         "|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        b, z = g["by_horizon"][str(h)], base[str(h)]
        if b["n"] == 0:
            s.append(f"| {h} 日 | 0 | – | – | – | – | {z['up_pct']}%（n={z['n']:,}）／{pct(z['median_pct'])} |")
            continue
        w = b["up_wilson95"]
        s.append(f"| {h} 日 | {b['n']} | {b['up']}/{b['n']}（{w[0]}–{w[1]}%） | {pct(b['median_pct'])}（{ci(b['median_boot95'])}） | "
                 f"{pct(b['min_pct'])}（{b['min_date']}） | {pct(b['max_pct'])}（{b['max_date']}） | {z['up_pct']}%（n={z['n']:,}）／{pct(z['median_pct'])} |")
    d = g["dd20"]
    if d["n"]:
        s.append(f"\n之後 20 個交易日內的最低收盤，對大漲當天收盤：中位數 {pct(d['median_pct'])}，最深 {pct(d['worst_pct'])}；"
                 f"{d['n']} 次裡 {d['closed_below_event_close']} 次曾收在大漲當天收盤之下、{d['at_least_5pct_below']} 次低 5% 以上、"
                 f"{d['at_least_10pct_below']} 次低 10% 以上（平常日：中位數 {pct(base['dd20']['median_pct'])}，低 10% 以上的比例 {base['dd20']['at_least_10pct_below_pct']}%，n={base['dd20']['n']:,}）。")
    return "\n".join(s)


def readme(r: dict) -> str:
    m, cov, base, G, now = r["method"], r["coverage"], r["baseline"], r["groups"], r["now"]
    R, Q = PRIMARY
    s = [f"# {STUDY_ID} — 加權指數大漲、成交金額卻比平常小的日子，之後 5／20／60 個交易日怎麼走（{cov['first_with_ratio']}–{cov['last']}）\n"]
    s.append("**這是描述統計，不是預測，也不是買賣訊號。** 下面每個比例都附次數和對照；區間很寬的地方，代表這些次數分不出差別。\n")
    s.append("## 數字怎麼算出來的\n")
    s.append(f"- 資料：證交所 FMTQIK（每日市場成交資訊）{cov['first']} 到 {cov['last']}，共 {cov['sessions']:,} 個交易日；"
             f"2026-11-13 以前用已追蹤的檔，2026-11-16 到 {cov['last']} 用 11-24 晚上抓的 11 月官方資料（兩份重疊的 {cov['merge']['overlap_sessions']} 天逐欄相同）。"
             f"每天的收盤變化都和證交所公布的漲跌點數相符（{cov['merge']['close_change_checks']:,} 天、誤差 ≤ 0.02 點）。")
    s.append("- 當日漲幅＝當天加權指數收盤 ÷ 前一個交易日收盤 − 1。")
    s.append(f"- 量比＝當天成交金額 ÷ 前 {LOOKBACK} 個交易日成交金額的算術平均（不含當天）。成交金額是 FMTQIK 的全市場成交金額（含 ETF、權證），單位新台幣元。"
             f"因為要前 {LOOKBACK} 天，第一個能算量比的日子是 {cov['first_with_ratio']}。")
    s.append(f"- 主定義：漲幅 ≥ {R}% 且量比 < {Q}。對照組：漲幅 ≥ {R}% 且量比 ≥ {HIGH_RATIO}（量大的大漲）、漲幅 ≥ {R}% 全部。")
    s.append(f"- 之後 h 日報酬＝大漲當天收盤到之後第 h 個交易日收盤（h＝5、20、60；用檔案裡的列數算，2000 年以前的週六盤和之後的補班週六都算一天）。")
    s.append(f"- 20 日內最低＝之後 {DD_WINDOW} 個交易日裡最低的收盤 ÷ 大漲當天收盤 − 1；只用收盤價。正值代表 20 天內從沒收在大漲當天之下。")
    s.append(f"- 「段數」：一個事件離上一個事件不到（含）{EPISODE_GAP} 個交易日，就算同一段行情。")
    s.append(f"- 區間：上漲比例用 Wilson 95%；中位數的 95% 區間是事件重抽樣（bootstrap）：n ≤ {EXACT_MAX_N} 的組精確列舉所有重抽組合（不靠亂數，n=8 有 6,435 種組合），"
             f"n > {EXACT_MAX_N} 的組抽 {BOOT:,} 次（種子 {SEED}）。區間端點＝累積機率第一次達到 2.5%、97.5% 的中位數。平常日＝從第 {LOOKBACK + 1} 個交易日起、每一天往後算同樣天數（台股長期往上，平常日本身就過半會漲，這是對照，不是結論）。")
    s.append(f"- 歷史事件不刪任何一筆、也不併群；{now['date']} 當天沒有之後的資料，單獨列在最後，不進任何統計。")
    s.append("- **定義是先看過資料才選的**：定義確定之前，已經有人在同一份資料上試過漲幅 3%、4%、5% 和量比 0.8、1.0 倍。所以主定義不是事先登記，下面一定要一起看敏感度表。\n")

    p = G["primary"]
    s.append(f"## 主結果：漲 ≥ {R}% 且量比 < {Q}（{p['n']} 次、{p['episodes']} 段行情）\n")
    s.append(group_table(p, base))
    p20 = p["by_horizon"]["20"]
    s.append(f"\n20 日中位數區間的下界 {p20['median_boot95'][0]:+.2f} 靠近 0：精確 bootstrap 算出重抽後中位數 ≤ 0 的機率是 "
             f"{100 * p20['median_le0_prob_exact']:.1f}%（5 日 {100 * p['by_horizon']['5']['median_le0_prob_exact']:.1f}%、"
             f"60 日 {100 * p['by_horizon']['60']['median_le0_prob_exact']:.1f}%）。這是「這 8 個數字重抽」的機率，不是之後會跌的機率。")
    ef = p["episode_first_only"]["by_horizon"]["20"]
    if ef.get("n"):
        s.append(f"\n每段行情只算第一天時（{p['episode_first_only']['n']} 次）：20 日後 {ef['up']}/{ef['n']} 次上漲，中位數 {pct(ef['median_pct'])}。")
    s.append("\n### 每一個事件\n")
    s.append("| 日期 | 星期 | 當日漲幅 | 收盤 | 量比 | 量比（60／120 日分母） | 股數量比 | 5 日後 | 20 日後 | 60 日後 | 20 日內最低（日期） | 漲跌幅制 | 量比窗口內週六數 | 段 |")
    s.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in r["events_primary"]:
        s.append(f"| {e['date']} | {e['weekday_zh']} | {pct(e['ret_pct'])} | {e['close']:,.2f} | {e['value_ratio']:.2f} | {e['value_ratio_60']:.2f}／{e['value_ratio_120']:.2f} | {e['shares_ratio']:.2f} | "
                 f"{pct(e['fwd5_pct'])} | {pct(e['fwd20_pct'])} | {pct(e['fwd60_pct'])} | {pct(e['dd20_pct'])}（{e['dd20_low_date'] or '–'}） | "
                 f"{e['limit_regime']} | {e['saturdays_in_ratio_window']} | {e['episode']} |")
    s.append("\n股數量比＝當天成交股數 ÷ 前 20 個交易日平均成交股數，只供參考，定義用的是成交金額。"
             "量比（60／120 日分母）＝當天成交金額 ÷ 前 60、120 個交易日平均，只供參考，定義用的是 20 日。"
             "「量比窗口內週六數」＝算 20 日平均的那 20 個交易日裡有幾個週六交易日（2000 年以前的週六盤和之後的補班週六；週六成交金額略低，可能讓平均略偏低、量比略偏高，見限制）。\n")
    nb = [(e["date"], x) for e in r["events_primary"] for x in e.get("other_big_up_within_20") or []]
    if nb:
        s.append("前後 20 個交易日內還有別的漲 ≥ 4% 日子的主定義事件（兩者之後的走勢大半重疊）：")
        s += [f"- {d}：{x['date']}（相差 {x['sessions_from_event']:+d} 個交易日，當日 {pct(x['ret_pct'])}，量比 {x['value_ratio']:.2f}）" for d, x in nb]
        s.append("")
    rg = r["primary_by_regime"]
    s.append("按漲跌幅制分（描述性切分，每邊只有幾次，不能當結論）：" + "；".join(
        f"{k} 制 {rg[k]['n']} 次，20 日後 {rg[k]['by_horizon']['20'].get('up', 0)}/{rg[k]['by_horizon']['20'].get('n', 0)} 上漲、中位數 {pct(rg[k]['by_horizon']['20'].get('median_pct'))}"
        for k in ("7%", "10%")) + "。" + regime_text(r) + "\n")
    s.append("| 時期 | 組 | 次數 | 5 日 上漲／n，中位數 | 20 日 上漲／n，中位數 | 60 日 上漲／n，中位數 |\n|---|---|---|---|---|---|")
    for k, lab in (("7%", "2015-06-01 以前（7% 制）"), ("10%", "2015-06-01 起（10% 制）")):
        for key, name in (("primary_by_regime", "主定義"), ("high_volume_by_regime", "量大（≥1.0）"), ("all_big_up_by_regime", "漲 ≥4% 全部")):
            g = r[key][k]
            cells = [("–" if not g["by_horizon"][str(h)].get("n") else
                      f"{g['by_horizon'][str(h)]['up']}/{g['by_horizon'][str(h)]['n']}，{pct(g['by_horizon'][str(h)]['median_pct'])}") for h in HORIZONS]
            s.append(f"| {lab} | {name} | {g['n']} | {cells[0]} | {cells[1]} | {cells[2]} |")
        bb = r["baseline_by_regime"][k]
        s.append(f"| {lab} | 平常日 | – | " + " | ".join(f"{bb[str(h)]['up_pct']}%（n={bb[str(h)]['n']:,}），{pct(bb[str(h)]['median_pct'])}" for h in HORIZONS) + " |")
    s.append("\n時期按事件當天（平常日按起算日）是否在 2015-06-01 以前分。\n")

    for key, title in (("high_volume", f"對照一：漲 ≥ {R}% 且量比 ≥ {HIGH_RATIO}（量大的大漲）"), ("all_big_up", f"對照二：漲 ≥ {R}% 全部")):
        g = G[key]
        s.append(f"## {title}（{g['n']} 次、{g['episodes']} 段行情）\n")
        s.append(group_table(g, base) + "\n")
    s.append(f"漲 ≥ {R}% 的日子按漲跌幅制分：7% 制（2015-06-01 以前）{r['events_all_big_up_count_by_regime']['7%']} 次，10% 制 {r['events_all_big_up_count_by_regime']['10%']} 次。\n")

    s.append("## 量小和量大的中位數差多少（解析度）\n")
    s.append("| 之後 | 主定義中位數 − 量大中位數 | 差的 bootstrap 95% 區間 | 能分辨的最小差距（區間半寬） | 差距 ÷ 最小差距 |\n|---|---|---|---|---|")
    for h in HORIZONS:
        d = r["median_difference_primary_vs_high_volume"][str(h)]
        s.append(f"| {h} 日 | {d['primary_median_minus_high_volume_median_pct']:+.2f} 個百分點 | {ci(d['boot95'])} 個百分點 | {d['smallest_resolvable_pct']:.2f} 個百分點 | {d['effect_to_bound']:.2f} 倍 |")
    s.append("\n區間包含 0，就表示用這麼少的次數，分不出量小和量大哪一組比較好；「差距 ÷ 最小差距」小於 1 倍，代表看到的差距還在這份樣本自己的雜訊範圍內。\n")
    s.append("| 之後 | 主定義上漲比例（n） | 平常日上漲比例 | 差（百分點） | Wilson 區間半寬 | 差 ÷ 半寬 | 平常日在 Wilson 區間內 |\n|---|---|---|---|---|---|---|")
    for h in HORIZONS:
        u = r["up_share_primary_vs_baseline"][str(h)]
        s.append(f"| {h} 日 | {u['primary_up_pct']}%（{G['primary']['by_horizon'][str(h)]['n']}） | {u['baseline_up_pct']}% | {u['effect_pp']:+.2f} | {u['smallest_resolvable_pp']:.2f} | {u['effect_to_bound']:.2f} 倍 | {'是' if u['baseline_inside_wilson'] else '否'} |")
    s.append("")

    s.append("## 敏感度表（換門檻後的結果）\n")
    s.append("每格：次數（段數）；之後 h 日：上漲次數／有資料的次數，中位數。")
    s.append("\n| 漲幅 ≥ | 量比 | 次數（段數） | 5 日 上漲／n，中位數 | 20 日 上漲／n，中位數 | 20 日上漲比例 Wilson 95% | 60 日 上漲／n，中位數 | 20 日內最低 中位數 | 20 日內低 10% 以上 |")
    s.append("|---|---|---|---|---|---|---|---|---|")
    for x in r["sensitivity"]:
        cells = []
        for h in HORIZONS:
            cells.append("–" if not x[f"n{h}"] else f"{x[f'up{h}']}/{x[f'n{h}']}，{pct(x[f'median{h}_pct'])}")
        w = x["up20_wilson95"]
        s.append(f"| {x['ret_min_pct']:.0f}% | {x['ratio']} | {x['n']}（{x['episodes']}） | {cells[0]} | {cells[1]} | "
                 f"{'–' if not w else f'{w[0]}–{w[1]}%'} | {cells[2]} | {pct(x['dd20_median_pct'])} | {x['dd20_at_least_10pct_below'] if x['dd20_at_least_10pct_below'] is not None else '–'} |")
    s.append(f"\n平常日（任何一天）：5／20／60 日上漲比例 {base['5']['up_pct']}%／{base['20']['up_pct']}%／{base['60']['up_pct']}%，"
             f"中位數 {pct(base['5']['median_pct'])}／{pct(base['20']['median_pct'])}／{pct(base['60']['median_pct'])}（n＝{base['5']['n']:,}／{base['20']['n']:,}／{base['60']['n']:,}）。\n")

    mi = now["mi_index"]
    s.append(f"## {now['date']}（星期{now['weekday_zh']}）這一天：不進統計\n")
    s.append(f"- 加權指數收 {now['close']:,.2f}，漲 {now['points']:+,.2f} 點（{pct(now['ret_pct'])}，前一日 {now['prev_close']:,.2f}）。"
             f"在 {now['return_rank_since']} 以來 {cov['sessions'] - 1:,} 個交易日的單日漲幅中排第 {now['return_rank']}。")
    s.append("- 單日漲幅前五名：" + "；".join(f"{t['rank']}. {t['date']} {pct(t['ret_pct'])}" for t in now["top5_returns"]) + "。")
    s.append(f"- 成交金額 {now['value_twd']:,} 元；前 {LOOKBACK} 個交易日（{now['avg20_window'][0]} 到 {now['avg20_window'][1]}）平均 {now['avg20_value_twd']:,} 元；"
             f"量比 {now['value_ratio']:.4f}。同樣算法的成交股數量比 {now['shares_ratio']:.2f}、成交筆數量比 {now['trades_ratio']:.2f}。"
             f"分母換成前 60、120 個交易日平均：{now['value_ratio_60']:.3f}、{now['value_ratio_120']:.3f} 倍。"
             + (f"成交金額是 {now['lowest_value_since']['date']}（{now['lowest_value_since']['value_twd']:,} 元）以來最低。" if now["lowest_value_since"] else ""))
    ss = now["shares_split"]
    s.append(f"- 成交股數 {now['volume_shares']:,} 股（{now['volume_shares'] / 1e8:.1f} 億股），其中 ETF {mi['shares_etf']:,} 股（{mi['shares_etf'] / 1e8:.1f} 億股，{ss['etf_pct']:.1f}%）、"
             f"一般股票 {mi['shares_common_stocks']:,} 股（{mi['shares_common_stocks'] / 1e8:.1f} 億股，{ss['common_stocks_pct']:.1f}%）。"
             f"成交金額裡 ETF 占 {100 * mi['value_etf_twd'] / now['value_twd']:.1f}%、一般股票占 {100 * mi['value_common_stocks_twd'] / now['value_twd']:.1f}%。股數量比接近 1、金額量比只有 0.44，所以要說「成交金額」，不要說「成交量」。")
    lr = now["big_up_days_with_lower_or_equal_ratio"]
    s.append(f"- 歷史上漲 ≥ {R}% 的 {G['all_big_up']['n']} 天裡，量比最低的是 {now['big_up_days_ratio_min']['date']}（{now['big_up_days_ratio_min']['value_ratio']:.2f}）；"
             + ("量比 ≤ 今天的有：" + "、".join(lr) if lr else "沒有一天量比和今天一樣低或更低") + "。")
    s.append(f"- 證交所 MI_INDEX「漲跌證券數合計」股票欄：上漲 {mi['stocks_up']:,} 檔（其中漲停 {mi['stocks_limit_up']}）、下跌 {mi['stocks_down']} 檔（其中跌停 {mi['stocks_limit_down']}）、"
             f"持平 {mi['stocks_unchanged']}、未成交 {mi['stocks_no_trade']}、無比價 {mi['stocks_no_comparable_price']}。"
             f"同一份資料的大盤統計：一般股票成交金額 {mi['value_common_stocks_twd']:,} 元、ETF {mi['value_etf_twd']:,} 元。")
    s.append(f"- {r['inference_not_tested']}")
    s.append(f"- 今天符合主定義：{'是' if now['qualifies_primary'] else '否'}；但之後的走勢還不存在，上面的歷史數字不是對今天的預測。\n")

    s.append("## 研究前探索數字的重算（2026-11-24 19:00 派工時列出）\n")
    s.append("| 探索時寫的 | 這次重算 | 一致 |\n|---|---|---|")
    s += [f"| {x['claim']} | {x['recomputed']} | {'是' if x['match'] else '**否**'} |" for x in r["work_order_recheck"]]
    s.append("\n不一致的項目以重算為準；重算只用收盤價（FMTQIK 沒有盤中最低）。\n")
    s.append("## 漲 ≥ 4% 的日子裡，同一段行情有好幾天的\n")
    s += [f"- " + "、".join(f"{c['date']}（{pct(c['ret_pct'])}，量比 {c['value_ratio']:.2f}）" for c in cl) for cl in r["clusters_all_big_up"]]
    s.append("")
    s.append("## 限制\n")
    s += [f"- {x}" for x in r["limitations"]]
    s.append(f"\n來源：`research/studies/{STUDY_ID}/source_manifest.json`；程式 `scripts/research/build_tx_low_volume_surge.py`；產生時間 {r['generated_at']}。\n")
    return "\n".join(s)


if __name__ == "__main__":
    main()
