#!/usr/bin/env python3
"""RS-XAUUSD-20261116-001 — gold calendar windows by start date and exit date (descriptive grid).

Definitions are the ones registered in the study's decision_log.md (registration commit c2d37f5,
2026-11-16 13:44:27 Asia/Taipei); this runner implements them and nothing else.

  window (start, exit, Y): from a start anchor in Y-1 to an exit in Y, Y = the Lunar New Year (LNY) year
  starts : 10-01, 10-15, 11-01, 11-15, 12-01, 12-15 of Y-1
  exits  : before_lny  = last trading day strictly before LNY
           lny_plus_5  = 5th trading day dated strictly after LNY
           lny_plus_10 = 10th trading day dated strictly after LNY
           feb_end     = last trading day on or before the last day of February of Y
           mar_end     = last trading day on or before 03-31 of Y
  daily rule  (Y = 2009..2026): start = last daily bar dated on or before the anchor; exits on daily bars
  weekly rule (Y = 1981..2008): a bar labelled Monday M closes on Friday F = M + 4; every end is the last
               weekly bar with F on or before the daily-rule date on the Monday-Friday calendar
               (holidays not removed); the lag in weekdays is reported per window and end
  return = end / start - 1; length = bars after the start bar up to the end bar;
  dip = lowest close among those bars / start - 1
  baseline (per cell) = every start bar from the cell's first window start on, held L bars
               (L = the cell's median length, a half rounded up), ending no later than its last end bar
  controls: (12-15, before_lny) and (12-01, before_lny) must reproduce RS-XAUUSD-20261115-001

usage: build_xauusd_start_exit_grid.py [--out-dir research/studies/RS-XAUUSD-20261116-001]
Every input path defaults to the locator in source_manifest.json; a locator 'git <commit>:<path>'
is read with `git show`. Every input's sha256 is checked against the manifest before use.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SID = "RS-XAUUSD-20261116-001"
PUBLISHED = "RS-XAUUSD-20261115-001"
Z = 1.959963984540054
DAILY = range(2009, 2027)
WEEKLY = range(1981, 2009)
FXIDC_YEARS = range(2014, 2027)
STARTS = [("10-01", (10, 1)), ("10-15", (10, 15)), ("11-01", (11, 1)),
          ("11-15", (11, 15)), ("12-01", (12, 1)), ("12-15", (12, 15))]
EXITS = ["before_lny", "lny_plus_5", "lny_plus_10", "feb_end", "mar_end"]
EXIT_ZH = {"before_lny": "春節前最後一個交易日", "lny_plus_5": "春節後第 5 個交易日",
           "lny_plus_10": "春節後第 10 個交易日", "feb_end": "2 月最後一個交易日", "mar_end": "3 月最後一個交易日"}
HALF_MARGIN_DIP = -5.0   # 10x: a close 5% below the start loses half of the margin
WIPE_DIP = -10.0         # 10x: a close 10% below the start loses all of it
LEVERAGE = [1, 2, 10]
LEVERAGE_CELLS = [("12-15", "before_lny"), ("12-15", "mar_end")]
SAMPLES = {"daily_2009_2026": "日線 2009–2026", "weekly_1981_2008": "週線 1981–2008",
           "pooled_46": "合併 46 個冬天（日線 18＋週線 28）"}

# Citations for 「為什麼會有季節性」: not inputs, not computed; see decision_log.md.
CONTEXT_SOURCES = [
    {"topic": "china_lny", "publisher": "World Gold Council (Goldhub, Ray Jia)",
     "url": "https://www.gold.org/goldhub/gold-focus/2026/02/china-gold-market-update-strong-start-2026",
     "published": "2026-02-12", "retrieved": "2026-11-16 13:40 Asia/Taipei",
     "quote": "With the Chinese New Year (CNY) holiday approaching (15 to 23 February), both gold jewellery demand and bullion investment may improve supported by self-rewarding and gifting-related purchases."},
    {"topic": "china_lny", "publisher": "World Gold Council (Goldhub, Ray Jia)",
     "url": "https://www.gold.org/goldhub/gold-focus/2021/03/chinese-gold-consumption-booms-during-2021-chinese-new-year-holiday",
     "published": "2021-03-05", "retrieved": "2026-11-16 13:40 Asia/Taipei",
     "quote": "In addition, gold bars and coins also attracted tremendous attention during the holiday as many retail investors took advantage of the local gold price dip ahead of the festival to enter the market for gifting and long-term saving."},
    {"topic": "india_diwali_weddings", "publisher": "World Gold Council (Goldhub, Kavita Chacko)",
     "url": "https://www.gold.org/goldhub/gold-focus/2025/11/india-gold-market-update-seasonal-strength",
     "published": "2025-11-21", "retrieved": "2026-11-16 13:40 Asia/Taipei",
     "quote": "Jewellers remain cautiously optimistic that the ongoing wedding season (November–March) will boost jewellery sales, given the expected large number of weddings."},
    {"topic": "india_diwali_weddings", "publisher": "World Gold Council (Goldhub, Kavita Chacko)",
     "url": "https://www.gold.org/goldhub/gold-focus/2024/11/indias-gold-market-update-festive-buying-holds-ground-face-high-prices",
     "published": "2024-11-18", "retrieved": "2026-11-16 13:40 Asia/Taipei",
     "quote": "In October, gold imports rose to $7.13bn from $4.39bn in the previous month, reflecting the seasonal uptick in demand for festivals and weddings."},
]


# ---------- inputs ----------
def read_bytes(locator: str) -> bytes:
    if locator.startswith("git "):
        commit, rel = locator[4:].split(":", 1)
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"],
                              capture_output=True, check=True).stdout
    return (ROOT / locator).read_bytes()


def load_inputs(manifest: dict) -> dict:
    out = {}
    for src in manifest["sources"]:
        data = read_bytes(src["locator"])
        got = hashlib.sha256(data).hexdigest()
        if got != src["sha256"] or len(data) != src["size_bytes"]:
            sys.exit(f"{src['role']}: sha256/size mismatch ({got}, {len(data)} bytes) at {src['locator']}")
        out[src["role"]] = data
    return out


def bars(data: bytes, weekly: bool) -> list:
    """[(label_date, close_date, close)] sorted; a duplicated label must carry the same close."""
    seen = {}
    for row in csv.DictReader(io.StringIO(data.decode())):
        d = dt.date.fromisoformat(row["time"][:10])
        c = float(row["close"])
        if d in seen and abs(seen[d] - c) > 1e-9:
            sys.exit(f"duplicated bar {d} with different closes {seen[d]} / {c}")
        seen[d] = c
    rows = []
    for d in sorted(seen):
        if weekly and d.weekday() != 0:
            sys.exit(f"weekly bar {d} is not labelled Monday")
        rows.append((d, d + dt.timedelta(days=4) if weekly else d, seen[d]))
    return rows


def lny_dates(data: bytes) -> dict:
    return {int(r["year"]): dt.date.fromisoformat(r["lunar_new_year"])
            for r in csv.DictReader(io.StringIO(data.decode()))}


# ---------- the weekday calendar (holidays not removed) ----------
def last_wd_on_or_before(d: dt.date) -> dt.date:
    while d.weekday() > 4:
        d -= dt.timedelta(days=1)
    return d


def nth_wd_after(d: dt.date, n: int) -> dt.date:
    k = 0
    while k < n:
        d += dt.timedelta(days=1)
        if d.weekday() <= 4:
            k += 1
    return d


def wd_lag(early: dt.date, target: dt.date) -> int:
    """Monday-Friday dates x with early < x <= target."""
    n, d = 0, early
    while d < target:
        d += dt.timedelta(days=1)
        if d.weekday() <= 4:
            n += 1
    return n


def anchor_date(y: int, md) -> dt.date:
    return dt.date(y - 1, *md)


def exit_target(y: int, lny: dt.date, ex: str) -> dt.date:
    """The daily-rule date on the weekday calendar."""
    if ex == "before_lny":
        return last_wd_on_or_before(lny - dt.timedelta(days=1))
    if ex == "lny_plus_5":
        return nth_wd_after(lny, 5)
    if ex == "lny_plus_10":
        return nth_wd_after(lny, 10)
    if ex == "feb_end":
        return last_wd_on_or_before(dt.date(y, 3, 1) - dt.timedelta(days=1))
    if ex == "mar_end":
        return last_wd_on_or_before(dt.date(y, 3, 31))
    raise ValueError(ex)


# ---------- the rules ----------
def last_on_or_before(rows, d: dt.date):
    idx = None
    for i, r in enumerate(rows):
        if r[1] <= d:
            idx = i
        else:
            break
    return idx


def daily_end(rows, y: int, lny: dt.date, ex: str):
    if ex == "before_lny":
        return last_on_or_before(rows, lny - dt.timedelta(days=1))
    if ex in ("lny_plus_5", "lny_plus_10"):
        n = 5 if ex == "lny_plus_5" else 10
        after = [i for i, r in enumerate(rows) if r[1] > lny]
        return after[n - 1] if len(after) >= n else None
    if ex == "feb_end":
        return last_on_or_before(rows, dt.date(y, 3, 1) - dt.timedelta(days=1))
    return last_on_or_before(rows, dt.date(y, 3, 31))


def window(rows, y: int, lny: dt.date, md, ex: str, weekly: bool):
    anchor = anchor_date(y, md)
    s = last_on_or_before(rows, anchor)
    target = exit_target(y, lny, ex)
    e = last_on_or_before(rows, target) if weekly else daily_end(rows, y, lny, ex)
    if s is None or e is None or e <= s or s == 0 or (anchor - rows[s][1]).days > 7:
        return None
    if not weekly and e == len(rows) - 1 and rows[e][1] < target:
        return None  # the series ends before the exit
    c0, c1 = rows[s][2], rows[e][2]
    inside = [r[2] for r in rows[s + 1:e + 1]]
    low = min(inside)
    w = {
        "year": y, "lunar_new_year": lny.isoformat(),
        "start_close_date": rows[s][1].isoformat(), "start_close": c0,
        "end_close_date": rows[e][1].isoformat(), "end_close": c1,
        "return_pct": (c1 / c0 - 1) * 100, "up": c1 > c0, "length_bars": e - s,
        "dip_pct": (low / c0 - 1) * 100, "dip_close_date": rows[s + 1 + inside.index(low)][1].isoformat(),
        "_s": s, "_e": e,
    }
    if weekly:
        w["lag_start_weekdays"] = wd_lag(rows[s][1], last_wd_on_or_before(anchor))
        w["lag_end_weekdays"] = wd_lag(rows[e][1], target)
    else:
        w["end_on_weekday_target"] = rows[e][1] == target
    return w


def windows(rows, years, lny, md, ex, weekly):
    out, missing = [], []
    for y in years:
        w = window(rows, y, lny[y], md, ex, weekly)
        (out.append(w) if w else missing.append(y))
    return out, missing


# ---------- statistics ----------
def wilson(k: int, n: int) -> list:
    p = k / n
    den = 1 + Z * Z / n
    mid = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [round((mid - half) * 100, 3), round((mid + half) * 100, 3)]


def quantile(xs, q):
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary(ws: list, weekly=None) -> dict:
    r = [w["return_pct"] for w in ws]
    n, k = len(ws), sum(w["up"] for w in ws)
    lo = min(ws, key=lambda w: w["return_pct"])
    hi = max(ws, key=lambda w: w["return_pct"])
    deep = min(ws, key=lambda w: w["dip_pct"])
    lengths = [w["length_bars"] for w in ws]
    out = {
        "n": n, "up": k, "down": n - k, "share_up_pct": round(k / n * 100, 3), "share_up_wilson95_pct": wilson(k, n),
        "mean_pct": st.mean(r), "median_pct": st.median(r), "p25_pct": quantile(r, 0.25), "p75_pct": quantile(r, 0.75),
        "best_pct": hi["return_pct"], "best_year": hi["year"], "worst_pct": lo["return_pct"], "worst_year": lo["year"],
        "deepest_dip_pct": deep["dip_pct"], "deepest_dip_year": deep["year"],
        "dip_deeper_than_5pct": sum(w["dip_pct"] <= HALF_MARGIN_DIP for w in ws),
        "dip_deeper_than_5pct_years": [w["year"] for w in ws if w["dip_pct"] <= HALF_MARGIN_DIP],
        "median_length_bars": st.median(lengths), "min_length_bars": min(lengths), "max_length_bars": max(lengths),
        "down_years": [w["year"] for w in ws if not w["up"]],
    }
    if weekly:
        out["max_lag_start_weekdays"] = max(w["lag_start_weekdays"] for w in ws)
        out["max_lag_end_weekdays"] = max(w["lag_end_weekdays"] for w in ws)
    return out


def baseline(rows, ws: list) -> dict:
    hold = math.floor(st.median(w["length_bars"] for w in ws) + 0.5)
    first, last = min(w["_s"] for w in ws), max(w["_e"] for w in ws)
    rets = [(rows[i + hold][2] / rows[i][2] - 1) * 100 for i in range(first, last - hold + 1)]
    return {"hold_bars": hold, "start_bars": len(rets),
            "first_start": rows[first][1].isoformat(), "last_start": rows[last - hold][1].isoformat(),
            "share_up_pct": round(sum(x > 0 for x in rets) / len(rets) * 100, 3), "median_pct": st.median(rets),
            "note": "every start bar in the period, overlapping; described, no interval"}


def ranks(xs: list) -> list:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list, b: list) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def clean(ws: list) -> list:
    return [{k: v for k, v in w.items() if not k.startswith("_")} for w in ws]


def compare(primary: list, other: list) -> dict:
    p = {w["year"]: w for w in primary}
    o = {w["year"]: w for w in other}
    years = sorted(set(p) & set(o))
    d = [abs(p[y]["return_pct"] - o[y]["return_pct"]) for y in years]
    return {"n": len(years), "up_primary": sum(p[y]["up"] for y in years), "up_other": sum(o[y]["up"] for y in years),
            "sign_differs_years": [y for y in years if p[y]["up"] != o[y]["up"]],
            "mean_abs_diff_pct_points": st.mean(d), "max_abs_diff_pct_points": max(d)}


# ---------- controls ----------
def controls(grid: dict, pub: dict) -> dict:
    out, fail = {}, []
    for key, sample, pub_ws in (("daily", "daily_2009_2026", pub["discovery"]["windows"]),
                                ("weekly", "weekly_1981_2008", pub["validation"]["windows"])):
        mine = {w["year"]: w for w in grid[sample]["12-15"]["before_lny"]["windows"]}
        diffs = [abs(mine[w["year"]]["return_pct"] - w["return_pct"]) for w in pub_ws]
        dates = all(mine[w["year"]]["start_close_date"] == w["start_close_date"]
                    and mine[w["year"]]["end_close_date"] == w["end_close_date"] for w in pub_ws)
        up = grid[sample]["12-15"]["before_lny"]["summary"]["up"]
        pub_up = sum(w["return_pct"] > 0 for w in pub_ws)
        ok = len(mine) == len(pub_ws) and max(diffs) <= 1e-9 and dates and up == pub_up
        out[f"dec15_before_lny_{key}"] = {"n": len(pub_ws), "up": up, "published_up": pub_up,
                                          "max_abs_diff_pct_points": max(diffs), "same_dates": dates, "pass": ok}
        fail += [] if ok else [f"dec15_before_lny_{key}"]
    rb = pub["robustness_start_dec1"]
    for key, sample, pk in (("daily", "daily_2009_2026", "discovery_daily"), ("weekly", "weekly_1981_2008", "validation_weekly")):
        s = grid[sample]["12-01"]["before_lny"]["summary"]
        ok = s["n"] == rb[pk]["n"] and s["up"] == rb[pk]["up"] and abs(s["median_pct"] - rb[pk]["median_pct"]) <= 1e-9
        out[f"dec1_before_lny_{key}"] = {"n": s["n"], "up": s["up"], "published_up": rb[pk]["up"],
                                         "median_pct": s["median_pct"], "published_median_pct": rb[pk]["median_pct"], "pass": ok}
        fail += [] if ok else [f"dec1_before_lny_{key}"]
    pooled = grid["pooled_46"]["12-15"]["before_lny"]["summary"]
    ok = pooled["n"] == 46 and pooled["up"] == 27
    out["dec15_before_lny_pooled"] = {"n": pooled["n"], "up": pooled["up"], "expected": "27 of 46", "pass": ok}
    fail += [] if ok else ["dec15_before_lny_pooled"]
    out["all_pass"] = not fail
    return out, fail


# ---------- leverage and the passbook spread ----------
def leverage(ws: list, k: int) -> dict:
    r = [k * w["return_pct"] for w in ws]
    dips = [k * w["dip_pct"] for w in ws]
    best = max(range(len(ws)), key=lambda i: r[i])
    worst = min(range(len(ws)), key=lambda i: r[i])
    deep = min(range(len(ws)), key=lambda i: dips[i])
    out = {"multiple": k, "n": len(ws), "best_pct": r[best], "best_year": ws[best]["year"],
           "median_pct": st.median(r), "worst_pct": r[worst], "worst_year": ws[worst]["year"],
           "deepest_dip_pct": dips[deep], "deepest_dip_year": ws[deep]["year"]}
    if k == 10:
        capped = [-100.0 if d <= -100 else x for x, d in zip(r, dips)]
        out["half_margin_lost_years"] = [{"year": w["year"], "dip_pct": w["dip_pct"], "dip_close_date": w["dip_close_date"],
                                          "levered_dip_pct": k * w["dip_pct"]} for w in ws if k * w["dip_pct"] <= -50]
        out["margin_gone_years"] = [w["year"] for w in ws if k * w["dip_pct"] <= -100]
        out["capped"] = {"what": "a year whose 10x dip reaches -100% is set to -100% (the position would not reach the end)",
                         "median_pct": st.median(capped), "worst_pct": min(capped),
                         "worst_year": ws[min(range(len(ws)), key=lambda i: capped[i])]["year"]}
    return out


def passbook(data: bytes) -> dict:
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    sp = {r["date"]: 1 - int(r["bank_buys_twd_per_g"]) / int(r["bank_sells_twd_per_g"]) for r in rows}
    today = [r for r in rows if r["date"] == "2026-11-16"][0]
    vals = sorted(sp.values())
    return {"quote_date": "2026-11-16", "quote_time": "13:29 (live page 掛牌時間)",
            "bank_buys_twd_per_g": int(today["bank_buys_twd_per_g"]), "bank_sells_twd_per_g": int(today["bank_sells_twd_per_g"]),
            "spread_pct": sp["2026-11-16"] * 100,
            "history_rows": len(rows), "history_from": rows[0]["date"], "history_to": rows[-1]["date"],
            "history_spread_min_pct": vals[0] * 100, "history_spread_median_pct": st.median(vals) * 100,
            "history_spread_max_pct": vals[-1] * 100,
            "source": "https://rate.bot.com.tw/gold?Lang=zh-TW (live) and https://rate.bot.com.tw/gold/chart/year/TWD/?Lang=zh-TW (history), retrieved 2026-11-16 13:39-13:45 Asia/Taipei"}


def net_of_spread(ws: list, s: float) -> dict:
    net = [((1 + w["return_pct"] / 100) * (1 - s) - 1) * 100 for w in ws]
    best = max(range(len(ws)), key=lambda i: net[i])
    worst = min(range(len(ws)), key=lambda i: net[i])
    return {"n": len(ws), "net_up": sum(x > 0 for x in net), "gross_up": sum(w["up"] for w in ws),
            "turned_down_years": [w["year"] for w, x in zip(ws, net) if w["up"] and x <= 0],
            "net_median_pct": st.median(net), "gross_median_pct": st.median(w["return_pct"] for w in ws),
            "net_best_pct": net[best], "net_best_year": ws[best]["year"],
            "net_worst_pct": net[worst], "net_worst_year": ws[worst]["year"]}


# ---------- README (generated, so it cannot disagree with results.json) ----------
def f2(x):
    return "—" if x is None else f"{x:+.2f}%"


def wl(i):
    return f"{i[0]:.1f}–{i[1]:.1f}%"


def readme(r: dict) -> str:
    g, c = r["grid"], r["controls"]
    L = [f"# {SID} — 黃金：起點 × 出場（6 個起點 × 5 個出場）", "",
         "由 `scripts/research/build_xauusd_start_exit_grid.py` 產生，數字全部來自同資料夾的 `results.json`；不要手改。",
         f"定義事先登記（{r['registration']}），本頁產生時間 {r['generated_at']}。", "",
         "## 這份研究在問什麼", "",
         "前一年 10/1、10/15、11/1、11/15、12/1、12/15 收盤（遇假日用前一個交易日）買黃金，在春節前最後一個交易日、春節後第 5、第 10 個交易日、"
         "2 月最後一個交易日、3 月最後一個交易日收盤賣，每一個組合（「格」）歷史上漲幾次、最好、中位數、最差、中途最深各是多少。"
         "2009–2026 用日線、1981–2008 用週線，再合起來看 46 個冬天。這是描述，不檢驗哪一格比較好，也不挑一格當結論。", "",
         "## 對照（和已發布的研究一致才往下看）", "",
         "| 對照 | 這次 | 已發布 | 一致 |", "|---|---|---|---|"]
    for k, v in c.items():
        if k == "all_pass":
            continue
        L.append(f"| {k} | {v['up']}／{v['n']} | {v.get('published_up', v.get('expected'))} | {'是' if v['pass'] else '否'} |")
    L += ["", f"已發布研究：{PUBLISHED}（12/15 × 春節前，與 12/1 × 春節前）。", ""]
    for sample, zh in SAMPLES.items():
        L += [f"## {zh}：漲幾次／中位數", "", "| 起點 | " + " | ".join(EXIT_ZH[e] for e in EXITS) + " |", "|---" * (len(EXITS) + 1) + "|"]
        for sk, _ in STARTS:
            cells = []
            for e in EXITS:
                s = g[sample][sk][e]["summary"]
                cells.append(f"{s['up']}／{s['n']}，{f2(s['median_pct'])}")
            L.append(f"| {sk} | " + " | ".join(cells) + " |")
        L += ["", "| 起點 → 出場 | n | 漲（Wilson 95%） | 最好（年） | 中位數 | 最差（年） | 中途最深（年） | 中途跌 5% 以上 | 長度中位數"
              + ("" if sample == "pooled_46" else " | 基準：任一天起抱同樣長度") + (" | 每一端最多比日線規則早幾個交易日（起點／終點）" if sample == "weekly_1981_2008" else "") + " |",
              "|---|---|---|---|---|---|---|---|---" + ("" if sample == "pooled_46" else "|---") + ("|---" if sample == "weekly_1981_2008" else "") + "|"]
        for sk, _ in STARTS:
            for e in EXITS:
                cell = g[sample][sk][e]
                s = cell["summary"]
                row = (f"| {sk} → {EXIT_ZH[e]} | {s['n']} | {s['up']}（{wl(s['share_up_wilson95_pct'])}） | {f2(s['best_pct'])}（{s['best_year']}） | "
                       f"{f2(s['median_pct'])} | {f2(s['worst_pct'])}（{s['worst_year']}） | {f2(s['deepest_dip_pct'])}（{s['deepest_dip_year']}） | "
                       f"{s['dip_deeper_than_5pct']} | {s['median_length_bars']:g}")
                if sample != "pooled_46":
                    b = cell["baseline"]
                    row += f" | 抱 {b['hold_bars']}：{b['share_up_pct']:.1f}% 漲、中位數 {f2(b['median_pct'])}"
                if sample == "weekly_1981_2008":
                    row += f" | {s['max_lag_start_weekdays']}／{s['max_lag_end_weekdays']}"
                L.append(row + " |")
        L += ["", "長度單位：日線是交易日，週線是週；合併的長度混了兩種單位，只當參考。" if sample == "pooled_46" else
              ("長度單位是交易日。" if sample.startswith("daily") else "長度單位是週；起點、終點收盤日是那一週的週五（週一標籤＋4 天），每一端只會比日線規則早、不會晚。"), ""]
    x = r["cross_cell"]
    L += ["## 格與格（描述，不是檢驗）", ""]
    for sample in ("daily_2009_2026", "weekly_1981_2008", "pooled_46"):
        rk = x["rank_of_dec15_before_lny"][sample]
        L.append(f"- {SAMPLES[sample]}：12/15 × 春節前在 30 格裡，漲的比例比它高的 {rk['share_up']['above']} 格、一樣 {rk['share_up']['equal']} 格、比它低 {rk['share_up']['below']} 格；"
                 f"中位數比它高的 {rk['median']['above']} 格。")
    for sample in ("daily_2009_2026", "weekly_1981_2008"):
        t = x["top_cells"][sample]
        L.append(f"- {SAMPLES[sample]} 漲的比例最高：" + "、".join(f"{a['start']} → {EXIT_ZH[a['exit']]}（{a['up']}／{a['n']}；另一段 {a['other_up']}／{a['other_n']}）" for a in t["share_up"])
                 + f"；中位數最高：" + "、".join(f"{a['start']} → {EXIT_ZH[a['exit']]}（{f2(a['median_pct'])}；另一段 {f2(a['other_median_pct'])}）" for a in t["median"]) + "。")
    ph = r["post_hoc_baseline_check"]["by_sample"]
    for sample, v in ph.items():
        L.append(f"- 事後補算、不能當結論：{SAMPLES[sample]} 30 格裡，漲的比例高過自己基準的 {v['share_up_above_baseline']} 格；Wilson 下緣也高過基準的 {v['lower_bound_above_baseline']} 格"
                 + ("（" + "、".join(f"{a['start']} → {EXIT_ZH[a['exit']]} {a['up']}／{a['n']}，下緣 {a['wilson_low_pct']:.1f}% 對基準 {a['baseline_share_up_pct']:.1f}%；另一段 {a['other_up']}／{a['other_n']}，基準 {a['other_baseline_share_up_pct']:.1f}%" for a in v["which"]) + "）" if v["which"] else "")
                 + "。30 格彼此重疊、沒有做多重比較校正，所以這只是描述。")
    L += [f"- 兩段時期 30 格的排名相關（Spearman）：中位數 {x['spearman_medians']:+.2f}、漲的比例 {x['spearman_share_up']:+.2f}。30 格共用同一批年份、天數重疊，不給區間。", ""]
    L += ["## 放大倍數（只看收盤；全額 k 倍、保證金＝名目的 1/k；不含利息、手續費、價差）", "",
          "| 格 | 樣本 | 倍數 | 最好（年） | 中位數 | 最差（年） | 中途最深（年） | 10 倍：中途虧掉保證金一半以上的年份 | 10 倍：保證金虧光的年份 |", "|---|---|---|---|---|---|---|---|---|"]
    for ck, cell in r["leverage"].items():
        for sample, rows in cell.items():
            for lv in rows:
                half = "、".join(f"{h['year']}（金價 {f2(h['dip_pct'])}，10 倍 {f2(h['levered_dip_pct'])}）" for h in lv.get("half_margin_lost_years", [])) if lv["multiple"] == 10 else ""
                gone = "、".join(map(str, lv.get("margin_gone_years", []))) if lv["multiple"] == 10 else ""
                L.append(f"| {ck} | {SAMPLES[sample]} | {lv['multiple']} | {f2(lv['best_pct'])}（{lv['best_year']}） | {f2(lv['median_pct'])} | "
                         f"{f2(lv['worst_pct'])}（{lv['worst_year']}） | {f2(lv['deepest_dip_pct'])}（{lv['deepest_dip_year']}） | {half or ('無' if lv['multiple'] == 10 else '')} | {gone or ('無' if lv['multiple'] == 10 else '')} |")
    L += ["", "2 倍＝金價漲跌 × 2 的近似，不是正二（槓桿 ETF）的真實淨值：正二每天重新調整倍數，結果跟路徑有關，還有管理費、期貨轉倉和台幣匯率。"
          "10 倍只看收盤；盤中最低比收盤更深，實際被追繳或斷頭會更早。保證金虧光的那一年，10 倍的結果另以 −100% 計（`capped`）。", ""]
    p, nb = r["passbook"], r["passbook_net"]
    L += ["## 黃金存摺一買一賣", "",
          f"台灣銀行黃金存摺 {p['quote_date']} {p['quote_time']}：本行賣出 {p['bank_sells_twd_per_g']:,}、本行買進 {p['bank_buys_twd_per_g']:,}（新台幣／1 公克），"
          f"一買一賣差 {p['spread_pct']:.2f}%（＝1 − 買進／賣出）；過去一年 {p['history_rows']} 個營業日介於 {p['history_spread_min_pct']:.2f}% 到 {p['history_spread_max_pct']:.2f}%（中位數 {p['history_spread_median_pct']:.2f}%）。", "",
          "| 12/15 × 春節前 | n | 毛漲幾次 | 扣價差後漲幾次 | 毛中位數 | 扣價差後中位數 | 扣價差後最好（年） | 扣價差後最差（年） | 扣價差後由漲變跌 |", "|---|---|---|---|---|---|---|---|---|"]
    for sample, v in nb.items():
        L.append(f"| {SAMPLES[sample]} | {v['n']} | {v['gross_up']} | {v['net_up']} | {f2(v['gross_median_pct'])} | {f2(v['net_median_pct'])} | "
                 f"{f2(v['net_best_pct'])}（{v['net_best_year']}） | {f2(v['net_worst_pct'])}（{v['net_worst_year']}） | {'、'.join(map(str, v['turned_down_years'])) or '無'} |")
    L += ["", "假設：兩端都用 2026-11-16 的買賣比例；金價報酬用 XAUUSD 代替台幣金價（美元兌台幣的變動沒算）。", ""]
    L += ["## 資料檢查", ""]
    dc = r["data_checks"]
    L.append(f"- 日線新舊兩版（2026-11-14 快照對 38dea7b，2009–2026）：{dc['daily_vs_38dea7b']['windows']} 段裡漲跌不同 {len(dc['daily_vs_38dea7b']['sign_differs'])} 段"
             + (("（" + "、".join(f"{a['start']}→{a['exit']} {a['year']}" for a in dc['daily_vs_38dea7b']['sign_differs']) + "）") if dc['daily_vs_38dea7b']['sign_differs'] else "") + "。")
    L.append(f"- Twelve Data 對 TradingView FX_IDC（2014–2026）：{dc['daily_vs_fxidc']['windows']} 段裡漲跌不同 {len(dc['daily_vs_fxidc']['sign_differs'])} 段"
             + (("（" + "、".join(f"{a['start']}→{a['exit']} {a['year']}" for a in dc['daily_vs_fxidc']['sign_differs']) + "）") if dc['daily_vs_fxidc']['sign_differs'] else "") + "。")
    off = dc["daily_end_not_on_weekday_target"]
    L.append(f"- 日線終點不在週一到週五推算的日期上：{off['windows']} 段（{len(off['cases'])} 個終點日，每個起點各一次：" + "、".join(f"{a['exit']} {a['year']}→{a['end_close_date']}" for a in off['cases']) + "；資料沒有那天的日線，規則照定義取前一根）。")
    wr = r["weekly_rule_check_2009_2026"]
    L.append(f"- 週線規則套在 2009–2026（對日線規則）：{wr['windows']} 段裡漲跌不同 {len(wr['sign_differs'])} 段，報酬差平均 {wr['mean_abs_diff_pct_points']:.2f} 個百分點。這是 1981–2008 週線表的粗細。")
    fm = dc["weekly_close_vs_daily_friday"]
    L.append("- 週線收盤對同一個週五的日線收盤（2008 起，按月份）：" + "；".join(f"{m} 月 {v['within_0_1pct']}／{v['n']}" for m, v in fm["by_month"].items()) + "。")
    L += ["", "## 下一段（2027 年春節那一段，只有日期）", "", "| 起點 | 日期 |", "|---|---|"]
    nx = r["next_window"]
    for sk, v in nx["starts"].items():
        L.append(f"| {sk} | {v['date']}（{v['weekday']}） |")
    L += ["", "| 出場 | 日期 |", "|---|---|"]
    for e, v in nx["exits"].items():
        L.append(f"| {EXIT_ZH[e]} | {v['date']}（{v['weekday']}） |")
    L += ["", f"2027 年春節 {nx['lunar_new_year']}（{nx['lunar_new_year_weekday']}）。日期按週一到週五推算，假日沒扣；不算報酬、不預測。", "",
          "## 「為什麼會有季節性」的公開說法（引用，不是本研究的結論）", ""]
    for s in r["context_sources"]:
        L.append(f"- {s['publisher']}，{s['published']}：{s['url']} — “{s['quote']}”")
    L += ["", "這些是世界黃金協會對實體需求的評論，不是價格研究，也沒有說這幾週金價會漲；本研究沒有檢驗它們。", "",
          "## 限制", ""] + [f"- {x}" for x in r["limitations"]] + ["", "## 方法細節", ""] + [f"- `{k}`：{x}" for k, x in r["method"].items()] + ["",
          "## 資料來源（Private）", "",
          "`source_manifest.json`：Twelve Data XAU/USD 日線與週線（2026-11-14 13:16 抓取的唯讀複本）、日線 Private git 38dea7b 舊版、TradingView FX_IDC:XAUUSD 日線匯出（2026-11-04 凍結）、"
          "香港天文台公曆農曆對照表 1981–2027、已發布研究 RS-XAUUSD-20261115-001 的 results.json（對照用）、台灣銀行黃金存摺牌價（2026-11-16 抄錄）。每個檔案的 SHA-256 在執行前核對。", ""]
    return "\n".join(L)


# ---------- main ----------
def main() -> None:
    public_inputs = Path(__file__).resolve().parent / "reproduction_inputs.json"
    public = public_inputs.is_file()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("reproduced") if public else Path("research/studies") / SID)
    a = ap.parse_args()
    out_dir = a.out_dir if a.out_dir.is_absolute() else ROOT / a.out_dir
    if public:
        m = json.loads(public_inputs.read_text())
        manifest = {"sources": [{"role": s["role"], "locator": path, "sha256": s["sha256"],
                                 "size_bytes": s["size_bytes"]} for path, s in m["inputs"].items()]}
    else:
        manifest = json.loads((Path("local-inputs") / SID / "source_manifest.json").read_text())
    raw = load_inputs(manifest)

    lny = lny_dates(raw["calendar_lunar_new_year"])
    daily = bars(raw["price_1d_xauusd"], weekly=False)
    weekly = bars(raw["price_1wk_xauusd"], weekly=True)
    daily_old = bars(raw["price_1d_xauusd_v38dea7b"], weekly=False)
    fxidc = bars(raw["price_1d_fxidc_frozen"], weekly=False)
    pub = json.loads(raw["published_results_rs_xauusd_20261115_001"])

    grid = {k: {} for k in SAMPLES}
    raw_ws = {}
    missing = {}
    for sk, md in STARTS:
        for k in SAMPLES:
            grid[k][sk] = {}
        for e in EXITS:
            dws, dm = windows(daily, DAILY, lny, md, e, weekly=False)
            wws, wm = windows(weekly, WEEKLY, lny, md, e, weekly=True)
            if dm or wm:
                missing[f"{sk}|{e}"] = {"daily": dm, "weekly": wm}
            raw_ws[(sk, e)] = (dws, wws)
            grid["daily_2009_2026"][sk][e] = {"summary": summary(dws), "baseline": baseline(daily, dws), "windows": clean(dws)}
            grid["weekly_1981_2008"][sk][e] = {"summary": summary(wws, weekly=True), "baseline": baseline(weekly, wws), "windows": clean(wws)}
            grid["pooled_46"][sk][e] = {"summary": summary(dws + wws),
                                        "note": "daily 2009-2026 and weekly 1981-2008 windows together; lengths mix trading days and weeks"}
    if missing:
        sys.exit(f"windows that cannot be built: {missing}")

    ctrl, fail = controls(grid, pub)
    if fail:
        print(json.dumps(ctrl, ensure_ascii=False, indent=1))
        sys.exit(f"control failed: {fail} — study blocked, nothing else is read")

    # cross-cell description
    cells = [(sk, e) for sk, _ in STARTS for e in EXITS]

    def val(sample, key):
        return [grid[sample][sk][e]["summary"][key] for sk, e in cells]

    def rank_of(sample, key):
        xs = val(sample, key)
        me = grid[sample]["12-15"]["before_lny"]["summary"][key]
        return {"above": sum(x > me for x in xs), "equal": sum(x == me for x in xs) - 1, "below": sum(x < me for x in xs)}

    def top(sample, other, key):
        xs = val(sample, key)
        best = max(xs)
        out = []
        for (sk, e), x in zip(cells, xs):
            if x == best:
                s, o = grid[sample][sk][e]["summary"], grid[other][sk][e]["summary"]
                out.append({"start": sk, "exit": e, "up": s["up"], "n": s["n"], "share_up_pct": s["share_up_pct"],
                            "median_pct": s["median_pct"], "other_up": o["up"], "other_n": o["n"],
                            "other_share_up_pct": o["share_up_pct"], "other_median_pct": o["median_pct"]})
        return out

    cross = {
        "rank_of_dec15_before_lny": {s: {"share_up": rank_of(s, "share_up_pct"), "median": rank_of(s, "median_pct")} for s in SAMPLES},
        "top_cells": {"daily_2009_2026": {"share_up": top("daily_2009_2026", "weekly_1981_2008", "share_up_pct"),
                                          "median": top("daily_2009_2026", "weekly_1981_2008", "median_pct")},
                      "weekly_1981_2008": {"share_up": top("weekly_1981_2008", "daily_2009_2026", "share_up_pct"),
                                           "median": top("weekly_1981_2008", "daily_2009_2026", "median_pct")}},
        "spearman_medians": spearman(val("daily_2009_2026", "median_pct"), val("weekly_1981_2008", "median_pct")),
        "spearman_share_up": spearman(val("daily_2009_2026", "share_up_pct"), val("weekly_1981_2008", "share_up_pct")),
        "note": "30 cells share years and overlapping days: descriptive, no interval, not a test",
    }

    # 事後補算 (added after the first run; not registered, not a verdict): cells whose Wilson lower bound
    # clears their own baseline, with the same cell's numbers in the other sample
    post_hoc = {"label": "事後補算，不能當結論: 30 overlapping cells, no family-wide correction; not registered", "by_sample": {}}
    for sample, other in (("daily_2009_2026", "weekly_1981_2008"), ("weekly_1981_2008", "daily_2009_2026")):
        hits = []
        for sk, e in cells:
            c, o = grid[sample][sk][e], grid[other][sk][e]
            if c["summary"]["share_up_wilson95_pct"][0] > c["baseline"]["share_up_pct"]:
                hits.append({"start": sk, "exit": e, "up": c["summary"]["up"], "n": c["summary"]["n"],
                             "wilson_low_pct": c["summary"]["share_up_wilson95_pct"][0], "baseline_share_up_pct": c["baseline"]["share_up_pct"],
                             "other_up": o["summary"]["up"], "other_n": o["summary"]["n"], "other_baseline_share_up_pct": o["baseline"]["share_up_pct"]})
        post_hoc["by_sample"][sample] = {"cells": 30, "lower_bound_above_baseline": len(hits), "which": hits,
                                         "share_up_above_baseline": sum(grid[sample][sk][e]["summary"]["share_up_pct"] > grid[sample][sk][e]["baseline"]["share_up_pct"] for sk, e in cells)}

    # weekly-rule check and data checks
    wr_diffs, wr_sign, wr_n, wr_cells = [], [], 0, {}
    dv_old, dv_fx = {"windows": 0, "sign_differs": []}, {"windows": 0, "sign_differs": []}
    for sk, md in STARTS:
        for e in EXITS:
            dws = raw_ws[(sk, e)][0]
            wk, wm = windows(weekly, DAILY, lny, md, e, weekly=True)
            c = compare(dws, wk)
            wr_cells[f"{sk}|{e}"] = c
            wr_n += c["n"]
            wr_sign += [{"start": sk, "exit": e, "year": y} for y in c["sign_differs_years"]]
            wk_by_year = {w["year"]: w for w in wk}
            wr_diffs += [abs(p["return_pct"] - wk_by_year[p["year"]]["return_pct"]) for p in dws if p["year"] in wk_by_year]
            old, _ = windows(daily_old, DAILY, lny, md, e, weekly=False)
            c2 = compare(dws, old)
            dv_old["windows"] += c2["n"]
            dv_old["sign_differs"] += [{"start": sk, "exit": e, "year": y} for y in c2["sign_differs_years"]]
            fx, fm_ = windows(fxidc, FXIDC_YEARS, lny, md, e, weekly=False)
            c3 = compare([w for w in dws if w["year"] in FXIDC_YEARS], fx)
            dv_fx["windows"] += c3["n"]
            dv_fx["sign_differs"] += [{"start": sk, "exit": e, "year": y} for y in c3["sign_differs_years"]]
            if fm_:
                dv_fx.setdefault("missing", []).append({"start": sk, "exit": e, "years": fm_})
    wr = {"windows": wr_n, "sign_differs": wr_sign, "mean_abs_diff_pct_points": st.mean(wr_diffs),
          "max_abs_diff_pct_points": max(wr_diffs), "per_cell": wr_cells}
    dmap = {r[1]: r[2] for r in daily}
    by_month = {}
    for _, f, c in weekly:
        if f < dt.date(2008, 1, 14) or f > daily[-1][1]:
            continue
        m = by_month.setdefault(f.month, {"n": 0, "within_0_1pct": 0, "no_daily_bar": 0, "max_abs_diff_pct": 0.0, "max_date": None})
        if f not in dmap:
            m["no_daily_bar"] += 1
            continue
        d = abs(c / dmap[f] - 1) * 100
        m["n"] += 1
        m["within_0_1pct"] += d <= 0.1
        if d > m["max_abs_diff_pct"]:
            m["max_abs_diff_pct"], m["max_date"] = d, f.isoformat()
    off_cases = sorted({(e, w["year"], w["end_close_date"]) for sk, _ in STARTS for e in EXITS
                        for w in raw_ws[(sk, e)][0] if not w["end_on_weekday_target"]})
    off = {"windows": sum(not w["end_on_weekday_target"] for sk, _ in STARTS for e in EXITS for w in raw_ws[(sk, e)][0]),
           "cases": [{"exit": e, "year": y, "end_close_date": d} for e, y, d in off_cases],
           "label": "事後補看 (data note, changes no number): the daily file has no bar on these dates' targets"}
    data_checks = {"daily_end_not_on_weekday_target": off, "daily_vs_38dea7b": dv_old, "daily_vs_fxidc": dv_fx,
                   "weekly_close_vs_daily_friday": {"from": "2008-01-18", "to": daily[-1][1].isoformat(),
                                                    "by_month": {str(k): by_month[k] for k in sorted(by_month)}}}

    # leverage and passbook
    lev = {}
    for sk, e in LEVERAGE_CELLS:
        dws, wws = raw_ws[(sk, e)]
        lev[f"{sk}|{e}"] = {s: [leverage(ws, k) for k in LEVERAGE] for s, ws in
                            (("daily_2009_2026", dws), ("weekly_1981_2008", wws), ("pooled_46", dws + wws))}
    pb = passbook(raw["bot_gold_passbook_twd"])
    dws, wws = raw_ws[("12-15", "before_lny")]
    pb_net = {s: net_of_spread(ws, pb["spread_pct"] / 100) for s, ws in
              (("daily_2009_2026", dws), ("weekly_1981_2008", wws), ("pooled_46", dws + wws))}

    # next window: dates only
    l27 = lny[2027]
    nxt = {"year": 2027, "lunar_new_year": l27.isoformat(), "lunar_new_year_weekday": l27.strftime("%A"),
           "starts": {sk: {"date": last_wd_on_or_before(anchor_date(2027, md)).isoformat(),
                           "weekday": last_wd_on_or_before(anchor_date(2027, md)).strftime("%A")} for sk, md in STARTS},
           "exits": {e: {"date": exit_target(2027, l27, e).isoformat(), "weekday": exit_target(2027, l27, e).strftime("%A")} for e in EXITS},
           "note": "weekday calendar, holidays not removed; dates only, no return, no forecast"}

    results = {
        "study_id": SID, "schema_version": 1, "revision": 1,
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "runner": "scripts/research/build_xauusd_start_exit_grid.py",
        "registration": "registered definitions committed 2026-11-16 13:44:27 Asia/Taipei (commit c2d37f5)",
        "instrument": "XAUUSD spot (Twelve Data XAU/USD)",
        "strategy_version": None,
        "method": {
            "window": "from a start anchor in Y-1 to an exit in Y, named by Y = the Lunar New Year year",
            "starts": "10-01, 10-15, 11-01, 11-15, 12-01, 12-15 of Y-1; the close of the last bar on or before the anchor",
            "exits": "before_lny = last trading day strictly before LNY; lny_plus_5 / lny_plus_10 = the 5th / 10th trading day dated strictly after LNY; feb_end = last trading day on or before the last day of February; mar_end = last trading day on or before 03-31",
            "lunar_new_year": "Hong Kong Observatory conversion table: the Jan-Feb date labelled 1st Lunar month",
            "daily_rule": "Y = 2009..2026 on daily bars; trading day = a daily bar of the file",
            "weekly_rule": "Y = 1981..2008; a weekly bar labelled Monday M closes on Friday M+4; each end = the last weekly bar whose Friday is on or before the daily-rule date on the Monday-Friday calendar (holidays not removed); lag_start_weekdays / lag_end_weekdays = Monday-Friday dates after the weekly close up to that date",
            "return": "end close / start close - 1, in percent",
            "length": "bars after the start bar up to and including the end bar (daily: trading days; weekly: weeks)",
            "dip": "lowest close among those bars / start close - 1 (closes only)",
            "share_up": "share of windows with end close > start close; Wilson 95% interval, z = 1.96",
            "quantiles": "linear interpolation between order statistics",
            "baseline": "per cell and sample: every start bar from the cell's first window start bar on, held L bars (L = the cell's median length, a half rounded up), the hold ending no later than the cell's last end bar; overlapping, no interval",
            "pooled": "the daily 2009-2026 and weekly 1981-2008 windows of the same cell together (n = 46)",
            "cross_cell": "rank of (12-15, before_lny) among the 30 cells; the top cells and their other-sample values; Spearman rank correlation (average ranks for ties) of the 30 medians and of the 30 shares up between the two samples; no test",
            "leverage": "k x the XAUUSD return and k x the dip, k = 1, 2, 10; full notional k x the money put up, margin = 1/k of notional, closes only, no interest, financing, fees or spread; 10x half margin lost = 10 x dip <= -50% (dip <= -5%); margin gone = 10 x dip <= -100% (dip <= -10%), capped at -100%",
            "passbook_spread": "s = 1 - bank buys / bank sells on 2026-11-16; net = (1 + r) x (1 - s) - 1 per window, the same ratio at both ends, XAUUSD return for the TWD gold price",
            "guard": "a window is built only if its start close is at most 7 calendar days before the anchor and is not the series' first bar; a daily window whose series ends before the exit is not built",
            "controls": "(12-15, before_lny) per-year returns equal to RS-XAUUSD-20261115-001 within 1e-9 points with the same dates, daily and weekly; (12-01, before_lny) counts and medians equal to its 12-01 robustness; pooled (12-15, before_lny) = 27 of 46",
        },
        "data_period": {"daily": {"from": daily[0][1].isoformat(), "to": daily[-1][1].isoformat(), "bars": len(daily)},
                        "weekly": {"from": weekly[0][0].isoformat(), "to": weekly[-1][0].isoformat(), "bars": len(weekly)}},
        "controls": ctrl,
        "grid": grid,
        "cross_cell": cross,
        "post_hoc_baseline_check": post_hoc,
        "weekly_rule_check_2009_2026": wr,
        "data_checks": data_checks,
        "leverage": lev,
        "passbook": pb,
        "passbook_net": pb_net,
        "next_window": nxt,
        "context_sources": CONTEXT_SOURCES,
        "limitations": [
            "Descriptive only: 30 start/exit cells, no test between them; the cells share years and overlap in days, so the best-looking cell is partly luck and is not a recommendation.",
            "The (12-15, before LNY) and (12-01, before LNY) cells were published before registration; month-level statistics of October to March were known from RS-MULTI-20260901-001.",
            "1981-2008 uses weekly closes: every end can sit some trading days earlier than the daily rule (max_lag_* per cell) and dips between Fridays are not seen.",
            "Pooled 46 mixes daily and weekly windows.",
            "Leverage is k x the close-to-close move: no intraday lows, financing, fees or spread; 2x is not a leveraged ETF's NAV.",
            "The passbook spread is one day's Bank of Taiwan quote applied at both ends, and the TWD gold price is replaced by XAUUSD (USD/TWD ignored); the history rows were transcribed from a rendered page, not saved byte for byte.",
            "The context sources describe physical demand; none of them is a price study, and this study does not test a cause.",
            "No independent local source checks the weekly closes before 2013.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")
    (out_dir / "README.md").write_text(readme(results))
    print(json.dumps({"controls_pass": ctrl["all_pass"],
                      "dec15_before_lny": {s: {k: grid[s]["12-15"]["before_lny"]["summary"][k] for k in ("up", "n", "median_pct")} for s in SAMPLES}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
