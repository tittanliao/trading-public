#!/usr/bin/env python3
"""RS-XAUUSD-20261115-001 — gold from the mid-December close to the last close before Lunar New Year.

Definitions are the ones registered in the study's decision_log.md (committed 2026-11-15 21:13:51, commit 82f29d5);
this runner implements them and nothing else.

  window Y   = from mid-December of Y-1 to just before Lunar New Year (LNY) of Y
  daily rule : start = close of the last daily bar dated <= 12-15 of Y-1;
               end   = close of the last daily bar dated <  LNY(Y)
  weekly rule: a bar labelled Monday M closes on Friday F = M + 4 days;
               start = last weekly bar with F <= 12-15 of Y-1; end = last weekly bar with F < LNY(Y)
  return = end / start - 1; length = bars after the start bar up to the end bar;
  dip = lowest close among those bars / start - 1
  discovery  = daily rule, Y = 2009..2026 (already seen by EP06 on 2026-11-05: not a test)
  validation = weekly rule, Y = 1981..2008 (registered before computing)
  baseline   = every start bar in the sample period held for L bars, L = median window length
               (a half is rounded up); daily period 2008-12-01 .. the 2026 window's end,
               weekly period F >= 1980-12-01 .. the 2008 window's end
  verdict    = 延續 if the Wilson 95% lower bound of the share up > baseline share up;
               沒有延續 if the share up <= baseline share up; otherwise 分不出

usage: build_xauusd_dec15_lny_window.py [--out-dir research/studies/RS-XAUUSD-20261115-001]
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
SID = "RS-XAUUSD-20261115-001"
Z = 1.959963984540054
DISCOVERY = range(2009, 2027)
VALIDATION = range(1981, 2009)
FXIDC_YEARS = range(2014, 2027)
SUBPERIODS = [(1981, 1990), (1991, 2000), (2001, 2008)]
DIP_LINE = -3.0


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
        loc = src["locator"]
        data = read_bytes(loc)
        got = hashlib.sha256(data).hexdigest()
        if got != src["sha256"] or len(data) != src["size_bytes"]:
            sys.exit(f"{src['role']}: sha256/size mismatch ({got}, {len(data)} bytes) at {loc}")
        out[src["role"]] = data
    return out


def bars(data: bytes, weekly: bool) -> list:
    """[(label_date, close_date, close)] sorted; duplicated labels must carry the same close."""
    seen = {}
    dups = []
    for row in csv.DictReader(io.StringIO(data.decode())):
        d = dt.date.fromisoformat(row["time"][:10])
        c = float(row["close"])
        if d in seen:
            if abs(seen[d] - c) > 1e-9:
                sys.exit(f"duplicated bar {d} with different closes {seen[d]} / {c}")
            dups.append(d.isoformat())
            continue
        seen[d] = c
    rows = []
    for d in sorted(seen):
        if weekly and d.weekday() != 0:
            sys.exit(f"weekly bar {d} is not labelled Monday")
        rows.append((d, d + dt.timedelta(days=4) if weekly else d, seen[d]))
    return rows, dups


def lny_dates(data: bytes) -> dict:
    return {int(r["year"]): dt.date.fromisoformat(r["lunar_new_year"])
            for r in csv.DictReader(io.StringIO(data.decode()))}


# ---------- the rule ----------
def last_index(rows, limit: dt.date, strict: bool) -> int:
    idx = None
    for i, (_, close_date, _) in enumerate(rows):
        if close_date < limit or (not strict and close_date == limit):
            idx = i
        else:
            break
    return idx


def window(rows, y: int, lny: dt.date, start_md=(12, 15), max_gap_days=None) -> dict:
    start_day = dt.date(y - 1, *start_md)
    s = last_index(rows, start_day, strict=False)
    e = last_index(rows, lny, strict=True)
    dec = last_index(rows, dt.date(y - 1, 12, 31), strict=False)
    if s is None or e is None or e <= s:
        return None
    gap = (start_day - rows[s][1]).days
    if max_gap_days is not None and gap > max_gap_days:
        return None  # the series does not reach back to this start
    if s == 0:
        return None  # the start bar is the series' first bar: cannot tell it is the right one
    c0, c1 = rows[s][2], rows[e][2]
    inside = [r[2] for r in rows[s + 1:e + 1]]
    low = min(inside)
    return {
        "year": y,
        "lunar_new_year": lny.isoformat(),
        "start_bar": rows[s][0].isoformat(),
        "start_close_date": rows[s][1].isoformat(),
        "start_close": c0,
        "end_bar": rows[e][0].isoformat(),
        "end_close_date": rows[e][1].isoformat(),
        "end_close": c1,
        "return_pct": (c1 / c0 - 1) * 100,
        "up": c1 > c0,
        "length_bars": e - s,
        "dip_pct": (low / c0 - 1) * 100,
        "dip_close_date": rows[s + 1 + inside.index(low)][1].isoformat(),
        "leg_a_pct": (rows[dec][2] / c0 - 1) * 100 if dec is not None and s < dec < e else None,
        "leg_b_pct": (c1 / rows[dec][2] - 1) * 100 if dec is not None and s < dec < e else None,
        "leg_split_close_date": rows[dec][1].isoformat() if dec is not None else None,
        "_s": s, "_e": e,
    }


def windows(rows, years, lny, start_md=(12, 15), max_gap_days=7):
    out = []
    for y in years:
        w = window(rows, y, lny[y], start_md, max_gap_days)
        if w is None:
            sys.exit(f"window {y} cannot be built from this series")
        out.append(w)
    return out


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


def summary(ws: list) -> dict:
    r = [w["return_pct"] for w in ws]
    k = sum(w["up"] for w in ws)
    n = len(ws)
    lo = min(ws, key=lambda w: w["return_pct"])
    hi = max(ws, key=lambda w: w["return_pct"])
    deep = min(ws, key=lambda w: w["dip_pct"])
    lengths = [w["length_bars"] for w in ws]
    return {
        "n": n, "up": k, "down": n - k,
        "share_up_pct": round(k / n * 100, 3),
        "share_up_wilson95_pct": wilson(k, n),
        "mean_pct": st.mean(r), "median_pct": st.median(r),
        "p25_pct": quantile(r, 0.25), "p75_pct": quantile(r, 0.75),
        "min_pct": lo["return_pct"], "min_year": lo["year"],
        "max_pct": hi["return_pct"], "max_year": hi["year"],
        "dip_deeper_than_3pct": sum(w["dip_pct"] < DIP_LINE for w in ws),
        "dip_deeper_than_3pct_wilson95_pct": wilson(sum(w["dip_pct"] < DIP_LINE for w in ws), n),
        "dip_deeper_than_3pct_years": [w["year"] for w in ws if w["dip_pct"] < DIP_LINE],
        "deepest_dip_pct": deep["dip_pct"], "deepest_dip_year": deep["year"],
        "median_length_bars": st.median(lengths),
        "min_length_bars": min(lengths), "max_length_bars": max(lengths),
        "down_years": [w["year"] for w in ws if not w["up"]],
    }


def baseline(rows, first_close_date: dt.date, last_index_incl: int, hold: int) -> dict:
    starts = [i for i, r in enumerate(rows) if r[1] >= first_close_date and i + hold <= last_index_incl]
    rets = [(rows[i + hold][2] / rows[i][2] - 1) * 100 for i in starts]
    return {
        "hold_bars": hold,
        "start_bars": len(starts),
        "first_start": rows[starts[0]][1].isoformat(),
        "last_start": rows[starts[-1]][1].isoformat(),
        "share_up_pct": round(sum(x > 0 for x in rets) / len(rets) * 100, 3),
        "median_pct": st.median(rets),
        "note": "every start bar in the period, overlapping; described, no interval",
    }


def verdict(summ: dict, base: dict) -> str:
    if summ["share_up_wilson95_pct"][0] > base["share_up_pct"]:
        return "延續"
    if summ["share_up_pct"] <= base["share_up_pct"]:
        return "沒有延續"
    return "分不出"


def hold_of(summ: dict) -> int:
    return math.floor(summ["median_length_bars"] + 0.5)


def legs(ws: list) -> dict:
    a = [w["leg_a_pct"] for w in ws if w["leg_a_pct"] is not None]
    b = [w["leg_b_pct"] for w in ws if w["leg_b_pct"] is not None]
    return {
        "leg_a": {"what": "start -> last close dated on or before 12-31", "n": len(a),
                  "up": sum(x > 0 for x in a), "up_wilson95_pct": wilson(sum(x > 0 for x in a), len(a)),
                  "median_pct": st.median(a)},
        "leg_b": {"what": "last December close -> end", "n": len(b),
                  "up": sum(x > 0 for x in b), "up_wilson95_pct": wilson(sum(x > 0 for x in b), len(b)),
                  "median_pct": st.median(b)},
    }


def short(ws: list) -> dict:
    s = summary(ws)
    return {k: s[k] for k in ("n", "up", "share_up_pct", "share_up_wilson95_pct", "median_pct",
                              "min_pct", "min_year", "max_pct", "max_year", "down_years")}


def compare(primary: list, other: list) -> dict:
    p = {w["year"]: w for w in primary}
    o = {w["year"]: w for w in other}
    years = sorted(set(p) & set(o))
    return {
        "years": [years[0], years[-1]], "n": len(years),
        "up_primary": sum(p[y]["up"] for y in years),
        "up_other": sum(o[y]["up"] for y in years),
        "sign_differs_years": [y for y in years if p[y]["up"] != o[y]["up"]],
        "mean_abs_diff_pct_points": st.mean(abs(p[y]["return_pct"] - o[y]["return_pct"]) for y in years),
        "max_abs_diff_pct_points": max(abs(p[y]["return_pct"] - o[y]["return_pct"]) for y in years),
        "max_abs_diff_year": max(years, key=lambda y: abs(p[y]["return_pct"] - o[y]["return_pct"])),
        "per_year": {str(y): {"primary_pct": p[y]["return_pct"], "other_pct": o[y]["return_pct"],
                              "other_start": o[y]["start_close_date"], "other_end": o[y]["end_close_date"]}
                     for y in years},
    }


def clean(ws: list) -> list:
    return [{k: v for k, v in w.items() if not k.startswith("_")} for w in ws]


# ---------- README (generated, so it cannot disagree with results.json) ----------
def f2(x):
    return "—" if x is None else f"{x:+.2f}%"


def wl(i):
    return f"{i[0]:.1f}–{i[1]:.1f}%"


def need_up(n: int, base_pct: float) -> int:
    """Smallest number up out of n whose Wilson lower bound clears the baseline (post hoc, for the README)."""
    return next(k for k in range(n + 1) if wilson(k, n)[0] > base_pct)


def readme(r: dict) -> str:
    d, v = r["discovery"], r["validation"]
    ds, vs, db, vb = d["summary"], v["summary"], d["baseline"], v["baseline"]
    g, dv, rb = r["weekly_rule_check_2009_2026"], r["data_versions"], r["robustness_start_dec1"]

    def ci(s):
        return f"{s['share_up_wilson95_pct'][0]:.1f}–{s['share_up_wilson95_pct'][1]:.1f}%"

    L = []
    L += [f"# {SID} — 黃金：12 月中收盤到春節前最後一個交易日", "",
          "由 `scripts/research/build_xauusd_dec15_lny_window.py` 產生，數字全部來自同資料夾的 `results.json`；不要手改。",
          f"定義事先登記（2026-11-15 21:13:51 提交 82f29d5，之後才寫程式），本頁產生時間 {r['generated_at']}。", "",
          "## 這份研究在問什麼", "",
          "從前一年 12 月 15 日收盤（遇假日用前一個交易日）買進黃金、抱到農曆春節前最後一個交易日收盤，歷史上漲的比例和幅度是多少；"
          "2009–2026 的日線結果，在 1981–2008 的週線上有沒有延續。",
          "2009–2026 這 18 段在登記前已經被 EP06 影片稿算過（14 次漲），所以只當「發現」、不是檢驗；1981–2008 的 28 段是登記後才第一次算的「驗證」。", "",
          "## 主結果", "",
          "| 樣本 | 年份（春節那年） | 段數 n | 漲 | 漲的比例（Wilson 95%） | 中位數 | 基準：同期任一天起抱同樣長度 | 判定 |",
          "|---|---|---|---|---|---|---|---|",
          f"| 發現（日線，已看過） | 2009–2026 | {ds['n']} | {ds['up']} | {ds['share_up_pct']:.1f}%（{ci(ds)}） | {f2(ds['median_pct'])} | "
          f"抱 {db['hold_bars']} 個交易日：{db['share_up_pct']:.1f}% 漲、中位數 {f2(db['median_pct'])}（{db['start_bars']} 個起點） | {d['verdict_if_it_were_a_test']}（不是檢驗） |",
          f"| 驗證（週線，登記後才算） | 1981–2008 | {vs['n']} | {vs['up']} | {vs['share_up_pct']:.1f}%（{ci(vs)}） | {f2(vs['median_pct'])} | "
          f"抱 {vb['hold_bars']} 週：{vb['share_up_pct']:.1f}% 漲、中位數 {f2(vb['median_pct'])}（{vb['start_bars']} 個起點） | **{v['verdict']}** |", "",
          "判定規則（登記）：漲的比例的 Wilson 95% 下緣高於基準＝延續；漲的比例不高於基準＝沒有延續；其他＝分不出。"
          f"要判定延續，漲的比例大約要比基準高出下緣到點估計的距離：n={ds['n']} 約 {ds['share_up_pct'] - ds['share_up_wilson95_pct'][0]:.0f} 個百分點，"
          f"n={vs['n']} 約 {vs['share_up_pct'] - vs['share_up_wilson95_pct'][0]:.0f} 個百分點（事後補算：驗證要 {vs['n']} 次漲 {need_up(vs['n'], vb['share_up_pct'])} 次，下緣才高過 {vb['share_up_pct']:.1f}%）。"
          "「沒有延續」是說沒有比同期任意抱同樣長度好，不是說比較差。", "",
          "| 樣本 | 平均 | 25 分位 | 75 分位 | 最差 | 最好 | 中間跌破起點 3% 以上（Wilson 95%） | 最深的中間回落 | 每段長度（中位數，最短–最長） |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, s, unit in (("發現 2009–2026", ds, "個交易日"), ("驗證 1981–2008", vs, "週")):
        L.append(f"| {name} | {f2(s['mean_pct'])} | {f2(s['p25_pct'])} | {f2(s['p75_pct'])} | {f2(s['min_pct'])}（{s['min_year']}） | "
                 f"{f2(s['max_pct'])}（{s['max_year']}） | {s['dip_deeper_than_3pct']}／{s['n']} 段（{wl(s['dip_deeper_than_3pct_wilson95_pct'])}） | {f2(s['deepest_dip_pct'])}（{s['deepest_dip_year']}） | "
                 f"{s['median_length_bars']:g} {unit}（{s['min_length_bars']}–{s['max_length_bars']}） |")
    L += ["", "跌的年份（春節那年）：發現 " + "、".join(map(str, ds["down_years"])) + "；驗證 " + "、".join(map(str, vs["down_years"])) + "。", ""]

    L += ["## 每一段", "", "| 春節那年 | 春節 | 起點收盤日 | 起點收盤 | 終點收盤日 | 終點收盤 | 報酬 | 長度 | 中間最低 |", "|---|---|---|---|---|---|---|---|---|"]
    for label, ws in (("發現", d["windows"]), ("驗證", v["windows"])):
        for w in ws:
            L.append(f"| {w['year']}（{label}） | {w['lunar_new_year']} | {w['start_close_date']} | {w['start_close']:.2f} | {w['end_close_date']} | "
                     f"{w['end_close']:.2f} | {f2(w['return_pct'])} | {w['length_bars']} | {f2(w['dip_pct'])} |")
    L += ["", "驗證的長度單位是週；週線的起點、終點收盤日是那一週的週五（週一標籤＋4 天）。", ""]

    L += ["## 登記的次要分析（描述，不進判定）", "",
          "| 項目 | 發現 2009–2026 | 驗證 1981–2008 |", "|---|---|---|"]
    for name, key in (("前段：起點 → 12 月最後一個收盤", "leg_a"), ("後段：12 月最後一個收盤 → 終點", "leg_b")):
        L.append(f"| {name} | {d['legs'][key]['up']}／{d['legs'][key]['n']} 漲（{wl(d['legs'][key]['up_wilson95_pct'])}），中位數 {f2(d['legs'][key]['median_pct'])} | "
                 f"{v['legs'][key]['up']}／{v['legs'][key]['n']} 漲（{wl(v['legs'][key]['up_wilson95_pct'])}），中位數 {f2(v['legs'][key]['median_pct'])} |")
    L += [f"| 改從 12 月 1 日起算 | {rb['discovery_daily']['up']}／{rb['discovery_daily']['n']} 漲（{wl(rb['discovery_daily']['share_up_wilson95_pct'])}），中位數 {f2(rb['discovery_daily']['median_pct'])} | "
          f"{rb['validation_weekly']['up']}／{rb['validation_weekly']['n']} 漲（{wl(rb['validation_weekly']['share_up_wilson95_pct'])}），中位數 {f2(rb['validation_weekly']['median_pct'])} |", "",
          f"起點（12 月 15 日）是影片稿當初自己挑的；換成 12 月 1 日起算，{ds['n']} 次漲 {ds['up']} 次變成 {rb['discovery_daily']['up']} 次、"
          f"中位數 {f2(ds['median_pct'])} 變成 {f2(rb['discovery_daily']['median_pct'])}，所以漲的次數跟起點綁在一起。", ""]
    L += ["驗證分三個年代（不能當結論）：" + "；".join(f"{k} {x['up']}／{x['n']} 漲（{wl(x['up_wilson95_pct'])}）、中位數 {f2(x['median_pct'])}" for k, x in v["subperiods"].items()) + "。", ""]

    L += ["## 資料檢查", "",
          "| 檢查 | 年份 | 段數 | 漲（主資料） | 漲（對照） | 方向不同的年份 | 報酬差（平均／最大，百分點） |", "|---|---|---|---|---|---|---|"]
    for name, c in (("日線：2026-11-14 快照 vs 38dea7b 舊版", dv["daily_vs_38dea7b"]),
                    ("日線：Twelve Data vs TradingView FX_IDC", dv["daily_vs_fxidc"]),
                    ("週線：2026-11-14 快照 vs 38dea7b 舊版", dv["weekly_vs_38dea7b"]),
                    ("週線規則 vs 日線規則（同一批年份）", g)):
        L.append(f"| {name} | {c['years'][0]}–{c['years'][1]} | {c['n']} | {c['up_primary']} | {c['up_other']} | "
                 f"{'、'.join(map(str, c['sign_differs_years'])) or '無'} | {c['mean_abs_diff_pct_points']:.2f}／{c['max_abs_diff_pct_points']:.2f}（{c['max_abs_diff_year'] if c['max_abs_diff_pct_points'] > 0 else '—'}） |")
    L += ["", "最後一列量的是驗證用的週線近似有多粗：同樣 18 年，用週線規則算和用日線規則算，方向不同的年份就是上表那幾年。", ""]

    n = r["next_window"]
    L += ["## 下一段（只有日期）", "",
          f"起點 {n['start_date']}（{n['start_weekday']}）收盤；{n['year']} 年春節 {n['lunar_new_year']} 是 {n['lunar_new_year_weekday']}，"
          "終點是春節前最後一個收盤（預期 2027-02-05 週五，若當天有交易）。不算報酬、不預測。", "",
          "## 限制", ""] + [f"- {x}" for x in r["limitations"]] + ["",
          "## 方法細節", ""] + [f"- `{k}`：{x}" for k, x in r["method"].items()] + ["",
          "## 資料來源（Private）", "",
          "`source_manifest.json`：Twelve Data XAU/USD 日線與週線（2026-11-14 13:16 抓取的唯讀複本，與 Private git 38dea7b 舊版）、"
          "TradingView FX_IDC:XAUUSD 日線匯出（2026-11-04 凍結）、香港天文台公曆農曆對照表 1981–2027（原檔 47 個）。每個檔案的 SHA-256 在執行前核對。", ""]
    return "\n".join(L)


# ---------- main ----------
def main() -> None:
    # Public form: when reproduction_inputs.json sits next to this script, each input is the file
    # it names under local-inputs/ (one per role, verified by sha256) and the output goes to
    # reproduced/, so the published package is never overwritten.
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
    daily, d_dups = bars(raw["price_1d_xauusd"], weekly=False)
    weekly, w_dups = bars(raw["price_1wk_xauusd"], weekly=True)
    daily_old, _ = bars(raw["price_1d_xauusd_v38dea7b"], weekly=False)
    weekly_old, _ = bars(raw["price_1wk_xauusd_v38dea7b"], weekly=True)
    fxidc, _ = bars(raw["price_1d_fxidc_frozen"], weekly=False)

    disc = windows(daily, DISCOVERY, lny)
    val = windows(weekly, VALIDATION, lny)
    wk_recent = windows(weekly, DISCOVERY, lny)
    s_disc, s_val = summary(disc), summary(val)

    b_disc = baseline(daily, dt.date(2008, 12, 1), disc[-1]["_e"], hold_of(s_disc))
    b_val = baseline(weekly, dt.date(1980, 12, 1), val[-1]["_e"], hold_of(s_val))

    rob_dec1 = {
        "discovery_daily": short(windows(daily, DISCOVERY, lny, start_md=(12, 1))),
        "validation_weekly": short(windows(weekly, VALIDATION, lny, start_md=(12, 1))),
    }
    subp = {}
    for y0, y1 in SUBPERIODS:
        ws = [w for w in val if y0 <= w["year"] <= y1]
        subp[f"{y0}-{y1}"] = {"n": len(ws), "up": sum(w["up"] for w in ws),
                             "up_wilson95_pct": wilson(sum(w["up"] for w in ws), len(ws)),
                             "median_pct": st.median(w["return_pct"] for w in ws),
                             "label": "不能當結論 (descriptive split)"}
    versions = {
        "daily_vs_38dea7b": compare(disc, windows(daily_old, DISCOVERY, lny)),
        "daily_vs_fxidc": compare([w for w in disc if w["year"] in FXIDC_YEARS],
                                  windows(fxidc, FXIDC_YEARS, lny)),
        "weekly_vs_38dea7b": compare(val, windows(weekly_old, VALIDATION, lny)),
    }
    grain = compare(disc, wk_recent)

    nxt_start = dt.date(2026, 12, 15)
    nxt_lny = lny[2027]
    results = {
        "study_id": SID,
        "schema_version": 1,
        "revision": 1,
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "runner": "scripts/research/build_xauusd_dec15_lny_window.py",
        "registration": "registered definitions committed 2026-11-15 21:13:51 Asia/Taipei (commit 82f29d5)",
        "instrument": "XAUUSD spot (Twelve Data XAU/USD)",
        "method": {
            "window": "from mid-December of Y-1 to just before Lunar New Year of Y, named by Y",
            "lunar_new_year": "Hong Kong Observatory conversion table: the Jan-Feb date labelled 1st Lunar month",
            "daily_rule": "start = close of the last daily bar dated on or before 12-15 of Y-1; end = close of the last daily bar dated strictly before LNY(Y)",
            "weekly_rule": "a weekly bar labelled Monday M closes on Friday M+4; start = last weekly bar whose Friday is on or before 12-15 of Y-1; end = last weekly bar whose Friday is strictly before LNY(Y); each end can be up to four trading days earlier than the daily rule",
            "return": "end close / start close - 1, in percent",
            "length": "bars after the start bar up to and including the end bar (daily: trading days; weekly: weeks)",
            "dip": "lowest close among those bars / start close - 1; weekly uses weekly closes only",
            "share_up": "share of windows with end close > start close; Wilson 95% interval, z = 1.96",
            "quantiles": "linear interpolation between order statistics",
            "baseline": "every start bar in the sample period held for L bars, L = the sample's median window length with a half rounded up; daily period 2008-12-01 to the 2026 window's end, weekly from the first bar whose Friday is on or after 1980-12-01 to the 2008 window's end; overlapping, so no interval",
            "verdict_rule": "延續 if the Wilson 95% lower bound of share up > baseline share up; 沒有延續 if share up <= baseline share up; otherwise 分不出",
            "legs": "leg A = start to the last close dated on or before 12-31 of Y-1 (weekly: last bar whose Friday is on or before 12-31); leg B = that close to the end",
            "start_gap_guard": "a window is built only if its start close is at most 7 calendar days before the start date and is not the series' first bar",
            "duplicates": "a bar label that appears twice must carry the same close; the second row is dropped",
        },
        "data_period": {
            "daily": {"from": daily[0][1].isoformat(), "to": daily[-1][1].isoformat(), "bars": len(daily)},
            "weekly": {"from": weekly[0][0].isoformat(), "to": weekly[-1][0].isoformat(), "bars": len(weekly),
                       "duplicated_labels_dropped": w_dups},
            "discovery_windows": [disc[0]["start_close_date"], disc[-1]["end_close_date"]],
            "validation_windows": [val[0]["start_close_date"], val[-1]["end_close_date"]],
        },
        "discovery": {"label": "daily 2009-2026, already seen by EP06 on 2026-11-05 — not a test",
                      "summary": s_disc, "baseline": b_disc, "verdict_if_it_were_a_test": verdict(s_disc, b_disc),
                      "legs": legs(disc), "windows": clean(disc)},
        "validation": {"label": "weekly 1981-2008, registered before computing",
                       "summary": s_val, "baseline": b_val, "verdict": verdict(s_val, b_val),
                       "legs": legs(val), "subperiods": subp, "windows": clean(val)},
        "weekly_rule_check_2009_2026": grain,
        "robustness_start_dec1": rob_dec1,
        "data_versions": versions,
        "next_window": {
            "year": 2027,
            "start_date": nxt_start.isoformat(), "start_weekday": nxt_start.strftime("%A"),
            "lunar_new_year": nxt_lny.isoformat(), "lunar_new_year_weekday": nxt_lny.strftime("%A"),
            "end_rule": "the last close before Lunar New Year (expected Friday 2027-02-05, if that is a trading day)",
            "note": "dates only; no return, no forecast",
        },
        "limitations": [
            "The 18 daily windows were seen before registration (EP06, 2026-11-05) and are discovery only; whether other start dates were tried then is unknown.",
            "The validation uses weekly bars: each end of a window can sit up to four trading days from where the daily rule would put it; the weekly-rule check on 2009-2026 measures how often that changes the sign.",
            "The provider revised its weekly history by up to 9.3% and its daily history by up to 1.8% between two fetches; data_versions reports how many window signs that changes.",
            "No independent local source checks the weekly closes before 2013; the FX_IDC check covers 2014-2026 daily only.",
            "The month study's weekly January (1981-2026 pooled) was known at registration and overlaps leg B.",
            "Baseline windows overlap and carry no interval; the verdict compares one interval with one number.",
            "Descriptive only: a calendar window is not a cause, a signal or a forecast.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "README.md").write_text(readme(results))
    print(json.dumps({
        "discovery": {k: s_disc[k] for k in ("n", "up", "share_up_pct", "share_up_wilson95_pct", "median_pct")},
        "discovery_baseline": {k: b_disc[k] for k in ("hold_bars", "share_up_pct", "median_pct")},
        "validation": {k: s_val[k] for k in ("n", "up", "share_up_pct", "share_up_wilson95_pct", "median_pct")},
        "validation_baseline": {k: b_val[k] for k in ("hold_bars", "share_up_pct", "median_pct")},
        "validation_verdict": results["validation"]["verdict"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
