#!/usr/bin/env python3
"""RS-TX-20261121-001 — TAIEX after a big weekly fall (A) and after a fall from the peak (B).

Pre-registered definitions (registered 2026-11-21 23:1x Asia/Taipei, before any result):
  Calendar: Yahoo ^TWII put on the TWSE calendar (same rule as build_tx_twse_calendar_prices.py,
    tol 0.5 points) through the last TWSE date; later Yahoo rows appended unchanged. date+close only.
  A: ISO-week close vs the last close before the week; event week <= -X (X = 10 primary; 8, 12).
     A qualifying week whose close is <= 20 sessions after the previous retained entry joins it.
     Entry = close of the first qualifying week. Forward 20/60/120/240-session returns; lowest close
     in sessions 1..240 (and 1..20, 1..60) vs entry, with the session number. Episode check: retained
     events within 240 sessions of an episode's first event form one episode.
  B: all-time high (running max since 1997-07-02) and one-year high (last 252 sessions incl. today).
     T = 20 primary; 15, 25. All-time: first close <= (1-T)*high within a stretch between new highs.
     One-year: first close <= (1-T)*252-high, re-armed after a new 252-session high.
  Baseline: every session as entry (not for viewers). Wilson 95% for up shares.
  "Now": events whose 20-session window (A) / episode (B) is not complete at the data end; excluded.

Usage: build_tx_week_crash_drawdown.py [--yahoo CSV] [--twse CSV] [--out-dir DIR] [--tables]
Writes only results.json under --out-dir; --tables prints the README tables to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_ID = "RS-TX-20261121-001"
HORIZONS = [20, 60, 120, 240]
A_THRESH = [8.0, 10.0, 12.0]
A_PRIMARY = 10.0
B_THRESH = [15.0, 20.0, 25.0]
B_PRIMARY = 20.0
JOIN_GAP = 20
EPISODE_GAP = 240
ONE_YEAR = 252
LOW_WINDOW = 240
TOL = 0.5


def wilson(k: int, n: int, z: float = 1.959964) -> list | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (cen - half), 2), round(100 * (cen + half), 2)]


def r2(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 2)


def load_prices(yahoo: str, twse: str) -> tuple[pd.DataFrame, dict]:
    y = pd.read_csv(yahoo, dtype={"date": str})[["date", "close"]]
    t = pd.read_csv(twse, dtype={"date": str})[["date", "taiex_close"]]
    t_last = t["date"].max()
    y_in, y_after = y[y["date"] <= t_last].copy(), y[y["date"] > t_last].copy()
    t = t[(t["date"] >= y["date"].min()) & (t["date"] <= t_last)]
    ys, ts = set(y_in["date"]), set(t["date"])
    tmap = dict(zip(t["date"], t["taiex_close"]))
    keep = y_in[y_in["date"].isin(ts)].copy()
    keep["source"] = "yahoo"
    diff = (keep["close"] - keep["date"].map(tmap)).abs()
    fix = diff > TOL
    replaced = keep.loc[fix, "date"].tolist()
    keep.loc[fix, "close"] = keep.loc[fix, "date"].map(tmap)
    keep.loc[fix, "source"] = "yahoo_close_from_twse"
    inserted = sorted(ts - ys)
    ins = pd.DataFrame({"date": inserted, "close": [tmap[d] for d in inserted], "source": "twse"})
    y_after["source"] = "yahoo_after_twse_file"
    m = pd.concat([keep, ins, y_after], ignore_index=True).sort_values("date").reset_index(drop=True)
    report = {
        "yahoo_rows": int(len(y)),
        "twse_file_last_date": t_last,
        "yahoo_rows_kept": int(len(keep)),
        "closes_replaced_by_twse": replaced,
        "twse_sessions_inserted": len(inserted),
        "twse_inserted_saturdays": int(sum(pd.Timestamp(d).dayofweek == 5 for d in inserted)),
        "yahoo_rows_dropped_not_twse_session": sorted(ys - ts),
        "yahoo_rows_appended_after_twse_file": y_after["date"].tolist(),
        "max_abs_close_diff_shared": r2(float(diff.max())),
        "sessions": int(len(m)),
        "first_date": m["date"].iloc[0],
        "last_date": m["date"].iloc[-1],
    }
    return m, report


def fwd(close: np.ndarray, i: int, h: int):
    return (close[i + h] / close[i] - 1) * 100 if i + h < len(close) else None


def low_after(close: np.ndarray, i: int, w: int):
    """Lowest close in sessions i+1..i+w vs close[i]; None unless the full window exists."""
    if i + w >= len(close):
        return None, None
    seg = close[i + 1:i + w + 1]
    j = int(np.argmin(seg))
    return (seg[j] / close[i] - 1) * 100, j + 1


def low_so_far(close: np.ndarray, i: int, w: int):
    seg = close[i + 1:min(i + w + 1, len(close))]
    if len(seg) == 0:
        return None, None
    j = int(np.argmin(seg))
    return (seg[j] / close[i] - 1) * 100, j + 1


def summarise(rows: list[dict], key: str, base: dict | None) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    a = np.array(vals)
    k = int((a > 0).sum())
    worst = min((r for r in rows if r.get(key) is not None), key=lambda r: r[key])
    best = max((r for r in rows if r.get(key) is not None), key=lambda r: r[key])
    out = {"n": n, "up": k, "down": n - k, "up_pct": r2(100 * k / n), "up_wilson95": wilson(k, n),
           "median": r2(np.median(a)), "min": r2(a.min()), "min_event": worst["id"],
           "max": r2(a.max()), "max_event": best["id"]}
    if base is not None:
        out["baseline_up_pct"] = base["up_pct"]
        out["baseline_median"] = base["median"]
        w = out["up_wilson95"]
        out["baseline_inside_interval"] = bool(w[0] <= base["up_pct"] <= w[1])
    return out


def baseline(close: np.ndarray) -> dict:
    out = {}
    for h in HORIZONS:
        r = (close[h:] / close[:-h] - 1) * 100
        out[str(h)] = {"n": int(len(r)), "up_pct": r2(100 * (r > 0).mean()), "median": r2(np.median(r))}
    lows = []
    for i in range(len(close) - LOW_WINDOW):
        lows.append(close[i + 1:i + LOW_WINDOW + 1].min() / close[i] - 1)
    lows = np.array(lows) * 100
    out["low240"] = {"n": int(len(lows)), "median": r2(np.median(lows)),
                     "share_below_minus20_pct": r2(100 * (lows <= -20).mean())}
    return out


# ---------------- A ----------------
def weekly(dates: list[str], close: np.ndarray) -> list[dict]:
    iso = [pd.Timestamp(d).isocalendar() for d in dates]
    keys = [(c[0], c[1]) for c in iso]
    weeks = []
    for i in range(len(dates)):
        if i + 1 == len(dates) or keys[i + 1] != keys[i]:
            weeks.append(i)  # index of last session of the week
    out = []
    for w in range(1, len(weeks)):
        i, p = weeks[w], weeks[w - 1]
        out.append({"week_close_idx": i, "week_close_date": dates[i], "prev_close_date": dates[p],
                    "prev_close": float(close[p]), "week_close": float(close[i]),
                    "change": (close[i] / close[p] - 1) * 100})
    return out


def study_a(dates, close, weeks, x: float, base) -> dict:
    q = [w for w in weeks if w["change"] <= -x]
    events = []
    for w in q:
        i = w["week_close_idx"]
        if events and i - events[-1]["_i"] <= JOIN_GAP:
            e = events[-1]
            e["weeks_absorbed"].append({"date": w["week_close_date"], "change": r2(w["change"])})
            if w["change"] < e["_worst"]:
                e["_worst"] = w["change"]
                e["largest_week_fall"] = r2(w["change"])
                e["largest_week_date"] = w["week_close_date"]
            continue
        events.append({"_i": i, "_worst": w["change"], "entry_date": w["week_close_date"],
                       "entry_close": r2(w["week_close"]), "prev_week_close_date": w["prev_close_date"],
                       "prev_week_close": r2(w["prev_close"]), "week_change": r2(w["change"]),
                       "weeks_absorbed": [{"date": w["week_close_date"], "change": r2(w["change"])}],
                       "largest_week_fall": r2(w["change"]), "largest_week_date": w["week_close_date"]})
    hist, now = [], []
    ep_first = None
    for e in events:
        i = e["_i"]
        e["id"] = e["entry_date"]
        e["sessions_after_entry_available"] = int(len(close) - 1 - i)
        for h in HORIZONS:
            e[f"ret{h}"] = r2(fwd(close, i, h))
        for w in (20, 60, LOW_WINDOW):
            v, d = low_after(close, i, w)
            e[f"low{w}"] = r2(v)
            e[f"low{w}_session"] = d
        if LOW_WINDOW + i < len(close):
            e["low240_date"] = dates[i + e["low240_session"]]
        lv, ld = low_so_far(close, i, LOW_WINDOW)
        e["low_so_far"], e["low_so_far_session"] = r2(lv), ld
        is_now = i + JOIN_GAP >= len(close) - 1 or (events[-1] is e and i + JOIN_GAP >= len(close) - 1)
        e["status"] = "now" if is_now else "history"
        if not is_now:
            if ep_first is None or i - ep_first > EPISODE_GAP:
                ep_first = i
                e["episode_first"] = True
            else:
                e["episode_first"] = False
            e["episode_id"] = dates[ep_first]
            hist.append(e)
        else:
            now.append(e)
    for e in events:
        e.pop("_i"), e.pop("_worst")
    summ = {str(h): summarise(hist, f"ret{h}", base[str(h)]) for h in HORIZONS}
    first = [e for e in hist if e["episode_first"]]
    summ_ep = {str(h): summarise(first, f"ret{h}", None) for h in HORIZONS}
    lows = [e["low240"] for e in hist if e["low240"] is not None]
    deepest = min((e for e in hist if e["low240"] is not None), key=lambda e: e["low240"], default=None)
    return {"threshold_pct": x, "qualifying_weeks": len(q), "events": len(hist), "episodes": len(first),
            "history": hist, "now": now, "summary": summ, "summary_episode_first_only": summ_ep,
            "low240_summary": {"n": len(lows), "median": r2(np.median(lows)) if lows else None,
                               "deepest": r2(deepest["low240"]) if deepest else None,
                               "deepest_event": deepest["id"] if deepest else None,
                               "count_at_or_below_minus10": int(sum(v <= -10 for v in lows)),
                               "count_at_or_below_minus20": int(sum(v <= -20 for v in lows)),
                               "count_no_lower_close": int(sum(v >= 0 for v in lows)),
                               "baseline_median": base["low240"]["median"]}}


# ---------------- B ----------------
def ref_high(close: np.ndarray, version: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    hi = np.empty(n)
    hi_idx = np.empty(n, dtype=int)
    if version == "all_time":
        m, mi = -1.0, 0
        for i in range(n):
            if close[i] > m:
                m, mi = close[i], i
            hi[i], hi_idx[i] = m, mi
    else:
        s = pd.Series(close)
        for i in range(n):
            lo = max(0, i - ONE_YEAR + 1)
            j = lo + int(np.argmax(close[lo:i + 1]))
            hi[i], hi_idx[i] = close[j], j
    return hi, hi_idx


def study_b(dates, close, version: str, t: float, base) -> dict:
    hi, hi_idx = ref_high(close, version)
    dd = close / hi - 1
    n = len(close)
    events = []
    armed = True
    for i in range(n):
        if dd[i] >= 0:  # new reference high (close equals the high) re-arms
            armed = True
            continue
        if armed and dd[i] <= -t / 100:
            armed = False
            events.append({"cross_idx": i, "high_idx": int(hi_idx[i]), "high_close": float(hi[i])})
    hist, now = [], []
    for e in events:
        i, h = e["cross_idx"], e["high_idx"]
        end = n - 1
        for j in range(i + 1, n):
            if dd[j] >= 0:
                end = j
                break
        complete = end < n - 1 or dd[n - 1] >= 0
        seg = close[i:end + 1] if complete else close[i:]
        k = int(np.argmin(seg))
        low_i = i + k
        rec = None
        for j in range(i + 1, n):
            if close[j] >= e["high_close"]:
                rec = j
                break
        row = {"id": dates[i], "high_date": dates[h], "high_close": r2(e["high_close"]),
               "cross_date": dates[i], "cross_close": r2(close[i]), "cross_drawdown": r2(100 * dd[i]),
               "sessions_high_to_cross": int(i - h),
               "episode_end_date": dates[end] if complete else None,
               "low_date": dates[low_i], "low_close": r2(close[low_i]),
               "low_vs_high": r2((close[low_i] / e["high_close"] - 1) * 100),
               "low_vs_cross": r2((close[low_i] / close[i] - 1) * 100),
               "sessions_cross_to_low": int(low_i - i), "sessions_high_to_low": int(low_i - h),
               "recovery_date": dates[rec] if rec is not None else None,
               "sessions_high_to_recovery": int(rec - h) if rec is not None else None,
               "sessions_cross_to_recovery": int(rec - i) if rec is not None else None,
               "sessions_after_cross_available": int(n - 1 - i)}
        if rec is not None:
            row["years_high_to_recovery"] = r2((pd.Timestamp(dates[rec]) - pd.Timestamp(dates[h])).days / 365.25)
        for hz in HORIZONS:
            row[f"ret{hz}"] = r2(fwd(close, i, hz))
        row["status"] = "history" if complete else "now"
        (hist if complete else now).append(row)
    for e in now:
        e["low_note"] = "lowest close so far (episode not finished)"
    summ = {str(h): summarise(hist, f"ret{h}", base[str(h)]) for h in HORIZONS}
    lv = [e["low_vs_high"] for e in hist]
    deepest = min(hist, key=lambda e: e["low_vs_high"], default=None)
    return {"version": version, "threshold_pct": t, "events": len(hist), "history": hist, "now": now,
            "summary": summ,
            "depth_summary": {"n": len(lv), "median_low_vs_high": r2(np.median(lv)) if lv else None,
                              "median_low_vs_cross": r2(np.median([e["low_vs_cross"] for e in hist])) if hist else None,
                              "median_sessions_cross_to_low": r2(np.median([e["sessions_cross_to_low"] for e in hist])) if hist else None,
                              "median_sessions_high_to_recovery": r2(np.median([e["sessions_high_to_recovery"] for e in hist if e["sessions_high_to_recovery"] is not None])) if hist else None,
                              "deepest": r2(deepest["low_vs_high"]) if deepest else None,
                              "deepest_event": deepest["id"] if deepest else None}}


def candidates(res: dict) -> dict:
    pos, worst = [], []
    def cell(study, label, block, neighbours):
        for h in HORIZONS:
            s = block["summary"][str(h)]
            if s.get("n", 0) and s["up"] > s["n"] / 2 and s["median"] > 0:
                pos.append({"study": study, "case": label, "horizon": h, "n": s["n"], "up": s["up"],
                            "median": s["median"], "up_wilson95": s["up_wilson95"],
                            "baseline_up_pct": s["baseline_up_pct"], "baseline_median": s["baseline_median"],
                            "neighbours": {k: {"n": v["summary"][str(h)].get("n"), "up": v["summary"][str(h)].get("up"),
                                               "median": v["summary"][str(h)].get("median")} for k, v in neighbours.items()}})
    a = res["A"]
    cell("A", "week <= -10%", a["10"], {"8": a["8"], "12": a["12"]})
    for ver in ("all_time", "one_year"):
        b = res["B"][ver]
        cell("B", f"{ver} high, -20%", b["20"], {"15": b["15"], "25": b["25"]})
    for x in a:
        blk = a[x]
        worst.append({"study": "A", "threshold": x,
                      "smallest_return": {h: [blk["summary"][h].get("min"), blk["summary"][h].get("min_event")] for h in blk["summary"]},
                      "deepest_low240": [blk["low240_summary"]["deepest"], blk["low240_summary"]["deepest_event"]]})
    for ver in res["B"]:
        for t, blk in res["B"][ver].items():
            worst.append({"study": "B", "version": ver, "threshold": t,
                          "smallest_return": {h: [blk["summary"][h].get("min"), blk["summary"][h].get("min_event")] for h in blk["summary"]},
                          "deepest_low_vs_high": [blk["depth_summary"]["deepest"], blk["depth_summary"]["deepest_event"]]})
    return {"rule": "primary thresholds (A 10%, B 20% both versions), each horizon: up > n/2 and median > 0",
            "positive": pos, "worst": worst}


def fmt(v, pct=True):
    if v is None:
        return "—"
    return f"{v:+.1f}%" if pct else str(v)


def tables(res: dict) -> str:
    L = []
    for x in ("10", "8", "12"):
        a = res["A"][x]
        L.append(f"\n### A 週跌 ≤ −{x}%：{a['events']} 個歷史事件（{a['qualifying_weeks']} 個達標週，{a['episodes']} 段行情）\n")
        L.append("| 進場週收盤日 | 當週跌幅 | 併入的達標週 | 20 日 | 60 日 | 120 日 | 240 日 | 20／60 日內最低 | 240 日內最低（第幾天） |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for e in a["history"] + a["now"]:
            ab = "、".join(f"{w['date']} {w['change']:+.1f}%" for w in e["weeks_absorbed"][1:]) or "—"
            low = f"{e['low240']:+.1f}%（{e['low240_session']}）" if e["low240"] is not None else \
                f"至今 {fmt(e['low_so_far'])}（{e['low_so_far_session']}）" if e["low_so_far"] is not None else "—"
            tag = "（現在）" if e["status"] == "now" else ""
            L.append(f"| {e['entry_date']}{tag} | {e['week_change']:+.1f}% | {ab} | {fmt(e['ret20'])} | {fmt(e['ret60'])} | {fmt(e['ret120'])} | {fmt(e['ret240'])} | {fmt(e['low20'])}／{fmt(e['low60'])} | {low} |")
        L.append("\n| 持有 | n | 漲 | 跌 | 中位數 | 最差（事件） | 最好 | 平常日漲的比例／中位數（不對觀眾） | Wilson 95% |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            s = a["summary"][str(h)]
            if s.get("n"):
                L.append(f"| {h} 日 | {s['n']} | {s['up']} | {s['down']} | {fmt(s['median'])} | {fmt(s['min'])}（{s['min_event']}） | {fmt(s['max'])} | {s['baseline_up_pct']}%／{fmt(s['baseline_median'])} | {s['up_wilson95'][0]}–{s['up_wilson95'][1]} |")
        se = a["summary_episode_first_only"]
        L.append("\n每段行情只算第一個事件：" + "；".join(f"{h} 日 {se[str(h)].get('up')}/{se[str(h)].get('n')} 漲、中位數 {fmt(se[str(h)].get('median'))}" for h in HORIZONS if se[str(h)].get("n")))
        l20 = [e["low20"] for e in a["history"] if e["low20"] is not None]
        L.append(f"\n20 日內收盤比進場低：{sum(v < 0 for v in l20)}/{len(l20)}；跌到進場 −5% 以下：{sum(v <= -5 for v in l20)}/{len(l20)}。")
        ls = a["low240_summary"]
        L.append(f"\n240 日內最低收盤（n={ls['n']}）：中位數 {fmt(ls['median'])}，最深 {fmt(ls['deepest'])}（{ls['deepest_event']}）；跌到 −10% 以下 {ls['count_at_or_below_minus10']} 次、−20% 以下 {ls['count_at_or_below_minus20']} 次、之後沒有更低收盤 {ls['count_no_lower_close']} 次；平常日中位數 {fmt(ls['baseline_median'])}（不對觀眾）。")
    for ver, name in (("all_time", "資料內最高收盤"), ("one_year", "一年內最高收盤")):
        for t in ("20", "15", "25"):
            b = res["B"][ver][t]
            L.append(f"\n### B {name}跌破 −{t}%：{b['events']} 個歷史事件\n")
            L.append("| 高點日（收盤） | 跌破日 | 最低日 | 最低相對高點 | 跌破後再跌 | 跌破→最低（交易日） | 20 日 | 60 日 | 120 日 | 240 日 | 回到高點日（高點起交易日／年） |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for e in b["history"] + b["now"]:
                rec = f"{e['recovery_date']}（{e['sessions_high_to_recovery']}／{e['years_high_to_recovery']} 年）" if e["recovery_date"] else "還沒回到"
                tag = "（現在，最低是至今）" if e["status"] == "now" else ""
                L.append(f"| {e['high_date']}（{e['high_close']:,.2f}） | {e['cross_date']}{tag} | {e['low_date']} | {fmt(e['low_vs_high'])} | {fmt(e['low_vs_cross'])} | {e['sessions_cross_to_low']} | {fmt(e['ret20'])} | {fmt(e['ret60'])} | {fmt(e['ret120'])} | {fmt(e['ret240'])} | {rec} |")
            L.append("\n| 持有（從跌破日） | n | 漲 | 跌 | 中位數 | 最差（事件） | 平常日（不對觀眾） | Wilson 95% |")
            L.append("|---|---|---|---|---|---|---|---|")
            for h in HORIZONS:
                s = b["summary"][str(h)]
                if s.get("n"):
                    L.append(f"| {h} 日 | {s['n']} | {s['up']} | {s['down']} | {fmt(s['median'])} | {fmt(s['min'])}（{s['min_event']}） | {s['baseline_up_pct']}%／{fmt(s['baseline_median'])} | {s['up_wilson95'][0]}–{s['up_wilson95'][1]} |")
            d = b["depth_summary"]
            L.append(f"\n深度（n={d['n']}）：最低相對高點中位數 {fmt(d['median_low_vs_high'])}，跌破後再跌中位數 {fmt(d['median_low_vs_cross'])}，跌破到最低中位數 {d['median_sessions_cross_to_low']} 個交易日，高點到回到高點中位數 {d['median_sessions_high_to_recovery']} 個交易日；最深 {fmt(d['deepest'])}（{d['deepest_event']}）。")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yahoo", default="local-inputs/TWII_1d_to_2026-11-20.csv")
    ap.add_argument("--twse", default="local-inputs/TAIEX_FMTQIK_1997-06-02_to_2026-11-13.csv")
    ap.add_argument("--out-dir", default="reproduced")
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()

    px, merge = load_prices(a.yahoo, a.twse)
    dates = px["date"].tolist()
    close = px["close"].to_numpy(dtype=float)
    base = baseline(close)
    weeks = weekly(dates, close)
    top = sorted(weeks, key=lambda w: w["change"])[:15]
    res_a = {f"{x:g}": study_a(dates, close, weeks, x, base) for x in A_THRESH}
    res_b = {ver: {f"{t:g}": study_b(dates, close, ver, t, base) for t in B_THRESH} for ver in ("all_time", "one_year")}
    res = {"A": res_a, "B": res_b}
    hi_all, hi_idx = ref_high(close, "all_time")
    out = {
        "study_id": STUDY_ID, "schema_version": 1, "revision": 1,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "strategy_version": None,
        "method": {"horizons_sessions": HORIZONS, "A_thresholds_pct": A_THRESH, "A_primary_pct": A_PRIMARY,
                   "A_join_gap_sessions": JOIN_GAP, "A_episode_gap_sessions": EPISODE_GAP,
                   "A_low_window_sessions": LOW_WINDOW, "B_thresholds_pct": B_THRESH, "B_primary_pct": B_PRIMARY,
                   "B_one_year_sessions": ONE_YEAR, "week": "ISO calendar week; close of its last session vs the last close before it",
                   "up": "forward return > 0", "interval": "Wilson 95% for up shares",
                   "calendar_tolerance_points": TOL, "price": "close only, price index without dividends"},
        "coverage": merge,
        "baseline_any_session": base,
        "largest_weekly_falls_top15": [{"date": w["week_close_date"], "change": r2(w["change"]),
                                        "prev_close_date": w["prev_close_date"]} for w in top],
        "now_state": {"last_date": dates[-1], "last_close": r2(close[-1]),
                      "all_time_high_date": dates[int(hi_idx[-1])], "all_time_high_close": r2(hi_all[-1]),
                      "drawdown_from_all_time_high": r2((close[-1] / hi_all[-1] - 1) * 100),
                      "last_week_change": r2(weeks[-1]["change"]), "last_week_prev_close_date": weeks[-1]["prev_close_date"]},
        "A": res_a, "B": res_b,
        "candidates": candidates(res),
        "limitations": [
            "Price index without dividends (event and baseline both price only).",
            "The all-time high is the highest close since 1997-07-02 (data start); the 1990 peak is not in the data.",
            "Few events per threshold; shares are reported as counts with Wilson intervals; no significance test.",
            "Forward windows of events within 240 sessions overlap (counted once per episode in the episode-first table).",
            "2026-11-16 close is Yahoo only (not checked against the exchange); 2026-11-17..20 checked by RS-TX-20261105-001 rev 4.",
        ],
    }
    od = Path("reproduced") if a.out_dir == "reproduced" else Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if a.tables:
        print(tables(res))


if __name__ == "__main__":
    main()
