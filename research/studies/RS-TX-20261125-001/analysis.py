#!/usr/bin/env python3
"""RS-TX-20261125-001 — TAIEX: the session after a day up 4% or more.

Design registered in research/studies/RS-TX-20261125-001/study.json (`design`) before any
next-session return of the historical up-4% days was computed. Definitions are inherited from
RS-TX-20261124-001 (its Data class, merge checks and episode grouping are imported, not copied):
  ret[i]     = close[i] / close[i-1] - 1, consecutive TWSE sessions (FMTQIK TAIEX close).
  ratio[i]   = value[i] / mean(value[i-20 .. i-1]) (FMTQIK 成交金額, whole market, day excluded).
  Event      : ret >= 4%, i >= 20 (first session with a ratio, as in RS-TX-20261124-001), i < last.
  r1[i]      = close[i+1] / close[i] - 1 (next session).
  give-back  = (close[i] - close[i+1]) / (close[i] - close[i-1]).
  D_T        : events with r1 <= -T%, T = 2, 3 (primary), 4; control C_T = events with r1 > -T%.
  Forward    : j = i + 1; from_up_close = close[j+h]/close[i] - 1; from_drop_close =
               close[j+h]/close[j] - 1; h = 5, 20, 60 sessions after the next session.
  Paths      : closes at offsets -20 .. +61 around i, close[i] = 100 (offset +61 = 60 sessions after the
               next session j, the same point as the h=60 return; R2 first version stopped at +60, fixed
               after research review r1, M1).
Added at R2 before the first run (not in the registered design, labelled so in results):
  back_above : first session k in j+1 .. j+60 with close[k] >= close[i] (sessions after j), or null.

Usage: build_tx_surge_next_day.py [--base CSV] [--update CSV] [--raw JSON] [--prior JSON] [--out-dir DIR]
Writes only results.json and README.md under --out-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tx_low_volume_surge as prev  # noqa: E402  (same definitions, imported on purpose)

STUDY_ID = "RS-TX-20261125-001"
PRIOR_ID = "RS-TX-20261124-001"
UP_MIN = 4.0
LOW_RATIO = 0.8
DROP_T = [2.0, 3.0, 4.0]
PRIMARY_T = 3.0
HORIZONS = [5, 20, 60]
PATH_PRE, PATH_POST = 20, 61  # +61 = j + 60, the h=60 point (review r1 M1)
BACK_WINDOW = 60
LIMIT_CHANGE = prev.LIMIT_CHANGE
DEFAULT_DIR = f"research/studies/{STUDY_ID}"
NOW_UP, NOW_NEXT = "2026-11-24", "2026-11-25"


def r2(v):
    return prev.r2(v)


def regime(d: str) -> str:
    return "7%" if d < LIMIT_CHANGE else "10%"


def raw_rows(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    assert d["stat"] == "OK", d["stat"]
    out = {}
    for row in d["data"]:
        y, m, dd = row[0].split("/")
        iso = f"{int(y) + 1911:04d}-{int(m):02d}-{int(dd):02d}"
        out[iso] = {"close": float(row[4].replace(",", "")), "points": float(row[5].replace(",", "")),
                    "value": int(row[2].replace(",", ""))}
    return out


def pct_stats(vals: list[float]) -> dict:
    a = np.array(vals)
    return {"n": len(a), "up": int((a > 0).sum()), "down": int((a < 0).sum()), "flat": int((a == 0).sum()),
            "median_pct": r2(np.median(a)), "mean_pct": r2(a.mean()),
            "p10_pct": r2(np.percentile(a, 10)), "p25_pct": r2(np.percentile(a, 25)),
            "p75_pct": r2(np.percentile(a, 75)), "p90_pct": r2(np.percentile(a, 90)),
            "min_pct": r2(a.min()), "max_pct": r2(a.max())}


def fwd_block(D, idx: list[int]) -> dict:
    out = {}
    for h in HORIZONS:
        valid = [i for i in idx if i + 1 + h <= D.last]
        blk = {"n": len(valid)}
        for key, base in (("from_up_close", lambda i: D.close[i]), ("from_drop_close", lambda i: D.close[i + 1])):
            if not valid:
                blk[key] = None
                continue
            v = [(D.close[i + 1 + h] / base(i) - 1) * 100 for i in valid]
            lo, hi = int(np.argmin(v)), int(np.argmax(v))
            up = sum(x > 0 for x in v)
            blk[key] = {"up": up, "down": sum(x < 0 for x in v), "up_wilson95": prev.wilson(up, len(v)),
                        "median_pct": r2(np.median(v)), "min_pct": r2(v[lo]), "min_date": D.dates[valid[lo]],
                        "max_pct": r2(v[hi]), "max_date": D.dates[valid[hi]]}
        out[str(h)] = blk
    return out


def back_above(D, i: int):
    j = i + 1
    for k in range(j + 1, min(j + BACK_WINDOW, D.last) + 1):
        if D.close[k] >= D.close[i]:
            return k - j, D.dates[k]
    return (None, None) if j + BACK_WINDOW <= D.last else ("not_yet", None)


def event_row(D, i: int, ep_of: dict) -> dict:
    j = i + 1
    r1 = (D.close[j] / D.close[i] - 1) * 100
    row = {"date": D.dates[i], "next_date": D.dates[j], "ret_pct": r2(100 * D.ret[i]),
           "points": r2(D.points[i]), "close": float(D.close[i]), "value_ratio": round(float(D.ratio[i]), 3),
           "next_ret_pct": r2(r1), "next_points": r2(D.points[j]), "next_close": float(D.close[j]),
           "next_value_ratio": round(float(D.ratio[j]), 3),
           "giveback_pct": r2(100 * (D.close[i] - D.close[j]) / (D.close[i] - D.close[i - 1])),
           "next_below_up_close": bool(D.close[j] < D.close[i]),
           "next_below_prior_close": bool(D.close[j] < D.close[i - 1]),
           "limit_regime": regime(D.dates[i]), "episode": ep_of[i]}
    for h in HORIZONS:
        ok = j + h <= D.last
        row[f"h{h}_from_up_close_pct"] = r2((D.close[j + h] / D.close[i] - 1) * 100) if ok else None
        row[f"h{h}_from_drop_close_pct"] = r2((D.close[j + h] / D.close[j] - 1) * 100) if ok else None
        row[f"h{h}_date"] = D.dates[j + h] if ok else None
    s, d = back_above(D, i)
    row["back_above_up_close_sessions"], row["back_above_up_close_date"] = s, d
    for T in DROP_T:
        row[f"drop_le_{int(T)}"] = bool(r1 <= -T)
    return row


def path(D, i: int) -> list:
    out = []
    for o in range(-PATH_PRE, PATH_POST + 1):
        k = i + o
        out.append(round(float(D.close[k] / D.close[i] * 100), 4) if 0 <= k <= D.last else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="local-inputs/TAIEX_FMTQIK_1997-06-02_to_2026-11-13.csv")
    ap.add_argument("--update", default="local-inputs/fmtqik_2026-11.csv")
    ap.add_argument("--raw", default="local-inputs/fmtqik_202611.json")
    ap.add_argument("--prior", default="local-inputs/results.json")
    ap.add_argument("--out-dir", default="reproduced")
    a = ap.parse_args()

    rows, merge = prev.load_series(a.base, a.update)
    D = prev.Data(rows)
    raw = raw_rows(a.raw)
    # the two days of the question equal the exchange's raw JSON
    for d in (NOW_UP, NOW_NEXT):
        k = D.dates.index(d)
        assert abs(D.close[k] - raw[d]["close"]) < 1e-9 and abs(D.points[k] - raw[d]["points"]) < 1e-9, d
        assert int(D.value[k]) == raw[d]["value"], d
    assert D.dates[D.last] == NOW_NEXT, D.dates[D.last]
    iu, inx = D.dates.index(NOW_UP), D.dates.index(NOW_NEXT)
    assert inx == iu + 1

    allev = [i for i in range(prev.LOOKBACK, D.last) if D.ret[i] >= UP_MIN / 100]
    early = [D.dates[i] for i in range(1, prev.LOOKBACK) if D.ret[i] >= UP_MIN / 100]
    hist = [i for i in allev if D.dates[i] < NOW_UP]
    low = [i for i in allev if D.ratio[i] < LOW_RATIO and D.dates[i] < NOW_UP]
    high = [i for i in hist if D.ratio[i] >= prev.HIGH_RATIO]
    P = json.load(open(a.prior, encoding="utf-8"))
    assert len(hist) == P["groups"]["all_big_up"]["n"] == 62, len(hist)
    assert [D.dates[i] for i in low] == [e["date"] for e in P["events_primary"]], "low-value set differs"
    assert [D.dates[i] for i in high] == [e["date"] for e in P["events_high_volume"]], "high-value set differs"
    assert allev[-1] == iu and len(allev) == 63

    eps = prev.episodes(allev)
    ep_of = {i: k + 1 for k, e in enumerate(eps) for i in e}
    r1 = {i: (D.close[i + 1] / D.close[i] - 1) * 100 for i in allev}
    gb = {i: 100 * (D.close[i] - D.close[i + 1]) / (D.close[i] - D.close[i - 1]) for i in allev}
    now_r1, now_gb = r1[iu], gb[iu]

    # Q1 next-session distribution
    day = (D.close[1:] / D.close[:-1] - 1) * 100  # every session's one-day change, 1997-06-03 .. last
    q1 = {"all": pct_stats([r1[i] for i in allev]), "historical_62": pct_stats([r1[i] for i in hist]),
          "by_regime": {g: pct_stats([r1[i] for i in allev if regime(D.dates[i]) == g]) for g in ("7%", "10%")},
          "thresholds": {}, "baseline_any_session": {"n": int(len(day)), "first": D.dates[1], "last": D.dates[-1],
                                                     "median_pct": r2(np.median(day)), "up": int((day > 0).sum())}}
    for T in DROP_T:
        k = [i for i in allev if r1[i] <= -T]
        kb = int((day <= -T).sum())
        q1["thresholds"][f"le_{int(T)}"] = {
            "threshold_pct": -T, "n_events": len(allev), "count": len(k), "share_pct": r2(100 * len(k) / len(allev)),
            "wilson95": prev.wilson(len(k), len(allev)), "count_excl_now": len([i for i in k if i != iu]),
            "dates": [D.dates[i] for i in k], "episodes": len(prev.episodes(k)),
            "by_regime": {g: len([i for i in k if regime(D.dates[i]) == g]) for g in ("7%", "10%")},
            "baseline_any_session_count": kb, "baseline_any_session_share_pct": r2(100 * kb / len(day))}
    srt = sorted(allev, key=lambda i: r1[i])
    q1["below_up_close_next"] = sum(r1[i] < 0 for i in allev)
    q1["below_prior_close_next"] = sum(D.close[i + 1] < D.close[i - 1] for i in allev)
    q1["worst10"] = [{"date": D.dates[i], "next_date": D.dates[i + 1], "ret_pct": r2(100 * D.ret[i]),
                      "next_ret_pct": r2(r1[i]), "giveback_pct": r2(gb[i])} for i in srt[:10]]
    now = {"up_date": NOW_UP, "next_date": NOW_NEXT, "up_ret_pct": r2(100 * D.ret[iu]), "up_points": float(D.points[iu]),
           "up_close": float(D.close[iu]), "up_value_ratio": round(float(D.ratio[iu]), 4),
           "next_ret_pct": round(now_r1, 4), "next_points": float(D.points[inx]), "next_close": float(D.close[inx]),
           "next_value_twd": int(D.value[inx]), "next_value_ratio": round(float(D.ratio[inx]), 4),
           "next_avg20_value_twd": int(round(D.avg20_value[inx])),
           "giveback_pct": round(now_gb, 4), "net_two_day_pct": r2((D.close[inx] / D.close[iu - 1] - 1) * 100),
           "rank_next_ret_among_63": 1 + sum(r1[i] < now_r1 for i in allev),
           "rank_giveback_among_63": 1 + sum(gb[i] > now_gb for i in allev),
           "giveback_top3_among_63": [{"date": D.dates[i], "giveback_pct": r2(gb[i])}
                                      for i in sorted(allev, key=lambda i: -gb[i])[:3]],
           "any_session_days_at_or_below_now": int((day <= now_r1).sum()),
           "any_session_days_at_or_below_now_since_2015_06_01": int(sum(
               1 for k in range(1, D.last + 1) if D.dates[k] >= LIMIT_CHANGE and day[k - 1] <= now_r1)),
           "raw_check": "close, change in points and value of 2026-11-24 and 2026-11-25 asserted equal to the raw FMTQIK JSON"}
    assert now["next_points"] == -1926.92 and now["next_close"] == 38195.43 and now["up_close"] == 40122.35

    # Q2 up-then-down groups
    q2 = {}
    for T in DROP_T:
        d_idx = [i for i in allev if r1[i] <= -T]
        c_idx = [i for i in allev if r1[i] > -T]
        q2[f"le_{int(T)}"] = {"threshold_pct": -T, "primary": T == PRIMARY_T,
                               "drop": {"n": len(d_idx), "episodes": len(prev.episodes(d_idx)),
                                        "with_forward_window": len([i for i in d_idx if i != iu]),
                                        "by_horizon": fwd_block(D, d_idx)},
                               "control": {"n": len(c_idx), "episodes": len(prev.episodes(c_idx)),
                                           "by_horizon": fwd_block(D, c_idx)}}
    d2 = [i for i in allev if r1[i] <= -2.0]
    events_drop = [event_row(D, i, ep_of) for i in d2]

    # Q3 eight low-value events
    low_rows = [event_row(D, i, ep_of) for i in low]
    q3 = {"n": len(low), "next_below_up_close": sum(r["next_below_up_close"] for r in low_rows),
          "next_le_minus3": sum(r["drop_le_3"] for r in low_rows), "next_le_minus2": sum(r["drop_le_2"] for r in low_rows),
          "next_ret_median_pct": r2(np.median([r1[i] for i in low])), "events": low_rows,
          "post_hoc_ninth_event_review_r1_M2": {
              "label": "post-hoc, added after research review r1: 2026-11-24 also has value ratio < 0.8, so it is the 9th event of the RS-TX-20261124-001 definition; it was excluded from the 8 only because it had no next session then",
              "event": event_row(D, iu, ep_of),
              "expected_next_le_minus3_in_8_at_historical_rate": r2(8 * len([i for i in hist if r1[i] <= -3]) / len(hist)),
              "historical_rate_basis": f"{len([i for i in hist if r1[i] <= -3])}/{len(hist)} historical up-4% days fell 3% or more the next session",
              "nine_next_le_minus3": sum(r["drop_le_3"] for r in low_rows) + 1}}
    assert D.ratio[iu] < LOW_RATIO

    # Q4 paths
    pidx = sorted(set(low) | set(d2) | {iu})
    paths = []
    for i in pidx:
        p = path(D, i)
        assert len(p) == PATH_PRE + PATH_POST + 1 and p[PATH_PRE] == 100.0
        assert abs(p[PATH_PRE + 1] - round(100 * D.close[i + 1] / D.close[i], 4)) < 1e-9
        assert abs(p[PATH_PRE + 1] - (100 + r1[i])) < 1e-3
        if i + 1 + 60 <= D.last:  # review r1 M1: the chart's last point is the h=60 return
            assert abs(p[PATH_PRE + 61] - round(100 * D.close[i + 61] / D.close[i], 4)) < 1e-9
            assert abs(p[PATH_PRE + 61] - 100 * (1 + (D.close[i + 61] / D.close[i] - 1))) < 1e-3
        paths.append({"date": D.dates[i], "ret_pct": r2(100 * D.ret[i]), "next_ret_pct": r2(r1[i]),
                      "low_value": i in low, "drop_le_2": r1[i] <= -2, "drop_le_3": r1[i] <= -3,
                      "drop_le_4": r1[i] <= -4, "is_now": i == iu, "limit_regime": regime(D.dates[i]),
                      "dates_first_last": [D.dates[max(i - PATH_PRE, 0)], D.dates[min(i + PATH_POST, D.last)]],
                      "base": f"close of {D.dates[i]} (offset 0) = 100",
                      "offset_start": -PATH_PRE,
                      "offset_end_with_data": min(PATH_POST, D.last - i),
                      "offset_of_60_sessions_after_next_session": 61 if i + 61 <= D.last else None,
                      "h60_from_up_close_index": round(float(D.close[i + 61] / D.close[i] * 100), 4) if i + 61 <= D.last else None,
                      "close_index": p})
    assert sum(p["is_now"] for p in paths) == 1

    tz = timezone(timedelta(hours=8))
    res = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(tz).isoformat(timespec="seconds"),
        "strategy_version": None,
        "method": {"up_min_pct": UP_MIN, "drop_thresholds_pct": [-t for t in DROP_T], "primary_drop_pct": -PRIMARY_T,
                   "horizons_after_next_session": HORIZONS, "low_value_ratio": LOW_RATIO, "lookback": prev.LOOKBACK,
                   "episode_gap": prev.EPISODE_GAP, "path_offsets": [-PATH_PRE, PATH_POST],
                   "percentiles": "numpy linear interpolation", "wilson_z": 1.959964,
                   "limit_change": LIMIT_CHANGE, "inherited_from": PRIOR_ID,
                   "added_at_r2_before_first_run": ["back_above_up_close_sessions (first close >= up-day close within 60 sessions after the next session)",
                                                     "next_below_prior_close (next close below the close before the up day)"],
                   "revised_after_review_r1": ["M1 path offsets -20..+61 (was -20..+60): +61 = 60 sessions after the next session = h=60 point; assertion added",
                                               "M2 q3_low_value_eight.post_hoc_ninth_event_review_r1_M2 (post-hoc, 2026-11-24 is the 9th low-value up day)",
                                               "S3 README prints no median for groups with n < 5; S4/S5 two limitation sentences added",
                                               "no change to design, events, thresholds or any other number"]},
        "coverage": {"first": D.dates[0], "last": D.dates[-1], "sessions": len(D.dates),
                     "first_with_ratio": D.dates[prev.LOOKBACK], "merge": merge,
                     "up4_before_first_with_ratio_not_counted": early},
        "now": now, "q1_next_session": q1, "q2_after_up_then_down": q2, "events_drop_le_2": events_drop,
        "q3_low_value_eight": q3, "q4_paths": {"offsets": list(range(-PATH_PRE, PATH_POST + 1)),
                                              "note": "close / up-day close * 100 (offset 0 = the up day = 100); offsets are sessions from the up day, so offset +1 is the next session (the down day) and offset +61 is 60 sessions after the down day = the h=60 point of the tables (review r1 M1); offset +60 is NOT the h=60 point; null beyond the data", "events": paths},
        "prior_check": {"study": PRIOR_ID, "historical_up4": len(hist), "low_value_dates_equal": True,
                        "high_value_dates_equal": True},
    }
    res["limitations_zh"] = limits(res)
    out = Path("reproduced") if a.out_dir == "reproduced" else Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=_np) + "\n", encoding="utf-8")
    (out / "README.md").write_text(readme(res), encoding="utf-8")
    print(f"wrote {out}/results.json and README.md; events {len(allev)}, D3 {q2['le_3']['drop']['n']}, paths {len(paths)}")


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def f(v, sign=True):
    return "—" if v is None else (f"{v:+.2f}%" if sign else f"{v:.2f}%")


def limits(r: dict) -> list[str]:
    q1, now = r["q1_next_session"], r["now"]
    t3 = q1["thresholds"]["le_3"]
    d2 = {e["date"]: e for e in r["events_drop_le_2"]}
    return [
        f"樣本極少：隔天跌 ≥3% 的歷史事件只有 {t3['count_excl_now']} 次（加上 11/24 共 {t3['count']} 次），"
        f"有之後 5／20／60 日資料的只有 {r['q2_after_up_then_down']['le_3']['drop']['with_forward_window']} 次；"
        "兩次走向相反，中位數只是兩個數的平均，不能當「通常會怎樣」。1/2 的 Wilson 95% 區間是 "
        f"{r['q2_after_up_then_down']['le_3']['drop']['by_horizon']['60']['from_up_close']['up_wilson95'][0]}–"
        f"{r['q2_after_up_then_down']['le_3']['drop']['by_horizon']['60']['from_up_close']['up_wilson95'][1]}%，等於什麼都分不出。",
        f"隔天跌 ≥3% 的比例 {t3['count']}/{t3['n_events']}（{t3['share_pct']}%，Wilson {t3['wilson95'][0]}–{t3['wilson95'][1]}%）"
        f"和任何一天跌 ≥3% 的比例 {t3['baseline_any_session_share_pct']}%（n={q1['baseline_any_session']['n']:,}）：區間包含平常日，分不出大漲後隔天比較容易大跌。",
        "事件彼此重疊：2020-03-20（隔天跌 3.73%）和量縮大漲 2020-03-24 只差 2 個交易日，畫圖的兩條線幾乎是同一段走勢；"
        "2003-02-17 同時是量縮大漲和隔天跌 ≥2% 的事件。漲 ≥4% 的 63 天只分成 31 段行情（相隔 ≤20 個交易日算同一段）。",
        f"漲跌幅限制 2015-06-01 從 7% 改成 10%：7% 制 {q1['by_regime']['7%']['n']} 次大漲、10% 制 {q1['by_regime']['10%']['n']} 次；"
        f"隔天跌 ≥3% 的 {t3['count']} 次裡 10% 制 {t3['by_regime']['10%']} 次、7% 制 {t3['by_regime']['7%']} 次。"
        "7% 制下一天最多跌 7%，前後的「大跌」不完全等同；這是事後看到的分布，不能當「10% 制以後比較容易隔天大跌」的結論。",
        "只用收盤價（FMTQIK 沒有盤中高低）：隔天盤中跌更深、或盤中曾翻紅都看不到；也沒有期貨夜盤。",
        "價格指數，不含股息、不扣成本；只是描述，不是預測，也不是買賣訊號。",
        "週六交易日（2000 年以前的週六盤、之後的補班週六）照 RS-TX-20261124-001 算成一個交易日；「隔天」就是檔案裡的下一列。",
        f"研究是因為 11/25 才做的：門檻 −2／−3／−4% 是在看歷史隔天漲跌之前定的，三個門檻都包含 11/25（{now['next_ret_pct']:.2f}%），"
        "但題目本身是看到今天才問的，「以前有沒有過」的答案天生偏向「很少」。",
        "研究登記後、第一次跑之前加了兩個欄位（隔天收盤是否低於大漲前一天收盤、60 日內第一次收回大漲日收盤要幾天），結果檔標為 R2 補加。"
        f"2003-02-17、2008-09-08 在 60 個交易日內都沒收回大漲日收盤；2001-12-12 用了 {d2['2001-12-12']['back_above_up_close_sessions']} 天、2020-03-20 用了 {d2['2020-03-20']['back_above_up_close_sessions']} 天。",
        "2008-09-08 和對照組裡的 2008-09-19 只差 9 個交易日，兩者 60 日窗口幾乎完全重疊；對照組最差值（2008-09-19）和隔天大跌組的 2008-09-08 是同一段跌勢，不是獨立的兩個樣本。",
        "漲 ≥4% 的日子本身多在高波動期，用「平常任何一天」當基準會讓大漲後的比例看起來偏高；較公平的高波動基準更高（事後補算，未登記，不在本研究結果檔），所以「大漲後比較容易大跌」不成立也講不出來。",
        "2026-11-24 本身也是量縮大漲（量比 0.44，符合 RS-TX-20261124-001 定義，是第 9 次），隔天 −4.80%；上一個研究的 8 次因為當時沒有隔天而不含它。8 次裡沒有跌 ≥3%，不能拿來說「量縮大漲隔天不會大跌」（以歷史比例 2/62 估，8 次裡預期只有約 0.26 次，0/8 什麼都證明不了）。",
        "畫圖序列的最後一點是大漲日後第 61 個交易日（＝大跌日後第 60 個），和表格 h=60 同一點；第 +60 點不是 h=60。2008-09-08、2020-03-20 兩次的 60 日走勢同時受當時整體市場環境影響（2008 年 9 月、2020 年 3 月），本研究沒有這些背景資料，無法分開，也不能把兩次當成區間或預期。",
        "狀態：研究審通過（confirmed）；描述統計，不是預測。",
    ]


def readme(r: dict) -> str:
    q1, q2, q3, now, cov = r["q1_next_session"], r["q2_after_up_then_down"], r["q3_low_value_eight"], r["now"], r["coverage"]
    t = q1["thresholds"]
    prev_worst = next(e for e in q1["worst10"] if e["date"] != now["up_date"])
    assert prev_worst["next_ret_pct"] == q1["historical_62"]["min_pct"]
    d3 = {e["date"]: e for e in r["events_drop_le_2"] if e["drop_le_3"] and e["date"] != now["up_date"]}
    early = "在那之前沒有漲 ≥4% 的日子" if not cov["up4_before_first_with_ratio_not_counted"] else "在那之前漲 ≥4% 但沒算進來的：" + "、".join(cov["up4_before_first_with_ratio_not_counted"])
    s = [f"# {STUDY_ID} — 加權指數漲 4% 以上的隔一個交易日：隔天大跌以前有過嗎、之後怎麼走（{cov['first']}–{cov['last']}）\n",
         "**狀態：研究審通過（confirmed）。描述統計，不是預測，也不是買賣訊號。**\n",
         "## 標題數字\n",
         f"- 1997 年以來漲 4% 以上的 {t['le_4']['n_events']} 次裡，隔天再跌 4% 以上的只有 {t['le_4']['count']} 次，就是 {now['up_date']} → {now['next_date']}"
         f"（{now['up_ret_pct']:+.2f}% 後 {now['next_ret_pct']:.2f}%，排第 {now['rank_next_ret_among_63']} 名）。之前最深是 {q1['historical_62']['min_pct']:.2f}%（{prev_worst['date']} → {prev_worst['next_date']}）。",
         f"- 隔天跌 3% 以上：{t['le_3']['count']} 次（{', '.join(t['le_3']['dates'])}）；跌 2% 以上：{t['le_2']['count']} 次。",
         f"- {now['next_date']} 收 {now['next_close']:,.2f}，跌 {-now['next_points']:,.2f} 點，吐回大漲那天漲幅的 {now['giveback_pct']:.1f}%（63 次裡第 {now['rank_giveback_among_63']} 大，前兩名是 {now['giveback_top3_among_63'][0]['date']} 的 {now['giveback_top3_among_63'][0]['giveback_pct']:.1f}%、{now['giveback_top3_among_63'][1]['date']} 的 {now['giveback_top3_among_63'][1]['giveback_pct']:.1f}%）；"
         f"兩天合計仍比 11/23 收盤高 {now['net_two_day_pct']:.2f}%。63 次裡隔天收盤低於大漲前一天收盤的：{q1['below_prior_close_next']} 次。\n",
         "## 數字怎麼算出來的\n",
         f"- 資料：證交所 FMTQIK {cov['first']} 到 {cov['last']}，{cov['sessions']:,} 個交易日；2026-11-13 以前用已追蹤的檔，之後用 2026-11-25 17:46 抓的 11 月官方 JSON（重疊 {cov['merge']['overlap_sessions']} 天逐欄相同；"
         f"每天收盤變化和公布漲跌點數相符，{cov['merge']['close_change_checks']:,} 天、誤差 ≤0.02 點）。11/24、11/25 的收盤、漲跌點數、成交金額由程式對原始 JSON 斷言相等。",
         f"- 大漲日：當天收盤 ÷ 前一交易日收盤 − 1 ≥ 4%，從第 21 個交易日（{cov['first_with_ratio']}，第一個能算量比的日子，沿用 RS-TX-20261124-001）起；{early}。歷史 62 次與上一個研究相同（程式斷言），加上 11/24 共 63 次。",
         "- 隔天漲跌＝大漲日下一個交易日收盤 ÷ 大漲日收盤 − 1。吐回比例＝隔天跌的點數 ÷ 大漲那天漲的點數。",
         "- 「隔天大跌」：主門檻隔天 ≤ −3%；敏感度 ≤ −2%、≤ −4%。對照組：大漲後隔天沒有跌到門檻的那些。",
         "- 之後 h 日（h＝5、20、60）：從隔天（大跌日）再往後數 h 個交易日的收盤，分別對大漲日收盤、對大跌日收盤算報酬。",
         "- 量縮大漲（8 次）：漲 ≥4% 且成交金額 < 前 20 個交易日平均（不含當天）的 0.8 倍，沿用 RS-TX-20261124-001，日期由程式斷言相同。",
         "- 比例的區間是 Wilson 95%；百分位用 numpy 線性內插。沒有做顯著性檢定。\n",
         "## 1. 大漲日隔天的漲跌（63 次）\n",
         "| 組 | n | 隔天漲 | 隔天跌 | 中位數 | 10% 分位 | 最差 | 最好 |", "|---|---|---|---|---|---|---|---|"]
    for name, g in (("全部（含 11/24）", q1["all"]), ("歷史 62 次", q1["historical_62"]), ("7% 制", q1["by_regime"]["7%"]), ("10% 制（2015-06-01 起）", q1["by_regime"]["10%"])):
        s.append(f"| {name} | {g['n']} | {g['up']} | {g['down']} | {f(g['median_pct'])} | {f(g['p10_pct'])} | {f(g['min_pct'])} | {f(g['max_pct'])} |")
    b = q1["baseline_any_session"]
    s += ["", f"| 隔天門檻 | 次數／63 | 比例（Wilson 95%） | 不含 11/24 | 段數 | 7% 制／10% 制 | 平常任何一天跌到門檻（n={b['n']:,}） |", "|---|---|---|---|---|---|---|"]
    for k, v in t.items():
        s.append(f"| ≤ {v['threshold_pct']:.0f}% | {v['count']} | {v['share_pct']}%（{v['wilson95'][0]}–{v['wilson95'][1]}） | {v['count_excl_now']} | {v['episodes']} | "
                 f"{v['by_regime']['7%']}／{v['by_regime']['10%']} | {v['baseline_any_session_count']}（{v['baseline_any_session_share_pct']}%） |")
    s += ["", f"11/25 的 {now['next_ret_pct']:.2f}% 在 63 次隔天漲跌裡排第 {now['rank_next_ret_among_63']}（最差）。放到 1997 年以來任何一天：跌幅 ≥ 這個數的有 {now['any_session_days_at_or_below_now']} 天（含 11/25），"
          f"2015-06-01 以後 {now['any_session_days_at_or_below_now_since_2015_06_01']} 天。11/25 成交金額 {now['next_value_twd']:,} 元，是前 20 日平均（{now['next_avg20_value_twd']:,} 元）的 {now['next_value_ratio']:.2f} 倍。",
          "", "隔天最差的 10 次：", "", "| 大漲日 | 漲幅 | 隔天 | 隔天漲跌 | 吐回 |", "|---|---|---|---|---|"]
    for e in q1["worst10"]:
        s.append(f"| {e['date']} | {f(e['ret_pct'])} | {e['next_date']} | {f(e['next_ret_pct'])} | {e['giveback_pct']:.1f}% |")
    s += ["", "## 2. 大漲、隔天大跌之後怎麼走\n", "逐筆（隔天 ≤ −2% 的 5 次；★＝也 ≤ −3%）：", "",
          "| 大漲日 | 漲幅 | 量比 | 隔天 | 隔天量比 | 吐回 | +5 對大漲日／對大跌日 | +20 | +60 | 收回大漲日收盤 | 制度 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in r["events_drop_le_2"]:
        bk = e["back_above_up_close_sessions"]
        bk = "還不知道" if bk == "not_yet" else ("60 日內沒有" if bk is None else f"{bk} 天（{e['back_above_up_close_date']}）")
        cols = " | ".join(f"{f(e[f'h{h}_from_up_close_pct'])}／{f(e[f'h{h}_from_drop_close_pct'])}" for h in (5, 20, 60))
        star = "★" if e["drop_le_3"] else ""
        s.append(f"| {e['date']}{star} | {f(e['ret_pct'])} | {e['value_ratio']:.2f} | {f(e['next_ret_pct'])} | {e['next_value_ratio']:.2f} | {e['giveback_pct']:.1f}% | {cols} | {bk} | {e['limit_regime']} |")
    s += ["", "分組（之後 h 日從大跌日那天往後數；「漲」＝收盤高於基準；最差、最好括號裡是大漲日日期）：", "",
          "| 門檻 | 組 | h | n | 對大漲日收盤：漲／n、中位數、最差、最好 | 對大跌日收盤：漲／n、中位數、最差、最好 |", "|---|---|---|---|---|---|"]
    for k, g in q2.items():
        for side, nm in (("drop", "隔天大跌"), ("control", "對照")):
            for h in ("5", "20", "60"):
                x = g[side]["by_horizon"][h]
                if not x["n"]:
                    s.append(f"| ≤ {g['threshold_pct']:.0f}% | {nm} | {h} | 0 | — | — |")
                    continue
                a_, c_ = x["from_up_close"], x["from_drop_close"]
                if x["n"] < 5:  # review r1 S3: a median of fewer than 5 values is not printed
                    a_, c_ = dict(a_, median_pct=None), dict(c_, median_pct=None)
                s.append(f"| ≤ {g['threshold_pct']:.0f}%{'（主）' if g['primary'] else ''} | {nm}（{g[side]['n']} 次） | {h} | {x['n']} | "
                         f"{a_['up']}/{x['n']}、{'中位數不列' if a_['median_pct'] is None else f(a_['median_pct'])}、{f(a_['min_pct'])}（{a_['min_date']}）、{f(a_['max_pct'])}（{a_['max_date']}） | "
                         f"{c_['up']}/{x['n']}、{'中位數不列' if c_['median_pct'] is None else f(c_['median_pct'])}、{f(c_['min_pct'])}（{c_['min_date']}）、{f(c_['max_pct'])}（{c_['max_date']}） |")
    s += ["", f"隔天 ≤ −3% 有後續資料的只有 {len(d3)} 次：" + "；".join(f"{d} 之後 60 日 {f(e['h60_from_up_close_pct'])}（對大漲日收盤）" for d, e in d3.items()) + "。只能逐筆講，不能講成「通常」。", "",
          f"## 3. 上一個研究的 8 次量縮大漲：隔天\n", f"8 次裡隔天收在大漲日收盤之下 {q3['next_below_up_close']} 次；跌 ≥2% {q3['next_le_minus2']} 次、跌 ≥3% {q3['next_le_minus3']} 次；隔天中位數 {f(q3['next_ret_median_pct'])}。", "",
          "| 大漲日 | 漲幅 | 量比 | 隔天 | 隔天漲跌 | 點數 | 收在大漲日之下 |", "|---|---|---|---|---|---|---|"]
    for e in q3["events"]:
        s.append(f"| {e['date']} | {f(e['ret_pct'])} | {e['value_ratio']:.2f} | {e['next_date']} | {f(e['next_ret_pct'])} | {e['next_points']:+,.2f} | {'是' if e['next_below_up_close'] else '否'} |")
    nn = q3["post_hoc_ninth_event_review_r1_M2"]["event"]
    s += ["", f"**事後補充（看了研究審第 1 輪才加，不在登記的設計裡）**：{nn['date']} 量比 {nn['value_ratio']:.2f}（< 0.8），也符合上一個研究的量縮大漲定義，是同一定義的第 9 次，隔天 {f(nn['next_ret_pct'])}。"
          f"上一個研究只有 8 次，是因為當時它還沒有隔天。9 次裡隔天跌 ≥3% 就是這 1 次。"
          f"（以歷史 62 次裡 2 次隔天跌 ≥3% 的比例估，8 次裡預期約 {q3['post_hoc_ninth_event_review_r1_M2']['expected_next_le_minus3_in_8_at_historical_rate']:.2f} 次跌 ≥3%，所以 0/8 證明不了什麼。）"]
    p = r["q4_paths"]["events"]
    s += ["", "## 4. 畫圖用逐筆序列\n", f"`results.json` → `q4_paths`：{len(p)} 個事件（8 次量縮大漲 ∪ 隔天 ≤ −2% 的 5 次 ∪ 11/24；2003-02-17 兩邊都有），"
          "每個事件大漲日前 20 到後 61 個交易日的收盤（偏移 −20..+61，共 82 點），大漲日收盤＝100；資料以外是 null（11/24 只有到 +1）。偏移是從大漲日起算，+1＝隔天（大跌日），**+61＝大跌日後第 60 個交易日，和第 2 節表格 h=60 同一點；+60 不是 h=60**。（第一版只畫到 +60，研究審 r1 M1 改成 +61。）程式斷言：長度 82、第 0 天＝100、第 +1 天＝100×(1＋隔天漲跌)、有資料的事件第 +61 點＝100×(1＋h60 報酬)。每筆另記 `base`、`offset_start`、`offset_end_with_data`、`dates_first_last`。"
          "每筆標了屬於哪一組（low_value、drop_le_2/3/4、is_now）。", "", "## 限制與不能說的話\n"]
    s += [f"- {x}" for x in r["limitations_zh"]]
    s += ["", "可以說：",
          "- 1997 年以來加權指數漲 4% 以上 63 次，隔天再跌 4% 以上的只有這一次；不過前一天漲得也多（+8.99%），按「吐回幾成」算，這次吐回 58%，2008 年 9 月和 2020 年 3 月那兩次吐得更多（66.6%、62.2%）。",
          "- 隔天跌 3% 以上之前只有兩次，剛好碰上 2008 年 9 月和 2020 年 3 月：一次之後 60 個交易日比大漲那天低 36.10%，一次高 25.06%。兩次方向完全相反，看不出這種走法之後會怎樣。（背景事件本研究沒有資料，要講須另附來源。）",
          "- 上一集那 8 次量縮大漲，隔天沒有一次跌超過 3%，最差 −2.12%；這次 11/24 也是量縮大漲（量比 0.44），隔天跌了 4.80%，是同一種大漲裡第一次跌這麼深。",
          "",
          "不能說：",
          "- 「大漲後隔天大跌，之後通常會…」；「大漲隔天大跌之後，可能跌 36%、也可能漲 25%」「最壞會跌三成多」「一半跌一半漲」：只有 2 次、方向相反，兩個個案不是範圍或預期，1/2 的 Wilson 約 9–91%。",
          "- 「史上最慘的大漲隔天」「前所未見的反轉」：−4.80% 排第 1 主要因為前一天漲得多；按吐回比例排第 3，1997 年以來任何一天跌 ≥4.80% 有 32 天（2015-06 以後 12 天）。",
          "- 「大漲後隔天比較容易大跌」「大漲後隔天大跌的機率是平常的兩倍」（4.76% vs 2.44%）：Wilson 1.63–13.09% 包含 2.44%；而且大漲日都在高波動期，較公平的基準更高（事後補算，未登記，不放進結果檔）。",
          "- 「大漲後（隔天沒大跌）60 天後通常比較高」：對照組 37/59＝62.7%，任何一天往後 61 列收盤比較高也約 61%（事後補算，未登記），分不出。",
          "- 「大漲後隔天一半會收黑」或用 8 次的 4/8 當訊號：全部 63 次隔天跌 25 次（39.7%），4/8 的 Wilson 21.5–78.5%，分不出。",
          "- 「量縮大漲隔天不會大跌」「這次打破規律所以…」：8 次裡預期本來就只有約 0.26 次跌 ≥3%，0/8 什麼都證明不了；11/24 自己就是第 9 次。",
          "- 「10% 漲跌幅以後比較常見」：事後切分，不能當結論。用盤中高低點講「跌更深」：只用收盤價。任何預測、買賣建議。",
          "- 2020-03-20 和 2020-03-24 是同一段行情（差 2 個交易日），2008-09-08 和對照組的 2008-09-19 窗口幾乎全重疊，不能當獨立例子。",
          "- 圖上線尾標 −36.10%／+25.06% 時，必須用偏移 +61 那一點（和這兩個數字同一點）；用 +60 那一點（−35.31%／+24.91%）標這兩個數字是錯的。",
          "", f"由 `scripts/research/build_tx_surge_next_day.py` 產生，生成時間 {r['generated_at']}。"]
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    main()
