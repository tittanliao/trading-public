#!/usr/bin/env python3
"""RS-TX-20261113-001 revision 2 — post-hoc (NOT pre-registered) descriptive additions.

Everything this script writes is labelled post hoc ("事後補看，不能當結論"). It does not change any
registered definition, threshold, clustering rule or number produced by build_tx_crash_rebound.py;
it only adds, after that runner has written results.json and README.md:
  - the denominator of every horizon (why 27 events give 26 or 24 shares, why the contrast's 101 gives 100);
  - "below the crash-day close": ever within the window (registered) versus still below on the last day (post hoc);
  - the raw triggers where both days moved >= 4% (fall <= -4%, rebound >= +4%) and their 20/60-session paths;
  - runs of the registered clustering that contain two or more raw triggers;
  - robustness: events with Day 2 in 2001 or later (no Saturday sessions), with a 2001+ baseline;
  - the data-calendar note (TWSE Saturday sessions before 2001) as an extra limitation;
  - split by the 20th session (still below the crash-day close or not): 60-session outcomes (viewer question, added later).

Usage: posthoc_tx_crash_rebound.py --prices CSV --merge-report JSON --study-dir DIR
Reads results.json in --study-dir, adds the key "posthoc_rev2", rewrites results.json, appends to README.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

D, U, GAP = 2.5, 2.0, 20
HORIZONS = [5, 20, 60, 250]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (cen - half), 2), round(100 * (cen + half), 2)]


def runs_of(ret, bounce=True):
    raw = [i for i in range(1, len(ret))
           if ret[i - 1] <= -D / 100 and ((ret[i] >= U / 100) if bounce else (ret[i] < U / 100))]
    runs, last = [], -10**9
    for i in raw:
        if (i - 1) - last > GAP:
            runs.append([i])
            last = i
        else:
            runs[-1].append(i)
    return raw, runs


def fwd(c, i, h):
    return round((c[i + h] / c[i] - 1) * 100, 2) if i + h < len(c) else None


def share(c, idx, h):
    r = [fwd(c, i, h) for i in idx if i + h < len(c)]
    k = sum(1 for x in r if x > 0)
    return {"n": len(r), "up": k, "up_pct": round(100 * k / len(r), 2) if r else None,
            "up_wilson95": wilson(k, len(r)), "median_pct": round(float(np.median(r)), 2) if r else None}


def calendar_summary(m: dict) -> dict:
    wk = {"Mon": "週一", "Tue": "週二", "Wed": "週三", "Thu": "週四", "Fri": "週五", "Sat": "週六", "Sun": "週日"}
    yrs = "、".join(f"{y} 年 {n} 天" for y, n in sorted(m["inserted_by_year"].items()))
    wds = "、".join(f"{wk[k]} {v} 天" for k, v in m["inserted_by_weekday"].items())
    diffs = "、".join(f"{x['date']}（Yahoo {x['yahoo']:,.2f}／證交所 {x['twse']:,.2f}）" for x in m["close_diff_over_tol"]) or "沒有"
    drop = "、".join(m["dropped_yahoo_only"]) or "沒有"
    txt = (f"revision 1 用的 Yahoo ^TWII 沒有任何週六的列，但證交所 2001 年以前週六也開盤，所以那段的「連續兩個交易日」不一定真的連續。"
           f"revision 2 用證交所官方「每日市場成交資訊」（FMTQIK，含每天的加權指數收盤）核對 {m['range'][0]} 到 {m['range'][1]} 的每一個交易日："
           f"兩邊同一天都有的 {m['shared']:,} 天，收盤差距最大 {m['close_diff_abs_max']:.2f} 點，超過 {m['tol_points']} 點的 {len(m['close_diff_over_tol'])} 天改用證交所收盤：{diffs}；"
           f"補進 Yahoo 缺的證交所交易日 {len(m['inserted_twse_only'])} 天（{yrs}；{wds}），收盤用證交所的數字；"
           f"拿掉 Yahoo 有、證交所那天沒開盤的列：{drop}。合併後共 {m['merged_rows']:,} 個交易日，事件、基準、對照組都在這份日曆上重算，定義不變。")
    lim = ["補進的交易日只有收盤（證交所這份資料不給開高低），本研究只用收盤，不受影響。",
           "交易日曆以證交所為準；同一天兩邊都有的，收盤沿用 Yahoo，只有差距超過門檻的那幾天改用證交所收盤（列在上面）。"]
    return {"readme_zh": txt, "limitations_zh": lim, "report": m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--merge-report", required=True)
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--twse", help="TWSE FMTQIK CSV; if it has the session after data_end, that exchange close is reported (not added to the data)")
    a = ap.parse_args()
    sd = Path(a.study_dir)
    res = json.loads((sd / "results.json").read_text())
    merge = json.loads(Path(a.merge_report).read_text())

    df = pd.read_csv(a.prices).sort_values("date").reset_index(drop=True)
    c = df["close"].to_numpy(float)
    d = df["date"].astype(str).tolist()
    ret = np.full(len(c), np.nan)
    ret[1:] = c[1:] / c[:-1] - 1
    raw, runs = runs_of(ret)
    kept = [r[0] for r in runs]
    craw, cruns = runs_of(ret, bounce=False)
    ckept = [r[0] for r in cruns]
    assert len(kept) == res["primary"]["n"] and len(ckept) == res["contrast"]["n"], "posthoc must see the runner's events"

    ph = {"label": "事後補看（revision 2, 2026-11-13）：不是事先登記的分析，不能當結論",
          "data_end": d[-1]}

    # 1. denominators
    def den(block, n_events, idx):
        out = {"events": n_events}
        for h in HORIZONS:
            nh = block["by_horizon"][str(h)]["n"]
            missing = [d[i] for i in idx if i + h >= len(c)]
            out[str(h)] = {"n": nh, "up": block["by_horizon"][str(h)].get("up"),
                           "without_window_day2": missing}
        return out
    ph["denominators"] = {
        "note": "Each share is up/n for its horizon; an event enters a horizon only when its Day 2 has h later sessions in the data.",
        "primary": den(res["primary"], res["primary"]["n"], kept),
        "contrast": den(res["contrast"], res["contrast"]["n"], ckept),
        "baseline": {str(h): res["baseline"][str(h)]["n"] for h in HORIZONS},
    }

    # 2. below the Day 1 close: ever within w (registered) vs on the last day w (post hoc)
    bel = {}
    for w in (20, 60):
        valid = [i for i in kept if i + w < len(c)]
        ever = [i for i in valid if c[i + 1:i + w + 1].min() < c[i - 1]]
        end = [i for i in valid if c[i + w] < c[i - 1]]
        bel[str(w)] = {"n": len(valid),
                       "ever_below_within": {"hits": len(ever), "pct": round(100 * len(ever) / len(valid), 2),
                                             "wilson95": wilson(len(ever), len(valid)), "registered": True},
                       "still_below_on_last_day": {"hits": len(end), "pct": round(100 * len(end) / len(valid), 2),
                                                   "wilson95": wilson(len(end), len(valid)), "registered": False,
                                                   "day2_dates": [d[i] for i in end]}}
    ph["below_day1_close"] = bel

    # 3. raw triggers with both moves >= 4%
    big = []
    for run in runs:
        for i in run:
            if ret[i - 1] <= -0.04 and ret[i] >= 0.04:
                big.append({"day1": d[i - 1], "day1_pct": round(100 * ret[i - 1], 2), "day2": d[i],
                            "day2_pct": round(100 * ret[i], 2), "day2_close": round(float(c[i]), 2),
                            "clustered_event": i == run[0], "event_day2": d[run[0]],
                            "fwd20_pct": fwd(c, i, 20), "fwd60_pct": fwd(c, i, 60),
                            "below_day1_within20": bool(c[i + 1:i + 21].min() < c[i - 1]) if i + 20 < len(c) else None})
    ph["both_moves_4pct"] = {"note": "Raw triggers with Day 1 <= -4% and Day 2 >= +4% (post hoc; too few for any share).",
                             "rows": big}

    # 4. runs with two or more raw triggers
    rr = []
    for run in runs:
        if len(run) < 2:
            continue
        rr.append({"triggers": [{"day1": d[i - 1], "day1_pct": round(100 * ret[i - 1], 2), "day2": d[i],
                                 "day2_pct": round(100 * ret[i], 2), "sessions_after_first_day2": i - run[0]} for i in run],
                   "from_first_day2": {"fwd20_pct": fwd(c, run[0], 20), "fwd60_pct": fwd(c, run[0], 60)},
                   "from_last_day2": {"fwd20_pct": fwd(c, run[-1], 20), "fwd60_pct": fwd(c, run[-1], 60)}})
    ph["repeat_runs"] = {"note": "Runs of the registered 20-session clustering that contain >= 2 raw triggers (post hoc).",
                         "raw_triggers": len(raw), "runs": len(runs), "rows": rr}

    # 5. 2001-onward robustness with its own baseline
    yrs = np.array([int(s[:4]) for s in d])
    k01 = [i for i in kept if yrs[i] >= 2001]
    base01 = [i for i in range(1, len(c)) if yrs[i] >= 2001]
    ph["since_2001"] = {"note": "Events with Day 2 in 2001 or later (TWSE stopped Saturday sessions in 2001); baseline = every session from 2001 on.",
                        "n_events": len(k01),
                        "by_horizon": {str(h): share(c, k01, h) for h in (5, 20, 60, 250)},
                        "baseline": {str(h): share(c, base01, h) for h in (5, 20, 60, 250)},
                        "below_day1_within20": sum(1 for i in k01 if i + 20 < len(c) and c[i + 1:i + 21].min() < c[i - 1]),
                        "below_day1_within20_n": sum(1 for i in k01 if i + 20 < len(c))}
    # 6. split by the 20th session: still below the Day 1 close or not (viewer question; post hoc)
    def split_row(i):
        return {"day1": d[i - 1], "day2": d[i], "fwd20_pct": fwd(c, i, 20), "fwd60_pct": fwd(c, i, 60),
                "fwd20_to_60_pct": round((c[i + 60] / c[i + 20] - 1) * 100, 2),
                "below_day1_on_day60": bool(c[i + 60] < c[i - 1])}
    v60 = [i for i in kept if i + 60 < len(c)]
    grp = {}
    for name, members in (("below_on_day20", [i for i in v60 if c[i + 20] < c[i - 1]]),
                          ("not_below_on_day20", [i for i in v60 if c[i + 20] >= c[i - 1]])):
        grp[name] = {"n": len(members),
                     "higher_than_day2_at_60": sum(1 for i in members if c[i + 60] > c[i]),
                     "up_from_day20_to_day60": sum(1 for i in members if c[i + 60] > c[i + 20]),
                     "below_day1_on_day60": sum(1 for i in members if c[i + 60] < c[i - 1])}
        if name == "below_on_day20":
            grp[name]["rows"] = [split_row(i) for i in members]
    ph["day20_line_split"] = {
        "note": ("Post hoc, not pre-registered; asked by a viewer at the channel's test screening. Events with a 60-session window, "
                 "split by whether the 20th-session close is still below the Day 1 (crash-day) close. "
                 "Counts only; n=7 and n=20 are too small to compare the two groups."),
        "registered": False, "source": "viewer test-screening question (2026-11-13)", **grp}
    ph["calendar"] = calendar_summary(merge)
    nxt = None
    if a.twse:
        t = pd.read_csv(a.twse, dtype={"date": str})
        t = t[t["date"] > d[-1]].sort_values("date")
        if len(t):
            x = t.iloc[0]
            need = res["now"]["day2_needed_close_primary"]
            nxt = {"date": x["date"], "twse_close": float(x["taiex_close"]),
                   "change_pct": round((float(x["taiex_close"]) / c[-1] - 1) * 100, 2),
                   "day2_needed_close_primary": need, "meets_day2": bool(float(x["taiex_close"]) >= need),
                   "in_study_data": False,
                   "note": "Exchange-published close of the next session; not in the study data (the Yahoo row for that date is not final)."}
    ph["next_session_exchange_close"] = nxt
    res["posthoc_rev2"] = ph
    (sd / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")

    # README appendix
    p = res["primary"]
    s = ["\n## revision 2 資料修正（交易所交易日曆）\n",
         ph["calendar"]["readme_zh"] + "\n",
         "\n## 分母（每個比例都是「更高的次數／能看到那麼多天的事件數」）\n",
         "| 之後 | 事件組 | 還看不到的事件（第 2 天） | 對照組（大跌但隔天沒大漲） | 還看不到的對照（第 2 天） | 平常日基準 n |",
         "|---|---|---|---|---|---|"]
    for h in HORIZONS:
        pp, cc = ph["denominators"]["primary"][str(h)], ph["denominators"]["contrast"][str(h)]
        s.append(f"| {h} 日 | {pp['up']}/{pp['n']} | {'、'.join(pp['without_window_day2']) or '—'} | {cc['up']}/{cc['n']} | "
                 f"{'、'.join(cc['without_window_day2']) or '—'} | {ph['denominators']['baseline'][str(h)]} |")
    s.append(f"\n事件組併群後 {p['n']} 筆、對照組 {res['contrast']['n']} 筆；比例的分母是各期間的 n，不是事件總數。\n")
    s.append("\n## 事後補看（不是事先登記的分析，不能當結論）\n")
    s.append("### 跌回大跌日收盤以下：「期間內曾經」與「最後一天還在下面」\n")
    s.append("| 期間 | n | 期間內曾經有一天收盤跌回（登記） | 最後一天收盤還在下面（事後補看） |\n|---|---|---|---|")
    for w in ("20", "60"):
        b = bel[w]
        e, l = b["ever_below_within"], b["still_below_on_last_day"]
        s.append(f"| {w} 日 | {b['n']} | {e['hits']}/{b['n']}＝{e['pct']}%（Wilson {e['wilson95'][0]}–{e['wilson95'][1]}） | "
                 f"{l['hits']}/{b['n']}＝{l['pct']}%（Wilson {l['wilson95'][0]}–{l['wilson95'][1]}） |")
    s.append("\n兩個數字都沒有登記平常日基準；大跌之後波動本來就大，不能說「比平常更容易跌回」。\n")
    s.append("### 跌、漲都 ≥ 4% 的原始觸發（次數太少，不算比例）\n")
    s.append("| 第 1 天 | 跌幅 | 第 2 天 | 漲幅 | 是否為併群後事件 | +20 日 | +60 日 | 20 日內曾跌回 |\n|---|---|---|---|---|---|---|---|")
    f = lambda v: "–" if v is None else f"{v:+.2f}%"
    for r in big:
        tag = "是" if r["clustered_event"] else f"否（併入 {r['event_day2']} 那筆）"
        bb = "–" if r["below_day1_within20"] is None else ("是" if r["below_day1_within20"] else "否")
        s.append(f"| {r['day1']} | {r['day1_pct']:+.2f}% | {r['day2']} | {r['day2_pct']:+.2f}% | {tag} | {f(r['fwd20_pct'])} | {f(r['fwd60_pct'])} | {bb} |")
    s.append("\n### 同一段行情內觸發兩次以上（依登記的 20 日併群規則）\n")
    s.append("| 段落 | 觸發（第 1 天／第 2 天，跌幅／漲幅，距第一次第 2 天幾個交易日） | 從第一次第 2 天 +20／+60 日 | 從最後一次第 2 天 +20／+60 日 |\n|---|---|---|---|")
    for k, r in enumerate(rr, 1):
        t = "；".join(f"{x['day1']}／{x['day2']}（{x['day1_pct']:+.2f}%／{x['day2_pct']:+.2f}%，+{x['sessions_after_first_day2']}）" for x in r["triggers"])
        s.append(f"| {k} | {t} | {f(r['from_first_day2']['fwd20_pct'])}／{f(r['from_first_day2']['fwd60_pct'])} | "
                 f"{f(r['from_last_day2']['fwd20_pct'])}／{f(r['from_last_day2']['fwd60_pct'])} |")
    s.append(f"\n原始觸發 {len(raw)} 次分成 {len(runs)} 段；{len(rr)} 段內觸發兩次以上。\n")
    s.append("### 只看 2001 年以後（第 2 天在 2001 年或之後；基準也只用 2001 年以後）\n")
    s.append("| 之後 | 事件：更高／n（Wilson 95%） | 中位數 | 2001 後平常日：更高比例 / 中位數 |\n|---|---|---|---|")
    for h in ("5", "20", "60", "250"):
        x, b = ph["since_2001"]["by_horizon"][h], ph["since_2001"]["baseline"][h]
        s.append(f"| {h} 日 | {x['up']}/{x['n']}＝{x['up_pct']}%（{x['up_wilson95'][0]}–{x['up_wilson95'][1]}） | {x['median_pct']:+.2f}% | {b['up_pct']}%（n={b['n']}）/ {b['median_pct']:+.2f}% |")
    s.append(f"\n2001 年以後 20 日內曾跌回大跌日收盤：{ph['since_2001']['below_day1_within20']}/{ph['since_2001']['below_day1_within20_n']}。\n")
    sp = ph["day20_line_split"]
    lo, hi = sp["below_on_day20"], sp["not_below_on_day20"]
    s.append("### 第 20 天還在大跌日收盤以下的那幾次，三個月後怎樣？（觀眾試聽的提問，事後補看）\n")
    s.append("有 60 日資料的事件，依「第 20 個交易日收盤是否仍低於大跌日（第 1 天）收盤」分兩組。只數次數，兩組各只有幾次，不拿來互相比。\n")
    s.append("| 第 20 天 | 次數 | 第 60 天比反彈日收盤高 | 第 20 天到第 60 天這段上漲 | 第 60 天仍低於大跌日收盤 |\n|---|---|---|---|---|")
    for lab, g in (("還在線下", lo), ("已回到線上", hi)):
        s.append(f"| {lab} | {g['n']} | {g['higher_than_day2_at_60']}/{g['n']} | {g['up_from_day20_to_day60']}/{g['n']} | {g['below_day1_on_day60']}/{g['n']} |")
    s.append("\n第 20 天還在線下的每一次（報酬都從反彈日收盤算）：\n")
    s.append("| 第 1 天 | 第 2 天 | +20 日 | +60 日 | 第 20→60 天 | 第 60 天仍在線下 |\n|---|---|---|---|---|---|")
    for r in lo["rows"]:
        s.append(f"| {r['day1']} | {r['day2']} | {f(r['fwd20_pct'])} | {f(r['fwd60_pct'])} | {f(r['fwd20_to_60_pct'])} | {'是' if r['below_day1_on_day60'] else '否'} |")
    s.append(f"\n線下 {lo['n']} 次、線上 {hi['n']} 次，樣本太小，不做「線下比線上差」之類的比較結論；這是看過結果之後才加的切法，不能當結論。\n")
    if nxt:
        s.append(f"\n### {nxt['date']}（資料之後的下一個交易日）\n")
        s.append(f"證交所已公告 {nxt['date']} 加權指數收盤 {nxt['twse_close']:,.2f}（較 {d[-1]} {nxt['change_pct']:+.2f}%），"
                 f"{'達到' if nxt['meets_day2'] else '沒有達到'}第 2 天門檻 {nxt['day2_needed_close_primary']:,.2f}。"
                 "這一天還沒有放進研究資料（Yahoo 那一列還沒定稿），上面所有數字都不含它。\n")
    s.append("\n## 補充限制（revision 2）\n")
    s += [f"- {x}" for x in ph["calendar"]["limitations_zh"]]
    s.append("- 「最後一天還在下面」、跌漲都 ≥ 4%、同段重複觸發、2001 年以後，全部是看過 revision 1 結果之後才加的事後補看，不能當結論。")
    s.append("- 「第 20 天還在線下的那幾次，三個月後怎樣」來自觀眾試聽的提問，也是事後補看；線下 n=7、線上 n=20，不做兩組比較。")
    s.append("- 「跌回大跌日收盤」沒有登記平常日基準；它只說明跌回去不罕見，不能說比平常更容易。")
    cneg = sum(1 for i in ckept if ret[i] < 0)
    cb = res["contrast"]["below_day1_close"]["20"]
    s.append(f"- 對照組「20 日內跌破第 1 天收盤 {cb['hits']}/{cb['n']}」不能跟事件組直接比：對照組第 2 天沒大漲，{len(ckept)} 筆裡有 {cneg} 筆第 2 天收盤就已經低於第 1 天收盤，起點離那條線近得多。\n")
    with open(sd / "README.md", "a") as fh:
        fh.write("\n".join(s) + "\n")
    print(json.dumps({"below": {w: (bel[w]["ever_below_within"]["hits"], bel[w]["still_below_on_last_day"]["hits"], bel[w]["n"]) for w in bel},
                      "big": [(r["day2"], r["clustered_event"]) for r in big], "repeat_runs": len(rr),
                      "since2001": {h: (v["up"], v["n"]) for h, v in ph["since_2001"]["by_horizon"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
