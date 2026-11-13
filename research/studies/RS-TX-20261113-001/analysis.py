#!/usr/bin/env python3
"""RS-TX-20261113-001 — TAIEX one-day fall then next-session rebound: forward outcomes.

Pre-registered definitions (registered 2026-11-13 before any result):
  Day 1 close-to-close <= -D, Day 2 (next session) close-to-close >= +U.
  Primary D/U = 2.5/2.0; robustness 2.0/1.5 and 3.0/2.5.
  Clustering: a trigger whose Day 1 is within 20 sessions after a retained trigger's Day 2 is dropped.
  Forward return from the Day 2 close to the close 5/20/60/250 sessions later.
  Baseline: every session. Contrast: Day 1 <= -D and Day 2 < +U ("fell, did not bounce").
  Bad half: share whose close fell below the Day 1 close within 20/60 sessions; median of negatives; worst.
  Halves: Day 2 date <= 2011 vs >= 2012 (descriptive only).

Usage: build_tx_crash_rebound.py [--prices CSV] [--out-dir DIR]
Writes only results.json and README.md under --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_ID = "RS-TX-20261113-001"
HORIZONS = [5, 20, 60, 250]
PRIMARY = (2.5, 2.0)
ROBUST = [(2.0, 1.5), (3.0, 2.5)]
CLUSTER_GAP = 20
BOOT = 10_000
SEED = 20261113


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (cen - half), 2), round(100 * (cen + half), 2))


def boot_median(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    med = np.median(x[idx], axis=1)
    return (round(float(np.percentile(med, 2.5)), 2), round(float(np.percentile(med, 97.5)), 2))


def triggers(ret: np.ndarray, d: float, u: float, bounce: bool) -> tuple[list[int], list[int]]:
    """Return (raw Day-2 indices, clustered Day-2 indices)."""
    raw = []
    for i in range(1, len(ret)):
        if ret[i - 1] <= -d / 100 and ((ret[i] >= u / 100) if bounce else (ret[i] < u / 100)):
            raw.append(i)
    kept, last_day2 = [], -10**9
    for i in raw:
        if (i - 1) - last_day2 > CLUSTER_GAP:
            kept.append(i)
            last_day2 = i
    return raw, kept


def horizon_stats(close: np.ndarray, dates: list[str], idx: list[int], h: int,
                  rng: np.random.Generator) -> dict:
    valid = [i for i in idx if i + h < len(close)]
    r = np.array([(close[i + h] / close[i] - 1) * 100 for i in valid])
    n = len(r)
    if n == 0:
        return {"n": 0}
    k = int((r > 0).sum())
    lo, hi = wilson(k, n)
    neg = r[r < 0]
    worst = int(np.argmin(r))
    out = {
        "n": n,
        "up": k,
        "up_pct": round(100 * k / n, 2),
        "up_wilson95": [lo, hi],
        "median_pct": round(float(np.median(r)), 2),
        "median_boot95": list(boot_median(r, rng)),
        "mean_pct": round(float(r.mean()), 2),
        "p10_pct": round(float(np.percentile(r, 10)), 2),
        "p25_pct": round(float(np.percentile(r, 25)), 2),
        "p75_pct": round(float(np.percentile(r, 75)), 2),
        "p90_pct": round(float(np.percentile(r, 90)), 2),
        "min_pct": round(float(r.min()), 2),
        "min_day2_date": dates[valid[worst]],
        "max_pct": round(float(r.max()), 2),
        "neg_n": int(len(neg)),
        "neg_median_pct": round(float(np.median(neg)), 2) if len(neg) else None,
    }
    return out


def below_day1(close: np.ndarray, idx: list[int], within: int) -> dict:
    valid = [i for i in idx if i + within < len(close)]
    hits = 0
    for i in valid:
        if close[i + 1:i + within + 1].min() < close[i - 1]:
            hits += 1
    n = len(valid)
    lo, hi = wilson(hits, n)
    return {"n": n, "hits": hits, "pct": round(100 * hits / n, 2) if n else None, "wilson95": [lo, hi]}


def event_rows(close, dates, ret, idx):
    rows = []
    for i in idx:
        row = {"day1": dates[i - 1], "day2": dates[i],
               "day1_pct": round(100 * ret[i - 1], 2), "day2_pct": round(100 * ret[i], 2),
               "day2_close": round(float(close[i]), 2)}
        for h in HORIZONS:
            row[f"fwd{h}_pct"] = round((close[i + h] / close[i] - 1) * 100, 2) if i + h < len(close) else None
        rows.append(row)
    return rows


def block(close, dates, ret, idx, rng, label):
    out = {"label": label, "n": len(idx), "by_horizon": {}}
    for h in HORIZONS:
        out["by_horizon"][str(h)] = horizon_stats(close, dates, idx, h, rng)
    out["below_day1_close"] = {str(w): below_day1(close, idx, w) for w in (20, 60)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="reproduced/prices_twse_calendar.csv")
    ap.add_argument("--out-dir", default="reproduced")
    a = ap.parse_args()

    df = pd.read_csv(a.prices).sort_values("date").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=float)
    dates = df["date"].astype(str).tolist()
    ret = np.full(len(close), np.nan)
    ret[1:] = close[1:] / close[:-1] - 1
    years = np.array([int(s[:4]) for s in dates])
    rng = np.random.default_rng(SEED)

    results = {
        "study_id": STUDY_ID,
        "schema_version": 1,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "strategy_version": None,
        "method": {
            "event": "Day1 close-to-close <= -D%, Day2 (next session) close-to-close >= +U%",
            "primary_D_U": PRIMARY, "robustness_D_U": ROBUST,
            "cluster_gap_sessions": CLUSTER_GAP,
            "forward_return": "close[Day2 + h] / close[Day2] - 1, h in sessions",
            "horizons": HORIZONS,
            "intervals": "Wilson 95% for shares; percentile bootstrap (10,000 resamples of events, seed 20261113) for medians",
            "baseline": "every session with a measurable forward window (unconditional)",
            "contrast": "Day1 <= -D% and Day2 < +U% (fell, did not bounce), same clustering",
        },
        "coverage": {"first": dates[0], "last": dates[-1], "sessions": len(close)},
        "baseline": {}, "primary": {}, "contrast": {}, "robustness": {}, "halves": {},
        "events_primary": [], "now": {}, "limitations": [],
    }

    all_idx = list(range(1, len(close)))
    for h in HORIZONS:
        results["baseline"][str(h)] = horizon_stats(close, dates, all_idx, h, rng)

    d, u = PRIMARY
    raw, kept = triggers(ret, d, u, True)
    results["primary"] = block(close, dates, ret, kept, rng, f"-{d}% then +{u}%")
    results["primary"]["raw_triggers"] = len(raw)
    results["events_primary"] = event_rows(close, dates, ret, kept)

    craw, ckept = triggers(ret, d, u, False)
    results["contrast"] = block(close, dates, ret, ckept, rng, f"-{d}% then < +{u}%")
    results["contrast"]["raw_triggers"] = len(craw)

    for rd, ru in ROBUST:
        rraw, rkept = triggers(ret, rd, ru, True)
        b = block(close, dates, ret, rkept, rng, f"-{rd}% then +{ru}%")
        b["raw_triggers"] = len(rraw)
        results["robustness"][f"{rd}_{ru}"] = b

    for name, mask in (("1997_2011", years <= 2011), ("2012_2026", years >= 2012)):
        sub = [i for i in kept if mask[i]]
        results["halves"][name] = block(close, dates, ret, sub, rng, name)
        results["halves"][name]["note"] = "descriptive split registered before results; not a conclusion"

    last = len(close) - 1
    results["now"] = {
        "data_end": dates[last],
        "last_close": round(float(close[last]), 2),
        "last_change_pct": round(100 * ret[last], 2),
        "prev_close": round(float(close[last - 1]), 2),
        "day1_qualifies_primary": bool(ret[last] <= -d / 100),
        "day2_needed_close_primary": round(float(close[last] * (1 + u / 100)), 2),
        "day2_known": False,
        "note": "The session after data_end is not in the data; whether the pair qualifies is decided only from the exchange close of that session.",
    }
    results["limitations"] = [
        "TAIEX is a price index without dividends; absolute long-hold returns understate a holder, but event and baseline are both price-only.",
        "Events are clustered (20 sessions) so 5- and 20-session windows do not overlap; 60- and 250-session windows of different events can still overlap, so those intervals are optimistic.",
        "Descriptive only; no causal claim. Thresholds were set so that the 2026-11-12 fall qualifies, registered before any count was computed.",
        "Small samples: the share and median carry their n and interval; a difference inside the interval is 'cannot be separated', not 'no difference'.",
    ]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text(readme(results))
    print(f"primary raw {len(raw)} kept {len(kept)}; contrast raw {len(craw)} kept {len(ckept)}; wrote {out}")


def tbl(bh: dict, base: dict) -> str:
    lines = ["| 之後 | n | 上漲比例（Wilson 95%） | 中位數報酬（bootstrap 95%） | p25 / p75 | 最差（Day 2 日期） | 平常日基準：上漲比例 / 中位數 |",
             "|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        s = bh.get(str(h), {})
        b = base[str(h)]
        if s.get("n", 0) == 0:
            lines.append(f"| {h} 日 | 0 | – | – | – | – | {b['up_pct']}% / {b['median_pct']}% |")
            continue
        lines.append(
            f"| {h} 日 | {s['n']} | {s['up_pct']}%（{s['up_wilson95'][0]}–{s['up_wilson95'][1]}） | "
            f"{s['median_pct']:+.2f}%（{s['median_boot95'][0]:+.2f}–{s['median_boot95'][1]:+.2f}） | "
            f"{s['p25_pct']:+.2f}% / {s['p75_pct']:+.2f}% | {s['min_pct']:+.2f}%（{s['min_day2_date']}） | "
            f"{b['up_pct']}%（n={b['n']}）/ {b['median_pct']:+.2f}% |")
    return "\n".join(lines)


def readme(r: dict) -> str:
    p, c, base = r["primary"], r["contrast"], r["baseline"]
    D, U = r["method"]["primary_D_U"]
    cov = r["coverage"]
    s = []
    s.append(f"# {STUDY_ID} — 台股一天大跌、隔天又大漲，接下來怎麼走？（{cov['first']}–{cov['last']}）\n")
    s.append("每個數字的來源：加權指數（^TWII）日收盤，連續兩個交易日的收盤對收盤漲跌幅。"
             f"事件＝第 1 天收盤跌 ≥ {D}%、第 2 天（下一個交易日）收盤漲 ≥ {U}%；"
             f"同一段行情裡第 1 天落在前一個保留事件第 2 天之後 {CLUSTER_GAP} 個交易日內的重複觸發併掉。"
             "報酬＝第 2 天收盤到之後第 h 個交易日收盤。比例附 n 與 Wilson 95% 區間；中位數附事件重抽樣 10,000 次的 95% 區間。"
             "「平常日基準」＝同一段歷史每一個交易日往後算同樣天數（台股長期本來就往上，這是對照，不是結論）。\n")
    s.append(f"## 主要結果（−{D}% 再 +{U}%；原始觸發 {p['raw_triggers']} 次，併群後 n={p['n']}）\n")
    s.append(tbl(p["by_horizon"], base))
    b20, b60 = p["below_day1_close"]["20"], p["below_day1_close"]["60"]
    s.append(f"\n壞的那一半：第 2 天之後 20 個交易日內收盤曾跌破第 1 天收盤的比例 {b20['pct']}%（{b20['hits']}/{b20['n']}，Wilson {b20['wilson95'][0]}–{b20['wilson95'][1]}）；"
             f"60 個交易日內 {b60['pct']}%（{b60['hits']}/{b60['n']}，Wilson {b60['wilson95'][0]}–{b60['wilson95'][1]}）。"
             + "".join(f" {h} 日負報酬事件 {p['by_horizon'][str(h)].get('neg_n')} 筆、其中位數 {p['by_horizon'][str(h)].get('neg_median_pct')}%；" for h in HORIZONS if p['by_horizon'][str(h)].get('n')) + "\n")
    s.append(f"\n## 對照：跌 ≥ {D}% 但隔天沒漲 ≥ {U}%（原始 {c['raw_triggers']}，併群後 n={c['n']}）\n")
    s.append(tbl(c["by_horizon"], base))
    cb20 = c["below_day1_close"]["20"]
    s.append(f"\n20 個交易日內跌破第 1 天收盤：{cb20['pct']}%（{cb20['hits']}/{cb20['n']}）。\n")
    s.append("\n## 穩健性（鄰近門檻，登記時就排好）\n")
    for k, b in r["robustness"].items():
        s.append(f"\n### {b['label']}（原始 {b['raw_triggers']}，併群後 n={b['n']}）\n")
        s.append(tbl(b["by_horizon"], base))
    s.append("\n## 前後期（登記的描述性切分，不能當結論）\n")
    for k, b in r["halves"].items():
        s.append(f"\n### {k.replace('_', '–')}（n={b['n']}）\n")
        s.append(tbl(b["by_horizon"], base))
    s.append("\n## 事件清單（主要定義，併群後）\n")
    s.append("| 第 1 天 | 跌幅 | 第 2 天 | 漲幅 | 第 2 天收盤 | +5 日 | +20 日 | +60 日 | +250 日 |\n|---|---|---|---|---|---|---|---|---|")
    for e in r["events_primary"]:
        f = lambda v: "–" if v is None else f"{v:+.2f}%"
        s.append(f"| {e['day1']} | {e['day1_pct']:+.2f}% | {e['day2']} | {e['day2_pct']:+.2f}% | {e['day2_close']:,.0f} | {f(e['fwd5_pct'])} | {f(e['fwd20_pct'])} | {f(e['fwd60_pct'])} | {f(e['fwd250_pct'])} |")
    n = r["now"]
    s.append(f"\n## 現在（資料截至 {n['data_end']}）\n")
    s.append(f"{n['data_end']} 收盤 {n['last_close']:,.2f}，較前一日 {n['last_change_pct']:+.2f}%（前一日 {n['prev_close']:,.2f}）。"
             f"第 1 天條件（≤ −{D}%）{'成立' if n['day1_qualifies_primary'] else '不成立'}。"
             f"下一個交易日收盤要 ≥ {n['day2_needed_close_primary']:,.2f} 才符合第 2 天條件（+{U}%）；該日收盤不在本資料內，不用新聞數字判定。\n")
    s.append("\n## 限制\n")
    s += [f"- {x}" for x in r["limitations"]]
    s.append(f"\n來源：`reproduction_inputs.json`；程式 `analysis.py`；產生時間 {r['generated_at']}。\n")
    return "\n".join(s)


def run_all() -> None:
    """Public form: calendar merge, registered runner, post-hoc additions (labelled), in that order."""
    import sys
    import build_tx_twse_calendar_prices as calendar_step
    import posthoc_tx_crash_rebound as posthoc_step
    yahoo = "local-inputs/TWII_1d_to_2026-11-12.csv"
    twse = "local-inputs/TAIEX_FMTQIK_1997-06-02_to_2026-11-13.csv"
    out = Path("reproduced")
    out.mkdir(parents=True, exist_ok=True)
    prices, report = str(out / "prices_twse_calendar.csv"), str(out / "calendar_merge_report.json")
    sys.argv = ["calendar", "--yahoo", yahoo, "--twse", twse, "--out", prices, "--report", report]
    calendar_step.main()
    sys.argv = ["runner", "--prices", prices, "--out-dir", str(out)]
    main()
    sys.argv = ["posthoc", "--prices", prices, "--merge-report", report, "--study-dir", str(out), "--twse", twse]
    posthoc_step.main()


if __name__ == "__main__":
    run_all()
