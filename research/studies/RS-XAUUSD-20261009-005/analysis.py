#!/usr/bin/env python3
"""RS-XAUUSD-20261009-005 — CFTC insight search: who is doing what, and what gold does next.

Design registered before running (research/studies/RS-XAUUSD-20261009-005/decision_log.md):
28 positioning conditions known at the report date × 9 outcomes measured from the Friday
publication close, discovery 2012-12 → 2019 and validation 2020 → 2026, labels fixed in
advance, a circular-shift max-|z| family correction, a report-week placebo and a gold-trend
regime check.

Usage:
    /opt/homebrew/bin/python3.12 scripts/research/build_cftc_insights.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import screen_harness as sh  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261009-005"
OUTPUT_DIR = Path("reproduced")
SERIES = Path("local-inputs/cftc_weekly_series_archive.csv")
DAILY = Path("local-inputs/FX_IDC_XAUUSD, 1D.csv")
TAIPEI = timezone(timedelta(hours=8))

START = pd.Timestamp("2012-12-11")
SPLIT = pd.Timestamp("2020-01-01")
SEED = 20261009
BLOCK = 4
DRAWS = 2000
SHIFTS = 499
MIN_SHIFT = 52
EXTREME_WINDOW, EXTREME_MIN = 156, 104
STREAK = 4
TREND_WEEKS = 40

GROUPS = {"mm": "管理基金（大戶）", "retail": "散戶（非報告者）", "prod_merc": "生產商／貿易商",
          "swap": "交換交易商", "other": "其他報告者"}
OUTCOMES = {
    "mon_tue": ("下週一二報酬", "direction"),
    "week1": ("公布後第 1 週報酬", "direction"),
    "week2": ("第 2 週報酬", "direction"),
    "week4": ("公布後 4 週報酬", "direction"),
    "shake": ("先跌再漲（洗盤）比例", "path"),
    "pump": ("先漲再跌比例", "path"),
    "abs1": ("第 1 週漲跌幅大小", "volatility"),
    "range1": ("第 1 週高低區間", "volatility"),
    "dip1": ("第 1 週最大回檔深度", "volatility"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_daily() -> pd.DataFrame:
    d = pd.read_csv(DAILY, parse_dates=["time"]).rename(columns={"time": "session"})
    return d.sort_values("session").reset_index(drop=True)[["session", "open", "high", "low", "close"]]


def weekly_frame(series: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    s = series.sort_values("report_date").reset_index(drop=True).copy()
    sessions = daily["session"].to_numpy()
    closes = daily["close"].to_numpy()
    last = daily["session"].max()

    def close_at(day: pd.Timestamp) -> float:
        if day > last:
            return np.nan
        i = np.searchsorted(sessions, np.datetime64(day), side="right") - 1
        return float(closes[i]) if i >= 0 else np.nan

    def window(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
        return daily[(daily["session"] > lo) & (daily["session"] <= hi)]

    rows = []
    for t in s["report_date"]:
        f = t + pd.Timedelta(days=3)
        c_prev, c_t, c_f = close_at(t - pd.Timedelta(days=7)), close_at(t), close_at(f)
        c_tue = close_at(t + pd.Timedelta(days=7))
        c_w1, c_w2, c_w4 = (close_at(f + pd.Timedelta(days=d)) for d in (7, 14, 28))
        w = window(f, f + pd.Timedelta(days=7)) if f + pd.Timedelta(days=7) <= last else window(f, f)
        h = window(t, f)
        pct = lambda a, b: 100 * (a / b - 1) if b and not np.isnan(a) and not np.isnan(b) else np.nan  # noqa: E731
        w1, w2 = pct(c_w1, c_f), pct(c_w2, c_w1)
        rows.append({
            "report_date": t,
            "gold_week_pct": pct(c_t, c_prev),
            "close_t": c_t,
            "mon_tue": pct(c_tue, c_f), "week1": w1, "week2": w2, "week4": pct(c_w4, c_f),
            "shake": float(w1 < 0 and w2 > 0) if not (np.isnan(w1) or np.isnan(w2)) else np.nan,
            "pump": float(w1 > 0 and w2 < 0) if not (np.isnan(w1) or np.isnan(w2)) else np.nan,
            "abs1": abs(w1) if not np.isnan(w1) else np.nan,
            "range1": 100 * (w["high"].max() - w["low"].min()) / c_f if len(w) and c_f else np.nan,
            "dip1": 100 * (w["low"].min() / c_f - 1) if len(w) and c_f else np.nan,
            # placebo: the report week, already history when the report is published
            "hist_dir": pct(c_f, c_t),
            "hist_range": 100 * (h["high"].max() - h["low"].min()) / c_t if len(h) and c_t else np.nan,
        })
    out = s.merge(pd.DataFrame(rows), on="report_date")
    out["trend_up"] = out["close_t"] > out["close_t"].rolling(TREND_WEEKS, min_periods=TREND_WEEKS).mean()
    return out


def conditions(d: pd.DataFrame) -> dict[str, dict]:
    """28 conditions, each a boolean array known at the report date."""
    c: dict[str, dict] = {}
    mm, rt, ot = d["mm_net_chg"], d["retail_net_chg"], d["other_net_chg"]
    gold = d["gold_week_pct"]

    def add(cid, family, label, mask):
        c[cid] = {"family": family, "label_zh": label, "mask": mask.fillna(False).to_numpy(bool)}

    add("J1", "joint", "大戶與散戶同時加多（你說的情況）", (mm > 0) & (rt > 0))
    add("J2", "joint", "大戶與散戶同時減多", (mm < 0) & (rt < 0))
    add("J3", "joint", "大戶加多、散戶減多", (mm > 0) & (rt < 0))
    add("J4", "joint", "大戶減多、散戶加多", (mm < 0) & (rt > 0))
    add("J5", "joint", "大戶、其他報告者、散戶全部加多", (mm > 0) & (ot > 0) & (rt > 0))
    add("J6", "joint", "大戶、其他報告者、散戶全部減多", (mm < 0) & (ot < 0) & (rt < 0))

    for g, label in GROUPS.items():
        level = d[f"{g}_net_pct_oi"]
        rank = level.rolling(EXTREME_WINDOW + 1, min_periods=EXTREME_MIN + 1).apply(
            lambda w: (w[:-1] < w[-1]).mean(), raw=True)
        add(f"X-{g}-hi", "extreme", f"{label}淨部位在過去三年最高 10%", rank >= 0.9)
        add(f"X-{g}-lo", "extreme", f"{label}淨部位在過去三年最低 10%", rank <= 0.1)

    for g, series, label in (("mm", mm, "大戶"), ("retail", rt, "散戶")):
        sign = np.sign(series).fillna(0).to_numpy()
        # run[i] = consecutive weeks ending at i with the same direction as week i
        run = np.ones(len(sign))
        for i in range(1, len(sign)):
            run[i] = run[i - 1] + 1 if sign[i] != 0 and sign[i] == sign[i - 1] else 1
        sign = pd.Series(sign, index=series.index)
        prev_sign = sign.shift(1)
        prior_run = pd.Series(run, index=series.index).shift(1)
        add(f"T-{g}-buy2sell", "turn", f"{label}連續加多 {STREAK} 週以上後轉為減多",
            (prev_sign > 0) & (sign < 0) & (prior_run >= STREAK))
        add(f"T-{g}-sell2buy", "turn", f"{label}連續減多 {STREAK} 週以上後轉為加多",
            (prev_sign < 0) & (sign > 0) & (prior_run >= STREAK))

    add("D-mm-up-cut", "divergence", "金價上週漲、大戶卻減多", (gold > 0) & (mm < 0))
    add("D-mm-down-add", "divergence", "金價上週跌、大戶卻加多", (gold < 0) & (mm > 0))
    add("D-retail-up-cut", "divergence", "金價上週漲、散戶卻減多", (gold > 0) & (rt < 0))
    add("D-retail-down-add", "divergence", "金價上週跌、散戶卻加多（接刀）", (gold < 0) & (rt > 0))

    oi = d["open_interest"].diff()
    add("O-up-up", "oi", "未平倉量增加、金價漲", (oi > 0) & (gold > 0))
    add("O-up-down", "oi", "未平倉量增加、金價跌", (oi > 0) & (gold < 0))
    add("O-down-up", "oi", "未平倉量減少、金價漲（空單回補）", (oi < 0) & (gold > 0))
    add("O-down-down", "oi", "未平倉量減少、金價跌（多單了結）", (oi < 0) & (gold < 0))
    return c


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def block_bootstrap_p(m: np.ndarray, v: np.ndarray, seed: int) -> float | None:
    """Moving-block bootstrap of the condition-minus-other difference, two-sided p.

    Same definition as screen_harness.block_bootstrap_effect (contiguous blocks of BLOCK
    weeks, resamples with fewer than 10 on either side dropped, +1 finite-resample
    correction), vectorised because this study runs it about a thousand times.
    """
    n = v.size
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / BLOCK)
    starts = rng.integers(0, max(n - BLOCK + 1, 1), size=(DRAWS, blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(DRAWS, -1)[:, :n]
    idx = np.minimum(idx, n - 1)
    mm, vv = m[idx], v[idx]
    n1 = mm.sum(axis=1)
    n0 = n - n1
    keep = (n1 >= 10) & (n0 >= 10)
    if not keep.any():
        return None
    s1 = np.where(mm, vv, 0).sum(axis=1)
    s0 = vv.sum(axis=1) - s1
    diff = (s1[keep] / n1[keep]) - (s0[keep] / n0[keep])
    above = int((diff > 0).sum())
    drawn = diff.size
    low, high = (above + 1) / (drawn + 1), (drawn - above + 1) / (drawn + 1)
    return round(2 * min(low, high, 0.5), 4)


def score(mask: np.ndarray, y: np.ndarray, seed: int) -> dict:
    ok = ~np.isnan(y)
    m, v = mask[ok], y[ok]
    a, b = v[m], v[~m]
    if a.size < 10 or b.size < 10:
        return {"n": int(a.size), "n_other": int(b.size), "effect": None}
    effect = float(a.mean() - b.mean())
    sigma = float(v.std(ddof=1))
    bound = sh.smallest_resolvable(a.size, b.size, sigma)
    p = block_bootstrap_p(m, v, seed)
    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return {"n": int(a.size), "n_other": int(b.size), "mean_with": round(float(a.mean()), 4),
            "mean_without": round(float(b.mean()), 4), "effect": round(effect, 4),
            "bound": bound, "p": p, "z": round(effect / se, 3) if se else None}


def label(disc: dict, val: dict, full: dict) -> str:
    if disc.get("effect") is None or val.get("effect") is None or full.get("effect") is None:
        return "樣本不足"
    same = np.sign(disc["effect"]) == np.sign(val["effect"]) and disc["effect"] != 0
    if (abs(disc["effect"]) > (disc["bound"] or 1e9) and (disc["p"] or 1) < 0.05
            and same and (val["p"] or 1) < 0.05):
        return "可用"
    if same and (full["p"] or 1) < 0.05:
        return "參考"
    return "無證據"


def family_max_z(d: pd.DataFrame, conds: dict, rng: np.random.Generator) -> tuple[float, list[float]]:
    """Circular-shift max-|z| null over every (condition, outcome) pair in discovery."""
    disc = (d["report_date"] < SPLIT).to_numpy()
    masks = np.array([c["mask"][disc] for c in conds.values()])
    ys = np.array([d.loc[disc, o].to_numpy(float) for o in OUTCOMES])
    n = masks.shape[1]

    def max_z(shift: int) -> float:
        best = 0.0
        for y in ys:
            ys_ = np.roll(y, shift)
            ok = ~np.isnan(ys_)
            for m in masks:
                mm, vv = m[ok], ys_[ok]
                a, b = vv[mm], vv[~mm]
                if a.size < 10 or b.size < 10:
                    continue
                se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
                if se:
                    best = max(best, abs(a.mean() - b.mean()) / se)
        return best

    observed = max_z(0)
    null = [max_z(int(rng.integers(MIN_SHIFT, n - MIN_SHIFT))) for _ in range(SHIFTS)]
    return observed, null


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    series = pd.read_csv(SERIES, parse_dates=["report_date"])
    daily = load_daily()
    d = weekly_frame(series, daily)
    conds = conditions(d)
    d = d[d["report_date"] >= START].reset_index(drop=True)
    for c in conds.values():
        c["mask"] = c["mask"][-len(d):]
    disc = (d["report_date"] < SPLIT).to_numpy()
    val = ~disc

    observed_max, null = family_max_z(d, conds, np.random.default_rng(SEED))
    null_arr = np.array(null)

    tests = []
    for cid, c in conds.items():
        for oid, (olabel, kind) in OUTCOMES.items():
            y = d[oid].to_numpy(float)
            stream = sh.stream_for(SEED, f"{cid}:{oid}").randrange(2**31)
            res_d = score(c["mask"][disc], y[disc], stream)
            res_v = score(c["mask"][val], y[val], stream + 3)
            res_f = score(c["mask"], y, stream + 4)
            placebo_col = "hist_range" if kind == "volatility" else "hist_dir"
            res_p = score(c["mask"], d[placebo_col].to_numpy(float), stream + 5)
            verdict = label(res_d, res_v, res_f)
            row = {"condition": cid, "condition_zh": c["label_zh"], "family": c["family"],
                   "outcome": oid, "outcome_zh": olabel, "kind": kind,
                   "discovery": res_d, "validation": res_v, "full": res_f, "placebo_report_week": res_p,
                   "family_adjusted_p": round(float((null_arr >= abs(res_d["z"])).mean()), 4)
                   if res_d.get("z") is not None else None,
                   "label": verdict}
            if verdict in ("可用", "參考"):
                trend = d["trend_up"].to_numpy(bool)
                up = score(c["mask"][trend], y[trend], stream + 1)
                down = score(c["mask"][~trend], y[~trend], stream + 2)
                row["regime"] = {"gold_uptrend": up, "gold_downtrend": down,
                                 "same_sign": (up.get("effect") is not None and down.get("effect") is not None
                                               and np.sign(up["effect"]) == np.sign(down["effect"]))}
            tests.append(row)

    # The page table: every result labelled 可用 / 參考, plus every outcome of the stated case (J1).
    highlights = []
    for t in tests:
        if t["label"] in ("可用", "參考") or t["condition"] == "J1":
            reg = t.get("regime") or {}
            highlights.append({
                "condition_zh": t["condition_zh"], "outcome_zh": t["outcome_zh"], "label": t["label"],
                "n": t["full"].get("n"),
                "effect_discovery": t["discovery"].get("effect"), "effect_validation": t["validation"].get("effect"),
                "effect_full": t["full"].get("effect"), "bound_full": t["full"].get("bound"),
                "p_full": t["full"].get("p"), "placebo_report_week": t["placebo_report_week"].get("effect"),
                "family_adjusted_p": t["family_adjusted_p"],
                "regime_same_sign": reg.get("same_sign"),
            })
    by_label: dict[str, int] = {}
    for t in tests:
        by_label[t["label"]] = by_label.get(t["label"], 0) + 1
    baseline = {o: {"discovery": round(float(np.nanmean(d.loc[disc, o])), 4),
                    "validation": round(float(np.nanmean(d.loc[val, o])), 4)} for o in OUTCOMES}
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "market": "XAUUSD", "strategy": "none — CFTC positioning insight search",
        "method": {"series": "cftc_weekly_series_archive.csv", "prices": "FX_IDC daily",
                   "weeks": int(len(d)), "discovery_weeks": int(disc.sum()), "validation_weeks": int(val.sum()),
                   "from": str(d["report_date"].min().date()), "to": str(d["report_date"].max().date()),
                   "split": str(SPLIT.date()), "block_weeks": BLOCK, "bootstrap": DRAWS, "seed": SEED,
                   "family_test": f"circular-shift max-|z|, {SHIFTS} shifts of at least {MIN_SHIFT} weeks",
                   "conditions": len(conds), "outcomes": len(OUTCOMES), "tests": len(tests)},
        "baseline": baseline,
        "family": {"observed_max_abs_z": round(observed_max, 3),
                   "null_median": round(float(np.median(null_arr)), 3),
                   "null_p95": round(float(np.quantile(null_arr, 0.95)), 3),
                   "family_p": round(float((null_arr >= observed_max).mean()), 4)},
        "by_label": by_label,
        "highlights": highlights,
        "tests": tests,
        "limitations": [
            "Weekly data: a pattern inside a week, or only in some regimes, is invisible in a weekly split.",
            "Prices start 2012-12; positions from 2006 are used only as percentile history.",
            "Labels were fixed before running; 252 correlated tests, protected by an out-of-sample half and a "
            "circular-shift family test rather than by a correction that assumes independence.",
            "No result changes strategy logic, risk or the analysis workflow without a separate decision.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n",
        encoding="utf-8")
    print(json.dumps({"weeks": len(d), "tests": len(tests), "by_label": by_label, "family": payload["family"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
