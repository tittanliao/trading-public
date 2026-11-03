"""RS-XAUUSD-20261103-001 — gold and the CPI surprise (pre-registered in decision_log.md, 2026-11-03).

Sample: the 33 CPI releases 2024-01..2026-10 with 30m data (same bars and alignment as RS-XAUUSD-20261102-002).
Surprise: actual minus consensus (cpi_surprise_2024_2026.csv, each value with a source URL).
Outcomes from the release-bar open: R30 (bar 1 close), R4h (bar 8 close), Rday (close of the 16:30 NY bar).
Rule backtests A–D as pre-registered; costs USD 1.5 round trip per oz (sensitivity 3.0).

usage: python3 scripts/research/build_xauusd_cpi_surprise.py [--dry]
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import hashlib
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

# Self-contained copy of the loaders and statistics used by RS-XAUUSD-20261102-002 (Public analysis.py must not
# import sibling runners). Same files, same bar alignment.
ROOT = Path(__file__).resolve().parents[3]
_EXPORTS = Path("local-inputs")
_M30 = ("FX_IDC_XAUUSD, 30.csv", "FX_IDC_XAUUSD, 30_to_2026-10-03.csv")
_M15 = "FX_IDC_XAUUSD, 15_to_2026-10-31.csv"
_NY = ZoneInfo("America/New_York")


def _sha(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _wilson(k: int, n: int):
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]


def _mean_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Difference in means with a 95% bound, 1.96 x Welch (unpooled) standard error."""
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = a.mean() - b.mean()
    return {"diff": round(float(d), 4), "bound_95": round(1.96 * se, 4), "resolved": bool(abs(d) > 1.96 * se)}


def _share(mask: np.ndarray) -> dict:
    k, n = int(mask.sum()), int(len(mask))
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "ci95_pct": _wilson(k, n)}


def _load_30m() -> pd.DataFrame:
    parts = [pd.read_csv(_EXPORTS / n, usecols=["time", "open", "high", "low", "close"]) for n in _M30]
    m15 = pd.read_csv(_EXPORTS / _M15, usecols=["time", "open", "high", "low", "close"])
    m15["time"] = pd.to_datetime(m15["time"], utc=True)
    r = m15.set_index("time").sort_index().resample("30min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    parts.append(r)
    allf = []
    for f in parts:
        f = f.copy()
        f["time"] = pd.to_datetime(f["time"], utc=True)
        allf.append(f)
    return pd.concat(allf).drop_duplicates("time", keep="first").set_index("time").sort_index().astype(float)


def _release_utc(day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(datetime(day.year, day.month, day.day, 8, 30, tzinfo=_NY)).tz_convert("UTC")


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


base = _NS(ROOT=ROOT, EXPORTS=_EXPORTS, M30_FILES=_M30, M15_FILE=_M15, NY=_NY,
                       TAIPEI_TZ=timezone(timedelta(hours=8)), sha=_sha, mean_diff=_mean_diff, share=_share,
                       load_30m=_load_30m, release_utc=_release_utc)
STUDY_ID = "RS-XAUUSD-20261103-001"
OUT = Path("reproduced")
SURPRISE = Path("local-inputs/cpi_surprise_2024_2026.csv")
BARS_4H = 8
COSTS = (1.5, 3.0)


def load_surprise() -> dict:
    rows = {}
    with open(SURPRISE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            def num(k):
                v = (r.get(k) or "").strip()
                return float(v) if v else None
            rows[r["release_date"]] = {k: num(k) for k in ("head_mom_act", "head_mom_cons", "core_mom_act", "core_mom_cons",
                                                         "head_yoy_act", "head_yoy_cons", "core_yoy_act", "core_yoy_cons")}
            rows[r["release_date"]]["consensus_source_class"] = r.get("consensus_source_class", "")
    return rows


def event_bars(m: pd.DataFrame, day: pd.Timestamp) -> dict | None:
    start = base.release_utc(day)
    if start not in m.index:
        return None
    i = m.index.get_loc(start)
    win = m.iloc[i:i + BARS_4H]
    if len(win) < BARS_4H or (win.index[-1] - win.index[0]) != pd.Timedelta(minutes=30 * (BARS_4H - 1)):
        return None
    close_bar = pd.Timestamp(datetime(day.year, day.month, day.day, 16, 30, tzinfo=base.NY)).tz_convert("UTC")
    if close_bar not in m.index:
        return None
    o = float(win["open"].iloc[0])
    c1 = float(win["close"].iloc[0])
    c8 = float(win["close"].iloc[-1])
    cd = float(m.loc[close_bar, "close"])
    return {"open": o, "c1": c1, "c8": c8, "cday": cd,
            "R30": (c1 / o - 1) * 100, "R4h": (c8 / o - 1) * 100, "Rday": (cd / o - 1) * 100}


def welch(a: np.ndarray, b: np.ndarray) -> dict | None:
    if len(a) < 2 or len(b) < 2:
        return None
    return base.mean_diff(a, b)


def ols_slope(x: np.ndarray, y: np.ndarray) -> dict | None:
    if len(x) < 3 or np.var(x) == 0:
        return None
    xm, ym = x.mean(), y.mean()
    b = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    resid = y - (ym + b * (x - xm))
    se = math.sqrt(float((resid ** 2).sum()) / (len(x) - 2) / float(((x - xm) ** 2).sum()))
    return {"slope_pct_per_0p1": round(b, 4), "bound_95": round(1.96 * se, 4), "resolved": bool(abs(b) > 1.96 * se),
            "n": int(len(x))}


def surprise_block(E: pd.DataFrame, col: str) -> dict:
    s = E[col]
    ok = E[s.notna()]
    hot, cool, inl = ok[ok[col] > 0], ok[ok[col] < 0], ok[ok[col] == 0]
    out = {"n": int(len(ok)), "n_hot": int(len(hot)), "n_inline": int(len(inl)), "n_cool": int(len(cool))}
    for k in ("R30", "R4h", "Rday"):
        out[k] = {
            "hot_mean": round(float(hot[k].mean()), 3) if len(hot) else None,
            "inline_mean": round(float(inl[k].mean()), 3) if len(inl) else None,
            "cool_mean": round(float(cool[k].mean()), 3) if len(cool) else None,
            "hot_minus_cool": welch(hot[k].values, cool[k].values),
        }
    # "with the surprise": hot -> gold down, cool -> gold up (in-line excluded)
    sig = ok[ok[col] != 0]
    for k in ("R30", "R4h", "Rday"):
        with_ = (np.sign(sig[k].values) == -np.sign(sig[col].values))
        out[f"{k}_with_surprise"] = base.share(with_)
    out["slope_R30"] = ols_slope((ok[col] * 10).values, ok["R30"].values)  # per 0.1 pp
    out["slope_R4h"] = ols_slope((ok[col] * 10).values, ok["R4h"].values)
    return out


def rule_stats(pnl_usd: list[float], pnl_pct: list[float], cost: float) -> dict:
    a = np.array(pnl_usd) - cost
    p = np.array(pnl_pct)
    n = len(a)
    if n == 0:
        return {"trades": 0}
    se = a.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {"trades": n, "win": base.share(a > 0), "mean_usd": round(float(a.mean()), 2),
            "mean_bound_95_usd": round(1.96 * se, 2) if n > 1 else None,
            "resolved_vs_0": bool(n > 1 and abs(a.mean()) > 1.96 * se),
            "total_usd": round(float(a.sum()), 1), "worst_usd": round(float(a.min()), 1),
            "mean_pct_gross": round(float(p.mean()), 3)}


def rules(E: pd.DataFrame, col: str) -> dict:
    out = {}
    for cost in COSTS:
        key = f"cost_{cost}"
        # B: follow bar 1 from bar-1 close to bar-8 close
        d = np.sign(E["R30"].values)
        m = d != 0
        B_usd = (d[m] * (E["c8"].values[m] - E["c1"].values[m])).tolist()
        B_pct = (d[m] * (E["c8"].values[m] / E["c1"].values[m] - 1) * 100).tolist()
        # C: trade the surprise from bar-1 close to bar-8 close (hot -> short, cool -> long)
        s = E[col].values
        ok = ~np.isnan(s) & (s != 0)
        dc = -np.sign(s[ok])
        C_usd = (dc * (E["c8"].values[ok] - E["c1"].values[ok])).tolist()
        C_pct = (dc * (E["c8"].values[ok] / E["c1"].values[ok] - 1) * 100).tolist()
        # D: trade the surprise from release open to bar-1 close (not executable: upper bound)
        D_usd = (dc * (E["c1"].values[ok] - E["open"].values[ok])).tolist()
        D_pct = (dc * (E["c1"].values[ok] / E["open"].values[ok] - 1) * 100).tolist()
        out[key] = {"A_no_position": {"trades": 0, "mean_usd": 0.0},
                    "B_follow_first_bar": rule_stats(B_usd, B_pct, cost),
                    "C_surprise_after_30m": rule_stats(C_usd, C_pct, cost),
                    "D_surprise_at_release_upper_bound": rule_stats(D_usd, D_pct, cost)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    sur = load_surprise()
    m = base.load_30m()
    rows, skipped = [], []
    for ds, v in sorted(sur.items()):
        b = event_bars(m, pd.Timestamp(ds))
        if b is None:
            skipped.append(ds)
            continue
        r = {"date": ds, **b, **v}
        r["core_surprise"] = None if v["core_mom_act"] is None or v["core_mom_cons"] is None else round(v["core_mom_act"] - v["core_mom_cons"], 1)
        r["head_surprise"] = None if v["head_mom_act"] is None or v["head_mom_cons"] is None else round(v["head_mom_act"] - v["head_mom_cons"], 1)
        rows.append(r)
    E = pd.DataFrame(rows)
    E["core_surprise"] = E["core_surprise"].astype(float)
    E["head_surprise"] = E["head_surprise"].astype(float)
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(base.TAIPEI_TZ).isoformat(timespec="seconds"),
        "method": {
            "sample": "CPI releases 2024-01..2026-10 with FX_IDC:XAUUSD 30m data (release bar opens 08:30 ET)",
            "surprise": "actual minus consensus MoM (0.1 pp steps); primary core, secondary headline; hot > 0, cool < 0",
            "outcomes": "R30 = bar 1 close, R4h = bar 8 close, Rday = close of the 16:30 NY bar; all vs release-bar open, %",
            "tests": "hot minus cool mean (1.96 x Welch (unpooled) SE); share moving with the surprise (Wilson); OLS slope per 0.1 pp; the 4h slope is post-hoc (only the 30m slope was pre-registered)",
            "rules": "B follow bar 1, C trade surprise after 30m, D trade surprise at release (upper bound, not executable); "
                     "entries and exits at 30m closes; costs USD per oz round trip",
        },
        "n_events": int(len(E)), "skipped": skipped,
        "core": surprise_block(E, "core_surprise"),
        "headline": surprise_block(E, "head_surprise"),
        "rules_core": rules(E, "core_surprise"),
        "rules_headline": rules(E, "head_surprise"),
        "consensus_source_classes": E["consensus_source_class"].value_counts().to_dict(),
        "events": [{k: (round(v, 3) if isinstance(v, float) and not math.isnan(v) else (None if isinstance(v, float) else v))
                    for k, v in r.items() if k in ("date", "R30", "R4h", "Rday", "core_surprise", "head_surprise",
                                                   "core_mom_act", "core_mom_cons", "head_mom_act", "head_mom_cons", "open")}
                   for r in rows],
        "limitations": [
            f"n = {int(E['core_surprise'].notna().sum())} releases with MoM figures ({len(E)} for rule B) split into hot / in-line / cool; most bounds are wide.",
            "Consensus figures come from secondary sources (polls quoted in news, or calendars), not an official series.",
            "Fills at 30m bar closes ignore slippage inside the bar; rule D (from the release open) is not executable.",
            "The delayed 2025-10-24 and 2026-02-13 releases are kept.",
            "Generic rule backtests for illustration; not trading advice.",
        ],
    }
    if a.dry:
        print(json.dumps({k: v for k, v in payload.items() if k != "events"}, ensure_ascii=False, indent=1, default=str))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")

    def loc(p: Path) -> str:
        return str(p.relative_to(ROOT))
    inputs = [{"role": f"price_30m_{k}", "source_system": "TradingView FX_IDC:XAUUSD chart export",
               "locator": loc(base.EXPORTS / n), **base.sha(base.EXPORTS / n), "immutable": True, "privacy": "private_raw_source"}
              for k, n in enumerate(base.M30_FILES, 1)]
    inputs += [{"role": "price_15m", "source_system": "TradingView FX_IDC:XAUUSD chart export",
                "locator": loc(base.EXPORTS / base.M15_FILE), **base.sha(base.EXPORTS / base.M15_FILE), "immutable": True,
                "privacy": "private_raw_source", "note": "resampled to 30m"},
               {"role": "cpi_surprise", "source_system": "BLS releases (actual) and pre-release consensus from news polls or calendars",
                "locator": loc(SURPRISE), **base.sha(SURPRISE), "immutable": False, "privacy": "public",
                "observed": "2026-11-03", "note": "every value carries its source URL in the CSV"}]
    (OUT / "source_manifest.json").write_text(json.dumps({"schema_version": 1, "study_id": STUDY_ID, "inputs": inputs},
                                                         ensure_ascii=False, indent=2) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
