#!/usr/bin/env python3
"""RS-TX-20261018-004 — the 2x Taiwan 50 barbells through 1997–2026 crashes, and five rebalancing rules.

Design fixed in research/studies/RS-TX-20261018-004/decision_log.md before any outcome. All in TWD. The loader
`returns()` is shared with RS-TX-20261018-005.

    python3.12 scripts/research/build_tw_barbell_stress.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-TX-20261018-004"
TAIPEI = timezone(timedelta(hours=8))
Y = Path("local-inputs")
F = Path("local-inputs")
GOLD_D = Path("local-inputs/XAUUSD_1d.csv")
GOLD_W = Path("local-inputs/XAUUSD_1wk.csv")
DIVIDEND, COST, SPLICE = 0.03, 0.0025, pd.Timestamp("2007-01-01")
PORTFOLIOS = {
    "P0": {"etf": 1.0},
    "P1": {"lev2": 0.5, "cash": 0.5},
    "P2": {"lev2": 0.5, "tbill": 0.5},
    "P3": {"lev2": 0.5, "gold": 0.5},
    "P4": {"lev2": 0.5, "gold": 0.25, "tbill": 0.25},
    "P5": {"etf": 0.5, "gold": 0.5},
}
LABELS = {"P0": "100% Taiwan 50", "P1": "50% 2x + 50% cash", "P2": "50% 2x + 50% T-bills", "P3": "50% 2x + 50% gold",
          "P4": "50% 2x + 25% gold + 25% T-bills", "P5": "50% Taiwan 50 + 50% gold"}
RULES = ("none", "annual", "quarterly", "band10", "band5")
CRASHES = {"dotcom_2000_2002": ("2000-01-01", "2002-12-31"), "gfc_2007_2009": ("2007-07-01", "2009-06-30"),
           "covid_2020": ("2020-01-01", "2020-06-30"), "rates_2022": ("2022-01-01", "2022-12-31")}
HALVES = {"1997_2013": ("1997-07-01", "2013-12-31"), "2014_2026": ("2014-01-01", "2026-12-31")}


def read(path: Path, col: str = "close") -> pd.Series:
    frame = pd.read_csv(path)
    key = next(k for k in ("date", "time", "observation_date") if k in frame.columns)
    idx = pd.to_datetime(frame[key], utc=True).dt.tz_localize(None).dt.normalize()
    values = pd.to_numeric(frame[col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=idx).dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def align(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Non-Taiwan closes dated D print after Taiwan's close on D (R3 review 2026-10-18): each is moved one day
    forward, so a Taiwan day sees only closes from before it."""
    series = series.set_axis(series.index + pd.Timedelta(days=1))
    return series.reindex(series.index.union(idx)).ffill(limit=10).reindex(idx)


def returns(drag: float = 0.02) -> pd.DataFrame:
    taiex = read(Y / "TWII_1d.csv")
    idx = taiex.index
    yfx = read(Y / "TWD_X_1d.csv")
    yfx = yfx[(yfx > 25) & (yfx < 40) & (yfx.index >= SPLICE)]
    ffx = read(F / "DEXTAUS.csv", "DEXTAUS")
    fx = align(pd.concat([ffx[ffx.index < SPLICE], yfx]), idx)
    fx = fx.fillna(align(ffx, idx))                   # Yahoo has no USD/TWD 2008-08-05 to 08-25; FRED fills it
    tb = read(F / "DTB3.csv", "DTB3")
    irx = read(Y / "IRX_1d.csv")
    rate = align(pd.concat([tb[tb.index < SPLICE], irx[irx.index >= SPLICE]]), idx)
    gd = read(GOLD_D)
    gw = read(GOLD_W)
    gw.index = gw.index + pd.Timedelta(days=4)     # a weekly bar is stamped Monday; its close is Friday's
    gold_usd = align(pd.concat([gw[gw.index < gd.index[0]], gd]), idx)
    out = pd.DataFrame(index=idx)
    out["etf"] = taiex.pct_change() + DIVIDEND / 252
    out["lev2"] = 2 * out["etf"] - drag / 252
    out["gold"] = (gold_usd * fx).pct_change()
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(0)
    out["tbill"] = (1 + rate.shift(1) / 100 * days / 360) * (fx / fx.shift(1)) - 1
    out["cash"] = 0.0
    out = out.iloc[1:]
    if out.isna().any().any():
        raise SystemExit(f"missing values after alignment: {out.isna().sum().to_dict()}")
    return out


def due(rule: str, day: pd.Timestamp, prev: pd.Timestamp | None, weight: float, target: float) -> bool:
    if rule == "none" or prev is None:
        return False
    if rule == "annual":
        return day.year != prev.year
    if rule == "quarterly":
        return day.quarter != prev.quarter or day.year != prev.year
    band = 0.10 if rule == "band10" else 0.05
    return abs(weight - target) > band


def simulate(r: pd.DataFrame, weights: dict, rule: str) -> tuple[pd.Series, int]:
    """Band rules test the close-to-close weight and rebalance at that close; calendar rules at the first close."""
    lead = next(k for k in ("lev2", "etf") if k in weights)
    holdings = dict(weights)
    values, prev, count = [], None, 0
    for day, row in r.iterrows():
        if rule in ("annual", "quarterly") and due(rule, day, prev, 0, 0):
            holdings, count = rebalance(holdings, weights), count + 1
        holdings = {k: v * (1 + row[k]) for k, v in holdings.items()}
        total = sum(holdings.values())
        if rule.startswith("band") and len(weights) > 1 and due(rule, day, prev, holdings[lead] / total, weights[lead]):
            holdings, count = rebalance(holdings, weights), count + 1
        values.append(sum(holdings.values()))
        prev = day
    return pd.Series(values, index=r.index), count


def rebalance(holdings: dict, weights: dict) -> dict:
    total = sum(holdings.values())
    traded = sum(abs(holdings[k] - total * w) for k, w in weights.items()) / 2
    total *= 1 - COST * traded / total
    return {k: total * w for k, w in weights.items()}


def stats(eq: pd.Series) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq[:trough].idxmax()
    recovered = eq[trough:][eq[trough:] >= eq[peak]]
    yearly = eq.resample("YE").last()
    yearly = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]), yearly]).pct_change().dropna()
    return {
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(float(dd.min() * 100), 2),
        "max_drawdown_peak": str(peak.date()), "max_drawdown_trough": str(trough.date()),
        "recovered_on": str(recovered.index[0].date()) if len(recovered) else None,
        "worst_year_pct": round(float(yearly.min() * 100), 2), "worst_year": int(yearly.idxmin().year),
        "cagr_over_max_dd": round(cagr / -dd.min(), 3),
        "final_multiple": round(float(eq.iloc[-1] / eq.iloc[0]), 2),
    }


def crash_dd(eq: pd.Series) -> dict:
    dd = eq / eq.cummax() - 1
    return {k: round(float(dd[a:b].min() * 100), 2) for k, (a, b) in CRASHES.items()}


def run(r: pd.DataFrame) -> dict:
    out = {"period": [str(r.index[0].date()), str(r.index[-1].date())], "by_rule": {}, "_curves": {}}
    for rule in RULES:
        block = {}
        for name, w in PORTFOLIOS.items():
            eq, count = simulate(r, w, rule)
            if rule == "annual":
                out["_curves"][name] = eq
            halves = {h: stats(eq[a:b]) for h, (a, b) in HALVES.items()}
            block[name] = stats(eq) | {"rebalances": count, "crash_drawdown_pct": crash_dd(eq), "halves": halves}
        out["by_rule"][rule] = block
    return out


def readings(res: dict) -> dict:
    annual = res["by_rule"]["annual"]
    p0 = annual["P0"]["max_drawdown_pct"]
    stress = {n: {"max_dd_pct": s["max_drawdown_pct"], "p0_max_dd_pct": p0,
                  "holds_up": bool(s["max_drawdown_pct"] > p0)} for n, s in annual.items() if n != "P0"}
    rules = {}
    for name in PORTFOLIOS:
        if name == "P0":
            continue
        rules[name] = {}
        for rule in RULES:
            if rule == "annual":
                continue
            better = all(
                res["by_rule"][rule][name]["halves"][h]["max_drawdown_pct"] > annual[name]["halves"][h]["max_drawdown_pct"]
                and res["by_rule"][rule][name]["halves"][h]["cagr_pct"] >= annual[name]["halves"][h]["cagr_pct"] - 0.5
                for h in HALVES)
            rules[name][rule] = "better_than_annual" if better else "no_consistent_difference"
    return {"stress": stress, "rules_vs_annual": rules}


def calibration(r: pd.DataFrame) -> dict:
    etf = read(Y / "0050_TW_1d.csv")
    etf = etf[etf.index >= "2014-01-02"]
    lev = read(Y / "00631L_TW_1d.csv")
    lev = lev[lev.index >= "2016-01-04"]

    def cagr(s):
        return round(((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1) * 100, 2)

    proxy = (1 + r["etf"][r.index > etf.index[0]]).cumprod()
    proxy2 = (1 + r["lev2"][r.index > lev.index[0]]).cumprod()
    return {"0050_2014_cagr_pct": cagr(etf), "proxy_2014_cagr_pct": cagr(proxy),
            "00631L_2016_cagr_pct": cagr(lev), "proxy_2x_2016_cagr_pct": cagr(proxy2),
            "daily_corr_proxy_vs_0050": round(float(r["etf"].reindex(etf.index).corr(etf.pct_change())), 3)}


def chart(res: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, eq in res["_curves"].items():
        s = res["by_rule"]["annual"][name]
        lw = 2.4 if name == "P0" else 1.3
        a1.plot(eq.index, eq / eq.iloc[0], linewidth=lw,
                label=f"{name} {LABELS[name]}: {s['cagr_pct']}%/yr, max DD {s['max_drawdown_pct']}%")
        a2.plot(eq.index, (eq / eq.cummax() - 1) * 100, linewidth=lw)
    for a, b in CRASHES.values():
        a2.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#999", alpha=0.15)
    a1.set_yscale("log"); a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=0.3)
    a1.set_title(f"Taiwan 50 proxy vs half-capital 2x barbells (TWD, annual rebalance), {res['period'][0]} to {res['period'][1]}")
    a2.set_ylabel("drawdown %"); a2.grid(alpha=0.3)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=100); plt.close(fig)


def sources() -> list[dict]:
    out = []
    for role, path, system in (
            ("price_1d_taiex", Y / "TWII_1d.csv", "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18"),
            ("price_1d_0050_calibration", Y / "0050_TW_1d.csv", "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18"),
            ("price_1d_00631L_calibration", Y / "00631L_TW_1d.csv", "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18"),
            ("fx_usdtwd_2007_on", Y / "TWD_X_1d.csv", "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18"),
            ("tbill_13w_2007_on", Y / "IRX_1d.csv", "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18"),
            ("fx_usdtwd_before_2007", F / "DEXTAUS.csv", "FRED DEXTAUS, fetched 2026-10-18"),
            ("tbill_3m_before_2007", F / "DTB3.csv", "FRED DTB3, fetched 2026-10-18"),
            ("price_1d_xauusd", GOLD_D, "repository tracked CSV"),
            ("price_1w_xauusd_before_2008", GOLD_W, "repository tracked CSV")):
        data = path.read_bytes()
        out.append({"role": role, "source_system": system, "locator": (path.relative_to(ROOT) if path.is_absolute() else path).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "immutable": False,
                    "privacy": "private_raw_source" if "xauusd" in role else "public_market_data"})
    return out


def write_manifest(study_id: str, generated_at: str) -> None:
    (Path("reproduced") / "source_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "study_id": study_id, "generated_at": generated_at,
         "timezone_assumptions": "TAIEX trading days; other series forward-filled up to 10 days.", "sources": sources()},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    r = returns()
    base = run(r)
    sens = run(returns(drag=0.04))
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "six fixed portfolios x five rebalancing rules (decision log 2026-10-18)",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md", "currency": "TWD",
                   "etf_proxy": "TAIEX daily return + 3.0%/252", "lev2": "2 x etf - 2.0%/252 (sensitivity 4.0%)",
                   "gold": "XAUUSD weekly before 2008-03-26, daily after, x USD/TWD",
                   "fx": "FRED DEXTAUS before 2007, Yahoo TWD=X after (25-40 filter)",
                   "tbill": "FRED DTB3 before 2007, ^IRX after; yield/100 x days/360 x FX",
                   "rebalance_cost": COST, "labels": LABELS, "crash_windows": CRASHES, "halves": HALVES},
        "correlation_daily": {"etf_gold": round(float(r["etf"].corr(r["gold"])), 3),
                              "etf_tbill": round(float(r["etf"].corr(r["tbill"])), 3)},
        "calibration": calibration(r),
        "results": {k: v for k, v in base.items() if k != "_curves"},
        "readings": readings(base),
        "sensitivity_drag_4pct_annual": {n: {k: s[k] for k in ("cagr_pct", "max_drawdown_pct")}
                                         for n, s in sens["by_rule"]["annual"].items()},
        "charts": [{"id": "stress", "file": "stress.png", "section": "seasonality",
                    "title": "Barbells through the 2000 and 2008 crashes, annual rebalance"}],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    directory = Path("reproduced")
    chart(base, directory / "charts/stress.png")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_manifest(STUDY_ID, payload["generated_at"])
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
