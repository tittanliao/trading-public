#!/usr/bin/env python3
"""RS-TX-20261018-003 — half capital in a 2x Taiwan 50, half in T-bills or gold, against 100% 0050.

Design fixed in research/studies/RS-TX-20261018-003/decision_log.md before any outcome. All in TWD.

    python3.12 scripts/research/build_tw_lev2_barbell.py
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

STUDY_ID = "RS-TX-20261018-003"
TAIPEI = timezone(timedelta(hours=8))
Y = Path("local-inputs")
GOLD = Path("local-inputs/XAUUSD_1d.csv")
DRAG, COST = 0.02, 0.0025
PORTFOLIOS = {
    "P0": {"etf": 1.0},
    "P1": {"lev2": 0.5, "cash": 0.5},
    "P2": {"lev2": 0.5, "tbill": 0.5},
    "P3": {"lev2": 0.5, "gold": 0.5},
    "P4": {"lev2": 0.5, "gold": 0.25, "tbill": 0.25},
    "P5": {"etf": 0.5, "gold": 0.5},
}
LABELS = {"P0": "100% 0050", "P1": "50% 2x + 50% cash", "P2": "50% 2x + 50% T-bills", "P3": "50% 2x + 50% gold",
          "P4": "50% 2x + 25% gold + 25% T-bills", "P5": "50% 0050 + 50% gold"}


def read(path: Path, col: str = "close") -> pd.Series:
    frame = pd.read_csv(path)
    key = "date" if "date" in frame.columns else "time"
    idx = pd.to_datetime(frame[key], utc=True).dt.tz_localize(None).dt.normalize()
    return pd.Series(frame[col].astype(float).to_numpy(), index=idx).sort_index()


def returns(start: str, real_lev: bool) -> pd.DataFrame:
    etf = read(Y / "0050_TW_1d.csv")
    etf = etf[etf.index >= start]
    idx = etf.index
    # R3 review 2026-10-18: New York closes dated D print after Taiwan's close on D, so each non-Taiwan series
    # is moved one day forward and a Taiwan day sees only closes from before it.
    lag = lambda s: s.set_axis(s.index + pd.Timedelta(days=1))  # noqa: E731
    fx = lag(read(Y / "TWD_X_1d.csv"))
    fx = fx[(fx > 25) & (fx < 40)].reindex(idx, method="ffill", limit=5).ffill(limit=5)
    gold_usd = lag(read(GOLD)).reindex(idx, method="ffill", limit=5)
    irx = lag(read(Y / "IRX_1d.csv")).reindex(idx, method="ffill", limit=5)
    out = pd.DataFrame(index=idx)
    out["etf"] = etf.pct_change()
    if real_lev:
        lev = read(Y / "00631L_TW_1d.csv").reindex(idx).ffill(limit=1)   # 2016-07-08 missing (typhoon day)
        out["lev2"] = lev.pct_change()
    else:
        out["lev2"] = 2 * out["etf"] - DRAG / 252
    out["gold"] = (gold_usd * fx).pct_change()
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(0)
    out["tbill"] = (1 + irx.shift(1) / 100 * days / 360) * (fx / fx.shift(1)) - 1
    out["cash"] = 0.0
    out = out.iloc[1:]
    if out.isna().any().any():
        raise SystemExit(f"missing values after alignment: {out.isna().sum().to_dict()}")
    return out


def simulate(r: pd.DataFrame, weights: dict, rebalance: bool) -> pd.Series:
    holdings = {k: w for k, w in weights.items()}
    values, last_year = [], None
    for day, row in r.iterrows():
        if rebalance and last_year is not None and day.year != last_year:
            total = sum(holdings.values())
            traded = sum(abs(holdings[k] - total * w) for k, w in weights.items()) / 2
            total *= 1 - COST * traded / total
            holdings = {k: total * w for k, w in weights.items()}
        last_year = day.year
        holdings = {k: v * (1 + row[k]) for k, v in holdings.items()}
        values.append(sum(holdings.values()))
    return pd.Series(values, index=r.index)


def stats(eq: pd.Series) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq[:trough].idxmax()
    under = (dd < 0).astype(int)
    runs = under.groupby((under != under.shift()).cumsum()).cumsum()
    yearly = eq.resample("YE").last().pct_change().dropna()
    first_year = eq[eq.index.year == eq.index[0].year]
    yearly = pd.concat([pd.Series([first_year.iloc[-1] / eq.iloc[0] - 1], index=[first_year.index[-1]]), yearly])
    return {
        "cagr_pct": round(cagr * 100, 2),
        "volatility_pct": round(float(eq.pct_change().std() * np.sqrt(252) * 100), 2),
        "max_drawdown_pct": round(float(dd.min() * 100), 2),
        "max_drawdown_peak": str(peak.date()), "max_drawdown_trough": str(trough.date()),
        "longest_under_water_days": int(runs.max()),
        "worst_year_pct": round(float(yearly.min() * 100), 2), "worst_year": int(yearly.idxmin().year),
        "cagr_over_max_dd": round(cagr / -dd.min(), 3),
        "final_multiple": round(float(eq.iloc[-1] / eq.iloc[0]), 3),
    }


def run(start: str, real_lev: bool) -> dict:
    r = returns(start, real_lev)
    out = {"period": [str(r.index[0].date()), str(r.index[-1].date())], "two_x": "real 00631L" if real_lev else "synthetic",
           "correlation_daily": {"etf_gold": round(float(r["etf"].corr(r["gold"])), 3),
                                 "etf_tbill": round(float(r["etf"].corr(r["tbill"])), 3)},
           "annual_rebalance": {}, "no_rebalance": {}, "_curves": {}}
    for name, w in PORTFOLIOS.items():
        eq = simulate(r, w, True)
        out["_curves"][name] = eq
        out["annual_rebalance"][name] = stats(eq)
        out["no_rebalance"][name] = stats(simulate(r, w, False))
    p0 = out["annual_rebalance"]["P0"]["max_drawdown_pct"]
    out["lower_drawdown_than_P0"] = [n for n, s in out["annual_rebalance"].items()
                                     if n != "P0" and s["max_drawdown_pct"] > p0]
    return out


def chart(res: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, eq in res["_curves"].items():
        s = res["annual_rebalance"][name]
        lw = 2.4 if name == "P0" else 1.4
        a1.plot(eq.index, eq / eq.iloc[0], linewidth=lw,
                label=f"{name} {LABELS[name]}: {s['cagr_pct']}%/yr, max DD {s['max_drawdown_pct']}%")
        a2.plot(eq.index, (eq / eq.cummax() - 1) * 100, linewidth=lw)
    a1.set_yscale("log"); a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=0.3)
    a1.set_title(f"0050 vs half-capital 2x barbells (TWD, annual rebalance), {res['period'][0]} to {res['period'][1]}")
    a2.set_ylabel("drawdown %"); a2.grid(alpha=0.3)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=100); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    synthetic = run("2014-01-02", False)
    real = run("2016-01-04", True)
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "six fixed portfolios (decision log 2026-10-18)",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md", "currency": "TWD",
                   "synthetic_2x": "2 x 0050 daily return - 2.0%/252", "tbill": "^IRX/100 x days/360, times USD/TWD",
                   "gold": "XAUUSD daily x USD/TWD", "rebalance_cost": COST, "labels": LABELS,
                   "fx_filter": "USD/TWD outside 25-40 dropped and forward-filled"},
        "synthetic_2014": {k: v for k, v in synthetic.items() if k != "_curves"},
        "real_00631L_2016": {k: v for k, v in real.items() if k != "_curves"},
        "lower_drawdown_on_both": sorted(set(synthetic["lower_drawdown_than_P0"]) & set(real["lower_drawdown_than_P0"])),
        "charts": [{"id": "barbell", "file": "barbell.png", "section": "seasonality", "title": "0050 against half-capital 2x barbells"}],
    }
    if args.dry_run:
        print(json.dumps({k: v for k, v in payload.items()}, ensure_ascii=False, indent=1))
        return 0
    directory = Path("reproduced")
    chart(synthetic, directory / "charts/barbell.png")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    srcs = []
    for role, path in (("price_1d_0050", Y / "0050_TW_1d.csv"), ("price_1d_00631L", Y / "00631L_TW_1d.csv"),
                       ("fx_usdtwd", Y / "TWD_X_1d.csv"), ("tbill_13w_yield", Y / "IRX_1d.csv"), ("price_1d_xauusd", GOLD)):
        data = path.read_bytes()
        srcs.append({"role": role, "source_system": "repository tracked CSV" if path == GOLD else "Yahoo Finance via yfinance 1.4.1, fetched 2026-10-18",
                     "locator": (path.relative_to(ROOT) if path.is_absolute() else path).as_posix(), "sha256": hashlib.sha256(data).hexdigest(),
                     "size_bytes": len(data), "immutable": False, "privacy": "public_market_data" if path != GOLD else "private_raw_source"})
    (directory / "source_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
         "timezone_assumptions": "Taiwan trading days; other series forward-filled up to 5 days.", "sources": srcs},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
