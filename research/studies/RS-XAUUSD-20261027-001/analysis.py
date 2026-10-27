#!/usr/bin/env python3
"""Gold against real rates, nominal rates and CPI on monthly data from 1971 (RS-XAUUSD-20261027-001).

Design fixed in research/studies/RS-XAUUSD-20261027-001/decision_log.md before any outcome was computed:
regime buckets with 12/24-month forward returns, gold priced against CPI at historic extremes, the rolling
co-movement with the 10-year real rate, and the five historical months closest to the latest one.

    python3.12 scripts/research/build_gold_rates_cpi_regimes.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261027-001"
WEEKLY = Path("local-inputs/XAUUSD_1wk.csv")
FRED_DIR = Path("local-inputs")
FRED_FILES = {"CPIAUCSL": "CPIAUCSL.csv", "FEDFUNDS": "FEDFUNDS.csv", "GS10": "GS10.csv",
              "DFII10": "DFII10_20261027.csv"}
TAIPEI = timezone(timedelta(hours=8))

# Frozen in the decision log.
HORIZONS = (12, 24)
PCT_FROM = "1975-01"
CORR_WINDOW = 36
ANALOG_FEATURES = ("gold_real_pct", "dd_ath", "real_short", "d12_real_short", "cpi_yoy", "fed_6m", "gold_12m")
ANALOG_N, ANALOG_GAP, ANALOG_EXCLUDE_LAST = 5, 24, 36
ANALOG_FORWARD = (6, 12, 24)
EXPENSIVE_PCT, EXPENSIVE_GAP = 90.0, 24


def r(value, digits=2):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), digits)


def rel(path: Path) -> str:
    """Repo-relative locator; the published copy runs from local-inputs/ outside the repo."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def month(ts) -> str:
    return pd.Period(ts, "M").strftime("%Y-%m")


def load_gold() -> pd.Series:
    frame = pd.read_csv(WEEKLY)
    stamp = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    # Weekly bars are stamped on Monday; the close belongs to that week's Friday.
    frame["m"] = (stamp + pd.Timedelta(days=4)).dt.to_period("M")
    return frame.groupby("m")["close"].last().astype(float)


def load_fred(series: str) -> pd.Series:
    frame = pd.read_csv(FRED_DIR / FRED_FILES[series])
    values = pd.to_numeric(frame[series], errors="coerce")
    stamp = pd.to_datetime(frame["observation_date"]).dt.to_period("M")
    return pd.Series(values.to_numpy(), index=stamp).dropna().groupby(level=0).mean()


def build_frame() -> pd.DataFrame:
    gold = load_gold()
    cpi, ff, gs10, tips = (load_fred(s) for s in ("CPIAUCSL", "FEDFUNDS", "GS10", "DFII10"))
    f = pd.DataFrame({"gold": gold})
    # CPI for month t is published in month t+1: at month-end t only CPI(t-1) is known.
    f["cpi"] = cpi.shift(1).reindex(f.index)
    f["fedfunds"] = ff.reindex(f.index)
    f["gs10"] = gs10.reindex(f.index)
    f["tips10"] = tips.reindex(f.index)
    f = f.dropna(subset=["gold", "cpi", "fedfunds", "gs10"])
    f["cpi_yoy"] = (f["cpi"] / f["cpi"].shift(12) - 1) * 100
    f["real_short"] = f["fedfunds"] - f["cpi_yoy"]
    f["real_10y_expost"] = f["gs10"] - f["cpi_yoy"]
    f["real_10y"] = f["tips10"].where(f["tips10"].notna(), f["real_10y_expost"])
    f["d12_real_short"] = f["real_short"] - f["real_short"].shift(12)
    f["fed_6m"] = f["fedfunds"] - f["fedfunds"].shift(6)
    f["gold_real"] = f["gold"] / f["cpi"]
    f["gold_real_pct"] = [
        100.0 * (f["gold_real"].iloc[: i + 1] <= f["gold_real"].iloc[i]).mean()
        if f.index[i] >= pd.Period(PCT_FROM, "M") else np.nan
        for i in range(len(f))
    ]
    f["dd_ath"] = (f["gold"] / f["gold"].cummax() - 1) * 100
    f["gold_12m"] = (f["gold"] / f["gold"].shift(12) - 1) * 100
    for h in HORIZONS:
        f[f"fwd{h}"] = (f["gold"].shift(-h) / f["gold"] - 1) * 100
        f[f"fwd{h}_real"] = ((f["gold"].shift(-h) / f["cpi"].shift(-h)) / f["gold_real"] - 1) * 100
    return f.dropna(subset=["cpi_yoy"])


def runs(mask: pd.Series) -> int:
    m = mask.fillna(False).astype(int).to_numpy()
    return int(((m[1:] == 1) & (m[:-1] == 0)).sum() + (m[0] == 1)) if len(m) else 0


def bucket_stats(f: pd.DataFrame, mask: pd.Series) -> dict:
    out = {"months": int(mask.sum()), "runs": runs(mask)}
    for h in HORIZONS:
        for kind in ("", "_real"):
            col = f"fwd{h}{kind}"
            allv = f[col].dropna()
            v = f.loc[mask, col].dropna()
            key = f"{h}m{kind}"
            if len(v) == 0:
                out[key] = {"n": 0}
                continue
            n_eff, base_eff = len(v) / h, len(allv) / h
            bound = 2.8 * allv.std(ddof=1) * math.sqrt(1 / max(n_eff, 1e-9) + 1 / base_eff)
            effect = v.mean() - allv.mean()
            out[key] = {"n": int(len(v)), "n_eff": r(n_eff, 1), "median_pct": r(v.median()),
                        "mean_pct": r(v.mean()), "win_rate_pct": r((v > 0).mean() * 100),
                        "baseline_mean_pct": r(allv.mean()), "effect_pct": r(effect),
                        "resolution_bound_pct": r(bound), "separates": bool(abs(effect) > bound)}
    return out


def regimes(f: pd.DataFrame) -> dict:
    defs = {
        "real_short": {"below_0": f["real_short"] < 0, "0_to_2": f["real_short"].between(0, 2),
                       "above_2": f["real_short"] > 2},
        "d12_real_short": {"falling_gt1": f["d12_real_short"] < -1,
                           "flat": f["d12_real_short"].between(-1, 1),
                           "rising_gt1": f["d12_real_short"] > 1},
        "fed_6m": {"cutting": f["fed_6m"] < -0.5, "flat": f["fed_6m"].between(-0.5, 0.5),
                   "hiking": f["fed_6m"] > 0.5},
        "cpi_yoy": {"le_4": f["cpi_yoy"] <= 4, "gt_4": f["cpi_yoy"] > 4},
    }
    return {name: {b: bucket_stats(f, m) for b, m in groups.items()} for name, groups in defs.items()}


def expensive(f: pd.DataFrame) -> dict:
    hit = f[f["gold_real_pct"] >= EXPENSIVE_PCT]
    episodes, last = [], None
    for idx, row in hit.iterrows():
        if last is not None and (idx - last).n < EXPENSIVE_GAP:
            last = idx
            continue
        episodes.append({"first_month": month(idx.to_timestamp()), "gold": r(row["gold"]),
                         "gold_real_pct": r(row["gold_real_pct"], 1), "real_short": r(row["real_short"]),
                         "cpi_yoy": r(row["cpi_yoy"]),
                         "fwd24_real_pct": r(row["fwd24_real"]), "fwd24_pct": r(row["fwd24"])})
        last = idx
    mask = f["gold_real_pct"] >= EXPENSIVE_PCT
    return {"threshold_pct": EXPENSIVE_PCT, "episodes": episodes, "stats": bucket_stats(f, mask)}


def comovement(f: pd.DataFrame) -> dict:
    x = f["gold_12m"]
    y = f["real_10y"] - f["real_10y"].shift(12)
    corr = x.rolling(CORR_WINDOW).corr(y)
    full = pd.concat([x, y], axis=1).dropna()
    by_decade = {}
    for start in range(1980, 2030, 10):
        part = full[(full.index >= pd.Period(f"{start}-01", "M")) & (full.index < pd.Period(f"{start + 10}-01", "M"))]
        if len(part) > 24:
            by_decade[f"{start}s"] = {"months": len(part), "corr": r(part.iloc[:, 0].corr(part.iloc[:, 1]), 3)}
    recent = corr.dropna()
    return {"window_months": CORR_WINDOW, "full_sample_corr": r(full.iloc[:, 0].corr(full.iloc[:, 1]), 3),
            "months": int(len(full)), "by_decade": by_decade,
            "latest": {"month": month(recent.index[-1].to_timestamp()), "corr": r(recent.iloc[-1], 3)},
            "series": {month(k.to_timestamp()): r(v, 3) for k, v in recent.items()}}


def level_gap(f: pd.DataFrame) -> dict:
    """Exploratory, added after the designed parts were seen (decision log): the 2006-2021 level fit of
    log(gold/CPI) on the TIPS real yield, and how far gold now sits from it."""
    d = f[f.index >= pd.Period("2006-01", "M")].dropna(subset=["tips10"])
    fit = d[d.index < pd.Period("2022-01", "M")]
    slope, intercept = np.polyfit(fit["real_10y"], np.log(fit["gold_real"]), 1)
    implied = np.exp(intercept + slope * d["real_10y"]) * d["cpi"]
    gap = (d["gold"] / implied - 1) * 100
    in_fit = gap[gap.index < pd.Period("2022-01", "M")]
    return {"label": "exploratory (not in the fixed design)", "fit_months": f"2006-01..2021-12 (n={len(fit)})",
            "slope_log_per_pp": r(slope, 4), "fit_r2": r(np.corrcoef(fit["real_10y"], np.log(fit["gold_real"]))[0, 1] ** 2, 3),
            "fit_period_max_abs_gap_pct": r(in_fit.abs().max(), 1),
            "by_month": {m: {"gold": r(d.loc[m, "gold"]), "real_10y": r(d.loc[m, "real_10y"]),
                             "implied_gold": r(implied.loc[m]), "gap_pct": r(gap.loc[m], 1)}
                         for m in ("2021-12", "2022-10", "2023-10", "2024-06", "2025-06", "2026-01",
                                   month(d.index[-1].to_timestamp()))}}


def analogs(f: pd.DataFrame) -> dict:
    cols = list(ANALOG_FEATURES)
    data = f.dropna(subset=cols)
    z = (data[cols] - data[cols].mean()) / data[cols].std(ddof=1)
    now = z.iloc[-1]
    cutoff = data.index[-1] - ANALOG_EXCLUDE_LAST
    dist = np.sqrt(((z[data.index <= cutoff] - now) ** 2).sum(axis=1)).sort_values()
    chosen = []
    for idx, d in dist.items():
        if all(abs((idx - c).n) >= ANALOG_GAP for c, _ in chosen):
            chosen.append((idx, d))
        if len(chosen) == ANALOG_N:
            break
    gold = f["gold"]
    rows = []
    for idx, d in chosen:
        row = data.loc[idx]
        pos = gold.index.get_loc(idx)
        fwd = {f"{h}m": (r((gold.iloc[pos + h] / gold.iloc[pos] - 1) * 100) if pos + h < len(gold) else None)
               for h in ANALOG_FORWARD}
        rows.append({"month": month(idx.to_timestamp()), "distance": r(d, 3),
                     **{c: r(row[c]) for c in cols}, "gold": r(row["gold"]), "forward_pct": fwd,
                     **{f"fwd_{k}_pct": v for k, v in fwd.items()}})
    last = data.iloc[-1]
    return {"features": cols, "now": {"month": month(data.index[-1].to_timestamp()),
                                      **{c: r(last[c]) for c in cols}, "gold": r(last["gold"]),
                                      "fedfunds": r(last["fedfunds"]), "gs10": r(last["gs10"]),
                                      "tips10": r(last["tips10"])},
            "analogs": rows}


def chart(f: pd.DataFrame, co: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    t = f.index.to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.4, 1]})
    a = axes[0]
    a.semilogy(t, f["gold_real"] / f["gold_real"].iloc[0] * 100, color="#b8860b", lw=1.6)
    a.set_ylabel("gold / CPI (1971 = 100, log)")
    a.set_title("Gold priced in CPI, the real short rate, and gold's co-movement with the 10-year real rate")
    b = axes[1]
    b.plot(t, f["real_short"], color="#1f77b4", lw=1.2, label="real short rate (Fed funds − CPI y/y)")
    b.plot(t, f["cpi_yoy"], color="#d62728", lw=1.0, alpha=0.8, label="CPI y/y")
    b.axhline(0, color="#888", lw=0.8)
    b.set_ylabel("%")
    b.legend(loc="upper right", fontsize=8)
    c = axes[2]
    s = pd.Series(co["series"])
    c.plot(pd.PeriodIndex(s.index, freq="M").to_timestamp(), s.values, color="#555", lw=1.2)
    c.axhline(0, color="#888", lw=0.8)
    c.set_ylabel(f"{CORR_WINDOW}m corr")
    c.set_ylim(-1, 1)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    f = build_frame()
    co = comovement(f)
    payload = {
        "study_id": STUDY_ID,
        "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "none — descriptive study of gold and US rates",
        "method": {
            "design": f"research/studies/{STUDY_ID}/decision_log.md (fixed 2026-10-27 before any outcome)",
            "gold": "last weekly close of each month (week assigned to the month of its Friday), nominal USD",
            "cpi_lag": "CPI of month t-1 is used at month-end t (publication lag)",
            "real_short": "FEDFUNDS monthly average − CPI y/y", "real_10y": "DFII10 monthly mean from 2003; GS10 − CPI y/y before",
            "gold_real_pct": f"expanding percentile of gold/CPI using only past months, from {PCT_FROM}",
            "horizons_months": HORIZONS,
            "resolution_bound": "2.8 × SD(all months) × √(1/n_eff + 1/N_eff), n_eff = months / horizon",
            "analogs": {"features": ANALOG_FEATURES, "z_score": "full sample", "n": ANALOG_N,
                        "min_gap_months": ANALOG_GAP, "exclude_last_months": ANALOG_EXCLUDE_LAST},
        },
        "coverage": {"first_month": month(f.index[0].to_timestamp()), "last_month": month(f.index[-1].to_timestamp()),
                     "months": int(len(f))},
        "part_a_regimes": regimes(f),
        "part_b_expensive": expensive(f),
        "part_c_comovement": {k: v for k, v in co.items() if k != "series"},
        "part_c_level_gap": level_gap(f),
        "part_d_analogs": analogs(f),
        "charts": [{"id": "gold_rates_cpi", "file": "gold_rates_cpi.png", "section": "market",
                    "title": "Gold/CPI, real short rate and CPI, and the 36-month gold/real-rate correlation"}],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
        return 0
    directory = Path("reproduced")
    chart(f, co, directory / "charts/gold_rates_cpi.png")
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    sources = [{"role": "price_1w", "source_system": "repository tracked CSV",
                "locator": rel(WEEKLY), **sha(WEEKLY), "immutable": False,
                "privacy": "private_raw_source", "note": "Month-end = last weekly close in the month."}]
    for sid, name in FRED_FILES.items():
        path = FRED_DIR / name
        sources.append({"role": f"fred_{sid.lower()}", "source_system": "FRED, St. Louis Fed (public)",
                        "locator": rel(path), **sha(path), "immutable": True,
                        "observed": "2026-10-27", "privacy": "public_source"})
    (directory / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
        "timezone_assumptions": "Monthly periods; gold weekly bars are Monday-stamped and assigned to the month of their Friday.",
        "sources": sources}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
