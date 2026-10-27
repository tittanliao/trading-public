#!/usr/bin/env python3
"""S1 V3.9 and how "clean" the trend was at entry (RS-XAUUSD-20261027-002).

Design fixed in research/studies/RS-XAUUSD-20261027-002/decision_log.md before any outcome was computed.
Cleanliness is Kaufman's efficiency ratio on daily closes that finished before the entry.

    python3.12 scripts/research/build_s1_trend_cleanliness.py [--dry-run]
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
import fail_pattern_toolkit as tk  # noqa: E402
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261027-002"
TRADES = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-10-09.csv")
DAILY = Path("local-inputs/XAUUSD_1d.csv")
WEEKLY = Path("local-inputs/XAUUSD_1wk.csv")
TAIPEI = timezone(timedelta(hours=8))

# Frozen in the decision log.
ER_DAYS, ER_WEEKS = 20, 13
Z80 = 1.96 + 0.8416  # the repository's 2.8: two-sided 5%, 80% power


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


def efficiency(closes: np.ndarray) -> float:
    path = np.abs(np.diff(closes)).sum()
    return float(abs(closes[-1] - closes[0]) / path) if path else float("nan")


def bars(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["start"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    return frame[["start", "close"]].sort_values("start").reset_index(drop=True)


def attach_er(trades: pd.DataFrame) -> pd.DataFrame:
    daily, weekly = bars(DAILY), bars(WEEKLY)
    daily["end"] = daily["start"] + pd.Timedelta(days=1)
    # Weekly bars start Monday 00:00Z; the week is final after Friday's close (Saturday 00:00Z).
    weekly["end"] = weekly["start"] + pd.Timedelta(days=5)
    rows = []
    for t in trades.itertuples():
        entry_utc = pd.Timestamp(t.entry_time) - pd.Timedelta(hours=8)  # export is Asia/Taipei
        d = daily[daily["end"] <= entry_utc]["close"].to_numpy()[-(ER_DAYS + 1):]
        w = weekly[weekly["end"] <= entry_utc]["close"].to_numpy()[-(ER_WEEKS + 1):]
        rows.append({"er20": efficiency(d) if len(d) == ER_DAYS + 1 else np.nan,
                     "dir20": np.sign(d[-1] - d[0]) if len(d) == ER_DAYS + 1 else np.nan,
                     "er13w": efficiency(w) if len(w) == ER_WEEKS + 1 else np.nan})
    return pd.concat([trades.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def group(frame: pd.DataFrame) -> dict:
    s = tk.stats(frame)
    s["mean_r"] = r(frame["r"].mean(), 4) if len(frame) else None
    return s


def effect(a: pd.DataFrame, b: pd.DataFrame, allr: pd.Series) -> dict:
    if len(a) < 2 or len(b) < 2:
        return {"effect_r": None, "resolution_bound_r": None, "separates": None}
    eff = a["r"].mean() - b["r"].mean()
    bound = Z80 * allr.std(ddof=1) * math.sqrt(1 / len(a) + 1 / len(b))
    return {"effect_r": r(eff, 4), "resolution_bound_r": r(bound, 4), "separates": bool(abs(eff) > bound)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = pd.read_csv(TRADES, encoding="utf-8-sig")
    raw = raw[raw["Date and time"] != "Open"]
    import io
    trades = tk.load_trades(io.StringIO(raw.to_csv(index=False)))
    trades["r"] = trades["return_pct"] / 0.5  # S1 stop = 0.5% of entry
    t = attach_er(trades).dropna(subset=["er20"])
    cuts = [float(t["er20"].quantile(q)) for q in (1 / 3, 2 / 3)]
    t["tercile"] = np.where(t["er20"] < cuts[0], "low", np.where(t["er20"] < cuts[1], "mid", "high"))
    med = float(t["er20"].median())
    t["shape"] = np.where(t["er20"] < med, "choppy", np.where(t["dir20"] > 0, "up_clean", "down_clean"))

    terciles = {k: group(t[t["tercile"] == k]) for k in ("low", "mid", "high")}
    terciles["high_vs_low"] = effect(t[t["tercile"] == "high"], t[t["tercile"] == "low"], t["r"])
    shapes = {k: group(t[t["shape"] == k]) for k in ("up_clean", "down_clean", "choppy")}
    shapes["up_clean_vs_choppy"] = effect(t[t["shape"] == "up_clean"], t[t["shape"] == "choppy"], t["r"])
    shapes["down_clean_vs_choppy"] = effect(t[t["shape"] == "down_clean"], t[t["shape"] == "choppy"], t["r"])

    t["month"] = t["entry_time"].dt.strftime("%Y-%m")
    months = t.groupby("month").agg(n=("r", "size"), net_r=("r", "sum"), net_usd=("net_pnl_usd", "sum"),
                                    mean_er20=("er20", "mean"), mean_er13w=("er13w", "mean"))
    rho = months["net_r"].rank().corr(months["mean_er20"].rank())
    rho_w = months["net_r"].rank().corr(months["mean_er13w"].rank())

    keep = t["tercile"] == "high"
    half = t["entry_time"] < t["entry_time"].iloc[len(t) // 2]
    def split(mask):
        return {"kept": group(t[mask & keep]), "removed": group(t[mask & ~keep]),
                **effect(t[mask & keep], t[mask & ~keep], t["r"])}
    filt = {"rule": f"take S1 only when ER20 >= {cuts[1]:.4f} (top tercile)", "all": split(pd.Series(True, index=t.index)),
            "first_half": split(half), "second_half": split(~half),
            "half_boundary": str(t["entry_time"].iloc[len(t) // 2])}

    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "S1 V3.9 (TradingView export 2026-10-09)",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md (fixed 2026-10-27 before any outcome)",
                   "er": "Kaufman efficiency ratio |c_t - c_t-n| / sum|c_i - c_i-1|",
                   "er20": f"{ER_DAYS} daily closes (TradingView XAUUSD 1D, UTC bars) that finished before entry",
                   "er13w": f"{ER_WEEKS} weekly closes that finished before entry",
                   "r": "Return % / 0.5 (S1 stop distance)", "tercile_cuts_er20": [r(c, 4) for c in cuts],
                   "median_er20": r(med, 4),
                   "resolution_bound": "2.8 × SD(R, all trades) × √(1/n1 + 1/n2)"},
        "coverage": {"trades": int(len(t)), "dropped_no_history": int(len(trades) - len(t)),
                     "first_entry": str(t["entry_time"].min()), "last_entry": str(t["entry_time"].max())},
        "all": group(t),
        "terciles": terciles,
        "shapes": shapes,
        "months": {"rank_corr_net_r_vs_er20": r(rho, 3), "rank_corr_net_r_vs_er13w": r(rho_w, 3),
                   "n_months": int(len(months)),
                   "rows": {m: {"n": int(v.n), "net_r": r(v.net_r), "net_usd": r(v.net_usd),
                                "mean_er20": r(v.mean_er20, 3), "mean_er13w": r(v.mean_er13w, 3)}
                            for m, v in months.iterrows()}},
        "filter_top_tercile": filt,
        "charts": [],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
        return 0
    Path("reproduced").mkdir(exist_ok=True); (Path("reproduced") / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    directory = Path("reproduced")
    (directory / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
        "timezone_assumptions": "Trade times are Asia/Taipei as exported; daily/weekly bars are UTC-stamped; a bar counts only once it has closed before the entry.",
        "sources": [
            {"role": "s1_trades", "source_system": "TradingView strategy export", "locator": rel(TRADES),
             **sha(TRADES), "immutable": True, "privacy": "private_raw_source"},
            {"role": "price_1d", "source_system": "repository tracked CSV", "locator": rel(DAILY),
             **sha(DAILY), "immutable": False, "privacy": "private_raw_source"},
            {"role": "price_1w", "source_system": "repository tracked CSV", "locator": rel(WEEKLY),
             **sha(WEEKLY), "immutable": False, "privacy": "private_raw_source"},
        ]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(directory / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
