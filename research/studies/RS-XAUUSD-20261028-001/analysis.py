#!/usr/bin/env python3
"""S1 V3.9 T1/T2/T3 confirmation entry on complete trade paths (RS-XAUUSD-20261028-001).

RS-XAUUSD-20260815-004 tested waiting 1, 2 or 3 bars for the signal low to hold before entering, but
its TradingView exports of the delayed modes were incomplete ("candidate-coverage mismatch"). This
study replays the same research Pine with PineTS on the same FX_IDC 30m bars, so every mode has its
complete path, and recomputes the comparison. The design (modes, window, metrics) is inherited from
that study; nothing is chosen here (decision_log.md).

    python3.12 scripts/research/build_s1_t123_complete_paths.py --replay   # Private: regenerate trades (PineTS)
    python3.12 scripts/research/build_s1_t123_complete_paths.py [--dry-run] # statistics from the trade files

A reader without PineTS can produce the four trade lists with TradingView (Strategy Tester → List of
trades) or any Pine runtime, as CSV with entry_time, exit_time, entry_price, exit_price, exit_id
(times Asia/Taipei, bar-open).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-XAUUSD-20261028-001"
STUDY_DIR = Path("reproduced")
TRADES_OFF = Path("local-inputs/s1-t123-off.csv")
TRADES_T1 = Path("local-inputs/s1-t123-t1.csv")
TRADES_T2 = Path("local-inputs/s1-t123-t2.csv")
TRADES_T3 = Path("local-inputs/s1-t123-t3.csv")
TV_OFF = Path("local-inputs/tv-export-off.csv")
TV_T1 = Path("local-inputs/tv-export-t1.csv")
TV_T2 = Path("local-inputs/tv-export-t2.csv")
TV_T3 = Path("local-inputs/tv-export-t3.csv")
TAIPEI = timezone(timedelta(hours=8))

# Inherited from RS-XAUUSD-20260815-004 (its window) and the repository's conventions.
WINDOW = ("2024-01-01 00:00", "2026-08-13 23:59")
AFTER = ("2026-08-14 00:00", "2026-10-08 23:59")
STOP_PCT = 0.5
Z80 = 2.8
MODES = {"OFF": (TRADES_OFF, TV_OFF), "T1": (TRADES_T1, TV_T1), "T2": (TRADES_T2, TV_T2), "T3": (TRADES_T3, TV_T3)}


def r(value, digits=3):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), digits)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def wilson(k: int, n: int) -> list:
    if not n:
        return [None, None]
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [r(100 * (centre - half), 2), r(100 * (centre + half), 2)]


def load(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["return_pct"] = (t["exit_price"] / t["entry_price"] - 1) * 100
    t["r"] = t["return_pct"] / STOP_PCT
    return t.sort_values("entry_time").reset_index(drop=True)


def stats(t: pd.DataFrame) -> dict:
    n = len(t)
    if not n:
        return {"n": 0}
    wins = int((t["r"] > 0).sum())
    gain, loss = t.loc[t["r"] > 0, "return_pct"].sum(), -t.loc[t["r"] < 0, "return_pct"].sum()
    equity = t["return_pct"].cumsum()
    return {"n": n, "win_rate_pct": r(100 * wins / n, 2), "win_rate_ci95_pct": wilson(wins, n),
            "profit_factor": r(gain / loss) if loss else None, "net_return_pct": r(t["return_pct"].sum(), 2),
            "mean_r": r(t["r"].mean()), "max_drawdown_pct": r((equity.cummax() - equity).max(), 2)}


def tv_export(path: Path) -> pd.DataFrame:
    tv = pd.read_csv(path, encoding="utf-8-sig")
    tv = tv[tv["Signal"] != "Open"]
    e = tv[tv["Type"] == "Entry long"].set_index("Trade number")
    x = tv[tv["Type"] == "Exit long"].set_index("Trade number")
    return pd.DataFrame({"entry_time": pd.to_datetime(e["Date and time"]), "entry_price": e["Price USD"],
                         "exit_time": pd.to_datetime(x["Date and time"]), "exit_price": x["Price USD"]}).dropna()


def coverage(replay: pd.DataFrame, tv: pd.DataFrame) -> dict:
    inside = replay[replay["entry_time"].between(tv["entry_time"].min(), tv["entry_time"].max())]
    m = tv.merge(inside, on="entry_time", how="outer", suffixes=("_tv", "_pt"), indicator=True)
    both = m[m["_merge"] == "both"]
    exact = both[(pd.to_datetime(both["exit_time_tv"]) == pd.to_datetime(both["exit_time_pt"]))
                 & ((both["entry_price_tv"] - both["entry_price_pt"]).abs() < 0.0015)
                 & ((both["exit_price_tv"] - both["exit_price_pt"]).abs() < 0.0015)]
    return {"tv_export_trades": int(len(tv)), "replay_trades_in_export_range": int(len(inside)),
            "matched_entries": int(len(both)), "exact_trades": int(len(exact)),
            "tv_only": int((m["_merge"] == "left_only").sum()), "replay_only": int((m["_merge"] == "right_only").sum())}


def replay() -> None:
    """Private only: regenerate the four trade lists with PineTS (scripts/research/pinets)."""
    pine = "research/studies/RS-XAUUSD-20260815-004/pine/S1-Awe-V3.9-T123-Research.pine"
    tool = ROOT / "scripts/research/pinets/replay_s1.py"
    TRADES_OFF.parent.mkdir(parents=True, exist_ok=True)
    control = subprocess.run([sys.executable, str(tool)], cwd=ROOT, capture_output=True, text=True)
    if control.returncode != 0:
        raise SystemExit("PineTS control against the formal V3.9 export failed; nothing replayed")
    for mode, (path, _) in MODES.items():
        args = [sys.executable, str(tool), "--source", pine, "--export", str(TV_OFF), "--out", str(path) + ".json"]
        if mode != "OFF":
            args += ["--set", "use_t_hold_filter=true", "--set", f't_hold_mode="{mode}"']
        subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        rows = pd.read_json(str(path) + ".json")
        rows[["entry_time", "exit_time", "entry_price", "exit_price", "exit_id"]].to_csv(path, index=False)
        Path(str(path) + ".json").unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", action="store_true", help="regenerate the trade lists with PineTS first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay()

    trades = {mode: load(path) for mode, (path, _) in MODES.items()}
    sd_all = pd.concat(t[t["entry_time"].between(*WINDOW)]["r"] for t in trades.values()).std(ddof=1)
    modes = {}
    for mode, t in trades.items():
        w = t[t["entry_time"].between(*WINDOW)]
        half = w["entry_time"].iloc[len(w) // 2]
        off = trades["OFF"][trades["OFF"]["entry_time"].between(*WINDOW)]
        bound = Z80 * sd_all * math.sqrt(1 / len(w) + 1 / len(off))
        modes[mode] = {"window": stats(w), "first_half": stats(w[w["entry_time"] < half]),
                       "second_half": stats(w[w["entry_time"] >= half]),
                       "after_window": stats(t[t["entry_time"].between(*AFTER)]),
                       "mean_r_minus_off": r(w["r"].mean() - off["r"].mean()) if mode != "OFF" else None,
                       "resolution_bound_r": r(bound) if mode != "OFF" else None,
                       "coverage_vs_tv_export_2026_08_16": coverage(t, tv_export(MODES[mode][1]))}
    w = {k: v["window"] for k, v in modes.items()}
    payload = {
        "study_id": STUDY_ID, "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "strategy_version": "S1 V3.9 research Pine with T1/T2/T3 switch (RS-XAUUSD-20260815-004)",
        "method": {"design": f"research/studies/{STUDY_ID}/decision_log.md (inherited from RS-XAUUSD-20260815-004)",
                   "replay": "PineTS 0.11.1 on TradingView FX_IDC:XAUUSD 30m bars; control 346/346 vs the formal V3.9 export",
                   "r": "(exit / entry - 1) × 100 / 0.5", "window": list(WINDOW), "after_window": list(AFTER),
                   "resolution_bound": "2.8 × SD(R, all modes in window) × √(1/n_mode + 1/n_OFF); modes share signals, so descriptive"},
        "headline": {"off_mean_r": w["OFF"]["mean_r"], "t1_mean_r": w["T1"]["mean_r"], "t2_mean_r": w["T2"]["mean_r"],
                     "t3_mean_r": w["T3"]["mean_r"], "off_n": w["OFF"]["n"], "t1_n": w["T1"]["n"],
                     "t2_n": w["T2"]["n"], "t3_n": w["T3"]["n"], "t3_profit_factor": w["T3"]["profit_factor"],
                     "t3_tv_export_n": modes["T3"]["coverage_vs_tv_export_2026_08_16"]["tv_export_trades"]},
        "modes": modes,
        "charts": [],
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def sha(path: Path) -> dict:
        data = path.read_bytes()
        return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    sources = [{"role": f"replayed_trades_{m.lower()}", "source_system": "PineTS 0.11.1 replay of the research Pine",
                "locator": rel(p), **sha(p), "immutable": False, "privacy": "private_derived"} for m, (p, _) in MODES.items()]
    sources += [{"role": f"tv_export_{m.lower()}_2026_08_16", "source_system": "TradingView Strategy Tester export",
                 "locator": rel(p), **sha(p), "immutable": True, "privacy": "private_raw_source"} for m, (_, p) in MODES.items()]
    (STUDY_DIR / "source_manifest.json").write_text(json.dumps({
        "schema_version": 1, "study_id": STUDY_ID, "generated_at": payload["generated_at"],
        "timezone_assumptions": "Trade times are Asia/Taipei bar-open times, as TradingView exports them.",
        "sources": sources}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": rel(STUDY_DIR / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
