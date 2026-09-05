#!/usr/bin/env python3
"""RS-XAUUSD-20260831-001 — does removing S2's Asia-session entries improve anything?

Deterministic. Reads the immutable S2 V3.2 trade export, buckets entries by Taipei session
using the same definition RS-XAUUSD-20260727-007 published, and tests one filter proposal.

The whole study turns on one control. Dropping the worst-performing third of any sample
raises the average of what is left, so "removing Asia improves the average" is not
evidence of anything on its own. The null shuffles session labels across trades while
preserving group sizes, which asks the only question that matters: does the session label
carry information beyond what any equally sized group would show?
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# The immutable S2 V3.2 trade export. Its sha256 is recorded in the study's
# source_manifest.json; a reader supplies their own copy at this path.
TRADES = Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv")
OUT = Path("reproduced")
TRIALS = 20000
SEED = 20260831
ROUND_TRIP_COST_PCT = 0.02
EXPORT_DATE = "2026-07-11"


def session_of(moment: datetime) -> str:
    """Taipei buckets, verbatim from RS-XAUUSD-20260727-007's published method."""
    minutes = moment.hour * 60 + moment.minute
    if 60 <= minutes <= 419:
        return "overnight"
    if 420 <= minutes <= 899:
        return "asia"
    if 900 <= minutes <= 1229:
        return "europe"
    return "us"


def load_trades() -> list[dict]:
    rows = list(csv.DictReader(TRADES.open(encoding="utf-8-sig")))
    by_number: dict[str, dict] = {}
    for row in rows:
        by_number.setdefault(row["Trade number"], {})[row["Type"]] = row
    trades = []
    for number, pair in sorted(by_number.items(), key=lambda kv: int(kv[0])):
        entry = pair.get("Entry long")
        exit_ = pair.get("Exit long")
        if not entry or not exit_:
            continue
        moment = datetime.strptime(entry["Date and time"], "%Y-%m-%d %H:%M")
        trades.append({
            "n": int(number),
            "entry_time": moment,
            "session": session_of(moment),
            "exit_date": exit_["Date and time"][:10],
            "return_pct": float(exit_["Return %"]),
            "pnl_usd": float(exit_["Net PnL USD"]),
        })
    trades.sort(key=lambda t: t["entry_time"])
    # RS-XAUUSD-20260727-007 published 157 trades, not 158: the final trade exits on
    # 2026-07-11, the export date itself, and was excluded. Matching that set exactly is
    # what makes this study's session numbers directly comparable to the published ones.
    trades = [t for t in trades if t["exit_date"] != EXPORT_DATE]
    return trades


def describe(group: list[dict]) -> dict:
    if not group:
        return {"n": 0}
    wins = [t for t in group if t["pnl_usd"] > 0]
    losses = [t for t in group if t["pnl_usd"] <= 0]
    gross_win = sum(t["pnl_usd"] for t in wins)
    gross_loss = -sum(t["pnl_usd"] for t in losses)
    returns = [t["return_pct"] for t in group]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1) if len(group) > 1 else 0.0
    return {
        "n": len(group),
        "wins": len(wins),
        "win_rate_pct": round(100 * len(wins) / len(group), 2),
        "mean_return_pct": round(mean, 4),
        "mean_return_net_of_cost_pct": round(mean - ROUND_TRIP_COST_PCT, 4),
        "sd_return_pct": round(variance ** 0.5, 4),
        "total_return_pct": round(sum(returns), 2),
        "net_pnl_usd": round(sum(t["pnl_usd"] for t in group), 2),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else None,
    }


def main() -> None:
    if not TRADES.is_file():
        raise SystemExit(f"trade export not found at {TRADES}")
    trades = load_trades()
    sessions = ["overnight", "asia", "europe", "us"]
    total_n = len(trades)
    total_pnl = sum(t["pnl_usd"] for t in trades)
    total_return = sum(t["return_pct"] for t in trades)

    by_session = {}
    for name in sessions:
        group = [t for t in trades if t["session"] == name]
        stats = describe(group)
        stats["share_of_trades_pct"] = round(100 * stats["n"] / total_n, 2)
        stats["share_of_net_pnl_pct"] = round(100 * stats["net_pnl_usd"] / total_pnl, 2)
        stats["share_of_total_return_pct"] = round(100 * stats["total_return_pct"] / total_return, 2)
        by_session[name] = stats

    baseline = describe(trades)
    kept = [t for t in trades if t["session"] != "asia"]
    dropped = [t for t in trades if t["session"] == "asia"]
    filtered = describe(kept)

    # The control. Shuffling the labels keeps every group size and asks whether "asia"
    # picks a worse set than an arbitrary set of the same size would.
    rng = random.Random(SEED)
    observed_gain = filtered["mean_return_pct"] - baseline["mean_return_pct"]
    observed_wr_gain = filtered["win_rate_pct"] - baseline["win_rate_pct"]
    returns = [t["return_pct"] for t in trades]
    wins = [1 if t["pnl_usd"] > 0 else 0 for t in trades]
    keep_n = len(kept)
    gain_at_least = 0
    wr_at_least = 0
    for _ in range(TRIALS):
        index = list(range(total_n))
        rng.shuffle(index)
        keep = index[:keep_n]
        mean_kept = sum(returns[i] for i in keep) / keep_n
        if mean_kept - baseline["mean_return_pct"] >= observed_gain:
            gain_at_least += 1
        wr_kept = 100 * sum(wins[i] for i in keep) / keep_n
        if wr_kept - baseline["win_rate_pct"] >= observed_wr_gain:
            wr_at_least += 1

    # Asia was not picked blind — it was picked after reading a published table of all four
    # sessions. So the honest test is not "is Asia unusual", it is "is the BEST of four
    # drops unusual", which is what a shuffle of the labels reproduces.
    per_session_gain = {}
    for name in sessions:
        keep_group = [t for t in trades if t["session"] != name]
        if not keep_group:
            continue
        per_session_gain[name] = round(
            describe(keep_group)["mean_return_pct"] - baseline["mean_return_pct"], 4)
    best_session = max(per_session_gain, key=per_session_gain.get)
    best_gain = per_session_gain[best_session]

    group_sizes = [len([t for t in trades if t["session"] == s]) for s in sessions]
    rng_family = random.Random(SEED + 1)
    family_at_least = 0
    null_best = []
    for _ in range(TRIALS):
        index = list(range(total_n))
        rng_family.shuffle(index)
        cursor = 0
        groups = []
        for size in group_sizes:
            groups.append(index[cursor:cursor + size])
            cursor += size
        best = max(
            (sum(returns[i] for i in index if i not in set(g)) / (total_n - len(g))
             - baseline["mean_return_pct"])
            for g in groups if len(g) < total_n
        )
        null_best.append(best)
        if best >= best_gain:
            family_at_least += 1
    null_best.sort()

    family = {
        "sessions_searched": len(sessions),
        "gain_by_dropped_session_pct": per_session_gain,
        "best_session_to_drop": best_session,
        "best_observed_gain_pct": best_gain,
        "null_median_best_gain_pct": round(null_best[TRIALS // 2], 4),
        "null_95th_best_gain_pct": round(null_best[int(TRIALS * 0.95)], 4),
        "p_best_gain_at_least_observed": round(family_at_least / TRIALS, 4),
        "note": ("Accounts for having chosen the session after seeing all four. The "
                 "single-session p-value does not."),
    }

    split = int(len(trades) * 0.7)
    periods = {}
    for label, part in (("early_70pct", trades[:split]), ("recent_30pct", trades[split:])):
        asia = [t for t in part if t["session"] == "asia"]
        rest = [t for t in part if t["session"] != "asia"]
        periods[label] = {
            "from": part[0]["entry_time"].strftime("%Y-%m-%d"),
            "to": part[-1]["entry_time"].strftime("%Y-%m-%d"),
            "asia": describe(asia),
            "not_asia": describe(rest),
        }

    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260831-001",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S2-Hammer V3.2",
        "method": {
            "trades": total_n,
            "session_timezone": "Asia/Taipei",
            "session_buckets": ("overnight=01:00-06:59; asia=07:00-14:59; "
                                "europe=15:00-20:29; us=20:30-00:59"),
            "session_definition_source": "RS-XAUUSD-20260727-007, reused verbatim",
            "session_assigned_on": "entry time",
            "null": ("session labels shuffled across trades preserving group sizes; asks "
                     "whether the label selects a worse set than an arbitrary set of the "
                     "same size"),
            "permutation_trials": TRIALS,
            "random_seed": SEED,
            "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
            "chronological_split": 0.7,
        },
        "baseline": baseline,
        "by_session": by_session,
        "drop_asia": {
            "kept": filtered,
            "dropped": describe(dropped),
            "trades_removed": len(dropped),
            "share_of_trades_removed_pct": round(100 * len(dropped) / total_n, 2),
            "mean_return_gain_pct": round(observed_gain, 4),
            "win_rate_gain_pct_points": round(observed_wr_gain, 2),
            "total_return_change_pct": round(filtered["total_return_pct"] - baseline["total_return_pct"], 2),
            "net_pnl_change_usd": round(filtered["net_pnl_usd"] - baseline["net_pnl_usd"], 2),
            "permutation_p_mean_return": round(gain_at_least / TRIALS, 4),
            "permutation_p_win_rate": round(wr_at_least / TRIALS, 4),
        },
        "family_correction": family,
        "chronological": periods,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"trades": total_n, "asia_n": len(dropped),
                      "p_mean_return": results["drop_asia"]["permutation_p_mean_return"],
                      "p_win_rate": results["drop_asia"]["permutation_p_win_rate"]}))


if __name__ == "__main__":
    main()
