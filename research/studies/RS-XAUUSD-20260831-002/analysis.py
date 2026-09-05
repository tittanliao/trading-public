#!/usr/bin/env python3
"""RS-XAUUSD-20260831-002 — why did S2 V3.2 weaken, and is the Asia effect separate?

Four published studies have seen S2 deteriorate by unrelated methods. This asks one
question: what is the decline actually made of?

The tool is a shift-share decomposition. A change in a weighted average has exactly three
sources — the parts got worse, the mix moved toward worse parts, or both — and separating
them is what decides whether "Asia is deteriorating" is a finding or a restatement of
"S2 is deteriorating". S1 over identical date ranges is the control: a decline that also
appears in S1 is about the market, not about S2.
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S2_TRADES = Path("local-inputs/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv")
S1_TRADES = Path("local-inputs/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv")
DAILY = Path("local-inputs/xauusd-1d.csv")
OUT = Path("reproduced")
TRIALS = 20000
SEED = 20260901
EXPORT_DATE = "2026-07-11"
SPLIT = 0.7


def session_of(moment: datetime) -> str:
    minutes = moment.hour * 60 + moment.minute
    if 60 <= minutes <= 419:
        return "overnight"
    if 420 <= minutes <= 899:
        return "asia"
    if 900 <= minutes <= 1229:
        return "europe"
    return "us"


def load_trades(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    pairs: dict[str, dict] = {}
    for row in rows:
        pairs.setdefault(row["Trade number"], {})[row["Type"]] = row
    trades = []
    for number, pair in pairs.items():
        entry, exit_ = pair.get("Entry long"), pair.get("Exit long")
        if not entry or not exit_ or exit_["Date and time"][:10] == EXPORT_DATE:
            continue
        moment = datetime.strptime(entry["Date and time"], "%Y-%m-%d %H:%M")
        trades.append({
            "entry_time": moment,
            "session": session_of(moment),
            "return_pct": float(exit_["Return %"]),
            "pnl_usd": float(exit_["Net PnL USD"]),
            "win": float(exit_["Net PnL USD"]) > 0,
        })
    trades.sort(key=lambda t: t["entry_time"])
    return trades


def stats(group: list[dict]) -> dict:
    if not group:
        return {"n": 0}
    returns = [t["return_pct"] for t in group]
    wins = sum(1 for t in group if t["win"])
    gross_win = sum(t["pnl_usd"] for t in group if t["pnl_usd"] > 0)
    gross_loss = -sum(t["pnl_usd"] for t in group if t["pnl_usd"] <= 0)
    return {
        "n": len(group),
        "win_rate_pct": round(100 * wins / len(group), 2),
        "mean_return_pct": round(sum(returns) / len(returns), 4),
        "total_return_pct": round(sum(returns), 2),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss else None,
    }


def by_session(group: list[dict], sessions: list[str]) -> dict:
    return {s: stats([t for t in group if t["session"] == s]) for s in sessions}


def gold_context(start: datetime, end: datetime) -> dict:
    """What the market itself was doing in a window, so 'the regime changed' is a number."""
    bars = []
    for row in csv.DictReader(DAILY.open(encoding="utf-8-sig")):
        moment = datetime.strptime(row["time"][:10], "%Y-%m-%d")
        if start <= moment <= end:
            bars.append({"c": float(row["close"]), "h": float(row["high"]),
                         "l": float(row["low"])})
    if len(bars) < 2:
        return {"sessions": len(bars)}
    closes = [b["c"] for b in bars]
    rets = [100 * (closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    ranges = [100 * (b["h"] - b["l"]) / b["c"] for b in bars]
    return {
        "sessions": len(bars),
        "total_move_pct": round(100 * (closes[-1] / closes[0] - 1), 2),
        "mean_daily_return_pct": round(mean, 4),
        "daily_volatility_pct": round(var ** 0.5, 4),
        "mean_daily_range_pct": round(sum(ranges) / len(ranges), 4),
        "up_day_share_pct": round(100 * sum(1 for r in rets if r > 0) / len(rets), 2),
    }


def main() -> None:
    for path in (S2_TRADES, S1_TRADES, DAILY):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    sessions = ["overnight", "asia", "europe", "us"]
    s2 = load_trades(S2_TRADES)
    s1_all = load_trades(S1_TRADES)

    split_at = int(len(s2) * SPLIT)
    early, recent = s2[:split_at], s2[split_at:]
    boundary = recent[0]["entry_time"]
    # S1 is cut by the same dates, not the same ratio: the control has to answer "what
    # happened in this window", and S1 trades far more often than S2.
    s1_early = [t for t in s1_all if t["entry_time"] < boundary]
    s1_recent = [t for t in s1_all if t["entry_time"] >= boundary]

    # Shift-share. The change in the overall mean splits into what happened inside each
    # session (within), how the mix of sessions moved (between), and the interaction.
    total_early, total_recent = len(early), len(recent)
    within = between = interaction = 0.0
    per_session = {}
    for name in sessions:
        a = [t for t in early if t["session"] == name]
        b = [t for t in recent if t["session"] == name]
        wa, wb = len(a) / total_early, len(b) / total_recent
        ma = sum(t["return_pct"] for t in a) / len(a) if a else 0.0
        mb = sum(t["return_pct"] for t in b) / len(b) if b else 0.0
        within += wa * (mb - ma)
        between += (wb - wa) * ma
        interaction += (wb - wa) * (mb - ma)
        per_session[name] = {
            "early": stats(a), "recent": stats(b),
            "weight_early_pct": round(100 * wa, 2),
            "weight_recent_pct": round(100 * wb, 2),
            "mean_change_pct": round(mb - ma, 4),
            "win_rate_change_pct_points": round(
                (stats(b).get("win_rate_pct", 0) or 0) - (stats(a).get("win_rate_pct", 0) or 0), 2),
        }
    observed_change = (sum(t["return_pct"] for t in recent) / total_recent
                       - sum(t["return_pct"] for t in early) / total_early)

    # Is a drop this size unusual for a sample this size? Shuffle the period labels.
    rng = random.Random(SEED)
    returns = [t["return_pct"] for t in s2]
    at_least = 0
    for _ in range(TRIALS):
        index = list(range(len(s2)))
        rng.shuffle(index)
        a = sum(returns[i] for i in index[:split_at]) / split_at
        b = sum(returns[i] for i in index[split_at:]) / (len(s2) - split_at)
        if (b - a) <= observed_change:
            at_least += 1

    results = {
        "schema_version": 1,
        "study_id": "RS-XAUUSD-20260831-002",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": "S2-Hammer V3.2, with S1 AweWithBB V3.9 as control",
        "method": {
            "s2_trades": len(s2),
            "s1_trades": len(s1_all),
            "chronological_split": SPLIT,
            "split_boundary": boundary.strftime("%Y-%m-%d"),
            "control": ("S1 cut by the same calendar boundary, not the same ratio, so it "
                        "answers what happened in this window"),
            "decomposition": ("shift-share: within-session change, between-session mix "
                              "change, and their interaction, summing to the total change"),
            "null": "period labels shuffled across trades, preserving group sizes",
            "permutation_trials": TRIALS,
            "random_seed": SEED,
            "session_definition_source": "RS-XAUUSD-20260727-007, reused verbatim",
        },
        "s2_periods": {
            "early": {"from": early[0]["entry_time"].strftime("%Y-%m-%d"),
                      "to": early[-1]["entry_time"].strftime("%Y-%m-%d"), **stats(early)},
            "recent": {"from": recent[0]["entry_time"].strftime("%Y-%m-%d"),
                       "to": recent[-1]["entry_time"].strftime("%Y-%m-%d"), **stats(recent)},
        },
        "s1_control": {
            "early": {"from": s1_early[0]["entry_time"].strftime("%Y-%m-%d"),
                      "to": s1_early[-1]["entry_time"].strftime("%Y-%m-%d"), **stats(s1_early)},
            "recent": {"from": s1_recent[0]["entry_time"].strftime("%Y-%m-%d"),
                       "to": s1_recent[-1]["entry_time"].strftime("%Y-%m-%d"), **stats(s1_recent)},
        },
        "by_session": per_session,
        "decomposition": {
            "total_mean_change_pct": round(observed_change, 4),
            "within_session_pct": round(within, 4),
            "between_session_mix_pct": round(between, 4),
            "interaction_pct": round(interaction, 4),
            "within_share_of_change_pct": round(100 * within / observed_change, 1),
            "between_share_of_change_pct": round(100 * between / observed_change, 1),
            "permutation_p_decline_at_least_observed": round(at_least / TRIALS, 4),
        },
        "gold_context": {
            "early": gold_context(early[0]["entry_time"], recent[0]["entry_time"]),
            "recent": gold_context(recent[0]["entry_time"], recent[-1]["entry_time"]),
        },
        "by_year": {
            str(year): stats([t for t in s2 if t["entry_time"].year == year])
            for year in sorted({t["entry_time"].year for t in s2})
        },
        "s1_by_year": {
            str(year): stats([t for t in s1_all if t["entry_time"].year == year])
            for year in sorted({t["entry_time"].year for t in s1_all})
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    d = results["decomposition"]
    print(json.dumps({"s2": len(s2), "s1": len(s1_all), "boundary": results["method"]["split_boundary"],
                      "total_change": d["total_mean_change_pct"],
                      "within": d["within_session_pct"], "between": d["between_session_mix_pct"],
                      "p": d["permutation_p_decline_at_least_observed"]}))


if __name__ == "__main__":
    main()
