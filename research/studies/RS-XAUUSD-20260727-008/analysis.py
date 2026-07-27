#!/usr/bin/env python3
"""Public reproduction method for the S2-Hammer V1.9 vs V3.2 gap report.

Per docs/RESEARCH_DEVELOPMENT_SPEC.md section 5, a gap study never recomputes the
underlying fail-pattern breakdown; it reads the two solo studies' results.json and
computes version deltas only. Point --v1-results / --v2-results at the sibling
published RS-XAUUSD-20260727-006 / -007 results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ENTRY_SLOTS_30M = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
BB_ZONE_ORDER = ["below_lower", "near_lower", "lower_mid", "near_middle", "upper_mid", "near_upper", "above_upper"]


def diff_table(v1: dict, v2: dict, keys: list[str]) -> dict:
    out = {}
    for key in keys:
        a, b = v1.get(key, {}), v2.get(key, {})
        wr_a, wr_b = a.get("win_rate_pct"), b.get("win_rate_pct")
        out[key] = {
            "v1_n": a.get("n", 0), "v1_win_rate_pct": wr_a, "v1_profit_factor": a.get("profit_factor"),
            "v2_n": b.get("n", 0), "v2_win_rate_pct": wr_b, "v2_profit_factor": b.get("profit_factor"),
            "win_rate_pct_diff_v2_minus_v1": round(wr_b - wr_a, 2) if (wr_a is not None and wr_b is not None) else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-results", type=Path, required=True)
    parser.add_argument("--v2-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v1 = json.loads(args.v1_results.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_results.read_text(encoding="utf-8"))

    baseline_diff = {
        "v1": v1["baseline"], "v2": v2["baseline"],
        "win_rate_pct_diff": round(v2["baseline"]["win_rate_pct"] - v1["baseline"]["win_rate_pct"], 2),
        "profit_factor_diff": round(v2["baseline"]["profit_factor"] - v1["baseline"]["profit_factor"], 3),
    }
    by_entry_30m_diff = diff_table(v1["by_entry_30m"], v2["by_entry_30m"], ENTRY_SLOTS_30M)

    fail_type_share_diff = {}
    for ft in ["immediate_loss", "false_breakout", "time_bleed", "normal_sl"]:
        a = v1["fail_pattern"]["by_type"].get(ft, {"pct": 0})
        b = v2["fail_pattern"]["by_type"].get(ft, {"pct": 0})
        fail_type_share_diff[ft] = {"v1_pct": a["pct"], "v2_pct": b["pct"], "diff": round(b["pct"] - a["pct"], 1)}

    bb_zone_diff = diff_table(v1["bb_zone"], v2["bb_zone"], BB_ZONE_ORDER)

    output = {
        "baseline_diff": baseline_diff,
        "by_entry_30m_diff": by_entry_30m_diff,
        "fail_type_share_diff": fail_type_share_diff,
        "bb_zone_diff": bb_zone_diff,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
