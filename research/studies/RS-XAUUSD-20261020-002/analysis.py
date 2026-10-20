#!/usr/bin/env python3
"""RS-XAUUSD-20261020-002 — does gold fall from 09:30 Taipei, above all after an Asian-morning S1?

Design: research/studies/RS-XAUUSD-20261020-002/decision_log.md. Measures live in build_s1_entry_reasons.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import build_s1_entry_reasons as er  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261020-002"
OUTPUT_DIR = Path("reproduced")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    trades, price, _enriched = er.load_inputs()
    generated = datetime.now(er.TAIPEI).isoformat(timespec="seconds")
    payload = {"schema_version": 1, "study_id": STUDY_ID, "generated_at": generated, **er.study_asia(trades, price)}
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    er.chart_asia(price, OUTPUT_DIR / "charts/slot_2h_returns.png")
    er.manifest(STUDY_ID, generated, OUTPUT_DIR)
    print(json.dumps({"written": str(OUTPUT_DIR / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
