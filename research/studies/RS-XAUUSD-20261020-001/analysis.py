#!/usr/bin/env python3
"""RS-XAUUSD-20261020-001 — US-session S1 with the last closed 30m bar above the upper band.

Design: research/studies/RS-XAUUSD-20261020-001/decision_log.md. Measures live in build_s1_entry_reasons.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import build_s1_entry_reasons as er  # noqa: E402

STUDY_ID = "RS-XAUUSD-20261020-001"
OUTPUT_DIR = Path("reproduced")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _trades, _price, enriched = er.load_inputs()
    generated = datetime.now(er.TAIPEI).isoformat(timespec="seconds")
    payload = {"schema_version": 1, "study_id": STUDY_ID, "generated_at": generated, **er.study_us(enriched)}
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    er.manifest(STUDY_ID, generated, OUTPUT_DIR)
    print(json.dumps({"written": str(OUTPUT_DIR / "results.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
