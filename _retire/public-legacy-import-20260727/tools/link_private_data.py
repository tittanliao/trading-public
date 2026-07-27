#!/usr/bin/env python3
"""Restore ignored legacy CSV paths as symlinks to sibling Private raw storage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path(
    os.environ.get("TRADING_PRIVATE_ROOT", PUBLIC_ROOT.parent / "trading-private")
).resolve()
MANIFEST = PRIVATE_ROOT / "config" / "legacy_trading_replatform_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["classification"] == "private_raw"
        ]
    failures: list[str] = []
    for row in rows:
        source = PRIVATE_ROOT / row["destination"]
        link = PUBLIC_ROOT / row["relative_path"]
        if not source.is_file() or sha256(source) != row["sha256"]:
            failures.append(f"invalid Private source: {row['relative_path']}")
            continue
        if not args.check:
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                failures.append(f"refusing to replace non-symlink: {row['relative_path']}")
                continue
            link.symlink_to(source)
        if not link.is_symlink() or link.resolve() != source.resolve():
            failures.append(f"missing or wrong data link: {row['relative_path']}")
    print(f"private data links: {len(rows)}")
    print(f"failures: {len(failures)}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
