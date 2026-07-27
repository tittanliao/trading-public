#!/usr/bin/env python3
"""Validate generated Public pages, catalog links, and privacy boundaries."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
GENERATED_PAGES = [
    ROOT / "index.html",
    ROOT / "xauusd/index.html",
    ROOT / "tx/index.html",
    ROOT / "research/index.html",
] + sorted((ROOT / "research/studies").glob("*/index.html"))
PROHIBITED_SUFFIXES = {".csv", ".doc", ".docx", ".xls", ".xlsx"}
PROHIBITED_EXACT = {"data/logs.json", "xauusd/signal_status.json"}


def git_files() -> list[str]:
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"]
    ).split(b"\0")
    staged = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--cached", "--name-only", "-z"]
    ).split(b"\0")
    return sorted({item.decode() for item in tracked + staged if item})


def main() -> int:
    failures: list[str] = []
    catalog = json.loads((ROOT / "site/catalog.json").read_text(encoding="utf-8"))
    for item in catalog["items"]:
        if not (ROOT / item["path"]).is_file():
            failures.append(f"missing catalog target: {item['path']}")

    href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.I)
    for page in GENERATED_PAGES:
        if not page.is_file():
            failures.append(f"missing generated page: {page.relative_to(ROOT)}")
            continue
        for href in href_pattern.findall(page.read_text(encoding="utf-8")):
            split = urlsplit(href)
            if split.scheme or split.netloc or href.startswith("#"):
                continue
            target = (page.parent / unquote(split.path)).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                failures.append(
                    f"broken generated link: {page.relative_to(ROOT)} -> {href}"
                )

    for relative in git_files():
        path = Path(relative)
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            failures.append(f"prohibited Public binary/raw file: {relative}")
        if relative in PROHIBITED_EXACT or ".claude/memory" in relative:
            failures.append(f"prohibited Public private-state path: {relative}")

    print(f"catalog items: {len(catalog['items'])}")
    print(f"generated pages: {len(GENERATED_PAGES)}")
    print(f"failures: {len(failures)}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
