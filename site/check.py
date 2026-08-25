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
WEEKLY_SOURCES = sorted((ROOT / "xauusd/weekly").glob("*/summary.json"))
GENERATED_PAGES = [
    ROOT / "index.html",
    ROOT / "xauusd/index.html",
    ROOT / "tx/index.html",
    ROOT / "research/index.html",
] + [
    # Pages that exist only when their source data does. Listed conditionally rather than
    # unconditionally: the glossary and the null registry are generated from optional
    # inputs, and a hard entry would fail the check on a checkout that lacks them. Both
    # went live once without being checked at all, which is how a 404 stayed invisible.
    page for page in (
        ROOT / "glossary/index.html",
        ROOT / "lessons/index.html",
        ROOT / "research/null-results/index.html",
    ) if page.is_file()
] + sorted((ROOT / "research/studies").glob("*/index.html")) + [
    source.parent / "index.html" for source in WEEKLY_SOURCES
]
if WEEKLY_SOURCES:
    GENERATED_PAGES.append(ROOT / "xauusd/weekly/index.html")
PROHIBITED_SUFFIXES = {".csv", ".doc", ".docx", ".xls", ".xlsx"}
PROHIBITED_EXACT = {"data/logs.json", "xauusd/signal_status.json"}
WEEKLY_KEYS = {
    "schema_version", "market", "forecast_week", "edition", "published_at",
    "publication_mode", "confidence", "source_count", "source_producers",
    "source_fingerprint", "data_cutoff", "market_summary", "scenario_comparison",
    "adopted_scenarios", "agreements", "disagreements", "key_levels",
    "strategy_plan", "event_risk", "recommendation", "evidence_limits", "disclaimer",
}
# Present only on an edition published as a translation. The 2026-W34 report was written in
# Chinese; everything from W35 is English, so this should stay rare.
WEEKLY_OPTIONAL_KEYS = {"language_note"}
PRIVATE_TOKENS = (
    "/users/", "reports/xauusd/weekly", "state/xauusd", "data/xauusd",
    "journal_locator", "input_set_id", "ledger_run_id", "resource_key",
    ".docx", ".gdoc", "private_provenance",
)

# Scanned on every generated page, not only weekly summaries. Thirty-four references to a
# private person reached thirteen published study pages before anyone looked, because the
# only leak scan ran on the weekly JSON.
PAGE_PRIVATE_TOKENS = PRIVATE_TOKENS + ("owner", "googledrive", "docs/strategy.md",
                                        "docs/policy.md", "decision_log", "handoff.md")


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

    for source in WEEKLY_SOURCES:
        relative = source.relative_to(ROOT)
        try:
            summary = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid weekly JSON: {relative}: {exc}")
            continue
        keys = set(summary)
        if not WEEKLY_KEYS <= keys or (keys - WEEKLY_KEYS) - WEEKLY_OPTIONAL_KEYS:
            failures.append(f"weekly summary key mismatch: {relative}")
        if summary.get("schema_version") != "1.0" or summary.get("market") != "XAUUSD":
            failures.append(f"weekly summary identity mismatch: {relative}")
        if source.parent.name != summary.get("forecast_week"):
            failures.append(f"weekly summary directory mismatch: {relative}")
        mode = summary.get("publication_mode")
        source_count = summary.get("source_count")
        producers = summary.get("source_producers")
        comparison = summary.get("scenario_comparison")
        if mode not in {"single_source", "multi_source"}:
            failures.append(f"invalid weekly publication mode: {relative}")
        if not isinstance(source_count, int) or source_count < 1:
            failures.append(f"invalid weekly source count: {relative}")
        if not isinstance(producers, list) or len(producers) != source_count:
            failures.append(f"weekly producer count mismatch: {relative}")
        if not isinstance(comparison, list) or len(comparison) != source_count:
            failures.append(f"weekly comparison count mismatch: {relative}")
        if mode == "single_source" and source_count != 1:
            failures.append(f"single-source weekly summary has multiple sources: {relative}")
        if mode == "multi_source" and (not isinstance(source_count, int) or source_count < 2):
            failures.append(f"multi-source weekly summary lacks multiple sources: {relative}")
        adopted = summary.get("adopted_scenarios")
        probabilities = (
            [item.get("probability") for item in adopted if isinstance(item, dict)]
            if isinstance(adopted, list) else []
        )
        if (
            not isinstance(adopted, list)
            or len(probabilities) != len(adopted)
            or any(not isinstance(value, int) for value in probabilities)
            or sum(probabilities) != 100
        ):
            failures.append(f"weekly adopted probabilities do not sum to 100: {relative}")
        lowered = source.read_text(encoding="utf-8").lower()
        leaked = [token for token in PRIVATE_TOKENS if token in lowered]
        if leaked:
            failures.append(f"weekly summary contains private token(s) {leaked}: {relative}")

    # A study ID printed as text is a citation the reader cannot follow and no link check
    # can see. The signal playbook cited a study that had no published page for days.
    study_pattern = re.compile(r"RS-[A-Z]+-\d{8}-\d{3}")
    published = {p.name for p in (ROOT / "research/studies").glob("*") if p.is_dir()}

    href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.I)
    for page in GENERATED_PAGES:
        if not page.is_file():
            failures.append(f"missing generated page: {page.relative_to(ROOT)}")
            continue
        page_text = page.read_text(encoding="utf-8")
        lowered_page = page_text.lower()
        for token in PAGE_PRIVATE_TOKENS:
            if token in lowered_page:
                failures.append(
                    f"private token {token!r} on generated page: {page.relative_to(ROOT)}"
                )
        for cited in sorted(set(study_pattern.findall(page_text))):
            if cited not in published:
                failures.append(
                    f"page cites an unpublished study: {page.relative_to(ROOT)} -> {cited}"
                )
        for href in href_pattern.findall(page_text):
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
    print(f"weekly summaries: {len(WEEKLY_SOURCES)}")
    print(f"failures: {len(failures)}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
