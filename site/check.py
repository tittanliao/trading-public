#!/usr/bin/env python3
"""Validate the clean-rebuild Public site: exact route allow-list, dead links, retired
routes actually gone, and no private-token leakage.

docs/PUBLIC_SITE_REBUILD_SPEC.md section 12 calls this "the site/privacy checker" — it is
the safety gate run after `site/build.py` and before commit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import POC_STUDIES, ROUTES, WEEKLY_FORECAST_WEEK  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GENERATED_PAGES = [ROOT / (r + "/index.html" if r else "index.html") for r in ROUTES]

# Routes this rebuild explicitly retires. Their presence after cutover is a failure —
# docs/PUBLIC_SITE_REBUILD_SPEC.md section 3/13: "retired routes should return 404".
RETIRED_PAGES = [
    ROOT / "jargon/index.html",
    ROOT / "lessons/index.html",
    ROOT / "tx/index.html",
    ROOT / "xauusd/index.html",
    ROOT / "v1/index.html",
    ROOT / "zh/index.html",
]
RETIRED_DIRS = [ROOT / "v1", ROOT / "zh", ROOT / "jargon", ROOT / "lessons"]

PRIVATE_TOKENS = (
    "/users/", "googledrive", "reports/xauusd/weekly", "state/xauusd", "data/xauusd",
    "journal_locator", "input_set_id", "ledger_run_id", ".docx", ".gdoc",
    "private_provenance", "the owner", "owner-directed", "owner-confirmed",
)


def fail(messages: list[str], text: str) -> None:
    messages.append(text)


def check_routes() -> list[str]:
    errors: list[str] = []
    for page in GENERATED_PAGES:
        if not page.is_file():
            fail(errors, f"missing canonical page: {page.relative_to(ROOT)}")
    if len(GENERATED_PAGES) != 8:
        fail(errors, f"expected exactly 8 canonical routes, ROUTES has {len(GENERATED_PAGES)}")
    return errors


def check_retired() -> list[str]:
    errors: list[str] = []
    for page in RETIRED_PAGES:
        if page.is_file():
            fail(errors, f"retired route still present: {page.relative_to(ROOT)}")
    for study_dir in (ROOT / "research/studies").iterdir():
        if not study_dir.is_dir() or study_dir.name in POC_STUDIES:
            continue
        page = study_dir / "index.html"
        if page.is_file():
            fail(errors, f"non-POC study page must be unlinked/404 in this phase: {page.relative_to(ROOT)}")
    for legacy in (ROOT / "xauusd/weekly/2026-W34", ROOT / "xauusd/weekly/2026-W35"):
        page = legacy / "index.html"
        if page.is_file():
            fail(errors, f"retired dated Weekly page still present: {page.relative_to(ROOT)}")
    return errors


def extract_links(html_text: str) -> list[str]:
    return re.findall(r'(?:href|src)="([^"]+)"', html_text)


def check_links_and_images() -> list[str]:
    errors: list[str] = []
    for page in GENERATED_PAGES:
        text = page.read_text(encoding="utf-8")
        for link in extract_links(text):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = (page.parent / unquote(urlsplit(link).path)).resolve()
            if not target.is_file() and not target.is_dir():
                fail(errors, f"broken link in {page.relative_to(ROOT)}: {link}")
    for study_id in POC_STUDIES:
        results = json.loads((ROOT / "research/studies" / study_id / "results.json").read_text(encoding="utf-8"))
        for chart in results.get("charts", []):
            image = ROOT / "research/studies" / study_id / "charts" / chart["file"]
            if not image.is_file():
                fail(errors, f"missing chart image for {study_id}: {chart['file']}")
    return errors


def check_privacy() -> list[str]:
    errors: list[str] = []
    scanned = list(GENERATED_PAGES)
    scanned.append(ROOT / f"xauusd/weekly/{WEEKLY_FORECAST_WEEK}/summary.json")
    for sid in POC_STUDIES:
        d = ROOT / "research/studies" / sid
        scanned += [d / "study.json", d / "results.json", d / "analysis.py"]
    for path in scanned:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in PRIVATE_TOKENS:
            if token in text:
                fail(errors, f"prohibited token '{token}' found in {path.relative_to(ROOT)}")
    return errors


def check_chart_count() -> tuple[int, list[str]]:
    errors: list[str] = []
    total = 0
    for sid, expected in zip(POC_STUDIES, (20, 5, 6)):
        results = json.loads((ROOT / "research/studies" / sid / "results.json").read_text(encoding="utf-8"))
        n = len(results.get("charts", []))
        total += n
        if n != expected:
            fail(errors, f"{sid}: expected {expected} charts, results.json declares {n}")
    return total, errors


def check_null_results() -> list[str]:
    errors: list[str] = []
    registry = json.loads((ROOT / "research/null-results/null_results.json").read_text(encoding="utf-8"))
    totals = registry.get("totals", {})
    hypotheses = [e for e in registry.get("entries", []) if e.get("kind") == "hypothesis"]
    if totals.get("hypotheses") != 63 or len(hypotheses) != 63:
        fail(errors, f"expected 63 registered hypotheses, found {len(hypotheses)} (totals says {totals.get('hypotheses')})")
    by_verdict = totals.get("by_verdict", {})
    expected_verdicts = {"no_evidence": 60, "underpowered": 2, "below_cost": 1}
    if by_verdict != expected_verdicts:
        fail(errors, f"verdict totals mismatch: expected {expected_verdicts}, got {by_verdict}")
    return errors


def main() -> int:
    errors: list[str] = []
    errors += check_routes()
    errors += check_retired()
    errors += check_links_and_images()
    errors += check_privacy()
    chart_total, chart_errors = check_chart_count()
    errors += chart_errors
    errors += check_null_results()

    print(json.dumps({
        "routes checked": len(GENERATED_PAGES),
        "poc chart total": chart_total,
        "failures": len(errors),
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
