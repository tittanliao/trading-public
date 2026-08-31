#!/usr/bin/env python3
"""Validate the Public site: canonical route allow-list, retired routes actually gone,
dead links, missing images, required report sections, and private-token leakage.

This is the gate run after `site/build.py` and before commit. Several checks here exist
because an independent review found the first cutover passed a weaker checker while
violating the contract anyway: a dated Weekly archive page was deleted while a Private
receipt still pointed at its URL, and the W36 report shipped without the four-week OHLC
section the spec requires. Both are now failures, not observations.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import POC_STUDIES, QUEUED_STUDIES, SUPERSEDED_STUDIES, routes, weekly_weeks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GENERATED_PAGES = [ROOT / (r + "/index.html" if r else "index.html") for r in routes()]

# Routes this rebuild retires. Presence after cutover is a failure.
RETIRED_PAGES = [
    ROOT / "jargon/index.html", ROOT / "lessons/index.html", ROOT / "tx/index.html",
    ROOT / "xauusd/index.html", ROOT / "v1/index.html", ROOT / "zh/index.html",
]

PRIVATE_TOKENS = (
    "/users/", "googledrive", "reports/xauusd/weekly", "state/xauusd", "data/xauusd",
    "journal_locator", "input_set_id", "ledger_run_id", ".docx", ".gdoc",
    "private_provenance", "owner",
)

EXPECTED_CHARTS = {
    "RS-XAUUSD-20260727-001": 20,
    "RS-XAUUSD-20260727-005": 5,
    "RS-XAUUSD-20260818-001": 6,
}


def check_routes(errors: list[str]) -> None:
    for page in GENERATED_PAGES:
        if not page.is_file():
            errors.append(f"missing canonical page: {page.relative_to(ROOT)}")
    # Every published Weekly edition must have a dated archive page: Private receipts
    # record those URLs as publication proof, so a missing one breaks a signed record.
    for week in weekly_weeks():
        page = ROOT / "xauusd/weekly" / week / "index.html"
        if not page.is_file():
            errors.append(f"published Weekly edition {week} has no archive page (breaks its receipt URL)")


def check_retired(errors: list[str]) -> None:
    for page in RETIRED_PAGES:
        if page.is_file():
            errors.append(f"retired route still present: {page.relative_to(ROOT)}")
    for sid in QUEUED_STUDIES:
        page = ROOT / "research/studies" / sid / "index.html"
        if page.is_file():
            errors.append(f"queued study must stay unlinked/404 this phase: {sid}")
    for sid in QUEUED_STUDIES + SUPERSEDED_STUDIES:
        # Evidence must survive even while the page does not. A superseded study is never
        # getting a page, which makes its evidence package the only remaining record.
        for required in ("study.json", "results.json"):
            if not (ROOT / "research/studies" / sid / required).is_file():
                errors.append(f"unpublished study lost its evidence package: {sid}/{required}")
    for sid in SUPERSEDED_STUDIES:
        page = ROOT / "research/studies" / sid / "index.html"
        if page.is_file():
            errors.append(f"superseded study must not have a page: {sid}")


def check_links_and_images(errors: list[str]) -> None:
    for page in GENERATED_PAGES:
        text = page.read_text(encoding="utf-8")
        for link in re.findall(r'(?:href|src)="([^"]+)"', text):
            if link.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            target = (page.parent / unquote(urlsplit(link).path)).resolve()
            if not target.exists():
                errors.append(f"broken link in {page.relative_to(ROOT)}: {link}")
    for sid, expected in EXPECTED_CHARTS.items():
        results = json.loads((ROOT / "research/studies" / sid / "results.json").read_text(encoding="utf-8"))
        charts = results.get("charts", [])
        if len(charts) != expected:
            errors.append(f"{sid}: expected {expected} charts, results.json declares {len(charts)}")
        for chart in charts:
            if not (ROOT / "research/studies" / sid / "charts" / chart["file"]).is_file():
                errors.append(f"missing chart image {sid}/{chart['file']}")


def check_weekly_sections(errors: list[str]) -> None:
    """A Weekly report must actually contain its required reader-facing sections."""
    required = ["市場摘要", "三劇本與機率", "關鍵價位", "事件風險", "共識", "分歧"]
    for week in weekly_weeks():
        page = ROOT / "xauusd/weekly" / week / "index.html"
        if not page.is_file():
            continue
        text = page.read_text(encoding="utf-8")
        for section in required:
            if section not in text:
                errors.append(f"{week} report is missing required section: {section}")
        summary = json.loads((page.parent / "summary.json").read_text(encoding="utf-8"))
        # four_week_overview became mandatory for editions published after the contract
        # change; older finalised artifacts are not retroactively invalidated.
        if summary.get("four_week_overview") and "四週回顧" not in text:
            errors.append(f"{week} has four_week_overview data but the page does not render it")


# The owner's account size. It reached the live site inside a published analysis.py even
# though that study's whole public form is normalised per unit of capital precisely so the
# account would not travel — the export rule only appended a comment beside the figure
# instead of replacing it. A scan is cheaper than remembering.
ACCOUNT_FIGURES = ("30,000", "30_000", "$30000")


def check_account_figures(errors: list[str]) -> None:
    for study_dir in (ROOT / "research/studies").iterdir():
        if not study_dir.is_dir():
            continue
        for path in study_dir.glob("*"):
            if path.suffix not in {".py", ".json", ".md", ".html"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in ACCOUNT_FIGURES:
                if token in text:
                    errors.append(
                        f"account size '{token}' present in {path.relative_to(ROOT)}; "
                        f"published research must state capital per unit, not the real account"
                    )


def check_privacy(errors: list[str]) -> None:
    scanned = list(GENERATED_PAGES)
    scanned += [ROOT / "xauusd/weekly" / w / "summary.json" for w in weekly_weeks()]
    # Pages serves the whole repository, so a queued or superseded study's evidence package
    # is as public as a published one — scan every package present, not just POC_STUDIES.
    # impact.md was missing from this list entirely, which is how "the owner" reached five
    # published packages: it is the one exported text that skipped the rewrite pass.
    for d in sorted((ROOT / "research/studies").iterdir()):
        if d.is_dir():
            scanned += [d / "study.json", d / "results.json", d / "analysis.py", d / "impact.md"]
    for path in scanned:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in PRIVATE_TOKENS:
            if token in text:
                errors.append(f"prohibited token '{token}' in {path.relative_to(ROOT)}")


def check_null_results(errors: list[str]) -> None:
    registry = json.loads((ROOT / "research/null-results/null_results.json").read_text(encoding="utf-8"))
    totals = registry.get("totals", {})
    hypotheses = [e for e in registry.get("entries", []) if e.get("kind") == "hypothesis"]
    if len(hypotheses) != 63 or totals.get("hypotheses") != 63:
        errors.append(f"expected 63 hypotheses, found {len(hypotheses)}")
    expected = {"no_evidence": 60, "underpowered": 2, "below_cost": 1}
    if totals.get("by_verdict") != expected:
        errors.append(f"verdict totals mismatch: expected {expected}, got {totals.get('by_verdict')}")


def check_tables(errors: list[str]) -> None:
    """Every table must be sortable and width-adaptive — a per-page regression here is
    exactly the kind of thing that silently degrades reading on a wide screen."""
    for page in GENERATED_PAGES:
        text = page.read_text(encoding="utf-8")
        tables = text.count("<table")
        sortable = text.count("<table data-sortable>")
        if tables != sortable:
            errors.append(f"{page.relative_to(ROOT)}: {tables - sortable} table(s) not sortable")
        if tables and text.count('class="table-wrap"') < tables:
            errors.append(f"{page.relative_to(ROOT)}: a table is not inside a width-adaptive wrapper")


def check_presentation_blocks(errors: list[str]) -> None:
    """Every declared presentation block must actually render.

    render_block returns "" when a block's source is absent, so a study whose evidence
    field was not whitelisted for Public export loses that whole section with no error
    anywhere — the page just quietly says less than it claims to. This happened to
    RS-XAUUSD-20260825-001, whose core evidence table vanished because the field existed
    in the Private results.json the blueprint was written against but not in the exported
    Public one. Validating against the Public data the page is actually built from is the
    only check that would have caught it.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build as gen
    for sid in POC_STUDIES:
        study, results = gen.load_study(sid)
        for index, block in enumerate(study.get("presentation", []), start=1):
            rendered = gen.render_block(block, study, results)
            if not rendered.strip():
                label = block.get("title") or block.get("source") or block["type"]
                errors.append(
                    f"{sid}: presentation block {index} ({block['type']}: {label}) renders "
                    f"nothing — its source is missing from the published data"
                )


def check_study_order(errors: list[str]) -> None:
    """Interpretation and limitations belong before the evidence links, not after."""
    for sid in POC_STUDIES:
        text = (ROOT / "research/studies" / sid / "index.html").read_text(encoding="utf-8")
        ev = text.find("Evidence 原始證據")
        interp = text.find("詮釋與實務意義")
        lim = text.find("限制與注意事項")
        if ev == -1:
            errors.append(f"{sid}: evidence links section missing")
            continue
        if interp != -1 and interp > ev:
            errors.append(f"{sid}: 詮釋與實務意義 renders after the evidence links")
        if lim != -1 and lim > ev:
            errors.append(f"{sid}: 限制與注意事項 renders after the evidence links")
        charts = text.find("Charts 圖表")
        first_table = text.find('<section class="data-block">')
        if charts != -1 and first_table != -1 and charts > first_table:
            errors.append(f"{sid}: charts render after the detailed tables")


def main() -> int:
    errors: list[str] = []
    for fn in (check_routes, check_retired, check_links_and_images, check_weekly_sections,
               check_privacy, check_account_figures, check_null_results, check_tables, check_presentation_blocks,
               check_study_order):
        fn(errors)
    print(json.dumps({
        "routes checked": len(GENERATED_PAGES),
        "weekly editions": len(weekly_weeks()),
        "poc chart total": sum(EXPECTED_CHARTS.values()),
        "queued studies": len(QUEUED_STUDIES),
        "superseded studies": len(SUPERSEDED_STUDIES),
        "failures": len(errors),
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
