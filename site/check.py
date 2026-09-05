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
from build import PUBLISHED_STUDIES, QUEUED_STUDIES, SUPERSEDED_STUDIES, routes, weekly_weeks  # noqa: E402

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


def check_backlog(errors: list[str]) -> None:
    """The backlog is only useful while its cross-references hold.

    Its whole value is the "already known" block on each item, so a reference that no longer
    resolves is worse than no reference: it tells a reader evidence exists where it does not.
    An item marked answered must name the study that answered it, for the same reason.
    """
    path = ROOT / "research/backlog/backlog.json"
    if not path.is_file():
        errors.append("research/backlog/backlog.json is missing")
        return
    backlog = json.loads(path.read_text(encoding="utf-8"))
    valid = {"open", "running", "answered", "closed"}
    live = set(PUBLISHED_STUDIES)
    seen: set[str] = set()
    for item in backlog.get("items", []):
        bid = item.get("id", "?")
        if bid in seen:
            errors.append(f"backlog {bid}: duplicate id")
        seen.add(bid)
        if item.get("status") not in valid:
            errors.append(f"backlog {bid}: unknown status {item.get('status')!r}")
        if item.get("status") == "answered" and not item.get("answered_by"):
            errors.append(f"backlog {bid}: answered but names no study")
        references = [r["study_id"] for r in item.get("prior_evidence", [])]
        if item.get("answered_by"):
            references.append(item["answered_by"])
        for sid in references:
            if sid not in live:
                errors.append(
                    f"backlog {bid}: cites {sid}, which has no published page to link to"
                )


def check_charts(errors: list[str]) -> None:
    """The chart contract, per the owner decision of 2026-08-31.

    A reader page shows a selected set of charts, each immediately followed by the table it
    explains — not every chart a study generated. So the rule is no longer "every declared
    chart must render". What must hold instead:

    - every chart a page shows is declared in that study's results.json;
    - the PNG behind every declared chart exists, whether or not the page shows it, because
      the evidence package is public and linked from the Evidence section;
    - no page shows the same chart twice, which is a blueprint copy-paste slip that renders
      cleanly and silently pads the page.
    """
    for sid in PUBLISHED_STUDIES:
        package = ROOT / "research/studies" / sid
        results = json.loads((package / "results.json").read_text(encoding="utf-8"))
        study = json.loads((package / "study.json").read_text(encoding="utf-8"))
        declared = {chart["file"] for chart in results.get("charts", [])}

        for chart in results.get("charts", []):
            if not (package / "charts" / chart["file"]).is_file():
                errors.append(f"{sid}: declared chart has no image: charts/{chart['file']}")

        shown: list[str] = []
        for index, block in enumerate(study.get("presentation", []), start=1):
            if block.get("type") != "evidence_pair":
                continue
            name = block.get("chart_file", "")
            if name not in declared:
                errors.append(
                    f"{sid}: presentation block {index} shows '{name}', which results.json "
                    "does not declare"
                )
            elif name in shown:
                errors.append(f"{sid}: presentation block {index} shows '{name}' twice")
            shown.append(name)


def check_weekly_sections(errors: list[str]) -> None:
    """A Weekly report must actually contain its required reader-facing sections."""
    required = ["市場摘要", "三劇本與機率", "關鍵價位", "事件風險", "分歧"]
    for week in weekly_weeks():
        page = ROOT / "xauusd/weekly" / week / "index.html"
        if not page.is_file():
            continue
        text = page.read_text(encoding="utf-8")
        summary = json.loads((page.parent / "summary.json").read_text(encoding="utf-8"))
        mode_heading = "共識" if summary.get("publication_mode") == "multi_source" else "可驗證事項"
        for section in required:
            if section not in text:
                errors.append(f"{week} report is missing required section: {section}")
        if mode_heading not in text:
            errors.append(f"{week} report is missing required section: {mode_heading}")
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
    # is as public as a published one — scan every package present, not just PUBLISHED_STUDIES.
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
    # 2026-09-05 independent audit finding 2: the two checks above are fixed constants, so
    # they stay green even when the file is stale relative to the studies actually
    # published here — which it was (39/102 entries here vs Private's live 46/109) for
    # some number of days with nothing catching it. This can only be checked from inside
    # Public, so it cross-references what Public itself already has: every published study
    # directory must have a `kind: study` entry in the null registry, and that entry's
    # `status` must match the study's own `study.json` — the exact drift that let two
    # superseded studies keep reading `confirmed` here after they were downgraded.
    by_id = {e["study_id"]: e for e in registry.get("entries", []) if e.get("kind") == "study"}
    for study_dir in sorted((ROOT / "research/studies").iterdir()):
        if not study_dir.is_dir():
            continue
        sid = study_dir.name
        study_path = study_dir / "study.json"
        if not study_path.is_file():
            continue
        entry = by_id.get(sid)
        if entry is None:
            errors.append(f"{sid}: published here but has no entry in null_results.json — "
                          f"export_null_registry.py needs a rerun")
            continue
        live_status = json.loads(study_path.read_text(encoding="utf-8")).get("status")
        if entry.get("status") != live_status:
            errors.append(f"{sid}: null_results.json status {entry.get('status')!r} does "
                          f"not match its own study.json status {live_status!r}")


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
    for sid in PUBLISHED_STUDIES:
        study, results = gen.load_study(sid)
        for index, block in enumerate(study.get("presentation", []), start=1):
            rendered = gen.render_block(block, study, results)
            if not rendered.strip():
                label = block.get("title") or block.get("source") or block["type"]
                errors.append(
                    f"{sid}: presentation block {index} ({block['type']}: {label}) renders "
                    f"nothing — its source is missing from the published data"
                )


def check_table_columns(errors: list[str]) -> None:
    """A column a blueprint asks for but the data does not have is dropped silently, so the
    table still renders and every other check passes — minus the evidence it was built to
    show. Every named column must resolve against the records it is applied to."""
    import build as gen
    for sid in PUBLISHED_STUDIES:
        study, results = gen.load_study(sid)
        for index, block in enumerate(study.get("presentation", []), start=1):
            columns = block.get("columns")
            source = block.get("table_source") if block["type"] == "evidence_pair" else block.get("source")
            if not columns or not source:
                continue
            value = gen.resolve_source(study, results, source)
            if not isinstance(value, (dict, list)):
                continue
            records = value.values() if isinstance(value, dict) else value
            available: set[str] = set()
            for record in records:
                if isinstance(record, dict):
                    available |= set(gen.flatten_record(record))
            missing = [c for c in columns if c not in available]
            if missing:
                errors.append(
                    f"{sid}: presentation block {index} ({source}) names columns absent "
                    f"from the data: {', '.join(missing)}"
                )


# Every stdlib/third-party top-level module name any published analysis.py currently
# imports. Extend this set, not the check, when a study legitimately needs a new one —
# the check's job is to catch a LOCAL name that resolves to nothing, and a name on this
# list is defined as never being that.
KNOWN_STDLIB_AND_THIRD_PARTY = {
    "__future__", "argparse", "base64", "bisect", "collections", "csv", "dataclasses",
    "datetime", "hashlib", "itertools", "json", "math", "matplotlib", "numpy", "os",
    "pandas", "pathlib", "random", "re", "shutil", "statistics", "sys", "typing",
    "zoneinfo",
}


def check_analysis_reproducibility(errors: list[str]) -> None:
    """Every published analysis.py must (a) parse, (b) write only to reproduced/ rather
    than overwriting its own package's results.json, and (c) import nothing that exists
    only in Private.

    2026-09-03 governance audit: 16 of 44 published analysis.py files did not parse at
    HEAD, because the exporter's path-rewrite regex could consume the newline after an
    output-path assignment and splice the next Python statement onto it. Nothing here had
    ever actually tried to parse the files this checker ships as "reproducible code" —
    the gate only scanned them for privacy tokens. That is necessary but not sufficient:
    a file can be free of private strings and still not be code.
    """
    import ast

    for sid in PUBLISHED_STUDIES:
        path = ROOT / "research/studies" / sid / "analysis.py"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{sid}: analysis.py does not parse ({exc.msg} at line {exc.lineno})")
            continue  # the checks below need a file that at least parses
        # The real invariant is narrower than "must write to reproduced/": a script that
        # writes no file at all (several print-only analyses, e.g. RS-XAUUSD-20260825-001)
        # carries zero overwrite risk and needs no reproduced/ path. What actually matters
        # is that IF the script writes a file, that file is not the package's own
        # results.json/report.html/charts — that was the real risk this check exists for.
        writes_any_file = bool(re.search(r"""\.write_text\(|\.write_bytes\(|open\([^)]*['"]w""", text))
        if writes_any_file and not re.search(r"""Path\(['"]reproduced""", text):
            errors.append(
                f"{sid}: analysis.py writes a file but not under reproduced/ — rerunning it "
                f"risks overwriting the package's own committed results.json"
            )
        # A live `import`/`from` statement is the bug; a docstring usage comment like
        # `python3.12 -m scripts.research.build_x` documenting the module's original
        # Private name is not — most hits here turned out to be exactly that, so the
        # check is scoped to statement position (start of line, ignoring indentation).
        if re.search(r"^\s*(?:import|from)\s+scripts\.research", text, re.MULTILINE):
            errors.append(
                f"{sid}: analysis.py imports a Private-only module (scripts.research...) "
                f"that does not exist in this repository"
            )
        # 2026-09-05 independent audit finding 1: the dotted-import check above only
        # catches one spelling. Ten published studies instead import a shared toolkit by
        # its BARE name (`import fail_pattern_toolkit`, `import screen_harness`, ...) —
        # the spelling the exporter's toolkit-inlining rewrite produces on success — but
        # were never actually re-exported, so the .py file the bare import needs was never
        # copied beside them. A parse-only check cannot see this: the file is syntactically
        # fine and the import is syntactically fine, it just resolves to nothing at
        # runtime. Walk the real AST (not a text regex — a docstring can read "from alpha
        # discovery..." and false-match a line-anchored regex) for every top-level
        # import/from, and require that any name which is neither a known stdlib/
        # third-party module nor a sibling file in this exact package directory be treated
        # as an error, not silently trusted.
        tree = ast.parse(text, filename=str(path))
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_names.add(node.module.split(".")[0])
        sibling_modules = {p.stem for p in path.parent.glob("*.py")}
        unresolved = imported_names - KNOWN_STDLIB_AND_THIRD_PARTY - sibling_modules
        # scripts.research.* itself is reported by the check above with a clearer message;
        # do not double-report it here under a generic "unresolved import" label.
        unresolved = {n for n in unresolved if n != "scripts"}
        if unresolved:
            errors.append(
                f"{sid}: analysis.py imports {sorted(unresolved)}, which resolve to "
                f"neither a known stdlib/third-party module nor a file in this package — "
                f"the toolkit was renamed on import but never copied beside it"
            )


def check_study_order(errors: list[str]) -> None:
    """Interpretation and limitations belong before the evidence links, not after."""
    for sid in PUBLISHED_STUDIES:
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
               check_backlog, check_charts, check_privacy, check_account_figures, check_null_results, check_tables, check_presentation_blocks,
               check_table_columns, check_analysis_reproducibility, check_study_order):
        fn(errors)
    print(json.dumps({
        "routes checked": len(GENERATED_PAGES),
        "weekly editions": len(weekly_weeks()),
        "backlog items": len(
            json.loads((ROOT / "research/backlog/backlog.json").read_text(encoding="utf-8"))["items"]
        ),
        "charts shown": sum(
            1 for sid in PUBLISHED_STUDIES
            for b in json.loads((ROOT / "research/studies" / sid / "study.json")
                                .read_text(encoding="utf-8")).get("presentation", [])
            if b.get("type") == "evidence_pair"
        ),
        "queued studies": len(QUEUED_STUDIES),
        "superseded studies": len(SUPERSEDED_STUDIES),
        "failures": len(errors),
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
