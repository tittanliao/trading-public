#!/usr/bin/env python3
"""Build deterministic landing pages from reviewed Public repository contents."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "04cd05734e6905561e113945948e848e106d26bb"
GENERATED = {
    Path("index.html"),
    Path("xauusd/index.html"),
    Path("tx/index.html"),
    Path("research/index.html"),
    Path("site/catalog.json"),
}
EXCLUDED_PARTS = {".git", ".github", "legacy-site", "site", "__pycache__"}
STUDY_ROOT = ROOT / "research/studies"


def title_for(path: Path) -> str:
    if path.suffix.lower() == ".html":
        text = path.read_text(encoding="utf-8", errors="ignore")[:100_000]
        match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        if match:
            return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()
    return path.stem.replace("_", " ").replace("-", " ").strip()


def section_for(relative: Path) -> str:
    if relative.parts and relative.parts[0] == "xauusd":
        return "xauusd"
    if relative.parts and relative.parts[0] == "tx":
        return "tx"
    return "research"


def catalog() -> dict[str, object]:
    items: list[dict[str, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        generated_study_page = (
            len(relative.parts) == 4
            and relative.parts[:2] == ("research", "studies")
            and relative.name == "index.html"
        )
        if relative in GENERATED or generated_study_page or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        extension = path.suffix.lower()
        if extension not in {".html", ".pine", ".py", ".json"}:
            continue
        if extension == ".html":
            kind = "report"
        elif extension == ".pine":
            kind = "pine"
        elif extension == ".py":
            kind = "python"
        else:
            kind = "result"
        items.append(
            {
                "path": relative.as_posix(),
                "title": title_for(path),
                "section": section_for(relative),
                "kind": kind,
            }
        )
    return {"schema_version": 1, "legacy_source_commit": SOURCE_COMMIT, "items": items}


def nav(prefix: str) -> str:
    return (
        '<nav class="nav">'
        f'<a href="{prefix}index.html">Overview</a>'
        f'<a href="{prefix}xauusd/">XAUUSD</a>'
        f'<a href="{prefix}tx/">TX</a>'
        f'<a href="{prefix}research/">Research files</a>'
        '</nav>'
    )


def document(title: str, eyebrow: str, lede: str, body: str, prefix: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="{html.escape(lede)}">
  <title>{html.escape(title)} · Trading Research</title>
  <link rel="stylesheet" href="{prefix}site/style.css">
</head>
<body>
  <header class="shell">
    <div class="eyebrow">{html.escape(eyebrow)}</div>
    <h1>{html.escape(title)}</h1>
    <p class="lede">{html.escape(lede)}</p>
    {nav(prefix)}
  </header>
  {body}
  <footer><div class="shell">Generated from reviewed repository contents. Research evidence, not trading advice.</div></footer>
  <script src="{prefix}site/app.js"></script>
</body>
</html>
"""


def card(item: dict[str, str], prefix: str) -> str:
    path = html.escape(prefix + item["path"])
    return (
        f'<a class="card" data-card href="{path}">'
        f'<div class="type">{html.escape(item["kind"])}</div>'
        f'<h3>{html.escape(item["title"])}</h3>'
        f'<p>{html.escape(item["path"])}</p></a>'
    )


def studies() -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if not STUDY_ROOT.is_dir():
        return found
    for manifest in sorted(STUDY_ROOT.glob("*/study.json"), reverse=True):
        study = json.loads(manifest.read_text(encoding="utf-8"))
        result_path = manifest.parent / "results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"study missing results.json: {manifest.parent}")
        study["_result"] = json.loads(result_path.read_text(encoding="utf-8"))
        study["_relative"] = manifest.parent.relative_to(ROOT).as_posix()
        found.append(study)
    return found


def study_card(study: dict[str, object], prefix: str = "../") -> str:
    headline = study["headline"]
    return (
        f'<a class="card study-card" data-card href="{html.escape(prefix + study["_relative"])}/">'
        f'<div class="type">{html.escape(study["status"])} · {html.escape(study["id"])}</div>'
        f'<h2>{html.escape(study["title"])}</h2>'
        f'<p>{html.escape(study["question"])}</p>'
        '<div class="mini-metrics">'
        f'<span><strong>{headline["trades"]}</strong> trades</span>'
        f'<span><strong>{headline["win_rate_pct"]}%</strong> WR</span>'
        f'<span><strong>{headline["profit_factor"]}</strong> PF</span>'
        '</div></a>'
    )


def metric(label: str, value: object, detail: str = "") -> str:
    return (
        '<div class="metric"><div class="metric-label">'
        f'{html.escape(label)}</div><div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div></div>'
    )


def result_table(title: str, rows: dict[str, dict[str, object]]) -> str:
    body = "".join(
        "<tr>"
        f"<td><strong>{html.escape(name)}</strong></td>"
        f"<td>{value['n']}</td><td>{value['win_rate_pct']}%</td>"
        f"<td>{value['profit_factor']}</td><td>{value['net_pnl_usd']:,.2f}</td>"
        f"<td>{value.get('advisory_delta', '—')}</td>"
        "</tr>"
        for name, value in rows.items()
    )
    return (
        f'<section class="report-section"><h2>{html.escape(title)}</h2>'
        '<div class="table-wrap"><table><thead><tr><th>Context</th><th>n</th>'
        '<th>WR</th><th>PF</th><th>Net USD</th><th>Score Δ</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div></section>"
    )


def study_page(study: dict[str, object]) -> str:
    result = study["_result"]
    baseline = result["baseline"]
    finding_html = "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )
    impacts = "".join(
        f'<li><strong>{html.escape(item["surface"])}</strong> · {html.escape(item["summary"])}</li>'
        for item in study["policy_impacts"]
    )
    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric("Closed trades", baseline["n"], "TradingView V3.9 export")
        + metric("Win rate", f'{baseline["win_rate_pct"]}%', f'95% CI {baseline["win_rate_ci95_pct"][0]}–{baseline["win_rate_ci95_pct"][1]}%')
        + metric("Profit factor", baseline["profit_factor"], f'Net ${baseline["net_pnl_usd"]:,.2f}')
        + metric("Macro coverage", f'{result["macro_coverage"]["pct"]}%', f'{result["macro_coverage"]["matched"]}/{baseline["n"]} trades')
        + '</div>'
        '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + result_table("Session context", result["by_session"])
        + result_table("Macro context", result["by_macro_verdict"])
        + '<section class="report-section"><h2>Impact on 請分析</h2>'
        f'<ul class="impact-list">{impacts}</ul>'
        '<p class="callout">Macro and session are advisory score adjustments. They are not entry permissions and cannot create or cancel a formal V3.9 signal.</p></section>'
        '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        '<p>Raw CSV and private decision conversations remain in trading-private. This public page contains reviewed aggregate results and the reproducible method only.</p>'
        '<div class="file-actions">'
        '<a href="results.json">Structured results</a><a href="analysis.py">Python method</a><a href="study.json">Study manifest</a>'
        '</div></section></main>'
    )
    return document(
        study["title"],
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        study["question"],
        body,
        "../../../",
    )


def section_page(data: dict[str, object], section: str, title: str, lede: str) -> str:
    items = [item for item in data["items"] if item["section"] == section]
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter by title or path" aria-label="Filter"></div></div>'
        f'<main class="shell"><div class="stats"><span class="stat"><strong>{len(items)}</strong> indexed files</span></div>'
        f'<div class="grid">{"".join(card(item, "../") for item in items)}</div></main>'
    )
    return document(title, section, lede, body, "../")


def research_page(data: dict[str, object], study_list: list[dict[str, object]]) -> str:
    items = [item for item in data["items"] if item["section"] == "research"]
    study_cards = "".join(study_card(study) for study in study_list)
    file_cards = "".join(card(item, "../") for item in items)
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter studies, reports, or code" aria-label="Filter"></div></div>'
        '<main class="shell"><h2 class="section-title">Adopted studies</h2>'
        f'<div class="grid study-grid">{study_cards}</div>'
        '<h2 class="section-title">Research files</h2>'
        f'<div class="grid">{file_cards}</div></main>'
    )
    return document(
        "Research studies",
        "Evidence → decision → workflow",
        "Reviewed studies preserve the question, reproducible method, aggregate result, and operational impact without publishing raw CSV or private conversation.",
        body,
        "../",
    )


def overview(data: dict[str, object], study_list: list[dict[str, object]]) -> str:
    items = data["items"]
    counts = {
        section: sum(item["section"] == section for item in items)
        for section in ("xauusd", "tx", "research")
    }
    cards = [
        ("xauusd/", "XAUUSD", "Gold strategy reports, Pine sources, and supporting research.", counts["xauusd"]),
        ("tx/", "TX", "Taiwan index futures experiments, reports, and code.", counts["tx"]),
        ("research/", "Research files", "Cross-market utilities, result JSON, and reproducible scripts.", counts["research"]),
        ("legacy-site/index.html", "Legacy site snapshot", "Archived mixed landing page for comparison; not the active workflow.", 6),
    ]
    collections = '<div class="grid">' + "".join(
        f'<a class="card" href="{href}"><div class="type">collection</div><h2>{title}</h2><p>{description} · {count} items</p></a>'
        for href, title, description, count in cards
    ) + '</div>'
    latest = "".join(study_card(study, "") for study in study_list[:3])
    body = (
        '<main class="shell">'
        '<h2 class="section-title">Research collections</h2>'
        f'{collections}'
        '<h2 class="section-title">Latest adopted studies</h2>'
        f'<div class="grid study-grid">{latest}</div>'
        '<section class="report-section"><h2>Evidence lifecycle</h2>'
        '<div class="pipeline"><span>CSV + hash</span><b>→</b><span>Python runner</span><b>→</b>'
        '<span>Structured result</span><b>→</b><span>Owner decision</span><b>→</b>'
        '<span>請分析 policy</span></div></section>'
        '</main>'
    )
    return document(
        "Trading research, without the maze.",
        "Reviewed public workspace",
        "A generated portal for published reports, code, and strategy evidence. Raw data and private model memory stay outside this repository.",
        body,
    )


def outputs(data: dict[str, object]) -> dict[Path, str]:
    study_list = studies()
    generated = {
        ROOT / "site/catalog.json": json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        ROOT / "index.html": overview(data, study_list),
        ROOT / "xauusd/index.html": section_page(data, "xauusd", "XAUUSD research", "Published gold strategy reports, Pine sources, and supporting evidence."),
        ROOT / "tx/index.html": section_page(data, "tx", "TX research", "Published Taiwan index futures experiments, reports, and code."),
        ROOT / "research/index.html": research_page(data, study_list),
    }
    for study in study_list:
        generated[ROOT / study["_relative"] / "index.html"] = study_page(study)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = outputs(catalog())
    failures: list[str] = []
    for path, expected in generated.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                failures.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
    print(f"generated files: {len(generated)}")
    print(f"drift: {len(failures)}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
