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
EXCLUDED_PARTS = {".git", ".github", "_retire", "site", "__pycache__"}
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
        f'<a href="{prefix}research/">Research</a>'
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


HEADLINE_LABELS = {
    "trades": "trades",
    "win_rate_pct": "WR",
    "profit_factor": "PF",
    "macro_coverage_pct": "macro cov.",
    "entry_slots_30m": "30m slots",
    "v34_trades": "V3.4 n",
    "v34_win_rate_pct": "V3.4 WR",
    "v34_profit_factor": "V3.4 PF",
    "v39_trades": "V3.9 n",
    "v39_win_rate_pct": "V3.9 WR",
    "v39_profit_factor": "V3.9 PF",
}


def headline_label(key: str) -> str:
    return HEADLINE_LABELS.get(key, key.replace("_", " "))


def headline_display(key: str, value: object) -> str:
    if isinstance(value, (int, float)) and key.endswith("_pct"):
        return f"{value}%"
    return str(value)


def study_card(study: dict[str, object], prefix: str = "../") -> str:
    headline = study["headline"]
    # A study may curate its own card_metrics (ordered headline keys); otherwise the
    # first three headline fields are shown, which matches every current single-version
    # study without requiring per-study special-casing here.
    keys = study.get("card_metrics") or list(headline)[:3]
    metrics_html = "".join(
        f'<span><strong>{html.escape(headline_display(key, headline[key]))}</strong> '
        f'{html.escape(headline_label(key))}</span>'
        for key in keys
        if key in headline
    )
    return (
        f'<a class="card study-card" data-card href="{html.escape(prefix + study["_relative"])}/">'
        f'<div class="type">{html.escape(study["status"])} · {html.escape(study["id"])}</div>'
        f'<h2>{html.escape(study["title"])}</h2>'
        f'<p>{html.escape(study["question"])}</p>'
        f'<div class="mini-metrics">{metrics_html}</div></a>'
    )


def metric(label: str, value: object, detail: str = "") -> str:
    return (
        '<div class="metric"><div class="metric-label">'
        f'{html.escape(label)}</div><div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div></div>'
    )


def result_table(
    title: str,
    rows: dict[str, dict[str, object]],
    *,
    show_adjustment: bool = True,
    note: str = "",
) -> str:
    body = "".join(
        (
            "<tr>"
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{value['n']}</td><td>{value['win_rate_pct']}%</td>"
            f"<td>{value['profit_factor']}</td><td>{value['net_pnl_usd']:,.2f}</td>"
        )
        + (
            f"<td>{value.get('advisory_delta', '—')}</td>"
            if show_adjustment
            else ""
        )
        + "</tr>"
        for name, value in rows.items()
    )
    adjustment_header = "<th>Score Δ</th>" if show_adjustment else ""
    note_html = f'<p class="section-note">{html.escape(note)}</p>' if note else ""
    return (
        f'<section class="report-section"><h2>{html.escape(title)}</h2>'
        f"{note_html}"
        '<div class="table-wrap"><table><thead><tr><th>Context</th><th>n</th>'
        f'<th>WR</th><th>PF</th><th>Net USD</th>{adjustment_header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div></section>"
    )


def value_or_dash(value: object, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


def entry_slot_table(rows: dict[str, dict[str, object]]) -> str:
    body = ""
    for slot, value in rows.items():
        interval = value.get("win_rate_ci95_pct")
        interval_text = (
            "—" if interval is None else f"{interval[0]}–{interval[1]}%"
        )
        delta = value.get("advisory_delta")
        delta_text = "—" if delta is None else f"{delta:+.1f}"
        delta_class = "score-neutral"
        if isinstance(delta, (int, float)) and delta > 0:
            delta_class = "score-positive"
        elif isinstance(delta, (int, float)) and delta < 0:
            delta_class = "score-negative"
        body += (
            "<tr>"
            f"<td><strong>{html.escape(slot)}</strong></td>"
            f"<td>{value['n']}</td>"
            f"<td>{value_or_dash(value.get('win_rate_pct'), '%')}</td>"
            f"<td>{interval_text}</td>"
            f"<td>{value_or_dash(value.get('profit_factor'))}</td>"
            f"<td>{value['net_pnl_usd']:,.2f}</td>"
            f'<td class="{delta_class}">{delta_text}</td>'
            "</tr>"
        )
    return (
        '<section class="report-section"><h2>30-minute entry timing</h2>'
        '<p class="section-note">Asia/Taipei bar-start time. All 48 slots are shown; '
        'low-n cells have wide uncertainty. Score deltas use a 30-trade prior and are '
        'capped at ±4 points, never as entry gates.</p>'
        '<div class="table-wrap tall-table"><table><thead><tr><th>30m slot</th><th>n</th>'
        '<th>WR</th><th>95% CI</th><th>PF</th><th>Net USD</th><th>Score Δ</th></tr>'
        f"</thead><tbody>{body}</tbody></table></div></section>"
    )


def comparison_entry_slot_table(comparison: dict[str, dict[str, object]]) -> str:
    body = ""
    for slot, item in comparison.items():
        diff = item.get("win_rate_pct_diff_v39_minus_v34")
        diff_text = "—" if diff is None else f"{diff:+.2f}pp"
        diff_class = "score-neutral"
        if isinstance(diff, (int, float)) and diff > 0:
            diff_class = "score-positive"
        elif isinstance(diff, (int, float)) and diff < 0:
            diff_class = "score-negative"
        body += (
            "<tr>"
            f"<td><strong>{html.escape(slot)}</strong></td>"
            f"<td>{item['v34_n']}</td>"
            f"<td>{value_or_dash(item.get('v34_win_rate_pct'), '%')}</td>"
            f"<td>{value_or_dash(item.get('v34_profit_factor'))}</td>"
            f"<td>{item['v39_n']}</td>"
            f"<td>{value_or_dash(item.get('v39_win_rate_pct'), '%')}</td>"
            f"<td>{value_or_dash(item.get('v39_profit_factor'))}</td>"
            f'<td class="{diff_class}">{diff_text}</td>'
            "</tr>"
        )
    return (
        '<section class="report-section"><h2>30-minute entry-slot comparison</h2>'
        '<p class="section-note">Asia/Taipei bar-start time. Both versions computed with the '
        "same deterministic method; most cells are low-n for both versions and differences "
        "should be read as descriptive, not as a stable timing edge.</p>"
        '<div class="table-wrap tall-table"><table><thead><tr><th>30m slot</th>'
        "<th>V3.4 n</th><th>V3.4 WR</th><th>V3.4 PF</th>"
        "<th>V3.9 n</th><th>V3.9 WR</th><th>V3.9 PF</th>"
        "<th>WR diff (V3.9−V3.4)</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div></section>"
    )


def study_page_comparison(study: dict[str, object]) -> str:
    result = study["_result"]
    versions = result["versions"]
    finding_html = "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )
    impacts = study.get("policy_impacts") or []
    impact_html = (
        "".join(
            f'<li><strong>{html.escape(item["surface"])}</strong> · {html.escape(item["summary"])}</li>'
            for item in impacts
        )
        if impacts
        else '<li class="empty">No active 請分析 policy change. Research/publication only for this study.</li>'
    )
    metric_html = "".join(
        metric(
            f'{version} n / WR / PF',
            f'{data["baseline"]["n"]} / {data["baseline"]["win_rate_pct"]}% / {data["baseline"]["profit_factor"]}',
            f'Net ${data["baseline"]["net_pnl_usd"]:,.2f}',
        )
        for version, data in versions.items()
    )
    body = (
        '<main class="shell report">'
        f'<div class="metric-grid">{metric_html}</div>'
        '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + "".join(
            result_table(
                f"{version} session summary",
                data["by_session"],
                show_adjustment=False,
                note="Descriptive only. See the 30-minute entry-slot comparison below for the primary evidence.",
            )
            for version, data in versions.items()
        )
        + comparison_entry_slot_table(result["comparison"]["by_entry_30m"])
        + "".join(
            result_table(f"{version} Macro context", data["by_macro_verdict"])
            for version, data in versions.items()
        )
        + '<section class="report-section"><h2>Impact on 請分析</h2>'
        f'<ul class="impact-list">{impact_html}</ul></section>'
        '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        "<p>Raw CSV and private decision conversations remain in trading-private. This public page "
        "contains reviewed aggregate results and the reproducible method only.</p>"
        '<div class="file-actions">'
        '<a href="results.json">Structured results</a><a href="analysis.py">Python method</a>'
        '<a href="study.json">Study manifest</a>'
        "</div></section></main>"
    )
    return document(
        study["title"],
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        study["question"],
        body,
        "../../../",
    )


def study_page(study: dict[str, object]) -> str:
    result = study["_result"]
    if "versions" in result:
        return study_page_comparison(study)
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
        + result_table(
            "Broad session summary",
            result["by_session"],
            show_adjustment=False,
            note="Descriptive only. Operational scoring uses the matching 30-minute slot below, not Asia/Europe/US labels.",
        )
        + entry_slot_table(result["by_entry_30m"])
        + result_table("Macro context", result["by_macro_verdict"])
        + '<section class="report-section"><h2>Impact on 請分析</h2>'
        f'<ul class="impact-list">{impacts}</ul>'
        '<p class="callout">Macro and the matching 30-minute slot are advisory score adjustments. They are not entry permissions and cannot create or cancel a formal V3.9 signal.</p></section>'
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


def section_page(
    study_list: list[dict[str, object]],
    market: str,
    title: str,
    lede: str,
) -> str:
    selected = [study for study in study_list if study["market"].lower() == market]
    cards = "".join(study_card(study) for study in selected)
    if not cards:
        cards = '<p class="empty">No active published study yet.</p>'
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter studies" aria-label="Filter"></div></div>'
        f'<main class="shell"><div class="stats"><span class="stat"><strong>{len(selected)}</strong> active studies</span></div>'
        f'<div class="grid study-grid">{cards}</div></main>'
    )
    return document(title, market, lede, body, "../")


def research_page(data: dict[str, object], study_list: list[dict[str, object]]) -> str:
    study_cards = "".join(study_card(study) for study in study_list)
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter studies" aria-label="Filter"></div></div>'
        '<main class="shell"><h2 class="section-title">Adopted studies</h2>'
        f'<div class="grid study-grid">{study_cards}</div></main>'
    )
    return document(
        "Research studies",
        "Evidence → decision → workflow",
        "Reviewed studies preserve the question, reproducible method, aggregate result, and operational impact without publishing raw CSV or private conversation.",
        body,
        "../",
    )


def overview(data: dict[str, object], study_list: list[dict[str, object]]) -> str:
    latest_xauusd = next(
        (study for study in study_list if study["market"].lower() == "xauusd"),
        None,
    )
    xauusd_href = (
        f'{latest_xauusd["_relative"]}/' if latest_xauusd else "xauusd/"
    )
    cards = [
        (xauusd_href, "XAUUSD Macro / timing analysis", "Open the latest human-readable S1 V3.9 study directly."),
        ("tx/", "TX studies", "Active reviewed Taiwan index futures studies."),
        ("research/", "All research", "Browse adopted human-readable study reports."),
    ]
    collections = '<div class="grid">' + "".join(
        f'<a class="card" href="{href}"><div class="type">analysis</div><h2>{title}</h2><p>{description}</p></a>'
        for href, title, description in cards
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
        ROOT / "xauusd/index.html": section_page(study_list, "xauusd", "XAUUSD studies", "Reviewed gold research presented as readable studies, not a repository tree."),
        ROOT / "tx/index.html": section_page(study_list, "tx", "TX studies", "Reviewed Taiwan index futures research presented as readable studies."),
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
