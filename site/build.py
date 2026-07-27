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
    badge = '<span class="badge-live">影響請分析</span>' if study.get("policy_impacts") else ""
    return (
        f'<a class="card study-card" data-card href="{html.escape(prefix + study["_relative"])}/">'
        f'<div class="type">{html.escape(study["status"])} · {html.escape(study["id"])}{badge}</div>'
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


def rank_score_cell(value: dict[str, object]) -> str:
    if value.get("low_sample"):
        return '<span class="score-neutral">0<sup class="low-sample-tag">low-n</sup></span>'
    score = value.get("rank_score")
    if score is None:
        return "—"
    css = "score-neutral"
    if score > 0:
        css = "score-positive"
    elif score < 0:
        css = "score-negative"
    return f'<span class="{css}">{score:+d}</span>'


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
            f"<td>{rank_score_cell(value)}</td>"
            if show_adjustment
            else ""
        )
        + "</tr>"
        for name, value in rows.items()
    )
    adjustment_header = "<th>Rank score</th>" if show_adjustment else ""
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
        body += (
            "<tr>"
            f"<td><strong>{html.escape(slot)}</strong></td>"
            f"<td>{value['n']}</td>"
            f"<td>{value_or_dash(value.get('win_rate_pct'), '%')}</td>"
            f"<td>{interval_text}</td>"
            f"<td>{value_or_dash(value.get('profit_factor'))}</td>"
            f"<td>{value['net_pnl_usd']:,.2f}</td>"
            f"<td>{rank_score_cell(value)}</td>"
            "</tr>"
        )
    return (
        '<section class="report-section"><h2>30-minute entry timing</h2>'
        '<p class="section-note">Asia/Taipei bar-start time. All 48 slots are shown; '
        'low-n cells (n&lt;5) score 0 and are marked low-n. Rank scores range −2…+2, '
        'never as entry gates.</p>'
        '<div class="table-wrap tall-table"><table><thead><tr><th>30m slot</th><th>n</th>'
        '<th>WR</th><th>95% CI</th><th>PF</th><th>Net USD</th><th>Rank score</th></tr>'
        f"</thead><tbody>{body}</tbody></table></div></section>"
    )


CHART_SECTION_LABELS = {
    "performance": "Performance",
    "fail_pattern": "Fail Pattern Breakdown",
    "timing_30m": "30-Minute Entry-Slot Timing",
    "pre_entry": "Pre-Entry Context — Immediate Loss",
    "kbar": "K-Bar Features at Entry",
    "bb": "Bollinger Band Position",
    "dxy": "DXY Context",
    "mtf": "Multi-Timeframe Alignment",
    "hold_time_streaks": "Hold Time & Streaks",
    "macro": "Macro Composite Context",
    "comparison": "Version Comparison",
}
CHART_SECTION_ORDER = list(CHART_SECTION_LABELS)


def chart_sections_html(charts: list[dict[str, str]]) -> str:
    """Render a study's results.json "charts" array grouped by section, generically —
    per docs/RESEARCH_DEVELOPMENT_SPEC.md section 7. Never a per-study hardcoded list."""
    by_section: dict[str, list[dict[str, str]]] = {}
    for chart in charts:
        by_section.setdefault(chart["section"], []).append(chart)
    blocks = []
    for section in CHART_SECTION_ORDER:
        items = by_section.get(section)
        if not items:
            continue
        images = "".join(
            f'<figure><img src="charts/{html.escape(c["file"])}" alt="{html.escape(c["title"])}" '
            f'style="max-width:100%"><figcaption>{html.escape(c["title"])}</figcaption></figure>'
            for c in items
        )
        blocks.append(
            f'<section class="report-section"><h2>{html.escape(CHART_SECTION_LABELS[section])}</h2>'
            f'<div class="chart-grid">{images}</div></section>'
        )
    return "".join(blocks)


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
        + impact_section_html(study)
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        "<p>Raw CSV and private decision conversations remain in trading-private. This public page "
        "contains reviewed aggregate results and the reproducible method only.</p>"
        + file_actions_html(study)
        + "</section></main>"
    )
    return document(
        study["title"],
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        study["question"],
        body,
        "../../../",
    )


def render_markdown_lite(text: str) -> str:
    """Minimal renderer scoped to the section 13.4 impact.md template: '# ' title
    (skipped, redundant with the section heading), '## ' mechanism headers, '- '
    bullets, and inline '**bold**'/'`code`' spans. Not a general Markdown parser."""

    def inline(segment: str) -> str:
        segment = html.escape(segment)
        segment = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", segment)
        segment = re.sub(r"`(.+?)`", r"<code>\1</code>", segment)
        return segment

    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{' '.join(paragraph)}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            blocks.append(f"<ul>{''.join(list_items)}</ul>")
            list_items.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
        elif stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            blocks.append(f"<h4>{inline(stripped[3:])}</h4>")
        elif stripped.startswith("# "):
            flush_paragraph()
            flush_list()
        elif stripped.startswith("- "):
            flush_paragraph()
            list_items.append(f"<li>{inline(stripped[2:])}</li>")
        else:
            flush_list()
            paragraph.append(inline(stripped))
    flush_paragraph()
    flush_list()
    return "".join(blocks)


def impact_section_html(study: dict[str, object]) -> str:
    impacts = study.get("policy_impacts") or []
    if not impacts:
        return (
            '<section class="report-section"><h2>Impact on 請分析</h2>'
            '<p class="callout">No active 請分析 policy change. '
            "Research/publication only for this study.</p></section>"
        )
    impact_md_path = STUDY_ROOT / study["id"] / "impact.md"
    if impact_md_path.is_file():
        body = f'<div class="impact-md">{render_markdown_lite(impact_md_path.read_text(encoding="utf-8"))}</div>'
    else:
        body = '<ul class="impact-list">' + "".join(
            f'<li><strong>{html.escape(item["surface"])}</strong> · {html.escape(item["summary"])}</li>'
            for item in impacts
        ) + "</ul>"
    return f'<section class="report-section"><h2>Impact on 請分析</h2>{body}</section>'


def file_actions_html(study: dict[str, object]) -> str:
    links = [
        '<a href="results.json">Structured results</a>',
        '<a href="analysis.py">Python method</a>',
        '<a href="study.json">Study manifest</a>',
    ]
    if (STUDY_ROOT / study["id"] / "impact.md").is_file():
        links.append('<a href="impact.md">Impact record</a>')
    return f'<div class="file-actions">{"".join(links)}</div>'


def fail_type_table(by_type: dict[str, dict[str, object]]) -> str:
    body = "".join(
        f"<tr><td><strong>{html.escape(name)}</strong></td><td>{v['count']}</td><td>{v['pct']}%</td></tr>"
        for name, v in by_type.items()
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th>fail_type</th><th>count</th><th>%</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def study_page_fail_pattern_solo(study: dict[str, object]) -> str:
    result = study["_result"]
    baseline = result["baseline"]
    finding_html = "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )
    kbar = result["kbar_coverage"]
    macro_html = ""
    if "by_macro_verdict" in result:
        macro_html = (
            chart_sections_html([c for c in result["charts"] if c["section"] == "macro"])
            + result_table(
                "Macro composite context", result["by_macro_verdict"],
                note="STRONG BUY/WAIT/NEUTRAL, read from the prior daily close (4-day max age). Advisory only.",
            )
        )
    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric("Closed trades", baseline["n"], f'{result["trade_period"]["start"][:10]} → {result["trade_period"]["end"][:10]}')
        + metric("Win rate", f'{baseline["win_rate_pct"]}%', f'95% CI {baseline["win_rate_ci95_pct"][0]}–{baseline["win_rate_ci95_pct"][1]}%')
        + metric("Profit factor", baseline["profit_factor"], f'Net ${baseline["net_pnl_usd"]:,.2f}')
        + metric("Max drawdown", f'${baseline["max_drawdown_usd"]:,.2f}', f'Max {baseline["max_consecutive_losses"]} consecutive losses')
        + '</div>'
        '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + chart_sections_html([c for c in result["charts"] if c["section"] in ("performance", "fail_pattern")])
        + f'<section class="report-section"><h2>Fail-Type Breakdown</h2>{fail_type_table(result["fail_pattern"]["by_type"])}</section>'
        + chart_sections_html([c for c in result["charts"] if c["section"] == "timing_30m"])
        + entry_slot_table(result["by_entry_30m"])
        + result_table(
            "Broad session summary", result["by_session"], show_adjustment=False,
            note="Descriptive only. The 30-minute entry-slot view above is the primary evidence.",
        )
        + chart_sections_html([c for c in result["charts"] if c["section"] in ("pre_entry", "kbar")])
        + f'<div class="note">K-bar coverage: {kbar["with_kbar_data"]}/{kbar["total_immediate_loss"]} '
        f'immediate_loss trades ({kbar["coverage_pct"]}%). Partial coverage — not a full-sample result.</div>'
        + chart_sections_html([c for c in result["charts"] if c["section"] == "bb"])
        + result_table("BB zone", result["bb_zone"], show_adjustment=False)
        + chart_sections_html([c for c in result["charts"] if c["section"] == "dxy"])
        + result_table("DXY RSI bucket", result["dxy"]["regime"]["by_bucket"], show_adjustment=False)
        + result_table("DXY 1D trend", result["dxy"]["regime"]["by_trend"], show_adjustment=False)
        + f'<div class="note">Avg 30-day rolling DXY–XAUUSD correlation: {result["dxy"]["avg_30d_correlation"]}</div>'
        + chart_sections_html([c for c in result["charts"] if c["section"] == "mtf"])
        + result_table("MTF HTF alignment", result["mtf"]["by_alignment"], show_adjustment=False)
        + result_table("MTF 4H RSI state", result["mtf"]["by_4h_state"], show_adjustment=False)
        + chart_sections_html([c for c in result["charts"] if c["section"] == "hold_time_streaks"])
        + macro_html
        + impact_section_html(study)
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        '<p>Raw CSV and private decision conversations remain in trading-private. This public page contains reviewed aggregate results, charts, and the reproducible method only.</p>'
        + file_actions_html(study)
        + '</section></main>'
    )
    return document(
        study["title"],
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        study["question"],
        body,
        "../../../",
    )


def gap_version_prefixes(baseline_diff: dict[str, object]) -> tuple[str, str]:
    """Every gap study's baseline_diff has exactly two per-version keys plus the two
    scalar diffs. Detecting them generically (instead of hardcoding v34/v39 or v1/v2)
    lets one renderer serve any strategy's gap report."""
    keys = [k for k in baseline_diff if k not in ("win_rate_pct_diff", "profit_factor_diff")]
    return keys[0], keys[1]


def gap_entry_slot_table(comparison: dict[str, dict[str, object]], p1: str, p2: str, label1: str, label2: str) -> str:
    diff_key = next(k for k in next(iter(comparison.values())) if k.startswith("win_rate_pct_diff_"))
    body = ""
    for slot, item in comparison.items():
        diff = item.get(diff_key)
        diff_text = "—" if diff is None else f"{diff:+.2f}pp"
        diff_class = "score-neutral"
        if isinstance(diff, (int, float)) and diff > 0:
            diff_class = "score-positive"
        elif isinstance(diff, (int, float)) and diff < 0:
            diff_class = "score-negative"
        body += (
            "<tr>"
            f"<td><strong>{html.escape(slot)}</strong></td>"
            f"<td>{item[f'{p1}_n']}</td>"
            f"<td>{value_or_dash(item.get(f'{p1}_win_rate_pct'), '%')}</td>"
            f"<td>{value_or_dash(item.get(f'{p1}_profit_factor'))}</td>"
            f"<td>{item[f'{p2}_n']}</td>"
            f"<td>{value_or_dash(item.get(f'{p2}_win_rate_pct'), '%')}</td>"
            f"<td>{value_or_dash(item.get(f'{p2}_profit_factor'))}</td>"
            f'<td class="{diff_class}">{diff_text}</td>'
            "</tr>"
        )
    return (
        '<section class="report-section"><h2>30-minute entry-slot comparison</h2>'
        '<p class="section-note">Asia/Taipei bar-start time. Both versions computed with the '
        "same deterministic method; most cells are low-n for both versions and differences "
        "should be read as descriptive, not as a stable timing edge.</p>"
        '<div class="table-wrap tall-table"><table><thead><tr><th>30m slot</th>'
        f"<th>{html.escape(label1)} n</th><th>{html.escape(label1)} WR</th><th>{html.escape(label1)} PF</th>"
        f"<th>{html.escape(label2)} n</th><th>{html.escape(label2)} WR</th><th>{html.escape(label2)} PF</th>"
        f"<th>WR diff ({html.escape(label2)}−{html.escape(label1)})</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div></section>"
    )


def study_page_gap(study: dict[str, object]) -> str:
    result = study["_result"]
    bd = result["baseline_diff"]
    p1, p2 = gap_version_prefixes(bd)
    version_labels = result.get("version_labels", {})
    label1, label2 = version_labels.get(p1, p1.upper()), version_labels.get(p2, p2.upper())
    finding_html = "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )
    fail_rows = "".join(
        f"<tr><td><strong>{html.escape(name)}</strong></td><td>{v[f'{p1}_pct']}%</td><td>{v[f'{p2}_pct']}%</td>"
        f"<td>{v['diff']:+.1f}pp</td></tr>"
        for name, v in result["fail_type_share_diff"].items()
    )
    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric(f"{label1} WR / PF", f'{bd[p1]["win_rate_pct"]}% / {bd[p1]["profit_factor"]}')
        + metric(f"{label2} WR / PF", f'{bd[p2]["win_rate_pct"]}% / {bd[p2]["profit_factor"]}')
        + metric(f"WR diff ({label2}−{label1})", f'{bd["win_rate_pct_diff"]:+.2f}pp')
        + metric(f"PF diff ({label2}−{label1})", f'{bd["profit_factor_diff"]:+.3f}')
        + '</div>'
        f'<div class="note">{html.escape(result["method"]["risk_parameter_caveat"])}</div>'
        '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + chart_sections_html(result["charts"])
        + gap_entry_slot_table(result["by_entry_30m_diff"], p1, p2, label1, label2)
        + '<section class="report-section"><h2>Fail-Type Share</h2>'
        f'<div class="table-wrap"><table><thead><tr><th>fail_type</th><th>{html.escape(label1)}</th><th>{html.escape(label2)}</th><th>diff</th></tr></thead>'
        f'<tbody>{fail_rows}</tbody></table></div></section>'
        + impact_section_html(study)
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        f'<p>{html.escape(result["method"]["basis"])}</p>'
        '<p>Raw CSV and private decision conversations remain in trading-private. This public page contains reviewed aggregate results, charts, and the reproducible method only.</p>'
        + file_actions_html(study)
        + '</section></main>'
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
    if "baseline_diff" in result:
        return study_page_gap(study)
    if "fail_pattern" in result:
        return study_page_fail_pattern_solo(study)
    raise ValueError(
        f'{study["id"]}: results.json matches none of the three section 5 contract '
        'shapes ("versions", "baseline_diff", "fail_pattern")'
    )


STATUS_SHEET_ORDER = ["confirmed", "progress", "pending"]
STATUS_SHEET_LABELS = {"confirmed": "Confirmed", "progress": "Progress", "pending": "Pending"}


def status_sheets_html(study_list: list[dict[str, object]], prefix: str = "../") -> str:
    """Groups cards into the three status sheets (section 13.1) — status IS the sheet,
    market-agnostic by construction. Replaces the single "Adopted studies" grid."""
    by_status: dict[str, list[dict[str, object]]] = {}
    for study in study_list:
        by_status.setdefault(study["status"], []).append(study)
    blocks = []
    for status in STATUS_SHEET_ORDER:
        items = by_status.get(status)
        if not items:
            continue
        cards = "".join(study_card(study, prefix) for study in items)
        blocks.append(
            f'<h2 class="section-title">{STATUS_SHEET_LABELS[status]} '
            f'<span class="sheet-count">({len(items)})</span></h2>'
            f'<div class="grid study-grid">{cards}</div>'
        )
    return "".join(blocks) if blocks else '<p class="empty">No active published study yet.</p>'


def section_page(
    study_list: list[dict[str, object]],
    market: str,
    title: str,
    lede: str,
) -> str:
    selected = [study for study in study_list if study["market"].lower() == market]
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter studies" aria-label="Filter"></div></div>'
        f'<main class="shell"><div class="stats"><span class="stat"><strong>{len(selected)}</strong> active studies</span></div>'
        f'{status_sheets_html(selected)}</main>'
    )
    return document(title, market, lede, body, "../")


def research_page(data: dict[str, object], study_list: list[dict[str, object]]) -> str:
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter studies" aria-label="Filter"></div></div>'
        f'<main class="shell">{status_sheets_html(study_list)}</main>'
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
