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
    Path("xauusd/weekly/index.html"),
    Path("tx/index.html"),
    Path("research/index.html"),
    Path("site/catalog.json"),
}
EXCLUDED_PARTS = {".git", ".github", "_retire", "site", "__pycache__"}
STUDY_ROOT = ROOT / "research/studies"
WEEKLY_ROOT = ROOT / "xauusd/weekly"


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
        generated_weekly_page = (
            len(relative.parts) == 4
            and relative.parts[:2] == ("xauusd", "weekly")
            and relative.name == "index.html"
        )
        if relative in GENERATED or generated_study_page or generated_weekly_page or EXCLUDED_PARTS.intersection(relative.parts):
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
    """Navigation by instrument, then by what you came to do.

    The previous nav offered Overview / XAUUSD / TX / Research, and a study about gold
    lived under Research rather than under XAUUSD — so a reader who wanted to know
    something about gold had two plausible doors and no way to tell which. Studies now sit
    inside the instrument they are about. What is left at the top level is the material
    that genuinely spans both: the lessons, and the vocabulary.
    """
    return (
        '<nav class="nav">'
        f'<a href="{prefix}index.html">\u9996\u9801</a>'
        f'<a href="{prefix}xauusd/">XAUUSD</a>'
        f'<a href="{prefix}tx/">TX</a>'
        f'<a href="{prefix}lessons/">\u4ec0\u9ebc\u6c92\u7528</a>'
        f'<a href="{prefix}glossary/">\u8853\u8a9e</a>'
        '</nav>'
    )


def version_switch(prefix: str, current: str = "v2") -> str:
    """A link back to the archived layout, kept because more re-layouts are expected.

    Only the navigation pages are archived. The studies themselves are identical across
    versions, and copying 5.9MB of charts to preserve a menu would be the wrong trade.
    """
    if current == "v1":
        return ""
    return (
        '<div class="version-switch">'
        f'<a href="{prefix}v1/">\u770b\u820a\u7248 v1.0</a>'
        "</div>"
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
    {version_switch(prefix)}
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


def weekly_summaries() -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if not WEEKLY_ROOT.is_dir():
        return found
    for source in WEEKLY_ROOT.glob("*/summary.json"):
        summary = json.loads(source.read_text(encoding="utf-8"))
        if summary.get("schema_version") != "1.0" or summary.get("market") != "XAUUSD":
            raise ValueError(f"invalid weekly summary: {source.relative_to(ROOT)}")
        if source.parent.name != summary.get("forecast_week"):
            raise ValueError(f"weekly directory/forecast mismatch: {source.relative_to(ROOT)}")
        summary["_relative"] = source.parent.relative_to(ROOT).as_posix()
        found.append(summary)
    return sorted(
        found,
        key=lambda item: (str(item["forecast_week"]), str(item["published_at"])),
        reverse=True,
    )


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
    "total_months": "months",
    "overall_win_rate_pct": "WR",
    "avg_chg_pts": "avg pts",
    "candidate_years": "years",
    "level_0382_win_rate_pct": "0.382 WR",
    "level_05_win_rate_pct": "0.5 WR",
    "level_0618_win_rate_pct": "0.618 WR",
    "s1_t1_hold_win_rate_pct": "S1 T+1 WR",
    "s1_t1_t2_hold_win_rate_pct": "S1 T+1/T+2 WR",
    "s2_t1_hold_win_rate_pct": "S2 T+1 WR",
    "s2_t1_t2_hold_win_rate_pct": "S2 T+1/T+2 WR",
    "s1_d1_down_minus_up_pp": "S1 DOWN−UP",
    "s2_d1_down_minus_up_pp": "S2 DOWN−UP",
    "s1_d1_down_win_rate_pct": "S1 D-1 DOWN WR",
    "s2_d1_down_win_rate_pct": "S2 D-1 DOWN WR",
    "cftc_reports": "CFTC reports",
    "s1_distinct_assigned_reports": "S1 assigned weeks",
    "s2_distinct_assigned_reports": "S2 assigned weeks",
    "s2_regime_win_rate_range_pp": "S2 regime range",
    "s1_t1_market_win_rate_pct": "T1 market WR",
    "s1_pullback_shadow_win_rate_pct": "0.15% pullback WR",
    "s1_pullback_shadow_profit_factor": "0.15% pullback PF",
    "s1_pullback_shadow_fill_rate_pct": "0.15% fill rate",
}


def headline_label(key: str) -> str:
    return HEADLINE_LABELS.get(key, key.replace("_", " "))


def headline_display(key: str, value: object) -> str:
    if isinstance(value, (int, float)) and key.endswith("_pct"):
        return f"{value}%"
    if isinstance(value, (int, float)) and key.endswith("_pp"):
        return f"{value}pp"
    return str(value)


THEME_LABELS = {
    "strategy_diagnostics": "Strategy diagnostics",
    "improvement_attempts": "Improvement attempts",
    "market_structure": "Market structure",
    "methodology": "Methodology",
}
THEME_BLURBS = {
    "strategy_diagnostics": "What the strategies actually do \u2014 how they win, how they lose, and what changed between versions.",
    "improvement_attempts": "Things tried in order to make them better. Almost all of these are negative results, and that is the finding.",
    "market_structure": "What the instrument itself looks like, independent of any strategy.",
    "methodology": "What this programme learned about how to test, usually by getting it wrong first.",
}
THEME_ORDER = ["strategy_diagnostics", "improvement_attempts", "market_structure", "methodology"]


def study_table_html(study_list, prefix="../"):
    """The same studies as a sortable table.

    Cards are readable and do not scale: at 28 studies a grid is a page of scrolling, and
    comparing two studies means holding one in your head while you find the other. A table
    puts theme, market, status and the headline metric in fixed columns so the eye can run
    down one of them, and sorting turns "which studies are confirmed" into one click.

    Both layouts are generated and CSS hides one. That costs a few KB of HTML and avoids
    re-rendering in JavaScript, which would mean the table only existed for readers with
    scripting enabled.
    """
    rows = []
    for study in study_list:
        headline = study.get("headline") or {}
        keys = study.get("card_metrics") or list(headline)[:1]
        metric_key = next((k for k in keys if k in headline), None)
        metric_text = (
            f"{headline_display(metric_key, headline[metric_key])} {headline_label(metric_key)}"
            if metric_key else "\u2014"
        )
        impact = "\u2713" if study.get("policy_impacts") else ""
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(prefix + study["_relative"])}/">'
            f'{html.escape(str(study["title"]))}</a></td>'
            f'<td>{html.escape(THEME_LABELS.get(str(study.get("theme")), "\u2014"))}</td>'
            f'<td>{html.escape(str(study.get("market", "")))}</td>'
            f'<td>{html.escape(str(study.get("status", "")))}</td>'
            f'<td>{html.escape(str(study.get("created_on", "")))}</td>'
            f'<td class="num">{html.escape(metric_text)}</td>'
            f'<td>{impact}</td>'
            "</tr>"
        )
    return (
        '<div data-view-table hidden><div class="table-wrap"><table data-sortable>'
        "<thead><tr>"
        '<th data-sort>Study</th><th data-sort>Theme</th><th data-sort>Market</th>'
        '<th data-sort>Status</th><th data-sort>Date</th><th data-sort>Headline</th>'
        '<th data-sort>\u5f71\u97ff\u8acb\u5206\u6790</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="section-note">\u9ede\u6b04\u4f4d\u6a19\u984c\u53ef\u6392\u5e8f\u3002'
        'Click a column heading to sort.</p></div>'
    )


def view_toggle_html() -> str:
    """Card / table switch. The choice is remembered per reader in localStorage."""
    return (
        '<div class="view-toggle" data-view-toggle>'
        '<button type="button" data-view="cards" aria-pressed="true">\u5361\u7247 Cards</button>'
        '<button type="button" data-view="table" aria-pressed="false">\u8868\u683c Table</button>'
        "</div>"
    )


def theme_sheets_html(study_list, prefix="../"):
    """Group by the question a study asked, not by how far through the process it is.

    Status \u2014 confirmed, progress, pending \u2014 is a workflow state. It tells a reader how
    complete the paperwork is, which is not what they came to find out. Grouping by theme
    also puts the negative results together, which is honest: improvement-attempts is the
    largest section and almost entirely nulls.
    """
    grouped = {}
    for study in study_list:
        grouped.setdefault(str(study.get("theme") or "market_structure"), []).append(study)
    blocks = []
    for theme in THEME_ORDER:
        items = grouped.get(theme)
        if not items:
            continue
        cards = "".join(study_card(study, prefix) for study in items)
        blocks.append(
            f'<h2 class="section-title">{html.escape(THEME_LABELS[theme])} '
            f'<span class="sheet-count">({len(items)})</span></h2>'
            f'<p class="section-note">{html.escape(THEME_BLURBS[theme])}</p>'
            f'<div class="grid study-grid">{cards}</div>'
        )
    for theme in [t for t in grouped if t not in THEME_ORDER]:
        cards = "".join(study_card(study, prefix) for study in grouped[theme])
        blocks.append(
            f'<h2 class="section-title">{html.escape(theme)}</h2>'
            f'<div class="grid study-grid">{cards}</div>'
        )
    grid = "".join(blocks) if blocks else '<p class="empty">No published study yet.</p>'
    return f'<div data-view-cards>{grid}</div>'


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
        # The card states what was found. The question it answered belongs on the study
        # page, where there is room for it; on a card it filled the space and said nothing.
        f'<p>{html.escape(str(study.get("card_summary") or study["question"]))}</p>'
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
    net_pnl_key: str = "net_pnl_usd",
    net_pnl_label: str = "Net USD",
) -> str:
    body = "".join(
        (
            "<tr>"
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{value['n']}</td><td>{value['win_rate_pct']}%</td>"
            f"<td>{value['profit_factor']}</td><td>{value[net_pnl_key]:,.2f}</td>"
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
        f'<th>WR</th><th>PF</th><th>{html.escape(net_pnl_label)}</th>{adjustment_header}</tr></thead>'
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
    "temporal_stability": "Temporal Stability",
    "seasonality": "Seasonality",
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
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
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


def summary_zh_html(study: dict[str, object]) -> str:
    """A short Chinese summary at the top of each study page.

    Not a translation of the page — deliberately. The owner reads these in English as
    practice, and a full Chinese version would simply replace it. Two or three sentences
    stating what was found gives something to check comprehension against, and costs about
    a twentieth of translating every finding.
    """
    text = str(study.get("summary_zh") or "").strip()
    if not text:
        return ""
    return (
        '<section class="report-section"><h2>中文摘要</h2>'
        f'<p>{html.escape(text)}</p>'
        '<p class="section-note">以下為英文原文。專有名詞見'
        '<a href="../../../glossary/">術語表</a>。</p></section>'
    )


def file_actions_html(study: dict[str, object]) -> str:
    links = [
        # First, deliberately. A reader who cannot get past the vocabulary cannot use any
        # of the others.
        '<a href="../../../glossary/">術語表 Glossary</a>',
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
    temporal_html = ""
    if "temporal_stability" in result:
        ts = result["temporal_stability"]
        holdout = ts["holdout_split"]
        flag = ts["degradation_flag"]
        flag_class = {"stable": "score-neutral", "improved": "score-positive", "degraded": "score-negative"}.get(flag, "score-neutral")
        in_s, held = holdout["in_sample"], holdout["held_out"]
        temporal_html = (
            chart_sections_html([c for c in result["charts"] if c["section"] == "temporal_stability"])
            + result_table(
                "Quarterly win rate (chronological)", ts["by_period"], show_adjustment=False,
                note="Descriptive only — not a re-optimized walk-forward. See the note below.",
            )
            + '<div class="note">'
            f'In-sample ({holdout["split_ratio"]*100:.0f}%, {html.escape(str(in_s["period"]["start"]))} → {html.escape(str(in_s["period"]["end"]))}): '
            f'n={in_s["n"]}, WR {in_s["win_rate_pct"]}%, PF {in_s["profit_factor"]}. '
            f'Held-out ({(1 - holdout["split_ratio"]) * 100:.0f}%, {html.escape(str(held["period"]["start"]))} → {html.escape(str(held["period"]["end"]))}): '
            f'n={held["n"]}, WR {held["win_rate_pct"]}%, PF {held["profit_factor"]}. '
            f'Degradation flag: <span class="{flag_class}">{html.escape(flag)}</span>. '
            f'{html.escape(result["method"].get("temporal_stability_limitation", ""))}'
            '</div>'
        )
    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric("Closed trades", baseline["n"], f'{result["trade_period"]["start"][:10]} → {result["trade_period"]["end"][:10]}')
        + metric("Win rate", f'{baseline["win_rate_pct"]}%', f'95% CI {baseline["win_rate_ci95_pct"][0]}–{baseline["win_rate_ci95_pct"][1]}%')
        + metric("Profit factor", baseline["profit_factor"], f'Net ${baseline["net_pnl_usd"]:,.2f}')
        + metric("Max drawdown", f'${baseline["max_drawdown_usd"]:,.2f}', f'Max {baseline["max_consecutive_losses"]} consecutive losses')
        + '</div>'
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
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
        + f'<div class="note">Avg 30-day rolling DXY–{html.escape(study["market"])} correlation: {result["dxy"]["avg_30d_correlation"]}</div>'
        + chart_sections_html([c for c in result["charts"] if c["section"] == "mtf"])
        + result_table("MTF HTF alignment", result["mtf"]["by_alignment"], show_adjustment=False)
        + result_table("MTF 4H RSI state", result["mtf"]["by_4h_state"], show_adjustment=False)
        + chart_sections_html([c for c in result["charts"] if c["section"] == "hold_time_streaks"])
        + macro_html
        + temporal_html
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
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
        + chart_sections_html(result["charts"])
        + gap_entry_slot_table(result["by_entry_30m_diff"], p1, p2, label1, label2)
        + '<section class="report-section"><h2>Fail-Type Share</h2>'
        f'<div class="table-wrap"><table><thead><tr><th>fail_type</th><th>{html.escape(label1)}</th><th>{html.escape(label2)}</th><th>diff</th></tr></thead>'
        f'<tbody>{fail_rows}</tbody></table></div></section>'
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


def month_seasonality_table(by_month: dict[str, dict[str, object]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(v['month_name'])}</strong></td><td>{v['n']}</td>"
        f'<td>{v["win_rate_pct"]}%</td><td>{v["avg_chg_pts"]:+.0f} pts</td>'
        f'<td>{v["avg_chg_pct"]:+.2f}%</td><td>{v["median_chg_pts"]:+.0f}</td>'
        f'<td>{v["best_chg_pts"]:+.0f}</td><td>{v["worst_chg_pts"]:+.0f}</td><td>{html.escape(v["bias"])}</td>'
        "</tr>"
        for _, v in sorted(by_month.items(), key=lambda kv: int(kv[0]))
    )
    return (
        '<section class="report-section"><h2>Seasonality by Calendar Month</h2>'
        '<p class="section-note">Win rate ≥55% → LONG bias, ≤45% → SHORT, else NEUTRAL. n = years present.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Month</th><th>n</th><th>WR</th>'
        "<th>Avg pts</th><th>Avg %</th><th>Median</th><th>Best</th><th>Worst</th><th>Bias</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def week_in_month_table(week_in_month: dict[str, dict[str, object]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(v['week_label'])}</strong></td><td>{v['n']}</td>"
        f'<td>{v["win_rate_pct"]}%</td><td>{v["avg_chg_pts"]:+.0f} pts</td><td>{v["median_chg_pts"]:+.0f}</td>'
        "</tr>"
        for _, v in sorted(week_in_month.items(), key=lambda kv: int(kv[0]))
    )
    return (
        '<section class="report-section"><h2>Week-in-Month Structure</h2>'
        '<div class="table-wrap"><table><thead><tr><th>Week</th><th>n</th><th>WR</th>'
        "<th>Avg pts</th><th>Median</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def year_month_heatmap_table(heatmap: dict[str, dict[str, object]]) -> str:
    years = sorted(heatmap.keys(), key=int)
    header = "<tr><th>Year</th>" + "".join(f"<th>{m}</th>" for m in range(1, 13)) + "</tr>"
    body = ""
    for year in years:
        row = f"<tr><td><strong>{html.escape(year)}</strong></td>"
        for month in range(1, 13):
            value = heatmap[year].get(str(month))
            if value is None:
                row += "<td>—</td>"
            else:
                css = "score-positive" if value > 0 else ("score-negative" if value < 0 else "score-neutral")
                row += f'<td class="{css}">{value:+.0f}</td>'
        row += "</tr>"
        body += row
    return (
        '<section class="report-section"><h2>Year × Month Heatmap (points)</h2>'
        f'<div class="table-wrap tall-table"><table><thead>{header}</thead><tbody>{body}</tbody></table></div></section>'
    )


def study_page_seasonality(study: dict[str, object]) -> str:
    result = study["_result"]
    overall = result["overall"]
    finding_html = "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )
    caveat = result["method"].get("continuous_contract_caveat")
    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric("Total months", overall["total_months"],
                 f'{result["data_period"]["start"]} → {result["data_period"]["end"]}')
        + metric("Overall win rate", f'{overall["overall_win_rate_pct"]}%', "Buy month-open, sell month-close")
        + metric("Avg monthly change", f'{overall["avg_chg_pts"]:+.0f} pts', result["instrument"])
        + "</div>"
        + (f'<div class="note">{html.escape(caveat)}</div>' if caveat else "")
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
        + chart_sections_html([c for c in result["charts"] if c["section"] == "seasonality"])
        + month_seasonality_table(result["by_month"])
        + week_in_month_table(result["week_in_month"])
        + year_month_heatmap_table(result["year_month_heatmap"])
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        "<p>Raw CSV and private decision conversations remain in trading-private. This public page "
        "contains reviewed aggregate results, charts, and the reproducible method only.</p>"
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


def study_page_fib_pullback(study: dict[str, object]) -> str:
    result = study["_result"]
    by_level = result["by_level"]
    finding_html = "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )
    caveat = result["method"].get("continuous_contract_caveat")
    metric_html = "".join(
        metric(
            f"{level} level",
            f'{data["win_rate_pct"]}%' if data["n"] else "no trades",
            f'n={data["n"]}' if data["n"] else "never triggered in this sample",
        )
        for level, data in by_level.items()
    )
    body = (
        '<main class="shell report">'
        f'<div class="metric-grid">{metric_html}</div>'
        + (f'<div class="note">{html.escape(caveat)}</div>' if caveat else "")
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
        + chart_sections_html(result["charts"])
        + result_table(
            "Win rate by Fibonacci retracement level", by_level,
            net_pnl_key="net_pnl_pts", net_pnl_label="Net pts", show_adjustment=False,
        )
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        f'<p>{html.escape(result["method"]["retracement_formula"])}</p>'
        "<p>Raw CSV and private decision conversations remain in trading-private. This public page "
        "contains reviewed aggregate results, charts, and the reproducible method only. Full "
        'year-by-year detail is in <a href="results.json">results.json</a>’s <code>yearly_detail</code>.</p>'
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


def context_program_metrics(study: dict[str, object]) -> str:
    headline = study["headline"]
    keys = study.get("card_metrics") or list(headline)[:4]
    return "".join(
        metric(headline_label(key), headline_display(key, headline[key]))
        for key in keys
        if key in headline
    )


def context_program_findings(study: dict[str, object]) -> str:
    return "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )


def context_program_limitations(result: dict[str, object]) -> str:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in result.get("limitations", []))
    return f'<section class="report-section"><h2>Evidence limits</h2><ul class="impact-list">{items}</ul></section>'


def pullback_replay_table(replay: dict[str, object]) -> str:
    body = ""
    for name, value in replay["policies"].items():
        metric = value["independent_signal_metrics"]
        recent = value["chronological_stability"]["recent_30pct_signal_cohort"]
        interval = metric["win_rate_wilson_95ci_pct"]
        body += (
            "<tr>"
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{metric['n']}</td>"
            f"<td>{value['fill_rate_of_t1_held_pct']}%</td>"
            f"<td>{metric['win_rate_pct']}% ({interval[0]}–{interval[1]}%)</td>"
            f"<td>{metric['profit_factor']}</td>"
            f"<td>{metric['average_pnl_usd']:,.2f}</td>"
            f"<td>{recent['n']} / {recent['win_rate_pct']}% / {recent['profit_factor']}</td>"
            f"<td>{value['paired_outcome_exact_p_value']}</td>"
            "</tr>"
        )
    return (
        '<section class="report-section"><h2>Candle-level T1 pullback replay</h2>'
        '<p class="section-note">The exit emulator matched all 472 OFF exit timestamps and IDs. '
        'The frozen 0.10% primary failed recent stability; 0.15% is post-output and shadow-only.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Policy</th><th>n</th><th>Fill rate</th>'
        '<th>WR (95% CI)</th><th>PF</th><th>Avg USD</th><th>Recent n / WR / PF</th><th>Paired p</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div></section>'
    )


FACTOR_LABELS = {
    "real_rate": "Real rate",
    "us10y": "US10Y",
    "dxy": "DXY",
    "vix": "VIX",
    "gold_trend": "Gold trend",
}


def noise_note(test: dict[str, object]) -> str:
    """State the separability verdict, not just the spread.

    A macro split reads as meaningful when only its win rates are shown. The spread has to
    be placed against what a no-effect null produces on the same group sizes, or the reader
    supplies the conclusion themselves.
    """
    if not test or not test.get("applicable"):
        return "Too few trades per group to test separability."
    verdict = "separable from noise" if test.get("separable") else "not separable from noise"
    return (
        f'Observed spread {test["observed_spread_pp"]}pp against a null median of '
        f'{test["null_median_spread_pp"]}pp over {test["trials"]:,} shuffles: '
        f'P(spread >= observed) = {test["p_spread_at_least_observed"]}, {verdict}.'
    )


def macro_gvz_section(strategy: str, gvz: dict[str, object]) -> str:
    """The GVZ threshold sweep, reported with its multiple-comparison correction.

    The sweep searched every threshold, so its best gap has to be compared against the best
    gap a sweep finds on unrelated data. Reporting the winning threshold alone would present
    a search artefact as a finding.
    """
    best = gvz["largest_gap_threshold"]
    test = gvz["permutation_test"]
    # This table carries no Net USD column. The sweep records only n, win rate and profit
    # factor per side, and reusing the standard table would have printed a 0.00 that reads
    # as "this split broke even" rather than "this split was never measured in dollars".
    rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(label)}</strong></td>"
        f'<td>{side["n"]}</td><td>{side["win_rate_pct"]}%</td>'
        f'<td>{side["profit_factor"]}</td>'
        "</tr>"
        for label, side in (
            (f'GVZ < {best["threshold"]}', best["below"]),
            (f'GVZ >= {best["threshold"]}', best["above"]),
        )
    )
    if test.get("applicable"):
        note = (
            f'Best threshold found by sweeping all candidates: gap {test["observed_best_gap_pp"]}pp. '
            f'A sweep over {test["trials"]:,} shuffles of the same data finds a median best gap of '
            f'{test["null_median_best_gap_pp"]}pp and a 95th percentile of '
            f'{test["null_95th_best_gap_pp"]}pp, so P(best gap >= observed) = '
            f'{test["p_best_gap_at_least_observed"]}. '
            + ("Survives the correction." if test.get("survives_multiple_comparison")
               else "Does not survive the multiple-comparison correction; this split is a "
                    "search artefact of having tested every threshold.")
            + f' The split also needs {best["min_detectable_pp"]}pp to resolve at these group sizes.'
        )
    else:
        note = "Too few trades to test the sweep against a null."
    return (
        f'<section class="report-section"><h2>{html.escape(strategy)} GVZ threshold sweep</h2>'
        f'<p class="section-note">{html.escape(note)}</p>'
        '<div class="table-wrap"><table><thead><tr><th>Context</th><th>n</th>'
        '<th>WR</th><th>PF</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div></section>'
    )


def macro_attribution_tables(strategies: dict[str, dict]) -> list[str]:
    tables = []
    for strategy, data in strategies.items():
        coverage = (
            f'{data["trades_with_macro"]}/{data["trades_total"]} trades carried macro values '
            f'({data["macro_coverage_pct"]}% coverage). Baseline n={data["baseline"]["n"]}, '
            f'WR {data["baseline"]["win_rate_pct"]}%, PF {data["baseline"]["profit_factor"]}.'
        )
        factor_rows: dict[str, dict] = {}
        for factor, block in data["by_factor"].items():
            label = FACTOR_LABELS.get(factor, factor)
            for group, metrics in block["groups"].items():
                factor_rows[f"{label} · {group}"] = metrics
        tables.append(result_table(f"{strategy} single macro factors", factor_rows,
                                   show_adjustment=False, note=coverage))
        for key, title in (("by_verdict", "composite verdict"), ("by_score", "composite score")):
            block = data.get(key)
            if block:
                tables.append(result_table(f"{strategy} {title}", block["groups"],
                                           show_adjustment=False,
                                           note=noise_note(block.get("noise_test", {}))))
        if data.get("gvz"):
            tables.append(macro_gvz_section(strategy, data["gvz"]))
    return tables


def study_page_context_program(study: dict[str, object]) -> str:
    """Render multi-strategy context studies from their shared aggregate shape.

    The strategy-specific tables are detected from result keys so later confirmation,
    completed-daily, or weekly-regime studies can reuse this page without an ID branch.
    """
    result = study["_result"]
    strategies = result["strategies"]
    first = next(iter(strategies.values()))
    tables = []
    if "rules" in first:
        for strategy, data in strategies.items():
            rule_rows = {
                "Baseline": data["rules"]["baseline"]["selected"],
                "T+1 holds signal low": data["rules"]["t1_hold_signal_low"]["selected"],
                "T+1/T+2 hold signal low": data["rules"]["t1_t2_hold_signal_low"]["selected"],
                "T+1 holds low and close": data["rules"]["t1_hold_low_and_close"]["selected"],
                "T+1/T+2 hold low and close": data["rules"]["t1_t2_hold_low_and_close"]["selected"],
            }
            holdout = data["chronological_holdout"]["held_out"]
            note = (
                f'Matched {data["coverage"]["matched_trades"]}/{data["coverage"]["total_trades"]} trades. '
                f'Held-out baseline n={holdout["baseline"]["n"]}, WR {holdout["baseline"]["win_rate_pct"]}%; '
                f'T+1 hold n={holdout["t1_hold_signal_low"]["selected_n"]}, '
                f'WR {holdout["t1_hold_signal_low"]["selected"]["win_rate_pct"]}%.'
            )
            tables.append(result_table(f"{strategy} confirmation screen", rule_rows, show_adjustment=False, note=note))
    elif "by_d1_direction" in first:
        for strategy, data in strategies.items():
            held = data["chronological_holdout"]["held_out"]
            note = (
                f'Held-out baseline n={held["baseline"]["n"]}, WR {held["baseline"]["win_rate_pct"]}%; '
                "all daily bars were fully completed before assignment."
            )
            tables.append(result_table(f"{strategy} prior completed day", data["by_d1_direction"], show_adjustment=False, note=note))
            tables.append(result_table(f"{strategy} D-2 → D-1 sequence", data["by_d2_d1_sequence"], show_adjustment=False))
    elif "by_net_oi_regime" in first:
        for strategy, data in strategies.items():
            note = (
                f'{data["coverage"]["distinct_reports"]} distinct conservatively available reports; '
                "several trades may share one weekly report."
            )
            tables.append(result_table(f"{strategy} Managed Money net/OI regime", data["by_net_oi_regime"], show_adjustment=False, note=note))
            tables.append(result_table(f"{strategy} crowding regime", data["by_crowding_regime"], show_adjustment=False))
    elif "by_factor" in first:
        tables.extend(macro_attribution_tables(strategies))
    else:
        raise ValueError(f'{study["id"]}: unknown multi-strategy context shape')

    body = (
        '<main class="shell report">'
        f'<div class="metric-grid">{context_program_metrics(study)}</div>'
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{context_program_findings(study)}</div></section>'
        + impact_section_html(study)
        + chart_sections_html(result.get("charts", []))
        + "".join(tables)
        + context_program_limitations(result)
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        '<p>All trading timestamps use Asia/Taipei. Raw CSV, source manifests, and private decision records are not published. '
        'This page contains reviewed aggregate results, one reproducible method, and pre-reviewed charts only.</p>'
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


def study_page_pullback_replay(study: dict[str, object]) -> str:
    result = study["_result"]
    validation = result["emulator_validation"]
    baseline = result["off_baseline_metrics"]
    body = (
        '<main class="shell report">'
        f'<div class="metric-grid">{context_program_metrics(study)}</div>'
        '<section class="report-section"><h2>Validation control</h2>'
        f'<p>The emulator matched {validation["exit_time_and_signal_match_n"]}/'
        f'{validation["n"]} OFF exit timestamps and IDs. Full OFF baseline: '
        f'WR {baseline["win_rate_pct"]}%, PF {baseline["profit_factor"]}, '
        f'average USD {baseline["average_pnl_usd"]}.</p></section>'
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{context_program_findings(study)}</div></section>'
        + impact_section_html(study)
        + pullback_replay_table(result)
        + context_program_limitations(result)
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(study["hypothesis"])}</p>'
        '<p>All timestamps use Asia/Taipei. Raw CSV, per-trade output, source manifests, '
        'and private decision records are not published. The Python method accepts '
        'locally authorized inputs and reproduces the reviewed aggregate.</p>'
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


def study_page_range_profile(study: dict[str, object]) -> str:
    """Intraday range-accumulation shape (RS-XAUUSD-20260823-001).

    A price-structure study rather than a strategy report, so it shares no field names with
    the fail-pattern/comparison/seasonality contracts and gets its own tables.
    """
    result = study["_result"]
    fam = result["families"]
    observed = fam["observed_profile"]
    nulls = fam["vs_shuffled_returns_null"]
    coverage = result["coverage"]
    head = study["headline"]

    finding_html = "".join(
        f'<article class="insight {html.escape(item["tone"])}"><strong>{html.escape(item["title"])}</strong>'
        f'<p>{html.escape(item["detail"])}</p></article>'
        for item in study["findings"]
    )

    def table(headers: list[str], rows: list[list[str]], caption: str, note: str = "") -> str:
        head_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
        )
        note_html = f'<p class="section-note">{html.escape(note)}</p>' if note else ""
        return (
            f'<section class="report-section"><h2>{html.escape(caption)}</h2>{note_html}'
            f'<div class="table-wrap"><table><thead><tr>{head_html}</tr></thead>'
            f"<tbody>{body_html}</tbody></table></div></section>"
        )

    marks = ["07:30", "09:00", "09:30", "11:00", "12:30", "15:30", "18:00",
             "20:30", "21:30", "22:30", "23:30", "01:30", "02:30", "04:30"]
    profile_rows = [
        [f"<strong>{slot}</strong>",
         f'{observed["train"]["range_completed_mean_pct"][slot]}%',
         f'{observed["valid"]["range_completed_mean_pct"][slot]}%',
         f'{observed["holdout"]["range_completed_mean_pct"][slot]}%',
         f'{nulls["holdout"]["excess_pct"][slot]:+}']
        for slot in marks
    ]

    consistent = sorted(
        (row for row in fam["increment_consistency"]["detail"] if row["min_abs_z"] >= 2.0),
        key=lambda row: -row["min_abs_z"],
    )
    consistent_rows = [
        [f'<strong>{row["slot"]}</strong>',
         "more than chance" if "more" in row["direction"] else "less than chance",
         " / ".join(f"{value:+}" for value in row["excess_pct"]),
         f'{row["min_abs_z"]}']
        for row in consistent
    ]

    clock = fam["us_clock_alignment"]["et_0830_release_slot"]
    clock_rows = [
        ["08:30 ET, US summer time", "20:30 Taipei",
         f'<strong>{clock["us_dst_on"]["mean_share_pct"]}%</strong>'],
        ["08:30 ET, US winter time", "21:30 Taipei",
         f'<strong>{clock["us_dst_off"]["mean_share_pct"]}%</strong>'],
        ["same Taipei slot, winter", "20:30 Taipei",
         f'{clock["same_slot_in_the_other_regime"]["20:30_when_dst_off"]}%'],
        ["same Taipei slot, summer", "21:30 Taipei",
         f'{clock["same_slot_in_the_other_regime"]["21:30_when_dst_on"]}%'],
    ]

    residual = fam["residual_and_extreme_risk"]["holdout"]
    residual_rows = [
        [f"<strong>{slot}</strong>",
         f'{residual[slot]["residual_range_mean_pct"]}%',
         f'{residual[slot]["residual_range_median_pct"]}%',
         f'{residual[slot]["new_extreme_after_pct"]}%']
        for slot in ["18:00", "20:30", "21:30", "22:30", "23:30", "00:30", "02:30", "03:30"]
    ]

    morning = fam["morning_conditioning"]
    morning_rows = [
        [f"<strong>{label.replace('_', ' ')}</strong>",
         f'{morning[label]["median_morning_ratio"]}',
         f'{morning[label]["median_day_ratio"]}',
         f'<strong>{morning[label]["median_rest_of_day_ratio"]}</strong>']
        for label in ["quiet_morning", "middle", "busy_morning"]
    ]

    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric("Sessions", coverage["sessions_used"],
                 f'{coverage["first_session"]} → {coverage["last_session"]}')
        + metric("Family permutation p", head["family_permutation_p_holdout"],
                 "all three periods; 200-shuffle resolution floor")
        + metric("Half hours beating chance",
                 f'{head["slots_consistent_and_abs_z_over_2_in_all_periods"]} of 48',
                 "sign-consistent and |z|>2 in every period")
        + "</div>"
        + summary_zh_html(study)
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
        + table(["Taipei", "train", "valid", "holdout", "holdout vs null"], profile_rows,
                "How the day fills",
                "Percent of the session's final range already traversed. The null shuffles "
                "each session's own bar-to-bar changes and rebuilds the path, so it keeps "
                "that day's volatility and the arcsine geometry of a random walk and "
                "destroys only when the large moves happened. All 48 slots are in "
                "results.json.")
        + table(["Taipei", "direction", "excess % (train / valid / holdout)", "min |z|"],
                consistent_rows,
                "Half hours that beat chance in every period",
                "32 of 48 slots are sign-consistent across all three periods against 12 "
                "expected by chance; these also clear |z| > 2 in each period.")
        + table(["release window", "clock slot", "share of the day's range"], clock_rows,
                "The busiest US half hour moves twice a year",
                "Taipei keeps no summer time and New York does. The peak of the US block is "
                "the 08:30 ET release window, so on a fixed Taipei clock it shifts by an "
                "hour — and the same Taipei slot carries five times less range in the other "
                "half of the year.")
        + table(["after", "mean residual", "median residual", "new daily extreme still arrives"],
                residual_rows,
                "What is left (holdout)",
                "Median residual reaches zero at 23:30: on more than half of sessions no new "
                "extreme arrives after that. It stays a probability, not a curfew.")
        + table(["morning tercile", "morning range", "full day", "rest of day"], morning_rows,
                "A busy morning says nothing about what is left",
                "Ratios against each session's trailing 20-session median range. Morning "
                "correlates with the full day at Spearman "
                f'{morning["spearman_morning_vs_full_day"]} and with the rest of the day at '
                f'{morning["spearman_morning_vs_rest_of_day"]} — it predicts the day only '
                "because it is part of it.")
        + '<section class="report-section"><h2>Limitations</h2>'
        + text_list(result["limitations"]) + "</section>"
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(str(study["hypothesis"]))}</p>'
        "<p>Descriptive and never directional: the profile says how much range has "
        "accumulated, not which way price moved. Raw CSV and private decision records remain "
        "in trading-private. This public page carries reviewed aggregate results and the "
        "reproducible method only — the published script reads its bars from a "
        "<code>local-inputs/</code> folder you supply.</p>"
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


NULL_REGISTRY = ROOT / "research/null-results/null_results.json"
GLOSSARY = ROOT / "site/glossary.json"


def glossary() -> list[dict[str, str]]:
    if not GLOSSARY.is_file():
        return []
    try:
        return json.loads(GLOSSARY.read_text(encoding="utf-8")).get("terms", [])
    except json.JSONDecodeError:
        return []


def glossary_page(terms: list[dict[str, str]]) -> str:
    """Bilingual vocabulary, written once for the whole site.

    The pages are in English because the owner is using them to practise reading it. A
    full Chinese translation would defeat that — given both, nobody reads the second one.
    What actually blocks comprehension is the domain vocabulary, and that vocabulary
    repeats: "win rate" appears 41 times across the studies, "holdout" 30. Defining each
    once here costs a fraction of translating them in place, and leaves the English as the
    thing being read.
    """
    rows = "".join(
        f"<tr><td><strong>{html.escape(t['en'])}</strong></td>"
        f"<td>{html.escape(t['zh'])}</td>"
        f"<td>{html.escape(t['gloss'])}</td></tr>"
        for t in terms
    )
    body = (
        '<main class="shell report">'
        '<section class="report-section"><h2>How to use this</h2>'
        "<p>研究頁面是英文的，因為擁有者用它們練習閱讀。這裡不是翻譯——是把每個技術詞"
        "定義一次，讓英文原文讀得下去。</p>"
        "<p>The study pages stay in English. This defines the vocabulary that blocks "
        "comprehension, once, so the English remains the thing being read.</p>"
        "<p><strong>三個最該先看的：</strong>"
        "<code>baseline</code>（沒有它，任何勝率都無法判讀）、"
        "<code>resolution bound</code>（決定一個「無證據」關掉了多少門）、"
        "<code>lookahead</code>（本站最大的假發現都來自它）。</p></section>"
        '<section class="report-section"><h2>Terms</h2>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>English</th><th>中文</th><th>說明</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div></section>"
        '<div class="file-actions"><a href="../research/">All studies</a>'
        '<a href="../research/null-results/">What did not work</a></div>'
        "</main>"
    )
    return document(
        "Glossary 術語表",
        "Bilingual reference",
        "Every technical term used across the studies, defined once in Chinese so the "
        "English pages stay readable.",
        body,
        "../",
    )

VERDICT_STYLE = {
    "no_evidence": ("bounded", "warn"),
    "below_cost": ("below cost", "warn"),
    "underpowered": ("untestable", "info"),
    "survives_screens": ("survived", "good"),
    "skipped": ("skipped", "info"),
}


def null_registry() -> dict[str, object] | None:
    if not NULL_REGISTRY.is_file():
        return None
    try:
        return json.loads(NULL_REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def null_results_page(registry: dict[str, object]) -> str:
    """The negative-results surface.

    Deliberately plain. Its readers are a person deciding whether a question is worth
    asking again, and a model deciding the same thing before spending a session on it —
    and for the second reader the JSON beside this page is the real interface. The page
    exists to make the JSON legible and to state, once and without hedging, that finding
    nothing is the result rather than the absence of one.
    """
    totals = registry["totals"]
    hypotheses = [e for e in registry["entries"] if e["kind"] == "hypothesis"]
    by_verdict = totals["by_verdict"]

    def chip(verdict: str, count: int) -> str:
        label, tone = VERDICT_STYLE.get(verdict, (verdict, "info"))
        return (f'<div class="metric"><div class="metric-label">{html.escape(label)}</div>'
                f'<div class="metric-value">{count}</div>'
                f'<div class="metric-detail">{html.escape(str(registry["how_to_read"]["verdicts"].get(verdict, "")))}</div></div>')

    rows = []
    for entry in sorted(hypotheses, key=lambda e: str(e["entry_id"])):
        label, tone = VERDICT_STYLE.get(str(entry["verdict"]), (str(entry["verdict"]), "info"))
        effect = entry.get("effect")
        bound = entry.get("smallest_resolvable_effect")
        ratio = ""
        if isinstance(effect, (int, float)) and isinstance(bound, (int, float)) and bound:
            ratio = f"{abs(effect) / bound:.2f}x"
        rows.append(
            "<tr>"
            f'<td><code>{html.escape(str(entry["entry_id"]).split(":")[-1])}</code></td>'
            f'<td>{html.escape(str(entry.get("claim") or ""))}</td>'
            f'<td>{html.escape(str(entry.get("origin") or ""))}</td>'
            f'<td class="num">{entry.get("n_condition") if entry.get("n_condition") is not None else "&mdash;"}</td>'
            f'<td class="num">{effect if effect is not None else "&mdash;"}</td>'
            f'<td class="num">{bound if bound is not None else "&mdash;"}</td>'
            f'<td class="num">{ratio or "&mdash;"}</td>'
            f'<td><span class="insight {tone}">{html.escape(label)}</span></td>'
            "</tr>"
        )

    gaps = registry.get("data_gaps") or []
    gap_html = "".join(
        f'<li><strong>{html.escape(str(g["family"]))}</strong> &mdash; {html.escape(str(g["gap"]))}</li>'
        for g in gaps
    ) or "<li>None recorded.</li>"

    families = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        + "".join(f'<td class="num">{counts.get(v, 0)}</td>'
                  for v in ("no_evidence", "below_cost", "underpowered", "survives_screens"))
        + "</tr>"
        for name, counts in sorted(totals["by_family"].items())
    )

    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric("Studies", totals["studies"], "most of them negative")
        + metric("Hypotheses on record", totals["hypotheses"], "each with its resolution bound")
        + metric("Survivors", by_verdict.get("survives_screens", 0),
                 "cleared every screen applied")
        + "</div>"
        '<section class="report-section"><h2>What this is</h2>'
        "<p>A record of questions that were asked of this data and answered with "
        "<em>no</em>. That is the finding, not a missing one. Most searches for a tradeable "
        "edge end this way, and the ones that do not are usually the ones nobody wrote "
        "down carefully enough to check.</p>"
        "<p>The number that makes an entry worth keeping is the "
        "<strong>smallest resolvable effect</strong>: the smallest thing the sample could "
        "have seen. A null with a wide bound rules out very little and leaves the question "
        "open. A null with a tight bound closes it. Flattening both into &ldquo;didn&rsquo;t "
        "work&rdquo; throws away the difference, so nothing here does that.</p>"
        "<p>The last column is that ratio &mdash; effect divided by bound. Below "
        "<code>1.00x</code> the observed effect is inside the noise floor of its own "
        "sample.</p></section>"
        '<section class="report-section"><h2>Verdicts</h2><div class="metric-grid">'
        + "".join(chip(v, c) for v, c in sorted(by_verdict.items()))
        + "</div></section>"
        '<section class="report-section"><h2>Hypotheses</h2>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>id</th><th>claim</th><th>origin</th><th>n</th><th>effect</th>"
        "<th>resolvable</th><th>ratio</th><th>verdict</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        '<p class="section-note">An <code>external_claim</code> was specified by someone '
        "else before this dataset was examined. Testing a claim you did not invent is a "
        "weaker form of data mining than testing one you did.</p></section>"
        '<section class="report-section"><h2>By family</h2>'
        '<div class="table-wrap"><table><thead><tr><th>family</th><th>bounded</th>'
        "<th>below cost</th><th>untestable</th><th>survived</th></tr></thead>"
        f"<tbody>{families}</tbody></table></div></section>"
        '<section class="report-section"><h2>Doors that were never opened</h2>'
        "<p>An <em>untestable</em> verdict is not a failure to find something. It is a "
        "failure to be able to look, and it names what looking would take. These are the "
        "cheapest places to make progress, because the blocker is data rather than "
        f"insight.</p><ul class=\"impact-list\">{gap_html}</ul></section>"
        '<section class="report-section"><h2>For machines</h2>'
        "<p>The registry is generated, not written, so it cannot drift from the studies it "
        "describes. It is published beside this page as JSON and is the intended interface "
        "for anything automated: read it, and skip what is already closed.</p>"
        '<div class="file-actions"><a href="null_results.json">Registry JSON</a>'
        '<a href="../">All studies</a><a href="../../glossary/">術語表 Glossary</a></div></section>'
        "</main>"
    )
    return document(
        "What did not work",
        "Negative results registry",
        "Questions asked of this data and answered with no, each carrying the smallest "
        "effect its sample could have resolved.",
        body,
        "../../",
    )


def study_page(study: dict[str, object]) -> str:
    result = study["_result"]
    if "versions" in result:
        return study_page_comparison(study)
    if "baseline_diff" in result:
        return study_page_gap(study)
    if "fail_pattern" in result:
        return study_page_fail_pattern_solo(study)
    if "by_month" in result:
        return study_page_seasonality(study)
    if "by_level" in result:
        return study_page_fib_pullback(study)
    if "policies" in result and "emulator_validation" in result:
        return study_page_pullback_replay(study)
    if "strategies" in result:
        return study_page_context_program(study)
    if "families" in result:
        return study_page_range_profile(study)
    raise ValueError(
        f'{study["id"]}: results.json matches none of the known report shapes '
        '("versions", "baseline_diff", "fail_pattern", "by_month", "by_level", "families")'
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


def weekly_card(summary: dict[str, object], href: str) -> str:
    mode = "Multi-source comparison" if summary["publication_mode"] == "multi_source" else "Single source"
    return (
        f'<a class="card" data-card href="{html.escape(href)}">'
        f'<div class="type">weekly · {html.escape(str(summary["forecast_week"]))}</div>'
        f'<h2>XAUUSD weekly outlook</h2><p>{html.escape(str(summary["market_summary"]))}</p>'
        '<div class="mini-metrics">'
        f'<span><strong>{summary["source_count"]}</strong> sources</span>'
        f'<span><strong>{html.escape(mode)}</strong></span>'
        f'<span><strong>{html.escape(str(summary["confidence"]))}</strong> confidence</span>'
        '</div></a>'
    )


def text_list(items: list[object], empty: str = "None recorded") -> str:
    if not items:
        return f'<p class="section-note">{html.escape(empty)}</p>'
    return '<ul class="impact-list">' + "".join(
        f'<li>{html.escape(str(item))}</li>' for item in items
    ) + '</ul>'


def weekly_comparison_table(summary: dict[str, object]) -> str:
    producers = [str(item["producer"]) for item in summary["scenario_comparison"]]
    directions: list[str] = []
    values: dict[str, dict[str, object]] = {}
    for source in summary["scenario_comparison"]:
        values[str(source["producer"])] = {
            str(item["direction"]): item["probability"] for item in source["scenarios"]
        }
        for item in source["scenarios"]:
            direction = str(item["direction"])
            if direction not in directions:
                directions.append(direction)
    head = "".join(f'<th>{html.escape(producer)}</th>' for producer in producers)
    rows = "".join(
        '<tr><td><strong>' + html.escape(direction) + '</strong></td>'
        + "".join(f'<td>{values[producer].get(direction, "—")}%</td>' for producer in producers)
        + '</tr>'
        for direction in directions
    )
    return (
        '<section class="report-section"><h2>Source scenario comparison</h2>'
        '<p class="section-note">Each column reproduces that eligible source’s probability; the adopted view below is resolved claim by claim, never by producer rank.</p>'
        f'<div class="table-wrap"><table><thead><tr><th>Direction</th>{head}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div></section>'
    )


def weekly_summary_page(
    summary: dict[str, object],
    archive: list[dict[str, object]],
    *,
    prefix: str,
    source_href: str,
    latest: bool,
) -> str:
    mode_label = "Multi-source comparison" if summary["publication_mode"] == "multi_source" else "Single source — no consensus claim"
    adopted = "".join(
        '<article class="insight info">'
        f'<strong>{html.escape(str(item["direction"]))} · {item["probability"]}%</strong>'
        f'<p><b>Conditions:</b> {html.escape(str(item["conditions"]))}</p>'
        f'<p><b>Invalidation:</b> {html.escape(str(item["invalidation"]))}</p>'
        f'<p><b>Targets:</b> {html.escape(str(item["targets"]))}</p></article>'
        for item in summary["adopted_scenarios"]
    )
    levels = "".join(
        f'<tr><td><strong>{html.escape(str(item["label"]))}</strong></td>'
        f'<td>{html.escape(str(item["value"]))}</td><td>{html.escape(str(item["basis"]))}</td></tr>'
        for item in summary["key_levels"]
    )
    strategies = "".join(
        f'<tr><td><strong>{html.escape(str(item["strategy"]))}</strong></td>'
        f'<td>{html.escape(str(item["stance"]))}</td><td>{html.escape(str(item["entry"]))}</td>'
        f'<td>{html.escape(str(item["stop"]))}</td><td>{html.escape(str(item["risk"]))}</td></tr>'
        for item in summary["strategy_plan"]
    )
    events = "".join(
        '<article class="insight warn">'
        f'<strong>{html.escape(str(item["name"]))}</strong>'
        f'<p>{html.escape(str(item["scheduled_at"]))}</p>'
        f'<p>{html.escape(str(item["handling"]))}</p></article>'
        for item in summary["event_risk"]
    )
    recommendation = summary["recommendation"]
    archive_cards = "".join(
        weekly_card(item, ("" if latest else "../") + str(item["forecast_week"]) + "/")
        for item in archive
    )
    body = (
        '<main class="shell report">'
        '<div class="metric-grid">'
        + metric("Forecast week", summary["forecast_week"], f'Edition {summary["edition"]}')
        + metric("Publication mode", mode_label, f'{summary["source_count"]} eligible source(s)')
        + metric("Confidence", summary["confidence"], f'Published {summary["published_at"]}')
        + metric("Data cutoff", summary["data_cutoff"], "Weekend research snapshot")
        + '</div>'
        + f'<div class="callout"><strong>{html.escape(str(recommendation["stance"]))}</strong>'
        f'<p>{html.escape(str(recommendation["summary"]))}</p>'
        f'<p><b>Changes when:</b> {html.escape(str(recommendation["invalidation"]))}</p></div>'
        + weekly_comparison_table(summary)
        + '<section class="report-section"><h2>Adopted scenario view</h2>'
        f'<div class="insight-grid">{adopted}</div></section>'
        + '<section class="report-section"><h2>Agreements</h2>' + text_list(summary["agreements"]) + '</section>'
        + '<section class="report-section"><h2>Disagreements and resolution</h2>' + text_list(summary["disagreements"]) + '</section>'
        + '<section class="report-section"><h2>Key levels</h2><div class="table-wrap"><table>'
        f'<thead><tr><th>Role</th><th>Level</th><th>Basis</th></tr></thead><tbody>{levels}</tbody></table></div></section>'
        + '<section class="report-section"><h2>Strategy plan</h2><div class="table-wrap"><table>'
        f'<thead><tr><th>Strategy</th><th>Stance</th><th>Entry gate</th><th>Stop</th><th>Risk</th></tr></thead><tbody>{strategies}</tbody></table></div></section>'
        + f'<section class="report-section"><h2>Event risk</h2><div class="insight-grid">{events}</div></section>'
        + '<section class="report-section"><h2>Evidence limits</h2>' + text_list(summary["evidence_limits"])
        + f'<p class="section-note">{html.escape(str(summary["disclaimer"]))}</p>'
        + f'<div class="file-actions"><a href="{html.escape(source_href)}">Reviewed summary JSON</a></div></section>'
        + '<h2 class="section-title">Weekly archive</h2><div class="grid">' + archive_cards + '</div>'
        + '</main>'
    )
    eyebrow = "Latest reviewed weekly outlook" if latest else "Reviewed weekly archive"
    return document(
        f'XAUUSD {summary["forecast_week"]} outlook',
        eyebrow,
        str(summary["market_summary"]),
        body,
        prefix,
    )


SIGNAL_PLAYBOOK = {
    "XAUUSD": {
        "intro": "\u8a0a\u865f\u5230\u4e86\u3002\u9019\u9801\u53ea\u56de\u7b54\u4e00\u4ef6\u4e8b\uff1a"
                 "\u73fe\u5728\u6709\u4ec0\u9ebc\u5df2\u77e5\u7684\u6771\u897f\uff0c\u80fd\u5e6b\u4f60"
                 "\u5224\u65b7\u9019\u7b46\u55ae\u3002",
        "checks": [
            {
                "title": "\u9032\u5834\u50f9\u5728\u5e03\u6797\u901a\u9053\u7684\u54ea\u88e1\uff08S1\uff09",
                "study": "RS-XAUUSD-20260823-002",
                "what": "%B > 1.0\uff08\u6536\u5728\u4e0a\u8ecc\u4e4b\u5916\uff09\u7684\u9032\u5834\uff0c"
                        "\u6b77\u53f2\u52dd\u7387 73.17%\uff08n=82\uff09\uff0c\u57fa\u6e96\u662f 55.93%\u3002"
                        "\u800c\u4e14\u5b83\u5728\u6a23\u672c\u5916\u66f4\u5f37\uff0c\u4e0d\u662f\u8b8a\u5f31\u3002",
                "caveat": "\u4f46\u62ff\u5b83\u7576\u904e\u6ffe\u5668\u6703\u8ce0\u9322\uff1a"
                          "\u53ea\u7559 17% \u7684\u9032\u5834\u3001 44% \u7684\u5831\u916c\u3002"
                          "\u5b83\u662f\u53c3\u8003\u8cc7\u8a0a\uff0c\u4e0d\u662f\u9598\u9580\u3002",
            },
            {
                "title": "\u73fe\u5728\u9019\u500b\u6642\u9593\u9ede\uff0c\u7576\u65e5\u9084\u5269\u591a\u5c11\u7a7a\u9593",
                "study": "RS-XAUUSD-20260823-001",
                "what": "\u53f0\u5317 23:30 \u4e4b\u5f8c\uff0c\u4e2d\u4f4d\u6578\u7684\u4e00\u5929"
                        "\u5df2\u7d93\u8d70\u5b8c\u4e86\uff08\u5269\u9918\u5340\u9593\u4e2d\u4f4d\u6578 0%\uff09\u3002"
                        "02:30 \u4e4b\u5f8c\u5e73\u5747\u53ea\u5269 4%\u3002",
                "caveat": "\u9019\u53ea\u8b1b\u300c\u9084\u6709\u591a\u5c11\u7a7a\u9593\u300d\uff0c"
                          "\u5b8c\u5168\u4e0d\u8b1b\u65b9\u5411\u3002",
            },
            {
                "title": "\u9019\u500b\u7b56\u7565\u672c\u4f86\u5c31\u9577\u4ec0\u9ebc\u6a23",
                "study": "RS-XAUUSD-20260727-007",
                "what": "S2 V3.2 \u57fa\u6e96\u52dd\u7387 47.13%\u3001\u7372\u5229\u56e0\u5b50 2.05\u3002"
                        "\u4f4e\u52dd\u7387\u9ad8\u8ce0\u7387\u662f\u6b63\u5e38\u5f62\u614b\uff0c"
                        "\u53ea\u770b\u52dd\u7387\u6703\u8aa4\u5224\u3002",
                "caveat": "S1 V3.9 \u7684\u57fa\u6e96\u662f 55.93% / PF 1.849\uff0c"
                          "\u5e73\u5747\u6301\u6709 30 \u6839\u3002",
            },
        ],
        "ruled_out": [
            "Macro \u7d9c\u5408\u5206\u6578\u3001GVZ \u9580\u6abb\u2014\u2014\u5df2\u65bc 2026-08-17 \u64a4\u92b7\uff0c\u4e0d\u8981\u518d\u7528\u5b83\u5011\u6e1b\u78bc",
            "30 \u5206\u9418\u6642\u69fd\u52dd\u7387\u2014\u2014\u8207\u96dc\u8a0a\u7121\u6cd5\u5340\u5206",
            "\u9031\u4e00\u6700\u5f31\u3001\u9031\u4e94\u6700\u5f37\u2014\u2014\u5728\u9019\u4efd\u8cc7\u6599\u4e0a\u4e0d\u6210\u7acb",
            "CFTC \u90e8\u4f4d\u2014\u2014\u76ee\u524d\u6a23\u672c\u770b\u4e0d\u51fa\u4efb\u4f55\u6771\u897f",
        ],
    },
    "TX": {
        "intro": "TX \u76ee\u524d\u53ea\u6709\u5b63\u7bc0\u6027\u8207\u56de\u6a94\u7d50\u69cb\u7684"
                 "\u521d\u6b65\u7814\u7a76\uff0c\u9084\u6c92\u6709\u8a0a\u865f\u5c64\u7d1a\u7684"
                 "\u5224\u65b7\u4f9d\u64da\u3002",
        "checks": [],
        "ruled_out": [],
    },
}


def signal_playbook_html(market: str, study_list: list[dict[str, object]], prefix: str) -> str:
    """What to look at when a signal arrives — the reason the owner opens this site.

    Everything else here is an archive organised for browsing. This is the one page with a
    task: a signal just fired, and the question is whether anything known raises or lowers
    confidence in taking it. It leads with the single finding that survived every screen,
    and it says in the same breath that the finding cannot be traded as a filter, because
    a number that raises win rate while destroying return is worse than useless if it is
    presented without that.

    The "already ruled out" list is here for the same reason: knowing what not to bother
    checking is a decision aid, and it is the largest thing this programme has produced.
    """
    book = SIGNAL_PLAYBOOK.get(market)
    if not book:
        return ""
    by_id = {str(study["id"]): study for study in study_list}

    checks = []
    for item in book["checks"]:
        study = by_id.get(item["study"])
        link = (
            f'<a href="{html.escape(prefix + study["_relative"])}/">{html.escape(item["study"])}</a>'
            if study else html.escape(item["study"])
        )
        checks.append(
            '<article class="insight good">'
            f'<strong>{html.escape(item["title"])}</strong>'
            f'<p>{html.escape(item["what"])}</p>'
            f'<p class="section-note">{html.escape(item["caveat"])}</p>'
            f'<p class="section-note">{link}</p>'
            "</article>"
        )
    ruled = "".join(f"<li>{html.escape(x)}</li>" for x in book["ruled_out"])
    ruled_block = (
        '<section class="report-section"><h2>\u4e0d\u7528\u518d\u770b\u7684</h2>'
        f'<ul class="impact-list">{ruled}</ul>'
        f'<p class="section-note">\u5b8c\u6574\u6e05\u55ae\u8207\u6bcf\u4e00\u9805\u7684'
        f'\u89e3\u6790\u4e0b\u9650\uff1a<a href="{prefix}lessons/">\u4ec0\u9ebc\u6c92\u7528</a></p>'
        "</section>"
    ) if ruled else ""

    return (
        '<section class="report-section"><h2>\u8a0a\u865f\u4f86\u4e86\uff0c\u5148\u770b\u9019\u88e1</h2>'
        f'<p>{html.escape(book["intro"])}</p>'
        + (f'<div class="insight-grid">{"".join(checks)}</div>' if checks
           else '<p class="section-note">\u5c1a\u7121\u53ef\u7528\u7684\u5224\u65b7\u4f9d\u64da\u3002</p>')
        + "</section>"
        + ruled_block
    )


def xauusd_page(study_list: list[dict[str, object]], weekly: list[dict[str, object]]) -> str:
    selected = [study for study in study_list if study["market"].lower() == "xauusd"]
    latest = weekly[0] if weekly else None
    weekly_html = (
        weekly_card(latest, "weekly/")
        if latest else '<p class="empty">No reviewed weekly outlook published yet.</p>'
    )
    body = (
        '<main class="shell">'
        + signal_playbook_html("XAUUSD", study_list, "../")
        + '<h2 class="section-title">\u672c\u9031\u5c55\u671b</h2>'
        f'<div class="grid">{weekly_html}</div>'
        + '<h2 class="section-title">XAUUSD \u7814\u7a76 '
        f'<span class="sheet-count">({len(selected)})</span></h2>'
        + '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="\u7be9\u9078 XAUUSD \u7814\u7a76" aria-label="Filter"></div></div>'
        + view_toggle_html()
        + theme_sheets_html(selected, "../")
        + study_table_html(selected, "../")
        + "</main>"
    )
    return document(
        "XAUUSD",
        "\u9ec3\u91d1",
        "\u8a0a\u865f\u4f86\u4e86\u8981\u770b\u4ec0\u9ebc\u3001\u672c\u9031\u5c55\u671b\uff0c"
        "\u4ee5\u53ca\u6240\u6709\u9ec3\u91d1\u7814\u7a76\u3002",
        body,
        "../",
    )


def section_page(
    study_list: list[dict[str, object]],
    market: str,
    title: str,
    lede: str,
) -> str:
    selected = [study for study in study_list if study["market"].lower() == market]
    body = (
        '<main class="shell">'
        + signal_playbook_html(market.upper(), study_list, "../")
        + f'<h2 class="section-title">{html.escape(market.upper())} \u7814\u7a76 '
        f'<span class="sheet-count">({len(selected)})</span></h2>'
        + '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="\u7be9\u9078\u7814\u7a76" aria-label="Filter"></div></div>'
        + view_toggle_html()
        + theme_sheets_html(selected, "../")
        + study_table_html(selected, "../")
        + "</main>"
    )
    return document(title, market, lede, body, "../")


def lessons_page(registry, study_list) -> str:
    """What was ruled out, and what the programme learned about testing.

    Split off from the instrument pages because it is the one section that genuinely spans
    both of them, and because it answers a different question: not "should I take this
    trade" but "has this already been tried". Those are different visits and putting them
    on the same page made each harder to find.
    """
    methodology = [s for s in study_list if s.get("theme") == "methodology"]
    cards = "".join(study_card(s, "../") for s in methodology)
    totals = (registry or {}).get("totals", {})
    banner = ""
    if totals:
        banner = (
            '<a class="card" data-card href="../research/null-results/">'
            '<div class="type">registry</div><h2>\u5b8c\u6574\u767b\u9304</h2>'
            "<p>\u6bcf\u4e00\u500b\u88ab\u554f\u904e\u4e26\u4e14\u5f97\u5230"
            "\u300c\u6c92\u6709\u300d\u7684\u554f\u984c\uff0c\u6bcf\u4e00\u7b46"
            "\u90fd\u5e36\u8457\u5b83\u7684\u89e3\u6790\u4e0b\u9650\u3002</p>"
            '<div class="mini-metrics">'
            f'<span><strong>{totals.get("hypotheses", 0)}</strong> hypotheses</span>'
            f'<span><strong>{totals.get("by_verdict", {}).get("survives_screens", 0)}</strong> survivors</span>'
            "</div></a>"
        )
    body = (
        '<main class="shell">'
        '<section class="report-section"><h2>\u70ba\u4ec0\u9ebc\u9019\u9801\u5b58\u5728</h2>'
        "<p>\u9019\u500b\u7814\u7a76\u8a08\u756b\u7684\u7522\u51fa\u5927\u90e8\u5206"
        "\u662f\u300c\u6c92\u6709\u300d\u3002\u90a3\u662f\u7d50\u8ad6\uff0c\u4e0d\u662f"
        "\u7f3a\u5c11\u7d50\u8ad6\u2014\u2014\u800c\u4e14\u77e5\u9053\u4ec0\u9ebc"
        "\u4e0d\u7528\u518d\u8a66\uff0c\u672c\u8eab\u5c31\u662f\u5224\u65b7\u4f9d\u64da\u3002</p>"
        "<p>\u6bcf\u4e00\u7b46\u90fd\u5e36\u8457<strong>\u89e3\u6790\u4e0b\u9650</strong>\uff1a"
        "\u9019\u500b\u6a23\u672c\u80fd\u5206\u8fa8\u7684\u6700\u5c0f\u5dee\u8ddd\u3002"
        "\u754c\u9650\u5bec\u7684\u300c\u7121\u8b49\u64da\u300d\u4ec0\u9ebc\u90fd"
        "\u6c92\u95dc\u6389\uff0c\u9019\u500b\u5340\u5225\u5f88\u91cd\u8981\u3002</p>"
        "</section>"
        + (f'<div class="grid">{banner}</div>' if banner else "")
        + (f'<h2 class="section-title">\u65b9\u6cd5\u8ad6 '
           f'<span class="sheet-count">({len(methodology)})</span></h2>'
           f'<p class="section-note">\u95dc\u65bc\u300c\u600e\u9ebc\u6e2c\u300d\u5b78\u5230'
           f'\u7684\u4e8b\uff0c\u901a\u5e38\u662f\u5148\u505a\u932f\u4e00\u6b21\u624d\u5b78\u5230\u7684\u3002</p>'
           f'<div class="grid study-grid">{cards}</div>' if methodology else "")
        + "</main>"
    )
    return document(
        "\u4ec0\u9ebc\u6c92\u7528",
        "negative results",
        "\u88ab\u6392\u9664\u7684\u641c\u5c0b\u7a7a\u9593\uff0c\u4ee5\u53ca\u6bcf\u4e00\u500b"
        "\u300c\u6c92\u6709\u300d\u5230\u5e95\u95dc\u6389\u4e86\u591a\u5c11\u9580\u3002",
        body,
        "../",
    )


def research_page(data: dict[str, object], study_list: list[dict[str, object]]) -> str:
    registry = null_registry()
    banner = ""
    if registry:
        totals = registry["totals"]
        # Placed above the study grid on purpose. The studies read as a list of things that
        # were tried; the registry is the only place that says how hard, and it is the
        # first thing worth knowing before asking a question of this data again.
        banner = (
            '<a class="card" data-card href="null-results/">'
            '<div class="type">registry</div><h2>What did not work</h2>'
            "<p>Every question asked of this data and answered with no, each carrying the "
            "smallest effect its sample could have resolved. Published as JSON so anything "
            "automated can skip what is already closed.</p>"
            '<div class="mini-metrics">'
            f'<span><strong>{totals["hypotheses"]}</strong> hypotheses</span>'
            f'<span><strong>{totals["by_verdict"].get("survives_screens", 0)}</strong> survivors</span>'
            f'<span><strong>{totals["studies"]}</strong> studies</span>'
            "</div></a>"
        )
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter studies" aria-label="Filter"></div></div>'
        f'<main class="shell">'
        + (f'<h2 class="section-title">Start here</h2><div class="grid">{banner}</div>'
           if banner else "")
        + view_toggle_html()
        + theme_sheets_html(study_list)
        + study_table_html(study_list)
        + "</main>"
    )
    return document(
        "Research studies",
        "Evidence → decision → workflow",
        "Reviewed studies preserve the question, reproducible method, aggregate result, and operational impact without publishing raw CSV or private conversation.",
        body,
        "../",
    )


def overview(
    data: dict[str, object],
    study_list: list[dict[str, object]],
    weekly: list[dict[str, object]],
) -> str:
    """The front door, organised by what the reader came to do.

    The previous homepage offered "XAUUSD" and "Research" as sibling entries, and a study
    about gold lived under the second one — so there were two plausible doors to the same
    thing and no way to tell which. The owner said as much: 我都不知道要點哪裡.

    The fix is that instruments are the top level and everything about an instrument lives
    inside it. What stays at this level is the one thing that spans both — what has been
    ruled out — and the vocabulary needed to read any of it.

    The ordering is the owner's actual journey, not the archive's structure: a signal
    arrives, and the first question is whether anything known makes this trade better or
    worse. That gets the largest card.
    """
    registry = null_registry()
    totals = (registry or {}).get("totals", {})
    counts = {}
    for study in study_list:
        counts[str(study.get("market", "")).upper()] = counts.get(
            str(study.get("market", "")).upper(), 0) + 1

    weekly_line = (
        f'{weekly[0]["forecast_week"]}\u5c55\u671b\u5df2\u767c\u5e03'
        if weekly else "\u5c1a\u672a\u767c\u5e03\u9031\u5831"
    )
    primary = (
        '<a class="card card-wide" data-card href="xauusd/">'
        '<div class="type">\u8a0a\u865f\u4f86\u4e86</div>'
        '<h2>XAUUSD \u9ec3\u91d1</h2>'
        "<p>\u9032\u5834\u524d\u5148\u770b\uff1a\u5e03\u6797\u4f4d\u7f6e\u7684\u6b77\u53f2"
        "\u52dd\u7387\u3001\u7576\u65e5\u9084\u5269\u591a\u5c11\u7a7a\u9593\u3001"
        "\u4ee5\u53ca\u54ea\u4e9b\u6771\u897f\u5df2\u7d93\u78ba\u5b9a\u6c92\u7528\u3002</p>"
        '<div class="mini-metrics">'
        f'<span><strong>{counts.get("XAUUSD", 0)}</strong> \u7bc7\u7814\u7a76</span>'
        f'<span><strong>{html.escape(weekly_line)}</strong></span>'
        "</div></a>"
        '<a class="card card-wide" data-card href="tx/">'
        '<div class="type">\u53e6\u4e00\u500b\u5546\u54c1</div>'
        '<h2>TX \u53f0\u6307\u671f</h2>'
        "<p>\u76ee\u524d\u53ea\u6709\u5b63\u7bc0\u6027\u8207\u56de\u6a94\u7d50\u69cb\u7684"
        "\u521d\u6b65\u7814\u7a76\uff0c\u9084\u6c92\u6709\u8a0a\u865f\u5c64\u7d1a\u7684"
        "\u5224\u65b7\u4f9d\u64da\u3002</p>"
        '<div class="mini-metrics">'
        f'<span><strong>{counts.get("TX", 0)}</strong> \u7bc7\u7814\u7a76</span>'
        "</div></a>"
    )

    secondary = [
        ("lessons/", "\u4ec0\u9ebc\u6c92\u7528",
         f'{totals.get("hypotheses", 0)} \u500b\u5047\u8a2d\u88ab\u6e2c\u904e\uff0c'
         f'{totals.get("by_verdict", {}).get("survives_screens", 0)} \u500b\u5b58\u6d3b\u3002'
         "\u77e5\u9053\u4ec0\u9ebc\u4e0d\u7528\u518d\u8a66\uff0c\u672c\u8eab\u5c31\u662f"
         "\u5224\u65b7\u4f9d\u64da\u3002"),
        ("glossary/", "\u8853\u8a9e\u8868",
         "\u7814\u7a76\u9801\u9762\u662f\u82f1\u6587\u7684\u3002\u9019\u88e1\u628a"
         "\u6bcf\u500b\u6280\u8853\u540d\u8a5e\u7528\u4e2d\u6587\u5b9a\u7fa9\u4e00\u6b21\u3002"),
        ("xauusd/weekly/", "\u9031\u5831",
         "\u6bcf\u9031\u7684\u95dc\u9375\u50f9\u4f4d\u3001\u5287\u672c\u8207\u4e8b\u4ef6\u98a8\u96aa\u3002"),
    ]
    minor = '<div class="grid">' + "".join(
        f'<a class="card" data-card href="{href}"><div class="type">\u53c3\u8003</div>'
        f'<h2>{title}</h2><p>{description}</p></a>'
        for href, title, description in secondary
    ) + "</div>"

    body = (
        '<main class="shell">'
        '<h2 class="section-title">\u4f60\u4ea4\u6613\u7684\u5546\u54c1</h2>'
        f'<div class="grid">{primary}</div>'
        '<h2 class="section-title">\u5176\u4ed6</h2>'
        f"{minor}"
        '<section class="report-section"><h2>\u9019\u500b\u7db2\u7ad9\u662f\u4ec0\u9ebc</h2>'
        "<p>\u4e00\u500b\u4ea4\u6613\u7814\u7a76\u8a08\u756b\u7684\u516c\u958b\u90e8\u5206\u3002"
        "\u5b83\u4e0d\u662f\u7d66\u5efa\u8b70\u7684\uff0c\u4e5f\u4e0d\u662f\u8a0e\u8ad6"
        "\u4ec0\u9ebc\u7b56\u7565\u6709\u6548\u2014\u2014\u5b83\u8a18\u9304\u7684\u662f"
        "\u54ea\u4e9b\u554f\u984c\u88ab\u554f\u904e\u3001\u7b54\u6848\u662f\u4ec0\u9ebc\uff0c"
        "\u4ee5\u53ca\u90a3\u500b\u7b54\u6848\u6709\u591a\u53ef\u9760\u3002</p>"
        "<p>\u5927\u90e8\u5206\u7684\u7b54\u6848\u662f\u300c\u6c92\u6709\u300d\u3002</p>"
        "</section>"
        "</main>"
    )
    return document(
        "\u4ea4\u6613\u7814\u7a76",
        "\u516c\u958b\u5de5\u4f5c\u5340",
        "\u8a0a\u865f\u4f86\u4e86\u8981\u770b\u4ec0\u9ebc\uff0c\u4ee5\u53ca\u54ea\u4e9b"
        "\u6771\u897f\u5df2\u7d93\u78ba\u5b9a\u6c92\u7528\u3002",
        body,
    )


def outputs(data: dict[str, object]) -> dict[Path, str]:
    study_list = studies()
    weekly = weekly_summaries()
    generated = {
        ROOT / "site/catalog.json": json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        ROOT / "index.html": overview(data, study_list, weekly),
        ROOT / "xauusd/index.html": xauusd_page(study_list, weekly),
        ROOT / "tx/index.html": section_page(
            study_list, "tx", "TX \u53f0\u6307\u671f", "\u53f0\u6307\u671f\u7814\u7a76\u3002"),
        ROOT / "research/index.html": research_page(data, study_list),
        ROOT / "lessons/index.html": lessons_page(null_registry(), study_list),
    }
    registry = null_registry()
    if registry:
        generated[ROOT / "research/null-results/index.html"] = null_results_page(registry)
    terms = glossary()
    if terms:
        generated[ROOT / "glossary/index.html"] = glossary_page(terms)
    if weekly:
        generated[ROOT / "xauusd/weekly/index.html"] = weekly_summary_page(
            weekly[0], weekly, prefix="../../", source_href=f'{weekly[0]["forecast_week"]}/summary.json', latest=True,
        )
        for summary in weekly:
            generated[ROOT / summary["_relative"] / "index.html"] = weekly_summary_page(
                summary, weekly, prefix="../../../", source_href="summary.json", latest=False,
            )
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
