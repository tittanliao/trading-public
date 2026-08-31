#!/usr/bin/env python3
"""Public site generator — clean rebuild (docs/PUBLIC_SITE_REBUILD_SPEC.md in trading-private).

Weekly-first, Research-second. Renders the canonical route allow-list below from reviewed
study.json/results.json data and published Weekly summary.json files — nothing else.

There is no per-result-shape renderer: every research page is composed from a small, named
vocabulary of presentation blocks (see `render_block`), driven by an ordered `presentation`
list stored in the study's own study.json. A new result shape is handled by writing a new
block list, not a new Python function.

Two conventions every table on this site inherits, so they cannot regress per-page:
  * every table is sortable by clicking a column header (`data-sort` carries the raw value,
    so formatted text like "1,234" or "[a, b]" never has to be re-parsed);
  * tables size to the available width and only scroll horizontally when the viewport is
    genuinely too narrow — a wide screen must never require sideways scrolling.

Usage:
    python3 site/build.py            write mode
    python3 site/build.py --check    verify current output matches; non-zero on drift
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

POC_STUDIES = [
    "RS-XAUUSD-20260727-001",
    "RS-XAUUSD-20260727-005",
    "RS-XAUUSD-20260727-007",
    "RS-XAUUSD-20260818-001",
    "RS-XAUUSD-20260823-001",
    "RS-XAUUSD-20260823-002",
]

# Every published Weekly edition keeps its dated archive page. This is not optional
# decoration: docs/BASE_WEEKLY_REPORT_WORKFLOW.md declares /xauusd/weekly/<week>/ a stable
# address, and Private release receipts record those exact URLs as proof of publication.
# The first cutover deleted the W34/W35 pages and broke the W35 receipt's URL; an
# independent review caught it. Discovering editions from the data keeps them in step.
def weekly_weeks() -> list[str]:
    weeks = sorted(p.parent.name for p in (ROOT / "xauusd/weekly").glob("*/summary.json"))
    return weeks


def latest_week() -> str:
    return weekly_weeks()[-1]


def routes() -> list[str]:
    return [
        "",
        "xauusd/weekly",
        *(f"xauusd/weekly/{w}" for w in weekly_weeks()),
        "research",
        "research/null-results",
        *(f"research/studies/{sid}" for sid in POC_STUDIES),
    ]


# Studies queued for a later migration phase (docs/PUBLIC_SITE_REBUILD_SPEC.md section 10).
# Named here only so the Research index can report the queue honestly without linking to a
# route that does not exist yet. All three phases are listed: reporting only 2A and 2B
# under-counted the queue by 13 studies in the first cutover.
PHASE_2A = [
    "RS-XAUUSD-20260825-001", "RS-XAUUSD-20260827-001",
]
PHASE_2B = [
    "RS-TX-20260728-001", "RS-TX-20260728-002", "RS-XAUUSD-20260727-003",
    "RS-XAUUSD-20260727-004", "RS-XAUUSD-20260727-006", "RS-XAUUSD-20260727-008",
    "RS-XAUUSD-20260815-001", "RS-XAUUSD-20260815-002", "RS-XAUUSD-20260815-003",
    "RS-XAUUSD-20260817-001",
]
PHASE_2C = [
    "RS-XAUUSD-20260727-002", "RS-XAUUSD-20260815-004", "RS-XAUUSD-20260818-002",
    "RS-XAUUSD-20260818-003", "RS-XAUUSD-20260818-004", "RS-XAUUSD-20260818-005",
    "RS-XAUUSD-20260819-001", "RS-XAUUSD-20260824-001", "RS-XAUUSD-20260824-002",
    "RS-XAUUSD-20260824-003", "RS-XAUUSD-20260824-004", "RS-XAUUSD-20260824-005",
    "RS-XAUUSD-20260824-006",
]
QUEUED_STUDIES = PHASE_2A + PHASE_2B + PHASE_2C


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def study_dir(study_id: str) -> Path:
    return ROOT / "research/studies" / study_id


def load_study(study_id: str) -> tuple[dict, dict]:
    d = study_dir(study_id)
    return load_json(d / "study.json"), load_json(d / "results.json")


def resolve_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def resolve_source(study: dict, results: dict, source: str) -> Any:
    """study.json first (reviewed narrative), then results.json (reproducible data), so a
    block never has to say which file its data lives in."""
    value = resolve_path(study, source)
    if value is None:
        value = resolve_path(results, source)
    return value


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        text = f"{value:,.4f}".rstrip("0").rstrip(".")
        return text if text and text != "-" else "0"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return f"[{fmt_cell(value[0])}, {fmt_cell(value[1])}]"
    if isinstance(value, list):
        return ", ".join(fmt_cell(v) for v in value)
    return esc(value)


def sort_key(value: Any) -> str:
    """Raw comparable value for a cell, emitted as data-sort so the client never has to
    re-parse formatted text. Numbers sort numerically; a [lo, hi] interval sorts by its
    lower bound; everything else sorts as lowercased text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(float(value))
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return repr(float(value[0]))
    return str(value).lower()


def is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, list) and bool(value) and isinstance(value[0], (int, float))


PREFERRED_COLUMNS = [
    "n", "wins", "win_rate_pct", "win_rate_ci95_pct", "profit_factor",
    "net_pnl_usd", "avg_pnl_usd", "avg_return_pct", "total_return_pct",
    "total_return_delta_pct", "win_rate_delta_pp", "rank_score",
    "count", "pct", "diff",
    "v34_n", "v34_win_rate_pct", "v34_profit_factor", "v34_pct",
    "v39_n", "v39_win_rate_pct", "v39_profit_factor", "v39_pct",
    "win_rate_pct_diff", "profit_factor_diff", "win_rate_pct_diff_v39_minus_v34",
    "scale", "filter_total_r", "share_of_baseline_pct", "preferred",
]
DROPPED_COLUMNS = {"low_sample", "rank_excluded_reason"}


def flatten_record(record: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if key in DROPPED_COLUMNS:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key not in DROPPED_COLUMNS:
                    out[f"{key}.{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


def ordered_columns(records: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for record in records:
        for key in record:
            seen.setdefault(key, None)
    columns = [c for c in PREFERRED_COLUMNS if c in seen]
    columns += [c for c in seen if c not in columns]
    return columns


def label_col(key: str) -> str:
    return esc(key.replace("_pct", " %").replace("_", " "))


def render_table(headers: list[str], rows: list[list[tuple[str, str, bool]]],
                 caption: str = "") -> str:
    """One table renderer for the whole site. `rows` is a list of rows; each cell is
    (display_html, sort_value, numeric). Every table is sortable and width-adaptive."""
    head = "".join(
        f'<th scope="col" data-type="{"number" if numeric else "text"}">'
        f'<button type="button" class="sort-btn">{label}<span class="sort-ind"></span></button></th>'
        for label, numeric in headers
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<{tag} data-sort="{attr(sortv)}"{" class=\"num\"" if numeric else ""}>{display}</{tag}>'
            for tag, display, sortv, numeric in row
        )
        body.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return (
        '<div class="table-wrap"><table data-sortable>' + cap
        + "<thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def render_records_table(records: dict[str, dict] | list[dict], row_label: str = "",
                         columns: list[str] | None = None,
                         record_keys: list[str] | None = None) -> str:
    if isinstance(records, dict):
        raw = [
            (key, flatten_record(value)) for key, value in records.items()
            if isinstance(value, dict) and (record_keys is None or key in record_keys)
        ]
    else:
        raw = [(str(i), flatten_record(value)) for i, value in enumerate(records) if isinstance(value, dict)]
    if not raw:
        return '<p class="empty-note">No rows.</p>'
    available_columns = ordered_columns([r for _, r in raw])
    columns = [column for column in columns if column in available_columns] if columns else available_columns
    # A dict source labels each row with its own key. A list source has no such key, so the
    # position index was being used — which renders a meaningless 0,1,2… column beside data
    # that already identifies itself (a slot, a decile, a threshold). When a list source
    # names its columns, promote the first one to the row label instead.
    if not isinstance(records, dict) and columns:
        label_column = columns[0]
        raw = [(fmt_cell(record.get(label_column)), record) for _, record in raw]
        columns = columns[1:]
        if not row_label:
            row_label = label_column.replace("_pct", " %").replace("_", " ")
    numeric_col = {
        c: any(is_numeric(record.get(c)) for _, record in raw) for c in columns
    }
    headers = [(esc(row_label), False)] + [(label_col(c), numeric_col[c]) for c in columns]
    rows = []
    for key, record in raw:
        cells = [("th", esc(key).replace("_", " "), key.lower(), False)]
        for c in columns:
            v = record.get(c)
            cells.append(("td", fmt_cell(v), sort_key(v), numeric_col[c]))
        rows.append(cells)
    return render_table(headers, rows)


# ---------------------------------------------------------------------------
# Presentation blocks
# ---------------------------------------------------------------------------

def block_metrics(value: dict, keys: list[str] | None = None) -> str:
    if keys:
        value = {key: value[key] for key in keys if key in value}
    items = "".join(
        f'<div class="metric"><div class="metric-value">{fmt_cell(v)}</div>'
        f'<div class="metric-label">{label_col(k)}</div></div>'
        for k, v in value.items()
    )
    klass = " metrics-featured" if keys else ""
    return f'<div class="metrics-strip{klass}">{items}</div>'


def block_findings(value: list[dict]) -> str:
    cards = "".join(
        f'<article class="finding-card"><h3>{esc(item.get("title_zh") or item.get("title", ""))}</h3>'
        f'<p>{esc(item.get("detail_zh") or item.get("detail", ""))}</p></article>'
        for item in value
    )
    return f'<div class="findings-grid">{cards}</div>'


def block_table(value: Any, title: str, columns: list[str] | None = None,
                record_keys: list[str] | None = None, takeaway: str = "") -> str:
    heading = f"<h3>{esc(title)}</h3>" if title else ""
    # A takeaway above the table gives a chartless study the same reading pattern as an
    # evidence_pair: Chinese context first, then the evidence. 17 of the 31 studies have no
    # charts at all, so this is the normal case for the rest of the migration, not a
    # special case — without it those pages would be a wall of unexplained tables.
    lead = f'<p class="evidence-takeaway">{esc(takeaway)}</p>' if takeaway else ""
    return f'<section class="data-block">{heading}{lead}{render_records_table(value, columns=columns, record_keys=record_keys)}</section>'


def block_prose(value: str) -> str:
    return f'<p class="prose">{esc(value)}</p>'


def block_charts(value: list[dict]) -> str:
    """Charts render full width and legible without a click. The link to the original PNG
    stays for anyone who wants to zoom further, but reading the report must not require it
    — the first cutover showed thumbnails below every table, which inverted the point of a
    visual report."""
    figures = []
    for chart in value:
        file_name = chart.get("file", "")
        title = chart.get("caption") or chart.get("title", file_name)
        href = f"charts/{file_name}"
        figures.append(
            f'<figure class="chart-figure">'
            f'<img src="{attr(href)}" alt="{attr(title)}" loading="lazy">'
            f'<figcaption>{esc(title)} '
            f'<a class="chart-full" href="{attr(href)}">開啟原圖 ↗</a></figcaption></figure>'
        )
    return f'<div class="chart-gallery">{"".join(figures)}</div>'


def block_metric_table(value: dict, title: str, keys: list[str]) -> str:
    rows = [[
        ("th", esc(label_col(key)), key.lower(), False),
        ("td", fmt_cell(value.get(key)), sort_key(value.get(key)), is_numeric(value.get(key))),
    ] for key in keys if key in value]
    if not rows:
        return ""
    return f'<section class="data-block compact-metric-table"><h3>{esc(title)}</h3>' \
           f'{render_table([("項目", False), ("數值", True)], rows)}</section>'


def block_evidence_pair(block: dict, study: dict, results: dict) -> str:
    """A research evidence unit keeps the chart immediately beside the table it explains.

    The chart is looked up by file name from results.json, so presentation metadata never
    duplicates captions or gives the page a second, editable version of evidence.
    """
    file_name = block.get("chart_file", "")
    chart = next((item for item in results.get("charts", []) if item.get("file") == file_name), None)
    value = resolve_source(study, results, block.get("table_source", ""))
    if not chart or value is None:
        return ""
    caption = chart.get("caption") or chart.get("title") or file_name
    href = f"charts/{file_name}"
    title = block.get("title", "")
    takeaway = block.get("takeaway_zh", "")
    table_title = block.get("table_title", "")
    return f'''<section class="evidence-unit">
<h2>{esc(title)}</h2>
<p class="evidence-takeaway">{esc(takeaway)}</p>
<figure class="chart-figure evidence-figure"><img src="{attr(href)}" alt="{attr(caption)}" loading="lazy">
<figcaption>{esc(caption)} <a class="chart-full" href="{attr(href)}">開啟原圖 ↗</a></figcaption></figure>
{block_table(value, table_title, block.get("columns"), block.get("record_keys"))}</section>'''


def block_limitations(value: list[str]) -> str:
    items = "".join(f"<li>{esc(item)}</li>" for item in value)
    return f'<ul class="limitations-list">{items}</ul>'


def block_evidence_links() -> str:
    links = [
        ("analysis.py — Method", "analysis.py"),
        ("results.json — Results", "results.json"),
        ("study.json — Study manifest", "study.json"),
    ]
    items = "".join(f'<li><a href="{href}">{esc(label)}</a></li>' for label, href in links)
    return f'<ul class="evidence-links">{items}</ul>'


BLOCK_TITLES = {
    "metrics": "Headline Metrics 關鍵數字",
    "findings": "Key Findings 重點發現",
    "charts": "Charts 圖表",
    "interpretation": "詮釋與實務意義",
    "limitations": "限制與注意事項",
    "evidence_links": "Evidence 原始證據",
}
SECTION_BLOCKS = {"metrics", "findings", "charts", "interpretation", "limitations", "evidence_links"}


def render_block(block: dict, study: dict, results: dict) -> str:
    kind = block["type"]
    title = block.get("title", BLOCK_TITLES.get(kind, ""))
    if kind == "evidence_pair":
        return block_evidence_pair(block, study, results)
    if kind == "evidence_links":
        body = block_evidence_links()
    else:
        value = resolve_source(study, results, block.get("source", ""))
        if value is None:
            return ""
        if kind == "metrics":
            body = block_metrics(value, block.get("keys"))
        elif kind == "findings":
            body = block_findings(value)
        elif kind in ("table", "comparison_table", "matrix_table"):
            return block_table(value, title, block.get("columns"), block.get("record_keys"),
                               block.get("takeaway_zh", ""))
        elif kind == "metric_table":
            return block_metric_table(value, title, block.get("keys", []))
        elif kind in ("prose", "interpretation"):
            body = block_prose(value)
        elif kind == "charts":
            body = block_charts(value)
        elif kind == "limitations":
            body = block_limitations(value)
        else:
            raise ValueError(f"unknown presentation block type: {kind}")
    if kind in SECTION_BLOCKS:
        heading = f"<h2>{esc(title)}</h2>" if title else ""
        return f'<section class="block-section">{heading}{body}</section>'
    return body


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------

SITE_NAME = "Trading Research"
SITE_NAME_ZH = "交易研究站"


def nav(depth: int) -> str:
    prefix = "../" * depth
    links = [
        ("Home", f"{prefix}index.html" if depth else "index.html"),
        ("Weekly", f"{prefix}xauusd/weekly/"),
        ("Research", f"{prefix}research/"),
    ]
    items = "".join(f'<a href="{attr(href)}">{label}</a>' for label, href in links)
    return f'<nav class="nav"><a class="brand" href="{attr(f"{prefix}index.html" if depth else "index.html")}">{SITE_NAME}</a><div class="nav-links">{items}</div></nav>'


def document(title: str, description: str, depth: int, body: str, wide: bool = False) -> str:
    prefix = "../" * depth
    shell = "shell wide" if wide else "shell"
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{attr(description)}">
<title>{esc(title)} · {SITE_NAME}</title>
<link rel="icon" href="{attr(prefix)}favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="{attr(prefix)}favicon.svg">
<link rel="stylesheet" href="{attr(prefix)}site/style.css">
</head>
<body>
<header class="{shell}">
{nav(depth)}
</header>
<main class="{shell}">
{body}
</main>
<footer class="{shell}"><p>研究證據，不是交易建議；執行前必須以即時 TradingView 確認。Research evidence, not trading advice.</p></footer>
<script src="{attr(prefix)}site/table.js" defer></script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def home_page() -> str:
    week = latest_week()
    s = load_json(ROOT / "xauusd/weekly" / week / "summary.json")
    zones = "".join(
        f'<li><span class="dir dir-{esc(z["direction"])}">{esc(z["direction"])}</span>'
        f'<strong>{z["probability"]}%</strong>{esc(z["conditions"][:56])}…</li>'
        for z in s["adopted_scenarios"]
    )
    key_zone = s["key_levels"][0]
    cards = []
    for sid in POC_STUDIES:
        study, results = load_study(sid)
        n = len(results.get("charts", []))
        cards.append(
            f'<a class="card" href="research/studies/{sid}/">'
            f'<div class="type">{esc(study.get("market", ""))} · {n} charts</div>'
            f'<h3>{esc(study.get("title", sid))}</h3></a>'
        )
    markets = sorted({load_study(s)[0].get("market", "") for s in POC_STUDIES + QUEUED_STUDIES})
    body = f"""
<section class="hero">
  <p class="eyebrow">{SITE_NAME}</p>
  <h1>{SITE_NAME_ZH}</h1>
  <p class="lede">{esc(" / ".join(m for m in markets if m))} 的週度展望與策略研究證據庫。
  不發訊號，只記錄問了什麼問題、答案是什麼、以及那個答案能相信到什麼程度。</p>
</section>

<section class="feature-card">
  <div class="type">Latest Weekly Report</div>
  <h2><a href="xauusd/weekly/{week}/">{esc(week)} 週報</a></h2>
  <p>{esc(s["recommendation"]["summary"][:150])}…</p>
  <ul class="scenario-preview">{zones}</ul>
  <p class="key-zone">關鍵決策帶　<strong>{esc(key_zone["value"])}</strong>　<span class="muted">{esc(key_zone["label"])}</span></p>
  <p class="more"><a href="xauusd/weekly/{week}/">閱讀完整週報 →</a></p>
</section>

<section class="block-section">
  <h2>Featured Research</h2>
  <div class="card-grid">{"".join(cards)}</div>
  <p class="more"><a href="xauusd/weekly/">所有週報 →</a>　<a href="research/">研究索引 →</a></p>
</section>
"""
    return document(SITE_NAME_ZH, "XAUUSD and TX weekly outlook and research evidence library.", 0, body)


# ---------------------------------------------------------------------------
# Weekly
# ---------------------------------------------------------------------------

def weekly_index_page() -> str:
    weeks = sorted(weekly_weeks(), reverse=True)   # newest first
    latest = weeks[0]
    s = load_json(ROOT / "xauusd/weekly" / latest / "summary.json")
    headers = [("Forecast Week", False), ("Market", False), ("Published", False),
               ("Mode", False), ("Confidence", False)]
    rows = []
    for w in weeks:
        d = load_json(ROOT / "xauusd/weekly" / w / "summary.json")
        published = str(d.get("published_at", ""))[:10]
        rows.append([
            ("th", f'<a href="{attr(w)}/">{esc(w)}</a>', w.lower(), False),
            ("td", esc(d.get("market", "")), str(d.get("market", "")).lower(), False),
            ("td", esc(published), published, False),
            ("td", esc(d.get("publication_mode", "")), str(d.get("publication_mode", "")).lower(), False),
            ("td", esc(d.get("confidence", "")), str(d.get("confidence", "")).lower(), False),
        ])
    body = f"""
<h1>Weekly</h1>
<p class="lede">依時間排列的週報，最新的在最上面。每一期都是由當時所有合格的獨立報告逐項比對後，
發布的單一審閱彙整版。點欄位標題可重新排序。</p>
<section class="feature-card">
  <div class="type">Latest</div>
  <h2><a href="{latest}/">{esc(latest)}</a></h2>
  <p>{esc(s["market_summary"][:170])}…</p>
  <p class="more"><a href="{latest}/">閱讀完整週報 →</a></p>
</section>
{render_table(headers, rows)}
"""
    return document("Weekly", "XAUUSD weekly outlook archive.", 2, body)


def weekly_report_page(week: str) -> str:
    s = load_json(ROOT / "xauusd/weekly" / week / "summary.json")

    # CFTC positioning evidence, when the edition published one. The image lets a reader
    # verify the figures quoted in the market summary rather than trust the transcription.
    cftc = ""
    if s.get("cftc_evidence"):
        ev = s["cftc_evidence"]
        cftc = (
            '<section class="block-section"><h2>CFTC 部位原始依據</h2>'
            f'<p class="evidence-takeaway">{esc(ev["note"])}</p>'
            '<figure class="chart-figure evidence-figure">'
            f'<img src="{attr(ev["file"])}" alt="CFTC positioning report {attr(ev["report_date"])}" loading="lazy">'
            f'<figcaption>CFTC Commitments of Traders · report date {esc(ev["report_date"])} '
            f'<a class="chart-full" href="{attr(ev["file"])}">開啟原圖 ↗</a>'
            '</figcaption></figure></section>'
        )

    four_week = ""
    if s.get("four_week_overview"):
        headers = [("Week", False), ("Open", True), ("High", True), ("Low", True),
                   ("Close", True), ("Change", False), ("Note", False)]
        rows = []
        for r in s["four_week_overview"]:
            rows.append([
                ("th", esc(r.get("week", "")), str(r.get("week", "")).lower(), False),
                ("td", fmt_cell(r.get("open")), sort_key(r.get("open")), True),
                ("td", fmt_cell(r.get("high")), sort_key(r.get("high")), True),
                ("td", fmt_cell(r.get("low")), sort_key(r.get("low")), True),
                ("td", fmt_cell(r.get("close")), sort_key(r.get("close")), True),
                ("td", esc(r.get("change", "")), str(r.get("change", "")), False),
                ("td", esc(r.get("note", "")), str(r.get("note", "")).lower(), False),
            ])
        four_week = f'<section class="block-section"><h2>四週回顧</h2>{render_table(headers, rows)}</section>'

    sc_headers = [("劇本", False), ("機率", True), ("條件", False), ("失準條件", False), ("目標", False)]
    sc_rows = [[
        ("th", f'<span class="dir dir-{esc(sc["direction"])}">{esc(sc["direction"])}</span>', sc["direction"], False),
        ("td", f'{sc["probability"]}%', sort_key(sc["probability"]), True),
        ("td", esc(sc["conditions"]), sc["conditions"].lower(), False),
        ("td", esc(sc["invalidation"]), sc["invalidation"].lower(), False),
        ("td", esc(sc["targets"]), sc["targets"].lower(), False),
    ] for sc in s["adopted_scenarios"]]

    lv_headers = [("區位", False), ("價位", False), ("依據", False)]
    lv_rows = [[
        ("th", esc(k["label"]), k["label"].lower(), False),
        ("td", esc(k["value"]), k["value"], False),
        ("td", esc(k["basis"]), k["basis"].lower(), False),
    ] for k in s["key_levels"]]

    sp_headers = [("策略", False), ("立場", False), ("Entry", False), ("SL", False), ("Risk", False)]
    sp_rows = [[
        ("th", esc(p["strategy"]), p["strategy"].lower(), False),
        ("td", esc(p["stance"]), p["stance"].lower(), False),
        ("td", esc(p["entry"]), p["entry"].lower(), False),
        ("td", esc(p["stop"]), p["stop"].lower(), False),
        ("td", esc(p["risk"]), p["risk"].lower(), False),
    ] for p in s["strategy_plan"]]

    ev_headers = [("事件", False), ("時間", False), ("處置", False)]
    ev_rows = [[
        ("th", esc(e["name"]), e["name"].lower(), False),
        ("td", esc(e["scheduled_at"]), str(e["scheduled_at"]), False),
        ("td", esc(e["handling"]), e["handling"].lower(), False),
    ] for e in s["event_risk"]]

    cmp_rows = []
    for c in s["scenario_comparison"]:
        cells = [("th", esc(c["producer"].capitalize()), c["producer"], False)]
        for sc in c["scenarios"]:
            cells.append(("td", f'<span class="dir dir-{esc(sc["direction"])}">{esc(sc["direction"])}</span> {sc["probability"]}%',
                          sort_key(sc["probability"]), True))
        cmp_rows.append(cells)
    cmp_headers = [("Producer", False)] + [(f"劇本 {i+1}", True) for i in range(len(cmp_rows[0]) - 1)]

    def bullets(items: list[str]) -> str:
        return "".join(f"<li>{esc(x)}</li>" for x in items)

    body = f"""
<div class="reading-rail weekly-intro">
<p class="eyebrow">{esc(s['forecast_week'])} · {esc(s['edition'])} · {esc(s['publication_mode'])} · confidence {esc(s['confidence'])}</p>
<h1>{esc(s['market'])} {esc(s['forecast_week'])} 週報</h1>
<p class="lede recommendation">{esc(s['recommendation']['summary'])}</p>
<p class="invalidation-note"><strong>轉折條件：</strong>{esc(s['recommendation']['invalidation'])}</p>
<section class="block-section"><h2>市場摘要</h2><p class="prose">{esc(s['market_summary'])}</p>
<p class="data-cutoff">資料截止：{esc(s['data_cutoff'])}</p></section>
</div>
{cftc}
{four_week}
<section class="block-section"><h2>三劇本與機率</h2>{render_table(sc_headers, sc_rows)}</section>
<section class="block-section"><h2>關鍵價位</h2>{render_table(lv_headers, lv_rows)}</section>
<section class="block-section"><h2>S1／S2 計畫</h2>{render_table(sp_headers, sp_rows)}</section>
<section class="block-section"><h2>事件風險</h2>{render_table(ev_headers, ev_rows)}</section>
<section class="block-section"><h2>Producer 劇本機率對照</h2>{render_table(cmp_headers, cmp_rows)}</section>
<section class="block-section reading-rail"><h2>共識</h2><ul class="limitations-list">{bullets(s['agreements'])}</ul></section>
<section class="block-section reading-rail"><h2>分歧</h2><ul class="limitations-list">{bullets(s['disagreements'])}</ul></section>
<section class="block-section reading-rail"><h2>未解決問題與證據限制</h2><ul class="limitations-list">{bullets(s['evidence_limits'])}</ul></section>
<section class="block-section reading-rail"><p class="disclaimer">{esc(s['disclaimer'])}</p></section>
"""
    return document(f"{s['market']} {s['forecast_week']} 週報", s["market_summary"][:150], 3, body, wide=True)


# ---------------------------------------------------------------------------
# Research index
# ---------------------------------------------------------------------------

def research_index_page() -> str:
    records = []
    for sid in POC_STUDIES:
        study, results = load_study(sid)
        n = len(results.get("charts", []))
        records.append({
            "id": sid,
            "title": study.get("title", sid),
            "market": study.get("market", ""),
            "theme": study.get("theme", "—"),
            "published": study.get("created_on", ""),
            "charts": n,
            "evidence": f"{n} charts" if n else "tables",
        })
    records.sort(key=lambda r: r["published"], reverse=True)   # newest first

    # Study ID rather than a publication date: the id is the stable identifier used
    # everywhere else (handoffs, receipts, the null registry), and it already encodes the
    # date. A separate Published column repeated that date without adding anything.
    headers = [("Title", False), ("Market", False), ("Study ID", False),
               ("Theme", False), ("Evidence", True)]
    rows = []
    for r in records:
        rows.append([
            ("th", f'<a href="studies/{r["id"]}/">{esc(r["title"])}</a>', r["title"].lower(), False),
            ("td", f'<span class="tag">{esc(r["market"])}</span>', r["market"].lower(), False),
            ("td", f'<code class="study-id">{esc(r["id"])}</code>', r["id"].lower(), False),
            ("td", esc(r["theme"]), r["theme"].lower(), False),
            ("td", esc(r["evidence"]), sort_key(r["charts"]), True),
        ])
    table = render_table(headers, rows)
    # data-market on the row is what the filter reads; render_table does not know about
    # filtering, so the attribute is injected here rather than complicating that renderer.
    for r in records:
        table = table.replace(
            f'<tr><th data-sort="{attr(r["title"].lower())}"',
            f'<tr data-market="{attr(r["market"])}"><th data-sort="{attr(r["title"].lower())}"',
            1,
        )

    markets = sorted({r["market"] for r in records if r["market"]})
    queued_markets: dict[str, int] = {}
    for sid in QUEUED_STUDIES:
        m = load_json(study_dir(sid) / "study.json").get("market", "")
        queued_markets[m] = queued_markets.get(m, 0) + 1
    all_markets = sorted(set(markets) | set(queued_markets))
    buttons = "".join(
        f'<button type="button" data-filter="{attr(m)}">{esc(m)}</button>' for m in all_markets
    )
    queued_line = "、".join(f"{esc(m)} {n} 篇" for m, n in sorted(queued_markets.items()))
    body = f"""
<h1>Research</h1>
<p class="lede">研究證據庫，涵蓋所有市場，最新的在最上面。點欄位標題可重新排序。</p>
<div class="filter-bar">
  <button type="button" data-filter="all" class="active">All</button>
  {buttons}
  <input type="search" id="research-search" placeholder="搜尋標題、主題…" aria-label="Search research">
</div>
<p class="more"><a href="null-results/">Null Results / 沒有效果的研究 →</a></p>
<div id="research-table">{table}</div>
<p class="empty-note">另有 {len(QUEUED_STUDIES)} 篇研究（{queued_line}）已完成並保留完整證據，
但詳細頁面排在後續遷移階段，本版尚未上線，因此不提供連結。</p>
<script>
(function () {{
  var scope = document.getElementById("research-table");
  if (!scope) return;
  var rows = Array.prototype.slice.call(scope.querySelectorAll("tbody tr"));
  var buttons = document.querySelectorAll(".filter-bar button");
  var search = document.getElementById("research-search");
  var active = "all";
  function apply() {{
    var q = search.value.trim().toLowerCase();
    rows.forEach(function (row) {{
      var market = row.getAttribute("data-market") || "";
      var text = row.textContent.toLowerCase();
      var ok = (active === "all" || market === active) && text.indexOf(q) !== -1;
      row.hidden = !ok;
    }});
  }}
  buttons.forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      buttons.forEach(function (b) {{ b.classList.remove("active"); }});
      btn.classList.add("active");
      active = btn.getAttribute("data-filter");
      apply();
    }});
  }});
  search.addEventListener("input", apply);
}})();
</script>
"""
    return document("Research", "XAUUSD and TX research evidence index.", 1, body, wide=True)


# ---------------------------------------------------------------------------
# Null results
# ---------------------------------------------------------------------------

def null_results_page() -> str:
    registry = load_json(ROOT / "research/null-results/null_results.json")
    entries = registry["entries"]
    totals = registry["totals"]
    hypotheses = [e for e in entries if e.get("kind") == "hypothesis"]
    study_level = [e for e in entries if e.get("kind") != "hypothesis"]
    live = set(POC_STUDIES)

    headers = [("Family", False), ("Claim", False), ("Verdict", False), ("Reason", False)]
    rows = []
    for e in hypotheses:
        sid = e.get("study_id")
        fam = esc(e.get("family", ""))
        if sid in live:
            fam = f'<a href="../studies/{sid}/">{fam}</a>'
        verdict = str(e.get("verdict", ""))
        rows.append([
            ("th", fam, str(e.get("family", "")).lower(), False),
            ("td", esc(e.get("claim", "")), str(e.get("claim", "")).lower(), False),
            ("td", f'<span class="verdict v-{esc(verdict)}">{esc(verdict)}</span>', verdict, False),
            ("td", esc(e.get("reason", "")), str(e.get("reason", "")).lower(), False),
        ])
    verdict_line = "　".join(f"{esc(k)} {v}" for k, v in totals["by_verdict"].items())
    body = f"""
<h1>Null Results / 沒有效果的研究</h1>
<p class="lede">{len(hypotheses)} 個已登記並逐一檢定的假設，每一個都附上該樣本能解析的最小效果。
{verdict_line}。另有 {len(study_level)} 篇研究以完整段落記錄，未逐一列在下表。點欄位標題可重新排序。</p>
{render_table(headers, rows)}
"""
    return document("Null Results", "63 registered null-result hypotheses.", 2, body, wide=True)


# ---------------------------------------------------------------------------
# Study detail
# ---------------------------------------------------------------------------

def study_page(study_id: str) -> str:
    study, results = load_study(study_id)
    blocks = "".join(render_block(b, study, results) for b in study["presentation"])
    question = study.get("question_zh") or study.get("question", "")
    body = f"""
<article class="study-report">
<p class="eyebrow">{esc(study_id)} · {esc(study.get("market", ""))} · {esc(study.get("created_on", ""))}</p>
<h1>{esc(study.get("title", study_id))}</h1>
<p class="lede research-question">{esc(question)}</p>
{blocks}
</article>
"""
    return document(study.get("title", study_id), question[:150], 3, body, wide=True)


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

# Two width tiers. Prose pages stay narrow because long lines are hard to read; table and
# chart pages use `.wide`, which grows with the viewport up to 1680px. The first cutover
# capped everything at 960px and set `white-space: nowrap` on every cell, so a wide screen
# still had to scroll sideways to see columns that would have fit easily. Horizontal
# scrolling is now a genuinely-too-narrow fallback, not the normal reading mode.
STYLE_CSS = """
:root{
  --bg:#0b0d12;--panel:#12151c;--panel2:#171b24;--line:#262b36;--line2:#333a48;
  --text:#e7e9ee;--muted:#9aa3b2;--cyan:#5ec8d8;--good:#6fcf97;--warn:#f2c94c;--bad:#eb5757;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;
  --shell:960px;--shell-wide:1680px;
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:var(--font);line-height:1.65;}
.shell{max-width:var(--shell);margin:0 auto;padding:0 24px;}
.shell.wide{max-width:var(--shell-wide);}
header.shell{padding-top:16px;}
main.shell{padding:24px 24px 72px;}
footer.shell{padding:22px 24px 44px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);}

.nav{display:flex;align-items:center;gap:20px;padding-bottom:14px;border-bottom:1px solid var(--line);flex-wrap:wrap;}
.nav .brand{color:var(--text);font-weight:700;text-decoration:none;letter-spacing:.01em;}
.nav-links{display:flex;gap:18px;font-size:.92rem;}
.nav-links a{color:var(--muted);text-decoration:none;}
.nav-links a:hover{color:var(--cyan);}

h1{font-size:1.85rem;margin:.25em 0 .35em;line-height:1.3;}
h2{font-size:1.22rem;margin:2em 0 .7em;border-top:1px solid var(--line);padding-top:1.1em;}
h3{font-size:1rem;margin:1.4em 0 .5em;color:var(--text);}
.eyebrow{color:var(--cyan);font-size:.76rem;letter-spacing:.05em;text-transform:uppercase;margin:0 0 .4em;}
.lede{color:var(--muted);font-size:1.03rem;max-width:74ch;}
.prose{max-width:78ch;color:var(--text);}
a{color:var(--cyan);}
.muted{color:var(--muted);}
.more{margin-top:1em;}

.hero{padding:18px 0 4px;}
.feature-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;margin:22px 0;}
.feature-card h2{border-top:none;margin:0 0 .5em;padding-top:0;}
.feature-card .type{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6em;}
.scenario-preview{list-style:none;padding:0;margin:14px 0;font-size:.92rem;}
.scenario-preview li{padding:6px 0;color:var(--muted);display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;}
.scenario-preview strong{color:var(--text);min-width:3.2em;}
.key-zone{margin-top:14px;}
.dir{display:inline-block;padding:1px 9px;border-radius:20px;font-size:.76rem;font-weight:600;}
.dir-bullish{background:rgba(111,207,151,.16);color:var(--good);}
.dir-bearish{background:rgba(235,87,87,.16);color:var(--bad);}
.dir-range{background:rgba(242,201,76,.16);color:var(--warn);}

.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;}
.card{display:block;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;text-decoration:none;color:var(--text);}
.card:hover{border-color:var(--cyan);}
.card .type{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;}
.card h3{margin:.45em 0 0;font-size:.98rem;line-height:1.45;}

.block-section{margin:1.4em 0;}
.data-block{margin:1.8em 0;}
.study-report > .data-block{max-width:980px;margin-left:auto;margin-right:auto;}
.reading-rail{max-width:800px;margin-left:auto;margin-right:auto;}
.weekly-intro{margin-bottom:2.2rem;}
.metrics-strip{display:flex;flex-wrap:wrap;gap:14px;}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;min-width:130px;flex:0 1 auto;}
.metric-value{font-size:1.34rem;font-weight:650;line-height:1.25;}
.metric-label{color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;margin-top:2px;}
.metrics-strip.metrics-featured{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));max-width:960px;}
.metrics-strip.metrics-featured .metric{min-width:0;width:100%;}
.compact-metric-table{max-width:980px;margin-left:auto;margin-right:auto;}

.findings-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px;}
.finding-card{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:18px;}
.finding-card h3{margin:0 0 .5em;font-size:1rem;line-height:1.45;}
.finding-card p{color:var(--muted);margin:0;font-size:.92rem;}

/* General chart galleries remain broad; evidence units use a calmer reading width. */
.chart-gallery{display:grid;grid-template-columns:1fr;gap:26px;}
.chart-figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;}
.chart-figure img{width:100%;height:auto;border-radius:8px;display:block;background:#fff;}
.chart-figure figcaption{color:var(--muted);font-size:.86rem;margin-top:10px;display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;}
.chart-full{font-size:.8rem;white-space:nowrap;}
.evidence-unit{max-width:980px;margin:3.4rem auto;}
.evidence-unit h2{margin-top:0;}
.evidence-takeaway{max-width:76ch;color:var(--muted);margin:.1rem 0 1rem;}
.evidence-figure{max-width:860px;margin:0 auto;}
.evidence-unit .data-block{max-width:980px;margin:1.4rem auto 0;}
@media (min-width:1400px){.chart-gallery.dense{grid-template-columns:repeat(2,1fr);}}

/* Tables: fill the available width, wrap prose columns, and only scroll when truly needed. */
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;margin:.6em 0;}
table{border-collapse:collapse;width:100%;font-size:.86rem;}
caption{text-align:left;padding:12px 14px 0;color:var(--muted);font-size:.84rem;}
th,td{padding:9px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;}
td.num,th[data-type="number"]{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}
thead th{background:var(--panel);color:var(--muted);font-weight:600;position:sticky;top:0;z-index:1;padding:0;}
thead th .sort-btn{width:100%;background:none;border:0;color:inherit;font:inherit;font-weight:600;
  padding:10px 14px;cursor:pointer;display:flex;align-items:center;gap:6px;text-align:left;}
th[data-type="number"] .sort-btn{justify-content:flex-end;}
thead th .sort-btn:hover{color:var(--cyan);}
.sort-ind{font-size:.7rem;opacity:.5;}
th[aria-sort="ascending"] .sort-ind::after{content:"▲";opacity:1;}
th[aria-sort="descending"] .sort-ind::after{content:"▼";opacity:1;}
th[aria-sort] .sort-btn{color:var(--cyan);}
tbody th{font-weight:500;color:var(--text);}
tbody tr:hover{background:var(--panel);}
tbody tr[hidden]{display:none;}
.study-id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;color:var(--muted);white-space:nowrap;}
.tag{display:inline-block;padding:1px 9px;border-radius:20px;background:var(--panel2);border:1px solid var(--line2);font-size:.76rem;}
.verdict{font-size:.78rem;padding:1px 8px;border-radius:20px;}
.v-no_evidence{background:rgba(154,163,178,.14);color:var(--muted);}
.v-underpowered{background:rgba(242,201,76,.16);color:var(--warn);}
.v-below_cost{background:rgba(235,87,87,.16);color:var(--bad);}

.limitations-list{color:var(--muted);font-size:.93rem;padding-left:1.25em;max-width:82ch;}
.limitations-list li{margin:.5em 0;}
.evidence-links{list-style:none;padding:0;display:flex;gap:18px;flex-wrap:wrap;}
.empty-note{color:var(--muted);font-size:.9rem;max-width:78ch;}

.filter-bar{display:flex;gap:8px;align-items:center;margin:18px 0;flex-wrap:wrap;}
.filter-bar button{background:var(--panel);border:1px solid var(--line);color:var(--muted);
  padding:6px 15px;border-radius:20px;cursor:pointer;font-size:.86rem;font-family:inherit;}
.filter-bar button:hover{color:var(--text);}
.filter-bar button.active{color:var(--bg);background:var(--cyan);border-color:var(--cyan);font-weight:600;}
.filter-bar input{margin-left:auto;background:var(--panel);border:1px solid var(--line);color:var(--text);
  padding:7px 13px;border-radius:9px;min-width:210px;font-family:inherit;font-size:.88rem;}

.recommendation{color:var(--text);font-size:1.06rem;max-width:78ch;}
.invalidation-note{color:var(--muted);max-width:78ch;}
.data-cutoff{color:var(--muted);font-size:.82rem;}
.disclaimer{color:var(--muted);font-size:.87rem;font-style:italic;}

@media (max-width:700px){
  .shell{padding:0 16px;}
  main.shell{padding:20px 16px 56px;}
  h1{font-size:1.5rem;}
  .metric{min-width:104px;padding:11px 13px;}
  .metric-value{font-size:1.15rem;}
  .metrics-strip.metrics-featured{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}
  .filter-bar input{margin-left:0;width:100%;}
  table{font-size:.82rem;}
}
"""

# Sorting is progressive enhancement: the table is already in a sensible order server-side
# (newest first), and every page-changing control is a real link, so a failed script never
# blocks navigation. Sort values come from data-sort, so formatted display text such as
# "1,234" or "[40.7, 56.2]" is never re-parsed in the browser.
TABLE_JS = """
(function () {
  function cellValue(row, index) {
    var cell = row.children[index];
    return cell ? (cell.getAttribute("data-sort") || cell.textContent.trim()) : "";
  }
  function comparator(index, numeric, dir) {
    return function (a, b) {
      var x = cellValue(a, index), y = cellValue(b, index);
      var r;
      if (numeric) {
        var nx = parseFloat(x), ny = parseFloat(y);
        var xb = isNaN(nx), yb = isNaN(ny);
        if (xb && yb) r = 0; else if (xb) r = 1; else if (yb) r = -1; else r = nx - ny;
      } else {
        r = x.localeCompare(y, undefined, { numeric: true, sensitivity: "base" });
      }
      return dir === "descending" ? -r : r;
    };
  }
  document.querySelectorAll("table[data-sortable]").forEach(function (table) {
    var head = table.tHead, body = table.tBodies[0];
    if (!head || !body) return;
    var headers = Array.prototype.slice.call(head.rows[0].cells);
    headers.forEach(function (th, index) {
      var btn = th.querySelector(".sort-btn");
      if (!btn) return;
      btn.addEventListener("click", function () {
        var numeric = th.getAttribute("data-type") === "number";
        var dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        headers.forEach(function (other) { other.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", dir);
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(comparator(index, numeric, dir));
        rows.forEach(function (row) { body.appendChild(row); });
      });
    });
  });
})();
"""


def build_catalog() -> dict:
    return {
        "schema_version": 2,
        "generated_by": "site/build.py",
        "routes": [f"/{r}/" if r else "/" for r in routes()],
        "weekly_editions": sorted(weekly_weeks(), reverse=True),
        "published_studies": POC_STUDIES,
        "migration_tracker": {
            "phase_1_published": POC_STUDIES,
            "phase_2a_queued": PHASE_2A,
            "phase_2b_queued": PHASE_2B,
            "phase_2c_queued": PHASE_2C,
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generated_pages() -> dict[str, str]:
    pages = {
        "index.html": home_page(),
        "xauusd/weekly/index.html": weekly_index_page(),
        "research/index.html": research_index_page(),
        "research/null-results/index.html": null_results_page(),
    }
    for week in weekly_weeks():
        pages[f"xauusd/weekly/{week}/index.html"] = weekly_report_page(week)
    for sid in POC_STUDIES:
        pages[f"research/studies/{sid}/index.html"] = study_page(sid)
    return pages


def assets() -> dict[str, str]:
    return {
        "site/style.css": STYLE_CSS,
        "site/table.js": TABLE_JS,
        "site/catalog.json": json.dumps(build_catalog(), ensure_ascii=False, indent=2) + "\n",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    outputs = {**generated_pages(), **assets()}
    if args.check:
        drift = [rel for rel, content in outputs.items()
                 if not (ROOT / rel).is_file() or (ROOT / rel).read_text(encoding="utf-8") != content]
        print(json.dumps({"generated files": len(outputs), "drift": len(drift), "paths": drift}))
        return 1 if drift else 0

    for rel, content in outputs.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"generated files": len(outputs), "drift": 0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
