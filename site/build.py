#!/usr/bin/env python3
"""Public site generator — clean rebuild (docs/PUBLIC_SITE_REBUILD_SPEC.md in trading-private).

Weekly-first, Research-second. Renders exactly the eight canonical routes named in the
spec's section 2, from reviewed study.json/results.json data and the published Weekly
summary.json — nothing else. There is no per-result-shape renderer: every research page is
composed from a small, named vocabulary of presentation blocks (see `render_block` below),
driven by an ordered `presentation` list stored in the study's own study.json. A new result
shape is handled by adding a block list, not a new Python function.

Usage:
    python3 site/build.py            write mode
    python3 site/build.py --check    validate the current output matches what this would
                                      generate; exits non-zero on any drift
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FAVICON = "📊"

# The exact eight canonical routes (docs/PUBLIC_SITE_REBUILD_SPEC.md section 2). Nothing
# else is written by this generator. Each entry is a route directory relative to ROOT;
# the file inside is always index.html.
WEEKLY_FORECAST_WEEK = "2026-W36"
POC_STUDIES = [
    "RS-XAUUSD-20260727-001",
    "RS-XAUUSD-20260727-005",
    "RS-XAUUSD-20260818-001",
]
ROUTES = [
    "",
    "xauusd/weekly",
    f"xauusd/weekly/{WEEKLY_FORECAST_WEEK}",
    "research",
    "research/null-results",
    *(f"research/studies/{sid}" for sid in POC_STUDIES),
]

# Studies queued for a later migration phase (docs/PUBLIC_SITE_REBUILD_SPEC.md section 10).
# Listed here only so the Research index and null-results page can name them without
# linking to a route that does not exist in this slice.
PHASE_2A = [
    "RS-XAUUSD-20260727-007", "RS-XAUUSD-20260823-001", "RS-XAUUSD-20260823-002",
    "RS-XAUUSD-20260825-001", "RS-XAUUSD-20260827-001",
]
PHASE_2B = [
    "RS-TX-20260728-001", "RS-TX-20260728-002", "RS-XAUUSD-20260727-003",
    "RS-XAUUSD-20260727-004", "RS-XAUUSD-20260727-006", "RS-XAUUSD-20260727-008",
    "RS-XAUUSD-20260815-001", "RS-XAUUSD-20260815-002", "RS-XAUUSD-20260815-003",
    "RS-XAUUSD-20260817-001",
]


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
    """Walk a dotted key path through nested dicts. Returns None if any segment is missing."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def resolve_source(study: dict, results: dict, source: str) -> Any:
    """A presentation block's `source` is looked up in study.json first, then results.json —
    study.json carries the reviewed Chinese narrative fields, results.json carries the
    reproducible data. A block never needs to say which file its data lives in."""
    value = resolve_path(study, source)
    if value is None:
        value = resolve_path(results, source)
    return value


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def fmt_cell(value: Any) -> str:
    """Render one table cell value. Lists of two numbers are read as a confidence interval;
    everything else is a plain formatted scalar. Never dumps a nested dict — a block that
    needs one is either flattened first or is not a `table`/`comparison_table` source."""
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


# Preferred column order for the generic records table. A record's own keys not in this
# list are appended afterward in first-seen order, so nothing is ever silently dropped —
# this list only controls which columns come first, not which columns exist.
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
# Keys that are bookkeeping rather than reader-facing evidence, dropped from every table.
DROPPED_COLUMNS = {"low_sample", "rank_excluded_reason"}


def flatten_record(record: dict) -> dict:
    """Flatten one level of nested dicts (e.g. {"v34": {...}, "v39": {...}}) into dotted
    keys, so the same renderer handles both flat and one-level-nested comparison shapes."""
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


def label_row(key: str) -> str:
    return esc(key).replace("_", " ")


def label_col(key: str) -> str:
    return esc(key.replace("_pct", " %").replace("_", " "))


def render_records_table(records: dict[str, dict] | list[dict], row_label: str = "") -> str:
    if isinstance(records, dict):
        rows = [(key, flatten_record(value)) for key, value in records.items() if isinstance(value, dict)]
    else:
        rows = [(str(i), flatten_record(value)) for i, value in enumerate(records) if isinstance(value, dict)]
    if not rows:
        return '<p class="empty-note">No rows.</p>'
    columns = ordered_columns([r for _, r in rows])
    head = f"<th>{esc(row_label)}</th>" + "".join(f"<th>{label_col(c)}</th>" for c in columns)
    body = []
    for key, record in rows:
        cells = "".join(f"<td>{fmt_cell(record.get(c))}</td>" for c in columns)
        body.append(f"<tr><th scope=\"row\">{label_row(key)}</th>{cells}</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr>' + head + "</tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Presentation blocks
# ---------------------------------------------------------------------------

def block_metrics(value: dict) -> str:
    items = "".join(
        f'<div class="metric"><div class="metric-value">{fmt_cell(v)}</div>'
        f'<div class="metric-label">{label_col(k)}</div></div>'
        for k, v in value.items()
    )
    return f'<div class="metrics-strip">{items}</div>'


def block_findings(value: list[dict]) -> str:
    cards = []
    for item in value:
        title = item.get("title_zh") or item.get("title", "")
        detail = item.get("detail_zh") or item.get("detail", "")
        cards.append(
            f'<article class="finding-card"><h3>{esc(title)}</h3><p>{esc(detail)}</p></article>'
        )
    return f'<div class="findings-grid">{"".join(cards)}</div>'


def block_table(value: Any, title: str) -> str:
    heading = f"<h3>{esc(title)}</h3>" if title else ""
    return f'<section class="data-block">{heading}{render_records_table(value)}</section>'


def block_comparison_table(value: dict, title: str) -> str:
    return block_table(value, title)


def block_matrix_table(value: dict, title: str) -> str:
    return block_table(value, title)


def block_prose(value: str) -> str:
    return f"<p>{esc(value)}</p>"


def block_charts(value: list[dict], study_id: str) -> str:
    figures = []
    for chart in value:
        file_name = chart.get("file", "")
        title = chart.get("title", file_name)
        href = f"charts/{file_name}"
        figures.append(
            f'<figure class="chart-figure"><a href="{esc(href)}">'
            f'<img src="{esc(href)}" alt="{esc(title)}" loading="lazy"></a>'
            f"<figcaption>{esc(title)}</figcaption></figure>"
        )
    return f'<div class="chart-gallery">{"".join(figures)}</div>'


def block_limitations(value: list[str]) -> str:
    items = "".join(f"<li>{esc(item)}</li>" for item in value)
    return f'<ul class="limitations-list">{items}</ul>'


def block_evidence_links(study_id: str) -> str:
    links = [
        ("analysis.py (Method)", "analysis.py"),
        ("results.json (Results)", "results.json"),
        ("study.json (Study manifest)", "study.json"),
    ]
    items = "".join(f'<li><a href="{href}">{esc(label)}</a></li>' for label, href in links)
    return f'<ul class="evidence-links">{items}</ul>'


BLOCK_TITLES = {
    "metrics": "Headline Metrics",
    "findings": "Key Findings 重點發現",
    "charts": "Charts",
    "limitations": "限制與注意事項",
    "evidence_links": "Evidence",
}


def render_block(block: dict, study: dict, results: dict, study_id: str) -> str:
    kind = block["type"]
    source = block.get("source")
    title = block.get("title", BLOCK_TITLES.get(kind, ""))
    if kind == "evidence_links":
        body = block_evidence_links(study_id)
    else:
        value = resolve_source(study, results, source)
        if value is None:
            return f'<section class="data-block"><p class="empty-note">Missing source: {esc(source)}</p></section>'
        if kind == "metrics":
            body = block_metrics(value)
        elif kind == "findings":
            body = block_findings(value)
        elif kind == "table":
            return block_table(value, title)
        elif kind == "comparison_table":
            return block_comparison_table(value, title)
        elif kind == "matrix_table":
            return block_matrix_table(value, title)
        elif kind == "prose":
            body = block_prose(value)
        elif kind == "charts":
            body = block_charts(value, study_id)
        elif kind == "limitations":
            body = block_limitations(value)
        else:
            raise ValueError(f"unknown presentation block type: {kind}")
    if kind in ("metrics", "findings", "charts", "limitations", "evidence_links"):
        heading = f"<h2>{esc(title)}</h2>" if title else ""
        return f'<section class="block-section">{heading}{body}</section>'
    return body


# ---------------------------------------------------------------------------
# Chrome: nav, document wrapper
# ---------------------------------------------------------------------------

def nav(depth: int) -> str:
    prefix = "../" * depth
    links = [
        ("Home", "index.html" if depth == 0 else f"{prefix}index.html"),
        ("Weekly", f"{prefix}xauusd/weekly/"),
        ("Research", f"{prefix}research/"),
    ]
    items = "".join(f'<a href="{href}">{label}</a>' for label, href in links)
    return f'<nav class="nav">{items}</nav>'


def document(title: str, description: str, depth: int, body: str) -> str:
    prefix = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{esc(description)}">
<title>{esc(title)}</title>
<link rel="stylesheet" href="{prefix}site/style.css">
</head>
<body>
<header class="shell">
{nav(depth)}
</header>
<main class="shell">
{body}
</main>
<footer class="shell"><p>研究證據，不是交易建議；執行前必須以即時 TradingView 確認。Research evidence, not trading advice.</p></footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

def home_page(weekly: dict) -> str:
    summary = weekly["public_summary"]
    scenarios = summary["adopted_scenarios"]
    zones = "".join(
        f'<li><strong>{esc(z["direction"])}</strong> {z["probability"]}% — {esc(z["conditions"][:60])}…</li>'
        for z in scenarios
    )
    key_zone = summary["key_levels"][0]
    poc_cards = "".join(
        f'<a class="card" href="research/studies/{sid}/">'
        f'<div class="type">Research</div><h3>{esc(load_json(study_dir(sid) / "study.json").get("title", sid))}</h3></a>'
        for sid in POC_STUDIES
    )
    body = f"""
<section class="hero">
  <p class="eyebrow">Trading Research</p>
  <h1>黃金劍盾研究站</h1>
  <p class="lede">XAUUSD 週報與策略研究的證據庫。不給訊號，只記錄問了什麼問題、答案是什麼、能相信到什麼程度。</p>
</section>
<section class="feature-card">
  <div class="type">Latest Weekly Report</div>
  <h2><a href="xauusd/weekly/{WEEKLY_FORECAST_WEEK}/">{esc(summary["forecast_week"])} 週報</a></h2>
  <p>{esc(summary["recommendation"]["summary"][:140])}…</p>
  <ul class="scenario-preview">{zones}</ul>
  <p class="key-zone">關鍵決策帶：<strong>{esc(key_zone["label"])} {esc(key_zone["value"])}</strong></p>
  <p><a href="xauusd/weekly/{WEEKLY_FORECAST_WEEK}/">閱讀完整週報 →</a></p>
</section>
<section class="block-section">
  <h2>Featured Research</h2>
  <div class="card-grid">{poc_cards}</div>
</section>
<section class="block-section">
  <p><a href="xauusd/weekly/">所有週報 →</a> · <a href="research/">研究索引 →</a></p>
</section>
"""
    return document("Trading Research", "XAUUSD weekly report and research evidence library.", 0, body)


# ---------------------------------------------------------------------------
# Weekly index and W36 report
# ---------------------------------------------------------------------------

def weekly_index_page(weekly: dict) -> str:
    summary = weekly["public_summary"]
    row = (
        f'<tr><td><a href="{summary["forecast_week"]}/">{esc(summary["forecast_week"])}</a></td>'
        f'<td>{esc(summary["market"])}</td><td>{esc(summary["published_at"])}</td>'
        f'<td>{esc(summary["confidence"])}</td></tr>'
    )
    body = f"""
<h1>Weekly</h1>
<p class="lede">依時間排列的 XAUUSD 週報。每一期都是由目前合格的獨立報告逐項比對後，發布的單一審閱彙整版。</p>
<section class="feature-card">
  <div class="type">Latest</div>
  <h2><a href="{summary["forecast_week"]}/">{esc(summary["forecast_week"])}</a></h2>
  <p>{esc(summary["market_summary"][:160])}…</p>
</section>
<div class="table-scroll"><table>
<thead><tr><th>Forecast Week</th><th>Market</th><th>Published</th><th>Confidence</th></tr></thead>
<tbody>{row}</tbody>
</table></div>
"""
    return document("Weekly", "XAUUSD weekly outlook archive.", 2, body)


def weekly_w36_page(weekly: dict) -> str:
    s = weekly["public_summary"]
    scenario_rows = "".join(
        f"<tr><th scope=\"row\">{esc(sc['direction'])} {sc['probability']}%</th>"
        f"<td>{esc(sc['conditions'])}</td><td>{esc(sc['invalidation'])}</td><td>{esc(sc['targets'])}</td></tr>"
        for sc in s["adopted_scenarios"]
    )
    level_rows = "".join(
        f"<tr><th scope=\"row\">{esc(k['label'])}</th><td>{esc(k['value'])}</td><td>{esc(k['basis'])}</td></tr>"
        for k in s["key_levels"]
    )
    plan_rows = "".join(
        f"<tr><th scope=\"row\">{esc(p['strategy'])}</th><td>{esc(p['stance'])}</td>"
        f"<td>{esc(p['entry'])}</td><td>{esc(p['stop'])}</td><td>{esc(p['risk'])}</td></tr>"
        for p in s["strategy_plan"]
    )
    event_rows = "".join(
        f"<tr><th scope=\"row\">{esc(e['name'])}</th><td>{esc(e['scheduled_at'])}</td><td>{esc(e['handling'])}</td></tr>"
        for e in s["event_risk"]
    )
    comparison_rows = "".join(
        f"<tr><th scope=\"row\">{esc(c['producer'].capitalize())}</th>"
        + "".join(f"<td>{esc(sc['direction'])} {sc['probability']}%</td>" for sc in c["scenarios"])
        + "</tr>"
        for c in s["scenario_comparison"]
    )
    agreements = "".join(f"<li>{esc(a)}</li>" for a in s["agreements"])
    disagreements = "".join(f"<li>{esc(a)}</li>" for a in s["disagreements"])
    unresolved = "".join(f"<li>{esc(a)}</li>" for a in s["evidence_limits"])
    body = f"""
<p class="eyebrow">{esc(s['forecast_week'])} · {esc(s['edition'])} · {esc(s['publication_mode'])} · confidence {esc(s['confidence'])}</p>
<h1>XAUUSD {esc(s['forecast_week'])} 週報</h1>
<p class="lede recommendation">{esc(s['recommendation']['summary'])}</p>
<p class="invalidation-note"><strong>轉折條件：</strong>{esc(s['recommendation']['invalidation'])}</p>

<section class="block-section"><h2>市場摘要</h2><p>{esc(s['market_summary'])}</p>
<p class="data-cutoff">資料截止：{esc(s['data_cutoff'])}</p></section>

<section class="block-section"><h2>三劇本與機率</h2>
<div class="table-scroll"><table>
<thead><tr><th>劇本</th><th>條件</th><th>失準條件</th><th>目標</th></tr></thead>
<tbody>{scenario_rows}</tbody></table></div></section>

<section class="block-section"><h2>關鍵價位</h2>
<div class="table-scroll"><table>
<thead><tr><th>區位</th><th>價位</th><th>依據</th></tr></thead>
<tbody>{level_rows}</tbody></table></div></section>

<section class="block-section"><h2>S1／S2 計畫</h2>
<div class="table-scroll"><table>
<thead><tr><th>策略</th><th>立場</th><th>Entry</th><th>SL</th><th>Risk</th></tr></thead>
<tbody>{plan_rows}</tbody></table></div></section>

<section class="block-section"><h2>事件風險</h2>
<div class="table-scroll"><table>
<thead><tr><th>事件</th><th>時間</th><th>處置</th></tr></thead>
<tbody>{event_rows}</tbody></table></div></section>

<section class="block-section"><h2>Claude / Codex 劇本機率對照</h2>
<div class="table-scroll"><table>
<thead><tr><th>Producer</th><th colspan="3">劇本（方向 機率%）</th></tr></thead>
<tbody>{comparison_rows}</tbody></table></div></section>

<section class="block-section"><h2>Claude / Codex 共識</h2><ul class="limitations-list">{agreements}</ul></section>
<section class="block-section"><h2>Claude / Codex 分歧</h2><ul class="limitations-list">{disagreements}</ul></section>
<section class="block-section"><h2>未解決問題與證據限制</h2><ul class="limitations-list">{unresolved}</ul></section>
<section class="block-section"><p class="disclaimer">{esc(s['disclaimer'])}</p></section>
"""
    return document(f"XAUUSD {s['forecast_week']} outlook", s["market_summary"][:150], 3, body)


# ---------------------------------------------------------------------------
# Research index
# ---------------------------------------------------------------------------

def research_index_page() -> str:
    live_rows = []
    for sid in POC_STUDIES:
        study, results = load_study(sid)
        market = "XAUUSD"
        n_charts = len(results.get("charts", []))
        evidence = f"{n_charts} charts" if n_charts else "tables"
        live_rows.append(
            f'<tr data-market="{market}"><td><a href="studies/{sid}/">{esc(study.get("title", sid))}</a></td>'
            f'<td>{market}</td><td>{esc(study.get("theme", "—"))}</td>'
            f'<td>{esc(study.get("created_on", "—"))}</td><td>{evidence}</td></tr>'
        )
    queued_note = (
        f"另有 {len(PHASE_2A) + len(PHASE_2B)} 篇研究排入後續遷移階段（{len(PHASE_2A)} 篇已有完整中文、"
        f"{len(PHASE_2B)} 篇待補中文），尚未於此版本上線，故不提供連結。"
    )
    body = f"""
<h1>Research</h1>
<p class="lede">研究證據庫，涵蓋每一個市場。</p>
<div class="filter-bar">
  <button type="button" data-filter="all" class="active">All</button>
  <button type="button" data-filter="XAUUSD">XAUUSD</button>
  <button type="button" data-filter="TX">TX</button>
  <input type="search" id="research-search" placeholder="Search…">
</div>
<p><a href="null-results/">Null Results / 沒有效果的研究 →</a></p>
<div class="table-scroll"><table id="research-table">
<thead><tr><th>Title</th><th>Market</th><th>Theme</th><th>Published</th><th>Evidence</th></tr></thead>
<tbody>{"".join(live_rows)}</tbody>
</table></div>
<p class="empty-note">{queued_note}</p>
<script>
(function () {{
  var table = document.getElementById("research-table");
  var rows = Array.prototype.slice.call(table.tBodies[0].rows);
  var buttons = document.querySelectorAll(".filter-bar button");
  var search = document.getElementById("research-search");
  var active = "all";
  function apply() {{
    var q = search.value.trim().toLowerCase();
    rows.forEach(function (row) {{
      var market = row.getAttribute("data-market");
      var text = row.textContent.toLowerCase();
      var matches = (active === "all" || market === active) && text.indexOf(q) !== -1;
      row.style.display = matches ? "" : "none";
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
    return document("Research", "XAUUSD and TX research evidence index.", 1, body)


# ---------------------------------------------------------------------------
# Null results
# ---------------------------------------------------------------------------

def null_results_page() -> str:
    registry = load_json(ROOT / "research/null-results/null_results.json")
    entries = registry["entries"]
    totals = registry["totals"]
    hypotheses = [e for e in entries if e.get("kind") == "hypothesis"]
    study_level = [e for e in entries if e.get("kind") != "hypothesis"]
    live_ids = set(POC_STUDIES)
    rows = []
    for e in hypotheses:
        sid = e.get("study_id")
        family = esc(e.get("family", ""))
        claim = esc(e.get("claim", ""))
        verdict = esc(e.get("verdict", ""))
        reason = esc((e.get("reason") or "")[:140])
        if sid in live_ids:
            family = f'<a href="../studies/{sid}/">{family}</a>'
        rows.append(f"<tr><td>{family}</td><td>{claim}</td><td>{verdict}</td><td>{reason}…</td></tr>")
    verdict_line = " · ".join(f"{esc(k)}: {v}" for k, v in totals["by_verdict"].items())
    body = f"""
<h1>Null Results / 沒有效果的研究</h1>
<p class="lede">{len(hypotheses)} 個已登記並逐一檢定的假設，每一個都附上該樣本能解析的最小效果——{verdict_line}。
另有 {len(study_level)} 篇研究以完整段落記錄，未逐一列在下表中。</p>
<div class="table-scroll"><table>
<thead><tr><th>Family</th><th>Claim</th><th>Verdict</th><th>Reason</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
"""
    return document("Null Results", "63 registered null-result hypotheses.", 2, body)


# ---------------------------------------------------------------------------
# Study detail page
# ---------------------------------------------------------------------------

def study_page(study_id: str) -> str:
    study, results = load_study(study_id)
    blocks = "".join(render_block(b, study, results, study_id) for b in study["presentation"])
    question = study.get("question_zh") or study.get("question", "")
    interpretation = study.get("interpretation_zh", "")
    body = f"""
<p class="eyebrow">{esc(study_id)}</p>
<h1>{esc(study.get("title", study_id))}</h1>
<p class="lede research-question">{esc(question)}</p>
{blocks}
{f'<section class="block-section"><h2>詮釋與實務意義</h2><p>{esc(interpretation)}</p></section>' if interpretation else ''}
"""
    return document(study.get("title", study_id), question[:150], 3, body)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

STYLE_CSS = """
:root{--bg:#0b0d12;--panel:#12151c;--panel2:#171b24;--line:#262b36;--text:#e7e9ee;
--muted:#9aa3b2;--cyan:#5ec8d8;--good:#6fcf97;--warn:#f2c94c;--bad:#eb5757;
--font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:var(--font);
line-height:1.6;}
.shell{max-width:960px;margin:0 auto;padding:0 20px;}
header.shell{padding-top:18px;}
.nav{display:flex;gap:18px;padding-bottom:14px;border-bottom:1px solid var(--line);
font-size:.92rem;}
.nav a{color:var(--muted);text-decoration:none;}
.nav a:hover{color:var(--cyan);}
main.shell{padding:28px 20px 60px;}
footer.shell{padding:24px 20px 40px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);}
h1{font-size:1.8rem;margin:.2em 0 .3em;}
h2{font-size:1.25rem;margin:1.6em 0 .6em;border-top:1px solid var(--line);padding-top:1em;}
h3{font-size:1.02rem;margin:.4em 0;}
.eyebrow{color:var(--cyan);font-size:.78rem;letter-spacing:.04em;text-transform:uppercase;
margin:0 0 .3em;}
.lede{color:var(--muted);font-size:1.02rem;max-width:60ch;}
a{color:var(--cyan);}
.hero{padding:20px 0 8px;}
.feature-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:22px;margin:20px 0;}
.feature-card h2{border-top:none;margin-top:0;padding-top:0;}
.scenario-preview{list-style:none;padding:0;margin:12px 0;color:var(--muted);font-size:.92rem;}
.scenario-preview li{padding:4px 0;}
.key-zone{color:var(--text);}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;}
.card{display:block;background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px;text-decoration:none;color:var(--text);}
.card:hover{border-color:var(--cyan);}
.card .type{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;}
.card h3{margin:.3em 0 0;font-size:1rem;}
.block-section{margin:1.2em 0;}
.data-block{margin:1.4em 0;}
.metrics-strip{display:flex;flex-wrap:wrap;gap:16px;}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:12px 16px;min-width:120px;}
.metric-value{font-size:1.3rem;font-weight:600;}
.metric-label{color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.03em;}
.findings-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}
.finding-card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
padding:16px;}
.finding-card h3{margin-top:0;}
.finding-card p{color:var(--muted);margin-bottom:0;font-size:.92rem;}
.table-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;}
table{border-collapse:collapse;width:100%;font-size:.86rem;white-space:nowrap;}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;}
thead th{color:var(--muted);font-weight:600;background:var(--panel);position:sticky;top:0;}
tbody th{font-weight:500;color:var(--text);}
tbody tr:hover{background:var(--panel);}
.chart-gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;}
.chart-figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:10px;}
.chart-figure img{width:100%;height:auto;border-radius:6px;display:block;}
.chart-figure figcaption{color:var(--muted);font-size:.8rem;margin-top:6px;text-align:center;}
.limitations-list{color:var(--muted);font-size:.92rem;padding-left:1.2em;}
.limitations-list li{margin:.4em 0;}
.evidence-links{list-style:none;padding:0;display:flex;gap:16px;flex-wrap:wrap;}
.empty-note{color:var(--muted);font-style:italic;}
.filter-bar{display:flex;gap:8px;align-items:center;margin:16px 0;flex-wrap:wrap;}
.filter-bar button{background:var(--panel);border:1px solid var(--line);color:var(--muted);
padding:6px 14px;border-radius:20px;cursor:pointer;font-size:.86rem;}
.filter-bar button.active{color:var(--bg);background:var(--cyan);border-color:var(--cyan);}
.filter-bar input{margin-left:auto;background:var(--panel);border:1px solid var(--line);
color:var(--text);padding:6px 12px;border-radius:8px;min-width:180px;}
.recommendation{color:var(--text);font-size:1.05rem;}
.invalidation-note{color:var(--muted);}
.data-cutoff{color:var(--muted);font-size:.82rem;}
.disclaimer{color:var(--muted);font-size:.86rem;font-style:italic;}
@media (max-width:640px){.metrics-strip{gap:10px;}.metric{min-width:100px;padding:10px 12px;}}
"""


# ---------------------------------------------------------------------------
# catalog.json
# ---------------------------------------------------------------------------

def build_catalog() -> dict:
    return {
        "schema_version": 2,
        "generated_by": "site/build.py (clean rebuild)",
        "routes": [f"/{r}/" if r else "/" for r in ROUTES],
        "poc_studies": POC_STUDIES,
        "weekly_forecast_week": WEEKLY_FORECAST_WEEK,
        "migration_tracker": {"phase_1": POC_STUDIES, "phase_2a": PHASE_2A, "phase_2b": PHASE_2B},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generated_pages() -> dict[str, str]:
    weekly = load_json(ROOT / "xauusd/weekly" / WEEKLY_FORECAST_WEEK / "summary.json")
    pages = {
        "index.html": home_page({"public_summary": weekly}),
        "xauusd/weekly/index.html": weekly_index_page({"public_summary": weekly}),
        f"xauusd/weekly/{WEEKLY_FORECAST_WEEK}/index.html": weekly_w36_page({"public_summary": weekly}),
        "research/index.html": research_index_page(),
        "research/null-results/index.html": null_results_page(),
    }
    for sid in POC_STUDIES:
        pages[f"research/studies/{sid}/index.html"] = study_page(sid)
    return pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    pages = generated_pages()
    catalog = build_catalog()

    if args.check:
        drift = []
        for rel, content in pages.items():
            path = ROOT / rel
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                drift.append(rel)
        style_path = ROOT / "site/style.css"
        if not style_path.is_file() or style_path.read_text(encoding="utf-8") != STYLE_CSS:
            drift.append("site/style.css")
        catalog_path = ROOT / "site/catalog.json"
        expected_catalog = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
        if not catalog_path.is_file() or catalog_path.read_text(encoding="utf-8") != expected_catalog:
            drift.append("site/catalog.json")
        print(json.dumps({"generated files": len(pages) + 2, "drift": len(drift), "paths": drift}))
        return 1 if drift else 0

    for rel, content in pages.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (ROOT / "site/style.css").write_text(STYLE_CSS, encoding="utf-8")
    (ROOT / "site/catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"generated files": len(pages) + 2, "drift": 0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
