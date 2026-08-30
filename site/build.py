#!/usr/bin/env python3
"""Build deterministic landing pages from reviewed Public repository contents."""

from __future__ import annotations

import argparse
import html
import json
import os
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


# ---------------------------------------------------------------------------
# Bilingual chrome. English is the default tree (unprefixed, unchanged paths);
# Chinese is a structurally identical mirror under /zh/. Everything in this dict is site
# chrome the author wrote directly, so it is translated in full here. Study content
# (title, question, findings, limitations) carries its own optional `_zh` fields on
# study.json and falls back to English when a study has not been translated yet — this
# is deliberately incremental, translated one study at a time rather than all at once.
# ---------------------------------------------------------------------------

CHROME = {
    "en": {
        "nav.home": "Home",
        "nav.xauusd": "XAUUSD",
        "nav.tx": "TX",
        "nav.lessons": "What Didn’t Work",
        "nav.jargon": "Jargon",
        "lang_switch_label": "中文",
        "footer": "Generated from reviewed repository contents. Research evidence, not "
                  "trading advice.",
        "untranslated_notice": (
            "This study's detailed analysis has not been translated to Chinese yet. "
            "Titles, questions and findings above are translated; the tables and "
            "narrative below are the English original."
        ),
        "home.title": "Trading Research",
        "home.eyebrow": "Public workspace",
        "home.lede": "What to check when a signal fires, and what has already been "
                    "ruled out.",
        "home.what_you_trade": "What you trade",
        "home.reference": "Reference",
        "home.signal_fired": "Signal fired",
        "home.xauusd_title": "XAUUSD Gold",
        "home.xauusd_desc": "Before an entry: the historical win rate by Bollinger "
                            "position, how much of the day's range is left, and what "
                            "has already been ruled out.",
        "home.studies_unit": "studies",
        "home.second_instrument": "Second instrument",
        "home.tx_title": "TX Taiwan Index Futures",
        "home.tx_desc": "Only preliminary work on seasonality and pullback structure "
                        "so far. Nothing at the signal layer yet.",
        "home.lessons_desc_tpl": "{n} hypotheses tested, {survived} survived. Knowing "
                                "what not to try again is itself something to decide "
                                "with.",
        "home.jargon_desc": "The site is in English. This defines every technical "
                            "term once, in Chinese, so it stays readable.",
        "home.weekly_title": "Weekly Report",
        "home.weekly_desc": "Key levels, scenarios and event risk, week by week.",
        "home.what_this_site_is": "What this site is",
        "home.what_this_site_p1": "The public half of a trading research programme. "
                                 "It does not give advice and it does not argue about "
                                 "which strategies work — it records which questions "
                                 "were asked, what the answer was, and how much that "
                                 "answer can be trusted.",
        "home.what_this_site_p2": "Most of the answers are no.",
        "home.weekly_published_tpl": "{week} outlook published",
        "home.no_weekly": "no weekly published yet",
        "home.card_reference": "Reference",
        "xauusd.eyebrow": "Gold",
        "xauusd.lede": "What to check when a signal fires, this week's outlook, and "
                      "every gold study.",
        "xauusd.this_week": "This week",
        "xauusd.studies_heading": "XAUUSD studies",
        "xauusd.filter_placeholder": "Filter XAUUSD studies",
        "xauusd.no_weekly_yet": "No reviewed weekly outlook published yet.",
        "section.filter_placeholder": "Filter studies",
        "section.studies_heading_tpl": "{market} studies",
        "playbook.start_here": "A signal fired — start here",
        "playbook.nothing_yet": "Nothing usable yet.",
        "playbook.not_worth_checking": "Not worth checking",
        "playbook.full_list_tpl": "The full list, each with its resolution bound: {link}",
        "lessons.title": "What Didn’t Work",
        "lessons.eyebrow": "negative results",
        "lessons.lede": "The search space that has been ruled out, and how much each "
                        "“no” actually closed.",
        "lessons.why_exists": "Why this page exists",
        "lessons.why_p1": "Most of what this programme produces is <em>no</em>. That "
                          "is a conclusion, not the absence of one — and knowing what "
                          "not to try again is itself something to decide with.",
        "lessons.why_p2": "Every entry carries its <strong>resolution bound</strong>: "
                          "the smallest difference that sample could have separated. "
                          "A “no evidence” with a wide bound closed nothing "
                          "at all, and that distinction is the whole point.",
        "lessons.registry_type": "registry",
        "lessons.registry_title": "The full registry",
        "lessons.registry_desc": "Every question that was asked and answered with no, "
                                 "each one carrying the smallest effect its sample "
                                 "could have resolved.",
        "lessons.hypotheses_unit": "hypotheses",
        "lessons.survivors_unit": "survivors",
        "lessons.methodology": "Methodology",
        "lessons.methodology_note": "What was learned about how to measure — usually "
                                    "by getting it wrong once first.",
        "null.title": "What did not work",
        "null.eyebrow": "Negative results registry",
        "null.lede": "Questions asked of this data and answered with no, each "
                     "carrying the smallest effect its sample could have resolved.",
        "research.title": "Research studies",
        "research.eyebrow": "Evidence → decision → workflow",
        "research.lede": "Reviewed studies preserve the question, reproducible method, "
                         "aggregate result, and operational impact without publishing "
                         "raw CSV or private conversation.",
        "research.start_here": "Start here",
        "research.registry_title": "What did not work",
        "research.registry_desc": "Every question asked of this data and answered with "
                                  "no, each carrying the smallest effect its sample "
                                  "could have resolved. Published as JSON so anything "
                                  "automated can skip what is already closed.",
        "research.studies_unit": "studies",
    },
    "zh": {
        "nav.home": "首頁",
        "nav.xauusd": "XAUUSD 黃金",
        "nav.tx": "TX 台指期",
        "nav.lessons": "什麼沒用",
        "nav.jargon": "術語",
        "lang_switch_label": "English",
        "footer": "內容產生自已審閱的倉庫內容。研究證據，非交易建議。",
        "untranslated_notice": (
            "這篇研究的詳細分析尚未翻譯成中文。上方的標題、問題與 Findings 已翻譯；"
            "下方的表格與敘述仍為英文原文。"
        ),
        "home.title": "交易研究",
        "home.eyebrow": "公開工作區",
        "home.lede": "訊號來了要看什麼，以及哪些東西已經確定沒用。",
        "home.what_you_trade": "你交易的商品",
        "home.reference": "參考",
        "home.signal_fired": "訊號來了",
        "home.xauusd_title": "XAUUSD 黃金",
        "home.xauusd_desc": "進場前先看：布林位置的歷史勝率、當日還剩多少空間，"
                            "以及哪些東西已經確定沒用。",
        "home.studies_unit": "篇研究",
        "home.second_instrument": "第二個商品",
        "home.tx_title": "TX 台指期",
        "home.tx_desc": "目前只有季節性與回檔結構的初步研究，還沒有訊號層級的判斷依據。",
        "home.lessons_desc_tpl": "{n} 個假設被測過，{survived} 個存活。知道什麼不用再試，"
                                "本身就是判斷依據。",
        "home.jargon_desc": "研究頁面是英文的。這裡把每個技術名詞用中文定義一次。",
        "home.weekly_title": "週報",
        "home.weekly_desc": "每週的關鍵價位、劇本與事件風險。",
        "home.what_this_site_is": "這個網站是什麼",
        "home.what_this_site_p1": "一個交易研究計畫的公開部分。它不是給建議的，也不是"
                                 "討論什麼策略有效——它記錄的是哪些問題被問過、答案是"
                                 "什麼，以及那個答案有多可靠。",
        "home.what_this_site_p2": "大部分的答案是「沒有」。",
        "home.weekly_published_tpl": "{week} 展望已發布",
        "home.no_weekly": "尚未發布週報",
        "home.card_reference": "參考",
        "xauusd.eyebrow": "黃金",
        "xauusd.lede": "訊號來了要看什麼、本週展望，以及所有黃金研究。",
        "xauusd.this_week": "本週展望",
        "xauusd.studies_heading": "XAUUSD 研究",
        "xauusd.filter_placeholder": "篩選 XAUUSD 研究",
        "xauusd.no_weekly_yet": "尚無已審閱的週報。",
        "section.filter_placeholder": "篩選研究",
        "section.studies_heading_tpl": "{market} 研究",
        "playbook.start_here": "訊號來了，先看這裡",
        "playbook.nothing_yet": "尚無可用的判斷依據。",
        "playbook.not_worth_checking": "不用再看的",
        "playbook.full_list_tpl": "完整清單與每一項的解析下限：{link}",
        "lessons.title": "什麼沒用",
        "lessons.eyebrow": "負面結果",
        "lessons.lede": "被排除的搜尋空間，以及每一個「沒有」到底關掉了多少門。",
        "lessons.why_exists": "為什麼這頁存在",
        "lessons.why_p1": "這個研究計畫的產出大部分是<em>沒有</em>。那是結論，不是"
                          "缺少結論——而且知道什麼不用再試，本身就是判斷依據。",
        "lessons.why_p2": "每一筆都帶著<strong>解析下限</strong>：這個樣本能分辨的"
                          "最小差距。界限寬的「無證據」什麼都沒關掉，這個區別很重要。",
        "lessons.registry_type": "登錄",
        "lessons.registry_title": "完整登錄",
        "lessons.registry_desc": "每一個被問過並且得到「沒有」的問題，每一筆都帶著它的"
                                 "解析下限。",
        "lessons.hypotheses_unit": "個假設",
        "lessons.survivors_unit": "個存活",
        "lessons.methodology": "方法論",
        "lessons.methodology_note": "關於「怎麼測」學到的事，通常是先做錯一次才學到的。",
        "null.title": "什麼沒有效",
        "null.eyebrow": "負面結果登錄",
        "null.lede": "問過這份資料並得到「沒有」的問題，每一筆都帶著它的樣本能分辨的"
                     "最小效應。",
        "research.title": "研究列表",
        "research.eyebrow": "證據 → 決策 → 工作流程",
        "research.lede": "已審閱的研究保留問題、可重現的方法、彙總結果與實務影響，"
                         "不發布原始 CSV 或私人對話。",
        "research.start_here": "先看這裡",
        "research.registry_title": "什麼沒有效",
        "research.registry_desc": "每一個問過這份資料並得到「沒有」的問題，每一筆都帶著"
                                  "它的樣本能分辨的最小效應。以 JSON 發布，讓自動化工具"
                                  "可以跳過已經關掉的問題。",
        "research.studies_unit": "篇研究",
    },
}


def t(key: str, lang: str) -> str:
    return CHROME[lang][key]


def zh(item: dict[str, object] | None, field: str, lang: str) -> str:
    """The `_zh` counterpart of a study.json/results.json prose field, with fallback.

    A study that has not been translated yet still renders correctly: the English field
    is used, and the page carries an honest notice rather than a silently mixed page.
    """
    if item is None:
        return ""
    if lang == "zh":
        value = item.get(f"{field}_zh")
        if value:
            return str(value)
    return str(item.get(field, ""))


def findings_html(study: dict[str, object], lang: str = "en") -> str:
    """The insight-card loop shared by every study-page renderer.

    Factored out because it was written out twelve times, once per report shape. A
    single point of translation here makes "Findings" bilingual on every page shape at
    once, rather than requiring twelve separate edits kept in sync by hand.
    """
    items = study.get("findings") or []
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = zh(item, "title", lang)
        detail = zh(item, "detail", lang)
        parts.append(
            f'<article class="insight {html.escape(str(item.get("tone", "info")))}">'
            f'<strong>{html.escape(title)}</strong>'
            f'<p>{html.escape(detail)}</p></article>'
        )
    return "".join(parts)

EXCLUDED_PARTS = {".git", ".github", "_retire", "site", "__pycache__", "v1", "zh"}
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


def nav(prefix: str, lang: str = "en") -> str:
    """Navigation by instrument, then by what you came to do.

    The previous nav offered Overview / XAUUSD / TX / Research, and a study about gold
    lived under Research rather than under XAUUSD — so a reader who wanted to know
    something about gold had two plausible doors and no way to tell which. Studies now sit
    inside the instrument they are about. What is left at the top level is the material
    that genuinely spans both: the lessons, and the vocabulary.
    """
    return (
        '<nav class="nav">'
        f'<a href="{prefix}index.html">{t("nav.home", lang)}</a>'
        f'<a href="{prefix}xauusd/">{t("nav.xauusd", lang)}</a>'
        f'<a href="{prefix}tx/">{t("nav.tx", lang)}</a>'
        f'<a href="{prefix}lessons/">{t("nav.lessons", lang)}</a>'
        f'<a href="{prefix}jargon/">{t("nav.jargon", lang)}</a>'
        '</nav>'
    )


def document(title: str, eyebrow: str, lede: str, body: str, prefix: str = "",
            lang: str = "en", untranslated_body: bool = False,
            html_language: str | None = None) -> str:
    """Every page's shell.

    Navigation chrome remains English on the canonical tree. A migrated research report
    can set ``html_language='zh-Hant'`` while keeping its English title and navigation.
    There is deliberately no language or layout switch: one URL is the source of truth.
    """
    html_lang = html_language or ("zh-Hant" if lang == "zh" else "en")
    asset_prefix = prefix if lang == "en" else prefix + "../"
    notice = (
        f'<div class="callout">{html.escape(t("untranslated_notice", lang))}</div>'
        if untranslated_body and lang == "zh" else ""
    )
    return f"""<!doctype html>
<html lang="{html_lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="{html.escape(lede)}">
  <title>{html.escape(title)} · Trading Research</title>
  <link rel="stylesheet" href="{asset_prefix}site/style.css">
</head>
<body>
  <header class="shell">
    <div class="eyebrow">{html.escape(eyebrow)}</div>
    <h1>{html.escape(title)}</h1>
    <p class="lede">{html.escape(lede)}</p>
    {nav(prefix, lang)}
  </header>
  {notice}
  {body}
  <footer><div class="shell">{html.escape(t("footer", lang))}</div></footer>
  <script src="{asset_prefix}site/app.js"></script>
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
    """Render the canonical, table-first study sheet."""
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
        '<div class="table-wrap research-table"><table data-sortable data-study-table>'
        "<thead><tr>"
        '<th data-sort>Study</th><th data-sort>Theme</th><th data-sort>Market</th>'
        '<th data-sort>Status</th><th data-sort>Date</th><th data-sort>Headline</th>'
        '<th data-sort>Changes practice</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="section-note">Click a column heading to sort.</p>'
    )


def view_toggle_html() -> str:
    """Card / table switch. The choice is remembered per reader in localStorage."""
    return (
        '<div class="view-toggle" data-view-toggle>'
        '<button type="button" data-view="cards" aria-pressed="true">Cards</button>'
        '<button type="button" data-view="table" aria-pressed="false">Table</button>'
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
    badge = '<span class="badge-live">Changes practice</span>' if study.get("policy_impacts") else ""
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


def study_page_comparison(study: dict[str, object], lang: str = "en") -> str:
    result = study["_result"]
    versions = result["versions"]
    finding_html = findings_html(study, lang)
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
        f'<p>{html.escape(zh(study, "hypothesis", lang))}</p>'
        "<p>Raw CSV and private decision conversations remain in trading-private. This public page "
        "contains reviewed aggregate results and the reproducible method only.</p>"
        + file_actions_html(study, lang)
        + "</section></main>"
    )
    return document(
        zh(study, "title", lang),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
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
            '<section class="report-section"><h2>Impact on practice</h2>'
            '<p class="callout">No active policy change. '
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
    return f'<section class="report-section"><h2>Impact on practice</h2>{body}</section>'


def study_asset_href(lang: str, study_id: str, filename: str) -> str:
    """Link to a study's own data file: raw results, method, charts.

    Never mirrored into zh/. study.json and results.json already carry `_zh` translation
    fields inline, and duplicating 30 studies' raw data and chart images into a second
    tree for no reason would repeat exactly the mistake v1/ was built to avoid -- that
    archive keeps only its navigation layer for the same reason. A Chinese study page's
    file links point back into the English-rooted study directory instead. Every study
    page sits at research/studies/<id>/, four directories below whichever tree's root,
    so the four ../ below is not a guess.
    """
    if lang == "en":
        return filename
    return f"../../../../research/studies/{study_id}/{filename}"


def en_link_jargon(lang: str) -> str:
    """jargon/ IS mirrored into zh/ (it is already bilingual), so a study page's own-
    tree relative path to it works in both trees without adjustment."""
    return "../../../jargon/"


def files_section_html(study: dict[str, object], lang: str,
                       include: tuple[str, ...] = ("results.json", "study.json")) -> str:
    """The bilingual "Files" section shared by the newer bespoke renderers.

    Factored out because it was written out four times with only the file list
    differing, which is exactly the kind of duplication that turns into four separate
    bugs the next time the link scheme changes -- as it just did, for zh/.
    """
    sid = str(study["id"])
    links = "".join(f'<a href="{study_asset_href(lang, sid, name)}">{name}</a>' for name in include)
    return (
        f'<section class="report-section"><h2>Files</h2>'
        f'<div class="file-actions">{links}'
        '<a href="../../null-results/">All null results</a>'
        f'<a href="{en_link_jargon(lang)}">{html.escape(t("nav.jargon", lang))}</a></div></section>'
    )


def file_actions_html(study: dict[str, object], lang: str = "en") -> str:
    sid = str(study["id"])
    links = [
        # First, deliberately. A reader who cannot get past the vocabulary cannot use any
        # of the others.
        f'<a href="{en_link_jargon(lang)}">{html.escape(t("nav.jargon", lang))}</a>',
        f'<a href="{study_asset_href(lang, sid, "results.json")}">Structured results</a>',
        f'<a href="{study_asset_href(lang, sid, "analysis.py")}">Python method</a>',
        f'<a href="{study_asset_href(lang, sid, "study.json")}">Study manifest</a>',
    ]
    if (STUDY_ROOT / study["id"] / "impact.md").is_file():
        links.append(f'<a href="{study_asset_href(lang, sid, "impact.md")}">Impact record</a>')
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


def study_page_fail_pattern_solo(study: dict[str, object], lang: str = "en") -> str:
    result = study["_result"]
    baseline = result["baseline"]
    finding_html = findings_html(study, lang)
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
        f'<p>{html.escape(zh(study, "hypothesis", lang))}</p>'
        '<p>Raw CSV and private decision conversations remain in trading-private. This public page contains reviewed aggregate results, charts, and the reproducible method only.</p>'
        + file_actions_html(study, lang)
        + '</section></main>'
    )
    return document(
        zh(study, "title", lang),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
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


def study_page_gap(study: dict[str, object], lang: str = "en") -> str:
    result = study["_result"]
    bd = result["baseline_diff"]
    p1, p2 = gap_version_prefixes(bd)
    version_labels = result.get("version_labels", {})
    label1, label2 = version_labels.get(p1, p1.upper()), version_labels.get(p2, p2.upper())
    finding_html = findings_html(study, lang)
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
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
        + chart_sections_html(result["charts"])
        + gap_entry_slot_table(result["by_entry_30m_diff"], p1, p2, label1, label2)
        + '<section class="report-section"><h2>Fail-Type Share</h2>'
        f'<div class="table-wrap"><table><thead><tr><th>fail_type</th><th>{html.escape(label1)}</th><th>{html.escape(label2)}</th><th>diff</th></tr></thead>'
        f'<tbody>{fail_rows}</tbody></table></div></section>'
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(zh(study, "hypothesis", lang))}</p>'
        f'<p>{html.escape(result["method"]["basis"])}</p>'
        '<p>Raw CSV and private decision conversations remain in trading-private. This public page contains reviewed aggregate results, charts, and the reproducible method only.</p>'
        + file_actions_html(study, lang)
        + '</section></main>'
    )
    return document(
        zh(study, "title", lang),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
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


def study_page_seasonality(study: dict[str, object], lang: str = "en") -> str:
    result = study["_result"]
    overall = result["overall"]
    finding_html = findings_html(study, lang)
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
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
        + chart_sections_html([c for c in result["charts"] if c["section"] == "seasonality"])
        + month_seasonality_table(result["by_month"])
        + week_in_month_table(result["week_in_month"])
        + year_month_heatmap_table(result["year_month_heatmap"])
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(zh(study, "hypothesis", lang))}</p>'
        "<p>Raw CSV and private decision conversations remain in trading-private. This public page "
        "contains reviewed aggregate results, charts, and the reproducible method only.</p>"
        + file_actions_html(study, lang)
        + "</section></main>"
    )
    return document(
        zh(study, "title", lang),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


def study_page_fib_pullback(study: dict[str, object], lang: str = "en") -> str:
    result = study["_result"]
    by_level = result["by_level"]
    finding_html = findings_html(study, lang)
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
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{finding_html}</div></section>'
        + impact_section_html(study)
        + chart_sections_html(result["charts"])
        + result_table(
            "Win rate by Fibonacci retracement level", by_level,
            net_pnl_key="net_pnl_pts", net_pnl_label="Net pts", show_adjustment=False,
        )
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(zh(study, "hypothesis", lang))}</p>'
        f'<p>{html.escape(result["method"]["retracement_formula"])}</p>'
        "<p>Raw CSV and private decision conversations remain in trading-private. This public page "
        "contains reviewed aggregate results, charts, and the reproducible method only. Full "
        f'year-by-year detail is in <a href="{study_asset_href(lang, str(study["id"]), "results.json")}">results.json</a>’s <code>yearly_detail</code>.</p>'
        + file_actions_html(study, lang)
        + "</section></main>"
    )
    return document(
        zh(study, "title", lang),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


def context_program_metrics(study: dict[str, object]) -> str:
    headline = study["headline"]
    keys = study.get("card_metrics") or list(headline)[:4]
    return "".join(
        metric(headline_label(key), headline_display(key, headline[key]))
        for key in keys
        if key in headline
    )


def context_program_findings(study: dict[str, object], lang: str = "en") -> str:
    return findings_html(study, lang)


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


def study_page_context_program(study: dict[str, object], lang: str = "en") -> str:
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
        return study_page_generic(study, lang)

    body = (
        '<main class="shell report">'
        f'<div class="metric-grid">{context_program_metrics(study)}</div>'
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{context_program_findings(study, lang)}</div></section>'
        + impact_section_html(study)
        + chart_sections_html(result.get("charts", []))
        + "".join(tables)
        + context_program_limitations(result)
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(zh(study, "hypothesis", lang))}</p>'
        '<p>All trading timestamps use Asia/Taipei. Raw CSV, source manifests, and private decision records are not published. '
        'This page contains reviewed aggregate results, one reproducible method, and pre-reviewed charts only.</p>'
        + file_actions_html(study, lang)
        + '</section></main>'
    )
    return document(
        zh(study, "title", lang),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


def study_page_pullback_replay(study: dict[str, object], lang: str = "en") -> str:
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
        + '<section class="report-section"><h2>Key findings</h2>'
        f'<div class="insight-grid">{context_program_findings(study, lang)}</div></section>'
        + impact_section_html(study)
        + pullback_replay_table(result)
        + context_program_limitations(result)
        + '<section class="report-section"><h2>Method and evidence boundary</h2>'
        f'<p>{html.escape(zh(study, "hypothesis", lang))}</p>'
        '<p>All timestamps use Asia/Taipei. Raw CSV, per-trade output, source manifests, '
        'and private decision records are not published. The Python method accepts '
        'locally authorized inputs and reproduces the reviewed aggregate.</p>'
        + file_actions_html(study, lang)
        + '</section></main>'
    )
    return document(
        zh(study, "title", lang),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


def study_page_range_profile(study: dict[str, object], lang: str = "en") -> str:
    """Canonical Chinese-body report for the intraday range study.

    This is the first migrated report shape: the title and site chrome stay English, while
    conclusions, notes and evidence read in Traditional Chinese. Long findings are rows,
    not cards, so the reader can scan verdicts down fixed columns.
    """
    result = study["_result"]
    fam = result["families"]
    observed = fam["observed_profile"]
    nulls = fam["vs_shuffled_returns_null"]
    coverage = result["coverage"]
    head = study["headline"]

    def table(headers: list[str], rows: list[list[str]], caption: str, note: str = "",
              table_class: str = "prose-table") -> str:
        head_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
        )
        note_html = f'<p class="section-note">{html.escape(note)}</p>' if note else ""
        return (
            f'<section class="report-section"><h2>{html.escape(caption)}</h2>{note_html}'
            f'<div class="table-wrap {html.escape(table_class)}"><table><thead><tr>{head_html}</tr></thead>'
            f"<tbody>{body_html}</tbody></table></div></section>"
        )

    tone_labels = {"good": "成立", "warn": "限制", "bad": "否定", "info": "描述"}
    finding_rows = [
        [tone_labels.get(str(item.get("tone", "info")), "描述"),
         f'<strong>{html.escape(zh(item, "title", "zh"))}</strong>',
         html.escape(zh(item, "detail", "zh"))]
        for item in study.get("findings", []) if isinstance(item, dict)
    ]

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
         "高於隨機" if "more" in row["direction"] else "低於隨機",
         " / ".join(f"{value:+}" for value in row["excess_pct"]),
         f'{row["min_abs_z"]}']
        for row in consistent
    ]

    clock = fam["us_clock_alignment"]["et_0830_release_slot"]
    clock_rows = [
        ["08:30 ET，美國夏令時間", "台北 20:30",
         f'<strong>{clock["us_dst_on"]["mean_share_pct"]}%</strong>'],
        ["08:30 ET，美國冬令時間", "台北 21:30",
         f'<strong>{clock["us_dst_off"]["mean_share_pct"]}%</strong>'],
        ["相同台北時段，冬令時間", "台北 20:30",
         f'{clock["same_slot_in_the_other_regime"]["20:30_when_dst_off"]}%'],
        ["相同台北時段，夏令時間", "台北 21:30",
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
        [f"<strong>{ {'quiet_morning': '安靜早盤', 'middle': '中段', 'busy_morning': '忙碌早盤'}[label] }</strong>",
         f'{morning[label]["median_morning_ratio"]}',
         f'{morning[label]["median_day_ratio"]}',
         f'<strong>{morning[label]["median_rest_of_day_ratio"]}</strong>']
        for label in ["quiet_morning", "middle", "busy_morning"]
    ]

    body = (
        '<main class="shell report">'
        + table(["樣本", "虛無檢定", "跨期一致時段"], [[
                    f'<strong>{coverage["sessions_used"]}</strong><br>{coverage["first_session"]} → {coverage["last_session"]}',
                    f'<strong>p = {head["family_permutation_p_holdout"]}</strong><br>三段資料皆達 200 次洗牌的解析下限',
                    f'<strong>{head["slots_consistent_and_abs_z_over_2_in_all_periods"]} / 48</strong><br>方向一致且每段 |z| &gt; 2',
                ]], "先看結論",
                "這是一份日內區間結構研究，不預測方向，也不改變 S1／S2 的進場規則。")
        + table(["判讀", "結論", "證據與限制"], finding_rows, "重點發現",
                "先讀結論欄，再用證據與限制欄判斷它能不能進入實務。",
                "prose-table findings-table")
        + '<section class="report-section"><h2>實務影響</h2>'
          '<p class="callout">不修改正式策略。可把時段輪廓當成風險背景，但不能當成方向訊號或固定門檻。</p></section>'
        + table(["台北時間", "訓練期", "驗證期", "留出期", "留出期相對虛無值"], profile_rows,
                "一天的區間如何填滿",
                "數字是當日最終區間已經走完的百分比。虛無模型會洗牌同一天的報酬並重建路徑，保留當日波動與棒數，只破壞大波動發生的時間。完整 48 格在 results.json。")
        + table(["台北時間", "方向", "超額百分點（訓練／驗證／留出）", "最小 |z|"],
                consistent_rows,
                "每一段都超越隨機的半小時",
                "48 格中有 32 格在三段資料方向一致，隨機預期只有 12 格；下表列出三段資料都達 |z| > 2 的 10 格。")
        + table(["公布時段", "時鐘位置", "占當日區間"], clock_rows,
                "美國最忙的半小時每年移動兩次",
                "台北沒有夏令時間，紐約有。美國 08:30 ET 的數據公布窗會在台北 20:30 與 21:30 之間切換；固定使用台北時鐘會混合兩個市場狀態。")
        + table(["此時之後", "平均剩餘區間", "中位剩餘區間", "仍出現新極值的交易日"],
                residual_rows,
                "還剩多少區間（留出期）",
                "23:30 之後的中位剩餘區間為零，表示過半交易日不再創新極值；這是機率敘述，不是宵禁。")
        + table(["早盤分組", "早盤區間比", "全日區間比", "其餘時段區間比"], morning_rows,
                "早盤很忙，不代表後面還會更忙",
                "各比值相對於該交易日前 20 日中位區間。早盤對全日 Spearman 為 "
                f'{morning["spearman_morning_vs_full_day"]}，對其餘時段只有 '
                f'{morning["spearman_morning_vs_rest_of_day"]}；它之所以能預示全日，只因為早盤本來就是全日的一部分。')
        + '<section class="report-section"><h2>限制</h2><ul class="impact-list">'
          '<li>這是描述性研究，不是預測研究，也沒有方向性。</li>'
          '<li>輪廓水位並不平穩；亞洲時段占比在樣本內持續上升，不能把合併樣本的固定門檻直接拿來使用。</li>'
          '<li>固定台北時間會因美國夏令時間而模糊美盤高峰，因此美國時段必須按紐約時鐘讀。</li>'
          '<li>交易日邊界定為台北 07:00；更換邊界會重新分配原始輪廓。</li>'
          '<li>成交量是 TradingView 現貨 tick volume，不是交易所成交量，只適合輪廓檢查。</li>'
          '<li>僅涵蓋一個商品、652 個交易日，且樣本處於強勁上升趨勢。</li>'
          '<li>任何結果都不修改正式 S1／S2 邏輯、即時風控或進場清單。</li>'
          '</ul></section>'
        + '<section class="report-section"><h2>方法與證據邊界</h2>'
        f'<p>{html.escape(zh(study, "hypothesis", "zh"))}</p>'
        '<p>這個輪廓只說明日內區間累積了多少，不說明價格往哪裡走。公開頁只包含已審閱的彙總結果與可重跑方法；原始資料與私人決策紀錄不在 Public。</p>'
        + file_actions_html(study, "en")
        + "</section></main>"
    )
    return document(
        str(study["title"]),
        f'{study["market"]} research · {study["status"]} · {study["id"]}',
        zh(study, "question", "zh"),
        body,
        "../../../",
        lang="en",
        html_language="zh-Hant",
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


def glossary_page(terms: list[dict[str, str]], lang: str = "en") -> str:
    """The one bilingual page on the site, and the reason the rest can stay English.

    Everything else is English. This page is the exception rather than a translation
    layer: given a full Chinese version of a study, nobody reads the English one, and
    reading the English is the point. What actually blocks comprehension is the domain
    vocabulary, and it repeats — "win rate" appears 41 times across the studies,
    "holdout" 30. Defining each term once costs a fraction of translating them in place.

    Named "Jargon" rather than "Glossary" because that is the word this vocabulary is
    called by in the industry the owner works in.
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
        "<p>Every other page on this site is in English. This one is not a translation of "
        "them — it defines the vocabulary that blocks comprehension, once, so the English "
        "stays the thing being read.</p>"
        "<p><strong>Three worth reading first:</strong> "
        "<code>baseline</code> — without it no win rate can be interpreted; "
        "<code>resolution bound</code> — it decides how much a “no evidence” actually "
        "closed; <code>lookahead</code> — every large false finding on this site came "
        "from it.</p></section>"
        '<section class="report-section"><h2>Terms</h2>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>English</th><th>中文</th><th>Gloss</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div></section>"
        '<div class="file-actions"><a href="../research/">All studies</a>'
        '<a href="../research/null-results/">What did not work</a></div>'
        "</main>"
    )
    return document(
        t("nav.jargon", lang),
        "Bilingual reference",
        "Every technical term used across the studies, defined once in Chinese so the "
        "English pages stay readable.",
        body,
        "../",
        lang=lang,
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


def null_results_page(registry: dict[str, object], lang: str = "en") -> str:
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
        f'<div class="file-actions"><a href="{en_link("../../", lang, "research/null-results/null_results.json")}">Registry JSON</a>'
        f'<a href="../">All studies</a><a href="../../jargon/">{html.escape(t("nav.jargon", lang))}</a></div></section>'
        "</main>"
    )
    return document(
        t("null.title", lang),
        t("null.eyebrow", lang),
        t("null.lede", lang),
        body,
        "../../",
        lang=lang,
        untranslated_body=True,
    )


def study_page_hypothesis_sweep(study: dict[str, object], lang: str = "en") -> str:
    """Hypothesis-sweep shape: many claims, one harness, mostly nulls.

    A sweep's page has a different job from a strategy report's. Nobody reads twenty rows
    looking for the winner — there isn't one. What a reader needs is to be able to check
    two things: that a null is a measurement rather than a shrug, and that the one
    good-looking number was tested honestly. So the resolution bound sits in the table
    beside every effect, and the win rate sits beside the baseline it must be read against.
    """
    result = study["_result"]
    coverage = result["coverage"]
    head = study["headline"]
    rows = {h["id"]: h for h in result["hypotheses"]}

    finding_html = findings_html(study, lang)

    def cell(value, spec="{}"):
        return "—" if value is None else html.escape(spec.format(value))

    sweep_rows = "".join(
        "<tr>"
        f'<td><code>{html.escape(h["id"])}</code></td>'
        f'<td>{html.escape(h["family"])}</td>'
        f'<td>{html.escape(h["claim"])}</td>'
        f'<td class="num">{h["n_condition"]}</td>'
        f'<td class="num">{cell(h.get("effect"), "{:+.4f}")}</td>'
        f'<td class="num">{cell(h.get("smallest_resolvable_effect"), "{:.4f}")}</td>'
        f'<td class="num">{cell(h.get("bootstrap_p_two_sided"))}</td>'
        f'<td class="num">{cell(h.get("win_rate_pct"), "{:.2f}")}</td>'
        f'<td class="num">{cell(h.get("baseline_win_rate_pct"), "{:.2f}")}</td>'
        f'<td>{html.escape(h["verdict"])}</td>'
        "</tr>"
        for h in result["hypotheses"]
    )

    consensus = result.get("consensus_analysis", {})
    vote_rows = "".join(
        "<tr>"
        f'<td class="num">{row["votes"]}</td><td class="num">{row["n"]}</td>'
        f'<td class="num">{cell(row.get("win_rate_pct"), "{:.2f}")}</td>'
        f'<td class="num">{cell(row.get("mean_return_pct"), "{:+.4f}")}</td>'
        "</tr>"
        for row in consensus.get("by_vote_count", [])
    )
    stability_rows = "".join(
        "<tr>"
        f'<td><code>{html.escape(c["id"])}</code></td>'
        f'<td>{html.escape(c["claim"])}</td>'
        f'<td class="num">{cell(c.get("pooled_win_rate_pct"), "{:.2f}")}</td>'
        f'<td class="num">{c["blocks_above_50"]}/{c["blocks_measured"]}</td>'
        "</tr>"
        for c in consensus.get("per_condition", [])
    )

    h18 = rows.get("h18", {})
    h17 = rows.get("h17", {})
    family = result.get("family_permutation", {})

    body = (
        '<main class="shell">'
        f'<section class="report-section"><h2>What was measured</h2>'
        f'<p>{html.escape(zh(study, "question", lang))}</p>'
        '<div class="mini-metrics">'
        f'<span><strong>{coverage["sessions"]}</strong> sessions</span>'
        f'<span><strong>{coverage["power_multiple"]}x</strong> the prior sweep</span>'
        f'<span><strong>{len(result["hypotheses"])}</strong> hypotheses</span>'
        f'<span><strong>{head.get("survivors", 0)}</strong> survivors</span>'
        f'<span><strong>{family.get("family_p")}</strong> family p</span>'
        "</div></section>"
        f'<section class="report-section"><h2>Findings</h2>{finding_html}</section>'

        '<section class="report-section"><h2>How to read a null here</h2>'
        "<p>Every row carries a <strong>resolution bound</strong>: the smallest difference "
        "these two samples could have separated at roughly 80% power, computed as "
        "<code>2.8 &times; &sigma; &times; &radic;(1/n₁ + 1/n₂)</code>. An effect inside its "
        "bound means <em>this sample cannot tell it from zero</em> — not that it is zero. "
        "A null with a bound is reusable evidence; a null without one is a shrug.</p>"
        "<p>The win rate is reported beside the baseline it has to be read against, never "
        "on its own. A rate above 50% is compatible with losing money, and the comparison "
        "column is usually where that becomes visible.</p></section>"

        '<section class="report-section"><h2>The twenty hypotheses</h2>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>id</th><th>family</th><th>claim</th><th>n</th><th>effect %</th>"
        "<th>bound %</th><th>boot p</th><th>win %</th><th>baseline %</th><th>verdict</th>"
        f"</tr></thead><tbody>{sweep_rows}</tbody></table></div></section>"

        '<section class="report-section"><h2>A 71% win rate that is not an edge</h2>'
        f'<p><code>h18</code> fires {h18.get("n_condition")} times and wins '
        f'<strong>{h18.get("win_rate_pct")}%</strong>. The sessions it is compared against '
        f'win <strong>{h18.get("baseline_win_rate_pct")}%</strong>, because the dollar '
        f'series covers only {h18.get("sessions_in_universe")} of the '
        f'{coverage["sessions"]} sessions in this study and gold rose through most of '
        "that window. In an era where most days win, winning on 71% of them is close to "
        "average.</p>"
        "<p>An earlier version of this study scored the same condition against all "
        f'{coverage["sessions"]} sessions. That produced a baseline of 54.56% and a '
        "bootstrap p of 0.032, and it would have been published as a survivor. Matching "
        "the comparison group to the condition's own era moved the baseline by "
        "<strong>5.38 points</strong> and the p-value to "
        f'{h18.get("bootstrap_p_two_sided")}.</p></section>'

        '<section class="report-section"><h2>The one thread worth more data</h2>'
        f'<p><code>h17</code> — falling 10-year real yields precede a stronger month for '
        f'gold — gives {h17.get("effect"):+.4f}% against a bound of '
        f'{h17.get("smallest_resolvable_effect"):.4f}%. It misses by 0.048 percentage '
        "points, and it is the only macro condition whose sign holds in all three "
        "chronological windows. It is also the textbook mechanism. That combination makes "
        "it a data problem rather than an idea problem.</p></section>"

        '<section class="report-section"><h2>Does a 50% win rate mean an edge?</h2>'
        f'<p>{html.escape(str(consensus.get("reading", "")))}</p>'
        "<h3>Does a condition stay above 50%, or only average above it?</h3>"
        '<div class="table-wrap"><table><thead><tr><th>id</th><th>claim</th>'
        "<th>pooled win %</th><th>blocks above 50%</th></tr></thead>"
        f"<tbody>{stability_rows}</tbody></table></div>"
        f'<p class="section-note"><strong>'
        f'{len(consensus.get("conditions_above_50_in_every_block", []))} of '
        f'{consensus.get("conditions_tested")}</strong> conditions stay above 50% in every '
        "one of five chronological blocks. Every one of them has a losing period hidden "
        "inside a winning average.</p>"
        "<h3>Does agreement help?</h3>"
        '<div class="table-wrap"><table><thead><tr><th>conditions agreeing</th>'
        "<th>sessions</th><th>win %</th><th>mean return %</th></tr></thead>"
        f"<tbody>{vote_rows}</tbody></table></div>"
        f'<p class="section-note">Monotone in votes: <strong>'
        f'{consensus.get("win_rate_monotone_in_votes")}</strong>. The highest-consensus '
        "bucket has the highest win rate in the table and the only negative mean return. "
        "The sessions behind that row are few enough that the magnitude is unstable; the "
        "direction is the opposite of what the rule predicts.</p></section>"

        '<section class="report-section"><h2>Limitations</h2><ul class="impact-list">'
        + "".join(f"<li>{html.escape(item)}</li>" for item in result.get("limitations", []))
        + "</ul></section>"
        + files_section_html(study, lang)
        + "</main>"
    )
    return document(
        zh(study, "title", lang),
        "Hypothesis sweep",
        zh(study, "card_summary", lang),
        body,
        "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


def study_page_preregistered(study: dict[str, object], lang: str = "en") -> str:
    """A pre-registered primary plus a family-corrected secondary set.

    This page's job is different again. There is one question that was written down before
    the data existed, and the reader has to be able to see that it really was — so the
    before/after table leads, with the predecessor's recorded numbers beside the new ones.
    Everything else is secondary and is labelled as such.
    """
    result = study["_result"]
    primary = result["primary"]
    prior = primary["prior"]
    rep = result["dollar_replication"]
    ice, twi = rep["results"]["dxy_ice"], rep["results"]["broad_twi"]
    head = study["headline"]

    finding_html = findings_html(study, lang)

    def num(value, spec="{}"):
        return "—" if value is None else html.escape(spec.format(value))

    signs = ", ".join(
        f'{v["effect"]:+.4f}' for v in primary["by_period"].values() if v["effect"] is not None
    )
    before_after = (
        "<tr><td>sessions in universe</td>"
        f'<td class="num">{prior["sessions"]}</td>'
        f'<td class="num">{primary["sessions_in_universe"]}</td></tr>'
        "<tr><td>effect</td>"
        f'<td class="num">{prior["effect"]:+.4f}%</td>'
        f'<td class="num">{primary["effect"]:+.4f}%</td></tr>'
        "<tr><td>resolution bound</td>"
        f'<td class="num">{prior["bound"]:.4f}%</td>'
        f'<td class="num">{primary["smallest_resolvable_effect"]:.4f}%</td></tr>'
        "<tr><td>win rate</td>"
        f'<td class="num">{prior["win_rate"]}%</td>'
        f'<td class="num">{primary["win_rate_pct"]}%</td></tr>'
        "<tr><td>baseline win rate</td>"
        f'<td class="num">{prior["baseline"]}%</td>'
        f'<td class="num">{primary["baseline_win_rate_pct"]}%</td></tr>'
        "<tr><td>sign across chronological thirds</td>"
        '<td class="num">+, +, +</td>'
        f'<td class="num">{html.escape(signs)}</td></tr>'
    )

    secondary_rows = "".join(
        "<tr>"
        f'<td><code>{html.escape(r["id"])}</code></td>'
        f'<td>{html.escape(r["family"])}</td>'
        f'<td>{html.escape(r["claim"])}</td>'
        f'<td class="num">{r["n_condition"]}</td>'
        f'<td class="num">{num(r.get("effect"), "{:+.4f}")}</td>'
        f'<td class="num">{num(r.get("smallest_resolvable_effect"), "{:.4f}")}</td>'
        f'<td class="num">{num(r.get("bootstrap_p_two_sided"))}</td>'
        f'<td class="num">{num(r.get("win_rate_pct"), "{:.2f}")}</td>'
        f'<td class="num">{num(r.get("baseline_win_rate_pct"), "{:.2f}")}</td>'
        f'<td>{html.escape(r["verdict"])}</td></tr>'
        for r in result["secondary"]
    )

    body = (
        '<main class="shell">'
        '<section class="report-section"><h2>What was measured</h2>'
        f'<p>{html.escape(zh(study, "question", lang))}</p>'
        '<div class="mini-metrics">'
        f'<span><strong>{head["sessions_before"]} → {head["sessions_after"]}</strong> sessions</span>'
        f'<span><strong>{head["bound_change_pct"]:+.1f}%</strong> bound change</span>'
        f'<span><strong>{html.escape(str(head["primary_verdict"]))}</strong> primary</span>'
        f'<span><strong>{result["family_permutation_secondary_only"]["family_p"]}</strong> secondary family p</span>'
        "</div></section>"

        '<section class="report-section"><h2>Why the primary result is not family-corrected</h2>'
        f'<p>{html.escape(str(result["design"]["why_no_family_correction_on_primary"]))} '
        "The seven secondary hypotheses below <em>are</em> corrected as a family, and the "
        "two numbers are reported separately so nobody has to take that argument on "
        "trust.</p></section>"

        f'<section class="report-section"><h2>Findings</h2>{finding_html}</section>'

        '<section class="report-section"><h2>The pre-registered hypothesis, before and after</h2>'
        f'<p>{html.escape(primary["claim"])}</p>'
        '<div class="table-wrap"><table><thead><tr><th></th>'
        f'<th>{html.escape(prior["study"])} {html.escape(prior["id"])}</th>'
        "<th>this study</th></tr></thead>"
        f"<tbody>{before_after}</tbody></table></div>"
        "<p class=\"section-note\">The single property that made this worth pursuing was "
        "that its sign held in all three chronological windows — the only macro condition "
        "in the predecessor that managed it. Adding the earlier years breaks exactly that. "
        "<strong>The stability was a feature of the sample, not of the relationship.</strong>"
        "</p></section>"

        '<section class="report-section"><h2>More data made the question harder</h2>'
        "<p>The bound is <code>2.8 &times; &sigma; &times; &radic;(1/n₁ + 1/n₂)</code>. "
        f'It rose {head["bound_change_pct"]:+.1f}%, and the two terms moved in opposite '
        "directions.</p>"
        '<div class="table-wrap"><table><thead><tr><th>term</th><th>change</th>'
        "<th>why</th></tr></thead><tbody>"
        f'<tr><td>&radic;(1/n₁ + 1/n₂)</td><td class="num">{head["sample_term_change_pct"]:+.1f}%</td>'
        "<td>the universe grew 39% but the condition group did not — an expanding "
        "percentile rank fires at whatever rate history dictates, 1.3% of 2018 and 31.5% "
        "of 2020 — and the bound is dominated by the smaller group</td></tr>"
        f'<tr><td>&sigma;</td><td class="num">{head["sigma_change_pct"]:+.1f}%</td>'
        "<td>the added years are noisier: gold's 20-session forward return had a standard "
        "deviation of 5.6351% across 2008-2012 against 4.3425% afterwards</td></tr>"
        "</tbody></table></div>"
        "<p class=\"section-note\"><strong>A bigger sample narrows a bound only if what it "
        "adds is no noisier than what it had.</strong> That is worth stating because &ldquo;get "
        "more data&rdquo; is the standard answer to an underpowered null, and here it was the "
        "wrong one.</p></section>"

        '<section class="report-section"><h2>The dollar extension does not replicate</h2>'
        f'<p>{html.escape(str(rep["question"]))}</p>'
        f'<p>The two indices are not the same instrument: correlation {rep["level_correlation"]} '
        f'on levels, a median gap of {rep["median_absolute_level_gap"]} index points. So the '
        f'longer one was asked to reproduce the shorter one on the {rep["overlap_sessions"]} '
        "sessions where both exist, before being allowed to extend anything.</p>"
        '<div class="table-wrap"><table><thead><tr><th>index</th><th>fires</th>'
        "<th>win rate</th><th>baseline</th><th>effect</th></tr></thead><tbody>"
        f'<tr><td>ICE DXY</td><td class="num">{ice["n_condition"]}</td>'
        f'<td class="num">{ice["win_rate_pct"]}%</td>'
        f'<td class="num">{ice["baseline_win_rate_pct"]}%</td>'
        f'<td class="num">{ice["effect_pct"]:+.4f}%</td></tr>'
        f'<tr><td>Broad trade-weighted</td><td class="num">{twi["n_condition"]}</td>'
        f'<td class="num">{twi["win_rate_pct"]}%</td>'
        f'<td class="num">{twi["baseline_win_rate_pct"]}%</td>'
        f'<td class="num">{twi["effect_pct"]:+.4f}%</td></tr>'
        "</tbody></table></div>"
        "<p class=\"section-note\">Same window, same construction, <strong>opposite signs</strong>. "
        "The extension is refused — and a result that flips sign when a closely related "
        "measurement instrument is swapped in was never a finding. This check was built to "
        "enable the result and it destroyed it instead.</p></section>"

        '<section class="report-section"><h2>Secondary hypotheses</h2>'
        '<p class="section-note">Corrected as a family. Family permutation p = '
        f'<strong>{result["family_permutation_secondary_only"]["family_p"]}</strong>.</p>'
        '<div class="table-wrap"><table><thead><tr><th>id</th><th>family</th><th>claim</th>'
        "<th>n</th><th>effect %</th><th>bound %</th><th>boot p</th><th>win %</th>"
        f"<th>baseline %</th><th>verdict</th></tr></thead><tbody>{secondary_rows}"
        "</tbody></table></div></section>"

        '<section class="report-section"><h2>Limitations</h2><ul class="impact-list">'
        + "".join(f"<li>{html.escape(item)}</li>" for item in result.get("limitations", []))
        + "</ul></section>"
        + files_section_html(study, lang, ("results.json", "study.json", "analysis.py"))
        + "</main>"
    )
    return document(
        zh(study, "title", lang), "Pre-registered test",
        zh(study, "card_summary", lang), body, "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


def study_page_robustness(study: dict[str, object], lang: str = "en") -> str:
    """One finding, measured several ways.

    The page has to keep two things separate that a reader will otherwise merge: the effect
    held every time, and the set of trades it selects did not. A table of win rates says
    the first and hides the second, so the per-trade agreement table is given equal weight
    rather than being a footnote.
    """
    result = study["_result"]
    v = result["variants"]
    ag = result["zone_agreement"]
    sim = result["instrument_similarity"]
    prior = result["prior"]
    cov = result["coverage"]
    head = study["headline"]

    finding_html = findings_html(study, lang)
    order = [k for k in ("A", "B", "C", "D") if k in v]
    variant_rows = "".join(
        "<tr>"
        f'<td><code>{html.escape(k)}</code></td>'
        f'<td>{html.escape(v[k]["description"])}</td>'
        f'<td class="num">{v[k]["n_above_upper"]}</td>'
        f'<td class="num">{v[k]["win_rate_above_pct"]:.2f}%</td>'
        f'<td class="num">{v[k]["win_rate_rest_pct"]:.2f}%</td>'
        f'<td class="num">{v[k]["gap_pct_points"]:+.2f}</td>'
        f'<td class="num">{v[k]["smallest_resolvable_gap_pct_points"]:.2f}</td>'
        f'<td class="num">{v[k]["permutation_p"]}</td>'
        f'<td>{html.escape(v[k]["verdict"])}</td></tr>'
        for k in order
    )
    money_rows = "".join(
        "<tr>"
        f'<td><code>{html.escape(k)}</code></td>'
        f'<td class="num">{v[k]["n_above_upper"]}</td>'
        f'<td class="num">{v[k]["win_rate_above_pct"]:.2f}%</td>'
        f'<td class="num">{v[k]["share_of_trades_kept_pct"]:.2f}%</td>'
        f'<td class="num">{v[k]["share_of_total_return_captured_pct"]:.2f}%</td></tr>'
        for k in order
    )
    agree_rows = "".join(
        "<tr>"
        f'<td>{html.escape(key.replace("_vs_", " vs "))}</td>'
        f'<td class="num">{row["same_zone_pct"]:.2f}%</td>'
        f'<td class="num">{row["above_upper_in_first"]}</td>'
        f'<td class="num">{row["above_upper_in_second"]}</td>'
        f'<td class="num">{row["above_upper_in_both"]}</td>'
        f'<td class="num">{row["jaccard"]}</td></tr>'
        for key, row in ag.items()
    )
    ab = ag["A_vs_B"]

    body = (
        '<main class="shell">'
        '<section class="report-section"><h2>What was measured</h2>'
        f'<p>{html.escape(zh(study, "question", lang))}</p>'
        '<div class="mini-metrics">'
        f'<span><strong>{head["variants_tested"]}</strong> measurements</span>'
        f'<span><strong>{head["survivors"]}</strong> survive</span>'
        f'<span><strong>{cov["trades_in_common_set"]}</strong> common trades</span>'
        f'<span><strong>{ab["same_zone_pct"]:.0f}%</strong> zone agreement, 30m vs 1h</span>'
        "</div></section>"
        f'<section class="report-section"><h2>Findings</h2>{finding_html}</section>'

        '<section class="report-section"><h2>The four measurements</h2>'
        f'<p class="section-note">All scored on the same {cov["trades_in_common_set"]} '
        f'trades — the ones every variant can price. {html.escape(str(cov["why_restricted"]))}. '
        "A difference measured on different trades would be a difference between samples."
        "</p>"
        '<div class="table-wrap"><table><thead><tr><th>variant</th><th>measurement</th>'
        "<th>n above upper</th><th>win rate</th><th>rest</th><th>gap (pp)</th>"
        f"<th>bound (pp)</th><th>perm p</th><th>verdict</th></tr></thead><tbody>{variant_rows}"
        "</tbody></table></div>"
        f'<p class="section-note"><strong>All {head["survivors"]} survive.</strong> The '
        f'effect held at {prior["win_rate"]}% on 30-minute bars in '
        f'{html.escape(prior["study"])}, and it holds under every re-measurement here.</p>'
        "</section>"

        '<section class="report-section"><h2>How much that is worth</h2>'
        "<p>Less than it sounds, and the number is here so nobody has to guess.</p>"
        '<div class="table-wrap"><table><thead><tr><th></th><th>this study (futures vs spot)</th>'
        "<th>the dollar test that inverted a finding</th></tr></thead><tbody>"
        f'<tr><td>level correlation</td><td class="num">{sim["level_correlation"]}</td>'
        f'<td class="num">{sim["comparison"]["dollar_test_level_correlation"]}</td></tr>'
        f'<tr><td>return correlation</td><td class="num">{sim["return_correlation"]}</td>'
        '<td class="num">—</td></tr></tbody></table></div>'
        "<p class=\"section-note\">Passing an instrument test between two quotes of the same "
        "metal is a much weaker statement than failing one between two genuinely different "
        "constructions. A real instrument test for a gold signal needs something that is "
        "not gold.</p></section>"

        '<section class="report-section"><h2>The effect is robust. The label is not.</h2>'
        '<div class="table-wrap"><table><thead><tr><th>pair</th><th>same zone</th>'
        "<th>above upper (first)</th><th>above upper (second)</th><th>in both</th>"
        f"<th>Jaccard</th></tr></thead><tbody>{agree_rows}</tbody></table></div>"
        f'<p>Read the first row. Thirty-minute and hourly spot — <em>the same instrument, '
        f'the same formula</em> — assign the same %B zone to only '
        f'<strong>{ab["same_zone_pct"]:.2f}%</strong> of trades. The 30-minute chart calls '
        f'{ab["above_upper_in_first"]} entries above the upper band; the hourly chart calls '
        f'{ab["above_upper_in_second"]}; they agree on {ab["above_upper_in_both"]}.</p>'
        '<p><strong>&ldquo;%B is above the upper band&rdquo; is not a property of the trade. '
        "It is a property of the chart you happen to have open.</strong> The statistical "
        "effect is real in every version; the label a person would act on is not stable "
        "between versions.</p>"
        "<p class=\"section-note\">This is invisible in a table of win rates, which is why "
        "per-trade agreement was measured rather than inferred from the headline numbers "
        "matching. Two of the variants posted identical win rates while sharing only 23 of "
        "the 28 entries each selected.</p></section>"

        '<section class="report-section"><h2>Selecting harder raised the win rate and gave up return</h2>'
        '<div class="table-wrap"><table><thead><tr><th>variant</th><th>n</th>'
        "<th>win rate</th><th>share of trades kept</th><th>share of total return captured</th>"
        f"</tr></thead><tbody>{money_rows}</tbody></table></div>"
        f'<p class="section-note">Variant B posts the best win rate at '
        f'{head["best_win_rate_pct"]}% while capturing '
        f'{head["best_win_rate_return_captured_pct"]}% of the return; variant A wins '
        f'{head["original_variant_win_rate_pct"]}% and captures '
        f'{head["original_variant_return_captured_pct"]}%. Tightening the selection bought '
        "win-rate points and sold return — the fifth independent instance of that trade in "
        "this programme, and it appeared here as a side effect of a test about something "
        "else.</p></section>"

        '<section class="report-section"><h2>Limitations</h2><ul class="impact-list">'
        + "".join(f"<li>{html.escape(item)}</li>" for item in result.get("limitations", []))
        + "</ul></section>"
        + files_section_html(study, lang, ("results.json", "study.json", "analysis.py", "impact.md"))
        + "</main>"
    )
    return document(
        zh(study, "title", lang), "Robustness check",
        zh(study, "card_summary", lang), body, "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


LABEL_OVERRIDES = {
    "n": "n",
    "ci95": "95% CI",
    "win_rate_ci95_pct": "win rate 95% CI",
    "pct": "%",
}


def humanise(key: str) -> str:
    """`win_rate_pct` -> `win rate %`. Table headers are read, not parsed."""
    if key in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[key]
    text = str(key).replace("_", " ")
    for suffix, replacement in ((" pct", " %"), (" usd", " USD"), (" r", " R")):
        if text.endswith(suffix):
            text = text[: -len(suffix)] + replacement
    return text


def is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def scalar_html(value: object) -> str:
    if value is None:
        return "<em>—</em>"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f'<span class="num">{html.escape(str(value))}</span>'
    return html.escape(str(value))


def inline_list(values: list) -> str | None:
    """A short list of scalars belongs on one line, not as a bulleted list.

    A confidence interval is the case that matters: `[51.42, 60.35]` rendered as a
    two-item bullet list is unreadable, and it appears hundreds of times.
    """
    if not values or not all(is_scalar(v) for v in values):
        return None
    if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
        return f'<span class="num">{values[0]} – {values[1]}</span>'
    if len(values) <= 8 and all(len(str(v)) <= 24 for v in values):
        return ", ".join(scalar_html(v) for v in values)
    return None


def uniform_rows(mapping: dict) -> list[str] | None:
    """If every value is a dict with the same keys, this is a table with named rows."""
    values = list(mapping.values())
    if len(values) < 2 or not all(isinstance(v, dict) for v in values):
        return None
    signatures = {tuple(v.keys()) for v in values}
    if len(signatures) != 1:
        return None
    columns = list(values[0].keys())
    if not columns or len(columns) > 12:
        return None
    if not all(is_scalar(cell) or isinstance(cell, list) for v in values for cell in v.values()):
        return None
    return columns


def table_from_mapping(mapping: dict, columns: list[str], first_header: str = "") -> str:
    # The row-label column is deliberately unheaded: the heading above the table already
    # says what the rows are, and repeating it reads as "levels | levels".
    head = f"<th>{html.escape(first_header)}</th>" + "".join(
        f"<th>{html.escape(humanise(c))}</th>" for c in columns
    )
    rows = "".join(
        "<tr>" + f"<td><strong>{html.escape(humanise(name))}</strong></td>" + "".join(
            f"<td>{render_value(row.get(c))}</td>" for c in columns
        ) + "</tr>"
        for name, row in mapping.items()
    )
    return (f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def render_value(value: object, depth: int = 0, name: str = "") -> str:
    """Render an arbitrary results payload as tables and definition lists.

    Written after five published pages turned out to be dumping raw JSON into <code>
    blocks — 366 of them — because the first version capped recursion depth and gave up.
    Depth is the wrong thing to cap: what matters is the shape. A dict whose values all
    share a key set is a table however deep it sits, and everything else decomposes into
    scalars, short inline lists, or named subsections.
    """
    if is_scalar(value):
        return scalar_html(value)

    if isinstance(value, list):
        if not value:
            return "<em>none</em>"
        # A chart manifest is a list of {file, caption}. Rendering it as a table of
        # filenames hands the reader the name of a picture that is sitting right there.
        if all(isinstance(item, dict) and str(item.get("file", "")).endswith(
                (".png", ".svg", ".jpg", ".jpeg")) for item in value):
            return '<div class="chart-grid">' + "".join(
                f'<figure class="chart"><img src="charts/{html.escape(str(item["file"]))}" '
                f'alt="{html.escape(str(item.get("caption") or item["file"]))}" loading="lazy">'
                f'<figcaption>{html.escape(str(item.get("caption") or ""))}</figcaption>'
                "</figure>"
                for item in value
            ) + "</div>"
        inline = inline_list(value)
        if inline is not None:
            return inline
        if all(isinstance(item, dict) for item in value):
            columns: list[str] = []
            for item in value:
                for key in item:
                    if key not in columns:
                        columns.append(key)
            if len(columns) <= 12:
                head = "".join(f"<th>{html.escape(humanise(c))}</th>" for c in columns)
                rows = "".join(
                    "<tr>" + "".join(
                        f"<td>{render_value(item.get(c), depth + 1)}</td>" for c in columns
                    ) + "</tr>"
                    for item in value
                )
                return (f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                        f"<tbody>{rows}</tbody></table></div>")
        return ('<ul class="impact-list">'
                + "".join(f"<li>{render_value(item, depth + 1)}</li>" for item in value)
                + "</ul>")

    if isinstance(value, dict):
        if not value:
            return "<em>none</em>"
        columns = uniform_rows(value)
        if columns:
            return table_from_mapping(value, columns)

        scalars = {k: v for k, v in value.items()
                   if is_scalar(v) or (isinstance(v, list) and inline_list(v) is not None)}
        nested = {k: v for k, v in value.items() if k not in scalars}

        parts = []
        if scalars:
            parts.append('<dl class="generic-block">' + "".join(
                f"<dt>{html.escape(humanise(k))}</dt><dd>{render_value(v, depth + 1, k)}</dd>"
                for k, v in scalars.items()
            ) + "</dl>")
        for key, child in nested.items():
            # A named subsection rather than a nested definition list: at three levels of
            # <dl> the indentation stops carrying meaning and the reader loses the key.
            heading = "h3" if depth == 0 else "h4"
            parts.append(f'<{heading} class="block-title">{html.escape(humanise(key))}</{heading}>'
                         + render_value(child, depth + 1, key))
        return "".join(parts)

    return html.escape(str(value))


GENERIC_SKIP = {
    "schema_version", "study_id", "generated_at", "market", "strategy", "method",
    "limitations", "coverage", "title",
}


def study_page_generic(study: dict[str, object], lang: str = "en") -> str:
    """The fallback every study shape lands on when it has no bespoke renderer.

    This exists because the dispatcher used to raise on an unrecognised shape, which meant
    a new kind of study could not be published until somebody wrote a page for it. Ten
    confirmed studies sat unpublished behind that, including the one the signal playbook
    cites. A plainer page beats an unpublishable one.
    """
    result = study["_result"]
    head = study.get("headline") or {}

    finding_html = findings_html(study, lang)
    metrics = "".join(
        f"<span><strong>{html.escape(str(head[key]))}</strong> "
        f"{html.escape(str(key).replace('_', ' '))}</span>"
        for key in (study.get("card_metrics") or [])
        if key in head
    )
    sections = "".join(
        f'<section class="report-section"><h2>'
        f"{html.escape(str(key).replace('_', ' '))}</h2>{render_value(value)}</section>"
        for key, value in result.items()
        if key not in GENERIC_SKIP
    )
    coverage = (
        f'<section class="report-section"><h2>Coverage</h2>'
        f'{render_value(result["coverage"])}</section>' if "coverage" in result else ""
    )

    body = (
        '<main class="shell">'
        '<section class="report-section"><h2>What was measured</h2>'
        f'<p>{html.escape(str(study.get("question") or ""))}</p>'
        + (f'<div class="mini-metrics">{metrics}</div>' if metrics else "")
        + "</section>"
        + (f'<section class="report-section"><h2>Findings</h2>{finding_html}</section>'
           if finding_html else "")
        + coverage
        + sections
        + ('<section class="report-section"><h2>Limitations</h2>'
           f'{render_value(result["limitations"])}</section>' if "limitations" in result else "")
        + ('<section class="report-section"><h2>Method</h2>'
           f'{render_value(result["method"])}</section>' if "method" in result else "")
        + files_section_html(study, lang, ("results.json", "study.json", "analysis.py"))
        + "</main>"
    )
    return document(
        zh(study, "title", lang), "Study",
        zh(study, "card_summary", lang), body, "../../../",
        lang=lang,
        untranslated_body=not study.get("body_translated_zh"),
    )


def study_page(study: dict[str, object], lang: str = "en") -> str:
    result = study["_result"]
    if "versions" in result:
        return study_page_comparison(study, lang)
    if "baseline_diff" in result:
        return study_page_gap(study, lang)
    if "fail_pattern" in result:
        return study_page_fail_pattern_solo(study, lang)
    if "by_month" in result:
        return study_page_seasonality(study, lang)
    if "by_level" in result:
        return study_page_fib_pullback(study, lang)
    if "policies" in result and "emulator_validation" in result:
        return study_page_pullback_replay(study, lang)
    if "strategies" in result:
        return study_page_context_program(study, lang)
    if isinstance(result.get("families"), dict) and "observed_profile" in result["families"]:
        return study_page_range_profile(study, lang)
    if "hypotheses" in result and "consensus_analysis" in result:
        return study_page_hypothesis_sweep(study, lang)
    if "primary" in result and "secondary" in result:
        return study_page_preregistered(study, lang)
    if "variants" in result and "zone_agreement" in result:
        return study_page_robustness(study, lang)
    # No bespoke renderer: fall back rather than refuse. Raising here meant a study could
    # not be published until someone wrote a page for its shape, and ten confirmed studies
    # accumulated behind that — one of them the study the signal playbook cites.
    return study_page_generic(study, lang)


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
        "intro": "A signal just fired. This page answers one question: what is already "
                 "known that bears on this trade?",
        "intro_zh": "訊號到了。這頁只回答一件事：現在有什麼已知的東西，能幫你判斷這筆單。",
        "checks": [
            {
                "title": "Where the entry sits in the Bollinger band (S1, on 30-minute bars)",
                "title_zh": "進場價在布林通道的哪裡（S1、看 30 分K）",
                "study": "RS-XAUUSD-20260823-002",
                "what": "Entries with %B above 1.0 — closing outside the upper band — won "
                        "73.17% historically (n=82) against a 55.93% baseline. It was "
                        "stronger out of sample, not weaker.",
                "what_zh": "%B > 1.0（收在上軌之外）的進場，歷史勝率 73.17%（n=82），基準是 "
                           "55.93%。而且它在樣本外更強，不是變弱。",
                "caveat": "Two things. One: as a filter it loses money — it keeps 17% of "
                          "entries and 44% of the return. Two: it has to be the 30-minute "
                          "chart. For the same trade, 30-minute and hourly agree on the %B "
                          "zone only 32% of the time; the 30-minute chart calls 71 entries "
                          "above the upper band, the hourly chart calls 28. A %B reading "
                          "without its bar size is not a reading.",
                "caveat_zh": "兩件事。一：拿它當過濾器會賠錢——只留 17% 的進場、44% 的報酬。"
                             "二：一定要看 30 分K。同一筆交易，30 分K 與 1 小時對 %B 區間的判定"
                             "只有 32% 一致；30 分K 認定 71 筆在上緣，1 小時只認 28 筆。沒說週期"
                             "的 %B 不是一個讀數。",
            },
            {
                "title": "How much of the day's range is left at this hour",
                "title_zh": "現在這個時間點，當日還剩多少空間",
                "study": "RS-XAUUSD-20260823-001",
                "what": "After 23:30 Taipei the median day is finished — median remaining "
                        "range 0%. After 02:30 the average is 4%.",
                "what_zh": "台北 23:30 之後，中位數的一天已經走完了（剩餘區間中位數 0%）。"
                           "02:30 之後平均只剩 4%。",
                "caveat": "This says how much room is left. It says nothing about direction.",
                "caveat_zh": "這只講「還有多少空間」，完全不講方向。",
            },
            {
                "title": "What this strategy looks like normally",
                "title_zh": "這個策略本來就長什麼樣",
                "study": "RS-XAUUSD-20260727-007",
                "what": "S2 V3.2 has a 47.13% baseline win rate and a 2.05 profit factor. A "
                        "low win rate with a high payoff is its normal shape; reading the "
                        "win rate alone will mislead you.",
                "what_zh": "S2 V3.2 基準勝率 47.13%、獲利因子 2.05。低勝率高賠率是正常形態，"
                           "只看勝率會誤判。",
                "caveat": "S1 V3.9's baseline is 55.93% at PF 1.849, holding about 30 bars.",
                "caveat_zh": "S1 V3.9 的基準是 55.93% / PF 1.849，平均持有 30 根。",
            },
        ],
        "ruled_out": [
            "The Macro composite score and the GVZ threshold — revoked 2026-08-17. Do not "
            "size down on them.",
            "30-minute slot win rates — indistinguishable from noise.",
            "Monday weakest, Friday strongest — not supported on this data.",
            "CFTC positioning — the available sample resolves nothing.",
        ],
        "ruled_out_zh": [
            "Macro 綜合分數、GVZ 門檻——已於 2026-08-17 撤銷，不要再用它們減碼",
            "30 分鐘時槽勝率——與雜訊無法區分",
            "週一最弱、週五最強——在這份資料上不成立",
            "CFTC 部位——目前樣本看不出任何東西",
        ],
    },
    "TX": {
        "intro": "TX has only preliminary work on seasonality and pullback structure. "
                 "Nothing at the signal layer yet.",
        "intro_zh": "TX 目前只有季節性與回檔結構的初步研究，還沒有訊號層級的判斷依據。",
        "checks": [],
        "ruled_out": [],
    },
}


def signal_playbook_html(market: str, study_list: list[dict[str, object]], prefix: str,
                         lang: str = "en") -> str:
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

    def pick(source: dict, field: str) -> str:
        if lang == "zh" and source.get(f"{field}_zh"):
            return str(source[f"{field}_zh"])
        return str(source.get(field, ""))

    checks = []
    for item in book["checks"]:
        study = by_id.get(item["study"])
        link = (
            f'<a href="{html.escape(prefix + study["_relative"])}/">{html.escape(item["study"])}</a>'
            if study else html.escape(item["study"])
        )
        checks.append(
            '<article class="insight good">'
            f'<strong>{html.escape(pick(item, "title"))}</strong>'
            f'<p>{html.escape(pick(item, "what"))}</p>'
            f'<p class="section-note">{html.escape(pick(item, "caveat"))}</p>'
            f'<p class="section-note">{link}</p>'
            "</article>"
        )
    ruled_items = book.get("ruled_out_zh") if lang == "zh" and book.get("ruled_out_zh") else book["ruled_out"]
    ruled = "".join(f"<li>{html.escape(x)}</li>" for x in ruled_items)
    lessons_link = f'<a href="{prefix}lessons/">{html.escape(t("nav.lessons", lang))}</a>' 
    ruled_block = (
        f'<section class="report-section"><h2>{t("playbook.not_worth_checking", lang)}</h2>'
        f'<ul class="impact-list">{ruled}</ul>'
        f'<p class="section-note">{t("playbook.full_list_tpl", lang).format(link=lessons_link)}</p>'
        "</section>"
    ) if ruled else ""

    return (
        f'<section class="report-section"><h2>{t("playbook.start_here", lang)}</h2>'
        f'<p>{html.escape(pick(book, "intro"))}</p>'
        + (f'<div class="insight-grid">{"".join(checks)}</div>' if checks
           else f'<p class="section-note">{t("playbook.nothing_yet", lang)}</p>')
        + "</section>"
        + ruled_block
    )
def en_link(prefix: str, lang: str, path: str) -> str:
    """Link to an English-only resource (currently: the weekly report) from either tree.

    `path` is always relative to the TREE ROOT, never to the calling page's own
    directory -- that is what lets one formula serve callers at any depth. For English,
    descending from the current page back to root and down into `path` is `prefix + path`.
    For Chinese, `path` lives only in the English tree, so one more step (`../`) escapes
    the zh/ wrapper before the same descent.

    The weekly report pipeline is a separate workflow (docs/WEEKLY_REPORT_WORKFLOW.md) and
    is not translated in this pass; this is how a Chinese page reaches it without
    publishing an untranslated page under a zh/ URL with no explanation.
    """
    if lang != "zh":
        return prefix + path
    return prefix + "../" + path


def xauusd_page(study_list: list[dict[str, object]], weekly: list[dict[str, object]],
                lang: str = "en") -> str:
    selected = [study for study in study_list if study["market"].lower() == "xauusd"]
    latest = weekly[0] if weekly else None
    weekly_html = (
        weekly_card(latest, en_link("../", lang, "xauusd/weekly/"))
        if latest else f'<p class="empty">{t("xauusd.no_weekly_yet", lang)}</p>'
    )
    body = (
        '<main class="shell">'
        + signal_playbook_html("XAUUSD", study_list, "../", lang)
        + f'<h2 class="section-title">{t("xauusd.this_week", lang)}</h2>'
        f'<div class="grid">{weekly_html}</div>'
        + f'<h2 class="section-title">{t("xauusd.studies_heading", lang)} '
        f'<span class="sheet-count">({len(selected)})</span></h2>'
        + '<div class="toolbar"><div class="shell"><input data-search type="search" '
        f'placeholder="{html.escape(t("xauusd.filter_placeholder", lang))}" aria-label="Filter"></div></div>'
        + study_table_html(selected, "../")
        + "</main>"
    )
    return document(
        "XAUUSD",
        t("xauusd.eyebrow", lang),
        t("xauusd.lede", lang),
        body,
        "../",
        lang=lang,
    )


def section_page(
    study_list: list[dict[str, object]],
    market: str,
    title: str,
    lede: str,
    lang: str = "en",
) -> str:
    selected = [study for study in study_list if study["market"].lower() == market]
    body = (
        '<main class="shell">'
        + signal_playbook_html(market.upper(), study_list, "../", lang)
        + f'<h2 class="section-title">{t("section.studies_heading_tpl", lang).format(market=html.escape(market.upper()))} '
        f'<span class="sheet-count">({len(selected)})</span></h2>'
        + '<div class="toolbar"><div class="shell"><input data-search type="search" '
        f'placeholder="{html.escape(t("section.filter_placeholder", lang))}" aria-label="Filter"></div></div>'
        + study_table_html(selected, "../")
        + "</main>"
    )
    return document(title, market, lede, body, "../", lang=lang)
def lessons_page(registry, study_list, lang: str = "en") -> str:
    """What was ruled out, and what the programme learned about testing.

    Split off from the instrument pages because it is the one section that genuinely spans
    both of them, and because it answers a different question: not "should I take this
    trade" but "has this already been tried". Those are different visits and putting them
    on the same page made each harder to find.
    """
    methodology = [x for x in study_list if x.get("theme") == "methodology"]
    totals = (registry or {}).get("totals", {})
    banner = ""
    if totals:
        banner = (
            '<a class="registry-strip" href="../research/null-results/">'
            f'<span><strong>{t("lessons.registry_title", lang)}</strong> — '
            f'{t("lessons.registry_desc", lang)}</span>'
            '<span class="registry-metrics">'
            f'<span><strong>{totals.get("hypotheses", 0)}</strong> {t("lessons.hypotheses_unit", lang)}</span>'
            f'<span><strong>{totals.get("by_verdict", {}).get("survives_screens", 0)}</strong> {t("lessons.survivors_unit", lang)}</span>'
            "</span></a>"
        )
    body = (
        '<main class="shell">'
        f'<section class="report-section"><h2>{t("lessons.why_exists", lang)}</h2>'
        f'<p>{t("lessons.why_p1", lang)}</p>'
        f'<p>{t("lessons.why_p2", lang)}</p>'
        "</section>"
        + (banner if banner else "")
        + (f'<h2 class="section-title">{t("lessons.methodology", lang)} '
           f'<span class="sheet-count">({len(methodology)})</span></h2>'
           f'<p class="section-note">{t("lessons.methodology_note", lang)}</p>'
           f'{study_table_html(methodology, "../")}' if methodology else "")
        + "</main>"
    )
    return document(
        t("lessons.title", lang),
        t("lessons.eyebrow", lang),
        t("lessons.lede", lang),
        body,
        "../",
        lang=lang,
    )


def research_page(data: dict[str, object], study_list: list[dict[str, object]],
                  lang: str = "en") -> str:
    registry = null_registry()
    banner = ""
    if registry:
        totals = registry["totals"]
        # Placed above the study grid on purpose. The studies read as a list of things that
        # were tried; the registry is the only place that says how hard, and it is the
        # first thing worth knowing before asking a question of this data again.
        banner = (
            '<a class="registry-strip" href="null-results/">'
            f'<span><strong>{t("research.registry_title", lang)}</strong> — '
            f'{t("research.registry_desc", lang)}</span>'
            '<span class="registry-metrics">'
            f'<span><strong>{totals["hypotheses"]}</strong> {t("lessons.hypotheses_unit", lang)}</span>'
            f'<span><strong>{totals["by_verdict"].get("survives_screens", 0)}</strong> {t("lessons.survivors_unit", lang)}</span>'
            f'<span><strong>{totals["studies"]}</strong> {t("research.studies_unit", lang)}</span>'
            "</span></a>"
        )
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        f'placeholder="{html.escape(t("section.filter_placeholder", lang))}" aria-label="Filter"></div></div>'
        f'<main class="shell">'
        + (f'<h2 class="section-title">{t("research.start_here", lang)}</h2>{banner}'
           if banner else "")
        + study_table_html(study_list)
        + "</main>"
    )
    return document(
        t("research.title", lang),
        t("research.eyebrow", lang),
        t("research.lede", lang),
        body,
        "../",
        lang=lang,
    )

def overview(
    data: dict[str, object],
    study_list: list[dict[str, object]],
    weekly: list[dict[str, object]],
    lang: str = "en",
) -> str:
    """The front door, organised by what the reader came to do.

    The previous homepage offered "XAUUSD" and "Research" as sibling entries, and a study
    about gold lived under the second one — so there were two plausible doors to the same
    thing and no way to tell which, which is exactly what the reader reported.

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
        t("home.weekly_published_tpl", lang).format(week=weekly[0]["forecast_week"])
        if weekly else t("home.no_weekly", lang)
    )
    primary = (
        '<a class="card card-wide" data-card href="xauusd/">'
        f'<div class="type">{t("home.signal_fired", lang)}</div>'
        f'<h2>{t("home.xauusd_title", lang)}</h2>'
        f'<p>{t("home.xauusd_desc", lang)}</p>'
        '<div class="mini-metrics">'
        f'<span><strong>{counts.get("XAUUSD", 0)}</strong> {t("home.studies_unit", lang)}</span>'
        f'<span><strong>{html.escape(weekly_line)}</strong></span>'
        "</div></a>"
        '<a class="card card-wide" data-card href="tx/">'
        f'<div class="type">{t("home.second_instrument", lang)}</div>'
        f'<h2>{t("home.tx_title", lang)}</h2>'
        f'<p>{t("home.tx_desc", lang)}</p>'
        '<div class="mini-metrics">'
        f'<span><strong>{counts.get("TX", 0)}</strong> {t("home.studies_unit", lang)}</span>'
        "</div></a>"
    )

    secondary = [
        ("lessons/", t("nav.lessons", lang),
         t("home.lessons_desc_tpl", lang).format(
             n=totals.get("hypotheses", 0),
             survived=totals.get("by_verdict", {}).get("survives_screens", 0))),
        ("jargon/", t("nav.jargon", lang), t("home.jargon_desc", lang)),
        (en_link("", lang, "xauusd/weekly/"), t("home.weekly_title", lang), t("home.weekly_desc", lang)),
    ]
    minor = '<div class="grid">' + "".join(
        f'<a class="card" data-card href="{href}"><div class="type">{t("home.card_reference", lang)}</div>'
        f'<h2>{title}</h2><p>{description}</p></a>'
        for href, title, description in secondary
    ) + "</div>"

    body = (
        '<main class="shell">'
        f'<h2 class="section-title">{t("home.what_you_trade", lang)}</h2>'
        f'<div class="grid">{primary}</div>'
        f'<h2 class="section-title">{t("home.reference", lang)}</h2>'
        f"{minor}"
        f'<section class="report-section"><h2>{t("home.what_this_site_is", lang)}</h2>'
        f'<p>{t("home.what_this_site_p1", lang)}</p>'
        f'<p>{t("home.what_this_site_p2", lang)}</p>'
        "</section>"
        "</main>"
    )
    return document(
        t("home.title", lang),
        t("home.eyebrow", lang),
        t("home.lede", lang),
        body,
        lang=lang,
    )



def outputs(data: dict[str, object]) -> dict[Path, str]:
    """Build the single canonical tree.

    Home and sheet/index chrome stay English. Research reports are migrated in place to
    English titles with Chinese bodies; no URL carries a language or layout version.
    """
    lang = "en"
    root = ROOT
    study_list = studies()
    weekly = weekly_summaries()
    generated: dict[Path, str] = {
        root / "index.html": overview(data, study_list, weekly, lang),
        root / "xauusd/index.html": xauusd_page(study_list, weekly, lang),
        root / "tx/index.html": section_page(
            study_list, "tx", "TX Taiwan Index Futures",
            "Studies on Taiwan index futures.", lang),
        root / "research/index.html": research_page(data, study_list, lang),
        root / "lessons/index.html": lessons_page(null_registry(), study_list, lang),
    }
    generated[root / "site/catalog.json"] = (
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    registry = null_registry()
    if registry:
        generated[root / "research/null-results/index.html"] = null_results_page(registry, lang)
    terms = glossary()
    if terms:
        generated[root / "jargon/index.html"] = glossary_page(terms, lang)
    if weekly:
        generated[root / "xauusd/weekly/index.html"] = weekly_summary_page(
            weekly[0], weekly, prefix="../../", source_href=f'{weekly[0]["forecast_week"]}/summary.json', latest=True,
        )
        for summary in weekly:
            generated[root / summary["_relative"] / "index.html"] = weekly_summary_page(
                summary, weekly, prefix="../../../", source_href="summary.json", latest=False,
            )
    for study in study_list:
        generated[root / study["_relative"] / "index.html"] = study_page(study, lang)

    # Old bilingual and layout-version URLs remain as tiny compatibility redirects. They
    # are not archives and are never linked from canonical pages; Git history is the
    # archive. Keeping redirects avoids broken bookmarks without maintaining two sites.
    canonical_html = [path for path in generated if path.suffix == ".html"]
    for target in canonical_html:
        relative = target.relative_to(ROOT)
        if relative.parts[:2] == ("xauusd", "weekly"):
            continue
        legacy = ROOT / "zh" / relative
        generated[legacy] = redirect_document(relative_href(legacy, target))

    v1_targets = {
        "index.html": ROOT / "index.html",
        "xauusd/index.html": ROOT / "xauusd/index.html",
        "xauusd/weekly/index.html": ROOT / "xauusd/weekly/index.html",
        "xauusd/weekly/2026-W34/index.html": ROOT / "xauusd/weekly/2026-W34/index.html",
        "xauusd/weekly/2026-W35/index.html": ROOT / "xauusd/weekly/2026-W35/index.html",
        "tx/index.html": ROOT / "tx/index.html",
        "research/index.html": ROOT / "research/index.html",
        "glossary/index.html": ROOT / "jargon/index.html",
    }
    for relative, target in v1_targets.items():
        legacy = ROOT / "v1" / relative
        if target in generated or target.is_file():
            generated[legacy] = redirect_document(relative_href(legacy, target))
    return generated


def relative_href(from_path: Path, to_path: Path) -> str:
    """A relative directory URL from one generated page to another."""
    rel = os.path.relpath(to_path.parent, start=from_path.parent)
    return "./" if rel == "." else rel.replace(os.sep, "/") + "/"


def redirect_document(target_href: str) -> str:
    target = html.escape(target_href, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="refresh" content="0;url={target}">
  <link rel="canonical" href="{target}">
  <title>Moved · Trading Research</title>
</head>
<body><p>This page moved to <a href="{target}">the canonical site</a>.</p></body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = catalog()
    generated = outputs(data)
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
