# trading-public

Reviewed trading research code, reports, and a generated GitHub Pages portal.

## Repository role

This is the active public publication workspace. Private governance, model memory, raw
data, binary source documents, and migration provenance live in sibling
`trading-private`. The old sibling `trading` repository is a read-only migration source.

## Website

Rebuilt clean on 2026-08-30 (`../trading-private/docs/PUBLIC_SITE_REBUILD_SPEC.md`).
Weekly is the primary product; Research is the evidence library behind it. Global
navigation is `Home / Weekly / Research`, and the generator writes only the canonical
route allow-list below — nothing else:

```text
/
/xauusd/weekly/
/xauusd/weekly/<forecast-week>/     (one page per published edition; W34, W35, W36)
/research/
/research/null-results/
/research/studies/<study-id>/       (one per published study; 30 at present)
```

Every published Weekly edition keeps a dated archive page. That is a contract, not a
preference: `docs/BASE_WEEKLY_REPORT_WORKFLOW.md` declares `/xauusd/weekly/<week>/` stable
and Private release receipts record those exact URLs as publication proof, so deleting one
breaks a signed record. `site/check.py` fails if a published `summary.json` has no page.

```sh
python3 site/build.py            # write mode
python3 site/build.py --check    # fails on any generated drift
python3 site/check.py            # route allow-list, dead links, images, privacy
```

The narrative and blueprint are authored in **sibling `trading-private`** and carried here
by `scripts/research/export_public_results.py`; editing the copy in this repository works
until the next export silently reverts it. Edit
`../trading-private/research/studies/<id>/study.json` (narrative, presentation blueprint) or
`xauusd/weekly/<week>/summary.json` (published verbatim by
`../trading-private/scripts/xauusd_weekly/publish_public_summary.py`), then regenerate.
Do not hand-edit any generated `index.html`, `site/style.css`, or `site/catalog.json` —
`site/catalog.json` is now the live route allow-list, not a repository-wide inventory.

Two conventions every table inherits from the shared renderer, so they cannot regress
page by page: each is sortable by clicking a column header (the raw value travels in
`data-sort`, so formatted text is never re-parsed), and each sizes to the available width,
scrolling sideways only when the viewport is genuinely too narrow. Charts render full
width and legible inline; opening the original PNG is a convenience, never a requirement.

A research report is composed from a small named vocabulary of presentation blocks
(`metrics`, `findings`, `table`, `comparison_table`, `matrix_table`, `metric_table`,
`evidence_pair`, `interpretation`, `limitations`, `evidence_links`) driven by an ordered
`presentation` list inside the study's own `study.json`. There is one page renderer for every result shape, not one
renderer per shape — a new shape is handled by writing a new block list, never a new
Python function. Chinese narrative fields (`question_zh`, `findings[].detail_zh`,
`limitations_zh`, `interpretation_zh`, …) sit beside their English source fields in the
same `study.json`; there is no separate language tree.

Reviewed research studies live under `research/studies/<study-id>/`. The 2026-08-30
rebuild and its migration are complete: 30 studies have a reader page and nothing is
queued. One study (`RS-XAUUSD-20260727-002`) keeps its **evidence package published and
its reader page withheld** — `study.json`, `results.json` and `analysis.py` are served
from this repository exactly like any other study, but it has no page and the Research
index says so, because it is `status: pending` and superseded by the -003/-004/-005 trio.
That is a withheld page, not an unpublished study.

A reader page shows a **selected** set of charts, each immediately followed by the table it
explains. A study may declare charts in `results.json` that no page shows; those PNGs stay
in the evidence package and are reachable through the Evidence section. `site/check.py`
enforces the contract in that direction: every chart a page shows must be declared, every
declared chart must have its PNG, and no page may show the same chart twice.

Reviewed XAUUSD weekly aggregates live under `xauusd/weekly/<forecast-week>/`. The
stable latest URL is `/xauusd/weekly/`; this address and its dated archive are the two
routes this rebuild explicitly kept in place rather than migrating — the Private
publication contract and every existing receipt already point at them. The narrative
language of a published `summary.json` follows the reviewed source reports verbatim; the
publisher performs sanitization and structural validation only and never translates.

Cross-repository implementation and Claude/Codex continuation are governed by
`../trading-private/docs/RESEARCH_DEVELOPMENT_SPEC.md`. The Public-only safety fallback
is `docs/RESEARCH_PUBLICATION_SPEC.md`.

GitHub Pages deploys the reviewed repository root after verifying generation drift,
links, images, and prohibited private/raw paths.

## Retired routes

`jargon/`, `lessons/`, `tx/`, `v1/`, `zh/`, `xauusd/index.html`, and the 28 non-Phase-1
study detail pages were removed on 2026-08-30 along with the language/version switch and
the card/table view toggle. They return 404 rather than redirecting to another layout.
Git history is the recovery path; no content was deleted from the sibling `research/`
evidence packages, only their now-unmigrated detail pages.

The 2026-08-18 removal of the earlier file-tree portal, and its `temp/memo.html`
indexing bug, predate this rebuild and are unrelated history.
