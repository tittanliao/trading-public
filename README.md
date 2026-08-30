# trading-public

Reviewed trading research code, reports, and a generated GitHub Pages portal.

## Repository role

This is the active public publication workspace. Private governance, model memory, raw
data, binary source documents, and migration provenance live in sibling
`trading-private`. The old sibling `trading` repository is a read-only migration source.

## Website

Rebuilt clean on 2026-08-31 (`../trading-private/docs/PUBLIC_SITE_REBUILD_SPEC.md`).
Weekly is the primary product; Research is the evidence library behind it. Global
navigation is `Home / Weekly / Research`, and the generator writes exactly eight
canonical routes — nothing else:

```text
/
/xauusd/weekly/
/xauusd/weekly/<forecast-week>/     (only 2026-W36 exists in this phase)
/research/
/research/null-results/
/research/studies/<study-id>/       (only the 3 Phase 1 studies exist in this phase)
```

```sh
python3 site/build.py            # write mode
python3 site/build.py --check    # fails on any generated drift
python3 site/check.py            # route allow-list, dead links, images, privacy
```

Edit `research/studies/<id>/study.json` (narrative, presentation blueprint) or
`xauusd/weekly/<week>/summary.json` (published verbatim by
`../trading-private/scripts/xauusd_weekly/publish_public_summary.py`), then regenerate.
Do not hand-edit any generated `index.html`, `site/style.css`, or `site/catalog.json` —
`site/catalog.json` is now the live route allow-list, not a repository-wide inventory.

A research report is composed from a small named vocabulary of presentation blocks
(`metrics`, `findings`, `table`, `comparison_table`, `matrix_table`, `charts`,
`limitations`, `evidence_links`) driven by an ordered `presentation` list inside the
study's own `study.json`. There is one page renderer for every result shape, not one
renderer per shape — a new shape is handled by writing a new block list, never a new
Python function. Chinese narrative fields (`question_zh`, `findings[].detail_zh`,
`limitations_zh`, `interpretation_zh`, …) sit beside their English source fields in the
same `study.json`; there is no separate language tree.

Reviewed research studies live under `research/studies/<study-id>/`. Every study's
evidence package (`study.json`, `results.json`, `analysis.py`, `charts/`) is retained in
this repository even while its detail page is not yet migrated — see the phase table in
`PUBLIC_SITE_REBUILD_SPEC.md` section 10. An unmigrated study's page returns 404 by
design; its evidence is not deleted, and the Research index does not link to it.

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
study detail pages were removed on 2026-08-31 along with the language/version switch and
the card/table view toggle. They return 404 rather than redirecting to another layout.
Git history is the recovery path; no content was deleted from the sibling `research/`
evidence packages, only their now-unmigrated detail pages.

The 2026-08-18 removal of the earlier file-tree portal, and its `temp/memo.html`
indexing bug, predate this rebuild and are unrelated history.
