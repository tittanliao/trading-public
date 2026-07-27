# trading-public

Reviewed trading research code, reports, and a generated GitHub Pages portal.

## Repository role

This is the active public publication workspace. Private governance, model memory, raw
data, binary source documents, and migration provenance live in sibling
`trading-private`. The old sibling `trading` repository is a read-only migration source.

## Website

The root landing pages are generated from the repository tree:

```sh
python3 site/build.py
python3 site/check.py
```

Edit publishable report/code sources, then regenerate. Do not hand-edit generated
`index.html`, `xauusd/index.html`, `tx/index.html`, `research/index.html`, or
`site/catalog.json`.

Reviewed research studies live under `research/studies/<study-id>/`. Each package has a
sanitized manifest, aggregate results, and a reproducible Python method. The build
creates a legacy-hub-style study card and detailed HTML report automatically; raw CSV
and private decision records remain in `trading-private`.

Cross-repository implementation and Claude/Codex continuation are governed by
`../trading-private/docs/RESEARCH_DEVELOPMENT_SPEC.md`. The Public-only safety fallback
is `docs/RESEARCH_PUBLICATION_SPEC.md`.

GitHub Pages deploys the reviewed repository root after verifying generation drift,
links, and prohibited private/raw paths.

## Retired import

The former file-tree portal, imported strategy scripts/reports, and compatibility tools
are preserved under `_retire/public-legacy-import-20260727/`. They are excluded from the
active catalog and navigation. Active pages are study-first and every visible card opens
a human-readable HTML analysis.
