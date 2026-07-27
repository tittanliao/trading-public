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

GitHub Pages deploys the reviewed repository root after verifying generation drift,
links, and prohibited private/raw paths.

## Local research data

Raw CSV files are intentionally not public. In a trusted sibling checkout, restore the
legacy relative data paths with:

```sh
python3 tools/link_private_data.py
python3 tools/link_private_data.py --check
```

The links point into sibling `trading-private/data/legacy-trading/raw/`; they are ignored
by Git and are never uploaded to Pages.

## Legacy snapshot

The former mixed landing pages and generators are preserved under `legacy-site/` for
comparison only. They are not the active authoring or deployment workflow.
