# Repository Instructions

## Role

This repository contains only intentionally published code, research outputs, and the
generated GitHub Pages site. Private memory, raw data, binary documents, governance, and
cross-model handoffs belong in sibling `trading-private`.

## Source of truth

When both sibling repositories are available, read `../trading-private/AGENTS.md` and
`../trading-private/docs/RESEARCH_DEVELOPMENT_SPEC.md` first. Otherwise read
`docs/RESEARCH_PUBLICATION_SPEC.md` and limit work to publication-only scope. Public
facts must be supported by reproducible code, versioned inputs, reports, and Git
history; model-specific memory is not a fact source.

## Safety

- Never copy `.claude/memory`, private journals, DOCX/XLSX, raw CSV, credentials, cookies,
  tokens, private keys, or unreviewed source documents into this repository.
- Do not change formal strategy logic, parameters, performance values, or research
  conclusions without explicit owner scope.
- Generated landing pages are rebuilt with `python3 site/build.py`; do not hand-edit them.
- Legacy sibling `trading` and Google Drive/XAUUSD are read-only sources.
- Before commit/push, run `python3 site/build.py --check` and `python3 site/check.py`.
