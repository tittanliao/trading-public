# Claude public workspace entry

This repository is the public publishing surface. In a sibling checkout, restore shared
context from `../trading-private/CLAUDE.md` and
`../trading-private/docs/RESEARCH_DEVELOPMENT_SPEC.md` before work. If the sibling
Private checkout is unavailable, use `docs/RESEARCH_PUBLICATION_SPEC.md` for
publication-only work and do not infer Private evidence or decisions.

Public updates must remain model-neutral and evidence-backed. Claude-only memory and
legacy instructions are not a Source of Truth. Use the Private `/update-research`
command for the cross-repository publishing checklist. Resume partial research only
from the Private study `handoff.md`.

Run before commit:

```sh
python3 site/build.py --check
python3 site/check.py
```
