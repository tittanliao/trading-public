# Claude public workspace entry

This repository is the public publishing surface. In a sibling checkout, restore shared
context from `../trading-private/CLAUDE.md` and `../trading-private/docs/ai/` before work.

Public updates must remain model-neutral and evidence-backed. Claude-only memory and
legacy instructions are not a Source of Truth. Use the Private `/update-research`
command for the cross-repository publishing checklist.

Run before commit:

```sh
python3 site/build.py --check
python3 site/check.py
```
