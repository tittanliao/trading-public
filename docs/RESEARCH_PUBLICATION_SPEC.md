# Research publication specification

The canonical cross-repository implementation contract is:

`../trading-private/docs/RESEARCH_DEVELOPMENT_SPEC.md`

Claude, Codex, or another executor must read that file before changing a study or the
research site. This Public document is the safe fallback for publication-only work.

## Public responsibility

`trading-public` contains only intentionally reviewed:

- `research/studies/<study-id>/study.json`;
- aggregate `results.json`;
- publishable `analysis.py`;
- site generator/source assets;
- generated indexes and study HTML.

Never publish raw CSV, absolute private paths, source manifests, journals, state
history, weekly model artifacts, private decision/handoff records, credentials, tokens,
cookies, or unreviewed documents.

The only weekly exception is the reviewed aggregate
`xauusd/weekly/<forecast-week>/summary.json` produced by the Private allow-list exporter.
It exposes comparisons and adopted conclusions, not any weekly model artifact or
private provenance. One source is labeled `single_source`; only multiple independent
same-input sources may be labeled `multi_source`.

## Implementation

1. Confirm the matching Private study is registered and at least `reproduced`.
2. Create the sanitized three-file Public source package.
3. Run the Public method against locally authorized inputs and compare every published
   headline/subgroup value.
4. Run:

```sh
python3 site/build.py
python3 site/build.py --check
python3 site/check.py
```

5. Review staged paths and scan for private strings and raw-data extensions.
6. Commit and push without history rewriting.
7. Confirm GitHub Pages success and HTTP 200 for root, research index, and study page.
8. Return the Public commit/deployment result to the Private decision and handoff
   records.

Generated HTML is never hand-edited. A new visualization or report shape is implemented
generically in `site/build.py`.
