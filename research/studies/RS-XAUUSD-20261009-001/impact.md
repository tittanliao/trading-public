# RS-XAUUSD-20261009-001 — live impact

What `請分析` does differently because of this study, adopted 2026-10-09.

## S1 forward check

- Surface: `請分析 S1 recommendation rule`
- Rule: S1 V3.9 is on a pre-registered forward check (PREREG-20261009-001): if its next 31 tester trades from 2026-10-09 sum to -1.50% or worse, or the running sum reaches -3.67% first, 請分析 stops recommending S1 entries until it is reviewed. -1.50% is the worst earlier 31-trade window.
- Citation: `research/studies/RS-XAUUSD-20261009-001/results.json` → `decline.rolling_windows.reference_min_total_return_pct`
- Scope: a recommendation rule. It changes no S1 logic or parameter and places, sizes or closes no order. Until the check reports, S1 is recommended as before at the fixed per-trade risk.
- Revocation: the check reports `continue`, or the rule is withdrawn.
