# Live impact — RS-XAUUSD-20260727-001 (revision 3)

Current as of 2026-07-28. This file is a snapshot of what `請分析` does differently
today; the history lives in the private decision record.

## 1. Macro composite verdict → S1 recommendation score

- **Surface**: `請分析` — "S1 recommendation score (advisory, not a trading gate)";
  restated in the strategy context reference "Macro context (S1, advisory only)".
- **Rule**: after the 0–100 core score, add the integer rank score of the Macro
  composite verdict read this session (`STRONG BUY +1`, `WAIT 0`, `NEUTRAL −1`), then
  clamp to 0–100.
- **Citation**: `research/studies/RS-XAUUSD-20260727-001/results.json`, key
  `by_macro_verdict[<verdict>].rank_score`. Report `n`, `win_rate_pct`,
  `profit_factor` and `win_rate_ci95_pct` from the same object.
- **Scope / limits**: advisory only. Range −1…+1. It never creates, blocks or cancels a
  formal V3.9 signal, never changes position size, and is not the S2 Macro
  position-sizing filter in the strategy context reference, which is a separate hard rule this study
  does not touch. If the live Macro reading is unavailable or its daily inputs are more
  than 4 days old, use `0` and mark the Macro evidence `unknown`.
- **Revocation**: a later registered study on current S1 data that supersedes the Macro
  breakdown, or an owner decision to remove Macro from the score. Revocation empties
  `policy_impacts`, sets `status: pending`, and regenerates this file and
  `docs/ADOPTED_RESEARCH.md`.

## 2. 30-minute entry slot → S1 recommendation score

- **Surface**: `請分析` — same section; restated in the trading profile reference
  session-context section.
- **Rule**: floor the fresh TradingView displayed time to its 30-minute bar start
  (`HH:00`/`HH:30`, Asia/Taipei), then add that slot's integer rank score (−2…+2).
- **Citation**: `research/studies/RS-XAUUSD-20260727-001/results.json`, key
  `by_entry_30m[HH:MM].rank_score`. Report `n`, `win_rate_pct`, `profit_factor` and
  `win_rate_ci95_pct` from the same object.
- **Scope / limits**: advisory only. Range −2…+2. No slot is a hard entry gate and a
  broad Asia/Europe/US average is never substituted. A slot with `low_sample: true`
  (`n < 5`: `04:00`, `05:30`, `06:00`, `07:00`, `18:00`) scores `0` and its timing
  evidence is reported as `low-sample`. `by_session[...]` is descriptive context and is
  never read by this rule.
- **Revocation**: as above; additionally, do not increase timing weight until a later
  registered study shows stable chronological/OOS slot effects.
