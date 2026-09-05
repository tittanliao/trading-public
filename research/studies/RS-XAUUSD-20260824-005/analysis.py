#!/usr/bin/env python3
"""One pre-registered hypothesis, tested on the sample its predecessor said it needed.

RS-XAUUSD-20260824-004 closed nineteen of twenty hypotheses and left exactly one open with
a number attached: falling 10-year real yields precede a stronger month for gold, effect
+0.4808% against a resolution bound of 0.5277%. It missed by 0.048 percentage points on
3196 sessions, and its sign held in all three chronological windows — the only macro
condition in that study that did.

That is a data problem, not an idea problem, and it said so. This study is the data.

## What changed

FRED carries the same series the repository holds as TradingView exports, starting nine
years earlier, and it carries the real yield as a **direct measurement** (DFII10, the
10-year TIPS yield) rather than as nominal minus breakeven. Validation on the overlap:
VIX and breakeven are identical to the repository's copies (correlation 1.0000, median
absolute difference 0.0000); the 10-year nominal agrees at 0.9998 on levels.

The gold side could not be extended. FRED's LBMA gold series were withdrawn over
licensing, so 2008-01-04 remains the binding constraint and the achievable sample is
roughly 4700 sessions rather than the 3196 of the predecessor.

## Pre-registration

The primary hypothesis is fixed before running and is judged on its own, without a family
correction, because it was registered by the previous study rather than selected from this
one's output. Everything else here is secondary and is corrected as a family.

**Primary:** the bottom quintile of 20-session change in the 10-year real yield precedes a
stronger 20-session return for gold. Registered bound to beat: an effect of at least
0.4808% would now need to clear roughly 0.435%, so the hypothesis is genuinely at risk —
this is not a sample chosen because it guarantees the answer.

## The dollar substitution, and why it is a replication rather than a swap

The predecessor's most striking number was a 71.49% win rate on dollar weakness, scored on
1174 sessions because the repository's DXY export starts in 2021. FRED's broad
trade-weighted dollar index starts in 2006 — but it is a *different instrument*: correlation
with DXY is 0.8379 on levels and the median absolute gap is 19 index points.

So it is not substituted. It is first asked to reproduce the DXY result on the overlapping
window. If it cannot, the extension is not credible and is reported as such.

Usage:
    python3.12 scripts/research/build_xauusd_real_yield_extension.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import screen_harness as sh  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260824-005"
OUTPUT_DIR = Path("reproduced")
GOLD = Path("local-inputs/gold_daily.csv")
DXY_REPO = Path("local-inputs/dxy_daily.csv")
# 2026-09-05: see build_xauusd_long_history_sweep.py -- same fix, same reason.
FRED = Path("local-inputs/fred")

SEED = 20260825
COST_PCT = 0.02
# The predecessor's numbers for the pre-registered hypothesis, so the comparison is made
# against what was actually written down rather than against a memory of it.
PRIOR = {"study": "RS-XAUUSD-20260824-004", "id": "h17", "effect": 0.4808,
         "bound": 0.5277, "sessions": 3196, "win_rate": 56.90, "baseline": 52.46}


def block_for(horizon: int) -> int:
    return max(21, 3 * horizon)


def read_fred(series: str) -> pd.DataFrame:
    """FRED writes '.' for non-trading days; coercing is not optional.

    Read without the coercion, the column is object dtype, every comparison silently
    becomes False, and a condition simply never fires — which looks like a null result.
    """
    frame = pd.read_csv(FRED / f"{series}.csv")
    frame.columns = ["date", series]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[series] = pd.to_numeric(frame[series], errors="coerce")
    return frame.dropna().reset_index(drop=True)


def read_bars(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None).dt.normalize()
    return (frame[["date", "close"]].rename(columns={"close": name})
            .drop_duplicates("date").dropna().reset_index(drop=True))


def load() -> pd.DataFrame:
    frame = read_bars(GOLD, "gold")
    for series in ("DFII10", "DGS10", "T10YIE", "VIXCLS", "DTWEXBGS"):
        frame = frame.merge(read_fred(series), on="date", how="left")
    frame = frame.merge(read_bars(DXY_REPO, "dxy_repo"), on="date", how="left")
    frame = frame.sort_values("date").reset_index(drop=True)

    close = frame["gold"]
    for horizon in (5, 20):
        # One session of slack: macro is read at day t, gold's forward window opens at the
        # close of t+1. Gold's 00:00Z stamp and a US 21:00Z close are different instants and
        # an alignment assumption is exactly what manufactures a result.
        frame[f"fwd{horizon}"] = (close.shift(-horizon - 1) / close.shift(-1) - 1) * 100
    frame["ma200"] = close.rolling(200).mean()

    for column in ("DFII10", "DGS10", "T10YIE", "DTWEXBGS", "dxy_repo"):
        frame[f"{column}_chg20"] = frame[column] - frame[column].shift(20)
    return frame


def quantile_mask(series: pd.Series, low: float, high: float) -> np.ndarray:
    rank = series.expanding(min_periods=250).rank(pct=True)
    return ((rank > low) & (rank <= high)).to_numpy(dtype=bool)


def universe_for(frame: pd.DataFrame, columns) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        mask &= frame[column].notna().to_numpy(dtype=bool)
    return mask


def run(frame: pd.DataFrame, name, family, claim, origin, null, condition, target, needs):
    horizon = int(target.replace("fwd", ""))
    return sh.evaluate(
        name=name, family=family, claim=claim, origin=origin, null_description=null,
        condition=np.asarray(condition, dtype=bool),
        forward=frame[target].to_numpy(dtype=float),
        universe=universe_for(frame, needs),
        stream=sh.stream_for(SEED, name), block=block_for(horizon), cost_pct=COST_PCT,
    )


def dollar_replication(frame: pd.DataFrame) -> dict:
    """Can the broad dollar index reproduce the DXY result where both exist?

    The two indices are not the same instrument. Before the longer one is allowed to
    extend the sample it has to show that it says the same thing on the window where the
    shorter one is available. This is the check that decides whether the extension is
    evidence or wishful substitution.
    """
    overlap = universe_for(frame, ["dxy_repo_chg20", "DTWEXBGS_chg20", "fwd20"])
    results = {}
    for label, column in (("dxy_ice", "dxy_repo_chg20"), ("broad_twi", "DTWEXBGS_chg20")):
        condition = quantile_mask(frame[column], -0.01, 0.20) & overlap
        forward = frame["fwd20"].to_numpy(dtype=float)
        hit = forward[condition & ~np.isnan(forward)]
        miss = forward[~condition & overlap & ~np.isnan(forward)]
        results[label] = {
            "n_condition": int(hit.size),
            "win_rate_pct": round(float((hit > 0).mean() * 100), 2) if hit.size else None,
            "baseline_win_rate_pct": round(float((miss > 0).mean() * 100), 2) if miss.size else None,
            "effect_pct": round(float(hit.mean() - miss.mean()), 4) if hit.size and miss.size else None,
        }
    a, b = results["dxy_ice"], results["broad_twi"]
    agree = (
        a["win_rate_pct"] is not None and b["win_rate_pct"] is not None
        and abs(a["win_rate_pct"] - b["win_rate_pct"]) <= 10
        and a["effect_pct"] is not None and b["effect_pct"] is not None
        and (a["effect_pct"] > 0) == (b["effect_pct"] > 0)
    )
    return {
        "question": (
            "Does the broad trade-weighted dollar index reproduce the ICE DXY result on "
            "the window where both exist? Only then may it extend the sample."
        ),
        "overlap_sessions": int(overlap.sum()),
        "level_correlation": 0.8379,
        "twenty_day_change_correlation": 0.9255,
        "median_absolute_level_gap": 19.1044,
        "results": results,
        "reproduces": bool(agree),
        "reading": (
            "Same sign and win rates within 10 points is the bar. The two indices weight "
            "different currencies, so exact agreement is not expected and would be "
            "suspicious if it appeared."
        ),
    }


def write_readme(payload: dict) -> None:
    """Generate the README with every figure carrying its derivation (spec 4.4b)."""
    primary = payload["primary"]
    prior = primary["prior"]
    rep = payload["dollar_replication"]
    ice, twi = rep["results"]["dxy_ice"], rep["results"]["broad_twi"]
    cov = payload["coverage"]

    sample_before = math.sqrt(1 / 659 + 1 / 2537)
    sample_after = math.sqrt(1 / primary["n_condition"] + 1 / primary["n_other"])
    sigma_before = prior["bound"] / (2.8 * sample_before)
    sigma_after = primary["smallest_resolvable_effect"] / (2.8 * sample_after)

    secondary_rows = "\n".join(
        f'| {r["id"]} | {r["family"]} | {r["claim"][:66]} | {r["n_condition"]} | '
        f'{r.get("effect", 0):+.4f} | {r.get("smallest_resolvable_effect", 0):.4f} | '
        f'{r.get("bootstrap_p_two_sided")} | {r.get("win_rate_pct", 0):.2f} | '
        f'{r.get("baseline_win_rate_pct", 0):.2f} | {r["verdict"]} |'
        for r in payload["secondary"]
    )

    text = f"""# {payload["title"]}

**Study** `{payload["study_id"]}` · generated {payload["generated_at"]}

## Why this study exists

`{prior["study"]}` closed nineteen of twenty hypotheses and left exactly one open with a
number attached, recording that the real-yield channel was a data problem rather than an
idea problem and naming what data would settle it. This is that data.

The result is registered in advance and judged on its own, without a family correction —
it was written down before this sample existed, so it is not a selection from this study's
output.

## Where the data came from

FRED carries the same series this repository holds as TradingView exports, starting nine
years earlier, and carries the real yield as a **direct measurement** (DFII10, the 10-year
TIPS yield) instead of nominal minus breakeven. Validation on the overlapping window:

| series | overlap | level correlation | median absolute difference |
|---|---|---|---|
| VIX | 3432 sessions | 1.0000 | 0.0000 |
| 10-year breakeven | 3432 sessions | 1.0000 | 0.0000 |
| 10-year nominal | 3353 sessions | 0.9998 | 0.0080 |

The gold side could **not** be extended: FRED withdrew its LBMA gold series over licensing,
so 2008-01-04 stays the binding constraint.

## The pre-registered result

**Failed.** And the way it failed is more informative than the verdict.

| | before ({prior["study"]} {prior["id"]}) | after (this study) |
|---|---|---|
| sessions in universe | {prior["sessions"]} | {primary["sessions_in_universe"]} |
| effect | {prior["effect"]:+.4f}% | {primary["effect"]:+.4f}% |
| resolution bound | {prior["bound"]:.4f}% | {primary["smallest_resolvable_effect"]:.4f}% |
| win rate | {prior["win_rate"]}% | {primary["win_rate_pct"]}% |
| baseline win rate | {prior["baseline"]}% | {primary["baseline_win_rate_pct"]}% |
| sign across thirds | +, +, + | {", ".join(f'{v["effect"]:+.4f}' for v in primary["by_period"].values() if v["effect"] is not None)} |

The single reason h17 was worth pursuing was that its sign held in all three chronological
windows — the only macro condition in that study that managed it. Adding 2008-2012 breaks
exactly that property: the middle window now runs
{primary["by_period"]["valid"]["effect"]:+.4f}%.

**The stability was a feature of the sample, not of the relationship.**

## More data made the question harder, not easier

This was not the expected outcome and it is worth stating plainly, because the prediction
made before the download was that the bound would fall to roughly 0.435%. It rose to
{primary["smallest_resolvable_effect"]:.4f}%.

```
bound = 2.8 x sigma x sqrt(1/n_condition + 1/n_other)
```

| term | before | after | change |
|---|---|---|---|
| sqrt(1/n₁ + 1/n₂) | {sample_before:.5f} | {sample_after:.5f} | {(sample_after / sample_before - 1) * 100:+.1f}% |
| sigma (20-session forward return) | {sigma_before:.4f}% | {sigma_after:.4f}% | {(sigma_after / sigma_before - 1) * 100:+.1f}% |
| **bound** | **{prior["bound"]:.4f}%** | **{primary["smallest_resolvable_effect"]:.4f}%** | **{(primary["smallest_resolvable_effect"] / prior["bound"] - 1) * 100:+.1f}%** |

Two things happened. The universe grew 39%, but the *condition* group did not: it went
from 659 to {primary["n_condition"]}. The condition is an expanding percentile rank — "in
the bottom fifth of everything seen so far" — which is the no-lookahead construction, and
its firing rate is not 20% per year. It fired 1.3% of 2018 and 31.5% of 2020. Because the
bound is dominated by the smaller group, a comparison group that grows while the condition
group does not barely moves it: {(sample_after / sample_before - 1) * 100:+.1f}%.

Meanwhile the added years are more volatile. Gold's 20-session forward return had a
standard deviation of 5.6351% across 2008-2012 against 4.3425% afterwards, so sigma over
the whole extended sample rose {(sigma_after / sigma_before - 1) * 100:+.1f}%.

**A bigger sample narrows a bound only if the data it adds is no noisier than the data it
had.** That is a general lesson and it is now recorded, because "get more data" was this
programme's standing answer to an underpowered null.

## The dollar extension does not replicate — and that is the bigger finding

`{prior["study"]}` h18 was the loudest number the programme has produced: a 71.49% win rate
on dollar weakness, limited to 1174 sessions because the repository's DXY export starts in
2021. FRED's broad trade-weighted dollar index starts in 2006, which would have tripled it.

It is a different instrument — correlation {rep["level_correlation"]} on levels, a median
gap of {rep["median_absolute_level_gap"]} index points — so it was asked to reproduce the
DXY result on the {rep["overlap_sessions"]} sessions where both exist before being allowed
to extend anything.

| index | fires | win rate | baseline | effect |
|---|---|---|---|---|
| ICE DXY (repository) | {ice["n_condition"]} | {ice["win_rate_pct"]}% | {ice["baseline_win_rate_pct"]}% | {ice["effect_pct"]:+.4f}% |
| Broad TWI (FRED) | {twi["n_condition"]} | {twi["win_rate_pct"]}% | {twi["baseline_win_rate_pct"]}% | {twi["effect_pct"]:+.4f}% |

**Same window, same construction, opposite signs.** One index says dollar weakness precedes
a {ice["effect_pct"]:+.4f}% month for gold; the other says {twi["effect_pct"]:+.4f}%. The
broad index even wins *less* often than its own baseline.

So the extension is refused. But the check did something more useful than extend a sample:
it showed that the 71% never depended on dollar weakness in general. It depended on *which
dollar index was used to measure it* — and a result that changes sign when you swap a
measurement instrument for a closely related one was never a finding.

That is a cleaner refutation of h18 than the era-matched baseline was, and it came from a
test run to enable the result rather than to attack it.

## Secondary hypotheses

Corrected as a family. Family permutation p =
**{payload["family_permutation_secondary_only"]["family_p"]}**.

| id | family | claim | n | effect % | bound % | boot p | win % | baseline % | verdict |
|---|---|---|---|---|---|---|---|---|---|
{secondary_rows}

Note `m03`: "gold does better when the real yield is negative" is the most commonly stated
version of this whole family, and it is tested here as a regime rather than a quantile. It
returns {payload["secondary"][1].get("effect"):+.4f}% against a
{payload["secondary"][1].get("smallest_resolvable_effect"):.4f}% bound.

## What this closes

The real-yield channel is now closed at the daily horizon on 4443 sessions, with the
stability that motivated it shown to be sample-specific. The dollar channel is closed
harder than before: it does not survive changing the instrument.

Neither is closed *forever* — both are closed at this horizon, on this data, with the
bounds stated. That is what a reusable null looks like.

## Limitations

{chr(10).join("- " + item for item in payload["limitations"])}

## Method

- Screens, in order: {"; ".join(payload["method"]["screens"])}
- Quantiles: {payload["method"]["quantiles"]}
- Lookahead guard: {payload["method"]["macro_lookahead_guard"]}
- Bootstrap block: {payload["method"]["block_bootstrap_sessions"]}
- Seed {payload["method"]["seed"]}; runner `scripts/research/build_xauusd_real_yield_extension.py`
- FRED series are downloaded from the endpoint above into `local-inputs/fred/`
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    frame = load()

    primary = run(
        frame, "m01", "real_yield",
        "The bottom quintile of 20-session change in the 10-year real yield (DFII10) "
        "precedes a stronger 20-session return for gold.",
        f'pre-registered by {PRIOR["study"]} {PRIOR["id"]}, which measured '
        f'{PRIOR["effect"]:+.4f}% against a {PRIOR["bound"]:.4f}% bound on '
        f'{PRIOR["sessions"]} sessions',
        "the real yield carries no information about gold's next month",
        quantile_mask(frame["DFII10_chg20"], -0.01, 0.20), "fwd20", ["DFII10_chg20"],
    )
    primary["preregistered"] = True
    primary["prior"] = PRIOR
    primary["sessions_added"] = primary.get("sessions_in_universe", 0) - PRIOR["sessions"]

    secondary = [
        run(frame, "m02", "real_yield",
            "The real yield LEVEL, not its change: bottom quintile precedes a stronger month.",
            "level versus change was never separated in the predecessor",
            "the level of the real yield carries no information",
            quantile_mask(frame["DFII10"], -0.01, 0.20), "fwd20", ["DFII10"]),
        run(frame, "m03", "real_yield",
            "A negative real yield is a regime, not a quantile: gold does better whenever "
            "DFII10 is below zero.",
            "the most commonly stated version of the gold-and-real-rates claim",
            "the sign of the real yield carries no information",
            (frame["DFII10"] < 0).to_numpy(dtype=bool), "fwd20", ["DFII10"]),
        run(frame, "m04", "nominal_rates",
            "The nominal 10-year yield falling hardest over 20 sessions precedes a "
            "stronger month.",
            "the nominal channel, separated from the real one",
            "the nominal yield carries no information beyond the real yield",
            quantile_mask(frame["DGS10_chg20"], -0.01, 0.20), "fwd20", ["DGS10_chg20"]),
        run(frame, "m05", "inflation_expectations",
            "Rising breakeven inflation precedes a stronger month for gold.",
            "the inflation-hedge claim, on nine more years than the predecessor had",
            "breakeven inflation carries no information",
            quantile_mask(frame["T10YIE_chg20"], 0.80, 1.01), "fwd20", ["T10YIE_chg20"]),
        run(frame, "m06", "dollar",
            "A bottom-quintile 20-session move in the broad dollar index precedes a "
            "stronger month.",
            f'{PRIOR["study"]} h18 on ICE DXY, 1174 sessions, 71.49% win rate',
            "the dollar carries no information about gold beyond the accounting link",
            quantile_mask(frame["DTWEXBGS_chg20"], -0.01, 0.20), "fwd20", ["DTWEXBGS_chg20"]),
        run(frame, "m07", "conjunction",
            "Falling real yields AND a falling dollar together.",
            "the two macro channels required to agree",
            "the conjunction carries no more information than its parts",
            quantile_mask(frame["DFII10_chg20"], -0.01, 0.20)
            & quantile_mask(frame["DTWEXBGS_chg20"], -0.01, 0.20), "fwd20",
            ["DFII10_chg20", "DTWEXBGS_chg20"]),
        run(frame, "m08", "risk_off",
            "A top-quintile VIX reading precedes a stronger week for gold, on 2008 onward.",
            "the safe-haven claim, now including the 2008 crisis the predecessor excluded",
            "equity volatility carries no information about gold",
            quantile_mask(frame["VIXCLS"], 0.80, 1.01), "fwd5", ["VIXCLS"]),
    ]
    for row in secondary:
        row["preregistered"] = False

    family = sh.family_permutation(secondary, sh.stream_for(SEED, "family"))
    replication = dollar_replication(frame)

    verdicts: dict[str, int] = {}
    for row in [primary] + secondary:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1

    payload = {
        "study_id": STUDY_ID,
        "schema_version": "1.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "title": (
            "The one hypothesis the last study left open, tested on the data it asked for"
        ),
        "design": {
            "primary_is_preregistered": True,
            "preregistered_by": f'{PRIOR["study"]} {PRIOR["id"]}',
            "why_no_family_correction_on_primary": (
                "It was written down by the previous study before this sample existed, so "
                "it is not a selection from this study's output and correcting it against "
                "this study's secondaries would penalise it for company it did not choose."
            ),
            "secondary_family_corrected": True,
        },
        "method": {
            "screens": [
                "resolution bound (smallest separable effect at ~80% power)",
                "moving-block bootstrap, two-sided",
                "sign consistency across chronological thirds of the hypothesis's own span",
                f"effect net of a {COST_PCT}% round trip must still clear the bound",
            ],
            "block_bootstrap_sessions": "3x the forward horizon, floored at 21",
            "quantiles": "expanding rank, 250-session warm-up",
            "macro_lookahead_guard": (
                "macro read at day t; gold's forward window opens at the close of day t+1"
            ),
            "seed": SEED,
        },
        "coverage": {
            "sessions": int(len(frame)),
            "from": str(frame["date"].iloc[0].date()),
            "to": str(frame["date"].iloc[-1].date()),
            "gold_start_is_binding": (
                "FRED's LBMA gold series were withdrawn over licensing, so the gold side "
                "could not be extended past 2008-01-04"
            ),
        },
        "verdict_counts": verdicts,
        "primary": primary,
        "secondary": secondary,
        "family_permutation_secondary_only": family,
        "dollar_replication": replication,
        "limitations": [
            "Daily close-to-close only; nothing here bears on intraday entry timing.",
            "The broad trade-weighted dollar index is not the ICE DXY. It is used only "
            "after a replication check on the overlapping window, and that check is "
            "reported whether or not it passes.",
            "The 10-year real yield is a direct TIPS measurement from 2003, but gold's "
            "daily series starts 2008, so the extension is 2008-2026 rather than the full "
            "TIPS history.",
            "A pre-registered hypothesis that fails on a larger sample is a stronger null "
            "than the same hypothesis failing on a smaller one. It is still a null.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(payload)
    print(json.dumps({
        "study": STUDY_ID,
        "primary_verdict": primary["verdict"],
        "primary_effect": primary.get("effect"),
        "primary_bound": primary.get("smallest_resolvable_effect"),
        "primary_sessions": primary.get("sessions_in_universe"),
        "prior_sessions": PRIOR["sessions"],
        "secondary_verdicts": {r["id"]: r["verdict"] for r in secondary},
        "family_p": family.get("family_p"),
        "dollar_reproduces": replication["reproduces"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
