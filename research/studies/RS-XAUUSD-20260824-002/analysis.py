#!/usr/bin/env python3
"""RS-XAUUSD-20260824-002 — CFTC positioning, and a placebo the publication lag hands us.

The owner supplied two hypotheses and one observation. The observation is the more valuable
of the three, and it shapes the whole study.

## The observation, and why it is a study design rather than a hypothesis

A COT report is dated Tuesday and published Friday afternoon US time — Saturday morning in
Taipei. By the time anyone reads it, Wednesday, Thursday and Friday have already traded.
The owner noticed this and asked to look at both windows separately.

That is a **built-in placebo test**, and most positioning research does not have one:

- `fwd_wed_fri` covers days that were *already history* when the report landed. Nothing in
  the report can have caused them. If positioning "predicts" this window, it is describing
  the move the positions were built during.
- `fwd_mon_tue` covers the first two sessions *after* publication — the only window a
  reader could have acted on.

A relationship that appears in the first and not the second is mechanical. One that appears
in the second is a claim. Reporting only a Tuesday-to-Tuesday weekly return, as almost
every COT study does, blends the two and cannot tell them apart.

## What the sample can and cannot do

Thirty-one weeks. Split into two groups of roughly fifteen, and with weekly gold returns
carrying a standard deviation near 2.5%, the smallest difference this sample can resolve is
around 2.5 percentage points *per week*. Almost nothing real is that large.

So the honest output of this study is mostly bounds, and it says so. What it is not is a
reason to skip the work: the series now exists, verified, and every future week extends it.
The CFTC publishes the complete disaggregated history back to 2009 — roughly 880 weeks —
and archiving it is the single change that would make these questions answerable.

Usage:
    python3.12 -m scripts.research.build_cftc_positioning
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "RS-XAUUSD-20260824-002"
OUTPUT_DIR = Path("reproduced")
SERIES = Path("local-inputs/cftc_weekly_series.csv")TAIPEI = timezone(timedelta(hours=8))

BOOTSTRAP = 4000
BLOCK = 4          # weeks; positioning is persistent so weekly rows are not independent
SEED = 20260824
MIN_GROUP = 6

# The two windows the publication lag separates, plus the blended one for contrast.
WINDOWS = {
    "fwd_wed_fri_pct": "already history when the report was published",
    "fwd_mon_tue_pct": "the only window a reader could act on",
    "fwd_week_pct": "Tuesday to Tuesday — blends both, which is why it misleads",
}


def rng(label: str) -> random.Random:
    return random.Random(f"{SEED}:{label}")


def smallest_resolvable(a: int, b: int, sigma: float) -> float | None:
    if a < 2 or b < 2 or sigma <= 0:
        return None
    return round(2.8 * sigma * math.sqrt(1 / a + 1 / b), 3)


def compare(values: pd.Series, condition: np.ndarray, stream: random.Random) -> dict:
    """Group difference with a block bootstrap and the bound that makes it readable."""
    usable = values.notna().to_numpy()
    series = values.to_numpy()[usable]
    flag = condition[usable]
    hit, miss = series[flag], series[~flag]
    if hit.size < MIN_GROUP or miss.size < MIN_GROUP:
        return {"n_condition": int(hit.size), "n_other": int(miss.size),
                "verdict": "underpowered",
                "reason": f"group sizes {hit.size}/{miss.size}; need {MIN_GROUP} each"}
    effect = float(hit.mean() - miss.mean())
    sigma = float(series.std(ddof=1))
    bound = smallest_resolvable(hit.size, miss.size, sigma)

    count = series.size
    starts = max(count - BLOCK, 1)
    draws = []
    for _ in range(BOOTSTRAP):
        index = []
        while len(index) < count:
            begin = stream.randrange(starts)
            index.extend(range(begin, min(begin + BLOCK, count)))
        index = np.array(index[:count])
        sample_flag, sample_values = flag[index], series[index]
        if sample_flag.sum() < 3 or (~sample_flag).sum() < 3:
            continue
        draws.append(sample_values[sample_flag].mean() - sample_values[~sample_flag].mean())
    draws = np.array(draws)
    share = float((draws > 0).mean()) if draws.size else None
    p_two = round(2 * min(share, 1 - share), 4) if share is not None else None

    resolvable = bound is not None and abs(effect) > bound
    return {
        "n_condition": int(hit.size), "n_other": int(miss.size),
        "mean_condition_pct": round(float(hit.mean()), 3),
        "mean_other_pct": round(float(miss.mean()), 3),
        "effect_pct": round(effect, 3),
        "smallest_resolvable_pct": bound,
        "effect_over_bound": round(abs(effect) / bound, 2) if bound else None,
        "bootstrap_p_two_sided": p_two,
        "verdict": (
            "resolvable" if resolvable and (p_two is not None and p_two <= 0.05)
            else "no_evidence"
        ),
        "reason": (
            f"effect {effect:+.3f}% vs a {bound:+.3f}% resolution bound"
            if not resolvable else f"clears the bound; bootstrap p={p_two}"
        ),
    }


def hypothesis(frame: pd.DataFrame, name: str, claim: str, origin: str,
               condition: np.ndarray, note: str | None = None) -> dict:
    """One condition evaluated across all three windows, so the placebo is always visible."""
    result = {
        "id": name, "claim": claim, "origin": origin,
        "condition_weeks": int(condition.sum()), "total_weeks": int(len(frame)),
        "windows": {},
    }
    if note:
        result["note"] = note
    for column, meaning in WINDOWS.items():
        outcome = compare(frame[column], condition, rng(f"{name}:{column}"))
        outcome["window_meaning"] = meaning
        result["windows"][column] = outcome

    acted = result["windows"]["fwd_mon_tue_pct"]
    history = result["windows"]["fwd_wed_fri_pct"]
    if acted.get("effect_pct") is not None and history.get("effect_pct") is not None:
        # The placebo reading. A larger effect in the already-happened window than in the
        # actionable one is the signature of a coincident relationship being mistaken for a
        # predictive one.
        result["placebo"] = {
            "already_history_effect_pct": history["effect_pct"],
            "actionable_effect_pct": acted["effect_pct"],
            "history_larger_in_magnitude": abs(history["effect_pct"]) > abs(acted["effect_pct"]),
            "reading": (
                "the relationship is mostly with the move the positions were built during, "
                "not with what came after"
                if abs(history["effect_pct"]) > abs(acted["effect_pct"])
                else "the actionable window carries at least as much as the historical one"
            ),
        }
    return result


def build_hypotheses(d: pd.DataFrame) -> list[dict]:
    """Owner hypotheses first, then the ones this study adds."""
    out = []

    def quantile_flag(column: str, q: float, above: bool = True) -> np.ndarray:
        threshold = d[column].quantile(q)
        return (d[column] > threshold if above else d[column] < threshold).to_numpy()

    # --- owner hypothesis 1 ---------------------------------------------------------
    # Stated as "retail long while managed money is short, then a rise followed by a fall".
    # The literal binary version fires twice in 31 weeks, so it is reported as underpowered
    # AND reformulated continuously: the divergence between retail and managed-money net
    # positioning, which is the same idea on a scale the sample can actually see.
    combo = ((d["retail_net_chg"] > 0) & (d["mm_net_chg"] < 0)).to_numpy()
    out.append(hypothesis(
        d, "h01_retail_long_mm_short_literal",
        "Retail adds length while managed money cuts it — the owner's stated combination.",
        "owner", combo,
        note="Fires twice in 31 weeks. Kept to record that the literal form is untestable "
             "here, not to claim a result; h02 is the same idea made continuous.",
    ))
    out.append(hypothesis(
        d, "h02_retail_vs_mm_divergence",
        "Top-quartile divergence between retail and managed-money net positioning "
        "(retail crowded long relative to the funds) precedes weakness.",
        "owner_reformulated", quantile_flag("retail_vs_mm_divergence", 0.75),
    ))

    # --- owner hypothesis 2 ---------------------------------------------------------
    out.append(hypothesis(
        d, "h03_producer_short_falls",
        "Producer/merchant shorts fall — less downside hedging — so the next week rises.",
        "owner", (d["prod_merc_short_chg"] < 0).to_numpy(),
    ))
    out.append(hypothesis(
        d, "h04_producer_short_falls_hard",
        "A top-quartile *reduction* in producer/merchant shorts precedes a rise.",
        "owner_sharpened", quantile_flag("prod_merc_short_chg", 0.25, above=False),
    ))

    # --- what a hedger's book actually is -------------------------------------------
    # Producers short to hedge production. If their book is a mechanical response to price
    # rather than a view on it, its correlation with the concurrent move should dominate
    # its correlation with the following one — which is what the placebo columns show.
    out.append(hypothesis(
        d, "h05_producer_short_rises",
        "Producer/merchant shorts rise — more hedging — so the next week falls.",
        "study", (d["prod_merc_short_chg"] > 0).to_numpy(),
    ))

    # --- retail as the classic contrarian indicator ----------------------------------
    out.append(hypothesis(
        d, "h06_retail_crowded_long",
        "Retail net length in its top quartile precedes weakness.",
        "external_claim", quantile_flag("retail_net_pct_oi", 0.75),
    ))
    out.append(hypothesis(
        d, "h07_retail_capitulates",
        "A bottom-quartile week for retail net positioning precedes strength.",
        "external_claim", quantile_flag("retail_net_pct_oi", 0.25, above=False),
    ))

    # --- managed money crowding ------------------------------------------------------
    out.append(hypothesis(
        d, "h08_mm_crowded_long",
        "Managed-money long/short ratio in its top quartile precedes weakness.",
        "external_claim", quantile_flag("mm_long_short_ratio", 0.75),
    ))
    out.append(hypothesis(
        d, "h09_mm_builds_length",
        "A top-quartile weekly build in managed-money net length precedes weakness.",
        "external_claim", quantile_flag("mm_net_chg", 0.75),
    ))

    # --- open interest and price together --------------------------------------------
    # The textbook reading: open interest rising with price is a healthy trend, rising
    # against price is distribution. It needs both series, which is why it is rarely tested.
    oi_up = d["total_changes"] > 0
    price_up = d["fwd_wed_fri_pct"].notna() & (d["close_at_report"].diff() > 0)
    out.append(hypothesis(
        d, "h10_oi_up_price_up",
        "Open interest and price rising together — new money confirming the trend.",
        "external_claim", (oi_up & price_up).to_numpy(),
    ))
    out.append(hypothesis(
        d, "h11_oi_up_price_down",
        "Open interest rising while price falls — new shorts, or distribution.",
        "external_claim", (oi_up & ~price_up).to_numpy(),
    ))

    # --- concentration ---------------------------------------------------------------
    # An unusual field the screenshots carry and the fetched CSV does not: how many traders
    # hold the positions. The same net length spread over fewer books is a different risk.
    d["mm_long_per_trader"] = d["mm_long"] / d["mm_long_traders"]
    out.append(hypothesis(
        d, "h12_mm_concentration",
        "Managed-money length concentrated in fewer books precedes a larger move.",
        "study", quantile_flag("mm_long_per_trader", 0.75),
        note="Uses the trader counts only the screenshots carry; the fetched CSV omits them.",
    ))

    # --- everyone on the same side ----------------------------------------------------
    agree = ((np.sign(d["retail_net_chg"]) == np.sign(d["mm_net_chg"]))
             & (np.sign(d["mm_net_chg"]) == np.sign(d["other_net_chg"]))
             & (d["mm_net_chg"] != 0))
    out.append(hypothesis(
        d, "h13_all_speculators_agree",
        "Retail, managed money and other reportables all move the same way in one week.",
        "study", agree.fillna(False).to_numpy(),
    ))

    # --- the dealer on the other side --------------------------------------------------
    out.append(hypothesis(
        d, "h14_swap_dealers_add_shorts",
        "Swap dealers adding shorts — the mirror of speculative length — precedes weakness.",
        "study", quantile_flag("swap_short_chg", 0.75),
    ))

    # --- level versus change ------------------------------------------------------------
    # Most COT commentary reads the level. If only the change matters, the level is noise
    # dressed as context, and the pair of hypotheses below is what separates them.
    out.append(hypothesis(
        d, "h15_mm_net_level_high",
        "The LEVEL of managed-money net length, top quartile.",
        "study", quantile_flag("mm_net_pct_oi", 0.75),
        note="Paired with h09, which tests the CHANGE. Level and change disagreeing is the "
             "useful outcome: it says which of the two the commentary should be reading.",
    ))

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=SERIES)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    d = pd.read_csv(args.series)
    d["report_date"] = pd.to_datetime(d["report_date"])
    results = build_hypotheses(d)

    # How often does the already-happened window carry more than the actionable one?
    placebos = [r["placebo"] for r in results if "placebo" in r]
    history_wins = sum(1 for p in placebos if p["history_larger_in_magnitude"])

    # The sharper version of the placebo reading. Counting how often one window beats the
    # other is weak — a coin flip gives 7 of 14. The question worth asking is whether the
    # two windows AGREE: if positioning genuinely carried information forward, a condition
    # associated with a rise in the days before publication should be associated with a
    # rise after it too, and the two effects would correlate positively across hypotheses.
    # A correlation near zero or negative says the apparent relationships do not survive
    # the moment of publication.
    pairs = [
        (p["already_history_effect_pct"], p["actionable_effect_pct"]) for p in placebos
    ]
    correlation = None
    same_sign = None
    if len(pairs) >= 5:
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((a - mx) * (b - my) for a, b in pairs)
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
        correlation = round(num / den, 3) if den else None
        same_sign = sum(1 for a, b in pairs if (a > 0) == (b > 0))

    # And the strongest apparent relationships specifically: do the biggest history-window
    # effects carry into the actionable window at all?
    ranked = sorted(
        ((r["id"], r["placebo"]) for r in results if "placebo" in r),
        key=lambda item: -abs(item[1]["already_history_effect_pct"]),
    )[:5]
    strongest = [
        {
            "id": name,
            "already_history_effect_pct": pl["already_history_effect_pct"],
            "actionable_effect_pct": pl["actionable_effect_pct"],
            "sign_flips_after_publication": (
                (pl["already_history_effect_pct"] > 0) != (pl["actionable_effect_pct"] > 0)
            ),
        }
        for name, pl in ranked
    ]

    verdicts: dict[str, int] = {}
    for r in results:
        v = r["windows"]["fwd_mon_tue_pct"]["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    payload = {
        "study_id": STUDY_ID,
        "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "market": "XAUUSD",
        "strategy": "none — CFTC positioning structure",
        "method": {
            "series": str(args.series.relative_to(ROOT)),
            "weeks": int(len(d)),
            "from": str(d["report_date"].min().date()),
            "to": str(d["report_date"].max().date()),
            "publication_lag": (
                "A report dated Tuesday is published Friday afternoon US time. Wednesday "
                "through Friday have already traded when it lands, which is what makes "
                "fwd_wed_fri a placebo window rather than a forecast."
            ),
            "bootstrap": BOOTSTRAP, "block_weeks": BLOCK, "seed": SEED,
            "min_group": MIN_GROUP,
        },
        "power": {
            "weeks": int(len(d)),
            "weekly_return_sd_pct": round(float(d["fwd_week_pct"].std(ddof=1)), 3),
            "typical_smallest_resolvable_pct": round(
                2.8 * float(d["fwd_week_pct"].std(ddof=1)) * math.sqrt(2 / (len(d) / 2)), 3
            ),
            "reading": (
                "With 31 weeks split in two, only differences of roughly this size per week "
                "can be separated from noise. Most real positioning effects are far smaller, "
                "so the expected outcome of this study is bounds rather than findings."
            ),
            "unlock": (
                "The CFTC publishes the complete disaggregated futures-only history back to "
                "September 2009 as annual archives — roughly 880 weeks against the 31 here. "
                "Archiving it is a data task and it is the single change that would make "
                "these questions answerable."
            ),
        },
        "placebo_summary": {
            "hypotheses_with_both_windows": len(placebos),
            "already_history_window_larger": history_wins,
            "correlation_between_windows": correlation,
            "same_sign_in_both_windows": same_sign,
            "strongest_history_effects": strongest,
            "reading": (
                "If positioning carried information across publication, a condition linked "
                "to a rise before the report landed would be linked to a rise after it, and "
                "the two windows would correlate positively across hypotheses. The counting "
                "measure is weak — a coin flip gives half — so the correlation and the "
                "sign-flip column on the strongest effects are the ones to read."
            ),
        },
        "actionable_window_verdicts": verdicts,
        "hypotheses": results,
        "limitations": [
            "Thirty-one weeks. Almost every result here is a bound, not a finding, and the "
            "bounds are wide.",
            "Transcribed from the owner's screenshots; all 132 field-values across the 12 "
            "weeks that overlap the official CSV matched exactly, and the series build fails "
            "if that ever stops being true.",
            "Two weeks are missing from the screenshot archive (2026-04-14 and 2026-05-05), "
            "so week-over-week changes spanning those gaps are computed across a two-week "
            "interval rather than one.",
            "Returns use the 07:00-Taipei session close, so a 'Tuesday close' is the close "
            "of the session that opened Tuesday morning Taipei time.",
            "No result changes formal S1 or S2 logic, live risk, or an entry checklist.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "study_id": STUDY_ID,
        "weeks": len(d),
        "hypotheses": len(results),
        "actionable_verdicts": verdicts,
        "placebo_history_larger": f"{history_wins}/{len(placebos)}",
        "typical_bound_pct": payload["power"]["typical_smallest_resolvable_pct"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
