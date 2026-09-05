#!/usr/bin/env python3
"""The screen battery, extracted so two studies cannot disagree about what a verdict means.

`build_xauusd_hypothesis_sweep.py` carries its own inline copy of this logic. That copy is
deliberately left alone: its numbers are published, and refactoring a runner underneath a
published result makes the result unreproducible for the sake of tidiness. New studies use
this module; the old one keeps its frozen copy.

The one thing parameterised here that was hard-coded there is the bootstrap block length.
That runner worked on 30-minute bars where a block of 12 spans six hours. On daily bars the
same 12 spans two and a half weeks, and a block has to be long enough to carry the series'
autocorrelation but short enough to leave many distinct blocks — so it has to move with the
bar size rather than being inherited.

A verdict is `survives_screens` only after clearing, in order:

1. **resolution** — the effect is larger than the smallest difference these two samples
   could separate. Skipping this is what turns "no effect" into "no evidence of an effect"
   and back again without anyone noticing which was measured.
2. **bootstrap** — a moving-block resample, so serial correlation is not counted as
   independent evidence.
3. **sign consistency** — the same direction in all three chronological windows. A pooled
   effect that flips sign is a period being described, not a condition.
4. **cost** — the effect NET of a round trip must still clear the resolution bound. The
   softer version of this test (gross effect vs cost) passes edges that are dead on
   arrival.
"""

from __future__ import annotations

import math
import random

import numpy as np


DEFAULT_BOOTSTRAP = 2000
DEFAULT_COST_PCT = 0.02


def stream_for(seed: int, label: str) -> random.Random:
    """A separate deterministic stream per hypothesis, so adding one does not move another."""
    return random.Random(f"{seed}:{label}")


def smallest_resolvable(count: int, other: int, sigma: float) -> float | None:
    """The effect size these two samples could separate at roughly 80% power.

    This is the number that makes a null readable. "No effect found" in a sample that could
    only ever have resolved 3% is a statement about the sample, not about gold.
    """
    if count < 2 or other < 2 or sigma <= 0:
        return None
    return round(2.8 * sigma * math.sqrt(1 / count + 1 / other), 4)


def block_bootstrap_effect(
    condition: np.ndarray,
    forward: np.ndarray,
    block: int,
    stream: random.Random,
    draws: int = DEFAULT_BOOTSTRAP,
) -> tuple[float | None, float | None]:
    """Resample contiguous blocks, returning (share above zero, two-sided p)."""
    count = forward.size
    # 2026-09-05 independent audit: `block` was never validated. A caller passing 0 or a
    # negative value sent `range(begin, min(begin + block, count))` empty, so the
    # `while len(index) < count` loop spun forever on a silently wrong argument. Fail on the
    # bad input instead of hanging on it.
    if block <= 0:
        raise ValueError(f"block must be a positive number of observations, got {block!r}")
    # 2026-09-05 independent audit: for block < count, a start index is valid whenever
    # begin + block <= count, i.e. begin in [0, count-block] -- that is count-block+1
    # values, not count-block. The previous `max(count - block, 1)` fed randrange one
    # short, so the final valid start (the only block containing the series' last
    # observation) was never drawn and that observation never entered the bootstrap.
    starts = max(count - block + 1, 1)
    collected = []
    for _ in range(draws):
        index: list[int] = []
        while len(index) < count:
            begin = stream.randrange(starts)
            index.extend(range(begin, min(begin + block, count)))
        picked = np.array(index[:count])
        sample_condition, sample_forward = condition[picked], forward[picked]
        if sample_condition.sum() < 10 or (~sample_condition).sum() < 10:
            continue
        collected.append(
            sample_forward[sample_condition].mean() - sample_forward[~sample_condition].mean()
        )
    if not collected:
        return None, None
    array = np.array(collected)
    share = float((array > 0).mean())
    # 2026-09-05 independent audit: this was `2 * min(share, 1 - share)`, which returns
    # exactly 0.0 when every resample lands on one side. A bootstrap of n draws cannot
    # establish p = 0; the most it can say is "smaller than roughly 2/(n+1)". Reporting 0.0
    # invites a reader to treat a finite resample as certainty, and it is the value most
    # likely to be quoted. Adding one to each tail count is the standard finite-resample
    # correction (Davison & Hinkley); it bounds p below by 2/(n+1) and leaves every p in the
    # interesting range essentially unchanged -- at the default 2000 draws it shifts a p by
    # at most 0.001, which is inside this function's own rounding.
    drawn = array.size
    above = int((array > 0).sum())
    low = (above + 1) / (drawn + 1)
    high = (drawn - above + 1) / (drawn + 1)
    return share, round(2 * min(low, high, 0.5), 4)


def evaluate(
    *,
    name: str,
    family: str,
    claim: str,
    origin: str,
    null_description: str,
    condition: np.ndarray,
    forward: np.ndarray,
    stream: random.Random,
    block: int,
    universe: np.ndarray | None = None,
    unit: str = "% forward return",
    cost_pct: float = DEFAULT_COST_PCT,
    minimum_fires: int = 20,
) -> dict:
    """One hypothesis, one verdict, with the bound that makes the verdict readable."""
    # `universe` is the set of sessions on which this hypothesis is even defined. It matters
    # because the comparison group is otherwise wrong in a way that is invisible: a VIX
    # condition tested against "every other session" is compared against years in which VIX
    # was not in the data at all, so the baseline describes a different era than the
    # condition does.
    usable = ~np.isnan(forward) & ~np.isnan(condition.astype(float))
    if universe is not None:
        usable = usable & universe
    condition, forward = condition[usable], forward[usable]

    # Chronological thirds of the sessions this hypothesis actually spans, not of some
    # longer series it happens to sit inside. Splitting on the outer series is how a
    # condition whose data begins in 2021 ends up with zero observations in two of three
    # windows and is then rejected for "inconsistency" it had no opportunity to show.
    count = forward.size
    period = np.array(["train"] * count, dtype=object)
    period[count // 3: 2 * count // 3] = "valid"
    period[2 * count // 3:] = "holdout"

    hit, miss = forward[condition], forward[~condition]
    result: dict = {
        "id": name,
        "family": family,
        "claim": claim,
        "origin": origin,
        "null": null_description,
        "unit": unit,
        "n_condition": int(hit.size),
        "n_other": int(miss.size),
    }
    if hit.size < minimum_fires or miss.size < minimum_fires:
        result.update({
            "verdict": "underpowered",
            "reason": (
                f"condition fires {hit.size} times against {miss.size}; below the "
                f"{minimum_fires} needed to say anything"
            ),
        })
        return result

    effect = float(hit.mean() - miss.mean())
    sigma = float(forward.std(ddof=1))
    bound = smallest_resolvable(hit.size, miss.size, sigma)
    share_above, two_sided = block_bootstrap_effect(condition, forward, block, stream)

    by_period = {}
    for label in ("train", "valid", "holdout"):
        mask = period == label
        window_hit, window_miss = forward[mask & condition], forward[mask & ~condition]
        by_period[label] = {
            "n_condition": int(window_hit.size),
            "effect": round(float(window_hit.mean() - window_miss.mean()), 4)
            if window_hit.size >= 5 and window_miss.size >= 5 else None,
        }
    signs = [row["effect"] for row in by_period.values() if row["effect"] is not None]
    consistent = len(signs) == 3 and (all(s > 0 for s in signs) or all(s < 0 for s in signs))

    # Win rate is reported for every hypothesis because the owner reasons in win rates, and
    # a study that only reports mean effects cannot be checked against that intuition. It is
    # never used as a screen: a rate above 50% is compatible with losing money, which this
    # programme has now demonstrated three separate ways.
    win_rate = float((hit > 0).mean() * 100)
    baseline_win_rate = float((miss > 0).mean() * 100)

    if bound is None:
        # 2026-09-05 independent audit: a constant-forward series (sigma=0) makes
        # smallest_resolvable() return None, and the old unconditional f"{bound:+.4f}"
        # then crashed formatting None. A bound of None means this sample cannot resolve
        # anything at all, which is itself the reportable fact.
        verdict, reason = "no_evidence", (
            "no resolvable bound: the forward series has zero variance in this sample"
        )
    elif abs(effect) <= bound:
        verdict, reason = "no_evidence", (
            f"effect {effect:+.4f} is inside the {bound:+.4f} these samples can separate"
        )
    elif two_sided is not None and two_sided > 0.05:
        verdict, reason = "no_evidence", (
            f"effect clears the resolution bound but bootstrap p={two_sided}"
        )
    elif not consistent:
        verdict, reason = "no_evidence", (
            "significant pooled but the sign does not hold across all three periods"
        )
    elif unit.endswith("% forward return") and (abs(effect) - cost_pct) <= bound:
        verdict, reason = "below_cost", (
            f"gross effect {effect:+.4f}% clears its bound, but net of a {cost_pct}% round "
            f"trip it is {abs(effect) - cost_pct:+.4f}%, inside the {bound:+.4f}% these "
            "samples can separate"
        )
    else:
        verdict, reason = "survives_screens", (
            "clears resolution, bootstrap, out-of-sample sign consistency and cost"
        )

    result.update({
        "sessions_in_universe": int(count),
        "effect": round(effect, 4),
        "baseline_mean": round(float(miss.mean()), 4),
        "win_rate_pct": round(win_rate, 2),
        "baseline_win_rate_pct": round(baseline_win_rate, 2),
        "win_rate_edge_pct": round(win_rate - baseline_win_rate, 2),
        "smallest_resolvable_effect": bound,
        "bootstrap_p_two_sided": two_sided,
        "bootstrap_share_above_zero": round(share_above, 4) if share_above is not None else None,
        "effect_net_of_cost": round(abs(effect) - cost_pct, 4)
        if unit.endswith("% forward return") else None,
        "by_period": by_period,
        "sign_consistent_all_periods": consistent,
        "verdict": verdict,
        "reason": reason,
    })
    return result


def family_minimum_p_correction(
    results: list[dict], stream: random.Random, draws: int = 2000
) -> dict:
    """How often would this many apparent winners appear if none of them were real?

    Twenty hypotheses at p<0.05 produce one 'discovery' by construction. This asks whether
    the best result in the family beats what the family's own noise produces.

    2026-09-05 independent audit: this was named `family_permutation` and its docstring
    implied it permutes the observed family. It does not -- it draws `count` fresh
    independent Uniform(0,1) values per trial and compares their minimum to the observed
    minimum p-value. That is a Monte Carlo Sidak/min-p correction under the assumption the
    per-hypothesis p-values are independent and uniform under the null; it is not a
    resample of the actual data and cannot see dependence between related hypotheses (e.g.
    two conditions built from overlapping windows or overlapping trades). Renamed to
    describe what it computes. The three sweeps currently calling this (`-20260824-004`,
    `-20260824-005`, `-20260825-001`) have no surviving hypothesis, so this defect does not
    presently change any published verdict -- but it can understate the true family-wide
    false-positive rate for a future family with correlated hypotheses, and a genuine
    permutation (shuffling condition labels through the whole harness per draw) would be
    needed before trusting a `survives_screens` verdict that depends on this correction.
    A backward-compatible alias, `family_permutation`, is kept so existing callers do not
    break; new callers should use the accurate name.
    """
    scored = [r for r in results if r.get("bootstrap_p_two_sided") is not None]
    if not scored:
        return {"tested": 0, "family_p": None}
    observed = min(r["bootstrap_p_two_sided"] for r in scored)
    count = len(scored)
    at_least = 0
    for _ in range(draws):
        best = min(stream.random() for _ in range(count))
        if best <= observed:
            at_least += 1
    return {
        "tested": count,
        "best_p_in_family": observed,
        # Same finite-resample correction as block_bootstrap_effect, and for the same
        # reason: `at_least / draws` reports 0.0 when no draw beats the observed minimum,
        # which claims more than 2000 draws can support.
        "family_p": round((at_least + 1) / (draws + 1), 4),
        "method": "monte_carlo_sidak_min_p_independent",
        "reading": (
            "the chance that a family of this size produces a result this strong when "
            "every member is noise, ASSUMING the per-hypothesis p-values are independent "
            "-- this is not a permutation of the observed data"
        ),
    }


# Backward-compatible name. Prefer family_minimum_p_correction in new code; see its
# docstring for why the old name overstated what this computes.
family_permutation = family_minimum_p_correction
