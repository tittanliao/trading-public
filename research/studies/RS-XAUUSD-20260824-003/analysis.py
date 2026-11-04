#!/usr/bin/env python3
"""RS-XAUUSD-20260824-003 — a stated selection rule, tested on 274 weeks of real data.

Three specific claims and one criticism were raised, and all four deserved a direct
answer rather than an argument about statistics:

1. Monday after the COT release breaks the low, sweeps liquidity, then rises.
2. A condition with a win rate above 50% is alpha.
3. Several conditions agreeing is alpha.
4. "I dislike you constantly saying not significant and then inventing a story."

Point 4 is fair and it shapes how this file works. Every claim below is tested with its
own stated criterion first, on its own data, and the result is shown before any
interpretation. Where the stated method and a conventional one disagree, both numbers are
printed side by side so the disagreement is visible rather than asserted.

The sample is no longer the constraint. The CFTC disaggregated history was downloaded with
permission — 503 weeks for gold, 2017 to 2026 — and joined to the daily series
gives **274 usable weeks**, up from the 31 that made the previous study unanswerable.

This is a post-hoc exploration, not a pre-registered test: the windows, the half split, the
six votes and the five-vote cutoff were chosen by the researcher.

## What the three tests found (figures are in results.json; this is a summary)

**The Monday sweep gap is a lookahead.** Weeks where Monday broke Friday's low finished
Mon+Tue positive 34.13% of the time against 64.19% otherwise, a -30.06 point gap. But
"Monday broke Friday's low" is not knowable until Monday is over, and the return being
measured *includes Monday*. Conditioning on Monday's close and measuring Tuesday alone (242
weeks, after dropping 32 weeks that have no separate Monday and Tuesday bar), the gap is
+1.80 points.

**">50% is alpha" is measured against a moving baseline.** The Mon+Tue baseline win rate is
45.99% in the first half of the sample and 54.74% in the second (137 weeks each). Of 118
distinct conditions (182 before removing conditions that select identical weeks), one
cleared 50% in the first half and 80.5% of them did in the second; the win rates of the
two halves correlate at -0.143.

**Consensus: higher in-sample win rate, no visible gain in return.** Five or more of six
positioning votes agreeing gives a 62.5% win rate (n=32) against 48.76%, a 13.74 point gap
with permutation p 0.1883. The mean weekly return gap is -0.1298 points (p 0.6497). The
walk-forward reconstruction (medians from prior weeks only, n=19) gives +13.83 points
(p 0.3393) and a return gap of +0.1570 (p 0.678). None of these gaps is separable from
chance at 0.05.

Inputs: see reproduction_inputs.json (the daily series is replaced weekly in the source repository).

Usage:
    python3.12 -m scripts.research.build_cftc_winrate_test
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import fail_pattern_toolkit as tk  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260824-003"
OUTPUT_DIR = Path("reproduced")
CFTC = Path("local-inputs/cftc_gold_disagg_2017_2026.csv")
DAILY_LOCATOR = "FX_IDC_XAUUSD, 1D.csv"
DAILY = Path("local-inputs/FX_IDC_XAUUSD, 1D.csv")
TAIPEI = timezone(timedelta(hours=8))
PERMUTATIONS = 20000
WARMUP = 30
SEED = 20260824
TIE_TOLERANCE = 1e-9   # permuted and observed gaps that are equal in exact arithmetic count as ties


def build_frame(cftc_path: Path = CFTC, daily_path: Path = DAILY) -> pd.DataFrame:
    """One row per COT report, joined to the sessions around its publication."""
    c = pd.read_csv(cftc_path)
    c["report_date"] = pd.to_datetime(c["report_date"])
    d = pd.read_csv(daily_path)
    d["time"] = pd.to_datetime(d["time"])
    d = d.sort_values("time").reset_index(drop=True)

    rows = []
    for _, r in c.iterrows():
        tuesday = r["report_date"]
        friday = tuesday + pd.Timedelta(days=3)     # publication
        monday = tuesday + pd.Timedelta(days=6)     # first session a reader could act in
        next_tuesday = tuesday + pd.Timedelta(days=7)
        before = d[d["time"] <= friday]
        after = d[d["time"] >= monday]
        close_next = d[d["time"] <= next_tuesday]
        if before.empty or after.empty or close_next.empty:
            continue
        b_fri, b_mon, b_tue = before.iloc[-1], after.iloc[0], close_next.iloc[-1]
        if b_mon["time"] <= b_fri["time"] or b_tue["time"] < b_mon["time"]:
            continue
        rows.append({
            "report_date": tuesday,
            "fri_low": b_fri["low"], "fri_close": b_fri["close"],
            "mon_low": b_mon["low"], "mon_close": b_mon["close"],
            "tue_close": b_tue["close"],
            # True when the "Monday" bar and the "Tuesday" bar are the same bar (no daily bar
            # on Monday or on Tuesday, e.g. a US holiday). Tuesday-only return is then 0 by
            # construction, not a measured flat day.
            "same_bar": bool(b_mon["time"] == b_tue["time"]),
            "mm_net": r["mm_long"] - r["mm_short"],
            "mm_net_chg": r.get("mm_long_chg", np.nan) - r.get("mm_short_chg", np.nan),
            "retail_net": r["retail_long"] - r["retail_short"],
            "retail_net_chg": r.get("retail_long_chg", np.nan) - r.get("retail_short_chg", np.nan),
            "pm_short_chg": r.get("prod_merc_short_chg", np.nan),
            "oi": r["open_interest"],
        })
    f = pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)
    # The window that matters for action: the two sessions after publication.
    f["mon_tue_ret"] = 100 * (f["tue_close"] / f["fri_close"] - 1)
    # The same window with Monday removed, which is what a Monday-close condition can use.
    f["tue_only_ret"] = 100 * (f["tue_close"] / f["mon_close"] - 1)
    f["mon_broke_fri_low"] = f["mon_low"] < f["fri_low"]
    f["mon_closed_up"] = f["mon_close"] > f["fri_close"]
    return f


def win_rate(series: pd.Series) -> float:
    return round(100 * float((series > 0).mean()), 2)


def block(series: pd.Series) -> dict:
    wins, losses = series[series > 0], series[series <= 0]
    return {
        "n": int(len(series)),
        "win_rate_pct": win_rate(series),
        "mean_pct": round(float(series.mean()), 4),
        "avg_win_pct": round(float(wins.mean()), 4) if len(wins) else None,
        "avg_loss_pct": round(float(losses.mean()), 4) if len(losses) else None,
        "payoff_ratio": (
            round(abs(float(wins.mean()) / float(losses.mean())), 3)
            if len(wins) and len(losses) and losses.mean() != 0 else None
        ),
        "total_pct": round(float(series.sum()), 2),
    }


def test_monday_sweep(f: pd.DataFrame) -> dict:
    """Claim 1, in both the circular form and the clean one."""
    swept, quiet = f[f["mon_broke_fri_low"]], f[~f["mon_broke_fri_low"]]
    recovered = f[f["mon_broke_fri_low"] & f["mon_closed_up"]]
    kept_falling = f[f["mon_broke_fri_low"] & ~f["mon_closed_up"]]

    # Weeks where the Monday and Tuesday bars are the same bar have a Tuesday-only return of
    # 0 by construction, which win_rate() counts as a loss. They are dropped from every
    # Tuesday-only comparison (clean version, sweep-and-recover); the circular version
    # measures Friday close to Tuesday close, which stays a valid return, so it keeps them.
    valid = f[~f["same_bar"]]
    swept_v, quiet_v = swept[~swept["same_bar"]], quiet[~quiet["same_bar"]]
    recovered_v = recovered[~recovered["same_bar"]]
    kept_falling_v = kept_falling[~kept_falling["same_bar"]]
    clean_gap = round(
        win_rate(swept_v["tue_only_ret"]) - win_rate(quiet_v["tue_only_ret"]), 2)
    circular_gap = round(
        win_rate(swept["mon_tue_ret"]) - win_rate(quiet["mon_tue_ret"]), 2)
    return {
        "claim": "Monday after the release breaks the low, sweeps liquidity, then rises.",
        "circular_version": {
            "note": (
                "Conditions on whether Monday broke Friday's low, then measures a return "
                "that includes Monday. The break is not knowable until Monday is over, so "
                "this asks whether a down Monday was down."
            ),
            "broke": block(swept["mon_tue_ret"]),
            "did_not_break": block(quiet["mon_tue_ret"]),
            "win_rate_gap_pct": circular_gap,
        },
        "clean_version": {
            "note": (
                "Same condition, evaluated at Monday's close, measuring Tuesday alone. "
                "Condition strictly precedes the window it predicts. Weeks without a "
                "separate Monday bar and Tuesday bar (same_bar) are excluded: their "
                "Tuesday-only return is 0 by construction."
            ),
            "weeks_excluded_same_bar": int(f["same_bar"].sum()),
            "weeks_used": int(len(valid)),
            "broke": block(swept_v["tue_only_ret"]),
            "did_not_break": block(quiet_v["tue_only_ret"]),
            "win_rate_gap_pct": clean_gap,
            "including_same_bar_weeks": {
                "note": "Earlier computation, kept for comparison: same-bar weeks counted "
                        "as 0% (a loss).",
                "broke": block(swept["tue_only_ret"]),
                "did_not_break": block(quiet["tue_only_ret"]),
                "win_rate_gap_pct": round(
                    win_rate(swept["tue_only_ret"]) - win_rate(quiet["tue_only_ret"]), 2),
            },
        },
        "sweep_and_recover": {
            "note": "The claimed shape: broke the low AND closed up — a real sweep. "
                    "Same-bar weeks excluded, as in the clean version.",
            "swept_and_recovered": block(recovered_v["tue_only_ret"]),
            "swept_and_kept_falling": block(kept_falling_v["tue_only_ret"]),
            "including_same_bar_weeks": {
                "swept_and_recovered": block(recovered["tue_only_ret"]),
                "swept_and_kept_falling": block(kept_falling["tue_only_ret"]),
            },
        },
        "reading": (
            f"The circular version shows a win-rate gap of {circular_gap:+.2f} points between "
            "weeks that broke Friday's low and weeks that did not. That gap comes from "
            "measuring a window that contains the condition. Evaluated at Monday's close "
            f"and measuring Tuesday alone, the gap is {clean_gap:+.2f} points."
        ),
    }


def make_conditions(df: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Single and paired conditions, all knowable at publication."""
    single = {}
    for column in ["mm_net", "mm_net_chg", "retail_net", "retail_net_chg", "pm_short_chg", "oi"]:
        series, ref = df[column], reference[column]
        if series.isna().all():
            continue
        single[f"{column}_up"] = series > 0
        single[f"{column}_q75"] = series > ref.quantile(0.75)
        single[f"{column}_q25"] = series < ref.quantile(0.25)
        single[f"{column}_above_med"] = series > ref.median()
    combined = dict(single)
    for a, b in itertools.combinations(list(single), 2):
        combined[f"{a}+{b}"] = single[a] & single[b]
    return combined


def _condition_summary(r: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    selected = r[r["train_win_rate"] > 50]
    return {
        "conditions_compared": int(len(r)),
        "first_half": {
            "from": str(train["report_date"].min().date()),
            "to": str(train["report_date"].max().date()),
            "conditions_over_50": int((r["train_win_rate"] > 50).sum()),
            "conditions_over_55": int((r["train_win_rate"] > 55).sum()),
        },
        "second_half": {
            "from": str(test["report_date"].min().date()),
            "to": str(test["report_date"].max().date()),
            "conditions_over_50": int((r["test_win_rate"] > 50).sum()),
            "share_over_50_pct": round(100 * float((r["test_win_rate"] > 50).mean()), 1),
        },
        "selected_in_first_half": int(len(selected)),
        "still_over_50_in_second": int((selected["test_win_rate"] > 50).sum()) if len(selected) else 0,
        "correlation_train_vs_test_win_rate": round(
            float(r["train_win_rate"].corr(r["test_win_rate"])), 3),
    }


def test_fifty_percent_rule(f: pd.DataFrame) -> dict:
    """Claim 2: a win rate above 50% is alpha. Tested by splitting the sample in half."""
    half = len(f) // 2
    train, test = f.iloc[:half], f.iloc[half:]
    ctrain, ctest = make_conditions(train, train), make_conditions(test, test)
    rows = []
    for name in ctrain:
        if name not in ctest:
            continue
        a, b = ctrain[name].fillna(False), ctest[name].fillna(False)
        if a.sum() < 15 or b.sum() < 15:
            continue
        rows.append({
            "condition": name,
            "train_win_rate": win_rate(train.loc[a, "mon_tue_ret"]),
            "test_win_rate": win_rate(test.loc[b, "mon_tue_ret"]),
            # The weeks each condition selects, in both halves: conditions that select the
            # same weeks in both halves are one condition, not several.
            "weeks_key": (a.to_numpy().tobytes(), b.to_numpy().tobytes()),
        })
    r_raw = pd.DataFrame(rows)
    r = r_raw.drop_duplicates(subset="weeks_key", keep="first").reset_index(drop=True)
    base_train = win_rate(train["mon_tue_ret"])
    base_test = win_rate(test["mon_tue_ret"])
    summary = _condition_summary(r, train, test)
    raw_summary = _condition_summary(r_raw, train, test)
    return {
        "claim": "A condition with a win rate above 50% is alpha.",
        "note": (
            "Conditions that select exactly the same weeks in both halves are counted once. "
            "The quantile and median cutoffs inside each condition are computed within each "
            "half (the second half uses its own cutoffs, not the first half's), so the "
            "second half is not a strict out-of-sample test of first-half cutoffs."
        ),
        "conditions_compared_before_deduplication": raw_summary["conditions_compared"],
        "baseline_win_rate_first_half": base_train,
        "baseline_win_rate_second_half": base_test,
        "baseline_shift_pct": round(base_test - base_train, 2),
        **summary,
        "before_deduplication": raw_summary,
        "reading": (
            f"The baseline moved {round(base_test - base_train, 2)} points between halves, so "
            "the same 50% threshold means 'beats the market' in one period and 'loses to it' "
            "in the other. A condition's win rate ranks it against the period, not against "
            "chance, and the period-to-period correlation of that rank is "
            f"{summary['correlation_train_vs_test_win_rate']}."
        ),
    }


def vote_count(df: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Six independent bullish positioning votes, all knowable at publication."""
    v = pd.DataFrame(index=df.index)
    v["mm_net_high"] = df["mm_net"] > reference["mm_net"].median()
    v["mm_adding"] = df["mm_net_chg"] > 0
    v["retail_low"] = df["retail_net"] < reference["retail_net"].median()
    v["retail_cutting"] = df["retail_net_chg"] < 0
    v["producer_short_cut"] = df["pm_short_chg"] < 0
    v["oi_high"] = df["oi"] > reference["oi"].median()
    return v.fillna(False).sum(axis=1)


def test_consensus(f: pd.DataFrame, stream: random.Random) -> dict:
    """Claim 3: several conditions agreeing is alpha."""
    f = f.copy()
    f["votes"] = vote_count(f, f)

    ladder = {}
    for k in range(0, 7):
        mask = f["votes"] == k
        if mask.sum() >= 8:
            ladder[str(k)] = block(f.loc[mask, "mon_tue_ret"])

    mask = (f["votes"] >= 5).to_numpy()
    hit, miss = f.loc[mask, "mon_tue_ret"], f.loc[~mask, "mon_tue_ret"]
    # Reported gap: difference of the two rounded win rates, as before. The permutation
    # test below compares against the UNROUNDED observed gap: comparing permuted gaps (which
    # are not rounded) with a gap rounded up by up to 0.005 drops the permutations that tie
    # with the observed one, which is the largest single mass, and understates p.
    observed = win_rate(hit) - win_rate(miss)
    values = f["mon_tue_ret"].to_numpy().copy()
    observed_exact = 100 * float((values[mask] > 0).mean()) - 100 * float((values[~mask] > 0).mean())
    observed_mean = float(hit.mean() - miss.mean())
    exceed = exceed_mean = 0
    for _ in range(PERMUTATIONS):
        stream.shuffle(values)
        gap = (100 * (values[mask] > 0).mean()) - (100 * (values[~mask] > 0).mean())
        if abs(gap) >= abs(observed_exact) - TIE_TOLERANCE:
            exceed += 1
        if abs(values[mask].mean() - values[~mask].mean()) >= abs(observed_mean):
            exceed_mean += 1

    # Walk-forward: every median is computed from prior weeks only.
    forward = []
    for i in range(len(f)):
        past = f.iloc[:i]
        if len(past) < WARMUP:
            forward.append(np.nan)
            continue
        row = f.iloc[i]
        # sum() over a list, deliberately: numpy booleans overload `+` as logical OR, so
        # chaining them with + silently yields 0 or 1 instead of a count. That bug produced
        # a walk-forward vote distribution of {0: 1, 1: 243} and an n=0 result that was
        # obviously wrong — the dangerous version is the one that lands on a plausible
        # number instead.
        forward.append(sum([
            bool(row["mm_net"] > past["mm_net"].median()),
            bool(row["mm_net_chg"] > 0),
            bool(row["retail_net"] < past["retail_net"].median()),
            bool(row["retail_net_chg"] < 0),
            bool(row["pm_short_chg"] < 0),
            bool(row["oi"] > past["oi"].median()),
        ]))
    f["votes_walk_forward"] = forward
    g = f.dropna(subset=["votes_walk_forward"])
    wf_mask = g["votes_walk_forward"] >= 5

    # The walk-forward version selects different weeks than the full-sample one and looks
    # better on both measures, so it gets its own permutation test rather than borrowing
    # the full-sample p. n=19 is thin and the test is what says how thin.
    wf_values = g["mon_tue_ret"].to_numpy().copy()
    wf_flag = wf_mask.to_numpy()
    wf_observed_wr = win_rate(g.loc[wf_mask, "mon_tue_ret"]) - win_rate(g.loc[~wf_mask, "mon_tue_ret"])
    wf_observed_wr_exact = (100 * float((wf_values[wf_flag] > 0).mean())
                            - 100 * float((wf_values[~wf_flag] > 0).mean()))
    wf_observed_mean = float(g.loc[wf_mask, "mon_tue_ret"].mean() - g.loc[~wf_mask, "mon_tue_ret"].mean())
    wf_exceed_wr = wf_exceed_mean = 0
    for _ in range(PERMUTATIONS):
        stream.shuffle(wf_values)
        gap_wr = (100 * (wf_values[wf_flag] > 0).mean()) - (100 * (wf_values[~wf_flag] > 0).mean())
        gap_mean = wf_values[wf_flag].mean() - wf_values[~wf_flag].mean()
        if abs(gap_wr) >= abs(wf_observed_wr_exact) - TIE_TOLERANCE:
            wf_exceed_wr += 1
        if abs(gap_mean) >= abs(wf_observed_mean):
            wf_exceed_mean += 1

    p_wr = round((exceed + 1) / (PERMUTATIONS + 1), 4)
    p_mean = round((exceed_mean + 1) / (PERMUTATIONS + 1), 4)
    p_wf_wr = round((wf_exceed_wr + 1) / (PERMUTATIONS + 1), 4)
    p_wf_mean = round((wf_exceed_mean + 1) / (PERMUTATIONS + 1), 4)

    def verdict(p: float) -> str:
        return "separable from chance at 0.05" if p < 0.05 else "not separable from chance at 0.05"

    return {
        "claim": "Several conditions agreeing at once is alpha.",
        "votes_definition": [
            "managed-money net above its median", "managed money adding net length",
            "retail net below its median", "retail cutting net length",
            "producer/merchant shorts falling", "open interest above its median",
        ],
        "by_vote_count": ladder,
        "vote_count_vs_return_correlation": round(
            float(np.corrcoef(f["votes"], f["mon_tue_ret"])[0, 1]), 3),
        "five_or_more": {
            "condition": block(hit),
            "other": block(miss),
            "win_rate_gap_pct": round(observed, 2),
            "mean_return_gap_pct": round(float(hit.mean() - miss.mean()), 4),
            "permutation_p": p_wr,
            "permutation_p_mean_return": p_mean,
            "permutations": PERMUTATIONS,
        },
        "five_or_more_walk_forward": {
            "note": "Medians from prior weeks only; no lookahead of any kind.",
            "warmup_weeks": WARMUP,
            "condition": block(g.loc[wf_mask, "mon_tue_ret"]),
            "other": block(g.loc[~wf_mask, "mon_tue_ret"]),
            "win_rate_gap_pct": round(wf_observed_wr, 2),
            "mean_return_gap_pct": round(wf_observed_mean, 4),
            "permutation_p_win_rate": p_wf_wr,
            "permutation_p_mean_return": p_wf_mean,
            "caveat": (
                "n=19. The walk-forward selection looks better than the full-sample one on "
                "both measures, which is the opposite of the usual direction and a reason "
                "to treat it as thin rather than as confirmation."
            ),
        },
        "opportunity_cost": {
            "note": "What trading only the consensus weeks would have captured.",
            "consensus_weeks": int(mask.sum()),
            "all_weeks": int(len(f)),
            "consensus_total_return_pct": round(float(hit.sum()), 2),
            "all_weeks_total_return_pct": round(float(f["mon_tue_ret"].sum()), 2),
            "share_of_opportunities_pct": round(100 * float(mask.mean()), 1),
            "share_of_return_captured_pct": round(
                100 * float(hit.sum() / f["mon_tue_ret"].sum()), 1),
        },
        "reading": (
            f"In sample, five or more votes gives a win-rate gap of {round(observed, 2):+.2f} "
            f"points (permutation p {p_wr}, {verdict(p_wr)}) and a mean weekly return gap of "
            f"{round(observed_mean, 4):+.4f} points (permutation p {p_mean}, {verdict(p_mean)}). "
            f"Walk-forward (n={int(wf_mask.sum())}): win-rate gap {round(wf_observed_wr, 2):+.2f} "
            f"(p {p_wf_wr}, {verdict(p_wf_wr)}), mean return gap {round(wf_observed_mean, 4):+.4f} "
            f"(p {p_wf_mean}, {verdict(p_wf_mean)}). Win rate and expectancy are different "
            "quantities; the gaps are described here, and a gap marked not separable is not "
            "evidence of either a gain or a loss."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    cftc_path, daily_path = CFTC, DAILY

    f = build_frame(cftc_path, daily_path)
    stream = random.Random(SEED)
    sweep = test_monday_sweep(f)
    fifty = test_fifty_percent_rule(f)
    consensus = test_consensus(f, stream)

    payload = {
        "study_id": STUDY_ID,
        "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "market": "XAUUSD",
        "strategy": "none — selection methodology tested on CFTC positioning",
        "method": {
            "cftc_source": "CFTC disaggregated futures-only history, downloaded "
                           "2026-08-24; 503 gold weeks, 2017-01-03 to 2026-08-18",
            "price_source": DAILY_LOCATOR,
            "price_input_sha256": hashlib.sha256(daily_path.read_bytes()).hexdigest(),
            "usable_weeks": int(len(f)),
            "from": str(f["report_date"].min().date()),
            "to": str(f["report_date"].max().date()),
            "windows": {
                "mon_tue_ret": "Friday close to next Tuesday close — the post-publication window",
                "tue_only_ret": "Monday close to Tuesday close — what a Monday-close condition can use",
            },
            "permutations": PERMUTATIONS, "warmup_weeks": WARMUP, "seed": SEED,
        },
        "claims_under_test": {
            "claim_1_monday_sweep": sweep,
            "claim_2_fifty_percent_rule": fifty,
            "claim_3_consensus": consensus,
        },
        "limitations": [
            f"{len(f)} weeks joined to a daily series starting 2021-05. The CFTC history "
            "reaches 2017 and 503 weeks; the binding constraint is now price data, not "
            "positioning.",
            "Returns are measured on daily closes. The intraday shape of the sweep "
            "hypothesis — break the low, then recover within the session — needs 30-minute "
            "data, which covers only 137 of these weeks.",
            "The consensus votes were chosen by the researcher, not optimised, and no record "
            "shows the five-vote cutoff was fixed before the by-vote table was seen. A search "
            "over vote definitions would find better ones and would need its own correction.",
            f"{int(f['same_bar'].sum())} weeks have no separate Monday bar and Tuesday bar "
            "(for example a US Monday holiday). They are excluded from the Tuesday-only "
            "comparisons and kept in the Friday-to-Tuesday ones.",
            "The claim-2 cutoffs (quantiles and medians) are computed inside each half, so "
            "the second half is not a strict out-of-sample test of first-half cutoffs.",
            "No result changes formal S1 or S2 logic, live risk, or an entry checklist.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps({
        "study_id": STUDY_ID,
        "weeks": len(f),
        "monday_sweep_gap_circular": sweep["circular_version"]["win_rate_gap_pct"],
        "monday_sweep_gap_clean": sweep["clean_version"]["win_rate_gap_pct"],
        "baseline_shift_between_halves": fifty["baseline_shift_pct"],
        "win_rate_correlation_across_halves": fifty["correlation_train_vs_test_win_rate"],
        "consensus_win_rate_gap": consensus["five_or_more"]["win_rate_gap_pct"],
        "consensus_mean_return_gap": consensus["five_or_more"]["mean_return_gap_pct"],
        "consensus_share_of_return": consensus["opportunity_cost"]["share_of_return_captured_pct"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
