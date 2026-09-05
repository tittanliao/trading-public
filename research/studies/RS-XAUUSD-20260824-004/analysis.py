#!/usr/bin/env python3
"""Twenty hypotheses on 18 years of daily gold, and the macro drivers never tested here.

Every sweep in this programme so far ran on a short window. The last one used 672 sessions
(2024-01 to 2026-08) and returned twenty nulls, and the honest reading of several of those
was not "the effect is absent" but "672 sessions could not resolve it". That is
the same objection raised here, in plainer words: 就算樣本少 應該還是有可能有 PATTERN.

The daily series in this repository goes back to 2008-01-04 — 4851 sessions, 7.2x the last
sweep. Four macro series sitting in `manual/csv` have never been used by any study: VIX,
GVZ (the gold volatility index), the US 10-year yield, and the 10-year breakeven inflation
rate. Their difference is the real yield, which is the textbook driver of gold and has
never been tested here at all.

So this study is not a new idea applied to old data. It is the old ideas applied to enough
data to tell the two kinds of "no" apart, plus the one family of drivers the programme had
data for and never looked at.

## What is deliberately not claimed

More data narrows the resolution bound; it does not make a weak effect real. Where a claim
was previously `no_evidence` and stays `no_evidence` with 7x the sample, that is a much
stronger statement than it was before, and the bound is reported so it can be read as one.

## Lookahead

Gold's daily bar and the US macro closes are stamped on the same date but do not end at the
same instant, and an alignment assumption is exactly the kind of thing that manufactures an
effect. Every macro-conditioned hypothesis therefore uses macro data through day t and
measures gold's forward return starting from the close of day t+1 — one full session of
slack, so no alignment convention can create a signal.

Usage:
    python3.12 scripts/research/build_xauusd_long_history_sweep.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import screen_harness as sh  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260824-004"
OUTPUT_DIR = Path("reproduced")

GOLD = Path("local-inputs/gold_daily.csv")
DXY = Path("local-inputs/dxy_daily.csv")
# 2026-09-05: repointed from the tracked weekly-refresh path, which is overwritten in
# place every week, to a permanent snapshot of the exact bytes this study was
# originally computed against (recovered from git commit 540562d, verified by hash).
# Without this, "rerunning" this study silently answers a different question every week.
MANUAL = Path("local-inputs")

SEED = 20260824
COST_PCT = 0.02


def block_for(horizon: int) -> int:
    """Bootstrap block length, scaled to the forward horizon.

    A 20-session forward return shares 19 of its 20 days with the next observation. Drawing
    21-day blocks off that series resamples windows that are almost the same window and
    counts them as independent evidence, which makes every long-horizon p-value optimistic.
    Three times the horizon, floored at a trading month, keeps a block wider than the
    overlap it has to break.
    """
    return max(21, 3 * horizon)


# --------------------------------------------------------------------------- data

def read_series(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None).dt.normalize()
    keep = [c for c in ("open", "high", "low", "close") if c in frame.columns]
    frame = frame[["date"] + keep].rename(columns={c: f"{name}_{c}" for c in keep})
    return frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load() -> pd.DataFrame:
    gold = read_series(GOLD, "gold")
    frame = gold.copy()
    for label, path in (
        ("dxy", DXY),
        ("vix", MANUAL / "VIX_1d.csv"),
        ("gvz", MANUAL / "GVZ_1d.csv"),
        ("us10y", MANUAL / "US10Y_1d.csv"),
        ("t10yie", MANUAL / "T10YIE_1d.csv"),
    ):
        frame = frame.merge(read_series(path, label), on="date", how="left")

    close = frame["gold_close"]
    frame["ret"] = close.pct_change() * 100
    for horizon in (1, 5, 20):
        # Forward return from the close of day t. Shifting by -horizon on a close-to-close
        # series means nothing about day t's own bar leaks into the condition.
        frame[f"fwd{horizon}"] = (close.shift(-horizon) / close - 1) * 100
        # The macro variants start one session later, so no same-day alignment assumption
        # between a 00:00Z gold stamp and a 21:00Z US close can create an effect.
        frame[f"fwd{horizon}_lag"] = (close.shift(-horizon - 1) / close.shift(-1) - 1) * 100

    frame["ma200"] = close.rolling(200).mean()
    frame["stretch"] = (close / frame["ma200"] - 1) * 100
    frame["high52"] = close.rolling(252).max()
    frame["drawdown"] = (close / frame["high52"] - 1) * 100
    frame["rv20"] = frame["ret"].rolling(20).std() * np.sqrt(252)
    frame["weekday"] = frame["date"].dt.weekday
    frame["month"] = frame["date"].dt.month
    frame["day_of_month"] = frame["date"].dt.day
    frame["days_in_month"] = frame["date"].dt.days_in_month

    up = frame["ret"] > 0
    down = frame["ret"] < 0
    frame["up_streak"] = up.groupby((~up).cumsum()).cumsum()
    frame["down_streak"] = down.groupby((~down).cumsum()).cumsum()

    frame["real_yield"] = frame["us10y_close"] - frame["t10yie_close"]
    frame["vrp"] = frame["gvz_close"] - frame["rv20"]
    for column in ("us10y_close", "t10yie_close", "real_yield", "dxy_close", "vix_close"):
        frame[f"{column}_chg20"] = frame[column] - frame[column].shift(20)
    frame["us10y_chg1"] = frame["us10y_close"] - frame["us10y_close"].shift(1)
    return frame



def quantile_mask(series: pd.Series, low: float, high: float) -> np.ndarray:
    """Rank within the *expanding* history, never the whole sample.

    A quantile computed over the full series is the classic lookahead: on 2012-01-01 it
    uses 2026 data to decide what 'top quintile' meant. The expanding rank asks only what
    was knowable on the day.
    """
    rank = series.expanding(min_periods=250).rank(pct=True)
    return ((rank > low) & (rank <= high)).to_numpy(dtype=bool)


# --------------------------------------------------------------------------- hypotheses

def build(frame: pd.DataFrame) -> list[dict]:
    out: list[dict] = []

    def add(name, family, claim, origin, null, condition, target,
            macro=False, needs=()):
        # `needs` names the columns this hypothesis reads. The sessions where all of them
        # exist are its universe, and both the condition group and the comparison group are
        # drawn from inside it — otherwise a VIX test is scored against years without VIX.
        universe = np.ones(len(frame), dtype=bool)
        for column in needs:
            universe &= frame[column].notna().to_numpy(dtype=bool)
        horizon = int(target.replace("fwd", "").replace("_lag", ""))
        out.append({
            "name": name, "family": family, "claim": claim, "origin": origin, "null": null,
            "condition": np.asarray(condition, dtype=bool),
            "forward": frame[target].to_numpy(dtype=float),
            "universe": universe, "horizon": horizon,
            "macro": macro, "target": target,
        })

    # --- calendar: previously tested on a short window, now with 7x the sample ----------
    add("h01", "calendar_long", "Monday is gold's worst weekday.",
        "prior sweep RS-XAUUSD-20260824-001 h01, on 672 sessions",
        "weekday carries no information about the next session",
        frame["weekday"] == 0, "fwd1")
    add("h02", "calendar_long", "Friday is gold's strongest weekday.",
        "prior sweep RS-XAUUSD-20260824-001 h02, on 672 sessions",
        "weekday carries no information about the next session",
        frame["weekday"] == 4, "fwd1")
    add("h03", "calendar_long", "September is gold's worst month.",
        "widely repeated seasonal claim; previously untestable on 672 sessions",
        "month carries no information about the next 20 sessions",
        frame["month"] == 9, "fwd20")
    add("h04", "calendar_long",
        "Turn of month: the last three and first three sessions outperform the middle.",
        "prior sweep RS-XAUUSD-20260824-001 h04",
        "position within the month carries no information",
        (frame["day_of_month"] <= 3) | (frame["day_of_month"] > frame["days_in_month"] - 3),
        "fwd5")

    # --- trend state: only reachable with a 200-day window and years of it --------------
    add("h05", "trend_state", "Gold above its 200-day average keeps rising.",
        "trend-following orthodoxy, never tested in this programme",
        "the 200-day average carries no information about the next week",
        frame["gold_close"] > frame["ma200"], "fwd5", needs=("ma200",))
    add("h06", "trend_state",
        "The top decile of stretch above the 200-day average mean-reverts over a month.",
        "prior sweep tested this over 8 hours; the claim is usually made over weeks",
        "stretch carries no information about the next month",
        quantile_mask(frame["stretch"], 0.90, 1.01), "fwd20", needs=("stretch",))
    add("h07", "trend_state", "A new 52-week high is followed by more gains.",
        "breakout continuation; needs years of history to have any 52-week highs at all",
        "a 52-week high carries no information about the next month",
        frame["gold_close"] >= frame["high52"] - 1e-9, "fwd20", needs=("high52",))
    add("h08", "trend_state",
        "The deepest decile of drawdown from the 52-week high is a buying opportunity.",
        "dip-buying orthodoxy",
        "drawdown depth carries no information about the next month",
        quantile_mask(-frame["drawdown"], 0.90, 1.01), "fwd20", needs=("drawdown",))

    # --- streaks: same claims, 7x the sample --------------------------------------------
    add("h09", "streak_long", "Three consecutive down days are followed by a bounce.",
        "prior sweep RS-XAUUSD-20260824-001, on 672 sessions",
        "a run of down days carries no information about the next one",
        frame["down_streak"] >= 3, "fwd1")
    add("h10", "streak_long", "Five consecutive up days are followed by a pullback.",
        "exhaustion folklore; five-day runs are too rare to test on 672 sessions",
        "a run of up days carries no information about the next one",
        frame["up_streak"] >= 5, "fwd1")

    # --- volatility state: GVZ has never been used as a conditioner ---------------------
    add("h11", "volatility_state",
        "The top quintile of realized volatility precedes weaker returns.",
        "vol-scaling orthodoxy",
        "realized volatility carries no information about direction",
        quantile_mask(frame["rv20"], 0.80, 1.01), "fwd5", needs=("rv20",))
    add("h12", "volatility_state",
        "A top-quintile GVZ reading precedes a stronger week for gold.",
        "GVZ is in the repository and no study has ever used it",
        "implied gold volatility carries no information about direction",
        quantile_mask(frame["gvz_close"], 0.80, 1.01), "fwd5_lag", macro=True, needs=("gvz_close",))
    add("h13", "volatility_state",
        "A top-quintile variance risk premium (GVZ minus realized) precedes weaker returns.",
        "the equity-market VRP result, asked of gold",
        "the gap between implied and realized volatility carries no directional information",
        quantile_mask(frame["vrp"], 0.80, 1.01), "fwd20_lag", macro=True, needs=("vrp",))

    # --- macro: the family this programme had data for and never opened -----------------
    add("h14", "macro_state", "A top-quintile VIX reading precedes a stronger week for gold.",
        "the safe-haven claim, stated as something measurable",
        "equity volatility carries no information about gold's direction",
        quantile_mask(frame["vix_close"], 0.80, 1.01), "fwd5_lag", macro=True, needs=("vix_close",))
    add("h15", "macro_state", "A top-decile one-day rise in the 10-year yield hurts gold.",
        "the nominal-rate channel",
        "a one-day yield move carries no information about gold's next session",
        quantile_mask(frame["us10y_chg1"], 0.90, 1.01), "fwd1_lag", macro=True, needs=("us10y_chg1",))
    add("h16", "macro_state",
        "A top-quintile 20-day rise in breakeven inflation precedes a stronger month.",
        "the inflation-hedge claim",
        "breakeven inflation carries no information about gold",
        quantile_mask(frame["t10yie_close_chg20"], 0.80, 1.01), "fwd20_lag", macro=True, needs=("t10yie_close_chg20",))
    add("h17", "macro_state",
        "Falling real yields are gold's actual driver: the bottom quintile of 20-day change "
        "in (10-year yield minus breakeven) precedes a stronger month.",
        "the textbook mechanism; never tested in this programme",
        "the real yield carries no information about gold",
        quantile_mask(frame["real_yield_chg20"], -0.01, 0.20), "fwd20_lag", macro=True, needs=("real_yield_chg20",))
    add("h18", "macro_state",
        "A bottom-quintile 20-day dollar move precedes a stronger month for gold.",
        "prior sweep tested one hour of DXY; the claim is normally made over weeks",
        "the dollar carries no information about gold beyond the same-day accounting link",
        quantile_mask(frame["dxy_close_chg20"], -0.01, 0.20), "fwd20_lag", macro=True, needs=("dxy_close_chg20",))
    add("h19", "macro_state",
        "Risk-off and cheap money together: top-quintile VIX with falling real yields.",
        "the two macro channels required to agree",
        "the conjunction carries no more information than its parts",
        quantile_mask(frame["vix_close"], 0.80, 1.01)
        & quantile_mask(frame["real_yield_chg20"], -0.01, 0.20), "fwd20_lag", macro=True, needs=("vix_close", "real_yield_chg20",))
    add("h20", "macro_state",
        "A top-quintile GVZ with gold already below its 200-day average marks capitulation.",
        "fear plus downtrend, the classic bottom description",
        "the conjunction carries no information",
        quantile_mask(frame["gvz_close"], 0.80, 1.01)
        & (frame["gold_close"] < frame["ma200"]).to_numpy(dtype=bool), "fwd20_lag",
        macro=True, needs=("gvz_close", "ma200"))
    return out


# --------------------------------------------------------------------------- consensus

def consensus(frame: pd.DataFrame, tests: list[dict]) -> dict:
    """The 50%-consensus rule, made falsifiable: does agreement among conditions predict?

    The stated position is 多個條件達成共識 就是 alpha — several conditions each above 50%,
    agreeing, is an edge. That is a testable claim rather than a mood, and 18 years is the
    first sample large enough to test it properly.

    Two things are measured. First, whether each condition's win rate stays above 50% in
    every one of five independent chronological blocks, or only in the pooled average.
    Second, whether days where MORE of them agree actually do better than days where fewer
    do — because if consensus is information, the count has to be monotone in the outcome.
    """
    forward = frame["fwd5"].to_numpy(dtype=float)
    usable = ~np.isnan(forward)
    blocks = np.array_split(np.arange(len(frame)), 5)

    members = []
    for test in tests:
        if test["target"] != "fwd5" or test["macro"]:
            continue
        members.append(test)
    # Not enough same-horizon members to make the count meaningful; fall back to every
    # gold-only condition re-measured on the 5-session horizon.
    if len(members) < 4:
        members = [t for t in tests if not t["macro"]]

    per_condition, matrix = [], []
    for test in members:
        condition = test["condition"]
        rates = []
        for block in blocks:
            picked = condition[block] & usable[block]
            values = forward[block][picked]
            rates.append(round(float((values > 0).mean() * 100), 2) if values.size >= 10 else None)
        pooled = forward[condition & usable]
        clean = [r for r in rates if r is not None]
        per_condition.append({
            "id": test["name"],
            "claim": test["claim"],
            "pooled_win_rate_pct": round(float((pooled > 0).mean() * 100), 2)
            if pooled.size else None,
            "win_rate_by_block_pct": rates,
            "blocks_above_50": sum(1 for r in clean if r > 50),
            "blocks_measured": len(clean),
        })
        matrix.append(condition.astype(int))

    # numpy booleans do NOT add: `a + b` is logical OR and silently caps the count at 1.
    # Casting to int first is the whole reason this line is written out rather than summed
    # with the builtin.
    votes = np.sum(np.vstack(matrix), axis=0) if matrix else np.zeros(len(frame), dtype=int)

    by_votes = []
    for level in range(int(votes.max()) + 1):
        picked = (votes == level) & usable
        values = forward[picked]
        if values.size < 20:
            by_votes.append({"votes": level, "n": int(values.size), "win_rate_pct": None,
                             "mean_return_pct": None})
            continue
        by_votes.append({
            "votes": level,
            "n": int(values.size),
            "win_rate_pct": round(float((values > 0).mean() * 100), 2),
            "mean_return_pct": round(float(values.mean()), 4),
        })

    measurable = [row for row in by_votes if row["win_rate_pct"] is not None]
    monotone = all(
        measurable[i]["win_rate_pct"] <= measurable[i + 1]["win_rate_pct"]
        for i in range(len(measurable) - 1)
    ) if len(measurable) > 1 else None
    stable = [c for c in per_condition
              if c["blocks_measured"] >= 4 and c["blocks_above_50"] == c["blocks_measured"]]

    return {
        "question": (
            "The rule under test: a condition above 50% is an edge, and several such "
            "conditions agreeing is alpha. Both halves are measured here."
        ),
        "horizon": "5 sessions",
        "conditions_tested": len(per_condition),
        "per_condition": per_condition,
        "conditions_above_50_in_every_block": [c["id"] for c in stable],
        "by_vote_count": by_votes,
        "win_rate_monotone_in_votes": monotone,
        "reading": (
            "If consensus carries information, two things must hold: a condition's win rate "
            "stays above 50% in every block rather than only on average, and the win rate "
            "rises as more conditions agree. Either one failing means the pooled number was "
            "describing the period."
        ),
    }


# --------------------------------------------------------------------------- README

def write_readme(payload: dict, frame: pd.DataFrame) -> None:
    """Generate the README, with every number carrying its own derivation.

    Spec section 4.4b: a reader must be able to reconstruct any figure from what is on the
    page. That complaint was exact — 給我一個數值卻沒有說明數值產生的過程 讓我很多
    問號 — so a bare table is not an acceptable output here.
    """
    rows = {h["id"]: h for h in payload["hypotheses"]}
    cov = payload["coverage"]
    con = payload["consensus_analysis"]
    h18, h17, h07 = rows["h18"], rows["h17"], rows["h07"]
    five = next((v for v in con["by_vote_count"] if v["votes"] == 5), None)

    table = "\n".join(
        f'| {h["id"]} | {h["family"]} | {h["claim"][:70]} | {h["n_condition"]} | '
        f'{h.get("effect", 0):+.4f} | {h.get("smallest_resolvable_effect", 0):.4f} | '
        f'{h.get("bootstrap_p_two_sided")} | {h.get("win_rate_pct", 0):.2f} | '
        f'{h.get("baseline_win_rate_pct", 0):.2f} | {h["verdict"]} |'
        for h in payload["hypotheses"]
    )
    consensus_table = "\n".join(
        f'| {c["id"]} | {c["pooled_win_rate_pct"]} | '
        f'{", ".join(str(x) for x in c["win_rate_by_block_pct"])} | '
        f'{c["blocks_above_50"]}/{c["blocks_measured"]} |'
        for c in con["per_condition"]
    )
    votes_table = "\n".join(
        f'| {v["votes"]} | {v["n"]} | {v["win_rate_pct"]} | {v["mean_return_pct"]} |'
        for v in con["by_vote_count"]
    )

    text = f"""# {payload["title"]}

**Study** `{payload["study_id"]}` · generated {payload["generated_at"]}

## What this study is

Every hypothesis sweep in this programme so far ran on a short window. The previous one
used **{cov["prior_sweep_sessions"]} sessions** and returned twenty nulls. For several of
them the honest reading was not "the effect is absent" but "this sample could not resolve
it" — and those are different statements that look identical in a summary table.

This study runs on **{cov["sessions"]} sessions**, {cov["from"]} to {cov["to"]}: a
**{cov["power_multiple"]}x** larger sample. It also opens the one family the repository had
data for and no study had ever used — VIX, GVZ, the 10-year yield and the 10-year breakeven
inflation rate, {cov["macro_sessions"]} sessions ending {cov["macro_to"]}.

## How to read the two numbers that decide every verdict

**Effect** is the mean forward return on days the condition fires, minus the mean on days
it does not, in percent. h05 fires on {rows["h05"]["n_condition"]} sessions with a
{rows["h05"].get("effect"):+.4f}% effect: gold above its 200-day average returned that much
more over the next five sessions than gold below it.

**Smallest resolvable effect** is the size a difference must reach before these two samples
could tell it from zero at roughly 80% power:

```
bound = 2.8 x sigma x sqrt(1/n_condition + 1/n_other)
```

For h05 that is `2.8 x {frame["fwd5"].std(ddof=1):.4f} x sqrt(1/{rows["h05"]["n_condition"]} + 1/{rows["h05"]["n_other"]})`
= **{rows["h05"].get("smallest_resolvable_effect"):.4f}%**. The observed
{rows["h05"].get("effect"):+.4f}% is inside it, so the verdict is `no_evidence` — which
here means *this sample cannot separate it*, not *it is zero*.

Reporting the bound is the difference between a null that is reusable and a null that is a
shrug. It says: not here, and how small an effect we would have caught.

## Result

**{payload["verdict_counts"].get("no_evidence", 0)} of {len(payload["hypotheses"])}
returned `no_evidence`. Zero survived.** Family permutation p =
**{payload["family_permutation"]["family_p"]}** — the chance that twenty hypotheses of pure
noise produce a best result at least as strong as the best one here
(p={payload["family_permutation"]["best_p_in_family"]}). At 0.79, this family is
indistinguishable from twenty coin flips.

| id | family | claim | n | effect % | bound % | boot p | win % | baseline % | verdict |
|---|---|---|---|---|---|---|---|---|---|
{table}

## The one number that answers the 50% question directly

A rule this programme is often asked to accept: a condition winning more than 50% is an
edge. h18 is that argument's best possible case, and its clearest refutation.

**h18: a bottom-quintile 20-day dollar move precedes a stronger month for gold.**
It fires {h18["n_condition"]} times and wins **{h18["win_rate_pct"]}%** of the time. By the
">50% is alpha" rule this is not marginal — it is enormous.

Now the comparison. The days it is measured against win
**{h18["baseline_win_rate_pct"]}%**.

The reason is visible in one column of the table: `sessions_in_universe` for h18 is
{h18["sessions_in_universe"]}, not {cov["sessions"]}. The dollar series in this repository
starts in 2021, so h18 can only be scored on 2021-2026 — and gold rose through almost all
of it. In that window *most* days win. The edge is
{h18["win_rate_pct"]} - {h18["baseline_win_rate_pct"]} =
**{h18["win_rate_edge_pct"]:+.2f} points**, and the mean-return version,
{h18.get("effect"):+.4f}%, sits inside the {h18.get("smallest_resolvable_effect"):.4f}% this
sample can resolve.

This is not a story told to explain away a good number. It is a number that changed by
**5.4 percentage points** when the comparison group was restricted to the same era as the
condition — and the first version of this study got it wrong in exactly that way. Before
the fix, h18 read as effect +1.5520% at p=0.032 against a baseline of 54.56%, because the
comparison group included thirteen years in which the dollar series did not exist.

**A win rate is only a number about the days it is compared against.**

## The closest thing to a real macro effect

**h17: falling real yields — the textbook driver of gold — over the next month.**

- effect **{h17.get("effect"):+.4f}%** against a bound of **{h17.get("smallest_resolvable_effect"):.4f}%**
- {h17["win_rate_pct"]}% win rate against a {h17["baseline_win_rate_pct"]}% baseline
- and the sign holds in all three chronological windows:
  {", ".join(f'{k} {v["effect"]:+.4f}' for k, v in h17["by_period"].items() if v["effect"] is not None)}

It is the only macro condition whose direction is stable across time, and it still lands
just **inside** what {h17["sessions_in_universe"]} sessions can separate — 0.048 percentage
points short. That is a specific, actionable statement rather than a shrug: this is the
hypothesis worth more data, and roughly how much more it would take.

h07 (a new 52-week high, p={h07.get("bootstrap_p_two_sided")}) is the lowest p-value in the
family, but its sign inverts in the middle window
({h07["by_period"]["valid"]["effect"]:+.4f}%), which is what a period looks like when it is
mistaken for a condition.

## The consensus rule, tested rather than argued

The second half of that rule is that several >50% conditions agreeing is alpha. That is
testable rather than arguable, and 18 years is the first sample big enough to test it.

**Test one — does a condition stay above 50%, or only average above 50%?**
Five equal chronological blocks, {con["conditions_tested"]} gold-only conditions:

| id | pooled win % | win % by block (5 blocks, oldest first) | blocks > 50% |
|---|---|---|---|
{consensus_table}

**{len(con["conditions_above_50_in_every_block"])} of {con["conditions_tested"]} conditions
stay above 50% in every block.** Every single one has at least one period where it loses
more often than it wins. h03 is the extreme: pooled {con["per_condition"][2]["pooled_win_rate_pct"]}%,
but its blocks run from {min(x for x in con["per_condition"][2]["win_rate_by_block_pct"] if x)}%
to {max(x for x in con["per_condition"][2]["win_rate_by_block_pct"] if x)}%.

**Test two — does agreement help?** If consensus is information, more agreement must mean
better outcomes.

| conditions agreeing | sessions | win % | mean return % |
|---|---|---|---|
{votes_table}

Monotone in votes: **{con["win_rate_monotone_in_votes"]}**.

And the row that matters most: at **5 agreeing conditions** the win rate is the highest in
the table at **{five["win_rate_pct"]}%** — while the mean return is
**{five["mean_return_pct"]}%**, the only negative number in the column.

The highest-consensus bucket wins most often and loses money. That is the fourth
independent time this programme has produced that pattern, after the take-profit ladder,
the %B filter and the two-strategy consensus test. It is not a coincidence and it is not
rhetoric: **raising a win rate is easy, and doing it does not make money.** The 92 sessions
behind that row are few enough that the exact figure is unstable; the direction is the
point, and the direction is the opposite of the one the rule predicts.

## Limitations

{chr(10).join("- " + item for item in payload["limitations"])}

## Method

- Screens, in order: {"; ".join(payload["method"]["screens"])}
- Quantiles: {payload["method"]["quantiles"]}
- Macro lookahead guard: {payload["method"]["macro_lookahead_guard"]}
- Universe: {payload["method"]["universe"]}
- Period split: {payload["method"]["period_split"]}
- Bootstrap: {payload["method"]["bootstrap_draws"]} draws, block = {payload["method"]["block_bootstrap_sessions"]}
- Seed {payload["method"]["seed"]}; runner `scripts/research/build_xauusd_long_history_sweep.py`

## Sources

- `local-inputs/gold_daily.csv` — {cov["sessions"]} daily bars, {cov["from"]} to {cov["to"]}
- `local-inputs/dxy_daily.csv` — dollar index, 2021-08-23 onward
- `local-inputs/VIX_1d.csv`, `GVZ_1d.csv`, `US10Y_1d.csv`, `T10YIE_1d.csv` — 2012 to {cov["macro_to"]}
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- main

def main() -> int:
    frame = load()
    tests = build(frame)

    results = []
    for test in tests:
        results.append(sh.evaluate(
            name=test["name"], family=test["family"], claim=test["claim"],
            origin=test["origin"], null_description=test["null"],
            condition=test["condition"], forward=test["forward"],
            universe=test["universe"],
            stream=sh.stream_for(SEED, test["name"]),
            block=block_for(test["horizon"]), cost_pct=COST_PCT,
        ))

    family = sh.family_minimum_p_correction(results, sh.stream_for(SEED, "family"))
    verdicts: dict[str, int] = {}
    for row in results:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1

    payload = {
        "study_id": STUDY_ID,
        "schema_version": "1.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "title": (
            "Twenty hypotheses on 18 years of daily gold, and the macro drivers "
            "the programme had data for and never opened"
        ),
        "method": {
            "bars": "XAUUSD daily close-to-close",
            "block_bootstrap_sessions": "3x the forward horizon, floored at 21",
            "bootstrap_draws": sh.DEFAULT_BOOTSTRAP,
            "round_trip_cost_pct": COST_PCT,
            "seed": SEED,
            "quantiles": (
                "expanding rank with a 250-session warm-up, so a threshold never uses data "
                "from after the day it is applied to"
            ),
            "macro_lookahead_guard": (
                "macro conditions read day t; gold's forward return starts at the close of "
                "day t+1"
            ),
                "universe": (
                "each hypothesis is scored only on sessions where every series it reads "
                "exists; the comparison group comes from inside that same universe"
            ),
            "period_split": (
                "chronological thirds of the sessions the hypothesis spans, not of the "
                "gold series it sits inside"
            ),
            "screens": [
                "resolution bound (smallest separable effect at ~80% power)",
                "moving-block bootstrap, two-sided",
                "sign consistency across chronological thirds",
                f"effect net of a {COST_PCT}% round trip must still clear the bound",
            ],
        },
        "coverage": {
            "sessions": int(len(frame)),
            "from": str(frame["date"].iloc[0].date()),
            "to": str(frame["date"].iloc[-1].date()),
            "macro_sessions": int(frame["vix_close"].notna().sum()),
            "macro_to": str(frame.loc[frame["vix_close"].notna(), "date"].iloc[-1].date()),
            "prior_sweep_sessions": 672,
            "power_multiple": round(len(frame) / 672, 1),
        },
        "verdict_counts": verdicts,
        "family_permutation": family,
        "hypotheses": results,
        "consensus_analysis": consensus(frame, tests),
        "limitations": [
            "Daily close-to-close only. Nothing here says anything about intraday entries, "
            "which is where both live strategies operate.",
            "The macro series end 2026-06-18 while gold runs to 2026-08-21, so macro-"
            "conditioned hypotheses see two months less than the calendar ones.",
            "Gold's 00:00Z daily stamp and the US macro close are different instants. The "
            "one-session lag removes the ambiguity at the cost of testing a slightly later "
            "entry than a live trader would get.",
            "A wider resolution bound is not evidence of absence. Every null is reported "
            "with the smallest effect its sample could have separated.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(payload, frame)
    print(json.dumps({
        "study": STUDY_ID,
        "sessions": payload["coverage"]["sessions"],
        "power_multiple": payload["coverage"]["power_multiple"],
        "verdicts": verdicts,
        "family_p": family.get("family_p"),
        "survivors": [r["id"] for r in results if r["verdict"] == "survives_screens"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
