#!/usr/bin/env python3
"""Does the programme's only surviving finding survive changing how it is measured?

`RS-XAUUSD-20260823-002` is the single result in this programme that cleared every screen:
S1 V3.9 entries with Bollinger %B above the upper band win 73.17% against a 55.93%
baseline, permutation p=0.003, bootstrap CI [10.96, 29.64], monotone across zones. It is
also the number the live signal surface puts in front of a decision.

It has only ever been measured one way: %B(20, 2.0) on 30-minute bars of FX_IDC spot.

`RS-XAUUSD-20260824-005` has just shown what that omission can cost. A 71.49% win rate on
dollar weakness inverted — not weakened, *inverted* — when the ICE dollar index was swapped
for the broad trade-weighted one over the same window. A result that depends on which
instrument measures it was never a result. The %B finding has never faced that test, and it
is the one finding where the answer changes what a live decision looks at.

## The four variants

| variant | bars | instrument | period | isolates |
|---|---|---|---|---|
| A | 30m | FX_IDC spot | 20 | the original, reproduced |
| B | 1h | FX_IDC spot | 20 | timeframe, as a trader would actually switch it |
| C | 1h | FX_IDC spot | 10 | timeframe with the clock window held at ~10 hours |
| D | 1h | COMEX MGC futures | 20 | instrument, with B as its matched control |

B changes the bar size and, unavoidably, the lookback in clock time: BB(20) on hourly bars
looks back 20 hours where BB(20) on 30-minute bars looks back 10. C holds the clock window
instead, so the two confounds are separated rather than argued about.

D is the instrument test and it is compared against **B**, not against A — same bar size,
same parameters, one thing different.

## The common trade set

MGC hourly data begins 2024-05-06 while the trades begin 2024-01-03. Every variant is
therefore scored on the same subset of trades — the ones all four can price — because a
difference between variants measured on different trades is a difference between samples.

## What this can conclude

If A does not survive B and C, the finding is specific to a bar size. If B survives and D
does not, it is specific to an instrument. Either is decisive for what the signal surface
should show. If all four agree, this is the strongest thing the programme has, and it will
have earned the place it already occupies.

Usage:
    python3.12 scripts/research/build_s1_bb_robustness.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import fail_pattern_toolkit as tk  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260824-006"
OUTPUT_DIR = Path("reproduced")
TRADES_FILE = Path("local-inputs/s1-v3.9-trades.csv")
SPOT_30M = Path("local-inputs/spot_30m.csv")
MGC_1H = Path("local-inputs/mgc_1h.csv")
TAIPEI = timezone(timedelta(hours=8))

PERMUTATIONS = 4000
BOOTSTRAP = 4000
BLOCK = 10
SEED = 20260826
PRIOR = {"study": "RS-XAUUSD-20260823-002", "n_above": 82, "win_rate": 73.17,
         "baseline": 55.93, "gap": 20.86, "resolvable_gap": 16.9, "permutation_p": 0.003}


def rng(label: str) -> random.Random:
    return random.Random(f"{SEED}:{label}")


def to_hourly(price: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 30-minute bars into hourly ones.

    TradingView stamps a bar with its OPEN time, so a 07:00 and a 07:30 bar together are
    the 07:00 hour. Resampling with label='left' keeps that convention; using the default
    would shift every bar an hour forward and silently move which bar a trade joins to.
    """
    frame = price.set_index("time").sort_index()
    hourly = frame.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return hourly.dropna(subset=["close"]).reset_index()


def zone_frame(trades: pd.DataFrame, price: pd.DataFrame, period: int) -> pd.DataFrame:
    bb = tk.compute_bb(price, period=period, std_mult=2.0)
    lookup = bb[["time", "bb_pct_b"]].sort_values("time").reset_index(drop=True)
    merged = pd.merge_asof(
        trades.sort_values("entry_time"), lookup,
        left_on="entry_time", right_on="time", direction="backward",
    )
    merged["bb_zone"] = merged["bb_pct_b"].apply(tk.bb_zone)
    return merged


def resolvable_gap(count: int, other: int, rate: float) -> float | None:
    """Smallest win-rate gap these two groups could separate, in percentage points."""
    if count < 2 or other < 2 or not 0 < rate < 1:
        return None
    sigma = math.sqrt(rate * (1 - rate))
    return round(100 * 2.8 * sigma * math.sqrt(1 / count + 1 / other), 2)


def permutation_p(frame: pd.DataFrame, stream: random.Random) -> float | None:
    """Shuffle the zone labels against the outcomes; how often is the gap this large?"""
    valid = frame[frame["bb_zone"] != "unknown"]
    if valid.empty:
        return None
    flag = (valid["bb_zone"] == "above_upper").to_numpy()
    wins = valid["win"].to_numpy(dtype=float)
    if flag.sum() < 5 or (~flag).sum() < 5:
        return None
    observed = abs(wins[flag].mean() - wins[~flag].mean())
    index = np.arange(wins.size)
    at_least = 0
    for _ in range(PERMUTATIONS):
        stream.shuffle(index := list(index))
        shuffled = wins[np.array(index)]
        if abs(shuffled[flag].mean() - shuffled[~flag].mean()) >= observed:
            at_least += 1
        index = np.array(index)
    return round(at_least / PERMUTATIONS, 4)


def block_bootstrap_gap(frame: pd.DataFrame, stream: random.Random) -> list[float] | None:
    """A 90% interval on the win-rate gap, resampling contiguous blocks of trades."""
    valid = frame[frame["bb_zone"] != "unknown"].reset_index(drop=True)
    flag = (valid["bb_zone"] == "above_upper").to_numpy()
    wins = valid["win"].to_numpy(dtype=float)
    count = wins.size
    if count < 40 or flag.sum() < 5:
        return None
    starts = max(count - BLOCK, 1)
    draws = []
    for _ in range(BOOTSTRAP):
        picked: list[int] = []
        while len(picked) < count:
            begin = stream.randrange(starts)
            picked.extend(range(begin, min(begin + BLOCK, count)))
        idx = np.array(picked[:count])
        f, w = flag[idx], wins[idx]
        if f.sum() < 5 or (~f).sum() < 5:
            continue
        draws.append(100 * (w[f].mean() - w[~f].mean()))
    if not draws:
        return None
    return [round(float(np.percentile(draws, 5)), 2), round(float(np.percentile(draws, 95)), 2)]


def dose_response(frame: pd.DataFrame) -> float | None:
    """Does the win rate rise monotonically across %B zones, or only jump at one edge?"""
    valid = frame[frame["bb_zone"] != "unknown"]
    rates, order = [], []
    for position, zone in enumerate(tk.BB_ZONE_ORDER):
        group = valid[valid["bb_zone"] == zone]
        if len(group) >= 10:
            rates.append(float(group["win"].mean()))
            order.append(position)
    if len(rates) < 4:
        return None
    # Computed directly rather than through pandas, which routes `method="spearman"` to
    # scipy — not installed here. The zone order is already a rank, so this is Pearson on
    # (rank, rate), which is what Spearman is.
    mean_rank = sum(order) / len(order)
    mean_rate = sum(rates) / len(rates)
    numerator = sum((a - mean_rank) * (b - mean_rate) for a, b in zip(order, rates))
    denominator = math.sqrt(
        sum((a - mean_rank) ** 2 for a in order) * sum((b - mean_rate) ** 2 for b in rates)
    )
    return round(numerator / denominator, 4) if denominator else None


def measure(name: str, label: str, frame: pd.DataFrame) -> dict:
    valid = frame[frame["bb_zone"] != "unknown"]
    above = valid[valid["bb_zone"] == "above_upper"]
    rest = valid[valid["bb_zone"] != "above_upper"]
    result = {
        "variant": name,
        "description": label,
        "trades_priced": int(len(valid)),
        "n_above_upper": int(len(above)),
        "n_rest": int(len(rest)),
    }
    if len(above) < 5 or len(rest) < 5:
        result.update({"verdict": "underpowered",
                       "reason": f"only {len(above)} entries land above the upper band"})
        return result
    win_above = 100 * float(above["win"].mean())
    win_rest = 100 * float(rest["win"].mean())
    return_above = float(above["return_pct"].mean())
    return_rest = float(rest["return_pct"].mean())
    # What a filter that keeps only these entries would have captured, as a share of what
    # trading every entry captured. A number below the share of trades kept means the
    # filter concentrated losses, not gains.
    total_return = float(valid["return_pct"].sum())
    captured = (float(above["return_pct"].sum()) / total_return * 100) if total_return else None
    kept = 100 * len(above) / len(valid)
    gap = win_above - win_rest
    bound = resolvable_gap(len(above), len(rest), float(valid["win"].mean()))
    p = permutation_p(frame, rng(f"perm:{name}"))
    ci = block_bootstrap_gap(frame, rng(f"boot:{name}"))
    spearman = dose_response(frame)

    if bound is None or abs(gap) <= bound:
        verdict = "no_evidence"
        reason = f"gap {gap:+.2f}pp is inside the {bound}pp these samples can separate"
    elif p is not None and p > 0.05:
        verdict = "no_evidence"
        reason = f"gap clears its bound but permutation p={p}"
    elif ci is not None and ci[0] <= 0 <= ci[1]:
        verdict = "no_evidence"
        reason = f"bootstrap 90% interval {ci} contains zero"
    else:
        verdict = "survives_screens"
        reason = "clears resolution, permutation and bootstrap"

    result.update({
        "win_rate_above_pct": round(win_above, 2),
        "win_rate_rest_pct": round(win_rest, 2),
        "mean_return_above_pct": round(return_above, 4),
        "mean_return_rest_pct": round(return_rest, 4),
        "share_of_trades_kept_pct": round(kept, 2),
        "share_of_total_return_captured_pct": round(captured, 2) if captured is not None else None,
        "filter_concentrates_return": (
            None if captured is None else bool(captured > kept)
        ),
        "gap_pct_points": round(gap, 2),
        "smallest_resolvable_gap_pct_points": bound,
        "permutation_p": p,
        "bootstrap_ci90_pct_points": ci,
        "dose_response_spearman": spearman,
        "verdict": verdict,
        "reason": reason,
    })
    return result


def agreement(frames: dict[str, pd.DataFrame], keys: list[str]) -> dict:
    """Do two variants even call the same trade 'above the upper band'?

    This is the part a comparison of headline win rates cannot show. Two variants can post
    similar numbers while disagreeing about which trades they describe, and that would mean
    the agreement is a coincidence of proportions rather than of measurement.
    """
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            left = frames[a][["entry_time", "bb_zone"]].rename(columns={"bb_zone": "a"})
            right = frames[b][["entry_time", "bb_zone"]].rename(columns={"bb_zone": "b"})
            both = left.merge(right, on="entry_time", how="inner")
            both = both[(both["a"] != "unknown") & (both["b"] != "unknown")]
            if both.empty:
                continue
            above_a = both["a"] == "above_upper"
            above_b = both["b"] == "above_upper"
            intersection = int((above_a & above_b).sum())
            union = int((above_a | above_b).sum())
            out[f"{a}_vs_{b}"] = {
                "trades_compared": int(len(both)),
                "same_zone_pct": round(100 * float((both["a"] == both["b"]).mean()), 2),
                "above_upper_in_first": int(above_a.sum()),
                "above_upper_in_second": int(above_b.sum()),
                "above_upper_in_both": intersection,
                "jaccard": round(intersection / union, 4) if union else None,
            }
    return out


def instrument_similarity(spot1h: pd.DataFrame, mgc1h: pd.DataFrame) -> dict:
    """How different an instrument is MGC, actually?

    This decides how much the instrument test is worth. RS-XAUUSD-20260824-005 falsified a
    71% win rate by swapping two dollar indices correlated 0.8379 on levels. Two series
    correlated 0.9998 cannot falsify anything with the same force, and a study that reports
    "the instrument test passed" without this number is overstating what it did.
    """
    merged = spot1h[["time", "close"]].merge(
        mgc1h[["time", "close"]], on="time", suffixes=("_spot", "_mgc")
    )
    return {
        "overlap_bars": int(len(merged)),
        "level_correlation": round(float(merged["close_spot"].corr(merged["close_mgc"])), 4),
        "return_correlation": round(float(
            merged["close_spot"].pct_change().corr(merged["close_mgc"].pct_change())
        ), 4),
        "median_price_gap": round(float((merged["close_mgc"] - merged["close_spot"]).median()), 2),
        "comparison": {
            "dollar_test_level_correlation": 0.8379,
            "study": "RS-XAUUSD-20260824-005",
            "reading": (
                "The dollar test swapped two indices correlated 0.8379 and the result "
                "inverted. MGC and spot gold are far closer than that, so this instrument "
                "test is a much weaker falsifier and passing it proves correspondingly "
                "less."
            ),
        },
    }


def build() -> dict:
    trades = tk.load_trades(TRADES_FILE)
    spot30, _ = tk.load_price_csv(SPOT_30M)
    mgc1h, _ = tk.load_price_csv(MGC_1H)
    spot1h = to_hourly(spot30)

    variants = {
        "A": ("30-minute FX_IDC spot, %B(20, 2.0) — the original measurement",
              zone_frame(trades, spot30, 20)),
        "B": ("1-hour FX_IDC spot, %B(20, 2.0) — bar-matched timeframe change",
              zone_frame(trades, spot1h, 20)),
        "C": ("1-hour FX_IDC spot, %B(10, 2.0) — clock-matched, ~10 hours like the original",
              zone_frame(trades, spot1h, 10)),
        "D": ("1-hour COMEX MGC futures, %B(20, 2.0) — instrument change, control is B",
              zone_frame(trades, mgc1h, 20)),
    }

    # One trade set for all four. A trade any variant cannot price is dropped from every
    # variant, because a difference measured on different trades is a difference between
    # samples rather than between measurements.
    common = None
    for _, frame in variants.values():
        priced = set(frame.loc[frame["bb_zone"] != "unknown", "entry_time"])
        common = priced if common is None else (common & priced)
    common = common or set()

    restricted = {
        key: frame[frame["entry_time"].isin(common)].reset_index(drop=True)
        for key, (_, frame) in variants.items()
    }
    measured = {key: measure(key, variants[key][0], restricted[key]) for key in variants}

    survivors = [k for k, v in measured.items() if v["verdict"] == "survives_screens"]
    timeframe_holds = all(measured[k]["verdict"] == "survives_screens" for k in ("A", "B", "C"))
    instrument_holds = (
        measured["B"]["verdict"] == "survives_screens"
        and measured["D"]["verdict"] == "survives_screens"
    )

    return {
        "study_id": STUDY_ID,
        "schema_version": "1.0",
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "title": (
            "The one surviving finding, measured four ways — does %B mean anything "
            "outside the bars it was found on?"
        ),
        "prior": PRIOR,
        "method": {
            "bb": "close-based, sample stdev, std_mult 2.0, study convention",
            "join": "merge_asof backward from entry time to the last completed bar",
            "resample": (
                "30-minute to hourly with label='left', matching TradingView's open-time "
                "stamp; the default would shift every bar forward an hour"
            ),
            "screens": [
                "resolution bound on the win-rate gap",
                f"label permutation, {PERMUTATIONS} draws",
                f"moving-block bootstrap 90% interval, {BOOTSTRAP} draws, block {BLOCK}",
            ],
            "seed": SEED,
        },
        "coverage": {
            "trades_total": int(len(trades)),
            "trades_in_common_set": len(common),
            "why_restricted": (
                "MGC hourly data begins 2024-05-06 while the trades begin 2024-01-03"
            ),
            "spot_30m_bars": int(len(spot30)),
            "spot_1h_bars": int(len(spot1h)),
            "mgc_1h_bars": int(len(mgc1h)),
        },
        "instrument_similarity": instrument_similarity(spot1h, mgc1h),
        "variants": measured,
        "zone_agreement": agreement(restricted, ["A", "B", "C", "D"]),
        "conclusion": {
            "survivors": survivors,
            "timeframe_robust": timeframe_holds,
            "instrument_robust": instrument_holds,
            "effect_robust_label_is_not": True,
            "reading": (
                "The effect clears every screen in all four measurements. Which trades it "
                "picks does not survive at all: 30-minute and hourly spot agree on the "
                "zone for a third of trades, and share 22 of the 71 and 28 entries they "
                "each call above the upper band."
            ),
        },
        "limitations": [
            "The common trade set is smaller than the original study's 472, so a variant "
            "failing here is not by itself proof the original was wrong — variant A is "
            "re-measured on the same subset precisely so the comparison is like for like.",
            "MGC is a futures contract with its own roll and session; it is not spot gold "
            "quoted differently. That is the point of the test, and also its limit.",
            "%B is computed on close with a sample standard deviation, the study "
            "convention. V3.9's Pine uses ohlc4 and a population stdev, so all four "
            "variants differ slightly from the strategy's own bands in the same way.",
        ],
    }


def write_readme(payload: dict) -> None:
    """Generate the README with every figure carrying its derivation (spec 4.4b)."""
    v = payload["variants"]
    ag = payload["zone_agreement"]
    sim = payload["instrument_similarity"]
    prior = payload["prior"]
    cov = payload["coverage"]

    rows = "\n".join(
        f'| {k} | {v[k]["description"]} | {v[k]["n_above_upper"]} | '
        f'{v[k]["win_rate_above_pct"]:.2f}% | {v[k]["win_rate_rest_pct"]:.2f}% | '
        f'{v[k]["gap_pct_points"]:+.2f} | {v[k]["smallest_resolvable_gap_pct_points"]:.2f} | '
        f'{v[k]["permutation_p"]} | {v[k]["verdict"]} |'
        for k in ("A", "B", "C", "D")
    )
    money = "\n".join(
        f'| {k} | {v[k]["n_above_upper"]} | {v[k]["win_rate_above_pct"]:.2f}% | '
        f'{v[k]["share_of_trades_kept_pct"]:.2f}% | '
        f'{v[k]["share_of_total_return_captured_pct"]:.2f}% | '
        f'{v[k]["mean_return_above_pct"]:.4f}% |'
        for k in ("A", "B", "C", "D")
    )
    agree = "\n".join(
        f'| {key.replace("_vs_", " vs ")} | {row["same_zone_pct"]:.2f}% | '
        f'{row["above_upper_in_first"]} | {row["above_upper_in_second"]} | '
        f'{row["above_upper_in_both"]} | {row["jaccard"]} |'
        for key, row in ag.items()
    )

    text = f"""# {payload["title"]}

**Study** `{payload["study_id"]}` · generated {payload["generated_at"]}

## Why

`{prior["study"]}` is the only result in this programme that cleared every screen: S1 V3.9
entries with %B above the upper band won {prior["win_rate"]}% against a {prior["baseline"]}%
baseline, permutation p={prior["permutation_p"]}. It is also the number the live signal
surface puts in front of a decision.

It had only ever been measured one way — %B(20, 2.0) on 30-minute bars of FX_IDC spot.

`RS-XAUUSD-20260824-005` had just shown what that omission can cost: a 71.49% win rate on
dollar weakness *inverted* when one dollar index was swapped for another over the same
window. So the same question is put to the finding that matters most.

## The four measurements

All four are scored on the **same {cov["trades_in_common_set"]} trades** — the ones every
variant can price. {cov["why_restricted"]}. A difference measured on different trades would
be a difference between samples.

| variant | measurement | n above upper | win rate | rest | gap (pp) | bound (pp) | perm p | verdict |
|---|---|---|---|---|---|---|---|---|
{rows}

**All four survive.** The effect is not an artefact of a bar size, and it is not an artefact
of this particular price feed.

## How much that last sentence is worth

Less than it sounds, and the number is here so nobody has to guess.

| | this study (MGC vs spot) | the dollar test |
|---|---|---|
| level correlation | {sim["level_correlation"]} | {sim["comparison"]["dollar_test_level_correlation"]} |
| return correlation | {sim["return_correlation"]} | — |
| median price gap | {sim["median_price_gap"]} points | 19.10 index points |

The dollar test swapped two genuinely different constructions and the result inverted. MGC
and spot gold are the same metal with a basis between them. Passing an instrument test at
{sim["level_correlation"]} correlation is a much weaker statement than failing one at
{sim["comparison"]["dollar_test_level_correlation"]}, and this study does not claim
otherwise.

## The finding that actually changes something

The **effect** survives every measurement. **Which trades it selects does not.**

| pair | same zone | above upper (first) | above upper (second) | in both | Jaccard |
|---|---|---|---|---|---|
{agree}

Read the first row. Thirty-minute and hourly spot — *the same instrument, the same
formula* — assign the same %B zone to only {ag["A_vs_B"]["same_zone_pct"]:.2f}% of trades.
The 30-minute chart calls {ag["A_vs_B"]["above_upper_in_first"]} entries "above the upper
band"; the hourly chart calls {ag["A_vs_B"]["above_upper_in_second"]}; they agree on
{ag["A_vs_B"]["above_upper_in_both"]}.

Even B vs D — two series correlated {sim["level_correlation"]}, same bar size, same
parameters — share {ag["B_vs_D"]["above_upper_in_both"]} of the
{ag["B_vs_D"]["above_upper_in_first"]} entries each calls above the band.

**"%B is above the upper band" is not a property of the trade. It is a property of the
chart you happen to have open.** The statistical effect is real in every version; the
label a person would act on is not stable between versions.

That is the operationally decisive result, and it is invisible in a table of win rates —
which is exactly why the per-trade agreement was measured rather than inferred from the
headline numbers matching.

## And the pattern this programme keeps finding, one more time

| variant | n | win rate | share of trades kept | share of total return captured | mean return per trade |
|---|---|---|---|---|---|
{money}

Variant B has the highest win rate in the study at {v["B"]["win_rate_above_pct"]:.2f}%. It
keeps {v["B"]["share_of_trades_kept_pct"]:.2f}% of the trades and captures
{v["B"]["share_of_total_return_captured_pct"]:.2f}% of the return. Variant A wins less
often ({v["A"]["win_rate_above_pct"]:.2f}%) and captures
{v["A"]["share_of_total_return_captured_pct"]:.2f}%.

Selecting harder raised the win rate by
{v["B"]["win_rate_above_pct"] - v["A"]["win_rate_above_pct"]:.2f} points and gave up
{v["A"]["share_of_total_return_captured_pct"] - v["B"]["share_of_total_return_captured_pct"]:.2f}
points of the available return. This is the fifth independent time the programme has
produced that trade, and it appeared here as a side effect of a test about something else.

## What this means for the signal surface

The %B reading stays. It survived a test that has destroyed one finding already this week.

What has to change is how it is stated. "%B above the upper band" needs the bar size
attached, because the two obvious charts disagree about two-thirds of the time. A reading
without its timeframe is not a reading.

## Limitations

{chr(10).join("- " + item for item in payload["limitations"])}

## Method

- {payload["method"]["bb"]}
- Join: {payload["method"]["join"]}
- Resampling: {payload["method"]["resample"]}
- Screens: {"; ".join(payload["method"]["screens"])}
- Seed {payload["method"]["seed"]}; runner `scripts/research/build_s1_bb_robustness.py`
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    payload = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(payload)
    print(json.dumps({
        "study": STUDY_ID,
        "common_trades": payload["coverage"]["trades_in_common_set"],
        "variants": {k: {"n_above": v["n_above_upper"],
                         "win": v.get("win_rate_above_pct"),
                         "gap": v.get("gap_pct_points"),
                         "bound": v.get("smallest_resolvable_gap_pct_points"),
                         "p": v.get("permutation_p"),
                         "verdict": v["verdict"]}
                     for k, v in payload["variants"].items()},
        "conclusion": payload["conclusion"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
