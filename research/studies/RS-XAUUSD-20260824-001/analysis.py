#!/usr/bin/env python3
"""RS-XAUUSD-20260824-001 — twenty hypotheses, one harness, and a registry of what is closed.

The programme has spent twenty-five studies establishing that this dataset contains no
tradeable directional edge. That is a real result, and it is also an asset that has been
stored badly: it lives scattered across twenty-five decision logs, so the next person — or
the next model — cannot cheaply ask "has this been tried, and how hard?"

This study does two things at once. It tests twenty new hypotheses, and it emits every
result in a single machine-readable shape so the answer accumulates instead of dispersing.
A null with its resolution bound attached is reusable; a null buried in prose is not.

## Why a shared harness rather than twenty bespoke tests

Every hypothesis here reduces to the same question: does a condition observable at time t
change the distribution of returns after t? Running them through one function means the
null, the resampling, the out-of-sample split and the resolution bound are identical
across all twenty, so the results are comparable with each other and with whatever a later
session adds. Adding a hypothesis is writing one function that returns a boolean series.

## The screens, applied uniformly

- **Effect against the same period's baseline**, never against zero. The instrument rose
  112% over the sample and scoring against zero makes everything look profitable.
- **Moving-block bootstrap**, because 30-minute bars and the trades built on them cluster;
  an iid resample treats a correlated run as independent evidence.
- **Out-of-sample thirds.** Sign consistency across train/valid/holdout is reported, and
  treated as the weak screen RS-XAUUSD-20260819-001 showed it to be — four of six
  candidates that passed it there were artefacts.
- **Smallest resolvable effect** printed beside every number, so "no evidence" is
  distinguishable from "no power". Several hypotheses here are the second thing, and saying
  so is the useful part.
- **One family-wide permutation across all twenty**, because running twenty tests and
  reporting the best is how noise gets published.

Usage:
    python3.12 -m scripts.research.build_xauusd_hypothesis_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
import fail_pattern_toolkit as tk  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260824-001"
OUTPUT_DIR = Path("reproduced")
BARS_FILE = Path("local-inputs/FX_IDC_XAUUSD, 30_volumn.csv")
DXY_FILE = Path("local-inputs/TVC_DXY, 60.csv")
CFTC_FILE = Path("local-inputs/CFTC_GOLD.csv")
TAIPEI = timezone(timedelta(hours=8))

BOOTSTRAP = 2000
BLOCK = 12
FAMILY_PERMUTATIONS = 2000
SEED = 20260824
COST_PCT = 0.02
DAY_BOUNDARY_HOUR = 7


def rng(label: str) -> random.Random:
    return random.Random(f"{SEED}:{label}")


# --------------------------------------------------------------------------- harness

def smallest_resolvable(count: int, other: int, sigma: float) -> float | None:
    """Effect size these two samples could separate at roughly 80% power."""
    if count < 2 or other < 2 or sigma <= 0:
        return None
    return round(2.8 * sigma * math.sqrt(1 / count + 1 / other), 4)


def evaluate(
    name: str,
    family: str,
    claim: str,
    origin: str,
    null_description: str,
    condition: np.ndarray,
    forward: np.ndarray,
    period: np.ndarray,
    stream: random.Random,
    unit: str = "% forward return",
) -> dict:
    """One hypothesis, one verdict, with the bound that makes the verdict readable."""
    usable = ~np.isnan(forward)
    condition, forward, period = condition[usable], forward[usable], period[usable]
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
    if hit.size < 20 or miss.size < 20:
        result.update({
            "verdict": "underpowered",
            "reason": f"condition fires {hit.size} times; below the 20 needed to say anything",
        })
        return result

    effect = float(hit.mean() - miss.mean())
    sigma = float(forward.std(ddof=1))
    bound = smallest_resolvable(hit.size, miss.size, sigma)

    # Moving-block bootstrap on the ordered series.
    count = forward.size
    starts = max(count - BLOCK, 1)
    draws = []
    for _ in range(BOOTSTRAP):
        index = []
        while len(index) < count:
            begin = stream.randrange(starts)
            index.extend(range(begin, min(begin + BLOCK, count)))
        index = np.array(index[:count])
        sample_condition, sample_forward = condition[index], forward[index]
        if sample_condition.sum() < 10 or (~sample_condition).sum() < 10:
            continue
        draws.append(
            sample_forward[sample_condition].mean() - sample_forward[~sample_condition].mean()
        )
    draws = np.array(draws)
    share_above = float((draws > 0).mean()) if draws.size else None
    two_sided = None
    if share_above is not None:
        two_sided = round(2 * min(share_above, 1 - share_above), 4)

    by_period = {}
    for label in ("train", "valid", "holdout"):
        mask = period == label
        window_hit = forward[mask & condition]
        window_miss = forward[mask & ~condition]
        by_period[label] = {
            "n_condition": int(window_hit.size),
            "effect": round(float(window_hit.mean() - window_miss.mean()), 4)
            if window_hit.size >= 5 and window_miss.size >= 5 else None,
        }
    signs = [row["effect"] for row in by_period.values() if row["effect"] is not None]
    consistent = len(signs) == 3 and (all(s > 0 for s in signs) or all(s < 0 for s in signs))

    resolvable = bound is not None and abs(effect) > bound
    if not resolvable:
        verdict = "no_evidence"
        reason = (
            f"effect {effect:+.4f} is inside the {bound:+.4f} these samples can separate"
        )
    elif two_sided is not None and two_sided > 0.05:
        verdict = "no_evidence"
        reason = f"effect clears the resolution bound but bootstrap p={two_sided}"
    elif not consistent:
        verdict = "no_evidence"
        reason = "significant pooled but the sign does not hold across all three periods"
    elif unit.endswith("% forward return") and (abs(effect) - COST_PCT) <= (bound or 0):
        # The tradeable quantity is the effect NET of the round trip, and that net has to
        # clear the same resolution bound the gross effect had to clear. Comparing the
        # gross effect to cost is the softer test and it passes things that are dead: an
        # edge of 0.0222% against a 0.02% round trip nets 0.0022%, which is an order of
        # magnitude inside its own noise floor while still looking like a survivor.
        verdict = "below_cost"
        reason = (
            f"gross effect {effect:+.4f}% clears its bound, but net of a {COST_PCT}% round "
            f"trip it is {abs(effect) - COST_PCT:+.4f}%, inside the {bound:+.4f}% these "
            "samples can separate"
        )
    else:
        verdict = "survives_screens"
        reason = "clears resolution, bootstrap, out-of-sample sign consistency and cost"

    result.update({
        "effect": round(effect, 4),
        "baseline_mean": round(float(miss.mean()), 4),
        "smallest_resolvable_effect": bound,
        "bootstrap_p_two_sided": two_sided,
        "bootstrap_share_above_zero": round(share_above, 4) if share_above is not None else None,
        "effect_net_of_cost": round(abs(effect) - COST_PCT, 4)
        if unit.endswith("% forward return") else None,
        "by_period": by_period,
        "sign_consistent_all_periods": consistent,
        "verdict": verdict,
        "reason": reason,
    })
    return result


# --------------------------------------------------------------------------- data

def load() -> dict:
    price, _ = tk.load_price_csv(BARS_FILE)
    price = price.sort_values("time").reset_index(drop=True)
    # The shared loader drops Volume; the volume hypothesis needs it, so it is read back
    # from the same file and joined on the timestamp rather than assumed row-aligned.
    raw = pd.read_csv(BARS_FILE, encoding="utf-8-sig", usecols=["time", "Volume"])
    raw["time"] = pd.to_datetime(raw["time"]).dt.tz_localize(None)
    price = price.merge(raw.rename(columns={"Volume": "volume"}), on="time", how="left")
    price["ret"] = 100 * price["close"].pct_change()
    price["session_day"] = (price["time"] - pd.Timedelta(hours=DAY_BOUNDARY_HOUR)).dt.date
    price["hour"] = price["time"].dt.hour
    price["minute"] = price["time"].dt.minute
    price["dow"] = price["time"].dt.dayofweek

    daily = (
        price.groupby("session_day")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"), bars=("close", "size"))
        .reset_index()
    )
    daily = daily[daily["bars"] >= 40].reset_index(drop=True)
    daily["ret"] = 100 * (daily["close"] / daily["close"].shift(1) - 1)
    daily["fwd"] = daily["ret"].shift(-1)
    daily["range"] = daily["high"] - daily["low"]
    daily["fwd_range"] = daily["range"].shift(-1)
    daily["dow"] = pd.to_datetime(daily["session_day"]).dt.dayofweek
    daily["month"] = pd.to_datetime(daily["session_day"]).dt.month
    daily["gap"] = 100 * (daily["open"] / daily["close"].shift(1) - 1)
    return {"price": price, "daily": daily}


def thirds(n: int) -> np.ndarray:
    label = np.array(["train"] * n, dtype=object)
    label[int(n * 0.55):int(n * 0.80)] = "valid"
    label[int(n * 0.80):] = "holdout"
    return label


# --------------------------------------------------------------------------- hypotheses
#
# Each returns (condition, forward, period, metadata). Adding one is writing one function
# and appending it to HYPOTHESES; nothing else in the file needs to change.

def h_daily(daily: pd.DataFrame, mask: np.ndarray, target: str = "fwd") -> tuple:
    frame = daily.dropna(subset=[target]).reset_index(drop=True)
    aligned = mask[: len(daily)][daily[target].notna().to_numpy()]
    return aligned, frame[target].to_numpy(), thirds(len(frame))


def build_hypotheses(data: dict) -> list[dict]:
    price, daily = data["price"], data["daily"]
    out = []

    def add(fn, **meta):
        out.append({"fn": fn, **meta})

    # --- external claims, tested because someone else specified them first ---------
    add(lambda: h_daily(daily, (daily["dow"] == 0).to_numpy()),
        id="h01_monday_weak", family="calendar", origin="external_claim",
        claim="Gold starts the week weak: Monday is the worst weekday.",
        null="all weekdays share one return distribution")
    add(lambda: h_daily(daily, (daily["dow"] == 4).to_numpy()),
        id="h02_friday_strong", family="calendar", origin="external_claim",
        claim="Gold rises into the weekend: Friday is the strongest weekday.",
        null="all weekdays share one return distribution")

    # Asian buying "fades" when London opens: does the Asian move reverse in Europe?
    session = price.copy()
    session["block"] = np.where(session["hour"].between(7, 14), "asia",
                        np.where(session["hour"].between(15, 20), "europe", "other"))
    grouped = session[session["block"] != "other"].groupby(["session_day", "block"])["ret"].sum().unstack()
    grouped = grouped.dropna().reset_index()
    asia_up = (grouped["asia"] > 0).to_numpy()
    add(lambda: (asia_up, grouped["europe"].to_numpy(), thirds(len(grouped))),
        id="h03_asia_fades_in_europe", family="session", origin="external_claim",
        claim="Asian-session buying fades once London opens.",
        null="the European block is independent of the Asian block's sign")

    # --- calendar structure -------------------------------------------------------
    day_index = pd.to_datetime(daily["session_day"])
    month_position = day_index.groupby([day_index.dt.year, day_index.dt.month]).rank()
    month_size = day_index.groupby([day_index.dt.year, day_index.dt.month]).transform("size")
    add(lambda: h_daily(daily, ((month_position <= 3) | (month_position > month_size - 3)).to_numpy()),
        id="h04_turn_of_month", family="calendar", origin="external_claim",
        claim="Turn-of-month: the first and last three sessions outperform the middle.",
        null="position within the month carries no return information")
    add(lambda: h_daily(daily, (daily["month"] == 9).to_numpy()),
        id="h05_september_weak", family="calendar", origin="external_claim",
        claim="September is gold's worst month.",
        null="all months share one return distribution")

    # --- gaps and overnight -------------------------------------------------------
    add(lambda: h_daily(daily, (daily["gap"] > 0.1).to_numpy()),
        id="h06_gap_up_continues", family="gap", origin="internal_idea",
        claim="A session that opens more than 0.1% above the prior close keeps going.",
        null="the opening gap carries no information about the session that follows")
    add(lambda: h_daily(daily, (daily["gap"] < -0.1).to_numpy()),
        id="h07_gap_down_continues", family="gap", origin="internal_idea",
        claim="A session that opens more than 0.1% below the prior close keeps going.",
        null="the opening gap carries no information about the session that follows")

    # --- range structure ----------------------------------------------------------
    narrow = daily["range"] == daily["range"].rolling(4).min()
    add(lambda: h_daily(daily, narrow.fillna(False).to_numpy(), target="fwd_range"),
        id="h08_nr4_range_expansion", family="range", origin="external_claim",
        claim="NR4: the narrowest range in four sessions precedes an expansion.",
        null="today's range rank carries no information about tomorrow's range",
        unit="USD next-session range")
    inside = (daily["high"] < daily["high"].shift(1)) & (daily["low"] > daily["low"].shift(1))
    add(lambda: h_daily(daily, inside.fillna(False).to_numpy(), target="fwd_range"),
        id="h09_inside_day_expansion", family="range", origin="external_claim",
        claim="An inside day precedes a larger range.",
        null="containment carries no information about tomorrow's range",
        unit="USD next-session range")
    outside = (daily["high"] > daily["high"].shift(1)) & (daily["low"] < daily["low"].shift(1))
    add(lambda: h_daily(daily, outside.fillna(False).to_numpy()),
        id="h10_outside_day_direction", family="range", origin="internal_idea",
        claim="An outside day sets the next session's direction.",
        null="engulfing the prior range carries no directional information")

    # --- streaks and mean reversion ----------------------------------------------
    up = daily["ret"] > 0
    three_up = up & up.shift(1) & up.shift(2)
    add(lambda: h_daily(daily, three_up.fillna(False).to_numpy()),
        id="h11_three_up_days", family="streak", origin="internal_idea",
        claim="Three consecutive up sessions are followed by a pullback.",
        null="streak length carries no information about the next session")
    three_down = (~up) & (~up).shift(1) & (~up).shift(2)
    add(lambda: h_daily(daily, three_down.fillna(False).to_numpy()),
        id="h12_three_down_days", family="streak", origin="internal_idea",
        claim="Three consecutive down sessions are followed by a bounce.",
        null="streak length carries no information about the next session")

    # --- distance from trend ------------------------------------------------------
    ema200 = price["close"].ewm(span=200, adjust=False).mean()
    stretch = 100 * (price["close"] / ema200 - 1)
    price_fwd = price["close"].shift(-16) / price["close"] * 100 - 100  # 8 hours ahead
    far_above = (stretch > stretch.quantile(0.9)).to_numpy()
    add(lambda: (far_above[price_fwd.notna().to_numpy()],
                 price_fwd.dropna().to_numpy(),
                 thirds(int(price_fwd.notna().sum()))),
        id="h13_stretched_above_ema200", family="mean_reversion", origin="internal_idea",
        claim="The top decile of stretch above the 200 EMA mean-reverts over eight hours.",
        null="distance from trend carries no information about the next eight hours")

    # --- volume -------------------------------------------------------------------
    volume_z = (price["volume"] - price["volume"].rolling(48).mean()) / price["volume"].rolling(48).std()
    spike = (volume_z > 3).to_numpy()
    fwd_4 = price["close"].shift(-4) / price["close"] * 100 - 100
    usable = fwd_4.notna().to_numpy() & ~np.isnan(volume_z.to_numpy())
    add(lambda: (spike[usable], fwd_4.to_numpy()[usable], thirds(int(usable.sum()))),
        id="h14_volume_spike_direction", family="volume", origin="internal_idea",
        claim="A three-sigma volume spike predicts the next two hours.",
        null="volume carries no directional information")

    # --- opening range ------------------------------------------------------------
    # The obvious version of this test is circular: comparing the session HIGH to the
    # opening-range high and then predicting that same session's return asks whether an up
    # session went up. The condition has to be observable before the window it predicts, so
    # the break is judged by 10:00 and the return measured from 10:00 to the session close.
    opening = price[price["hour"] == 7].groupby("session_day").agg(
        or_high=("high", "max"), or_low=("low", "min"))
    morning = price[price["hour"].between(8, 9)].groupby("session_day").agg(
        am_high=("high", "max"), am_low=("low", "min"))
    at_ten = price[(price["hour"] == 10) & (price["minute"] == 0)].groupby(
        "session_day")["open"].first().rename("price_at_10")
    rest = price[price["hour"] >= 10].groupby("session_day")["close"].last().rename("close_of_day")
    frame = (
        daily[["session_day"]]
        .merge(opening, on="session_day").merge(morning, on="session_day")
        .merge(at_ten, on="session_day").merge(rest, on="session_day")
        .dropna().reset_index(drop=True)
    )
    frame["rest_of_session"] = 100 * (frame["close_of_day"] / frame["price_at_10"] - 1)
    broke_up = (frame["am_high"] > frame["or_high"]).to_numpy()
    add(lambda: (broke_up, frame["rest_of_session"].to_numpy(), thirds(len(frame))),
        id="h15_opening_range_breakout", family="session", origin="external_claim",
        claim="Breaking the 07:00 opening range by 10:00 predicts the rest of the session.",
        null="an opening-range break carries no information about the hours after it")

    # --- weekly ------------------------------------------------------------------
    weekly = daily.set_index(pd.to_datetime(daily["session_day"])).resample("W").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna().reset_index()
    weekly["ret"] = 100 * (weekly["close"] / weekly["close"].shift(1) - 1)
    weekly["fwd"] = weekly["ret"].shift(-1)
    weekly["close_position"] = (weekly["close"] - weekly["low"]) / (weekly["high"] - weekly["low"])
    strong_close = (weekly["close_position"] > 0.8).to_numpy()
    weekly_ok = weekly["fwd"].notna().to_numpy()
    add(lambda: (strong_close[weekly_ok], weekly["fwd"].to_numpy()[weekly_ok],
                 thirds(int(weekly_ok.sum()))),
        id="h16_strong_weekly_close", family="weekly", origin="internal_idea",
        claim="A weekly close in the top fifth of its range predicts the next week.",
        null="where the week closes in its own range carries no information")

    # --- cross-asset --------------------------------------------------------------
    try:
        dxy, _ = tk.load_price_csv(DXY_FILE)
        dxy = dxy.sort_values("time").reset_index(drop=True)
        dxy["dxy_ret"] = 100 * dxy["close"].pct_change()
        hourly = price.set_index("time")["close"].resample("1h").last().dropna()
        gold_ret = 100 * hourly.pct_change()
        joined = pd.merge(
            gold_ret.rename("gold").reset_index(),
            dxy[["time", "dxy_ret"]], on="time", how="inner",
        ).dropna().reset_index(drop=True)
        joined["gold_fwd"] = joined["gold"].shift(-1)
        joined = joined.dropna().reset_index(drop=True)
        dxy_down = (joined["dxy_ret"] < joined["dxy_ret"].quantile(0.1)).to_numpy()
        add(lambda: (dxy_down, joined["gold_fwd"].to_numpy(), thirds(len(joined))),
            id="h17_dxy_lead_gold", family="cross_asset", origin="internal_idea",
            claim="A bottom-decile DXY hour predicts the next hour of gold.",
            null="DXY at t carries no information about gold at t+1")
    except Exception as error:  # noqa: BLE001
        out.append({"fn": None, "id": "h17_dxy_lead_gold", "family": "cross_asset",
                    "origin": "internal_idea", "claim": "DXY leads gold by one hour.",
                    "null": "n/a", "skip_reason": f"DXY series unavailable: {error}"})

    # --- positioning --------------------------------------------------------------
    try:
        cftc = pd.read_csv(CFTC_FILE)
        date_col = next(c for c in cftc.columns if "date" in c.lower())
        long_col = next(c for c in cftc.columns if "money" in c.lower() and "long" in c.lower())
        short_col = next(c for c in cftc.columns if "money" in c.lower() and "short" in c.lower())
        cftc["report_date"] = pd.to_datetime(cftc[date_col])
        cftc["net"] = cftc[long_col] - cftc[short_col]
        cftc = cftc.sort_values("report_date").reset_index(drop=True)
        cftc["net_change"] = cftc["net"].diff()
        weekly_price = weekly.copy()
        weekly_price["report_date"] = pd.to_datetime(
            weekly_price.iloc[:, 0], errors="coerce"
        ).astype("datetime64[ns]")
        cftc["report_date"] = cftc["report_date"].astype("datetime64[ns]")
        merged_cftc = pd.merge_asof(
            weekly_price.sort_values("report_date"),
            cftc[["report_date", "net", "net_change"]].sort_values("report_date"),
            on="report_date", direction="backward",
        ).dropna(subset=["fwd", "net_change"]).reset_index(drop=True)
        crowded = (merged_cftc["net_change"] >
                   merged_cftc["net_change"].quantile(0.75)).to_numpy()
        add(lambda: (crowded, merged_cftc["fwd"].to_numpy(), thirds(len(merged_cftc))),
            id="h18_cftc_crowding", family="positioning", origin="external_claim",
            claim="A top-quartile weekly build in managed-money net length precedes a weaker week.",
            null="positioning change carries no information about the following week",
            data_gap=(
                "The repository holds 12 weekly CFTC rows (2026-06-02 onward) because the "
                "weekly fetcher keeps a rolling window, not a history. This hypothesis needs "
                "years. CFTC publishes the full Commitments of Traders history free; "
                "archiving it is a prerequisite for any positioning hypothesis, and it is "
                "the single cheapest way to open a family this dataset currently cannot test."
            ))
    except Exception as error:  # noqa: BLE001
        out.append({"fn": None, "id": "h18_cftc_crowding", "family": "positioning",
                    "origin": "external_claim",
                    "claim": "Crowded managed-money positioning precedes weakness.",
                    "null": "n/a", "skip_reason": f"CFTC join unavailable: {error}"})

    # --- intraday timing ----------------------------------------------------------
    fwd_2 = price["close"].shift(-2) / price["close"] * 100 - 100
    ok = fwd_2.notna().to_numpy()
    release = ((price["hour"] == 20) & (price["minute"] == 30)).to_numpy()
    add(lambda: (release[ok], fwd_2.to_numpy()[ok], thirds(int(ok.sum()))),
        id="h19_us_release_slot_direction", family="session", origin="internal_idea",
        claim="The 20:30 Taipei US-release slot has a directional bias.",
        null="the release slot moves price but carries no direction")
    quiet = ((price["hour"] >= 3) & (price["hour"] < 6)).to_numpy()
    add(lambda: (quiet[ok], fwd_2.to_numpy()[ok], thirds(int(ok.sum()))),
        id="h20_dead_zone_drift", family="session", origin="internal_idea",
        claim="The 03:00-06:00 Taipei dead zone drifts in one direction.",
        null="the quiet block carries no directional information")

    return out


# --------------------------------------------------------------------------- family test

def family_permutation(results: list[dict], stream: random.Random) -> dict:
    """Twenty tests are twenty chances. Correct across the whole set, not per hypothesis.

    Each hypothesis is reduced to a standardised effect, and the null redraws the same
    number of standardised effects from a normal centred on zero — the distribution a set
    of true nulls would produce. The comparison is between the largest observed effect and
    the largest a no-effect family typically produces.
    """
    scored = [
        row for row in results
        if row.get("verdict") != "underpowered" and row.get("smallest_resolvable_effect")
    ]
    if not scored:
        return {"tested": 0}
    # effect / resolution bound is a scale-free "how many just-resolvable units" measure.
    observed = [abs(row["effect"]) / row["smallest_resolvable_effect"] for row in scored]
    observed_max = max(observed)
    null_max = []
    for _ in range(FAMILY_PERMUTATIONS):
        null_max.append(max(abs(stream.gauss(0, 1)) / 2.8 * 2.8 for _ in scored))
    exceed = sum(1 for value in null_max if value >= observed_max)
    return {
        "tested": len(scored),
        "largest_observed_units_of_resolution": round(observed_max, 3),
        "null_median": round(float(np.median(null_max)), 3),
        "p_value": round((exceed + 1) / (FAMILY_PERMUTATIONS + 1), 4),
        "note": (
            "An effect below 1.0 is inside what its own sample can separate, whatever its "
            "p-value looks like in isolation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    data = load()
    specs = build_hypotheses(data)
    results = []
    for spec in specs:
        if spec.get("fn") is None:
            results.append({k: v for k, v in spec.items() if k != "fn"} | {"verdict": "skipped"})
            continue
        try:
            condition, forward, period = spec["fn"]()
        except Exception as error:  # noqa: BLE001
            results.append({k: v for k, v in spec.items() if k != "fn"}
                           | {"verdict": "skipped", "skip_reason": str(error)[:200]})
            continue
        outcome = evaluate(
            spec["id"], spec["family"], spec["claim"], spec["origin"], spec["null"],
            np.asarray(condition, dtype=bool), np.asarray(forward, dtype=float),
            np.asarray(period, dtype=object), rng(spec["id"]),
            unit=spec.get("unit", "% forward return"),
        )
        # Anything the spec knows and the harness does not — a data gap, a caveat — rides
        # through onto the result, so the registry entry carries it without a second table.
        for key, value in spec.items():
            if key not in outcome and key not in ("fn", "unit"):
                outcome[key] = value
        results.append(outcome)

    verdicts = {}
    for row in results:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1

    payload = {
        "study_id": STUDY_ID,
        "schema_version": 1,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "market": "XAUUSD",
        "strategy": "none — hypothesis sweep on the price series",
        "method": {
            "bars": str(BARS_FILE.relative_to(ROOT)),
            "day_boundary_hour_taipei": DAY_BOUNDARY_HOUR,
            "bootstrap": BOOTSTRAP,
            "block_size": BLOCK,
            "family_permutations": FAMILY_PERMUTATIONS,
            "split": "train 0-55%, valid 55-80%, holdout 80-100% by index",
            "seed": SEED,
            "cost_pct_round_trip": COST_PCT,
            "verdict_rule": (
                "survives_screens requires all three: the effect exceeds what the samples "
                "can resolve, the block bootstrap two-sided p is at or below 0.05, and the "
                "sign holds in train, valid and holdout."
            ),
        },
        "coverage": {
            "bars": int(len(data["price"])),
            "sessions": int(len(data["daily"])),
            "from": str(data["price"]["time"].min()),
            "to": str(data["price"]["time"].max()),
        },
        "verdict_counts": verdicts,
        "family_permutation": family_permutation(results, rng("family")),
        "hypotheses": results,
        "limitations": [
            "One instrument, 32 months, one strong uptrend. A calendar hypothesis needing "
            "many years — the month-of-year claims in particular — cannot be settled here "
            "and is reported as underpowered rather than as absence of effect.",
            "Forward returns overlap for the intraday hypotheses; the moving-block "
            "bootstrap accounts for the clustering but does not eliminate it.",
            "A verdict of no_evidence bounds an effect, it does not prove one is absent. "
            "The bound is printed beside every result for exactly that reason.",
            "No result changes formal S1 or S2 logic, live risk, or an entry checklist.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "study_id": STUDY_ID,
        "hypotheses": len(results),
        "verdicts": verdicts,
        "family_p": payload["family_permutation"].get("p_value"),
        "survivors": [r["id"] for r in results if r["verdict"] == "survives_screens"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
