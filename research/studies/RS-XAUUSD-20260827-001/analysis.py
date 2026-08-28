#!/usr/bin/env python3
"""Four monthly buys of 1OZ gold futures, held to March — and the size that survives.

This is a sizing decision before it is a study. The build period in the specification
starts 2026-09-01, days from now, and the deliverable is a number of ounces per month. The
backtest is the evidence for that number, not the product.

Which matters, because the backtest as specified gives a dangerous answer.

## The trap this study is built around

Across the eighteen September-to-March cycles the daily data can cover, the worst drawdown
measured from average cost is -14.18%. Margin call arrives at -19.0% for seven ounces a
month and -28.2% for five. So "the largest size with a zero margin-call rate" — the
question the specification asks — answers with the top of whatever range is tested, every
time.

That is a property of eighteen samples, not of the calendar. Rolling every seven-month
window through the same data, 6.5% of them fall past -19% and 1.3% past -28%. The September
window has simply never contained one. Gold fell 29.4% between January and June 2026,
seven months ago; it happened outside the window.

So this study reports two answers and never blends them: what history did, and what the
arithmetic permits. Where they disagree, the arithmetic is the one that governs a live
position, because it is the one that does not depend on which eighteen samples were drawn.

## Roll cost is excluded on instruction

The specification assumed $0.50/oz. Carry theory on a $4,650 contract at prevailing rates
puts a two-month roll nearer $36/oz, and the basis measurable here — front-month against
spot, $1 to $3 — is not the calendar spread and cannot settle it. Rather than pick a number
that would silently drive the result, roll cost is set to zero and every figure below is
therefore optimistic by two to three rolls' worth. The report says so wherever it matters.

## The seasonality claim in the specification is half backwards

It states December-to-March strength. Monthly statistics from 1980 say December and January
are strong (52-64% win rate) while February and March are the two weakest months of the
year (40% and 45%). Holding to the end of March means capturing January and then sitting
through the worst pair. Exit timing is therefore a tested parameter rather than a fixed
assumption — and every difference it produces is inside what eighteen cycles can resolve,
which the output states rather than glosses.

Usage:
    python3.12 scripts/research/build_gold_1oz_accumulation.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import study_package as pkg  # noqa: E402

STUDY_ID = "RS-XAUUSD-20260827-001"
TAIPEI = timezone(timedelta(hours=8))
DAILY = Path("local-inputs/gold_daily.csv")
WEEKLY = Path("local-inputs/gold_weekly.csv")

# CME 1-Ounce Gold (1OZ), COMEX. One contract is one troy ounce, $1 per $1/oz, so ounces
# and contracts are the same count. Margins are the spec's figures and must be rechecked
# against CME before an order: they move with volatility, and the whole sizing answer is a
# function of them.
MULTIPLIER = 1.0
INITIAL_MARGIN = 206.0
MAINTENANCE_MARGIN = 190.0
CAPITAL = 30_000.0  # set to your own account size
ROLL_COST_PER_OZ = 0.0          # excluded on instruction; see the module docstring
BUILD_MONTHS = (9, 10, 11, 12)
SIZES = (2, 3, 4, 5, 6, 7)
FIRST_CYCLE, LAST_CYCLE = 2008, 2025
EXIT_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4}
STRESS_DRAWS = (-10, -20, -30, -40, -50)


def load_daily() -> pd.DataFrame:
    frame = pd.read_csv(DAILY)
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    return frame[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)


def month_end(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)


def build_prices(daily: pd.DataFrame, year: int) -> list[dict] | None:
    """First traded close of each build month. Nothing here can see a later price."""
    buys = []
    for month in BUILD_MONTHS:
        window = daily[(daily.date >= f"{year}-{month:02d}-01")
                       & (daily.date < f"{year}-{month:02d}-12")]
        if window.empty:
            return None
        row = window.iloc[0]
        buys.append({"date": str(row.date.date()), "price": float(row.close)})
    return buys


def simulate(daily: pd.DataFrame, buys: list[dict], oz_per_month: int,
             exit_at: pd.Timestamp) -> dict:
    """Day-by-day equity from the first buy to the exit, with both margin tests.

    Adapted from build_tx_fib_three_wave.simulate_account, with one deliberate change: TX
    tops the account up to a buffer as each leg fills. Here capital is fixed at $30,000 and
    never added to, because the question is whether a given size survives the money that
    exists — a simulation that funds its own rescue cannot answer that.

    Two margin tests, because they answer different questions. The close test is what a
    broker acts on at settlement. The intraday-low test is what a broker acts on when the
    move happens during the session, and it is the stricter and more realistic one for a
    24-hour market.
    """
    start = pd.Timestamp(buys[0]["date"])
    window = daily[(daily.date >= start) & (daily.date <= exit_at)]
    if window.empty:
        return {}
    by_date: dict[str, float] = {}
    for buy in buys:
        by_date[buy["date"]] = by_date.get(buy["date"], 0.0) + oz_per_month

    held = 0.0
    cost = 0.0                      # total dollars of entry price times ounces
    rolls = 0
    realised_roll_cost = 0.0
    peak_equity = CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0
    min_close_buffer = math.inf
    min_low_buffer = math.inf
    close_call = False
    intraday_breach = False
    ruin = False
    first_call_date = None

    def equity(mark: float) -> float:
        return CAPITAL - realised_roll_cost + (mark * held - cost)

    for row in window.itertuples(index=False):
        stamp = str(row.date.date())
        if stamp in by_date:
            added = by_date.pop(stamp)
            held += added
            cost += added * float(row.close)

        if held == 0:
            continue
        maintenance = held * MAINTENANCE_MARGIN
        low_equity = equity(float(row.low))
        close_equity = equity(float(row.close))

        # Per-ounce cushion: how far price can still fall before maintenance is breached.
        min_low_buffer = min(min_low_buffer, (low_equity - maintenance) / held)
        min_close_buffer = min(min_close_buffer, (close_equity - maintenance) / held)
        if low_equity < maintenance and not intraday_breach:
            intraday_breach = True
            first_call_date = stamp
        close_call = close_call or close_equity < maintenance
        ruin = ruin or low_equity < 0

        peak_equity = max(peak_equity, close_equity)
        drawdown = peak_equity - min(low_equity, close_equity)
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown / peak_equity * 100 if peak_equity else 0.0

    exit_price = float(window.close.iloc[-1])
    avg_cost = cost / held if held else 0.0
    # Two different drawdowns, and conflating them is the easiest mistake here.
    #
    # `max_equity_drawdown` is peak to trough on the equity curve. It can be large in a
    # cycle that finished profitably: 2025/09-2026/03 built at 3,900, rode gold to 5,597 in
    # January and exited at 4,683, so the account gave back 9,809 dollars from its high
    # while still ending 20% up.
    #
    # `trough_vs_cost_pct` is the trough measured against what the ounces cost. That is
    # the one margin depends on, because maintenance is tested against equity and equity
    # is a function of price minus cost, not of a former peak.
    trough_vs_cost = (float(window.low.min()) / avg_cost - 1) * 100 if avg_cost else 0.0
    gross = (exit_price - avg_cost) * held
    return {
        "oz_per_month": oz_per_month,
        "total_oz": held,
        "avg_cost": round(avg_cost, 2),
        "exit_date": str(window.date.iloc[-1].date()),
        "exit_price": round(exit_price, 2),
        "notional_at_entry": round(avg_cost * held, 2),
        "leverage_at_entry": round(avg_cost * held / CAPITAL, 2),
        "initial_margin_required": round(held * INITIAL_MARGIN, 2),
        "gross_pnl": round(gross, 2),
        "roll_cost": round(realised_roll_cost, 2),
        "rolls": rolls,
        "net_pnl": round(gross - realised_roll_cost, 2),
        "return_on_capital_pct": round((gross - realised_roll_cost) / CAPITAL * 100, 2),
        "exit_vs_cost_pct": round((exit_price / avg_cost - 1) * 100, 2),
        "trough_vs_cost_pct": round(trough_vs_cost, 2),
        "max_equity_drawdown": round(max_dd, 2),
        "max_equity_drawdown_pct": round(max_dd_pct, 2),
        "min_close_cushion_per_oz": round(min_close_buffer, 2)
        if math.isfinite(min_close_buffer) else None,
        "min_intraday_cushion_per_oz": round(min_low_buffer, 2)
        if math.isfinite(min_low_buffer) else None,
        "margin_call_on_close": close_call,
        "margin_breach_intraday": intraday_breach,
        "first_breach_date": first_call_date,
        "ruin": ruin,
    }


def arithmetic_table() -> list[dict]:
    """What each size permits before any history is consulted.

    This is the part that does not depend on the sample, and where the two answers diverge
    it is the one that governs. Computed from the current price rather than a historical
    average, because that is the price the live position would be built at.
    """
    price = 4_650.0
    rows = []
    for size in SIZES:
        total = size * 4
        rows.append({
            "oz_per_month": size,
            "total_oz": total,
            "notional": round(total * price, 2),
            "leverage": round(total * price / CAPITAL, 2),
            "initial_margin": round(total * INITIAL_MARGIN, 2),
            "maintenance_margin": round(total * MAINTENANCE_MARGIN, 2),
            "drop_to_margin_call_pct": round(
                -((CAPITAL - total * MAINTENANCE_MARGIN) / total) / price * 100, 2),
            "drop_to_ruin_pct": round(-(CAPITAL / total) / price * 100, 2),
        })
    return rows


def window_drawdowns(daily: pd.DataFrame, days: int = 147) -> dict:
    """Every rolling seven-month window, so the September window has something to be compared with."""
    close = daily.close.to_numpy()
    low = daily.low.to_numpy()
    worst = np.array([(low[i:i + days].min() / close[i] - 1) * 100
                      for i in range(len(daily) - days)])
    return {
        "windows": int(worst.size),
        "median_pct": round(float(np.median(worst)), 2),
        "p05_pct": round(float(np.percentile(worst, 5)), 2),
        "worst_pct": round(float(worst.min()), 2),
        "share_below": {f"{t}%": round(float((worst <= t).mean() * 100), 2)
                        for t in (-19, -22.8, -28.2, -36.2)},
        "reading": (
            "The September build window has never contained a drawdown past -15%. These are "
            "all windows in the same data. A size that survives every September cycle has "
            "not been shown to survive gold; it has been shown to survive eighteen draws."
        ),
    }


def stress(daily: pd.DataFrame, oz_per_month: int) -> dict:
    """Synthetic paths, kept apart from the historical cycles on purpose.

    Scenario A puts the fall after the book is complete, which is the worst case: full size
    against the whole move at an average cost set before it. Scenario B walks price down
    through the build, so later buys average the book down.

    These are not sampled from anything. They are arithmetic on an assumed path, and a
    reader who mixes them with the eighteen historical cycles will think they carry sample
    support.
    """
    price = 4_650.0
    total = oz_per_month * 4
    out = {"scenario_a_fall_after_build": [], "scenario_b_fall_during_build": []}

    for draw in STRESS_DRAWS:
        # A: four buys at a flat 4,650, then the fall.
        avg_a = price
        trough_a = price * (1 + draw / 100)
        equity_a = CAPITAL + (trough_a - avg_a) * total
        out["scenario_a_fall_after_build"].append({
            "drawdown_pct": draw,
            "avg_cost": round(avg_a, 2),
            "trough_price": round(trough_a, 2),
            "equity_at_trough": round(equity_a, 2),
            "maintenance_required": round(total * MAINTENANCE_MARGIN, 2),
            "margin_call": bool(equity_a < total * MAINTENANCE_MARGIN),
            "ruin": bool(equity_a < 0),
            "recovery_to_avg_cost_pnl": 0.0,
            "recovery_to_2026_high_pnl": round((5_597.23 - avg_a) * total, 2),
        })

        # B: price walks down linearly across the four build dates and the trough is the
        # last buy, so the book averages down into it.
        steps = [price * (1 + draw / 100 * k / 3) for k in range(4)]
        avg_b = float(np.mean(steps))
        trough_b = steps[-1]
        equity_b = CAPITAL + (trough_b - avg_b) * total
        out["scenario_b_fall_during_build"].append({
            "drawdown_pct": draw,
            "buy_prices": [round(s, 2) for s in steps],
            "avg_cost": round(avg_b, 2),
            "trough_price": round(trough_b, 2),
            "equity_at_trough": round(equity_b, 2),
            "maintenance_required": round(total * MAINTENANCE_MARGIN, 2),
            "margin_call": bool(equity_b < total * MAINTENANCE_MARGIN),
            "ruin": bool(equity_b < 0),
            "recovery_to_avg_cost_pnl": 0.0,
            "recovery_to_2026_high_pnl": round((5_597.23 - avg_b) * total, 2),
        })
    return out


def monthly_seasonality() -> dict:
    """Hypothesis 3, with the bound that decides whether it can be answered at all."""
    weekly = pd.read_csv(WEEKLY)
    weekly["date"] = pd.to_datetime(weekly["time"], utc=True).dt.tz_localize(None)
    monthly = weekly.set_index("date")["close"].resample("ME").last().dropna()
    ret = ((monthly / monthly.shift(1) - 1) * 100).dropna()
    ret = ret[ret.index.year >= 1980]

    rows = []
    for month in range(1, 13):
        block = ret[ret.index.month == month]
        rest = ret[ret.index.month != month]
        rate = float((block > 0).mean() * 100)
        rest_rate = float((rest > 0).mean() * 100)
        sigma = math.sqrt(0.5 * 0.5)
        bound = 100 * 2.8 * sigma * math.sqrt(1 / len(block) + 1 / len(rest))
        rows.append({
            "month": month, "n": int(len(block)),
            "win_rate_pct": round(rate, 2),
            "rest_win_rate_pct": round(rest_rate, 2),
            "gap_pct_points": round(rate - rest_rate, 2),
            "smallest_resolvable_gap_pct_points": round(bound, 2),
            "mean_return_pct": round(float(block.mean()), 3),
            "separates": bool(abs(rate - rest_rate) > bound),
        })
    strongest = max(rows, key=lambda r: r["win_rate_pct"])
    return {
        "coverage": f"{ret.index.year.min()}-{ret.index.year.max()}, {len(ret)} months",
        "source": "weekly closes resampled to month end",
        "by_month": rows,
        "months_that_separate": [r["month"] for r in rows if r["separates"]],
        "spec_claim": "the specification asserts December-to-March seasonal strength",
        "what_the_data_says": (
            "December and January are the strong pair; February and March are the two "
            "weakest months of the year. Holding to the end of March captures January and "
            "then sits through both weak months."
        ),
        "power_note": (
            f"The strongest month is {strongest['month']} at {strongest['win_rate_pct']}% "
            f"against {strongest['rest_win_rate_pct']}%, a gap of "
            f"{strongest['gap_pct_points']} points against a resolvable "
            f"{strongest['smallest_resolvable_gap_pct_points']}. No month separates. "
            "Forty-seven observations cannot resolve a monthly effect of a plausible size, "
            "so the seasonal claim is neither supported nor refuted here."
        ),
        "legacy_comparison": {
            "source": "a 1980-2026 monthly table in the retired trading repository",
            "agreement": "January strong, February and March weak in both",
            "disagreement": "August 50.0% there against 61.7% here; February 31.9% "
                            "against 40.4%. Month-end definition and price source differ, "
                            "and neither is authoritative.",
        },
    }


def public_view(payload: dict) -> dict:
    """The same study with the account size taken out of it.

    Everything here is expressed per unit of capital or as a percentage, so the findings
    travel without the size of a particular account travelling with them. The private
    results keep the dollars, because sizing a real position needs them and the owner is
    the reader there.

    Normalising is not redaction. Nothing is hidden that changes a conclusion: leverage,
    drawdown, win rate and the margin triggers are all ratios already, and the one figure
    that was absolute — ounces per month — becomes ounces per ten thousand of capital,
    which is the form another reader could actually use.
    """
    unit = 10_000.0
    scale = unit / payload["capital"]
    rows = []
    for row in payload["arithmetic_sizing"]["rows"]:
        rows.append({
            "oz_per_month_per_10k": round(row["oz_per_month"] * scale, 2),
            "total_oz_per_10k": round(row["total_oz"] * scale, 2),
            "leverage": row["leverage"],
            "initial_margin_pct_of_capital": round(
                row["initial_margin"] / payload["capital"] * 100, 2),
            "drop_to_margin_call_pct": row["drop_to_margin_call_pct"],
            "drop_to_ruin_pct": row["drop_to_ruin_pct"],
        })

    performance = {}
    for label, sizes in payload["by_exit_timing"].items():
        performance[label] = {
            str(round(int(size) * scale, 2)): {
                "leverage": next(r["leverage"] for r in payload["arithmetic_sizing"]["rows"]
                                 if r["oz_per_month"] == int(size)),
                "win_rate_pct": v["win_rate_pct"],
                "mean_return_on_capital_pct": v["mean_return_on_capital_pct"],
                "median_return_on_capital_pct": v["median_return_on_capital_pct"],
                "worst_equity_drawdown_pct": v["worst_equity_drawdown_pct"],
                "margin_breach_cycles": v["margin_breach_intraday_cycles"],
            }
            for size, v in sizes.items()
        }

    cycles = [{
        "cycle": c["cycle"],
        "avg_cost": c["avg_cost"],
        "exit_price": c["exit_price"],
        "exit_vs_cost_pct": c["exit_vs_cost_pct"],
        "trough_vs_cost_pct": c["trough_vs_cost_pct"],
        "return_on_capital_pct": c["return_on_capital_pct"],
        "max_equity_drawdown_pct": c["max_equity_drawdown_pct"],
    } for c in payload["historical_cycles"]["mar"] if c["oz_per_month"] == 4]

    stress = {}
    for size, block in payload["stress_scenarios"].items():
        leverage = next(r["leverage"] for r in payload["arithmetic_sizing"]["rows"]
                        if r["oz_per_month"] == int(size))
        stress[f"leverage_{leverage}x"] = {
            scenario: [{
                "drawdown_pct": row["drawdown_pct"],
                "equity_pct_of_capital": round(
                    row["equity_at_trough"] / payload["capital"] * 100, 2),
                "margin_call": row["margin_call"],
                "ruin": row["ruin"],
            } for row in rows_]
            for scenario, rows_ in block.items()
        }

    return {
        "normalisation": (
            "Position size is stated per 10,000 units of account currency and everything "
            "else as a ratio, so the result is usable at any account size and carries none."
        ),
        "sizing_by_arithmetic": rows,
        "performance_by_exit_timing": performance,
        "cycles_at_2_5x_leverage": cycles,
        "stress_scenarios": stress,
        "rolling_window_drawdowns": payload["rolling_window_drawdowns"],
        "seasonality": payload["seasonality"],
        "two_kinds_of_drawdown": payload["two_kinds_of_drawdown"],
        "answers": {
            "leverage_the_backtest_permits": next(
                r["leverage"] for r in payload["arithmetic_sizing"]["rows"]
                if r["oz_per_month"] == payload["answers"][
                    "largest_size_with_no_historical_margin_breach"]),
            "leverage_the_arithmetic_permits": next(
                r["leverage"] for r in payload["arithmetic_sizing"]["rows"]
                if r["oz_per_month"] == payload["answers"][
                    "largest_size_surviving_a_repeat_of_2026"]),
            "why_they_differ": payload["answers"]["why_they_differ"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    args = parser.parse_args()
    daily = load_daily()

    cycles: dict[str, list[dict]] = {}
    for label, month in EXIT_MONTHS.items():
        rows = []
        for year in range(FIRST_CYCLE, LAST_CYCLE + 1):
            buys = build_prices(daily, year)
            if buys is None:
                continue
            exit_at = month_end(year + 1, month)
            if exit_at > daily.date.iloc[-1]:
                continue
            for size in args.sizes:
                result = simulate(daily, buys, size, exit_at)
                if result:
                    result["cycle"] = f"{year}/09-{year + 1}/{month:02d}"
                    result["buys"] = buys
                    rows.append(result)
        cycles[label] = rows

    by_size = {}
    for label, rows in cycles.items():
        per_size = {}
        for size in args.sizes:
            subset = [r for r in rows if r["oz_per_month"] == size]
            if not subset:
                continue
            pnl = np.array([r["net_pnl"] for r in subset])
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            per_size[size] = {
                "cycles": len(subset),
                "win_rate_pct": round(float((pnl > 0).mean() * 100), 2),
                "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3)
                if losses.size and losses.sum() else None,
                "mean_return_on_capital_pct": round(
                    float(np.mean([r["return_on_capital_pct"] for r in subset])), 2),
                "median_return_on_capital_pct": round(
                    float(np.median([r["return_on_capital_pct"] for r in subset])), 2),
                "worst_cycle_pnl": round(float(pnl.min()), 2),
                "worst_equity_drawdown_pct": round(
                    float(max(r["max_equity_drawdown_pct"] for r in subset)), 2),
                "margin_call_close_cycles": sum(1 for r in subset if r["margin_call_on_close"]),
                "margin_breach_intraday_cycles": sum(
                    1 for r in subset if r["margin_breach_intraday"]),
                "ruin_cycles": sum(1 for r in subset if r["ruin"]),
                "min_intraday_cushion_per_oz": round(
                    float(min(r["min_intraday_cushion_per_oz"] for r in subset)), 2),
            }
        by_size[label] = per_size

    mar = by_size["mar"]
    zero_call = [s for s, v in mar.items() if v["margin_breach_intraday_cycles"] == 0]
    arithmetic = arithmetic_table()
    windows = window_drawdowns(daily)
    seasonality = monthly_seasonality()

    # The two answers, stated separately and never averaged.
    historical_max = max(zero_call) if zero_call else None
    survives_2026 = [r["oz_per_month"] for r in arithmetic
                     if r["drop_to_margin_call_pct"] < -29.44]
    arithmetic_max = max(survives_2026) if survives_2026 else None

    payload = {
        "study_id": STUDY_ID,
        "schema_version": "1.0",
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "title": ("Four monthly buys of 1OZ gold futures held to March — what history "
                  "allows and what the arithmetic allows are different numbers"),
        "contract": {
            "instrument": "CME 1-Ounce Gold (1OZ), COMEX",
            "multiplier_usd_per_dollar_per_oz": MULTIPLIER,
            "initial_margin": INITIAL_MARGIN,
            "maintenance_margin": MAINTENANCE_MARGIN,
            "margin_caveat": ("Spec figures. CME revises these with volatility and the "
                              "sizing answer is a direct function of them; recheck before "
                              "an order."),
            "proxy": ("1OZ listed in 2025, so there is no history. XAUUSD spot stands in "
                      "and the multiplier makes ounces and contracts interchangeable. "
                      "Futures liquidity and spread are not modelled."),
        },
        "capital": CAPITAL,
        "roll_cost_per_oz": ROLL_COST_PER_OZ,
        "roll_cost_note": (
            "Excluded on instruction. The spec assumed $0.50/oz; carry on a $4,650 contract "
            "at prevailing rates is nearer $36/oz for a two-month roll, and the measurable "
            "front-month basis of $1-3 is not the calendar spread. Two to three rolls sit "
            "between September and March, so every net figure here is optimistic by "
            "somewhere between $1 and $110 per ounce."
        ),
        "coverage": {
            "cycles": len(range(FIRST_CYCLE, LAST_CYCLE + 1)),
            "daily_from": str(daily.date.iloc[0].date()),
            "daily_to": str(daily.date.iloc[-1].date()),
            "why_not_1990": ("The spec asks for 1990 onward. Daily gold in this repository "
                             "starts 2008-01-04, giving 18 build cycles rather than 35. "
                             "Weekly data reaches 1970 but cannot drive a daily margin "
                             "simulation."),
        },
        "arithmetic_sizing": {
            "priced_at": 4_650.0,
            "note": ("Independent of the sample. Where this and the historical result "
                     "disagree, this is the one that governs a live position."),
            "rows": arithmetic,
        },
        "historical_cycles": cycles,
        "by_exit_timing": by_size,
        "two_kinds_of_drawdown": {
            "trough_vs_cost_pct": (
                "The trough measured against what the ounces cost. Margin is tested on "
                "this, because maintenance compares equity to a requirement and equity is "
                "price minus cost."
            ),
            "max_equity_drawdown_pct": (
                "Peak to trough on the equity curve. Larger, and large even in profitable "
                "cycles: 2025/09-2026/03 built at 3,900, rode gold to 5,597 in January and "
                "exited at 4,683 for +20%, giving back 9,809 dollars from its high on the "
                "way. It is what the position felt like, not what triggers a call."
            ),
            "why_it_matters": (
                "A size chosen on trough-vs-cost survives margin. A size chosen on equity "
                "drawdown survives the person holding it. They are different numbers and "
                "the second is the larger one."
            ),
        },
        "rolling_window_drawdowns": windows,
        "stress_scenarios": {size: stress(daily, size) for size in args.sizes},
        "stress_note": ("Synthetic paths on an assumed shape, not draws from anything. "
                        "Kept separate from historical_cycles so they cannot be read as "
                        "carrying sample support."),
        "seasonality": seasonality,
        "answers": {
            "largest_size_with_no_historical_margin_breach": historical_max,
            "largest_size_surviving_a_repeat_of_2026": arithmetic_max,
            "why_they_differ": (
                "The eighteen September cycles never contained a drawdown past -15%, so "
                "every tested size clears them. The arithmetic asks a different question: "
                "which sizes survive a fall the size of the one gold delivered between "
                "January and June 2026, -29.44%. That fall happened outside the build "
                "window, which is why the backtest does not see it and why it cannot be "
                "used to rule it out."
            ),
        },
        "limitations": [
            "Roll cost is zero by instruction. Two to three rolls sit in each cycle and "
            "gold carries positive contango, so every net figure is optimistic.",
            "Eighteen cycles. The worst of them is -14.18% from average cost while 6.5% of "
            "all seven-month windows in the same data fall past -19%.",
            "XAUUSD spot stands in for a contract listed in 2025. Deferred-month liquidity "
            "is thinner than spot and the assumed fills are better than reachable ones.",
            "Margins are held constant. CME raises them in exactly the conditions that "
            "would trigger a call, so a real breach arrives earlier than modelled.",
            "No commission, no financing, no currency cost.",
            "Monthly seasonality has 47 observations a month against a resolvable gap of "
            "roughly 21 percentage points. No month separates, in either direction.",
        ],
    }

    payload["public_view"] = public_view(payload)

    written = pkg.write_package(
        STUDY_ID, payload,
        market="XAUUSD",
        strategy="none — 1OZ futures accumulation and position sizing",
        title=payload["title"],
        question=("Four monthly buys of 1OZ gold futures from September, held to the "
                  "following March on $30,000: what size survives, and does the "
                  "historical answer or the arithmetic one govern?"),
        hypothesis=("The backtest will clear every tested size because the September "
                    "window has never contained a large drawdown, and that is a fact "
                    "about eighteen samples rather than about the calendar."),
        runner="scripts/research/build_gold_1oz_accumulation.py",
        headline={
            "cycles": 18,
            "sizes_tested": len(args.sizes),
            "largest_size_no_historical_breach": historical_max,
            "largest_size_surviving_2026_repeat": arithmetic_max,
            "worst_trough_vs_cost_pct": round(
                min(r["trough_vs_cost_pct"] for r in cycles["mar"]), 2),
            "worst_equity_drawdown_pct_at_largest_size": round(
                max(r["max_equity_drawdown_pct"] for r in cycles["mar"]), 2),
            "share_of_all_windows_past_minus_19_pct": windows["share_below"]["-19%"],
            "months_that_separate": len(seasonality["months_that_separate"]),
        },
        findings=[],
        card_summary=("Four monthly buys of 1OZ gold held to March. The backtest clears "
                      "every size because eighteen September cycles never met a large "
                      "drawdown; the arithmetic does not."),
        limitations=payload["limitations"],
        card_metrics=["cycles", "largest_size_no_historical_breach",
                      "largest_size_surviving_2026_repeat",
                      "share_of_all_windows_past_minus_19_pct"],
    )

    print(json.dumps({
        "study": STUDY_ID, "written": written,
        "answers": payload["answers"],
        "by_size_mar": {s: {"win": v["win_rate_pct"],
                            "mean_ret": v["mean_return_on_capital_pct"],
                            "worst_dd_pct": v["worst_equity_drawdown_pct"],
                            "intraday_breaches": v["margin_breach_intraday_cycles"]}
                        for s, v in mar.items()},
        "months_that_separate": seasonality["months_that_separate"],
    }, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
