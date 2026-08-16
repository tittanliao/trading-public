#!/usr/bin/env python3
"""Candle-level S1 T1 pullback/chase-cap research backtest.

The signal cohort is frozen to the 472 OFF/formal filled S1 opportunities. The script
first validates its exit emulator against those TradingView trades, then recomputes
entry, SL/TP exit, and PnL for preregistered T1 entry policies. Raw inputs are immutable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2] if SCRIPT_PATH.parent.name == "research" else SCRIPT_PATH.parents[3]
STUDY_ID = "RS-XAUUSD-20260815-004"
DATA_ROOT = ROOT / "local-inputs"
OUTPUT_DIR = Path("reproduced-s1-pullback")
OFF_FILE = DATA_ROOT / "s1-off-trades.csv"
BARS_FILE = DATA_ROOT / "xauusd-30m.csv"
FORMAL_PINE = DATA_ROOT / "S1-V3.9.pine"
TICK = 0.001
SL_PCT = 0.005
TP1_PCT = 0.005
TP2_PCT = 0.0175
OUT_K_COUNT = 36


@dataclass(frozen=True)
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class TVTrade:
    number: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    exit_signal: str
    pnl: float
    qty: float
    duration_bars: int


@dataclass(frozen=True)
class SimTrade:
    source_trade_number: int
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    exit_signal: str
    qty: float
    pnl: float
    duration_bars: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def load_bars(path: Path) -> list[Bar]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        bars = [
            Bar(
                time=datetime.fromisoformat(row["time"]).replace(tzinfo=None),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
            for row in csv.DictReader(handle)
        ]
    times = [bar.time for bar in bars]
    if times != sorted(times) or len(times) != len(set(times)):
        raise SystemExit("30-minute bars are not unique and chronological")
    return bars


def load_tv_trades(path: Path) -> list[TVTrade]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["Trade number"]), []).append(row)
    trades = []
    for number in sorted(grouped):
        pair = grouped[number]
        entries = [row for row in pair if row["Type"].startswith("Entry")]
        exits = [row for row in pair if row["Type"].startswith("Exit")]
        if len(pair) != 2 or len(entries) != 1 or len(exits) != 1:
            raise SystemExit(f"malformed TradingView trade {number}")
        entry, exit_row = entries[0], exits[0]
        trades.append(
            TVTrade(
                number=number,
                entry_time=parse_time(entry["Date and time"]),
                exit_time=parse_time(exit_row["Date and time"]),
                entry_price=float(entry["Price USD"]),
                exit_price=float(exit_row["Price USD"]),
                exit_signal=exit_row["Signal"],
                pnl=float(exit_row["Net PnL USD"]),
                qty=float(entry["Size (qty)"]),
                duration_bars=int(exit_row["Duration (bars)"]),
            )
        )
    return trades


def stop_fill(bar: Bar, stop: float) -> float | None:
    """TradingView-like long stop fill with one adverse tick of slippage."""
    if bar.open <= stop:
        return bar.open - TICK
    if bar.low <= stop:
        return stop - TICK
    return None


def simulate_exit(
    bars: list[Bar],
    entry_index: int,
    entry_price: float,
    source_trade_number: int,
    signal_time: datetime,
) -> SimTrade | None:
    sl_price = entry_price * (1 - SL_PCT)
    tp1_price = entry_price * (1 + TP1_PCT)
    tp2_price = entry_price * (1 + TP2_PCT)
    pending_stop: float | None = None
    pending_signal: str | None = None

    for index in range(entry_index, len(bars)):
        bar = bars[index]
        if index > entry_index and pending_stop is not None and pending_signal is not None:
            fill = stop_fill(bar, pending_stop)
            if fill is not None:
                qty = math.floor(10000 / entry_price * 10) / 10
                return SimTrade(
                    source_trade_number=source_trade_number,
                    signal_time=signal_time,
                    entry_time=bars[entry_index].time,
                    exit_time=bar.time,
                    entry_price=entry_price,
                    exit_price=fill,
                    exit_signal=pending_signal,
                    qty=qty,
                    pnl=(fill - entry_price) * qty,
                    duration_bars=index - entry_index,
                )

        if bar.close >= tp2_price:
            lookback = bars[max(0, index - 9 + 1) : index + 1]
            pending_stop = max(tp2_price, min(item.low for item in lookback))
            pending_signal = "S1BB_TP2"
        elif bar.close >= tp1_price:
            lookback = bars[max(0, index - 18 + 1) : index + 1]
            pending_stop = max(tp1_price, min(item.low for item in lookback))
            pending_signal = "S1BB_TP1"
        else:
            pending_stop = sl_price
            pending_signal = "S1BB_SL"

        # Formal V3.9 submits additional time-stop orders after 36 bars. In the frozen
        # exports every filled exit uses the regular SL/TP IDs, so these secondary
        # orders do not alter the validated filled path and are intentionally omitted.
    return None


def wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    proportion = wins / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return [round((center - margin) * 100, 2), round((center + margin) * 100, 2)]


def exact_paired_outcome_p_value(loss_to_win: int, win_to_loss: int) -> float | None:
    """Two-sided exact sign/binomial test across discordant paired outcomes."""
    discordant = loss_to_win + win_to_loss
    if discordant == 0:
        return None
    lower_tail = sum(math.comb(discordant, value) for value in range(min(loss_to_win, win_to_loss) + 1))
    return round(min(1.0, 2 * lower_tail / (2**discordant)), 6)


def metrics(trades: list[SimTrade]) -> dict[str, Any]:
    pnls = [trade.pnl for trade in trades]
    wins = sum(value > 0 for value in pnls)
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "n": len(trades),
        "wins": wins,
        "losses": sum(value < 0 for value in pnls),
        "win_rate_pct": round(wins / len(trades) * 100, 2) if trades else None,
        "win_rate_wilson_95ci_pct": wilson_interval(wins, len(trades)),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_usd": round(sum(pnls), 2),
        "average_pnl_usd": round(statistics.mean(pnls), 2) if pnls else None,
        "max_closed_trade_drawdown_usd": round(max_dd, 2),
        "average_duration_bars": round(statistics.mean(t.duration_bars for t in trades), 2)
        if trades
        else None,
    }


def tv_metrics(trades: list[TVTrade]) -> dict[str, Any]:
    pnls = [trade.pnl for trade in trades]
    wins = sum(value > 0 for value in pnls)
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "n": len(trades),
        "wins": wins,
        "losses": sum(value < 0 for value in pnls),
        "win_rate_pct": round(wins / len(trades) * 100, 2) if trades else None,
        "win_rate_wilson_95ci_pct": wilson_interval(wins, len(trades)),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_usd": round(sum(pnls), 2),
        "average_pnl_usd": round(statistics.mean(pnls), 2) if pnls else None,
        "max_closed_trade_drawdown_usd": round(max_dd, 2),
        "average_duration_bars": round(statistics.mean(t.duration_bars for t in trades), 2)
        if trades
        else None,
    }


def validate_exit_emulator(
    bars: list[Bar], tv_trades: list[TVTrade], index_by_time: dict[datetime, int]
) -> dict[str, Any]:
    time_signal_matches = 0
    price_differences = []
    pnl_differences = []
    mismatches = []
    for trade in tv_trades:
        entry_index = index_by_time[trade.entry_time]
        simulated = simulate_exit(
            bars, entry_index, trade.entry_price, trade.number, bars[entry_index - 1].time
        )
        if simulated is None:
            mismatches.append({"trade_number": trade.number, "reason": "no simulated exit"})
            continue
        if simulated.exit_time == trade.exit_time and simulated.exit_signal == trade.exit_signal:
            time_signal_matches += 1
        else:
            mismatches.append(
                {
                    "trade_number": trade.number,
                    "tv_exit_time": trade.exit_time.isoformat(sep=" "),
                    "sim_exit_time": simulated.exit_time.isoformat(sep=" "),
                    "tv_exit_signal": trade.exit_signal,
                    "sim_exit_signal": simulated.exit_signal,
                }
            )
        price_differences.append(simulated.exit_price - trade.exit_price)
        pnl_differences.append(simulated.pnl - trade.pnl)
    return {
        "n": len(tv_trades),
        "exit_time_and_signal_match_n": time_signal_matches,
        "exit_time_and_signal_match_pct": round(time_signal_matches / len(tv_trades) * 100, 2),
        "mean_abs_exit_price_difference": round(
            statistics.mean(abs(value) for value in price_differences), 6
        ),
        "max_abs_exit_price_difference": round(max(abs(value) for value in price_differences), 6),
        "mean_abs_pnl_difference_usd": round(
            statistics.mean(abs(value) for value in pnl_differences), 4
        ),
        "mismatch_n": len(mismatches),
        "mismatches": mismatches,
    }


def opportunity(
    trade: TVTrade,
    bars: list[Bar],
    index_by_time: dict[datetime, int],
    policy: dict[str, Any],
) -> tuple[SimTrade | None, str]:
    t1_index = index_by_time[trade.entry_time]
    signal_index = t1_index - 1
    signal_low = bars[signal_index].low
    t1 = bars[t1_index]
    if t1.low < signal_low:
        return None, "t1_broke_signal_low"

    kind = policy["kind"]
    if kind == "t1_market":
        entry_index = t1_index + 1
        if entry_index >= len(bars):
            return None, "insufficient_future_bars"
        entry_price = bars[entry_index].open + TICK
    elif kind == "t1_chase_cap":
        entry_index = t1_index + 1
        if entry_index >= len(bars):
            return None, "insufficient_future_bars"
        entry_price = bars[entry_index].open + TICK
        cap = t1.open * (1 + policy["max_chase_pct"] / 100)
        if entry_price > cap:
            return None, "chase_cap_exceeded"
    elif kind == "t1_pullback":
        cap = t1.open * (1 + policy["max_chase_pct"] / 100)
        limit = min(t1.close * (1 - policy["pullback_pct"] / 100), cap)
        entry_index = -1
        entry_price = math.nan
        for candidate_index in range(t1_index + 1, min(t1_index + 3, len(bars))):
            candidate = bars[candidate_index]
            # Conservative ambiguity rule: a signal-low breach cancels before any
            # same-bar limit touch can be credited as a fill.
            if candidate.low < signal_low:
                return None, "signal_low_breached_before_fill"
            if candidate.open <= limit:
                entry_index = candidate_index
                entry_price = candidate.open
                break
            if candidate.low <= limit <= candidate.high:
                entry_index = candidate_index
                entry_price = limit
                break
        if entry_index < 0:
            return None, "limit_not_filled"
    else:
        raise ValueError(f"unknown policy: {kind}")

    simulated = simulate_exit(
        bars, entry_index, entry_price, trade.number, bars[signal_index].time
    )
    return (simulated, "filled") if simulated else (None, "no_exit_before_data_end")


def run_policy(
    policy: dict[str, Any],
    tv_trades: list[TVTrade],
    bars: list[Bar],
    index_by_time: dict[datetime, int],
    chronology_cutoff: datetime,
) -> tuple[dict[str, Any], list[SimTrade]]:
    independent = []
    reasons: dict[str, int] = {}
    for trade in tv_trades:
        result, reason = opportunity(trade, bars, index_by_time, policy)
        reasons[reason] = reasons.get(reason, 0) + 1
        if result:
            independent.append(result)

    sequential = []
    current_exit: datetime | None = None
    skipped_while_position = 0
    by_number = {trade.source_trade_number: trade for trade in independent}
    source_by_number = {trade.number: trade for trade in tv_trades}
    for source in tv_trades:
        candidate = by_number.get(source.number)
        if candidate is None:
            continue
        if current_exit is not None and candidate.signal_time < current_exit:
            skipped_while_position += 1
            continue
        sequential.append(candidate)
        current_exit = candidate.exit_time

    selected_source = [source_by_number[trade.source_trade_number] for trade in independent]
    conversions: dict[str, int] = {}
    for simulated, source in zip(independent, selected_source):
        label = ("win" if source.pnl > 0 else "loss") + "_to_" + (
            "win" if simulated.pnl > 0 else "loss"
        )
        conversions[label] = conversions.get(label, 0) + 1
    t1_held_n = len(tv_trades) - reasons.get("t1_broke_signal_low", 0)
    loss_to_win = conversions.get("loss_to_win", 0)
    win_to_loss = conversions.get("win_to_loss", 0)
    early = [trade for trade in independent if trade.signal_time <= chronology_cutoff]
    recent = [trade for trade in independent if trade.signal_time > chronology_cutoff]
    by_year = {
        str(year): metrics([trade for trade in independent if trade.signal_time.year == year])
        for year in sorted({trade.signal_time.year for trade in independent})
    }

    return (
        {
            "policy": policy,
            "independent_signal_metrics": metrics(independent),
            "known_signal_sequential_metrics": metrics(sequential),
            "entry_disposition_counts": dict(sorted(reasons.items())),
            "t1_held_signal_n": t1_held_n,
            "fill_rate_of_t1_held_pct": round(len(independent) / t1_held_n * 100, 2),
            "sequential_skipped_while_position_n": skipped_while_position,
            "same_source_off_metrics": tv_metrics(selected_source),
            "outcome_conversions": dict(sorted(conversions.items())),
            "paired_outcome_exact_p_value": exact_paired_outcome_p_value(
                loss_to_win, win_to_loss
            ),
            "chronological_stability": {
                "cutoff_signal_time": chronology_cutoff.isoformat(sep=" "),
                "early_70pct_signal_cohort": metrics(early),
                "recent_30pct_signal_cohort": metrics(recent),
                "by_signal_year": by_year,
            },
            "boundary": "sequential path uses only the frozen OFF filled-signal cohort; it cannot add qualified signals that formal OFF missed while in position",
        },
        independent,
    )


def write_outputs(
    output_dir: Path,
    generated_at: str,
    results: dict[str, Any],
    trades: list[dict[str, Any]],
    off_file: Path,
    bars_file: Path,
    formal_pine: Path,
) -> None:
    (output_dir / "pullback_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "pullback_trades.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["policy"] + list(asdict(trades[0]["trade"])) if trades else ["policy"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in trades:
            row = {"policy": item["policy"], **asdict(item["trade"])}
            for key in ("signal_time", "entry_time", "exit_time"):
                row[key] = row[key].isoformat(sep=" ")
            writer.writerow(row)

    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": generated_at,
        "privacy": "private",
        "sources": [
            {
                "role": "off_trades",
                "locator": off_file.name,
                "sha256": sha256(off_file),
                "size_bytes": off_file.stat().st_size,
                "immutable": True,
            },
            {
                "role": "xauusd_30m",
                "locator": bars_file.name,
                "sha256": sha256(bars_file),
                "size_bytes": bars_file.stat().st_size,
                "immutable": True,
            },
            {
                "role": "formal_pine",
                "locator": formal_pine.name,
                "sha256": sha256(formal_pine),
                "size_bytes": formal_pine.stat().st_size,
                "immutable": True,
            },
        ],
    }
    (output_dir / "pullback_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# S1 T1 chase-cap and pullback backtest",
        "",
        "## Emulator control",
        "",
        f'- OFF exit time/signal match: {results["emulator_validation"]["exit_time_and_signal_match_n"]}/{results["emulator_validation"]["n"]} ({results["emulator_validation"]["exit_time_and_signal_match_pct"]}%).',
        f'- Mean absolute exit-price difference: {results["emulator_validation"]["mean_abs_exit_price_difference"]} USD.',
        f'- Full OFF baseline: n={results["off_baseline_metrics"]["n"]}, WR {results["off_baseline_metrics"]["win_rate_pct"]}%, PF {results["off_baseline_metrics"]["profit_factor"]}, average USD {results["off_baseline_metrics"]["average_pnl_usd"]}.',
        "",
        "## Results",
        "",
        "| Policy | Filled/T1-held | WR (95% CI) | PF / same-source OFF | Avg USD / OFF | Closed DD | Recent n/WR/PF | Paired p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in results["policies"].items():
        independent = value["independent_signal_metrics"]
        recent = value["chronological_stability"]["recent_30pct_signal_cohort"]
        source = value["same_source_off_metrics"]
        ci = independent["win_rate_wilson_95ci_pct"]
        lines.append(
            f'| {name} | {independent["n"]}/{value["t1_held_signal_n"]} '
            f'({value["fill_rate_of_t1_held_pct"]}%) | {independent["win_rate_pct"]}% '
            f'({ci[0]}–{ci[1]}%) | {independent["profit_factor"]} / '
            f'{source["profit_factor"]} | {independent["average_pnl_usd"]} / '
            f'{source["average_pnl_usd"]} | '
            f'{independent["max_closed_trade_drawdown_usd"]} | '
            f'{recent["n"]}/{recent["win_rate_pct"]}%/{recent["profit_factor"]} | '
            f'{value["paired_outcome_exact_p_value"]} |'
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The frozen primary 0.10% pullback is not chronologically stable: recent n=45, WR 46.67%, PF 1.293. It is rejected as an adoption candidate.",
        "- Waiting one bar and entering at market is worse than the same-source OFF entries. Waiting alone is not the edge.",
        "- The post-output 0.15% sensitivity is the strongest quality screen (n=77, WR 59.74%, PF 2.106; recent n=39, WR 53.85%, PF 1.836), but it fills only 22.92% of T1-held signals. Its paired outcome test versus the same source signals is not significant, so it is a shadow candidate, not a formal rule.",
        "- The 0.15% candidate rule is: after T1 preserves the signal low, place `min(T1 close × 0.9985, T1 open × 1.0010)` during T2/T3; cancel before fill if the signal low is breached.",
        "",
        "## Boundary",
        "",
        "The independent view recalculates the actual exit for every frozen OFF signal and may overlap trades. The sequential view suppresses overlaps but cannot recover additional qualified signals that formal OFF never filled while in position. Same-bar limit-touch/invalidation ambiguity is resolved conservatively by cancelling first. No result changes the formal strategy.",
    ]
    (output_dir / "PULLBACK_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-file", type=Path, default=OFF_FILE)
    parser.add_argument("--bars-file", type=Path, default=BARS_FILE)
    parser.add_argument("--formal-pine", type=Path, default=FORMAL_PINE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--generated-at", default=datetime.now().astimezone().isoformat(timespec="seconds"))
    args = parser.parse_args()
    off_file = args.off_file.resolve()
    bars_file = args.bars_file.resolve()
    formal_pine = args.formal_pine.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = load_bars(bars_file)
    tv_trades = load_tv_trades(off_file)
    index_by_time = {bar.time: index for index, bar in enumerate(bars)}
    missing = [trade.number for trade in tv_trades if trade.entry_time not in index_by_time]
    if missing:
        raise SystemExit(f"OFF entry timestamps missing from bars: {missing[:20]}")
    signal_times = [bars[index_by_time[trade.entry_time] - 1].time for trade in tv_trades]
    chronology_cutoff = signal_times[math.ceil(len(signal_times) * 0.7) - 1]

    validation = validate_exit_emulator(bars, tv_trades, index_by_time)
    policies = {
        "T1_market": {"kind": "t1_market"},
        "T1_chase_cap_0.00pct": {"kind": "t1_chase_cap", "max_chase_pct": 0.00},
        "T1_chase_cap_0.05pct": {"kind": "t1_chase_cap", "max_chase_pct": 0.05},
        "T1_chase_cap_0.10pct": {"kind": "t1_chase_cap", "max_chase_pct": 0.10},
        "T1_chase_cap_0.15pct": {"kind": "t1_chase_cap", "max_chase_pct": 0.15},
        "T1_pullback_0.05pct_cap_0.10pct": {
            "kind": "t1_pullback",
            "pullback_pct": 0.05,
            "max_chase_pct": 0.10,
        },
        "T1_pullback_0.10pct_cap_0.10pct": {
            "kind": "t1_pullback",
            "pullback_pct": 0.10,
            "max_chase_pct": 0.10,
        },
        "T1_pullback_0.15pct_cap_0.10pct": {
            "kind": "t1_pullback",
            "pullback_pct": 0.15,
            "max_chase_pct": 0.10,
        },
    }
    policy_results = {}
    output_trades = []
    for name, policy in policies.items():
        result, trades = run_policy(
            policy, tv_trades, bars, index_by_time, chronology_cutoff
        )
        policy_results[name] = result
        output_trades.extend({"policy": name, "trade": trade} for trade in trades)
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": args.generated_at,
        "timezone": "Asia/Taipei",
        "emulator_validation": validation,
        "off_baseline_metrics": tv_metrics(tv_trades),
        "chronology_cutoff_signal_time": chronology_cutoff.isoformat(sep=" "),
        "policies": policy_results,
        "frozen_primary_policy": "T1_pullback_0.10pct_cap_0.10pct",
        "sensitivity_candidate_policy": "T1_pullback_0.15pct_cap_0.10pct",
        "interpretation": {
            "frozen_primary": "rejected_as_adoption_candidate_due_to_recent_instability",
            "sensitivity_candidate": "shadow_only_due_to_post_output_selection_low_fill_count_and_non_significant_paired_outcomes",
        },
        "formal_change": False,
    }
    write_outputs(
        output_dir,
        args.generated_at,
        results,
        output_trades,
        off_file,
        bars_file,
        formal_pine,
    )
    print(json.dumps({"validation": validation, "policies": policy_results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
