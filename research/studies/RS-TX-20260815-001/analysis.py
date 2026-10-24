#!/usr/bin/env python3
"""Build a no-lookahead, three-wave TX/TMF Fibonacci pullback study.

The price source is a TradingView-exported MXF1! continuous contract. Historical PnL is
scaled synthetically to TMF at TWD 10/point because TMF did not exist before 2024.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-TX-20260815-001"
DATA_FILES = {
    "daily": "TAIFEX_DLY_MXF1!, 1D.csv",
    "four_hour": "TAIFEX_DLY_MXF1!, 240.csv",
    "one_hour": "TAIFEX_DLY_MXF1!, 60.csv",
    "thirty_minute": "TAIFEX_DLY_MXF1!, 30.csv",
}
SOURCE_SHA256 = {
    "daily": "9228ba40db820c17e294fe62540c5b83a4809f84d40e704eec172d553a6ddc9b",
    "four_hour": "3482ea7d10fafb83e1f2b2c69b290917de7343c5149329a9144726e6086ab69f",
    "one_hour": "f9ee58a97297a4946e005b42e350fa449298846b0963012393ff546ee9bbd869",
    "thirty_minute": "45ea58882d076698180816476d3755f07d4d8644f66a6252f4f4b53d84ada704",
}
LEVELS = ("0.618", "0.5", "0.382")
RATIOS = {"0.618": 0.618, "0.5": 0.5, "0.382": 0.382}
LOTS = {"0.618": 3, "0.5": 9, "0.382": 3}
MULTIPLIER_TWD = 10
INITIAL_MARGIN_TWD = 31_800
MAINTENANCE_MARGIN_TWD = 24_400
BUFFER_POINTS = 5_000
STOP_RECOVERY_BARS = 10
MIN_INTERPRETABLE_N = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"source hash mismatch: {path}: expected {expected}, got {actual}")


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_ohlcv(path: Path, daily: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} missing fields: {sorted(missing)}")
    frame["time"] = pd.to_datetime(frame["time"])
    if getattr(frame["time"].dt, "tz", None) is not None:
        frame["time"] = frame["time"].dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if frame[["open", "high", "low", "close", "Volume"]].isna().any().any():
        raise SystemExit(f"{path} contains null OHLCV")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise SystemExit(f"{path} contains invalid high")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise SystemExit(f"{path} contains invalid low")
    frame.attrs["daily"] = daily
    return frame


def wilson(wins: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = wins / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return [round(100 * max(0, centre - margin), 2), round(100 * min(1, centre + margin), 2)]


def trade_stats(trades: list[dict]) -> dict:
    n = len(trades)
    wins = sum(item["pnl_pts_per_contract"] > 0 for item in trades)
    losses = sum(item["pnl_pts_per_contract"] < 0 for item in trades)
    gross_profit = sum(max(0.0, item["pnl_twd"]) for item in trades)
    gross_loss = abs(sum(min(0.0, item["pnl_twd"]) for item in trades))
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100 * wins / n, 2) if n else None,
        "win_rate_ci95_pct": wilson(wins, n),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_pts_per_contract": round(sum(item["pnl_pts_per_contract"] for item in trades), 2),
        "net_pnl_twd": round(sum(item["pnl_twd"] for item in trades), 2),
        "avg_pnl_pts_per_contract": round(sum(item["pnl_pts_per_contract"] for item in trades) / n, 2) if n else None,
        "avg_pnl_twd": round(sum(item["pnl_twd"] for item in trades) / n, 2) if n else None,
        "low_sample": n < MIN_INTERPRETABLE_N,
    }


def on_or_before(frame: pd.DataFrame, target: date) -> pd.Series | None:
    subset = frame[frame["time"] <= pd.Timestamp(target)]
    return None if subset.empty else subset.iloc[-1]


def first_year_row(frame: pd.DataFrame, year: int) -> pd.Series | None:
    subset = frame[frame["time"].dt.year == year]
    return None if subset.empty else subset.iloc[0]


def fib_prices(jan_open: float, anchor_high: float) -> dict[str, float]:
    spread = anchor_high - jan_open
    return {level: round(jan_open + RATIOS[level] * spread, 2) for level in LEVELS}


def choose_fill(row: pd.Series, level_price: float) -> float:
    # A resting buy limit receives the opening price when the market gaps below it.
    return round(min(float(row["open"]), level_price), 2)


def simulate_account(
    daily: pd.DataFrame,
    entries: dict[str, dict],
    exit_date: str,
    exit_price: float,
) -> dict:
    active_entries = [dict(value, level=level, lots=LOTS[level]) for level, value in entries.items() if value["triggered"]]
    if not active_entries:
        return {
            "funded_capital_twd": 0.0,
            "max_drawdown_twd": 0.0,
            "max_drawdown_pct": 0.0,
            "max_intraday_drawdown_twd": 0.0,
            "max_intraday_drawdown_pct": 0.0,
            "min_close_buffer_points": None,
            "min_intraday_buffer_points": None,
            "maintenance_call_close": False,
            "maintenance_breach_intraday_low": False,
        }
    by_date: dict[str, list[dict]] = {}
    for entry in active_entries:
        by_date.setdefault(entry["entry_date"], []).append(entry)
    start = min(pd.Timestamp(item["entry_date"]) for item in active_entries)
    end = pd.Timestamp(exit_date)
    window = daily[(daily["time"] >= start) & (daily["time"] <= end)]
    positions: list[dict] = []
    funded = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    max_intraday_drawdown = 0.0
    max_intraday_drawdown_pct = 0.0
    min_close_buffer = math.inf
    min_low_buffer = math.inf
    close_call = False
    intraday_breach = False

    def equity(mark: float) -> float:
        return funded + sum((mark - item["entry_price"]) * item["lots"] * MULTIPLIER_TWD for item in positions)

    for row in window.itertuples(index=False):
        row_date = str(row.time.date())
        for item in sorted(by_date.get(row_date, []), key=lambda value: LEVELS.index(value["level"])):
            mark_before_funding = item["entry_price"]
            positions.append(item)
            lots = sum(position["lots"] for position in positions)
            target_equity = lots * (MAINTENANCE_MARGIN_TWD + BUFFER_POINTS * MULTIPLIER_TWD)
            top_up = max(0.0, target_equity - equity(mark_before_funding))
            funded += top_up
            peak_equity = max(peak_equity, equity(mark_before_funding))

        lots = sum(position["lots"] for position in positions)
        if lots == 0:
            continue
        maintenance = lots * MAINTENANCE_MARGIN_TWD
        low_equity = equity(float(row.low))
        close_mark = exit_price if row_date == exit_date else float(row.close)
        close_equity = equity(close_mark)
        low_buffer = (low_equity - maintenance) / (lots * MULTIPLIER_TWD)
        close_buffer = (close_equity - maintenance) / (lots * MULTIPLIER_TWD)
        min_low_buffer = min(min_low_buffer, low_buffer)
        min_close_buffer = min(min_close_buffer, close_buffer)
        intraday_breach = intraday_breach or low_equity < maintenance
        close_call = close_call or close_equity < maintenance
        intraday_drawdown = peak_equity - low_equity
        max_intraday_drawdown = max(max_intraday_drawdown, intraday_drawdown)
        if peak_equity > 0:
            max_intraday_drawdown_pct = max(max_intraday_drawdown_pct, intraday_drawdown / peak_equity * 100)
        peak_equity = max(peak_equity, close_equity)
        drawdown = peak_equity - close_equity
        max_drawdown = max(max_drawdown, drawdown)
        if peak_equity > 0:
            max_drawdown_pct = max(max_drawdown_pct, drawdown / peak_equity * 100)

    return {
        "funded_capital_twd": round(funded, 2),
        "max_drawdown_twd": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "max_intraday_drawdown_twd": round(max_intraday_drawdown, 2),
        "max_intraday_drawdown_pct": round(max_intraday_drawdown_pct, 2),
        "min_close_buffer_points": round(min_close_buffer, 2),
        "min_intraday_buffer_points": round(min_low_buffer, 2),
        "maintenance_call_close": close_call,
        "maintenance_breach_intraday_low": intraday_breach,
    }


def build_cycle(daily: pd.DataFrame, year: int, cutoff: pd.Timestamp) -> dict | None:
    jan = first_year_row(daily, year)
    jul_candidates = daily[(daily["time"].dt.year == year) & (daily["time"] <= pd.Timestamp(date(year, 7, 1)))]
    if jan is None or jul_candidates.empty:
        return None
    jul = jul_candidates.iloc[-1]
    jan_open = float(jan["open"])
    jul_close = float(jul["close"])
    qualified = jul_close > jan_open
    initial_high_rows = daily[(daily["time"] >= jan["time"]) & (daily["time"] <= jul["time"])]
    initial_high_row = initial_high_rows.loc[initial_high_rows["high"].idxmax()]
    rolling_high = float(initial_high_row["high"])
    rolling_high_date = str(initial_high_row["time"].date())
    frozen_high: float | None = None
    frozen_high_date: str | None = None
    entries = {level: {"triggered": False} for level in LEVELS}
    previous_close = jul_close
    final_target = pd.Timestamp(date(year + 1, 3, 31))
    window = daily[(daily["time"] > jul["time"]) & (daily["time"] <= min(final_target, cutoff))]

    if qualified:
        for _, row in window.iterrows():
            active_high = frozen_high if frozen_high is not None else rolling_high
            prices = fib_prices(jan_open, active_high)
            first_triggered_today = False
            for level in LEVELS:
                if entries[level]["triggered"]:
                    continue
                level_price = prices[level]
                if previous_close >= level_price and float(row["low"]) < level_price:
                    if frozen_high is None:
                        frozen_high = rolling_high
                        frozen_high_date = rolling_high_date
                        prices = fib_prices(jan_open, frozen_high)
                        first_triggered_today = True
                        level_price = prices[level]
                    entries[level] = {
                        "triggered": True,
                        "entry_date": str(row["time"].date()),
                        "level_price": level_price,
                        "entry_price": choose_fill(row, level_price),
                        "lots": LOTS[level],
                        "same_day_with_first_wave": first_triggered_today or entries["0.618"].get("entry_date") == str(row["time"].date()),
                    }
            if frozen_high is None and float(row["high"]) > rolling_high:
                rolling_high = float(row["high"])
                rolling_high_date = str(row["time"].date())
            previous_close = float(row["close"])

    anchor_high = frozen_high if frozen_high is not None else rolling_high
    anchor_date = frozen_high_date if frozen_high_date is not None else rolling_high_date
    level_prices = fib_prices(jan_open, anchor_high) if qualified else {level: None for level in LEVELS}
    scheduled = on_or_before(daily[daily["time"] >= jul["time"]], date(year + 1, 3, 31))
    complete = scheduled is not None and scheduled["time"] <= cutoff and scheduled["time"].year == year + 1
    variants: dict[str, dict] = {}
    provisional_mark = None

    if complete and any(item["triggered"] for item in entries.values()):
        scheduled_date = str(scheduled["time"].date())
        scheduled_price = float(scheduled["close"])
        stop_date = scheduled_date
        stop_price = scheduled_price
        stop_reason = "scheduled_mar31"
        recovery = None
        deepest = entries["0.382"]
        if deepest["triggered"]:
            deepest_level = level_prices["0.382"]
            after = daily[(daily["time"] >= pd.Timestamp(deepest["entry_date"])) & (daily["time"] <= scheduled["time"])]
            below_count = 0
            breach_date = None
            recovered_dates: list[str] = []
            longest_below_run = 0
            for _, stop_row in after.iterrows():
                if float(stop_row["close"]) < deepest_level:
                    if below_count == 0:
                        breach_date = str(stop_row["time"].date())
                    below_count += 1
                    longest_below_run = max(longest_below_run, below_count)
                    if below_count >= STOP_RECOVERY_BARS:
                        stop_date = str(stop_row["time"].date())
                        stop_price = float(stop_row["close"])
                        stop_reason = "10_consecutive_closes_below_0382"
                        break
                elif below_count:
                    recovered_dates.append(str(stop_row["time"].date()))
                    below_count = 0
                    breach_date = None
            recovery = {
                "rule": "after the deepest layer exists, any close below 0.382 arms the test; 10 consecutive closes below exits all layers, while a close at/above resets the count",
                "longest_consecutive_closes_below": longest_below_run,
                "recovery_dates": recovered_dates,
                "terminal_open_breach_date": breach_date,
            }

        for variant, (exit_date, exit_price, reason) in {
            "no_stop": (scheduled_date, scheduled_price, "scheduled_mar31"),
            "with_stop": (stop_date, stop_price, stop_reason),
        }.items():
            layer_results = {}
            total_pnl = 0.0
            total_lots = 0
            weighted_entry = 0.0
            for level, entry in entries.items():
                if not entry["triggered"]:
                    layer_results[level] = {"triggered": False}
                    continue
                points = round(exit_price - entry["entry_price"], 2)
                pnl_twd = round(points * LOTS[level] * MULTIPLIER_TWD, 2)
                total_pnl += pnl_twd
                total_lots += LOTS[level]
                weighted_entry += entry["entry_price"] * LOTS[level]
                layer_results[level] = {
                    **entry,
                    "exit_date": exit_date,
                    "exit_price": round(exit_price, 2),
                    "pnl_pts_per_contract": points,
                    "pnl_twd": pnl_twd,
                    "win": points > 0,
                }
            average_entry = round(weighted_entry / total_lots, 2) if total_lots else None
            account = simulate_account(daily, entries, exit_date, exit_price)
            variants[variant] = {
                "exit_date": exit_date,
                "exit_price": round(exit_price, 2),
                "exit_reason": reason,
                "lots": total_lots,
                "average_entry": average_entry,
                "pnl_twd": round(total_pnl, 2),
                "return_on_funded_capital_pct": round(total_pnl / account["funded_capital_twd"] * 100, 2) if account["funded_capital_twd"] else None,
                "layers": layer_results,
                "account": account,
            }
        variants["with_stop"]["recovery_test"] = recovery
    elif not complete and any(item["triggered"] for item in entries.values()):
        marked = daily[daily["time"] <= cutoff].iloc[-1]
        active = [(level, entry) for level, entry in entries.items() if entry["triggered"]]
        total_lots = sum(LOTS[level] for level, _ in active)
        average_entry = sum(entry["entry_price"] * LOTS[level] for level, entry in active) / total_lots
        mark_price = float(marked["close"])
        provisional_mark = {
            "as_of": str(marked["time"].date()),
            "lots": total_lots,
            "average_entry": round(average_entry, 2),
            "mark_price": round(mark_price, 2),
            "unrealized_pnl_pts_per_contract_at_average": round(mark_price - average_entry, 2),
            "unrealized_pnl_twd": round(sum((mark_price - entry["entry_price"]) * LOTS[level] * MULTIPLIER_TWD for level, entry in active), 2),
            "account": simulate_account(daily, entries, str(marked["time"].date()), mark_price),
            "performance_status": "provisional_excluded_from_statistics",
        }

    return {
        "year": year,
        "jan_open_date": str(jan["time"].date()),
        "jan_open": round(jan_open, 2),
        "jul1_reference_date": str(jul["time"].date()),
        "jul1_close": round(jul_close, 2),
        "jan_to_jul1_return_pct": round((jul_close - jan_open) / jan_open * 100, 2),
        "qualified": qualified,
        "anchor_high": round(anchor_high, 2) if qualified else None,
        "anchor_high_date": anchor_date if qualified else None,
        "anchor_frozen_on_first_wave": frozen_high is not None,
        "level_prices": level_prices,
        "entries": entries,
        "complete": complete,
        "variants": variants,
        "provisional_mark": provisional_mark,
    }


def summarize(cycles: list[dict], variant: str) -> dict:
    complete = [cycle for cycle in cycles if cycle["complete"]]
    layer_trades = {level: [] for level in LEVELS}
    annual = []
    for cycle in complete:
        value = cycle["variants"].get(variant)
        if value is None:
            continue
        annual.append({"year": cycle["year"], **{key: value[key] for key in ("lots", "average_entry", "exit_date", "exit_price", "exit_reason", "pnl_twd", "return_on_funded_capital_pct")}, **value["account"]})
        for level in LEVELS:
            layer = value["layers"][level]
            if layer["triggered"]:
                layer_trades[level].append(layer)
    annual_wins = sum(item["pnl_twd"] > 0 for item in annual)
    annual_gross_profit = sum(max(0, item["pnl_twd"]) for item in annual)
    annual_gross_loss = abs(sum(min(0, item["pnl_twd"]) for item in annual))
    return {
        "by_level": {level: trade_stats(layer_trades[level]) for level in LEVELS},
        "annual_combined": {
            "n": len(annual),
            "wins": annual_wins,
            "win_rate_pct": round(100 * annual_wins / len(annual), 2) if annual else None,
            "win_rate_ci95_pct": wilson(annual_wins, len(annual)),
            "profit_factor": round(annual_gross_profit / annual_gross_loss, 3) if annual_gross_loss else None,
            "net_pnl_twd": round(sum(item["pnl_twd"] for item in annual), 2),
            "avg_pnl_twd": round(sum(item["pnl_twd"] for item in annual) / len(annual), 2) if annual else None,
            "worst_account_drawdown_twd": round(max((item["max_drawdown_twd"] for item in annual), default=0), 2),
            "worst_account_drawdown_pct": round(max((item["max_drawdown_pct"] for item in annual), default=0), 2),
            "worst_intraday_drawdown_twd": round(max((item["max_intraday_drawdown_twd"] for item in annual), default=0), 2),
            "worst_intraday_drawdown_pct": round(max((item["max_intraday_drawdown_pct"] for item in annual), default=0), 2),
            "maintenance_call_years_close": [item["year"] for item in annual if item["maintenance_call_close"]],
            "maintenance_breach_years_intraday_low": [item["year"] for item in annual if item["maintenance_breach_intraday_low"]],
            "low_sample": len(annual) < MIN_INTERPRETABLE_N,
        },
        "yearly": annual,
    }


def volatility_context(daily: pd.DataFrame) -> dict:
    previous_close = daily["close"].shift(1)
    true_range = pd.concat(
        [daily["high"] - daily["low"], (daily["high"] - previous_close).abs(), (daily["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14).mean()
    latest = daily.iloc[-1]
    by_year = {}
    for year in range(2013, int(latest["time"].year) + 1):
        indexes = daily.index[daily["time"].dt.year == year]
        if len(indexes) == 0:
            continue
        values = true_range.loc[indexes]
        by_year[str(year)] = {
            "n": len(values),
            "median_true_range_pts": round(float(values.median()), 2),
            "p75_true_range_pts": round(float(values.quantile(0.75)), 2),
            "days_true_range_ge_1000_pct": round(float((values >= 1000).mean() * 100), 2),
        }
    return {
        "latest_complete_date": str(latest["time"].date()),
        "latest_close": float(latest["close"]),
        "latest_atr14_pts": round(float(atr14.iloc[-1]), 2),
        "one_thousand_points_as_atr14": round(1000 / float(atr14.iloc[-1]), 3),
        "one_thousand_points_as_price_pct": round(1000 / float(latest["close"]) * 100, 3),
        "loss_twd_by_position": {str(lots): lots * MULTIPLIER_TWD * 1000 for lots in (3, 12, 15)},
        "by_year": by_year,
    }


def chart_outputs(package: Path, summaries: dict) -> list[dict]:
    chart_dir = package / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts = []
    no_stop = summaries["no_stop"]
    with_stop = summaries["with_stop"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = list(range(len(LEVELS)))
    width = 0.36
    ax.bar([item - width / 2 for item in x], [no_stop["by_level"][level]["win_rate_pct"] or 0 for level in LEVELS], width, label="No stop")
    ax.bar([item + width / 2 for item in x], [with_stop["by_level"][level]["win_rate_pct"] or 0 for level in LEVELS], width, label="10-close recovery stop")
    ax.set_xticks(x, LEVELS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Three-wave layer win rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "layer_win_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    charts.append({"id": "layer_win_rate", "file": f"charts/{path.name}", "title": "Three-wave layer win rate"})

    years = sorted({item["year"] for item in no_stop["yearly"]} | {item["year"] for item in with_stop["yearly"]})
    nmap = {item["year"]: item["pnl_twd"] for item in no_stop["yearly"]}
    smap = {item["year"]: item["pnl_twd"] for item in with_stop["yearly"]}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = list(range(len(years)))
    ax.bar([item - width / 2 for item in x], [nmap.get(year, 0) for year in years], width, label="No stop")
    ax.bar([item + width / 2 for item in x], [smap.get(year, 0) for year in years], width, label="With stop")
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_xticks(x, [str(year) for year in years], rotation=45)
    ax.set_ylabel("Annual combined PnL (TWD)")
    ax.set_title("Weighted 3/9/3 annual outcomes")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = chart_dir / "annual_pnl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    charts.append({"id": "annual_pnl", "file": f"charts/{path.name}", "title": "Weighted 3/9/3 annual outcomes"})
    return charts


def format_stat(value: dict) -> str:
    ci = value["win_rate_ci95_pct"]
    ci_text = "unknown" if ci is None else f"{ci[0]}–{ci[1]}%"
    pf = "—" if value["profit_factor"] is None else value["profit_factor"]
    wr = "unknown" if value["win_rate_pct"] is None else f'{value["win_rate_pct"]}%'
    avg = "unknown" if value["avg_pnl_twd"] is None else f'{value["avg_pnl_twd"]:,.0f}'
    return f'| {value["n"]} | {wr} | {ci_text} | {pf} | {value["net_pnl_twd"]:,.0f} | {avg} |'


def write_readme(package: Path, results: dict) -> None:
    lines = [
        "# TX/TMF no-lookahead three-wave Fibonacci pullback", "",
        "Research-only study on a TradingView-exported MXF1! continuous contract. Historical amounts are synthetic TMF scaling at TWD 10/point; they are not historical TMF execution results.", "",
        "## Frozen method", "",
        "- A year qualifies when the July-1 reference close is above the first trading-day open.",
        "- The anchor begins as the known year-to-date high through July 1, then updates only from prior completed days until the first 0.618 trigger; that first trigger freezes all three prices.",
        "- Levels use `jan_open + ratio × (anchor_high - jan_open)`, so a decline reaches 0.618, then 0.5, then 0.382.",
        "- Lots are 3 / +9 / +3. A daily low below a level triggers; a gap below fills at the opening price, otherwise at the level.",
        "- Primary exit is the close on/before the following March 31. After the deepest layer exists, the stop variant exits all layers following 10 consecutive closes below 0.382; any close at/above resets the count.",
        "- Capital is topped up at each fill until account equity equals maintenance margin plus a 5,000-point buffer for every open lot.", "",
    ]
    for variant, label in (("no_stop", "No stop"), ("with_stop", "10-close recovery stop")):
        summary = results["summaries"][variant]
        lines += [f"## {label}", "", "| Level | n | WR | 95% CI | PF | Net TWD | Avg TWD |", "|---|---:|---:|---|---:|---:|---:|"]
        for level in LEVELS:
            lines.append(f"| {level} " + format_stat(summary["by_level"][level]))
        annual = summary["annual_combined"]
        lines += ["", f'- Combined years: n={annual["n"]}, WR={annual["win_rate_pct"]}%, PF={annual["profit_factor"]}, net TWD {annual["net_pnl_twd"]:,.0f}.', f'- Worst close-equity drawdown: TWD {annual["worst_account_drawdown_twd"]:,.0f} ({annual["worst_account_drawdown_pct"]}%); worst intraday-low stress drawdown: TWD {annual["worst_intraday_drawdown_twd"]:,.0f} ({annual["worst_intraday_drawdown_pct"]}%).', f'- Close maintenance-call years: {annual["maintenance_call_years_close"] or "none"}; intraday-low breach years: {annual["maintenance_breach_years_intraday_low"] or "none"}.', ""]
    context = results["volatility_context"]
    lines += [
        "## 1,000-point risk context", "",
        f'- At the latest complete close {context["latest_close"]:,.0f}, 1,000 points is {context["one_thousand_points_as_price_pct"]}% and only {context["one_thousand_points_as_atr14"]} of the 14-day ATR ({context["latest_atr14_pts"]:,.0f} points).',
        f'- A 1,000-point adverse move is TWD {context["loss_twd_by_position"]["3"]:,} at 3 lots, TWD {context["loss_twd_by_position"]["12"]:,} at 12 lots, and TWD {context["loss_twd_by_position"]["15"]:,} at 15 lots.',
        f'- In 2026, {context["by_year"]["2026"]["days_true_range_ge_1000_pct"]}% of complete daily bars have true range at least 1,000 points.', "",
    ]
    provisional = next((cycle for cycle in results["cycles"] if cycle["year"] == 2026), None)
    if provisional is not None:
        mark = provisional["provisional_mark"]
        lines += [
            "## Provisional 2026 path", "",
            f'- Corrected levels: 0.618 `{provisional["level_prices"]["0.618"]:,.1f}`, 0.5 `{provisional["level_prices"]["0.5"]:,.1f}`, 0.382 `{provisional["level_prices"]["0.382"]:,.1f}`.',
            f'- Only the first 3-lot layer triggered. At the {mark["as_of"]} mark of {mark["mark_price"]:,.0f}, it is +{mark["unrealized_pnl_pts_per_contract_at_average"]:,.1f} points / synthetic TWD {mark["unrealized_pnl_twd"]:,.0f}.',
            f'- Before recovery, intraday-low account stress reached TWD {mark["account"]["max_intraday_drawdown_twd"]:,.0f}. This incomplete cycle is excluded from WR/PF/PnL statistics.', "",
        ]
    lines += ["## Limitations", ""] + [f"- {item}" for item in results["limitations"]]
    (package / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(package: Path, results: dict) -> None:
    images = []
    for chart in results["charts"]:
        encoded = base64.b64encode((package / chart["file"]).read_bytes()).decode("ascii")
        images.append(f'<h2>{chart["title"]}</h2><img alt="{chart["title"]}" src="data:image/png;base64,{encoded}">')
    findings = "".join(f"<li>{item}</li>" for item in results["insights"])
    limits = "".join(f"<li>{item}</li>" for item in results["limitations"])
    html = f'''<!doctype html><meta charset="utf-8"><title>TX/TMF three-wave Fibonacci study</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}}img{{max-width:100%;height:auto}}li{{margin:.45rem 0}}</style>
<h1>TX/TMF no-lookahead three-wave Fibonacci pullback</h1><p>Private research; no live rule or broker action.</p>
<h2>Insights</h2><ul>{findings}</ul>{''.join(images)}<h2>Limitations</h2><ul>{limits}</ul>'''
    (package / "report.html").write_text(html, encoding="utf-8")


def write_manifest(package: Path, data_root: Path, cutoff: str) -> None:
    sources = []
    for role, filename in DATA_FILES.items():
        path = data_root / filename
        sources.append({
            "role": role,
            "source_system": "TradingView CSV export",
            "locator": f"local-inputs/{filename}",
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "immutable": True,
            "privacy": "private_raw_source",
            "observed_at": "2026-08-15",
            "timezone": "Asia/Taipei",
            "analysis_use": "primary daily backtest" if role == "daily" else "coverage and lower-timeframe availability audit",
        })
    json_dump(package / "source_manifest.json", {"schema_version": 1, "study_id": STUDY_ID, "privacy": "private", "data_cutoff": cutoff, "sources": sources})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("local-inputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced"))
    parser.add_argument("--data-cutoff", default="2026-08-14")
    parser.add_argument("--generated-at", default=datetime.now().astimezone().isoformat(timespec="seconds"))
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    package = args.output_dir.resolve()
    cutoff = pd.Timestamp(args.data_cutoff)

    for role, filename in DATA_FILES.items():
        verify(data_root / filename, SOURCE_SHA256[role])
    frames = {role: load_ohlcv(data_root / filename, role == "daily") for role, filename in DATA_FILES.items()}
    daily_all = frames["daily"]
    excluded = daily_all[daily_all["time"] > cutoff]
    daily = daily_all[daily_all["time"] <= cutoff].copy().reset_index(drop=True)
    cycles = [cycle for year in range(2013, 2027) if (cycle := build_cycle(daily, year, cutoff)) is not None]
    summaries = {variant: summarize(cycles, variant) for variant in ("no_stop", "with_stop")}
    volatility = volatility_context(daily)
    package.mkdir(parents=True, exist_ok=True)

    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": args.generated_at,
        "market": "TX",
        "execution_model": "synthetic TMF at TWD 10/point",
        "method": {
            "timezone": "Asia/Taipei / TradingView exchange session labels",
            "data_cutoff": args.data_cutoff,
            "qualification": "jul1 reference close > first trading-day open",
            "anchor": "known YTD high through July 1, then prior-completed-day rolling high until first 0.618 trigger; freeze thereafter",
            "formula": "jan_open + ratio * (anchor_high - jan_open)",
            "level_order": list(LEVELS),
            "lots": LOTS,
            "trigger": "previous close >= level and current daily low < level",
            "fill": "current open if gap below level, otherwise level",
            "same_day_multi_level": "all crossed levels fill on the same daily bar",
            "no_stop_exit": "close on or before following March 31",
            "stop_exit": "after 0.382 entry, any close below frozen 0.382 arms the test; exit all after 10 consecutive closes below, while a close at/above resets the count",
            "margin": {"initial_twd_per_lot": INITIAL_MARGIN_TWD, "maintenance_twd_per_lot": MAINTENANCE_MARGIN_TWD, "normalized_as_of": "2026-07-06", "historical_application": "fixed-current-margin scenario, not historical schedule"},
            "funding": "at each fill, top up account equity to maintenance plus 5,000 points times TWD10 times open lots",
            "costs": "gross; commission, tax, slippage beyond gap handling, and explicit roll cost are unknown/not deducted",
        },
        "data_quality": {
            "daily_rows_used": len(daily),
            "daily_start": str(daily["time"].min().date()),
            "daily_end": str(daily["time"].max().date()),
            "excluded_incomplete_or_post_cutoff_daily_rows": excluded[["time", "open", "high", "low", "close"]].assign(time=lambda x: x["time"].dt.strftime("%Y-%m-%d")).to_dict("records"),
            "zero_volume_daily_dates": daily.loc[daily["Volume"] <= 0, "time"].dt.strftime("%Y-%m-%d").tolist(),
            "lower_timeframe_coverage": {role: {"rows": len(frame), "start": frame["time"].min().isoformat(), "end": frame["time"].max().isoformat()} for role, frame in frames.items() if role != "daily"},
        },
        "cycles": cycles,
        "summaries": summaries,
        "volatility_context": volatility,
        "insights": [],
        "limitations": [
            "Only 13 complete decision years (2013–2025) exist; every level and combined result is low sample.",
            "TMF launched in 2024. Pre-launch amounts scale MXF1! prices synthetically at TWD 10/point and cannot reproduce historical TMF liquidity, fills, or slippage.",
            "MXF1! is a rolling continuous front-month contract. Back-adjustment setting is unknown, and roll jumps/basis can affect anchors, triggers, exits, and drawdown.",
            "The daily bar cannot establish intraday high/low order. Using only the prior completed day's anchor prevents lookahead but may differ from an intraday implementation.",
            "Daily close proxies settlement for maintenance checks; intraday-low breach is a stress flag, not proof of an actual broker margin call.",
            "The fixed 2026 margin scenario is applied to all years for current-capital comparability; it is not the historical margin schedule.",
            "Results are gross of commission, transaction tax, explicit roll cost, and additional slippage.",
            "The 2026 cycle is provisional and excluded from WR/PF/PnL statistics because its March 2027 exit is unavailable.",
        ],
    }
    no_stop = summaries["no_stop"]["annual_combined"]
    with_stop = summaries["with_stop"]["annual_combined"]
    results["insights"] = [
        f'No-stop combined outcomes: n={no_stop["n"]}, WR={no_stop["win_rate_pct"]}%, PF={no_stop["profit_factor"]}, net TWD {no_stop["net_pnl_twd"]:,.0f}.',
        f'10-bar recovery-stop outcomes: n={with_stop["n"]}, WR={with_stop["win_rate_pct"]}%, PF={with_stop["profit_factor"]}, net TWD {with_stop["net_pnl_twd"]:,.0f}.',
        f'One thousand points equals only {volatility["one_thousand_points_as_atr14"]} ATR14 at the latest complete bar, but costs TWD {volatility["loss_twd_by_position"]["15"]:,} at 15 lots.',
        "The 3/9/3 schedule is a midpoint-conviction design: 12 of 15 lots are already open at 0.5; the final 0.382 layer reduces the full-book average to exactly the 0.5 level but adds little protection before it triggers.",
        "A 5,000-point maintenance buffer implies a full-book margin line exactly 5,000 points below the funded mark, not the separately supplied 31,454 estimate unless additional capital is deposited.",
    ]
    results["charts"] = chart_outputs(package, summaries)
    json_dump(package / "results.json", results)
    write_manifest(package, data_root, args.data_cutoff)
    write_readme(package, results)
    write_report(package, results)
    print(json.dumps({"study_id": STUDY_ID, "cycles": len(cycles), "complete_years": [cycle["year"] for cycle in cycles if cycle["complete"]], "provisional_years": [cycle["year"] for cycle in cycles if not cycle["complete"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
