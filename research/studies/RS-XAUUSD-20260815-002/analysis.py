#!/usr/bin/env python3
"""Public reproduction method for the XAUUSD confirmation/context studies.

Raw files are intentionally not published. Supply locally authorized TradingView
List-of-Trades exports for S1 V3.9 and S2 V3.2 plus the price/CFTC inputs required by
the selected study. The script reproduces the published aggregate JSON; chart PNGs are
reviewed static artifacts and are not regenerated here.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TAIPEI = ZoneInfo("Asia/Taipei")
NEW_YORK = ZoneInfo("America/New_York")
STUDY_MODES = {
    "RS-XAUUSD-20260815-001": "post_signal",
    "RS-XAUUSD-20260815-002": "daily_context",
    "RS-XAUUSD-20260815-003": "cftc_regime",
}
TRADE_COLUMN_ALIASES = {
    "trade_id": ["Trade number", "Trade #"],
    "type": ["Type"],
    "datetime": ["Date and time"],
    "signal": ["Signal"],
    "price": ["Price USD"],
    "size_qty": ["Size (qty)"],
    "net_pnl_usd": ["Net PnL USD", "Net P&L USD"],
    "return_pct": ["Return %", "Net P&L %"],
    "mfe_pct": ["Favorable excursion %"],
    "mae_pct": ["Adverse excursion %"],
}


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pick(columns: list[str], canonical: str) -> str:
    for alias in TRADE_COLUMN_ALIASES[canonical]:
        if alias in columns:
            return alias
    raise KeyError(f"missing trade column: {canonical}")


def load_trades(path: Path, strategy: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("﻿").strip() for column in frame.columns]
    rename = {pick(list(frame.columns), key): key for key in TRADE_COLUMN_ALIASES}
    frame = frame.rename(columns=rename)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    entries = frame[frame["type"] == "Entry long"][
        ["trade_id", "datetime", "signal", "price", "size_qty"]
    ].rename(columns={"datetime": "entry_time", "signal": "entry_signal", "price": "entry_price"})
    exits = frame[frame["type"] == "Exit long"][
        ["trade_id", "datetime", "signal", "price", "net_pnl_usd", "return_pct", "mfe_pct", "mae_pct"]
    ].rename(columns={"datetime": "exit_time", "signal": "exit_signal", "price": "exit_price"})
    trades = entries.merge(exits, on="trade_id", how="inner")
    trades = trades[trades["exit_signal"].astype(str).str.upper() != "OPEN"].copy()
    trades["hold_bars"] = ((trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 1800).astype(int)
    trades["win"] = trades["net_pnl_usd"] > 0
    trades["result"] = np.select(
        [trades["net_pnl_usd"] > 0, trades["net_pnl_usd"] < 0],
        ["win", "loss"], default="breakeven",
    )
    trades["strategy"] = strategy
    return trades.sort_values("entry_time").reset_index(drop=True)


def load_price(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [column.strip() for column in frame.columns]
    time_col = next(column for column in frame.columns if "time" in column.lower())
    frame[time_col] = pd.to_datetime(frame[time_col], utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    frame = frame.rename(columns={time_col: "time"})
    return frame[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def wilson(wins: int, count: int, z: float = 1.96) -> list[float] | None:
    if count == 0:
        return None
    proportion = wins / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
    return [round(100 * (centre - margin / denominator), 2), round(100 * (centre + margin / denominator), 2)]


def stat(frame: pd.DataFrame) -> dict:
    count = len(frame)
    if count == 0:
        value = {
            "n": 0, "wins": 0, "win_rate_pct": None, "win_rate_ci95_pct": None,
            "profit_factor": None, "net_pnl_usd": 0.0, "avg_pnl_usd": None,
            "avg_return_pct": None,
        }
    else:
        wins = int((frame["net_pnl_usd"] > 0).sum())
        gross_profit = frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
        gross_loss = abs(frame.loc[frame["net_pnl_usd"] < 0, "net_pnl_usd"].sum())
        value = {
            "n": count,
            "wins": wins,
            "win_rate_pct": round(100 * wins / count, 2),
            "win_rate_ci95_pct": wilson(wins, count),
            "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
            "net_pnl_usd": round(float(frame["net_pnl_usd"].sum()), 2),
            "avg_pnl_usd": round(float(frame["net_pnl_usd"].mean()), 2),
            "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
        }
    value["low_sample"] = count < 20
    return value


def grouped(frame: pd.DataFrame, key: str, report_key: str | None = None) -> dict:
    result = {}
    for name, part in frame.groupby(key, observed=True):
        item = stat(part)
        if report_key:
            item["distinct_reports"] = int(part[report_key].nunique())
        result[str(name)] = item
    return result


def chronological_groups(frame: pd.DataFrame, keys: list[str], report_key: str | None = None) -> dict:
    split = int(len(frame) * 0.7)
    result = {"split_ratio": 0.7}
    for label, part in (("in_sample", frame.iloc[:split]), ("held_out", frame.iloc[split:])):
        result[label] = {
            "period": {
                "start": part["entry_time"].min().isoformat() if len(part) else None,
                "end": part["entry_time"].max().isoformat() if len(part) else None,
            },
            "baseline": stat(part),
            "groups": {key: grouped(part, key, report_key) for key in keys},
        }
    return result


def attach_post_signal(trades: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    positions = {value: index for index, value in enumerate(price["time"])}
    records = []
    for row in trades.itertuples(index=False):
        index = positions.get(row.entry_time)
        if index is None or index < 1 or index + 2 >= len(price):
            continue
        signal, t1, t2, t3 = [price.iloc[index + offset] for offset in (-1, 0, 1, 2)]
        record = row._asdict()
        record.update({
            "signal_time": signal["time"],
            "signal_bar_low": float(signal["low"]),
            "signal_bar_close": float(signal["close"]),
            "t1_time": t1["time"], "t1_low": float(t1["low"]), "t1_close": float(t1["close"]),
            "t2_time": t2["time"], "t2_low": float(t2["low"]), "t2_close": float(t2["close"]),
            "t3_time": t3["time"], "t3_open": float(t3["open"]),
            "t1_break_signal_low": bool(t1["low"] < signal["low"]),
            "t1_t2_break_signal_low": bool(min(t1["low"], t2["low"]) < signal["low"]),
            "t1_close_holds": bool(t1["close"] >= signal["close"]),
            "t2_close_holds": bool(t2["close"] >= signal["close"]),
            "wait1_entry_observable": bool(row.exit_time > t2["time"]),
            "wait2_entry_observable": bool(row.exit_time > t3["time"]),
            "wait1_entry_delta_pct": round(100 * (float(t2["open"]) / row.entry_price - 1), 5),
            "wait2_entry_delta_pct": round(100 * (float(t3["open"]) / row.entry_price - 1), 5),
        })
        records.append(record)
    return pd.DataFrame(records)


def rule_summary(frame: pd.DataFrame, mask: pd.Series) -> dict:
    selected, rejected = frame[mask], frame[~mask]
    baseline, selected_stats = stat(frame), stat(selected)
    losses_filtered = int((rejected["net_pnl_usd"] < 0).sum())
    return {
        "selected": selected_stats,
        "selected_n": len(selected),
        "rejected_n": len(rejected),
        "kept_pct": round(100 * len(selected) / len(frame), 2) if len(frame) else None,
        "losses_filtered": losses_filtered,
        "wins_filtered": int((rejected["net_pnl_usd"] > 0).sum()),
        "rejected_loss_pct": round(100 * losses_filtered / len(rejected), 2) if len(rejected) else None,
        "win_rate_change_pp": round(selected_stats["win_rate_pct"] - baseline["win_rate_pct"], 2) if len(selected) and len(frame) else None,
        "profit_factor_change": round(selected_stats["profit_factor"] - baseline["profit_factor"], 3) if selected_stats["profit_factor"] is not None and baseline["profit_factor"] is not None else None,
    }


def post_signal_rules(frame: pd.DataFrame) -> dict:
    masks = {
        "baseline": pd.Series(True, index=frame.index),
        "t1_hold_signal_low": ~frame["t1_break_signal_low"],
        "t1_t2_hold_signal_low": ~frame["t1_t2_break_signal_low"],
        "t1_hold_low_and_close": (~frame["t1_break_signal_low"]) & frame["t1_close_holds"],
        "t1_t2_hold_low_and_close": (~frame["t1_t2_break_signal_low"]) & frame["t2_close_holds"],
    }
    return {name: rule_summary(frame, mask) for name, mask in masks.items()}


def post_signal_holdout(frame: pd.DataFrame) -> dict:
    split = int(len(frame) * 0.7)
    result = {"split_ratio": 0.7}
    for label, part in (("in_sample", frame.iloc[:split]), ("held_out", frame.iloc[split:])):
        result[label] = {
            "period": {
                "start": part["entry_time"].min().isoformat() if len(part) else None,
                "end": part["entry_time"].max().isoformat() if len(part) else None,
            },
            "baseline": stat(part),
            "t1_hold_signal_low": rule_summary(part, ~part["t1_break_signal_low"]),
            "t1_t2_hold_signal_low": rule_summary(part, ~part["t1_t2_break_signal_low"]),
        }
    return result


def distribution(values: pd.Series) -> dict:
    values = values.dropna()
    return {
        "n": len(values),
        "mean_pct": round(float(values.mean()), 4) if len(values) else None,
        "median_pct": round(float(values.median()), 4) if len(values) else None,
        "p10_pct": round(float(values.quantile(0.1)), 4) if len(values) else None,
        "p90_pct": round(float(values.quantile(0.9)), 4) if len(values) else None,
    }


def build_post_signal(frames: dict[str, pd.DataFrame], price: pd.DataFrame, study_id: str, generated_at: str) -> dict:
    matched_frames = {label: attach_post_signal(frame, price) for label, frame in frames.items()}
    result = {
        "schema_version": 1, "study_id": study_id, "generated_at": generated_at,
        "strategy": "S1 V3.9 and S2 V3.2",
        "method": {
            "timezone": "Asia/Taipei",
            "signal_bar_definition": "The 30m bar immediately before List-of-Trades entry_time; both Pine strategies use default next-bar-open processing, confirmed by source and entry-price/open matching.",
            "primary_hypothesis": "A long trade whose next completed 30m bar breaks the signal-bar low has a worse original strategy outcome.",
            "secondary_hypothesis": "Holding that low through the next two completed bars improves original-outcome selection.",
            "no_lookahead": "T+1 can only be acted on at T+2 open; T+1/T+2 can only be acted on at T+3 open.",
            "counterfactual_boundary": "No Pine replay. Selected stats retain original TradingView exits/PnL and measure screening association, not delayed-entry strategy PnL.",
            "minimum_interpretable_group_n": 20,
        },
        "strategies": {},
        "limitations": [
            "30m price history starts 2025-09-17, so only overlapping trades are matched.",
            "The trade export contains filled trades, not every raw signal; filtered/in-position signals are absent.",
            "Original outcomes and exits are retained. Exact delayed-entry WR/PF requires a Pine replay with revised entry timing.",
            "Subgroup comparisons are descriptive and multiple rules are not authorization for parameter optimization.",
        ],
        "charts": [{"id": "post_signal_rule_winrate", "file": "post_signal_rule_winrate.png", "title": "Post-Signal Rule Win Rate", "section": "timing_30m"}],
    }
    for label, frame in frames.items():
        matched = matched_frames[label]
        result["strategies"][label] = {
            "full_baseline": stat(frame),
            "coverage": {"total_trades": len(frame), "matched_trades": len(matched), "pct": round(100 * len(matched) / len(frame), 2)},
            "matched_baseline": stat(matched),
            "rules": post_signal_rules(matched),
            "chronological_holdout": post_signal_holdout(matched),
            "wait_1_bar_entry": {
                "observable_n": int(matched["wait1_entry_observable"].sum()),
                "entry_price_delta": distribution(matched.loc[matched["wait1_entry_observable"], "wait1_entry_delta_pct"]),
            },
            "wait_2_bars_entry": {
                "observable_n": int(matched["wait2_entry_observable"].sum()),
                "entry_price_delta": distribution(matched.loc[matched["wait2_entry_observable"], "wait2_entry_delta_pct"]),
            },
        }
    return result


def attach_daily(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    bars = daily.copy()
    bars["available_at"] = bars["time"] + pd.Timedelta(days=1)
    bars["direction"] = np.select(
        [bars["close"] > bars["open"], bars["close"] < bars["open"]],
        ["UP", "DOWN"], default="FLAT",
    )
    available = bars["available_at"].to_numpy()
    records = []
    for row in trades.itertuples(index=False):
        index = int(np.searchsorted(available, np.datetime64(row.entry_time), side="right") - 1)
        if index < 1:
            continue
        record = row._asdict()
        record.update({
            "d1_time": bars.iloc[index]["time"],
            "d1_direction": bars.iloc[index]["direction"],
            "d1_return_pct": round(100 * (bars.iloc[index]["close"] / bars.iloc[index]["open"] - 1), 5),
            "d2_time": bars.iloc[index - 1]["time"],
            "d2_direction": bars.iloc[index - 1]["direction"],
            "d2_return_pct": round(100 * (bars.iloc[index - 1]["close"] / bars.iloc[index - 1]["open"] - 1), 5),
            "d2_d1_sequence": f'{bars.iloc[index - 1]["direction"]}_{bars.iloc[index]["direction"]}',
        })
        records.append(record)
    return pd.DataFrame(records)


def daily_interaction(daily_frame: pd.DataFrame, signal_frame: pd.DataFrame) -> dict:
    columns = ["trade_id", "strategy", "t1_break_signal_low", "t1_t2_break_signal_low"]
    joined = daily_frame.merge(signal_frame[columns], on=["trade_id", "strategy"], how="inner")
    joined["d1_x_t1"] = joined["d1_direction"] + "|" + np.where(joined["t1_break_signal_low"], "T1_BREAK", "T1_HOLD")
    joined["d1d2_x_t2"] = joined["d2_d1_sequence"] + "|" + np.where(joined["t1_t2_break_signal_low"], "T1T2_BREAK", "T1T2_HOLD")
    return {
        "coverage": len(joined),
        "by_d1_x_t1": grouped(joined, "d1_x_t1"),
        "by_d1d2_x_t2": grouped(joined, "d1d2_x_t2"),
    }


def build_daily(frames: dict[str, pd.DataFrame], price30: pd.DataFrame, daily: pd.DataFrame, study_id: str, generated_at: str) -> dict:
    signal_frames = {label: attach_post_signal(frame, price30) for label, frame in frames.items()}
    result = {
        "schema_version": 1, "study_id": study_id, "generated_at": generated_at,
        "strategy": "S1 V3.9 and S2 V3.2",
        "method": {
            "timezone": "Asia/Taipei",
            "daily_availability": "TradingView daily bar timestamp + 24h; only fully completed bars available before entry are used.",
            "direction": "UP when close>open, DOWN when close<open, otherwise FLAT.",
            "sequence_order": "D-2 then D-1",
            "interaction": "Cross-tabulation with pre-registered T+1/T+2 signal-low hold; probabilities are not multiplied.",
            "minimum_interpretable_group_n": 20,
        },
        "strategies": {},
        "limitations": [
            "Daily context is associative, not causal.",
            "D-1/D-2 sequence and post-signal interaction create low-n cells, especially for S2 V3.2.",
            "The +24h availability rule is deliberately conservative and avoids using an unfinished daily candle.",
            "No daily-context result changes a live signal or formal strategy rule without a separate adoption decision.",
        ],
        "charts": [{"id": "prior_day_direction_winrate", "file": "prior_day_direction_winrate.png", "title": "Prior-Day Direction Win Rate", "section": "comparison"}],
    }
    for label, frame in frames.items():
        attached = attach_daily(frame, daily)
        result["strategies"][label] = {
            "coverage": {"total_trades": len(frame), "matched_trades": len(attached), "pct": round(100 * len(attached) / len(frame), 2)},
            "baseline": stat(attached),
            "by_d1_direction": grouped(attached, "d1_direction"),
            "by_d2_d1_sequence": grouped(attached, "d2_d1_sequence"),
            "chronological_holdout": chronological_groups(attached, ["d1_direction", "d2_d1_sequence"]),
            "post_signal_interaction": daily_interaction(attached, signal_frames[label]),
        }
    return result


def load_cftc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["report_date"] = pd.to_datetime(frame["report_date_as_yyyy_mm_dd"]).dt.date
    numeric = [
        "open_interest_all", "change_in_open_interest_all", "m_money_positions_long_all",
        "m_money_positions_short_all", "m_money_positions_spread",
        "change_in_m_money_long_all", "change_in_m_money_short_all",
        "change_in_m_money_spread",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric)
    standard, conservative = [], []
    for report_date in frame["report_date"]:
        friday = report_date + timedelta(days=3)
        standard.append(datetime.combine(friday, time(15, 30), tzinfo=NEW_YORK).astimezone(TAIPEI).replace(tzinfo=None))
        conservative.append(datetime.combine(report_date + timedelta(days=7), time(0, 0)))
    frame["standard_release_at_taipei"] = standard
    frame["available_at"] = conservative
    frame["mm_net"] = frame["m_money_positions_long_all"] - frame["m_money_positions_short_all"]
    frame["mm_net_change"] = frame["change_in_m_money_long_all"] - frame["change_in_m_money_short_all"]
    frame["mm_net_oi_ratio"] = frame["mm_net"] / frame["open_interest_all"]
    frame["net_oi_regime"] = np.select(
        [
            (frame["mm_net_change"] >= 0) & (frame["change_in_open_interest_all"] >= 0),
            (frame["mm_net_change"] >= 0) & (frame["change_in_open_interest_all"] < 0),
            (frame["mm_net_change"] < 0) & (frame["change_in_open_interest_all"] >= 0),
        ],
        ["NET_UP_OI_UP", "NET_UP_OI_DOWN", "NET_DOWN_OI_UP"], default="NET_DOWN_OI_DOWN",
    )
    percentiles = []
    for index in range(len(frame)):
        history = frame.loc[:index, "mm_net_oi_ratio"]
        percentiles.append(float((history <= history.iloc[-1]).mean()) if len(history) >= 52 else np.nan)
    frame["expanding_crowding_percentile"] = percentiles
    frame["crowding_regime"] = np.select(
        [frame["expanding_crowding_percentile"] >= 0.8, frame["expanding_crowding_percentile"] <= 0.2],
        ["CROWDED_LONG", "LOW_NET_LONG"], default="MID_RANGE",
    )
    return frame


def attach_cftc(trades: pd.DataFrame, cftc: pd.DataFrame) -> pd.DataFrame:
    available = cftc["available_at"].to_numpy()
    records = []
    for row in trades.itertuples(index=False):
        index = int(np.searchsorted(available, np.datetime64(row.entry_time), side="right") - 1)
        if index < 51:
            continue
        report = cftc.iloc[index]
        lag_days = (row.entry_time - report["available_at"]).total_seconds() / 86400
        record = row._asdict()
        record.update({
            "cftc_report_date": report["report_date"].isoformat(),
            "cftc_available_at": report["available_at"],
            "cftc_standard_release_at_taipei": report["standard_release_at_taipei"],
            "mm_net": int(report["mm_net"]),
            "mm_net_change": int(report["mm_net_change"]),
            "oi_change": int(report["change_in_open_interest_all"]),
            "net_oi_regime": report["net_oi_regime"],
            "crowding_regime": report["crowding_regime"],
            "days_since_conservative_available": round(lag_days, 3),
            "availability_lag_bucket": "D0_2" if lag_days < 3 else ("D3_6" if lag_days < 7 else "D7_PLUS"),
        })
        records.append(record)
    return pd.DataFrame(records)


def build_cftc(frames: dict[str, pd.DataFrame], cftc: pd.DataFrame, study_id: str, generated_at: str) -> dict:
    result = {
        "schema_version": 1, "study_id": study_id, "generated_at": generated_at,
        "strategy": "S1 V3.9 and S2 V3.2",
        "method": {
            "timezone": "Asia/Taipei",
            "report_definition": "CFTC Disaggregated Futures Only, GOLD code 088691, Managed Money.",
            "standard_release_reference": "Tuesday positions are normally released Friday 15:30 America/New_York.",
            "primary_no_lookahead_availability": "Conservative report_date + 7 calendar days at 00:00 Asia/Taipei; later than the normal Friday release and robust to ordinary holiday delay.",
            "net_change": "change_in_m_money_long_all - change_in_m_money_short_all",
            "crowding": "Expanding percentile of Managed Money net/open interest using only reports through the assigned report; minimum 52 reports.",
            "minimum_interpretable_group_n": 20,
        },
        "cftc_period": {"start": cftc["report_date"].min().isoformat(), "end": cftc["report_date"].max().isoformat(), "reports": len(cftc)},
        "strategies": {},
        "limitations": [
            "CFTC is weekly regime context, not a 30-minute timing signal.",
            "The conservative +7-day availability rule sacrifices some freshness to prevent release-time leakage.",
            "Trade-level confidence intervals do not adjust for clustering of several trades under the same CFTC report; distinct report counts are shown.",
            "The CFTC snapshot begins in 2022 only to provide a 52-report expanding-percentile warm-up before 2024 trades.",
            "No CFTC subgroup becomes a live filter without separate approval and chronological validation.",
        ],
        "charts": [{"id": "cftc_net_oi_regime_winrate", "file": "cftc_net_oi_regime_winrate.png", "title": "CFTC Net/OI Regime Win Rate", "section": "macro"}],
    }
    for label, frame in frames.items():
        attached = attach_cftc(frame, cftc)
        result["strategies"][label] = {
            "coverage": {
                "total_trades": len(frame), "matched_trades": len(attached),
                "pct": round(100 * len(attached) / len(frame), 2),
                "distinct_reports": int(attached["cftc_report_date"].nunique()),
            },
            "baseline": stat(attached),
            "by_net_oi_regime": grouped(attached, "net_oi_regime", "cftc_report_date"),
            "by_crowding_regime": grouped(attached, "crowding_regime", "cftc_report_date"),
            "by_availability_lag": grouped(attached, "availability_lag_bucket", "cftc_report_date"),
            "chronological_holdout": chronological_groups(attached, ["net_oi_regime", "crowding_regime"], "cftc_report_date"),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", choices=sorted(STUDY_MODES), default=Path(__file__).resolve().parent.name)
    parser.add_argument("--s1-trades", type=Path, required=True)
    parser.add_argument("--s2-trades", type=Path, required=True)
    parser.add_argument("--price-30m", type=Path)
    parser.add_argument("--price-1d", type=Path)
    parser.add_argument("--cftc", type=Path)
    parser.add_argument("--generated-at", default=datetime.now(TAIPEI).isoformat(timespec="seconds"))
    parser.add_argument("--output", type=Path, default=Path("reproduced-results.json"))
    args = parser.parse_args()
    mode = STUDY_MODES[args.study_id]
    frames = {
        "S1 V3.9": load_trades(args.s1_trades, "S1 V3.9"),
        "S2 V3.2": load_trades(args.s2_trades, "S2 V3.2"),
    }
    if mode == "post_signal":
        if not args.price_30m:
            parser.error("--price-30m is required for the post-signal study")
        result = build_post_signal(frames, load_price(args.price_30m), args.study_id, args.generated_at)
    elif mode == "daily_context":
        if not args.price_30m or not args.price_1d:
            parser.error("--price-30m and --price-1d are required for the daily-context study")
        result = build_daily(frames, load_price(args.price_30m), load_price(args.price_1d), args.study_id, args.generated_at)
    else:
        if not args.cftc:
            parser.error("--cftc is required for the CFTC study")
        result = build_cftc(frames, load_cftc(args.cftc), args.study_id, args.generated_at)
    json_dump(args.output, result)
    print(json.dumps({"study_id": args.study_id, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
