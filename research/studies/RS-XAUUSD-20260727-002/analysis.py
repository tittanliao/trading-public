#!/usr/bin/env python3
"""Re-run S1 AweWithBB 30-minute-slot and Macro context for both V3.4 and V3.9.

The legacy ``trading`` repository is read-only. This script records hashes for every
input and writes derived research artifacts only inside ``trading-private``. It
generalizes ``analyze_s1_v39_context.py`` to run the same deterministic method across
two TradingView "List of Trades" export schemas (V3.4's older column names and V3.9's
current ones) so results are directly comparable at 30-minute granularity. No
fail-pattern (fail_type/DXY/MTF/K-bar) breakdown is computed here by request;
session and Macro context are retained because they are part of the existing 001
methodology, not the fail-pattern report.

Provenance-only, serving only the superseded RS-XAUUSD-20260727-002 (status: pending,
never migrated to the current fail-pattern report format). Not rerun or extended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LEGACY = Path("../trading")
DEFAULT_OUTPUT = Path("reproduced")
MA_LENGTH = 50
MAX_MACRO_AGE = pd.Timedelta(days=4)
STUDY_ID = "RS-XAUUSD-20260727-002"
ENTRY_SLOTS_30M = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in (0, 30)]

# Column aliases across TradingView export schema versions.
COLUMN_ALIASES = {
    "trade_id": ["Trade number", "Trade #"],
    "type": ["Type"],
    "datetime": ["Date and time"],
    "price": ["Price USD"],
    "size_qty": ["Size (qty)"],
    "signal": ["Signal"],
    "net_pnl_usd": ["Net PnL USD", "Net P&L USD"],
    "return_pct": ["Return %", "Net P&L %"],
}

VERSIONS = {
    "V3.4": "xauusd/XAUUSD-Long-S1-AweWithBB/S1-Awe-V3.4_FX_IDC_XAUUSD_2026-04-26.csv",
    "V3.9": "xauusd/XAUUSD-Long-S1-AweWithBB/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_column(columns: list[str], canonical: str) -> str:
    for alias in COLUMN_ALIASES[canonical]:
        if alias in columns:
            return alias
    raise KeyError(f"no column found for {canonical!r} among {columns}")


def load_daily(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("﻿").strip() for column in frame.columns]
    time_column = next(column for column in frame.columns if "time" in column.lower())
    frame[time_column] = pd.to_datetime(frame[time_column], utc=False)
    if frame[time_column].dt.tz is not None:
        frame[time_column] = frame[time_column].dt.tz_localize(None)
    return (
        frame[[time_column, "close"]]
        .rename(columns={time_column: "time", "close": value_name})
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lstrip("﻿").strip() for column in frame.columns]
    columns = list(frame.columns)
    rename = {pick_column(columns, canonical): canonical for canonical in COLUMN_ALIASES}
    frame = frame.rename(columns=rename)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    entries = frame[frame["type"] == "Entry long"][
        ["trade_id", "datetime", "price", "size_qty"]
    ].rename(columns={"datetime": "entry_time", "price": "entry_price"})
    exits = frame[frame["type"] == "Exit long"][
        ["trade_id", "datetime", "signal", "net_pnl_usd", "return_pct"]
    ].rename(columns={"datetime": "exit_time", "signal": "exit_signal"})
    trades = entries.merge(exits, on="trade_id", how="inner")
    trades = trades[trades["exit_signal"].astype(str).str.upper() != "OPEN"].copy()
    trades["win"] = trades["net_pnl_usd"] > 0
    trades["entry_hour"] = trades["entry_time"].dt.hour
    trades["entry_slot_30m"] = trades["entry_time"].dt.strftime("%H:%M")
    minute_of_day = trades["entry_time"].dt.hour * 60 + trades["entry_time"].dt.minute
    trades["session"] = np.select(
        [
            minute_of_day.between(7 * 60, 15 * 60 - 1),
            minute_of_day.between(15 * 60, 20 * 60 + 29),
            (minute_of_day >= 20 * 60 + 30) | (minute_of_day < 60),
        ],
        ["asia", "europe", "us"],
        default="overnight",
    )
    trades["year_group"] = np.where(trades["entry_time"].dt.year < 2026, "2024-2025", "2026")
    return trades.sort_values("entry_time").reset_index(drop=True)


def build_macro(paths: dict[str, Path]) -> pd.DataFrame:
    series = {
        "us10y": load_daily(paths["us10y"], "us10y"),
        "t10yie": load_daily(paths["t10yie"], "t10yie"),
        "dxy": load_daily(paths["dxy"], "dxy"),
        "vix": load_daily(paths["vix"], "vix"),
        "gold": load_daily(paths["gold"], "gold"),
    }
    base = series.pop("us10y")
    for column, frame in series.items():
        base = pd.merge_asof(
            base.sort_values("time"),
            frame.sort_values("time"),
            on="time",
            direction="backward",
            tolerance=pd.Timedelta(days=4),
        )
    base["real_rate"] = base["us10y"] - base["t10yie"]
    for column in ["real_rate", "us10y", "dxy", "vix", "gold"]:
        base[f"ma50_{column}"] = base[column].rolling(MA_LENGTH, min_periods=MA_LENGTH).mean()
    base["pt_real"] = np.where(base["real_rate"] < base["ma50_real_rate"], 2, 0)
    base["pt_10y"] = np.where(base["us10y"] < base["ma50_us10y"], 1, 0)
    base["pt_dxy"] = np.where(base["dxy"] < base["ma50_dxy"], 1, 0)
    base["pt_vix"] = np.where(base["vix"] > base["ma50_vix"], 1, 0)
    base["pt_trend"] = np.where(base["gold"] > base["ma50_gold"], 1, 0)
    required = [f"ma50_{column}" for column in ["real_rate", "us10y", "dxy", "vix", "gold"]]
    valid = base[required].notna().all(axis=1)
    base["macro_score"] = np.where(
        valid,
        base[["pt_real", "pt_10y", "pt_dxy", "pt_vix", "pt_trend"]].sum(axis=1),
        np.nan,
    )
    base["macro_verdict"] = np.select(
        [base["macro_score"] <= 2, base["macro_score"].between(3, 4), base["macro_score"] >= 5],
        ["WAIT", "NEUTRAL", "STRONG BUY"],
        default="N/A",
    )
    return base.dropna(subset=["macro_score"]).reset_index(drop=True)


def attach_macro(trades: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    left = trades.sort_values("entry_time")
    right = macro.rename(columns={"time": "macro_time"}).sort_values("macro_time")
    joined = pd.merge_asof(
        left,
        right[["macro_time", "macro_score", "macro_verdict"]],
        left_on="entry_time",
        right_on="macro_time",
        direction="backward",
        tolerance=MAX_MACRO_AGE,
    )
    return joined


def wilson_interval(wins: int, count: int, z: float = 1.96) -> list[float] | None:
    if count == 0:
        return None
    proportion = wins / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
    return [round(100 * (centre - margin / denominator), 2), round(100 * (centre + margin / denominator), 2)]


def stats(frame: pd.DataFrame) -> dict:
    count = len(frame)
    if count == 0:
        return {"n": 0, "wins": 0, "win_rate_pct": None, "win_rate_ci95_pct": None,
                "profit_factor": None, "net_pnl_usd": 0.0, "avg_pnl_usd": None,
                "avg_return_pct": None}
    wins = int(frame["win"].sum())
    gross_profit = frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
    gross_loss = abs(frame.loc[frame["net_pnl_usd"] < 0, "net_pnl_usd"].sum())
    return {
        "n": count,
        "wins": wins,
        "win_rate_pct": round(100 * wins / count, 2),
        "win_rate_ci95_pct": wilson_interval(wins, count),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "net_pnl_usd": round(float(frame["net_pnl_usd"].sum()), 2),
        "avg_pnl_usd": round(float(frame["net_pnl_usd"].mean()), 2),
        "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
    }


def grouped_stats(frame: pd.DataFrame, column: str) -> dict:
    return {str(key): stats(group) for key, group in frame.groupby(column, observed=True)}


def entry_slot_stats(frame: pd.DataFrame) -> dict:
    return {slot: stats(frame[frame["entry_slot_30m"] == slot]) for slot in ENTRY_SLOTS_30M}


def recommendation_delta(group: dict, baseline: dict, prior_n: int = 30, max_points: float = 10.0) -> float | None:
    if not group["n"] or not baseline["avg_return_pct"]:
        return None
    weight = group["n"] / (group["n"] + prior_n)
    shrunk = weight * group["avg_return_pct"] + (1 - weight) * baseline["avg_return_pct"]
    relative = (shrunk - baseline["avg_return_pct"]) / abs(baseline["avg_return_pct"])
    return round(max(-max_points, min(max_points, relative * 10)), 1)


def analyze_version(trade_path: Path, macro: pd.DataFrame) -> dict:
    trades = load_trades(trade_path)
    joined = attach_macro(trades, macro)
    matched = joined.dropna(subset=["macro_score"]).copy()
    baseline = stats(trades)
    by_entry_30m = entry_slot_stats(trades)
    by_macro = grouped_stats(matched, "macro_verdict")
    for item in by_entry_30m.values():
        item["advisory_delta"] = recommendation_delta(item, baseline, max_points=4.0)
    for item in by_macro.values():
        item["advisory_delta"] = recommendation_delta(item, baseline)
    return {
        "trade_period": {
            "start": str(trades["entry_time"].min()),
            "end": str(trades["exit_time"].max()),
        },
        "macro_coverage": {
            "matched": len(matched),
            "unmatched": len(trades) - len(matched),
            "pct": round(100 * len(matched) / len(trades), 2) if len(trades) else None,
        },
        "baseline": baseline,
        "by_session": grouped_stats(trades, "session"),
        "by_entry_30m": by_entry_30m,
        "by_macro_verdict": by_macro,
        "by_period": grouped_stats(trades, "year_group"),
    }


def compare_entry_30m(v34: dict, v39: dict) -> dict:
    comparison = {}
    for slot in ENTRY_SLOTS_30M:
        a = v34["by_entry_30m"][slot]
        b = v39["by_entry_30m"][slot]
        wr_diff = (
            round(b["win_rate_pct"] - a["win_rate_pct"], 2)
            if a["win_rate_pct"] is not None and b["win_rate_pct"] is not None
            else None
        )
        comparison[slot] = {
            "v34_n": a["n"], "v34_win_rate_pct": a["win_rate_pct"], "v34_profit_factor": a["profit_factor"],
            "v39_n": b["n"], "v39_win_rate_pct": b["win_rate_pct"], "v39_profit_factor": b["profit_factor"],
            "win_rate_pct_diff_v39_minus_v34": wr_diff,
        }
    return comparison


def markdown_report(result: dict) -> str:
    v34 = result["versions"]["V3.4"]
    v39 = result["versions"]["V3.9"]
    lines = [
        f"# {STUDY_ID} — S1 AweWithBB V3.4 vs V3.9, 30-minute entry-slot comparison",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "## Scope",
        "",
        f"- V3.4 TradingView closed trades: **{v34['baseline']['n']}**, "
        f"{v34['trade_period']['start']} to {v34['trade_period']['end']}.",
        f"- V3.9 TradingView closed trades: **{v39['baseline']['n']}**, "
        f"{v39['trade_period']['start']} to {v39['trade_period']['end']}.",
        "- Entry timestamps are TradingView export time, interpreted as Asia/Taipei.",
        "- No fail-pattern (fail_type/DXY/MTF/K-bar) breakdown; this study covers session, "
        "30-minute entry slot, and Macro context only, per the agreed scope for this rerun.",
        "- **Risk-parameter caveat**: V3.4 used 0.5% SL with TP2 = 2R; V3.9 uses a different "
        "exit structure. Win-rate/PF differences between versions "
        "may reflect exit-rule changes, not only timing/regime differences — do not read this "
        "as a pure like-for-like backtest.",
        "- V3.4 is superseded (the confirmed active version is V3.9). This study is historical "
        "comparison only; it does not reinstate V3.4 or change any active `請分析` rule.",
        "",
        "## Baseline",
        "",
        "| Version | n | WR | PF | Net |",
        "|---|---:|---:|---:|---:|",
        f"| V3.4 | {v34['baseline']['n']} | {v34['baseline']['win_rate_pct']}% | "
        f"{v34['baseline']['profit_factor']} | ${v34['baseline']['net_pnl_usd']:,.2f} |",
        f"| V3.9 | {v39['baseline']['n']} | {v39['baseline']['win_rate_pct']}% | "
        f"{v39['baseline']['profit_factor']} | ${v39['baseline']['net_pnl_usd']:,.2f} |",
        "",
        "## 30-minute entry-slot comparison (Asia/Taipei)",
        "",
        "| Slot | V3.4 n | V3.4 WR | V3.4 PF | V3.9 n | V3.9 WR | V3.9 PF | WR diff (V3.9-V3.4) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(v, suffix=""):
        return "—" if v is None else f"{v}{suffix}"
    for slot, item in result["comparison"]["by_entry_30m"].items():
        lines.append(
            f"| {slot} | {item['v34_n']} | {fmt(item['v34_win_rate_pct'], '%')} | "
            f"{fmt(item['v34_profit_factor'])} | {item['v39_n']} | "
            f"{fmt(item['v39_win_rate_pct'], '%')} | {fmt(item['v39_profit_factor'])} | "
            f"{fmt(item['win_rate_pct_diff_v39_minus_v34'], 'pp')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Both versions now report every metric at 30-minute entry-slot granularity; no broad "
        "session or hourly bucket is treated as the primary evidence for future studies.",
        "- Most 30-minute cells are low-n for both versions; treat differences as descriptive, "
        "not as a basis for a new hard rule.",
        "- V3.9 is the only version eligible to affect the live S1 advisory score "
        "(`research/studies/RS-XAUUSD-20260727-001/`). V3.4 numbers here are context only.",
        "",
        "## Source provenance",
        "",
    ]
    for source in result["sources"]:
        lines.append(f"- `{source['path']}` — SHA-256 `{source['sha256']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    legacy = args.legacy_root.resolve()
    output = args.output_dir.resolve()

    trade_paths = {version: legacy / rel for version, rel in VERSIONS.items()}
    daily_dir = legacy / "xauusd/csv"
    macro_paths = {
        "us10y": daily_dir / "TVC_US10Y, 1D.csv",
        "t10yie": daily_dir / "FRED_T10YIE, 1D.csv",
        "dxy": daily_dir / "20260711/TVC_DXY, 1D.csv",
        "vix": daily_dir / "TVC_VIX, 1D.csv",
        "gold": daily_dir / "20260711/FX_IDC_XAUUSD, 1D.csv",
    }
    source_paths = [*trade_paths.values(), *macro_paths.values()]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    macro = build_macro(macro_paths)
    versions = {version: analyze_version(path, macro) for version, path in trade_paths.items()}
    comparison = {"by_entry_30m": compare_entry_30m(versions["V3.4"], versions["V3.9"])}

    generated_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
    result = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": generated_at,
        "strategy": "S1 AweWithBB V3.4 vs V3.9",
        "method": {
            "macro_score": "real_rate<MA50 +2; US10Y<MA50 +1; DXY<MA50 +1; VIX>MA50 +1; XAUUSD>MA50 +1",
            "macro_labels": "WAIT=0-2; NEUTRAL=3-4; STRONG BUY=5-6",
            "macro_assignment": "latest prior daily observation, maximum age 4 days",
            "session_timezone": "Asia/Taipei (TradingView export-time assumption)",
            "session_buckets": "overnight=01:00-06:59; asia=07:00-14:59; europe=15:00-20:29; us=20:30-00:59 (descriptive only)",
            "primary_granularity": "30-minute entry slot (HH:00/HH:30), the standard granularity for this and future studies",
            "advisory_delta": "average-return empirical Bayes shrinkage toward each version's own baseline, prior_n=30; Macro capped ±10 and 30-minute entry slots capped ±4",
            "column_normalization": "V3.4 export uses 'Trade #'/'Net P&L USD'/'Net P&L %'; V3.9 export uses 'Trade number'/'Net PnL USD'/'Return %'. Both mapped to the same canonical schema before computation.",
        },
        "versions": versions,
        "comparison": comparison,
        "sources": [{"path": path.name, "sha256": sha256(path)} for path in source_paths],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "v34_baseline": versions["V3.4"]["baseline"],
        "v39_baseline": versions["V3.9"]["baseline"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
