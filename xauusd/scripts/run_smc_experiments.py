"""
SMC (Smart Money Concepts) strategy experiment runner.

Runs 10 long + 10 short SMC strategies on XAUUSD 30-min data and prints
a ranked comparison table. Results saved to XAUUSD-SMC-Experiments/results.json.

Usage（20260705 移至 scripts/ 子資料夾）:
    python3 xauusd/scripts/run_smc_experiments.py   # from trading/ root
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

# ── Path setup: 找出 xauusd/ 目錄（本檔可能在 xauusd/ 或 xauusd/scripts/ 下）──
_here = Path(__file__).parent
_xauusd_dir = _here if (_here / "experiments").exists() else _here.parent
sys.path.insert(0, str(_xauusd_dir))

from analysis.config import PRICE_CSV, PRICE_CSV_4H
from analysis import loader
from analysis.mtf_analysis import prepare_htf_filter
from experiments.engine import run_backtest, run_backtest_short, summary, Trade
from experiments.runner import score
from experiments.smc_indicators import precompute, describe
from experiments.strategies_smc import make_strategies

OUT_DIR = _xauusd_dir / "XAUUSD-SMC-Experiments"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")
    print(f"  {'ID':<4} {'Group':<7} {'Description':<40} {'N':>4} {'WR':>6} {'PF':>6} {'PnL%':>6}")
    print(f"  {'─'*4} {'─'*7} {'─'*40} {'─'*4} {'─'*6} {'─'*6} {'─'*6}")
    for r in rows:
        n   = r["total"]
        wr  = f"{r['win_rate']:.1%}"   if n > 0 else "  N/A"
        pf  = f"{r['profit_factor']:.3f}" if n > 0 else "  N/A"
        pnl = f"{r['net_pnl_pct']:+.2f}" if n > 0 else "  N/A"
        print(f"  {r['id']:<4} {r['group']:<7} {r['description']:<40} {n:>4} {wr:>6} {pf:>6} {pnl:>6}")


def _run_all(price, strategies, engine_fn, htf_filter=None):
    rows = []
    trades_map = {}
    for sid, (fn, group, desc) in strategies.items():
        trades = engine_fn(price, fn, htf_filter=htf_filter)
        s = summary(trades)
        trades_map[sid] = trades
        rows.append({"id": sid, "group": group, "description": desc, **s})
    return rows, trades_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  XAUUSD SMC Strategy Experiments  (FVG / Order Block / Liq. Sweep)")
    print("=" * 72)

    # ── 1. Load price data ────────────────────────────────────────────────
    print("\n[1/4] Loading price data...")
    if not PRICE_CSV.exists():
        print(f"  ERROR: {PRICE_CSV} not found.")
        sys.exit(1)

    price = loader.load_price(PRICE_CSV)
    print(f"  30m : {len(price)} bars  "
          f"({price['time'].min().date()} → {price['time'].max().date()})")

    xau_4h = loader.load_price(PRICE_CSV_4H) if PRICE_CSV_4H.exists() else None
    htf_filter = prepare_htf_filter(xau_4h) if xau_4h is not None else None
    if xau_4h is not None:
        print(f"  4H  : {len(xau_4h)} bars  (HTF filter enabled)")
    else:
        print("  4H  : not found — HTF filter disabled")

    # ── 2. Pre-compute SMC arrays ─────────────────────────────────────────
    print("\n[2/4] Pre-computing SMC arrays...")
    smc = precompute(price)
    describe(smc, price)

    STRATEGIES_LONG, STRATEGIES_SHORT = make_strategies(smc)

    # ── 3. Run backtests ──────────────────────────────────────────────────
    print("\n[3/4] Running backtests...")

    long_rows,  long_trades  = _run_all(price, STRATEGIES_LONG,  run_backtest,       htf_filter)
    short_rows, short_trades = _run_all(price, STRATEGIES_SHORT, run_backtest_short, htf_filter)

    long_df  = pd.DataFrame(long_rows).set_index("id")
    short_df = pd.DataFrame(short_rows).set_index("id")

    scored_long  = score(long_df)
    scored_short = score(short_df)

    # ── 4. Print results ──────────────────────────────────────────────────
    _print_table(long_rows,  "LONG  strategies  (M01–M10)  [SL 0.5% / TP 1.0%  R:R 2:1]")
    _print_table(short_rows, "SHORT strategies  (M11–M20)  [SL 0.5% / TP 1.0%  R:R 2:1]")

    print(f"\n{'─' * 72}")
    print("  TOP 5 LONG (composite score)")
    print(f"{'─' * 72}")
    for rank, (sid, row) in enumerate(scored_long.head(5).iterrows(), 1):
        n  = int(row["total"])
        wr = f"{row['win_rate']:.1%}"
        pf = f"{row['profit_factor']:.3f}"
        print(f"  #{rank}  {sid}  {row['description']:<40}  N={n:>3}  WR={wr}  PF={pf}  score={row['score']:.3f}")

    print(f"\n{'─' * 72}")
    print("  TOP 5 SHORT (composite score)")
    print(f"{'─' * 72}")
    for rank, (sid, row) in enumerate(scored_short.head(5).iterrows(), 1):
        n  = int(row["total"])
        wr = f"{row['win_rate']:.1%}"
        pf = f"{row['profit_factor']:.3f}"
        print(f"  #{rank}  {sid}  {row['description']:<40}  N={n:>3}  WR={wr}  PF={pf}  score={row['score']:.3f}")

    # ── 5. Key insights ───────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  SMC INSIGHT SUMMARY")
    print(f"{'─' * 72}")

    for label, df in [("LONG", scored_long), ("SHORT", scored_short)]:
        viable = df[(df["total"] >= 10) & (df["profit_factor"] > 1.0)]
        if not viable.empty:
            best = viable.index[0]
            r = viable.loc[best]
            print(f"  [{label}] Best viable: {best} — {r['description']}")
            print(f"         N={int(r['total'])}  WR={r['win_rate']:.1%}  PF={r['profit_factor']:.3f}  PnL={r['net_pnl_pct']:+.2f}%")
        else:
            print(f"  [{label}] No strategy with ≥10 trades and PF > 1.0 found.")

    # ── 6. Save results.json ──────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "results.json"

    def _to_records(df: pd.DataFrame, direction: str) -> list[dict]:
        records = []
        for rank, (sid, row) in enumerate(df.iterrows(), 1):
            records.append({
                "rank":          rank,
                "id":            sid,
                "direction":     direction,
                "group":         str(row.get("group", "")),
                "description":   str(row.get("description", "")),
                "n_trades":      int(row.get("total", 0)),
                "win_rate":      round(float(row.get("win_rate", 0)) * 100, 1),
                "profit_factor": round(min(float(row.get("profit_factor", 0)), 99.0), 3),
                "net_pnl_pct":   round(float(row.get("net_pnl_pct", 0)), 2),
                "avg_hold_bars": round(float(row.get("avg_hold_bars", 0)), 1),
                "max_consec_loss": int(row.get("max_consec_loss", 0)),
                "score":         round(float(row.get("score", 0)), 3),
            })
        return records

    payload = {
        "params": {
            "sl_pct":        0.005,
            "tp_pct":        0.010,
            "max_hold_bars": 48,
            "fvg_lookback":  100,
            "ob_confirm_bars": 12,
            "ob_move_pct":   0.003,
            "ob_lookback":   60,
            "liq_lookback":  30,
        },
        "long":  _to_records(scored_long,  "long"),
        "short": _to_records(scored_short, "short"),
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved → {out}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
