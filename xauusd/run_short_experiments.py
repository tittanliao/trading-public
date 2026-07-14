"""
Entry point for the 20-strategy short experiment.

Steps:
    1. Load 30-min price data + 4H HTF data
    2. Run baseline backtests (no filter) + HTF-filtered backtests (skip 4H bullish)
    3. Score and rank strategies
    4. Generate HTML report with MTF filter comparison + per-strategy 4H analysis
    5. Generate Pine Script files

Usage:
    python3.12 run_short_experiments.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from analysis.config import PRICE_CSV, PRICE_CSV_4H, XAUUSD_CSV_1D
from analysis import loader
from analysis.mtf_analysis import prepare_htf_filter
from experiments import runner
from experiments.report import generate as generate_html
from experiments.pine_generator_short import generate_all as generate_pine

OUT_DIR  = Path(__file__).parent / "XAUUSD-Short-Experiments"
HTML_OUT = OUT_DIR / "report.html"


def _build_comparison(base: pd.DataFrame, filt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in base.index:
        b, f = base.loc[sid], filt.loc[sid]
        rows.append({
            "strategy_id":       sid,
            "win_rate_base":     b["win_rate"],
            "win_rate_filtered": f["win_rate"],
            "trades_base":       b["total"],
            "trades_filtered":   f["total"],
            "pf_base":           min(b["profit_factor"], 99),
            "pf_filtered":       min(f["profit_factor"], 99),
        })
    return pd.DataFrame(rows).set_index("strategy_id")


def main() -> None:
    print("=" * 60)
    print("  XAUUSD 20-Strategy Short Experiment")
    print("=" * 60)

    # ── Load price data ───────────────────────────────────────────
    print(f"\n[1/5] Loading price data...")
    price = loader.load_price(PRICE_CSV)
    print(f"      30m: {len(price)} bars  "
          f"({price['time'].min().date()} → {price['time'].max().date()})")

    if len(price) < 60:
        print("ERROR: Need at least 60 bars.")
        sys.exit(1)

    xau_4h = loader.load_price(PRICE_CSV_4H)  if PRICE_CSV_4H.exists()  else None
    xau_1d = loader.load_price(XAUUSD_CSV_1D) if XAUUSD_CSV_1D.exists() else None
    htf_filter = prepare_htf_filter(xau_4h) if xau_4h is not None else None

    if xau_4h is not None:
        print(f"      4H : {len(xau_4h)} bars  "
              f"({xau_4h['time'].min().date()} → {xau_4h['time'].max().date()})")
    else:
        print("      4H : not found — HTF filter disabled")

    # ── Baseline backtests ────────────────────────────────────────
    print("\n[2/5] Running baseline short backtests (no HTF filter)...")
    results_base, trades_map = runner.run_all_short(price)
    scored = runner.score(results_base)

    # ── HTF-filtered backtests ────────────────────────────────────
    mtf_comparison = None
    if htf_filter is not None:
        print("\n[3/5] Running HTF-filtered backtests (skip 4H bullish)...")
        results_filt, _ = runner.run_all_short(price, htf_filter=htf_filter)
        mtf_comparison = _build_comparison(results_base, results_filt)

        avg_delta = (
            mtf_comparison["win_rate_filtered"] - mtf_comparison["win_rate_base"]
        ).mean()
        improved = (
            mtf_comparison["win_rate_filtered"] > mtf_comparison["win_rate_base"]
        ).sum()
        print(f"      Avg ΔWin Rate: {avg_delta:+.1%}  |  "
              f"{improved}/{len(mtf_comparison)} strategies improved")
    else:
        print("\n[3/5] Skipping HTF filter (no 4H data)")

    # ── Print ranking ─────────────────────────────────────────────
    print("\n  --- Strategy Ranking (baseline) ---")
    display_cols = ["group", "total", "win_rate", "profit_factor",
                    "net_pnl_pct", "max_consec_loss", "score"]
    print(scored[display_cols].to_string(float_format="{:.3f}".format))

    best = scored.index[0]
    print(f"\n  [BEST] {best}  (score={scored.loc[best,'score']:.3f})")

    # ── HTML report ───────────────────────────────────────────────
    print("\n[4/5] Generating HTML report...")
    OUT_DIR.mkdir(exist_ok=True)
    generate_html(
        scored, trades_map, price, HTML_OUT,
        mtf_comparison=mtf_comparison,
        price_4h=xau_4h,
        price_1d=xau_1d,
        direction="short",
    )

    # ── Save results.json ─────────────────────────────────────────
    results_list = []
    for rank_i, (sid, row) in enumerate(scored.iterrows(), start=1):
        results_list.append({
            "rank":           rank_i,
            "code":           sid,
            "name":           str(row.get("description", sid)),
            "group":          str(row.get("group", "")),
            "n_trades":       int(row.get("total", 0)),
            "win_rate":       round(float(row.get("win_rate", 0)) * 100, 1),
            "profit_factor":  round(float(row.get("profit_factor", 0)), 3),
            "net_pnl_pct":    round(float(row.get("net_pnl_pct", 0)) * 100, 2),
            "score":          round(float(row.get("score", 0)), 3),
        })
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({"direction": "short", "results": results_list}, f,
                  ensure_ascii=False, indent=2)

    # ── Pine Script files ─────────────────────────────────────────
    print("\n[5/5] Generating Pine Script files...")
    generate_pine()

    print(f"\n{'=' * 60}")
    print(f"  Done.")
    print(f"  HTML report  : {HTML_OUT}")
    print(f"  Pine scripts : {OUT_DIR / 'pine'}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
