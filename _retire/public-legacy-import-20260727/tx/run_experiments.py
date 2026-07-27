#!/usr/bin/env python3
"""
Run all 20 long strategies on MTX 30m data.

Usage:
    py -3.11 run_experiments.py
    py -3.11 run_experiments.py --sl 40 --tp 80
    py -3.11 run_experiments.py --csv "TAIFEX_MXF1!, 30.csv"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.runner import run_all_long
from experiments.report import generate
from experiments.pine_generator import generate_all

OUTPUT_DIR = Path(__file__).parent / "TX-Long-Experiments"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="TAIFEX_DLY_MXF1!, 30.csv")
    p.add_argument("--sl", type=int, default=30)
    p.add_argument("--tp", type=int, default=60)
    args = p.parse_args()

    print(f"Loading {args.csv}  SL={args.sl}pts  TP={args.tp}pts")
    results = run_all_long(csv_file=args.csv, sl_pts=args.sl, tp_pts=args.tp)

    print("\n=== Top 5 Long Strategies ===")
    for r in results[:5]:
        print(f"  [{r.get('rank','-')}] {r['code']} {r['name']:30s}  "
              f"WR={r['win_rate']}%  PF={r['profit_factor']:.3f}  "
              f"PnL=NT${r['net_pnl_ntd']:,.0f}  Score={r.get('score',0):.3f}")

    generate(OUTPUT_DIR, direction="long")
    generate_all()
    print(f"\n✓ Done.  Open {OUTPUT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
