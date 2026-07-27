#!/usr/bin/env python3
"""Run all 20 short strategies on MTX 30m data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.runner import run_all_short
from experiments.report import generate
from experiments.pine_generator_short import generate_all

OUTPUT_DIR = Path(__file__).parent / "TX-Short-Experiments"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="TAIFEX_DLY_MXF1!, 30.csv")
    p.add_argument("--sl", type=int, default=30)
    p.add_argument("--tp", type=int, default=60)
    args = p.parse_args()

    print(f"Loading {args.csv}  SL={args.sl}pts  TP={args.tp}pts")
    results = run_all_short(csv_file=args.csv, sl_pts=args.sl, tp_pts=args.tp)

    print("\n=== Top 5 Short Strategies ===")
    for r in results[:5]:
        print(f"  [{r.get('rank','-')}] {r['code']} {r['name']:30s}  "
              f"WR={r['win_rate']}%  PF={r['profit_factor']:.3f}  "
              f"PnL=NT${r['net_pnl_ntd']:,.0f}  Score={r.get('score',0):.3f}")

    generate(OUTPUT_DIR, direction="short")
    generate_all()
    print(f"\n✓ Done.  Open {OUTPUT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
