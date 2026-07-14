"""
Entry point: full fail-pattern analysis for all XAUUSD strategies.

Usage:
    python3.12 main.py                  # all strategies
    python3.12 main.py S1-AweWithBB     # single strategy by id

Output per strategy:
    reports/<id>/          PNG charts (reference copies)
    <strategy-folder>/report.html   self-contained HTML report
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from analysis.config import (
    STRATEGIES, PRICE_CSV, PRICE_CSV_60M, PRICE_CSV_4H,
    DXY_CSV_1D, DXY_CSV_30, XAUUSD_CSV_1D,
)
from analysis import loader, metrics, fail_patterns, pre_entry, charts, report, dxy_analysis, mtf_analysis

REPORTS = Path(__file__).parent / "reports"


def _load_all_data():
    """Load all price and DXY data; returns a data dict."""
    return {
        "dxy_1d":   loader.load_dxy(DXY_CSV_1D)       if DXY_CSV_1D.exists()    else None,
        "dxy_30":   loader.load_dxy(DXY_CSV_30)       if DXY_CSV_30.exists()    else None,
        "xau_1d":   loader.load_price(XAUUSD_CSV_1D)  if XAUUSD_CSV_1D.exists() else None,
        "xau_60m":  loader.load_price(PRICE_CSV_60M)  if PRICE_CSV_60M.exists() else None,
        "xau_4h":   loader.load_price(PRICE_CSV_4H)   if PRICE_CSV_4H.exists()  else None,
    }


def run_strategy(cfg: dict, data: dict | None = None) -> None:
    dxy_1d    = (data or {}).get("dxy_1d")
    dxy_30    = (data or {}).get("dxy_30")
    xauusd_1d = (data or {}).get("xau_1d")
    xau_60m   = (data or {}).get("xau_60m")
    xau_4h    = (data or {}).get("xau_4h")
    name    = cfg["id"]
    version = cfg["version"]
    csv_path = cfg["folder"] / cfg["trades_csv"]

    print(f"\n{'=' * 60}")
    print(f"  {name}  v{version}")
    print(f"{'=' * 60}")

    # ── Load & basic metrics ──────────────────────────────────────
    trades = loader.load_trades(csv_path)
    summ   = metrics.summary(trades)

    print(f"  Trades          : {summ['total_trades']}")
    print(f"  Win Rate        : {summ['win_rate']:.1%}")
    print(f"  Profit Factor   : {summ['profit_factor']:.2f}")
    print(f"  Net P&L         : ${summ['net_pnl']:,.2f}")
    print(f"  Max Drawdown    : ${metrics.max_drawdown(trades):,.2f}")
    print(f"  Avg Hold        : {summ['avg_hold_bars']:.1f} bars")
    print(f"  Max Consec Loss : {summ['max_consec_losses']}")

    # ── Fail pattern classification ───────────────────────────────
    classified = fail_patterns.classify_fail(trades)
    print(f"\n  --- Fail Patterns ({len(classified)} losses) ---")
    print(fail_patterns.fail_type_summary(classified).to_string())

    sess = fail_patterns.session_stats(trades)
    print(f"\n  --- Session Stats ---")
    print(sess[["total", "wins", "win_rate", "avg_pnl"]].to_string())

    print(f"\n  --- Fail by Session ---")
    print(fail_patterns.fail_by_session(classified).to_string())

    # ── Pre-entry analysis (immediate_loss) ───────────────────────
    imm_losses = classified[classified["fail_type"] == "immediate_loss"].copy()
    profile, enriched = {}, pd.DataFrame()

    if len(imm_losses):
        trades_ctx = pre_entry.add_trade_context(
            trades.merge(classified[["trade_id", "fail_type"]], on="trade_id", how="left")
        )
        profile = pre_entry.immediate_loss_profile(trades_ctx)

        print(f"\n  --- Immediate Loss: Entry Hour ---")
        print(profile["entry_hour"].to_string())

        price    = loader.load_price(PRICE_CSV)
        enriched = pre_entry.enrich_with_kbars(imm_losses, price)
        cov      = pre_entry.kbar_coverage(enriched)
        print(f"\n  --- K-Bar Coverage: {cov['with_kbar_data']}/{cov['total_losses']} "
              f"({cov['coverage_pct']}%) ---")
        if cov["with_kbar_data"] > 0:
            cols = ["trade_id", "entry_time", "rsi", "rsi_vs_ma",
                    "prev_1_dir", "prev_3_green", "momentum_3"]
            print(enriched.dropna(subset=["rsi"])[cols].to_string())

    # ── BB position analysis ──────────────────────────────────────
    from analysis.bb_analysis import enrich_trades_with_bb, bb_stats
    bb_enriched = enrich_trades_with_bb(trades, loader.load_price(PRICE_CSV))
    bb_stats_out = bb_stats(bb_enriched)
    print(f"\n  --- BB Position Analysis ---")
    print(bb_stats_out["by_zone"][["total", "win_rate", "avg_pnl"]].to_string())

    # ── Divergence analysis ───────────────────────────────────────
    from analysis.divergence import enrich_trades_with_divergence, divergence_stats
    div_enriched = enrich_trades_with_divergence(trades, loader.load_price(PRICE_CSV), lookback_bars=12)
    div_stats_out = divergence_stats(div_enriched)
    print(f"\n  --- Pre-entry Divergence (lookback 12 bars = 6h) ---")
    if not div_stats_out["by_bull_div"].empty:
        print("  Bull Div:", div_stats_out["by_bull_div"][["total","win_rate"]].to_string())

    # ── Multi-Timeframe analysis ──────────────────────────────────
    htf_enriched_out, htf_stats_result = None, None
    if xau_4h is not None or xauusd_1d is not None:
        print(f"\n  --- Multi-Timeframe Analysis ---")
        trades_with_fail = trades.merge(
            classified[["trade_id", "fail_type"]], on="trade_id", how="left"
        )
        htf_enriched_out = mtf_analysis.enrich_trades_with_htf(
            trades_with_fail,
            price_60m=xau_60m,
            price_4h=xau_4h,
            price_1d=xauusd_1d,
        )
        htf_stats_result = mtf_analysis.htf_stats(htf_enriched_out)

        if "coverage" in htf_stats_result:
            print(htf_stats_result["coverage"].to_string())
        if "by_alignment" in htf_stats_result and not htf_stats_result["by_alignment"].empty:
            print(f"\n  Win Rate by HTF Alignment:")
            print(htf_stats_result["by_alignment"].to_string())
        if "by_4h_state" in htf_stats_result and not htf_stats_result["by_4h_state"].empty:
            print(f"\n  Win Rate by 4H State:")
            print(htf_stats_result["by_4h_state"].to_string())
        if "by_4h_bucket" in htf_stats_result and not htf_stats_result["by_4h_bucket"].empty:
            print(f"\n  Win Rate by 4H RSI Bucket:")
            print(htf_stats_result["by_4h_bucket"].to_string())
        if "fail_by_4h" in htf_stats_result and not htf_stats_result["fail_by_4h"].empty:
            print(f"\n  Fail Type % by 4H State:")
            print(htf_stats_result["fail_by_4h"].to_string())

    # ── DXY analysis ──────────────────────────────────────────────
    dxy_stats_out, corr_df_out = None, None
    if dxy_1d is not None:
        enriched_dxy = dxy_analysis.enrich_trades_with_dxy(trades, dxy_1d, dxy_30)
        dxy_stats_out = dxy_analysis.dxy_regime_stats(enriched_dxy)
        print(f"\n  --- DXY Win Rate by RSI Bucket ---")
        print(dxy_stats_out["by_bucket"].to_string())
        print(f"\n  --- DXY Win Rate by Trend ---")
        print(dxy_stats_out["by_trend"].to_string())
        if xauusd_1d is not None:
            corr_df_out = dxy_analysis.dxy_correlation_stats(xauusd_1d, dxy_1d)
            avg_c = corr_df_out["rolling_corr"].dropna().mean()
            print(f"\n  --- DXY×XAUUSD Avg 30D Correlation: {avg_c:.3f} ---")

    # ── PNG charts (reference folder) ────────────────────────────
    out_dir = REPORTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if htf_stats_result:
        for fname, fig in [
            ("htf_alignment.png",  charts.htf_alignment_bar(htf_stats_result, name)),
            ("htf_4h_state.png",   charts.htf_4h_state_bar(htf_stats_result, name)),
            ("htf_4h_bucket.png",  charts.htf_bucket_heatmap(htf_stats_result, name)),
        ]:
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt; plt.close(fig)

    png_jobs = [
        ("equity_curve.png",    charts.equity_curve(trades, name)),
        ("fail_types.png",      charts.fail_type_breakdown(classified, name)),
        ("mfe_dist.png",        charts.mfe_distribution(classified, name)),
        ("mae_vs_mfe.png",      charts.mae_vs_mfe_scatter(classified, name)),
        ("session_heatmap.png", charts.session_heatmap(sess, name)),
        ("hourly_winrate.png",  charts.hourly_winrate(fail_patterns.hourly_stats(trades), name)),
        ("consec_losses.png",   charts.consecutive_loss_hist(metrics.consecutive_losses(trades), name)),
        ("hold_time_dist.png",  charts.hold_time_dist(trades, name)),
    ]
    if profile:
        png_jobs += [
            ("pre_entry_hour.png",  charts.pre_entry_hour(profile, name)),
            ("pre_entry_dow.png",   charts.pre_entry_dow(profile, name)),
            ("pre_entry_prev.png",  charts.pre_entry_prev_result(profile, name)),
            ("pre_entry_tsw.png",   charts.pre_entry_tsw(profile, name)),
            ("kbar_features.png",   charts.kbar_feature_summary(enriched, name)),
        ]

    for filename, fig in png_jobs:
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  PNG charts → {out_dir}/")

    # ── HTML report → strategy folder ────────────────────────────
    html_path = cfg["folder"] / "report.html"
    report.generate(
        strategy_id=name,
        version=version,
        trades=trades,
        classified=classified,
        sess_stats=sess,
        profile=profile,
        enriched=enriched,
        out_path=html_path,
        dxy_stats=dxy_stats_out,
        corr_df=corr_df_out,
        htf_enriched=htf_enriched_out,
        htf_stats_out=htf_stats_result,
        bb_stats_out=bb_stats_out,
        div_stats_out=div_stats_out,
    )


def main() -> None:
    target_ids = set(sys.argv[1:])
    strategies = [
        cfg for cfg in STRATEGIES
        if not target_ids or cfg["id"] in target_ids
    ]
    if not strategies:
        available = [c["id"] for c in STRATEGIES]
        print(f"No match. Available: {available}")
        sys.exit(1)

    data = _load_all_data()
    for label, key in [("DXY 1D", "dxy_1d"), ("DXY 30m", "dxy_30"),
                        ("XAUUSD 1D", "xau_1d"), ("XAUUSD 60m", "xau_60m"), ("XAUUSD 4H", "xau_4h")]:
        df = data.get(key)
        if df is not None:
            print(f"[{label}] {len(df)} bars  "
                  f"({df['time'].min().date()} → {df['time'].max().date()})")

    for cfg in strategies:
        try:
            run_strategy(cfg, data=data)
        except FileNotFoundError as exc:
            print(f"[SKIP] {cfg['id']}: {exc}", file=sys.stderr)

    print(f"\nDone. HTML reports in each strategy folder.")


if __name__ == "__main__":
    main()
