#!/usr/bin/env python3
"""Build a single-strategy-version fail-pattern report using the project's standard
30-minute-granularity contract. Despite the filename (kept for
provenance continuity with RS-XAUUSD-20260727-003/-004), STUDY_CONFIG covers any
strategy/market, not only S1.

Selector is `--study-id`, not `--version`: two studies can share a version (e.g.
RS-XAUUSD-20260727-001 and -004 are both S1 V3.9), so version alone is ambiguous.

Usage:
    /opt/homebrew/bin/python3.12 scripts/research/build_s1_fail_pattern_solo.py --study-id RS-XAUUSD-20260727-003
    /opt/homebrew/bin/python3.12 scripts/research/build_s1_fail_pattern_solo.py --study-id RS-XAUUSD-20260727-004
    /opt/homebrew/bin/python3.12 scripts/research/build_s1_fail_pattern_solo.py --study-id RS-XAUUSD-20260727-006
    /opt/homebrew/bin/python3.12 scripts/research/build_s1_fail_pattern_solo.py --study-id RS-XAUUSD-20260727-007
    /opt/homebrew/bin/python3.12 scripts/research/build_s1_fail_pattern_solo.py --study-id RS-XAUUSD-20260727-001

Writes results.json, report.html, README.md, and charts/*.png into the study's
research/studies/<id>/ directory. study.json/source_manifest.json/decision_log.md/
handoff.md are authored separately (they carry decision context this runner
does not have).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import fail_pattern_toolkit as tk  # noqa: E402
import temporal_stability_toolkit as tst  # noqa: E402

DEFAULT_LEGACY = Path("../trading")

STUDY_CONFIG = {
    "RS-XAUUSD-20260727-001": {
        "version": "V3.9",
        "strategy_id": "S1-AweWithBB",
        "trade_csv": "xauusd/XAUUSD-Long-S1-AweWithBB/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv",
        "with_macro": True,
        "policy_impact": True,
        "with_temporal_stability": True,
    },
    "RS-XAUUSD-20260727-003": {
        "version": "V3.4",
        "strategy_id": "S1-AweWithBB",
        "trade_csv": "xauusd/XAUUSD-Long-S1-AweWithBB/S1-Awe-V3.4_FX_IDC_XAUUSD_2026-04-26.csv",
        "with_macro": False,
        "policy_impact": False,
    },
    "RS-XAUUSD-20260727-004": {
        "version": "V3.9",
        "strategy_id": "S1-AweWithBB",
        "trade_csv": "xauusd/XAUUSD-Long-S1-AweWithBB/S1-Awe-V3.9_FX_IDC_XAUUSD_2026-07-11.csv",
        "with_macro": False,
        "policy_impact": False,
    },
    "RS-XAUUSD-20260727-006": {
        "version": "V1.9",
        "strategy_id": "S2-Hammer",
        "trade_csv": "xauusd/XAUUSD-Long-S2-Hammer/S2-Pullback-V1.9_FX_IDC_XAUUSD_2026-07-11.csv",
        "with_macro": False,
        "policy_impact": False,
    },
    "RS-XAUUSD-20260727-007": {
        "version": "V3.2",
        "strategy_id": "S2-Hammer",
        "trade_csv": "xauusd/XAUUSD-Long-S2-Hammer/S2-Hammer-V3.2_FX_IDC_XAUUSD_2026-07-11.csv",
        "with_macro": False,
        "policy_impact": False,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9; color:#2c3e50; margin:0; padding:0; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 1.8em; margin-bottom: 4px; }
  h2 { font-size: 1.2em; border-bottom: 2px solid #3498db; padding-bottom: 4px; margin-top: 36px; color: #2980b9; }
  .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 24px; }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .kpi { background: white; border-radius: 10px; padding: 16px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.1); min-width: 140px; flex: 1; }
  .kpi-label { font-size: .8em; color: #7f8c8d; text-transform: uppercase; letter-spacing:.05em; }
  .kpi-value { font-size: 1.6em; font-weight: 700; margin-top: 2px; }
  .pos { color: #27ae60; } .neg { color: #e74c3c; } .neu { color: #2980b9; }
  .card { background: white; border-radius: 10px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-top: 20px; }
  .tbl { border-collapse: collapse; width: 100%; font-size: .85em; }
  .tbl th { background: #ecf0f1; padding: 6px 10px; text-align: left; }
  .tbl td { padding: 5px 10px; border-top: 1px solid #ecf0f1; }
  .note { background:#fef9e7; border-left:4px solid #f1c40f; padding:10px 16px; border-radius:4px; margin-top:12px; font-size:.88em; color:#7d6608; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media(max-width:700px){ .chart-grid { grid-template-columns: 1fr; } }
  footer { text-align:center; color:#bdc3c7; font-size:.8em; margin-top:40px; padding-top:16px; border-top:1px solid #ecf0f1; }
</style>
"""


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0;">'


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _stats_rows(stats_dict: dict, order: list[str] | None = None) -> list[list]:
    keys = order if order else list(stats_dict.keys())
    rows = []
    for key in keys:
        v = stats_dict.get(key)
        if not v or not v.get("n"):
            continue
        row = [key, v["n"], f'{v["win_rate_pct"]}%', v["profit_factor"] or "-", f'${v["net_pnl_usd"]:,.2f}']
        if "rank_score" in v:
            marker = " (low-sample)" if v.get("low_sample") else ""
            row.append(f'{v["rank_score"]:+d}{marker}')
        rows.append(row)
    return rows


def build(study_id: str, legacy_root: Path, output_dir: Path) -> dict:
    cfg = STUDY_CONFIG[study_id]
    version = cfg["version"]
    strategy_id = cfg["strategy_id"]
    market = study_id.split("-")[1]
    with_macro = cfg["with_macro"]
    legacy = legacy_root.resolve()
    trade_path = legacy / cfg["trade_csv"]
    csv_dir = legacy / "xauusd/csv/20260711"
    csv_root = legacy / "xauusd/csv"
    price_paths = {
        "price_30m": csv_dir / "FX_IDC_XAUUSD, 30.csv",
        "price_60m": csv_dir / "FX_IDC_XAUUSD, 60.csv",
        "price_4h": csv_dir / "FX_IDC_XAUUSD, 240.csv",
        "price_1d": csv_dir / "FX_IDC_XAUUSD, 1D.csv",
        "dxy_1d": csv_dir / "TVC_DXY, 1D.csv",
    }
    macro_only_paths = {
        "macro_us10y": csv_root / "TVC_US10Y, 1D.csv",
        "macro_t10yie": csv_root / "FRED_T10YIE, 1D.csv",
        "macro_vix": csv_root / "TVC_VIX, 1D.csv",
    }
    source_paths = {"trades": trade_path, **price_paths}
    if with_macro:
        source_paths.update(macro_only_paths)
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    trades = tk.load_trades(trade_path)
    price_30m, rsi_prov_30m = tk.load_price_csv(price_paths["price_30m"])
    price_60m, rsi_prov_60m = tk.load_price_csv(price_paths["price_60m"])
    price_4h, rsi_prov_4h = tk.load_price_csv(price_paths["price_4h"])
    price_1d, rsi_prov_1d = tk.load_price_csv(price_paths["price_1d"])
    dxy_1d, rsi_prov_dxy = tk.load_price_csv(price_paths["dxy_1d"])

    baseline = tk.summary(trades)
    classified = tk.classify_fail(trades)
    fail_summary = tk.fail_type_summary(classified)
    session_stats = tk.grouped_stats(trades, "session")
    fail_by_session = tk.fail_by_session(classified)
    entry_30m = tk.entry_slot_30m_stats(trades)

    ctx = tk.add_trade_context(trades, classified)
    profile = tk.immediate_loss_profile(ctx)
    kbar_enriched = tk.enrich_with_kbars(classified, price_30m)
    kbar_cov = tk.kbar_coverage(kbar_enriched)

    bb_enriched = tk.enrich_trades_with_bb(trades, price_30m)
    bb_stats_out = tk.bb_stats(bb_enriched)

    dxy_enriched = tk.enrich_trades_with_dxy(trades, dxy_1d)
    dxy_stats_out = tk.dxy_regime_stats(dxy_enriched)
    corr_df = tk.dxy_correlation_stats(price_1d, dxy_1d)
    avg_corr = round(float(corr_df["rolling_corr"].dropna().mean()), 3)

    htf_enriched = tk.enrich_trades_with_htf(trades, price_60m, price_4h, price_1d)
    htf_stats_out = tk.htf_stats(htf_enriched)

    streaks = tk.consecutive_losses(trades)

    temporal_stability_block = None
    if cfg.get("with_temporal_stability"):
        by_period = tst.quarterly_bucket_stats(trades)
        holdout = tst.chronological_holdout(trades, split_ratio=0.7)
        temporal_stability_block = {
            "by_period": by_period,
            "holdout_split": holdout,
            "degradation_flag": tst.degradation_flag(holdout),
        }

    macro_block = None
    if with_macro:
        macro = tk.build_macro_composite({
            "us10y": macro_only_paths["macro_us10y"],
            "t10yie": macro_only_paths["macro_t10yie"],
            "dxy": price_paths["dxy_1d"],
            "vix": macro_only_paths["macro_vix"],
            "gold": price_paths["price_1d"],
        })
        macro_joined = tk.attach_macro(trades, macro)
        macro_matched = macro_joined.dropna(subset=["macro_score"])
        by_macro_verdict = tk.grouped_stats(macro_matched, "macro_verdict")
        by_macro_score = tk.grouped_stats(macro_matched.assign(macro_score=macro_matched["macro_score"].astype(int)), "macro_score")
        macro_session_interaction = {
            f"{verdict}|{session}": tk.stats(group)
            for (verdict, session), group in macro_matched.groupby(["macro_verdict", "session"], observed=True)
        }
        macro_block = {
            "macro_period": {"start": str(macro["time"].min()), "end": str(macro["time"].max())},
            "macro_coverage": tk.macro_coverage(macro_joined),
            "by_macro_verdict": by_macro_verdict,
            "by_macro_score": by_macro_score,
            "macro_session_interaction": macro_session_interaction,
        }

    # ---- Live-impact rank scores (spec section 13.2) — policy-impacting studies only ----
    if cfg["policy_impact"]:
        tk.compute_rank_scores(entry_30m, ordered_keys=tk.ENTRY_SLOTS_30M)
        tk.compute_rank_scores(session_stats, ordered_keys=["asia", "europe", "us"])
        tk.mark_descriptive_only(session_stats, [k for k in ["overnight"] if k in session_stats])
        if macro_block is not None:
            tk.compute_rank_scores(macro_block["by_macro_verdict"], ordered_keys=list(macro_block["by_macro_verdict"].keys()))

    # ---- Charts ----
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts_manifest = []

    def save(fig, chart_id, title, section):
        path = charts_dir / f"{chart_id}.png"
        path.write_bytes(tk.fig_to_png_bytes(fig))
        charts_manifest.append({"id": chart_id, "file": f"{chart_id}.png", "title": title, "section": section})

    save(tk.chart_equity_curve(trades, strategy_id, version), "equity_curve", "Equity Curve & Drawdown", "performance")
    save(tk.chart_fail_type_breakdown(classified, strategy_id, version), "fail_type_breakdown", "Fail Type Breakdown", "fail_pattern")
    save(tk.chart_mfe_distribution(classified, strategy_id, version), "mfe_distribution", "MFE% Distribution", "fail_pattern")
    save(tk.chart_mae_vs_mfe(classified, strategy_id, version), "mae_vs_mfe", "MAE vs MFE", "fail_pattern")
    save(tk.chart_entry_slot_30m_winrate(entry_30m, strategy_id, version), "entry_slot_30m_winrate", "Win Rate by 30-Minute Entry Slot", "timing_30m")
    save(tk.chart_pre_entry_slot_30m(profile["entry_slot_30m"], strategy_id, version), "immediate_loss_by_slot_30m", "Immediate Loss vs All Trades by 30-Min Slot", "timing_30m")
    save(tk.chart_pre_entry_categorical(profile["entry_dow"], "Day-of-Week", strategy_id, version), "pre_entry_dow", "Immediate Loss by Day of Week", "pre_entry")
    save(tk.chart_pre_entry_categorical(profile["prev_result"], "Previous Trade Result", strategy_id, version), "pre_entry_prev_result", "Immediate Loss by Previous Result", "pre_entry")
    save(tk.chart_pre_entry_categorical(profile["trades_since_win"], "Trades Since Last Win", strategy_id, version), "pre_entry_tsw", "Immediate Loss by Trades Since Win", "pre_entry")
    save(tk.chart_kbar_features(kbar_enriched, strategy_id, version), "kbar_features", "K-Bar Features at Entry", "kbar")
    save(tk.chart_bb_zone_winrate(bb_stats_out, strategy_id, version), "bb_zone_winrate", "Win Rate by BB Zone", "bb")
    save(tk.chart_dxy_winrate(dxy_stats_out, strategy_id, version), "dxy_winrate", "DXY Context vs Win Rate", "dxy")
    save(tk.chart_dxy_correlation(corr_df, strategy_id, version, market), "dxy_correlation", f"DXY x {market} Rolling Correlation", "dxy")
    save(tk.chart_htf_alignment(htf_stats_out, strategy_id, version), "htf_alignment", "Win Rate by HTF Alignment", "mtf")
    save(tk.chart_htf_4h_state(htf_stats_out, strategy_id, version), "htf_4h_state", "Win Rate by 4H RSI State", "mtf")
    save(tk.chart_htf_bucket_heatmap(htf_stats_out, strategy_id, version), "htf_4h_bucket", "Win Rate by 4H RSI Bucket", "mtf")
    save(tk.chart_hold_time_dist(trades, strategy_id, version), "hold_time_dist", "Hold Time Distribution", "hold_time_streaks")
    save(tk.chart_consecutive_losses(streaks, strategy_id, version), "consecutive_losses", "Consecutive Loss Streaks", "hold_time_streaks")
    if macro_block is not None:
        save(tk.chart_macro_verdict_winrate(macro_block["by_macro_verdict"], strategy_id, version), "macro_verdict_winrate", "Win Rate by Macro Composite Verdict", "macro")
    if temporal_stability_block is not None:
        save(
            tst.chart_quarterly_stability(trades, temporal_stability_block["by_period"], 0.7, strategy_id, version),
            "quarterly_stability", "Quarterly Win Rate — Chronological Holdout", "temporal_stability",
        )

    generated_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
    method = {
        "fail_pattern_thresholds": {
            "immediate_loss_mfe_pct": tk.IMMEDIATE_LOSS_MFE_PCT,
            "false_breakout_mae_mfe_ratio": tk.FALSE_BREAKOUT_MAE_MFE_RATIO,
            "time_bleed_min_bars": tk.TIME_BLEED_MIN_BARS,
        },
        "session_timezone": "Asia/Taipei (TradingView export-time assumption)",
        "session_buckets": "overnight=01:00-06:59; asia=07:00-14:59; europe=15:00-20:29; us=20:30-00:59 (descriptive only)",
        "primary_time_granularity": "30-minute entry slot (HH:00/HH:30, Asia/Taipei)",
        "bb_params": "period=20, std_mult=2.0, computed on 30m close",
        "dxy_params": "1D RSI bucket/trend(20D SMA)/RSI-vs-MA momentum; 30-day rolling DXY-XAUUSD daily return correlation",
        "mtf_params": "60m/4H/1D RSI(14) state (bullish/bearish/neutral) and bucket, HTF alignment = count of bullish TFs",
        "rsi_provenance": {
            "price_30m": rsi_prov_30m, "price_60m": rsi_prov_60m, "price_4h": rsi_prov_4h,
            "price_1d": rsi_prov_1d, "dxy_1d": rsi_prov_dxy,
        },
    }
    if macro_block is not None:
        method["macro_score"] = "real_rate<MA50 +2; US10Y<MA50 +1; DXY<MA50 +1; VIX>MA50 +1; XAUUSD>MA50 +1"
        method["macro_labels"] = "WAIT=0-2; NEUTRAL=3-4; STRONG BUY=5-6"
        method["macro_assignment"] = "latest prior daily observation, maximum age 4 days"
    if cfg["policy_impact"]:
        method["scoring_method"] = "integer rank score"
    if temporal_stability_block is not None:
        method["temporal_stability_limitation"] = tst.TEMPORAL_STABILITY_LIMITATION
        method["temporal_stability_buckets"] = "calendar quarter (YYYY-Qn) of entry_time; most recent bucket may be partial"
        method["temporal_stability_holdout"] = "chronological split, first 70% entry-time-ordered trades = in_sample, last 30% = held_out"
        method["temporal_stability_degradation_rule"] = (
            "degraded: held_out.win_rate_pct < in_sample.win_rate_ci95_pct[0] OR held_out.profit_factor < 1.0; "
            "improved: held_out.win_rate_pct > in_sample.win_rate_ci95_pct[1] AND held_out.profit_factor > in_sample.profit_factor; "
            "else stable"
        )

    results = {
        "schema_version": 1,
        "study_id": study_id,
        "generated_at": generated_at,
        "strategy": f"{strategy_id} {version}",
        "method": method,
        "trade_period": {"start": str(trades["entry_time"].min()), "end": str(trades["exit_time"].max())},
        "baseline": baseline,
        "fail_pattern": {"total_losses": len(classified), "by_type": fail_summary, "by_session": fail_by_session},
        "by_session": session_stats,
        "by_entry_30m": entry_30m,
        "immediate_loss_profile": profile,
        "kbar_coverage": kbar_cov,
        "bb_zone": bb_stats_out,
        "dxy": {"regime": dxy_stats_out, "avg_30d_correlation": avg_corr},
        "mtf": htf_stats_out,
        "hold_time_streaks": {
            "avg_hold_bars": baseline["avg_hold_bars"],
            "max_consecutive_losses": baseline["max_consecutive_losses"],
            "streak_lengths": streaks.tolist(),
        },
        "charts": charts_manifest,
        "sources": [{"role": role, "path": str(path), "sha256": sha256(path)} for role, path in source_paths.items()],
    }
    if macro_block is not None:
        results.update(macro_block)
    if temporal_stability_block is not None:
        results["temporal_stability"] = temporal_stability_block
    results = tk.to_json_safe(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(results, charts_dir), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(results), encoding="utf-8")
    return results


def render_html(results: dict, charts_dir: Path) -> str:
    b64 = {c["id"]: base64_of(charts_dir / c["file"]) for c in results["charts"]}
    baseline = results["baseline"]
    strategy = results["strategy"]
    kpis = [
        ("Trades", baseline["n"], "neu"),
        ("Win Rate", f'{baseline["win_rate_pct"]}%', "pos" if baseline["win_rate_pct"] >= 50 else "neg"),
        ("Profit Factor", baseline["profit_factor"], "pos" if (baseline["profit_factor"] or 0) >= 1.5 else "neu"),
        ("Net P&L", f'${baseline["net_pnl_usd"]:,.0f}', "pos" if baseline["net_pnl_usd"] >= 0 else "neg"),
        ("Max Drawdown", f'${baseline["max_drawdown_usd"]:,.0f}', "neg"),
        ("Max Consec Loss", baseline["max_consecutive_losses"], "neu"),
        ("Avg Hold", f'{baseline["avg_hold_bars"]:.0f} bars', "neu"),
    ]
    kpi_html = '<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value {c}">{v}</div></div>' for l, v, c in kpis
    ) + "</div>"

    fail_rows = [[k, v["count"], f'{v["pct"]}%'] for k, v in results["fail_pattern"]["by_type"].items()]
    session_rows = _stats_rows(results["by_session"], ["asia", "europe", "us", "overnight"])
    entry_30m_rows = _stats_rows(results["by_entry_30m"], tk.ENTRY_SLOTS_30M)
    bb_rows = _stats_rows(results["bb_zone"], tk.BB_ZONE_ORDER)
    dxy_bucket_rows = _stats_rows(results["dxy"]["regime"]["by_bucket"])
    mtf_align_rows = _stats_rows(results["mtf"]["by_alignment"])

    has_rank_score = "rank_score" in results["by_entry_30m"].get(tk.ENTRY_SLOTS_30M[0], {})
    session_headers = ["session", "n", "WR", "PF", "Net"] + (["Score"] if has_rank_score else [])
    entry_headers = ["slot", "n", "WR", "PF", "Net"] + (["Score"] if has_rank_score else [])

    macro_html = ""
    if "by_macro_verdict" in results:
        macro_rows = _stats_rows(results["by_macro_verdict"], ["WAIT", "NEUTRAL", "STRONG BUY"])
        macro_headers = ["verdict", "n", "WR", "PF", "Net"] + (["Score"] if has_rank_score else [])
        macro_html = f"""
  <h2>Macro Composite Context</h2>
  <div class="card">
    {_img(b64.get("macro_verdict_winrate", ""))}
    {_table(macro_headers, macro_rows)}
    <div class="note">Macro coverage: {results["macro_coverage"]["matched"]}/{results["macro_coverage"]["matched"] + results["macro_coverage"]["unmatched"]} trades ({results["macro_coverage"]["pct"]}%).</div>
  </div>
"""

    temporal_html = ""
    if "temporal_stability" in results:
        ts = results["temporal_stability"]
        period_rows = _stats_rows(ts["by_period"])
        holdout = ts["holdout_split"]
        flag = ts["degradation_flag"]
        flag_class = {"stable": "neu", "improved": "pos", "degraded": "neg"}.get(flag, "neu")
        temporal_html = f"""
  <h2>Temporal Stability — Chronological Holdout</h2>
  <div class="card">
    {_img(b64.get("quarterly_stability", ""))}
    {_table(["quarter", "n", "WR", "PF", "Net"], period_rows)}
    <div class="note">
      In-sample ({holdout['split_ratio']*100:.0f}%, {holdout['in_sample']['period']['start']} &rarr; {holdout['in_sample']['period']['end']}):
      n={holdout['in_sample']['n']}, WR {holdout['in_sample']['win_rate_pct']}%, PF {holdout['in_sample']['profit_factor']}.
      Held-out ({(1-holdout['split_ratio'])*100:.0f}%, {holdout['held_out']['period']['start']} &rarr; {holdout['held_out']['period']['end']}):
      n={holdout['held_out']['n']}, WR {holdout['held_out']['win_rate_pct']}%, PF {holdout['held_out']['profit_factor']}.
      Degradation flag: <span class="{flag_class}"><b>{flag}</b></span>.
      {results['method'].get('temporal_stability_limitation', '')}
    </div>
  </div>
"""

    generated_at = results["generated_at"]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy} — Fail Pattern Report (30-min)</title>
{CSS}
</head>
<body>
<div class="wrap">
  <h1>{strategy}</h1>
  <div class="meta">Generated {generated_at} &nbsp;|&nbsp; {results["trade_period"]["start"]} → {results["trade_period"]["end"]}</div>
  {kpi_html}

  <h2>Equity Curve & Drawdown</h2>
  <div class="card">{_img(b64["equity_curve"])}</div>

  <h2>Fail Pattern Breakdown</h2>
  <div class="card">
    <div class="chart-grid">{_img(b64["fail_type_breakdown"])}{_img(b64["mfe_distribution"])}</div>
    {_img(b64["mae_vs_mfe"])}
    {_table(["fail_type", "count", "pct"], fail_rows)}
  </div>

  <h2>Session Summary {"(scored; overnight descriptive)" if has_rank_score else "(descriptive)"}</h2>
  <div class="card">{_table(session_headers, session_rows)}</div>

  <h2>30-Minute Entry-Slot Timing (primary evidence)</h2>
  <div class="card">
    {_img(b64["entry_slot_30m_winrate"])}
    {_img(b64["immediate_loss_by_slot_30m"])}
    {_table(entry_headers, entry_30m_rows)}
  </div>
  {macro_html}
  {temporal_html}
  <h2>Pre-Entry Context — Immediate Loss</h2>
  <div class="card">
    <div class="chart-grid">{_img(b64["pre_entry_dow"])}{_img(b64["pre_entry_prev_result"])}</div>
    {_img(b64["pre_entry_tsw"])}
  </div>

  <h2>K-Bar Features at Entry</h2>
  <div class="card">
    {_img(b64["kbar_features"])}
    <div class="note">K-Bar coverage: {results["kbar_coverage"]["with_kbar_data"]}/{results["kbar_coverage"]["total_immediate_loss"]} immediate_loss trades ({results["kbar_coverage"]["coverage_pct"]}%).</div>
  </div>

  <h2>Bollinger Band Position</h2>
  <div class="card">{_img(b64["bb_zone_winrate"])}{_table(["zone", "n", "WR", "PF", "Net"], bb_rows)}</div>

  <h2>DXY Context</h2>
  <div class="card">
    {_img(b64["dxy_winrate"])}
    {_img(b64["dxy_correlation"])}
    <div class="note">Avg 30-day rolling DXY-XAUUSD correlation: {results["dxy"]["avg_30d_correlation"]}</div>
    {_table(["DXY RSI bucket", "n", "WR", "PF", "Net"], dxy_bucket_rows)}
  </div>

  <h2>Multi-Timeframe Alignment</h2>
  <div class="card">
    {_img(b64["htf_alignment"])}
    {_img(b64["htf_4h_state"])}
    {_img(b64["htf_4h_bucket"])}
    {_table(["HTF alignment", "n", "WR", "PF", "Net"], mtf_align_rows)}
  </div>

  <h2>Hold Time & Streaks</h2>
  <div class="card"><div class="chart-grid">{_img(b64["hold_time_dist"])}{_img(b64["consecutive_losses"])}</div></div>

  <footer>XAUUSD Strategy Fail-Pattern Toolkit (30-min contract) &nbsp;·&nbsp; {generated_at}</footer>
</div>
</body>
</html>"""
    return html


def base64_of(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode()


def render_readme(results: dict) -> str:
    baseline = results["baseline"]
    lines = [
        f"# {results['study_id']} — {results['strategy']} fail-pattern report",
        "",
        f"Generated: `{results['generated_at']}`",
        "",
        "## Scope",
        "",
        f"- Closed trades: **{baseline['n']}**, {results['trade_period']['start']} to {results['trade_period']['end']}.",
        "- 30-minute entry slot is the primary time axis (the project's standard granularity).",
        f"- K-bar coverage for immediate_loss trades: {results['kbar_coverage']['coverage_pct']}%.",
    ]
    if "by_macro_verdict" in results:
        lines.append(f"- Macro composite coverage: {results['macro_coverage']['pct']}% (section 5.1 item 10).")
    if "scoring_method" in results.get("method", {}):
        lines.append("- This study is policy-impacting: by_entry_30m/by_session/by_macro_verdict carry integer rank_score (section 13.2). See `impact.md`.")
    if "temporal_stability" in results:
        ts = results["temporal_stability"]
        lines.append(
            f"- Temporal stability (section 5.1 item 11): chronological 70/30 holdout — "
            f"degradation flag **{ts['degradation_flag']}**. Not a re-optimized walk-forward; see `method.temporal_stability_limitation`."
        )
    lines += [
        "",
        "## Baseline",
        "",
        f"- WR **{baseline['win_rate_pct']}%**, PF **{baseline['profit_factor']}**, "
        f"net **${baseline['net_pnl_usd']:,.2f}**, max drawdown **${baseline['max_drawdown_usd']:,.2f}**.",
        "",
        "## Fail pattern",
        "",
    ]
    for name, v in results["fail_pattern"]["by_type"].items():
        lines.append(f"- `{name}`: {v['count']} ({v['pct']}%)")
    lines += [
        "",
        "## Interpretation",
        "",
        "- 30-minute slot and DXY/MTF/BB context are descriptive evidence, not entry gates.",
        "- See `report.html` for the full chart-embedded report and `results.json` for structured data.",
        "",
        "## Source provenance",
        "",
    ]
    for source in results["sources"]:
        lines.append(f"- `{source['role']}`: `{source['path']}` — SHA-256 `{source['sha256']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True, choices=list(STUDY_CONFIG))
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("reproduced")
    results = build(args.study_id, args.legacy_root, output_dir.resolve())
    print(json.dumps({
        "study_id": args.study_id, "output": str(output_dir),
        "baseline": results["baseline"], "chart_count": len(results["charts"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
