"""
Visualization helpers. Every public function returns a matplotlib Figure.

Usage:
    fig = charts.equity_curve(trades, "S1-AweWithBB")
    fig.savefig("reports/equity.png", dpi=150)
    # or in a notebook: display(fig)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analysis import metrics as m

# --- Colour palette ---
_WIN = "#2ecc71"
_LOSS = "#e74c3c"
_NEUTRAL = "#95a5a6"
_BLUE = "#3498db"
_DXY = "#e67e22"

FAIL_COLORS = {
    "immediate_loss": "#c0392b",
    "false_breakout": "#e67e22",
    "time_bleed": "#8e44ad",
    "normal_sl": "#7f8c8d",
}


# ---------------------------------------------------------------------------
# Equity & Drawdown
# ---------------------------------------------------------------------------

def equity_curve(trades: pd.DataFrame, title: str = "") -> plt.Figure:
    dd = m.drawdown_series(trades)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(trades["entry_time"], trades["cum_pnl_usd"],
             linewidth=1.5, color=_BLUE)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cumulative P&L (USD)")
    ax1.set_title(f"Equity Curve — {title}")

    ax2.fill_between(trades["entry_time"], dd, 0, color=_LOSS, alpha=0.5)
    ax2.set_ylabel("Drawdown (USD)")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fail patterns
# ---------------------------------------------------------------------------

def fail_type_breakdown(classified_losses: pd.DataFrame, title: str = "") -> plt.Figure:
    counts = classified_losses["fail_type"].value_counts()
    colors = [FAIL_COLORS.get(k, _NEUTRAL) for k in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Fail Pattern Breakdown — {title}")
    ax.set_ylabel("# Trades")
    fig.tight_layout()
    return fig


def mfe_distribution(classified_losses: pd.DataFrame, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    for ftype, group in classified_losses.groupby("fail_type"):
        ax.hist(group["mfe_pct"], bins=20, alpha=0.65,
                label=ftype, color=FAIL_COLORS.get(ftype, _NEUTRAL))
    ax.set_title(f"MFE% Distribution of Losses — {title}")
    ax.set_xlabel("MFE %")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def mae_vs_mfe_scatter(classified_losses: pd.DataFrame, title: str = "") -> plt.Figure:
    """Scatter plot of MAE% vs MFE% coloured by fail_type — shows trade quality at a glance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for ftype, group in classified_losses.groupby("fail_type"):
        ax.scatter(group["mfe_pct"], group["mae_pct"],
                   label=ftype, color=FAIL_COLORS.get(ftype, _NEUTRAL),
                   alpha=0.6, s=30)
    ax.set_xlabel("MFE %")
    ax.set_ylabel("MAE %")
    ax.set_title(f"MAE vs MFE for Losses — {title}")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Session & Time analysis
# ---------------------------------------------------------------------------

def session_heatmap(session_df: pd.DataFrame, title: str = "") -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    session_df["win_rate"].plot.bar(ax=ax1, color=_BLUE, edgecolor="white")
    ax1.set_title("Win Rate by Session")
    ax1.set_ylabel("Win Rate")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="gray", linewidth=1, linestyle="--")

    colors = [_WIN if v >= 0 else _LOSS for v in session_df["avg_pnl"]]
    session_df["avg_pnl"].plot.bar(ax=ax2, color=colors, edgecolor="white")
    ax2.set_title("Avg P&L by Session (USD)")
    ax2.axhline(0, color="gray", linewidth=0.8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def hourly_winrate(hourly_df: pd.DataFrame, title: str = "") -> plt.Figure:
    colors = [_WIN if v >= 0.5 else _LOSS for v in hourly_df["win_rate"]]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(hourly_df.index, hourly_df["win_rate"], color=colors, edgecolor="white")
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="50%")
    ax.set_title(f"Win Rate by Entry Hour (UTC+8) — {title}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Win Rate")
    ax.set_xticks(range(0, 24))
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streak analysis
# ---------------------------------------------------------------------------

def consecutive_loss_hist(streak_series: pd.Series, title: str = "") -> plt.Figure:
    if streak_series.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    max_val = int(streak_series.max())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(streak_series, bins=range(1, max_val + 2),
            align="left", color=_LOSS, edgecolor="white")
    ax.set_title(f"Consecutive Loss Streaks — {title}")
    ax.set_xlabel("Streak Length (trades)")
    ax.set_ylabel("Occurrences")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hold-time distribution
# ---------------------------------------------------------------------------

def hold_time_dist(trades: pd.DataFrame, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    wins = trades[trades["result"] == "win"]["hold_bars"]
    losses = trades[trades["result"] == "loss"]["hold_bars"]
    bins = range(0, int(trades["hold_bars"].max()) + 2, 4)  # group by 2-hour buckets
    ax.hist(wins, bins=bins, alpha=0.65, label="Win", color=_WIN)
    ax.hist(losses, bins=bins, alpha=0.65, label="Loss", color=_LOSS)
    ax.set_title(f"Hold Time Distribution (30-min bars) — {title}")
    ax.set_xlabel("Bars held")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pre-entry context (Layer 1 — trade-data only)
# ---------------------------------------------------------------------------

def pre_entry_hour(profile: dict, title: str = "") -> plt.Figure:
    """
    Side-by-side bar chart: immediate_loss rate vs all-trades rate by entry hour.
    A tall immediate_loss bar at a given hour means that hour is over-represented in failures.
    """
    df = profile["entry_hour"]
    x = df.index
    width = 0.4
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x - width / 2, df["immediate_loss"], width, label="immediate_loss", color=_LOSS, alpha=0.8)
    ax.bar(x + width / 2, df["all_trades"], width, label="all trades", color=_BLUE, alpha=0.6)
    ax.set_title(f"Entry Hour Distribution — immediate_loss vs all trades ({title})")
    ax.set_xlabel("Hour (UTC+8)")
    ax.set_ylabel("Proportion")
    ax.set_xticks(range(0, 24))
    ax.legend()
    fig.tight_layout()
    return fig


def pre_entry_dow(profile: dict, title: str = "") -> plt.Figure:
    df = profile["entry_dow"]
    x = range(len(df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], df["immediate_loss"], width,
           label="immediate_loss", color=_LOSS, alpha=0.8)
    ax.bar([i + width / 2 for i in x], df["all_trades"], width,
           label="all trades", color=_BLUE, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index)
    ax.set_title(f"Day-of-Week Distribution — immediate_loss vs all trades ({title})")
    ax.set_ylabel("Proportion")
    ax.legend()
    fig.tight_layout()
    return fig


def pre_entry_prev_result(profile: dict, title: str = "") -> plt.Figure:
    df = profile["prev_result"]
    x = range(len(df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], df["immediate_loss"], width,
           label="immediate_loss", color=_LOSS, alpha=0.8)
    ax.bar([i + width / 2 for i in x], df["all_trades"], width,
           label="all trades", color=_BLUE, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index)
    ax.set_title(f"Previous Trade Result — before immediate_loss vs all trades ({title})")
    ax.set_ylabel("Proportion")
    ax.legend()
    fig.tight_layout()
    return fig


def pre_entry_tsw(profile: dict, title: str = "") -> plt.Figure:
    """trades_since_win — are immediate losses more likely after losing streaks?"""
    df = profile["trades_since_win"]
    x = range(len(df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - width / 2 for i in x], df["immediate_loss"], width,
           label="immediate_loss", color=_LOSS, alpha=0.8)
    ax.bar([i + width / 2 for i in x], df["all_trades"], width,
           label="all trades", color=_BLUE, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index)
    ax.set_title(f"Consecutive Non-Wins Before Entry ({title})")
    ax.set_xlabel("Trades since last win")
    ax.set_ylabel("Proportion")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pre-entry K-bar features (Layer 2 — requires price data overlap)
# ---------------------------------------------------------------------------

def kbar_feature_summary(enriched: pd.DataFrame, title: str = "") -> plt.Figure:
    """
    Shows RSI, momentum_3, and prev_3_green for enriched immediate_loss trades.
    Displays a placeholder when no K-bar data is available.
    """
    df = enriched.dropna(subset=["rsi"])
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5,
                "No K-bar data available.\nExport OHLCV+RSI from TradingView to enable this chart.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(f"K-Bar Features at Entry — {title}")
        ax.axis("off")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # RSI distribution
    axes[0].hist(df["rsi"], bins=15, color=_LOSS, edgecolor="white")
    axes[0].axvline(30, color="green",  linestyle="--", linewidth=1, label="RSI 30")
    axes[0].axvline(50, color="gray",   linestyle="--", linewidth=1, label="RSI 50")
    axes[0].axvline(70, color="orange", linestyle="--", linewidth=1, label="RSI 70")
    axes[0].set_title("RSI at Entry")
    axes[0].set_xlabel("RSI value")
    axes[0].legend(fontsize=8)

    # 3-bar momentum
    axes[1].hist(df["momentum_3"], bins=15, color=_NEUTRAL, edgecolor="white")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("3-Bar Price Momentum % Before Entry")
    axes[1].set_xlabel("Momentum %")

    # Green bar count
    counts = df["prev_3_green"].value_counts().sort_index()
    axes[2].bar(counts.index.astype(str), counts.values, color=_BLUE, edgecolor="white")
    axes[2].set_title("Green Bars in Last 3 Before Entry")
    axes[2].set_xlabel("# Green bars (0–3)")
    axes[2].set_ylabel("Count")

    fig.suptitle(f"K-Bar Context at Immediate-Loss Entry ({len(df)} trades) — {title}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# DXY analysis charts
# ---------------------------------------------------------------------------

def dxy_winrate_chart(dxy_stats: dict, title: str = "") -> plt.Figure:
    """
    3-panel chart: win rate by DXY RSI bucket, by DXY trend, by RSI momentum.
    dxy_stats is the dict returned by dxy_analysis.dxy_regime_stats().
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: by RSI bucket
    df_b = dxy_stats.get("by_bucket", pd.DataFrame())
    if not df_b.empty:
        colors = [_WIN if v >= 0.5 else _LOSS for v in df_b["win_rate"]]
        labels = [l.split("(")[0] for l in df_b.index]
        axes[0].bar(range(len(df_b)), df_b["win_rate"], color=colors, edgecolor="white")
        axes[0].set_xticks(range(len(df_b)))
        axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        axes[0].axhline(0.5, color="gray", linewidth=1, linestyle="--")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Win Rate by DXY RSI Bucket")
        axes[0].set_ylabel("Win Rate")
        for i, (wr, n) in enumerate(zip(df_b["win_rate"], df_b["total"])):
            axes[0].text(i, wr + 0.02, f"{wr:.0%}\nn={n}", ha="center", fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)

    # Panel 2: by trend
    df_t = dxy_stats.get("by_trend", pd.DataFrame())
    if not df_t.empty:
        colors = [_WIN if v >= 0.5 else _LOSS for v in df_t["win_rate"]]
        axes[1].bar(df_t.index, df_t["win_rate"], color=colors, edgecolor="white")
        axes[1].axhline(0.5, color="gray", linewidth=1, linestyle="--")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Win Rate: DXY Trend Direction")
        for i, (idx, row) in enumerate(df_t.iterrows()):
            axes[1].text(i, row["win_rate"] + 0.02,
                         f"{row['win_rate']:.0%}\nn={row['total']}", ha="center", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)

    # Panel 3: by momentum (RSI vs MA)
    df_m = dxy_stats.get("by_momentum", pd.DataFrame())
    if not df_m.empty:
        colors = [_WIN if v >= 0.5 else _LOSS for v in df_m["win_rate"]]
        labels = [l.split(" ")[0] for l in df_m.index]
        axes[2].bar(range(len(df_m)), df_m["win_rate"], color=colors, edgecolor="white")
        axes[2].set_xticks(range(len(df_m)))
        axes[2].set_xticklabels(labels, rotation=10, ha="right", fontsize=8)
        axes[2].axhline(0.5, color="gray", linewidth=1, linestyle="--")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Win Rate: DXY RSI vs MA")
        for i, (idx, row) in enumerate(df_m.iterrows()):
            axes[2].text(i, row["win_rate"] + 0.02,
                         f"{row['win_rate']:.0%}\nn={row['total']}", ha="center", fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)

    fig.suptitle(f"DXY Context vs Win Rate — {title}")
    fig.tight_layout()
    return fig


def dxy_correlation_chart(corr_df: pd.DataFrame, title: str = "") -> plt.Figure:
    """
    Two-panel chart:
      Top: rolling N-day correlation between DXY and XAUUSD daily returns
      Bottom: DXY and XAUUSD normalised close (indexed to 100 at first common date)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})

    ax1.plot(corr_df["time"], corr_df["rolling_corr"], color=_DXY, linewidth=1.2)
    ax1.axhline(0,    color="gray",  linewidth=0.8, linestyle="--")
    ax1.axhline(-0.5, color=_WIN,   linewidth=0.8, linestyle=":", label="-0.5 threshold")
    ax1.axhline( 0.5, color=_LOSS,  linewidth=0.8, linestyle=":", label="+0.5 threshold")
    ax1.set_ylabel("Rolling Correlation")
    ax1.set_title(f"DXY × XAUUSD Rolling Return Correlation — {title}")
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(fontsize=8)
    ax1.fill_between(corr_df["time"], corr_df["rolling_corr"], 0,
                     where=corr_df["rolling_corr"] < 0, alpha=0.25, color=_WIN, label="negative")
    ax1.fill_between(corr_df["time"], corr_df["rolling_corr"], 0,
                     where=corr_df["rolling_corr"] > 0, alpha=0.25, color=_LOSS, label="positive")

    ax2.plot(corr_df["time"], corr_df["dxy_ret"].cumsum() * 100,
             color=_DXY, linewidth=1.2, label="DXY cum. ret %")
    ax2r = ax2.twinx()
    ax2r.plot(corr_df["time"], corr_df["xau_ret"].cumsum() * 100,
              color=_BLUE, linewidth=1.2, label="XAUUSD cum. ret %")
    ax2.set_ylabel("DXY Cum. Return %", color=_DXY)
    ax2r.set_ylabel("XAUUSD Cum. Return %", color=_BLUE)
    ax2.set_xlabel("Date")
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="upper left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-Timeframe (MTF) charts
# ---------------------------------------------------------------------------

_HTF_STATE_COLORS = {
    "bullish":  "#27ae60",
    "neutral":  "#f39c12",
    "bearish":  "#e74c3c",
    "unknown":  "#bdc3c7",
}
_ALIGN_COLORS = ["#e74c3c", "#e67e22", "#3498db", "#27ae60"]


def htf_alignment_bar(stats: dict, title: str = "") -> plt.Figure:
    """Bar chart: win rate by HTF alignment score."""
    df = stats.get("by_alignment")
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No HTF data available", ha="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    labels    = [str(l) for l in df.index]
    win_rates = df["win_rate"] * 100
    totals    = df["total"]
    colors    = (_ALIGN_COLORS + _ALIGN_COLORS)[: len(labels)]

    bars = ax.bar(labels, win_rates, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, n, wr in zip(bars, totals, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"n={n}\n{wr:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("HTF Alignment (number of aligned timeframes)")
    ax.set_ylabel("Win Rate %")
    ax.set_title(f"Win Rate by HTF Alignment — {title}")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    return fig


def htf_4h_state_bar(stats: dict, title: str = "") -> plt.Figure:
    """Grouped chart: win rate by 4H RSI state + fail type breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: win rate by 4H state
    df_state = stats.get("by_4h_state")
    ax = axes[0]
    if df_state is not None and not df_state.empty:
        ordered = [s for s in ["bearish", "neutral", "bullish"] if s in df_state.index]
        df_plot = df_state.loc[ordered]
        colors  = [_HTF_STATE_COLORS[s] for s in ordered]
        bars    = ax.bar(ordered, df_plot["win_rate"] * 100, color=colors,
                         edgecolor="white", linewidth=0.8)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        for bar, (state, row) in zip(bars, df_plot.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"n={int(row['total'])}\n{row['win_rate']*100:.1f}%",
                    ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Win Rate by 4H RSI State — {title}")
    ax.set_ylabel("Win Rate %")
    ax.set_ylim(0, 110)

    # Right: fail type % by 4H state (stacked bar)
    df_fail = stats.get("fail_by_4h")
    ax2 = axes[1]
    if df_fail is not None and not df_fail.empty:
        states    = [s for s in ["bearish", "neutral", "bullish"] if s in df_fail.index]
        fail_cols = [c for c in df_fail.columns if c in FAIL_COLORS]
        bottom    = np.zeros(len(states))
        for fc in fail_cols:
            vals = [df_fail.loc[s, fc] * 100 if s in df_fail.index and fc in df_fail.columns else 0
                    for s in states]
            ax2.bar(states, vals, bottom=bottom, label=fc,
                    color=FAIL_COLORS[fc], edgecolor="white", linewidth=0.5)
            bottom += np.array(vals)
        ax2.set_title(f"Fail Type % by 4H State — {title}")
        ax2.set_ylabel("% of Losses")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.set_ylim(0, 115)
    else:
        ax2.text(0.5, 0.5, "Insufficient loss data", ha="center", transform=ax2.transAxes)
        ax2.set_title("Fail Type % by 4H State")

    fig.tight_layout()
    return fig


def htf_bucket_heatmap(stats: dict, title: str = "") -> plt.Figure:
    """Heatmap: win rate by 4H RSI bucket."""
    df = stats.get("by_4h_bucket")
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No 4H data available", ha="center", transform=ax.transAxes)
        return fig

    bucket_order = ["oversold(<30)", "low(30-50)", "high(50-70)", "overbought(>70)"]
    df_plot = df.reindex([b for b in bucket_order if b in df.index])

    fig, ax = plt.subplots(figsize=(10, 3))
    wr     = df_plot["win_rate"].values.astype(float)
    totals = df_plot["total"].values.astype(int)
    labels = list(df_plot.index)

    im = ax.imshow([wr], cmap="RdYlGn", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yticks([])

    for i, (w, n) in enumerate(zip(wr, totals)):
        ax.text(i, 0, f"{w*100:.1f}%\nn={n}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if (w < 0.38 or w > 0.62) else "black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Win Rate")
    ax.set_title(f"Win Rate by 4H RSI Bucket — {title}")
    fig.tight_layout()
    return fig
