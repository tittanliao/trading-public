#!/usr/bin/env python3
"""
Cross-commodity shared analysis:
  1. Hourly entry heatmap — win-rate & avg profit by day-of-week × hour
  2. 30min RSI golden/death cross & divergence filter at hourly entry

Run from the trading/ root:
    python3.12 shared/run_shared_analysis.py
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT_JSON = Path(__file__).parent / "shared_results.json"

COMMODITIES = {
    "xauusd": {
        "name": "XAUUSD 黃金",
        "csv_60m": ROOT / "xauusd/csv/FX_IDC_XAUUSD, 60.csv",
        "csv_30m": ROOT / "xauusd/csv/FX_IDC_XAUUSD, 30.csv",
        "unit": "pct",
        "unit_label": "%",
    },
    "tx": {
        "name": "TX 台指期 (MTX)",
        "csv_60m": ROOT / "tx/csv/TAIFEX_DLY_MXF1!, 60.csv",
        "csv_30m": ROOT / "tx/csv/TAIFEX_MXF1!, 30.csv",
        "unit": "pts",
        "unit_label": "pts",
    },
}

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

RSI_STATE_ORDER = [
    'overall', 'golden_cross', 'death_cross',
    'bullish_div', 'bearish_div', 'rsi_above_ma', 'rsi_below_ma',
]
RSI_STATE_LABELS = {
    'overall':      'All',
    'golden_cross': 'GoldenX',
    'death_cross':  'DeathX',
    'bullish_div':  'BullDiv',
    'bearish_div':  'BearDiv',
    'rsi_above_ma': 'RSI>MA',
    'rsi_below_ma': 'RSI<MA',
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=False)
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('Asia/Taipei')
    else:
        df['time'] = df['time'].dt.tz_convert('Asia/Taipei')
    return df.sort_values('time').reset_index(drop=True)


def add_rsi_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_rsi = 'RSI' in df.columns and 'RSI-based MA' in df.columns

    if has_rsi:
        rsi = pd.to_numeric(df['RSI'], errors='coerce').ffill()
        rsima = pd.to_numeric(df['RSI-based MA'], errors='coerce').ffill()
        above = rsi > rsima
        prev_above = above.shift(1)
        df['rsi_above_ma'] = above
        df['golden_cross'] = above & (~prev_above.fillna(False))
        df['death_cross'] = (~above) & prev_above.fillna(True)
    else:
        df['rsi_above_ma'] = False
        df['golden_cross'] = False
        df['death_cross'] = False

    for sig, col in [('bullish_div', 'Regular Bullish'), ('bearish_div', 'Regular Bearish')]:
        if col in df.columns:
            df[sig] = df[col].fillna('').astype(str).str.strip() != ''
        else:
            df[sig] = False

    return df


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def make_heatmap(matrix: np.ndarray, counts: np.ndarray,
                  hours: list, title: str,
                  vmin: float, vmax: float,
                  cmap: str, fmt: str, suffix: str) -> str:
    n_hours = len(hours)
    fig_h = max(3.5, n_hours * 0.32 + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    masked = np.ma.masked_where(np.isnan(matrix), matrix)
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    ax.set_xticks(range(5))
    ax.set_xticklabels(DAY_NAMES, fontsize=9)
    ax.set_yticks(range(n_hours))
    ax.set_yticklabels([f'{h:02d}:00' for h in hours], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    vcenter = (vmax + vmin) / 2
    for i in range(n_hours):
        for j in range(5):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            n = int(counts[i, j])
            text = f'{v:{fmt}}{suffix}'
            if n < 5:
                text += '*'
            brightness = (v - vmin) / max(vmax - vmin, 1e-9)
            txt_color = 'white' if brightness < 0.25 or brightness > 0.85 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=7, color=txt_color, fontweight='bold')

    plt.tight_layout()
    return fig_to_b64(fig)


def make_rsi_bar(rsi_stats: dict, commodity_name: str, unit_label: str, unit: str) -> str:
    states = [s for s in RSI_STATE_ORDER if s in rsi_stats]
    labels = [RSI_STATE_LABELS[s] for s in states]
    wr_vals = [rsi_stats[s]['win_rate'] for s in states]
    ret_vals = [rsi_stats[s]['avg_ret'] for s in states]
    counts = [rsi_stats[s]['count'] for s in states]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    label_name = commodity_name.split()[0]
    fig.suptitle(f'{label_name} — Hourly Entry: 30m RSI Filter Effect', fontsize=11, fontweight='bold')

    # Win rate chart
    ax = axes[0]
    colors = ['#2e7d32' if v >= 50 else '#c62828' for v in wr_vals]
    bars = ax.bar(labels, wr_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(50, color='#666', linestyle='--', linewidth=1, alpha=0.6, label='Breakeven 50%')
    ax.set_ylabel('Win Rate (%)', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title('Win Rate by 30m RSI State', fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(axis='x', labelsize=8)
    for bar, v, n in zip(bars, wr_vals, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 1.5, 95),
                f'{v:.1f}%\nn={n}', ha='center', va='bottom', fontsize=7)

    # Avg return chart
    ax = axes[1]
    colors = ['#2e7d32' if v >= 0 else '#c62828' for v in ret_vals]
    bars = ax.bar(labels, ret_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='#666', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylabel(f'Avg Return ({unit_label})', fontsize=9)
    ax.set_title(f'Avg Return ({unit_label}) by 30m RSI State', fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    fmt_ret = '.2f' if unit == 'pct' else '.0f'
    for bar, v, n in zip(bars, ret_vals, counts):
        offset = abs(max(ret_vals) - min(ret_vals)) * 0.03 + 1e-9
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (offset if v >= 0 else -offset * 3),
                f'{v:{fmt_ret}}\nn={n}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    return fig_to_b64(fig)


def analyze_commodity(cid: str, cfg: dict) -> dict:
    print(f"  [{cid}] loading data...")
    df60 = load_csv(cfg['csv_60m'])
    df30 = load_csv(cfg['csv_30m'])
    df30 = add_rsi_signals(df30)

    # Forward returns
    df60 = df60.copy()
    df60['next_close'] = df60['close'].shift(-1)
    df60['return_pct'] = (df60['next_close'] - df60['close']) / df60['close'] * 100
    df60['return_pts'] = df60['next_close'] - df60['close']
    df60 = df60.dropna(subset=['next_close']).copy()
    df60['dow'] = df60['time'].dt.dayofweek
    df60['hour'] = df60['time'].dt.hour
    df60['win'] = df60['return_pct'] > 0
    df60 = df60[df60['dow'] < 5].copy()

    ret_col = 'return_pct' if cfg['unit'] == 'pct' else 'return_pts'

    # ── Heatmaps ───────────────────────────────────────────────────
    hours = sorted(df60['hour'].unique())
    wr_mat = np.full((len(hours), 5), np.nan)
    ret_mat = np.full((len(hours), 5), np.nan)
    cnt_mat = np.zeros((len(hours), 5))

    for i, h in enumerate(hours):
        for j in range(5):
            sub = df60[(df60['hour'] == h) & (df60['dow'] == j)]
            if len(sub) >= 2:
                wr_mat[i, j] = float(sub['win'].mean() * 100)
                ret_mat[i, j] = float(sub[ret_col].mean())
                cnt_mat[i, j] = len(sub)

    print(f"  [{cid}] generating heatmaps...")
    label_name = 'XAUUSD' if cid == 'xauusd' else 'TX MTX'
    wr_b64 = make_heatmap(
        wr_mat, cnt_mat, hours,
        f'{label_name} — Hourly Win Rate (%) [entry close, exit next close]',
        vmin=30, vmax=70, cmap='RdYlGn', fmt='.0f', suffix='%'
    )
    ret_fmt = '.2f' if cfg['unit'] == 'pct' else '.0f'
    ret_vabs = 0.15 if cfg['unit'] == 'pct' else 30
    ret_b64 = make_heatmap(
        ret_mat, cnt_mat, hours,
        f'{label_name} — Hourly Avg Return ({cfg["unit_label"]}) [entry close, exit next close]',
        vmin=-ret_vabs, vmax=ret_vabs, cmap='RdYlGn',
        fmt=ret_fmt, suffix=cfg["unit_label"]
    )

    # ── RSI Cross Filter ───────────────────────────────────────────
    print(f"  [{cid}] computing RSI filter...")
    df30_slim = df30[['time', 'RSI', 'RSI-based MA',
                       'golden_cross', 'death_cross',
                       'bearish_div', 'bullish_div', 'rsi_above_ma']].copy()
    df30_slim.columns = ['time30', 'rsi_30m', 'rsima_30m',
                          'golden_cross', 'death_cross',
                          'bearish_div', 'bullish_div', 'rsi_above_ma']

    df60s = df60.sort_values('time').reset_index(drop=True)
    df30s = df30_slim.sort_values('time30').reset_index(drop=True)
    df30s = df30s.rename(columns={'time30': 'time'})
    df_m = pd.merge_asof(df60s, df30s, on='time', direction='backward')

    # Only rows where 30m RSI data is available
    has_rsi = df_m['rsi_30m'].notna()

    state_masks = {
        'overall':      pd.Series(True, index=df_m.index),
        'golden_cross': df_m['golden_cross'].fillna(False),
        'death_cross':  df_m['death_cross'].fillna(False),
        'bullish_div':  df_m['bullish_div'].fillna(False),
        'bearish_div':  df_m['bearish_div'].fillna(False),
        'rsi_above_ma': df_m['rsi_above_ma'].fillna(False) & has_rsi,
        'rsi_below_ma': (~df_m['rsi_above_ma'].fillna(False)) & has_rsi,
    }

    rsi_stats = {}
    for state, mask in state_masks.items():
        sub = df_m[mask]
        if len(sub) < 2:
            continue
        rsi_stats[state] = {
            'label': RSI_STATE_LABELS[state],
            'count': int(len(sub)),
            'win_rate': float(sub['win'].mean() * 100),
            'avg_ret': float(sub[ret_col].mean()),
            'total_ret': float(sub[ret_col].sum()),
        }

    rsi_b64 = make_rsi_bar(rsi_stats, cfg['name'], cfg['unit_label'], cfg['unit'])

    # Best/worst cells for summary
    valid_mask = ~np.isnan(wr_mat) & (cnt_mat >= 5)
    best_wr = worst_wr = None
    if valid_mask.any():
        best_idx = np.unravel_index(np.nanargmax(np.where(valid_mask, wr_mat, np.nan)), wr_mat.shape)
        worst_idx = np.unravel_index(np.nanargmin(np.where(valid_mask, wr_mat, np.nan)), wr_mat.shape)
        best_wr = {
            'hour': int(hours[best_idx[0]]), 'dow': DAY_NAMES[best_idx[1]],
            'win_rate': float(wr_mat[best_idx]), 'avg_ret': float(ret_mat[best_idx]),
            'count': int(cnt_mat[best_idx]),
        }
        worst_wr = {
            'hour': int(hours[worst_idx[0]]), 'dow': DAY_NAMES[worst_idx[1]],
            'win_rate': float(wr_mat[worst_idx]), 'avg_ret': float(ret_mat[worst_idx]),
            'count': int(cnt_mat[worst_idx]),
        }

    return {
        'name': cfg['name'],
        'unit_label': cfg['unit_label'],
        'unit': cfg['unit'],
        'n_total': int(len(df60)),
        'n_rsi_overlap': int(has_rsi.sum()),
        'hours': [int(h) for h in hours],
        'wr_heatmap_b64': wr_b64,
        'ret_heatmap_b64': ret_b64,
        'rsi_filter_b64': rsi_b64,
        'rsi_stats': rsi_stats,
        'best_wr': best_wr,
        'worst_wr': worst_wr,
    }


def run():
    print("=== Shared Analysis ===")
    results = {}
    for cid, cfg in COMMODITIES.items():
        results[cid] = analyze_commodity(cid, cfg)

    OUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f"\nSaved → {OUT_JSON}")
    return results


if __name__ == '__main__':
    run()
