#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_week.py — Analyze a week of multi-machine fileiotest data.

Reads the combined CSV from all machines, groups by connection type
(direct wire vs university switch), and produces statistical comparisons,
time-of-day/day-of-week breakdowns, and publication-quality plots.

Usage:
    python3 analyze_week.py <all_results.csv> [--output-dir ./analysis]
    python3 analyze_week.py all_results.csv --direct-wire <wire-source-hostname>

Dependencies:
    pip install pandas matplotlib scipy
"""

__author__ = 'João Tonini / Claude'
__version__ = '0.3'

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

# ─── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

COLOR_DIRECT = '#2171b5'
COLOR_SWITCH = '#cb181d'
COLOR_DIRECT_LIGHT = '#9ecae1'
COLOR_SWITCH_LIGHT = '#fc9272'

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def normalize_rate(rate_str) -> float:
    """Convert pv rate strings to Mbit/s.

    Handles George's format: 763Mib/s, 31.8MiB/s, etc.
    """
    import re
    if not isinstance(rate_str, str) or not rate_str.strip():
        return np.nan

    m = re.match(r'([\d.]+)\s*([A-Za-z/]+)', rate_str.strip())
    if not m:
        return np.nan

    value = float(m.group(1))
    unit = m.group(2)

    # Bit-based (pv -8 flag or default): Mib/s, Gib/s, Kib/s
    if 'ib/s' in unit:
        if unit[0] in 'Gg': return value * 1000.0
        elif unit[0] in 'Mm': return value
        elif unit[0] in 'Kk': return value / 1000.0
        else: return value / 1_000_000.0

    # Byte-based: MiB/s, GiB/s, KiB/s
    if 'iB/s' in unit or 'B/s' in unit:
        if unit[0] in 'Gg': return value * 8.0 * 1024.0
        elif unit[0] in 'Mm': return value * 8.0
        elif unit[0] in 'Kk': return value * 8.0 / 1024.0
        else: return value * 8.0 / (1024.0 * 1024.0)

    return np.nan


def load_and_prepare(filepath: str, direct_wire: str) -> pd.DataFrame:
    """Load combined CSV and add derived columns."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['day_name'] = df['timestamp'].dt.day_name().str[:3]
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5

    # Connection type
    df['connection'] = df['source'].apply(
        lambda s: 'Direct Wire' if s == direct_wire else 'University Switch'
    )

    # Normalize rates to Mbit/s
    rate_cols = {
        'cold_cache_rate': 'cold_cache_mbps',
        'true_write_rate': 'true_write_mbps',
        'hot_cache_run1_rate': 'hot_run1_mbps',
        'hot_cache_run2_rate': 'hot_run2_mbps',
        'hot_cache_run3_rate': 'hot_run3_mbps',
    }
    for src, dst in rate_cols.items():
        if src in df.columns:
            df[dst] = df[src].apply(normalize_rate)

    # Average of 3 hot-cache runs
    hot_cols = ['hot_run1_mbps', 'hot_run2_mbps', 'hot_run3_mbps']
    available_hot = [c for c in hot_cols if c in df.columns]
    if available_hot:
        df['hot_cache_avg_mbps'] = df[available_hot].mean(axis=1)

    # Ensure numeric columns
    for col in ['ping_avg_ms', 'ping_min_ms', 'ping_max_ms', 'ping_mdev_ms',
                'ping_loss_pct',
                'cold_cache_retrans', 'cold_cache_timeouts', 'cold_cache_outrsts',
                'hot_cache_retrans', 'hot_cache_timeouts', 'hot_cache_outrsts',
                'true_write_retrans', 'true_write_timeouts', 'true_write_outrsts']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Total TCP retransmits across all phases
    retrans_cols = ['cold_cache_retrans', 'hot_cache_retrans', 'true_write_retrans']
    available_retrans = [c for c in retrans_cols if c in df.columns]
    if available_retrans:
        df['total_retrans'] = df[available_retrans].sum(axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0


def effect_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"


def significance_report(df: pd.DataFrame) -> str:
    direct = df[df['connection'] == 'Direct Wire']
    switch = df[df['connection'] == 'University Switch']

    lines = []
    lines.append("=" * 76)
    lines.append("STATISTICAL COMPARISON — Direct Wire vs University Switch")
    lines.append(f"  Direct Wire:       {direct['source'].nunique()} machine(s), "
                 f"{len(direct)} samples")
    lines.append(f"  University Switch: {switch['source'].nunique()} machine(s), "
                 f"{len(switch)} samples")
    lines.append(f"  Date range:        {df['timestamp'].min()} → {df['timestamp'].max()}")
    lines.append("=" * 76)

    metrics = [
        ('cold_cache_mbps', 'Cold-Cache Throughput (Mbit/s)'),
        ('true_write_mbps', 'True-Write Throughput (Mbit/s)'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg Throughput (Mbit/s)'),
        ('ping_avg_ms', 'Ping Avg Latency (ms)'),
        ('ping_mdev_ms', 'Ping Jitter / mdev (ms)'),
        ('cold_cache_retrans', 'Cold-Cache TCP Retransmits'),
        ('true_write_retrans', 'True-Write TCP Retransmits'),
        ('total_retrans', 'Total TCP Retransmits (all phases)'),
    ]

    for col, label in metrics:
        if col not in df.columns:
            continue
        va = direct[col].dropna()
        vb = switch[col].dropna()

        if len(va) < 3 or len(vb) < 3:
            lines.append(f"\n--- {label} ---")
            lines.append("  Insufficient data")
            continue

        d = cohens_d(va, vb)
        t_stat, t_p = stats.ttest_ind(va, vb, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(va, vb, alternative='two-sided')
        sig = '***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else ''

        lines.append(f"\n--- {label} ---")
        lines.append(f"  {'Direct Wire':>20s}:  mean={va.mean():10.2f}  "
                     f"std={va.std():8.2f}  median={va.median():10.2f}")
        lines.append(f"  {'University Switch':>20s}:  mean={vb.mean():10.2f}  "
                     f"std={vb.std():8.2f}  median={vb.median():10.2f}")
        lines.append(f"  Difference (A-B):     {va.mean() - vb.mean():+.2f}")
        lines.append(f"  Welch's t-test:       t={t_stat:.3f}, p={t_p:.2e}  {sig}")
        lines.append(f"  Mann-Whitney U:       U={u_stat:.1f}, p={u_p:.2e}")
        lines.append(f"  Cohen's d:            {d:+.3f}  ({effect_label(d)})")

    # Weekday vs Weekend for university switch
    sw_wd = switch[~switch['is_weekend']]
    sw_we = switch[switch['is_weekend']]
    if 'cold_cache_mbps' in switch.columns and len(sw_wd) > 3 and len(sw_we) > 3:
        va = sw_wd['cold_cache_mbps'].dropna()
        vb = sw_we['cold_cache_mbps'].dropna()
        t_stat, t_p = stats.ttest_ind(va, vb, equal_var=False)
        lines.append(f"\n--- University Switch: Weekday vs Weekend (Cold-Cache) ---")
        lines.append(f"  {'Weekday':>20s}:  mean={va.mean():10.2f}  std={va.std():8.2f}")
        lines.append(f"  {'Weekend':>20s}:  mean={vb.mean():10.2f}  std={vb.std():8.2f}")
        lines.append(f"  Welch's t-test:       t={t_stat:.3f}, p={t_p:.2e}")

    lines.append("\n" + "=" * 76)
    lines.append("Significance: * p<0.05  ** p<0.01  *** p<0.001")
    lines.append("=" * 76)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_throughput_timeseries(df, output_dir):
    col = 'cold_cache_mbps'
    if col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    direct = df[df['connection'] == 'Direct Wire']
    switch = df[df['connection'] == 'University Switch']

    for src in switch['source'].unique():
        subset = switch[switch['source'] == src].sort_values('timestamp')
        ax.plot(subset['timestamp'], subset[col], color=COLOR_SWITCH_LIGHT,
                linewidth=0.5, alpha=0.4)

    switch_mean = switch.groupby('timestamp')[col].mean().sort_index()
    ax.plot(switch_mean.index, switch_mean.values, color=COLOR_SWITCH,
            linewidth=1.5, label='Switch (mean)', zorder=3)

    for src in direct['source'].unique():
        subset = direct[direct['source'] == src].sort_values('timestamp')
        ax.plot(subset['timestamp'], subset[col], color=COLOR_DIRECT,
                linewidth=1.5, label='Direct Wire', zorder=4)

    ax.set_ylabel('Throughput (Mbit/s)')
    ax.set_title('Cold-Cache Throughput — Full Week')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '01_timeseries_week.png'))
    plt.close(fig)
    print("  Saved 01_timeseries_week.png")


def plot_hourly_profile(df, output_dir):
    col = 'cold_cache_mbps'
    if col not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, is_we, title in [(ax1, False, 'Weekdays'), (ax2, True, 'Weekends')]:
        subset = df[df['is_weekend'] == is_we]
        for conn, color, light in [
            ('Direct Wire', COLOR_DIRECT, COLOR_DIRECT_LIGHT),
            ('University Switch', COLOR_SWITCH, COLOR_SWITCH_LIGHT),
        ]:
            grp = subset[subset['connection'] == conn]
            if len(grp) == 0:
                continue
            hourly = grp.groupby(grp['timestamp'].dt.hour)[col]
            means = hourly.mean()
            stds = hourly.std().fillna(0)
            ax.plot(means.index, means, color=color, marker='o', linewidth=2,
                    label=conn, zorder=3)
            ax.fill_between(means.index, means - stds, means + stds,
                            color=light, alpha=0.3)
        ax.set_xlabel('Hour of Day')
        ax.set_title(title)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)])
        ax.legend(fontsize=9)

    ax1.set_ylabel('Throughput (Mbit/s)')
    fig.suptitle('Cold-Cache Throughput by Hour — Weekday vs Weekend',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(output_dir, '02_hourly_profile.png'))
    plt.close(fig)
    print("  Saved 02_hourly_profile.png")


def plot_day_of_week(df, output_dir):
    col = 'cold_cache_mbps'
    if col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    positions_d = np.arange(7) * 2.5 - 0.4
    positions_s = np.arange(7) * 2.5 + 0.4

    for conn, positions, color, light in [
        ('Direct Wire', positions_d, COLOR_DIRECT, COLOR_DIRECT_LIGHT),
        ('University Switch', positions_s, COLOR_SWITCH, COLOR_SWITCH_LIGHT),
    ]:
        subset = df[df['connection'] == conn]
        data_by_day = []
        for d in range(7):
            vals = subset[subset['day_of_week'] == d][col].dropna().values
            data_by_day.append(vals if len(vals) > 0 else [np.nan])
        ax.boxplot(data_by_day, positions=positions, widths=0.6,
                   patch_artist=True,
                   boxprops=dict(facecolor=light, edgecolor=color, linewidth=1),
                   medianprops=dict(color='black', linewidth=1.5),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color),
                   flierprops=dict(markeredgecolor=color, markersize=2))

    ax.set_xticks(np.arange(7) * 2.5)
    ax.set_xticklabels(DAY_NAMES)
    ax.set_ylabel('Throughput (Mbit/s)')
    ax.set_title('Cold-Cache Throughput by Day of Week')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLOR_DIRECT_LIGHT, edgecolor=COLOR_DIRECT, label='Direct Wire'),
        Patch(facecolor=COLOR_SWITCH_LIGHT, edgecolor=COLOR_SWITCH, label='University Switch'),
    ])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '03_day_of_week.png'))
    plt.close(fig)
    print("  Saved 03_day_of_week.png")


def plot_distributions(df, output_dir):
    metrics = [
        ('cold_cache_mbps', 'Cold-Cache\nThroughput (Mbit/s)'),
        ('true_write_mbps', 'True-Write\nThroughput (Mbit/s)'),
        ('ping_avg_ms', 'Ping Avg\nLatency (ms)'),
        ('cold_cache_retrans', 'Cold-Cache\nTCP Retransmits'),
    ]
    available = [(c, l) for c, l in metrics
                 if c in df.columns and df[c].notna().sum() > 0]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available):
        data_d = df[df['connection'] == 'Direct Wire'][col].dropna().values
        data_s = df[df['connection'] == 'University Switch'][col].dropna().values

        if len(data_d) > 1:
            vp = ax.violinplot([data_d], positions=[0.8], showmedians=False,
                               showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(COLOR_DIRECT_LIGHT)
                pc.set_edgecolor(COLOR_DIRECT)
                pc.set_alpha(0.6)
        if len(data_s) > 1:
            vp = ax.violinplot([data_s], positions=[1.2], showmedians=False,
                               showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(COLOR_SWITCH_LIGHT)
                pc.set_edgecolor(COLOR_SWITCH)
                pc.set_alpha(0.6)

        box_data, box_pos = [], []
        if len(data_d) > 0:
            box_data.append(data_d); box_pos.append(0.8)
        if len(data_s) > 0:
            box_data.append(data_s); box_pos.append(1.2)
        if box_data:
            bp = ax.boxplot(box_data, positions=box_pos, widths=0.15,
                            patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5),
                            flierprops=dict(markersize=2))
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(COLOR_DIRECT_LIGHT if box_pos[i] < 1.0
                                  else COLOR_SWITCH_LIGHT)

        ax.set_ylabel(label)
        xlabels, xticks = [], []
        if len(data_d) > 0: xlabels.append('Direct\nWire'); xticks.append(0.8)
        if len(data_s) > 0: xlabels.append('Univ.\nSwitch'); xticks.append(1.2)
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlim(0.4, 1.6)

    fig.suptitle('Distribution Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(output_dir, '04_distributions.png'))
    plt.close(fig)
    print("  Saved 04_distributions.png")


def plot_latency_timeseries(df, output_dir):
    if 'ping_avg_ms' not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for conn, color in [('Direct Wire', COLOR_DIRECT),
                        ('University Switch', COLOR_SWITCH)]:
        subset = df[df['connection'] == conn].sort_values('timestamp')
        if len(subset) == 0:
            continue
        if conn == 'University Switch':
            grouped = subset.groupby('timestamp').agg(
                avg=('ping_avg_ms', 'mean'), mdev=('ping_mdev_ms', 'mean')
            ).sort_index()
            ax1.plot(grouped.index, grouped['avg'], color=color,
                     linewidth=1, alpha=0.8, label=conn)
            ax2.plot(grouped.index, grouped['mdev'], color=color,
                     linewidth=1, alpha=0.8, label=conn)
        else:
            ax1.plot(subset['timestamp'], subset['ping_avg_ms'], color=color,
                     linewidth=1.5, label=conn, zorder=3)
            ax2.plot(subset['timestamp'], subset['ping_mdev_ms'], color=color,
                     linewidth=1.5, label=conn, zorder=3)

    ax1.set_ylabel('Avg Latency (ms)')
    ax1.set_title('Ping Latency — Full Week')
    ax1.legend()
    ax2.set_ylabel('Jitter / mdev (ms)')
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '05_latency_week.png'))
    plt.close(fig)
    print("  Saved 05_latency_week.png")


def plot_tcp_retransmits(df, output_dir):
    col = 'cold_cache_retrans'
    if col not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: retransmits over time
    for conn, color in [('Direct Wire', COLOR_DIRECT),
                        ('University Switch', COLOR_SWITCH)]:
        subset = df[df['connection'] == conn].sort_values('timestamp')
        if len(subset) == 0:
            continue
        if conn == 'University Switch':
            grouped = subset.groupby('timestamp')[col].mean().sort_index()
            ax1.plot(grouped.index, grouped.values, color=color,
                     linewidth=1, alpha=0.8, label=conn)
        else:
            ax1.plot(subset['timestamp'], subset[col], color=color,
                     linewidth=1.5, label=conn, zorder=3)

    ax1.set_ylabel('TCP RetransSegs (cold-cache)')
    ax1.set_title('TCP Retransmits Over Time')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    ax1.xaxis.set_major_locator(mdates.DayLocator())

    # Right: scatter — retransmits vs throughput
    mbps_col = 'cold_cache_mbps'
    if mbps_col in df.columns:
        for conn, color, marker in [('Direct Wire', COLOR_DIRECT, 'o'),
                                     ('University Switch', COLOR_SWITCH, 's')]:
            subset = df[df['connection'] == conn].dropna(subset=[mbps_col, col])
            if len(subset) == 0:
                continue
            ax2.scatter(subset[col], subset[mbps_col], c=color, marker=marker,
                        s=12, alpha=0.4, label=conn)
        ax2.set_xlabel('TCP RetransSegs')
        ax2.set_ylabel('Cold-Cache Throughput (Mbit/s)')
        ax2.set_title('Retransmits vs Throughput')
        ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '06_tcp_retransmits.png'))
    plt.close(fig)
    print("  Saved 06_tcp_retransmits.png")


def plot_per_machine_summary(df, output_dir):
    col = 'cold_cache_mbps'
    if col not in df.columns:
        return

    summary = df.groupby('source')[col].agg(['mean', 'std']).sort_values(
        'mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.4)))
    colors = []
    for src in summary.index:
        conn = df[df['source'] == src]['connection'].iloc[0]
        colors.append(COLOR_DIRECT if conn == 'Direct Wire' else COLOR_SWITCH)

    ax.barh(range(len(summary)), summary['mean'], xerr=summary['std'],
            color=colors, edgecolor='white', linewidth=0.5,
            capsize=3, error_kw={'linewidth': 0.8})

    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index)
    ax.set_xlabel('Mean Cold-Cache Throughput (Mbit/s)')
    ax.set_title('Per-Machine Throughput Summary')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLOR_DIRECT, label='Direct Wire'),
        Patch(facecolor=COLOR_SWITCH, label='University Switch'),
    ], loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '07_per_machine.png'))
    plt.close(fig)
    print("  Saved 07_per_machine.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    default_direct = ''
    try:
        from load_config import load_config
        cfg = load_config()
        default_direct = cfg.get('wire', {}).get('source_machine', default_direct)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description='Analyze a week of multi-machine fileiotest data.')
    parser.add_argument('csv', help='Combined results CSV (from orchestrator collect)')
    parser.add_argument('--output-dir', default='./analysis',
                        help='Output directory (default: ./analysis)')
    parser.add_argument('--direct-wire', default=default_direct,
                        help='Hostname of the direct-wire machine (reads from config.toml if available)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df = load_and_prepare(args.csv, args.direct_wire)

    n_machines = df['source'].nunique()
    n_samples = len(df)
    n_direct = len(df[df['connection'] == 'Direct Wire'])
    n_switch = len(df[df['connection'] == 'University Switch'])
    days = (df['timestamp'].max() - df['timestamp'].min()).days

    print(f"  {n_machines} machines, {n_samples} total samples over {days} days")
    print(f"  Direct wire ({args.direct_wire}): {n_direct} samples")
    print(f"  University switch ({n_machines - 1} machines): {n_switch} samples")

    print("\nRunning significance tests...")
    report = significance_report(df)
    print(report)

    report_path = os.path.join(args.output_dir, 'significance_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Report saved → {report_path}")

    print("\nGenerating plots...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_throughput_timeseries(df, args.output_dir)
        plot_hourly_profile(df, args.output_dir)
        plot_day_of_week(df, args.output_dir)
        plot_distributions(df, args.output_dir)
        plot_latency_timeseries(df, args.output_dir)
        plot_tcp_retransmits(df, args.output_dir)
        plot_per_machine_summary(df, args.output_dir)

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()
