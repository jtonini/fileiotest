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

__author__ = 'João Tonini'
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
        df['hot_cache_min_mbps'] = df[available_hot].min(axis=1)
        df['hot_cache_max_mbps'] = df[available_hot].max(axis=1)
        df['hot_cache_std_mbps'] = df[available_hot].std(axis=1)

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
    phases = [
        ('cold_cache_mbps', 'Cold-Cache'),
        ('hot_cache_avg_mbps', 'Hot-Cache (avg of 3 runs)'),
        ('true_write_mbps', 'True Write'),
    ]
    available = [(c, l) for c, l in phases if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(16, 4 * len(available)),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]

    direct = df[df['connection'] == 'Direct Wire']
    switch = df[df['connection'] == 'University Switch']

    for ax, (col, label) in zip(axes, available):
        for src in switch['source'].unique():
            subset = switch[switch['source'] == src].sort_values('timestamp')
            ax.plot(subset['timestamp'], subset[col], color=COLOR_SWITCH_LIGHT,
                    linewidth=0.5, alpha=0.4)

        if len(switch) > 0:
            switch_mean = switch.groupby('timestamp')[col].mean().sort_index()
            ax.plot(switch_mean.index, switch_mean.values, color=COLOR_SWITCH,
                    linewidth=1.5, label='Switch (mean)', zorder=3)

        for src in direct['source'].unique():
            subset = direct[direct['source'] == src].sort_values('timestamp')
            ax.plot(subset['timestamp'], subset[col], color=COLOR_DIRECT,
                    linewidth=1.5, label='Direct Wire', zorder=4)

        ax.set_ylabel('Mbit/s')
        ax.set_title(label)
        ax.legend(fontsize=9, loc='upper right')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.suptitle('Throughput — Full Week (All Phases)', fontsize=13, fontweight='bold')
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, '01_timeseries_week.png'))
    plt.close(fig)
    print("  Saved 01_timeseries_week.png")


def plot_hourly_profile(df, output_dir):
    phases = [
        ('cold_cache_mbps', 'Cold-Cache'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg'),
        ('true_write_mbps', 'True Write'),
    ]
    available = [(c, l) for c, l in phases if c in df.columns]
    if not available:
        return

    fig, axes_grid = plt.subplots(len(available), 2, figsize=(14, 4 * len(available)),
                                  sharey='row')
    if len(available) == 1:
        axes_grid = [axes_grid]

    for row, (col, phase_label) in enumerate(available):
        for col_idx, (is_we, title) in enumerate([(False, 'Weekdays'), (True, 'Weekends')]):
            ax = axes_grid[row][col_idx]
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
            ax.set_xticks(range(0, 24, 3))
            ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)])
            if col_idx == 0:
                ax.set_ylabel(f'{phase_label}\n(Mbit/s)')
            if row == 0:
                ax.set_title(title)
            ax.legend(fontsize=8)

    fig.suptitle('Throughput by Hour — Weekday vs Weekend (All Phases)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, '02_hourly_profile.png'))
    plt.close(fig)
    print("  Saved 02_hourly_profile.png")


def plot_day_of_week(df, output_dir):
    phases = [
        ('cold_cache_mbps', 'Cold-Cache'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg'),
        ('true_write_mbps', 'True Write'),
    ]
    available = [(c, l) for c, l in phases if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 5 * len(available)),
                             sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (col, phase_label) in zip(axes, available):
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
        ax.set_ylabel(f'{phase_label}\n(Mbit/s)')
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=COLOR_DIRECT_LIGHT, edgecolor=COLOR_DIRECT, label='Direct Wire'),
            Patch(facecolor=COLOR_SWITCH_LIGHT, edgecolor=COLOR_SWITCH, label='University Switch'),
        ], fontsize=9)

    fig.suptitle('Throughput by Day of Week — All Phases',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, '03_day_of_week.png'))
    plt.close(fig)
    print("  Saved 03_day_of_week.png")


def plot_distributions(df, output_dir):
    metrics = [
        ('cold_cache_mbps', 'Cold-Cache\nThroughput (Mbit/s)'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg\nThroughput (Mbit/s)'),
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
    phases = [
        ('cold_cache_mbps', 'Cold-Cache'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg'),
        ('true_write_mbps', 'True Write'),
    ]
    available = [(c, l) for c, l in phases if c in df.columns]
    if not available:
        return

    machines = df.groupby('source')[available[0][0]].mean().sort_values().index.tolist()
    n_machines = len(machines)
    n_phases = len(available)

    fig, ax = plt.subplots(figsize=(12, max(4, n_machines * 0.5)))

    bar_height = 0.7 / n_phases
    phase_colors = ['#4292c6', '#ef6548', '#78c679']

    for i, (col, label) in enumerate(available):
        summary = df.groupby('source')[col].agg(['mean', 'std']).reindex(machines)
        positions = np.arange(n_machines) + i * bar_height - (n_phases - 1) * bar_height / 2

        colors = []
        for src in machines:
            conn = df[df['source'] == src]['connection'].iloc[0]
            if conn == 'Direct Wire':
                colors.append(phase_colors[i])
            else:
                colors.append(phase_colors[i])

        ax.barh(positions, summary['mean'], height=bar_height,
                xerr=summary['std'], color=phase_colors[i], alpha=0.8,
                edgecolor='white', linewidth=0.5,
                capsize=2, error_kw={'linewidth': 0.6},
                label=label)

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels(machines)
    ax.set_xlabel('Mean Throughput (Mbit/s)')
    ax.set_title('Per-Machine Throughput Summary — All Phases')

    # Add connection type markers
    for idx, src in enumerate(machines):
        conn = df[df['source'] == src]['connection'].iloc[0]
        marker = '*' if conn == 'Direct Wire' else ''
        if marker:
            ax.annotate(marker, xy=(0, idx), fontsize=14, fontweight='bold',
                        ha='right', va='center', xytext=(-5, 0),
                        textcoords='offset points')

    ax.legend(loc='lower right', fontsize=9)
    from matplotlib.patches import Patch
    handles = ax.get_legend_handles_labels()[0]
    wire_note = Patch(facecolor='none', edgecolor='none', label='* = Direct Wire')
    ax.legend(handles=handles + [wire_note], loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '07_per_machine.png'))
    plt.close(fig)
    print("  Saved 07_per_machine.png")


def plot_phase_decomposition(df, output_dir):
    """Stacked bar chart decomposing throughput differences between phases.

    For each machine (or connection group), shows:
      - True Write throughput (the baseline — slowest, includes all bottlenecks)
      - Receiver disk cost  = Hot Cache - True Write
      - Sender disk cost    = Cold Cache delta from Hot Cache (if cold < hot)

    This decomposes: where does the time go?
    """
    needed = ['hot_cache_avg_mbps', 'true_write_mbps', 'cold_cache_mbps']
    if not all(c in df.columns for c in needed):
        return

    # Group by source
    summary = df.groupby('source').agg(
        hot_mean=('hot_cache_avg_mbps', 'mean'),
        cold_mean=('cold_cache_mbps', 'mean'),
        write_mean=('true_write_mbps', 'mean'),
        connection=('connection', 'first'),
    ).sort_values('hot_mean', ascending=True)

    machines = summary.index.tolist()
    n = len(machines)

    # Decompose into stacked components
    true_write = summary['write_mean'].values
    disk_write_cost = np.maximum(0, summary['hot_mean'].values - summary['write_mean'].values)
    disk_read_cost = np.maximum(0, summary['hot_mean'].values - summary['cold_mean'].values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, n * 0.5)),
                                    gridspec_kw={'width_ratios': [2, 1]})

    # Left: stacked horizontal bars
    y = np.arange(n)
    colors_stack = ['#41ab5d', '#fc8d59', '#4292c6']

    ax1.barh(y, true_write, height=0.6, color=colors_stack[0],
             edgecolor='white', linewidth=0.5, label='True Write (net + disk write)')
    ax1.barh(y, disk_write_cost, height=0.6, left=true_write,
             color=colors_stack[1], edgecolor='white', linewidth=0.5,
             label='Receiver disk write cost')
    # Show cold < hot as a marker (sender disk read slows things down)
    for i, cost in enumerate(disk_read_cost):
        if cost > 0.5:  # only annotate meaningful differences
            ax1.plot(summary['cold_mean'].values[i], i, 'v', color='#e31a1c',
                     markersize=8, zorder=5)

    ax1.set_yticks(y)
    ax1.set_yticklabels(machines)
    ax1.set_xlabel('Throughput (Mbit/s)')
    ax1.set_title('Throughput Decomposition by Phase')

    # Add connection type markers
    for idx, src in enumerate(machines):
        conn = summary.loc[src, 'connection']
        if conn == 'Direct Wire':
            ax1.annotate('*', xy=(0, idx), fontsize=14, fontweight='bold',
                         ha='right', va='center', xytext=(-5, 0),
                         textcoords='offset points')

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=colors_stack[0], label='True Write throughput'),
        Patch(facecolor=colors_stack[1], label='+ Receiver disk write cost\n  (hot cache - true write)'),
        Line2D([0], [0], marker='v', color='#e31a1c', linestyle='None',
               markersize=8, label='Cold cache rate\n  (sender reads from disk)'),
        Patch(facecolor='none', edgecolor='none', label='* = Direct Wire'),
    ]
    ax1.legend(handles=handles, loc='lower right', fontsize=8)

    # Right: summary by connection type
    for conn, color, offset in [('Direct Wire', COLOR_DIRECT, -0.15),
                                 ('University Switch', COLOR_SWITCH, 0.15)]:
        grp = summary[summary['connection'] == conn]
        if len(grp) == 0:
            continue

        means = [grp['cold_mean'].mean(), grp['hot_mean'].mean(), grp['write_mean'].mean()]
        labels = ['Cold\nCache', 'Hot\nCache', 'True\nWrite']
        x = np.arange(3) + offset

        ax2.bar(x, means, width=0.25, color=color, alpha=0.8,
                edgecolor='white', label=conn)

        # Annotate deltas
        if len(means) == 3 and means[1] > 0:
            # Hot → True Write delta
            delta_write = means[1] - means[2]
            if abs(delta_write) > 0.5:
                mid_y = (means[1] + means[2]) / 2
                ax2.annotate(f'{delta_write:+.1f}',
                             xy=(2 + offset, mid_y), fontsize=7,
                             ha='center', color=color, fontweight='bold')

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Cold\nCache', 'Hot\nCache', 'True\nWrite'])
    ax2.set_ylabel('Mean Throughput (Mbit/s)')
    ax2.set_title('Phase Comparison\n(by connection type)')
    ax2.legend(fontsize=8)

    fig.suptitle('Bottleneck Decomposition — Where Does the Time Go?',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, '08_phase_decomposition.png'))
    plt.close(fig)
    print("  Saved 08_phase_decomposition.png")


def plot_hot_cache_spread(df, output_dir):
    """Hot-cache measurement reliability plot.

    For each sample, 3 hot-cache runs are taken. This plot shows:
      - Individual run values as faint dots
      - Per-sample mean as a line
      - Min-max shading (range of the 3 runs per sample)
      - 95% and 99% confidence interval bands (across samples over time)

    This demonstrates measurement consistency and statistical reliability.
    """
    needed = ['hot_cache_avg_mbps', 'hot_cache_min_mbps', 'hot_cache_max_mbps',
              'hot_run1_mbps', 'hot_run2_mbps', 'hot_run3_mbps']
    if not all(c in df.columns for c in needed):
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    for conn, color, light in [
        ('Direct Wire', COLOR_DIRECT, COLOR_DIRECT_LIGHT),
        ('University Switch', COLOR_SWITCH, COLOR_SWITCH_LIGHT),
    ]:
        subset = df[df['connection'] == conn].sort_values('timestamp')
        if len(subset) == 0:
            continue

        ts = subset['timestamp']
        avg = subset['hot_cache_avg_mbps']
        lo = subset['hot_cache_min_mbps']
        hi = subset['hot_cache_max_mbps']

        ax = axes[0]

        # Individual runs as faint dots
        for run_col in ['hot_run1_mbps', 'hot_run2_mbps', 'hot_run3_mbps']:
            ax.scatter(ts, subset[run_col], color=color, s=6, alpha=0.15, zorder=1)

        # Min-max shaded range (spread of 3 runs per sample)
        ax.fill_between(ts, lo, hi, color=light, alpha=0.3,
                        label=f'{conn} run spread (min-max)', zorder=2)

        # Mean line
        ax.plot(ts, avg, color=color, linewidth=1.5,
                label=f'{conn} mean', zorder=4)

        # Rolling confidence intervals (using expanding window across samples)
        if len(subset) >= 5:
            # Compute rolling stats for CI bands
            rolling = subset['hot_cache_avg_mbps'].expanding(min_periods=3)
            roll_mean = rolling.mean()
            roll_std = rolling.std()
            roll_n = rolling.count()

            # 95% CI (z=1.96) and 99% CI (z=2.576)
            se = roll_std / np.sqrt(roll_n)
            ci95_lo = roll_mean - 1.96 * se
            ci95_hi = roll_mean + 1.96 * se
            ci99_lo = roll_mean - 2.576 * se
            ci99_hi = roll_mean + 2.576 * se

            ax.fill_between(ts, ci99_lo, ci99_hi, color=color, alpha=0.08,
                            label=f'{conn} 99% CI', zorder=3)
            ax.fill_between(ts, ci95_lo, ci95_hi, color=color, alpha=0.15,
                            label=f'{conn} 95% CI', zorder=3)

        # Bottom panel: per-sample spread (max - min)
        spread = hi - lo
        axes[1].bar(ts, spread, width=pd.Timedelta(minutes=2), color=color,
                    alpha=0.6, label=f'{conn} run spread')

    axes[0].set_ylabel('Hot-Cache Throughput (Mbit/s)')
    axes[0].set_title('Hot-Cache Runs — Measurement Spread and Confidence Intervals')
    axes[0].legend(fontsize=8, loc='upper right', ncol=2)

    axes[1].set_ylabel('Spread (Mbit/s)\n(max - min of 3 runs)')
    axes[1].set_xlabel('Time')
    axes[1].legend(fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d %H:%M'))
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '09_hot_cache_spread.png'))
    plt.close(fig)
    print("  Saved 09_hot_cache_spread.png")


def plot_heatmap(df, output_dir):
    """Hour-of-day x day-of-week heatmap for each connection type.

    Color-coded throughput makes business-hours degradation
    immediately visible at a glance.
    """
    phases = [
        ('cold_cache_mbps', 'Cold-Cache'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg'),
        ('true_write_mbps', 'True Write'),
    ]
    available = [(c, l) for c, l in phases if c in df.columns]
    if not available:
        return

    connections = []
    if len(df[df['connection'] == 'Direct Wire']) > 0:
        connections.append('Direct Wire')
    if len(df[df['connection'] == 'University Switch']) > 0:
        connections.append('University Switch')
    if not connections:
        return

    n_conn = len(connections)
    n_phase = len(available)

    fig, axes = plt.subplots(n_phase, n_conn,
                             figsize=(6 * n_conn, 3.5 * n_phase),
                             squeeze=False)

    for row, (col, phase_label) in enumerate(available):
        # Get global min/max for consistent color scale across connection types
        vmin = df[col].quantile(0.02) if df[col].notna().sum() > 0 else 0
        vmax = df[col].quantile(0.98) if df[col].notna().sum() > 0 else 1

        for ci, conn in enumerate(connections):
            ax = axes[row][ci]
            subset = df[df['connection'] == conn]

            # Build pivot: rows=hour (0-23), cols=day (Mon-Sun)
            subset = subset.copy()
            subset['hour_int'] = subset['timestamp'].dt.hour
            pivot = subset.pivot_table(
                values=col, index='hour_int', columns='day_of_week',
                aggfunc='mean'
            )
            # Ensure full 24h x 7day grid
            pivot = pivot.reindex(index=range(24), columns=range(7))

            im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                           vmin=vmin, vmax=vmax, interpolation='nearest',
                           origin='upper')

            ax.set_yticks(range(0, 24, 3))
            ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
            ax.set_xticks(range(7))
            ax.set_xticklabels(DAY_NAMES, fontsize=9)

            if row == 0:
                ax.set_title(f'{conn}', fontsize=11, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(f'{phase_label}\nHour of Day')

            fig.colorbar(im, ax=ax, shrink=0.8, label='Mbit/s')

    fig.suptitle('Throughput Heatmap — Hour of Day x Day of Week',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, '10_heatmap.png'))
    plt.close(fig)
    print("  Saved 10_heatmap.png")


def plot_cdf(df, output_dir):
    """Cumulative distribution function (CDF) for all throughput phases.

    Standard in network benchmarking — shows "X% of transfers
    achieved at least Y Mbit/s". Makes it easy to state things like
    "95% of direct wire transfers exceeded 4 Gbps."
    """
    phases = [
        ('cold_cache_mbps', 'Cold-Cache'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg'),
        ('true_write_mbps', 'True Write'),
    ]
    available = [(c, l) for c, l in phases if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available),
                             figsize=(5.5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, phase_label) in zip(axes, available):
        for conn, color, ls in [
            ('Direct Wire', COLOR_DIRECT, '-'),
            ('University Switch', COLOR_SWITCH, '-'),
        ]:
            subset = df[df['connection'] == conn][col].dropna().sort_values()
            if len(subset) == 0:
                continue

            n = len(subset)
            cdf = np.arange(1, n + 1) / n

            ax.plot(subset.values, cdf, color=color, linewidth=2,
                    linestyle=ls, label=conn)

            # Mark key percentiles
            for pct, marker, ms in [(0.50, 'o', 6), (0.95, 's', 6), (0.99, 'D', 5)]:
                idx = int(pct * n) - 1
                if 0 <= idx < n:
                    ax.plot(subset.values[idx], pct, marker=marker, color=color,
                            markersize=ms, zorder=5)

        ax.set_xlabel(f'{phase_label} Throughput (Mbit/s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(phase_label)

        # Reference lines at key percentiles
        for pct, label in [(0.50, '50th'), (0.95, '95th'), (0.99, '99th')]:
            ax.axhline(pct, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.text(ax.get_xlim()[0], pct + 0.01, label, fontsize=7,
                    color='gray', alpha=0.7)

        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9)

    fig.suptitle('Cumulative Distribution Function (CDF) — Throughput',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, '11_cdf.png'))
    plt.close(fig)
    print("  Saved 11_cdf.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    default_direct = ''
    try:
        from load_config import load_config
        cfg = load_config()
        default_direct = cfg.get('wire', {}).get('source_machine', default_direct)
    except (Exception, SystemExit):
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
        plot_phase_decomposition(df, args.output_dir)
        plot_hot_cache_spread(df, args.output_dir)
        plot_heatmap(df, args.output_dir)
        plot_cdf(df, args.output_dir)

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()
