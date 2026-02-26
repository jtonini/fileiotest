#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_iperf3.py — Analysis of iperf3 + fileiotest data with plots.

Companion to analyze_week.py — uses the same visual style.

Includes:
  - Time-sync filtering to thais_wire window
  - iperf3 summary per machine (theoretical max)
  - Overhead calculation: iperf3 vs fileiotest phases
  - NIC tier breakdown with iperf3
  - Wire vs switch comparison
  - Comprehensive plots saved to output directory

Usage:
    python3 analyze_iperf3.py all_results.csv [--outdir analysis_iperf3]
"""

__author__ = 'Joao Tonini'
__version__ = '0.1'

import sys
import os
import argparse
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

# ─── Style (matches analyze_week.py) ─────────────────────────────────
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

# Colors — same as analyze_week.py
COLOR_DIRECT = '#2171b5'
COLOR_SWITCH = '#cb181d'
COLOR_DIRECT_LIGHT = '#9ecae1'
COLOR_SWITCH_LIGHT = '#fc9272'

# Additional colors for iperf3 / phases
COLOR_IPERF3 = '#41ab5d'      # green (matches phase_decomposition)
COLOR_COLD = '#4292c6'         # blue (matches per_machine phase colors)
COLOR_HOT = '#ef6548'          # red-orange
COLOR_WRITE = '#78c679'        # light green

# NIC tier colors
COLOR_1G = '#4292c6'
COLOR_2_5G = '#fe9929'
COLOR_5G = '#e31a1c'

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# ─── NIC speed tiers (Mbps) ──────────────────────────────────────────
NIC_SPEEDS = {
    'aamy': 1000, 'alexis': 1000, 'boyi': 1000,
    'josh': 1000, 'justin': 1000, 'khanh': 1000,
    'camryn': 2500, 'irene2': 2500,
    'cooper': 5000, 'thais_wire': 5000,
    'kevin': 100, 'mayer': 100, 'evan': 100,
}

WIRE_SOURCE = 'thais_wire'
DROP_SOURCES = ['sarah', 'evan', 'hamilton', 'kevin', 'mayer']


# ─── Helpers ──────────────────────────────────────────────────────────

def normalize_rate(rate_str) -> float:
    """Convert pv rate strings to Mbit/s (same logic as analyze_week.py)."""
    import re
    if not isinstance(rate_str, str) or not rate_str.strip():
        return np.nan
    m = re.match(r'([\d.]+)\s*([A-Za-z/]+)', rate_str.strip())
    if not m:
        return np.nan
    value = float(m.group(1))
    unit = m.group(2)
    if 'ib/s' in unit:
        if unit[0] in 'Gg': return value * 1000.0
        elif unit[0] in 'Mm': return value
        elif unit[0] in 'Kk': return value / 1000.0
        else: return value / 1_000_000.0
    if 'iB/s' in unit or 'B/s' in unit:
        if unit[0] in 'Gg': return value * 8.0 * 1024.0
        elif unit[0] in 'Mm': return value * 8.0
        elif unit[0] in 'Kk': return value * 8.0 / 1024.0
        else: return value * 8.0 / (1024.0 * 1024.0)
    return np.nan


def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0


def effect_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"


def load_data(filepath):
    """Load CSV and compute derived columns."""
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df[~df['source'].isin(DROP_SOURCES)].copy()

    # Time columns
    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['day_name'] = df['timestamp'].dt.day_name().str[:3]
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5

    # Normalize rates to Mbit/s (same as analyze_week.py)
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

    # Hot cache averages
    hot_cols = ['hot_run1_mbps', 'hot_run2_mbps', 'hot_run3_mbps']
    available_hot = [c for c in hot_cols if c in df.columns]
    if available_hot:
        df['hot_cache_avg_mbps'] = df[available_hot].mean(axis=1)

    # iperf3 to Mbit/s
    df['iperf3_sender_mbps'] = pd.to_numeric(df['iperf3_sender_bps'], errors='coerce') / 1e6
    df['iperf3_receiver_mbps'] = pd.to_numeric(df['iperf3_receiver_bps'], errors='coerce') / 1e6
    df['iperf3_retrans'] = pd.to_numeric(df['iperf3_retransmits'], errors='coerce')

    # Connection type
    df['connection'] = df['source'].apply(
        lambda s: 'Direct Wire' if s == WIRE_SOURCE else 'University Switch'
    )

    # NIC tier
    df['nic_speed'] = df['source'].map(NIC_SPEEDS).fillna(0).astype(int)
    df['nic_tier'] = df['nic_speed'].map({
        1000: '1G', 2500: '2.5G', 5000: '5G', 100: '100M'
    })

    # Ping / retransmits as numeric
    for col in ['ping_avg_ms', 'ping_min_ms', 'ping_max_ms', 'ping_mdev_ms',
                'ping_loss_pct',
                'cold_cache_retrans', 'hot_cache_retrans', 'true_write_retrans']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def time_sync_filter(df):
    """Filter to only samples within thais_wire time window."""
    wire = df[df['source'] == WIRE_SOURCE]
    if wire.empty:
        print("WARNING: No thais_wire data found, skipping time sync")
        return df
    t_min = wire['timestamp'].min()
    t_max = wire['timestamp'].max()
    synced = df[(df['timestamp'] >= t_min) & (df['timestamp'] <= t_max)].copy()
    print(f"  Time window: {t_min} -> {t_max}")
    print(f"  Samples: {len(df)} -> {len(synced)} (time-synced)")
    return synced


def print_separator(title):
    print(f"\n{'=' * 76}")
    print(f"  {title}")
    print(f"{'=' * 76}")


# ─── Text analysis ────────────────────────────────────────────────────

def summary_table(df):
    print_separator("PER-MACHINE SUMMARY")
    print("\n  All rates in Mbit/s (median values)")
    print(f"  {'-' * 90}")
    print(f"  {'Machine':<14} {'Conn':<7} {'NIC':<6} {'N':>4}  "
          f"{'iperf3':>8} {'Cold':>8} {'Hot':>8} {'Write':>8} {'Ping':>8}")
    print(f"  {'-' * 90}")
    for src in sorted(df['source'].unique()):
        m = df[df['source'] == src]
        nic = NIC_SPEEDS.get(src, '?')
        conn = 'WIRE' if src == WIRE_SOURCE else 'SWITCH'
        ping = pd.to_numeric(m['ping_avg_ms'], errors='coerce').median()
        print(f"  {src:<14} {conn:<7} {nic:>4}M  {len(m):>4}  "
              f"{m['iperf3_sender_mbps'].median():>8.0f} "
              f"{m['cold_cache_mbps'].median():>8.0f} "
              f"{m['hot_cache_avg_mbps'].median():>8.0f} "
              f"{m['true_write_mbps'].median():>8.0f} "
              f"{ping:>8.2f}" if not pd.isna(ping) else f"{'-':>8}")
    print(f"  {'-' * 90}")


def overhead_analysis(df):
    print_separator("OVERHEAD ANALYSIS: iperf3 (theoretical) vs fileiotest (real-world)")
    for conn in ['Direct Wire', 'University Switch']:
        m = df[df['connection'] == conn]
        if m.empty:
            continue
        iperf = m['iperf3_sender_mbps'].dropna()
        cold = m['cold_cache_mbps'].dropna()
        hot = m['hot_cache_avg_mbps'].dropna()
        true_w = m['true_write_mbps'].dropna()
        print(f"\n  {conn} ({len(m)} samples)")
        print(f"  {'-' * 60}")
        print(f"  {'Metric':<25} {'Median':>10} {'Mean':>10} {'Std':>10}")
        print(f"  {'-' * 60}")
        for name, series in [('iperf3 (raw)', iperf), ('Cold cache', cold),
                             ('Hot cache', hot), ('True write', true_w)]:
            if len(series) > 0:
                print(f"  {name:<25} {series.median():>10.0f} "
                      f"{series.mean():>10.0f} {series.std():>10.0f}")
        print(f"  {'-' * 60}")
        if len(iperf) > 0 and iperf.median() > 0:
            for name, series in [('Cold cache', cold), ('Hot cache', hot),
                                 ('True write', true_w)]:
                if len(series) > 0:
                    overhead = (1 - series.median() / iperf.median()) * 100
                    print(f"  {name + ' overhead':<25} {overhead:>9.1f}%  (vs iperf3)")


def nic_tier_analysis(df):
    print_separator("NIC TIER ANALYSIS (Switch machines only)")
    switch = df[df['connection'] == 'University Switch']
    if switch.empty:
        return
    print(f"\n  {'Tier':<8} {'N':>5}  {'iperf3':>10} {'Cold':>10} "
          f"{'Hot':>10} {'Write':>10} {'Ping_ms':>10}")
    print(f"  {'-' * 75}")
    for tier in ['1G', '2.5G', '5G']:
        t = switch[switch['nic_tier'] == tier]
        if t.empty:
            continue
        print(f"  {tier:<8} {len(t):>5}  "
              f"{t['iperf3_sender_mbps'].median():>10.0f} "
              f"{t['cold_cache_mbps'].median():>10.0f} "
              f"{t['hot_cache_avg_mbps'].median():>10.0f} "
              f"{t['true_write_mbps'].median():>10.0f} "
              f"{t['ping_avg_ms'].median():>10.2f}")
    print(f"  {'-' * 75}")
    print("  All rates in Mbit/s (median)")


def wire_vs_switch_report(df):
    print_separator("WIRE vs SWITCH COMPARISON")
    wire = df[df['connection'] == 'Direct Wire']
    switch = df[df['connection'] == 'University Switch']
    if wire.empty or switch.empty:
        print("  Insufficient data for comparison")
        return
    print(f"\n  Wire: {len(wire)} samples ({wire['source'].nunique()} machine)")
    print(f"  Switch: {len(switch)} samples ({switch['source'].nunique()} machines)")
    metrics = [
        ('iperf3 (sender)', 'iperf3_sender_mbps'),
        ('Cold cache', 'cold_cache_mbps'),
        ('Hot cache (avg)', 'hot_cache_avg_mbps'),
        ('True write', 'true_write_mbps'),
    ]
    print(f"\n  {'Metric':<20} {'Wire Med':>10} {'Switch Med':>10} "
          f"{'Delta':>10} {'p-value':>10} {'Cohen d':>10}")
    print(f"  {'-' * 75}")
    for name, col in metrics:
        w = wire[col].dropna()
        s = switch[col].dropna()
        if len(w) < 2 or len(s) < 2:
            continue
        w_med, s_med = w.median(), s.median()
        delta = w_med - s_med
        t_stat, p_val = scipy_stats.ttest_ind(w, s, equal_var=False)
        d = cohens_d(w, s)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {name:<20} {w_med:>10.0f} {s_med:>10.0f} "
              f"{delta:>+10.0f} {p_val:>9.1e} {d:>+9.2f} {sig}")
    w_ping = wire['ping_avg_ms'].dropna()
    s_ping = switch['ping_avg_ms'].dropna()
    if len(w_ping) > 1 and len(s_ping) > 1:
        t_stat, p_val = scipy_stats.ttest_ind(w_ping, s_ping, equal_var=False)
        d = cohens_d(w_ping, s_ping)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {'Ping (ms)':<20} {w_ping.median():>10.2f} {s_ping.median():>10.2f} "
              f"{w_ping.median()-s_ping.median():>+10.2f} {p_val:>9.1e} {d:>+9.2f} {sig}")
    print(f"  {'-' * 75}")
    print("  Rates in Mbit/s | *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


def retransmit_summary(df):
    print_separator("TCP RETRANSMITS")
    print("\n  iperf3:")
    for label, m in [('Wire', df[df['connection'] == 'Direct Wire']),
                     ('Switch', df[df['connection'] == 'University Switch'])]:
        retrans = m['iperf3_retrans'].dropna()
        if len(retrans) == 0:
            continue
        print(f"    {label}: median={retrans.median():.0f}  mean={retrans.mean():.1f}  "
              f"max={retrans.max():.0f}  zero={int((retrans==0).sum())}/{len(retrans)} "
              f"({(retrans==0).mean()*100:.0f}%)")
    print("\n  fileiotest (per phase):")
    for label, m in [('Wire', df[df['connection'] == 'Direct Wire']),
                     ('Switch', df[df['connection'] == 'University Switch'])]:
        if m.empty:
            continue
        print(f"    {label}:")
        for phase, col in [('Cold', 'cold_cache_retrans'),
                           ('Hot', 'hot_cache_retrans'),
                           ('Write', 'true_write_retrans')]:
            vals = m[col].dropna()
            if len(vals) > 0:
                print(f"      {phase:<8} median={vals.median():.0f}  "
                      f"mean={vals.mean():.1f}  total={vals.sum():.0f}")


def weekday_vs_weekend(df):
    print_separator("WEEKDAY vs WEEKEND (Switch only)")
    switch = df[df['connection'] == 'University Switch'].copy()
    if switch.empty:
        return
    for period, label in [(False, 'Weekday'), (True, 'Weekend')]:
        m = switch[switch['is_weekend'] == period]
        if m.empty:
            continue
        cold = m['cold_cache_mbps'].dropna()
        print(f"  {label}: {len(m)} samples, cold cache median = {cold.median():.0f} Mbit/s")
    wd = switch[~switch['is_weekend']]['cold_cache_mbps'].dropna()
    we = switch[switch['is_weekend']]['cold_cache_mbps'].dropna()
    if len(wd) > 1 and len(we) > 1:
        t, p = scipy_stats.ttest_ind(wd, we, equal_var=False)
        print(f"  Difference: p={p:.4f} "
              f"{'(significant)' if p < 0.05 else '(not significant)'}")


# ─── Plotting functions (analyze_week.py style) ──────────────────────

def plot_01_per_machine_bars(df, outdir):
    """Per-machine grouped bar chart: iperf3 vs cold/hot/write."""
    wire_machines = sorted(df[df['source'] == WIRE_SOURCE]['source'].unique())
    switch_df = df[df['source'] != WIRE_SOURCE].groupby('source')['nic_speed'] \
        .first().sort_values(ascending=False)
    machines = list(wire_machines) + list(switch_df.index)
    n = len(machines)

    fig, ax = plt.subplots(figsize=(14, max(5, n * 0.6)))

    bar_height = 0.18
    phase_colors = [COLOR_IPERF3, COLOR_COLD, COLOR_HOT, COLOR_WRITE]
    phase_labels = ['iperf3 (raw)', 'Cold cache', 'Hot cache', 'True write']
    phase_cols = ['iperf3_sender_mbps', 'cold_cache_mbps',
                  'hot_cache_avg_mbps', 'true_write_mbps']

    y = np.arange(n)

    for i, (col, label, color) in enumerate(zip(phase_cols, phase_labels, phase_colors)):
        means = []
        stds = []
        for src in machines:
            vals = df[df['source'] == src][col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)
        offset = (i - 1.5) * bar_height
        ax.barh(y + offset, means, height=bar_height, xerr=stds,
                color=color, alpha=0.8, edgecolor='white', linewidth=0.5,
                capsize=2, error_kw={'linewidth': 0.6}, label=label)

    # NIC link speed markers
    for idx, src in enumerate(machines):
        nic = NIC_SPEEDS.get(src, 0)
        ax.plot(nic, idx, '|', color='black', markersize=15,
                markeredgewidth=2, zorder=5)

    # Wire annotation
    for idx, src in enumerate(machines):
        conn = df[df['source'] == src]['connection'].iloc[0]
        if conn == 'Direct Wire':
            ax.annotate('*', xy=(0, idx), fontsize=14, fontweight='bold',
                        ha='right', va='center', xytext=(-5, 0),
                        textcoords='offset points')

    # Labels with NIC tier
    labels = []
    for src in machines:
        nic = NIC_SPEEDS.get(src, 0)
        tier = {100: '100M', 1000: '1G', 2500: '2.5G', 5000: '5G'}.get(nic, '?')
        labels.append(f"{src} ({tier})")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Throughput (Mbit/s)')
    ax.set_title('Per-Machine Throughput: iperf3 vs fileiotest\n'
                 '(bars = mean +/- std, | = NIC link speed, * = direct wire)')

    handles = ax.get_legend_handles_labels()[0]
    nic_marker = Line2D([0], [0], marker='|', color='black', linestyle='None',
                        markersize=10, markeredgewidth=2, label='NIC link speed')
    wire_note = Patch(facecolor='none', edgecolor='none', label='* = Direct Wire')
    ax.legend(handles=handles + [nic_marker, wire_note],
              loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, '01_per_machine_iperf3_vs_fileiotest.png'))
    plt.close(fig)
    print("  Saved 01_per_machine_iperf3_vs_fileiotest.png")


def plot_02_overhead_waterfall(df, outdir):
    """Overhead waterfall chart for wire and switch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    phase_colors = [COLOR_IPERF3, COLOR_COLD, COLOR_HOT, COLOR_WRITE]

    for ax, (conn, label) in zip(axes, [
        ('Direct Wire', 'Direct Wire (5G)'),
        ('University Switch', 'University Switch (mixed NICs)')
    ]):
        m = df[df['connection'] == conn]
        if m.empty:
            continue

        iperf_med = m['iperf3_sender_mbps'].median()
        cold_med = m['cold_cache_mbps'].median()
        hot_med = m['hot_cache_avg_mbps'].median()
        write_med = m['true_write_mbps'].median()

        categories = ['iperf3\n(raw)', 'Cold\ncache', 'Hot\ncache', 'True\nwrite']
        values = [iperf_med, cold_med, hot_med, write_med]

        bars = ax.bar(categories, values, color=phase_colors, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        for i, val in enumerate(values):
            if i == 0:
                ax.text(i, val + iperf_med * 0.02, f'{val:.0f}',
                        ha='center', fontsize=10, fontweight='bold')
            else:
                overhead = (1 - val / iperf_med) * 100
                ax.text(i, val + iperf_med * 0.02,
                        f'{val:.0f}\n({overhead:.0f}% loss)',
                        ha='center', fontsize=9)

        ax.axhline(y=iperf_med, color=COLOR_IPERF3, linestyle='--',
                   alpha=0.4, linewidth=1)
        ax.set_ylabel('Throughput (Mbit/s)')
        ax.set_title(f'{label}\n({len(m)} samples)')
        ax.set_ylim(0, iperf_med * 1.2)

    fig.suptitle('Protocol Overhead: iperf3 (theoretical max) vs '
                 'fileiotest (real-world file transfer)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, '02_overhead_waterfall.png'))
    plt.close(fig)
    print("  Saved 02_overhead_waterfall.png")


def plot_03_nic_tier_boxplots(df, outdir):
    """NIC tier boxplots with wire — all 4 metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('iperf3 (raw bandwidth)', 'iperf3_sender_mbps', axes[0, 0]),
        ('Cold cache (disk -> network)', 'cold_cache_mbps', axes[0, 1]),
        ('Hot cache (RAM -> network)', 'hot_cache_avg_mbps', axes[1, 0]),
        ('True write (network -> disk)', 'true_write_mbps', axes[1, 1]),
    ]

    tier_info = [
        ('1G', COLOR_1G, '#9ecae1'),
        ('2.5G', COLOR_2_5G, '#fee391'),
        ('5G', COLOR_5G, '#fc9272'),
        ('5G\n(wire)', COLOR_DIRECT, COLOR_DIRECT_LIGHT),
    ]

    for title, col, ax in metrics:
        data_groups = []
        positions = []
        colors_edge = []
        colors_fill = []
        labels = []
        pos = 1

        for tier_label, edge_c, fill_c in tier_info:
            if 'wire' in tier_label:
                vals = df[df['source'] == WIRE_SOURCE][col].dropna()
            else:
                vals = df[(df['nic_tier'] == tier_label) &
                          (df['connection'] == 'University Switch')][col].dropna()
            if len(vals) > 0:
                data_groups.append(vals.values)
                positions.append(pos)
                colors_edge.append(edge_c)
                colors_fill.append(fill_c)
                labels.append(tier_label)
            pos += 1

        if not data_groups:
            continue

        bp = ax.boxplot(data_groups, positions=positions, widths=0.6,
                        patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        for patch, ec, fc in zip(bp['boxes'], colors_edge, colors_fill):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(1.2)

        # NIC speed dashed lines
        tier_limits = {'1G': 1000, '2.5G': 2500, '5G': 5000, '5G\n(wire)': 5000}
        for p, lbl in zip(positions, labels):
            limit = tier_limits.get(lbl, 0)
            ax.plot([p - 0.35, p + 0.35], [limit, limit], 'k--',
                    alpha=0.3, linewidth=1)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Mbit/s')
        ax.set_title(title)

        for p, grp in zip(positions, data_groups):
            ax.text(p, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                    f'n={len(grp)}', ha='center', fontsize=8, color='gray')

    fig.suptitle('Throughput by NIC Tier and Connection Type\n'
                 '(dashed line = NIC link speed)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, '03_nic_tier_boxplots.png'))
    plt.close(fig)
    print("  Saved 03_nic_tier_boxplots.png")


def plot_04_distributions(df, outdir):
    """Wire vs switch violin+box distributions (analyze_week.py style)."""
    metrics = [
        ('iperf3_sender_mbps', 'iperf3\n(Mbit/s)'),
        ('cold_cache_mbps', 'Cold-Cache\n(Mbit/s)'),
        ('hot_cache_avg_mbps', 'Hot-Cache\n(Mbit/s)'),
        ('true_write_mbps', 'True-Write\n(Mbit/s)'),
        ('ping_avg_ms', 'Ping Avg\n(ms)'),
    ]
    available = [(c, l) for c, l in metrics
                 if c in df.columns and df[c].notna().sum() > 0]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available),
                             figsize=(4.5 * len(available), 5))
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
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlim(0.4, 1.6)

    fig.suptitle('Distribution Comparison — Wire vs Switch (with iperf3)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, '04_distributions.png'))
    plt.close(fig)
    print("  Saved 04_distributions.png")


def plot_05_timeseries(df, outdir):
    """Time series of iperf3 and cold cache (analyze_week.py style)."""
    phases = [
        ('iperf3_sender_mbps', 'iperf3 Raw Bandwidth'),
        ('cold_cache_mbps', 'Cold-Cache Throughput'),
        ('hot_cache_avg_mbps', 'Hot-Cache Avg Throughput'),
    ]

    fig, axes = plt.subplots(len(phases), 1, figsize=(16, 4 * len(phases)),
                             sharex=True)

    direct = df[df['connection'] == 'Direct Wire']
    switch = df[df['connection'] == 'University Switch']

    for ax, (col, label) in zip(axes, phases):
        # Individual switch machine traces
        for src in switch['source'].unique():
            subset = switch[switch['source'] == src].sort_values('timestamp')
            ax.plot(subset['timestamp'], subset[col], color=COLOR_SWITCH_LIGHT,
                    linewidth=0.5, alpha=0.4)

        # Switch mean
        if len(switch) > 0:
            sw_mean = switch.groupby('timestamp')[col].mean().sort_index()
            ax.plot(sw_mean.index, sw_mean.values, color=COLOR_SWITCH,
                    linewidth=1.5, label='Switch (mean)', zorder=3)

        # Wire
        for src in direct['source'].unique():
            subset = direct[direct['source'] == src].sort_values('timestamp')
            ax.plot(subset['timestamp'], subset[col], color=COLOR_DIRECT,
                    linewidth=1.5, label='Direct Wire', zorder=4)

        ax.set_ylabel('Mbit/s')
        ax.set_title(label)
        ax.legend(fontsize=9, loc='upper right')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d %H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.suptitle('Throughput Over Time — iperf3 + fileiotest',
                 fontsize=13, fontweight='bold')
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, '05_timeseries.png'))
    plt.close(fig)
    print("  Saved 05_timeseries.png")


def plot_06_retransmits(df, outdir):
    """Retransmit comparison (analyze_week.py style)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: iperf3 retransmits over time
    for conn, color in [('Direct Wire', COLOR_DIRECT),
                        ('University Switch', COLOR_SWITCH)]:
        subset = df[df['connection'] == conn].sort_values('timestamp')
        if len(subset) == 0:
            continue
        if conn == 'University Switch':
            grouped = subset.groupby('timestamp')['iperf3_retrans'].mean().sort_index()
            ax1.plot(grouped.index, grouped.values, color=color,
                     linewidth=1, alpha=0.8, label=conn)
        else:
            ax1.plot(subset['timestamp'], subset['iperf3_retrans'], color=color,
                     linewidth=1.5, label=conn, zorder=3)

    ax1.set_ylabel('iperf3 TCP Retransmits')
    ax1.set_title('iperf3 Retransmits Over Time')
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))

    # Right: scatter — iperf3 retransmits vs throughput
    for conn, color, marker in [('Direct Wire', COLOR_DIRECT, 'o'),
                                 ('University Switch', COLOR_SWITCH, 's')]:
        subset = df[df['connection'] == conn].dropna(
            subset=['iperf3_sender_mbps', 'iperf3_retrans'])
        if len(subset) == 0:
            continue
        ax2.scatter(subset['iperf3_retrans'], subset['iperf3_sender_mbps'],
                    c=color, marker=marker, s=12, alpha=0.4, label=conn)
    ax2.set_xlabel('iperf3 TCP Retransmits')
    ax2.set_ylabel('iperf3 Throughput (Mbit/s)')
    ax2.set_title('Retransmits vs Throughput')
    ax2.legend(fontsize=9)

    fig.suptitle('iperf3 TCP Retransmit Analysis', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, '06_retransmits.png'))
    plt.close(fig)
    print("  Saved 06_retransmits.png")


def plot_07_ping_latency(df, outdir):
    """Ping latency — time series and CDF (analyze_week.py style)."""
    if 'ping_avg_ms' not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: time series
    for conn, color in [('Direct Wire', COLOR_DIRECT),
                        ('University Switch', COLOR_SWITCH)]:
        subset = df[df['connection'] == conn].sort_values('timestamp')
        if len(subset) == 0:
            continue
        if conn == 'University Switch':
            grouped = subset.groupby('timestamp')['ping_avg_ms'].mean().sort_index()
            ax1.plot(grouped.index, grouped.values, color=color,
                     linewidth=1, alpha=0.8, label=conn)
        else:
            ax1.plot(subset['timestamp'], subset['ping_avg_ms'], color=color,
                     linewidth=1.5, label=conn, zorder=3)
    ax1.set_ylabel('Avg Latency (ms)')
    ax1.set_title('Ping Latency Over Time')
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))

    # Right: CDF
    for conn, color in [('University Switch', COLOR_SWITCH),
                        ('Direct Wire', COLOR_DIRECT)]:
        vals = df[df['connection'] == conn]['ping_avg_ms'].dropna().sort_values()
        if len(vals) > 0:
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax2.plot(vals, cdf, color=color, linewidth=2,
                     label=f'{conn} (n={len(vals)})')
    ax2.set_xlabel('Ping RTT (ms)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Ping Latency CDF')
    ax2.legend(fontsize=9)
    for pct in [0.50, 0.95]:
        ax2.axhline(pct, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    fig.suptitle('Network Latency Analysis', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, '07_ping_latency.png'))
    plt.close(fig)
    print("  Saved 07_ping_latency.png")


def plot_08_cdf_all_phases(df, outdir):
    """CDF for all phases including iperf3 (analyze_week.py style)."""
    phases = [
        ('iperf3_sender_mbps', 'iperf3 Raw'),
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
        for conn, color in [('Direct Wire', COLOR_DIRECT),
                            ('University Switch', COLOR_SWITCH)]:
            subset = df[df['connection'] == conn][col].dropna().sort_values()
            if len(subset) == 0:
                continue
            n = len(subset)
            cdf = np.arange(1, n + 1) / n
            ax.plot(subset.values, cdf, color=color, linewidth=2, label=conn)
            # Percentile markers
            for pct, marker, ms in [(0.50, 'o', 6), (0.95, 's', 6)]:
                idx = int(pct * n) - 1
                if 0 <= idx < n:
                    ax.plot(subset.values[idx], pct, marker=marker,
                            color=color, markersize=ms, zorder=5)

        ax.set_xlabel(f'{phase_label} (Mbit/s)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(phase_label)
        for pct, plabel in [(0.50, '50th'), (0.95, '95th')]:
            ax.axhline(pct, color='gray', linewidth=0.5, linestyle=':',
                       alpha=0.5)
            ax.text(ax.get_xlim()[0], pct + 0.01, plabel, fontsize=7,
                    color='gray', alpha=0.7)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9)

    fig.suptitle('Cumulative Distribution Function (CDF) — All Metrics',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, '08_cdf_all_phases.png'))
    plt.close(fig)
    print("  Saved 08_cdf_all_phases.png")


def generate_all_plots(df, outdir):
    print_separator("GENERATING PLOTS")
    os.makedirs(outdir, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_01_per_machine_bars(df, outdir)
        plot_02_overhead_waterfall(df, outdir)
        plot_03_nic_tier_boxplots(df, outdir)
        plot_04_distributions(df, outdir)
        plot_05_timeseries(df, outdir)
        plot_06_retransmits(df, outdir)
        plot_07_ping_latency(df, outdir)
        plot_08_cdf_all_phases(df, outdir)
    print(f"\n  All plots saved to: {outdir}/")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze iperf3 + fileiotest data with plots.')
    parser.add_argument('csv', help='Path to all_results.csv')
    parser.add_argument('--outdir', default='analysis_iperf3',
                        help='Output directory for plots (default: analysis_iperf3)')
    args = parser.parse_args()

    print("=" * 76)
    print("  Network Performance Analysis -- iperf3 + fileiotest")
    print("=" * 76)

    print(f"\nLoading: {args.csv}")
    df = load_data(args.csv)
    n_machines = df['source'].nunique()
    n_samples = len(df)
    n_direct = len(df[df['connection'] == 'Direct Wire'])
    n_switch = len(df[df['connection'] == 'University Switch'])
    print(f"  {n_machines} machines, {n_samples} samples")
    print(f"  Direct Wire ({WIRE_SOURCE}): {n_direct} samples")
    print(f"  University Switch ({n_machines - 1} machines): {n_switch} samples")

    print("\nApplying time-sync filter (thais_wire window)...")
    df = time_sync_filter(df)

    iperf_valid = df['iperf3_sender_mbps'].notna().sum()
    print(f"\n  iperf3 coverage: {iperf_valid}/{len(df)} samples "
          f"({iperf_valid/len(df)*100:.0f}%)")

    summary_table(df)
    overhead_analysis(df)
    nic_tier_analysis(df)
    wire_vs_switch_report(df)
    retransmit_summary(df)
    weekday_vs_weekend(df)

    generate_all_plots(df, args.outdir)

    print(f"\n{'=' * 76}")
    print(f"  Analysis complete. Plots in: {args.outdir}/")
    print(f"{'=' * 76}\n")


if __name__ == '__main__':
    main()
