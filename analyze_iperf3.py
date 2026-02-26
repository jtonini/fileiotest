#!/usr/bin/env python3
"""analyze_iperf3.py — Analysis of iperf3 + fileiotest data with plots.

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

import sys
import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ─── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'iperf3': '#2196F3',      # blue
    'cold': '#FF9800',        # orange
    'hot': '#F44336',         # red
    'write': '#4CAF50',       # green
    'wire': '#9C27B0',        # purple
    'switch': '#607D8B',      # blue-grey
    '1G': '#42A5F5',
    '2.5G': '#FFA726',
    '5G': '#EF5350',
    '5G_wire': '#AB47BC',
}

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


def parse_rate_to_mbits(rate_str):
    """Convert rate string like '276MiB/s' to Mbit/s."""
    if pd.isna(rate_str) or rate_str == '':
        return np.nan
    s = str(rate_str).strip().rstrip('/s')
    try:
        if 'GiB' in s:
            return float(s.replace('GiB', '')) * 8 * 1024
        elif 'MiB' in s:
            return float(s.replace('MiB', '')) * 8 * 1024 / 1000
        elif 'KiB' in s:
            return float(s.replace('KiB', '')) * 8 / 1000
        elif 'GB' in s:
            return float(s.replace('GB', '')) * 8 * 1000
        elif 'MB' in s:
            return float(s.replace('MB', '')) * 8
        elif 'kB' in s:
            return float(s.replace('kB', '')) * 8 / 1000
        else:
            return float(s) * 8
    except (ValueError, TypeError):
        return np.nan


def load_data(filepath):
    """Load CSV and compute derived columns."""
    df = pd.read_csv(filepath, parse_dates=['timestamp'])

    # Filter
    df = df[~df['source'].isin(DROP_SOURCES)].copy()

    # Convert rates to Mbit/s
    for col in ['cold_cache_rate', 'true_write_rate',
                'hot_cache_run1_rate', 'hot_cache_run2_rate', 'hot_cache_run3_rate']:
        df[col + '_mbits'] = df[col].apply(parse_rate_to_mbits)

    # Average hot cache
    hot_cols = ['hot_cache_run1_rate_mbits', 'hot_cache_run2_rate_mbits', 'hot_cache_run3_rate_mbits']
    df['hot_cache_avg_mbits'] = df[hot_cols].mean(axis=1)

    # iperf3 to Mbit/s
    df['iperf3_sender_mbits'] = pd.to_numeric(df['iperf3_sender_bps'], errors='coerce') / 1e6
    df['iperf3_receiver_mbits'] = pd.to_numeric(df['iperf3_receiver_bps'], errors='coerce') / 1e6
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

    # Ping
    df['ping_avg'] = pd.to_numeric(df['ping_avg_ms'], errors='coerce')

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
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ─── Text analysis functions ─────────────────────────────────────────

def summary_table(df):
    """Per-machine summary with iperf3 and fileiotest."""
    print_separator("PER-MACHINE SUMMARY")
    rows = []
    for src in sorted(df['source'].unique()):
        m = df[df['source'] == src]
        n = len(m)
        nic = NIC_SPEEDS.get(src, '?')
        conn = 'WIRE' if src == WIRE_SOURCE else 'SWITCH'
        iperf_med = m['iperf3_sender_mbits'].median()
        cold_med = m['cold_cache_rate_mbits'].median()
        hot_med = m['hot_cache_avg_mbits'].median()
        true_med = m['true_write_rate_mbits'].median()
        ping_med = m['ping_avg'].median()
        rows.append({
            'Machine': src, 'Conn': conn, 'NIC': f"{nic}M", 'N': n,
            'iperf3': f"{iperf_med:.0f}", 'Cold': f"{cold_med:.0f}",
            'Hot': f"{hot_med:.0f}", 'Write': f"{true_med:.0f}",
            'Ping_ms': f"{ping_med:.2f}" if not pd.isna(ping_med) else '-',
        })
    tbl = pd.DataFrame(rows)
    print("\n  All rates in Mbit/s (median values)")
    print(f"  {'-' * 90}")
    print(f"  {'Machine':<14} {'Conn':<7} {'NIC':<6} {'N':>4}  {'iperf3':>8} {'Cold':>8} {'Hot':>8} {'Write':>8} {'Ping':>8}")
    print(f"  {'-' * 90}")
    for _, r in tbl.iterrows():
        print(f"  {r['Machine']:<14} {r['Conn']:<7} {r['NIC']:<6} {r['N']:>4}  {r['iperf3']:>8} {r['Cold']:>8} {r['Hot']:>8} {r['Write']:>8} {r['Ping_ms']:>8}")
    print(f"  {'-' * 90}")


def overhead_analysis(df):
    """iperf3 vs fileiotest overhead per connection type."""
    print_separator("OVERHEAD ANALYSIS: iperf3 (theoretical) vs fileiotest (real-world)")
    for conn in ['Direct Wire', 'University Switch']:
        m = df[df['connection'] == conn]
        if m.empty:
            continue
        iperf = m['iperf3_sender_mbits'].dropna()
        cold = m['cold_cache_rate_mbits'].dropna()
        hot = m['hot_cache_avg_mbits'].dropna()
        true_w = m['true_write_rate_mbits'].dropna()
        print(f"\n  {conn} ({len(m)} samples)")
        print(f"  {'-' * 60}")
        print(f"  {'Metric':<25} {'Median':>10} {'Mean':>10} {'Std':>10}")
        print(f"  {'-' * 60}")
        for name, series in [('iperf3 (raw)', iperf), ('Cold cache', cold),
                             ('Hot cache', hot), ('True write', true_w)]:
            if len(series) > 0:
                print(f"  {name:<25} {series.median():>10.0f} {series.mean():>10.0f} {series.std():>10.0f}")
        print(f"  {'-' * 60}")
        if len(iperf) > 0 and iperf.median() > 0:
            for name, series in [('Cold cache', cold), ('Hot cache', hot), ('True write', true_w)]:
                if len(series) > 0:
                    overhead = (1 - series.median() / iperf.median()) * 100
                    print(f"  {name + ' overhead':<25} {overhead:>9.1f}%  (vs iperf3)")


def nic_tier_analysis(df):
    """iperf3 and fileiotest by NIC tier for switch machines only."""
    print_separator("NIC TIER ANALYSIS (Switch machines only)")
    switch = df[df['connection'] == 'University Switch']
    if switch.empty:
        return
    print(f"\n  {'Tier':<8} {'N':>5}  {'iperf3':>10} {'Cold':>10} {'Hot':>10} {'Write':>10} {'Ping_ms':>10}")
    print(f"  {'-' * 75}")
    for tier in ['1G', '2.5G', '5G']:
        t = switch[switch['nic_tier'] == tier]
        if t.empty:
            continue
        print(f"  {tier:<8} {len(t):>5}  "
              f"{t['iperf3_sender_mbits'].median():>10.0f} "
              f"{t['cold_cache_rate_mbits'].median():>10.0f} "
              f"{t['hot_cache_avg_mbits'].median():>10.0f} "
              f"{t['true_write_rate_mbits'].median():>10.0f} "
              f"{t['ping_avg'].median():>10.2f}")
    print(f"  {'-' * 75}")
    print("  All rates in Mbit/s (median)")


def wire_vs_switch(df):
    """Statistical comparison: wire vs switch."""
    print_separator("WIRE vs SWITCH COMPARISON")
    wire = df[df['connection'] == 'Direct Wire']
    switch = df[df['connection'] == 'University Switch']
    if wire.empty or switch.empty:
        print("  Insufficient data for comparison")
        return
    print(f"\n  Wire: {len(wire)} samples ({wire['source'].nunique()} machine)")
    print(f"  Switch: {len(switch)} samples ({switch['source'].nunique()} machines)")
    metrics = [
        ('iperf3 (sender)', 'iperf3_sender_mbits'),
        ('Cold cache', 'cold_cache_rate_mbits'),
        ('Hot cache (avg)', 'hot_cache_avg_mbits'),
        ('True write', 'true_write_rate_mbits'),
    ]
    print(f"\n  {'Metric':<20} {'Wire Med':>10} {'Switch Med':>10} {'Delta':>10} {'p-value':>10} {'Cohen d':>10}")
    print(f"  {'-' * 75}")
    for name, col in metrics:
        w = wire[col].dropna()
        s = switch[col].dropna()
        if len(w) < 2 or len(s) < 2:
            continue
        w_med, s_med = w.median(), s.median()
        delta = w_med - s_med
        t_stat, p_val = scipy_stats.ttest_ind(w, s, equal_var=False)
        pooled_std = np.sqrt((w.std()**2 + s.std()**2) / 2)
        d = (w.mean() - s.mean()) / pooled_std if pooled_std > 0 else 0
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {name:<20} {w_med:>10.0f} {s_med:>10.0f} {delta:>+10.0f} {p_val:>9.1e} {d:>+9.2f} {sig}")
    w_ping = wire['ping_avg'].dropna()
    s_ping = switch['ping_avg'].dropna()
    if len(w_ping) > 1 and len(s_ping) > 1:
        t_stat, p_val = scipy_stats.ttest_ind(w_ping, s_ping, equal_var=False)
        pooled_std = np.sqrt((w_ping.std()**2 + s_ping.std()**2) / 2)
        d = (w_ping.mean() - s_ping.mean()) / pooled_std if pooled_std > 0 else 0
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {'Ping (ms)':<20} {w_ping.median():>10.2f} {s_ping.median():>10.2f} {w_ping.median()-s_ping.median():>+10.2f} {p_val:>9.1e} {d:>+9.2f} {sig}")
    print(f"  {'-' * 75}")
    print("  Rates in Mbit/s | *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


def iperf3_retransmit_summary(df):
    """iperf3 retransmit comparison."""
    print_separator("IPERF3 TCP RETRANSMITS")
    wire = df[df['connection'] == 'Direct Wire']
    switch = df[df['connection'] == 'University Switch']
    for label, m in [('Wire', wire), ('Switch', switch)]:
        retrans = m['iperf3_retrans'].dropna()
        if len(retrans) == 0:
            continue
        print(f"\n  {label}: {len(retrans)} samples")
        print(f"    Median: {retrans.median():.0f}  Mean: {retrans.mean():.1f}  "
              f"Max: {retrans.max():.0f}  Zero-retransmit: {(retrans == 0).sum()}/{len(retrans)} "
              f"({(retrans == 0).mean()*100:.0f}%)")


def fileiotest_retransmit_summary(df):
    """Fileiotest TCP retransmit comparison by phase."""
    print_separator("FILEIOTEST TCP RETRANSMITS (per phase)")
    for label, m in [('Wire', df[df['connection'] == 'Direct Wire']),
                     ('Switch', df[df['connection'] == 'University Switch'])]:
        if m.empty:
            continue
        print(f"\n  {label}:")
        for phase, col in [('Cold cache', 'cold_cache_retrans'),
                           ('Hot cache', 'hot_cache_retrans'),
                           ('True write', 'true_write_retrans')]:
            vals = pd.to_numeric(m[col], errors='coerce').dropna()
            if len(vals) > 0:
                print(f"    {phase:<15} median={vals.median():.0f}  mean={vals.mean():.1f}  "
                      f"total={vals.sum():.0f}  zero={((vals==0).sum())}/{len(vals)}")


def weekday_vs_weekend(df):
    """Check for time-of-day or weekday effects."""
    print_separator("WEEKDAY vs WEEKEND (Switch only)")
    switch = df[df['connection'] == 'University Switch'].copy()
    if switch.empty or 'timestamp' not in switch.columns:
        return
    switch['dow'] = switch['timestamp'].dt.dayofweek
    switch['is_weekend'] = switch['dow'] >= 5
    for period, label in [(False, 'Weekday'), (True, 'Weekend')]:
        m = switch[switch['is_weekend'] == period]
        if m.empty:
            continue
        cold = m['cold_cache_rate_mbits'].dropna()
        print(f"  {label}: {len(m)} samples, cold cache median = {cold.median():.0f} Mbit/s")
    wd = switch[~switch['is_weekend']]['cold_cache_rate_mbits'].dropna()
    we = switch[switch['is_weekend']]['cold_cache_rate_mbits'].dropna()
    if len(wd) > 1 and len(we) > 1:
        t, p = scipy_stats.ttest_ind(wd, we, equal_var=False)
        print(f"  Difference: p={p:.4f} {'(significant)' if p < 0.05 else '(not significant)'}")


# ─── Plotting functions ───────────────────────────────────────────────

def plot_01_per_machine_bars(df, outdir):
    """Per-machine grouped bar chart: iperf3 vs cold/hot/write."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Sort: wire first, then by NIC speed desc, then alphabetical
    wire_machines = sorted(df[df['source'] == WIRE_SOURCE]['source'].unique())
    switch_df = df[df['source'] != WIRE_SOURCE].groupby('source')['nic_speed'].first().sort_values(ascending=False)
    machines = list(wire_machines) + list(switch_df.index)

    x = np.arange(len(machines))
    width = 0.18

    medians = {'iperf3': [], 'cold': [], 'hot': [], 'write': []}
    for src in machines:
        m = df[df['source'] == src]
        medians['iperf3'].append(m['iperf3_sender_mbits'].median())
        medians['cold'].append(m['cold_cache_rate_mbits'].median())
        medians['hot'].append(m['hot_cache_avg_mbits'].median())
        medians['write'].append(m['true_write_rate_mbits'].median())

    ax.bar(x - 1.5*width, medians['iperf3'], width, label='iperf3 (raw)',
           color=COLORS['iperf3'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x - 0.5*width, medians['cold'], width, label='Cold cache',
           color=COLORS['cold'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + 0.5*width, medians['hot'], width, label='Hot cache',
           color=COLORS['hot'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + 1.5*width, medians['write'], width, label='True write',
           color=COLORS['write'], alpha=0.85, edgecolor='white', linewidth=0.5)

    # NIC speed markers
    nic_speeds = [NIC_SPEEDS.get(m, 0) for m in machines]
    ax.scatter(x, nic_speeds, marker='_', s=400, color='black', zorder=5,
               linewidths=2, label='NIC link speed')

    # Highlight wire machine
    for i, m in enumerate(machines):
        if m == WIRE_SOURCE:
            ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color=COLORS['wire'])
            ax.text(i, max(medians['iperf3'][i], nic_speeds[i]) * 1.02,
                    'WIRE', ha='center', fontsize=9, fontweight='bold', color=COLORS['wire'])

    # Labels with NIC tier
    labels = []
    for m in machines:
        nic = NIC_SPEEDS.get(m, 0)
        tier = {100: '100M', 1000: '1G', 2500: '2.5G', 5000: '5G'}.get(nic, '?')
        labels.append(f"{m}\n({tier})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Throughput (Mbit/s)')
    ax.set_title('Per-Machine Throughput: iperf3 (theoretical) vs fileiotest (real-world)\nMedian values, all rates in Mbit/s')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(max(medians['iperf3']), max(nic_speeds)) * 1.15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    plt.tight_layout()
    path = os.path.join(outdir, '01_per_machine_iperf3_vs_fileiotest.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_02_overhead_waterfall(df, outdir):
    """Overhead waterfall chart for wire and switch."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (conn, conn_label) in zip(axes, [('Direct Wire', 'Direct Wire (5G)'),
                                               ('University Switch', 'University Switch (mixed NICs)')]):
        m = df[df['connection'] == conn]
        if m.empty:
            continue

        iperf_med = m['iperf3_sender_mbits'].median()
        cold_med = m['cold_cache_rate_mbits'].median()
        hot_med = m['hot_cache_avg_mbits'].median()
        write_med = m['true_write_rate_mbits'].median()

        categories = ['iperf3\n(raw)', 'Cold\ncache', 'Hot\ncache', 'True\nwrite']
        values = [iperf_med, cold_med, hot_med, write_med]
        colors_list = [COLORS['iperf3'], COLORS['cold'], COLORS['hot'], COLORS['write']]

        bars = ax.bar(categories, values, color=colors_list, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        for i, (cat, val) in enumerate(zip(categories, values)):
            if i == 0:
                ax.text(i, val + iperf_med * 0.02, f'{val:.0f}',
                        ha='center', fontsize=10, fontweight='bold')
            else:
                overhead = (1 - val / iperf_med) * 100
                ax.text(i, val + iperf_med * 0.02, f'{val:.0f}\n({overhead:.0f}% loss)',
                        ha='center', fontsize=9)

        ax.axhline(y=iperf_med, color=COLORS['iperf3'], linestyle='--', alpha=0.4, linewidth=1)
        ax.set_ylabel('Throughput (Mbit/s)')
        ax.set_title(f'{conn_label}\n({len(m)} samples)')
        ax.set_ylim(0, iperf_med * 1.2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    fig.suptitle('Protocol Overhead: iperf3 (theoretical max) vs fileiotest (real-world file transfer)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, '02_overhead_waterfall.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_03_nic_tier_boxplots(df, outdir):
    """NIC tier boxplots with iperf3 overlay — switch + wire."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ('iperf3 (raw bandwidth)', 'iperf3_sender_mbits', axes[0, 0]),
        ('Cold cache (disk -> network)', 'cold_cache_rate_mbits', axes[0, 1]),
        ('Hot cache (RAM -> network)', 'hot_cache_avg_mbits', axes[1, 0]),
        ('True write (network -> disk)', 'true_write_rate_mbits', axes[1, 1]),
    ]

    tier_colors = [COLORS['1G'], COLORS['2.5G'], COLORS['5G'], COLORS['5G_wire']]

    for title, col, ax in metrics:
        data_groups = []
        labels = []

        for tier in ['1G', '2.5G', '5G']:
            vals = df[(df['nic_tier'] == tier) & (df['connection'] == 'University Switch')][col].dropna()
            if len(vals) > 0:
                data_groups.append(vals.values)
                labels.append(tier)

        wire_vals = df[df['source'] == WIRE_SOURCE][col].dropna()
        if len(wire_vals) > 0:
            data_groups.append(wire_vals.values)
            labels.append('5G\n(wire)')

        if not data_groups:
            continue

        bp = ax.boxplot(data_groups, labels=labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        for i, patch in enumerate(bp['boxes']):
            c = tier_colors[i] if i < len(tier_colors) else tier_colors[-1]
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        tier_limits = {'1G': 1000, '2.5G': 2500, '5G': 5000, '5G\n(wire)': 5000}
        for i, lbl in enumerate(labels):
            limit = tier_limits.get(lbl, 0)
            ax.plot([i + 0.6, i + 1.4], [limit, limit], 'k--', alpha=0.3, linewidth=1)

        ax.set_ylabel('Mbit/s')
        ax.set_title(title)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

        for i, grp in enumerate(data_groups):
            ax.text(i + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                    f'n={len(grp)}', ha='center', fontsize=8, color='gray')

    fig.suptitle('Throughput by NIC Tier and Connection Type\n(dashed line = NIC link speed)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, '03_nic_tier_boxplots.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_04_wire_vs_switch_violin(df, outdir):
    """Wire vs switch violin comparison for all metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 7))

    metrics = [
        ('iperf3', 'iperf3_sender_mbits'),
        ('Cold cache', 'cold_cache_rate_mbits'),
        ('Hot cache', 'hot_cache_avg_mbits'),
        ('True write', 'true_write_rate_mbits'),
    ]

    for ax, (title, col) in zip(axes, metrics):
        wire_data = df[df['connection'] == 'Direct Wire'][col].dropna().values
        switch_data = df[df['connection'] == 'University Switch'][col].dropna().values

        if len(wire_data) == 0 or len(switch_data) == 0:
            continue

        parts = ax.violinplot([switch_data, wire_data], positions=[1, 2],
                              showmeans=True, showmedians=True, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS['switch'] if i == 0 else COLORS['wire'])
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('red')

        for pos, data, color in [(1, switch_data, COLORS['switch']),
                                  (2, wire_data, COLORS['wire'])]:
            med = np.median(data)
            ax.text(pos, med, f'  {med:.0f}', va='center', fontsize=9,
                    fontweight='bold', color=color)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Switch\n(9 machines)', 'Wire\n(thais_wire)'])
        ax.set_title(title)
        ax.set_ylabel('Mbit/s')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    fig.suptitle('Wire vs Switch: Throughput Distributions\n(red line = median, black line = mean)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, '04_wire_vs_switch_violin.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_05_timeseries(df, outdir):
    """Time series of iperf3 and cold cache throughput."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for idx, (ax, title, col) in enumerate(zip(
        axes,
        ['iperf3 Raw Bandwidth Over Time', 'fileiotest Cold Cache Throughput Over Time'],
        ['iperf3_sender_mbits', 'cold_cache_rate_mbits']
    )):
        for src in sorted(df['source'].unique()):
            m = df[df['source'] == src].sort_values('timestamp')
            vals = m[col].dropna()
            if len(vals) == 0:
                continue
            is_wire = src == WIRE_SOURCE
            color = COLORS['wire'] if is_wire else None
            alpha = 1.0 if is_wire else 0.5
            lw = 2 if is_wire else 1
            marker = 'o' if is_wire else '.'
            ms = 4 if is_wire else 2
            ax.plot(m.loc[vals.index, 'timestamp'], vals, marker=marker,
                    markersize=ms, label=src, alpha=alpha, linewidth=lw, color=color)

        ax.set_ylabel('Throughput (Mbit/s)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, ncol=1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    axes[1].set_xlabel('Time')
    fig.suptitle('Throughput Stability Over Collection Window',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, '05_timeseries.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_06_retransmits(df, outdir):
    """Retransmit comparison: iperf3 and fileiotest."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # iperf3 retransmits
    ax = axes[0]
    wire_ret = df[df['connection'] == 'Direct Wire']['iperf3_retrans'].dropna()
    switch_ret = df[df['connection'] == 'University Switch']['iperf3_retrans'].dropna()
    data = []
    labels = []
    if len(switch_ret) > 0:
        data.append(switch_ret.values)
        labels.append(f'Switch\n(n={len(switch_ret)})')
    if len(wire_ret) > 0:
        data.append(wire_ret.values)
        labels.append(f'Wire\n(n={len(wire_ret)})')
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        bp_colors = [COLORS['switch'], COLORS['wire']][:len(data)]
        for patch, c in zip(bp['boxes'], bp_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax.set_ylabel('TCP Retransmits per 10s test')
    ax.set_title('iperf3 TCP Retransmits')

    # fileiotest retransmits by phase
    ax = axes[1]
    phases = ['cold_cache_retrans', 'hot_cache_retrans', 'true_write_retrans']
    phase_labels = ['Cold', 'Hot', 'Write']

    x_pos = np.arange(len(phase_labels))
    width = 0.35

    wire_meds = []
    switch_meds = []
    for col in phases:
        w = pd.to_numeric(df[df['connection'] == 'Direct Wire'][col], errors='coerce').dropna()
        s = pd.to_numeric(df[df['connection'] == 'University Switch'][col], errors='coerce').dropna()
        wire_meds.append(w.median() if len(w) > 0 else 0)
        switch_meds.append(s.median() if len(s) > 0 else 0)

    ax.bar(x_pos - width/2, switch_meds, width, label='Switch', color=COLORS['switch'], alpha=0.8)
    ax.bar(x_pos + width/2, wire_meds, width, label='Wire', color=COLORS['wire'], alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel('Median TCP Retransmits per sample')
    ax.set_title('fileiotest TCP Retransmits by Phase')
    ax.legend()

    fig.suptitle('TCP Retransmit Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, '06_retransmits.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_07_ping_latency(df, outdir):
    """Ping latency comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Box plot by machine
    ax = axes[0]
    machines = sorted(df['source'].unique(),
                      key=lambda s: (s != WIRE_SOURCE, NIC_SPEEDS.get(s, 0)))
    data = []
    labels = []
    colors_list = []
    for src in machines:
        vals = df[df['source'] == src]['ping_avg'].dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(src)
            colors_list.append(COLORS['wire'] if src == WIRE_SOURCE else COLORS['switch'])

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, vert=True,
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='.', markersize=2))
        for patch, c in zip(bp['boxes'], colors_list):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax.set_ylabel('Ping RTT (ms)')
    ax.set_title('Ping Latency by Machine')
    ax.tick_params(axis='x', rotation=45)

    # CDF
    ax = axes[1]
    for conn, color, label in [('University Switch', COLORS['switch'], 'Switch'),
                                ('Direct Wire', COLORS['wire'], 'Wire')]:
        vals = df[df['connection'] == conn]['ping_avg'].dropna().sort_values()
        if len(vals) > 0:
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, color=color, linewidth=2, label=f'{label} (n={len(vals)})')
    ax.set_xlabel('Ping RTT (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Ping Latency CDF')
    ax.legend()

    fig.suptitle('Network Latency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, '07_ping_latency.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_08_cdf_all_phases(df, outdir):
    """CDF for all phases: wire vs switch."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('iperf3 Raw Bandwidth', 'iperf3_sender_mbits', axes[0, 0]),
        ('Cold Cache', 'cold_cache_rate_mbits', axes[0, 1]),
        ('Hot Cache (avg)', 'hot_cache_avg_mbits', axes[1, 0]),
        ('True Write', 'true_write_rate_mbits', axes[1, 1]),
    ]

    for title, col, ax in metrics:
        for conn, color, lbl in [('University Switch', COLORS['switch'], 'Switch'),
                                  ('Direct Wire', COLORS['wire'], 'Wire')]:
            vals = df[df['connection'] == conn][col].dropna().sort_values()
            if len(vals) > 0:
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, cdf, color=color, linewidth=2,
                        label=f'{lbl} (n={len(vals)})')

        ax.set_xlabel('Mbit/s')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    fig.suptitle('Cumulative Distribution: Wire vs Switch\nAll metrics in Mbit/s',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, '08_cdf_all_phases.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_plots(df, outdir):
    """Generate all plots."""
    print_separator("GENERATING PLOTS")
    os.makedirs(outdir, exist_ok=True)

    plot_01_per_machine_bars(df, outdir)
    plot_02_overhead_waterfall(df, outdir)
    plot_03_nic_tier_boxplots(df, outdir)
    plot_04_wire_vs_switch_violin(df, outdir)
    plot_05_timeseries(df, outdir)
    plot_06_retransmits(df, outdir)
    plot_07_ping_latency(df, outdir)
    plot_08_cdf_all_phases(df, outdir)

    print(f"\n  All plots saved to: {outdir}/")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze iperf3 + fileiotest data')
    parser.add_argument('csv', help='Path to all_results.csv')
    parser.add_argument('--outdir', default='analysis_iperf3',
                        help='Output directory for plots (default: analysis_iperf3)')
    args = parser.parse_args()

    print("=" * 72)
    print("  Network Performance Analysis -- iperf3 + fileiotest")
    print("=" * 72)

    print(f"\nLoading: {args.csv}")
    df = load_data(args.csv)
    print(f"  Total: {len(df)} samples from {df['source'].nunique()} machines")
    print(f"  Sources: {', '.join(sorted(df['source'].unique()))}")

    # Time sync
    print("\nApplying time-sync filter (thais_wire window)...")
    df = time_sync_filter(df)

    # iperf3 coverage
    iperf_valid = df['iperf3_sender_mbits'].notna().sum()
    print(f"\n  iperf3 coverage: {iperf_valid}/{len(df)} samples ({iperf_valid/len(df)*100:.0f}%)")

    # Text analysis
    summary_table(df)
    overhead_analysis(df)
    nic_tier_analysis(df)
    wire_vs_switch(df)
    iperf3_retransmit_summary(df)
    fileiotest_retransmit_summary(df)
    weekday_vs_weekend(df)

    # Plots
    generate_all_plots(df, args.outdir)

    print(f"\n{'=' * 70}")
    print("  Analysis complete.")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
