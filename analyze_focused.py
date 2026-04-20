#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_focused.py — Key findings from Parish Lab NAS performance tests.

Three focused plots:
  1. thais→cooper (5G↔5G) vs thais→sarahvaughan (5G→1G NAS) — NAS bottleneck
  2. josh campus vs intranet — contention and latency difference
  3. Ping latency: campus switch vs private intranet

Usage:
    python3 analyze_focused.py [results_dir]
"""

__author__ = 'Joao Tonini'
__version__ = '1.0'

import sys
import os
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

# Colors — consistent with analyze_iperf3.py
COLOR_SV = '#2171b5'
COLOR_COOPER = '#7570b3'
COLOR_CAMPUS = '#4292c6'
COLOR_INTRANET = '#41ab5d'
COLOR_NKC = '#d94801'
COLOR_COLD = '#4292c6'
COLOR_HOT = '#ef6548'
COLOR_WRITE = '#78c679'


def rate_to_mibs(rate_str) -> float:
    if not isinstance(rate_str, str) or not rate_str.strip():
        return np.nan
    m = re.match(r'([\d.]+)\s*([A-Za-z/]+)', rate_str.strip())
    if not m:
        return np.nan
    value = float(m.group(1))
    unit = m.group(2)
    if 'iB/s' in unit or 'B/s' in unit:
        if unit[0] in 'Gg': return value * 1024.0
        elif unit[0] in 'Mm': return value
        elif unit[0] in 'Kk': return value / 1024.0
        else: return value / (1024.0 * 1024.0)
    return np.nan


def bps_to_mbps(bps) -> float:
    try: return float(bps) / 1_000_000
    except (ValueError, TypeError): return np.nan


def machine_from_label(label):
    return (label
            .replace('_campus_nkc_simul', '')
            .replace('_campus_simul', '')
            .replace('_intranet_simul', '')
            .replace('_campus_nkc', '')
            .replace('_campus', '')
            .replace('_intranet', '')
            .replace('_to_cooper', ''))


def dest_from_label(label):
    if 'nkc' in label: return 'natkingcole'
    if 'to_cooper' in label: return 'cooper'
    return 'sarahvaughan'


def path_from_label(label):
    return 'intranet' if 'intranet' in label else 'campus'


def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0


def effect_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    return "large"


def load_csv(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    rate_cols = ['cold_cache_rate', 'hot_cache_run1_rate', 'hot_cache_run2_rate',
                 'hot_cache_run3_rate', 'true_write_rate']
    for col in rate_cols:
        if col in df.columns:
            df[f'{col}_mibs'] = df[col].apply(rate_to_mibs)
    for col in ['iperf3_sender_bps', 'iperf3_receiver_bps']:
        if col in df.columns:
            df[f'{col}_mbps'] = df[col].apply(bps_to_mbps)
    hot_mibs = [c for c in df.columns if 'hot_cache_run' in c and c.endswith('_mibs')]
    if hot_mibs:
        df['hot_cache_avg_mibs'] = df[hot_mibs].mean(axis=1)
    for col in ['ping_avg_ms', 'ping_min_ms', 'ping_max_ms', 'ping_mdev_ms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['machine'] = df['source'].apply(machine_from_label)
    df['dest'] = df['source'].apply(dest_from_label)
    df['path'] = df['source'].apply(path_from_label)
    return df


# ─── Plot 1: thais→cooper vs thais→SV ────────────────────────────────

def plot_nas_bottleneck(df_sv, df_cooper, outdir):
    """Show the NAS 1G cap: thais→cooper (5G↔5G) vs thais→SV (5G→1G NAS)."""
    thais_sv = df_sv[(df_sv['machine'] == 'thais') & (df_sv['dest'] == 'sarahvaughan')]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    metrics = [
        ('iperf3_sender_bps_mbps', 'iperf3 Throughput\n(Mbps)'),
        ('cold_cache_rate_mibs', 'Cold Cache Read\n(MiB/s)'),
        ('hot_cache_avg_mibs', 'Hot Cache Read\n(MiB/s)'),
        ('true_write_rate_mibs', 'True Write\n(MiB/s)'),
    ]

    for (col, ylabel), ax in zip(metrics, axes):
        sv_data = thais_sv[col].dropna()
        cooper_data = df_cooper[col].dropna()
        if sv_data.empty or cooper_data.empty:
            ax.set_title(ylabel)
            continue

        bp = ax.boxplot([sv_data.values, cooper_data.values],
                        labels=['thais -> SV\n(5G src, 1G NAS)', 'thais -> cooper\n(5G <-> 5G)'],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=1.5))
        bp['boxes'][0].set_facecolor(COLOR_SV); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(COLOR_COOPER); bp['boxes'][1].set_alpha(0.7)

        # Medians annotation
        sv_med = sv_data.median()
        cooper_med = cooper_data.median()
        ax.text(1, sv_med, f' {sv_med:.0f}', va='center', fontsize=9, color=COLOR_SV, fontweight='bold')
        ax.text(2, cooper_med, f' {cooper_med:.0f}', va='center', fontsize=9, color=COLOR_COOPER, fontweight='bold')

        d = cohens_d(cooper_data, sv_data)
        ax.text(0.5, 0.02, f"Cohen's d = {d:.1f} ({effect_label(d)})",
                ha='center', transform=ax.transAxes, fontsize=9,
                style='italic', color='#555555')

        ax.set_ylabel(ylabel, fontsize=11)

    fig.suptitle('NAS Hardware Bottleneck: Intel X553 NIC caps at 1 Gbps\n'
                 'thais (5G NIC) -> sarahvaughan (1G NAS) vs thais -> cooper (5G workstation)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/key_01_nas_bottleneck.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ key_01_nas_bottleneck.png')


# ─── Plot 2: Campus vs Intranet (josh) ───────────────────────────────

def plot_campus_vs_intranet(df_sv, outdir):
    """josh campus vs intranet — same machine, same NICs, different paths."""
    josh_campus = df_sv[(df_sv['machine'] == 'josh') & (df_sv['path'] == 'campus') &
                        (df_sv['dest'] == 'sarahvaughan')]
    josh_intranet = df_sv[(df_sv['machine'] == 'josh') & (df_sv['path'] == 'intranet') &
                          (df_sv['dest'] == 'sarahvaughan')]

    if josh_campus.empty or josh_intranet.empty:
        print('  ⚠ Missing josh campus/intranet data')
        return

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    metrics = [
        ('cold_cache_rate_mibs', 'Cold Cache Read\n(MiB/s)'),
        ('hot_cache_avg_mibs', 'Hot Cache Read\n(MiB/s)'),
        ('true_write_rate_mibs', 'True Write\n(MiB/s)'),
        ('ping_avg_ms', 'Ping Latency\n(ms)'),
    ]

    for (col, ylabel), ax in zip(metrics, axes):
        campus_data = josh_campus[col].dropna()
        intranet_data = josh_intranet[col].dropna()
        if campus_data.empty or intranet_data.empty:
            ax.set_title(ylabel)
            continue

        bp = ax.boxplot([campus_data.values, intranet_data.values],
                        labels=['josh\ncampus switch', 'josh\nprivate switch'],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=1.5))
        bp['boxes'][0].set_facecolor(COLOR_CAMPUS); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(COLOR_INTRANET); bp['boxes'][1].set_alpha(0.7)

        campus_med = campus_data.median()
        intranet_med = intranet_data.median()
        ax.text(1, campus_med, f' {campus_med:.1f}', va='center', fontsize=9,
                color=COLOR_CAMPUS, fontweight='bold')
        ax.text(2, intranet_med, f' {intranet_med:.1f}', va='center', fontsize=9,
                color=COLOR_INTRANET, fontweight='bold')

        d = cohens_d(intranet_data, campus_data)
        ax.text(0.5, 0.02, f"d = {d:.2f} ({effect_label(d)})",
                ha='center', transform=ax.transAxes, fontsize=9,
                style='italic', color='#555555')

        ax.set_ylabel(ylabel, fontsize=11)

    fig.suptitle('Campus Switch vs Private Intranet: josh -> sarahvaughan\n'
                 'Same machine, same 1G NICs, different network paths (simultaneous test)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/key_02_campus_vs_intranet.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ key_02_campus_vs_intranet.png')


# ─── Plot 3: Latency comparison all machines ─────────────────────────

def plot_latency(df_all, outdir):
    """Ping latency: campus vs intranet, all machines."""
    sv = df_all[df_all['dest'] == 'sarahvaughan']
    nkc = df_all[df_all['dest'] == 'natkingcole']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: Box plot by path (campus vs intranet)
    ax = axes[0]
    campus = sv[sv['path'] == 'campus']['ping_avg_ms'].dropna()
    intranet = sv[sv['path'] == 'intranet']['ping_avg_ms'].dropna()

    plot_data = [campus.values, intranet.values]
    plot_labels = [f'Campus switch\n(n={len(campus)})',
                   f'Private switch\n(n={len(intranet)})']
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))
    bp['boxes'][0].set_facecolor(COLOR_CAMPUS); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLOR_INTRANET); bp['boxes'][1].set_alpha(0.7)

    campus_med = campus.median()
    intranet_med = intranet.median()
    ax.text(1, campus_med, f'  {campus_med:.3f} ms', va='center', fontsize=10,
            color=COLOR_CAMPUS, fontweight='bold')
    ax.text(2, intranet_med, f'  {intranet_med:.3f} ms', va='center', fontsize=10,
            color=COLOR_INTRANET, fontweight='bold')

    d = cohens_d(campus, intranet)
    ax.text(0.5, 0.02, f"Cohen's d = {d:.2f} ({effect_label(d)})",
            ha='center', transform=ax.transAxes, fontsize=9,
            style='italic', color='#555555')

    ax.set_ylabel('Ping avg latency (ms)', fontsize=11)
    ax.set_title('Latency: Campus vs Private Switch\n(all machines -> sarahvaughan)',
                 fontsize=12, fontweight='bold')

    # Panel 2: Per-machine latency comparison (SV vs NKC campus)
    ax = axes[1]
    machines = sorted(set(sv[sv['path'] == 'campus']['machine'].unique()) &
                      set(nkc['machine'].unique()))

    sv_lat = [sv[(sv['machine'] == m) & (sv['path'] == 'campus')]['ping_avg_ms'].median()
              for m in machines]
    nkc_lat = [nkc[nkc['machine'] == m]['ping_avg_ms'].median()
               for m in machines]
    sv_lat = [v if pd.notna(v) else 0 for v in sv_lat]
    nkc_lat = [v if pd.notna(v) else 0 for v in nkc_lat]

    x = np.arange(len(machines))
    w = 0.35
    ax.bar(x - w/2, sv_lat, w, label='sarahvaughan', color=COLOR_SV, alpha=0.8)
    ax.bar(x + w/2, nkc_lat, w, label='natkingcole', color=COLOR_NKC, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(machines, rotation=45, ha='right')
    ax.set_ylabel('Ping avg latency (ms)', fontsize=11)
    ax.set_title('Per-Machine Latency: SV vs NKC\n(campus switch)',
                 fontsize=12, fontweight='bold')
    ax.legend()

    fig.suptitle('Network Latency Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/key_03_latency.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ key_03_latency.png')


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else './all_results_sarahvaughan'
    outdir = os.path.join(results_dir, 'analysis')
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading data from {results_dir}...')

    campus_csv = os.path.join(results_dir, 'all_campus_simul.csv')
    intranet_csv = os.path.join(results_dir, 'all_intranet_simul.csv')
    nkc_csv = os.path.join(results_dir, 'all_campus_nkc_simul.csv')
    cooper_csv = os.path.join(results_dir, 'thais_cooper', 'results_thais_to_cooper.csv')

    dfs = []
    for label, path in [('SV Campus', campus_csv), ('SV Intranet', intranet_csv),
                         ('NKC Campus', nkc_csv)]:
        if os.path.isfile(path):
            d = load_csv(path)
            dfs.append(d)
            print(f'  {label:12s}: {len(d)} samples')
        else:
            print(f'  ⚠ No {label} data')

    df_cooper = pd.DataFrame()
    if os.path.isfile(cooper_csv):
        df_cooper = load_csv(cooper_csv)
        print(f'  {"Cooper":12s}: {len(df_cooper)} samples')

    if not dfs:
        print('ERROR: No data.'); sys.exit(1)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f'  {"Total":12s}: {len(df_all)} samples\n')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if not df_cooper.empty:
            plot_nas_bottleneck(df_all, df_cooper, outdir)
        plot_campus_vs_intranet(df_all, outdir)
        plot_latency(df_all, outdir)

    print(f'\nDone! Key plots saved to {outdir}/')


if __name__ == '__main__':
    main()
