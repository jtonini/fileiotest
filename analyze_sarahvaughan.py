#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_sarahvaughan.py — Analysis of fileiotest results for NAS tests.

Companion to analyze_iperf3.py — uses the same visual style and conventions.

Generates:
  01. iperf3 throughput by machine (box plots, colored by NIC tier)
  02. Transfer rates by machine (cold/hot/write grouped bars)
  03. Campus vs intranet for dual-homed machines
  04. thais → sarahvaughan vs thais → cooper comparison
  05. Time series over 24hr test period
  06. Retransmit analysis
  07. sarahvaughan vs natkingcole per-machine comparison
  08. sarahvaughan vs natkingcole box plots
  09. Summary statistics table (CSV + console)

Usage:
    python3 analyze_sarahvaughan.py [results_dir] [--outdir DIR]

    results_dir defaults to ./all_results_sarahvaughan/
"""

__author__ = 'Joao Tonini'
__version__ = '0.2'

import sys
import os
import re
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

COLOR_COLD = '#4292c6'
COLOR_HOT = '#ef6548'
COLOR_WRITE = '#78c679'
COLOR_IPERF3 = '#41ab5d'
COLOR_1G = '#4292c6'
COLOR_2_5G = '#fe9929'
COLOR_5G = '#e31a1c'
COLOR_INTRANET = '#41ab5d'
COLOR_BASELINE = '#7570b3'
COLOR_SV = '#2171b5'
COLOR_NKC = '#d94801'

TIER_COLORS = {
    '5G': COLOR_5G, '2.5G': COLOR_2_5G, '1G': COLOR_1G,
    'intranet': COLOR_INTRANET, 'baseline': COLOR_BASELINE,
}
TIER_ORDER = ['5G', '2.5G', '1G', 'intranet', 'baseline']

NIC_SPEEDS = {
    'thais': 5000, 'cooper': 5000, 'hamilton': 5000,
    'camryn': 2500, 'irene2': 2500,
    'aamy': 1000, 'alexis': 1000, 'boyi': 1000,
    'josh': 1000, 'justin': 1000, 'khanh': 1000,
    'mayer': 1000,
}

SV_CAMPUS_SPEED = 1000
SV_INTRANET_SPEED = 1000
NKC_CAMPUS_SPEED = 1000


def normalize_rate(rate_str) -> float:
    if not isinstance(rate_str, str) or not rate_str.strip():
        return np.nan
    m = re.match(r'([\d.]+)\s*([A-Za-z/]+)', rate_str.strip())
    if not m:
        return np.nan
    value = float(m.group(1))
    unit = m.group(2)
    if 'iB/s' in unit or 'B/s' in unit:
        if unit[0] in 'Gg': return value * 8.0 * 1024.0
        elif unit[0] in 'Mm': return value * 8.0
        elif unit[0] in 'Kk': return value * 8.0 / 1024.0
        else: return value * 8.0 / (1024.0 * 1024.0)
    if 'ib/s' in unit:
        if unit[0] in 'Gg': return value * 1000.0
        elif unit[0] in 'Mm': return value
        elif unit[0] in 'Kk': return value / 1000.0
        else: return value / 1_000_000.0
    return np.nan


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


def tier_from_label(label):
    if 'intranet' in label: return 'intranet'
    if 'to_cooper' in label: return 'baseline'
    machine = machine_from_label(label)
    speed = NIC_SPEEDS.get(machine, 1000)
    if speed >= 5000: return '5G'
    elif speed >= 2500: return '2.5G'
    return '1G'


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
            df[f'{col}_mbps'] = df[col].apply(normalize_rate)
    for col in ['iperf3_sender_bps', 'iperf3_receiver_bps']:
        if col in df.columns:
            df[f'{col}_mbps'] = df[col].apply(bps_to_mbps)
    hot_mibs = [c for c in df.columns if 'hot_cache_run' in c and c.endswith('_mibs')]
    if hot_mibs:
        df['hot_cache_avg_mibs'] = df[hot_mibs].mean(axis=1)
    hot_mbps = [c for c in df.columns if 'hot_cache_run' in c and c.endswith('_mbps')]
    if hot_mbps:
        df['hot_cache_avg_mbps'] = df[hot_mbps].mean(axis=1)
    for col in ['iperf3_retransmits', 'cold_cache_retrans', 'hot_cache_retrans',
                'true_write_retrans']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['machine'] = df['source'].apply(machine_from_label)
    df['tier'] = df['source'].apply(tier_from_label)
    df['path'] = df['source'].apply(path_from_label)
    df['dest'] = df['source'].apply(dest_from_label)
    df['nic_speed'] = df['machine'].map(NIC_SPEEDS).fillna(1000).astype(int)
    return df


# ─── Plot Functions ───────────────────────────────────────────────────

def plot_01_iperf3_by_machine(df, outdir):
    fig, ax = plt.subplots(figsize=(16, 7))
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    sv = df[df['dest'] == 'sarahvaughan']
    groups = sv.groupby(['machine', 'tier', 'path']).size().reset_index(name='n')
    groups['tier_rank'] = groups['tier'].map(tier_rank)
    groups = groups.sort_values(['tier_rank', 'machine', 'path'])
    data, labels, colors = [], [], []
    for _, row in groups.iterrows():
        m, t, p = row['machine'], row['tier'], row['path']
        vals = sv[(sv['machine'] == m) & (sv['path'] == p)]['iperf3_sender_bps_mbps'].dropna()
        if len(vals) == 0: continue
        data.append(vals.values)
        suffix = ' (inet)' if p == 'intranet' else ''
        labels.append(f'{m}{suffix}\n({NIC_SPEEDS.get(m, "?")}M)')
        colors.append(TIER_COLORS.get(t, '#95a5a6'))
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(y=SV_CAMPUS_SPEED, color='red', linestyle='--', alpha=0.5,
               label=f'SV campus NIC ({SV_CAMPUS_SPEED} Mbps)')
    legend_elements = [Patch(facecolor=TIER_COLORS[t], alpha=0.7, label=t)
                       for t in TIER_ORDER if t in sv['tier'].unique()]
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', alpha=0.5,
                                  label=f'SV campus NIC ({SV_CAMPUS_SPEED}M)'))
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_title('iperf3 Throughput by Machine -> sarahvaughan (simultaneous)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(f'{outdir}/01_iperf3_by_machine.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 01_iperf3_by_machine.png')


def plot_02_transfer_rates(df, outdir):
    fig, ax = plt.subplots(figsize=(16, 7))
    sv = df[df['dest'] == 'sarahvaughan']
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    agg = (sv.groupby(['machine', 'tier', 'path'])
           .agg(cold=('cold_cache_rate_mibs', 'median'),
                hot=('hot_cache_avg_mibs', 'median'),
                write=('true_write_rate_mibs', 'median'))
           .reset_index())
    agg['tier_rank'] = agg['tier'].map(tier_rank)
    agg = agg.sort_values(['tier_rank', 'machine', 'path']).reset_index(drop=True)
    x = np.arange(len(agg))
    w = 0.25
    ax.bar(x - w, agg['cold'], w, label='Cold Cache Read', color=COLOR_COLD, alpha=0.8)
    ax.bar(x, agg['hot'], w, label='Hot Cache Read (avg)', color=COLOR_HOT, alpha=0.8)
    ax.bar(x + w, agg['write'], w, label='True Write', color=COLOR_WRITE, alpha=0.8)
    labels = [f"{r['machine']}{' (inet)' if r['path'] == 'intranet' else ''}" for _, r in agg.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    max_1g = 1000.0 / 8.0
    ax.axhline(y=max_1g, color='red', linestyle='--', alpha=0.5,
               label=f'1G theoretical max ({max_1g:.0f} MiB/s)')
    ax.legend(loc='upper right')
    ax.set_ylabel('Transfer Rate (MiB/s)')
    ax.set_title('Median Transfer Rates by Machine -> sarahvaughan',
                 fontsize=13, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(f'{outdir}/02_transfer_rates_by_machine.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 02_transfer_rates_by_machine.png')


def plot_03_campus_vs_intranet(df, outdir):
    sv = df[df['dest'] == 'sarahvaughan']
    dual_machines = ['josh', 'justin', 'mayer']
    dual = sv[sv['machine'].isin(dual_machines)]
    if dual.empty:
        print('  ⚠ No dual-homed data'); return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [('iperf3_sender_bps_mbps', 'iperf3 (Mbps)'),
               ('cold_cache_rate_mibs', 'Cold Cache Read (MiB/s)'),
               ('true_write_rate_mibs', 'True Write (MiB/s)')]
    for (col, ylabel), ax in zip(metrics, axes):
        campus_vals = [dual[(dual['machine'] == m) & (dual['path'] == 'campus')][col].median() or 0 for m in dual_machines]
        intranet_vals = [dual[(dual['machine'] == m) & (dual['path'] == 'intranet')][col].median() or 0 for m in dual_machines]
        campus_vals = [v if pd.notna(v) else 0 for v in campus_vals]
        intranet_vals = [v if pd.notna(v) else 0 for v in intranet_vals]
        x = np.arange(len(dual_machines)); w = 0.35
        ax.bar(x - w/2, campus_vals, w, label='Campus', color=COLOR_1G, alpha=0.8)
        ax.bar(x + w/2, intranet_vals, w, label='Intranet', color=COLOR_INTRANET, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(dual_machines)
        ax.set_ylabel(ylabel); ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend()
    fig.suptitle('Campus vs Intranet: Dual-Homed Machines -> sarahvaughan',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/03_campus_vs_intranet.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 03_campus_vs_intranet.png')


def plot_04_thais_comparison(df, df_cooper, outdir):
    thais_sv = df[(df['machine'] == 'thais') & (df['dest'] == 'sarahvaughan')]
    if thais_sv.empty or df_cooper.empty:
        print('  ⚠ Missing thais data'); return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    comparisons = [('iperf3_sender_bps_mbps', 'iperf3 Throughput (Mbps)'),
                   ('cold_cache_rate_mibs', 'Cold Cache Read (MiB/s)'),
                   ('hot_cache_avg_mibs', 'Hot Cache Read (MiB/s)')]
    for (col, ylabel), ax in zip(comparisons, axes):
        sv_data = thais_sv[col].dropna()
        cooper_data = df_cooper[col].dropna()
        if sv_data.empty or cooper_data.empty:
            ax.set_title(ylabel); continue
        bp = ax.boxplot([sv_data.values, cooper_data.values],
                        labels=['-> sarahvaughan\n(5G -> 1G dest)', '-> cooper\n(5G <-> 5G)'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(COLOR_SV); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(COLOR_BASELINE); bp['boxes'][1].set_alpha(0.7)
        d = cohens_d(cooper_data, sv_data)
        ax.text(0.5, 0.02, f"Cohen's d = {d:.2f} ({effect_label(d)})",
                ha='center', transform=ax.transAxes, fontsize=9, style='italic', color='#555555')
        ax.set_ylabel(ylabel); ax.set_title(ylabel, fontsize=12, fontweight='bold')
    fig.suptitle('thais: sarahvaughan (SV campus 1G) vs cooper (5G<->5G campus switch)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/04_thais_sv_vs_cooper.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 04_thais_sv_vs_cooper.png')


def plot_05_timeseries(df, outdir):
    sv = df[df['dest'] == 'sarahvaughan']
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax = axes[0]
    for tier in TIER_ORDER:
        subset = sv[sv['tier'] == tier]
        if subset.empty: continue
        color = TIER_COLORS[tier]
        for machine in subset['machine'].unique():
            md = subset[subset['machine'] == machine].sort_values('timestamp')
            ax.plot(md['timestamp'], md['iperf3_sender_bps_mbps'],
                    color=color, alpha=0.2, linewidth=0.7)
        tier_data = subset.sort_values('timestamp').set_index('timestamp')
        tier_med = tier_data['iperf3_sender_bps_mbps'].resample('30min').median()
        ax.plot(tier_med.index, tier_med.values, color=color, linewidth=2.5, label=tier)
    ax.legend(loc='upper right')
    ax.set_ylabel('iperf3 Throughput (Mbps)')
    ax.set_title('iperf3 Throughput Over Time -> sarahvaughan (simultaneous)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    ax = axes[1]
    for tier in ['1G', 'intranet']:
        subset = sv[sv['tier'] == tier]
        if subset.empty: continue
        color = TIER_COLORS[tier]
        tier_data = subset.sort_values('timestamp').set_index('timestamp')
        cold_med = tier_data['cold_cache_rate_mibs'].resample('30min').median()
        hot_med = tier_data['hot_cache_avg_mibs'].resample('30min').median()
        ax.plot(cold_med.index, cold_med.values, color=color, linewidth=2, linestyle='-',
                label=f'{tier} cold cache')
        ax.plot(hot_med.index, hot_med.values, color=color, linewidth=2, linestyle='--',
                alpha=0.7, label=f'{tier} hot cache')
    ax.legend(loc='upper right')
    ax.set_ylabel('Transfer Rate (MiB/s)')
    ax.set_title('Transfer Rates Over Time (1G campus & intranet)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    fig.tight_layout()
    fig.savefig(f'{outdir}/05_timeseries.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 05_timeseries.png')


def plot_06_retransmits(df, outdir):
    sv = df[df['dest'] == 'sarahvaughan']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    ax = axes[0]
    agg = (sv.groupby(['machine', 'tier', 'path'])
           .agg(med=('iperf3_retransmits', 'median'), mx=('iperf3_retransmits', 'max'))
           .reset_index())
    agg['tier_rank'] = agg['tier'].map(tier_rank)
    agg = agg.sort_values(['tier_rank', 'machine', 'path']).reset_index(drop=True)
    x = np.arange(len(agg))
    colors = [TIER_COLORS.get(t, '#95a5a6') for t in agg['tier']]
    ax.bar(x, agg['med'], color=colors, alpha=0.7)
    ax.errorbar(x, agg['med'],
                yerr=[np.zeros(len(agg)), (agg['mx'] - agg['med']).clip(lower=0)],
                fmt='none', color='black', capsize=3, alpha=0.5)
    labels = [f"{r['machine']}{' (i)' if r['path'] == 'intranet' else ''}" for _, r in agg.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Retransmits')
    ax.set_title('iperf3 Retransmits (median + max)', fontsize=12, fontweight='bold')
    ax = axes[1]
    agg2 = (sv.groupby(['machine', 'tier', 'path'])
            .agg(med=('cold_cache_retrans', 'median'), mx=('cold_cache_retrans', 'max'))
            .reset_index())
    agg2['tier_rank'] = agg2['tier'].map(tier_rank)
    agg2 = agg2.sort_values(['tier_rank', 'machine', 'path']).reset_index(drop=True)
    x = np.arange(len(agg2))
    colors = [TIER_COLORS.get(t, '#95a5a6') for t in agg2['tier']]
    ax.bar(x, agg2['med'], color=colors, alpha=0.7)
    labels = [f"{r['machine']}{' (i)' if r['path'] == 'intranet' else ''}" for _, r in agg2.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Retransmits')
    ax.set_title('Cold Cache Transfer Retransmits (median)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/06_retransmits.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 06_retransmits.png')


def plot_07_sv_vs_nkc_bars(df, outdir):
    sv = df[(df['dest'] == 'sarahvaughan') & (df['path'] == 'campus')]
    nkc = df[df['dest'] == 'natkingcole']
    if sv.empty or nkc.empty:
        print('  ⚠ No SV vs NKC comparison data'); return
    machines = sorted(set(sv['machine'].unique()) & set(nkc['machine'].unique()))
    if not machines:
        print('  ⚠ No common machines for SV vs NKC'); return
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    metrics = [('iperf3_sender_bps_mbps', 'iperf3 (Mbps)'),
               ('cold_cache_rate_mibs', 'Cold Cache Read (MiB/s)'),
               ('true_write_rate_mibs', 'True Write (MiB/s)')]
    for (col, ylabel), ax in zip(metrics, axes):
        sv_vals = [sv[sv['machine'] == m][col].median() for m in machines]
        nkc_vals = [nkc[nkc['machine'] == m][col].median() for m in machines]
        sv_vals = [v if pd.notna(v) else 0 for v in sv_vals]
        nkc_vals = [v if pd.notna(v) else 0 for v in nkc_vals]
        x = np.arange(len(machines)); w = 0.35
        ax.bar(x - w/2, sv_vals, w, label='sarahvaughan', color=COLOR_SV, alpha=0.8)
        ax.bar(x + w/2, nkc_vals, w, label='natkingcole', color=COLOR_NKC, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(machines, rotation=45, ha='right')
        ax.set_ylabel(ylabel); ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend()
    fig.suptitle('sarahvaughan vs natkingcole: Per-Machine Comparison (campus, simultaneous)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/07_sv_vs_nkc_bars.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 07_sv_vs_nkc_bars.png')


def plot_08_sv_vs_nkc_boxplots(df, outdir):
    sv = df[(df['dest'] == 'sarahvaughan') & (df['path'] == 'campus')]
    nkc = df[df['dest'] == 'natkingcole']
    if sv.empty or nkc.empty:
        print('  ⚠ No SV vs NKC data for boxplots'); return
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    metrics = [('iperf3_sender_bps_mbps', 'iperf3 (Mbps)'),
               ('cold_cache_rate_mibs', 'Cold Cache (MiB/s)'),
               ('hot_cache_avg_mibs', 'Hot Cache (MiB/s)'),
               ('true_write_rate_mibs', 'True Write (MiB/s)')]
    for (col, ylabel), ax in zip(metrics, axes):
        sv_data = sv[col].dropna()
        nkc_data = nkc[col].dropna()
        if sv_data.empty and nkc_data.empty:
            ax.set_title(ylabel); continue
        plot_data, plot_labels = [], []
        if not sv_data.empty: plot_data.append(sv_data.values); plot_labels.append('SV')
        if not nkc_data.empty: plot_data.append(nkc_data.values); plot_labels.append('NKC')
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.5)
        box_colors = [COLOR_SV, COLOR_NKC][:len(plot_data)]
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        if not sv_data.empty and not nkc_data.empty:
            d = cohens_d(sv_data, nkc_data)
            ax.text(0.5, 0.02, f"d = {d:.2f} ({effect_label(d)})",
                    ha='center', transform=ax.transAxes, fontsize=9,
                    style='italic', color='#555555')
        ax.set_ylabel(ylabel); ax.set_title(ylabel, fontsize=12, fontweight='bold')
    fig.suptitle('sarahvaughan vs natkingcole: Aggregate Performance (campus)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{outdir}/08_sv_vs_nkc_boxplots.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✓ 08_sv_vs_nkc_boxplots.png')


def print_summary(df_all, df_cooper, outdir):
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    rows = []
    for (machine, tier, path, dest), g in df_all.groupby(['machine', 'tier', 'path', 'dest']):
        nic = NIC_SPEEDS.get(machine, '?')
        dest_short = 'SV' if dest == 'sarahvaughan' else 'NKC'
        rows.append({
            'Machine': machine, 'Dest': dest_short, 'Path': path,
            'NIC': f'{nic}M', 'Tier': tier,
            'iperf3 Med': f"{g['iperf3_sender_bps_mbps'].median():.0f}"
                          if g['iperf3_sender_bps_mbps'].notna().any() else 'N/A',
            'Cold': f"{g['cold_cache_rate_mibs'].median():.1f}",
            'Hot': f"{g['hot_cache_avg_mibs'].median():.1f}",
            'Write': f"{g['true_write_rate_mibs'].median():.1f}",
            'Retrans': f"{g['iperf3_retransmits'].median():.0f}"
                       if g['iperf3_retransmits'].notna().any() else 'N/A',
            'N': len(g),
            '_tier_rank': tier_rank.get(tier, 99),
            '_dest_rank': 0 if dest == 'sarahvaughan' else 1,
        })
    if not df_cooper.empty:
        g = df_cooper
        rows.append({
            'Machine': 'thais->cooper', 'Dest': 'cooper', 'Path': 'campus',
            'NIC': '5G<->5G', 'Tier': 'baseline',
            'iperf3 Med': f"{g['iperf3_sender_bps_mbps'].median():.0f}",
            'Cold': f"{g['cold_cache_rate_mibs'].median():.1f}",
            'Hot': f"{g['hot_cache_avg_mibs'].median():.1f}",
            'Write': f"{g['true_write_rate_mibs'].median():.1f}",
            'Retrans': f"{g['iperf3_retransmits'].median():.0f}",
            'N': len(g), '_tier_rank': 99, '_dest_rank': 2,
        })
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(['_dest_rank', '_tier_rank', 'Machine', 'Path'])
    summary = summary.drop(columns=['_tier_rank', '_dest_rank'])
    csv_path = f'{outdir}/summary_stats.csv'
    summary.to_csv(csv_path, index=False)
    print('\n  Summary Statistics:')
    print('  ' + '-' * 120)
    print(summary.to_string(index=False))
    print('  ' + '-' * 120)
    print(f'\n  ✓ {csv_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results_dir', nargs='?', default='./all_results_sarahvaughan')
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()
    results_dir = args.results_dir
    outdir = args.outdir or os.path.join(results_dir, 'analysis')
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
            print(f'  {label:12s}: {len(d)} samples, {d["machine"].nunique()} machines')
        else:
            print(f'  ⚠ No {label} data at {path}')

    df_cooper = pd.DataFrame()
    if os.path.isfile(cooper_csv):
        df_cooper = load_csv(cooper_csv)
        print(f'  {"Cooper":12s}: {len(df_cooper)} samples (thais->cooper baseline)')

    if not dfs:
        print('ERROR: No data found.'); sys.exit(1)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f'  {"Total":12s}: {len(df_all)} samples')
    print(f'\nGenerating plots -> {outdir}/')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plot_01_iperf3_by_machine(df_all, outdir)
        plot_02_transfer_rates(df_all, outdir)
        plot_03_campus_vs_intranet(df_all, outdir)
        if not df_cooper.empty:
            plot_04_thais_comparison(df_all, df_cooper, outdir)
        plot_05_timeseries(df_all, outdir)
        plot_06_retransmits(df_all, outdir)
        has_nkc = df_all['dest'].eq('natkingcole').any()
        if has_nkc:
            plot_07_sv_vs_nkc_bars(df_all, outdir)
            plot_08_sv_vs_nkc_boxplots(df_all, outdir)
        print_summary(df_all, df_cooper, outdir)

    print(f'\nDone! All plots saved to {outdir}/')


if __name__ == '__main__':
    main()
