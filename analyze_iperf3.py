#!/usr/bin/env python3
"""analyze_iperf3.py — Preliminary analysis of iperf3 + fileiotest data.

Includes:
  - Time-sync filtering to thais_wire window
  - iperf3 summary per machine (theoretical max)
  - Overhead calculation: iperf3 vs fileiotest phases
  - NIC tier breakdown with iperf3
  - Wire vs switch comparison

Usage:
    python3 analyze_iperf3.py all_results.csv
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

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
    print(f"  Time window: {t_min} → {t_max}")
    print(f"  Samples: {len(df)} → {len(synced)} (time-synced)")
    return synced


def print_separator(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


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
        ping_med = pd.to_numeric(m['ping_avg_ms'], errors='coerce').median()

        rows.append({
            'Machine': src,
            'Conn': conn,
            'NIC': f"{nic}M",
            'N': n,
            'iperf3': f"{iperf_med:.0f}",
            'Cold': f"{cold_med:.0f}",
            'Hot': f"{hot_med:.0f}",
            'Write': f"{true_med:.0f}",
            'Ping_ms': f"{ping_med:.2f}" if not pd.isna(ping_med) else '-',
        })

    tbl = pd.DataFrame(rows)
    print("\n  All rates in Mbit/s (median values)")
    print(f"  {'─' * 90}")
    print(f"  {'Machine':<14} {'Conn':<7} {'NIC':<6} {'N':>4}  {'iperf3':>8} {'Cold':>8} {'Hot':>8} {'Write':>8} {'Ping':>8}")
    print(f"  {'─' * 90}")
    for _, r in tbl.iterrows():
        print(f"  {r['Machine']:<14} {r['Conn']:<7} {r['NIC']:<6} {r['N']:>4}  {r['iperf3']:>8} {r['Cold']:>8} {r['Hot']:>8} {r['Write']:>8} {r['Ping_ms']:>8}")
    print(f"  {'─' * 90}")


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
        print(f"  {'─' * 60}")
        print(f"  {'Metric':<25} {'Median':>10} {'Mean':>10} {'Std':>10}")
        print(f"  {'─' * 60}")

        for name, series in [('iperf3 (raw)', iperf), ('Cold cache', cold),
                             ('Hot cache', hot), ('True write', true_w)]:
            if len(series) > 0:
                print(f"  {name:<25} {series.median():>10.0f} {series.mean():>10.0f} {series.std():>10.0f}")

        print(f"  {'─' * 60}")

        # Overhead percentages
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
    print(f"  {'─' * 75}")

    for tier in ['1G', '2.5G', '5G']:
        t = switch[switch['nic_tier'] == tier]
        if t.empty:
            continue
        print(f"  {tier:<8} {len(t):>5}  "
              f"{t['iperf3_sender_mbits'].median():>10.0f} "
              f"{t['cold_cache_rate_mbits'].median():>10.0f} "
              f"{t['hot_cache_avg_mbits'].median():>10.0f} "
              f"{t['true_write_rate_mbits'].median():>10.0f} "
              f"{pd.to_numeric(t['ping_avg_ms'], errors='coerce').median():>10.2f}")
    print(f"  {'─' * 75}")
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

    print(f"\n  {'Metric':<20} {'Wire Med':>10} {'Switch Med':>10} {'Δ':>10} {'p-value':>10} {'Cohen d':>10}")
    print(f"  {'─' * 75}")

    for name, col in metrics:
        w = wire[col].dropna()
        s = switch[col].dropna()
        if len(w) < 2 or len(s) < 2:
            continue

        w_med = w.median()
        s_med = s.median()
        delta = w_med - s_med

        # Welch's t-test
        t_stat, p_val = scipy_stats.ttest_ind(w, s, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt((w.std()**2 + s.std()**2) / 2)
        d = (w.mean() - s.mean()) / pooled_std if pooled_std > 0 else 0

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {name:<20} {w_med:>10.0f} {s_med:>10.0f} {delta:>+10.0f} {p_val:>9.1e} {d:>+9.2f} {sig}")

    # Ping comparison
    w_ping = pd.to_numeric(wire['ping_avg_ms'], errors='coerce').dropna()
    s_ping = pd.to_numeric(switch['ping_avg_ms'], errors='coerce').dropna()
    if len(w_ping) > 1 and len(s_ping) > 1:
        t_stat, p_val = scipy_stats.ttest_ind(w_ping, s_ping, equal_var=False)
        pooled_std = np.sqrt((w_ping.std()**2 + s_ping.std()**2) / 2)
        d = (w_ping.mean() - s_ping.mean()) / pooled_std if pooled_std > 0 else 0
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {'Ping (ms)':<20} {w_ping.median():>10.2f} {s_ping.median():>10.2f} {w_ping.median()-s_ping.median():>+10.2f} {p_val:>9.1e} {d:>+9.2f} {sig}")

    print(f"  {'─' * 75}")
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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <all_results.csv>")
        sys.exit(1)

    filepath = sys.argv[1]

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Network Performance Analysis — iperf3 + fileiotest                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    print(f"\nLoading: {filepath}")
    df = load_data(filepath)
    print(f"  Total: {len(df)} samples from {df['source'].nunique()} machines")
    print(f"  Sources: {', '.join(sorted(df['source'].unique()))}")

    # Time sync
    print("\nApplying time-sync filter (thais_wire window)...")
    df = time_sync_filter(df)

    # iperf3 coverage
    iperf_valid = df['iperf3_sender_mbits'].notna().sum()
    print(f"\n  iperf3 coverage: {iperf_valid}/{len(df)} samples ({iperf_valid/len(df)*100:.0f}%)")

    summary_table(df)
    overhead_analysis(df)
    nic_tier_analysis(df)
    wire_vs_switch(df)
    iperf3_retransmit_summary(df)
    fileiotest_retransmit_summary(df)
    weekday_vs_weekend(df)

    print(f"\n{'═' * 70}")
    print("  Analysis complete.")
    print(f"{'═' * 70}\n")


if __name__ == '__main__':
    main()
