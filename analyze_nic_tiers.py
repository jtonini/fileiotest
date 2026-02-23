#!/usr/bin/env python3
"""analyze_nic_tiers.py — Break down throughput by NIC link speed.

Usage:
    python3 analyze_nic_tiers.py all_results/all_results.csv
"""

import sys
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── NIC speed mapping (Mbit/s) ──────────────────────────────────────
NIC_SPEED = {
    "evan":     100,
    "kevin":    100,
    "mayer":    100,
    "aamy":     1000,
    "alexis":   1000,
    "boyi":     1000,
    "josh":     1000,
    "justin":   1000,
    "khanh":    1000,
    "camryn":   2500,
    "irene2":   2500,
    "cooper":   5000,
    "hamilton":  5000,
    "thais":    5000,
    "sarah":    5000,  # wire (5G ↔ 5G)
}

TIER_LABELS = {
    100:  "100 Mbit",
    1000: "1 Gbit",
    2500: "2.5 Gbit",
    5000: "5 Gbit",
}

TIER_COLORS = {
    "100 Mbit":   "#d62728",
    "1 Gbit":     "#1f77b4",
    "2.5 Gbit":   "#ff7f0e",
    "5 Gbit":     "#2ca02c",
    "Wire (5G)":  "#9467bd",
}


def parse_rate(s):
    """Convert pv rate string like '74.4MiB/s' to MiB/s float."""
    if pd.isna(s):
        return None
    m = re.match(r"([\d.]+)\s*(MiB|GiB|KiB)/s", str(s))
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "GiB":
        val *= 1024
    elif unit == "KiB":
        val /= 1024
    return val * 8  # Convert MiB/s → Mbit/s


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Parse rates to Mbit/s
    df["cold_mbit"] = df["cold_cache_rate"].apply(parse_rate)
    df["true_write_mbit"] = df["true_write_rate"].apply(parse_rate)

    # Average hot cache across 3 runs
    for col in ["hot_cache_run1_rate", "hot_cache_run2_rate", "hot_cache_run3_rate"]:
        df[col + "_mbit"] = df[col].apply(parse_rate)
    hot_cols = [c + "_mbit" for c in ["hot_cache_run1_rate", "hot_cache_run2_rate", "hot_cache_run3_rate"]]
    df["hot_cache_avg_mbit"] = df[hot_cols].mean(axis=1)

    # Assign NIC speed tier
    df["nic_mbit"] = df["source"].map(NIC_SPEED)
    df["tier"] = df["nic_mbit"].map(TIER_LABELS)
    # Mark wire separately
    df.loc[df["source"] == "sarah", "tier"] = "Wire (5G)"

    return df


def plot_tier_boxplots(df, out_prefix):
    """Box plots of throughput by NIC tier for each phase."""
    phases = [
        ("cold_mbit", "Cold-Cache Throughput"),
        ("hot_cache_avg_mbit", "Hot-Cache Avg Throughput"),
        ("true_write_mbit", "True-Write Throughput"),
    ]

    tier_order = ["100 Mbit", "1 Gbit", "2.5 Gbit", "5 Gbit", "Wire (5G)"]
    present_tiers = [t for t in tier_order if t in df["tier"].unique()]

    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 6))
    if len(phases) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, phases):
        data = []
        labels = []
        colors = []
        for tier in present_tiers:
            vals = df.loc[df["tier"] == tier, col].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append(f"{tier}\n(n={len(vals)})")
                colors.append(TIER_COLORS.get(tier, "#999"))

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("Mbit/s")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Throughput by NIC Link Speed", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{out_prefix}_tier_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tier_per_machine(df, out_prefix):
    """Horizontal bar chart grouped by NIC tier."""
    phases = [
        ("cold_mbit", "Cold-Cache", "#1f77b4"),
        ("hot_cache_avg_mbit", "Hot-Cache Avg", "#ff7f0e"),
        ("true_write_mbit", "True Write", "#2ca02c"),
    ]

    tier_order = ["Wire (5G)", "5 Gbit", "2.5 Gbit", "1 Gbit", "100 Mbit"]
    present_tiers = [t for t in tier_order if t in df["tier"].unique()]

    # Build ordered machine list grouped by tier
    machines = []
    tier_boundaries = []
    for tier in present_tiers:
        tier_machines = (
            df[df["tier"] == tier]
            .groupby("source")["cold_mbit"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        if tier_machines:
            tier_boundaries.append((len(machines), tier))
            machines.extend(tier_machines)

    fig, ax = plt.subplots(figsize=(12, max(6, len(machines) * 0.5)))

    y_pos = np.arange(len(machines))
    bar_height = 0.25

    for i, (col, label, color) in enumerate(phases):
        means = []
        stds = []
        for machine in machines:
            vals = df.loc[df["source"] == machine, col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)

        ax.barh(
            y_pos + (i - 1) * bar_height,
            means,
            bar_height,
            xerr=stds,
            label=label,
            color=color,
            alpha=0.7,
            capsize=2,
        )

    # Add tier labels and dividers
    for idx, tier in tier_boundaries:
        ax.axhline(y=idx - 0.5, color="gray", linewidth=0.5, linestyle="--")
        # Find the last machine in this tier for label placement
        next_idx = len(machines)
        for other_idx, _ in tier_boundaries:
            if other_idx > idx:
                next_idx = other_idx
                break
        mid = (idx + next_idx - 1) / 2
        ax.text(
            -0.02, mid, tier,
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=9, fontweight="bold",
            color=TIER_COLORS.get(tier, "#333"),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(machines)
    ax.set_xlabel("Mean Throughput (Mbit/s)")
    ax.set_title("Per-Machine Throughput — Grouped by NIC Speed", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = f"{out_prefix}_tier_per_machine.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tier_timeseries(df, out_prefix):
    """Throughput over time colored by NIC tier."""
    phases = [
        ("cold_mbit", "Cold-Cache"),
        ("hot_cache_avg_mbit", "Hot-Cache Avg"),
        ("true_write_mbit", "True Write"),
    ]

    tier_order = ["100 Mbit", "1 Gbit", "2.5 Gbit", "5 Gbit", "Wire (5G)"]
    present_tiers = [t for t in tier_order if t in df["tier"].unique()]

    fig, axes = plt.subplots(len(phases), 1, figsize=(14, 4 * len(phases)), sharex=True)

    for ax, (col, title) in zip(axes, phases):
        for tier in present_tiers:
            tier_df = df[df["tier"] == tier].copy()
            if tier_df[col].dropna().empty:
                continue

            # Resample to 30-min median per tier
            tier_df = tier_df.set_index("timestamp")
            resampled = tier_df[col].resample("30min").median().dropna()

            ax.plot(
                resampled.index, resampled.values,
                label=tier,
                color=TIER_COLORS.get(tier, "#999"),
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_ylabel("Mbit/s")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Throughput Over Time — by NIC Speed Tier", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{out_prefix}_tier_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_tier_summary(df):
    """Print summary statistics by tier."""
    tier_order = ["100 Mbit", "1 Gbit", "2.5 Gbit", "5 Gbit", "Wire (5G)"]
    present_tiers = [t for t in tier_order if t in df["tier"].unique()]

    phases = [
        ("cold_mbit", "Cold-Cache"),
        ("hot_cache_avg_mbit", "Hot-Cache Avg"),
        ("true_write_mbit", "True Write"),
        ("ping_avg_ms", "Ping Avg (ms)"),
    ]

    print("\n" + "=" * 80)
    print("THROUGHPUT SUMMARY BY NIC SPEED TIER")
    print("=" * 80)

    for col, label in phases:
        print(f"\n--- {label} ---")
        print(f"  {'Tier':<12} {'Mean':>10} {'Std':>10} {'Median':>10} {'N':>8} {'Machines':>10}")
        for tier in present_tiers:
            vals = df.loc[df["tier"] == tier, col].dropna()
            n_machines = df.loc[df["tier"] == tier, "source"].nunique()
            if len(vals) > 0:
                print(
                    f"  {tier:<12} {vals.mean():>10.1f} {vals.std():>10.1f} "
                    f"{vals.median():>10.1f} {len(vals):>8} {n_machines:>10}"
                )

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_nic_tiers.py <results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_prefix = csv_path.rsplit(".", 1)[0]

    print(f"Loading {csv_path}...")
    df = load_data(csv_path)
    print(f"  {len(df)} samples from {df['source'].nunique()} machines")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    print_tier_summary(df)

    print("\nGenerating plots...")
    plot_tier_boxplots(df, out_prefix)
    plot_tier_per_machine(df, out_prefix)
    plot_tier_timeseries(df, out_prefix)

    print("\nDone!")


if __name__ == "__main__":
    main()
