#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_run.py — Parse the stats file from fileiotest.sh.

Reads the structured stats file that fileiotest.sh produces (3rd arg),
plus an optional ping log from the collector. Outputs a single CSV row.

Stats file format (produced by George's fileiotest.sh):
    <date>
    user@host to dest
    Initial stats
    TcpRetransSegs     0    0.0
    ...
    === COLD-CACHE RUN ===
    bytes=95.4MiB rate=31.8MiB/s
    TcpRetransSegs     12   0.0
    ...
    === 3 HOT-CACHE RUNS ===
    bytes=... rate=...       (run 1)
    bytes=... rate=...       (run 2)
    bytes=... rate=...       (run 3)
    TcpRetransSegs     ...
    ...
    === TRUE WRITE ===
    bytes=... rate=...
    TcpRetransSegs     ...
    ...

Usage:
    python3 parse_run.py <statsfile> <timestamp> <num_files> <source_label> [ping_log]
"""

__author__ = 'João Tonini / Claude'
__version__ = '0.3'

import csv
import re
import sys
import os

# Section markers in the stats file
SECTION_MARKERS = {
    'cold_cache':  r'===\s*COLD-CACHE\s+RUN\s*===',
    'hot_cache':   r'===\s*3\s+HOT-CACHE\s+RUNS\s*===',
    'true_write':  r'===\s*TRUE\s+WRITE\s*===',
}

# nstat TCP fields we track
TCP_FIELDS = ['TcpRetransSegs', 'TCPTimeouts', 'TcpOutRsts']


def parse_pv_line(line: str) -> dict:
    """Parse a pv output line in the new tagged format.

    Format: bytes=95.4MiB rate=31.8MiB/s
    """
    result = {'bytes': '', 'rate': ''}

    m = re.search(r'bytes=\s*(\S+)', line)
    if m:
        result['bytes'] = m.group(1)

    m = re.search(r'rate=\s*(\S+)', line)
    if m:
        result['rate'] = m.group(1)

    return result


def parse_nstat_lines(lines: list) -> dict:
    """Parse nstat output lines for TCP counters.

    nstat format:
        TcpRetransSegs     12     0.0

    Returns: {'TcpRetransSegs': 12, 'TCPTimeouts': 0, 'TcpOutRsts': 0}
    """
    result = {}
    for line in lines:
        for field in TCP_FIELDS:
            if field.lower() in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result[field] = int(float(parts[1]))
                    except (ValueError, IndexError):
                        result[field] = 0
    return result


def split_stats_file(text: str) -> dict:
    """Split the stats file into sections.

    Returns dict with keys: 'initial', 'cold_cache', 'hot_cache', 'true_write'
    Each value is a list of lines in that section.
    """
    lines = text.splitlines()
    sections = {'initial': []}
    current = 'initial'

    for line in lines:
        stripped = line.strip()
        matched = False
        for section_name, pattern in SECTION_MARKERS.items():
            if re.match(pattern, stripped):
                current = section_name
                sections[current] = []
                matched = True
                break
        if not matched:
            if current in sections:
                sections[current].append(stripped)
            else:
                sections[current] = [stripped]

    return sections


def parse_section_cold_or_write(lines: list) -> dict:
    """Parse a section with 1 pv line + nstat output.

    Used for COLD-CACHE and TRUE WRITE sections.
    """
    result = {
        'bytes': '', 'rate': '',
        'TcpRetransSegs': '', 'TCPTimeouts': '', 'TcpOutRsts': '',
    }

    # Find pv line (contains "bytes=" and "rate=")
    pv_lines = [l for l in lines if 'bytes=' in l and 'rate=' in l]
    if pv_lines:
        pv = parse_pv_line(pv_lines[0])
        result['bytes'] = pv['bytes']
        result['rate'] = pv['rate']

    # Find nstat lines
    nstat_lines = [l for l in lines
                   if any(f.lower() in l.lower() for f in TCP_FIELDS)]
    tcp = parse_nstat_lines(nstat_lines)
    for field in TCP_FIELDS:
        if field in tcp:
            result[field] = tcp[field]

    return result


def parse_section_hot_cache(lines: list) -> dict:
    """Parse the hot-cache section with 3 pv lines + nstat output.

    Returns rates for each run plus combined TCP stats.
    """
    result = {
        'run1_bytes': '', 'run1_rate': '',
        'run2_bytes': '', 'run2_rate': '',
        'run3_bytes': '', 'run3_rate': '',
        'TcpRetransSegs': '', 'TCPTimeouts': '', 'TcpOutRsts': '',
    }

    # Find all pv lines
    pv_lines = [l for l in lines if 'bytes=' in l and 'rate=' in l]
    for i, pv_line in enumerate(pv_lines[:3]):
        pv = parse_pv_line(pv_line)
        run = i + 1
        result[f'run{run}_bytes'] = pv['bytes']
        result[f'run{run}_rate'] = pv['rate']

    # Find nstat lines
    nstat_lines = [l for l in lines
                   if any(f.lower() in l.lower() for f in TCP_FIELDS)]
    tcp = parse_nstat_lines(nstat_lines)
    for field in TCP_FIELDS:
        if field in tcp:
            result[field] = tcp[field]

    return result


def parse_ping_log(filepath: str) -> dict:
    """Parse ping -c -q output for round-trip stats."""
    result = {
        'ping_min_ms': '',
        'ping_avg_ms': '',
        'ping_max_ms': '',
        'ping_mdev_ms': '',
        'ping_loss_pct': '',
    }

    if not filepath or not os.path.isfile(filepath):
        return result

    try:
        with open(filepath, 'r', errors='replace') as f:
            text = f.read()
    except Exception:
        return result

    m = re.search(r'(\d+(?:\.\d+)?)%\s+packet loss', text)
    if m:
        result['ping_loss_pct'] = m.group(1)

    m = re.search(
        r'rtt min/avg/max/mdev\s*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms',
        text
    )
    if m:
        result['ping_min_ms'] = m.group(1)
        result['ping_avg_ms'] = m.group(2)
        result['ping_max_ms'] = m.group(3)
        result['ping_mdev_ms'] = m.group(4)

    return result


def build_csv_row(timestamp: str, num_files: str, source: str,
                  cold: dict, hot: dict, write: dict,
                  ping: dict) -> list:
    """Build a flat CSV row."""
    row = [timestamp, source, num_files]

    # Ping
    row.extend([
        ping.get('ping_min_ms', ''),
        ping.get('ping_avg_ms', ''),
        ping.get('ping_max_ms', ''),
        ping.get('ping_mdev_ms', ''),
        ping.get('ping_loss_pct', ''),
    ])

    # Cold cache
    row.extend([
        cold.get('bytes', ''),
        cold.get('rate', ''),
        cold.get('TcpRetransSegs', ''),
        cold.get('TCPTimeouts', ''),
        cold.get('TcpOutRsts', ''),
    ])

    # Hot cache (3 runs)
    for run in [1, 2, 3]:
        row.extend([
            hot.get(f'run{run}_bytes', ''),
            hot.get(f'run{run}_rate', ''),
        ])
    # Hot cache TCP stats (combined for all 3 runs)
    row.extend([
        hot.get('TcpRetransSegs', ''),
        hot.get('TCPTimeouts', ''),
        hot.get('TcpOutRsts', ''),
    ])

    # True write
    row.extend([
        write.get('bytes', ''),
        write.get('rate', ''),
        write.get('TcpRetransSegs', ''),
        write.get('TCPTimeouts', ''),
        write.get('TcpOutRsts', ''),
    ])

    return row


def csv_header() -> list:
    """Return the CSV header row."""
    header = [
        'timestamp', 'source', 'num_files',
        # Ping
        'ping_min_ms', 'ping_avg_ms', 'ping_max_ms',
        'ping_mdev_ms', 'ping_loss_pct',
        # Cold cache
        'cold_cache_bytes', 'cold_cache_rate',
        'cold_cache_retrans', 'cold_cache_timeouts', 'cold_cache_outrsts',
        # Hot cache (3 runs)
        'hot_cache_run1_bytes', 'hot_cache_run1_rate',
        'hot_cache_run2_bytes', 'hot_cache_run2_rate',
        'hot_cache_run3_bytes', 'hot_cache_run3_rate',
        'hot_cache_retrans', 'hot_cache_timeouts', 'hot_cache_outrsts',
        # True write
        'true_write_bytes', 'true_write_rate',
        'true_write_retrans', 'true_write_timeouts', 'true_write_outrsts',
    ]
    return header


def main():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <statsfile> <timestamp> <num_files> "
              f"<source_label> [ping_log]",
              file=sys.stderr)
        sys.exit(1)

    statsfile = sys.argv[1]
    timestamp = sys.argv[2]
    num_files = sys.argv[3]
    source_label = sys.argv[4]
    ping_logfile = sys.argv[5] if len(sys.argv) > 5 else None

    with open(statsfile, 'r', errors='replace') as f:
        stats_text = f.read()

    sections = split_stats_file(stats_text)

    cold = parse_section_cold_or_write(sections.get('cold_cache', []))
    hot = parse_section_hot_cache(sections.get('hot_cache', []))
    write = parse_section_cold_or_write(sections.get('true_write', []))
    ping = parse_ping_log(ping_logfile) if ping_logfile else {}

    row = build_csv_row(timestamp, num_files, source_label,
                        cold, hot, write, ping)

    writer = csv.writer(sys.stdout)
    writer.writerow(row)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--header':
        writer = csv.writer(sys.stdout)
        writer.writerow(csv_header())
        sys.exit(0)

    main()
