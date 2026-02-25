#!/usr/bin/env python3
"""parse_run.py — Parse a single fileiotest stats file into a CSV row.

Reads the structured output from fileiotest.sh which contains:
  - Per-phase TCP retransmit deltas (from nstat)
  - Per-phase pv throughput lines

Stats file format:
    <date line>
    <user@host>
    Initial stats
    TcpRetransSegs  ...
    === COLD-CACHE RUN ===
    bytes=28.6MiB rate=[31.2MiB/s]
    TcpRetransSegs  ...
    === 3 HOT-CACHE RUNS ===
    bytes=... rate=[...]       (run 1)
    bytes=... rate=[...]       (run 2)
    bytes=... rate=[...]       (run 3)
    TcpRetransSegs  ...
    === TRUE WRITE ===
    bytes=... rate=[...]
    TcpRetransSegs  ...

Usage:
    python3 parse_run.py <statsfile> <timestamp> <num_files> <source_label> [ping_log] [iperf3_log]
    python3 parse_run.py --header
"""

import sys
import os
import re
import csv
import json

# ─── Constants ────────────────────────────────────────────────────────

SECTION_MARKERS = {
    'cold_cache': r'===\s*COLD-CACHE\s+RUN\s*===',
    'hot_cache':  r'===\s*3\s+HOT-CACHE\s+RUNS\s*===',
    'true_write': r'===\s*TRUE\s+WRITE\s*===',
}

TCP_FIELDS = ['TcpRetransSegs', 'TcpExtTCPTimeouts', 'TcpOutRsts']

TCP_FIELD_MAP = {
    'tcpretranssegs':    'TcpRetransSegs',
    'tcpexttcptimeouts': 'TcpExtTCPTimeouts',
    'tcpoutrsts':        'TcpOutRsts',
}


# ─── Parsing helpers ──────────────────────────────────────────────────

def parse_pv_line(line: str) -> dict:
    """Parse a pv output line like:
        bytes=28.6MiB rate=[31.2MiB/s]    (bracketed)
        bytes=95.4MiB rate= 763Mib/s      (unbracketed)
        bytes= 953MiB rate=[ 278MiB/s]    (space-padded brackets)
    """
    result = {'bytes': '', 'rate': ''}
    if not line or 'bytes=' not in line:
        return result
    m = re.search(r'bytes=\s*(\S+)', line)
    if m:
        result['bytes'] = m.group(1).strip('[]')
    m = re.search(r'rate=\s*(\[.*?\]|\S+)', line)
    if m:
        result['rate'] = m.group(1).strip('[]')
    return result


def parse_nstat_lines(lines: list) -> dict:
    """Extract TCP counters from nstat-style lines."""
    result = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            field_lower = parts[0].lower()
            if field_lower in TCP_FIELD_MAP:
                canonical = TCP_FIELD_MAP[field_lower]
                try:
                    result[canonical] = int(float(parts[1]))
                except (ValueError, IndexError):
                    pass
    return result


def tcp_delta(after: dict, before: dict) -> dict:
    """Compute per-phase TCP deltas from cumulative counters."""
    delta = {}
    for field in TCP_FIELDS:
        a = after.get(field, 0)
        b = before.get(field, 0)
        delta[field] = max(0, a - b)
    return delta


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


def get_tcp_from_lines(lines: list) -> dict:
    """Extract TCP stats from a section's lines."""
    tcp_lines = [l for l in lines if any(
        k in l.lower() for k in TCP_FIELD_MAP
    )]
    return parse_nstat_lines(tcp_lines)


def parse_pv_from_lines(lines: list) -> list:
    """Extract all pv output lines from a section."""
    return [l for l in lines if 'bytes=' in l and 'rate=' in l]


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
        r'rtt min/avg/max/mdev\s*=\s*'
        r'([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)',
        text
    )
    if m:
        result['ping_min_ms'] = m.group(1)
        result['ping_avg_ms'] = m.group(2)
        result['ping_max_ms'] = m.group(3)
        result['ping_mdev_ms'] = m.group(4)
    return result


def parse_iperf3_log(filepath: str) -> dict:
    """Parse iperf3 JSON output for bandwidth and retransmit stats.

    Returns dict with:
        iperf3_sender_bps:    sender bitrate in bits/sec
        iperf3_receiver_bps:  receiver bitrate in bits/sec
        iperf3_retransmits:   total TCP retransmits during test
    """
    result = {
        'iperf3_sender_bps': '',
        'iperf3_receiver_bps': '',
        'iperf3_retransmits': '',
    }
    if not filepath or not os.path.isfile(filepath):
        return result
    try:
        with open(filepath, 'r', errors='replace') as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception):
        return result

    # iperf3 JSON structure: end -> sum_sent / sum_received
    end = data.get('end', {})
    if not end:
        return result

    sum_sent = end.get('sum_sent', {})
    sum_received = end.get('sum_received', {})

    if sum_sent:
        bps = sum_sent.get('bits_per_second', '')
        if bps != '':
            result['iperf3_sender_bps'] = f'{bps:.0f}'
        retrans = sum_sent.get('retransmits', '')
        if retrans != '':
            result['iperf3_retransmits'] = str(retrans)

    if sum_received:
        bps = sum_received.get('bits_per_second', '')
        if bps != '':
            result['iperf3_receiver_bps'] = f'{bps:.0f}'

    return result


def build_csv_row(timestamp: str, num_files: str, source: str,
                  cold_pv: dict, cold_tcp: dict,
                  hot_pvs: list, hot_tcp: dict,
                  write_pv: dict, write_tcp: dict,
                  ping: dict, iperf3: dict) -> list:
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
    # iperf3
    row.extend([
        iperf3.get('iperf3_sender_bps', ''),
        iperf3.get('iperf3_receiver_bps', ''),
        iperf3.get('iperf3_retransmits', ''),
    ])
    # Cold cache
    row.extend([
        cold_pv.get('bytes', ''),
        cold_pv.get('rate', ''),
        cold_tcp.get('TcpRetransSegs', 0),
        cold_tcp.get('TcpExtTCPTimeouts', 0),
        cold_tcp.get('TcpOutRsts', 0),
    ])
    # Hot cache (3 runs)
    for i in range(3):
        if i < len(hot_pvs):
            row.extend([hot_pvs[i].get('bytes', ''), hot_pvs[i].get('rate', '')])
        else:
            row.extend(['', ''])
    # Hot cache TCP deltas (combined for all 3 runs)
    row.extend([
        hot_tcp.get('TcpRetransSegs', 0),
        hot_tcp.get('TcpExtTCPTimeouts', 0),
        hot_tcp.get('TcpOutRsts', 0),
    ])
    # True write
    row.extend([
        write_pv.get('bytes', ''),
        write_pv.get('rate', ''),
        write_tcp.get('TcpRetransSegs', 0),
        write_tcp.get('TcpExtTCPTimeouts', 0),
        write_tcp.get('TcpOutRsts', 0),
    ])
    return row


def csv_header() -> list:
    """Return the CSV header row."""
    header = [
        'timestamp', 'source', 'num_files',
        # Ping
        'ping_min_ms', 'ping_avg_ms', 'ping_max_ms',
        'ping_mdev_ms', 'ping_loss_pct',
        # iperf3
        'iperf3_sender_bps', 'iperf3_receiver_bps', 'iperf3_retransmits',
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
              f"<source_label> [ping_log] [iperf3_log]",
              file=sys.stderr)
        sys.exit(1)

    statsfile = sys.argv[1]
    timestamp = sys.argv[2]
    num_files = sys.argv[3]
    source_label = sys.argv[4]
    ping_logfile = sys.argv[5] if len(sys.argv) > 5 else None
    iperf3_logfile = sys.argv[6] if len(sys.argv) > 6 else None

    with open(statsfile, 'r', errors='replace') as f:
        stats_text = f.read()

    sections = split_stats_file(stats_text)

    # Get cumulative TCP stats from each section
    tcp_initial = get_tcp_from_lines(sections.get('initial', []))
    tcp_cold_cum = get_tcp_from_lines(sections.get('cold_cache', []))
    tcp_hot_cum = get_tcp_from_lines(sections.get('hot_cache', []))
    tcp_write_cum = get_tcp_from_lines(sections.get('true_write', []))

    # Compute per-phase deltas
    cold_tcp = tcp_delta(tcp_cold_cum, tcp_initial)
    hot_tcp = tcp_delta(tcp_hot_cum, tcp_cold_cum)
    write_tcp = tcp_delta(tcp_write_cum, tcp_hot_cum)

    # Parse pv throughput lines
    cold_pv_lines = parse_pv_from_lines(sections.get('cold_cache', []))
    cold_pv = parse_pv_line(cold_pv_lines[0]) if cold_pv_lines else {}

    hot_pv_lines = parse_pv_from_lines(sections.get('hot_cache', []))
    hot_pvs = [parse_pv_line(l) for l in hot_pv_lines[:3]]

    write_pv_lines = parse_pv_from_lines(sections.get('true_write', []))
    write_pv = parse_pv_line(write_pv_lines[0]) if write_pv_lines else {}

    # Parse ping log
    ping = parse_ping_log(ping_logfile) if ping_logfile else {}

    # Parse iperf3 log
    iperf3 = parse_iperf3_log(iperf3_logfile) if iperf3_logfile else {}

    row = build_csv_row(timestamp, num_files, source_label,
                        cold_pv, cold_tcp,
                        hot_pvs, hot_tcp,
                        write_pv, write_tcp,
                        ping, iperf3)

    writer = csv.writer(sys.stdout)
    writer.writerow(row)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--header':
        writer = csv.writer(sys.stdout)
        writer.writerow(csv_header())
        sys.exit(0)
    main()
