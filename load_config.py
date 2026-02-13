#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_config.py — Read config.toml for fileiotest scripts.

Can be used in two ways:

1. As a Python module:
    from load_config import load_config
    cfg = load_config()
    print(cfg['destination']['machine'])

2. From bash (outputs shell export statements):
    eval "$(python3 load_config.py)"
    echo $DEST_MACHINE   # → myhost

Supports Python 3.9+ (uses tomllib if available, falls back to
a simple parser for our limited TOML subset).
"""

__author__ = 'João Tonini / Claude'
__version__ = '0.2'

import os
import re
import sys

CONFIG_FILE = 'config.toml'


def _find_config() -> str:
    """Find config.toml relative to this script or cwd."""
    if os.path.isfile(CONFIG_FILE):
        return CONFIG_FILE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, CONFIG_FILE)
    if os.path.isfile(candidate):
        return candidate
    return ''


def _simple_toml_parse(text: str) -> dict:
    """Minimal TOML parser for our flat config structure."""
    config = {}
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if '#' in stripped and not stripped.startswith('"'):
            in_quote = False
            for i, ch in enumerate(stripped):
                if ch == '"':
                    in_quote = not in_quote
                elif ch == '#' and not in_quote:
                    stripped = stripped[:i].rstrip()
                    break
        cleaned.append(stripped)

    full_text = '\n'.join(cleaned)
    section_pattern = re.compile(r'^\[(\w+)\]\s*$', re.MULTILINE)
    sections = section_pattern.split(full_text)

    i = 1
    while i < len(sections) - 1:
        section_name = sections[i]
        section_body = sections[i + 1]
        config[section_name] = _parse_section(section_body)
        i += 2

    return config


def _parse_section(body: str) -> dict:
    result = {}
    body = _collapse_arrays(body)
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not m:
            continue
        key = m.group(1)
        raw_value = m.group(2).strip()
        if raw_value.startswith('['):
            items = re.findall(r'"([^"]*)"', raw_value)
            result[key] = items
        elif raw_value.startswith('"') and raw_value.endswith('"'):
            result[key] = raw_value.strip('"')
        elif raw_value.isdigit():
            result[key] = int(raw_value)
        elif re.match(r'^\d+\.\d+$', raw_value):
            result[key] = float(raw_value)
        elif raw_value.lower() in ('true', 'false'):
            result[key] = raw_value.lower() == 'true'
        else:
            result[key] = raw_value
    return result


def _collapse_arrays(text: str) -> str:
    result = []
    in_array = False
    array_buf = []
    for line in text.splitlines():
        stripped = line.strip()
        if in_array:
            array_buf.append(stripped)
            if ']' in stripped:
                result.append(' '.join(array_buf))
                array_buf = []
                in_array = False
        elif '= [' in stripped and ']' not in stripped:
            in_array = True
            array_buf = [stripped]
        else:
            result.append(stripped)
    if array_buf:
        result.append(' '.join(array_buf))
    return '\n'.join(result)


def load_config(config_path: str = '') -> dict:
    """Load and parse config.toml."""
    if not config_path:
        config_path = _find_config()
    if not config_path:
        print(f"ERROR: {CONFIG_FILE} not found.", file=sys.stderr)
        print(f"  Run: python3 setup_wizard.py", file=sys.stderr)
        sys.exit(1)
    with open(config_path, 'r') as f:
        text = f.read()
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        pass
    try:
        import tomli
        return tomli.loads(text)
    except ImportError:
        pass
    return _simple_toml_parse(text)


def config_to_shell(cfg: dict) -> str:
    """Convert config dict to shell export statements."""
    exports = []

    # Destination (campus network)
    d = cfg.get('destination', {})
    exports.append(f'export DEST_MACHINE="{d.get("machine", "")}"')
    exports.append(f'export DEST_USER="{d.get("user", "")}"')
    dest = f'{d.get("user", "")}@{d.get("machine", "")}'
    exports.append(f'export DEST="{dest}"')

    # Wire connection (point-to-point)
    w = cfg.get('wire', {})
    exports.append(f'export WIRE_SOURCE_MACHINE="{w.get("source_machine", "")}"')
    exports.append(f'export WIRE_DEST_IP="{w.get("dest_ip", "")}"')
    exports.append(f'export WIRE_SOURCE_IP="{w.get("source_ip", "")}"')
    wire_dest = f'{d.get("user", "")}@{w.get("dest_ip", "")}'
    exports.append(f'export WIRE_DEST="{wire_dest}"')

    # Switch source machines
    s = cfg.get('source', {})
    switch_machines = s.get('switch_machines', [])
    arr = ' '.join(f'"{m}"' for m in switch_machines)
    exports.append(f'export SWITCH_MACHINES=({arr})')

    # All sources (wire + switch)
    all_sources = [w.get('source_machine', '')] + switch_machines
    all_arr = ' '.join(f'"{m}"' for m in all_sources if m)
    exports.append(f'export ALL_SOURCES=({all_arr})')

    # Paths
    p = cfg.get('paths', {})
    exports.append(f'export DEPLOY_DIR="{p.get("deploy_dir", "~/fileiotest")}"')
    exports.append(f'export RESULTS_DIR="{p.get("results_dir", "./collector_results")}"')

    # Collection
    c = cfg.get('collection', {})
    exports.append(f'export NUM_FILES="{c.get("num_files", "10")}"')
    exports.append(f'export INTERVAL_MIN="{c.get("interval_min", 15)}"')
    exports.append(f'export DURATION_HR="{c.get("duration_hr", 168)}"')
    exports.append(f'export PING_COUNT="{c.get("ping_count", 20)}"')

    return '\n'.join(exports)


if __name__ == '__main__':
    cfg = load_config()
    print(config_to_shell(cfg))
