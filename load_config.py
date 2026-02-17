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

Requires Python 3.11+ (tomllib) or 'pip install tomli' for older versions.
"""

__author__ = 'João Tonini'
__version__ = '0.3'

import os
import sys

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print(
            "ERROR: Python 3.11+ required (for tomllib), or install tomli:\n"
            "  pip install tomli",
            file=sys.stderr
        )
        sys.exit(1)

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


def load_config(config_path: str = '') -> dict:
    """Load and parse config.toml.

    Raises FileNotFoundError if config.toml is not found.
    """
    if not config_path:
        config_path = _find_config()
    if not config_path:
        raise FileNotFoundError(
            f"{CONFIG_FILE} not found. Run: python3 setup_wizard.py"
        )
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


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
    try:
        cfg = load_config()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(config_to_shell(cfg))
