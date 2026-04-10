#!/usr/bin/env bash
# launch_collector_sv.sh — Start resumable collector in background.
# Deployed and called by orchestrator_sarahvaughan.sh.
#
# Usage: bash launch_collector_sv.sh <dest> <source_label> <num_files> <interval> <duration> <ping_count>

set -u

DEST="$1"
SOURCE_LABEL="$2"
NUM_FILES="${3:-100}"
INTERVAL_MIN="${4:-15}"
DURATION_HR="${5:-24}"
PING_COUNT="${6:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NUM_FILES INTERVAL_MIN DURATION_HR PING_COUNT SOURCE_LABEL
export RESULTS_DIR="./collector_results"
export PATH="$SCRIPT_DIR:/usr/local/bin:/usr/bin:/usr/sbin:$PATH"

nohup bash ./collector_resumable.sh "$DEST" \
    > "${SCRIPT_DIR}/collector_${SOURCE_LABEL}.log" 2>&1 </dev/null &

echo $!
