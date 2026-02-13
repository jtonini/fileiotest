#!/usr/bin/env bash
# collector.sh — Weekly data collector for fileiotest benchmarks
#
# Runs fileiotest.sh at regular intervals, captures ping latency
# alongside George's stats file, and parses everything into CSV.
#
# fileiotest.sh now handles:
#   - Remote cache clearing on the destination
#   - TCP retransmit stats via nstat (per-phase deltas)
#   - Structured output to a stats file
#
# This collector adds:
#   - Ping latency per sample
#   - Scheduling (run every N minutes for M hours)
#   - CSV aggregation via parse_run.py
#
# Usage:
#   ./collector.sh <user@ipaddress> [options]
#
# Options (via environment variables):
#   NUM_FILES       Comma-separated list of file counts to cycle through.
#                   Default: "10"  (single fixed payload)
#   INTERVAL_MIN    Minutes between samples. Default: 15
#   DURATION_HR     Total collection duration in hours. Default: 168 (1 week)
#   RESULTS_DIR     Where to store results. Default: ./collector_results
#   PING_COUNT      Number of pings per sample. Default: 20
#   SOURCE_LABEL    Label for this source machine. Default: $(hostname -s)
#
set -euo pipefail

# ─── Arguments ────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <user@ipaddress>"
    echo ""
    echo "Environment variables for configuration:"
    echo "  NUM_FILES     Comma-separated file counts (default: 10)"
    echo "  INTERVAL_MIN  Minutes between runs (default: 15)"
    echo "  DURATION_HR   Total hours to collect (default: 168 = 1 week)"
    echo "  RESULTS_DIR   Output directory (default: ./collector_results)"
    echo "  PING_COUNT    Pings per sample (default: 20)"
    echo "  SOURCE_LABEL  Label for source machine (default: hostname)"
    exit 1
fi

DEST="$1"
DEST_HOST="${DEST#*@}"

# ─── Configuration ────────────────────────────────────────────────────
IFS=',' read -ra FILE_COUNTS <<< "${NUM_FILES:-10}"
INTERVAL_MIN="${INTERVAL_MIN:-15}"
DURATION_HR="${DURATION_HR:-168}"
RESULTS_DIR="${RESULTS_DIR:-./collector_results}"
PING_COUNT="${PING_COUNT:-20}"
SOURCE_LABEL="${SOURCE_LABEL:-$(hostname -s)}"
INTERVAL_SEC=$(( INTERVAL_MIN * 60 ))
TOTAL_SAMPLES=$(( (DURATION_HR * 60) / INTERVAL_MIN ))

# Find fileiotest.sh relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILEIOTEST="$SCRIPT_DIR/fileiotest.sh"
PARSER="$SCRIPT_DIR/parse_run.py"

if [[ ! -x "$FILEIOTEST" ]]; then
    echo "ERROR: fileiotest.sh not found or not executable at $FILEIOTEST"
    exit 1
fi

if [[ ! -f "$PARSER" ]]; then
    echo "ERROR: parse_run.py not found at $PARSER"
    exit 1
fi

# ─── Setup ────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR/logs"

CSV_FILE="$RESULTS_DIR/results_${SOURCE_LABEL}.csv"
SUMMARY_FILE="$RESULTS_DIR/collection_summary_${SOURCE_LABEL}.txt"

if [[ ! -f "$CSV_FILE" ]]; then
    python3 "$PARSER" --header > "$CSV_FILE"
fi

START_TIME="$(date -Iseconds)"
cat > "$SUMMARY_FILE" <<EOF
=== Collection Configuration ===
Started:            $START_TIME
Source:             $SOURCE_LABEL ($(hostname))
Destination:        $DEST
File counts:        ${FILE_COUNTS[*]}
Interval:           ${INTERVAL_MIN} minutes
Duration:           ${DURATION_HR} hours
Total samples:      $TOTAL_SAMPLES
Ping count:         $PING_COUNT per sample
Results dir:        $RESULTS_DIR
Kernel:             $(uname -r)
EOF

DAYS=$(( DURATION_HR / 24 ))
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  fileiotest Collector — ${SOURCE_LABEL}                     "
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Source:         ${SOURCE_LABEL} → ${DEST}"
echo "║  File counts:    ${FILE_COUNTS[*]}"
echo "║  Interval:       every ${INTERVAL_MIN} min"
echo "║  Duration:       ${DAYS} days ($TOTAL_SAMPLES samples)"
echo "║  Results:        $CSV_FILE"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Trap for graceful shutdown ───────────────────────────────────────
RUNNING=true
cleanup() {
    echo ""
    echo "[$(date -Iseconds)] ${SOURCE_LABEL}: Caught signal — stopping after current run."
    RUNNING=false
}
trap cleanup SIGINT SIGTERM

# ─── Main collection loop ────────────────────────────────────────────
sample=0
failures=0

while [[ $sample -lt $TOTAL_SAMPLES ]] && [[ "$RUNNING" == "true" ]]; do
    idx=$(( sample % ${#FILE_COUNTS[@]} ))
    NUM="${FILE_COUNTS[$idx]}"

    TIMESTAMP="$(date -Iseconds)"
    SAFE_TS="$(date +%Y%m%d_%H%M%S)"
    LOGFILE="$(cd "$RESULTS_DIR" && pwd)/logs/${SOURCE_LABEL}_run_${SAFE_TS}_n${NUM}.log"
    STATSFILE="$(cd "$RESULTS_DIR" && pwd)/logs/${SOURCE_LABEL}_stats_${SAFE_TS}_n${NUM}.txt"

    sample=$(( sample + 1 ))
    echo "[${TIMESTAMP}] ${SOURCE_LABEL}: Sample ${sample}/${TOTAL_SAMPLES} — NUM=${NUM}"

    # ── Ping latency ─────────────────────────────────────────────────
    PINGLOG="$(cd "$RESULTS_DIR" && pwd)/logs/${SOURCE_LABEL}_ping_${SAFE_TS}.log"
    ping -c "$PING_COUNT" -q "$DEST_HOST" > "$PINGLOG" 2>&1 || true

    # ── Run the benchmark ────────────────────────────────────────────
    # fileiotest.sh now takes 3 args: numfiles, destination, stats-file
    # It handles remote cache clearing and TCP stats internally.
    WORKDIR="$(mktemp -d)"
    cp "$SCRIPT_DIR/randomfiles.py" "$WORKDIR/"

    run_ok=true
    (
        cd "$WORKDIR"
        "$FILEIOTEST" "$NUM" "$DEST" "$STATSFILE"
    ) > "$LOGFILE" 2>&1 || run_ok=false

    rm -rf "$WORKDIR"

    # ── Parse and record ─────────────────────────────────────────────
    if [[ "$run_ok" == "true" ]]; then
        python3 "$PARSER" "$STATSFILE" "$TIMESTAMP" "$NUM" \
            "$SOURCE_LABEL" "$PINGLOG" \
            >> "$CSV_FILE" 2>/dev/null \
            && echo "  ✓ Parsed → $CSV_FILE" \
            || echo "  ⚠ Parse warning — raw log at $LOGFILE"
    else
        failures=$(( failures + 1 ))
        echo "  ✗ Run failed — see $LOGFILE"
    fi

    # ── Sleep until next interval ────────────────────────────────────
    if [[ $sample -lt $TOTAL_SAMPLES ]] && [[ "$RUNNING" == "true" ]]; then
        remaining=$INTERVAL_SEC
        while [[ $remaining -gt 0 ]] && [[ "$RUNNING" == "true" ]]; do
            chunk=$(( remaining > 10 ? 10 : remaining ))
            sleep "$chunk"
            remaining=$(( remaining - chunk ))
        done
    fi
done

# ─── Final summary ───────────────────────────────────────────────────
END_TIME="$(date -Iseconds)"
cat >> "$SUMMARY_FILE" <<EOF

=== Collection Complete ===
Ended:          $END_TIME
Samples run:    $sample
Failures:       $failures
CSV file:       $CSV_FILE
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ${SOURCE_LABEL}: Collection complete."
echo "  Samples: ${sample}  |  Failures: ${failures}"
echo "  CSV:     $CSV_FILE"
echo "════════════════════════════════════════════════════════════════"
