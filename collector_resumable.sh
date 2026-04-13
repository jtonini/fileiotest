#!/usr/bin/env bash
# collector_resumable.sh — Drop-in replacement for collector.sh with resume support.
#
# This is a thin wrapper that patches the sample counter to resume
# from the last completed sample. It reads the existing CSV file
# (if any), counts the data rows, and adjusts the starting sample
# number accordingly.
#
# If the destination is temporarily unreachable (e.g., cable swap),
# the collector will log failures but keep running. When the
# destination comes back, sampling resumes automatically.
#
# Usage: Same as collector.sh
#   export SOURCE_LABEL='machine_campus'
#   export NUM_FILES='100' INTERVAL_MIN='15' DURATION_HR='24' PING_COUNT='20'
#   bash collector_resumable.sh 'root@sarahvaughan'
#
# Or via launch_collector.sh (which sets these env vars).

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────
DEST="${1:?Usage: $0 user@host}"
DEST_HOST="${DEST#*@}"

IFS=',' read -ra FILE_COUNTS <<< "${NUM_FILES:-10}"
INTERVAL_MIN="${INTERVAL_MIN:-15}"
DURATION_HR="${DURATION_HR:-24}"
RESULTS_DIR="${RESULTS_DIR:-./collector_results}"
PING_COUNT="${PING_COUNT:-20}"
SOURCE_LABEL="${SOURCE_LABEL:-$(hostname -s)}"
INTERVAL_SEC=$(( INTERVAL_MIN * 60 ))
TOTAL_SAMPLES=$(( (DURATION_HR * 60) / INTERVAL_MIN ))

# iperf3 configuration
IPERF3_PORT="${IPERF3_PORT:-5201}"
IPERF3_DURATION="${IPERF3_DURATION:-10}"
IPERF3_RETRIES="${IPERF3_RETRIES:-3}"

# Find scripts relative to this file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILEIOTEST="$SCRIPT_DIR/fileiotest.sh"
PARSER="$SCRIPT_DIR/parse_run.py"
IPERF_BIN="$SCRIPT_DIR/iperf3"

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

# ─── Resume logic ────────────────────────────────────────────────────
# Count existing data rows (subtract 1 for header)
EXISTING_ROWS=$(( $(wc -l < "$CSV_FILE") - 1 ))
if [[ $EXISTING_ROWS -lt 0 ]]; then
    EXISTING_ROWS=0
fi

if [[ $EXISTING_ROWS -ge $TOTAL_SAMPLES ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ${SOURCE_LABEL}: Already have ${EXISTING_ROWS}/${TOTAL_SAMPLES} samples."
    echo "  Collection is already complete. Nothing to do."
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
fi

START_SAMPLE=$EXISTING_ROWS
REMAINING_SAMPLES=$(( TOTAL_SAMPLES - START_SAMPLE ))

START_TIME="$(date -Iseconds)"

if [[ $START_SAMPLE -eq 0 ]]; then
    # Fresh start — write summary
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
iperf3:             ${IPERF3_DURATION}s test to port ${IPERF3_PORT}
Results dir:        $RESULTS_DIR
Kernel:             $(uname -r)
EOF
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  fileiotest Collector (resumable) — ${SOURCE_LABEL}         "
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Source:         ${SOURCE_LABEL} → ${DEST}"
    echo "║  File counts:    ${FILE_COUNTS[*]}"
    echo "║  Interval:       every ${INTERVAL_MIN} min"
    echo "║  Duration:       ${DURATION_HR}hr ($TOTAL_SAMPLES samples)"
    echo "║  iperf3:         ${IPERF3_DURATION}s to port ${IPERF3_PORT}"
    echo "║  Results:        $CSV_FILE"
    echo "╚══════════════════════════════════════════════════════════════╝"
else
    # Resuming — append to summary
    cat >> "$SUMMARY_FILE" <<EOF

=== Resumed ===
Resumed at:         $START_TIME
Existing samples:   $START_SAMPLE
Remaining:          $REMAINING_SAMPLES
EOF
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  fileiotest Collector — RESUMING ${SOURCE_LABEL}            "
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Existing samples:  ${START_SAMPLE}/${TOTAL_SAMPLES}"
    echo "║  Remaining:         ${REMAINING_SAMPLES} samples"
    echo "║  Source:            ${SOURCE_LABEL} → ${DEST}"
    echo "║  Results:           $CSV_FILE"
    echo "╚══════════════════════════════════════════════════════════════╝"
fi
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
sample=$START_SAMPLE
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

    # ── Connectivity check ───────────────────────────────────────────
    # Quick check if destination is reachable before spending time on tests
    if ! ping -c 1 -W 3 "$DEST_HOST" >/dev/null 2>&1; then
        echo "  ⚠ ${DEST_HOST} unreachable — skipping sample (will retry next interval)"
        failures=$(( failures + 1 ))
        sample=$(( sample - 1 ))  # don't count this as a completed sample

        # Sleep until next interval
        if [[ "$RUNNING" == "true" ]]; then
            remaining=$INTERVAL_SEC
            while [[ $remaining -gt 0 ]] && [[ "$RUNNING" == "true" ]]; do
                chunk=$(( remaining > 10 ? 10 : remaining ))
                sleep "$chunk"
                remaining=$(( remaining - chunk ))
            done
        fi
        continue
    fi

    # ── Ping latency ─────────────────────────────────────────────────
    PINGLOG="$(cd "$RESULTS_DIR" && pwd)/logs/${SOURCE_LABEL}_ping_${SAFE_TS}.log"
    ping -c "$PING_COUNT" -q "$DEST_HOST" > "$PINGLOG" 2>&1 || true

    # ── iperf3 bandwidth test ────────────────────────────────────────
    IPERFLOG="$(cd "$RESULTS_DIR" && pwd)/logs/${SOURCE_LABEL}_iperf3_${SAFE_TS}.json"
    if [[ -x "$IPERF_BIN" ]] || command -v iperf3 >/dev/null 2>&1; then
        IPERF_CMD="${IPERF_BIN}"
        [[ -x "$IPERF_CMD" ]] || IPERF_CMD="iperf3"
        # Stagger iperf3 start to avoid collisions in simultaneous mode
        sleep $(( RANDOM % 45 ))

        iperf_ok=false
        for attempt in $(seq 1 "$IPERF3_RETRIES"); do
            "$IPERF_CMD" -c "$DEST_HOST" -p "$IPERF3_PORT" -t "$IPERF3_DURATION" -J \
                > "$IPERFLOG" 2>/dev/null && { iperf_ok=true; break; }
            sleep $(( RANDOM % 30 + 10 ))
        done
        if [[ "$iperf_ok" == "false" ]]; then
            echo '{}' > "$IPERFLOG"
            echo "  ⚠ iperf3 failed after $IPERF3_RETRIES attempts"
        fi
    else
        echo '{}' > "$IPERFLOG"
    fi

    # ── Run the benchmark ────────────────────────────────────────────
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
            "$SOURCE_LABEL" "$PINGLOG" "$IPERFLOG" \
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
Samples run:    $(( sample - START_SAMPLE ))
Total samples:  $sample
Failures:       $failures
CSV file:       $CSV_FILE
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ${SOURCE_LABEL}: Collection complete."
echo "  Samples: ${sample}/${TOTAL_SAMPLES}  |  Failures: ${failures}"
echo "  CSV:     $CSV_FILE"
echo "════════════════════════════════════════════════════════════════"
