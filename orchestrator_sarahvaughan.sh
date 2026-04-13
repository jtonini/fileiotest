#!/usr/bin/env bash
# orchestrator_sarahvaughan.sh — Deploy and run fileiotest collectors
#                                 targeting sarahvaughan (NFS server)
#
# Runs from jonimitchell as zeus. Deploys collector to each workstation
# and launches tests in two modes:
#   1. Sequential — one machine at a time (clean baseline)
#   2. Simultaneous — all machines at once (contention test)
#
# Each machine gets campus test (→ 141.166.222.28).
# josh, mayer, justin also get intranet test (→ 192.168.0.6).
#
# The collector is resumable: if a test is interrupted (e.g., cable swap
# on sarahvaughan), just re-run this script and it picks up where it
# left off.
#
# Usage:
#   ./orchestrator_sarahvaughan.sh deploy          # push files to all machines
#   ./orchestrator_sarahvaughan.sh sequential      # run 24hr per machine, one at a time
#   ./orchestrator_sarahvaughan.sh simultaneous    # run all machines at once for 24hr
#   ./orchestrator_sarahvaughan.sh status          # check progress
#   ./orchestrator_sarahvaughan.sh stop            # stop all collectors
#   ./orchestrator_sarahvaughan.sh collect         # gather all CSVs to jonimitchell
#   ./orchestrator_sarahvaughan.sh iperf-start     # start iperf3 daemon on sarahvaughan
#   ./orchestrator_sarahvaughan.sh iperf-check     # check iperf3 on sarahvaughan

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────

SARAHVAUGHAN_CAMPUS="141.166.222.28"
SARAHVAUGHAN_INTRANET="192.168.0.6"

# Campus switch machines → sarahvaughan campus IP
CAMPUS_MACHINES=(
    "thais"
    "cooper"
    "hamilton"
    "camryn"
    "irene2"
    "aamy"
    "alexis"
    "boyi"
    "josh"
    "justin"
    "khanh"
)

# Intranet machines → sarahvaughan intranet IP
INTRANET_MACHINES=(
    "josh"
    "mayer"
    "justin"
)

DEPLOY_DIR="~/fileiotest"
DEPLOY_USER="root"
NUM_FILES="100"
INTERVAL_MIN="15"
DURATION_HR="24"
PING_COUNT="20"

# NIC speeds (Mbps) — for reference in status output
declare -A NIC_SPEEDS=(
    [thais]=5000 [cooper]=5000 [hamilton]=5000
    [camryn]=2500 [irene2]=2500
    [aamy]=1000 [alexis]=1000 [boyi]=1000
    [josh]=1000 [justin]=1000 [khanh]=1000
    [josh_i]=1000 [mayer_i]=1000 [justin_i]=1000
    [sarahvaughan_campus]=1000 [sarahvaughan_intranet]=1000
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_DIR="$SCRIPT_DIR/all_results_sarahvaughan"

# Files to deploy
DEPLOY_FILES=(
    "fileiotest.sh"
    "randomfiles.py"
    "collector_resumable.sh"
    "parse_run.py"
    "load_config.py"
    "launch_collector_sv.sh"
)

# ─── Helpers ──────────────────────────────────────────────────────────
info()  { echo "  [INFO]  $*"; }
ok()    { echo "  [  OK]  $*"; }
warn()  { echo "  [WARN]  $*"; }
fail()  { echo "  [FAIL]  $*"; }

ssh_cmd() {
    local host="$1"; shift
    timeout 15 ssh -o ConnectTimeout=5 -o BatchMode=yes "${DEPLOY_USER}@${host}" "$@" 2>/dev/null
}

label_campus()  { echo "${1}_campus"; }
label_intranet(){ echo "${1}_intranet"; }

# ─── Deploy ───────────────────────────────────────────────────────────
do_deploy() {
    # Collect unique machines
    declare -A all_machines
    for h in "${CAMPUS_MACHINES[@]}"; do all_machines[$h]=1; done
    for h in "${INTRANET_MACHINES[@]}"; do all_machines[$h]=1; done

    echo "Deploying fileiotest to ${#all_machines[@]} machines..."
    echo ""

    for host in $(echo "${!all_machines[@]}" | tr ' ' '\n' | sort); do
        printf "  %-12s " "$host"
        if ssh_cmd "$host" "mkdir -p ${DEPLOY_DIR}"; then
            # Copy files
            scp_ok=true
            for f in "${DEPLOY_FILES[@]}"; do
                src="$SCRIPT_DIR/$f"
                if [[ -f "$src" ]]; then
                    scp -q -o BatchMode=yes "$src" \
                        "${DEPLOY_USER}@${host}:${DEPLOY_DIR}/" 2>/dev/null || { scp_ok=false; break; }
                else
                    warn "missing $f locally"
                    scp_ok=false
                    break
                fi
            done

            if [[ "$scp_ok" == "true" ]]; then
                ssh_cmd "$host" "chmod +x ${DEPLOY_DIR}/{fileiotest.sh,collector_resumable.sh,parse_run.py,launch_collector_sv.sh}" \
                    && ssh_cmd "$host" "ln -sf ${DEPLOY_DIR}/collector_resumable.sh ${DEPLOY_DIR}/collector.sh 2>/dev/null; \
                                        test -L ${DEPLOY_DIR}/iperf3 || ln -sf /usr/bin/iperf3 ${DEPLOY_DIR}/iperf3 2>/dev/null; true" \
                    && ok "deployed" \
                    || fail "chmod failed"
            else
                fail "scp failed"
            fi
        else
            fail "ssh failed"
        fi
    done
    echo ""
}

# ─── Launch a single collector on a remote machine ────────────────────
launch_one() {
    local host="$1"
    local dest="$2"
    local label="$3"
    local duration="${4:-$DURATION_HR}"

    # Check if already running
    if ssh_cmd "$host" "pgrep -f 'collector_resumable.sh.*${dest}'" >/dev/null 2>&1; then
        warn "${label} already running on ${host}"
        return 0
    fi

    ssh_cmd "$host" \
        "cd ${DEPLOY_DIR} && \
         export SOURCE_LABEL='${label}' NUM_FILES='${NUM_FILES}' \
                INTERVAL_MIN='${INTERVAL_MIN}' DURATION_HR='${duration}' \
                PING_COUNT='${PING_COUNT}' RESULTS_DIR='./collector_results' \
                PATH='${DEPLOY_DIR}:/usr/local/bin:/usr/bin:/usr/sbin:\$PATH' && \
         nohup bash ./collector_resumable.sh '${DEPLOY_USER}@${dest}' \
             > collector_${label}.log 2>&1 </dev/null &
        echo \$!"
    sleep 1
    if ssh_cmd "$host" "pgrep -f 'collector_resumable.sh.*${label}'" >/dev/null 2>&1; then
        ok "${label} started on ${host}"
    else
        fail "${label} launch failed on ${host}"
    fi
}

# ─── Wait for a collector to finish ──────────────────────────────────
wait_for_collector() {
    local host="$1"
    local label="$2"
    local max_wait=$(( DURATION_HR * 3600 + 1800 ))  # duration + 30min buffer
    local elapsed=0
    local check_interval=300  # check every 5 minutes

    while [[ $elapsed -lt $max_wait ]]; do
        if ! ssh_cmd "$host" "pgrep -f 'collector_resumable.sh.*${label}'" >/dev/null 2>&1; then
            ok "${label} on ${host} completed"
            return 0
        fi

        # Show progress
        local samples
        samples=$(ssh_cmd "$host" "wc -l < ${DEPLOY_DIR}/collector_results/results_${label}.csv 2>/dev/null" 2>/dev/null || echo "1")
        samples=$(( samples - 1 ))
        local total=$(( (DURATION_HR * 60) / INTERVAL_MIN ))
        echo "  [$(date +%H:%M)] ${label}: ${samples}/${total} samples"

        sleep "$check_interval"
        elapsed=$(( elapsed + check_interval ))
    done

    warn "${label} did not finish within expected time"
    return 1
}

# ─── Sequential ───────────────────────────────────────────────────────
do_sequential() {
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SEQUENTIAL MODE: One machine at a time, ${DURATION_HR}hr each"
    echo "  Campus: ${#CAMPUS_MACHINES[@]} machines → ${SARAHVAUGHAN_CAMPUS}"
    echo "  Intranet: ${#INTRANET_MACHINES[@]} machines → ${SARAHVAUGHAN_INTRANET}"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # Ensure deployed
    do_deploy

    # Campus tests — one at a time
    echo "── Campus switch tests ──────────────────────────────────────"
    for host in "${CAMPUS_MACHINES[@]}"; do
        local label
        label=$(label_campus "$host")
        local nic_speed="${NIC_SPEEDS[$host]:-?}"

        echo ""
        echo ">>> Starting ${label} (NIC: ${nic_speed}Mbps → SV campus: ${NIC_SPEEDS[sarahvaughan_campus]}Mbps)"
        launch_one "$host" "$SARAHVAUGHAN_CAMPUS" "$label"
        wait_for_collector "$host" "$label"
    done

    # Intranet tests — one at a time
    echo ""
    echo "── Intranet tests ───────────────────────────────────────────"
    for host in "${INTRANET_MACHINES[@]}"; do
        local label
        label=$(label_intranet "$host")
        local nic_speed="${NIC_SPEEDS[${host}_i]:-?}"

        echo ""
        echo ">>> Starting ${label} (NIC: ${nic_speed}Mbps → SV intranet: ${NIC_SPEEDS[sarahvaughan_intranet]}Mbps)"
        launch_one "$host" "$SARAHVAUGHAN_INTRANET" "$label"
        wait_for_collector "$host" "$label"
    done

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Sequential tests complete!"
    echo "  Run '$0 collect' to gather results."
    echo "═══════════════════════════════════════════════════════════════"
}

# ─── Simultaneous ─────────────────────────────────────────────────────
do_simultaneous() {
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SIMULTANEOUS MODE: All machines at once, ${DURATION_HR}hr"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # These need different labels to avoid overwriting sequential results
    # Use _campus_simul and _intranet_simul suffixes

    # Ensure deployed
    do_deploy

    echo "Launching all campus collectors..."
    for host in "${CAMPUS_MACHINES[@]}"; do
        local label="${host}_campus_simul"
        printf "  %-12s " "$host"
        launch_one "$host" "$SARAHVAUGHAN_CAMPUS" "$label"
    done

    echo ""
    echo "Launching all intranet collectors..."
    for host in "${INTRANET_MACHINES[@]}"; do
        local label="${host}_intranet_simul"
        printf "  %-12s " "$host"
        launch_one "$host" "$SARAHVAUGHAN_INTRANET" "$label"
    done

    TOTAL_SAMPLES=$(( (DURATION_HR * 60) / INTERVAL_MIN ))
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  All collectors launched. ${TOTAL_SAMPLES} samples over ${DURATION_HR}hr."
    echo "  Use '$0 status' to monitor progress."
    echo "  Use '$0 stop' to stop early."
    echo "  Use '$0 collect' to gather results when done."
    echo "═══════════════════════════════════════════════════════════════"
}

# ─── Status ───────────────────────────────────────────────────────────
do_status() {
    echo "Collector status (→ sarahvaughan):"
    echo ""
    printf "  %-12s  %-8s  %-22s  %-10s  %-8s  %s\n" \
        "MACHINE" "NIC" "LABEL" "STATUS" "SAMPLES" "PATH"

    local total_samples=$(( (DURATION_HR * 60) / INTERVAL_MIN ))

    # Campus
    for host in "${CAMPUS_MACHINES[@]}"; do
        for suffix in "campus" "campus_simul"; do
            local label="${host}_${suffix}"
            local nic_speed="${NIC_SPEEDS[$host]:-?}"

            # Check if CSV exists
            local csv_exists
            csv_exists=$(ssh_cmd "$host" "test -f ${DEPLOY_DIR}/collector_results/results_${label}.csv && echo yes || echo no" 2>/dev/null || echo "no")

            if [[ "$csv_exists" == "yes" ]]; then
                local status samples
                if ssh_cmd "$host" "pgrep -f 'collector_resumable.*${label}'" >/dev/null 2>&1; then
                    status="RUNNING"
                else
                    status="stopped"
                fi
                samples=$(ssh_cmd "$host" "wc -l < ${DEPLOY_DIR}/collector_results/results_${label}.csv" 2>/dev/null || echo "1")
                samples=$(( samples - 1 ))
                printf "  %-12s  %-8s  %-22s  %-10s  %3d/%-4d  %s\n" \
                    "$host" "${nic_speed}M" "$label" "$status" "$samples" "$total_samples" "campus"
            fi
        done
    done

    # Intranet
    for host in "${INTRANET_MACHINES[@]}"; do
        for suffix in "intranet" "intranet_simul"; do
            local label="${host}_${suffix}"
            local nic_speed="${NIC_SPEEDS[${host}_i]:-?}"

            local csv_exists
            csv_exists=$(ssh_cmd "$host" "test -f ${DEPLOY_DIR}/collector_results/results_${label}.csv && echo yes || echo no" 2>/dev/null || echo "no")

            if [[ "$csv_exists" == "yes" ]]; then
                local status samples
                if ssh_cmd "$host" "pgrep -f 'collector_resumable.*${label}'" >/dev/null 2>&1; then
                    status="RUNNING"
                else
                    status="stopped"
                fi
                samples=$(ssh_cmd "$host" "wc -l < ${DEPLOY_DIR}/collector_results/results_${label}.csv" 2>/dev/null || echo "1")
                samples=$(( samples - 1 ))
                printf "  %-12s  %-8s  %-22s  %-10s  %3d/%-4d  %s\n" \
                    "$host" "${nic_speed}M" "$label" "$status" "$samples" "$total_samples" "intranet"
            fi
        done
    done
    echo ""
}

# ─── Stop ─────────────────────────────────────────────────────────────
do_stop() {
    echo "Stopping all sarahvaughan collectors..."
    echo ""

    declare -A all_machines
    for h in "${CAMPUS_MACHINES[@]}"; do all_machines[$h]=1; done
    for h in "${INTRANET_MACHINES[@]}"; do all_machines[$h]=1; done

    for host in $(echo "${!all_machines[@]}" | tr ' ' '\n' | sort); do
        printf "  %-12s " "$host"
        if ssh_cmd "$host" "pkill -TERM -f 'collector_resumable.sh'" 2>/dev/null; then
            ok "stop signal sent"
        else
            info "no collector running"
        fi
    done
    echo ""
}

# ─── Collect ──────────────────────────────────────────────────────────
do_collect() {
    echo "Collecting results → ${COLLECT_DIR}/"
    echo ""

    mkdir -p "$COLLECT_DIR"

    declare -A all_machines
    for h in "${CAMPUS_MACHINES[@]}"; do all_machines[$h]=1; done
    for h in "${INTRANET_MACHINES[@]}"; do all_machines[$h]=1; done

    for host in $(echo "${!all_machines[@]}" | tr ' ' '\n' | sort); do
        printf "  %-12s " "$host"
        local host_dir="$COLLECT_DIR/$host"
        mkdir -p "$host_dir"

        # Grab all CSV and summary files for this host
        scp -q -o BatchMode=yes \
            "${DEPLOY_USER}@${host}:${DEPLOY_DIR}/collector_results/results_${host}_*.csv" \
            "$host_dir/" 2>/dev/null \
        && scp -q -o BatchMode=yes \
            "${DEPLOY_USER}@${host}:${DEPLOY_DIR}/collector_results/collection_summary_${host}_*.txt" \
            "$host_dir/" 2>/dev/null \
        && ok "collected" \
        || warn "no results or scp failed"
    done

    # Merge all CSVs
    echo ""
    echo "Merging CSVs..."

    for mode in campus campus_simul intranet intranet_simul; do
        COMBINED="$COLLECT_DIR/all_${mode}.csv"
        head_written=false

        for host_dir in "$COLLECT_DIR"/*/; do
            for csv in "$host_dir"/results_*_${mode}.csv; do
                [[ -f "$csv" ]] || continue
                if [[ "$head_written" == "false" ]]; then
                    cat "$csv" > "$COMBINED"
                    head_written=true
                else
                    tail -n +2 "$csv" >> "$COMBINED"
                fi
            done
        done

        if [[ -f "$COMBINED" ]]; then
            local total_rows=$(( $(wc -l < "$COMBINED") - 1 ))
            ok "${mode}: $COMBINED ($total_rows samples)"
        fi
    done

    echo ""
    echo "Results ready for analysis."
}

# ─── iperf3 management on sarahvaughan ────────────────────────────────
do_iperf_start() {
    echo "Starting iperf3 daemon on sarahvaughan..."
    ssh -o ConnectTimeout=5 root@sarahvaughan \
        "pgrep iperf3 >/dev/null 2>&1 && echo 'Already running' || (nohup iperf3 -s -D && echo 'Started')"
}

do_iperf_check() {
    echo "iperf3 on sarahvaughan:"
    ssh -o ConnectTimeout=5 root@sarahvaughan \
        "sockstat -l | grep 5201 || echo 'NOT RUNNING'"
}

# ─── Main ─────────────────────────────────────────────────────────────
case "${1:-help}" in
    deploy)       do_deploy ;;
    sequential)   do_sequential ;;
    simultaneous) do_simultaneous ;;
    status)       do_status ;;
    stop)         do_stop ;;
    collect)      do_collect ;;
    iperf-start)  do_iperf_start ;;
    iperf-check)  do_iperf_check ;;
    *)
        echo "Usage: $0 {deploy|sequential|simultaneous|status|stop|collect|iperf-start|iperf-check}"
        echo ""
        echo "Commands:"
        echo "  deploy        Push collector to all workstations"
        echo "  sequential    Run 24hr per machine, one at a time (baseline)"
        echo "  simultaneous  Run all machines at once for 24hr (contention)"
        echo "  status        Show running/stopped status and sample counts"
        echo "  stop          Gracefully stop all collectors"
        echo "  collect       Gather all CSVs to jonimitchell and merge"
        echo "  iperf-start   Start iperf3 daemon on sarahvaughan"
        echo "  iperf-check   Check iperf3 status on sarahvaughan"
        echo ""
        echo "Machines:"
        echo "  Campus (→ ${SARAHVAUGHAN_CAMPUS}):   ${CAMPUS_MACHINES[*]}"
        echo "  Intranet (→ ${SARAHVAUGHAN_INTRANET}): ${INTRANET_MACHINES[*]}"
        echo ""
        echo "Test plan:"
        echo "  1. $0 deploy"
        echo "  2. $0 sequential    # ~14 days for all machines"
        echo "  3. $0 simultaneous  # 24hr contention test"
        echo "  4. $0 collect       # merge all results"
        exit 1
        ;;
esac
