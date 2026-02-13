# fileiotest — Network Performance Benchmarking

Measures and compares network throughput between lab workstations connected via a direct wire versus a managed network switch. Designed to collect a week of data across multiple machines simultaneously, then produce statistical comparisons and plots.

## Architecture

```
            ┌──────┐  Cat5e (no switch)  ┌──────┐
            │ �PC   │════════════════════ │ �PC   │
            │wire-  │   private subnet   │ dest  │
            │ src   │                    │       │
            └──────┘                     └───┬───┘
                                             │
                                        ┌────┴────┐
                                        │ Network │
                                        │ Switch  │
                                        └────┬────┘
                          ┌──────────────────┼──────────────────┐
                          │                  │                  │
                      ┌──────┐          ┌──────┐          ┌──────┐
                      │ �PC   │          │ �PC   │   ...    │ �PC   │
                      │ws_sw1│          │ws_sw2│          │ws_swN│
                      └──────┘          └──────┘          └──────┘

  ┌──────┐
  │ �PC   │  control machine
  │ctrl  │  (orchestrator.sh deploys + manages ws_sw collectors)
  └──────┘  wire-src runs its own collector independently (separate physical network)
```

**Two test paths, one destination:**

- **Direct wire** (wire-src → dest): Machine-to-machine point-to-point Ethernet on a private subnet, no switch involved. Baseline measurement.
- **Network switch** (N workstations → dest): Traffic goes through the managed network switch infrastructure. This is what we're evaluating.

## Repository Contents

| File | Description |
|---|---|
| `fileiotest.sh` | Transfer benchmark — generates random files, clears caches on both sender and receiver, measures throughput via `pv`, captures TCP retransmit stats via `nstat` |
| `randomfiles.py` | Generates random 10MB test files (called by `fileiotest.sh`) |
| `setup_wizard.py` | Interactive wizard that generates `config.toml` with your site-specific machine names, IPs, and paths |
| `load_config.py` | Reads `config.toml` for use by both Python and bash scripts |
| `config.toml.example` | Template showing the config structure (safe to commit) |
| `collector.sh` | Per-machine collector — runs `fileiotest.sh` every N minutes, adds ping latency, writes structured CSV |
| `parse_run.py` | Parses the stats file from `fileiotest.sh` + ping logs into CSV rows |
| `orchestrator.sh` | Deploys and manages collectors across all switch workstations from a control machine |
| `analyze_week.py` | Statistical analysis + 7 plots |
| `.gitignore` | Keeps `config.toml` and results out of version control |

## Prerequisites

**On all lab machines (source workstations + destination):**

- `vmtouch` (can be built from source — see `fileiotest.sh`)
- `pv` (pipe viewer)
- `nstat` (from `iproute2`, typically pre-installed)
- Python 3.9+
- Passwordless SSH access between machines (`ssh-copy-id`)
- Privileges to run `echo 3 > /proc/sys/vm/drop_caches`

**On the control machine and for analysis:**

- Python 3.9+
- `pandas`, `matplotlib`, `scipy` (for `analyze_week.py`)

```bash
pip install pandas matplotlib scipy
```

## Setup

### 1. Clone the repo

On the **control machine** (for switch workstations):

```bash
git clone <repo-url>
cd fileiotest
```

On the **wire source machine** (done separately):

```bash
git clone <repo-url>
cd fileiotest
```

### 2. Run the setup wizard

```bash
python3 setup_wizard.py
```

The wizard asks for:

- **Destination machine** hostname and SSH user
- **Wire connection** source hostname and private IPs
- **Switch workstations** (space or comma separated list)
- **Paths** (deploy directory, results directory)
- **Collection parameters** (file count, interval, duration, ping count)

It validates SSH connectivity and writes `config.toml` (which is gitignored).

Run the wizard on **both** the control machine and the wire source machine.

### 3. Verify SSH access

Make sure the configured user can SSH without a password from:

- Control machine → all switch workstations
- Control machine → destination
- Wire source → destination (via the private IP)
- Each switch workstation → destination

## Running the Collection

### Start switch workstations (from control machine)

```bash
./orchestrator.sh start
```

This will:

1. Deploy the repo to all switch workstations via SCP
2. Launch a background collector on each workstation
3. Print the exact command to run on the wire source machine

### Start the wire test (from wire source)

SSH into the wire source and run the command printed by the orchestrator. It will look something like:

```bash
cd ~/fileiotest
nohup env \
  SOURCE_LABEL='<wire-source-hostname>' \
bash ./collector.sh '<user>@<wire-dest-ip>' \
  > collector_wire.log 2>&1 &
```

Start this at roughly the same time as the switch workstations.

### Monitor progress

```bash
./orchestrator.sh status
```

Output:

```
  MACHINE       STATUS      SAMPLES     CONNECTION
  wire-src      RUNNING     288         direct-wire
  ws_sw1        RUNNING     288         ws-thru-switch
  ws_sw2        RUNNING     287         ws-thru-switch
  ...
```

### Stop early (if needed)

```bash
# Stop switch workstations from control machine:
./orchestrator.sh stop

# Stop wire source (from that machine directly):
pkill -TERM -f collector.sh
```

Collectors handle `SIGTERM` gracefully — they finish the current run before exiting.

## Collecting Results

After the collection period (default: 1 week):

```bash
./orchestrator.sh collect
```

This SCPs all CSVs from every machine and merges them:

```
all_results/
├── <wire-src>/
│   └── results_<wire-src>.csv
├── <ws_sw1>/
│   └── results_<ws_sw1>.csv
├── ...
└── all_results.csv          ← combined, ready for analysis
```

## Analysis

```bash
python3 analyze_week.py all_results/all_results.csv
```

### Output

**`analysis/significance_report.txt`** — Statistical comparison including Welch's t-test, Mann-Whitney U, and Cohen's d effect size for each metric, plus a weekday vs weekend breakdown.

**7 plots in `analysis/`:**

| Plot | What it shows |
|---|---|
| `01_timeseries_week.png` | Throughput over the full week, all machines overlaid |
| `02_hourly_profile.png` | Mean ± σ by hour of day — weekday vs weekend side by side |
| `03_day_of_week.png` | Box plots by day of week |
| `04_distributions.png` | Violin + box plots for throughput, latency, TCP retransmits |
| `05_latency_week.png` | Ping latency + jitter over the full week |
| `06_tcp_retransmits.png` | Retransmits over time + scatter vs throughput |
| `07_per_machine.png` | Horizontal bar chart ranking all machines |

### Custom options

```bash
# Different output directory:
python3 analyze_week.py all_results.csv --output-dir ./report_figures

# Override the direct-wire machine name:
python3 analyze_week.py all_results.csv --direct-wire <wire-source-hostname>
```

## Configuration Reference

`config.toml` is generated by the wizard and **never committed** to version control.
See `config.toml.example` for the structure.

Environment variables can override collection parameters at runtime:

```bash
NUM_FILES="5,10,20" INTERVAL_MIN=30 ./collector.sh <user>@<dest>
```

## CSV Schema

Each row represents one sample from one machine (27 columns):

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 timestamp of the sample |
| `source` | Source machine hostname |
| `num_files` | Number of 10MB files transferred |
| `ping_min_ms` | Minimum ping RTT (ms) |
| `ping_avg_ms` | Average ping RTT (ms) |
| `ping_max_ms` | Maximum ping RTT (ms) |
| `ping_mdev_ms` | Ping jitter / std deviation (ms) |
| `ping_loss_pct` | Packet loss percentage |
| `cold_cache_bytes` | Bytes transferred (cold cache) |
| `cold_cache_rate` | Transfer rate (cold cache, from `pv`) |
| `cold_cache_retrans` | TCP retransmit segments (cold cache phase) |
| `cold_cache_timeouts` | TCP timeouts (cold cache phase) |
| `cold_cache_outrsts` | TCP resets (cold cache phase) |
| `hot_cache_run{1,2,3}_bytes` | Bytes transferred per hot-cache run |
| `hot_cache_run{1,2,3}_rate` | Transfer rate per hot-cache run |
| `hot_cache_retrans` | TCP retransmits (combined 3 hot-cache runs) |
| `hot_cache_timeouts` | TCP timeouts (combined 3 hot-cache runs) |
| `hot_cache_outrsts` | TCP resets (combined 3 hot-cache runs) |
| `true_write_bytes` | Bytes transferred (true write to disk) |
| `true_write_rate` | Transfer rate (true write) |
| `true_write_retrans` | TCP retransmits (true write phase) |
| `true_write_timeouts` | TCP timeouts (true write phase) |
| `true_write_outrsts` | TCP resets (true write phase) |

## Quick Reference

```bash
# First time setup (on control machine and wire source):
python3 setup_wizard.py

# Start the week-long collection:
./orchestrator.sh start          # switch workstations (from control machine)
# then start wire source separately (command printed by orchestrator)

# Check on things:
./orchestrator.sh status

# After the week:
./orchestrator.sh collect
python3 analyze_week.py all_results/all_results.csv
```
