# fileiotest -- Network Performance Benchmarking

Measures and compares network throughput between lab workstations connected via a direct wire versus a managed network switch. Designed to collect a week of data across multiple machines simultaneously, then produce statistical comparisons and plots.

## Architecture

```
            +------+  Cat5e (no switch)  +------+
            |  WS  |====================|  WS  |
            |wire- |   private subnet   | dest |
            | src  |                    |      |
            +------+                    +--+---+
                                           |
                                      +----+----+
                                      | Network |
                                      | Switch  |
                                      +----+----+
                        +------------------+------------------+
                        |                  |                  |
                    +------+          +------+          +------+
                    |  WS  |          |  WS  |   ...    |  WS  |
                    |ws_sw1|          |ws_sw2|          |ws_swN|
                    +------+          +------+          +------+

  +------+
  |  WS  |  control machine
  | ctrl |  (orchestrator.sh deploys + manages ws_sw collectors)
  +------+  wire-src runs its own collector independently (separate physical network)
```

**Two test paths, one destination:**

- **Direct wire** (wire-src -> dest): Machine-to-machine point-to-point Ethernet on a private subnet, no switch involved. Baseline measurement.
- **Network switch** (N workstations -> dest): Traffic goes through the managed network switch infrastructure. This is what we're evaluating.

## How the Test Works

Each run of `fileiotest.sh` measures the transfer speed of a known payload from one
workstation to another under three distinct caching conditions. Understanding what
happens at each step is important because caching at any layer (filesystem, kernel,
or disk) can inflate throughput numbers and mask the true network performance.

### Step by step

**1. Dependency check.** The script verifies that `vmtouch` and `pv` are available.
If `vmtouch` is missing, it clones the repository and builds it from source. If `pv`
is missing, it installs it via the system package manager. Both tools are essential:
`vmtouch` gives precise control over what lives in the kernel page cache, and `pv`
monitors the data stream to report transfer rates.

**2. Generate random test files.** `randomfiles.py` creates N files of 10 MB each,
filled with random printable characters. Using random data is deliberate -- it
prevents compression from skewing throughput numbers, since SSH can negotiate
compression on the transport layer. The files are written to the current directory
with a `.iotest` suffix.

**3. Lock files in sender memory.** `vmtouch -t` forces all `.iotest` files into
the sender's page cache. Since the files were just written, they are likely already
resident, but this step guarantees it. A second `vmtouch` call (without `-t`) can
be uncommented to verify residency if needed.

**4. Snapshot initial TCP counters.** Before any transfer begins, the script records
`TcpRetransSegs`, `TcpOutRsts`, and `TcpExtTCPTimeouts` from `nstat`. These
counters are cumulative, so later readings are compared against this baseline to
compute deltas per phase.

**5. Cold-cache run.** The sender flushes its page cache
(`echo 3 > /proc/sys/vm/drop_caches`), and the receiver does the same via SSH.
With both caches empty, the data must be read from disk on the sender and written
through the network -- no shortcuts. The files are streamed through `pv` into
`ssh -T` to the destination. `pv` reports the transfer rate. TCP counters are
captured again after this phase.

**6. Three hot-cache runs.** The files are loaded back into the sender's page cache
with `vmtouch -t`. This time, the sender reads from memory rather than disk,
isolating network performance from disk I/O. The transfer repeats three times with a
60-second sleep between runs to capture variability over time. TCP counters are
captured after all three runs complete.

**7. True-write run.** Caches are flushed again on both ends, but this time the
destination writes the incoming data to disk instead of discarding it (`/dev/null`).
This captures the end-to-end cost including the receiver's disk I/O, which matters
for real workloads like copying datasets between machines. TCP counters are captured
one final time.

**8. Cleanup.** The test files are removed from both the sender and the receiver.

### Why three conditions matter

The cold-cache, hot-cache, and true-write phases isolate different bottlenecks:

- **Cold-cache** measures what happens when both sender and receiver start from scratch -- disk read on the sender, network transfer, discard on the receiver. This is the most conservative throughput number.
- **Hot-cache** removes the sender's disk from the equation. If hot-cache rates are significantly higher than cold-cache, disk I/O on the sender is a bottleneck.
- **True-write** adds the receiver's disk back in. If true-write rates are lower than cold-cache, the receiver's storage is slower than the sender's. Comparing all three reveals where the actual constraint lives.

### What the output looks like

A single run produces a stats file like this:

```
Fri Feb 13 11:04:06 AM EST 2026
cazuza@badenpowell to zeus@jonimitchell
Initial stats
TcpRetransSegs                  1503               0.0
TcpOutRsts                      457799             0.0
TcpExtTCPTimeouts               705                0.0
=== COLD-CACHE RUN ===
bytes=28.6MiB rate=[31.2MiB/s]
TcpRetransSegs                  1503               0.0
TcpOutRsts                      457799             0.0
TcpExtTCPTimeouts               705                0.0
=== 3 HOT-CACHE RUNS ===
bytes=28.6MiB rate=[31.6MiB/s]
bytes=28.6MiB rate=[31.5MiB/s]
bytes=28.6MiB rate=[31.9MiB/s]
TcpRetransSegs                  1504               0.0
TcpOutRsts                      457843             0.0
TcpExtTCPTimeouts               705                0.0
=== TRUE WRITE ===
bytes=28.6MiB rate=[32.1MiB/s]
TcpRetransSegs                  1504               0.0
TcpOutRsts                      457849             0.0
TcpExtTCPTimeouts               705                0.0
```

The `collector.sh` wrapper runs this repeatedly on a schedule, adds ping latency
measurements, and feeds everything into `parse_run.py` to produce structured CSV
rows for analysis.

## Repository Contents

| File | Description |
|---|---|
| `fileiotest.sh` | Transfer benchmark -- generates random files, clears caches on both sender and receiver, measures throughput via `pv`, captures TCP retransmit stats via `nstat` |
| `randomfiles.py` | Generates random 10MB test files (called by `fileiotest.sh`) |
| `setup_wizard.py` | Interactive wizard that generates `config.toml` with your site-specific machine names, IPs, and paths |
| `load_config.py` | Reads `config.toml` for use by both Python and bash scripts |
| `config.toml.example` | Template showing the config structure (safe to commit) |
| `collector.sh` | Per-machine collector -- runs `fileiotest.sh` every N minutes, adds ping latency, writes structured CSV |
| `parse_run.py` | Parses the stats file from `fileiotest.sh` + ping logs into CSV rows |
| `orchestrator.sh` | Deploys and manages collectors across all switch workstations from a control machine |
| `analyze_week.py` | Statistical analysis + 11 plots |
| `.gitignore` | Keeps `config.toml` and results out of version control |

## Prerequisites

**On all lab machines (source workstations + destination):**

- `vmtouch` (can be built from source -- see `fileiotest.sh`)
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

## Step-by-Step Guide

### Step 1: Clone the repo

On the **control machine** (for switch workstations):

```bash
git clone <repo-url>
cd fileiotest
```

On the **wire source machine** (done separately since it is on a different physical network):

```bash
git clone <repo-url>
cd fileiotest
```

### Step 2: Run the setup wizard

On **both** the control machine and the wire source machine:

```bash
python3 setup_wizard.py
```

The wizard asks for:

- **Destination machine** hostname and SSH user
- **Wire connection** source hostname and private IPs (e.g., 10.0.0.x)
- **Switch workstations** (space or comma separated list)
- **Paths** (deploy directory, results directory)
- **Collection parameters** (file count, interval, duration, ping count)

It validates SSH connectivity and writes `config.toml` (which is gitignored and never committed).

### Step 3: Verify SSH access

Make sure the configured user can SSH without a password from:

- Control machine -> all switch workstations
- Control machine -> destination
- Wire source -> destination (via the private IP)
- Each switch workstation -> destination

```bash
# Example: test from control machine
ssh <user>@<switch-ws-1> hostname
ssh <user>@<dest> hostname
```

### Step 4: Quick sanity test (optional but recommended)

Before launching the full week-long collection, verify the pipeline works end-to-end from one machine:

```bash
# One-shot test of the benchmark itself
./fileiotest.sh 1 <user>@<dest> /tmp/test_stats.txt
cat /tmp/test_stats.txt

# Quick collector test (2 samples, 1 min apart)
NUM_FILES="1" INTERVAL_MIN=1 DURATION_HR=1 \
  SOURCE_LABEL="test" \
  bash ./collector.sh '<user>@<dest>'

# Check the output
cat ./collector_results/results_test.csv

# Test the analysis pipeline
python3 analyze_week.py ./collector_results/results_test.csv --direct-wire none
ls ./analysis/
```

If all of that works, you are ready for the real run.

### Step 5: Start the switch workstations (from control machine)

```bash
./orchestrator.sh start
```

This will:

1. Deploy the repo to all switch workstations via SCP
2. Launch a background collector on each workstation
3. Print the exact command to run on the wire source machine

### Step 6: Start the wire test (from wire source)

SSH into the wire source and run the command printed by the orchestrator. It will look something like:

```bash
cd ~/fileiotest
nohup env \
  SOURCE_LABEL='<wire-source-hostname>' \
bash ./collector.sh '<user>@<wire-dest-ip>' \
  > collector_wire.log 2>&1 &
```

Start this at roughly the same time as the switch workstations.

### Step 7: Monitor progress

From the control machine:

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

### Step 8: Stop early (if needed)

```bash
# Stop switch workstations from control machine:
./orchestrator.sh stop

# Stop wire source (from that machine directly):
pkill -TERM -f collector.sh
```

Collectors handle `SIGTERM` gracefully -- they finish the current run before exiting.

### Step 9: Collect results

After the collection period (default: 1 week):

```bash
./orchestrator.sh collect
```

This SCPs all CSVs from every machine and merges them:

```
all_results/
  <wire-src>/
    results_<wire-src>.csv
  <ws_sw1>/
    results_<ws_sw1>.csv
  ...
  all_results.csv          <-- combined, ready for analysis
```

### Step 10: Run the analysis

```bash
python3 analyze_week.py all_results/all_results.csv
```

Or with custom options:

```bash
# Different output directory
python3 analyze_week.py all_results.csv --output-dir ./report_figures

# Override the direct-wire machine name
python3 analyze_week.py all_results.csv --direct-wire <wire-source-hostname>
```

All outputs land in `./analysis/` (or your custom directory).

## Understanding the Output

### Significance Report (`significance_report.txt`)

The text report provides statistical comparisons between the direct wire and university switch for every metric. For each metric you will see:

- **Mean and std**: Average and standard deviation for each connection type. A large difference in means suggests a real performance gap.
- **Welch's t-test**: Tests whether the means are significantly different. Look at the p-value -- p < 0.05 means the difference is statistically significant, p < 0.001 means it is highly significant. Stars indicate significance level (* p<0.05, ** p<0.01, *** p<0.001).
- **Mann-Whitney U**: A non-parametric alternative that does not assume normal distribution. Useful for confirming the t-test result, especially if data is skewed.
- **Cohen's d**: Effect size, independent of sample size. Tells you how big the difference actually is: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), or large (>0.8). For your use case, expect large effects.
- **Weekday vs Weekend**: Tests whether the switch performance differs between business days and weekends. A significant result here points to campus network congestion as the root cause.

### Plot Interpretation Guide

**11 plots in `analysis/`:**

| Plot | What it shows |
|---|---|
| `01_timeseries_week.png` | Throughput over the full week, all machines overlaid |
| `02_hourly_profile.png` | Mean +/- std by hour of day, weekday vs weekend side by side |
| `03_day_of_week.png` | Box plots by day of week |
| `04_distributions.png` | Violin + box plots for throughput, latency, TCP retransmits |
| `05_latency_week.png` | Ping latency + jitter over the full week |
| `06_tcp_retransmits.png` | Retransmits over time + scatter vs throughput |
| `07_per_machine.png` | Horizontal bar chart ranking all machines |
| `08_phase_decomposition.png` | Bottleneck breakdown: true write baseline, disk write cost, sender disk cost |
| `09_hot_cache_spread.png` | Hot-cache 3-run spread with 95% and 99% confidence intervals |
| `10_heatmap.png` | Hour x day-of-week throughput heatmap (green=fast, red=slow) |
| `11_cdf.png` | Cumulative distribution function with 50th/95th/99th percentile markers |

#### Plot 01 -- Full Week Time Series

Shows throughput for every machine over the entire collection period. Each phase (cold cache, hot cache, true write) gets its own subplot. The direct wire appears as a blue line, switch machines as red. Look for: daily throughput dips during business hours on the switch, and stable performance on the direct wire regardless of time.

#### Plot 02 -- Hourly Profile

Averages all samples by hour of day, split into weekday and weekend panels. Each phase gets its own row. The shaded band is one standard deviation. Look for: a dip in the switch curve during 9am-5pm on weekdays that recovers on weekends. If the direct wire stays flat while the switch dips, the cause is campus network congestion, not the machines themselves.

#### Plot 03 -- Day of Week

Box plots for each day of the week, showing the spread and median for each connection type. Look for: Monday-Friday boxes that are lower and wider (more variable) than Saturday-Sunday boxes for the switch machines.

#### Plot 04 -- Distributions

Violin plots overlaid with box plots for throughput (all three phases), ping latency, and TCP retransmits. Look for: clear separation between the direct wire and switch distributions. The switch violin should be wider (more variable) and shifted toward worse values.

#### Plot 05 -- Latency Time Series

Ping latency (average and jitter/mdev) over the full week. Look for: latency spikes on the switch path that correlate with throughput dips in Plot 01. The direct wire should show consistently low, flat latency.

#### Plot 06 -- TCP Retransmits

Left panel: retransmits over time. Right panel: scatter plot of retransmits vs throughput. Look for: the scatter showing a negative correlation (more retransmits = lower throughput) on the switch path. This is the smoking gun for copper/infrastructure degradation -- packet loss causes retransmits which throttle TCP throughput.

#### Plot 07 -- Per-Machine Summary

Horizontal bar chart ranking all machines by mean throughput across all three phases. Look for: the direct wire machine at the top, with all switch machines clustered far below it.

#### Plot 08 -- Phase Decomposition

Breaks down where time is lost at each step. The green bar is the true write throughput (network + receiver disk). The orange bar stacked on top is the receiver disk write cost (the difference between hot cache and true write). Red triangles mark cold cache rates, showing the sender disk read penalty. The right panel compares phases side by side by connection type with annotated deltas. Look for: how much throughput the switch loses compared to the direct wire at each phase, and whether the bottleneck is network or disk.

#### Plot 09 -- Hot Cache Spread and Confidence Intervals

Each sample measures hot cache throughput 3 times. This plot shows the raw spread (faint dots for individual runs, shaded band for min-to-max range per sample) and confidence intervals that narrow as samples accumulate. Look for: narrow per-sample spread (the 3 runs are consistent, meaning measurements are reliable) and tight confidence intervals (you have enough data to be sure of the result). If the switch shows wider per-sample spread than the direct wire, that indicates network instability.

#### Plot 10 -- Heatmap

Hour-of-day (rows) by day-of-week (columns) grid, color-coded from green (fast) to red (slow). One grid per connection type per phase. This is the most immediately visual plot. Look for: a red block in the 9am-5pm weekday area on the switch heatmap, while the direct wire heatmap stays uniformly green. This tells the business-hours congestion story at a glance.

#### Plot 11 -- CDF (Cumulative Distribution Function)

Standard network benchmarking plot. The x-axis is throughput, the y-axis is cumulative probability (0 to 1). Read it as: "what fraction of all transfers achieved at most this throughput?" One curve per connection type, one panel per phase. Markers at the 50th, 95th, and 99th percentiles. Look for: the direct wire curve far to the right (high throughput) and steep (consistent). The switch curve far to the left (lower throughput) and more gradual (more variable). The horizontal distance between the two curves at any percentile is the performance penalty. Key statement for proposals: "At the 95th percentile, direct wire delivers X Mbit/s while the switch delivers Y Mbit/s."

### What the Three Phases Mean

Each sample runs three types of transfers to isolate different bottlenecks:

- **Cold cache**: Files are evicted from memory on both sender and receiver before transfer. The sender reads from disk, transfers over the network, and the receiver discards the data. This is the realistic worst case -- what a user experiences when transferring files not already in RAM.
- **Hot cache**: Files are pre-loaded into the sender's RAM. No disk reads -- the sender streams directly from memory to the network. The receiver discards the data. This isolates the network as the sole bottleneck. Run 3 times with 60-second pauses for consistency. If hot cache shows degradation through the switch, it is 100% network.
- **True write**: Like hot cache on the sender, but the receiver writes the data to an actual file instead of discarding it. This adds the receiver's disk write speed as a factor.

The differences between phases decompose the bottlenecks: hot cache minus true write is the receiver disk write penalty, hot cache minus cold cache is the sender disk read penalty.

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

# Quick sanity test:
./fileiotest.sh 1 <user>@<dest> /tmp/test_stats.txt
NUM_FILES="1" INTERVAL_MIN=1 DURATION_HR=1 SOURCE_LABEL="test" bash ./collector.sh '<user>@<dest>'
python3 analyze_week.py ./collector_results/results_test.csv --direct-wire none

# Start the week-long collection:
./orchestrator.sh start          # switch workstations (from control machine)
# then start wire source separately (command printed by orchestrator)

# Check on things:
./orchestrator.sh status

# After the week:
./orchestrator.sh collect
python3 analyze_week.py all_results/all_results.csv

# Outputs:
ls ./analysis/
```
