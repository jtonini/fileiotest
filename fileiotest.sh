#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 {numfiles} {user@(host|ipaddress)} {stats-file-name} "
    echo " "
    echo " The stats file will contain tagged numbers suitable for stats processing."
    echo " "
    exit 1
fi

NUM="$1"
DEST="$2"
STATS_FILE="$3"

unlink $STATS_FILE 2>/dev/null || true

# Check for vmtouch on this distro. It is not in
# the core OS of every Linux system.

if ! command -v vmtouch >/dev/null 2>&1; then
    echo "vmtouch not installed."
    echo "It can be found here: git clone --depth 1 https://github.com/hoytech/vmtouch.git"
    exit 2
fi

# Check on pv, too.
if ! command -v pv >/dev/null 2>&1; then
    echo "pv not installed. Installing it."
    sudo dnf -y install pv
fi

export PV_FLAGS DEST

find_python()
{
  for py in python3 python3.9; do
    if command -v "$py" >/dev/null &&
       "$py" -c 'import sys; exit(sys.version_info < (3,9))'
    then
      echo "$py"
      return 0
    fi
  done
  return 1
}

PYTHON=$(find_python) || {
  echo "Python 3.9+ required."
  exit 1
}


# Build files filled with junk. This bit of code requires python3.9+
# First, see if python3 is pointing at python3.9 or higher, and then
# try python3.9 specifically.
eval "$PYTHON" ./randomfiles.py -n "$NUM"

# Try blowing out the whole cache. If it doesn't work, not a big deal.
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true

# Explicitly remove from the cache the files we created.
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -e


###############################################################################
# Record the user stats
###############################################################################
date >> $STATS_FILE
echo `hostname` to $DEST >> $STATS_FILE
echo "Initial stats" >> $STATS_FILE
nstat -az | egrep -i 'TcpRetransSegs|TCPTimeouts|OutRsts' >> $STATS_FILE

# Here is the UGLY UGLY business end of the code.
run_transfer()
{
  set -o pipefail

  find . -type f -name "*.iotest" -print0 \
    | xargs -0 -r cat \
    | pv -f -i 1 -F "bytes=%b rate=%r " \
        2> >(tr '\r' '\n' | awk 'NF{last=$0} END{print last}' > pv.last) \
    | ssh -T "$DEST" "cat > /dev/null"
}


###############################################################################
# First run. Comments in the first run only -- same commands are used in the
# next two runs.
###############################################################################
echo "=== COLD-CACHE RUN ===" >> $STATS_FILE

# Attempt to empty the cache on the remote machine.
ssh -T $DEST 'echo 3 | sudo tee /proc/sys/vm/drop_caches'

# Invoke the transfer.
run_transfer
cat pv.last >> $STATS_FILE

# Record the stats.
nstat -a | egrep -i 'TcpRetransSegs|TCPTimeouts|OutRsts' >> $STATS_FILE

# Read the files into memory.
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -t
###############################################################################
###############################################################################
echo "=== HOT-CACHE RUN ===" >> $STATS_FILE
ssh -T $DEST 'echo 3 | sudo tee /proc/sys/vm/drop_caches'
run_transfer
cat pv.last >> $STATS_FILE
nstat -a | egrep -i 'TcpRetransSegs|TCPTimeouts|OutRsts' >> $STATS_FILE

###############################################################################
###############################################################################
run_transfer_and_write() {
  set -o pipefail

  find . -type f -name "*.iotest" -print0 \
    | xargs -0 -r cat \
    | pv -f -i 1 -F "bytes=%b rate=%r " \
        2> >(tr '\r' '\n' | awk 'NF{last=$0} END{print last}' > pv.last) \
    | ssh -T "$DEST" "cat > ./this_file_is_junk"
}


echo "=== TRUE WRITE ===" >> $STATS_FILE
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -e
ssh -T $DEST 'echo 3 | sudo tee /proc/sys/vm/drop_caches'
run_transfer_and_write
cat pv.last >> $STATS_FILE
nstat -a | egrep -i 'TcpRetransSegs|TCPTimeouts|OutRsts' >> $STATS_FILE
ssh "$DEST" "rm -f ./this_file_is_junk"

rm -f ./*.iotest

