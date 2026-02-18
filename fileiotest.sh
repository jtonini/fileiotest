#!/usr/bin/env bash
set -euo pipefail

TESTINGHOST="billieholiday"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 {numfiles} {user@(host|ipaddress)} {stats-file-name} "
    echo " "
    echo " The stats file will contain tagged numbers suitable for stats processing."
    echo " "
    exit 1
fi

testing()
{
    if [ "$(hostname -s)" == "$TESTINGHOST" ]; then
        true
    else
        false
    fi
}

###
# Some utility functions.
###
remote_OS()
{
    if [ -z "$1" ]; then
        echo "Usage: remote_OS [user@]hostname"
        echo " returns {unknown|linux|UNIX|OS-name-literal}"
        false
        return
    fi

    OS=$(ssh "$1" 'uname -s')
    if [ "$?" -ne 0 ]; then
        echo "unknown"
    elif [ "$OS" == "Linux" ]; then
        echo "linux"
    elif [ "$OS" == "FreeBSD" ]; then
        echo "UNIX"
    else
        echo "$OS"
    fi
}

ipfromhostname()
{
    if [ -z "$1" ]; then
        cat<<EOF
    usage: ipfromhostname somestring

    somestring can be user@host, user@IP.AD.DR.ESS, host, or IP.AD.DR.ESS.

    returns normalized IP address if it is available.
EOF
    return 1
    fi

    string="$1"
    # Chop the user@ part.
    host="${string##*@}"

    # If the remaining part is an IP address, we are done.
    if [[ "$host" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
        echo "$host"
        return 0
    fi


    # Check locally available info first. This will work if the
    # current host has recently refreshed host info even if
    # DNS is down.
    ipaddress=$(getent ahostsv4 "$host" | head -1 | awk '{print $1}')
    if [ ! -z "$ipaddress" ]; then
        echo "$ipaddress"
        return 0
    fi

    # Ask the question the usual way.
    ipaddress=$(host -W 2 -t A "$host" | awk '{print $NF}')
    if [ ! -z "$ipaddress" ]; then
        echo "$ipaddress"
        return 0
    fi

    # Ping might work if DNS is messed up
    ipaddress=$(ping -n -c 1 -W 1 "$host" | head -2 | tail -1 | awk '{print $4}')
    ipaddress=${ipaddress//:}
    if [ ! -z "$ipaddress" ]; then
        echo "$ipaddress"
        return 0
    fi

    return 2
}

###
# Local variables.
###
NUM="$1"
DEST="$2"
STATS_FILE="$3"
DEST_OS=$(remote_OS $DEST)
HOST=$(ipfromhostname "$DEST")
USER=${DEST%@*}
DEST="$USER"@"$HOST"
SSH_OPTS="-T -F /dev/null"

###
# export-ing the variables eliminates the need to pass them
# as arguments.
###
export NUM DEST STATS_FILE DEST_OS HOST USER DEST SSH_OPTS

###
# remove the STATS_FILE if it is there.
# I have never seen unlink aliased, but rm almost always is.
###
testing && echo "removing old STATS_FILE"
unlink $STATS_FILE 2>/dev/null || true

###
# Check for vmtouch on this distro. It is not in
# the core OS of every Linux system.
###
if ! command -v vmtouch >/dev/null 2>&1; then
    echo "vmtouch not installed."
    echo "It can be found here: git clone --depth 1 https://github.com/hoytech/vmtouch.git"
    exit 2
fi

###
# Check on pv, too. It can be dnf installed.
###
if ! command -v pv >/dev/null 2>&1; then
    echo "pv not installed. Installing it."
    sudo dnf -y install pv
fi


###
# The random file generator requires Python3.9+. It is possible
# for this function to fail (i.e., the computer *has* Python 3.9,
# but the function misses it.
###
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

testing && echo "Python is $PYTHON"

###
# Build files filled with junk, and force them to be written
# to disk.
###
eval "$PYTHON" ./randomfiles.py -n "$NUM"
sync

testing && echo "Random files of data built."

###
# Try blowing out the whole cache. If it doesn't work, not a big deal.
###
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
testing && echo "local cache flushed."

###############################################################################
# Record the user stats
###############################################################################
date >> $STATS_FILE
echo `whoami`@`hostname` to $DEST >> $STATS_FILE
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
    | ssh $SSH_OPTS "$DEST" "cat > /dev/null"
}


###############################################################################
# First run. Comments in the first run only -- same commands are used in the
# next two runs.
###############################################################################

echo "=== COLD-CACHE RUN ===" | tee -a $STATS_FILE
###
# Explicitly remove from the cache the files we created.
###
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -e

###
# Attempt to empty the cache on the remote machine if it is Linux. If our
# user on the remote Linux machine is not root, then forgive the failure.
###
if [ "$DEST_OS" == "linux" ]; then
    ssh $SSH_OPTS $DEST 'echo 3 | sudo tee /proc/sys/vm/drop_caches' || true
fi


# Invoke the transfer.
run_transfer
cat pv.last >> $STATS_FILE

# Record the stats.
nstat -a | egrep -i 'TcpRetransSegs|TCPTimeouts|OutRsts' >> $STATS_FILE

###############################################################################
# Now we do three hot-cache runs separated by 60 seconds.
# We reverse course and explicitly read our test files into
# memory.
###############################################################################
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -t
echo "=== 3 HOT-CACHE RUNS ===" | tee -a $STATS_FILE
ssh $SSH_OPTS $DEST 'echo 3 | sudo tee /proc/sys/vm/drop_caches'
run_transfer
cat pv.last >> $STATS_FILE
echo "Zzzleeping for 60 seconds."
sleep 60
run_transfer
cat pv.last >> $STATS_FILE
echo "ZZzleeping for 60 seconds."
sleep 60
run_transfer
cat pv.last >> $STATS_FILE
nstat -a | egrep -i 'TcpRetransSegs|TCPTimeouts|OutRsts' >> $STATS_FILE

###############################################################################
# This final exercise will write the files on the remote computer.
###############################################################################
run_transfer_and_write()
{
  set -o pipefail

  find . -type f -name "*.iotest" -print0 \
    | xargs -0 -r cat \
    | pv -f -i 1 -F "bytes=%b rate=%r " \
        2> >(tr '\r' '\n' | awk 'NF{last=$0} END{print last}' > pv.last) \
    | ssh $SSH_OPTS "$DEST" "cat > ./this_file_is_junk"

  if [ "$DEST_OS" == "linux" ]; then
    ssh $SSH_OPTS "$DEST" 'sync'
  fi
}


echo "=== TRUE WRITE ===" | tee -a $STATS_FILE
###
# Erase our own cache.
###
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -e

if [ "$DEST_OS" == "linux" ]; then
    ssh $SSH_OPTS $DEST 'echo 3 | sudo tee /proc/sys/vm/drop_caches'
fi

run_transfer_and_write
cat pv.last >> $STATS_FILE
nstat -a | egrep -i 'TcpRetransSegs|TCPTimeouts|OutRsts' >> $STATS_FILE

###
# Clean up the remote.
###
ssh "$DEST" "rm -f ./this_file_is_junk"

rm -f ./*.iotest

