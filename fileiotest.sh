#!/usr/bin/env bash
set -euo pipefail

# Best to [re]build vmtouch for this distro. It is not in
# the core OS.
if ! command -v vmtouch >/dev/null 2>&1; then
    echo "vmtouch not installed. Installing it"
    sudo dnf install -y gcc make git
    git clone https://github.com/hoytech/vmtouch.git
    cd vmtouch
    make
    sudo install -m 0755 vmtouch /usr/local/bin/
fi

# pv has the same limitation.
if ! command -v pv >/dev/null 2>&1; then
    echo "pv not installed. Installing it."
    dnf -y install pv
fi

if [ -z "$2" ]; then
    echo "Usage: $0 {numfiles} {user@ipaddress}"
    exit 1
fi

NUM="$1"
DEST="$2"

# Does pv support bits as a UOM on this platform?
if pv --help 2>&1 | grep -q -- '-8'; then
  export PV_FLAGS="-ra8tpe -i 1"
else
  export PV_FLAGS="-rabtpe -i 1"
fi

# Create new files of random data.
/usr/bin/time -v python ./randomfiles.py -n "$NUM"

# Try blowing out the cache entirely. Doesn't hurt to try, yes?
sudo echo 1 > /proc/sys/vm/drop_caches || true

# Load them into cache
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -t
# find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch

# Stats.
/usr/bin/time -v bash -o pipefail -c '
  find . -type f -name "*.iotest" -print0 \
  | xargs -0 pv '"$PV_FLAGS" ' \
  | ssh -T ' "$DEST" ' "cat > /dev/null" '

# remove the files from cache
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -e

# rerun, same files.
/usr/bin/time -v bash -o pipefail -c '
  find . -type f -name "*.iotest" -print0 \
  | xargs -0 pv '"$PV_FLAGS" ' \
  | ssh -T ' "$DEST" ' "cat > /dev/null" '


rm -fr *.iotest
