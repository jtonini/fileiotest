#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {numfiles} {user@ipaddress}"
  exit 1
fi

NUM="$1"
DEST="$2"

# Best to [re]build vmtouch for this distro. It is not in
# the core OS of every Linux system.
if ! command -v vmtouch >/dev/null 2>&1; then
  echo "vmtouch not installed. Building/installing it."
  sudo dnf install -y gcc make git

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  git clone --depth 1 https://github.com/hoytech/vmtouch.git "$tmpdir/vmtouch"
  make -C "$tmpdir/vmtouch"
  sudo install -m 0755 "$tmpdir/vmtouch/vmtouch" /usr/local/bin/
fi

# Check on pv, too.
if ! command -v pv >/dev/null 2>&1; then
  echo "pv not installed. Installing it."
  sudo dnf -y install pv
fi

# If bits are supported as a UOM, let's use them.
if pv --help 2>&1 | grep -q -- '-8'; then
  PV_FLAGS='-ra8tpe -i 1'
else
  PV_FLAGS='-rabtpe -i 1'
fi
export PV_FLAGS DEST

# Build files filled with junk.
/usr/bin/time -v python ./randomfiles.py -n "$NUM"

# Try blowing out the whole cache. If it doesn't work, not a big deal.
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true

# Read the files into memory.
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -t

# Here is the business end of the code.
run_transfer() {
  /usr/bin/time -v bash -o pipefail -c '
    find . -type f -name "*.iotest" -print0 \
      | xargs -0 cat \
      | pv $PV_FLAGS \
      | ssh -T "$DEST" "cat > /dev/null"
  '
}

echo "=== HOT-CACHE RUN ==="
run_transfer

# Explicitly remove the files we cached.
find . -type f -name '*.iotest' -print0 | xargs -0 vmtouch -e

echo "=== COLD-CACHE RUN ==="
run_transfer

rm -f ./*.iotest

