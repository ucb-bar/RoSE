#!/bin/bash
# Pull the sweep's per-cell JSONs off the EC2 sweep instance and onto garden.
#
# The instance is reachable ONLY through the FireSim manager, and only the
# manager holds AWS credentials, so every hop goes:
#   garden -> ubuntu@3.88.218.39 (manager) -> ubuntu@192.168.0.44 (sweep box)
# base64 over the nested ssh avoids quoting the inner command twice.
set -euo pipefail

MGR=${MGR:-ubuntu@3.88.218.39}
BOX=${BOX:-ubuntu@192.168.0.44}
KEY=${KEY:-$HOME/.ssh/firesim.pem}          # the key as named on GARDEN
RKEY=${RKEY:-'~/.ssh/firesim.pem'}          # the same key as named ON THE MANAGER
DEST=${1:-$(dirname "$0")/../results}
SSHO="-o BatchMode=yes -o StrictHostKeyChecking=no"

mkdir -p "$DEST"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# One tarball with both cell sets: the main 4-solver grid, and the
# supplementary MILP arm that re-ran the buildable cells with a 600 s optimizer
# limit instead of the workloads' own 120 s.
ssh $SSHO -i "$KEY" "$MGR" \
  "ssh $SSHO -i $RKEY $BOX 'cd /home/ubuntu && tar czf sweep_cells.tgz sweep/cells \
     sweep/results/critical_path.json sweep/results/milp_model_size.json \
     \$( [ -d sweep600/cells ] && echo sweep600/cells )'" >/dev/null

ssh $SSHO -i "$KEY" "$MGR" \
  "scp -q $SSHO -i $RKEY $BOX:/home/ubuntu/sweep_cells.tgz /tmp/sweep_cells.tgz && base64 -w0 /tmp/sweep_cells.tgz" \
  > "$tmp/cells.b64"

base64 -d "$tmp/cells.b64" > "$tmp/cells.tgz"
rm -rf "$DEST/cells" "$DEST/cells_milp600"
tar xzf "$tmp/cells.tgz" -C "$tmp"
mv "$tmp/sweep/cells" "$DEST/cells"
[ -d "$tmp/sweep600/cells" ] && mv "$tmp/sweep600/cells" "$DEST/cells_milp600"
for f in critical_path.json milp_model_size.json; do
  [ -f "$tmp/sweep/results/$f" ] && cp "$tmp/sweep/results/$f" "$DEST/$f"
done
echo "pulled $(ls "$DEST/cells" | wc -l) per-cell JSONs into $DEST/cells"
[ -d "$DEST/cells_milp600" ] && \
  echo "pulled $(ls "$DEST/cells_milp600" | wc -l) supplementary MILP cells into $DEST/cells_milp600"
exit 0
