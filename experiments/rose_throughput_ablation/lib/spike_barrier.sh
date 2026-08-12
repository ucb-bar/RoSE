#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# spike_barrier.sh — measure ONE co-sim sync-barrier operating point on the
# functional Spike RoSE bridge (NO FPGA, NO GPU, NO Isaac).
#
# This is the runnable-now substrate for the SOCKET-FIX ablation axis: it drives
# the REAL tracked deploy/hephaestus synchronizer + socket_thread against the
# rose_spike_sim bridge running a camera-free guest, and decomposes the per-grant
# barrier into send / rtt / pickup with high-res timestamps (lib/seam_instr_run2.py).
#
# The only lever this script flips is ROSE_SYNC_RECV_TIMEOUT:
#   0.1   = OLD synchronizer poll (recv-before-tx backoff)  -> ~115 ms/grant
#   0.001 = NEW committed fast poll (txqueue-before-recv)    -> ~16-21 ms/grant
# (The txqueue-before-recv reorder itself is committed at 72fa33e; this env var
#  reproduces the old-vs-new REGIME against that committed code.)
#
# Emits a per-grant CSV (grant,A,B,C,D,send_ms,rtt_ms,pickup_ms,total_ms) and a
# one-line summary "SUMMARY <tag> <recv_timeout> <median_total_ms> <grants>".
# ---------------------------------------------------------------------------
set -u

ROSE_ROOT="${ROSE_ROOT:-/scratch/dima/rose-infra/RoSE}"
BUNDLE="$ROSE_ROOT/experiments/rose_throughput_ablation"
PY="${ROSE_PY:-$ROSE_ROOT/deploy/.venv-rose/bin/python}"
SPIKE_BIN="${ROSE_SPIKE_BIN:-$ROSE_ROOT/soc/sim/rose_spike_sim}"
GUEST_ELF="${ROSE_GUEST_ELF:-$ROSE_ROOT/soc/sim/zephyr_rose_builds/reqrsp/zephyr/zephyr.elf}"
PATTERN_YAML="${ROSE_PATTERN_YAML:-$ROSE_ROOT/deploy/config/config_gym_PatternEnv-v0.yaml}"
OUTDIR="${ROSE_ABLATION_OUT:-$BUNDLE/run_out}"

# The lever + knobs (all defaulted; override from run_ablation.sh):
RECV_TIMEOUT="${ROSE_SYNC_RECV_TIMEOUT:-0.001}"
TAG="${TAG:-newsock}"
PORT="${PORT:-10031}"
STEP="${STEP:-1000}"          # tiny cycle budget so Spike retires it ~instantly ->
FREQ="${FREQ:-1000000}"       #   per-grant wall ~= pure barrier overhead
MAXG="${MAXG:-40}"            # grants to time (3 warmup skipped in the stats)
BOOT_WAIT="${BOOT_WAIT:-4}"
TIMEOUT_S="${TIMEOUT_S:-120}"

mkdir -p "$OUTDIR"
for f in "$PY" "$SPIKE_BIN" "$GUEST_ELF" "$PATTERN_YAML"; do
  if [ ! -e "$f" ]; then echo "[spike_barrier] MISSING prerequisite: $f" >&2; exit 3; fi
done

SYNC_LOG="$OUTDIR/sync_${TAG}.log"
SPIKE_LOG="$OUTDIR/spike_${TAG}.log"
SEAM_CSV="$OUTDIR/seam_${TAG}.csv"

echo "[spike_barrier] tag=$TAG recv_timeout=${RECV_TIMEOUT}s step=$STEP port=$PORT maxg=$MAXG"

# 1) start the REAL tracked synchronizer under the non-invasive seam instrumentation
export ROSE_ROOT ROSE_GYM_ENV=PatternEnv-v0 \
       ROSE_FIRESIM_STEP="$STEP" ROSE_FIRESIM_FREQ="$FREQ" \
       ROSE_SYNC_PORT="$PORT" ROSE_SYNC_WATCHDOG_S=0 \
       ROSE_SYNC_RECV_TIMEOUT="$RECV_TIMEOUT" \
       SEAM_MAX_GRANTS="$MAXG" SEAM_OUT="$SEAM_CSV" ROSE_RX_DEBUG=1
cd "$ROSE_ROOT/deploy/hephaestus"
"$PY" "$BUNDLE/lib/seam_instr_run2.py" --yaml_path "$PATTERN_YAML" > "$SYNC_LOG" 2>&1 &
SYNC=$!
sleep "$BOOT_WAIT"

# 2) start the Spike RoSE bridge on the same port with the camera-free guest.
#    (No chipyard env.sh needed: rose_spike_sim resolves its libs via rpath; run
#     from soc/sim so librose_spike.so resolves.)
cd "$ROSE_ROOT/soc/sim"
export LD_LIBRARY_PATH="$ROSE_ROOT/soc/sim:${LD_LIBRARY_PATH:-}"
ROSE_SPIKE_DEBUG=1 stdbuf -oL -eL "$SPIKE_BIN" -p 1 \
  --rose-base=0x2000 --rose-irq=3 --rose-dma-base=0x90000000 \
  --rose-nreqrsp=2 --rose-ndma=1 --rose-port="$PORT" \
  "$GUEST_ELF" > "$SPIKE_LOG" 2>&1 &
SPIKE=$!

# 3) wait for the synchronizer to reach MAXG grants and self-dump, then clean up
for i in $(seq 1 "$TIMEOUT_S"); do
  if ! kill -0 $SYNC 2>/dev/null; then echo "[spike_barrier] sync finished ~${i}s"; break; fi
  sleep 1
done
kill $SPIKE 2>/dev/null; pkill -P $SPIKE 2>/dev/null; kill $SYNC 2>/dev/null

# 4) emit the machine-readable summary line (median total_ms over non-warmup grants)
if [ -s "$SEAM_CSV" ]; then
  "$PY" - "$SEAM_CSV" "$TAG" "$RECV_TIMEOUT" <<'PYEOF'
import sys, statistics
csv, tag, to = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [l.strip().split(",") for l in open(csv)][1:]
tot = [float(r[8]) for r in rows[3:] if len(r) >= 9]  # total_ms, 3 warmup skipped
if tot:
    print(f"SUMMARY {tag} {to} {statistics.median(tot):.3f} {len(tot)}")
else:
    print(f"SUMMARY {tag} {to} NA 0")
PYEOF
else
  echo "SUMMARY $TAG $RECV_TIMEOUT NA 0"
fi
