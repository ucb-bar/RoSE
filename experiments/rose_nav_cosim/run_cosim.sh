#!/usr/bin/env bash
# Run the RoSE <-> IsaacLab spike lockstep co-sim: the on-SoC flight stack (fused vision +
# EKF + TinyMPC -> motor thrusts) flying the Isaac warehouse gate course, REAL Isaac cameras.
#
# Two modes:
#   MODE=short  (default) - settle + a few gate ticks, quick sanity that the fresh clone flies.
#   MODE=full             - traverse the full 4-gate course.
#
# Prereqs (see REPRODUCE_ROSE_NAV_COSIM.md):
#   - guest ELF built (experiments/rose_nav_cosim/build_guest.sh)
#   - rose_spike_sim built (soc/src/main/cc/rose_spike/build.sh -> soc/sim/rose_spike_sim)
#   - IsaacLab + env_isaaclab conda env installed (external prereq)
#
# Env overrides:
#   ISAAC_PY   - python from the env_isaaclab conda env
#                (default: /scratch2/dima/miniforge3/envs/env_isaaclab/bin/python)
#   ELF        - guest ELF (default: build_guest/zephyr/zephyr.elf)
#   OUT_DIR    - run artifacts dir (default: experiments/rose_nav_cosim/run_out)
#   MODE       - short | full
#   ROSE_WH_SEED (default 1000)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSE="$(cd "$HERE/../.." && pwd)"

ISAAC_PY="${ISAAC_PY:-/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python}"
ELF="${ELF:-$HERE/build_guest/zephyr/zephyr.elf}"
OUT_DIR="${OUT_DIR:-$HERE/run_out}"
MODE="${MODE:-short}"
SEED="${ROSE_WH_SEED:-1000}"

[ -x "$ISAAC_PY" ] || { echo "ERROR: ISAAC_PY not found/executable: $ISAAC_PY"; exit 1; }
[ -f "$ELF" ] || { echo "ERROR: guest ELF not found: $ELF (run build_guest.sh)"; exit 1; }
[ -x "$ROSE/soc/sim/rose_spike_sim" ] || { echo "ERROR: rose_spike_sim missing (build it)"; exit 1; }

if [ "$MODE" = "full" ]; then
  MAX_SIM_TIME=3000; SPIKE_TIMEOUT=3400; WALL=3500; EPLEN=60
else
  MAX_SIM_TIME=250;  SPIKE_TIMEOUT=600;  WALL=700;  EPLEN=60
fi

SYNC="$OUT_DIR/sync.log"; SIM="$OUT_DIR/sim.log"; TRAJ="$OUT_DIR/traj.csv"
CHASE="$OUT_DIR/chase_frames"; FPV="$OUT_DIR/fpv_frames"
rm -rf "$OUT_DIR"; mkdir -p "$CHASE" "$FPV"

pkill -f "run_sync_only.py" 2>/dev/null || true
pkill -f rose_spike_sim 2>/dev/null || true
sleep 2

echo "[run_cosim] MODE=$MODE SEED=$SEED ELF=$ELF"
cd "$ROSE/deploy/hephaestus"
ROSE_GYM_ENV=WarehouseThrustEnv-v0 ROSE_VISION=1 ROSE_WH_OBST=0 ROSE_FREEZE=1 ROSE_WH_SEED="$SEED" \
  ROSE_ISAAC_CAMERA=1 ROSE_VIDEO_DIR="$CHASE" ROSE_FPV_DIR="$FPV" ROSE_VIDEO_DECIM=2 \
  ROSE_TRAJ_CSV="$TRAJ" ROSE_WH_EPLEN="$EPLEN" ROSE_MAX_SIM_TIME="$MAX_SIM_TIME" \
  ROSE_SYNC_WATCHDOG_S=300 PYTHONPATH="$ROSE/deploy/hephaestus" \
  "$ISAAC_PY" run_sync_only.py --yaml_path "$ROSE/deploy/config/config_gym_WarehouseThrustEnv-v0.yaml" > "$SYNC" 2>&1 &
SPID=$!
echo "[run_cosim] sync pid=$SPID booting Isaac (this can take a few minutes)..."
for i in $(seq 1 480); do
  grep -q "listening on" "$SYNC" 2>/dev/null && { echo "[run_cosim] listening ~${i}s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "[run_cosim] SYNC DIED"; tail -40 "$SYNC"; exit 1; }
  sleep 1
done
grep -q "listening on" "$SYNC" || { echo "[run_cosim] TIMEOUT boot"; kill $SPID 2>/dev/null; exit 1; }

ROSE_ISA=rv64gcv_zicntr_zihpm_zfh_zvfh ROSE_SPIKE_TIMEOUT="$SPIKE_TIMEOUT" timeout "$WALL" \
  bash "$ROSE/soc/sim/run_spike_rose_lockstep.sh" "$ELF" 1 > "$SIM" 2>&1
echo "[run_cosim] --- spike done ---"
kill $SPID 2>/dev/null || true
pkill -f "run_sync_only.py" 2>/dev/null || true
pkill -f rose_spike_sim 2>/dev/null || true

echo "=== guest boot/est/fused (tail) ==="; grep -E "nav_controller:|nav\[|fused:|Illegal|mcause" "$SIM" | tail -20
echo "=== GATE events ==="; grep -E "\[GATE\]" "$SYNC" || echo "(none yet)"
echo "=== frame counts ==="; echo "chase=$(ls "$CHASE" 2>/dev/null | wc -l) fpv=$(ls "$FPV" 2>/dev/null | wc -l)"
echo "[run_cosim] artifacts in $OUT_DIR"
