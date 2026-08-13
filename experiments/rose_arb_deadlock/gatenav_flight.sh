#!/usr/bin/env bash
# GATE-NAV DELIVERABLE flight on the DELIVERY-FIXED bitstream (2026-08-13, held-valid
# handshake; validated 2020 iters / 91k words / zero drops). Runs the EXACT spike
# 3/4-gate config (docs/GATE_NAV_SPIKE_TO_FPGA_PORT.md §2c) on the FPGA co-sim, with:
#   - ROSE_MAX_SIM_TIME=250  : headroom for ~12s physics = 3 gates (grant-time cap; §3)
#   - ROSE_TRAJ_CSV          : REAL physics measurement (grant-iters != physics under FREEZE)
#   - chase camera -> ROSE_VIDEO_DIR (f_*.jpg) for the recorded-video deliverable
# Watches [GATE] passed markers (sync log) + physics time (traj CSV col t).
set -uo pipefail
ROSE=/scratch/dima/rose-infra/RoSE
SP=/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/d4827fc8-b516-4227-b71d-79192ba241cd/scratchpad
ISAAC_PY=/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python
GARDEN_IP=136.152.139.10
FS1=vnikiforov@firesim1.millennium.berkeley.edu
FSIMDIR=$ROSE/soc/sim/chipyard/sims/firesim
ULIVE=/scratch/vnikiforov/rose-fsim-run/sim_slot_0/uartlog
SYNC_LOG=$SP/gn_sync.log; INFRA_LOG=$SP/gn_infra.log; RW_LOG=$SP/gn_rw.log
DRV_LOG=$SP/gn_driver_uartlog.txt
CHASE_DIR=$SP/gn_chase; FPV_DIR=$SP/gn_fpv; TRAJ=$SP/gn_traj.csv
rm -rf "$CHASE_DIR" "$FPV_DIR"; mkdir -p "$CHASE_DIR" "$FPV_DIR"; rm -f "$TRAJ"

export ROSE_ARB_TRACE=20000       # light heartbeat (delivery already proven; watch for regressions)
export ROSE_MAX_SIM_TIME=250      # grant-time cap (s): headroom for 3 gates (~12s physics), §3
export ROSE_WH_EPLEN=60           # episode length (seconds physics) — matches spike
export ROSE_SYNC_WATCHDOG_S=400
WALL=5400                         # 90 min wall cap (30MHz is slow)
say(){ echo "[$(date '+%H:%M:%S')] $*"; }

fsim(){ ( set +u; cd "$FSIMDIR" && source env.sh >/dev/null 2>&1 && source sourceme-manager.sh --skip-ssh-setup >/dev/null 2>&1 \
          && cd deploy && ROSE_SYNC_HOST=$GARDEN_IP ROSE_SYNC_PORT=10001 ROSE_ARB_TRACE=$ROSE_ARB_TRACE "$@" ); }

cleanup(){
  say "cleanup..."
  [ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null
  pkill -f 'run_sync_only.py --yaml_path .*WarehouseThrustEnv' 2>/dev/null
  fsim firesim kill -c config_runtime_firesim1.yaml >/dev/null 2>&1 || true
  sleep 3
  ALIVE=$(ssh -o BatchMode=yes "$FS1" 'ps aux|grep FireSim-xilinx|grep vnikiforov|grep -v grep|wc -l' 2>/dev/null)
  say "cleanup done (firesim1 sims=$ALIVE); chase=$(ls "$CHASE_DIR" 2>/dev/null|wc -l) frames; traj=$([ -f "$TRAJ" ]&&wc -l <"$TRAJ"||echo 0) rows"
}
trap cleanup EXIT INT TERM

# ---- pre-hygiene ----
ALIVE=$(ssh -o BatchMode=yes "$FS1" 'ps aux|grep FireSim-xilinx|grep vnikiforov|grep -v grep|wc -l' 2>/dev/null)
[ "${ALIVE:-1}" != 0 ] && { say "FAIL: firesim1 sim alive pre-run ($ALIVE)"; exit 1; }
ssh -o BatchMode=yes "$FS1" "rm -f $ULIVE" 2>/dev/null

# ---- start sync: EXACT spike gate config + chase cam + traj CSV ----
: > "$SYNC_LOG"
say "start sync (spike 3-gate config: WarehouseThrust seed1000 FREEZE vision, chase cam, TRAJ)"
( cd "$ROSE/deploy/hephaestus" && \
  ROSE_GYM_ENV=WarehouseThrustEnv-v0 ROSE_VISION=1 ROSE_WH_OBST=0 ROSE_FREEZE=1 ROSE_WH_SEED=1000 \
  ROSE_ISAAC_CAMERA=1 ROSE_VIDEO_DIR="$CHASE_DIR" ROSE_FPV_DIR="$FPV_DIR" ROSE_VIDEO_DECIM=2 \
  ROSE_RENDER_HZ=10 ROSE_TRAJ_CSV="$TRAJ" ROSE_SERVE_DEBUG=0 \
  ROSE_SYNC_RECV_TIMEOUT=0.001 ROSE_FIRESIM_STEP=5000000 ROSE_FIRESIM_FREQ=1000000000 \
  ROSE_SYNC_HOST=$GARDEN_IP PYTHONPATH="$ROSE/deploy/hephaestus" \
  "$ISAAC_PY" -u run_sync_only.py --yaml_path "$ROSE/deploy/config/config_gym_WarehouseThrustEnv-v0.yaml" ) > "$SYNC_LOG" 2>&1 &
SYNC_PID=$!
for i in $(seq 1 600); do
  grep -q "listening on" "$SYNC_LOG" 2>/dev/null && { say "sync listening after ~${i}s"; break; }
  kill -0 $SYNC_PID 2>/dev/null || { say "SYNC DIED during boot"; tail -25 "$SYNC_LOG"; exit 2; }
  sleep 1
done
grep -q "listening on" "$SYNC_LOG" || { say "sync boot timeout"; exit 3; }

# ---- infrasetup immediately before runworkload (delivery-fixed Saturn MMIO driver) ----
say "infrasetup firesim1 (delivery-fixed bitstream)..."
if fsim firesim infrasetup -c config_runtime_firesim1.yaml > "$INFRA_LOG" 2>&1; then say "infrasetup OK"; else say "infrasetup FAILED"; tail -20 "$INFRA_LOG"; exit 4; fi

# ---- runworkload; watch [GATE] + physics time ----
say "runworkload (gate-nav flight begins; watching [GATE] + physics)..."
fsim firesim runworkload -c config_runtime_firesim1.yaml > "$RW_LOG" 2>&1 &
FLIGHT_START=$(date +%s); last_hb=0; gates=0
while true; do
  now=$(date +%s); el=$((now-FLIGHT_START))
  if ! kill -0 $SYNC_PID 2>/dev/null; then say "sync exited (co-sim ended) at ~${el}s wall"; break; fi
  [ $el -ge $WALL ] && { say "WALL cap ${WALL}s — ending"; break; }
  ng=$(grep -acE "\[GATE\] passed" "$SYNC_LOG" 2>/dev/null)
  if [ "${ng:-0}" -gt "$gates" ]; then
    gates=$ng; say "  *** GATE $gates *** $(grep -aE "\[GATE\] passed" "$SYNC_LOG" | tail -1)"
  fi
  if [ $((el - last_hb)) -ge 30 ]; then
    last_hb=$el
    pt=$(tail -1 "$TRAJ" 2>/dev/null | awk -F, '{print $3"s x="$4" y="$5" z="$6" goal="$14" gates="$15}')
    iter=$(grep -aoE "Stepping simulation: [0-9]+ iters" "$SYNC_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | tail -1)
    say "  t+${el}s grant_iter=${iter:-0} gates=$gates | physics: ${pt:-<no traj row yet>}"
  fi
  sleep 4
done

# ---- harvest: driver uartlog, [GATE] summary, video frames ----
say "=== flight ended; harvesting ==="
scp -o BatchMode=yes "$FS1:$ULIVE" "$DRV_LOG" 2>/dev/null && say "driver uartlog -> $DRV_LOG"
echo "=== ALL [GATE] passed ==="; grep -aE "\[GATE\] passed" "$SYNC_LOG" 2>/dev/null
echo "=== final traj rows ==="; tail -3 "$TRAJ" 2>/dev/null
echo "=== chase frames: $(ls "$CHASE_DIR" 2>/dev/null | wc -l) ==="
say "RESULT: gates=$(grep -acE '\[GATE\] passed' "$SYNC_LOG") chase_frames=$(ls "$CHASE_DIR" 2>/dev/null|wc -l) traj_rows=$([ -f "$TRAJ" ]&&wc -l <"$TRAJ"||echo 0)"
