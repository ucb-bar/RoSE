#!/usr/bin/env bash
# ARBITER-TRACE flight: real WarehouseThrustEnv-v0 gate-nav co-sim on the split
# (firesim1 U250 FPGA + garden Isaac), with the instrumented driver's ROSE_ARB_TRACE
# heartbeat streaming the in-bitstream arbiter counters. Goal: localize WHERE the 0x42
# reqrsp response dies when the guest WFI-hangs (~211 physics steps).
#   sh flatlines while grants flow  -> arbiter stuck mid-packet (framing) on last_cmd
#   ch2~=(tx-rx0-rx1) flatlines      -> ch2 delivery dead
#   all advance                      -> guest-side hang
# Crash-safe: one infrasetup immediately before one runworkload; cleanup on any exit.
set -uo pipefail
ROSE=/scratch/dima/rose-infra/RoSE
SP=/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/d4827fc8-b516-4227-b71d-79192ba241cd/scratchpad
ISAAC_PY=/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python
GARDEN_IP=136.152.139.10
FS1=vnikiforov@firesim1.millennium.berkeley.edu
FSIMDIR=$ROSE/soc/sim/chipyard/sims/firesim
ULIVE=/scratch/vnikiforov/rose-fsim-run/sim_slot_0/uartlog
SYNC_LOG=$SP/arb_sync.log; INFRA_LOG=$SP/arb_infra.log; RW_LOG=$SP/arb_rw.log
ARB_LOG=$SP/arb_driver_uartlog.txt
FPV_DIR=$SP/arb_fpv; mkdir -p "$FPV_DIR"
VIDEO_DIR=$ROSE/deploy/hephaestus/logs

export ROSE_ARB_TRACE=10000      # driver heartbeat every N tick()s (forwarded via runtime_config.py)
export ROSE_MAX_SIM_TIME=45      # sim-time cap (s): enough for pre-hang + long flatline + clean end
export ROSE_WH_EPLEN=200
export ROSE_SYNC_WATCHDOG_S=250
WALL=800
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
  say "cleanup done (firesim1 sims=$ALIVE)"
}
trap cleanup EXIT INT TERM

# ---- pre-hygiene ----
ALIVE=$(ssh -o BatchMode=yes "$FS1" 'ps aux|grep FireSim-xilinx|grep vnikiforov|grep -v grep|wc -l' 2>/dev/null)
[ "${ALIVE:-1}" != 0 ] && { say "FAIL: firesim1 sim alive pre-run ($ALIVE)"; exit 1; }
ssh -o BatchMode=yes "$FS1" "rm -f $ULIVE" 2>/dev/null
ls "$VIDEO_DIR"/recording-*.avi >/dev/null 2>&1 && mv "$VIDEO_DIR"/recording-*.avi "$SP/" 2>/dev/null

# ---- start sync (Isaac boots ~1-4 min; chase cam ON; SERVE trace ON to correlate) ----
: > "$SYNC_LOG"
say "start sync (WarehouseThrustEnv, FREEZE, vision, SERVE trace, split->firesim1)"
( cd "$ROSE/deploy/hephaestus" && \
  ROSE_GYM_ENV=WarehouseThrustEnv-v0 ROSE_VISION=1 ROSE_WH_OBST=0 ROSE_FREEZE=1 ROSE_WH_SEED=1000 \
  ROSE_RENDER_HZ=10 ROSE_ISAAC_CAMERA=1 ROSE_VIDEO_DIR="$FPV_DIR" ROSE_SERVE_DEBUG=1 \
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

# ---- infrasetup (immediately before runworkload; ships the instrumented driver) ----
say "infrasetup firesim1 (nav guest, instrumented Saturn MMIO driver)..."
if fsim firesim infrasetup -c config_runtime_firesim1.yaml > "$INFRA_LOG" 2>&1; then say "infrasetup OK"; else say "infrasetup FAILED"; tail -20 "$INFRA_LOG"; exit 4; fi

# ---- runworkload (driver dials back to garden:10001; ARB heartbeat streams to uartlog) ----
say "runworkload (instrumented flight begins)..."
fsim firesim runworkload -c config_runtime_firesim1.yaml > "$RW_LOG" 2>&1 &
FLIGHT_START=$(date +%s); last_hb=0
while true; do
  now=$(date +%s); el=$((now-FLIGHT_START))
  if ! kill -0 $SYNC_PID 2>/dev/null; then say "sync exited (co-sim ended) at ~${el}s wall"; break; fi
  [ $el -ge $WALL ] && { say "WALL cap ${WALL}s — ending"; break; }
  if [ $((el - last_hb)) -ge 20 ]; then
    last_hb=$el
    iter=$(grep -aoE "Stepping simulation: [0-9]+ iters" "$SYNC_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | tail -1)
    arb=$(ssh -o BatchMode=yes "$FS1" "grep -a '\[ARB\] sh=' $ULIVE 2>/dev/null | tail -1" 2>/dev/null)
    say "  t+${el}s grant_iter=${iter:-0} | ${arb:-<no ARB line yet>}"
  fi
  sleep 3
done

# ---- fetch driver uartlog ([ARB] heartbeat + [ROSE DRIVER] cmds) + video ----
say "=== flight ended; fetching driver uartlog + video ==="
scp -o BatchMode=yes "$FS1:$ULIVE" "$ARB_LOG" 2>/dev/null && say "driver uartlog -> $ARB_LOG ($(wc -l < "$ARB_LOG") lines)"
VID=$(ls -t "$VIDEO_DIR"/recording-*.avi 2>/dev/null | head -1)
say "RESULT: last grant_iter=$(grep -aoE 'Stepping simulation: [0-9]+ iters' "$SYNC_LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | tail -1) video=${VID:-NONE}"
echo "=== last 6 [ARB] heartbeats ==="; grep -a "\[ARB\] sh=" "$ARB_LOG" 2>/dev/null | tail -6
echo "=== last 14 sync SERVE lines ==="; grep -aE "\[SERVE\]" "$SYNC_LOG" 2>/dev/null | tail -14
