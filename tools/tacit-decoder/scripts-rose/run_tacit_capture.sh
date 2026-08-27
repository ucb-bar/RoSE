#!/usr/bin/env bash
# Capture a TACIT instruction trace from the FireSim FPGA. Runs the TACIT-enabled hello-world
# guest on the Saturn+RoSE+TACIT bitstream (local U250). The guest's l_trace encoder (target=
# RawByte sink, id 0) emits the trace; the ported TacitBridge streams it off-FPGA; tacit.cc
# writes tacit0.out in the sim run dir. The RoSE bridge just needs cycle grants -> a free-run
# sync (no FREEZE). Guest prints "Hello World!" then reboots (HTIF) -> sim ends -> trace flushed.
set -uo pipefail
ROSE=/scratch/dima/rose-infra/RoSE
SP=/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/d4827fc8-b516-4227-b71d-79192ba241cd/scratchpad
ISAAC_PY=/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python
FSIMDIR=$ROSE/soc/sim/chipyard/sims/firesim
SIMDIR=$ROSE/soc/sim/firesim_run_temp
RUNTIME=config_runtime_local_tacit.yaml
PORT=10076; SYNC_HOST=127.0.0.1
SYNC_LOG=$SP/tacit_sync.log; INFRA_LOG=$SP/tacit_infra.log; RW_LOG=$SP/tacit_rw.log
rm -f "$SYNC_LOG" "$INFRA_LOG" "$RW_LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
fsim(){ ( set +u; cd "$FSIMDIR" && source env.sh >/dev/null 2>&1 && source sourceme-manager.sh --skip-ssh-setup >/dev/null 2>&1 \
          && cd deploy && ROSE_SYNC_HOST=$SYNC_HOST ROSE_SYNC_PORT=$PORT "$@" ); }
cleanup(){
  say "cleanup..."; [ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null
  pkill -f 'run_sync_only.py --yaml_path .*WarehouseThrustEnv' 2>/dev/null
  fsim firesim kill -c "$RUNTIME" >/dev/null 2>&1 || true; sleep 3; say "cleanup done"
}
trap cleanup EXIT INT TERM
LOCAL=$(ps aux 2>/dev/null | grep -i FireSim-xilinx | grep -v grep | wc -l)
[ "${LOCAL:-1}" != 0 ] && { say "FAIL: local FireSim sim already running"; exit 1; }

say "start free-run sync (grants cycles; guest is standalone TACIT hello-world)"
( cd "$ROSE/deploy/hephaestus" && \
  ROSE_GYM_ENV=WarehouseThrustEnv-v0 ROSE_VISION=0 ROSE_WH_OBST=0 ROSE_WH_SEED=1000 \
  ROSE_SYNC_RECV_TIMEOUT=0.001 ROSE_FIRESIM_STEP=5000000 ROSE_FIRESIM_FREQ=1000000000 \
  ROSE_SYNC_HOST=$SYNC_HOST ROSE_SYNC_PORT=$PORT PYTHONPATH="$ROSE/deploy/hephaestus" \
  "$ISAAC_PY" -u run_sync_only.py --yaml_path "$ROSE/deploy/config/config_gym_WarehouseThrustEnv-v0.yaml" ) > "$SYNC_LOG" 2>&1 &
SYNC_PID=$!
for i in $(seq 1 600); do
  grep -q "listening on" "$SYNC_LOG" 2>/dev/null && { say "sync listening after ~${i}s"; break; }
  kill -0 $SYNC_PID 2>/dev/null || { say "SYNC DIED"; tail -20 "$SYNC_LOG"; exit 2; }
  sleep 1
done
grep -q "listening on" "$SYNC_LOG" || { say "sync boot timeout"; exit 3; }

say "infrasetup LOCAL U250 (program TACIT bitstream)..."
if fsim firesim infrasetup -c "$RUNTIME" > "$INFRA_LOG" 2>&1; then say "infrasetup OK"; else say "infrasetup FAILED"; tail -25 "$INFRA_LOG"; exit 4; fi

say "runworkload (TACIT guest boots, traces, exits)..."
fsim firesim runworkload -c "$RUNTIME" > "$RW_LOG" 2>&1 &
RW_BG=$!
# find the live uartlog + watch for Hello World / sim end
ULIVE=""; START=$(date +%s)
while true; do
  el=$(( $(date +%s) - START ))
  [ -z "$ULIVE" ] && ULIVE=$(find "$SIMDIR" -name uartlog -newermt '-3 min' 2>/dev/null | head -1)
  hw=$(grep -aic 'Hello World' "$ULIVE" 2>/dev/null || echo 0)
  # runworkload returns when the sim ends (guest reboot/HTIF)
  if ! kill -0 $RW_BG 2>/dev/null; then say "runworkload finished at ~${el}s"; break; fi
  [ $el -ge 600 ] && { say "timeout 600s — ending"; break; }
  [ $((el % 20)) -eq 0 ] && say "  t+${el}s  hello_world=$hw  uartlog=${ULIVE:-<none>}"
  sleep 5
done
say "=== harvest ==="
[ -n "$ULIVE" ] && { echo "--- uartlog tail ---"; tail -15 "$ULIVE" 2>/dev/null | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g'; }
TOUT=$(find "$SIMDIR" -name 'tacit0.out' -newermt '-15 min' 2>/dev/null | head -1)
if [ -n "$TOUT" ]; then cp "$TOUT" "$SP/tacit0.out"; say "TACIT trace extracted: $SP/tacit0.out ($(stat -c%s "$SP/tacit0.out") bytes)"; else say "NO tacit0.out found under $SIMDIR"; find "$SIMDIR" -name 'tacit*' 2>/dev/null | head; fi
