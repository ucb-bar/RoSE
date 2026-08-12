#!/usr/bin/env bash
# ===========================================================================
# run_ablation.sh — fully-automated RoSE co-sim throughput ablation harness.
#
# Ablates every simulation-throughput architectural lever we tuned, as a proper
# ablation (baseline all-old -> single-lever flips -> all-new) PLUS a granularity
# sweep. Splits RUNNABLE-NOW rows (no FPGA/GPU) from FPGA/GPU-GATED rows.
#
# LEVERS (each an independent toggle, no rebuild needed except the datapath bitstream):
#   1 sync granularity   ROSE_FIRESIM_STEP  {1M,5M,10M,50M,100M,500M}
#   2 socket poll fix     ROSE_SYNC_RECV_TIMEOUT  {0.1 old, 0.001 new}   <- RUN NOW (spike)
#   3 lazy camera render  ROSE_RENDER_HZ {0 old,10 new} + ROSE_ISAAC_CAMERA <- GPU-gated
#   4 datapath            MMIO-Saturn vs DMA-Saturn bitstream            <- FPGA-gated
#   5 sync mode           free-running (minimal_sync) vs barrier (run_sync_only)
#   6 environment         PatternEnv-v0 (camera-free) vs WarehouseThrustEnv-v0 (Isaac)
#
# MODES
#   --now       (default) run the socket-fix barrier axis on the functional Spike
#               bridge, then assemble results/ablation_results.csv + report data.
#   --render    run the Isaac-standalone env.step render timing (GPU). Auto-skips
#               (and keeps the measured-prior numbers) if a GPU job is active.
#   --fpga      run the on-silicon effective-MHz configs (granularity sweep, socket-
#               fixed, MMIO-vs-DMA datapath) on firesim1/garden. GUARDED: prints the
#               exact recipe and only executes with ROSE_ALLOW_FPGA=1.
#   --analyze   just re-run the analysis (no measurement).
#   --all       --now + --render + --fpga (each still individually guarded).
#
# Re-runnable from a clean state: measured-now rows regenerate; measured-prior and
# computed rows come from data/measured_prior.json (analyze.py labels every source).
# ===========================================================================
set -u
ROSE_ROOT="${ROSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ROSE_ROOT
BUNDLE="$ROSE_ROOT/experiments/rose_throughput_ablation"
PY3="${PY3:-python3}"
MODE="${1:---now}"

banner(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
run_now(){
  banner "LEVER 2: socket-fix barrier axis  (Spike bridge, no FPGA/GPU)"
  # PatternEnv-v0 camera-free + reqrsp guest; firesim_step=1000 so per-grant wall ~= pure barrier.
  ROSE_ROOT="$ROSE_ROOT" TAG=newsock PORT="${PORT_NEW:-10071}" \
    ROSE_SYNC_RECV_TIMEOUT=0.001 MAXG="${MAXG:-40}" TIMEOUT_S=90 \
    bash "$BUNDLE/lib/spike_barrier.sh" || echo "[run_now] newsock point failed (non-fatal)"
  ROSE_ROOT="$ROSE_ROOT" TAG=oldsock PORT="${PORT_OLD:-10072}" \
    ROSE_SYNC_RECV_TIMEOUT=0.1  MAXG="${MAXG:-40}" TIMEOUT_S=120 \
    bash "$BUNDLE/lib/spike_barrier.sh" || echo "[run_now] oldsock point failed (non-fatal)"
  echo "[run_now] seam CSVs -> $BUNDLE/run_out/seam_{old,new}sock.csv"
}

# ---------------------------------------------------------------------------
gpu_busy(){
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  # busy if any compute process is resident OR utilization/mem is non-trivial
  local m u
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${m:-0}" -gt 4000 ] || [ "${u:-0}" -gt 30 ]
}

run_render(){
  banner "LEVER 3: lazy camera render axis  (Isaac-standalone env.step, GPU)"
  # Default is DEFER: the GPU is shared with a live capstone flight, so we do NOT
  # launch a competing Isaac session unless explicitly authorized AND the GPU is free.
  if [ "${ROSE_ALLOW_GPU:-0}" != 1 ] || gpu_busy || [ -z "${ISAAC_PY:-}" ]; then
    echo "[run_render] DEFERRED — keeping measured-prior env.step numbers (31.0 / 8.1 / 6.06 ms)"
    echo "[run_render]   from data/measured_prior.json (camera_render_timing.md direct GPU timing)."
    echo "[run_render]   To run later on a FREE GPU:  ISAAC_PY=/path/to/env_isaaclab/bin/python ROSE_ALLOW_GPU=1 $0 --render"
    return 0
  fi
  echo "[run_render] would time WarehouseThrustEnv-v0 env.step at ROSE_RENDER_HZ in {0,10} + physics floor."
  echo "[run_render]   ROSE_RENDER_HZ=0  ROSE_ISAAC_CAMERA=1 $ISAAC_PY <time_env.py render1>   # old"
  echo "[run_render]   ROSE_RENDER_HZ=10                      $ISAAC_PY <time_env.py lazy>      # new"
  echo "[run_render] (uses the committed lazy-render toggle in warehouse_thrust_env.py; b4c946c)"
  # The scratchpad time_env.py harness produced the committed numbers; wire it here when a GPU is free.
}

# ---------------------------------------------------------------------------
run_fpga(){
  banner "LEVERS 1/4/5: on-silicon effective-MHz  (FPGA — SCRIPTED, guarded)"
  local HWDB_MMIO="alveo_u250_firesim-rocket-saturn-with-rose-fast"
  local CFG_DMA="RoseTLRocketSaturnDMAMMIOOnlyConfig"
  cat <<EOF
[run_fpga] These configs need the U250 FPGA. They are SCRIPTED but NOT run here
           (garden's FPGA is live with a capstone flight; do NOT contend).

  Free-running ceiling (sync_mode=free):
    deploy/hephaestus/minimal_sync.py            # continuous grant, no barrier -> 59.9 MHz

  Granularity sweep (sync_mode=barrier, MMIO Saturn, old socket):
    for STEP in 1000000 5000000 10000000 50000000 100000000 500000000; do
      ROSE_FIRESIM_STEP=\$STEP ROSE_SYNC_RECV_TIMEOUT=0.1 \\
        python deploy/hephaestus/run_sync_only.py --yaml_path config_gym_PatternEnv-v0.yaml &
      firesim runworkload   # bench:spin guest, hwdb=$HWDB_MMIO
    done

  Socket-fixed + datapath (5M step):
    ROSE_SYNC_RECV_TIMEOUT=0.001  hwdb=$HWDB_MMIO       # MMIO 60 MHz
    ROSE_SYNC_RECV_TIMEOUT=0.001  config=$CFG_DMA       # DMA-Saturn (60 MHz building)

  Fill the measured columns later on firesim1/garden:
    ROSE_ALLOW_FPGA=1 $0 --fpga
EOF
  if [ "${ROSE_ALLOW_FPGA:-0}" = 1 ]; then
    echo "[run_fpga] ROSE_ALLOW_FPGA=1 set — but this harness intentionally leaves the"
    echo "[run_fpga] firesim runworkload invocation to the operator (classifier-gated FPGA)."
  fi
}

# ---------------------------------------------------------------------------
analyze(){
  banner "ANALYZE -> results/ablation_results.csv + results/waterfall.json"
  "$PY3" "$BUNDLE/lib/analyze.py"
}

case "$MODE" in
  --now)     run_now; analyze ;;
  --render)  run_render; analyze ;;
  --fpga)    run_fpga; analyze ;;
  --analyze) analyze ;;
  --all)     run_now; run_render; run_fpga; analyze ;;
  *) echo "usage: $0 [--now|--render|--fpga|--analyze|--all]"; exit 2 ;;
esac
banner "DONE ($MODE).  Report: $BUNDLE/report.html"
