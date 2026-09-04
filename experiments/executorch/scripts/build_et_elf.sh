#!/usr/bin/env bash
# Build ONE ExecuTorch Zephyr ELF carrying N baked .pte models, targeted at
# f2_quad_hetero_norose_tacit_q31_60mhz (the bitstream the ModelBlaster sweeps
# in experiments/sweep3net ran on, so the cycle counts are comparable).
#
#   build_et_elf.sh <tag> <model1> [model2 ...]
#     model = a tag under experiments/executorch/pte/ (e.g. dronet_int8)
#
# Env knobs:
#   MB_XNN_PROFILE  OFF (default) = clean end-to-end timing;
#                   ON  = per-operator `>>, <op>, <cycles>` lines. Those are
#                   clean rdcycle deltas, but each one is an HTIF console write
#                   (~1e6 cycles on FireSim) INSIDE the execute() bracket, so
#                   the per-model total from a profiling build is worthless.
#                   Build both; take totals from OFF, per-op from ON.
#   MB_EXEC_ITERS   execute() repeats (default 3: iter0 cold, rest warm).
#   MB_ET_POOL_MB   ExecuTorch method_allocation_pool (default 256).
#   PRISTINE=1      force a from-scratch build (~15 min); default is incremental.
#
# Each build dir gets its OWN ExecuTorch cmake-out (-DET_BUILD_DIR_PATH). The
# sample's default puts it INSIDE the source tree (third-party/executorch/
# cmake-out), which every build dir then shares -- so a profiling build and a
# clean build silently overwrite each other's ExecuTorch/XNNPACK objects and
# you link whichever ran last. That is invisible in the log and changes the
# measurement.
#
# Toolchain split, deliberate and load-bearing:
#   * west / cmake / ninja / Zephyr SDK come from the RoSE in-tree env, so the
#     ELF is built against RoSE's zephyr (rose-2-dev) and its board defs.
#   * `flatc` and -DPYTHON_EXECUTABLE come from a SEPARATE ExecuTorch env
#     (executorch 1.0.1 / torch 2.9). The RoSE in-tree env is torch 2.13 with no
#     executorch, and executorch 1.0.1 pins torch>=2.9,<2.10 -- it MUST NOT be
#     installed there; every ModelBlaster build in the repo depends on that env.
set -uo pipefail
TAG=${1:?tag}; shift
R=/scratch/dima/rose-infra/RoSE
ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw
E=$R/experiments/executorch
SDIR=$ZCS/samples/executorch
ETENV=/scratch2/dima/misc_sw/XPU-RT/zephyr-chipyard-sw/tools/miniforge3/envs/zephyr
BD=${BUILD_DIR:-/scratch/dima/et_rose_build/$TAG}
mkdir -p "$E/elf" "$E/logs" "$(dirname "$BD")"
LOG=$E/logs/build_$TAG.log
exec > "$LOG" 2>&1
echo "### build $TAG models=$* $(date -u +%FT%TZ)"

set +u
source $ZCS/scripts/activate_conda.sh
source $ZCS/scripts/set_envvars_sdk.sh
set -u
export PATH="$PATH:$ETENV/bin"        # appended: only `flatc` is taken from here
ET_PY=$ETENV/bin/python
export TMPDIR=${TMPDIR:-/scratch/dima/et_rose_build/tmp}; mkdir -p "$TMPDIR"

# --- bake the .pte files + their REAL inputs into one translation unit -------
MARGS=()
for m in "$@"; do
  p=$E/pte/$m.pte; io=$E/pte/$m.io.npz
  [ -f "$p" ] || { echo "ABORT: no $p"; exit 1; }
  MARGS+=(--model "$m=$p")
  [ -f "$io" ] && MARGS+=(--io "$m=$io")
done
"$ET_PY" "$SDIR/model/gen_multi_pte.py" \
  --out "$SDIR/executor_runner/models_pte.c" \
  --header "$SDIR/executor_runner/models_pte.h" "${MARGS[@]}" \
  || { echo "ABORT gen_multi_pte"; exit 1; }

# ram0 = 1 GiB. The default DT ram0 cannot hold a 256 MB ET pool, and the
# failure mode is a mid-execute allocation fault rather than a link error.
OVL=$BD.ram0.overlay; mkdir -p "$(dirname "$OVL")"
echo '&ram0 { reg = < 0x80000000 0x40000000 >; };' > "$OVL"

P=""; { [ "${PRISTINE:-0}" = 1 ] || [ ! -f "$BD/build.ninja" ]; } && P="-p"
west build $P -b chipyard_riscv64/rocketchip_virt_riscv64 "$SDIR/executor_runner/" \
  --build-dir "$BD" -- \
  -DET_BUILD_DIR_PATH="$BD/et-cmake-out" \
  -DMB_MULTI_MODEL=ON \
  -DMB_XNN_PROFILE=${MB_XNN_PROFILE:-OFF} \
  -DMB_EXEC_ITERS=${MB_EXEC_ITERS:-3} \
  -DMB_ET_POOL_MB=${MB_ET_POOL_MB:-256} \
  -DMB_ET_PIN_HART=2 -DMB_ET_THREADS=1 -DMB_XNN_VLENB=${MB_XNN_VLENB:-32} \
  -DEXECUTORCH_LOG_LEVEL=${ET_LOG_LEVEL:-Error} \
  -DMB_ET_SAMPLE_OUTPUT=${MB_ET_SAMPLE_OUTPUT:-OFF} -DMB_ET_SAMPLE_N=${MB_ET_SAMPLE_N:-64} \
  -DXNNPACK_ENABLE_RISCV_VECTOR=ON -DXNNPACK_ENABLE_RISCV_GEMMINI=OFF \
  -DPYTHON_EXECUTABLE="$ET_PY" \
  -DEXTRA_CONF_FILE="$SDIR/executor_runner/firesim_quad_hetero.conf" \
  -DEXTRA_DTC_OVERLAY_FILE="$OVL"
RC=$?
echo "#### west rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|Error|undefined reference" "$LOG" | tail -20; echo "#### ABORT build"; exit 1; }

# --- gates. Every one of these has a silent failure mode. -------------------
# Read them from the BUILD ARTIFACTS, not the log: an incremental build does not
# re-run cmake configure, so the "Merged configuration"/message(STATUS) lines are
# absent from a rerun even though the settings are in force.
grep -q "^CONFIG_MP_MAX_NUM_CPUS=4$" "$BD/zephyr/.config" || {
  echo "#### ABORT MP_MAX_NUM_CPUS != 4 -- Zephyr SMP boot spinwaits for every configured hart"; exit 1; }
grep -q "^CONFIG_RISCV_ISA_EXT_V=y$" "$BD/zephyr/.config" || {
  echo "#### ABORT V extension off -- XNNPACK RVV kernels would trap at the first vector op"; exit 1; }
APPCMD=$(python3 -c "
import json,sys
d=json.load(open('$BD/compile_commands.json'))
print(next((e['command'] for e in d if e['file'].endswith('riscv_executor_runner.cpp')), ''))" 2>/dev/null)
grep -q "DMB_ET_PIN_HART=2" <<<"$APPCMD" || {
  echo "#### ABORT MB_ET_PIN_HART not applied -- ET would run on a Gemmini-only hart"; exit 1; }
grep -q "DMB_ET_THREADS=1" <<<"$APPCMD" || {
  echo "#### ABORT MB_ET_THREADS not applied -- pool workers would land on harts 0/1"; exit 1; }
# The delegated path must actually be XNNPACK-with-RVV, not the scalar fallback.
grep -q "XNN_ENABLE_RISCV_VECTOR=1" "$BD/compile_commands.json" || {
  echo "#### ABORT XNN_ENABLE_RISCV_VECTOR is not 1 -- this would silently measure scalar XNNPACK"; exit 1; }
grep -q "XNN_RISCV_VLENB=" "$BD/compile_commands.json" || {
  echo "#### ABORT XNN_RISCV_VLENB unset -- XNNPACK would probe VLEN with vsetvli in a static"
  echo "####       initializer, which runs on boot hart 0 (no vector unit) and traps"; exit 1; }
# The RVV micro-kernels are compiled per-source at -march=rv64gcv; if the option
# were off the ELF would still link and run, just scalar and ~silently slower.
if ! ${CROSS_COMPILE:-riscv64-zephyr-elf-}objdump -d "$BD/zephyr/zephyr.elf" 2>/dev/null \
      | grep -qE "^\s+[0-9a-f]+:\s+[0-9a-f ]+\s+v(setvli|setivli|le8|le32|se8|se32|fmacc|mul|add|nclip)"; then
  echo "#### WARN could not confirm vector instructions in the ELF (objdump)"
fi
cp "$BD/zephyr/zephyr.elf" "$E/elf/$TAG.elf"
echo "#### stashed $E/elf/$TAG.elf $(stat -c%s "$E/elf/$TAG.elf") bytes"
echo "############ BUILDDONE $TAG ############"
