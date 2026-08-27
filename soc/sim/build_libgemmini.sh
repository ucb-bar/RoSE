#!/usr/bin/env bash
# Build the spike Gemmini model (libgemmini.so) for a given Gemmini shape.
#
# WHY THIS EXISTS
# ---------------
# spike's Gemmini model requantizes with the ACC_SCALE macro straight out of
# gemmini_params.h (libgemmini/gemmini.cc:1849 -- `acc_t scaled = ACC_SCALE(value, scale)`).
# So the header libgemmini is COMPILED against silently decides the model's numerics.
# Build it against an f32-acc_scale header while the RTL elaborates a Q0.31 config and
# spike and hardware disagree with no error anywhere -- exactly the class of bug the
# Q0.31 work exists to remove.
#
# This script keeps the two in lockstep: the header is generated from the *same*
# GemminiArrayConfig the RTL elaborates from (gemmini.GemminiHeaderGen calls the pure
# config.generateHeader(), no FIRRTL needed -- seconds, not minutes), then libgemmini.so
# is rebuilt against it.  Works for ANY gemmini shape, not just the ones with a named
# config class.
#
# USAGE
#   soc/sim/build_libgemmini.sh [opts]
#
#   --base <name>          gemmini base config. default: q31ws
#                          q31 | q31ws | q31ws_32x32 | q31ws_32x32_acc | q31_32x32_both
#                          | default | lean   (last two are the stock f32 configs)
#   --mesh <N>             override meshRows=meshColumns=N  (any dimension)
#   --sp-capacity <KB>     override scratchpad capacity
#   --acc-capacity <KB>    override accumulator capacity
#   --dma-buswidth <bits>  override DMA bus width
#   --dataflow <WS|OS|BOTH>
#   --header <path>        skip generation, use this header verbatim
#   --name <label>         snapshot dir name (default: derived from base+mesh)
#   --install              also copy the .so into $RISCV/lib (what spike loads by default)
#   --keep-staged          leave the generated header staged in libgemmini/ (default:
#                          restore the submodule's original so it stays clean)
#
# EXAMPLES
#   # 16x16 Q0.31 WS -- matches RoseTLDualRocketSaturnGemminiQ31WsConfig
#   soc/sim/build_libgemmini.sh --base q31ws --install
#
#   # arbitrary shape
#   soc/sim/build_libgemmini.sh --base q31ws --mesh 32 --acc-capacity 128 --dma-buswidth 256
#
#   # A/B against the stock f32 acc_scale model
#   soc/sim/build_libgemmini.sh --base default --name fp32_16x16
#
# Snapshots land in soc/sim/gemmini_models/<name>/{gemmini_params.h,libgemmini.so,meta.txt}
# so you can keep several shapes side by side and point spike at one via its lib path.
set -euo pipefail

ROSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHIPYARD="${ROSE_ROOT}/soc/sim/chipyard"
LIBGEMMINI="${CHIPYARD}/generators/gemmini/software/libgemmini"
SNAP_ROOT="${ROSE_ROOT}/soc/sim/gemmini_models"

BASE="q31ws"; MESH=""; SPCAP=""; ACCCAP=""; DMABUS=""; DATAFLOW=""
HEADER_OVERRIDE=""; NAME=""; DO_INSTALL=""; KEEP_STAGED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)          BASE="$2"; shift 2 ;;
    --mesh)          MESH="$2"; shift 2 ;;
    --sp-capacity)   SPCAP="$2"; shift 2 ;;
    --acc-capacity)  ACCCAP="$2"; shift 2 ;;
    --dma-buswidth)  DMABUS="$2"; shift 2 ;;
    --dataflow)      DATAFLOW="$2"; shift 2 ;;
    --header)        HEADER_OVERRIDE="$2"; shift 2 ;;
    --name)          NAME="$2"; shift 2 ;;
    --install)       DO_INSTALL=1; shift ;;
    --keep-staged)   KEEP_STAGED=1; shift ;;
    -h|--help)       sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; $d'; exit 0 ;;
    *) echo "build_libgemmini: unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$NAME" ]] && NAME="${BASE}${MESH:+_${MESH}x${MESH}}"
SNAP_DIR="${SNAP_ROOT}/${NAME}"
mkdir -p "${SNAP_DIR}"

# chipyard env.sh sets RISCV (libgemmini's Makefile requires it) and puts the
# conda JDK 20 on PATH.  NOTE: source it WITHOUT a pipe -- piping subshells it, PATH
# never applies, sbt then runs on system Java 21 and Scala 2.12.17 dies with
# "bad constant pool index".  set +u because chipyard's conda hook derefs unset vars.
set +u
# shellcheck disable=SC1091
source "${CHIPYARD}/env.sh" >/dev/null 2>&1 || true
set -u
if [[ -z "${RISCV:-}" ]]; then
  echo "build_libgemmini: FATAL: RISCV unset after sourcing ${CHIPYARD}/env.sh" >&2; exit 1
fi
echo "[libgemmini] RISCV=${RISCV}"

# ---- 1. obtain the header -------------------------------------------------
GEN_HEADER="${SNAP_DIR}/gemmini_params.h"
if [[ -n "${HEADER_OVERRIDE}" ]]; then
  echo "[libgemmini] using supplied header: ${HEADER_OVERRIDE}"
  cp "${HEADER_OVERRIDE}" "${GEN_HEADER}"
else
  ARGS=(--base "${BASE}" --out "${GEN_HEADER}")
  [[ -n "$MESH"     ]] && ARGS+=(--mesh "$MESH")
  [[ -n "$SPCAP"    ]] && ARGS+=(--sp-capacity "$SPCAP")
  [[ -n "$ACCCAP"   ]] && ARGS+=(--acc-capacity "$ACCCAP")
  [[ -n "$DMABUS"   ]] && ARGS+=(--dma-buswidth "$DMABUS")
  [[ -n "$DATAFLOW" ]] && ARGS+=(--dataflow "$DATAFLOW")
  echo "[libgemmini] generating header: ${ARGS[*]}"
  ( cd "${CHIPYARD}" && sbt -batch "gemmini/runMain gemmini.GemminiHeaderGen ${ARGS[*]}" ) \
    | grep -E 'gemmini-header-gen|\[error\]'
fi
[[ -f "${GEN_HEADER}" ]] || { echo "build_libgemmini: header not produced" >&2; exit 1; }

# ---- 2. stage it into libgemmini and rebuild ------------------------------
ORIG_HEADER="$(mktemp)"
cp "${LIBGEMMINI}/gemmini_params.h" "${ORIG_HEADER}"
restore_header() {
  if [[ -z "${KEEP_STAGED}" ]]; then
    cp "${ORIG_HEADER}" "${LIBGEMMINI}/gemmini_params.h"
    rm -f "${ORIG_HEADER}"
  fi
}
trap restore_header EXIT

cp "${GEN_HEADER}" "${LIBGEMMINI}/gemmini_params.h"
echo "[libgemmini] rebuilding libgemmini.so"
( cd "${LIBGEMMINI}" && make clean >/dev/null 2>&1 || true; cd "${LIBGEMMINI}" && make ) \
  || { echo "build_libgemmini: libgemmini build FAILED" >&2; exit 1; }

# ---- 3. snapshot ----------------------------------------------------------
cp "${LIBGEMMINI}/libgemmini.so" "${SNAP_DIR}/libgemmini.so"
{
  echo "name:        ${NAME}"
  echo "base:        ${BASE}"
  echo "overrides:   mesh=${MESH:-<base>} sp=${SPCAP:-<base>} acc=${ACCCAP:-<base>} dma=${DMABUS:-<base>} dataflow=${DATAFLOW:-<base>}"
  echo "built:       $(date -Iseconds)"
  echo "chipyard:    $(cd "${CHIPYARD}" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "libgemmini:  $(cd "${LIBGEMMINI}" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "--- header key defines ---"
  grep -E '^#define DIM|^typedef .*(elem_t|acc_t|acc_scale_t);|ACC_SCALE_T_IS_FLOAT|^#define ACC_SCALE_IDENTITY' "${GEN_HEADER}" || true
} > "${SNAP_DIR}/meta.txt"

echo "[libgemmini] snapshot -> ${SNAP_DIR}/"
sed -n '/--- header key defines ---/,$p' "${SNAP_DIR}/meta.txt"

# ---- 4. optional install --------------------------------------------------
if [[ -n "${DO_INSTALL}" ]]; then
  cp "${LIBGEMMINI}/libgemmini.so" "${RISCV}/lib/libgemmini.so"
  echo "[libgemmini] installed -> ${RISCV}/lib/libgemmini.so"
fi
echo "[libgemmini] OK"
