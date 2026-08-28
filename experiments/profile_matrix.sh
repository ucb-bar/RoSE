#!/usr/bin/env bash
# profile_matrix.sh -- build N models against ONE kernel set and profile them all
# on the FPGA in a single sweep, emitting the operator x model matrix.
#
# WHY THIS EXISTS
# ---------------
# ModelBlaster ships one kernel per operator, shared by every model. A change to
# `conv2d_s8` lands in dronet, yolov8n and fused_full simultaneously. Measuring
# it on one model and shipping is how `maxpool2d_s8` nearly went out with its
# stride-1 padded path (yolov8n's SPPF) exercised by no model's build at all.
#
# So: an operator owner should never see one model. This runs the whole matrix
# from a single curated kernel set and reports every operator across every model.
#
# USAGE
#   ./profile_matrix.sh --tag mychange --curated /path/to/curated
#   ./profile_matrix.sh --tag base --models "dronet yolov8_nano" --target rvv
#
# OUTPUT
#   experiments/matrix/<tag>/matrix.csv   model,op,instances,cycles,pct_of_model
#   experiments/matrix/<tag>/<model>.uartlog
#   plus a printed operator x model table.
set -uo pipefail

ZCS="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw"
MB="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
# examples/*/run.sh shells out to `python` (generate_skeleton/generate_kernels)
# and needs the in-tree Zephyr SDK on PATH; neither is on a bare login shell's
# PATH. Activate them here so this script is runnable standalone rather than
# failing deep inside stage [2/5] with a bare "python: command not found".
if ! command -v python >/dev/null 2>&1; then
  set +u
  source "${ZCS}/scripts/activate_conda.sh"  >/dev/null 2>&1
  source "${ZCS}/scripts/set_envvars_sdk.sh" >/dev/null 2>&1
  set -u
fi
# `modelblaster` has no __init__.py-based install into the zephyr conda env's
# site-packages (no .pth/egg-link) -- it only resolves as a namespace package
# when its PARENT dir (zephyr-chipyard-sw, i.e. $ZCS) is on sys.path. But
# _run_lib.sh cd's into $MB (REPO_ROOT = the modelblaster dir itself) before
# calling `python -m modelblaster.pipeline...`, so cwd alone doesn't supply
# that parent dir. Confirmed by hand: cwd=$MB + `python -m
# modelblaster.pipeline.generate_skeleton --help` -> ModuleNotFoundError;
# same command with PYTHONPATH=$ZCS -> works. Export it unconditionally
# (idempotent, harmless if already correct).
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
OUT_ROOT="/scratch/dima/rose-infra/RoSE/experiments/matrix"
MGR="ubuntu@3.88.218.39"
KEY="$HOME/.ssh/firesim.pem"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -i "$KEY")
HW="f2_dual_small_norose_tacit_q31_60mhz"   # 2.99x faster than 20mhz, cycle-identical
TREE="/home/ubuntu/chipyard-rose"

TAG=""; CURATED=""; TARGET="rvv"; QUANT="int8"
MODELS="dronet yolov8_nano fused_full mlp_generic"
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)     TAG="$2"; shift 2 ;;
    --curated) CURATED="$2"; shift 2 ;;
    --target)  TARGET="$2"; shift 2 ;;
    --quant)   QUANT="$2"; shift 2 ;;
    --models)  MODELS="$2"; shift 2 ;;
    --hw)      HW="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$TAG" ]] && { echo "--tag is required" >&2; exit 2; }

DEST="$OUT_ROOT/$TAG"
mkdir -p "$DEST"
echo "=== profile_matrix: tag=$TAG target=$TARGET quant=$QUANT hw=$HW"
echo "=== models: $MODELS"
[[ -n "$CURATED" ]] && echo "=== curated: $CURATED" || echo "=== curated: (none -- kernels will be REFERENCE, not curated)"

# ---------------------------------------------------------------- 1. build
declare -A ELF
for m in $MODELS; do
  # _run_lib.sh's BUILD_DIR is always "${TARGET}${BUILD_SUFFIX}" where
  # BUILD_SUFFIX is "_firesim" iff RUNNER=firesim -- it is NEVER promoted
  # to a "_f16" variant on disk even when GEN_TARGET auto-promotes for
  # fused_full's fp16 tail (that promotion only changes which backend
  # generates the kernels, not the build directory name). Mirroring a
  # separate "_f16" path here was wrong and pointed at a directory that
  # never gets created.
  elf="$MB/examples/$m/$QUANT/build/${TARGET}_firesim/zephyr/zephyr.elf"
  if [[ "$SKIP_BUILD" == "1" && -f "$elf" ]]; then
    echo "--- $m: reusing existing ELF"
  else
    echo "--- building $m (target=$TARGET) ..."
    ( cd "$MB" && \
      MODEL_NAME="$m" BACKEND=reference TARGET="$TARGET" QUANT="$QUANT" \
      OPTIMIZE=0 STOP_AFTER=build RUNNER=firesim \
      GLOBAL_CURATED_DIR="${CURATED:-}" \
      bash "examples/$m/run.sh" ) > "$DEST/$m.build.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "    BUILD FAILED (rc=$rc) -- see $DEST/$m.build.log"
      # Compile poisoning downgrades every op sorting before the culprit, so
      # surface it rather than silently profiling a degraded build.
      grep -iE "BUILD failure, not a numeric mismatch|west build failed" \
        "$DEST/$m.build.log" 2>/dev/null | head -2 | sed 's/^/      /'
      continue
    fi
  fi
  [[ -f "$elf" ]] || { echo "    no ELF at $elf"; continue; }
  ELF[$m]="$elf"
  echo "    ok: $(stat -c %s "$elf") bytes"
done

[[ ${#ELF[@]} -eq 0 ]] && { echo "no ELFs built; nothing to profile"; exit 1; }

# ---------------------------------------------------------------- 2. submit
declare -A JOB RES
for m in "${!ELF[@]}"; do
  remote="/home/ubuntu/pm_${TAG}_${m}.elf"
  res="/home/ubuntu/pm_${TAG}_${m}_res"
  scp -q -o BatchMode=yes -i "$KEY" "${ELF[$m]}" "$MGR:$remote" || { echo "scp failed for $m"; continue; }
  "${SSH[@]}" "$MGR" "rm -rf $res && mkdir -p $res" >/dev/null 2>&1
  out=$("${SSH[@]}" "$MGR" "cd ~/fpga_queue && export FQ_SOCKET=/var/lib/fq/fq.sock && \
        timeout 60 ./bin/fq submit --tree $TREE --hw-config $HW --elf $remote \
        --timeout 3000 --results $res 2>&1 | head -1")
  id=$(grep -oE 'job [0-9]+' <<<"$out" | grep -oE '[0-9]+' | head -1)
  JOB[$m]="$id"; RES[$m]="$res"
  echo "--- submitted $m -> job ${id:-?}"
  sleep 12   # stagger: same-second dispatch onto lanes still tearing down has raced before
done

# ---------------------------------------------------------------- 3. wait
echo "=== waiting for ${#JOB[@]} jobs ..."
for i in $(seq 1 180); do
  pending=0
  for m in "${!JOB[@]}"; do
    st=$("${SSH[@]}" "$MGR" "grep -E 'job ${JOB[$m]} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1")
    [[ -z "$st" ]] && pending=$((pending+1))
  done
  [[ $pending -eq 0 ]] && break
  sleep 20
done

# ---------------------------------------------------------------- 4. collect
: > "$DEST/matrix.csv"
echo "model,op,instances,cycles,pct_of_model" >> "$DEST/matrix.csv"
for m in "${!JOB[@]}"; do
  verdict=$("${SSH[@]}" "$MGR" "grep -E 'job ${JOB[$m]} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1")
  echo "--- $m: ${verdict:-NO TERMINAL STATE}"
  case "$verdict" in *"DONE rc=0"*) ;; *) echo "    skipping (not DONE)"; continue ;; esac

  "${SSH[@]}" "$MGR" "find ${RES[$m]} -name uartlog | head -1 | xargs -r cat" > "$DEST/$m.uartlog" 2>/dev/null
  [[ -s "$DEST/$m.uartlog" ]] || { echo "    no uartlog"; continue; }

  # Provenance: job ids are reused across daemon restarts, so verify the ELF
  # name FireSim embeds in the guest command line AND the harness model line.
  gotelf=$(grep -ao 'pm_[A-Za-z0-9_]*\.elf' "$DEST/$m.uartlog" | head -1)
  gotmodel=$(grep -o 'harness: model=[a-z0-9_]*' "$DEST/$m.uartlog" | head -1 | cut -d= -f2)
  if [[ "$gotelf" != "pm_${TAG}_${m}.elf" || "$gotmodel" != "$m" ]]; then
    echo "    PROVENANCE MISMATCH: elf='$gotelf' model='$gotmodel' -- DISCARDING"
    continue
  fi
  awk -F, -v M="$m" '
    /^[0-9]+,/ && NF>=5 { agg[$3]+=$NF; n[$3]++; tot+=$NF }
    END { for (o in agg) printf "%s,%s,%d,%d,%.2f\n", M, o, n[o], agg[o], 100*agg[o]/tot }
  ' "$DEST/$m.uartlog" | sort -t, -k4 -rn >> "$DEST/matrix.csv"
  echo "    ok: $(awk -F, '/^[0-9]+,/ && NF>=5 {t+=$NF} END{printf "%d", t}' "$DEST/$m.uartlog") cycles"
done

# ---------------------------------------------------------------- 5. report
echo
echo "=== OPERATOR x MODEL (cycles) ==="
python3 - "$DEST/matrix.csv" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("  (no data)"); raise SystemExit
models = sorted({r["model"] for r in rows})
ops    = sorted({r["op"] for r in rows})
cyc = collections.defaultdict(dict)
for r in rows: cyc[r["op"]][r["model"]] = int(r["cycles"])
w = max(len(o) for o in ops) + 2
print("  " + "op".ljust(w) + "".join(m.rjust(16) for m in models))
tot = collections.defaultdict(int)
for o in ops:
    line = "  " + o.ljust(w)
    for m in models:
        v = cyc[o].get(m)
        line += (f"{v:,}".rjust(16) if v is not None else "-".rjust(16))
        if v: tot[m] += v
    print(line)
print("  " + "-" * (w + 16*len(models)))
print("  " + "TOTAL".ljust(w) + "".join(f"{tot[m]:,}".rjust(16) for m in models))
PY
echo
echo "=== wrote $DEST/matrix.csv"
