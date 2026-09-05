#!/usr/bin/env bash
# Drive the wl_sweep families that the micro-ROS harness CAN express.
#
# Builds are strictly SERIAL: build_microros.sh wipes the inner colcon state of
# the shared libmicroros submodule regardless of BUILD_DIR, so two concurrent
# builds corrupt each other. Runs go one at a time behind them.
#
# Periods come from soc/sw/xpu-rt/data/toplevel/wl_sweep/networks_<fam>_*.json,
# rounded to the harness's integer-millisecond timer. The one-shot network is
# listed FIRST so it binds to NET_A, matching the harness convention, and is
# pinned to the gemmini hart; the periodic ones share an rvv hart, which is the
# configuration that exercises co-scheduling.
set -uo pipefail
R=/scratch/dima/rose-infra/RoSE
MGR=ubuntu@3.88.218.39; KEY=~/.ssh/firesim.pem
HW=f2_quad_hetero_norose_tacit_q31_60mhz_pcim
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=25 -i $KEY)

run_family() {
  local tag=$1 models=$2 quants=$3 pinbs=$4 harts=$5 periods=$6 to=${7:-2400}
  echo "########## $tag  models=$models periods=$periods  $(date -u +%FT%TZ)"
  MODELS=$models QUANTS=$quants PIN_BACKENDS=$pinbs PIN_HARTS=$harts \
  PERIODS_MS=$periods MICROROS_2EXEC_BC=0 MICROROS_2EXEC_FUSE_BC=0 \
    bash $R/experiments/microros/build_microros.sh "$tag"
  local st; st=$(grep -oE 'BUILDDONE|### ABORT.*' $R/experiments/microros/logs/build_$tag.log 2>/dev/null | tail -1)
  echo "  build=$st"
  [ "$st" = BUILDDONE ] || { grep -oE "error: .{0,80}" $R/experiments/microros/logs/build_$tag.log | sort -u | head -3; return 1; }

  # pick a free lane that is not the degraded f2-07
  local lane host
  read -r lane host < <("${SSH[@]}" $MGR 'cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq lanes 2>/dev/null | awk "\$4==\"FREE\" && \$1!=\"f2-07\" {print \$1, \$NF; exit}"' </dev/null 2>/dev/null)
  [ -z "${lane:-}" ] && { echo "  no free lane; waiting"; sleep 120; read -r lane host < <("${SSH[@]}" $MGR 'cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq lanes 2>/dev/null | awk "\$4==\"FREE\" && \$1!=\"f2-07\" {print \$1, \$NF; exit}"' </dev/null 2>/dev/null); }
  [ -z "${lane:-}" ] && { echo "  STILL no free lane, skipping $tag"; return 1; }
  echo "  lane=$lane host=$host"
  "${SSH[@]}" $MGR "mkdir -p /home/ubuntu/r/$tag" </dev/null
  scp -q -o BatchMode=yes -o StrictHostKeyChecking=no -i $KEY \
      $R/experiments/microros/elf/$tag.elf $MGR:/home/ubuntu/r/$tag/z.elf || { echo "  scp FAILED"; return 1; }
  "${SSH[@]}" $MGR "bash /home/ubuntu/f2_pcim_tacit_run.sh $tag $lane $host $HW $to" </dev/null \
      > $R/experiments/microros/logs/run_$tag.log 2>&1
  echo "  --- $tag result ---"
  "${SSH[@]}" $MGR "grep -E '^\[[a-z0-9_]+\] done|TRACE_MAGIC_AUDIT|\*\*\* PASSED|mcause: [0-9]+' /home/ubuntu/r/$tag/uartlog 2>/dev/null | head -8" </dev/null
}

run_family perception_heavy yolov8_nano_sf,mlp_control_sd int8,fp32 gemmini_q31,rvv 0,2 0,4
run_family saturation      yolov8_nano_se,dronet_sf,mlp_control_sf int8,int8,fp32 gemmini_q31,rvv,rvv 0,2,2 0,19,3
run_family vint_intro      vint,mlp_control_sf int8,fp32 gemmini_q31,rvv 0,2 0,13 3000
run_family vint_multi      vint,dronet_se,mlp_control_sd int8,int8,fp32 gemmini_q31,rvv,rvv 0,2,2 0,63,3 3000
echo "########## WL FAMILY SWEEP DONE ##########"
