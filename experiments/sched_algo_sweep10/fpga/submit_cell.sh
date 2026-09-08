#!/usr/bin/env bash
# Dispatch ONE built sched_algo_sweep10 ELF to the fq pool and pull its uartlog
# back. Same shape as experiments/workload_gen/submit_workload.sh -- only the
# elf/ and res_/ roots differ.
#
#   submit_cell.sh <tag>          e.g. submit_cell.sh s10_control_mix_gempair_base_cpsat
#
# NOT run by the build pass. The build pass never submits.
set -uo pipefail
TAG=$1
R=/scratch/dima/rose-infra/RoSE; OUT=$R/experiments/sched_algo_sweep10
mkdir -p $OUT/fpga/logs $OUT/res
exec > $OUT/fpga/logs/fq_$TAG.log 2>&1
MGR=${FQ_MGR:-ubuntu@3.88.218.39}; KEY=${FQ_KEY:-~/.ssh/firesim.pem}
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY)
HW=${FQ_HW:-f2_quad_hetero_norose_tacit_q31_60mhz}
echo "### submit $TAG $(date -u +%FT%TZ)"
scp -q -o BatchMode=yes -i $KEY "$OUT/elf/$TAG.elf" $MGR:/home/ubuntu/$TAG.elf \
  || { echo "ABORT scp"; exit 1; }
"${SSH[@]}" $MGR "rm -rf /home/ubuntu/s10res_$TAG && mkdir -p /home/ubuntu/s10res_$TAG" </dev/null
O=$("${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; timeout 60 ./bin/fq submit \
  --tree /home/ubuntu/chipyard-rose --hw-config $HW --elf /home/ubuntu/$TAG.elf \
  --timeout 5400 --results /home/ubuntu/s10res_$TAG 2>&1 | head -3" </dev/null)
echo "#### submit: $O"
JID=$(grep -oE 'job [0-9]+' <<<"$O" | grep -oE '[0-9]+' | head -1)
echo "#### jid=$JID"; [ -z "$JID" ] && { echo "ABORT no jid"; exit 1; }
for i in $(seq 1 400); do
  sleep 20
  "${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq status 2>/dev/null" </dev/null \
    | grep -qE "^ *$JID " || break
done
mkdir -p $OUT/res/$TAG
scp -q -o BatchMode=yes -i $KEY -r $MGR:/home/ubuntu/s10res_$TAG/. $OUT/res/$TAG/ 2>/dev/null
UL=$(find $OUT/res/$TAG -name uartlog | head -1); echo "#### uartlog=${UL:-NONE}"
# fq copies from the run host's sim_slot_*/ and those survive between jobs, so a
# cell can silently collect the PREVIOUS job's uartlog. The embedded schedule
# tag is the only thing that tells you which ELF actually ran.
if [ -n "$UL" ]; then
  S=$(grep -oE 'xpurt-runner: schedule=[^ ]+' "$UL" | head -1 | cut -d= -f2)
  [ "$S" = "$TAG" ] && echo "#### identity OK ($S)" \
    || echo "#### IDENTITY MISMATCH: uartlog says '$S', expected '$TAG'"
  N=$(python3 -c "
import json,glob
p=[x for x in json.load(open('$OUT/fpga/elf_plan.json')) if x['tag']=='$TAG']
print(p[0]['dispatches'] if p else -1)")
  echo "#### schedule dispatches=$N"
  grep -E "entries=|MODELBLASTER_VERIFY|PASSED|FAILED" "$UL" | head -4
fi
echo "############ FQDONE $TAG ############"
