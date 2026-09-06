#!/usr/bin/env bash
# Submit ONE quad_debug ELF to an fq lane and pull the uartlog back.
#   submit.sh <tag> <elf>
# Everything it writes stays under experiments/quad_debug/ -- the wl_sweep
# tree is owned by another campaign that is running right now.
# Remote paths are prefixed qd_ / qdres_ for the same reason.
set -uo pipefail
TAG=$1; ELF=$2
R=/scratch/dima/rose-infra/RoSE; OUT=$R/experiments/quad_debug
mkdir -p $OUT/logs $OUT/res_$TAG
exec > $OUT/logs/fq_$TAG.log 2>&1
MGR=ubuntu@3.88.218.39; KEY=~/.ssh/firesim.pem
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY)
HW=f2_quad_hetero_norose_tacit_q31_60mhz
echo "### submit $TAG $(date -u +%FT%TZ) elf=$ELF"
scp -q -o BatchMode=yes -i $KEY "$ELF" $MGR:/home/ubuntu/qd_$TAG.elf || { echo "ABORT scp"; exit 1; }
"${SSH[@]}" $MGR "rm -rf /home/ubuntu/qdres_$TAG && mkdir -p /home/ubuntu/qdres_$TAG" </dev/null
O=$("${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; timeout 60 ./bin/fq submit \
  --tree /home/ubuntu/chipyard-rose --hw-config $HW --elf /home/ubuntu/qd_$TAG.elf \
  --timeout 5400 --results /home/ubuntu/qdres_$TAG 2>&1 | head -3" </dev/null)
echo "#### submit: $O"
JID=$(grep -oE 'job [0-9]+' <<<"$O" | grep -oE '[0-9]+' | head -1)
echo "#### jid=$JID"; [ -z "$JID" ] && { echo "ABORT no jid"; exit 1; }
for i in $(seq 1 400); do
  sleep 20
  "${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq status 2>/dev/null" </dev/null \
    | grep -qE "^ *$JID " || break
done
scp -q -o BatchMode=yes -i $KEY -r $MGR:/home/ubuntu/qdres_$TAG/. $OUT/res_$TAG/ 2>/dev/null
UL=$(find $OUT/res_$TAG -name uartlog | head -1); echo "#### uartlog=${UL:-NONE}"
# Three-part gate. fq copies from the run host's sim_slot_*/, which survives
# between jobs, so a cell can silently collect the PREVIOUS job's uartlog --
# stale lane output otherwise reads as success.
if [ -n "$UL" ]; then
  S=$(grep -oE 'xpurt-runner: schedule=[^ ]+' "$UL" | head -1 | cut -d= -f2)
  [ "$S" = "$TAG" ] && echo "#### identity OK ($S)" || echo "#### IDENTITY MISMATCH: uartlog says '$S', expected '$TAG'"
  echo "#### fault lines: $(grep -c 'mcause' "$UL")"
  grep -E "entries=|MODELBLASTER_VERIFY|mcause" "$UL" | head -8
  echo "#### trace rows: $(grep -cE '^[0-9]+,' "$UL")"
fi
echo "############ FQDONE $TAG ############"
