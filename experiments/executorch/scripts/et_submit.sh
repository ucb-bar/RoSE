#!/usr/bin/env bash
# Ship one ExecuTorch ELF to the shared AWS manager, run it on an fq lane on the
# SAME bitstream the ModelBlaster sweeps used, pull the uartlog back.
#   et_submit.sh <tag>
# Modelled on experiments/shard_dim/scripts/sweep_submit.sh -- deliberately, so
# an ET run and an MB run differ in nothing but the ELF.
#
# Shared-infrastructure rules observed here: never `firesim kill`, never cancel
# another user's job, never terminate a lane host. LANE defaults to f2-05
# because f2-07 is on degraded hardware.
set -uo pipefail
TAG=$1
R=/scratch/dima/rose-infra/RoSE; OUT=$R/experiments/executorch
mkdir -p $OUT/logs $OUT/res
exec > $OUT/logs/fq_$TAG.log 2>&1
MGR=ubuntu@3.88.218.39; KEY=~/.ssh/firesim.pem
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY)
HW=f2_quad_hetero_norose_tacit_q31_60mhz
LANE=${LANE:-f2-05}; LANEARG=""; [ -n "$LANE" ] && LANEARG="--lane $LANE"
TMO=${FQ_TIMEOUT:-3000}
STAMP="et-$TAG-$(date +%s | tail -c 6)"
echo "### submit $TAG lane=$LANE stamp=$STAMP $(date -u +%FT%TZ)"
[ -f "$OUT/elf/$TAG.elf" ] || { echo "ABORT no elf"; exit 1; }
echo "#### elf mtime=$(date -r $OUT/elf/$TAG.elf -u +%FT%TZ) bytes=$(stat -c%s $OUT/elf/$TAG.elf)"
scp -q -o BatchMode=yes -i $KEY "$OUT/elf/$TAG.elf" $MGR:/home/ubuntu/et_$TAG.elf || { echo "ABORT scp"; exit 1; }
"${SSH[@]}" $MGR "rm -rf /home/ubuntu/etres_$TAG && mkdir -p /home/ubuntu/etres_$TAG" </dev/null
O=$("${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; timeout 60 ./bin/fq submit \
  --tree /home/ubuntu/chipyard-rose --hw-config $HW --elf /home/ubuntu/et_$TAG.elf \
  --workload $STAMP $LANEARG --timeout $TMO --results /home/ubuntu/etres_$TAG 2>&1 | head -3" </dev/null)
echo "#### submit: $O"
JID=$(grep -oE 'job [0-9]+' <<<"$O" | grep -oE '[0-9]+' | head -1)
echo "#### jid=$JID"; [ -z "$JID" ] && { echo "ABORT no jid"; exit 1; }
for i in $(seq 1 240); do
  sleep 20
  "${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq status 2>/dev/null" </dev/null \
    | grep -qE "^ *$JID " || break
done
rm -rf $OUT/res/$TAG; mkdir -p $OUT/res/$TAG
scp -q -o BatchMode=yes -i $KEY -r $MGR:/home/ubuntu/etres_$TAG/. $OUT/res/$TAG/ 2>/dev/null
UL=$(find $OUT/res/$TAG -name uartlog | head -1); echo "#### uartlog=${UL:-NONE}"
if [ -n "$UL" ]; then
  # Provenance: fq reuses job ids across daemon restarts, and a lane can hand
  # back the PREVIOUS occupant's uartlog if it was polled too early.
  grep -q "$STAMP" "$UL" || echo "#### WARN uartlog does not carry stamp $STAMP -- provenance unverified"
  grep -E "MB_ET_HART|MB_ET_THREADPOOL|MB_INPUT_BAKED|EXECUTORCH_EXECUTE_CYCLES|checksum=|mcause|PASSED|FAILED" "$UL" | head -40
fi
echo "############ FQDONE $TAG ############"
