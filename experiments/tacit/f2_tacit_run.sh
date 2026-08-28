#!/usr/bin/env bash
# Runs on the AWS F2 MANAGER. Submits one fq job, waits, then pulls the TACIT
# trace files off the run host -- fq's firesim backend hardcodes
# common_simulation_outputs=["uartlog"], so tacit<N>.out is NEVER copied back
# by runworkload; it only exists in <simulation_dir>/sim_slot_0/ on the host
# that actually ran the job.
set -uo pipefail
TAG="$1"; TO="${2:-1200}"
HW="f2_dual_small_norose_tacit_q31_60mhz"
TREE="/home/ubuntu/chipyard-rose"
R="/home/ubuntu/r/$TAG"
export FQ_SOCKET=/var/lib/fq/fq.sock
cd ~/fpga_queue

# Clear stale traces on EVERY run host first: sim_slot_*/ survives between
# jobs, so a run whose sim never started would otherwise hand back the
# PREVIOUS occupant's trace -- the same cross-agent mis-attribution fq's
# argv_clear_uartlog guards against for the uartlog.
HOSTS=$(python3 -c "
import yaml,sys
p=yaml.safe_load(open('/var/lib/fq/pool.yaml'))
print(' '.join(h for l in p['lanes'] for h in (l.get('hosts') or [])))
")
echo "== run hosts: $HOSTS"
for h in $HOSTS; do
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=10 -i ~/firesim.pem ubuntu@$h \
      'rm -f /home/ubuntu/sim_slot_*/tacit*.out' </dev/null >/dev/null 2>&1 &
done
wait
echo "== cleared stale tacit*.out"

./bin/fq submit --tree "$TREE" --hw-config "$HW" --elf "$R/z.elf" \
   --timeout "$TO" --results "$R" --wait 2>&1 | tail -25
echo "== fq submit returned $?"

echo "== hunting tacit*.out on the run hosts =="
mkdir -p "$R/tacit"
for h in $HOSTS; do
  found=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=10 -i ~/firesim.pem ubuntu@$h \
      'ls -l /home/ubuntu/sim_slot_*/tacit*.out 2>/dev/null' </dev/null 2>/dev/null)
  if [ -n "$found" ]; then
    echo "--- $h ---"; echo "$found"
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -i ~/firesim.pem "ubuntu@$h:/home/ubuntu/sim_slot_*/tacit*.out" "$R/tacit/" 2>/dev/null
  fi
done
echo "== collected =="
ls -l "$R/tacit/" 2>/dev/null || echo "NOTHING COLLECTED"
echo "== uartlog tail =="
tail -40 "$R/uartlog" 2>/dev/null || echo "no uartlog"
