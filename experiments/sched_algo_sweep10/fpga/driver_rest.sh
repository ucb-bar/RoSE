#!/usr/bin/env bash
# Finish the 1056-row set unattended: wait for the CP-SAT emissions, dedupe
# EVERYTHING on schedule content, then build every distinct schedule that has
# no ELF yet. Idempotent -- safe to re-run; --skip-existing keeps finished ELFs.
set -uo pipefail
F=/scratch/dima/rose-infra/RoSE/experiments/sched_algo_sweep10/fpga
cd $F && source env.sh
exec >> $F/logs/driver_rest.log 2>&1
echo "=== driver_rest start $(date -u +%FT%TZ)"

# 1a. the tranche-1 build pass owns examples/xpurt_s10_w1..6; two build passes
#     sharing those trees would overwrite each other's objects, so wait it out.
while pgrep -f "build_all.py --plan elf_plan_t1" >/dev/null; do sleep 30; done
echo "--- tranche-1 builds done $(date -u +%FT%TZ)"
# 1b. wait out any emission still running
while pgrep -f "emit_all.py --jobs jobs_rest_cpsat" >/dev/null; do sleep 30; done
echo "--- emissions done $(date -u +%FT%TZ)"

# 2. dedupe over every emission we have, and re-run the cheap/cpsat emitters
#    for anything still missing (a solver that crashed gets one retry, then is
#    recorded in unbuildable.json rather than silently dropped).
python3 plan.py jobs --stage all --out jobs_all.json
python3 - <<'PY'
import json, os
jobs=json.load(open('jobs_all.json')); miss=[]
for a,w,s in jobs:
    tag=f"{a}__{w}__{s.replace(':','-')}"
    if not os.path.exists(f'schedules/{tag}.meta.json'): miss.append([a,w,s])
json.dump(miss, open('jobs_missing.json','w'))
print(f"missing emissions: {len(miss)}")
PY
if [ "$(python3 -c "import json;print(len(json.load(open('jobs_missing.json'))))")" != 0 ]; then
  python3 emit_all.py --jobs jobs_missing.json --outdir schedules \
      --out emitted_missing.json --cheap-workers 8 --cpsat-parallel 4
fi
python3 plan.py dedupe --emitted emitted_t1.json emitted_rest_cheap.json \
    emitted_rest_cpsat.json $( [ -f emitted_missing.json ] && echo emitted_missing.json ) \
    --out elf_plan.json
python3 winners_local.py --out winners_local.json
python3 dispatch_order.py --out dispatch_order.txt
echo "--- dedupe done $(date -u +%FT%TZ)"

# 3. build everything not already built
python3 build_all.py --plan elf_plan.json --out built_all.json --workers 6 --skip-existing
echo "--- builds done $(date -u +%FT%TZ)"
python3 winners_local.py --out winners_local.json
python3 summarize.py > $F/SUMMARY.txt 2>&1
cat $F/SUMMARY.txt
echo "=== driver_rest end $(date -u +%FT%TZ)"
