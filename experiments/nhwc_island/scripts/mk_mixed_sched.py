#!/usr/bin/env python3
"""Serial schedule with a per-op-kind slot override.

  mk_mixed_sched.py OUT.json --graph G --model M --slot CPU_P#0
                    --on CPU_E#0:nchw_to_nhwc_s8,nhwc_to_nchw_s8

Same hard-chained serial order as shard_dim's mk_serial_sched.py -- one dispatch
in flight, no concurrency tax -- but the named op kinds are placed on a different
slot. That is the one thing making a relayout a DISPATCH buys that nothing else
does: on this SoC the two hart types have different ISAs (harts 0/1 Rocket +
Gemmini, no vector unit; harts 2/3 Rocket + Saturn), so placing the conversion
changes which instructions can implement it, not merely when it runs.
"""
import argparse, json
a = argparse.ArgumentParser()
a.add_argument("out"); a.add_argument("--graph", required=True)
a.add_argument("--model", required=True); a.add_argument("--slot", default="CPU_P#0")
a.add_argument("--on", default="", help="SLOT:kind,kind")
a = a.parse_args()
alt_slot, alt_kinds = None, set()
if a.on:
    alt_slot, kinds = a.on.split(":", 1)
    alt_kinds = set(kinds.split(","))
g = json.load(open(a.graph))
ops = [o for o in g["ops"] if o.get("dispatch_id") is not None]
by = {o["dispatch_id"]: o for o in ops}
key = {d: f"{a.model}0_dispatch_{d}" for d in by}
seen, order = set(), []
def vis(d):
    if d in seen: return
    seen.add(d)
    for q in by[d].get("depends_on", []):
        if q in by: vis(q)
    order.append(d)
for d in by: vis(d)
disp, t, prev = {}, 0.0, None
for d in order:
    o = by[d]
    deps = [key[q] for q in o.get("depends_on", []) if q in by]
    if prev is not None and key[prev] not in deps:
        deps.append(key[prev])
    slot = alt_slot if o["op"] in alt_kinds else a.slot
    disp[key[d]] = {"id": d, "ordinal": 1, "total": 1, "dependencies": deps,
                    "hardware_target": slot, "start_time": round(t, 6),
                    "duration": 0.01, "job_name": f"{a.model}0",
                    "module_name": o.get("name", f"dispatch_{d}")}
    t += 0.01; prev = d
slots = sorted({v["hardware_target"] for v in disp.values()})
json.dump({"dot_file": "nhwc_island.json", "dispatches": disp,
           "metadata": {"makespan": t, "num_operations": len(disp),
                        "machines": slots, "machine_combinations": [slots],
                        "profile_hw": {"CPU_P": "gemmini_q31", "CPU_E": "V256D128_rvv"}}},
          open(a.out, "w"), indent=1)
print(f"[mixed_sched] {a.out}: {len(disp)} dispatches, slots={slots}, "
      f"{sum(1 for v in disp.values() if v['hardware_target']==alt_slot)} on {alt_slot}")
