#!/usr/bin/env python3
"""Schedule EVERY dispatch (all tiles of all ops) serially on ONE slot.

Serial placement is deliberate: it measures work conservation with zero
concurrency tax, isolating study axes A/B (split degree / tile width) from
axis D (placement).  mk_serial_sched.py OUT.json --graph G --model M --slot CPU_E#0
"""
import argparse, json
a = argparse.ArgumentParser()
a.add_argument("out"); a.add_argument("--graph", required=True)
a.add_argument("--model", required=True); a.add_argument("--slot", default="CPU_E#0")
a = a.parse_args()
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
disp, t = {}, 0.0
prev = None
for d in order:
    o = by[d]
    deps = [key[q] for q in o.get("depends_on", []) if q in by]
    if prev is not None and key[prev] not in deps:
        deps.append(key[prev])          # hard chain: one dispatch in flight
    disp[key[d]] = {"id": d, "ordinal": 1, "total": 1, "dependencies": deps,
                    "hardware_target": a.slot, "start_time": round(t, 6),
                    "duration": 0.01, "job_name": f"{a.model}0",
                    "module_name": o.get("name", f"dispatch_{d}")}
    t += 0.01; prev = d
json.dump({"dot_file": "shard_dim.json", "dispatches": disp,
           "metadata": {"makespan": t, "num_operations": len(disp),
                        "machines": [a.slot], "machine_combinations": [[a.slot]],
                        "profile_hw": {"CPU_P": "gemmini_q31", "CPU_E": "V256D128_rvv"}}},
          open(a.out, "w"), indent=1)
print(f"[serial_sched] {a.out}: {len(disp)} dispatches, all on {a.slot}, hard-chained")
