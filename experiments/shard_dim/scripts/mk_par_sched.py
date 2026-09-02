#!/usr/bin/env python3
"""Greedy list schedule over the hetero cores, using MEASURED per-dispatch costs.

The serial schedule (`mk_serial_sched.py`) answers "what does the split COST".
This one answers "what does it BUY": tiles of one op are independent, so a
scheduler can place them on different harts and the makespan -- not the summed
work -- is what the model actually takes.

  mk_par_sched.py OUT.json --graph G --model M --costs costs.json
                  [--slots CPU_P#0=gemmini_q31,CPU_E#0=rvv]

`costs.json` is {backend: {dispatch_id: microseconds}} measured on the SAME
binary family (the serial runs), so the schedule is built from this hardware's
real numbers rather than a cost model. A dispatch missing a cost for a backend
is simply not placeable there.

The emitted schedule pins each dispatch's predecessor ON ITS OWN SLOT as an
extra dependency, so the runtime's ordering matches the one that was planned;
`start_time` alone is advisory.
"""
import argparse, json, heapq

a = argparse.ArgumentParser()
a.add_argument("out"); a.add_argument("--graph", required=True)
a.add_argument("--model", required=True); a.add_argument("--costs", required=True)
a.add_argument("--slots", default="CPU_P#0=gemmini_q31,CPU_E#0=rvv")
a = a.parse_args()

g = json.load(open(a.graph))
ops = [o for o in g["ops"] if o.get("dispatch_id") is not None]
by = {o["dispatch_id"]: o for o in ops}
costs = {k: {int(d): float(v) for d, v in vv.items()}
         for k, vv in json.load(open(a.costs)).items()}
slots = [(s.split("=")[0], s.split("=")[1]) for s in a.slots.split(",")]

# Only conv/linear are backend-specific; everything else runs wherever it is
# placed. Cost falls back to the other backend's measurement when a kind has no
# number of its own, which is what the profile actually gives us for the
# gemmini-side elementwise ops.
def cost(did, kind):
    c = costs.get(kind, {}).get(did)
    if c is not None:
        return c
    for k in costs:
        if did in costs[k]:
            return costs[k][did]
    return 1.0

npred = {d: len([q for q in by[d].get("depends_on", []) if q in by]) for d in by}
succ = {d: [] for d in by}
for d in by:
    for q in by[d].get("depends_on", []):
        if q in by:
            succ[q].append(d)

ready = {d for d in by if npred[d] == 0}
done_at, place, order = {}, {}, {s: [] for s, _ in slots}
free = {s: 0.0 for s, _ in slots}
start = {}
while ready:
    # earliest-completion-first over (ready dispatch, slot)
    best = None
    for d in ready:
        dep_rdy = max([done_at[q] for q in by[d].get("depends_on", []) if q in by],
                      default=0.0)
        for s, kind in slots:
            st = max(dep_rdy, free[s])
            fin = st + cost(d, kind)
            if best is None or fin < best[0] or (fin == best[0] and d < best[2]):
                best = (fin, s, d, st)
    fin, s, d, st = best
    ready.discard(d)
    place[d], start[d], done_at[d] = s, st, fin
    free[s] = fin
    order[s].append(d)
    for q in succ[d]:
        npred[q] -= 1
        if npred[q] == 0:
            ready.add(q)

key = {d: f"{a.model}0_dispatch_{d}" for d in by}
disp = {}
for d in sorted(by):
    deps = [key[q] for q in by[d].get("depends_on", []) if q in by]
    i = order[place[d]].index(d)
    if i:                                   # serialize this slot
        prev = order[place[d]][i - 1]
        if key[prev] not in deps:
            deps.append(key[prev])
    disp[key[d]] = {"id": d, "ordinal": 1, "total": 1, "dependencies": deps,
                    "hardware_target": place[d], "start_time": round(start[d] / 1e6, 9),
                    "duration": round((done_at[d] - start[d]) / 1e6, 9),
                    "job_name": f"{a.model}0",
                    "module_name": by[d].get("name", f"dispatch_{d}")}
mk = max(done_at.values())
json.dump({"dot_file": "shard_dim.json", "dispatches": disp,
           "metadata": {"makespan": mk / 1e6, "num_operations": len(disp),
                        "machines": [s for s, _ in slots],
                        "machine_combinations": [[s for s, _ in slots]],
                        "profile_hw": {"CPU_P": "gemmini_q31",
                                       "CPU_E": "V256D128_rvv"}}},
          open(a.out, "w"), indent=1)
print(f"[par_sched] {a.out}: {len(disp)} dispatches, predicted makespan "
      f"{mk:.1f} us, slots " +
      ", ".join(f"{s}:{len(order[s])}" for s, _ in slots))
