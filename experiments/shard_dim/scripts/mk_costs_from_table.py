#!/usr/bin/env python3
"""Per-dispatch cost JSON for ANY candidate graph, from the study's measured
per-tile times.  mk_costs_from_table.py <graph.json> <out.json>

mk_costs.py needs two serial runs OF THAT EXACT TREE; this needs none, because
every (conv, axis, backend) tile time was already measured by the 12 serial runs
and a candidate graph is just a re-selection of those. Non-conv dispatches come
from the unsplit B0 runs, where they are unaffected by any split.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mk_mixed as M

graph, out = sys.argv[1], sys.argv[2]
tiles, other = M.cost_table()
g = json.load(open(graph))
costs, missing = {"rvv": {}, "gemmini_q31": {}}, []
for op in g["ops"]:
    d = op.get("dispatch_id")
    if d is None:
        continue
    nm = op["name"].split(".tile_")[0]
    ax = (op.get("split_from") or {}).get("axis", "none")
    t = int(op["name"].split(".tile_")[1]) if ".tile_" in op["name"] else 0
    for be in ("rvv", "gemmini_q31"):
        if op["op"] == "conv2d_s8":
            v = tiles.get((nm, ax, be))
            # A single-tile ("pad only") conv has no measured entry of its own;
            # fall back to the unsplit time so the PLACEMENT is still sane. That
            # only affects the predicted makespan, never the measured one.
            if v is None or t >= len(v):
                v = tiles[(nm, "none", be)]
                missing.append((nm, ax, be))
                t2 = 0
            else:
                t2 = t
            costs[be][str(d)] = round(v[t2], 3)
        else:
            costs[be][str(d)] = round(other.get((nm, be), other.get((nm, "rvv"), 1.0)), 3)
json.dump(costs, open(out, "w"), indent=1)
print(f"[costs] {out}: {len(costs['rvv'])} dispatches"
      + (f"  ({len(set(missing))} conv cells fell back to unsplit)" if missing else ""))
