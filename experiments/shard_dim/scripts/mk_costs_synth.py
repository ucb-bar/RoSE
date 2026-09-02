#!/usr/bin/env python3
"""{backend: {dispatch_id: us}} for a SPLIT graph, from the measured per-tile
table of the OH-vs-OC study.

mk_costs.py needs two serial FPGA runs of the tree it is costing. That is the
right thing when the tree is new, but every (conv, axis, tile) cell in a
best-axis tree has ALREADY been measured -- ohA/ohB supplied the OH tiles,
ocA/L1-L3 the OC tiles, B0 the unsplit ops. This maps those measurements onto a
freshly split graph by NAME, so a mixed-axis tree can be scheduled from real
hardware numbers without spending two more FPGA runs per arm on re-measuring
cells that are already in the table.

  mk_costs_synth.py OUT.json --graph SPLIT_GRAPH.json

The axis of each tile is read from the graph's own `split_from.axis`, so the
costs cannot silently disagree with the partition that was actually built.
"""
import argparse, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

a = argparse.ArgumentParser()
a.add_argument("out"); a.add_argument("--graph", required=True)
a = a.parse_args()

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "mk_mixed.py")).read().replace("\nmain()\n", "\n")
_mm = types.ModuleType("mm")
# mk_mixed.py does sys.path.insert(..., dirname(abspath(__file__))) at import;
# exec() into a fresh module namespace supplies no __file__, so set it.
_mm.__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mk_mixed.py")
exec(compile(_src, "mk_mixed.py", "exec"), _mm.__dict__)
tiles, other = _mm.cost_table()

g = json.load(open(a.graph))
costs = {"rvv": {}, "gemmini_q31": {}}
missing = []
for op in g["ops"]:
    did = op.get("dispatch_id")
    if did is None:
        continue
    nm = op["name"]
    base = nm.split(".tile_")[0]
    sf = op.get("split_from") or {}
    axis = sf.get("axis", "none")
    idx = int(sf["tile"]) if "tile" in sf else 0
    for be in ("rvv", "gemmini_q31"):
        if op.get("op") == "conv2d_s8":
            v = tiles.get((base, axis, be))
            if v is None or idx >= len(v):
                missing.append((nm, axis, be)); continue
            costs[be][str(did)] = round(v[idx], 3)
        else:
            v = other.get((base, be), other.get((base, "rvv")))
            if v is None:
                missing.append((nm, axis, be)); continue
            costs[be][str(did)] = round(v, 3)
if missing:
    for m in missing[:10]:
        print(f"  MISSING measured cost: {m}")
    sys.exit(f"[mk_costs_synth] {len(missing)} cells have no measurement -- "
             f"refusing to emit a table with modelled numbers in it")
json.dump(costs, open(a.out, "w"), indent=1)
print(f"[mk_costs_synth] -> {a.out}  "
      f"rvv={len(costs['rvv'])} gemmini_q31={len(costs['gemmini_q31'])} dispatches")
