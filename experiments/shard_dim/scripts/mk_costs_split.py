#!/usr/bin/env python3
"""Project measured UNSPLIT per-dispatch costs onto a SPLIT graph.

  mk_costs_split.py OUT.json --graph SPLIT_GRAPH --costs UNSPLIT_COSTS

`mk_costs.py` gives {backend: {dispatch_id: us}} keyed by the ids of the graph it
measured. `apply_split_hint` RENUMBERS every id at and after an inserted tile, so
feeding those costs to a split graph silently misattributes them -- an unsplit op
picks up a neighbour's cost, and a tile picks up whatever now sits at its index.
Nothing errors; the schedule is just built on fiction.

Names are stable across the rewrite (a tile is `<parent>.tile_N`), so the
mapping goes through names. Tiles are then priced with the SAME axis-aware model
the production scheduler uses (xpu-rt/profile_loader.py), so a schedule built
here and the runtime's own view of it agree by construction.

`mk_costs_synth.py` does this for dronet from its measured per-tile cell table --
prefer that where it applies, since it needs no model at all.
"""
import argparse, json, os, sys

sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/xpu-rt")
import profile_loader as PL

a = argparse.ArgumentParser()
a.add_argument("out"); a.add_argument("--graph", required=True)
a.add_argument("--costs", required=True)
a.add_argument("--unsplit-graph", default=None,
               help="graph the costs were measured on; defaults to deriving "
                    "parent ids from the split graph's own split_from records")
a = a.parse_args()

g = json.load(open(a.graph))
costs = {k: {int(d): float(v) for d, v in vv.items()}
         for k, vv in json.load(open(a.costs)).items()}

# parent id per name, from the unsplit graph when given, else from split_from
name_to_id = {}
if a.unsplit_graph:
    for o in json.load(open(a.unsplit_graph))["ops"]:
        if o.get("dispatch_id") is not None:
            name_to_id[o["name"]] = o["dispatch_id"]
else:
    for o in g["ops"]:
        sf = o.get("split_from") or {}
        if sf.get("op_id") is not None:
            name_to_id[o["name"].split(".tile_")[0]] = sf["op_id"]
    for o in g["ops"]:
        d = o.get("dispatch_id")
        if d is not None and not (o.get("split_from") or {}):
            name_to_id.setdefault(o["name"], d)

out = {be: {} for be in costs}
missing = []
zero_cost = set()
for o in g["ops"]:
    d = o.get("dispatch_id")
    if d is None:
        continue
    sf = o.get("split_from") or {}
    base = o["name"].split(".tile_")[0]
    pid = name_to_id.get(base)
    for be in costs:
        c = costs[be].get(pid) if pid is not None else None
        if c is None:
            # Missing on EVERY backend means the op emits no kernel call at all
            # -- an alias/view like chunk2_c1, which yolov8_nano has 8 of (155
            # graph ops, 147 profile rows). Those are genuinely zero cost and
            # must be placeable, or the whole graph becomes unschedulable.
            #
            # Missing on only SOME backends is the opposite: a real coverage
            # hole, where the op runs but nobody measured it there. Refuse that
            # one, because silently costing it zero would make the scheduler
            # believe a backend is free at exactly the op it cannot run well.
            if all(costs[b2].get(pid) is None for b2 in costs):
                out[be][str(d)] = 0.0
                zero_cost.add(o["name"])
            else:
                missing.append((o["name"], be))
            continue
        if not sf:
            out[be][str(d)] = round(c, 3); continue
        ax = sf.get("axis")
        if ax == "OH" and sf.get("window_rows") and sf.get("parent_IH") is not None:
            pih = int(sf["parent_IH"]) + 2 * int(sf.get("parent_PH", 0))
            kh = int((o.get("shape") or {}).get("KH", 0) or 0)
            v = c * (sf["window_rows"] / pih) * PL._oh_copy_tax(be, kh > 1)
        else:
            w = next((sf[k] for k in ("tile_oc", "tile_n", "tile_c")
                      if sf.get(k) is not None), None)
            tot = next((sf[k] for k in ("parent_OC", "parent_N", "parent_n",
                                        "parent_C") if sf.get(k) is not None),
                       None)
            if w and tot:
                q = PL._quantum_for(be) if ax == "OC" else None
                v = (c * (-(-w // q)) / max(1, -(-int(tot) // q))) if q else c * w / int(tot)
            else:
                v = c / max(1, sf.get("n_splits", 1))
        out[be][str(d)] = round(v, 3)

if missing:
    for m in missing[:8]:
        print(f"  MISSING cost: {m}")
    sys.exit(f"[mk_costs_split] {len(missing)} (op, backend) cells have no "
             f"measured parent -- refusing to emit a partial table")
json.dump(out, open(a.out, "w"), indent=1)
print(f"[mk_costs_split] -> {a.out}  " +
      "  ".join(f"{be}={len(v)}" for be, v in out.items()) +
      (f"  ({len(zero_cost)} kernel-less alias ops costed 0)" if zero_cost else ""))
