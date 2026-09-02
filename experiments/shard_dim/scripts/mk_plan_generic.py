#!/usr/bin/env python3
"""Choose a split axis per operator for ANY network, by minimising predicted
makespan over a given slot set.

`mk_mixed.py` does this for dronet but is hardwired to it: dronet's conv names,
its ten convs, and a hand-built table of which measured run supplied which cell.
This generalises that to any graph, and prices tiles with the SAME model the
production scheduler uses (`xpu-rt/profile_loader.py`), so a plan chosen here and
a schedule built later agree by construction rather than by coincidence.

  mk_plan_generic.py --graph G --costs C --model M --slots S [--max-splits N]
                     [--axes OC,OH,N] [--out plan.json]

`costs` is {backend: {dispatch_id: microseconds}} measured on THIS hardware
(mk_costs.py from two serial runs). A dispatch with no cost for a backend is not
placeable there.

Search is greedy: repeatedly take the single (op, axis) split that most reduces
the makespan, stop when none helps. Greedy is the baseline the goal wants
compared against a MILP; keeping it here, honest and simple, is the point.
"""
import argparse, collections, copy, itertools, json, os, sys

sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/xpu-rt")
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
import profile_loader as PL
from modelblaster.pipeline.apply_split_hint import apply_split_hint, SplitHintError

AXIS_DIM = {"OC": "OC", "OH": "OH", "N": "N"}


def makespan(graph, cost):
    """Greedy earliest-completion list schedule. Returns (makespan_us, place)."""
    ops = [o for o in graph["ops"] if o.get("dispatch_id") is not None]
    by = {o["dispatch_id"]: o for o in ops}
    npred = {d: len([q for q in by[d].get("depends_on", []) if q in by]) for d in by}
    succ = collections.defaultdict(list)
    for d in by:
        for q in by[d].get("depends_on", []):
            if q in by:
                succ[q].append(d)
    ready = {d for d in by if npred[d] == 0}
    free, done, place = collections.defaultdict(float), {}, {}
    while ready:
        best = None
        for d in ready:
            rdy = max([done[q] for q in by[d].get("depends_on", []) if q in by], default=0.0)
            for slot, kind in SLOTS:
                c = cost(by[d], kind)
                if c is None:
                    continue
                fin = max(rdy, free[slot]) + c
                if best is None or fin < best[0] or (fin == best[0] and d < best[2]):
                    best = (fin, slot, d)
        if best is None:
            raise SystemExit("[mk_plan] a dispatch is not placeable on any slot")
        fin, slot, d = best
        ready.discard(d); done[d] = fin; free[slot] = fin; place[d] = slot
        for q in succ[d]:
            npred[q] -= 1
            if npred[q] == 0:
                ready.add(q)
    return max(done.values()), place


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--graph", required=True); a.add_argument("--costs", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--slots", default="CPU_P#0=gemmini_q31,CPU_E#0=rvv")
    a.add_argument("--axes", default="OC,OH,N")
    a.add_argument("--max-splits", type=int, default=64)
    a.add_argument("--out", default="/tmp/plan_generic.json")
    a = a.parse_args()

    global SLOTS
    SLOTS = [tuple(x.split("=")) for x in a.slots.split(",")]
    g0 = json.load(open(a.graph))
    costs = {k: {int(d): float(v) for d, v in vv.items()}
             for k, vv in json.load(open(a.costs)).items()}
    axes = a.axes.split(",")

    # Which (op, axis) pairs are even legal? Ask the splitter rather than
    # duplicating its rules -- it is the thing that will actually build the tree,
    # and it raises SplitHintError with a reason when an axis does not apply.
    ops = [o for o in g0["ops"] if o.get("dispatch_id") is not None]
    cand = []
    for o in ops:
        sh = o.get("shape") or {}
        for ax in axes:
            dim = sh.get(AXIS_DIM[ax])
            if not isinstance(dim, int) or dim < 2:
                continue
            w = [dim // 2, dim - dim // 2]
            try:
                apply_split_hint(g0, [{"op": o["dispatch_id"], "n_splits": 2,
                                       "tile_sizes": w, "axis": ax}])
            except (SplitHintError, SystemExit, KeyError, ValueError):
                continue
            cand.append((o["dispatch_id"], o["name"], ax, w))
    print(f"[mk_plan] {len(ops)} dispatches, {len(cand)} legal (op, axis) splits")

    # dispatch_id -> cost is keyed by the ORIGINAL graph's ids, but
    # apply_split_hint RENUMBERS every id after an inserted tile, so a
    # dispatch's new id does not index the cost table -- including for ops that
    # were never split. Names are stable (a tile is just "<parent>.tile_N"), so
    # map through names instead. Getting this wrong makes unsplit ops look
    # unplaceable, which is how it first showed up.
    orig_id = {o["name"]: o["dispatch_id"] for o in ops}

    def cost_for(hints):
        gg = apply_split_hint(g0, hints) if hints else g0
        parent = {}
        for o in gg["ops"]:
            d = o.get("dispatch_id")
            if d is None:
                continue
            sf = o.get("split_from") or {}
            base_name = o["name"].split(".tile_")[0]
            parent[d] = (orig_id.get(base_name), sf, o)
        def c(op, kind):
            d = op["dispatch_id"]
            pid, sf, oo = parent[d]
            base = costs.get(kind, {}).get(pid)
            if base is None:
                # No cost on ANY backend => the op emits no kernel call (an
                # alias/view such as chunk2_c1). Zero, and placeable anywhere.
                # No cost on THIS backend only => a genuine coverage hole, and
                # the op is legitimately not placeable here.
                if all(costs.get(b, {}).get(pid) is None for b in costs):
                    return 0.0
                return None
            if not sf:
                return base
            ax = sf.get("axis")
            if ax == "OH" and sf.get("window_rows") and sf.get("parent_IH") is not None:
                pih = int(sf["parent_IH"]) + 2 * int(sf.get("parent_PH", 0))
                kh = int((oo.get("shape") or {}).get("KH", 0) or 0)
                return base * (sf["window_rows"] / pih) * PL._oh_copy_tax(kind, kh > 1)
            w = sf.get("tile_oc", sf.get("tile_n"))
            tot = sf.get("parent_OC", sf.get("parent_N"))
            if w and tot:
                q = PL._quantum_for(kind)
                if q:
                    return base * (-(-w // q)) / max(1, -(-int(tot) // q))
                return base * w / int(tot)
            return base / max(1, sf.get("n_splits", 1))
        return gg, c

    chosen, hints = [], []
    gg, c = cost_for(hints)
    cur, _ = makespan(gg, c)
    print(f"[mk_plan] unsplit predicted makespan {cur:.1f} us")
    while len(chosen) < a.max_splits:
        best = None
        for (did, nm, ax, w) in cand:
            if any(h["op"] == did for h in hints):
                continue
            trial = hints + [{"op": did, "n_splits": 2, "tile_sizes": w, "axis": ax}]
            try:
                gg, c = cost_for(trial)
                m, _ = makespan(gg, c)
            except Exception:
                continue
            if m < cur - 0.5 and (best is None or m < best[0]):
                best = (m, did, nm, ax, w, trial)
        if not best:
            break
        cur, did, nm, ax, w, hints = best[0], best[1], best[2], best[3], best[4], best[5]
        chosen.append(f"{did}:{ax}:{w[0]},{w[1]}")
        print(f"  + {nm:<28} did={did:<4} {ax} {w}   -> {cur:.1f} us")
    print(f"\n[mk_plan] {len(chosen)} splits, predicted {cur:.1f} us")
    print("mk_split args:\n  " + " ".join(chosen))
    json.dump({"model": a.model, "slots": a.slots, "predicted_us": cur,
               "n_candidates": len(cand), "mk_split_args": chosen},
              open(a.out, "w"), indent=1)
    print(f"[mk_plan] -> {a.out}")


main()
