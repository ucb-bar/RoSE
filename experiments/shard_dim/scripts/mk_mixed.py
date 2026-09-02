#!/usr/bin/env python3
"""Choose a per-operator split AXIS by minimising PREDICTED makespan, from
measured per-tile costs, then emit the plan for mk_split.py.

Every number in the cost table is measured on this bitstream: the unsplit and
per-axis tile times come out of the 12 serial runs of this study (B0 / ocA /
L1-L3 / ohA / ohB), so the only modelled thing here is the PLACEMENT, and the
FPGA run that follows is what checks it.

Why a search rather than "pick the axis with the best work ratio": work ratio is
a per-backend quantity and the schedule decides the backend. On this SoC gemmini
is 1.5-14x faster than rvv on every dronet conv, so most convs want a gemmini
hart -- where OC wins -- and the axis that wins in isolation is not necessarily
the axis that wins in the schedule.
"""
import glob, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ul import parse_uartlog, conv_rows

MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
RES = "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/results"
#: Slot set to plan for. Overridable so the same measured cost table can be
#: replanned for a different machine (e.g. an rvv PAIR, where OH's lack of a
#: blocking quantum is what breaks the OC=32 convs out of their 1.0x floor)
#: without re-measuring:  MK_MIXED_SLOTS="CPU_E#0=rvv,CPU_E#1=rvv"
SLOTS = [tuple(x.split("=")) for x in os.environ["MK_MIXED_SLOTS"].split(",")] \
    if os.environ.get("MK_MIXED_SLOTS") else \
    [("CPU_P#0", "gemmini_q31"), ("CPU_P#1", "gemmini_q31"),
     ("CPU_E#0", "rvv"), ("CPU_E#1", "rvv")]
#: which run measured which (conv, axis). All 2-way, all serial-on-one-hart.
SRC = {"OC": {**{f"conv_modules.{i}": "oc:ocA" for i in range(4)},
              "conv_modules.4": "aln:L1", "conv_modules.5": "aln:L3",
              "conv_modules.6": "aln:L2", "conv_modules.7": "aln:L1",
              "conv_modules.8": "aln:L3", "conv_modules.9": "aln:L2"},
       "OH": {**{f"conv_modules.{i}": "oc:ohA" for i in range(4)},
              **{f"conv_modules.{i}": "oc:ohB" for i in range(4, 10)}}}


#: gemmini's OH tiles were re-measured after the ZERO-COPY kernel landed (fq
#: 470/471), which removed the gather/scatter the OH wrapper used to need. That
#: moved gemmini OH from losing all 10 cells to winning 6, so planning against
#: the pre-zero-copy ohAP/ohBP runs picks all-OC for a reason that no longer
#: exists. Those runs carry the SAME partitions, so the substitution is exact.
#: rvv is unaffected -- it still uses the portable gather/scatter path.
SRC_ZEROCOPY_P = {**{f"conv_modules.{i}": "oc:zc2A" for i in range(4)},
                  **{f"conv_modules.{i}": "oc:zc2B" for i in range(4, 10)}}

def load(kind_tag, arm):
    kind, tag = kind_tag.split(":")
    d = (f"{RES}/aln/res_aln{tag}{arm}" if kind == "aln" else f"{RES}/oh/res_{tag}{arm}")
    uls = glob.glob(os.path.join(d, "**", "uartlog"), recursive=True)
    return parse_uartlog(max(uls, key=os.path.getsize))


def cost_table():
    """{(conv, axis, backend): [per-tile us]} and {(other_op_did, backend): us}."""
    tiles, other = {}, {}
    base = {a: load("aln:B0", a) for a in ("E", "P")}
    for arm, be in (("E", "rvv"), ("P", "gemmini_q31")):
        for nm, rows in conv_rows(base[arm]).items():
            tiles[(nm, "none", be)] = [rows[0][1]]
        for did, rec in base[arm]["prof"].items():
            if rec["op"] != "conv2d_s8":
                other[(rec["name"], be)] = rec["us"]
    cache = {}
    for axis in ("OC", "OH"):
        for nm, src in SRC[axis].items():
            for arm, be in (("E", "rvv"), ("P", "gemmini_q31")):
                if axis == "OH" and be == "gemmini_q31":
                    src, arm = SRC_ZEROCOPY_P[nm], ""   # zc2* are P-arm only
                if (src, arm) not in cache:
                    cache[(src, arm)] = conv_rows(load(src, arm))
                r = cache[(src, arm)].get(nm)
                if r and len(r) > 1:
                    tiles[(nm, axis, be)] = [t[1] for t in r]
    return tiles, other


def makespan(graph, costfn):
    """Greedy earliest-completion list schedule; returns (makespan, placement)."""
    ops = [o for o in graph["ops"] if o.get("dispatch_id") is not None]
    by = {o["dispatch_id"]: o for o in ops}
    npred = {d: len([q for q in by[d].get("depends_on", []) if q in by]) for d in by}
    succ = {d: [] for d in by}
    for d in by:
        for q in by[d].get("depends_on", []):
            if q in by:
                succ[q].append(d)
    ready = {d for d in by if npred[d] == 0}
    free = {s: 0.0 for s, _ in SLOTS}
    done, place = {}, {}
    while ready:
        best = None
        for d in ready:
            rdy = max([done[q] for q in by[d].get("depends_on", []) if q in by], default=0.0)
            for s, kind in SLOTS:
                fin = max(rdy, free[s]) + costfn(by[d], kind)
                if best is None or fin < best[0] or (fin == best[0] and d < best[2]):
                    best = (fin, s, d)
        fin, s, d = best
        ready.discard(d); done[d] = fin; free[s] = fin; place[d] = s
        for q in succ[d]:
            npred[q] -= 1
            if npred[q] == 0:
                ready.add(q)
    return max(done.values()), place


def main():
    from modelblaster.pipeline.apply_split_hint import apply_split_hint
    tiles, other = cost_table()
    g0 = json.load(open(f"{MB}/examples/dronet_armB/int8/generated/graph.json"))
    convs = [o for o in g0["ops"] if o.get("op") == "conv2d_s8"]
    did_of = {o["name"]: o["dispatch_id"] for o in convs}
    OHdim = {o["name"]: o["shape"]["OH"] for o in convs}
    OCdim = {o["name"]: o["shape"]["OC"] for o in convs}

    def build(assign):
        hints = []
        for nm, ax in assign.items():
            if ax == "none":
                continue
            if ax == "OC":
                w = [OCdim[nm] // 2, OCdim[nm] - OCdim[nm] // 2]
            else:
                w = [OHdim[nm] // 2, OHdim[nm] - OHdim[nm] // 2]
            hints.append({"op": did_of[nm], "n_splits": 2, "tile_sizes": w, "axis": ax})
        return apply_split_hint(g0, hints) if hints else json.loads(json.dumps(g0))

    def costfn_for(assign):
        def f(op, kind):
            nm = op["name"].split(".tile_")[0]
            if op["op"] != "conv2d_s8":
                return other.get((nm, kind), other.get((nm, "rvv"), 1.0))
            ax = assign.get(nm, "none")
            t = int(op["name"].split(".tile_")[1]) if ".tile_" in op["name"] else 0
            v = tiles.get((nm, ax, kind))
            return v[t] if v and t < len(v) else tiles[(nm, "none", kind)][0]
        return f

    assign = {nm: "none" for nm in did_of}
    cur, _ = makespan(build(assign), costfn_for(assign))
    print(f"predicted makespan, all unsplit: {cur:.1f} us")
    improved = True
    while improved:
        improved = False
        best = None
        for nm in did_of:
            for ax in ("OC", "OH"):
                if assign[nm] == ax:
                    continue
                if (nm, ax, "rvv") not in tiles or (nm, ax, "gemmini_q31") not in tiles:
                    continue
                trial = dict(assign); trial[nm] = ax
                m, _ = makespan(build(trial), costfn_for(trial))
                if m < cur - 0.5 and (best is None or m < best[0]):
                    best = (m, nm, ax)
        if best:
            cur, nm, ax = best
            assign[nm] = ax
            print(f"  + {nm:<16} -> {ax}   predicted makespan {cur:.1f} us")
            improved = True
    print(f"\nchosen assignment (predicted makespan {cur:.1f} us):")
    plan = []
    for nm in sorted(did_of, key=lambda n: did_of[n]):
        ax = assign[nm]
        if ax == "none":
            print(f"  {nm:<16} did={did_of[nm]:<3} UNSPLIT")
            continue
        d = OCdim[nm] if ax == "OC" else OHdim[nm]
        w = [d // 2, d - d // 2]
        plan.append(f"{did_of[nm]}:{ax}:{w[0]},{w[1]}")
        print(f"  {nm:<16} did={did_of[nm]:<3} {ax} {w}")
    print("\nmk_split args:\n  " + " ".join(plan))
    json.dump({"assignment": assign, "predicted_makespan_us": cur,
               "mk_split_args": plan}, open(os.environ.get("MK_MIXED_OUT", f"{RES}/oh/mixed_plan.json"), "w"), indent=1)


main()
