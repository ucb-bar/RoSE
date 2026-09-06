#!/usr/bin/env python3
"""Generate a candidate set of XPU-RT workload JSONs that stress scheduling.

  mk_workloads.py [--emit] [--out DIR] [--pairs rvvpair,gempair,hetero]

The existing data/toplevel/3net_pairs/*.json are three hand-written files that
differ only in hardware.machines. This generates a FAMILY of workloads that
vary the things that actually make a scheduling problem hard -- period,
instance count, network mix, and the spread of per-instance cost -- across the
scaled rungs from mk_scaled_variants.py plus ViNT as the large model.

PERIODS ARE DERIVED, NOT GUESSED. A period shorter than one instance's own
runtime is infeasible and the solver will either fail or silently drop work,
which produces a "hard" workload for the wrong reason. Each model's cost is
estimated analytically from its IR (MACs for conv/linear, elements for
pointwise) and ANCHORED to the measured runtime of the default rung on each
backend, so the estimate carries real units. Periods are then set as a
multiple of the slowest-backend cost, which is what makes a workload tight but
satisfiable.

Families, from tight low-latency loops to large models run together:
  tight_loop      many tiny instances at short periods -- dispatch churn
  control_mix     the classic 3net shape: control loop + perception
  perception_heavy one large detector + a light control loop
  scale_ladder    the SAME network at six scales at once -- duration spread
  bimodal         extreme period ratio: one very slow model + a 1 ms loop
  vint_intro      ViNT (large) beside a tight control loop
  vint_multi      ViNT + dronet + mlp -- large model in a multi-model task
  saturation      deliberately over-subscribed; tests graceful degradation
  depth_chain     fastdepth -> dronet ALONE: the dependency's cost, attributable
  depth_nav       that pipeline + an independent control loop (the real shape)
  depth_contended the pipeline against a competing detector that depends on nothing

DEPENDENT PIPELINES. A family may declare `edges`, which workload_factory
turns into real precedence: the producer's sink dispatches become predecessors
of the consumer's source dispatches, and for two periodic networks with equal
instance counts it pairs instance i to instance i. That is a scheduling
constraint, not a data path -- no tensor crosses the edge, exactly as in the
hand-written data/toplevel/networks_deps*.json. The models are still chosen so
the pairing is physically meaningful: fastdepth emits a 1x128x128 depth map and
dronet_sf is the 128-input rung, so the two agree spatially (dronet's stem
takes 3 channels to fastdepth's 1; examples/fastdepth_dronet is the fused,
data-exact realisation of the same pipeline).

A chain ticks at ONE rate, so its members share a period derived from the SUM
of their costs -- see the chain handling in main(). Giving each member a period
from its own cost would let the consumer run more often than its producer.
"""
import argparse, json, os, glob, sys, collections

R = "/scratch/dima/rose-infra/RoSE"
MB = f"{R}/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
GEN = "zephyr-chipyard-sw/gen"
TARGET = "firesim_f2_rocket_saturn"
#: cpu_p is the Gemmini-attached hart count, cpu_e the Saturn/RVV one. The
#: three 2-hart entries isolate a backend (or mix exactly one of each) so a
#: result can be attributed; "quad" is the whole f2_quad_hetero part, all four
#: harts at once, which is what the chip actually offers. Tiles are still 2-way
#: there, so a quad run buys its extra parallelism BETWEEN operators while each
#: split operator still spans two harts -- 4-way tiling is a separate question.
PAIRS = {"rvvpair": {"cpu_p": 0, "cpu_e": 2},
         "gempair": {"cpu_p": 2, "cpu_e": 0},
         "hetero":  {"cpu_p": 1, "cpu_e": 1},
         "quad":    {"cpu_p": 2, "cpu_e": 2}}

# Measured ms for the DEFAULT rung, single hart, from the fresh serial runs.
# These anchor the analytic estimate so periods come out in real units.
ANCHOR = {                       # model -> (gemmini ms, rvv ms)
    "mlp_control": (0.547, 0.596),
    "dronet":      (4.988, 7.912),
    "yolov8_nano": (90.045, 167.603),
    "vint":        (5272.755, 17028.698),
    # fastdepth, int8, measured this session (114 dispatches, single hart).
    # The ONLY model here whose gemmini arm is slower than its rvv arm: its
    # depthwise kernel is scalar_tap_ranges (a per-channel 5x5 has no reduction
    # dimension for a systolic array) and relu6_s8 falls back to reference.
    "fastdepth":   (230.33, 79.80),
}
DEFAULT_RUNG = {"mlp_control": "sd", "dronet": "se", "yolov8_nano": "sd"}

#: MEASURED single-hart cost (gemmini ms, rvv ms), from the serialP/serialE
#: runs in experiments/sweep3net. These OVERRIDE the analytic estimate below
#: wherever a rung appears here, because the estimate is not merely imprecise
#: on some rungs -- it is wrong by more than an order of magnitude.
#:
#: WHY THIS EXISTS. cost_ms() scales MAC counts from a network's DEFAULT rung,
#: which implicitly assumes every rung reaches the same curated kernels. It
#: does not: dronet_sa (32px) and dronet_sb (48px) resolve 7/7 conv picks to
#: REFERENCE scalar on BOTH backends where sc..sg get curated ones, so they run
#: ~61x and ~64x slower than the estimate predicts (0.60 vs 36.8 ms measured on
#: rvv). tight_loop derives its periods from dronet_sa, so it shipped windows
#: of 2.281 ms against a 35.256 ms critical path -- infeasible by construction,
#: in both arms and all four machine pairs. Every heuristic scheduler silently
#: returned a schedule that missed them; only the MILP reported `infeasible`.
#: See the sched_algo_sweep10 entry in experiments/kernel_opt_log.jsonl.
MEASURED = {
    "dronet":         (4.988, 7.912),   "dronet_sa":      (37.027, 36.813),
    "dronet_sb":      (103.193, 102.192), "dronet_sc":    (2.269, 2.734),
    "dronet_sd":      (4.913, 4.827),   "dronet_se":      (4.992, 7.903),
    "dronet_sf":      (8.064, 9.389),   "dronet_sg":      (15.835, 12.528),
    "fastdepth":      (230.644, 80.061),
    "mlp_control":    (0.547, 0.596),   "mlp_control_sa": (0.045, 0.044),
    "mlp_control_sb": (0.065, 0.045),   "mlp_control_sd": (0.529, 0.164),
    "mlp_control_sf": (1.915, 0.479),
    "vint":           (5257.460, 17021.522),
    "yolov8_nano":    (90.029, 167.588), "yolov8_nano_sc": (60.209, 109.840),
    "yolov8_nano_se": (130.794, 227.647), "yolov8_nano_sf": (180.720, 308.324),
    "yolov8_nano_sh": (3662.874, 568.417),
}


def macs(graph):
    """Analytic cost proxy: MACs for conv/linear, elements for everything else."""
    tot = 0
    for o in graph.get("ops", []):
        s = o.get("shape") or {}
        op = o.get("op", "")
        try:
            if op.startswith("conv2d") and "OC" in s:
                oh = int(s.get("OH") or 0); ow = int(s.get("OW") or oh)
                tot += (int(s["OC"]) * oh * ow * int(s.get("IC", 1))
                        * int(s.get("KH", 1)) * int(s.get("KW", 1)))
            elif op.startswith("linear") and "N" in s:
                tot += int(s["M"]) * int(s["N"]) * int(s["K"])
            elif "n" in s:
                tot += int(s["n"])
            elif "C" in s and "H" in s:
                tot += int(s["C"]) * int(s["H"]) * int(s["W"])
        except (KeyError, TypeError, ValueError):
            pass
    return tot


def discover():
    """{name: {quant, graph, macs, base}} for every model with an IR on disk."""
    out = {}
    for g in glob.glob(f"{MB}/examples/*/*/generated/graph.json"):
        parts = g.split("/")
        name, quant = parts[-4], parts[-3]
        # only the pristine trees: skip split/probe/variant-of-variant trees
        if any(t in name for t in ("_sw_", "_iso", "_ax", "bestax", "_ec",
                                   "_aln", "_hwc", "_oh", "_oc", "_zc", "_kcov",
                                   "_kopt", "_qshard", "arm", "bnfold", "_bx",
                                   "_e2e", "_pad", "_shard", "sweep", "_conv",
                                   "_regenprobe", "_ctl", "_v2", "_64", "_xref")):
            continue
        base = name.split("_s")[0] if "_s" in name and name.split("_s")[-1] in "abcdefgh" else name
        if base not in ANCHOR:
            continue
        try:
            gr = json.load(open(g))
        except Exception:
            continue
        # A model with BOTH quants lowered (fastdepth is the first) would
        # otherwise let glob order pick one, and the loser silently wins about
        # half the time -- yielding a dispatch_deps_path for a quant that was
        # never emitted. int8 is the sweep's canonical quant; mlp_control is
        # fp32-only and still resolves, because this only breaks a tie.
        prev = out.get(name)
        if prev and prev["quant"] == "int8" and quant != "int8":
            continue
        out[name] = {"quant": quant, "graph": g, "macs": macs(gr), "base": base,
                     "ops": len(gr.get("ops", []))}
    return out


def cost_ms(info, models, name=None):
    """(gemmini ms, rvv ms): measured where we have it, else scaled analytically."""
    if name in MEASURED:
        return MEASURED[name]
    base = info["base"]
    ref = models.get(f"{base}_{DEFAULT_RUNG[base]}") if base in DEFAULT_RUNG else None
    ref = ref or models.get(base)
    if not ref or not ref["macs"]:
        return ANCHOR[base]
    r = info["macs"] / ref["macs"]
    g, v = ANCHOR[base]
    return (g * r, v * r)


def net_entry(nid, name, info, period=None, instances=None):
    e = {"id": nid, "identifier": name,
         "dispatch_deps_path":
             f"{GEN}/vmfb/{name}/{TARGET}/gemmini_q31/{name}.{info['quant']}/"
             f"{name}.{info['quant']}_dispatch_graph.json"}
    if period is not None:
        e["period"] = round(period, 3)
        e["window_duration"] = round(period, 3)
    if instances is not None:
        e["num_instances"] = instances
    return e


def build(models):
    """[(family, {net: (period_mult|None, instances)})] -- periods derived below."""
    have = lambda n: n in models
    F = []

    def add(name, spec, edges=()):
        """One family. `edges` are network-level {from,to} precedence pairs."""
        F.append((name, spec, [dict(e) for e in edges]))

    # Each spec: name -> (instances, period as a MULTIPLE of that model's own
    # worst-backend cost; None = aperiodic/one-shot)
    if have("mlp_control_sa") and have("dronet_sa"):
        add("tight_loop", {"mlp_control_sa": (16, 3.0), "mlp_control_sb": (8, 4.0),
                            "dronet_sa": (4, 6.0)})
    if have("mlp_control_sd") and have("dronet_se") and have("yolov8_nano_sc"):
        add("control_mix", {"mlp_control_sd": (8, 4.0), "dronet_se": (4, 5.0),
                             "yolov8_nano_sc": (1, None)})
    if have("yolov8_nano_sf") and have("mlp_control_sd"):
        add("perception_heavy", {"yolov8_nano_sf": (1, None),
                                 "mlp_control_sd": (4, 8.0)})
    ladder = {f"dronet_s{k}": (1, None) for k in "bcdefg" if have(f"dronet_s{k}")}
    if len(ladder) >= 4:
        add("scale_ladder", ladder)
    if have("yolov8_nano_sh") and have("mlp_control_sa"):
        add("bimodal", {"yolov8_nano_sh": (1, None), "mlp_control_sa": (32, 2.0)})
    if have("vint"):
        # mlp_control_sf, not a smaller rung: sa/sb/sd have nothing above the
        # ~5 us launch floor, so pairing ViNT (which we deliberately leave
        # unsplit) with one of those makes the sharded arm byte-identical to
        # the base arm -- 6 FPGA runs that can only ever report 1.00x.
        if have("mlp_control_sf"):
            add("vint_intro", {"vint": (1, None), "mlp_control_sf": (8, 6.0)})
        if have("dronet_se") and have("mlp_control_sd"):
            add("vint_multi", {"vint": (1, None), "dronet_se": (4, 8.0),
                                "mlp_control_sd": (8, 6.0)})
    if have("yolov8_nano_se") and have("dronet_sf") and have("mlp_control_sf"):
        add("saturation", {"yolov8_nano_se": (2, None), "dronet_sf": (6, 2.0),
                           "mlp_control_sf": (16, 1.5)})
    # fastdepth -> dronet: a genuine data-dependent perception pipeline, and
    # the only family set here with a cross-network edge. Instance counts on
    # the two ends are EQUAL so workload_factory pairs tick i to tick i rather
    # than falling back to its different-rates "newest closed producer" rule.
    if have("fastdepth") and have("dronet_sf"):
        chain = [{"from": "fastdepth", "to": "dronet_sf"}]
        pipe = {"fastdepth": (2, 1.5), "dronet_sf": (2, 1.5)}
        add("depth_chain", dict(pipe), chain)
        if have("mlp_control_sd"):
            add("depth_nav", {**pipe, "mlp_control_sd": (16, 4.0)}, chain)
        if have("yolov8_nano_sc"):
            add("depth_contended", {**pipe, "yolov8_nano_sc": (1, None)}, chain)
    return F


def chain_groups(spec, edges):
    """{name: [names in its dependency chain]} -- union-find over the edges."""
    parent = {n: n for n in spec}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    grp = collections.defaultdict(list)
    for n in spec:
        grp[find(n)].append(n)
    return {n: grp[find(n)] for n in spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=f"{R}/soc/sw/xpu-rt/data/toplevel/wl_sweep")
    ap.add_argument("--pairs", default="rvvpair,gempair,hetero,quad")
    a = ap.parse_args()

    models = discover()
    print(f"  discovered {len(models)} models with an IR on disk")
    fams = build(models)
    pairs = [p for p in a.pairs.split(",") if p in PAIRS]
    n = 0
    print(f"\n  {'workload':<26}{'pair':<9}{'networks':<46}{'est busy ms':>12}")
    print("  " + "-" * 96)
    for fam, spec, edges in fams:
        for pair in pairs:
            # (slowest, fastest) backend cost for each model on THIS pair
            cost = {}
            for name in spec:
                g, v = cost_ms(models[name], models, name)
                cands = ([g] if PAIRS[pair]["cpu_p"] else []) + \
                        ([v] if PAIRS[pair]["cpu_e"] else [])
                cost[name] = (max(cands), min(cands))
            chains = chain_groups(spec, edges)
            nets, nid, busy = {}, 0, 0.0
            for name, (inst, pmult) in spec.items():
                worst, best = cost[name]
                grp = chains[name]
                if len(grp) > 1 and pmult is not None:
                    # A pipeline ticks at ONE rate. The period has to cover the
                    # whole chain's latency, not this member's share of it --
                    # otherwise the consumer gets a period shorter than its
                    # producer's runtime and can never actually be fed.
                    worst = sum(cost[m][0] for m in grp)
                    pmult = max(spec[m][1] for m in grp if spec[m][1] is not None)
                per = worst * pmult if pmult else None
                nets[name] = net_entry(nid, name, models[name], per, inst)
                busy += best * inst
                nid += 1
            doc = {
                "_comment": (f"workload family '{fam}' on the {pair} machine "
                             f"configuration ({PAIRS[pair]['cpu_p']} gemmini + "
                             f"{PAIRS[pair]['cpu_e']} rvv harts). "
                             f"@generated by experiments/workload_gen/mk_workloads.py. "
                             f"Periods are derived from each model's analytic MAC count "
                             f"anchored to the measured single-hart runtime of its "
                             f"default rung, so they are tight but satisfiable."
                             + (f" Networks joined by `edges` form a dependency "
                                f"chain and share one derived period, covering "
                                f"the whole chain's latency." if edges else "")),
                "hardware": {
                    "machines": PAIRS[pair],
                    "profile_hw": {"cpu_p": "gemmini_q31", "cpu_e": "V256D128_rvv"},
                    "profile": {"target": TARGET, "topo_tag": "topo_0",
                                "topo_tag_override": True, "gen_root": GEN},
                    "p_core_speedup": 1.0},
                "scheduler": {"random_seed": 42, "solver_verbosity": 2,
                              "time_limit": 120, "use_profiled": True,
                              "prune_periodic": True,
                              "restrict_makespan_to_nonperiodic": False},
                "networks": nets}
            if edges:
                doc["edges"] = edges
            label = ",".join(f"{k.split('_')[-1] if '_s' in k else k}x{v['num_instances']}"
                             if "num_instances" in v else k.split("_")[-1]
                             for k, v in nets.items())
            if edges:
                label += "  [" + ",".join(f"{e['from']}->{e['to']}" for e in edges) + "]"
            print(f"  {fam:<26}{pair:<9}{label[:45]:<46}{busy:>12.1f}")
            if a.emit:
                os.makedirs(a.out, exist_ok=True)
                with open(f"{a.out}/networks_{fam}_{pair}.json", "w") as f:
                    json.dump(doc, f, indent=1)
                n += 1
    print(f"\n  {'wrote '+str(n)+' workload JSONs to '+a.out if a.emit else '(dry run -- pass --emit)'}")


main()
