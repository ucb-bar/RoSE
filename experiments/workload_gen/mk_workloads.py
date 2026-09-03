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
"""
import argparse, json, os, glob, sys, collections

R = "/scratch/dima/rose-infra/RoSE"
MB = f"{R}/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
GEN = "zephyr-chipyard-sw/gen"
TARGET = "firesim_f2_rocket_saturn"
PAIRS = {"rvvpair": {"cpu_p": 0, "cpu_e": 2},
         "gempair": {"cpu_p": 2, "cpu_e": 0},
         "hetero":  {"cpu_p": 1, "cpu_e": 1}}

# Measured ms for the DEFAULT rung, single hart, from the fresh serial runs.
# These anchor the analytic estimate so periods come out in real units.
ANCHOR = {                       # model -> (gemmini ms, rvv ms)
    "mlp_control": (0.547, 0.596),
    "dronet":      (4.988, 7.912),
    "yolov8_nano": (90.045, 167.603),
    "vint":        (5272.755, 17028.698),
}
DEFAULT_RUNG = {"mlp_control": "sd", "dronet": "se", "yolov8_nano": "sd"}


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
        out[name] = {"quant": quant, "graph": g, "macs": macs(gr), "base": base,
                     "ops": len(gr.get("ops", []))}
    return out


def cost_ms(info, models):
    """(gemmini ms, rvv ms) scaled from the base network's measured default."""
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
    # Each spec: name -> (instances, period as a MULTIPLE of that model's own
    # worst-backend cost; None = aperiodic/one-shot)
    if have("mlp_control_sa") and have("dronet_sa"):
        F.append(("tight_loop", {"mlp_control_sa": (16, 3.0), "mlp_control_sb": (8, 4.0),
                                 "dronet_sa": (4, 6.0)}))
    if have("mlp_control_sd") and have("dronet_se") and have("yolov8_nano_sc"):
        F.append(("control_mix", {"mlp_control_sd": (8, 4.0), "dronet_se": (4, 5.0),
                                  "yolov8_nano_sc": (1, None)}))
    if have("yolov8_nano_sf") and have("mlp_control_sd"):
        F.append(("perception_heavy", {"yolov8_nano_sf": (1, None),
                                       "mlp_control_sd": (4, 8.0)}))
    ladder = {f"dronet_s{k}": (1, None) for k in "bcdefg" if have(f"dronet_s{k}")}
    if len(ladder) >= 4:
        F.append(("scale_ladder", ladder))
    if have("yolov8_nano_sh") and have("mlp_control_sa"):
        F.append(("bimodal", {"yolov8_nano_sh": (1, None), "mlp_control_sa": (32, 2.0)}))
    if have("vint"):
        # mlp_control_sf, not a smaller rung: sa/sb/sd have nothing above the
        # ~5 us launch floor, so pairing ViNT (which we deliberately leave
        # unsplit) with one of those makes the sharded arm byte-identical to
        # the base arm -- 6 FPGA runs that can only ever report 1.00x.
        if have("mlp_control_sf"):
            F.append(("vint_intro", {"vint": (1, None), "mlp_control_sf": (8, 6.0)}))
        if have("dronet_se") and have("mlp_control_sd"):
            F.append(("vint_multi", {"vint": (1, None), "dronet_se": (4, 8.0),
                                     "mlp_control_sd": (8, 6.0)}))
    if have("yolov8_nano_se") and have("dronet_sf") and have("mlp_control_sf"):
        F.append(("saturation", {"yolov8_nano_se": (2, None), "dronet_sf": (6, 2.0),
                                 "mlp_control_sf": (16, 1.5)}))
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=f"{R}/soc/sw/xpu-rt/data/toplevel/wl_sweep")
    ap.add_argument("--pairs", default="rvvpair,gempair,hetero")
    a = ap.parse_args()

    models = discover()
    print(f"  discovered {len(models)} models with an IR on disk")
    fams = build(models)
    pairs = [p for p in a.pairs.split(",") if p in PAIRS]
    n = 0
    print(f"\n  {'workload':<26}{'pair':<9}{'networks':<46}{'est busy ms':>12}")
    print("  " + "-" * 96)
    for fam, spec in fams:
        for pair in pairs:
            nets, nid, busy = {}, 0, 0.0
            for name, (inst, pmult) in spec.items():
                info = models[name]
                g, v = cost_ms(info, models)
                # the cost on the SLOWEST backend this pair actually has
                cands = ([g] if PAIRS[pair]["cpu_p"] else []) + \
                        ([v] if PAIRS[pair]["cpu_e"] else [])
                worst = max(cands)
                per = worst * pmult if pmult else None
                nets[name] = net_entry(nid, name, info, per, inst)
                busy += min(cands) * inst
                nid += 1
            doc = {
                "_comment": (f"workload family '{fam}' on the {pair} machine pair. "
                             f"@generated by experiments/workload_gen/mk_workloads.py. "
                             f"Periods are derived from each model's analytic MAC count "
                             f"anchored to the measured single-hart runtime of its "
                             f"default rung, so they are tight but satisfiable."),
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
            label = ",".join(f"{k.split('_')[-1] if '_s' in k else k}x{v['num_instances']}"
                             if "num_instances" in v else k.split("_")[-1]
                             for k, v in nets.items())
            print(f"  {fam:<26}{pair:<9}{label[:45]:<46}{busy:>12.1f}")
            if a.emit:
                os.makedirs(a.out, exist_ok=True)
                with open(f"{a.out}/networks_{fam}_{pair}.json", "w") as f:
                    json.dump(doc, f, indent=1)
                n += 1
    print(f"\n  {'wrote '+str(n)+' workload JSONs to '+a.out if a.emit else '(dry run -- pass --emit)'}")


main()
