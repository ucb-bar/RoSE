#!/usr/bin/env python3
"""Emit ONE sweep10 (arm, workload, solver) schedule as a `scheduled_*.json`
the modelblaster xpurt codegen can ingest.

Why this exists rather than `scripts/run_xpurt_schedule.py`:

  * `run_xpurt_schedule.py` only exposes four solvers (milp, greedy,
    greedy_periodic, decomposed). Ten of the twelve sweep10 entries -- and
    every one of the winners -- come from the standalone XPU-RT solver tree
    (`metaheuristics.py`, `cpsat_scheduler.py`), which that script cannot
    reach.
  * `run_xpurt_schedule.py` also wraps greedy in a periodic-instance
    refinement loop and post-trims the schedule. sweep10 does neither: it
    builds the workload ONCE at the spec's own `num_instances` and hands that
    same instance to every solver.

So the workload build here is byte-for-byte `sweep10_runner.build` (which is
itself verbatim `wl_sweep_bench.build`), and only the emit step is new: the
solver's `(t, alpha)` goes straight into `postprocessing.output_scheduled_json`
-- the same function `run_xpurt_schedule.py` uses, unchanged and shared by both
trees -- so the JSON the codegen sees is the format it already accepts.

Net effect: every ELF built from these schedules corresponds to exactly one row
of `results/all_results.json`, with no harness difference in between.

Usage:
  XPURT_CODE_ROOT=/scratch2/dima/misc_sw/XPU-RT \
  XPURT_DATA_ROOT=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt \
  emit_schedule.py --arm wl_sweep --name networks_bimodal_gempair \
                   --solver pso --out /path/scheduled_x.json
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

CODE = os.environ["XPURT_CODE_ROOT"]
DATA = os.environ["XPURT_DATA_ROOT"]
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "xpu-rt"))

from workload_factory import create_workload_from_network_hierarchy, build_machine_combinations
from profile_loader import load_profiled_processing_times
from postprocessing import output_scheduled_json
from schedule_decoder import DecoderContext, evaluate

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "scripts", "sweep10_runner.py")


def _load_runner():
    """Reuse sweep10_runner's make_solver/validate verbatim -- the winners have
    to come out of the same code that produced the predictions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep10_runner", os.path.abspath(_RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(spec_path):
    """Verbatim sweep10_runner.build, but keeps the profile side-channels that
    `output_scheduled_json` needs for per-dispatch `module_name`."""
    nd = json.load(open(spec_path))
    hw = nd["hardware"]
    machines, combos = build_machine_combinations(
        {k.upper(): v for k, v in hw["machines"].items()})
    phw = {k.lower(): v for k, v in hw["profile_hw"].items()}
    combo_hw = [phw[c[0].split("#")[0].lower()] for c in combos]
    tt = np.zeros((len(machines), len(machines)))
    prof = hw.get("profile", {})
    tt_override = (prof.get("topo_tag_per_hw")
                   or (prof.get("topo_tag") if prof.get("topo_tag_override") else None))
    pt, prof_p, prof_e, prof_by_net = load_profiled_processing_times(
        networks=nd["networks"], repo_base_path=DATA, machine_combinations=combos,
        combo_hw=combo_hw, profile_target=prof.get("target", "firesim_f2_rocket_saturn"),
        cpu_p_profile_hw=phw.get("cpu_p", "gemmini_q31"),
        cpu_e_profile_hw=phw.get("cpu_e", "V256D128_rvv"),
        rng=np.random.default_rng(42), p_core_speedup=float(hw.get("p_core_speedup", 1.0)),
        topo_tag_override=tt_override)
    w = create_workload_from_network_hierarchy(
        networks_data=nd, repo_base_path=DATA, machines=machines, transfer_times=tt,
        p_core_speedup=float(hw.get("p_core_speedup", 1.0)), random_seed=42,
        processing_times=pt, machine_combinations=combos)
    return w, nd, prof_p, prof_e, prof_by_net


def _best_of_fast(w):
    """`best-of-fast` is not a solver in `make_solver` -- sweep10 SYNTHESISED it
    in the analyzer from the six sub-second heuristics' own rows
    (`sweep10_analyze.py:86`, feasible-then-fastest). Reproduce that selection
    here so the row gets a real schedule, and so its dedupe identity with the
    member it picks is measured rather than assumed."""
    import greedy_scheduler as gs
    import metaheuristics as mh
    ctx = DecoderContext(w)
    cands = []
    for name, f in (("greedy", gs.greedy_schedule),
                    ("greedy_periodic", gs.greedy_periodic_schedule),
                    ("greedy_reserved", gs.greedy_reserved_schedule),
                    ("decomposed", gs.decomposed_schedule),
                    ("heft", mh.heft_schedule),
                    ("heft_edf", mh.heft_edf_schedule)):
        try:
            t, al = f(w)
            o, ms, _ = evaluate(ctx, t, al, True)
            cands.append(((1 if ms > 0 else 0, ms, o), name, (t, al)))
        except Exception:
            continue
    cands.sort(key=lambda x: x[0])
    if not cands:
        raise RuntimeError("best-of-fast: no member returned a schedule")
    return cands[0][2], cands[0][1]


def _sched_hash(doc):
    """Dedupe key = the op -> (combination, start, duration) assignment, i.e.
    everything the codegen reads out of the schedule. NOT the objective: two
    different assignments can share a makespan, and collapsing those would
    silently drop coverage."""
    disp = doc["dispatches"]
    items = sorted(disp.items()) if isinstance(disp, dict) else \
        sorted(((str(i), d) for i, d in enumerate(disp)))
    h = hashlib.sha256()
    for k, d in items:
        h.update(k.encode())
        h.update(b"\0")
        h.update(json.dumps(d, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)          # wl_sweep | wl_sweep_shard
    ap.add_argument("--name", required=True)         # networks_<family>_<cfg>
    ap.add_argument("--solver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-out", default=None)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--cpsat-time", type=float, default=60.0)
    ap.add_argument("--cpsat-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runner = _load_runner()
    spec = os.path.join(DATA, "data", "toplevel", a.arm, a.name + ".json")
    w, nd, prof_p, prof_e, prof_by_net = build(spec)
    ctx = DecoderContext(w)

    picked = None
    t0 = time.perf_counter()
    if a.solver == "best-of-fast":
        (t, alpha), picked = _best_of_fast(w)
    else:
        fn = runner.make_solver(a.solver, a.budget, a.cpsat_time, a.cpsat_workers, a.seed)
        t, alpha = fn(w)
    wall = round(time.perf_counter() - t0, 3)
    obj, misses, all_end = evaluate(ctx, t, alpha, True)
    val = runner.validate(ctx, t, alpha)

    phw = {k.upper(): v for k, v in nd["hardware"]["profile_hw"].items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    output_scheduled_json(
        combined_workload=w, t=t, alpha=alpha, output_path=a.out,
        profiled_times_p=prof_p, profiled_times_e=prof_e,
        profile_hw=phw, profiled_times_by_network=prof_by_net)

    doc = json.load(open(a.out))
    disp = doc["dispatches"]
    vals = list(disp.values()) if isinstance(disp, dict) else disp
    rec = dict(arm=a.arm, workload=a.name, solver=a.solver, seed=a.seed,
               ops=int(ctx.n), periodic_ops=int(ctx.periodic.sum()),
               combos=int(ctx.n_combos), wall_s=wall,
               objective=round(float(obj), 6), all_ops=round(float(all_end), 6),
               misses=int(misses), validation=val,
               dispatches=len(vals),
               json_makespan=round(max(x["start_time"] + x["duration"] for x in vals), 6),
               sched_hash=_sched_hash(doc), picked=picked,
               schedule=os.path.abspath(a.out))
    print(json.dumps(rec))
    if a.meta_out:
        json.dump(rec, open(a.meta_out, "w"), indent=1)


if __name__ == "__main__":
    main()
