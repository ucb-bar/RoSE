"""One (arm, workload, solver) solve, evaluated and validated.

Built as a separate process per job on purpose: the metaheuristics and CP-SAT
have very different core appetites (CP-SAT asks for `workers` search threads,
everything else is single-threaded), so the dispatcher can only size the pool
correctly if a job is a process it can count.

The ten solvers are exactly the set `scripts/solver_study/wl_sweep_bench.py`
defines; the workload build is that file's `build()`, unchanged, so numbers are
comparable with the Sep-3 `wl_sweep_baseline.json` run.
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np

CODE = os.environ["XPURT_CODE_ROOT"]
DATA = os.environ["XPURT_DATA_ROOT"]
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "xpu-rt"))

from workload_factory import create_workload_from_network_hierarchy, build_machine_combinations
from profile_loader import load_profiled_processing_times
from schedule_decoder import DecoderContext, evaluate
import greedy_scheduler as gs
import metaheuristics as mh


def build(spec_path):
    """Verbatim from wl_sweep_bench.build — same seeds, same profile target."""
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
    pt, *_ = load_profiled_processing_times(
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
    return w, nd


def make_solver(name, budget, cpsat_time, cpsat_workers, seed=0):
    if name == "greedy":          return lambda w: gs.greedy_schedule(w)
    if name == "greedy_periodic": return lambda w: gs.greedy_periodic_schedule(w)
    if name == "greedy_reserved": return lambda w: gs.greedy_reserved_schedule(w)
    if name == "decomposed":      return lambda w: gs.decomposed_schedule(w)
    if name == "heft":            return lambda w: mh.heft_schedule(w)
    if name == "heft_edf":        return lambda w: mh.heft_edf_schedule(w)
    if name == "pso":             return lambda w: mh.pso_schedule(w, time_budget=budget, seed=seed)
    if name == "sa":              return lambda w: mh.sa_schedule(w, time_budget=budget, seed=seed)
    if name == "cpsat":
        from cpsat_scheduler import cpsat_schedule
        return lambda w: cpsat_schedule(w, time_limit=cpsat_time, workers=cpsat_workers,
                                        random_seed=seed)
    if name == "cpsat:warm":
        from cpsat_scheduler import cpsat_schedule
        return lambda w: cpsat_schedule(w, time_limit=cpsat_time, workers=cpsat_workers,
                                        random_seed=seed,
                                        warm_start=mh.heft_edf_schedule(w))
    if name == "cpsat:warmbest":
        # Supplementary variant, NOT one of the ten. `cpsat:warm` hints from
        # heft_edf unconditionally, and on depth_contended heft_edf is 119%
        # WORSE than greedy -- so the hint drops CP-SAT into a bad basin and
        # the warm solve still loses to a 0.05 s heuristic. Hinting from
        # whichever cheap heuristic is actually best (feasible-then-fastest)
        # costs ~0.5 s and removes that failure mode by construction.
        from cpsat_scheduler import cpsat_schedule
        from schedule_decoder import DecoderContext as _DC, evaluate as _ev

        def _best(w):
            c = _DC(w)
            cands = []
            for f in (gs.greedy_schedule, gs.greedy_periodic_schedule,
                      gs.greedy_reserved_schedule, gs.decomposed_schedule,
                      mh.heft_schedule, mh.heft_edf_schedule):
                try:
                    t, al = f(w)
                    o, ms, _ = _ev(c, t, al, True)
                    cands.append(((1 if ms > 0 else 0, ms, o), (t, al)))
                except Exception:
                    continue
            cands.sort(key=lambda x: x[0])
            return cands[0][1] if cands else None

        return lambda w: cpsat_schedule(w, time_limit=cpsat_time, workers=cpsat_workers,
                                        random_seed=seed, warm_start=_best(w))
    raise ValueError(f"unknown solver {name}")


def validate(ctx, t, alpha, tol=1e-6):
    """Independent feasibility audit of the float schedule the solver returned.

    `evaluate` only scores makespan and window misses; it never asks whether
    the schedule is *executable*. CP-SAT in particular solves on an integer
    microsecond grid and hands back floats, so precedence and no-overlap have
    to be re-checked in the float arithmetic the rest of the pipeline uses.
    """
    n = ctx.n
    combo = np.argmax(alpha, axis=1)
    dur = ctx.dur[np.arange(n), combo]
    n_inf = int(np.sum(~np.isfinite(dur)))
    d = np.where(np.isfinite(dur), dur, 0.0)
    end = t + d

    prec_viol, prec_worst = 0, 0.0
    for i in range(n):
        for p in ctx.pred[i]:
            need = end[p] + ctx.transfer[ctx.first_machine[combo[p]]][ctx.first_machine[combo[i]]]
            if need > t[i] + tol:
                prec_viol += 1
                prec_worst = max(prec_worst, float(need - t[i]))

    # A combination occupies EVERY machine in it, so overlap is per machine,
    # not per combination index.
    per_machine = defaultdict(list)
    for i in range(n):
        if d[i] <= 0:
            continue
        for m in ctx.combos[combo[i]]:
            per_machine[m].append((float(t[i]), float(end[i])))
    ov, ov_worst = 0, 0.0
    for lst in per_machine.values():
        lst.sort()
        cur = -np.inf
        for s, e in lst:
            if s < cur - tol:
                ov += 1
                ov_worst = max(ov_worst, float(cur - s))
            cur = max(cur, e)

    neg = int(np.sum(t < -tol))
    early = int(np.sum(t < ctx.min_start - tol))
    return dict(prec_viol=prec_viol, prec_worst_ms=round(prec_worst, 9),
                overlap_viol=ov, overlap_worst_ms=round(ov_worst, 9),
                inf_dur_assign=n_inf, neg_start=neg, before_min_start=early)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--cpsat-time", type=float, default=60.0)
    ap.add_argument("--cpsat-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rec = dict(arm=a.arm, workload=a.name, solver=a.solver, seed=a.seed)
    spec = os.path.join(DATA, "data", "toplevel", a.arm, a.name + ".json")
    try:
        t0 = time.perf_counter()
        w, nd = build(spec)
        rec["build_s"] = round(time.perf_counter() - t0, 3)
        ctx = DecoderContext(w)
        rec["ops"] = int(ctx.n)
        rec["combos"] = int(ctx.n_combos)
        rec["periodic_ops"] = int(ctx.periodic.sum())
        rec["lanes"] = (f"{nd['hardware']['profile_hw'].get('cpu_p')}"
                        f"+{nd['hardware']['profile_hw'].get('cpu_e')}")
    except Exception as e:
        rec["error"] = f"build: {type(e).__name__}: {e}"
        json.dump(rec, open(a.out, "w"), indent=1)
        return

    fn = make_solver(a.solver, a.budget, a.cpsat_time, a.cpsat_workers, a.seed)
    t0 = time.perf_counter()
    try:
        t, alpha = fn(w)
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        obj, misses, all_end = evaluate(ctx, t, alpha, True)
        rec.update(objective=round(float(obj), 6), all_ops=round(float(all_end), 6),
                   misses=int(misses))
        rec["validation"] = validate(ctx, t, alpha)
        if a.solver.startswith("cpsat"):
            import cpsat_scheduler
            rec["cpsat"] = dict(cpsat_scheduler.LAST_SOLVE)
    except Exception as e:
        import traceback
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
