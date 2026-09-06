"""Seed-repeat run: how much of a solver's margin is just search noise?

CP-SAT with `num_search_workers=8` is NOT reproducible -- the result depends on
thread interleaving -- and pso/sa are stochastic by construction. Without a
repeat the study cannot say whether, e.g., cpsat:warm beating cpsat by 1% is a
real effect of the warm start or the spread of two nondeterministic solves.
Re-runs the four stochastic solvers at extra seeds on a family-spanning subset.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

FAMILIES = ["bimodal", "control_mix", "depth_chain", "depth_contended", "depth_nav",
            "perception_heavy", "saturation", "scale_ladder", "vint_intro", "vint_multi"]
SUBSET = ([("wl_sweep", f"networks_{f}_gempair") for f in FAMILIES] +
          [("wl_sweep_shard", f"networks_{f}_quad") for f in FAMILIES])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--cpsat-python", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--cpsat-parallel", type=int, default=12)
    ap.add_argument("--cheap-workers", type=int, default=88)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep10_runner.py")
    env = dict(os.environ)
    env.update(XPURT_CODE_ROOT=a.code_root, XPURT_DATA_ROOT=a.data_root,
               XPURT_CPSAT_PYTHON=a.cpsat_python)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"
    seeds = [int(x) for x in a.seeds.split(",") if x]

    def run(job):
        arm, name, solver, seed = job
        op = os.path.join(a.outdir, f"{arm}__{name}__{solver.replace(':','-')}__s{seed}.json")
        if os.path.exists(op) and os.path.getsize(op) > 0:
            return op, "cached"
        cmd = [sys.executable, runner, "--arm", arm, "--name", name, "--solver", solver,
               "--out", op, "--seed", str(seed), "--budget", "20.0", "--cpsat-time", "60.0"]
        t0 = time.time()
        try:
            subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        except subprocess.TimeoutExpired:
            json.dump(dict(arm=arm, workload=name, solver=solver, seed=seed,
                           error="TimeoutExpired"), open(op, "w"))
        return op, f"{time.time()-t0:.1f}s"

    cheap = [(arm, n, s, sd) for arm, n in SUBSET for s in ("pso", "sa") for sd in seeds]
    heavy = [(arm, n, s, sd) for arm, n in SUBSET for s in ("cpsat", "cpsat:warm") for sd in seeds]
    for label, batch, par in (("cheap", cheap, a.cheap_workers), ("cpsat", heavy, a.cpsat_parallel)):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=par) as ex:
            list(ex.map(run, batch))
        print(f"{label}: {len(batch)} jobs in {time.time()-t0:.0f}s", flush=True)
    print("VARIANCE DONE", flush=True)


if __name__ == "__main__":
    main()
