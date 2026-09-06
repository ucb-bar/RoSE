"""Fan the 880 (arm x workload x solver) solves across the box.

Two pools, not one: CP-SAT sets `num_search_workers=8` inside its own
subprocess, so a CP-SAT job costs eight cores while every other solver costs
one. Running them in a single 96-wide pool would oversubscribe by ~8x during
the CP-SAT phase and inflate every wall-clock number in the study -- which is
one of the things being measured.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

CHEAP = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
         "heft", "heft_edf", "pso", "sa"]
HEAVY = ["cpsat", "cpsat:warm", "cpsat:warmbest"]


def jobs(data_root, arms):
    out = []
    for arm in arms:
        d = os.path.join(data_root, "data", "toplevel", arm)
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                out.append((arm, f[:-5]))
    return out


def run(job, outdir, runner, budget, cpsat_time, cpsat_workers, env):
    arm, name, solver = job
    tag = f"{arm}__{name}__{solver.replace(':', '-')}"
    op = os.path.join(outdir, tag + ".json")
    if os.path.exists(op) and os.path.getsize(op) > 0:
        return tag, "cached"
    cmd = [sys.executable, runner, "--arm", arm, "--name", name, "--solver", solver,
           "--out", op, "--budget", str(budget), "--cpsat-time", str(cpsat_time),
           "--cpsat-workers", str(cpsat_workers)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=cpsat_time + 900)
    except subprocess.TimeoutExpired:
        json.dump(dict(arm=arm, workload=name, solver=solver,
                       error="TimeoutExpired: runner exceeded "
                             f"{cpsat_time + 900:.0f}s",
                       wall_s=round(time.time() - t0, 2)), open(op, "w"), indent=1)
        return tag, "TIMEOUT"
    if p.returncode != 0 and not os.path.exists(op):
        json.dump(dict(arm=arm, workload=name, solver=solver,
                       error=f"runner rc={p.returncode}",
                       stderr=(p.stderr or "")[-3000:],
                       wall_s=round(time.time() - t0, 2)), open(op, "w"), indent=1)
    return tag, f"{time.time() - t0:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--cpsat-python", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arms", default="wl_sweep,wl_sweep_shard")
    ap.add_argument("--solvers", default="")
    ap.add_argument("--cheap-workers", type=int, default=90)
    ap.add_argument("--cpsat-workers", type=int, default=8)
    ap.add_argument("--cpsat-parallel", type=int, default=12)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--cpsat-time", type=float, default=60.0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep10_runner.py")
    env = dict(os.environ)
    env.update(XPURT_CODE_ROOT=a.code_root, XPURT_DATA_ROOT=a.data_root,
               XPURT_CPSAT_PYTHON=a.cpsat_python)
    # One BLAS thread per job: these are tiny dense ops and a 96-way pool of
    # 96-thread BLAS pools is pure contention.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[v] = "1"

    arms = [x for x in a.arms.split(",") if x]
    wls = jobs(a.data_root, arms)
    sel = set(x for x in a.solvers.split(",") if x)
    cheap = [(arm, n, s) for arm, n in wls for s in CHEAP if not sel or s in sel]
    heavy = [(arm, n, s) for arm, n in wls for s in HEAVY if not sel or s in sel]
    print(f"{len(wls)} workload-arms; {len(cheap)} cheap jobs, {len(heavy)} cpsat jobs",
          flush=True)

    for label, batch, par in (("cheap", cheap, a.cheap_workers),
                              ("cpsat", heavy, a.cpsat_parallel)):
        if not batch:
            continue
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=par) as ex:
            futs = [ex.submit(run, j, a.outdir, runner, a.budget, a.cpsat_time,
                              a.cpsat_workers, env) for j in batch]
            for f in futs:
                try:
                    tag, msg = f.result()
                except Exception as e:
                    tag, msg = "?", f"DISPATCH-FAIL {type(e).__name__}: {e}"
                done += 1
                if done % 25 == 0 or done == len(batch):
                    print(f"  [{label}] {done}/{len(batch)}  {time.time()-t0:.0f}s  last={tag} {msg}",
                          flush=True)
        print(f"{label} phase done in {time.time()-t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
