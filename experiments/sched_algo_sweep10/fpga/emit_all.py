#!/usr/bin/env python3
"""Emit schedules for a set of (arm, workload, solver) rows, one process each,
and record the content hash of every emission so identical schedules can share
one ELF.

Two pools, for the same reason `sweep10_dispatch.py` has two: CP-SAT asks for
8 search threads inside its own process (measured at 750% CPU), everything else
is single-threaded. Running them at one width either starves the heuristics or
oversubscribes the box 8x.
"""
import argparse, json, os, subprocess, sys, time
import concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
EMIT = os.path.join(HERE, "emit_schedule.py")
PY = os.environ.get("EMIT_PYTHON", sys.executable)


def one(job):
    arm, wl, solver, outdir = job
    tag = f"{arm}__{wl}__{solver.replace(':', '-')}"
    out = os.path.join(outdir, tag + ".json")
    meta = os.path.join(outdir, tag + ".meta.json")
    if os.path.exists(meta) and os.path.exists(out):
        try:
            return json.load(open(meta)) | {"cached": True}
        except Exception:
            pass
    t0 = time.time()
    p = subprocess.run([PY, EMIT, "--arm", arm, "--name", wl, "--solver", solver,
                        "--out", out, "--meta-out", meta],
                       capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip().splitlines()
        return dict(arm=arm, workload=wl, solver=solver, tag=tag,
                    error=err[-1] if err else f"rc={p.returncode}",
                    traceback="\n".join(err[-12:]), elapsed=round(time.time() - t0, 1))
    rec = json.load(open(meta))
    rec["tag"] = tag
    rec["elapsed"] = round(time.time() - t0, 1)
    json.dump(rec, open(meta, "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON list of [arm, workload, solver]")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cheap-workers", type=int, default=24)
    ap.add_argument("--cpsat-parallel", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = json.load(open(a.jobs))
    cheap = [(r[0], r[1], r[2], a.outdir) for r in rows if not r[2].startswith("cpsat")]
    heavy = [(r[0], r[1], r[2], a.outdir) for r in rows if r[2].startswith("cpsat")]
    out = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max(1, a.cheap_workers)) as ex1, \
         cf.ThreadPoolExecutor(max(1, a.cpsat_parallel)) as ex2:
        futs = [ex1.submit(one, j) for j in cheap] + [ex2.submit(one, j) for j in heavy]
        done = 0
        for f in cf.as_completed(futs):
            out.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    ok = [r for r in out if "error" not in r]
    print(f"{len(ok)}/{len(out)} emitted -> {a.out}")
    for r in out:
        if "error" in r:
            print(f"  FAIL {r['tag']}: {r['error']}")


if __name__ == "__main__":
    main()
