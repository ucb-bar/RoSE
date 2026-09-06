#!/usr/bin/env python3
"""
Scheduler-ALGORITHM sweep over the wl_sweep workload families.

Runs run_xpurt_schedule.py for every (arm, workload, solver) cell, each in its
OWN working directory.  That isolation is not cosmetic: the script writes
`schedules/scheduled_<basename>_<solvertag>_profiled.json` and
`plots/<basename>_<solvertag>_profiled.png` relative to CWD, and the basename
does NOT carry the arm -- `wl_sweep/networks_bimodal_gempair.json` and
`wl_sweep_shard/networks_bimodal_gempair.json` share a basename, so a shared
CWD would silently overwrite one arm's result with the other's.

For each cell we record: solver status, predicted makespan (max over dispatches
of start+duration), operation count, wall-clock solve time, per-machine and
per-machine-kind busy time / utilisation, and periodic-window (period) misses.

Usage:
  sweep_driver.py --arms base shard --solvers greedy milp --jobs 48
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                wait)

REPO = os.path.expanduser("~/xpu-rt")
SCRIPT = os.path.join(REPO, "scripts", "run_xpurt_schedule.py")
ARM_DIR = {
    "base": os.path.join(REPO, "data", "toplevel", "wl_sweep"),
    "shard": os.path.join(REPO, "data", "toplevel", "wl_sweep_shard"),
}
SOLVER_TAG = {
    "milp": "",
    "greedy": "_greedy",
    "greedy_periodic": "_greedy_periodic",
    "decomposed": "_decomposed",
}
EPS = 1e-6

# family/pair split: networks_<family>_<pair>.json, pair in this set
PAIRS = ("gempair", "hetero", "quad", "rvvpair")


def split_basename(base: str) -> tuple[str, str]:
    stem = base[len("networks_"):] if base.startswith("networks_") else base
    for p in PAIRS:
        if stem.endswith("_" + p):
            return stem[: -(len(p) + 1)], p
    return stem, "unknown"


def periodic_spec(networks_json: str) -> tuple[dict, dict]:
    """Return ({net_id: {period, window, start_time, declared_instances}}, raw hw cfg)."""
    with open(networks_json) as f:
        d = json.load(f)
    per = {}
    for nid, info in d.get("networks", {}).items():
        if info.get("period") is not None and info.get("window_duration") is not None:
            per[nid] = {
                "period": float(info["period"]),
                "window": float(info["window_duration"]),
                "start_time": float(info.get("start_time", 0.0) or 0.0),
                "declared_instances": info.get("num_instances"),
            }
    return per, d.get("hardware", {})


def instance_index(job_name: str, net_ids) -> tuple[str, int] | None:
    """`mlp_control_sa7` -> ('mlp_control_sa', 7). Longest prefix wins."""
    for nid in sorted(net_ids, key=len, reverse=True):
        if job_name.startswith(nid):
            rest = job_name[len(nid):]
            if rest.isdigit():
                return nid, int(rest)
    return None


def analyse_schedule(sched_path: str, networks_json: str) -> dict:
    with open(sched_path) as f:
        sched = json.load(f)
    disp = sched["dispatches"]
    meta = sched.get("metadata", {})
    per, hw = periodic_spec(networks_json)

    makespan = 0.0
    total_busy = 0.0
    busy_by_machine: dict[str, float] = {}
    ops_by_machine: dict[str, int] = {}
    # per periodic instance span
    inst_span: dict[tuple[str, int], list[float]] = {}
    ops_periodic = 0

    for name, info in disp.items():
        st = float(info["start_time"])
        du = float(info["duration"])
        end = st + du
        makespan = max(makespan, end)
        tgt = info["hardware_target"]
        busy_by_machine[tgt] = busy_by_machine.get(tgt, 0.0) + du
        ops_by_machine[tgt] = ops_by_machine.get(tgt, 0) + 1
        total_busy += du
        jn = info.get("job_name", "")
        hit = instance_index(jn, per.keys()) if per else None
        if hit is not None:
            ops_periodic += 1
            key = hit
            cur = inst_span.get(key)
            if cur is None:
                inst_span[key] = [st, end]
            else:
                cur[0] = min(cur[0], st)
                cur[1] = max(cur[1], end)

    # machine KIND (CPU_P / CPU_E) roll-up
    profile_hw = meta.get("profile_hw", {})
    busy_by_kind: dict[str, float] = {}
    machines = meta.get("machines", [])
    n_by_kind: dict[str, int] = {}
    for m in machines:
        kind = m.split("#")[0]
        n_by_kind[kind] = n_by_kind.get(kind, 0) + 1
        busy_by_kind.setdefault(kind, 0.0)
    for tgt, b in busy_by_machine.items():
        for part in tgt.split("+"):
            kind = part.split("#")[0]
            busy_by_kind[kind] = busy_by_kind.get(kind, 0.0) + b / len(tgt.split("+"))

    util_by_machine = {m: (busy_by_machine.get(m, 0.0) / makespan if makespan > 0 else 0.0)
                       for m in machines}
    util_by_kind = {}
    for kind, b in busy_by_kind.items():
        cap = makespan * max(1, n_by_kind.get(kind, 1))
        util_by_kind[kind] = (b / cap) if cap > 0 else 0.0

    # periodic window misses
    misses = []
    inst_counts: dict[str, int] = {}
    for (nid, i), (s, e) in sorted(inst_span.items()):
        inst_counts[nid] = inst_counts.get(nid, 0) + 1
        spec = per[nid]
        w_open = spec["start_time"] + i * spec["period"]
        w_close = w_open + spec["window"]
        late = e - w_close
        early = w_open - s
        if late > 1e-6 or early > 1e-6:
            misses.append({
                "network": nid, "instance": i,
                "window": [w_open, w_close],
                "actual": [s, e],
                "late_by_ms": max(0.0, late),
                "early_by_ms": max(0.0, early),
            })

    return {
        "makespan_pred_ms": makespan,
        "makespan_meta_ms": meta.get("makespan"),
        "num_operations": len(disp),
        "num_periodic_operations": ops_periodic,
        "num_nonperiodic_operations": len(disp) - ops_periodic,
        "total_busy_ms": total_busy,
        "machines": machines,
        "profile_hw": profile_hw,
        "busy_by_machine_ms": busy_by_machine,
        "util_by_machine": util_by_machine,
        "busy_by_kind_ms": busy_by_kind,
        "util_by_kind": util_by_kind,
        "n_machines_by_kind": n_by_kind,
        "ops_by_machine": ops_by_machine,
        "periodic_instances_retained": inst_counts,
        "periodic_declared_instances": {k: v["declared_instances"] for k, v in per.items()},
        "periodic_periods_ms": {k: v["period"] for k, v in per.items()},
        "period_misses": misses,
        "num_period_misses": len(misses),
        "any_period_missed": bool(misses),
        "worst_late_ms": max([m["late_by_ms"] for m in misses], default=0.0),
    }


STATUS_RE = re.compile(r"^Status:\s+(.*)$", re.M)
OPTVAL_RE = re.compile(r"^Optimal value:\s+(.*)$", re.M)
TOTOPS_RE = re.compile(r"Total operations:\s+(\d+)")
# cvxpy verbose banner: splits model build from the actual optimizer call.
COMPILE_RE = re.compile(r"Compilation took ([0-9.eE+-]+) seconds")
SOLVETIME_RE = re.compile(
    r"Solver \(including time spent in interface\) took ([0-9.eE+-]+) seconds")
# MOSEK mixed-integer summary
MIPGAP_RE = re.compile(r"Relative gap\s*(?:\(%\))?\s*:\s*([0-9.eE+-]+)")
MIPOBJ_RE = re.compile(r"Objective of best integer solution\s*:\s*([0-9.eE+-]+)")
MIPBOUND_RE = re.compile(r"Best objective bound\s*:\s*([0-9.eE+-]+)")
MSKTIME_RE = re.compile(r"Optimizer terminated\. Time:\s*([0-9.eE+-]+)")


def _rlimit_preexec(mem_bytes: int):
    def _f():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    return _f


def run_cell(cell: dict) -> dict:
    arm, base, solver = cell["arm"], cell["basename"], cell["solver"]
    workdir = cell["workdir"]
    os.makedirs(workdir, exist_ok=True)
    njson = os.path.join(ARM_DIR[arm], base + ".json")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = cell["mplconfig"]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[v] = str(cell["blas_threads"])

    cmd = [sys.executable, "-u", SCRIPT, "--networks-json", njson,
           "--solver", solver, "--time-limit", str(cell["time_limit"])]
    if solver == "milp":
        # verbosity 1 turns on the cvxpy/MOSEK banner, which is the only
        # place the model-build time, the optimizer time and the MIP gap
        # are reported separately.  It does not change the solve.
        cmd += ["--solver-verbosity", "1"]
    if cell.get("taskset_cpus"):
        cmd = ["taskset", "-c", cell["taskset_cpus"]] + cmd

    pre = None
    if cell.get("mem_limit_gb"):
        pre = _rlimit_preexec(int(cell["mem_limit_gb"] * (1 << 30)))

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                              text=True, timeout=cell.get("hard_timeout", 3600),
                              preexec_fn=pre)
        rc, so, se = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = -9
        so = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        se = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    wall = time.time() - t0

    class _P:  # keep the rest of the function shape
        pass
    proc = _P()
    proc.returncode, proc.stdout, proc.stderr = rc, so, se
    log = proc.stdout + "\n=== STDERR ===\n" + proc.stderr
    with open(os.path.join(workdir, "run.log"), "w") as f:
        f.write(log)

    fam, pair = split_basename(base)
    out = {
        "cell_id": cell["cell_id"],
        "family": fam, "pair": pair, "arm": arm, "solver": solver,
        "basename": base, "networks_json": njson,
        "returncode": proc.returncode,
        "wall_s": wall,
        "timed_out": timed_out,
        "time_limit_s": cell["time_limit"],
        "blas_threads": cell["blas_threads"],
    }
    # cvxpy's verbose banner and MOSEK's own log go to STDERR, not stdout, so
    # these have to be matched against the combined stream.  (`Status:` and
    # `Optimal value:` above are the script's own prints, on stdout.)
    both = proc.stdout + "\n" + proc.stderr
    for key, rx in (("build_s", COMPILE_RE), ("solve_s", SOLVETIME_RE),
                    ("mip_rel_gap", MIPGAP_RE), ("mip_best_obj", MIPOBJ_RE),
                    ("mip_best_bound", MIPBOUND_RE), ("mosek_time_s", MSKTIME_RE)):
        m = rx.search(both)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    m = STATUS_RE.search(proc.stdout)
    out["solver_status"] = m.group(1).strip() if m else ("heuristic" if solver != "milp" else None)
    m = OPTVAL_RE.search(proc.stdout)
    if m:
        try:
            out["milp_objective"] = float(m.group(1).strip())
        except ValueError:
            out["milp_objective"] = m.group(1).strip()
    tots = TOTOPS_RE.findall(proc.stdout)
    if tots:
        out["ops_reported_prescheduling"] = int(tots[-1])

    sched_path = os.path.join(workdir, "schedules",
                              f"scheduled_{base}{SOLVER_TAG[solver]}_profiled.json")
    if proc.returncode == 0 and os.path.exists(sched_path):
        try:
            out.update(analyse_schedule(sched_path, njson))
            out["ok"] = True
            out["schedule_json"] = sched_path
        except Exception as e:  # noqa: BLE001
            out["ok"] = False
            out["error"] = f"analyse failed: {type(e).__name__}: {e}"
    else:
        out["ok"] = False
        tail = (proc.stdout[-1500:] + proc.stderr[-3000:])
        out["error"] = "run failed (rc=%d) or no schedule json" % proc.returncode
        out["tail"] = tail

    with open(os.path.join(cell["celldir"], cell["cell_id"] + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base", "shard"])
    ap.add_argument("--solvers", nargs="+",
                    default=["greedy", "greedy_periodic", "decomposed", "milp"])
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--cpus-per-job", type=int, default=0,
                    help="if >0, taskset each worker onto this many cpus")
    ap.add_argument("--hard-timeout", type=int, default=3600)
    ap.add_argument("--mem-limit-gb", type=float, default=0.0,
                    help="RLIMIT_AS backstop per cell, GB (0 = none)")
    ap.add_argument("--ops-map", default=None,
                    help="combined_*.json from an earlier (heuristic) run; used "
                         "to order cells smallest-workload-first")
    ap.add_argument("--deadline-s", type=float, default=0.0,
                    help="stop SUBMITTING new cells after this many seconds "
                         "(0 = no deadline). Cells never submitted are recorded "
                         "as not_attempted_budget.")
    ap.add_argument("--root", default=os.path.expanduser("~/sweep"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    root = args.root
    celldir = os.path.join(root, "cells")
    workroot = os.path.join(root, "work")
    mplcfg = os.path.join(root, "mplconfig")
    for d in (celldir, workroot, mplcfg):
        os.makedirs(d, exist_ok=True)

    ncpu = os.cpu_count() or 1
    cells = []
    for arm in args.arms:
        for fn in sorted(os.listdir(ARM_DIR[arm])):
            if not fn.endswith(".json"):
                continue
            base = fn[:-5]
            fam, _pair = split_basename(base)
            if args.families and fam not in args.families:
                continue
            for solver in args.solvers:
                cid = f"{arm}__{base}__{solver}"
                wd = os.path.join(workroot, cid)
                if args.fresh and os.path.isdir(wd):
                    shutil.rmtree(wd)
                cells.append({
                    "cell_id": cid, "arm": arm, "basename": base, "solver": solver,
                    "workdir": wd, "celldir": celldir, "mplconfig": mplcfg,
                    "time_limit": args.time_limit,
                    "blas_threads": args.blas_threads,
                    "hard_timeout": args.hard_timeout,
                    "mem_limit_gb": args.mem_limit_gb,
                })
    # Order smallest-workload-first: the MILP model is O(n^2) in cvxpy scalar
    # constraints, so small workloads are where it has any chance of finishing,
    # and running them first means a budget cut only ever loses the cells that
    # were least likely to produce a schedule.
    if args.ops_map and os.path.exists(args.ops_map):
        with open(args.ops_map) as fh:
            prev = json.load(fh)
        ops = {}
        for r in prev:
            if r.get("ok") and r.get("num_operations"):
                k = (r["arm"], r["basename"])
                ops[k] = max(ops.get(k, 0), r["num_operations"])
        for c in cells:
            c["ops_hint"] = ops.get((c["arm"], c["basename"]), 10 ** 9)
        cells.sort(key=lambda c: (c["ops_hint"], c["cell_id"]))

    # cpu pinning: hand each worker a disjoint slice
    if args.cpus_per_job > 0:
        for i, c in enumerate(cells):
            slot = i % max(1, args.jobs)
            lo = (slot * args.cpus_per_job) % ncpu
            hi = min(ncpu - 1, lo + args.cpus_per_job - 1)
            c["taskset_cpus"] = f"{lo}-{hi}" if hi > lo else str(lo)

    print(f"[driver] {len(cells)} cells, jobs={args.jobs}, ncpu={ncpu}, "
          f"time_limit={args.time_limit}s, blas_threads={args.blas_threads}, "
          f"cpus_per_job={args.cpus_per_job or 'unpinned'}", flush=True)

    results = []
    t0 = time.time()
    done = 0
    pending = list(cells)
    inflight: dict = {}

    def _record(r: dict) -> None:
        results.append(r)
        ms = r.get("makespan_pred_ms")
        print(f"[{len(results)}/{len(cells)}] {r['cell_id']} ok={r.get('ok')} "
              f"status={r.get('solver_status')} "
              f"makespan={ms if ms is None else round(ms, 3)} "
              f"wall={round(r.get('wall_s', 0) or 0, 1)}s "
              f"{r.get('error', '')}", flush=True)

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        while pending or inflight:
            # Fill free slots, unless the submission deadline has passed.
            while (pending and len(inflight) < args.jobs
                   and not (args.deadline_s and (time.time() - t0) > args.deadline_s)):
                c = pending.pop(0)
                inflight[ex.submit(run_cell, c)] = c
            if not inflight:
                break
            done_futs, _ = wait(set(inflight), return_when=FIRST_COMPLETED)
            for fut in done_futs:
                c = inflight.pop(fut)
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001
                    r = {"cell_id": c["cell_id"], "arm": c["arm"], "solver": c["solver"],
                         "basename": c["basename"], "ok": False,
                         "family": split_basename(c["basename"])[0],
                         "pair": split_basename(c["basename"])[1],
                         "error": f"{type(e).__name__}: {e}"}
                    with open(os.path.join(celldir, c["cell_id"] + ".json"), "w") as fh:
                        json.dump(r, fh, indent=1)
                _record(r)

    # Anything still pending was cut by the wall-clock budget, not by the solver.
    for c in pending:
        r = {"cell_id": c["cell_id"], "arm": c["arm"], "solver": c["solver"],
             "basename": c["basename"], "ok": False,
             "family": split_basename(c["basename"])[0],
             "pair": split_basename(c["basename"])[1],
             "ops_hint": c.get("ops_hint"),
             "error": "not_attempted_budget"}
        with open(os.path.join(celldir, c["cell_id"] + ".json"), "w") as fh:
            json.dump(r, fh, indent=1)
        _record(r)

    outp = os.path.join(root, f"combined_{args.tag}.json")
    with open(outp, "w") as f:
        json.dump(sorted(results, key=lambda r: r["cell_id"]), f, indent=1)
    print(f"[driver] wrote {outp}; total wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
