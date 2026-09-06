#!/usr/bin/env python3
"""Backfill the MILP cost/gap fields into the per-cell JSONs from run.log.

cvxpy's verbose banner and MOSEK's own log are written to STDERR, so the first
version of the driver -- which matched them against stdout only -- left
build_s / solve_s / the MIP gap empty.  Everything needed is still in each
cell's run.log, so this reads it back rather than re-running any solve.
"""
import glob
import json
import os
import re
import sys

RX = {
    "build_s": re.compile(r"Compilation took ([0-9.eE+-]+) seconds"),
    "solve_s": re.compile(r"Solver \(including time spent in interface\) took "
                          r"([0-9.eE+-]+) seconds"),
    "mosek_time_s": re.compile(r"Optimizer terminated\. Time:\s*([0-9.eE+-]+)"),
    "mip_best_obj": re.compile(r"Objective of best integer solution\s*:\s*([0-9.eE+-]+)"),
    "mip_best_bound": re.compile(r"Best objective bound\s*:\s*([0-9.eE+-]+)"),
}
MSK_STATUS = re.compile(r"Problem status\s*:\s*(\S+)")
# cvxpy prints this before it starts reducing, so it is available even for the
# cells that died during compilation -- which is exactly where the MILP's limit
# is, so it is the number that locates the wall.
MODEL_RE = re.compile(r"Your problem has (\d+) variables, (\d+) constraints")
# last reduction cvxpy announced before the process died
REDUCT_RE = re.compile(r"Applying reduction (\w+)")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit("usage: backfill_milp_metrics.py <cells dir> <work dir>")
    cells_dir, work_dir = sys.argv[1], sys.argv[2]
    n = 0
    for cf in sorted(glob.glob(os.path.join(cells_dir, "*.json"))):
        with open(cf) as fh:
            cell = json.load(fh)
        log = os.path.join(work_dir, cell["cell_id"], "run.log")
        txt = ""
        if os.path.exists(log):
            with open(log, errors="replace") as fh:
                txt = fh.read()
        changed = False
        for key, rx in (RX.items() if cell.get("solver") == "milp" else ()):
            if cell.get(key) is None:
                m = rx.search(txt)
                if m:
                    try:
                        cell[key] = float(m.group(1))
                        changed = True
                    except ValueError:
                        pass
        m = MODEL_RE.search(txt)
        if m and cell.get("milp_n_variables") is None:
            cell["milp_n_variables"] = int(m.group(1))
            cell["milp_n_constraints"] = int(m.group(2))
            changed = True
        red = REDUCT_RE.findall(txt)
        if red and not cell.get("last_cvxpy_reduction"):
            cell["last_cvxpy_reduction"] = red[-1]
            changed = True
        m = MSK_STATUS.search(txt)
        if m and not cell.get("mosek_problem_status"):
            cell["mosek_problem_status"] = m.group(1)
            changed = True
        # relative MIP gap, from the two objective numbers MOSEK reports
        o, b = cell.get("mip_best_obj"), cell.get("mip_best_bound")
        if cell.get("mip_rel_gap") is None and o and b:
            cell["mip_rel_gap"] = abs(o - b) / max(abs(o), 1e-12)
            changed = True
        # classify the failure so the table says WHY, not just "not ok"
        # re-run over an already-classified set: "other" is the catch-all, so
        # let a later, better rule replace it
        if not cell.get("ok") and cell.get("failure_reason") in (None, "other"):
            tail = cell.get("tail", "") + txt[-4000:]
            rc = cell.get("returncode")
            st = (cell.get("solver_status") or "").lower()
            if "infeasible" in st and "optimal" not in st:
                # MOSEK proved no schedule exists inside the declared periodic
                # windows.  That is a statement about the WORKLOAD, not a solver
                # failure -- and it is one the three heuristics never make: they
                # return a schedule that silently misses the windows instead.
                cell["failure_reason"] = "milp_proved_infeasible"
            elif "Infeasible precedence" in tail:
                cell["failure_reason"] = "infeasible_precedence_windows"
            elif "err_space" in tail or "large memory allocation has failed" in tail:
                # MOSEK ran out of address space inside task.optimize(): the
                # model BUILT, the branch-and-bound did not fit.
                cell["failure_reason"] = "out_of_memory_mosek_solve"
            elif "MemoryError" in tail or "Unable to allocate" in tail:
                cell["failure_reason"] = "out_of_memory_model_build"
            elif cell.get("timed_out"):
                cell["failure_reason"] = "wall_clock_timeout"
            elif rc in (-11, -6, -9, 139, 134, 137):
                # RLIMIT_AS denies an allocation inside MOSEK/numpy's C code,
                # which aborts rather than raising MemoryError into Python.
                # Same cause as the clean MemoryError above: the model did not
                # fit the per-cell address-space budget.
                cell["failure_reason"] = "out_of_memory_model_build"
            elif "profile_loader: required profile data is missing" in tail:
                cell["failure_reason"] = "missing_profile_data"
            elif cell.get("error") == "not_attempted_budget":
                cell["failure_reason"] = "not_attempted_budget"
            else:
                cell["failure_reason"] = "other"
            changed = True
        if changed:
            with open(cf, "w") as fh:
                json.dump(cell, fh, indent=1)
            n += 1
    print(f"backfilled {n} cell JSONs")


if __name__ == "__main__":
    main()
