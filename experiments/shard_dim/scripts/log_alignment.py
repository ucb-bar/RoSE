#!/usr/bin/env python3
"""Append the alignment-axis result to experiments/kernel_opt_log.jsonl.

  log_alignment.py <cells.csv> "<jobs>" "<json list of notes>"

Follows the existing schema (ts / experiment / platform / ... / notes).
Appends only -- never rewrites.
"""
import csv, json, sys, datetime, statistics, os
LOG = "/scratch/dima/rose-infra/RoSE/experiments/kernel_opt_log.jsonl"


def cls(r):
    a, e = r["aligned"] == "True", r["even"] == "True"
    return "aligned_even" if (a and e) else ("aligned_uneven" if a else "misaligned")


def main():
    cells = [r for r in csv.DictReader(open(sys.argv[1])) if r["work_ratio"]]
    jobs, notes = sys.argv[2], json.loads(sys.argv[3])
    by = {}
    for r in cells:
        k = (r["backend"], f"OC{r['OC']}", r["partition"])
        by.setdefault(k, []).append(float(r["work_ratio"]))
    grid = {}
    for (be, oc, part), v in sorted(by.items()):
        row = next(r for r in cells if r["backend"] == be and f"OC{r['OC']}" == oc
                   and r["partition"] == part)
        grid.setdefault(be, {}).setdefault(oc, {})[part] = {
            "class": cls(row), "n": len(v),
            "work_ratio_mean": round(statistics.fmean(v), 3),
            "work_ratio_spread": round(max(v) - min(v), 3),
            "pred_slab": float(row["pred_slab"])}
    summ = {}
    for be in ("rvv", "gemmini_q31"):
        sub = [r for r in cells if r["backend"] == be]
        if not sub:
            continue
        e = sorted(abs(float(r["err_slab"])) for r in sub)
        summ[be] = {"slab_model_abs_err": {
            "mean": round(statistics.fmean(e), 3),
            "median": round(statistics.median(e), 3),
            "max": round(e[-1], 3), "n_cells": len(e)}}
        for c in ("aligned_even", "aligned_uneven", "misaligned"):
            v = [float(r["work_ratio"]) for r in sub if cls(r) == c]
            if v:
                summ[be][c] = {"n": len(v), "work_ratio_mean": round(statistics.fmean(v), 3),
                               "min": round(min(v), 3), "max": round(max(v), 3)}
    entry = {
        "ts": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "shard_dim_alignment_v1",
        "platform": ("AWS F2 f2.6xlarge x4, f2_quad_hetero_norose_tacit_q31_60mhz "
                     "(agfi-0662319f4b07483a7), 4 harts: 0/1 rocket+gemmini_q31, "
                     "2/3 rocket+saturn V256D128"),
        "job": jobs,
        "question": ("Is split cost driven by ALIGNMENT (each tile width a whole "
                     "multiple of the backend blocking quantum) rather than by "
                     "EVENNESS (equal tiles)?"),
        "placement": "every tile serial on one hart (work conservation, no concurrency tax)",
        "metric": "work_ratio = sum(tile times)/unsplit time; 1.0 = split is free",
        "model": ("slab model: work_ratio = sum_i ceil(w_i/Q)/ceil(OC/Q); "
                  "Q=32 rvv (vsetvlmax_e32m4 at VLEN=256), Q=16 gemmini (DIM)"),
        "grid": grid,
        "summary": summ,
        "config": "experiments/shard_dim/configs/alignment_jobs.json",
        "raw": f"experiments/shard_dim/results/{os.path.basename(sys.argv[1])}",
        "notes": notes,
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"appended shard_dim_alignment_v1 entry ({sum(1 for _ in open(LOG))} lines)")


main()
