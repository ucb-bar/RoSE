#!/usr/bin/env python3
"""Append one shard_dim result to experiments/kernel_opt_log.jsonl.

Follows the existing 374-entry schema: ts / experiment / platform, then
free-form result fields and a `notes` list. Appends only -- never rewrites.
"""
import csv, json, sys, datetime, os
LOG = "/scratch/dima/rose-infra/RoSE/experiments/kernel_opt_log.jsonl"
def main():
    csv_path, job_id, backend, slot = sys.argv[1:5]
    rows = list(csv.DictReader(open(csv_path)))
    split = [r for r in rows if int(r["k"]) > 1]
    errs = sorted(abs(float(r["err"])) for r in split if r["err"])
    grid = {}
    for r in split:
        grid.setdefault(f"OC{r['OC']}", {})[f"k{r['k']}"] = {
            "tile_w": int(r["tile_w"]), "work_ratio": float(r["work_ratio"]),
            "predicted": float(r["predicted"]) if r["predicted"] else None,
            "cost_P2_us": float(r["cost_P2_us"]), "gain_P2_pct": float(r["gain_P2_pct"]),
            "cost_P4_us": float(r["cost_P4_us"]), "gain_P4_pct": float(r["gain_P4_pct"])}
    entry = {
        "ts": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "shard_dim_v1",
        "platform": ("AWS F2 f2.6xlarge, f2_quad_hetero_norose_tacit_q31_60mhz "
                     "(agfi-0662319f4b07483a7), 4 harts: 0/1 rocket+gemmini_q31, "
                     "2/3 rocket+saturn V256D128"),
        "job": f"fq {job_id}",
        "question": "impact of SHARDING DIMENSION (split degree k -> tile width w) on per-op cost",
        "backend_under_test": backend,
        "placement": f"all tiles serial on {slot} (isolates work conservation from concurrency tax)",
        "metric": ("work_ratio = sum(tile times)/unsplit time; 1.0 = perfect split, k = full "
                   "duplication. cost_P<n> = ceil(k/n)*tile_time, the makespan of packing k "
                   "tiles onto n harts; this SoC gives P=2 per same-kind pair."),
        "model": ("rvv: work_ratio(k,OC) = k*ceil((OC/k)/32)/ceil(OC/32), "
                  "V=32 = vsetvlmax_e32m4() at VLEN=256; source-verified in "
                  "kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c"),
        "grid": grid,
        "model_abs_err_x": {"mean": round(sum(errs)/len(errs), 3) if errs else None,
                            "median": round(errs[len(errs)//2], 3) if errs else None,
                            "max": round(errs[-1], 3) if errs else None,
                            "n_cells": len(errs)},
        "config": "experiments/shard_dim/configs/{study,jobs}.json",
        "raw": f"experiments/shard_dim/results/{os.path.basename(csv_path)}",
        "notes": [],
    }
    entry["notes"] = json.loads(sys.argv[5]) if len(sys.argv) > 5 else []
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"appended shard_dim_v1 entry to {LOG} (now {sum(1 for _ in open(LOG))} entries)")
main()
