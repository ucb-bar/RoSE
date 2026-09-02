#!/usr/bin/env python3
"""Build a split tree carrying a DIFFERENT split degree k per operator.

That is the budget trick behind this study: one binary yields a whole
(axis_extent x k) grid, because every operator can be tiled independently.

  mk_sweep.py <model> <src_exdir> <quant> <dst_exdir> <did>:<k> [<did>:<k> ...]
"""
import json, os, shutil, subprocess, sys
MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
sys.path.insert(0, os.path.dirname(MB))
from modelblaster.pipeline.apply_split_hint import apply_split_hint

def main():
    model, src_ex, quant, dst_ex = sys.argv[1:5]
    plan = {}
    for tok in sys.argv[5:]:
        d, k = tok.split(":"); plan[int(d)] = int(k)
    SRC = f"{MB}/examples/{src_ex}/{quant}/generated"
    DST = f"{MB}/examples/{dst_ex}/{quant}/generated"
    os.makedirs(DST, exist_ok=True)
    for f in ("weights.npz", "io.npz"):
        shutil.copy2(f"{SRC}/{f}", f"{DST}/{f}")
    for be in ("gemmini_q31", "rvv"):
        os.makedirs(f"{DST}/{be}", exist_ok=True)
        for f in ("kernels.c", "kernels.h", "kernel_picks.json"):
            shutil.copy2(f"{SRC}/{be}/{f}", f"{DST}/{be}/{f}")
    g = json.load(open(f"{SRC}/graph.json"))
    out = apply_split_hint(g, [{"op": d, "n_splits": k} for d, k in sorted(plan.items())])
    json.dump(out, open(f"{DST}/graph.json", "w"), indent=2)
    n = sum(1 for o in out["ops"] if o.get("dispatch_id") is not None)
    print(f"[mk_sweep] {model}: plan={plan} -> {n} dispatches")
    for be in ("gemmini_q31", "rvv"):
        gen = f"{DST}/{be}"
        keep = {f: open(f"{gen}/{f}", "rb").read()
                for f in ("kernels.c", "kernels.h", "kernel_picks.json")}
        r = subprocess.run(
            [sys.executable, "-m", "modelblaster.pipeline.generate_skeleton",
             "--ir", f"{DST}/graph.json", "--weights", f"{DST}/weights.npz",
             "--io", f"{DST}/io.npz", "--out-dir", gen, "--backend", be],
            cwd=MB, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": os.path.dirname(MB)})
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"skeleton {be} failed")
        for f, d in keep.items(): open(f"{gen}/{f}", "wb").write(d)
        print(f"[mk_sweep] {be}: skeleton regenerated, kernels preserved")
main()
