#!/usr/bin/env python3
"""Build a split tree carrying an OC and/or OH partition per op.

Generalises mk_align.py (which only knew OC) to the spatial axis:

  mk_split.py <src_exdir> <quant> <dst_exdir> <did>:<axis>:<w1>,<w2>[,...] ...

    axis = OC  -> output-channel tiles (the existing splitter)
    axis = OH  -> output-ROW tiles (spatial); widths are output row counts

Kernels are COPIED from the source tree, never regenerated: generate_skeleton
does not emit kernels.c, and a regenerated kernel_picks.json would silently
change the conv algorithm (see MB_DRIFT_ATOL in the study notes).
"""
import json, os, shutil, subprocess, sys
MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
sys.path.insert(0, os.path.dirname(MB))
from modelblaster.pipeline.apply_split_hint import apply_split_hint

KEEP = ("kernels.c", "kernels.h", "kernel_picks.json")
BACKENDS = ("gemmini_q31", "rvv")


def axis_total(op, axis):
    sh = op.get("shape") or {}
    if axis == "OC":
        return int(sh["OC"])
    if axis == "N":
        # linear / linear_s8 split along output features. Written for the dronet
        # conv study, this function knew only OC and OH, so an N request fell
        # through to the OH branch and died on a missing IH.
        return int(sh["N"])
    OH = sh.get("OH")
    if OH is None:
        OH = (int(sh["IH"]) + 2 * int(sh["PH"]) - int(sh["KH"])) // int(sh["SH"]) + 1
    return int(OH)


def main():
    src_ex, quant, dst_ex = sys.argv[1:4]
    plan = {}
    for tok in sys.argv[4:]:
        d, axis, ws = tok.split(":")
        plan[int(d)] = (axis, [int(x) for x in ws.split(",")])
    SRC = f"{MB}/examples/{src_ex}/{quant}/generated"
    DST = f"{MB}/examples/{dst_ex}/{quant}/generated"
    os.makedirs(DST, exist_ok=True)
    for f in ("weights.npz", "io.npz"):
        shutil.copy2(f"{SRC}/{f}", f"{DST}/{f}")
    for be in BACKENDS:
        os.makedirs(f"{DST}/{be}", exist_ok=True)
        for f in KEEP:
            shutil.copy2(f"{SRC}/{be}/{f}", f"{DST}/{be}/{f}")

    g = json.load(open(f"{SRC}/graph.json"))
    by = {o["dispatch_id"]: o for o in g["ops"] if o.get("dispatch_id") is not None}
    hints = []
    for d, (axis, ws) in sorted(plan.items()):
        tot = axis_total(by[d], axis)
        if sum(ws) != tot:
            raise SystemExit(f"did {d} ({by[d]['name']}): {axis} partition {ws} "
                             f"sums to {sum(ws)} but {axis}={tot}")
        hints.append({"op": d, "n_splits": len(ws), "tile_sizes": ws, "axis": axis})
    out = apply_split_hint(g, hints)
    json.dump(out, open(f"{DST}/graph.json", "w"), indent=2)
    n = sum(1 for o in out["ops"] if o.get("dispatch_id") is not None)
    print(f"[mk_split] {dst_ex}: {len(hints)} ops split -> {n} dispatches")
    for d, (axis, ws) in sorted(plan.items()):
        sh = by[d]["shape"]
        tot = axis_total(by[d], axis)
        if axis == "OC":
            rv = sum(-(-w // 32) for w in ws); rv0 = -(-tot // 32)
            gm = sum(-(-w // 16) for w in ws); gm0 = -(-tot // 16)
            extra = (f"rvv_slabs={rv}/{rv0} gem_blocks={gm}/{gm0}")
        elif axis == "N":
            # linear output-feature split: no quantum, no halo, nothing to
            # report but the partition itself.
            extra = f"even={'yes' if len(set(ws)) == 1 else 'no'}"
        else:
            KH, SH, IH = int(sh["KH"]), int(sh["SH"]), int(sh["IH"])
            wins = [(w - 1) * SH + KH for w in ws]
            extra = (f"windows={wins} in_rows={sum(wins)} vs IH+2P="
                     f"{IH + 2 * int(sh['PH'])} halo_dup="
                     f"{sum(wins) / (IH + 2 * int(sh['PH'])):.4f}")
        print(f"    did={d:<3} {by[d]['name']:<16} {axis}={tot:<4} part={ws}  {extra}")

    for be in BACKENDS:
        gen = f"{DST}/{be}"
        keep = {f: open(f"{gen}/{f}", "rb").read() for f in KEEP}
        r = subprocess.run(
            [sys.executable, "-m", "modelblaster.pipeline.generate_skeleton",
             "--ir", f"{DST}/graph.json", "--weights", f"{DST}/weights.npz",
             "--io", f"{DST}/io.npz", "--out-dir", gen, "--backend", be],
            cwd=MB, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": os.path.dirname(MB)})
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"skeleton {be} failed")
        for f, d in keep.items():
            open(f"{gen}/{f}", "wb").write(d)
        print(f"[mk_split] {be}: skeleton regenerated, kernels preserved")


main()
