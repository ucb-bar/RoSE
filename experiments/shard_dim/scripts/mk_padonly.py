#!/usr/bin/env python3
"""The padding-materialisation CONTROL: every conv becomes a SINGLE OH tile.

The OH split's tiles run a conv with PH=PW=0 over a pre-padded window. On rvv
that keeps kernel_conv2d_s8 on its "all taps in bounds" vectorised strip path,
which the UNSPLIT call often cannot take (for OW=4, MB_CONV_TILE=4, PW=1 the
only strip starts at iw0=-1). So part of the measured OH win may be the
PRE-PADDING, not the row split -- and the two are confounded in every OH arm.

This tree separates them. Each conv is rewritten as one tile covering ALL of its
output rows: same gather, same PH=PW=0 kernel call, same scatter, but k=1, so
there is no split and no halo. Its work ratio against the unsplit baseline is
therefore exactly the pre-padding effect plus the copy cost, with the split's
contribution removed by construction:

    OH_k2_ratio  =  padding_effect  +  split_contribution
    padonly_ratio = padding_effect                          <- this tree

  mk_padonly.py <src_exdir> <quant> <dst_exdir>
"""
import copy, json, os, shutil, subprocess, sys
MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
KEEP = ("kernels.c", "kernels.h", "kernel_picks.json")
BACKENDS = ("gemmini_q31", "rvv")


def pad_only(op):
    """One OH 'tile' spanning the whole output. Same arithmetic as
    apply_split_hint._split_conv2d_s8_oh with widths == [OH]."""
    sh = op["shape"]
    IH, IW = int(sh["IH"]), int(sh["IW"])
    KH, SH, PH, PW = int(sh["KH"]), int(sh["SH"]), int(sh["PH"]), int(sh["PW"])
    OH = int(sh["OH"])
    lo = -PH
    win = (OH - 1) * SH + KH
    t = copy.deepcopy(op)
    t["shape"] = dict(sh)
    t["shape"].update({"IH": win, "PH": 0, "OH": OH, "IW": IW + 2 * PW, "PW": 0})
    t["outputs"] = [f"{op['outputs'][0]}.tile_0"]
    t["name"] = op["name"] + ".tile_0"
    t["split_from"] = {
        "op_id": op["dispatch_id"], "tile": 0, "n_splits": 1, "axis": "OH",
        "tile_oh": OH, "tile_offset_OH": 0,
        "in_row_lo": lo, "window_rows": win,
        "pad_top": max(0, -lo), "pad_bot": max(0, lo + win - IH),
        "in_rows": win - max(0, -lo) - max(0, lo + win - IH),
        "parent_IH": IH, "parent_IW": IW, "parent_OH": OH,
        "parent_PH": PH, "parent_PW": PW}
    return t


def main():
    src_ex, quant, dst_ex = sys.argv[1:4]
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
    tensors = g.setdefault("tensors", {})
    n = 0
    for i, op in enumerate(g["ops"]):
        if op.get("op") != "conv2d_s8":
            continue
        parent_out = op["outputs"][0]
        g["ops"][i] = pad_only(op)
        p = tensors.get(parent_out)
        if p and "shape" in p:
            e = {"shape": list(p["shape"]), "dtype": p.get("dtype", "i8"),
                 "split_from": parent_out, "tile": 0, "n_splits": 1,
                 "elem_offset": 0, "alias_kind": "row_window"}
            if p.get("quant") is not None:
                e["quant"] = copy.deepcopy(p["quant"])
            tensors[f"{parent_out}.tile_0"] = e
        n += 1
    json.dump(g, open(f"{DST}/graph.json", "w"), indent=2)
    ndisp = sum(1 for o in g["ops"] if o.get("dispatch_id") is not None)
    print(f"[mk_padonly] {dst_ex}: {n} convs pre-padded (k=1), {ndisp} dispatches "
          f"(UNCHANGED from unsplit -- no split, so the dispatch count is identical)")
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
            print(r.stdout[-2500:], r.stderr[-2500:]); raise SystemExit(f"skeleton {be} failed")
        for f, d in keep.items():
            open(f"{gen}/{f}", "wb").write(d)
        print(f"[mk_padonly] {be}: skeleton regenerated, kernels preserved")


main()
