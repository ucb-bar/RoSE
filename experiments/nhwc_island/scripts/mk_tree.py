#!/usr/bin/env python3
"""Stage an NHWC-island example tree (or its NCHW baseline) from dronet.

  mk_tree.py <dst_exdir> {nchw|nhwc} [--align64]

Same shape as experiments/shard_dim/scripts/mk_split.py: build the graph, then
regenerate kernels.c AND the skeleton for every backend. Unlike mk_split.py the
kernels are NOT copied from a source tree -- an NHWC graph needs different
kernels (the nhwc conv/maxpool/batchnorm entry points plus the two relayouts),
so both arms regenerate them through the same code path. That is what makes the
A/B honest: the only input that differs between the two trees is the graph.

MODELBLASTER_CURATED_VERIFY=0 is deliberate. The per-curated-kernel verify is a
whole-model spike build+run per kernel (7 ops x ~30s); the gate that actually
matters here is the whole-model spike run over the FINISHED tree, which
scripts/spike_verify.sh does once, against the PyTorch goldens in io.npz. Also,
with verify ON, MB_DRIFT_ATOL would have to be set to 2 for gemmini_tiled_conv
to be accepted at all -- and setting it would silently relax the bit-exactness
requirement on batchnorm and maxpool, which is the opposite of what this
experiment needs.
"""
import argparse, json, os, shutil, subprocess, sys

R = "/scratch/dima/rose-infra/RoSE"
MB = f"{R}/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
ZCS = os.path.dirname(MB)
SRC_EX = "dronet"
QUANT = "int8"
BACKENDS = ("gemmini_q31", "rvv")
HINT = f"{R}/experiments/nhwc_island/cfg/layout_hint_dronet.json"


def run(cmd, env=None, cwd=MB):
    e = {**os.environ, "PYTHONPATH": ZCS}
    e.update(env or {})
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-6000:]); print(r.stderr[-6000:], file=sys.stderr)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dst"); ap.add_argument("layout", choices=("nchw", "nhwc"))
    ap.add_argument("--align64", action="store_true")
    a = ap.parse_args()

    SRC = f"{MB}/examples/{SRC_EX}/{QUANT}/generated"
    DST = f"{MB}/examples/{a.dst}/{QUANT}/generated"
    os.makedirs(DST, exist_ok=True)
    for f in ("weights.npz", "io.npz"):
        shutil.copy2(f"{SRC}/{f}", f"{DST}/{f}")

    if a.layout == "nhwc":
        out = run([sys.executable, "pipeline/assign_layouts.py",
                   f"{SRC}/graph.json", f"{DST}/graph.json",
                   "--hint", HINT, "--model", "dronet", "--report"])
        print(out.strip())
    else:
        shutil.copy2(f"{SRC}/graph.json", f"{DST}/graph.json")

    env = {"MODELBLASTER_CURATED_VERIFY": "0"}
    if a.align64:
        env["MB_BUF_ALIGN64"] = "1"
    if a.layout == "nhwc":
        env["MB_REQUIRE_NHWC"] = "1"

    for be in BACKENDS:
        gen = f"{DST}/{be}"
        os.makedirs(gen, exist_ok=True)
        run([sys.executable, "-m", "modelblaster.pipeline.generate_kernels",
             "--ir", f"{DST}/graph.json", "--out-dir", gen,
             "--backend", "reference", "--target", be, "--quant", QUANT,
             "--global-curated-dir", f"{MB}/kernels",
             # Required by generate_kernels whenever the target's verify_method
             # is the spike harness, even with MODELBLASTER_CURATED_VERIFY=0.
             "--repo-root", MB, "--harness-dir", f"{MB}/harness",
             "--io", f"{DST}/io.npz",
             "--build-dir", f"{MB}/examples/{a.dst}/{QUANT}/build/verify_{be}"],
            env=env)
        out = run([sys.executable, "-m", "modelblaster.pipeline.generate_skeleton",
                   "--ir", f"{DST}/graph.json", "--weights", f"{DST}/weights.npz",
                   "--io", f"{DST}/io.npz", "--out-dir", gen, "--backend", be],
                  env=env)
        picks = json.load(open(f"{gen}/kernel_picks.json"))["picks"]
        print(f"[mk_tree] {a.dst}/{be}: "
              + " ".join(f"{k}={v.get('algorithm') or v['source']}"
                         for k, v in sorted(picks.items())))
        for line in out.splitlines():
            if line.startswith("[act_layout]"):
                print("           " + line)
    print(f"[mk_tree] wrote {DST}")


main()
