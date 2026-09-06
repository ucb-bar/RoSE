#!/usr/bin/env python3
"""Prove the fast int8 simulator is BIT-EXACT with the shipped reference.

Two independent checks, because each is weak alone:

  A) SUBSTITUTION. Extract the same trained FastDepth twice -- once with the
     stock five-deep-loop `_sim_*` primitives, once with int8_fast.patch()
     applied -- and require every artifact to be byte-identical. Run at a
     reduced input resolution (default 64) purely so the stock simulator
     finishes: the OP KINDS, the weights, the calibration path and every
     requantize are identical to the 224 graph, only the spatial extents
     differ, so this exercises exactly the code being replaced.

  B) INTERPRETER. Run the standalone batched `run_ir` over the FULL 224 graph
     on the exact input baked into io.npz, and require its output to equal
     io.npz's golden byte for byte. That closes the gap A leaves: A validates
     the patched primitives inside extract_int8, B validates the separate
     batch interpreter the accuracy harness actually uses, at full size.
"""
import argparse, json, os, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--full-ir", default=os.path.join(C.ROOT, "ir", "base_c1"))
ap.add_argument("--work", default="/tmp/fd_verify")
ap.add_argument("--skip-a", action="store_true")
ap.add_argument("--per-channel", action="store_true")
ap.add_argument("--per-channel-depthwise", action="store_true")
a = ap.parse_args()

C.apply_trained_env()
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
import int8_fast

ok = True

if not a.skip_a:
    # A) same extraction, stock vs patched primitives.
    os.environ["MODELBLASTER_FASTDEPTH_INPUT"] = str(a.size)
    os.environ.pop("MODELBLASTER_FASTDEPTH_CALIB")     # 224-only bank
    from modelblaster.models import fastdepth as fdm
    from modelblaster.pipeline.extract_graph import extract
    shutil.rmtree(a.work, ignore_errors=True)
    for arm in ("stock", "fast"):
        if arm == "fast":
            int8_fast.patch()
        m, s = fdm.get_model(), fdm.get_sample_input()
        t0 = time.time()
        extract(m, s, name="fastdepth", out_dir=f"{a.work}/{arm}", quant="int8",
                per_channel=a.per_channel,
                per_channel_depthwise=a.per_channel_depthwise)
        print(f"[A] {arm:<5} extract {time.time() - t0:8.1f}s", flush=True)
    for f in ("graph.json", "weights.npz", "io.npz"):
        p, q = f"{a.work}/stock/{f}", f"{a.work}/fast/{f}"
        if f.endswith(".json"):
            same = json.load(open(p)) == json.load(open(q))
        else:
            zp, zq = np.load(p), np.load(q)
            same = (sorted(zp.files) == sorted(zq.files)
                    and all(np.array_equal(zp[k], zq[k]) for k in zp.files))
        print(f"[A] {f:<12} identical={same}")
        ok &= same

# B) batch interpreter vs the shipped 224 golden.
ir = json.load(open(os.path.join(a.full_ir, "graph.json")))
W = dict(np.load(os.path.join(a.full_ir, "weights.npz")))
io = np.load(os.path.join(a.full_ir, "io.npz"))
x = io["input"].reshape(1, 3, 224, 224).astype(np.int8)
t0 = time.time()
y = int8_fast.run_ir(ir, W, x).reshape(-1)
g = io["output"].reshape(-1)
same = np.array_equal(y, g)
print(f"[B] run_ir vs io.npz golden: identical={same}  "
      f"(max|diff|={int(np.abs(y.astype(int) - g.astype(int)).max())}, "
      f"{time.time() - t0:.1f}s for 1 frame)")
ok &= same
print("VERIFY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
