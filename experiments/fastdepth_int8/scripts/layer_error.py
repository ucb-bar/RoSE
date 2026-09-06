#!/usr/bin/env python3
"""Per-layer attribution: WHERE does the int8 FastDepth lose the fp32 signal?

Runs the bn-folded fp32 graph and the int8 graph on the SAME frames and, at
every IR tensor, compares the dequantised int8 activation against the fp32 one.

Two quantities per tensor, because they answer different questions:

  sqnr_db   10*log10(||fp32||^2 / ||fp32 - deq(int8)||^2): total error carried
            at that point, i.e. how far the network has already drifted.
  clip%     fraction of fp32 values whose magnitude exceeds the tensor's
            calibrated range (max_abs from the calibration set). These are
            values the int8 tensor CANNOT represent -- the signature of a
            calibration range set too narrow.
  step_db   SQNR of a hypothetical PERFECT quantisation of the fp32 tensor at
            this tensor's own scale. It is the best SQNR this scale could give
            with zero upstream error, so sqnr_db far below step_db means the
            damage arrived from earlier layers rather than being made here.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--ir", required=True)
ap.add_argument("-n", type=int, default=8)
ap.add_argument("--tag", default="layer_error")
a = ap.parse_args()

C.apply_trained_env()
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
import int8_fast
from modelblaster.models import fastdepth as fdm
from modelblaster.pipeline import extract_graph as EG

ir = json.load(open(os.path.join(a.ir, "graph.json")))
W = dict(np.load(os.path.join(a.ir, "weights.npz")))
scales = {k: float(v["quant"]["scale"]) for k, v in ir["tensors"].items()}

rgb, gt, _ = C.load_val(a.n)
x = torch.from_numpy(rgb)

# fp32 reference activations from the SAME graph the quantiser saw: symbolic
# trace + bn folding, so tensor names line up 1:1 with the IR's.
gm = torch.fx.symbolic_trace(fdm.get_model())
EG._fold_conv_bn(gm)
cap = EG._CaptureTensors(gm)
with torch.no_grad():
    cap.run(x)
fp32 = {k: v.detach().numpy().astype(np.float64) for k, v in cap.tensors.items()}
fp32[ir["input"]["tensor"]] = rgb.astype(np.float64)

s_in = scales[ir["input"]["tensor"]]
xq = np.clip(np.round(rgb / s_in), -127, 127).astype(np.int8)
_, act = int8_fast.run_ir(ir, W, xq, keep_all=True)

order = [ir["input"]["tensor"]] + [o["outputs"][0] for o in ir["ops"]]
rows = []
for name in order:
    if name not in fp32 or name not in act:
        continue
    f = fp32[name].reshape(-1)
    s = scales[name]
    d = act[name].reshape(-1).astype(np.float64) * s
    if f.size != d.size:
        continue
    err = f - d
    sqnr = 10 * np.log10(max((f ** 2).sum(), 1e-300) / max((err ** 2).sum(), 1e-300))
    rng = 127.0 * s
    clip = float((np.abs(f) > rng).mean()) * 100.0
    ideal = f - np.clip(np.round(f / s), -127, 127) * s
    step = 10 * np.log10(max((f ** 2).sum(), 1e-300) / max((ideal ** 2).sum(), 1e-300))
    rows.append(dict(tensor=name, sqnr_db=float(sqnr), step_db=float(step),
                     clip_pct=clip, scale=s, max_abs=float(np.abs(f).max())))

kind = {o["outputs"][0]: o["op"] for o in ir["ops"]}
print(f"{'tensor':<30}{'op':<22}{'sqnr_dB':>9}{'step_dB':>9}{'clip%':>8}{'d_sqnr':>8}")
prev = None
for r in rows:
    d = "" if prev is None else f"{r['sqnr_db'] - prev:+8.2f}"
    print(f"{r['tensor']:<30}{kind.get(r['tensor'], 'input'):<22}"
          f"{r['sqnr_db']:9.2f}{r['step_db']:9.2f}{r['clip_pct']:8.3f}{d:>8}")
    prev = r["sqnr_db"]
print(C.save(a.tag, {"ir": a.ir, "n": a.n, "rows": rows}))
