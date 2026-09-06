#!/usr/bin/env python3
"""fp32 FastDepth baseline on NYU val, PER-IMAGE protocol.

Deliberately re-derives the baseline here rather than quoting the training
log: the int8 delta is only meaningful if both arms see the identical frames
through the identical metric code (fd_common), and the training log's number
came from a different loader on a different box.
"""
import argparse, json, sys, time
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fd_common as C

ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, default=None, help="frames (default: all)")
ap.add_argument("--tag", default="fp32_full")
ap.add_argument("--save-pred", default=None, help="npz of fp32 predictions")
a = ap.parse_args()

C.apply_trained_env()
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
from modelblaster.models import fastdepth as fdm

model = fdm.get_model()
rgb, gt, names = C.load_val(a.n)
print(f"[fp32] {len(rgb)} frames", flush=True)
rows, preds = [], []
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(rgb), 16):
        y = model(torch.from_numpy(rgb[s:s + 16])).numpy()[:, 0]
        preds.append(y.astype(np.float32))
        for i in range(y.shape[0]):
            rows.append(C.per_image(y[i], gt[s + i]))
preds = np.concatenate(preds)
ev = C.aggregate(rows)
print(C.fmt(ev, "fp32"))
print(f"   bands near {ev['rmse_0.5_2']:.3f} mid {ev['rmse_2_5']:.3f} far {ev['rmse_5_10']:.3f}")
print(f"   {time.time() - t0:.1f}s")
print(C.save(a.tag, {"arm": "fp32", "n": len(rgb), "metrics": ev}))
if a.save_pred:
    np.savez(a.save_pred, pred=preds, names=names)
