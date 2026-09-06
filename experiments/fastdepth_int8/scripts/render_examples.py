#!/usr/bin/env python3
"""Qualitative comparison: RGB, ground truth, fp32 PyTorch, int8 ModelBlaster.

Frames are chosen at PERCENTILES of the fixed-int8 per-image RMSE, not by eye.
Picking the prettiest outputs would make a quantisation study meaningless -- the
point is to show what a typical frame looks like AND what a bad one looks like,
so the p90 row is deliberately included.

Depth panels in one row share a colour scale (that row's ground-truth range), so
the columns are directly comparable; a per-panel autoscale would hide exactly
the range errors quantisation causes.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_common as C

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
MIN_D, MAX_D = 0.5, 10.0
CMAP = "magma"

R = os.path.join(os.path.dirname(HERE), "results")
rgb, gt, names = C.load_val()
fp32 = np.load(f"{R}/pred_fp32_e0.npz")
stock = np.load(f"{R}/pred_int8_e0_c1.npz")
fixed = np.load(f"{R}/pred_int8_e0_c8pcdwca.npz")
# Assert the four sources are in the same frame order rather than assuming it:
# silently misaligning predictions with ground truth would produce a plausible
# figure of the wrong thing, which is the worst failure mode for a qualitative
# comparison.
assert list(fp32["names"]) == list(names) == list(stock["names"]) == list(fixed["names"]), \
    "prediction files are not in val order; align by name before plotting"
P32, PST, PFX = fp32["pred"], stock["pred"], fixed["pred"]

def rmse(p, g):
    m = (g > MIN_D) & (g < MAX_D)
    return float(np.sqrt(((np.clip(p[m], MIN_D, MAX_D) - g[m]) ** 2).mean()))

err = np.array([rmse(PFX[i], gt[i]) for i in range(len(gt))])
pct = [10, 30, 50, 70, 90]
sel = [int(np.argsort(err)[int(round(p / 100 * (len(err) - 1)))]) for p in pct]

cols = ["RGB input", "ground truth", "fp32 (PyTorch)",
        "int8 stock", "int8 fixed", "|int8 fixed − fp32|"]
fig, axes = plt.subplots(len(sel), len(cols), figsize=(3.05 * len(cols), 3.05 * len(sel)))
for r, i in enumerate(sel):
    im = (rgb[i] * STD + MEAN).clip(0, 1).transpose(1, 2, 0)
    g = gt[i]
    vmin, vmax = float(g[g > MIN_D].min()), float(g.max())
    panels = [(im, None), (g, (vmin, vmax)), (P32[i], (vmin, vmax)),
              (PST[i], (vmin, vmax)), (PFX[i], (vmin, vmax)),
              (np.abs(PFX[i] - P32[i]), (0, 0.5))]
    for c, (dat, lim) in enumerate(panels):
        ax = axes[r, c]
        if lim is None:
            ax.imshow(dat)
        else:
            h = ax.imshow(dat, cmap="viridis" if c == 5 else CMAP,
                          vmin=lim[0], vmax=lim[1])
            if c in (4, 5):
                plt.colorbar(h, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[c], fontsize=10)
    axes[r, 0].set_ylabel(f"p{pct[r]}  {names[i]}\nRMSE {err[i]:.2f} m", fontsize=8)
    for c, p in ((3, PST[i]), (4, PFX[i])):
        axes[r, c].set_xlabel(f"RMSE {rmse(p, g):.2f} m", fontsize=8)
    axes[r, 2].set_xlabel(f"RMSE {rmse(P32[i], g):.2f} m", fontsize=8)

fig.suptitle("FastDepth on NYU Depth V2 — fp32 PyTorch vs ModelBlaster int8 "
             "(epoch-0 checkpoint, 654-frame val split)\n"
             "rows are RMSE percentiles of the fixed-int8 model, not hand-picked",
             fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = os.path.join(os.path.dirname(HERE), "plots", "fastdepth_int8_examples.png")
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"selected frames: {[names[i] for i in sel]}")
print(f"per-image RMSE  fp32 {np.mean([rmse(P32[i],gt[i]) for i in sel]):.3f}  "
      f"stock {np.mean([rmse(PST[i],gt[i]) for i in sel]):.3f}  "
      f"fixed {np.mean([rmse(PFX[i],gt[i]) for i in sel]):.3f}  (these 5 frames)")
