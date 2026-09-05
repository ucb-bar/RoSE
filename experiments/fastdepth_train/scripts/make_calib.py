#!/usr/bin/env python3
"""Build the int8 calibration bank from real NYU val frames.

Must apply EXACTLY the preprocessing training used -- square centre crop,
resize, then ImageNet normalisation. A calibration set preprocessed differently
from the training input produces activation ranges the deployed model never
sees, and nothing downstream would flag it: the int8 graph still verifies
bit-exact against its own golden.
"""
import argparse, glob, h5py, numpy as np, torch, torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/ubuntu/nyu_h5/val")
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--out", default="/home/ubuntu/fastdepth_calib.npz")
a = ap.parse_args()

files = sorted(glob.glob(f"{a.data}/**/*.h5", recursive=True))
# spread across the split rather than taking the first N, which would be one
# contiguous run of frames from a single scene
idx = np.linspace(0, len(files) - 1, a.n).astype(int)
out = []
for i in idx:
    with h5py.File(files[i], "r") as f:
        rgb = np.asarray(f["rgb"], dtype=np.float32).transpose(1, 2, 0) / 255.0
    h, w = rgb.shape[:2]
    c = min(h, w)
    y0, x0 = (h - c) // 2, (w - c) // 2
    t = torch.from_numpy(rgb[y0:y0 + c, x0:x0 + c]).permute(2, 0, 1)[None]
    t = F.interpolate(t, (a.size, a.size), mode="bilinear", align_corners=False)[0]
    out.append(((t - MEAN) / STD).numpy())
arr = np.stack(out).astype(np.float32)
np.savez(a.out, samples=arr)
print(f"wrote {a.out}  shape={arr.shape}  range=[{arr.min():.3f},{arr.max():.3f}] "
      f"mean={arr.mean():.4f} std={arr.std():.4f}  (from {len(files)} val frames)")
