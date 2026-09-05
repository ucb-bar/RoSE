#!/usr/bin/env python3
"""Train FastDepth on NYU Depth V2, for weights the int8 path can quantize.

Why this exists: models/fastdepth.py ships with weights=None, so every depth
number it has produced so far is a shape/throughput result and nothing else.
Quantization error is only meaningful against trained weights -- an int8 model
built from random init tells you about the arithmetic, not the network.

DATASET. The canonical FastDepth tarball (datasets.lids.mit.edu) is dead, as is
the sparse-to-dense one on the same host, and cs.nyu.edu 404s. The live source
is the HF mirror sayakpaul/nyu_depth_v2 -- and its tar shards turn out to hold
the ORIGINAL HDF5 files (train|val/official/NNNNN.h5, keys "rgb" uint8 3x480x640
and "depth" float32 in metres, already inpainted so there are no zero holes).
It is a repack of the dead tarball, not a re-derivation, which is why the split
is exactly 47,584 / 654.

We read those .h5 files directly instead of going through `datasets`: the repo
also ships a loading script, and datasets>=3.0 refuses scripts outright
("Dataset scripts are no longer supported"). Reading the archive members is
both simpler and immune to that policy changing again.

The model is imported from models/fastdepth.py rather than redefined, so the
checkpoint loads into the deployment path by construction. Train with
MODELBLASTER_FASTDEPTH_* set to the configuration you intend to deploy.
"""
import argparse, glob, json, os, sys, time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

MAX_DEPTH = 10.0    # NYU Depth V2 is captured to ~10 m
MIN_DEPTH = 0.5     # below this the Kinect returns noise; standard eval floor
#: The encoder is ImageNet-pretrained, so it expects ImageNet-normalised input.
#: Feeding it raw [0,1] RGB does not fail, it just quietly wastes the pretrained
#: features -- the first layers see a distribution they were never fitted on.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def check_units(paths):
    """Assert the depth really is metres before spending GPU hours on it.

    The files say float32 and the samples read 0.7-8.5 m, but a future reshuffle
    of the mirror could swap in millimetres or a uint16 encoding, and nothing
    downstream would notice -- the loss would just be large and the model would
    train to something useless. So this is a hard gate, not a guess.
    """
    mn, mx = 1e9, -1e9
    for p in paths:
        with h5py.File(p, "r") as f:
            d = np.asarray(f["depth"], dtype=np.float32)
        mn, mx = min(mn, float(d.min())), max(mx, float(d.max()))
    if not (0.0 <= mn and mx <= 12.0):
        raise SystemExit(
            f"depth range [{mn:.3f}, {mx:.3f}] is not metres for NYU (expect "
            f"~0-10). Refusing to train: a units mismatch trains silently.")
    return mn, mx


class NYU(Dataset):
    """Square centre crop then resize, so depth geometry is not stretched."""

    def __init__(self, files, size, train):
        self.files, self.size, self.train = files, size, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with h5py.File(self.files[i], "r") as f:
            rgb = np.asarray(f["rgb"], dtype=np.float32).transpose(1, 2, 0) / 255.0
            dep = np.asarray(f["depth"], dtype=np.float32)
        h, w = dep.shape[:2]
        c = min(h, w)
        y0, x0 = (h - c) // 2, (w - c) // 2
        rgb = rgb[y0:y0 + c, x0:x0 + c]
        dep = dep[y0:y0 + c, x0:x0 + c]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        dep = torch.from_numpy(dep)[None, None]
        rgb = F.interpolate(rgb, (self.size, self.size), mode="bilinear", align_corners=False)[0]
        # nearest for depth: bilinear would blend across the invalid (0) holes
        # and invent depth at object boundaries.
        dep = F.interpolate(dep, (self.size, self.size), mode="nearest")[0]
        if self.train:
            if torch.rand(1).item() < 0.5:
                rgb, dep = torch.flip(rgb, [2]), torch.flip(dep, [2])
            # brightness jitter belongs in [0,1] space, before normalisation
            rgb = (rgb * (0.8 + 0.4 * torch.rand(1).item())).clamp(0, 1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return rgb, dep


def metrics(pred, gt):
    """Standard NYU depth metrics on valid pixels only.

    Also reports RMSE split by depth band. d1 rose while RMSE worsened on the
    first attempt, which is what L1 does when it trades a minority of far
    pixels for the near majority -- the bands make that visible instead of
    leaving it as a guess.
    """
    m = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if m.sum() == 0:
        return None
    p, g = pred[m].clamp(MIN_DEPTH, MAX_DEPTH), gt[m]
    r = torch.max(p / g, g / p)
    bands = {}
    for lo, hi in ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0)):
        b = (g >= lo) & (g < hi)
        bands[f"rmse_{lo:g}_{hi:g}"] = (torch.sqrt(((p[b] - g[b]) ** 2).mean()).item()
                                        if b.any() else float("nan"))
    return dict(**bands,
        d1=(r < 1.25).float().mean().item(),
        d2=(r < 1.25 ** 2).float().mean().item(),
        d3=(r < 1.25 ** 3).float().mean().item(),
        rmse=torch.sqrt(((p - g) ** 2).mean()).item(),
        rel=((p - g).abs() / g).mean().item(),
        log10=(torch.log10(p) - torch.log10(g)).abs().mean().item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="/home/ubuntu/fastdepth_ckpt")
    ap.add_argument("--data", default="/home/ubuntu/nyu_h5")
    ap.add_argument("--limit-train", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, "/home/ubuntu/fastdepth_train")
    import models.fastdepth as fdm
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_files = sorted(glob.glob(f"{a.data}/train/**/*.h5", recursive=True))
    va_files = sorted(glob.glob(f"{a.data}/val/**/*.h5", recursive=True))
    if not tr_files or not va_files:
        raise SystemExit(f"no .h5 under {a.data}; run prepare_data first")
    print(f"  train={len(tr_files)} val={len(va_files)}", flush=True)
    mn, mx = check_units(va_files[:20])
    print(f"  depth units verified: metres, observed [{mn:.3f}, {mx:.3f}]", flush=True)

    if a.limit_train:
        tr_files = tr_files[:a.limit_train]
    dl_tr = DataLoader(NYU(tr_files, a.size, True), batch_size=a.batch, shuffle=True,
                       num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(NYU(va_files, a.size, False), batch_size=a.batch,
                       shuffle=False, num_workers=4, pin_memory=True)

    model = fdm.get_model().to(dev)

    # Start the head at the dataset's mean depth instead of at zero.
    # The head is a plain linear conv, so an untrained model predicts ~0 for a
    # target that averages ~2.9 m -- it must burn its early steps learning a
    # large constant offset before it can learn any structure, and until then it
    # emits negative depths. Seeding the bias puts the model at the
    # constant-mean baseline on step 0 (d1 ~= 0.47) so every step after that
    # buys structure. This lives in the trainer, not the model: it only sets
    # values, so the deployed graph and its op set are untouched.
    with torch.no_grad():
        samp = []
        for f in tr_files[:200]:
            with h5py.File(f, "r") as fh:
                samp.append(float(np.asarray(fh["depth"], dtype=np.float32).mean()))
        mean_depth = float(np.mean(samp))
        model.head.bias.fill_(mean_depth)
    print(f"  head bias seeded to mean depth {mean_depth:.3f} m", flush=True)

    model.train()
    n = sum(p.numel() for p in model.parameters())
    print(f"  model params {n:,}  skips={getattr(model,'skips',None)}  size={a.size}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    best, best_rmse, hist = -1.0, 1e9, []

    for ep in range(a.epochs):
        model.train(); tl = 0.0; nb = 0; t0 = time.time()
        for rgb, dep in dl_tr:
            rgb, dep = rgb.to(dev, non_blocking=True), dep.to(dev, non_blocking=True)
            valid = (dep > MIN_DEPTH) & (dep < MAX_DEPTH)
            if valid.sum() == 0:
                continue
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                out = model(rgb)
                loss = (out - dep).abs()[valid].mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
            if nb % 200 == 0:
                print(f"    ep{ep} step {nb}/{len(dl_tr)} loss {tl/nb:.4f}", flush=True)
        model.eval(); acc = []
        with torch.no_grad():
            for rgb, dep in dl_va:
                rgb, dep = rgb.to(dev), dep.to(dev)
                m = metrics(model(rgb).float(), dep)
                if m: acc.append(m)
        ev = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}
        ev.update(epoch=ep, train_loss=tl / max(nb, 1), secs=round(time.time() - t0))
        hist.append(ev)
        print(f"  epoch {ep}: loss {ev['train_loss']:.4f}  d1 {ev['d1']:.4f} "
              f"rmse {ev['rmse']:.4f} rel {ev['rel']:.4f}  "
              f"[near {ev['rmse_0.5_2']:.3f} mid {ev['rmse_2_5']:.3f} "
              f"far {ev['rmse_5_10']:.3f}]  ({ev['secs']}s)", flush=True)
        json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "epoch": ep, "metrics": ev},
                   f"{a.out}/last.pt")
        if ev["d1"] > best:
            best = ev["d1"]
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": ev},
                       f"{a.out}/best.pt")
            print(f"    new best d1 {best:.4f} -> best.pt", flush=True)
        # Kept separately so the d1 rule cannot silently discard the model with
        # the lowest absolute error, which is the one a depth CONSUMER wants.
        if ev["rmse"] < best_rmse:
            best_rmse = ev["rmse"]
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": ev},
                       f"{a.out}/best_rmse.pt")
            print(f"    new best rmse {best_rmse:.4f} -> best_rmse.pt", flush=True)
    print(f"DONE best_d1={best:.4f}")


main()
