"""Shared plumbing for the FastDepth int8 accuracy study.

One module so the fp32 arm and the int8 arm cannot drift apart: the frames,
the valid-pixel mask, and the metric definitions are literally the same code
for both. The metric definitions mirror
experiments/fastdepth_train/scripts/eval_fastdepth.py exactly (PER-IMAGE, then
averaged -- pooling a batch into one RMSE reports a systematically higher
number by Jensen and is not the published protocol).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # experiments/fastdepth_int8
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

MIN_DEPTH, MAX_DEPTH = 0.5, 10.0

#: The exact env the checkpoint was trained under. Set as a block: a single
#: mismatched knob changes the architecture and load_state_dict(strict=True)
#: either fails or -- worse for a width knob -- silently builds a different
#: net that still loads.
TRAINED_ENV = {
    "MODELBLASTER_FASTDEPTH_WIDTH_MULT": "1.0",
    "MODELBLASTER_FASTDEPTH_DECODER_CH": "512",
    "MODELBLASTER_FASTDEPTH_INPUT": "224",
    "MODELBLASTER_FASTDEPTH_SKIPS": "1",
    "MODELBLASTER_FASTDEPTH_PRETRAINED": "1",
    "MODELBLASTER_FASTDEPTH_CALIB": os.path.join(DATA, "calib32.npz"),
}

#: Which trained checkpoint to study. epoch0 is the DEFAULT and the primary
#: export target: it is the better model in metres (rmse 0.63 vs 1.10) and its
#: predictions occupy ~[1.2, 5.8] m instead of saturating the 10 m ceiling, so
#: the same 255 int8 codes cover about half the range -- i.e. half the
#: quantisation step. epoch12 wins on d1 only, which is a ratio threshold that
#: does not penalise the over-dispersion. Override with FD_CKPT=epoch12.
CKPTS = {"epoch0": os.path.join(DATA, "ckpt_epoch0.pt"),
         "epoch12": os.path.join(DATA, "ckpt_epoch12.pt")}


def ckpt_name() -> str:
    return os.environ.get("FD_CKPT", "epoch0")


def apply_trained_env(calib: "str | None" = None) -> None:
    for k, v in TRAINED_ENV.items():
        os.environ[k] = v
    name = ckpt_name()
    if name not in CKPTS:
        raise SystemExit(f"FD_CKPT={name!r} not in {sorted(CKPTS)}")
    os.environ["MODELBLASTER_FASTDEPTH_CKPT"] = CKPTS[name]
    if calib is not None:
        os.environ["MODELBLASTER_FASTDEPTH_CALIB"] = calib


def load_val(n: "int | None" = None):
    """NYU val frames, preprocessed on the GPU box with the SAME crop/resize/
    normalise as eval_fastdepth.py. rgb is float16-stored (relative eps 1e-3,
    ~5000x finer than the int8 input step we are studying) and returned as
    float32; depth is metres at native float32."""
    rgb, dep, names = [], [], []
    for shard in sorted(glob.glob(os.path.join(DATA, "nyu_val_*.npz"))):
        z = np.load(shard)
        rgb.append(z["rgb"]); dep.append(z["depth"]); names.append(z["names"])
        if n is not None and sum(r.shape[0] for r in rgb) >= n:
            break
    rgb = np.concatenate(rgb).astype(np.float32)
    dep = np.concatenate(dep).astype(np.float32)
    names = np.concatenate(names)
    if n is not None:
        rgb, dep, names = rgb[:n], dep[:n], names[:n]
    return rgb, dep, names


def per_image(pred: np.ndarray, gt: np.ndarray) -> "dict | None":
    """Metrics for ONE image. pred/gt are 2-D metres. Identical formulae to
    eval_fastdepth.py's per_image (predictions clamped into the eval band, as
    the published protocol does)."""
    m = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if not m.any():
        return None
    p = np.clip(pred[m], MIN_DEPTH, MAX_DEPTH).astype(np.float64)
    g = gt[m].astype(np.float64)
    r = np.maximum(p / g, g / p)
    out = dict(d1=float((r < 1.25).mean()),
               d2=float((r < 1.25 ** 2).mean()),
               d3=float((r < 1.25 ** 3).mean()),
               rmse=float(np.sqrt(((p - g) ** 2).mean())),
               rel=float((np.abs(p - g) / g).mean()),
               log10=float(np.abs(np.log10(p) - np.log10(g)).mean()))
    for lo, hi in ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0)):
        b = (g >= lo) & (g < hi)
        out[f"rmse_{lo:g}_{hi:g}"] = (
            float(np.sqrt(((p[b] - g[b]) ** 2).mean())) if b.any() else float("nan"))
    return out


def aggregate(rows: list) -> dict:
    rows = [r for r in rows if r is not None]
    with np.errstate(invalid="ignore"):
        return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def fmt(ev: dict, label: str = "") -> str:
    return (f"{label:<22} d1 {ev['d1']:.4f}  d2 {ev['d2']:.4f}  d3 {ev['d3']:.4f}  "
            f"rmse {ev['rmse']:.4f}  rel {ev['rel']:.4f}  log10 {ev['log10']:.4f}")


def save(tag: str, payload: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, f"{tag}.json")
    payload = dict(payload, ckpt=ckpt_name())
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    return p
