#!/usr/bin/env python
"""Flax octo-small-1.5 checkpoint -> PyTorch state_dict for models/octo_small.py.

Runs in the shared `zephyr` env (torch, no jax): it consumes `ref/params.npz`,
which `dump_jax_ref.py` produced on the JAX side. Nothing is installed anywhere
and the two envs never have to coexist.

    PYTHONPATH=$ZCS python convert_weights.py [--out octo_small_torch.pt]

The three layout facts that make or break this, all VERIFIED rather than
assumed (see validate.py):

  Dense    flax kernel is (in, out); torch Linear.weight is (out, in)  -> .T
  Conv     flax kernel is (kH, kW, Cin, Cout); torch is (Cout, Cin, kH, kW)
  MHA      flax q/k/v kernels are (features, heads, head_dim) and out is
           (heads, head_dim, features); the head axes flatten to h*head_dim
           in C order, which is exactly torch's (heads, head_dim) view of a
           flat (features,) axis -> reshape(384, 384) then .T

and the two transforms that are folded in at conversion time:

  StdConv  weight standardisation `(w - mean) / (std + 1e-5)` over the
           (kH, kW, Cin) axes, i.e. per output channel. Upstream applies this
           on every parameter READ, so for a frozen checkpoint it is part of
           the weights. Skipping it produces a model that runs, looks
           reasonable, and is wrong by a per-channel affine.
The constant -1.0 goal half of the stem input is NOT folded into the first
conv's bias: those convs are zero-padded, so the constant's contribution is
spatially varying in the pad ring rather than a per-channel offset. See the
note in models/octo_small.py::ImageTokenizer.
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib
from pathlib import Path

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
# .../RoSE/experiments/octo_port -> .../RoSE -> the zephyr-chipyard-sw tree that
# holds the `modelblaster` package. Honour PYTHONPATH=$ZCS if already set.
_ZCS = HERE.parents[1] / "soc/sw/xpu-rt/zephyr-chipyard-sw"
if str(_ZCS) not in sys.path:
    sys.path.insert(0, str(_ZCS))

from modelblaster.models.octo_small import (  # noqa: E402
    STD_CONV_EPS, STEM_FEATURES, _cfg,
)

TF = "octo_transformer"
BT = f"{TF}/BlockTransformer_0/Transformer_0"
DM = "heads_action/diffusion_model"


def weight_standardize(w: np.ndarray) -> np.ndarray:
    """Upstream `vit_encoders.weight_standardize(w, axis=[0,1,2], eps=1e-5)`.

    `w` is HWIO, so the reduction is over kernel-h, kernel-w and input-channel:
    one mean and one std per OUTPUT channel. `np.std` is population std
    (ddof=0), matching `jnp.std`.
    """
    w = w.astype(np.float64)
    w = w - w.mean(axis=(0, 1, 2), keepdims=True)
    w = w / (w.std(axis=(0, 1, 2), keepdims=True) + STD_CONV_EPS)
    return w.astype(np.float32)


def dense(p: dict, prefix: str) -> dict:
    """flax Dense -> torch Linear."""
    out = {"weight": torch.from_numpy(p[f"{prefix}/kernel"].T.copy())}
    bk = f"{prefix}/bias"
    if bk in p:
        out["bias"] = torch.from_numpy(p[bk].copy())
    return out


def conv(p: dict, prefix: str, standardize: bool) -> dict:
    """flax Conv (HWIO) -> torch Conv2d (OIHW), optionally weight-standardised."""
    k = p[f"{prefix}/kernel"]
    if standardize:
        k = weight_standardize(k)
    w = np.transpose(k, (3, 2, 0, 1)).copy()
    out = {"weight": torch.from_numpy(w)}
    bk = f"{prefix}/bias"
    if bk in p:
        out["bias"] = torch.from_numpy(p[bk].copy())
    return out


def norm(p: dict, prefix: str) -> dict:
    """flax LayerNorm/GroupNorm store `scale`, torch stores `weight`."""
    return {"weight": torch.from_numpy(p[f"{prefix}/scale"].copy()),
            "bias": torch.from_numpy(p[f"{prefix}/bias"].copy())}


def attention(p: dict, prefix: str) -> dict:
    """flax MultiHeadDotProductAttention -> four torch Linears."""
    sd = {}
    for name in ("query", "key", "value"):
        k = p[f"{prefix}/{name}/kernel"]           # (features, heads, head_dim)
        f = k.shape[0]
        sd[f"{name}.weight"] = torch.from_numpy(k.reshape(f, -1).T.copy())
        b = p[f"{prefix}/{name}/bias"]             # (heads, head_dim)
        sd[f"{name}.bias"] = torch.from_numpy(b.reshape(-1).copy())
    k = p[f"{prefix}/out/kernel"]                  # (heads, head_dim, features)
    f = k.shape[-1]
    sd["out.weight"] = torch.from_numpy(k.reshape(-1, f).T.copy())
    sd["out.bias"] = torch.from_numpy(p[f"{prefix}/out/bias"].copy())
    return sd


def stem(p: dict, flax_prefix: str) -> dict:
    """SmallStem16: 4 x (StdConv + GroupNorm) then the 1x1 `embedding` conv."""
    sd = {}
    for i in range(len(STEM_FEATURES)):
        c = conv(p, f"{flax_prefix}/StdConv_{i}", standardize=True)
        for k, v in c.items():
            sd[f"convs.{i}.{k}"] = v
        for k, v in norm(p, f"{flax_prefix}/GroupNorm_{i}").items():
            sd[f"norms.{i}.{k}"] = v
    for k, v in conv(p, f"{flax_prefix}/embedding", standardize=False).items():
        sd[f"embedding.{k}"] = v
    return sd


def convert(params: dict, cfg: dict) -> dict:
    p = params
    sd: dict[str, torch.Tensor] = {}

    def put(prefix: str, d: dict):
        for k, v in d.items():
            sd[f"{prefix}{k}"] = v

    # ---- observation tokenizers ----------------------------------------
    put("backbone.tok_primary.stem.",
        stem(p, f"{TF}/observation_tokenizers_primary/SmallStem16_0"))
    put("backbone.proj_primary.", dense(p, f"{TF}/obs_primary_projection"))
    sd["backbone.pos_primary"] = torch.from_numpy(
        p[f"{TF}/obs_primary_pos_embedding"].copy())

    if cfg["wrist"]:
        put("backbone.tok_wrist.stem.",
            stem(p, f"{TF}/observation_tokenizers_wrist/SmallStem16_0"))
        put("backbone.proj_wrist.", dense(p, f"{TF}/obs_wrist_projection"))
        sd["backbone.pos_wrist"] = torch.from_numpy(
            p[f"{TF}/obs_wrist_pos_embedding"].copy())

    # ---- language + readout --------------------------------------------
    put("backbone.proj_language.", dense(p, f"{TF}/task_language_projection"))
    sd["backbone.pos_language"] = torch.from_numpy(
        p[f"{TF}/task_language_pos_embedding"].copy())
    sd["backbone.pos_readout"] = torch.from_numpy(
        p[f"{TF}/readout_action_pos_embedding"].copy())

    # ---- transformer ----------------------------------------------------
    for i in range(cfg["layers"]):
        b = f"{BT}/encoderblock_{i}"
        t = f"backbone.transformer.blocks.{i}."
        put(t + "norm1.", norm(p, f"{b}/LayerNorm_0"))
        put(t + "norm2.", norm(p, f"{b}/LayerNorm_1"))
        put(t + "attn.", attention(p, f"{b}/MultiHeadDotProductAttention_0"))
        put(t + "mlp.fc1.", dense(p, f"{b}/MlpBlock_0/Dense_0"))
        put(t + "mlp.fc2.", dense(p, f"{b}/MlpBlock_0/Dense_1"))
    put("backbone.transformer.encoder_norm.", norm(p, f"{BT}/encoder_norm"))

    # ---- diffusion score network ---------------------------------------
    # FourierFeatures computes `x @ w.T` with w = kernel (out/2, 1); torch
    # Linear(1, out/2, bias=False) computes `x @ weight.T` with the same
    # (out/2, 1) weight, so this one is a straight copy, NOT a transpose.
    sd["score.time_preprocess.linear.weight"] = torch.from_numpy(
        p[f"{DM}/time_preprocess/kernel"].copy())
    put("score.cond1.", dense(p, f"{DM}/cond_encoder/Dense_0"))
    put("score.cond2.", dense(p, f"{DM}/cond_encoder/Dense_1"))
    put("score.fc_in.", dense(p, f"{DM}/reverse_network/Dense_0"))
    for i in range(3):
        b = f"{DM}/reverse_network/MLPResNetBlock_{i}"
        t = f"score.blocks.{i}."
        put(t + "norm.", norm(p, f"{b}/LayerNorm_0"))
        put(t + "fc1.", dense(p, f"{b}/Dense_0"))
        put(t + "fc2.", dense(p, f"{b}/Dense_1"))
    put("score.fc_out.", dense(p, f"{DM}/reverse_network/Dense_1"))
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(HERE / "ref/params.npz"))
    ap.add_argument("--out", default=str(HERE / "octo_small_torch.pt"))
    args = ap.parse_args()

    if not os.path.isfile(args.params):
        raise SystemExit(
            f"{args.params} missing. Produce it first on the JAX side:\n"
            f"  /scratch2/dima/miniforge3/envs/octo/bin/python "
            f"{HERE}/dump_jax_ref.py")

    cfg = _cfg()
    with np.load(args.params) as z:
        params = {k: z[k] for k in z.files if "hf_model" not in k}
    print(f"loaded {len(params)} policy leaves "
          f"({sum(v.size for v in params.values()):,} params) "
          f"[T5 leaves dropped by design]")

    sd = convert(params, cfg)
    n = sum(v.numel() for v in sd.values())
    print(f"converted -> {len(sd)} torch tensors, {n:,} params")

    # Round-trip check: does the port's own module accept this exactly?
    from modelblaster.models.octo_small import OctoSmall  # noqa: PLC0415
    model = OctoSmall(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.endswith("mask_bias")]
    if real_missing or unexpected:
        print(f"!! missing={real_missing}\n!! unexpected={unexpected}")
        return 1
    print("state_dict matches the module exactly (no missing/unexpected keys)")

    torch.save(sd, args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
