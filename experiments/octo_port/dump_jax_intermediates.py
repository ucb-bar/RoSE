#!/usr/bin/env python
"""Dump the JAX attention mask + per-submodule intermediates for the SAME
deterministic input that dump_jax_ref.py used. Used to localise conversion
errors to a specific layer instead of only seeing a bad final number.

    /scratch2/dima/miniforge3/envs/octo/bin/python dump_jax_intermediates.py
"""
import os
import sys
from pathlib import Path

import numpy as np

OCTO_REPO = "/scratch2/dima/misc_sw/octo_work/octo"
CKPT = os.environ.get(
    "OCTO_CKPT",
    str(Path.home() / ".cache/huggingface/hub/models--rail-berkeley--octo-small-1.5"
        / "snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58"))
OUT = Path(__file__).resolve().parent / "ref"
sys.path.insert(0, OCTO_REPO)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax.numpy as jnp  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

model = OctoModel.load_pretrained(CKPT)
io = np.load(OUT / "ref_io.npz")

B, W = 1, 2
observations = {
    "image_primary": jnp.asarray(io["img_primary"]),
    "image_wrist": jnp.asarray(io["img_wrist"]),
    "timestep_pad_mask": jnp.asarray(io["timestep_pad_mask"]),
    "pad_mask_dict": {
        "image_primary": jnp.ones((B, W), dtype=bool),
        "image_wrist": jnp.ones((B, W), dtype=bool),
        "timestep": jnp.ones((B, W), dtype=bool),
    },
}
tasks = {
    "language_instruction": jnp.asarray(io["lang"]),
    "pad_mask_dict": {"language_instruction": jnp.ones((B,), dtype=bool)},
}

_, state = model.module.apply(
    {"params": model.params},
    observations, tasks, jnp.asarray(io["timestep_pad_mask"]),
    train=False, method="octo_transformer",
    capture_intermediates=True, mutable=["intermediates"],
)

flat = flatten_dict(state["intermediates"], sep="/")
save = {}
print(f"{len(flat)} captured intermediates")
for k, v in sorted(flat.items()):
    leaf = v[0] if isinstance(v, tuple) else v
    if hasattr(leaf, "tokens"):          # TokenGroup
        leaf = leaf.tokens
    try:
        a = np.asarray(leaf)
    except Exception:
        continue
    if a.dtype == object or a.ndim == 0:
        continue
    save[k] = a

# Print only the ones we will actually assert on, to keep this readable.
INTERESTING = (
    "attention_mask",
    "observation_tokenizers_primary/SmallStem16_0/StdConv_0/__call__",
    "observation_tokenizers_primary/SmallStem16_0/GroupNorm_0/__call__",
    "observation_tokenizers_primary/SmallStem16_0/embedding/__call__",
    "observation_tokenizers_primary/__call__",
    "observation_tokenizers_wrist/__call__",
    "obs_primary_projection/__call__",
    "task_language_projection/__call__",
    "Transformer_0/encoderblock_0/__call__",
    "Transformer_0/encoderblock_11/__call__",
    "Transformer_0/encoder_norm/__call__",
    "Transformer_0/__call__",
)
for k in sorted(save):
    if any(s in k for s in INTERESTING):
        print(f"  {k:95s} {str(save[k].shape):22s} {save[k].dtype}")

np.savez(OUT / "ref_intermediates.npz", **save)
print("wrote", OUT / "ref_intermediates.npz", f"({len(save)} arrays)")
