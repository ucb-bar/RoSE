#!/usr/bin/env python
"""Dump the octo-small-1.5 Flax checkpoint + a JAX reference forward pass.

Runs in the JAX/Flax env (NOT the shared zephyr env):
    /scratch2/dima/miniforge3/envs/octo/bin/python dump_jax_ref.py

Outputs into ref/:
  params.npz         - every Flax leaf, flattened with '/' separated keys
  param_tree.json    - {key: [shape, dtype, numel]} for inspection
  ref_io.npz         - deterministic inputs + reference activations/outputs
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OCTO_REPO = "/scratch2/dima/misc_sw/octo_work/octo"
CKPT = os.environ.get(
    "OCTO_CKPT",
    str(Path.home() / ".cache/huggingface/hub/models--rail-berkeley--octo-small-1.5"
        / "snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58"),
)
OUT = Path(__file__).resolve().parent / "ref"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, OCTO_REPO)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402

from octo.model.octo_model import OctoModel  # noqa: E402

print("loading", CKPT, flush=True)
model = OctoModel.load_pretrained(CKPT)

# ---------------------------------------------------------------- param dump
flat = flatten_dict(model.params, sep="/")
tree = {}
npz = {}
for k, v in flat.items():
    a = np.asarray(v)
    tree[k] = [list(a.shape), str(a.dtype), int(a.size)]
    npz[k] = a
np.savez(OUT / "params.npz", **npz)
with open(OUT / "param_tree.json", "w") as f:
    json.dump(tree, f, indent=1, sort_keys=True)

total = sum(v[2] for v in tree.values())
t5 = sum(v[2] for k, v in tree.items() if "hf_model" in k)
print(f"\nPARAMS total={total:,}  t5={t5:,}  policy={total - t5:,}")
by_top = {}
for k, v in tree.items():
    top = k.split("/")[0] if "hf_model" not in k else "T5(hf_model)"
    by_top[top] = by_top.get(top, 0) + v[2]
for k in sorted(by_top, key=lambda x: -by_top[x]):
    print(f"  {k:40s} {by_top[k]:>12,}")

# --------------------------------------------------------------- example spec
print("\nEXAMPLE BATCH SPEC")
print(json.dumps(jax.tree_util.tree_map(
    lambda x: [list(np.shape(x)), str(np.asarray(x).dtype)],
    model.example_batch), indent=1, default=str))
print("\nPRETTY SPEC\n", model.get_pretty_spec())

# ------------------------------------------------------------------ ref input
# Deterministic synthetic observation + a *precomputed* T5 language embedding.
# The language embedding is what the port takes as an input tensor: we feed the
# LanguageTokenizer a raw (batch, 16, 768) array, which the tokenizer accepts
# directly (see LanguageTokenizer.__call__ -> isinstance(..., jax.Array) path),
# bypassing T5 entirely. That is exactly the deployment contract.
rng = np.random.RandomState(0)
B, W = 1, 2
img_p = rng.randint(0, 256, size=(B, W, 256, 256, 3)).astype(np.uint8)
img_w = rng.randint(0, 256, size=(B, W, 128, 128, 3)).astype(np.uint8)
lang = rng.randn(B, 16, 768).astype(np.float32) * 0.5
timestep_pad_mask = np.ones((B, W), dtype=bool)

observations = {
    "image_primary": jnp.asarray(img_p),
    "image_wrist": jnp.asarray(img_w),
    "timestep_pad_mask": jnp.asarray(timestep_pad_mask),
    "pad_mask_dict": {
        "image_primary": jnp.ones((B, W), dtype=bool),
        "image_wrist": jnp.ones((B, W), dtype=bool),
        "timestep": jnp.ones((B, W), dtype=bool),
    },
}
tasks = {
    "language_instruction": jnp.asarray(lang),
    "pad_mask_dict": {"language_instruction": jnp.ones((B,), dtype=bool)},
}

outs = model.module.apply(
    {"params": model.params},
    observations,
    tasks,
    jnp.asarray(timestep_pad_mask),
    train=False,
    method="octo_transformer",
)
print("\nTRANSFORMER OUTPUT GROUPS")
for k, v in outs.items():
    print(f"  {k:28s} tokens={tuple(v.tokens.shape)} mask={tuple(v.mask.shape)}")

readout = outs["readout_action"]
embeddings = np.asarray(readout.tokens.mean(axis=-2))  # use_map=False -> mean pool
print("embeddings", embeddings.shape)

# One deterministic score-network evaluation (the thing the DDPM loop calls 20x)
t_in = np.full((B, W, 1), 7.0, dtype=np.float32)
noisy = (rng.randn(B, W, 28) * 0.3).astype(np.float32)
head = model.module.bind({"params": model.params}).heads["action"]
eps_pred = np.asarray(head(outs, time=jnp.asarray(t_in),
                           noisy_actions=jnp.asarray(noisy), train=False))
print("eps_pred", eps_pred.shape)

# Full deterministic DDPM rollout (fixed rng) for the end-to-end check.
actions = np.asarray(head.predict_action(
    outs, rng=jax.random.PRNGKey(42), train=False, embodiment_action_dim=7))
print("actions", actions.shape)

betas = np.asarray(head.betas)
alphas = np.asarray(head.alphas)
alpha_hats = np.asarray(head.alpha_hats)

save = dict(
    img_primary=img_p, img_wrist=img_w, lang=lang,
    timestep_pad_mask=timestep_pad_mask,
    embeddings=embeddings,
    time=t_in, noisy_actions=noisy, eps_pred=eps_pred,
    actions=actions,
    betas=betas, alphas=alphas, alpha_hats=alpha_hats,
)
for k, v in outs.items():
    save[f"out_{k}_tokens"] = np.asarray(v.tokens)
    save[f"out_{k}_mask"] = np.asarray(v.mask)
np.savez(OUT / "ref_io.npz", **save)
print("\nwrote", OUT / "params.npz", OUT / "ref_io.npz")
