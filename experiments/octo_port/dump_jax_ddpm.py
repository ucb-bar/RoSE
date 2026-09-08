#!/usr/bin/env python
"""Dump the exact noise sequence JAX's DDPM sampler consumes, plus its actions.

    /scratch2/dima/miniforge3/envs/octo/bin/python dump_jax_ddpm.py

`DiffusionActionHead.predict_action` draws its noise inside a `jax.lax.scan`,
so the only way to compare the 20-step loop against a torch reimplementation is
to reproduce the PRNG key chain OUTSIDE the scan and hand both sides the same
arrays -- JAX's Threefry and torch's Philox/MT are different generators, so
"same seed" means nothing across them.

The key chain is re-derived here and then CHECKED: we re-run the loop in numpy
with the dumped noise and require it to reproduce upstream's own
`predict_action` output. If that check passes, the dumped noise really is the
noise upstream used, and comparing torch against it is meaningful.
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

model = OctoModel.load_pretrained(CKPT)
io = np.load(OUT / "ref_io.npz")
B, W = 1, 2
STEPS, AFLAT = 20, 28

observations = {
    "image_primary": jnp.asarray(io["img_primary"]),
    "image_wrist": jnp.asarray(io["img_wrist"]),
    "timestep_pad_mask": jnp.asarray(io["timestep_pad_mask"]),
    "pad_mask_dict": {"image_primary": jnp.ones((B, W), dtype=bool),
                      "image_wrist": jnp.ones((B, W), dtype=bool),
                      "timestep": jnp.ones((B, W), dtype=bool)},
}
tasks = {"language_instruction": jnp.asarray(io["lang"]),
         "pad_mask_dict": {"language_instruction": jnp.ones((B,), dtype=bool)}}
outs = model.module.apply({"params": model.params}, observations, tasks,
                          jnp.asarray(io["timestep_pad_mask"]), train=False,
                          method="octo_transformer")
head = model.module.bind({"params": model.params}).heads["action"]

SEED = 42
actions_ref = np.asarray(head.predict_action(
    outs, rng=jax.random.PRNGKey(SEED), train=False, embodiment_action_dim=7))

# --- replicate the key chain exactly as predict_action walks it -------------
rng = jax.random.PRNGKey(SEED)
rng, key = jax.random.split(rng)
noise = np.asarray(jax.random.normal(key, (B, W, AFLAT)))
step_noise = np.empty((STEPS, B, W, AFLAT), dtype=np.float32)
for i in range(STEPS):                      # scan iterates time = 19..0
    rng, key = jax.random.split(rng)
    step_noise[i] = np.asarray(jax.random.normal(key, (B, W, AFLAT)))

# --- self-check: numpy loop with the dumped noise must match upstream -------
betas, alphas, alpha_hats = (np.asarray(head.betas), np.asarray(head.alphas),
                             np.asarray(head.alpha_hats))
x = noise.copy()
for i, t in enumerate(range(STEPS - 1, -1, -1)):
    eps = np.asarray(head(outs, time=jnp.full((B, W, 1), float(t)),
                          noisy_actions=jnp.asarray(x), train=False))
    x = (1 / np.sqrt(alphas[t])) * (
        x - ((1 - alphas[t]) / np.sqrt(1 - alpha_hats[t])) * eps)
    if t > 0:
        x = x + np.sqrt(betas[t]) * step_noise[i]
    x = np.clip(x, -head.max_action, head.max_action)
replay = x.reshape(B, W, 4, 7)[:, -1]
err = float(np.abs(replay - actions_ref).max())
print(f"key-chain self-check: max_abs(replay vs upstream predict_action) = {err:.3e}")
if err > 1e-5:
    raise SystemExit(
        f"key chain NOT reproduced (err={err:.3e}); the dumped noise is not what "
        f"upstream used, so a torch comparison against it would be meaningless.")

np.savez(OUT / "ref_ddpm.npz", noise=noise, step_noise=step_noise,
         actions=actions_ref, seed=np.array(SEED))
print("actions", actions_ref.shape, "\n", actions_ref)
print("wrote", OUT / "ref_ddpm.npz")
