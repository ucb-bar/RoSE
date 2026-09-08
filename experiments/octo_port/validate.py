#!/usr/bin/env python
"""Numerical validation of the PyTorch Octo port against the JAX reference.

Runs in the shared `zephyr` env (torch, no jax); consumes the `.npz` files
`dump_jax_ref.py` / `dump_jax_intermediates.py` produced on the JAX side.

    PYTHONPATH=$ZCS python validate.py

Reports max-abs-error AND cosine similarity per stage and per output, because
either one alone hides a different failure: cosine similarity stays ~1.0 under
a global scale error (exactly what a missed StdConv standardisation looks
like), and max-abs-error alone says nothing about whether the direction of a
384-d embedding survived.

Stage-by-stage rather than end-to-end on purpose. A single final number tells
you the port is broken but not where; these localise it to one layer.
"""
from __future__ import annotations

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

sys.path.insert(0, str(HERE))                     # sibling convert_weights.py

from modelblaster.models.octo_small import (  # noqa: E402
    OctoSmall, _cfg,
)

REF = HERE / "ref"
FAILED: list[str] = []


# One uniform relative tolerance for every check, rather than a hand-picked
# atol per tensor. These activations span |max| from 0.55 to 202, so a single
# absolute threshold is either meaningless at the small end or spuriously
# strict at the large end -- the first run of this script "failed" only
# stem/StdConv_1, whose |max| is 105 and whose RELATIVE error (6.1e-06) was
# identical to the 29 checks that passed. Error is gated on
# max_abs/|ref|max, i.e. pure fp32 accumulation-order noise, which is what
# a reimplementation on a different framework should show.
#
# 5e-5 is ~6x headroom over the worst relative error actually observed
# (8.1e-06) and is nowhere near loose enough to hide a real bug: the
# conv-bias-fold error this port hit registered rel = 2.5e-01.
RTOL = 5.0e-5


def compare(name: str, a: np.ndarray, b: np.ndarray, atol: float = 0.0) -> None:
    """a = port, b = JAX reference. Gated on relative error (see RTOL)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    mae = float(np.abs(a - b).max())
    rel = mae / max(float(np.abs(b).max()), 1e-12)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(a @ b / denom) if denom > 0 else 1.0
    ok = rel <= RTOL
    if not ok:
        FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'} {name:52s} "
          f"max_abs={mae:.3e}  rel={rel:.3e}  cos={cos:.9f}  "
          f"|ref|max={np.abs(b).max():.4g}")


def main() -> int:
    cfg = _cfg()
    if cfg["part"] != "full" or cfg["window"] != 2 or cfg["primary"] != 256 \
            or cfg["wrist"] != 128 or cfg["layers"] != 12 or cfg["goal"]:
        raise SystemExit(
            "validate.py compares against a reference dumped for the stock "
            "config (window=2, primary=256, wrist=128, 12 layers, no goal "
            "images, PART=full). Unset the MODELBLASTER_OCTO_* knobs.")

    io = np.load(REF / "ref_io.npz")
    inter = np.load(REF / "ref_intermediates.npz")

    model = OctoSmall(cfg).eval()
    sd = torch.load(HERE / "octo_small_torch.pt", map_location="cpu",
                    weights_only=True)
    model.load_state_dict(sd, strict=False)

    # ---------------------------------------------------------------- masks
    print("\n[1] attention mask — must be bit-exact, it is a constant")
    ref_mask = inter["octo_transformer/BlockTransformer_0/attention_mask"]
    got = model.backbone.mask_bias.numpy()[0, 0] == 0.0
    ref = ref_mask[0, 0].astype(bool)
    n_bad = int((got != ref).sum())
    print(f"  {'ok ' if n_bad == 0 else 'FAIL'} mask {ref.shape}  "
          f"mismatched entries = {n_bad}  "
          f"(allowed fraction ref={ref.mean():.4f} port={got.mean():.4f})")
    if n_bad:
        FAILED.append("attention_mask")

    # ------------------------------------------------------- goal-fold check
    print("\n[2] conv0 — port's 6-channel path vs one built straight from the "
          "Flax kernel\n    (also demonstrates why the bias-fold shortcut is "
          "wrong for a padded conv)")
    with np.load(REF / "params.npz") as z:
        pk = ("octo_transformer/observation_tokenizers_primary/SmallStem16_0"
              "/StdConv_0")
        k6 = z[f"{pk}/kernel"]
        b6 = z[f"{pk}/bias"]
    from convert_weights import weight_standardize  # noqa: PLC0415
    w6 = torch.from_numpy(np.transpose(weight_standardize(k6),
                                       (3, 2, 0, 1)).copy())
    bb6 = torch.from_numpy(b6.copy())
    g = torch.Generator().manual_seed(7)
    xo = torch.randn(2, 3, 64, 64, generator=g)
    c0 = model.backbone.tok_primary.stem.convs[0]
    with torch.no_grad():
        x6 = torch.cat([xo, torch.full_like(xo, -1.0)], 1)
        full6 = torch.nn.functional.conv2d(x6, w6, bb6, stride=2, padding=1)
        port = torch.nn.functional.conv2d(x6, c0.weight, c0.bias,
                                          stride=2, padding=1)
        # what the (wrong) bias fold would have produced
        fold_b = bb6 + (-1.0) * w6[:, 3:].sum(dim=(1, 2, 3))
        folded3 = torch.nn.functional.conv2d(xo, w6[:, :3].contiguous(), fold_b,
                                             stride=2, padding=1)
    compare("conv0 port vs flax kernel", port.numpy(), full6.numpy(), 1e-6)
    interior = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    print(f"  -- bias-fold shortcut: max_abs overall = "
          f"{(folded3 - full6).abs().max():.4e}, but on the INTERIOR only "
          f"{(folded3[interior] - full6[interior]).abs().max():.4e}"
          f"  <- exact inside, wrong in the pad ring")

    # ------------------------------------------------------------ stem stages
    print("\n[3] SmallStem16 stages (primary) — NHWC ref transposed to NCHW")
    img_p = torch.from_numpy(io["img_primary"]).permute(0, 1, 4, 2, 3).float()
    img_w = torch.from_numpy(io["img_wrist"]).permute(0, 1, 4, 2, 3).float()
    lang = torch.from_numpy(io["lang"])

    tok = model.backbone.tok_primary
    with torch.no_grad():
        x = img_p.reshape(-1, 3, 256, 256) / 127.5 - 1.0
        # the stem sees 6 channels: obs and the constant -1.0 goal half
        x = torch.cat([x, tok.goal_const.expand_as(x)], dim=1)
        acts = {}
        for i, (cv, nm) in enumerate(zip(tok.stem.convs, tok.stem.norms)):
            x = cv(x)
            acts[f"StdConv_{i}"] = x.clone()
            x = nm(x)
            acts[f"GroupNorm_{i}"] = x.clone()
            x = torch.relu(x)
        x = tok.stem.embedding(x)
        acts["embedding"] = x.clone()

    base = "octo_transformer/observation_tokenizers_primary/SmallStem16_0"
    for key, atol in (("StdConv_0", 2e-4), ("GroupNorm_0", 2e-4),
                      ("StdConv_1", 5e-4), ("GroupNorm_1", 5e-4),
                      ("StdConv_2", 2e-3), ("GroupNorm_2", 2e-3),
                      ("StdConv_3", 5e-3), ("GroupNorm_3", 5e-3),
                      ("embedding", 5e-3)):
        rk = f"{base}/{key}/__call__"
        if rk not in inter:
            continue
        compare(f"stem/{key}", acts[key].permute(0, 2, 3, 1).numpy(),
                inter[rk], atol)

    print("\n[4] tokenizer outputs + projections")
    with torch.no_grad():
        tp = model.backbone.tok_primary(img_p)
        tw = model.backbone.tok_wrist(img_w)
        pp = model.backbone.proj_primary(tp)
        pl = model.backbone.proj_language(lang)
    compare("tokenizer_primary (B,W,256,512)", tp.numpy(),
            inter[f"{base.replace('/SmallStem16_0','')}/__call__"], 5e-3)
    compare("tokenizer_wrist   (B,W,64,512)", tw.numpy(),
            inter["octo_transformer/observation_tokenizers_wrist/__call__"], 5e-3)
    compare("obs_primary_projection", pp.numpy(),
            inter["octo_transformer/obs_primary_projection/__call__"], 5e-3)
    compare("task_language_projection", pl.numpy(),
            inter["octo_transformer/task_language_projection/__call__"], 1e-5)

    print("\n[5] transformer — per-block, 690 tokens x 384")
    bt = "octo_transformer/BlockTransformer_0/Transformer_0"
    with torch.no_grad():
        w = cfg["window"]
        bk = model.backbone
        task = bk.proj_language(lang) + bk.pos_language
        groups = [bk.proj_primary(tp) + bk.pos_primary[:, :w],
                  bk.proj_wrist(tw) + bk.pos_wrist[:, :w],
                  torch.cat([task.unsqueeze(1)] * w, dim=1),
                  bk.pos_readout[:, :w].expand_as(
                      (bk.proj_primary(tp))[:, :, :1, :])]
        ts = torch.cat(groups, dim=2).reshape(-1, w * bk.n_per_step, 384)
        h = torch.cat([task, ts], dim=1)
        for i, blk in enumerate(bk.transformer.blocks):
            h = blk(h, bk.mask_bias)
            rk = f"{bt}/encoderblock_{i}/__call__"
            if rk in inter and i in (0, 1, 5, 10, 11):
                compare(f"encoderblock_{i}", h.numpy(), inter[rk], 2e-2)
        enc = bk.transformer.encoder_norm(h)
    compare("encoder_norm (transformer out)", enc.numpy(),
            inter[f"{bt}/encoder_norm/__call__"], 2e-2)

    print("\n[6] readout embedding — the score network's conditioning input")
    with torch.no_grad():
        emb = model.backbone(img_p, img_w, lang)
    compare("readout embedding (B,W,384)", emb.numpy(), io["embeddings"], 2e-2)

    print("\n[7] score network — one denoising evaluation (eps_pred)")
    noisy = torch.from_numpy(io["noisy_actions"])
    t_in = torch.from_numpy(io["time"])
    with torch.no_grad():
        eps_ref_emb = model.score(torch.from_numpy(io["embeddings"]), noisy, t_in)
        eps_full = model(img_p, img_w, lang, noisy, t_in)
    # Driven by the JAX embedding: isolates the score net from any backbone drift.
    compare("eps_pred | JAX embedding", eps_ref_emb.numpy(), io["eps_pred"], 1e-4)
    # End-to-end through the port's own backbone.
    compare("eps_pred | port embedding (end-to-end)",
            eps_full.numpy(), io["eps_pred"], 5e-3)

    print("\n[8] diffusion schedule constants")
    for k in ("betas", "alphas", "alpha_hats"):
        compare(k, getattr(model, k).numpy(), io[k], 1e-6)

    print("\n[9] full 20-step DDPM rollout (same injected noise on both sides)")
    if (REF / "ref_ddpm.npz").is_file():
        d = np.load(REF / "ref_ddpm.npz")
        with torch.no_grad():
            act = model.sample_actions(
                torch.from_numpy(io["embeddings"]),
                torch.from_numpy(d["noise"]),
                torch.from_numpy(d["step_noise"]))
        compare("actions (B,4,7) | JAX embedding", act.numpy(), d["actions"], 1e-3)
        with torch.no_grad():
            act2 = model.sample_actions(
                emb, torch.from_numpy(d["noise"]),
                torch.from_numpy(d["step_noise"]))
        compare("actions (B,4,7) | port embedding", act2.numpy(), d["actions"], 5e-2)
    else:
        print("  -- ref/ref_ddpm.npz not present; run dump_jax_ddpm.py")

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
