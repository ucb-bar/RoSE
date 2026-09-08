# Octo → ModelBlaster port: scoping note

Written before any model code, from the checkpoint itself (not from the paper
and not from the upstream README). Every number below was read out of
`config.json` / the restored Flax param tree, or measured by running the JAX
model — see `dump_jax_ref.py` and `ref/param_tree.json`.

## 1. Exact variant

| | |
|---|---|
| Variant | **`octo-small-1.5`** (`rail-berkeley/octo-small-1.5`) |
| Checkpoint | `~/.cache/huggingface/hub/models--rail-berkeley--octo-small-1.5/snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58` |
| Step | `300000` (orbax `CheckpointManager`, single `default/checkpoint` blob, 546 MB) |
| Backbone size | ViT-S: `token_embedding_size=384`, 12 layers, 6 heads, `mlp_dim=1536` |
| Attention | `use_correct_attention=True` (the post-December mask convention) |
| Position embeddings | pre-added to tokens; `add_position_embedding=False` inside the Transformer |
| Dropout | 0.0 everywhere (`dropout_rate=0.0`, `attention_dropout_rate=0.0`) — eval == train graph |

## 2. Parameter count (measured, from the restored tree)

```
total (incl. T5)        136,670,604
  T5 text encoder       109,628,544     <- NOT PORTED, see §7
  policy                 27,042,060     <- this is the port
```

Policy breakdown:

```
transformer (12 x Encoder1DBlock + encoder_norm)   21,294,336
diffusion action head (score network)               1,703,564
obs tokenizer primary (SmallStem16)                 1,058,048
obs tokenizer wrist   (SmallStem16)                 1,058,048
obs_primary_pos_embedding  [1,10,256,384]             983,040
task_language_projection   [768->384]                 295,296
obs_wrist_pos_embedding    [1,10, 64,384]             245,760
obs_primary_projection     [512->384]                 196,992
obs_wrist_projection       [512->384]                 196,992
task_language_pos_embedding [1,16,384]                  6,144
readout_action_pos_embedding [1,10,1,384]               3,840
```

267 policy leaves. The `_pos_embedding` tensors are sized for
`max_horizon=10` and sliced to the live window at run time.

## 3. Action head — **DiffusionActionHead**, not MSE

This is the part that materially changes the port, so it is spelled out.

```
DiffusionActionHead(readout_key="readout_action", use_map=False,
                    action_dim=7, action_horizon=4,
                    time_dim=32, num_blocks=3, hidden_dim=256,
                    use_layer_norm=True, diffusion_steps=20,
                    n_diffusion_samples=1, max_action=5.0)
```

* `use_map=False` → the readout embedding is a **mean over the readout tokens**.
  `readouts={"action": 1}`, i.e. there is exactly **one** readout token per
  timestep, so that mean is over a length-1 axis: it is a squeeze, not a
  reduction. No MAPHead, no `probe` parameter in the tree (confirmed: no
  `map_head` key exists). One less op to port.
* Score network (`octo/model/components/diffusion.py::create_diffusion_model`):
  * `time_preprocess` = `FourierFeatures(32, learnable=True)`, one param
    `kernel [16,1]`; computes `concat(cos(2πtwᵀ), sin(2πtwᵀ))` → 32.
  * `cond_encoder` = `MLP((64, 32))`, swish between, **no** activation on the
    final layer (`activate_final=False`), no LayerNorm (`use_layer_norm`
    defaults False on `MLP`).
  * `reverse_network` = `MLPResNet(num_blocks=3, out_dim=28, hidden_dim=256,
    use_layer_norm=True, activation=swish)`:
    `Dense(256)` → 3 × [`LayerNorm` → `Dense(1024)` → swish → `Dense(256)` → +residual]
    → swish → `Dense(28)`.
  * Input to `reverse_network` is `concat([cond_enc(32), obs_enc(384), noisy_actions(28)])`
    = **444**, which matches `reverse_network/Dense_0/kernel [444,256]`. Good
    independent confirmation that the wiring is understood.
* **The denoising loop matters more than the network**, exactly as flagged.
  The transformer runs **once**; the score network runs **20 times** (`diffusion_steps=20`),
  in a DDPM ancestral-sampling loop:
  ```
  x_T ~ N(0, I)                                     # (B, W, 28)
  for t = 19 .. 0:
      eps = score(embeddings, x_t, t)
      x = (1/sqrt(alphas[t])) * (x - ((1-alphas[t])/sqrt(1-alpha_hats[t])) * eps)
      x = x + (t>0) * sqrt(betas[t]) * z ,  z ~ N(0,I)
      x = clip(x, -5, 5)
      x = where(action_mask, x, sqrt(1-alpha_hats[t]) * z)     # masked dims
  actions = rearrange(x, "... (h a) -> ... h a", h=4, a=7)[..., -1, :, :]
  ```
  `betas` is a fixed cosine schedule (`cosine_beta_schedule(20, s=0.008)`) — a
  **constant**, folded into the port as a buffer. With
  `embodiment_action_dim=7 == action_dim`, `action_mask` is all-True and the
  `where` is a no-op; it only bites for embodiments with fewer than 7 dims.
* Cost consequence: the score net is 1.70 M params ≈ 1.70 MMAC/step → ~34 MMAC
  for the whole loop, versus **~19 GMAC** for one transformer pass (see §6).
  So the loop is ~0.2% of the FLOPs. The right decomposition is
  **backbone-once + score-net-×20**, and the loop is *not* worth unrolling into
  one giant static graph. (It *could* be: 20 iterations is a compile-time
  constant and the only per-step non-constant is the noise `z`, which would
  become 20 extra input tensors. It is just pointless.)
* The two per-step RNG draws (`z`) are the only stochasticity. For validation I
  drive both sides from the *same* pre-drawn noise so the comparison is
  deterministic rather than distributional.

## 4. Observation window, resolutions, tokenizers

```
window_size (horizon)   2          (max_horizon = 10)
image_primary           (B, 2, 256, 256, 3)  uint8
image_wrist             (B, 2, 128, 128, 3)  uint8
```

Both observation tokenizers are `ImageTokenizer` + `SmallStem16`, and both have
`task_stack_keys=["image_primary"]` / `["image_wrist"]`. **That is a trap worth
stating**: `ImageTokenizer` concatenates the *task* (goal) image onto the
*observation* image along the channel axis, and when the task has no goal image
(the language-conditioned case, which is ours) it substitutes
`zeros_like(observations[k][:,0])`. So

* the stem's input is **6 channels**, not 3 — confirmed by
  `StdConv_0/kernel [3,3,6,32]`;
* `normalize_images` (`x/127.5 - 1.0`) is applied to the *whole* 6-channel
  stack, so the zero-filled goal half becomes **-1.0**, not 0.0. Feeding zeros
  post-normalisation would be wrong.

`SmallStem16`: 4 × [`StdConv(3×3, stride 2, pad 1)` → `GroupNorm` → `relu`] with
features `(32, 96, 192, 384)`, then a `1×1` `Conv` "embedding" → 512 channels
(`patch_size//16 = 1`, so the "patchify" conv is a pointwise 1×1).

Token counts (verified against a real JAX forward pass):

| group | shape | tokens/timestep |
|---|---|---|
| `obs_primary` | 256→128→64→32→**16**, 16×16 | **256** |
| `obs_wrist` | 128→64→32→16→**8**, 8×8 | **64** |
| `obs_task_language` | repeated language (`repeat_task_tokens=True`) | **16** |
| `readout_action` | learned, content-free | **1** |
| | | **337 / timestep** |

Sequence fed to the transformer: 16 prefix (`task_language`) + 2 × 337 =
**690 tokens**, d=384. Confirmed live:
`readout_action tokens=(1,2,1,384)`, `obs=(1,2,336,384)`.

`repeat_task_tokens=True` means the language tokens appear **twice**: once as
the prefix group and again tiled across every timestep. Both copies come from
the *same* `task_language_projection` + `task_language_pos_embedding`
parameters, so it is a tile, not a second projection.

## 5. Language embedding shape — the input the port takes

`LanguageTokenizer(encoder="t5-base", finetune_encoder=False)`. Its
`__call__` has two paths, and the one we use is already in upstream:

```python
if not isinstance(tasks["language_instruction"], (jax.Array, np.ndarray)):
    tokens = self.hf_model(**tasks["language_instruction"]).last_hidden_state
else:
    tokens = tasks["language_instruction"]          # <- pass the embedding in
```

So handing the policy a raw array bypasses T5 entirely, with no upstream patch.
The text processor is `HFTokenizer(t5-base, max_length=16, padding="max_length",
truncation=True)`, and T5-base's hidden size is 768, therefore:

> **language embedding = `(batch, 16, 768)` float32.**

It is then `task_language_projection` (`768→384`) + a `[1,16,384]` position
embedding. This is the port's language input, computed once offline per task
instruction.

## 6. Compute shape of the thing being ported

Per transformer pass, at window=2 with both cameras (690 tokens, 12 layers):

```
QKV + out projections   4 × 384 × 384 × 690 × 12   ≈  4.07 GMAC
MLP (384→1536→384)      2 × 384 ×1536 × 690 × 12   ≈  8.14 GMAC
attention scores + AV   2 × 690² × 384 × 12        ≈  4.39 GMAC
                                                    ≈ 16.6 GMAC
```
plus the two conv stems (~0.9 GMAC for 256² primary + 128² wrist) → **~17.5 GMAC
per control step**, against ~34 MMAC for the entire 20-step denoising loop.

This is the single most important engineering fact in the note: **the port's
cost is a 690-token ViT-S, and the token count is dominated by the primary
camera (256 of 337 tokens/timestep).** The obvious knobs, in order of
effectiveness, are: drop the wrist camera (−64 tok/step), window=1 (halves the
timestep tokens), and a smaller primary resolution (128² → 64 tokens/step).
The port exposes all three as env-var shape knobs so this is measurable rather
than argued about.

For calibration against the prior JAX numbers in
`/scratch2/dima/misc_sw/octo_work/results_*.json`: those are **octo-base**
(202 M total / 92.8 M policy, ViT-B d=768), CPU median 768 ms / GPU 83 ms, and
`results_CPU_8.json` sweeps window=1 → 152.6 ms, window=2 → 299.8 ms. Our
target is ~3.4× smaller in the transformer, so those figures are an upper
bound, not a baseline.

## 7. What is deliberately NOT ported, and why

**The T5-base text encoder (109,628,544 params = 80% of the full checkpoint,
54% of octo-base's).** A robot executing a fixed task has a fixed language
instruction; `finetune_encoder=False` means T5 is frozen, so its output for a
given instruction is a constant. It is computed **once, offline**, and handed to
the policy as a `(1, 16, 768)` constant tensor. Upstream already supports
exactly this (§5) — this is not a shortcut or an approximation, it is the
deployment path. **The port is complete without it.** Anyone reading this later
should not file the missing T5 as a gap.

Also not ported, because this checkpoint does not use them: `MAPHead`
(`use_map=False`), `FilmConditioning` (`use_film=False`, no `task_film_keys`),
`TokenLearner` (`use_token_learner=False`), `unet.py` /
`UNetDDPMActionHead`, `BinTokenizer` / `LowdimObsTokenizer`, and the
`ViTResnet` / `PatchEncoder` stems. Confirmed by absence from the param tree,
not by reading the config alone.

## 8. Judgement: is this materially harder than the brief implied?

Harder in two specific places, neither a blocker:

1. **`StdConv` weight standardisation.** The stems' convs are not plain convs:
   `StdConv.param` overrides parameter *read* to return
   `(w - mean) / (std + 1e-5)` over axes `[0,1,2]` (HWIO → per-output-channel
   over kernel×kernel×in_ch). For a frozen checkpoint this is a pure function
   of the weights, so it **folds into the converted tensors at conversion time**
   and costs nothing at run time. It does mean a naive
   "copy kernel, transpose to OIHW" conversion is silently wrong — the weights
   would be off by a per-channel scale and shift. This is exactly the class of
   quiet failure the brief warned about, so it gets an explicit unit check.
2. **`GroupNorm` is not in ModelBlaster's op vocabulary.** 8 instances (4 per
   stem × 2 stems). `extract_graph_export.py`'s `_NEW_COMPUTE` has
   `layer_norm`, not `group_norm`. Options, cheapest first: (a) rewrite as a
   reshape + `layer_norm` over the (group, C/g, H, W) tail — mathematically
   identical since flax GroupNorm normalises over channels-within-group *and*
   spatial; (b) add a `group_norm` op kind. The port takes (a) as an
   env-selectable lowering so the graph lands on ops that already exist.

Two epsilon mismatches that would otherwise silently degrade the port, found by
reading flax rather than by debugging:

* flax `nn.LayerNorm` default `epsilon=1e-6`; torch `nn.LayerNorm` default
  `eps=1e-5`. **Must** pass `eps=1e-6`.
* flax `nn.GroupNorm` default `epsilon=1e-6`, `num_groups=32`; torch
  `nn.GroupNorm` default `eps=1e-5`. **Must** pass `eps=1e-6`.

And one numerics detail: flax masks attention logits with
`jnp.finfo(float32).min` (-3.4e38), not `-inf`. The port matches that literal so
softmax rounding agrees bit-for-bit rather than approximately.

The attention mask itself is worth noting as *easier* than it looks: it is
generated by pure NumPy from group sizes and `AttentionRule`s, combined with the
pad mask. With both cameras present, language present and no padded timesteps —
the deployment case — it is a **compile-time constant** `(690, 690)` boolean. The
port bakes it as a buffer and asserts it equals the mask JAX produces.

**Verdict: proceed.** No part of this needs a framework feature ModelBlaster
lacks, and nothing here justifies stopping. The diffusion head is a 20-step loop
over a 1.7 M-param MLP, which is a scheduling question, not a porting obstacle.

## 9. Reference environment

No package was installed, upgraded or removed anywhere. The JAX reference runs
in the **pre-existing** `/scratch2/dima/miniforge3/envs/octo` env
(jax 0.4.20 / flax 0.7.5 / numpy 1.24.3 / tf 2.15.0, **no torch**); the port
runs in the shared `zephyr` env (torch, no jax). Because no single env has both,
the two sides communicate through `.npz` on disk — which is strictly better for
this job anyway, since it makes the comparison reproducible and rerunnable
without a JAX install.

`/scratch2/dima/misc_sw/octo_work/` was treated as read-only throughout.
