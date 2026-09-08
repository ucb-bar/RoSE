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

---

## 10. Deliverable 4 — fp32 extraction: BLOCKED in both extractors

**fp32 extraction does not currently succeed, for two independent reasons in
two different files. Neither is a defect in the port.** The graph itself is
fully covered — see §11 — so the missing piece is an emitter, not an op.

Everything below was produced by *running* the extractors, not by reading them.
Artifacts are under `inventory/`.

### 10a. `pipeline/extract_graph.py --quant fp32` (torch.fx) — structurally unusable

The model traces cleanly under `torch.fx.symbolic_trace`, and 220 of its 352
fx nodes classify as supported. It then hard-raises. Two blockers, both
`raise NotImplementedError`, neither with a bypass flag:

```
extract_graph.py:5382   get_attr nodes not supported yet: backbone_pos_language
extract_graph.py:5369   unsupported call_method (transpose/unflatten/reshape/...)
```

Full fx-level rejection census for the faithful config (`op_inventory.py`):

```
[supported] 220        [REJECTED] 132
  86  nn.Linear          48  method:transpose
  30  fn:add             36  method:unflatten
  28  nn.LayerNorm       18  get_attr            <- position embeddings + mask
  12  fn:sdpa            12  method:flatten
  12  fn:gelu             8  method:reshape
  10  nn.Conv2d           3  method:expand_as
   8  nn.GroupNorm        2  fn:sub
   8  fn:relu             2  method:permute
   7  fn:cat              1  method:unsqueeze
   6  fn:getitem          1  fn:cos
   6  fn:mul              1  fn:sin
   5  fn:sigmoid
   2  fn:truediv
```

**110 of the 132 rejections are pure tensor reshaping** —
`transpose`/`unflatten`/`flatten`/`reshape`/`permute`/`unsqueeze`/`expand_as`
as *methods*. `extract_graph.py` only accepts two `call_method` targets at all
(`chunk`, `flip`, `:5280-5371`); everything else raises. Splitting `(B, L, 384)`
into `(B, 6, L, 64)` heads and back is not optional in an attention block, so
this rejects **any** transformer, not just Octo.

The other 18 (34 with `GN=layernorm`) are `get_attr`: Octo's four learned
positional-embedding tables and the constant attention-mask buffer. Any tensor
that is an `nn.Parameter`/buffer consumed by *arithmetic* rather than owned by
a recognised module becomes `get_attr` in fx. **Every ViT-shaped model has
learned position embeddings**, so this too is a whole-model-class blocker.

Note the direction of the `GN` knob here: `MODELBLASTER_OCTO_GN=layernorm`
makes the fx path *worse* (132 → 184 rejections), because fx handles
`nn.LayerNorm` the module but not `F.layer_norm` the function. `GN=native` is
correct for fx; `GN=layernorm` is correct for export. The knob is not cosmetic.

### 10b. `pipeline/extract_graph_export.py` (torch.export) — refuses fp32 by design

This is the extractor whose vocabulary actually fits (it treats all 148 alias
nodes as free, which is exactly what fx rejects). It ingests the model
end-to-end. But:

```
extract_graph_export.py:1976   raise SystemExit("--quant fp32 not implemented
                               in the export path yet")
```

checked *before* the model is even loaded. `--quant int8` and `--quant fp16`
are implemented; fp32 is not.

Two small enabling changes were needed and made, because the export path
whitelists models in two places:

```
extract_graph_export.py:1941   --model choices += "octo_small"
extract_graph_export.py:1930   _import_model_module(): added the octo_small branch
```

`_load_model` itself is generic (`get_model()` + `get_sample_input()`), so
that is the whole change. With it, `--inventory-only` runs and the extractor
writes its own classification — reproduced in §11.

### 10c. What it would take

In rough order of effort, cheapest first:

1. **Run the port at `--quant fp16` through the export path.** Already
   unblocked by 10b's two lines; needs no new op. This is the shortest route
   to real IR + a scheduled graph, and fp16 is what ViNT's conv encoder
   already uses.
2. **Add an fp32 mode to the export walker.** It is the *same* walker: it
   already carries an `fp16` branch that skips scale/multiplier/shift and
   casts weights, and `op_suffix`/`tensor_dtype`/`weight_np_dtype` are already
   parameterised on quant. fp32 is that branch with an empty suffix and no
   cast. This is the principled fix and it lifts the restriction for every
   future model, not just Octo.
3. **Teach the fx path aliases + `get_attr`.** Much larger: `_ALIAS`-style
   pass-through for the shape methods plus constant-tensor handling. Only
   worth it if the fx path has to stay the fp32 route.

I did **not** do (2). It is a change to a shared 2,375-line pipeline file that
every other model's int8 flow goes through, and the brief scoped this task to
the port plus a report. Reporting it precisely is the deliverable; silently
rewriting the shared extractor is not. Flagging it as the recommended next
step, with the evidence above.

## 11. Deliverable 5 — op-coverage gap report

### 11a. The histogram, from the extractor itself

`inventory/export_faithful/op_inventory.txt` (613 aten nodes) and
`inventory/export_covered/op_inventory.txt` (648 nodes), both written by
`extract_graph_export.py --inventory-only`:

| class | faithful | covered (`GN=layernorm NORM=0 TIME=lut`) |
|---|---|---|
| supported | 141 | 145 |
| new | 63 | 76 |
| alias (free) | 116 | 148 |
| tail (host scalar) | 2 | 0 |
| **UNKNOWN** | **12** | **0** |

Faithful config, per op:

```
supported  86 linear   30 add   10 conv2d   8 relu   7 cat
new        28 layer_norm   12 sdpa   12 gelu   6 mul   5 sigmoid
alias      48 transpose  36 unflatten  12 flatten  8 reshape  5 slice
            3 expand_as  2 permute  1 unsqueeze  1 select
tail        2 div
UNKNOWN     8 group_norm   2 sub   1 cos   1 sin
```

**Every one of the 12 unknowns is removable by a knob, exactly, with no
accuracy cost — and the covered config verifies at 0 UNKNOWN.** The
extractor confirms this itself: it prints a `!!! UNKNOWN ops` banner and
`sys.exit(1)` in the faithful config, and prints nothing in the covered one.

### 11b. The four gaps and what each costs

| gap | count | status | fix |
|---|---|---|---|
| `group_norm` | 8 | **exists in fx, missing in export, no int8 kernel anywhere** | `GN=layernorm` (exact) |
| `sub` | 2 | image normalisation only | `NORM=0` (pre-normalised input) |
| `cos`, `sin` | 2 | int8-only in the repo, no fp32 spec | `TIME=lut` (exact) |

* **`group_norm`** is the interesting one, because it is *half* present. The fx
  extractor supports it (`extract_graph.py:628`, `:4175`) and there is a
  curated vectorised RVV fp32 kernel (`kernels/rvv/rvv_group_norm_direct.c`,
  validated `35_GroupNorm PASS 7.03e-06`), plus an auto-synthesised
  `group_norm_f16`. But it is absent from the export extractor's tables, and
  **there is no `group_norm_s8` at all** — so an int8 Octo must either gain
  one or pin GroupNorm to fp16 via the mixed-precision path, the same way ViNT
  pins its goal encoder. `GN=layernorm` sidesteps all of it and is exact
  (measured `max_abs=1.788e-06` vs `nn.GroupNorm`, §12).
* **`cos`/`sin`** exist only as `cos_s8`/`sin_s8` (int8, curated
  `rvv_*_s8_rvv_memo_lut_gather.c`); there is no fp32 spec. Rather than add
  one, `TIME=lut` deletes the need: the time-conditioning branch is a frozen
  function of the timestep and DDPM visits exactly 20 integer timesteps, so
  FourierFeatures + the 2-layer cond MLP collapse to a **20×32 host-side
  table** (`ScoreNet.cond_table`). Measured bit-exact (`max_abs=0.0`) against
  the on-the-fly branch, and the full 20-step rollout still matches JAX at
  `1.25e-06`. Same reasoning as not porting T5.
* **`sub`** is only the `x/127.5 - 1` normalisation. Worth noting the
  normalisation *cannot* be folded into conv0: the scale folds cleanly but the
  `-1` shift does not, for the same zero-padding reason the goal-channel fold
  fails (§8.3).

### 11c. Curated vs reference kernels for Octo's op set

Selection: `generate_kernels.py` probes
`kernels/<target>/<backend>_<op>_<algorithm>.c` first, verifies it against the
model's real shapes, and falls back to `spec.reference_impl` on a miss
(`--global-curated-dir modelblaster/kernels`).

| op | curated? | where |
|---|---|---|
| `conv2d` fp32 | yes | `kernels/rvv/rvv_conv2d_rvv_oc_blocked.c` |
| `linear` fp32 | yes | `kernels/rvv/rvv_linear_direct.c` |
| `layer_norm` fp32 | yes | rvv `direct` |
| `group_norm` fp32 | yes | rvv `direct` (**rvv only**; no s8 variant) |
| `gelu` / `gelu_exact` | yes | rvv `direct` (both) |
| `softmax` fp32 | yes | rvv `direct` |
| `relu`, `sigmoid` fp32 | yes | rvv `direct` |
| `matmul`/`_ta`/`_tb`/`bmm` | yes | rvv `direct` each |
| `mean_dim` | yes | rvv `direct` (unused here — see below) |
| **`sdpa`** | **NO** | reference only, on every backend |
| **`add` fp32** | **NO** | `add_s8`/`add_f16` only |
| **`mul` fp32** | **NO** | `mul_s8`/`mul_f16`/`mul_c1_*` only |
| **`cat*_c1` fp32** | **NO** | only an f16 curated variant exists |

So on RVV fp32 the port's four uncurated ops are **`sdpa` (12), `add` (30),
`mul` (6), `cat` (7)** — they fall back to the scalar reference compiled with
`-march=rv64gcv`. `add`/`mul`/`cat` are memory-bound elementwise/copy work
that auto-vectorises acceptably; **`sdpa` is the one that matters**, and it is
worse than "uncurated":

> the reference `sdpa` (`reference_kernels.py:12386`) supports **no mask**, no
> dropout and no custom scale, and declares a VLA `float scores[S]` on the
> stack. Octo's attention is **masked** — a 690×690 blockwise-causal mask —
> and S=690. So the existing `sdpa` kernel is not merely slow for this model,
> it is *semantically wrong* and would silently drop the mask.

The port's answer is already in place: `MODELBLASTER_OCTO_ATTN=matmul` writes
attention out as `matmul → mul_scalar → add(mask) → softmax → matmul`, all of
which have curated RVV fp32 kernels, and it is equivalent to the sdpa form to
round-off (`max_abs=1.073e-06`, §12). That is the same decomposition
`extract_graph_export.py:81-84` already applies to ViNT's sdpa. **Anyone
scheduling this model on real hardware should use `ATTN=matmul`, or add a
masked `sdpa` spec.** The mask is additive and pre-baked as a buffer, so the
`add` is a plain elementwise add against a constant.

`mean_dim` is listed only to record that the port does **not** need it: the
readout mean-pool is over a length-1 axis (`readouts={"action":1}`) and lowers
to a `select`, which is a free alias.

## 12. Equivalence of the lowering knobs (measured, not asserted)

Each knob claims to be mathematically identical to the faithful form. Verified
end-to-end on the port's own output rather than argued:

```
GN=layernorm vs nn.GroupNorm     max_abs=1.788e-06  rel=9.955e-07  cos=1.000000000
ATTN=matmul  vs sdpa             max_abs=1.073e-06  rel=5.973e-07  cos=1.000000119
TIME=lut     vs fourier branch   max_abs=0.000e+00  (bit-exact)
TIME=lut     eps_pred vs JAX     max_abs=8.345e-07  rel=4.622e-07
TIME=lut     20-step actions     max_abs=1.252e-06  rel=1.244e-06
```

`NORM=0` is definitionally identical (it moves `x/127.5-1` out of the graph).

## 13. On-target validation, and the ten bugs it took to get there

`bash modelblaster/examples/octo_small/run.sh` with `QUANT=int8` runs the port
end to end on a board. §1-12 was measured in PyTorch or against JAX; this is
the generated C, executing the extracted graph on a target.

**Configuration (all of it in `examples/octo_small/run.sh`):** 316 ops,
`--per-channel` weight scales, 8 calibration samples of real BridgeData
frames, `MODELBLASTER_ACT_PERCENTILE=99.99`, `SPLITFC=1`, plus the §1-12
coverage knobs (`GN=layernorm TIME=lut NORM=0 ATTN=matmul`).

```
RUNNER=spike    max_abs_err=8 of 127 int8 LSBs   cos=0.9713   ~16 min
RUNNER=native   max_abs_err=7                    cos=0.9757   ~12 s host
```

316 dispatches run to completion on spike_riscv64, on the same IR in both
cases (for the spike build the `inspect_tensors` list is stripped from
`graph.json` first -- the dumps go out over HTIF at roughly 100 lines/s, so
the native ladder's 1.8 M lines would take hours; stripping them changes no
arithmetic).

Per-tensor tracking against the PyTorch fp32 capture, input to output:

```
conv2d_4      cos 0.9935     matmul_1       cos 0.9672
add_5             0.9934     add_46             0.9865
cat_4             0.9934     layer_norm_32      0.9822
layer_norm_8      0.9940     select             0.9730
                             linear_84 (out)    0.9756
```

**The int8 golden is the QUANTIZED FP32 REFERENCE**, not a simulation of the
int8 graph (`extract_graph_export.py` io.npz emit:
`_quantize_per_tensor_sym(captured_fp32, scale)`). `max_abs_err` therefore
measures the whole network's quantization error in LSBs of the output scale,
and a zero-tolerance PASS is not reachable for a real quantized model. Read
the number, not the verdict line.

`RUNNER=native` (`native_sim/native/64`) is what made this affordable: the
whole 690-token model runs in ~12 s instead of spike's ~15 min, with the same
in-binary `MODELBLASTER_VERIFY` compare. Two fixes were needed before that
board could build at all -- `fence rw, rw` in the harness and `rdcycle` in the
generated `model.c` are not x86 instructions -- so `RUNNER=native` had never
actually built anything.

### 13.1 Method

`--inspect <tensors>` dumps chosen intermediates from the running binary with
their scale, and writes `inspect_ref.npz` with the PyTorch fp32 values at the
same tensors. Walking a ladder of checkpoints and reading the first place the
cosine collapses localizes a bug to one op.
`experiments/octo_port/onchip/inspect_compare.py` does the comparison;
`xtarget_compare.py` diffs two runs of the same IR against each other.

### 13.2 Lowering bugs (8)

Every one was silent: the build succeeded, no kernel was reported missing,
and the answer was wrong.

| # | what was wrong | localized by |
|---|---|---|
| 1 | `transpose`/`permute` aliased even when it reorders elements (a ViT head split, a stem NCHW->NHWC) | `linear_1` cos 0.0196 -> 0.9991 |
| 2 | `matmul_s8` has no batch loop, so 6 heads collapsed to one head's work over the wrong elements | -- |
| 3 | K-transposes materialized (12 x 2.9 MB) instead of using the kernel's `transpose_b` | memory only |
| 4 | `layer_norm` took K from the *aliased buffer's* last dim, so GroupNormLN normalized 128 elements instead of its 16384-element group | M=8192,K=128 -> M=64,K=16384 |
| 5 | channel-broadcast `mul`/`add` hidden behind a rank-3 view, lowered to a flat elementwise op walking C*H*W elements of a length-C weight | first stem norm all-zero |
| 6 | the shared mask `(1,1,S,S)` added to `(1,heads,S,S)` scores by flat index | -- |
| 7 | `expand_as` and narrowing `slice`/`select` were aliases despite changing the element count | `cat4_c1_s8` read 33 MB past a 98 KB buffer (segfault) |
| 8 | a rank-4 `cat` assumed axis 1 whatever the real axis was | segfault |

and two that were not shape bugs:

* `generate_skeleton.ptr_for` had no case for a **weight consumed as a plain
  positional input** -- a GroupNorm gamma into `mul_c1_s8`, a positional
  embedding into `slice_c_s8`, the mask into `add_tile_s8`. It fell through to
  an intermediate scratch buffer, which is zeroed, so those ops read all zeros.
* `softmax_s8`'s output scale was hardcoded to `1/127`. Over 690 keys a typical
  attention weight is ~1e-3, i.e. **below half an LSB** -- whole attention rows
  quantized to zero. Using the calibrated scale instead was the single largest
  fix of the set.

New kernels this needed: `permute4_s8`, `matmul_b_s8`, `add_tile_s8`,
`add_c1_s8`, all four bit-exact against numpy
(`pipeline/tests/test_broadcast_and_permute_kernels.py`). The alias-vs-copy
predicates are pinned in `pipeline/tests/test_export_alias_decisions.py`.

**Correction to §11.** It says the readout mean-pool "lowers to a `select`,
which is a free alias". It does not: `x[:, :, -1, :]` takes 1 token out of 337
along axis 2, which is a strided gather. It is a `slice_c_s8` copy now.

### 13.3 Quantization bugs in the port itself (2)

Found by linting the extracted graph for ops whose output scale is much
coarser than an input scale.

* **`NORM=0` was feeding the model raw [0,255].** The knob moves `/127.5 - 1`
  out of the graph, which makes pre-normalized values the input contract, but
  `get_sample_input` returned [0,255] regardless. So the stem ran 128x
  overdriven, and `goal_const` (= `normalize_images(0)` = -1.0) sat in a
  different domain from the image it is concatenated with: that cat's shared
  scale came out 2.008 = 255/127 and the -1.0 goal channels quantized to a
  single LSB. Every activation scale downstream was fitted to that.
* **The score net's `Linear(concat([cond(32), obs(384), action(28)]))`** forces
  one per-tensor scale on three blocks whose measured scales are 0.010857 /
  0.085178 / 0.0053646, so `noisy_actions` -- the variable the diffusion loop
  is denoising -- reached the matmul with 8 of its 127 levels.
  `MODELBLASTER_OCTO_SPLITFC=1` replaces it with three summed Linears
  (identical arithmetic, measured 3.1e-06 / 5.0e-07 against the fused form in
  fp32, same order as the other knobs in §12).

### 13.4 Calibration is the largest single lever, and it is fragile

`get_calibration_spec` has named a `bridge_episodes` loader since the port
landed, but **no such loader was registered**, so every extraction fell back
to one `torch.randint` sample -- and the spec's `rolling_window` composer
would have produced `(B, window*3, R, R)` instead of `(B, window, 3, R, R)`,
the same element count tokenized wrongly. Both are fixed
(`mb_datasets/bridge_episodes.py`, `synthetic.py`, `window_stack`).

Note what calibrating on one sample was hiding: it was the SAME sample the
golden is computed from, so the reported error was optimistic. Moving to 8
real frames with a held-in-distribution test frame took the output cosine
from 0.9585 to 0.8575 before clipping. The clipping sweep, on those 8 frames:

```
max-abs (no clip)  cos 0.8575   max_abs_err 22
99.999                 0.9085                14
99.99                  0.9757                 7
99.95                  0.9354                18
99.9                   0.9254                19
99.0                   0.7332                95
```

99.99 is **empirically tuned on this calibration set, not a principled
quantile**: the extractor estimates the percentile from at most 2048 elements
per tensor per sample, so for the 2.8 M-element attention tensors it is closer
to "max of a 16 k draw" than to a real 99.99th percentile -- which is why the
neighbours are not monotonic. A 3.5x spread in `max_abs_err` across adjacent
settings is itself the finding: per-tensor int8 PTQ on this model is fragile,
and re-tuning is required if the calibration set changes.

### 13.5 Spike and native are not bit-comparable, and the reason is `expf`

Same IR, same reference kernels, and the outputs differ. Neither target is
broken. Localized with `xtarget_compare.py`:

```
native vs spike, same IR:  linear  add  cat_2  slice_4    0.00% differ
                           layer_norm_33                 91.80%, max 18 lsb
                           final output                  69.64%, max 24 lsb
```

Ops 0..56 -- the language branch and BOTH image stems, including 10 convs, 8
`layer_norm_s8`, 8 `mul_c1_s8`, 8 `add_c1_s8`, 3 `add_s8`, 2 `permute4_s8`, 4
`slice_c_s8` -- are **bit-identical across targets**. The first op that calls
`expf` is #68 (`softmax`), and divergence starts there. `expf` is the only
operation in these kernels that is not exactly specified: `sqrtf`, `roundf`
and float `+ - * /` are all correctly rounded and contribute nothing.

FP contraction was the other candidate and is NOT the cause. gcc's default
`-ffp-contract=fast` does fuse the multiply-accumulates in the kernels that
dequantize to float, asymmetrically -- counted in the disassembly of one
`kernels.c`, RISC-V gets 5 fused ops in `layer_norm_s8` and 1 each in
`add_s8` / `add_c1_s8` / `add_tile_s8`, x86-64 gets none -- and driving
`add_s8` directly with the model's own scale triples moves up to 0.785% of
its int8 outputs by 1 LSB. But the bit-identical prefix above contains 51 of
those fused ops and is bit-identical anyway, and rebuilding native with
`-mfma -ffp-contract=fast` (new `EXTRA_KERNEL_CFLAGS` knob) only moves
`max_abs_err` 26 -> 20 against spike's 13. So contraction is a real
asymmetry that in practice does not flip int8 outputs here.

The rule: for an int8 graph whose kernels dequantize to float, do not compare
two targets bit-for-bit. Compare each against the fp32 reference with a
tolerance. `EXTRA_KERNEL_CFLAGS='-ffp-contract=off'` on both targets removes
the contraction half if a stricter comparison is ever wanted.

### 13.6 Independent checks of the integer path

Two numpy replicas, both reproducing the device dump **bit-for-bit**, so the
IR's own numbers are confirmed against something that is not the pipeline:

* `linear_s8_pc` at op 0 -- int32 accumulate, then the Q0.31
  `(acc*mult + 2^30) >> 31` requantize with the per-channel multiplier and
  shift read straight out of `weights.npz`. Confirms the extracted weights,
  multipliers and shifts are what the kernel consumes.
* the `slice_5` -> `select` chain -- `[16:690]` of the 690-token sequence,
  then token 336 of 337 per timestep, replicated from the dumped
  `layer_norm_32` values through both requantizes. This is the pair the old
  alias treatment got wrong (a non-zero-start slice and a `select`
  reinterpreted onto axis 1 of `N=2, IC=337`), so it is worth pinning
  directly rather than only through the end-to-end cosine.

### 13.7 The deployment decomposition also extracts

`MODELBLASTER_OCTO_PART` splits the graph the way a scheduler would run it
(backbone once, score 20x). Both halves extract with 0 pending kernels:

```
PART=full      316 ops
PART=backbone  290 ops
PART=score      23 ops   inputs: obs_enc(768) + noisy_actions(56) + t(64)
```

Op mix of the full graph: `linear_s8_pc` 85, `permute4_s8` 50,
`layer_norm_s8` 36, `add_s8` 32, `matmul_b_s8` 24, `add_tile_s8` 12,
`softmax_s8` 12, `gelu_s8` 12, `conv2d_s8_pc` 10, `mul_c1_s8` 8,
`add_c1_s8` 8, `relu_s8` 8, `slice_c_s8` 6, `cat2_c1_s8` 4, `sigmoid_s8` 4,
`mul_s8` 4, `cat4_c1_s8` 1.

### 13.8 Known gaps

* `PART=score` has no calibration source -- its inputs are the backbone's
  output plus the diffusion state. `get_calibration_spec` returns None for it
  rather than calibrating on noise; it should be calibrated from a
  `PART=full` run's captured `obs_enc`.
* `img_wrist` is calibrated on the primary camera's frames: these episodes are
  single-camera, and the loader records the substitution in the item meta.
* Intermediate buffers are **240 MiB** with no liveness reuse -- every tensor
  gets its own static array (510 of them in `buffers.c`). 2.1 MiB of that is
  constant tensors that also get a redundant scratch buffer nothing reads.
  The spike build fits the 256 MB ram0 at 82%.
* Where the time goes, from the spike profile (shares are the reliable part;
  the absolute numbers are spike's cycle model, not silicon, and note that
  the per-op `rdcycle` deltas sum to 331 G while `WALL_CYCLES` reads
  3.31 G -- exactly 100x apart, because the two use different clocks):

  ```
  linear_s8_pc    85 dispatches   67.3%
  matmul_b_s8     24              17.9%
  conv2d_s8_pc    10              11.3%
  softmax_s8      12               2.3%
  gelu_s8         12               0.5%
  everything else                 <1%   (permute4_s8, 50 dispatches: 0.04%)
  ```

  96.5% is in linear / matmul / conv, which is where the curated RVV and
  Gemmini kernels already are -- `conv2d_s8_pc` and `linear_s8_pc` have them,
  `matmul_b_s8` does not. The 50 `permute4_s8` copies cost 0.04% of the time,
  so removing them (a strided-A `matmul_b_s8` could read the head split in
  place) is a ~9.5 MB memory win, not a speed one.
* `MODELBLASTER_OCTO_ATTN=sdpa` now **refuses to extract** rather than
  silently dropping the mask, exactly as §11 predicted. `ATTN=matmul` is the
  only route, as §11 recommended.
* No FPGA run yet.

## 14. fp16 on vectorized RVV

`QUANT=fp16 TARGET=rvv` lowers the port onto the `rvv_f16` backend
(`-march=rv64gcv_zfh_zvfh`, spike `--isa=rv64gcv_zicntr_zfh_zvfh`).
**328 op records, 0 pending kernels, 18 of the 19 op kinds on curated
vectorized RVV+Zvfh kernels** -- only `cat4_c1_f16` (1 dispatch) is still
the scalar reference.

### 14.1 fp16 is a much better fit for this model than int8

```
                    max_abs_err   max_rel_err   cos vs reference   verdict
fp16 (native)         0.00537        0.0877         0.999998        PASS
int8 (native)         7 of 127       --             0.9756          FAIL*
```

`*` and the int8 FAIL is structural, not a defect: the int8 golden this
extractor emits IS the quantized fp32 reference, so a zero-tolerance pass
needs the int8 graph to reproduce fp32 exactly (§13). The fp16 golden is
the model's own `.half()` forward, so its PASS is a real gate -- and the
output error is ~60x tighter than int8's on the same outputs.

That is worth stating plainly after §13.4: per-tensor int8 PTQ on this
model needed a real calibration set AND a tuned activation clip to reach
cos 0.976, and was fragile to both. fp16 needs neither and lands at
0.999998.

### 14.2 What was already there, and what was missing

The `rvv_f16` backend and 42 curated kernels predate this work (`47fd721`
"25 curated kernels closing the gemmini_q31 and rvv_f16 coverage gaps",
`bc25c91` "Merge ModelBlaster's kernelbench/fp16 line", `95d7be7` "rvv:
the AVL a vsetvl is given, and two kernels that were silently wrong"),
including vectorized `linear_f16` and `conv2d_f16`. Missing on every ref,
checked post-fetch against `origin/main` and every branch:

* `bmm_tb_f16` did not exist as an OP. The existing family was
  {matmul, matmul_tb} x {., bmm} with no batched-transposed member, so a
  multi-head Q @ K.T had to materialize the transpose.
* `permute4_f16`, `add_tile_f16`, `add_c1_f16`, `mul_scalar_f16` did not
  exist either -- the fp16 counterparts of the s8 ops §13 added.
* `layer_norm_f16`, `softmax_f16`, `gelu_f16`, `bmm_f16` existed as
  REFERENCE only, with no curated kernel on any target.

### 14.3 Measured, all-reference vs all-curated

Same IR, same model (`LAYERS=1 WINDOW=1` so the reference baseline
finishes), spike per-op cycles. Both runs **PASS** against the PyTorch
fp16 golden (reference `max_abs_err=0.00406`, curated `0.00748`).

```
op                  n    reference     curated   speedup
conv2d_f16         10     16545.6M      280.7M     59.0x
linear_f16         19      6963.9M      169.8M     41.0x
bmm_tb_f16          1       441.1M       54.6M      8.1x   <- new op + kernel
bmm_f16             1       432.3M       24.1M     17.9x   <- new kernel
softmax_f16         1        83.3M        2.4M     34.8x   <- new kernel
gelu_f16            1        77.6M        1.5M     50.9x   <- new kernel
layer_norm_f16     14        51.4M        4.2M     12.3x   <- new kernel
relu_f16            8        19.0M        0.3M     72.8x
add_c1_f16          8        13.7M        0.8M     17.7x   <- new op + kernel
mul_c1_f16          8        13.7M        0.3M     46.0x
add_tile_f16        1         9.0M        0.4M     25.6x   <- new op + kernel
mul_scalar_f16      1         9.0M        0.4M     25.6x   <- new op + kernel
cat2_c1_f16         3         7.5M        0.1M     66.1x
permute4_f16        6         5.0M        0.3M     15.9x   <- new op + kernel
add_f16            10         4.4M        0.1M     54.1x
slice_c_f16         6         3.0M        0.1M     60.8x
cat4_c1_f16         1         1.6M        1.6M      1.0x   (still reference)
sigmoid_f16         4         0.3M        0.0M     34.9x
mul_f16             4         0.0M        0.0M     50.4x
TOTAL                     24681.4M      541.4M     45.6x
```

`WALL_CYCLES` independently reads 246.81M vs 5.41M -- also 45.6x. The two
counters differ by exactly 100x because they use different clocks (the
per-op numbers are `rdcycle` deltas, `WALL_CYCLES` is `k_cycle_get_64`);
the RATIO is what either one is good for. Same trap as §13.8.

**Amdahl drove the work order, and it is worth recording.** With only
`linear_f16` and `conv2d_f16` curated, the remaining scalar ops were 29%
of the model. Vectorizing the two big GEMMs promoted everything else, so
the second tier (softmax 10%, gelu 9%, layer_norm 6%) mattered far more
than the int8 profile in §13.8 suggested -- there softmax was 2.3%.

### 14.4 One judgement call I got wrong first

I first shipped softmax with only passes 1 and 3 vectorized, leaving the
`expf` pass scalar, on the grounds that a polynomial exp would put an
approximation in the op feeding every attention weight. It measured
**1.15x**, i.e. it left 72.6 of 83.3 Mcycles on the table and made softmax
the largest remaining scalar op in the model.

The caution did not survive contact with the numbers. **The reference
already rounds every exp to fp16 on the way out** (`output[k] =
(_Float16)e`, ~5e-4 relative), so the vector exp's 1.6e-6 fp32 error is
three orders of magnitude below error the reference itself introduces.
Measured over 200 random rows of K=690: fp16 output max|d| 7.6e-06,
row-sum max rel err 6.2e-07, 0.04% of elements one fp16 ulp apart, against
an fp16 verify atol of 1e-2. Vectorizing the exp took softmax to **34.8x**.

The same vector exp is what makes `gelu_f16` 50.9x, via
`tanh(a) = 1 - 2/(exp(2a)+1)`. Both kernels disassemble to zero libm
calls. Frequent 1-ulp fp16 differences are expected there and are not a
defect: `expf`/`tanhf` are not correctly rounded, so the reference's own
last bits move with the libm -- which is exactly why §13.5's spike and
native builds are not bit-identical.

Two kernels are deliberately exact rather than fast: softmax's row max is
`vfredmax` plus ONE scalar multiply (monotonic for `input_scale >= 0`, so
the max of the scaled values IS the scaled max), and `permute4_f16` cannot
change a value at all.

### 14.5 Two silent-fallback bugs found on the way

Both made a run labelled `rvv_f16` actually measure scalar code, and
neither said so.

* **ram0 was sized from the baked io.** `io * 3 + 128 MB` is a good proxy
  only when activations are small relative to the io; this model inverts
  it, ~1 MB of io against 479 MB of fp16 intermediates. The derived size
  came out UNDER the stock 256 MB, so no overlay was written and the link
  failed with `region 'RAM' overflowed by 244299616 bytes`. On the
  curated-verify path that link error is caught and reported as the KERNEL
  failing, so **all 18 curated picks silently became reference**.
  `generate_skeleton` now writes `footprint.json` with the .bss it
  declared, `_run_lib.sh` sizes from io + buffers + weights (704 MiB
  here), and the verify builds get the overlay through
  `MODELBLASTER_EXTRA_CMAKE_ARGS`.
* **`_check_conv_family_layout_agreement` made `rvv_f16` unbuildable.** It
  compared `conv2d_s8`'s weight layout ('ihwoc') against `conv2d_f16`'s
  ('oihw') and exited -- a combination its own SCOPE note calls legitimate.
  Retired; see the commit for why nothing is left for it to protect. The
  same guard had also made `rvv_x60`'s packing untestable (4 pre-existing
  test errors, now passing). Related: a 4-D tensor the IR claims for
  nothing is no longer conv-packed -- rank 4 is not the same question as
  "is a conv filter", and this model's mask and positional tables are
  rank 4.

Also: `MB_DRIFT_ATOL` is now int8-only in `run.sh`. It is measured in int8
LSBs and it is the one knob that LOOSENS a verify gate; under `QUANT=fp16`
it was taking the curated verify atol from 0.01 to **2.0** on outputs
spanning +/-2.

### 14.6 Status and gaps

* All 18 curated kernels PASS their per-kernel verify against the
  reference at the spec shapes (0 FAILs).
* Reduced-size (`LAYERS=1 WINDOW=1`) fp16 on spike with all 18: PASS,
  `max_abs_err=0.00748`, 5.41M wall cycles.
* Full-size (690-token, 12-layer) fp16 on spike with all 18: **PASS**,
  `max_abs_err=0.00732`, 7.99 G rdcycles over 328 dispatches.
* `cat4_c1_f16` now has a curated kernel too (19/19), added after the
  measurements below, so it is not in them.
* No FPGA run at fp16.

### 14.7 Full-size profile: it is now 97% GEMM

```
op                 Gcycles   share
linear_f16           3.618   45.3%
bmm_tb_f16           2.503   31.3%
bmm_f16              1.101   13.8%
conv2d_f16           0.561    7.0%
softmax_f16          0.105    1.3%
gelu_f16             0.036    0.4%
layer_norm_f16       0.025    0.3%
everything else     <0.05    <0.3% each
TOTAL                7.99 G
```

Two things worth taking from this.

**The second tier is gone.** softmax, gelu and layer_norm were 25% of the
model when only linear and conv were curated (14.3); they are now 2.0%
combined. The seven kernels §14.4 describes did their job and are no
longer where the time is.

**`bmm_tb_f16` is the top remaining target, and its 8.1x is the lowest of
the new kernels -- those two facts are the same fact.** At 31.3% it is the
second-largest cost in the model, and its K-reduction form reduces over
K = head_dim = 64, which at f16m2 on VLEN=256 is only two vector
iterations before a `vfredusum`. The reduction overhead is amortized over
almost nothing, unlike `linear_f16` (41x) whose K is 384 or 1536. An
N-lanes or register-tiled form that keeps several output columns live per
pass would amortize it properly; that is the measurement to take next,
not a guess.

Note also that conv2d is only 7.0% at full size against 52% on the
reduced model -- the stem runs once while the transformer runs twelve
times, so `LAYERS=1 WINDOW=1` flatters conv and the A/B in 14.3 should be
read per-op, not as a whole-model figure for the real model.
