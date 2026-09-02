# Tensor layout in the ModelBlaster IR — a design study

**Status:** design study. Nothing here is implemented. No source file was modified to write it.
**Scope:** how tensor LAYOUT (NCHW / NHWC, and blocked variants) should be represented in the
ModelBlaster IR so that it *cascades* between ops instead of every kernel converting on entry and exit.
**Repo root for all citations:** `/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/`
(paths below are written `modelblaster/...` relative to that root unless marked otherwise).

---

## 1. The problem, in measured cycles

The gemmini `conv2d_s8` kernel was instrumented with `rdcycle` brackets around its three phases and run
on AWS F2 (all-gemmini dronet int8, unsplit). The instrumentation is
`MB_GEM_PHASE_TRACE` / `MB_GEM_PH_EMIT` at
`modelblaster/kernels/gemmini/gemmini_conv2d_s8_gemmini_tiled_conv.c:113-137`; the raw log is
`RoSE/experiments/shard_dim/results/oh/res_ph0/uartlog`, lines beginning `MB_GEMPHASE,`
(fields: `hart, IC, IH, IW, OC, in_transpose_cycles, gemmini_cycles, out_transpose_cycles`).

Ten conv dispatches, summed:

| phase | cycles | share |
|---|---:|---:|
| input transpose NCHW→NHWC | 852,788 | **34.3%** |
| gemmini itself | 318,487 | **12.8%** |
| output transpose NHWC→NCHW | 1,317,580 | **52.9%** |
| total | 2,488,855 | 100% |

**87.2% of gemmini conv time is layout conversion.** These are the only measured numbers in this
document; everything else below that carries a figure is arithmetic on this table and is labelled as such.

### 1.1 Where the 3.53× comes from

If NHWC were chained through the conv section, only two conversions survive: the model input, and the
return to NCHW before `flatten`/`linear`. Reconstructing that from the table:

```
kept = gemmini(318,487) + conv0 input transpose(364,810) + one tail conversion(~21,800)
     = 705,097 cycles
2,488,855 / 705,097 = 3.530x        removed = 71.7% of conv cycles
```

The `~21,800` is the measured cost of an output transpose at the tail geometry (`OC=128, OHW=16`:
21,716 and 21,877 cycles in the log). The reconstruction lands on the stated 3.53× to three significant
figures, which is a useful consistency check but *not* an independent measurement.

Two facts from the same table shape everything that follows:

- **conv0 dominates.** Its output transpose alone (971,937) is **39.1%** of all conv cycles — the kernel's
  own comment at `gemmini_conv2d_s8_gemmini_tiled_conv.c:538-541` says the same. Its *input* transpose
  (364,810) is **51.7% of the entire surviving budget** after chaining. Chaining NHWC is worth 3.53×; the
  next lever after that is the model input surface, not more chaining.
- **The transposes are memory-bound at a stable rate.** Dividing cycles by bytes moved:
  conv0's byte-at-a-time path runs at 9.7 cycles/byte on both sides; the `IC%4==0` packed input path runs
  at **6.0–6.7 cycles/byte**; output transposes run at **9.8–10.6 cycles/byte** on tensors ≥1.5 KB and
  degrade to 30–43 cycles/byte on the 512 B–1.5 KB tail layers (fixed loop/cold-miss overhead over a
  512 KB workspace). *(Derived, not measured directly.)* A cost model for an explicit relayout op has a
  defensible constant to start from: **~10 cycles/byte, plus a fixed term of roughly 20 kcycles.**

### 1.2 The measurement understates the prize

The phase log brackets only `conv2d_s8`. The gemmini maxpool kernel transposes as well —
`modelblaster/kernels/gemmini_q31/gemmini_q31_maxpool2d_s8_gemmini_tiled_conv_pool.c:164` (`NCHW -> NHWC
into ws_input`) and `:198` (`NHWC -> NCHW`). dronet has one maxpool inside the proposed island, and its
two transposes are *not* in the 2.49 M total. So 3.53× on conv cycles is a floor for the island, not a ceiling.

---

## 2. How layout is represented today

### 2.1 Weights: already a modelled, machine-readable, codegen-time concept

This is the precedent the study was asked to take seriously, and it is stronger than "a transform exists".
There is a whole three-part contract:

1. **Declaration, per algorithm.** `AlgorithmCandidate.weight_layout: str = "oihw"` —
   `modelblaster/pipeline/reference_kernels.py:102-108`. Legal values `"oihw"` (PyTorch `[OC,IC,KH,KW]`),
   `"ihwoc"` (`[IC,KH,KW,OC]`, for vectorising over OC), `"hwio"` (`[KH,KW,IC,OC]`, for gemmini's
   `tiled_conv_auto`). ~20 declaration sites.
2. **Resolution, per op × backend, with hard failure on disagreement.**
   `_conv_weight_layout_for_op` (`generate_skeleton.py:1122-1168`) resolves in three steps: target-affined
   algorithms win; else the op follows the backend-wide `conv2d_s8` layout if its C branches on a pack
   macro (`_op_follows_backend_conv_layout`, `:1116-1120`); else OIHW. The backend-wide fallback is
   `_conv_weight_layout_for_backend` (`:1003-1042`), which consults `backend_lineage` so a variant such as
   `rvv_x60` inherits `rvv`'s answer — the comment at `:1023-1029` records that not doing so cost
   `max_abs_err=57` "with no error anywhere in the build". Cross-op conflicts raise `SystemExit`
   (`_check_conv_family_layout_agreement`, `:1046-1103`).
3. **Application, at codegen, once, to a constant array.** `_backend_pack_weight`
   (`generate_skeleton.py:1193-1231`) applies `_LAYOUT_PERMUTATION = {"hwio": (2,3,1,0), "ihwoc":
   (1,2,3,0)}` (`:1182-1190`) via `np.transpose` and emits the permuted literal into `weights.c`
   (`:1904-1918`). Runtime cost: **zero**. The kernel's matching read is selected by a compile define in
   `Backend.kernel_cflags` — `-DMODELBLASTER_GEMMINI_HWIO_WEIGHTS=1` at
   `modelblaster/pipeline/backends.py:419`, `-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1` at `:119`.
4. **A gate that refuses a mismatch rather than tolerating it.**
   `_assert_curated_layout_contract` (`modelblaster/pipeline/generate_kernels.py:52-94`), called before any
   numeric check at `:1056` and `:1694`. Its docstring is the governing principle for this whole area:
   *"Verify cannot distinguish 'this kernel is buggy' from 'this kernel was given weights in a layout it
   never agreed to'"* and *"Do NOT loosen the curated verify tolerance: the fallback it triggers is
   numerically correct and therefore invisible end-to-end."*

The design principle embodied here is worth stating explicitly, because it is not the obvious one:
**the permutation is never applied at compare time. It is applied once at codegen, and both sides of every
comparison are then made to physically agree.** There is no "compare in layout A then permute" anywhere in
the tree.

### 2.2 Activations: prose in `semantics`, plus load-bearing assumptions scattered across the pipeline

Layout for activations exists only as English. `CONV2D_S8.semantics` says
`"Layout (NCHW for activations, OIHW for weights)"` — `reference_kernels.py:1920`;
`MAXPOOL2D_S8` says `"Layout (NCHW)"` at `:3527`. There are ~40 such prose statements in
`reference_kernels.py`, 11 in `generate_skeleton.py`, 8 in `extract_graph.py`, 5 in `apply_split_hint.py`.

The IR carries none of it. A tensor record is exactly:

```json
"conv_modules_0": {"shape": [1, 32, 56, 56], "dtype": "i8",
                   "quant": {"scale": 0.0305884823, "zero_point": 0}}
```

(`modelblaster/examples/dronet/int8/generated/graph.json`). `shape` is an *ordered list* for tensors but a
*named dict* for ops (`{"N":1,"IC":3,"IH":112,...}`), so op geometry cannot express axis order at all.
NCHW is implied by convention and by every consumer independently re-deriving it.

Where that convention is load-bearing in code, not comments:

- `generate_skeleton.py:2199-2205` — an OC split tile's output alias is `elem_offset = oc0 * OH * OW`,
  correct only because "[N, OC, OH, OW] is NCHW".
- `generate_skeleton.py:2250` — the per-tensor alias pass, same reasoning.
- `generate_skeleton.py:596` (`parallel_cbs_shard_fn`) — the intra-op OC shard writes
  `c->out + oc0 * OH * OW`; `:592-593` "The output is NCHW, so an OC slice is a contiguous run of planes."
  Same at `:696` and `:779`.
- `apply_split_hint.py:507-515` — `_register_tile_tensors` maps split axis → tensor dim with a hardcoded
  `axis_dim = 1` for OC and `2` for OH, i.e. NCHW positions.
- `apply_split_hint.py:241` and `generate_skeleton.py:1395-1400` — the entire justification for
  `_OH_TILE_HELPER` existing: "in NCHW neither end of a row band is contiguous."
- The two transposes inside every gemmini conv (`gemmini_conv2d_s8_gemmini_tiled_conv.c:338` and `:530`)
  and every gemmini maxpool — the thing this document is about.

### 2.3 The asymmetry that makes activations genuinely harder than weights

Three facts, all verified, that together define the design space:

**(a) `model.c` is emitted per backend over the *whole* graph; the runtime picks.**
`generate()` takes a single `--backend` string (`generate_skeleton.py:4327-4340`) and writes one `model.c`
into one out-dir; callers loop (`modelblaster/scripts/run_xpurt_k1.sh:284-330`). Four such files exist for
dronet today (`examples/dronet/int8/generated/{scalar,rvv,gemmini_q31,gemmini_q31_rvv}/model.c`). Each
contains a dispatch table covering **every** dispatch (`generate_skeleton.py:4104-4114`), with uncovered ops
falling back to the scalar `reference_impl` — `KernelSpec.__post_init__` guarantees the algorithm queue is
never empty (`reference_kernels.py:136-160`), so no op is ever *unavailable* on a backend, only slow.
Selection happens at run time by string compare on the schedule entry:
`generate_xpurt_main.py:310-320` emits `else if (strcmp(e_->impl, "<kind>") == 0) {
MODEL_<UMID>_DISPATCH_FNS_<BS>[e_->dispatch_id](...); }`.

So: **codegen does not and cannot know which backend a given dispatch will run on.** The IR's
`hardware_target` field is `"any"` on all 5,190 ops across every `examples/*/int8/generated/graph.json` in
the tree; placement lives in a separate XPU-RT schedule JSON, per dispatch
(`ingest_xpurt_schedule.py:1-36`, `:554-567`), solved offline by a MOSEK MIP over measured per-backend
cycles (`scripts/run_xpurt_scheduler_multi.py:136-137`, `:192-202`).

**(b) Weights get a per-backend copy. Activations do not.**
`_weight_name` suffixes every weight symbol with the backend (`generate_skeleton.py:977-1001`) precisely
so two backends can hold the same logical tensor in two physical layouts; the docstring records the
`max_abs_err=51` wrong-answer bug from before the suffix existed. `weights.c` is compiled per backend
(`modelblaster/harness_xpurt/CMakeLists.txt:192-211`).

Intermediate activation buffers are the opposite. `buffers.c` is compiled **exactly once per model, not
per backend** (`generate_skeleton.py:4155-4172`, `CMakeLists.txt:213-231`), with unsuffixed external
symbols, because — `_buf_name` docstring, `generate_skeleton.py:112-123` — "file-static buffers would give
each backend its own private copy and cross-backend dispatches would read zeroed scratch."

**(c) A layout disagreement on an activation is size-identical, and therefore silent.**
Buffers are sized `_prod(tensors[t]["shape"])` (`generate_skeleton.py:2404-2407`). NCHW and NHWC of the
same tensor have identical element counts. There is no link error, no size assert, no runtime tag — just
wrong numbers. This is the same failure class as the `max_abs_err=51` and `max_abs_err=57` bugs already
documented in the tree, and it is why any design here needs a hard codegen gate rather than a convention.

**Summary of the asymmetry.** A weight is read-only, constant, and duplicable per backend at zero runtime
cost. An activation is a single shared runtime buffer, written by one dispatch and read by others that may
land on different backends, whose size does not distinguish layouts. *Layout for activations must be a
per-tensor, backend-independent decision, and every backend must be able to honour it.*

---

## 3. What the island actually looks like (dronet)

From `examples/dronet/int8/generated/graph.json`, the op chain is:

```
d0  conv2d_s8      x            -> conv_modules_0
d1  maxpool2d_s8   conv_modules_0 -> maxpool1
d2  batchnorm2d_s8 maxpool1     -> relu_modules_0
d3..d16  conv/conv/conv/add, bn, conv/conv/conv/add, bn, conv/conv/conv/add   (9 convs, 3 adds, 2 bns)
--  view (flatten) add_2        -> flatten                (zero-cost alias, generate_skeleton.py:2161-2162)
d17 relu_s8        flatten      -> relu_modules_6
--  view (dropout) -> dropout1
d18 linear_s8      dropout1     -> linear1
d19 linear_s8      dropout1     -> linear2
d20 sigmoid_s8
```

Op-by-op feasibility (taken as input per the brief, spot-checked where cheap):

| op | count | signature evidence | verdict |
|---|---:|---|---|
| `add_s8` | 3 | `(a, b, output, int n, ...)` — flat `n`, no shape dims — `reference_kernels.py:3685-3689` | **layout-polymorphic by signature.** No change at all. |
| `relu_s8` | 1 | `(input, output, int n)` — `:1829` | same |
| `batchnorm2d_s8` | 3 | `(input, scale, bias, output, N, C, H, W, ...)`, indexes `((n*C+c)*H+h)*W+w` — `:3860-3862` | needs an NHWC variant. Per-channel affine with C innermost is *more* vector-friendly (one `vle8` of C contiguous channels, one `vle32` of the scale vector). |
| `maxpool2d_s8` | 1 | `(input, output, N, C, IH, IW, ...)`, `"Layout (NCHW)"` — `:3523-3536` | needs a variant; the gemmini implementation is already NHWC internally (`gemmini_q31_maxpool2d_s8_gemmini_tiled_conv_pool.c:164`, `:198`) so its variant is a *deletion* of two transposes, not new code. |
| `conv2d_s8` | 10 | the subject | needs an NHWC entry point; the gemmini one is a deletion of two transposes. |
| `linear_s8` | 2 | `(input, weight, bias, output, M, K, N, ...)` — `[M,K] x [N,K]` — `:1409-1423` | the island boundary. See §3.1. |
| `view` ×2 | — | codegen alias, no runtime op | see §3.1 |

The `add_s8`/`relu_s8` observation generalises usefully: **the op set already partitions itself into
layout-polymorphic ops (flat-`n` signature) and layout-bound ops (`N,C,H,W` signature).** A machine-readable
layout attribute can be *derived* for most of the op table by inspecting `KernelSpec.signature` rather
than hand-annotated, which shrinks the annotation burden a lot.

### 3.1 The tail: a `flatten` under NHWC, and a free way out

`view` is a pure codegen alias — `generate_skeleton.py:2161-2162` maps the output name onto the input's
buffer; no bytes move, and `_zero_cost_ops` at `:2270` keeps it out of the profile. That stays true under
NHWC. What changes is the *element order* the following `linear_s8` sees: `[H,W,C]` instead of `[C,H,W]`.

Two ways to close it:

- **(a) A relayout before the flatten.** One conversion on a `[1,128,4,4]` tensor — the ~21,800 cycles in
  the 3.53× reconstruction. Simple, obviously correct.
- **(b) Permute the linear weights' K axis at codegen.** `linear_s8` weight is `[N, K]` with
  `K = C*H*W`; permuting K from `(C,H,W)` to `(H,W,C)` order on a constant array makes the NHWC flat
  vector read correctly at **zero runtime cost** — exactly the `_backend_pack_weight` trick, extended
  from a filter to a K-axis. Both dronet linears consume the same tensor, so one permutation serves both.
  This would take the island to `318,487 + 364,810 = 683,297` cycles, i.e. **3.64×** (arithmetic on §1).

Option (b) is the design-coherent one — it says the same thing about layout that the weight machinery
already says — but it is *not* free to implement: `_backend_pack_weight` early-returns on anything that is
not 4-D (`generate_skeleton.py:1201-1202`), so a K-axis permutation needs a new mechanism keyed on the
*consumed tensor's* layout rather than on the weight's rank. It is also worth checking (I have not) that a
K-permutation composes safely with the linear N-split, which slices weight *rows*
(`_n_tile_operands`, `generate_skeleton.py:162-176`) and should be orthogonal. Treat (b) as a
stage-3 refinement worth ~3% and a lot of conceptual tidiness, not as a stage-1 requirement.

### 3.2 The head: the model input is half the remaining budget

conv0's input transpose is 364,810 cycles = 51.7% of the post-chaining budget. `x` is the model *input*,
which is a function argument, not a `buffers.c` buffer (`generate_skeleton.py:2397-2399` excludes inputs),
so its layout is an interface contract with the harness, not an internal decision.

**Speculation, unverified:** on RoSE the camera frame arrives over the bridge DMA, so the delivered layout
is a choice rather than a given, and a camera naturally produces interleaved (HWC) pixels. If the sensor
path delivered HWC, the island cost would be `318,487 + ~21,800` and the arithmetic gives ~7.3×. I have not
checked what the `ucbbar,rose-camera` driver actually writes, whether `IC=3` NHWC (not 4-byte aligned;
`gemmini_conv2d_s8_gemmini_tiled_conv.c:143-152` explains why the packed store path needs `IC%4==0`) is
efficient for gemmini's mvin, or what it would do to the fp32/int8 quantise step. Flagging it as the
highest-value follow-on question, not as a plan.

---

## 4. Constraints any design must satisfy

**C1 — Sound under arbitrary placement.** Because every backend's `model.c` contains every dispatch and
placement is bound by `strcmp` at run time (§2.3a), a design may not assume a dispatch's backend. Either
every backend honours the tensor's declared layout, or the design inserts something that makes it honour it.

**C2 — One physical layout per tensor.** Follows from the single shared `buffers.c` (§2.3b). Per-backend
activation layouts are not representable without duplicating and synchronising buffers, which would
reintroduce the copies we are trying to delete.

**C3 — Fail loudly, never silently.** A layout mismatch is size-identical and numerically catastrophic
(§2.3c). The tree's established answer to this class is a `SystemExit` tripwire, not a runtime check —
`generate_skeleton.py:2222-2236` (unknown split axis), `:1046-1103` (cross-op weight layout),
`generate_kernels.py:52-94` (curated layout contract).

**C4 — The end-to-end golden compare must stay meaningful.** See §7. This is the constraint that most
constrains the design, and it points at one specific answer: the model's input and output surfaces stay NCHW.

**C5 — A graph without the feature must emit byte-identical output.** An explicit discipline in this tree:
`generate_skeleton.py:4074-4080` and `:1352-1353` both note that a model not using a feature must produce
a byte-identical `model.c` so baselines stay comparable. A `layout` key absent from a tensor must mean
exactly what today means.

**C6 — Don't break sharding, and be honest about which axis it helps.** See §6.

---

## 5. The options

### Option 1 — Layout as a tensor attribute in the IR, with a layout-assignment pass

`ir["tensors"][name]["layout"] = "nchw" | "nhwc"`, absent ⇒ `"nchw"`. A pass reads the graph, chooses a
layout per tensor, and inserts explicit conversions where producer and consumer disagree.

*Soundness under arbitrary placement:* good, **provided** the "explicit conversion" and the "every backend
honours it" question are answered — which Option 1 alone does not answer. On its own it is a
representation, not a mechanism.

*Blast radius:* the IR itself is remarkably permissive. It is plain JSON with **no schema class, no
validator, no unknown-key rejection** anywhere in `pipeline/`; every applier is `copy.deepcopy` + mutate
(`apply_split_hint.py:558-565`), so a new key survives every rewrite for free. `"version": 1` is written
(`extract_graph.py:6`) and never read as a gate. Only two sites index `ir["tensors"]` directly
(`generate_skeleton.py:2041`, `:3913`). The real cost is not parsing — it is the *gating sets*
(`_SPLITTABLE_BY_AXIS` / `_OH_SLICEABLE_CONV_OPS` / `_OC_SLICEABLE_CONV_OPS`) which this codebase
repeatedly warns must stay in sync or you get "a clean build with a wrong answer"
(`generate_skeleton.py:140-152`).

*Verification:* neutral — it inherits whatever §7 decides.

*Incremental:* yes, trivially. Absent key ⇒ today's behaviour ⇒ C5 satisfied for free.

*Sharding:* the attribute is exactly what `apply_split_hint._register_tile_tensors:507-515` needs to stop
hardcoding `axis_dim`.

**Verdict: necessary, not sufficient.** This is the representation layer; it needs 2 and 3 on top.

### Option 2 — An explicit `relayout` op in the graph

A real op with a `KernelSpec`, a `dispatch_id`, and a row in the profile CSV.

*Soundness:* excellent, and for a non-obvious reason. Because it is a dispatch, it is emitted into
**every** backend's `model.c` like any other op, so whichever hart the schedule lands it on performs the
conversion. It needs no knowledge of placement.

*Schedulability and splittability — the strongest argument.* Today the transpose is 87% of a conv's time
and is invisible: it is inside `kernel_conv2d_s8`, so the MIP charges it to the conv and cannot move it,
overlap it, or parallelise it. As a dispatch it becomes: (i) visible to the offline MOSEK solver, which
already consumes measured per-op-per-backend cycles from `benchmarks/profile_db/` — no new cost modelling
needed, it just profiles like anything else; (ii) **splittable**, and a transpose is embarrassingly
parallel. NCHW→NHWC splits cleanly on H: each shard reads `IC` runs of `rows*IW` and writes one contiguous
`rows*IW*IC` band. *Arithmetic, not measured:* conv0's 364,810-cycle input relayout sharded 4 ways is worth
roughly 270 kcycles, which is 38% of the entire post-chaining budget — a bigger prize than anything left in
the conv itself.

*Blast radius:* moderate but well-trodden. A new op means a `KernelSpec` (`reference_kernels.py`), an entry
in `KERNEL_SPECS` (`:12035`), `_gen_inputs_*` and an `argtypes_factory` in `verify_kernel.py`, a call
emitter in `generate_skeleton.py`, and curated kernels. Adding an op is a routine operation in this tree —
there are ~140 `KernelSpec`s. Note there is **no** transpose/permute/relayout op today; this is genuinely new.

*Verification:* the best of any option. A relayout op has a trivially checkable oracle
(`nhwc_to_nchw(nchw_to_nhwc(x)) == x` bit-exact) and its NCHW reference impl is four lines.

*Incremental:* yes. It can land with layout assignment still doing nothing — the op exists, is verified,
is profiled, and is never emitted into a graph. That is a genuinely small and reversible stage 1.

#### 5.2a Hardware-boosted relayout — placement changes the ISA, not just the timeline

*Added 2026-09-02 after review. This strengthens Option 2 beyond what §5.2 argued, and partly answers the
doubt recorded in §10 ("placing it does not obviously help unless the solver can overlap it").*

That doubt considered only **overlap**. There is a second, larger effect: on this SoC the two hart types
have **different ISAs**, so placing a relayout changes what instructions can implement it.

* Harts 0/1 are Rocket + `Q31WsGemminiConfig` with **no vector unit**. Harts 2/3 are Rocket +
  `saturn.rocket.WithRocketVectorUnit(256, 128, ...)` (`RoSEConfigs.scala:444-470`).
* The measured ~9.7 cycles/byte for the transposes (§1.1) is therefore **not** "a transpose costs 9.7 c/B".
  It is "a transpose costs 9.7 c/B *on the one hart type that cannot vectorise it*" — because today the
  conversion is welded to the conv, and the conv is on gemmini.
* Saturn has exactly the primitives a relayout wants, and our rvv kernels already use them: `vlse8`,
  `vsse8`, `vlse32`, `vsse32` strided loads and stores appear across `kernels/rvv/*.c`.
* **RVV segment load/store (`vlseg`/`vsseg`, NF = 2..8) is the native AoS↔SoA instruction family — which is
  precisely NCHW↔NHWC.** `IC == 8` is a single `vsseg8e8`; `IC == 32` is four passes. **Nothing in the tree
  uses segment ops today** (0 occurrences across `kernels/rvv/`), so this is unexploited hardware.

Option 2 therefore buys three things, in decreasing order of confidence:

1. **A different ISA** for the conversion (scalar Rocket → Saturn RVV). Independent of overlap, and the
   largest effect if strided/segment throughput is good.
2. **Splittability** — as argued in §5.2; a relayout is embarrassingly parallel on H.
3. **Overlap** — weakest on a 2-hart dronet, as §10 says. Though the phase data makes it less marginal than
   it looks: gemmini itself is only 12.8% of "gemmini conv time", so the gemmini harts are mostly stalled on
   memory while the rvv harts sit idle whenever the solver places convs on gemmini. Moving conversion work
   to the idle side fills both.

**This is a hypothesis with a mechanism, not a result.** Saturn's strided-access throughput is unmeasured
here, and some vector implementations decompose a strided access into one memory operation per element, in
which case the win is far smaller than the ISA suggests. `vsseg` being entirely unused in this tree means it
is also unproven on this hardware.

**Consequence for stage 1 (§9):** stage 1 already lands curated relayout kernels for *both* `rvv` and
`gemmini` and profiles them independently, so it answers this before anything depends on the answer. Add one
thing: write **two** rvv candidates — a strided `vlse8`/`vsse8` version and a `vsseg`-based version — and let
the curator pick. Algorithm selection is already verify-gated and measured, so this costs one extra kernel
file and tests the segment hypothesis directly rather than by argument.

*Cost:* it materialises a buffer where a fused in-kernel transpose used the kernel's own workspace. Note
that the traffic is identical (the gemmini kernel *already* writes a separate `ws_input`/`ws_output`,
`gemmini_conv2d_s8_gemmini_tiled_conv.c:255-262`) — but the buffer becomes a `buffers.c` entry rather than
a 512 KB per-hart static, which is arguably an improvement in memory accounting.

**Verdict: the mechanism the design should be built on.** Options 1 and 2 are not alternatives —
1 is the decision, 2 is how the decision is materialised.

### Option 3 — Per-op layout requirements declared in the kernel spec

`AlgorithmCandidate.act_layouts: tuple[str, ...] = ("nchw",)`, an exact mirror of the existing
`weight_layout` field at `reference_kernels.py:102-108`, plus a resolver mirroring
`_conv_weight_layout_for_op` (`generate_skeleton.py:1122-1168`) and a `SystemExit` gate mirroring
`_assert_curated_layout_contract` (`generate_kernels.py:52-94`).

*Soundness:* this is what turns Option 1 from a representation into a checkable contract. At codegen,
`emit_model(ir, out_dir, platform, backend)` knows **both** the tensor's declared layout and *this*
backend — so for every (op, backend) pair it can decide: emit the native NHWC kernel, or emit a shim, or
`SystemExit`. Because it runs once per backend and every backend's `model.c` is emitted, the check is
*total over placements* even though no single invocation knows the placement. **This is the key soundness
argument of the whole design.**

*Blast radius:* small and mechanical. Most entries can be *derived* from `KernelSpec.signature` (flat-`n`
⇒ polymorphic; see §3), so only the `N,C,H,W`-shaped ops need annotation.

*Incremental:* yes — default `("nchw",)` reproduces today exactly.

**Verdict: adopt, as the declaration layer.**

### Option 4 — Runtime layout tag on the buffer, consumers materialise what they need

*Soundness:* fail-safe under any placement, which is its appeal.

*Why it is wrong here:* it puts a branch and a potential full-tensor copy on every kernel entry, decided at
run time, so the cost model can no longer predict a dispatch's cost — the MIP's per-op cycle numbers become
placement-history-dependent. It thrashes exactly when placement alternates, which is the case the
heterogeneous scheduler is *designed* to produce. And it needs a tag next to each buffer, which means either
mutating `buffers.c`'s shape (a shared, size-derived TU) or a side table — new shared mutable state on an SMP
system where the tree has already been bitten by exactly that (`gemmini_conv2d_s8_gemmini_tiled_conv.c:62-88`,
the per-hart workspace fix).

The deeper objection: this design *chooses at run time something the compiler already knows*. Codegen has
the tensor layout and the backend; the only fact it lacks is placement, and Option 3's totality check
removes the need for it.

**Verdict: reject**, but keep one idea from it — the *fallback* it implies (convert on entry when you can't
consume the layout) is right; it just belongs at codegen, not at run time. That is the shim in §6.

### Option 5 — Islands with explicit boundaries

Not really a separate mechanism — it is what Options 1+2+3 produce. Worth naming as the *policy*: the
assignment pass computes maximal connected subgraphs whose ops all have an NHWC path, and inserts relayout
dispatches on the boundary edges. The policy question ("which islands?") is separable from the mechanism
and can start as: *only islands the pass can prove profitable, only `conv2d_s8` / `maxpool2d_s8` /
`batchnorm2d_s8` / flat-`n` ops, never crossing the model input or output surface.*

### Option 6 — A blocked layout (NCHWc / `[N, C/c, H, W, c]`)

Considered and rejected for stage 1. This is the standard answer in TVM and oneDNN and it has a real
advantage here: an OC slice at block granularity stays contiguous, so it would preserve the OC split and
the OC shard that plain NHWC breaks (§6). But `tiled_conv_auto` consumes **dense NHWC**
(`gemmini_conv2d_s8_gemmini_tiled_conv.c:509-518`), so a blocked layout would not remove the transpose —
which is the entire point of the exercise. It also does not make an OH row band contiguous (it is `C/c`
runs, not one), so it does not deliver the sharding win either. Revisit only if a future accelerator wants
blocked input.

---

## 6. Interaction with sharding — the honest trade

This cuts both ways and the document would be dishonest to present only the upside.

**NHWC deletes the OH tax.** `_OH_TILE_HELPER` (`generate_skeleton.py:1389-1450`) exists solely because
"in NCHW neither end of a row band is contiguous" (`:1395-1400`). It emits a gather of
`IC * win_rows * (IW + 2*PW)` bytes (materialising the conv's padding), then the kernel, then a scatter of
`OC * oh_rows * OW` bytes in `OC` separate `memcpy`s (`:1445-1448`). Under NHWC a row band is **one
contiguous run** on both sides: the gather becomes a pointer offset and the scatter disappears. The
`_OH_TILE_HELPER_ZEROCOPY` variant (`:1453-1502`) and the whole `mb_gem_ohwin` window machinery in the
gemmini kernel (`gemmini_conv2d_s8_gemmini_tiled_conv.c:163-233`, ~70 lines of windowed transpose walk)
become dead code. That machinery exists *because* the kernel already transposes and can be taught the
parent's stride — the comment at `generate_skeleton.py:1462-1465` says so outright. It is a hand-rolled
special case of what a first-class layout concept generalises.

**NHWC creates an OC tax — the exact mirror.** Under `[N, OH, OW, OC]` an OC slice is strided, not
contiguous. That breaks two things:

- the graph-level OC split's output alias, `elem_offset = oc0 * OH * OW`
  (`generate_skeleton.py:2199-2205`) — the `offset_aliases` mechanism holds a single scalar offset and
  **cannot express a strided view at all**;
- the intra-op OC shard, `c->out + oc0 * OH * OW` (`:596`).

So a naive NHWC island would silently corrupt any OC-split or OC-sharded conv inside it. Per C3 this must
be a `SystemExit` in the assignment pass, not a comment. There is a precedent for the *graceful* form too:
when the weight layout makes an OC slice non-contiguous, the codegen compiles the shard path **out** and
falls back to a serial call with an explanatory comment rather than emitting something wrong
(`generate_skeleton.py:893-916`, `_serial_only_wrapper` at `:920`). That is the pattern to copy.

**Net effect on axis choice.** The memory note records the current best-axis result: OC is quantum-limited
(rvv 32, gemmini 16), OH is copy-taxed by NCHW, best-axis dronet = 1.88× on 2 rvv harts vs 1.37× all-OC on
F2. NHWC **flips the preference**: OH becomes free and OC becomes taxed. The two changes are not
independent and should not be evaluated independently — a sharding sweep run on an NCHW build does not
predict the NHWC build's best axis.

**A unifying fix worth noting (speculative, out of scope for stage 1).** Both problems are the same
problem: `offset_aliases` is offset-only. Generalising it to a strided view `(offset, stride, count)` would
make *both* axes work in *both* layouts and would delete `_OH_TILE_HELPER` for NCHW too. That is a larger,
more valuable refactor than the layout work itself, and it is a reason to be careful about how much
special-case code the layout work adds on top of the current alias mechanism.

**The relayout op is itself shardable**, which partially compensates: see §5 Option 2.

**The E axis is the exception that proves the framing (added 2026-09-02).** Everything above is about
*which* axis a layout taxes, and both of the axes it discusses are geometric: `OC` names a channel, `OH`
names a row, and a layout decides whether either is contiguous. The pointwise split axis `E` names neither
— it is a flat range of elements out of the tensor's storage — and a contiguous byte range stays contiguous
under **every** permutation of the axes, because permuting axes changes which logical coordinates the
elements carry, not which elements a `[lo, hi)` range covers. A pointwise op does not read coordinates, so
the tile is correct in nchw, nhwc, or anything else. It is the only axis in `apply_split_hint` that carries
no layout guard, and `act_layout.LAYOUT_AGNOSTIC_OPS` is the same fact arrived at from the other direction
— the ops that need no relayout are exactly the ops whose split needs no layout.

This matters for staging: the pointwise and pool splits (which cover `silu_s8` ×57 in yolov8n and
`maxpool2d_s8` at 28.9% of DroNet's gemmini time) can be taken *before* the NHWC migration without
inheriting any of the axis-flip risk above. The pool's `C` axis is not exempt — it is a channel range with
exactly the conv's `OC` contiguity claim, and it carries the same guard by name.

---

## 7. Verification — a first-class constraint, not an afterthought

The survey of the verify machinery produced three facts that dominate this section.

**(V1) `verify_kernel.py` is NCHW to the bone, but only matters for `scalar`.**
Every `_gen_inputs_*` hardcodes NCHW buffer shapes (`verify_kernel.py:205-214` conv2d,
`:298-307` conv2d_s8, `:310-319` maxpool2d_s8, `:330-336` batchnorm2d_s8), the compare at `:927` is
index-for-index, and the failure diagnostic unravels against `out_ref.shape` (`:931-932`). There is no
layout hook — the only `layout`-adjacent grep hit in the file is a `transpose_b` matmul operand flag at
`:691`. But `VERIFY_HOST_CTYPES` is used by exactly two backends, `scalar` and `scalar_f16`
(`backends.py:105`, `:212`), neither of which repacks anything.

**(V2) For every vector backend the "per-kernel" gate is already a whole-model golden compare.**
`VERIFY_SPIKE_HARNESS` (`generate_kernels.py:691-818`) builds the entire model with the candidate kernel
substituted and compares the **final model output** against the PyTorch golden in `io.npz`. Consequence:
an NHWC *intermediate* is invisible to it. That is simultaneously reassuring (the gate does not
false-alarm on a layout change) and dangerous (it does not localise).

**(V3) There is no intermediate-tensor observability to build on.** `io.npz` contains only inputs and
outputs (`extract_graph_export.py:2244-2345`). There is no `MB_DUMP`, no per-op checksum, no CRC anywhere
in `pipeline/`, `scripts/`, `validation/`, `runtime/`, `harness*/`. The one aggregate that exists —
`OUTPUT_SUMMARY sum= abs_sum= min= max=` at `generate_xpurt_main.py:548-549` — is **permutation-invariant**
and therefore worthless as a layout check.

And the escape hatches that do exist are all scalar tolerances (`Backend.atol_override`,
`AccuracyClass`/`ACCURACY_CLASS_ATOL` at `reference_kernels.py:64-69`, `MB_DRIFT_ATOL` at
`generate_kernels.py:734-767`). None can absorb a permutation: an NHWC tensor compared elementwise against
NCHW is wrong at essentially every position by the full int8 dynamic range, so the tolerance would have to
be ~255 and would then accept anything.

### The plan

**1. Keep the model input and output surfaces NCHW.** Non-negotiable, and it is what preserves the entire
existing gate. `io.npz`, `test_golden.bin` (`generate_skeleton.py:4206-4208`), the in-binary compare
(`harness/src/main.c:136-148`, `generate_multi_main.py:83-106`, `generate_xpurt_main.py:441-460`), and the
host parsers (`validation/runner_common.py:52-68`, `:342-357`) all continue to work **unmodified**.
An NHWC island that round-trips at its boundaries is bit-identical end-to-end to the NCHW build — for int8
this is exact, since a permutation is a bijection and no arithmetic changes.

**2. Per-kernel verify for NHWC ops: permute in the harness, compare in one canonical layout.**
Give `verify_kernel` a layout parameter sourced from the algorithm's declared `act_layouts`; for an NHWC
candidate, permute the NCHW-generated inputs into NHWC, run the candidate, permute its output back to
NCHW, and compare **bit-exactly** against the unmodified NCHW `reference_impl` run on the original inputs.

This deliberately departs from the tree's stated principle (§2.1: "make both sides physically agree, never
permute at compare time"). The departure is defensible and the reason should be recorded: that principle
exists because a *weight* permutation happens in production codegen, so permuting at compare time would
have meant the test and the shipped artifact disagreed. Here the permutation is a lossless bijection
applied symmetrically inside a test harness on int8 data; the comparison is still index-for-index in a
single canonical layout, and the alternative — a second, NHWC `reference_impl` for every layout-bound op —
is more code to get wrong for no additional assurance. Worth a short comment at the call site saying so.

**3. A relayout round-trip identity test.** `nhwc_to_nchw(nchw_to_nhwc(x)) == x`, bit-exact, at every shape
in the model plus `extra_shapes`. Cheap, and it catches the whole class of relayout indexing bugs before
they reach a model.

**4. New: per-dispatch activation hashing (`MB_ACT_CRC`).** This is the one piece of genuinely new
infrastructure the design needs, and it is needed because of V2+V3: today a layout bug in the middle of the
island surfaces as "final output wrong", with 21 dispatches to bisect by hand. Proposal: an optional build
flag that emits `MB_ACTCRC,<dispatch_id>,<crc32>` after each dispatch, with the host comparator reading
each tensor's declared layout from `graph.json` and canonicalising to NCHW before comparing against a
PyTorch per-layer reference. This is useful well beyond layout work — it is the missing intermediate
observability the tree has never had (`notes/observability_gaps.md` is the natural place to record it).
Cost: a codegen change in `emit_model`, a golden-generation change in `extract_graph_export.py` to dump
per-layer activations, and a host comparator. Non-trivial; do it in stage 2, not stage 1.

**5. A layout-coverage gate.** The most likely failure of this whole design is *not* a wrong answer — it is
the pass silently not engaging, producing a correct-but-NCHW build indistinguishable from the baseline
except in cycles. That is precisely the hazard `_assert_curated_layout_contract` warns about
(`generate_kernels.py:62-67`: "the fallback it triggers is numerically correct and therefore invisible
end-to-end"), and the tree's existing answer for the analogous kernel-coverage case is
`scripts/check_kernel_coverage.py:1-52`, which fails a vector-labelled build that silently ran scalar.
Do the same: a build labelled NHWC must assert the expected count of NHWC tensors and relayout dispatches,
and fail otherwise.

---

## 8. Recommendation

**Adopt Options 1 + 2 + 3 together. They are three layers of one design, not three alternatives.**

| layer | mechanism | mirrors |
|---|---|---|
| **Declare** | `AlgorithmCandidate.act_layouts: tuple[str,...] = ("nchw",)` — which activation layouts this algorithm's C can consume and produce. Derived from `KernelSpec.signature` where possible (flat-`n` ⇒ all layouts). | `weight_layout`, `reference_kernels.py:102-108` |
| **Decide** | `ir["tensors"][t]["layout"] = "nchw" \| "nhwc"`, absent ⇒ `"nchw"`. Chosen by a new applier `pipeline/assign_layouts.py`, in the shape of `apply_shard_hint.py` (reads `graph.json`, writes `graph.json`, records provenance under `_rewrite`). | `apply_shard_hint.py:124`, `:128-129` |
| **Materialise** | `nchw_to_nhwc_s8` / `nhwc_to_nchw_s8` as real ops with `dispatch_id`s, inserted on island boundary edges. Schedulable, profilable, splittable on H. | any `KernelSpec` |
| **Enforce** | At codegen, for each (op, backend): if the tensor's layout ∉ `act_layouts(op, backend)`, emit a **shim** (relayout into per-hart scratch → NCHW kernel → relayout out) — the same generated code as the relayout kernel, at a second call site. If no shim is possible, `SystemExit`. | `_assert_curated_layout_contract`, `generate_kernels.py:52-94`; `_serial_only_wrapper`, `generate_skeleton.py:893-920` |

### Why this and not the others

- It satisfies **C1** without knowing placement, because the check is total over backends: `emit_model` runs
  once per backend, and *each run* either emits a native kernel or a shim. Whichever `strcmp` wins at run
  time, the code it selects is correct.
- It satisfies **C2** by construction: layout is per tensor, decided once, backend-independent.
- It satisfies **C3** with a `SystemExit` in the pattern the tree already uses three times.
- It satisfies **C4** by keeping the model surfaces NCHW, so `io.npz` and every golden comparator are untouched.
- It satisfies **C5** for free: no `layout` key ⇒ no relayout dispatches ⇒ byte-identical `model.c`.
- It makes the 87% **visible**. The single worst property of today's arrangement is not that the transposes
  are expensive; it is that they are *hidden inside a kernel*, so the MIP charges them to the conv and can
  neither move them nor split them. Everything good in this design follows from making them a dispatch.

### The shim, specifically

The shim is what makes arbitrary placement safe, so it deserves precision. For backend B lacking an NHWC
`conv2d_s8`, B's `model.c` emits `relayout_to_nchw(T_in, scratch); kernel_conv2d_s8(scratch, ...,
scratch_out); relayout_to_nhwc(scratch_out, T_out);`. Notes:

- The scratch is **file-static in `model.c`**, not in `buffers.c`. Correct because it never crosses a
  dispatch boundary and therefore never crosses a backend. It **must be per-hart**, indexed the way
  `MB_GEM_WS_SLOT` is (`gemmini_conv2d_s8_gemmini_tiled_conv.c:78-86`) — the tree has already been bitten
  once by a shared static workspace under concurrent shards.
- The shim's cost is *not* modelled analytically. It is measured, because a shimmed op simply profiles
  slower on that backend in `benchmarks/profile_db/<network>__<target>__<quant>.jsonl`
  (`benchmarks/profile_db.py:22`, `:114`) and the MIP consumes those numbers directly
  (`scripts/run_xpurt_scheduler_multi.py:136-137`, `:192-202`). No new cost model.
- Consequence for correctness reasoning: **placement can never produce a wrong answer, only a slower one.**

### 8.1 Breaking the layout ↔ cost ↔ placement cycle

The cycle is real: layout affects cost, cost drives placement, placement determines what layout is wanted.
The design breaks it in three ways, in decreasing order of importance:

1. **By decoupling correctness from the cycle entirely.** The shim means that any layout assignment is
   *correct* under any placement. So the cycle is a pure optimisation loop, not a soundness loop. This is
   the whole reason for preferring the shim over the alternative rule ("only choose NHWC where every
   backend has a native NHWC kernel"), which would be sound but would yield an empty island in any build
   that includes rvv until someone writes rvv NHWC kernels.
2. **By making the loop measured rather than modelled.** ModelBlaster does not estimate per-op cost — it
   *measures* it, per (network, target), on the target
   (`benchmarks/profile_db.py:1-90`; host cycles are explicitly forbidden as an input,
   `generate_skeleton.py:3993-3997`). So the honest loop is: assign layouts → generate → **profile** →
   solve → run. One pass. The second solve sees true costs, including shims, because they were measured.
3. **By ordering the passes and bounding the iteration.** Run the layout pass *before* placement, using the
   conservative all-NCHW conv cost (which is an upper bound on any layout's conv cost, since the current
   kernel already pays both conversions). Then place, then optionally re-run the layout pass once against
   the resulting placement, then place once more. **Stop there.** This is a heuristic with no optimality
   guarantee and the document should not pretend otherwise; the guarantee we get is correctness under any
   placement plus a monotone first step, not a fixed point.

The principled alternative — making layout a decision variable inside the MOSEK MIP alongside placement —
is the right long-term answer and the wrong stage-1 answer. It would need a relayout cost on every graph
edge and a per-(op, backend, layout) cost matrix, roughly squaring the model, for a problem that is already
a MIP over ~21 dispatches × N backends. Note it; don't build it.

---

## 9. Staged migration

### Stage 1 — small, reversible, delivers a measurement (no IR change at all)

Land **only the relayout op**, with layout assignment doing nothing.

- Add `NCHW_TO_NHWC_S8` / `NHWC_TO_NCHW_S8` `KernelSpec`s to `reference_kernels.py` with reference impls
  (~10 lines each), `extra_shapes` covering dronet's geometries, `argtypes_factory`, and `_gen_inputs_*` +
  round-trip identity in `verify_kernel.py`.
- Add curated kernels for `rvv` and `gemmini` — the gemmini one is a *lift* of the existing blocked
  transpose loops out of `gemmini_conv2d_s8_gemmini_tiled_conv.c:338-446` and `:530-598`, which are already
  tuned (the `mb_gem_put4` packed-store path, the `TB=32` blocking whose removal cost conv0 46% per the
  comment at `:551-558`). This is code motion, not new tuning.
- Do **not** touch the IR, the assignment pass, `emit_model`'s call emitters, or any existing kernel.

**What it buys:** an independently profiled, verified, splittable relayout kernel, and a real
cycles-per-byte curve to replace the ~10 cycles/byte estimate in §1.1 — which is the number every later
stage's cost reasoning depends on. **What it risks:** nothing. Zero graphs change; `model.c` is
byte-identical for every existing model (C5); reverting is deleting two files and two spec entries.

### Stage 2 — the representation and the gate, still no behaviour change

- `AlgorithmCandidate.act_layouts` (default `("nchw",)`), the per-op resolver, and the codegen
  `SystemExit` gate. With every default at `"nchw"`, the gate is a no-op that proves the plumbing.
- `ir["tensors"][t]["layout"]`, read by nothing but the gate.
- `assign_layouts.py` present but defaulting to "assign nothing".
- Teach `apply_split_hint._register_tile_tensors:507-515` to read `layout` instead of hardcoding
  `axis_dim`, and add the OC-under-NHWC `SystemExit` (§6) *before* any NHWC tensor can exist.
- `MB_ACT_CRC` per-dispatch hashing (§7.4) — build it here, while NCHW is still the only layout, so the
  tool itself is validated against a known-good build.

Still byte-identical output. Still fully reversible.

### Stage 3 — one island, one model, one backend, opt-in

- NHWC entry points for `conv2d_s8`, `maxpool2d_s8`, `batchnorm2d_s8` on gemmini. For conv and maxpool this
  is *deleting* the transposes and declaring `act_layouts=("nhwc",)`; only batchnorm needs new C.
- Shim generation in `emit_model` for backends without a native NHWC path.
- `assign_layouts` policy: maximal islands over `{conv2d_s8, maxpool2d_s8, batchnorm2d_s8}` + flat-`n` ops,
  never crossing the model input/output surface, never containing an OC-split or OC-sharded op.
- Gate the whole thing behind an explicit hint file (the tree's established pattern —
  `modelblaster.split_hints/v1`, `apply_split_hint.py:58`; `modelblaster.shard_hints/v1`,
  `apply_shard_hint.py:54`) so it is opt-in per model.
- Validate: dronet, all-gemmini, F2. Expect ~3.5× on conv cycles and bit-identical model output.

### Stage 4 — generalise

Layout-coverage gate (§7.5); the linear K-axis permutation (§3.1b); re-run the sharding axis sweep under
NHWC (§6 — the existing best-axis result does not carry over); investigate the sensor-side input layout
(§3.2); consider generalising `offset_aliases` to strided views (§6).

---

## 10. What could go wrong, and what I am unsure about

**Things I believe are right but have not verified in code:**

- That `emit_model` has a clean insertion point for shim emission around every layout-bound op's call site.
  I read the conv2d_s8 emitter (`generate_skeleton.py:2993-3033`) but not the batchnorm or maxpool ones,
  and the fused-op path (`sub_ops`) stores geometry on `sub_ops[0]` rather than on the op
  (`apply_split_hint.py:376-386`), which the survey flags as a recurring source of "the accessor finds
  nothing and the feature silently falls back". A `layout` attribute will hit that same two-places problem.
- That the linear K-axis permutation (§3.1b) composes with the linear N-split. I reasoned it does (N slices
  rows, K permutes columns) but did not check `_n_tile_operands`.
- That an `IC=3` NHWC input is actually good for gemmini's mvin. The packed-store path explicitly requires
  `IC%4==0` (`gemmini_conv2d_s8_gemmini_tiled_conv.c:143-152`), and conv0 — the layer that matters most —
  has `IC=3`.

**Things that could make the 3.53× not materialise:**

- The estimate assumes the NHWC conv is otherwise identical to today's. It will not be exactly: removing
  the transposes changes the working set (`ws_input`/`ws_output` disappear, and the conv reads and writes
  `buffers.c` arrays directly), which changes cache behaviour in a direction I cannot predict. The output
  transposes measured at 30–43 cycles/byte on small tensors (§1.1) are evidence that cache effects here are
  large and not obviously monotone.
- `tiled_conv_auto` is handed `ws_input`/`ws_output` today (`:509-518`). Handing it the model's own buffers
  instead requires those buffers to satisfy whatever alignment and aliasing the ROCC DMA needs. The kernel
  currently declares `__attribute__((aligned(64)))` on its workspaces (`:257-260`); `buffers.c` emits plain
  arrays with no alignment attribute (`generate_skeleton.py:2404-2407`). **This is a concrete thing to check
  before stage 3**, and if it forces a copy back into an aligned workspace, a large part of the win evaporates.
- The `gemmini_fence()` / `gemmini_flush(0)` sequencing (`:521-527`) was added because "the post-conv
  NHWC->NCHW read and the next op's `gemmini_flush` race with in-flight mvout DMAs and corrupt memory
  (FireSim Saturn: mcause=1, mepc=0)". Removing the transpose removes the read that motivated the fence's
  placement. The fence must stay; it would be easy to lose it while deleting the loop around it.

**Things that could make the design unsound in a way I have not caught:**

- **The silent size-identity (§2.3c) is the central danger.** Every safeguard proposed here is a codegen
  check, and codegen checks only fire on the paths that run. An op reached by a path the gate does not
  cover produces a plausible wrong answer with no error anywhere in the build — the exact signature of the
  `max_abs_err=51` and `max_abs_err=57` bugs already in this tree. I would want the stage-2 gate to be
  *deny-by-default* (every op must positively declare a layout it supports; unknown ⇒ `SystemExit`) rather
  than allow-by-default.
- The `_zero_cost_ops` set (`generate_skeleton.py:2270`: `view`, `chunk2_c1*`) and the offset-alias machinery
  assume aliases are layout-preserving. `view` is (bytes don't move); `chunk2_c1` is a **channel** chunk, and
  under NHWC a channel chunk is strided, not a contiguous offset. dronet has no `chunk2_c1`, but yolov8 does
  (`extract_graph.py:5087`). Any island containing one must be refused.
- Concatenation: `cat2/3/4_c1_s8` are channel-axis concats whose codegen assumes contiguous planes
  (`verify_kernel.py:262-278`, `:368-374`). Same problem, same answer — refuse, for now.

**Things I am unsure about at the design level:**

- Whether the shim is the right default, or whether it is a trap. It guarantees correctness, but it also
  guarantees that a bad layout assignment degrades *quietly* into extra copies rather than failing. That is
  the "numerically correct and therefore invisible" hazard again, one level up. The layout-coverage gate
  (§7.5) is my proposed mitigation; I am not confident it is sufficient.
- Whether the deviation from "never permute at compare time" (§7.2) is really as safe as I argued. It is
  exact for int8 and for any bit-exact op. It would *not* be safe for an op whose NHWC implementation
  changes accumulation order in a way that matters for float — which is not the current island, but would
  be if this ever extends to `rvv_f16`.
- Whether making relayout a schedulable dispatch actually helps the MIP or just gives it more to chew on.
  A relayout is a pure serial dependency between two ops; splitting it helps, but placing it does not
  obviously help unless the solver can overlap it with an unrelated dispatch. On a 2-hart dronet there may
  be nothing to overlap with. The value of Option 2 may be mostly *visibility* and *splittability* rather
  than *placement freedom*, and I have stated it that way above.
  **Partly answered — see §5.2a (added 2026-09-02).** That reasoning holds only for overlap. Placement also
  selects the *ISA*: the gemmini harts have no vector unit, the Saturn harts do, so placing a relayout on
  hart 2/3 makes `vlse8`/`vsse8` — and potentially the unused `vsseg` segment family, which is literally an
  AoS↔SoA instruction — available to it. That is a per-op speedup independent of overlap. Still unmeasured;
  stage 1 measures it.
- Whether stage 1 is too small. It delivers no speedup, only a measurement and a verified kernel. The case
  for it is that the ~10 cycles/byte constant is load-bearing for every later decision and is currently
  inferred from a kernel whose loops are entangled with a gemmini call. If the team would rather take on
  more risk for a faster answer, stages 1 and 3 can be merged and stage 2's gate written afterwards — but
  then the first NHWC build lands without the `SystemExit` tripwire that stage 2 exists to provide, and
  §2.3c says what that costs.

---

## 11. Stage 1 results (measured 2026-09-02, fq 480-484)

Stage 1 landed: six curated relayout kernels, `max_abs_err = 0` on every row, profiled on F2.
Artefacts `experiments/relayout/`; log entries 390-391. **Four things below change decisions made
earlier in this document.**

### 11.1 The prize is 7.0x, not 3.53x

§1 computed 3.53x on the assumption that the two surviving conversions — conv0's NCHW input and the
return to NCHW before flatten/linear — still cost what they cost *inside* the conv. They do not. As a
relayout dispatch on a Saturn hart using segment ops they run ~11x cheaper per byte:

| | scalar (in-conv) | rvv_seg relayout |
|---|---:|---:|
| gemmini compute (unchanged) | 318,487 | 318,487 |
| conv0 input conversion | 364,810 | 33,096 |
| tail output conversion | 21,700 | 1,969 |
| **post-chaining total** | **704,997** | **353,552** |

Against today's 2,488,855 cycles that is **3.53x → 7.04x**, and the conversion share of what remains
falls from 55% to 10%. The residue becomes gemmini compute, which is the thing worth paying for.

### 11.2 §5.2a was half right — it is segments, not vectors

The doubt in §10 said the ISA advantage might be illusory. For plain strided access it largely **is**:
`vlse8`/`vsse8` buy only ~2x over the scalar loop. The win is the **segment** family:

| variant | c/B (n2h / h2n) | vs scalar |
|---|---|---|
| `ref_scalar` | 12.25 / 8.44 | — |
| `gem_tb32` (the lifted blocked nest) | 8.03 / 6.03 | 1.5x |
| `rvv_strided` | 4.14 / 2.86 | 2.0x |
| **`rvv_seg`** | **0.85 / 0.90** | **8.0x** |

`vsseg`/`vlseg` were unused anywhere in this tree before stage 1. The advantage is a step function in
C — 7-15x where C <= 8, 3.0x where C >= 16 — because NF is legal for every value 2..8, so **conv0's
C=3 is a single `vsseg3e8` on the fast path**, not a fallback. The layer that matters most is the one
the ISA suits best.

### 11.3 Two cost-model assumptions here were wrong

* **There is no ~20 kcycle fixed per-relayout term.** Least squares over 17 shapes puts the intercept
  under 600 cycles — two orders of magnitude smaller. A cost model carrying the larger figure would
  refuse to split a relayout that splits perfectly well.
* **The 30-43 c/B measured on small tensors is NOT in the transpose.** A standalone relayout with 512 KB
  flushed before every timed call peaks at 10.1 c/B on the worst real shape. That blowup belongs to
  something else inside `conv2d_s8` and should not be attributed to layout conversion.

### 11.4 §9's stage-1 recipe cites code that no longer exists

It points at a `mb_gem_put4` packed-store path and a comment about blocking removal costing conv0 46%.
Both were reverted before this document was written — the packed-store experiments measured 0.99x and
0.76x and were backed out (log entry `gemmini_transpose_phase_attribution`). The lift itself is
unaffected; the blocked `TB=32` nests are at `gemmini_conv2d_s8_gemmini_tiled_conv.c:307` and `:450`.

### 11.5 Caveat on the scalar baseline

Arms A and B ran byte-identical ELFs and agreed to 0.00%, which measures FPGA determinism, not
robustness. A rebuilt arm (buffers +256 B in BSS) shifted the *scalar* kernel by -12.4%/+17.2%
uniformly across shapes while leaving its **pooled** cost unchanged (7.03 → 7.05) and every ratio above
intact. So quote the scalar floor pooled; the scalar per-direction asymmetry is a property of the
binary, not the algorithm. The RVV direction asymmetry is stable to two decimals.
