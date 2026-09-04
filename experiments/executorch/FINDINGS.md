# ExecuTorch on RVV — the four campaign models on `f2_quad_hetero_norose_tacit_q31_60mhz`

`mlp_control`, `dronet`, `yolov8_nano` and `vint` taken through the ExecuTorch
flow onto the Saturn RVV harts of the F2 quad-hetero bitstream, and compared
against the ModelBlaster numbers measured on the **same bitstream**.

Everything below is measured on real RTL through the shared `fq` queue unless it
says otherwise. Job ids are in the tables; raw uartlogs are in
`experiments/executorch/res/<tag>/`, per-model export reports in
`experiments/executorch/pte/<tag>.json`.

---

## 1. The RVV-path question, answered

**ExecuTorch reaches RVV through XNNPACK, and only through XNNPACK.** This is
not a scalar-only measurement — but it is not a fully vectorized one either.
Three facts from the source and the toolchain settle it:

1. **The RVV micro-kernels really are compiled with the vector extension**, per
   source file — `third-party/executorch/backends/xnnpack/third-party/XNNPACK/CMakeLists.txt:1000`:

   ```cmake
   SET_PROPERTY(SOURCE ${ALL_RVV_MICROKERNEL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=rv64gcv -mabi=lp64d ")
   ```

   Per-source `COMPILE_FLAGS` land **last** on the command line, after Zephyr's
   own `-march`, so they win. Verified on a real compile line: the flags for
   `f32-argmaxpool-9p8x-rvv-u1v.c` end `… -march=rv64imafdc_zicsr_zifencei … -march=rv64gcv`.
   The linked ELF contains **1708 vector instructions**.

2. **Kernel selection is compile-time, so it cannot silently fall back.** The
   ucb-bar fork patched `src/configs/hardware-config.c:254` to
   `const bool use_riscv_vector = XNN_ENABLE_RISCV_VECTOR;` with the `getauxval`
   hwcap probe commented out. With `-DXNNPACK_ENABLE_RISCV_VECTOR=ON`
   (`XNN_ENABLE_RISCV_VECTOR=1`, gated in `build_et_elf.sh`),
   `xnn_arch_riscv_vector` is set and the `_rvv` ukernels are the registered
   ones. There is no cpuinfo path on bare metal that could quietly pick scalar.

3. **Everything NOT delegated to XNNPACK is scalar, by construction.** The
   ExecuTorch subbuild's base flags are `-march=rv64imafdc` (no `v`), and Zephyr
   additionally passes `-fno-tree-vectorize -fno-tree-loop-vectorize
   -fno-tree-slp-vectorize`. ExecuTorch's **portable kernels therefore cannot be
   vectorized at all** — not by hand, not by the compiler.

So the quantity that matters per model is **how much of the graph XNNPACK
actually takes** (§2). Short version: *the convolutions and GEMMs are real RVV;
the glue is scalar.*

One correction to the tree: `XNNPACK_ENABLE_RISCV_GEMMINI`, which
`samples/executorch/executor_runner/CMakeLists.txt` sets to ON, **does not exist
in any XNNPACK in this repo** — no `gemmini` string in any XNNPACK CMakeLists,
no `*gemmini*` source anywhere. It is a no-op. ExecuTorch has no Gemmini path
here at all; Gemmini is reachable only through ModelBlaster's `gemmini_q31`
backend. Harts 0/1 of this bitstream are consequently unusable by ExecuTorch.

---

## 2. Export feasibility — what each model lowers to

**All four export.** Delegates = XNNPACK subgraphs (RVV); undelegated =
ExecuTorch portable kernels (scalar). `getitem` is free (it only unpacks a
multi-output node), so it is excluded from the "real ops" column.

| model | quant | delegates | undeleg. nodes | of which real ops | .pte |
|---|---|---:|---:|---|---:|
| `mlp_control` | fp32 | 4 | 7 | **3 × `elu`** | 189 KB |
| `dronet` | int8 | 6 | 18 | **3 × `_native_batch_norm_legit_no_training`**, `view_copy`, `_clone_dim_order` | 342 KB |
| `yolov8_nano` | int8 | 11 | 43 | **8 × `split_with_sizes_copy`, 2 × `upsample_nearest2d`** | 3.2 MB |
| `vint` | int8 | 69 | 288 | **72 × `view_copy`, 23 × `_clone_dim_order`, 17 × `expand_copy`, 12 × `select_copy`, 8 × `native_layer_norm`, 8 × `mul.Scalar`, 8 × `logical_not`**, +13 more kinds | 24 MB |

- **`mlp_control` is fragmented by `elu`.** XNNPACK does not take ELU, so a
  4-layer MLP becomes 4 delegated GEMM segments separated by 3 scalar
  activations. At this size the graph is almost all boundary.
- **`dronet` is the cleanest fit** — 6 delegated segments cover all 10 convs;
  only the 3 standalone BatchNorms and 2 layout ops fall out.
- **`yolov8_nano`** loses the C2f channel splits and both upsamples.
- **`vint` is the worst case by a wide margin**: 69 delegates against 288
  undelegated nodes. The transformer tail (LayerNorm, the
  `logical_not`/`where`/`any` attention-mask arithmetic) and the very heavy
  `view_copy`/`expand_copy` traffic are all portable-scalar. That is a
  structural mismatch between a ViT-style model and an XNNPACK-only backend,
  not a tuning problem.

---

## 3. Measured — ExecuTorch vs ModelBlaster, one RVV hart, same bitstream

ET = `rdcycle` around `method->execute()`, 3 iterations (iter 0 cold, warm =
min of the two warm iters), `MB_XNN_PROFILE=OFF`, threadpool clamped to 1,
pinned to hart 2. MB = the `rdcycle` sum of the per-op profile from
`experiments/sweep3net/res_<model>_serialE_base` — every op on one `rvv` hart.
MB's independent mtime-based wall counter agrees with that sum to within ~1% for
all four models, so MB's inter-op overhead is negligible and the two numbers are
measuring the same thing.

| model | ET cold | ET warm | ET warm @1 GHz | MB rvv 1-hart | MB ops | **ET/MB** |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_control` fp32 | 1,600,835 | **1,512,815** | 1.51 ms | 578,541 (fq 597) | 7 | **2.61×** |
| `dronet` int8 | 15,714,339 | **13,941,927** | 13.94 ms | 7,855,964 (fq 601) | 21 | **1.77×** |
| `yolov8_nano` int8 | 310,728,541 | **303,143,017** | 303.14 ms | 167,132,539 (fq 616) | 147 | **1.81×** |
| `vint` int8 | 689,050,860 | **673,136,658** | 673.14 ms | 17,019,615,052 (fq 617) | 605 | **0.040×** |

ET jobs: **fq 851** (mlp_control + dronet + yolov8_nano, one boot),
**fq 852** (vint), **fq 853** (per-op profiling rerun of the first three).
Cycles are the measured quantity; the ms column assumes the 1 GHz nominal target
clock implied by `CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC=1000000` together with the
observed 1000:1 rdcycle-to-mtime ratio.

### Reading the three convnet rows

ExecuTorch is **1.8–2.6× slower than ModelBlaster** on `mlp_control`, `dronet`
and `yolov8_nano`. The ordering is consistent and the gap narrows as the model
gets more convolutional — 2.61× on the tiny ELU-fragmented MLP, ~1.8× on both
convnets — which is what you expect if the delegated RVV kernels are broadly
competitive and the loss is at the segment boundaries (layout conversions,
quantize/dequantize converts, portable-scalar ops, and ExecuTorch interpreter
dispatch between delegates). §4 quantifies that split.

Note this **reverses** the earlier `notes/flow_comparison.md` result, which had
ExecuTorch *ahead* of ModelBlaster on DroNet (13.67M vs 15.87M) on the old
dual-Rocket-Saturn U250 bitstream. ExecuTorch has not regressed: it is at
13.94M here, essentially the same number. **ModelBlaster moved** — from 15.87M
to 7.86M on the same network — via the NHWC/curated-kernel work. The ET number
is a useful fixed reference point precisely because it did not change.

### The `vint` row is not an ExecuTorch win

ExecuTorch is **25× faster than ModelBlaster on ViNT**, and that is a statement
about the ModelBlaster baseline, not about ExecuTorch. MB's 17.0 G cycles break
down as:

| MB op | n | cycles | share |
|---|---:|---:|---:|
| `conv2d_s8_pc` | 65 | 7,711,434,853 | 45.3% |
| `conv2d_f16` | 65 | 5,113,399,855 | 30.0% |
| `linear_s8_pc` | 20 | 1,346,435,276 | 7.9% |
| `sigmoid_s8` | 65 | 987,203,544 | 5.8% |
| `depthwise_conv2d_s8` | 16 | 492,283,179 | 2.9% |

The two conv sets are disjoint by dispatch name (0 overlap), i.e. this is the
mixed-precision ViNT config — one EfficientNet encoder int8, the other fp16 —
not duplicated work. The problem is the *rate*: MB's largest single dispatch,
`conv2d_86` (N=6, IC=16, 32×42, OC=96, 1×1), is 12.4 M MACs in **504,753,595
cycles ≈ 41 cycles/MAC**. On a VLEN=256 int8 machine that is roughly two orders
of magnitude off what the hardware can do, so `conv2d_s8_pc` is evidently not
resolving to a curated kernel. **The ViNT MB baseline is kernel-limited, not
architecture-limited**, and this ExecuTorch run is an independent lower bound
showing ~25× of headroom sitting in it. Even comparing only MB's int8 half
(7.71 G, 65 convs) against ET's *entire* model gives 11.4×.

---

## 4. Per-operator breakdown

Per-operator timings come from a second build with `MB_XNN_PROFILE=ON`
(**fq 859**), which makes the XNNPACK delegate create its runtime with
`XNN_FLAG_BASIC_PROFILING` and emit a `>>, <op>, <cycles>` line per operator per
invoke. Two caveats, both important:

* That build's **whole-model totals are worthless** — profiling logging sits
  inside the `execute()` bracket and inflates it ~500× on a HTIF console. Totals
  in §3 come from the profiling-**off** build; only the per-op deltas come from
  here. Percentages below are stated against the clean total.
* `XNN_FLAG_BASIC_PROFILING` also changes XNNPACK's execution plan (see §5), so
  this is an attribution of a *closely related* plan, not the exact one timed in §3.

**Where the time goes.** "In delegates" is the sum of XNNPACK's own per-op
timings for one invoke across all delegates; the remainder is ExecuTorch
interpreter dispatch plus the portable-scalar ops from §2.

| model | clean warm | in XNNPACK delegates | outside delegates | delegate reshape |
|---|---:|---:|---:|---:|
| `mlp_control` fp32 | 1,512,815 | 67,226 (**4.4%**) | 1,445,589 (**95.6%**) | 83,067 |
| `dronet` int8 | 13,941,927 | 8,290,411 (59.5%) | 5,651,516 (40.5%) | 571,073 |
| `yolov8_nano` int8 | 303,143,017 | 169,948,067 (56.1%) | 133,194,950 (43.9%) | 3,964,546 |

**`mlp_control` is the extreme case: 95.6% of its time is not in XNNPACK at
all.** The four GEMMs cost 67 k cycles combined; everything else is the three
scalar ELUs and the runtime between four delegate boundaries.

### Op-for-op against ModelBlaster

Same networks, same hart. ET rows are XNNPACK op names; MB rows are its own
kernels (`experiments/sweep3net/res_<model>_serialE_base`).

**`mlp_control` fp32** — MB 578,541 total (linear 531,075 / 4 ops, elu 47,466 / 3)

| | ET | MB | |
|---|---:|---:|---|
| the 4 dense layers | **67,226** (`Fully Connected (NC, F32) GEMM`) | 531,075 (`linear`) | **ET 7.9× faster** |
| everything else | 1,445,589 | 47,466 (`elu`) | ET 30× more |

**`dronet` int8** — MB 7,855,964 total

| | ET | MB | |
|---|---:|---:|---|
| the 10 convolutions | **4,063,025** (`Convolution (NHWC, QC8) IGEMM`, n=10) | 7,209,291 (`conv2d_s8`, n=10) | **ET 1.77× faster** |
| max pool | 2,097,137 (n=1) | 239,338 (n=1) | **ET 8.8× slower** |
| NCHW↔NHWC transposes | 1,147,513 (n=14) | — | ET-only cost |
| clamp / convert | 837,853 | 451 (`relu_s8`) | ET-only cost |
| residual adds | 55,882 (n=3) | 356,525 (n=3) | ET 6.4× faster |
| outside delegates | 5,651,516 | ~0 | 3 portable BatchNorms + dispatch |

**`yolov8_nano` int8** — MB 167,132,539 total

| | ET | MB | |
|---|---:|---:|---|
| the 63 convolutions | **77,864,014** (39 `Convolution … IGEMM` + 24 `Fully Connected … GEMM`; the 1×1s lower to GEMM) | 155,470,762 (`conv2d_s8`, n=63) | **ET 2.0× faster** |
| SiLU | **80,226,946** (57 `Sigmoid` + 57 `Multiply`) | 4,794,103 (`silu_s8`, n=57) | **ET 16.7× slower** |
| max pool | 160,741 (n=3) | 3,655,642 (n=3) | ET 22× faster |
| transposes / converts | 9,776,784 | — | ET-only cost |
| outside delegates | 133,194,950 | ~0 | splits, upsamples + dispatch |

**This is the whole story of the 1.8× gap, and it is not the kernels.**
XNNPACK's RVV convolutions beat ModelBlaster's hand-written ones by **1.8×
(dronet) to 2.0× (yolov8_nano)** — the op counts line up exactly (10 vs 10,
63 vs 39+24), so this is a like-for-like comparison. ExecuTorch then gives all
of it back, three ways:

1. **Unfused transcendentals.** XNNPACK decomposes SiLU into `Sigmoid` +
   `Multiply`; its Sigmoid alone is 78.3 M cycles for 57 calls (1.37 M each)
   against ModelBlaster's fused `silu_s8` at 84 k each. That single op is
   **75.4 M cycles — 55% of ExecuTorch's entire 136 M excess on yolov8_nano.**
2. **Layout conversion.** XNNPACK runs NHWC inside an NCHW graph, so every
   delegate boundary pays `Transpose` and `Convert` — 1.15 M on dronet, 9.8 M on
   yolov8_nano, which ModelBlaster (NCHW-int8 end to end) simply does not have.
3. **Everything outside the delegates: 40–96% of total time.** This is the
   portable-scalar ops of §2 plus ExecuTorch interpreter dispatch. ModelBlaster's
   AOT-generated `run_model()` has essentially none of it — its mtime wall
   counter matches its per-op rdcycle sum to ~1%.

Aggregating: on `dronet` ExecuTorch is **ahead by 3.1 M cycles on convolution and
behind by 9.2 M on everything else**. That is the 1.77×.

---

## 5. Numerics

Two different questions, and they need separating.

**(a) Does the RVV hardware compute what the exported program says?** Compared
element-sum-wise against the *same* `.pte` executed by ExecuTorch's host runtime
on x86 with the *same* input (the runner now bakes the real input tensors; the
uartlog confirms with `MB_INPUT_BAKED=<model> n=… bytes0=…`).

| model | output | HW checksum | ET host (x86) | abs diff | rel diff |
|---|---:|---:|---:|---:|---:|
| `mlp_control` fp32 | 0 | 14.791794 | 14.791794 | 4.2e-07 | 2.8e-08 |
| `dronet` int8 | 0 | -0.030273 | -0.030273 | 4.4e-07 | 1.4e-05 |
| `dronet` int8 | 1 | 0.478202 | 0.478202 | 3.4e-07 | 7.2e-07 |
| `vint` int8 | 0 | 8.653941 | 8.653941 | 1.5e-07 | 1.8e-08 |
| `vint` int8 | 1 | 8.706648 | 8.706648 | 1.7e-07 | 2.0e-08 |
| `yolov8_nano` int8 | 0 | -448952.978705 | -448818.939519 | **134.04** | **3.0e-04** |
| `yolov8_nano` int8 | 1 | -97813.484649 | -97843.815654 | **30.33** | **3.1e-04** |
| `yolov8_nano` int8 | 2 | -19112.570846 | -19133.433173 | **20.86** | **1.1e-03** |

`mlp_control`, `dronet` and `vint` agree with the host to **print precision** —
the differences are exactly the rounding of `%f`'s 6 decimals, so RVV is
reproducing the host as exactly as this can resolve. **`yolov8_nano` does not**:
~3e-4 relative on sums over 57,600 / 14,400 / 3,600 elements, three orders of
magnitude above print noise.

A checksum can hide compensating errors, so a second run (fq 853) dumped
per-element samples (`MB_ET_SAMPLE_OUTPUT`, 128 positions per output: the first
64 plus a strided sweep):

| model | out | sampled | max abs \|HW−host\| | mean | elements differing |
|---|---:|---:|---:|---:|---:|
| `mlp_control` fp32 | 0 | 4 | 4.0e-07 | 3.0e-07 | 0 (all within print precision) |
| `dronet` int8 | 0,1 | 1,1 | 4.4e-07 | — | 0 (within print precision) |
| `yolov8_nano` int8 | 0 | 128 | **0.368** | 0.048 | **40 / 128** |
| `yolov8_nano` int8 | 1 | 128 | **0.648** | 0.078 | **55 / 128** |
| `yolov8_nano` int8 | 2 | 128 | **0.640** | 0.089 | **59 / 128** |

**The mechanism is pinned: 100% of the yolov8_nano differences are exact integer
multiples of that head's output quantization step** (0.122745, 0.129619,
0.127989 for the three heads), at ratios 1–5. So the RVV int8 path is producing
*integer* output values that differ from the x86 host by one to five
quantization steps on 30–46% of elements. That is requantization / accumulation-
order divergence inside the quantized convolution, **not** fp noise, not a
layout bug, and not print precision. It is the same class the earlier
KernelBench sweep flagged (*all* its checksum mismatches were conv /
conv-transposed; activations, matmul and softmax matched exactly), so it is the
expected risk area rather than a surprise. Which side is "right" is not
established — see §6.

**The measurement is exactly reproducible.** fq 857 re-ran the *identical* ELF
from fq 851 and reproduced **every cycle count and every checksum bit-for-bit**
(1,512,815 / 13,941,927 / 303,143,017; checksums 14.791794, −0.030273, 0.478202,
−448952.978705, −97813.484649, −19112.570846). ExecuTorch on this hardware is
deterministic.

### …but `yolov8_nano`'s result depends on the BUILD, and one build is host-exact

Across four runs of the same `.pte` on the same input, `mlp_control` and
`dronet` returned **byte-identical** checksums every time (14.791794 /
−0.030273 / 0.478202). `yolov8_nano` did not:

| run | build | out0 | out1 | out2 |
|---|---|---:|---:|---:|
| fq 851 | clean, `ITERS=3` | −448952.978705 | −97813.484649 | −19112.570846 |
| fq 857 | **same ELF as fq 851** | −448952.978705 | −97813.484649 | −19112.570846 |
| fq 853 | profiling, `ITERS=2`, sampled output | −449113.776738 | −98100.851909 | −19073.278003 |
| fq 859 | profiling, `ITERS=2`, `LOG_LEVEL=Info` | **−448818.939519** | **−97843.815654** | **−19133.433173** |
| — | **x86 host reference** | **−448818.939519** | **−97843.815654** | **−19133.433173** |

Three observations, and together they change the diagnosis:

- **It is deterministic per binary** — fq 857 reproduces fq 851 bit-for-bit,
  cycles included. So this is not a race and not uninitialised *randomness*.
- **It varies with things that must not affect arithmetic.** fq 853 and fq 859
  differ only in log level and whether the sampled-output code is compiled in —
  pure code-size/layout changes — and they give different answers.
- **fq 859 matches the x86 host to all six printed digits on all three
  outputs.** So the RVV micro-kernels *can* produce the host-exact result; the
  hardware and the kernels are not intrinsically wrong.

That combination — deterministic per binary, sensitive to memory layout, one
layout exactly right — is the signature of a **memory bug (an uninitialised or
stale read, or buffer aliasing at a delegate boundary)**, not of a rounding
convention. It is confined to `yolov8_nano`, the model whose graph is cut by 8
`split_with_sizes_copy` and 2 `upsample_nearest2d` portable ops between
delegates — exactly where such a bug would live. That the errors land on exact
quantization-step multiples is then a consequence of the corrupted value being
an int8 that gets dequantized, not evidence of a rounding difference.

This supersedes the first reading of the per-element data (that the divergence
was requantization rounding). **`yolov8_nano` results from this ExecuTorch flow
should not be trusted until it is fixed**; `mlp_control`, `dronet` and `vint`
are unaffected and host-exact. Note the cycle counts are not in question — the
clean-build timings reproduce exactly (fq 857) — so §3 and §4 stand.

**(b) How far is each model from its PyTorch fp32 golden?** This is the
quantization question and is a property of the export, not of the hardware
(measured on the host, `pte/<tag>.json`, same input):

| model | quant | max abs err | mean abs err | cosine | numel |
|---|---|---:|---:|---:|---:|
| `mlp_control` | fp32 | 6.68e-06 | 1.87e-06 | 0.99999999999 | 4 |
| `dronet` | int8 | 6.60e-03 | 3.41e-03 | 0.99991 | 2 |
| `yolov8_nano` | int8 | 3.069 | 0.207 | 0.99962 | 75,600 |
| `vint` | int8 | 0.568 | 0.124 | 0.99770 | 21 |

`mlp_control` fp32 lands at max_abs_err 6.68e-06, **the same value ModelBlaster's
own in-binary verify reports** (`MODELBLASTER_VERIFY [mlp_control] max_abs_err=6.67572021e-06`)
— the two flows produce the same fp32 result on the same input. The int8 rows
are XNNPACK pt2e symmetric per-channel PTQ and are **not** directly comparable to
MB's verify numbers, which are reported in each model's own output dtype
(MB dronet reports `max_abs_err=4` in the int8 output domain; MB yolov8_nano is
bit-exact `max_abs_err=0` against its own int8 reference). Comparing the two
flows' *quantization quality* would need both driven from one shared golden;
that was not done here.

---

## 6. What could NOT be established

- **The `yolov8_nano` memory bug is localized but not found.** §5 establishes
  *that* it is layout-dependent and *where* to look (delegate boundaries around
  the 8 `split_with_sizes_copy` / 2 `upsample_nearest2d` portable ops), and that
  one build is host-exact — but the actual offending buffer was not identified.
  Doing so needs an op-by-op intermediate dump, or a build with the ET pool
  poisoned to a non-zero pattern to make the stale read obvious. Not attempted.
- **No per-operator data for `vint`.** The profiling build covers only the three
  smaller models. ViNT is where the delegate/portable split matters most (69 vs
  288), so the attribution of its 673 M cycles between XNNPACK RVV and
  portable-scalar ops is **not** measured. Its `.pte` is 24 MB and the profiling
  log volume would be large; it was left out for time, not for any technical
  reason.
- **The per-op attribution is from a slightly different execution plan.**
  `XNN_FLAG_BASIC_PROFILING` changes what XNNPACK does (§5 proves it: the
  numerics change). §4's split between delegated and non-delegated time is
  therefore approximate for the clean build, though the delegate `invoke`
  markers and the per-op sums agree to <0.01% within the profiling build, so it
  is self-consistent.
- **Multi-hart ExecuTorch is untested here, and expected to be unusable.** Every
  run pins to one hart with a 1-thread pool. The threadpool cannot be given
  affinity, so on this heterogeneous bitstream any pool worker landing on hart
  0/1 executes vector code on a tile without a vector unit. Making ExecuTorch
  use harts 2 *and* 3 requires an affinity-aware pthreadpool, which was not
  attempted. (Prior work on the homogeneous bitstream also found 2-core ET ~175×
  slower on LeNet and faulting in XNNPACK weight packing on MobileNetV2.)
- **ELF size / load cost is not characterized.** The ET ELFs are 91–113 MB
  against ModelBlaster's few MB, essentially all static XNNPACK. That is a real
  deployment cost, and it inflates FireSim setup time, but it was not measured
  as a metric.
- **No ExecuTorch-vs-ModelBlaster comparison of quantization quality**, for the
  reason in §5(b): the two flows verify against different references.
- **`vint`'s ModelBlaster baseline was taken as found.** The conclusion that
  `conv2d_s8_pc` is falling back to a slow kernel is inferred from cycles/MAC,
  not from a kernel-pick dump. Confirming it means re-running MB vint with the
  kernel-pick gate that `experiments/shard_dim/scripts/build_one.sh` applies to
  Gemmini.
- **Scaled variants (`<net>_s[a-h]`) were not swept.** Only the four base models.

---

## 7. Porting work — what had to change, and why

The `samples/executorch` tree targeted a **dual-Rocket-Saturn** bitstream where
every hart had a vector unit. `f2_quad_hetero_norose_tacit_q31_60mhz` is
heterogeneous: harts 0,1 are Rocket+Gemmini `rv64imafdc` with **no vector
unit**; harts 2,3 are Rocket+Saturn `V256D128`. Four things broke, each of them
silently or misleadingly.

1. **XNNPACK's VLEN probe runs before `main()` and trapped on the boot hart.**
   *(fq 850: `mcause: 2, Illegal instruction, mtval 0xc0075d7`, immediately
   after the Zephyr banner.)* `mtval` decodes as `vsetvli a1, zero, e8, m1, ta,
   ma` — exactly the probe in `hardware-config.c`. It runs that early because
   ExecuTorch constructs its backend in a **C++ static initializer**
   (`XNNPACKBackend.cpp:207`, `auto cls = XnnpackBackend();`) whose constructor
   calls `xnn_initialize()`. Static initializers run on the boot hart — hart 0,
   with no vector unit — so **no amount of application-level thread pinning can
   fix it**; the trap precedes any application code.
   *Fix:* build-time VLENB. `hardware-config.c` takes `-DXNN_RISCV_VLENB=<n>`
   (32 = Saturn VLEN 256/8) in place of the probe — it is the only vector
   instruction on the `xnn_initialize()` path. Plumbed as `-DMB_XNN_VLENB` in
   `executor_runner/CMakeLists.txt`, defaulted to 32 by `build_et_elf.sh`, and
   gated: an unset value aborts the build.

2. **`hardware-config.c` would not even assemble.** RoSE's Zephyr has
   `CONFIG_RISCV_V_KERNEL_ONLY`, which strips `v` from the *global* `-march` and
   adds it back only per-file. `hardware-config.c` is not a micro-kernel source,
   so it never got the per-file flag and its `vsetvli` failed at assembly time
   (`unrecognized opcode … extension 'v' required`). On a target whose global
   `-march` already carries `v` it assembles by accident, which is why this had
   never surfaced. *Fix:* add that one file to the `-march=rv64gcv` per-source set.

3. **The XNNPACK threadpool would have scheduled vector work onto Gemmini
   harts.** ExecuTorch sizes its pool from `sysconf(_SC_NPROCESSORS_ONLN)` =
   `CONFIG_MP_MAX_NUM_CPUS` = 4 (`extension/threadpool/threadpool.cpp:101`) and
   pthreadpool workers carry no affinity. The samples' usual answer —
   `MP_MAX_NUM_CPUS=1`, making every `parallelize_*` run inline — **hangs on this
   bitstream**, because Zephyr's SMP boot spin-waits for every configured hart.
   *Fix:* boot all 4 harts, then recover the same inline behaviour with
   `-DMB_ET_THREADS=1` (`ThreadPool::_unsafe_reset_threadpool(1)`) and run the ET
   region on a thread pinned to hart 2 via `-DMB_ET_PIN_HART=2`.
   `k_thread_cpu_pin()` on a *running* thread is a silent no-op, so the runner
   creates the thread `K_FOREVER`, pins, then starts — the idiom from
   `modelblaster/harness_tacit/src/main.c`. Both are asserted at runtime in the
   uartlog: `MB_ET_THREADPOOL from=4 to=1 ok=1` and `MB_ET_HART=2`.

4. **Inputs were not real.** The stock runner fills every float input with 1.0,
   which makes comparison against a PyTorch golden meaningless. `gen_multi_pte.py`
   now bakes the model's actual input tensors alongside the `.pte` and the runner
   feeds them (`--io <tag>=<tag>.io.npz`; confirmed by `MB_INPUT_BAKED` in the
   uartlog).

Two further traps found and fixed, both of the silent-wrong-artifact kind:

- **Wrong ModelBlaster tree imported.** The ExecuTorch env carries an editable
  install (`_editable_impl_modelblaster.pth`) pointing at an older ModelBlaster
  tree, and RoSE's `modelblaster/` has no top-level `__init__.py`, so it can only
  be a *namespace* package — and Python prefers a regular package found anywhere
  on `sys.path` over a namespace portion at `sys.path[0]`. A plain
  `sys.path.insert(0, MB_REPO)` therefore silently exported the **wrong tree's
  models**, which differ (`mlp_control` 15, `dronet` 20, `yolov8_nano` 162 diff
  lines). The exporter now binds the package root explicitly and asserts the
  resolved file lies under `MB_REPO`, printing `[provenance] <model> <- <path>`.
  The session's first three exports were made with the wrong tree and discarded.
- **A profiling build produced no profile.** The `>>, <op>, <cycles>` lines —
  the only reason to build with `ENABLE_XNNPACK_PROFILING` — went through
  `ET_LOG(Info)`, so at the sample's usual `EXECUTORCH_LOG_LEVEL=Error` they are
  compiled out and the profiling build silently emits **nothing** (observed:
  fq 853, 0 op lines). Raising the level to `Info` gets them back but also
  unleashes every other Info/Debug line in ExecuTorch; on FireSim the console is
  HTIF at a few KB/s, so the run floods and approaches the queue timeout (fq 859
  took 44 min against a 50 min limit). `XNNProfiler.cpp` now emits those lines
  with `printf`, so they survive at any log level and nothing else comes with
  them.
- **Shared ExecuTorch build directory.** The sample defaults `ET_BUILD_DIR_PATH`
  to `third-party/executorch/cmake-out`, *inside the source tree*, so every west
  build dir shares it — a profiling build and a clean build overwrite each
  other's ExecuTorch/XNNPACK objects and you link whichever ran last, with no
  sign in the log. Each build now gets its own via `-DET_BUILD_DIR_PATH`.

Build-time gates were added for each of these (`build_et_elf.sh`), because every
one fails silently: hart count, `CONFIG_RISCV_ISA_EXT_V`, `MB_ET_PIN_HART`,
`MB_ET_THREADS`, `XNN_ENABLE_RISCV_VECTOR=1`, `XNN_RISCV_VLENB` set, and
vector instructions present in the linked ELF. They read **build artifacts**, not
the build log — an incremental build does not re-run cmake configure, so the
`message(STATUS)` lines are absent from a rerun even though the settings hold.

### Files

| path | what |
|---|---|
| `experiments/executorch/scripts/export_mb_models.py` | MB model → `.pte` + host golden + delegate/portable report |
| `experiments/executorch/scripts/export_all.sh` | driver; pins the separate ET env |
| `experiments/executorch/scripts/build_et_elf.sh` | multi-model Zephyr ELF for the quad-hetero bitstream + gates |
| `experiments/executorch/scripts/et_submit.sh` | scp → `fq submit` → collect (mirrors `sweep_submit.sh`) |
| `experiments/executorch/scripts/et_parse.py` | uartlog → per-model results, joined against the MB baseline |
| `samples/executorch/executor_runner/riscv_executor_runner.cpp` | hart pinning, threadpool clamp, baked inputs |
| `samples/executorch/executor_runner/firesim_quad_hetero.conf` | Zephyr overlay for this bitstream |
| `samples/executorch/executor_runner/CMakeLists.txt` | `MB_ET_PIN_HART`, `MB_ET_THREADS`, `MB_XNN_VLENB` |
| `samples/executorch/model/gen_multi_pte.py` | bakes real inputs beside each `.pte` |
| `third-party/executorch/.../XNNPACK/CMakeLists.txt` | `-march=rv64gcv` for `hardware-config.c` |
| `third-party/executorch/.../XNNPACK/src/configs/hardware-config.c` | `XNN_RISCV_VLENB` build-time VLEN |
| `third-party/executorch/backends/xnnpack/runtime/profiling/XNNProfiler.cpp` | per-op lines via `printf`, not `ET_LOG(Info)` |
| `experiments/executorch/patches/third_party_rvv_hetero.patch` | **the three third-party edits, exported** — see below |


> **The three `third-party/` edits are NOT tracked by git.** `third-party/{executorch,XNNPACK}`
> are gitlinks, so `git status` in `zephyr-chipyard-sw` shows nothing for them and
> a fresh `git submodule update` silently discards all three — after which the
> build either fails to assemble or, worse, still links and measures the wrong
> thing. They are exported to
> `experiments/executorch/patches/third_party_rvv_hetero.patch` (with a README
> explaining each) and must be re-applied on any fresh checkout.

---

## 8. Reproducing

```bash
E=/scratch/dima/rose-infra/RoSE/experiments/executorch
bash $E/scripts/export_all.sh mlp_control:fp32 dronet:int8 yolov8_nano:int8 vint:int8
bash $E/scripts/build_et_elf.sh et3 mlp_control_fp32 dronet_int8 yolov8_nano_int8
bash $E/scripts/et_submit.sh et3
python3 $E/scripts/et_parse.py $E/res/et3
```

### Environment — nothing shared was modified

The RoSE in-tree conda env
(`soc/sw/xpu-rt/zephyr-chipyard-sw/tools/miniforge3/envs/zephyr`) is **torch 2.13
with no executorch installed**, and executorch 1.0.1 pins `torch>=2.9,<2.10`.
It was **not** touched. Instead:

* Host-side export uses a **separate venv**, `experiments/executorch/.venv`,
  created with `--system-site-packages` on top of an existing unrelated
  ExecuTorch env
  (`/scratch2/dima/misc_sw/XPU-RT/zephyr-chipyard-sw/tools/miniforge3/envs/zephyr`
  — executorch 1.0.1 / torch 2.9.0+cpu / torchao 0.14.0). Exactly two packages
  were installed **into the venv**: `efficientnet_pytorch` and
  `warmup_scheduler`, both needed to unpickle the ViNT checkpoint.
* The ELF build uses the **RoSE in-tree env** for west/cmake/ninja/Zephyr SDK
  (cmake 3.31.8 satisfies executorch's `>=3.29`), taking only `flatc` and
  `-DPYTHON_EXECUTABLE` from the ExecuTorch env.
* `third-party/{executorch,XNNPACK}` were tracked by the superproject but never
  checked out. They were populated from a local mirror **at exactly the commits
  the superproject records** (executorch `54af43112` branch `zephyr-1.0.1`,
  XNNPACK `43e0628fb5`); their `.git` gitlink files were removed, so they are
  plain directories rather than initialised submodules.
