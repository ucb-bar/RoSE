# Global curated kernel library

Hand-written or expert-curated kernel implementations for use across all
models. These are model-agnostic — they implement the standard kernel
signature from `reference_kernels.py` with no shape-specific assumptions.

## **Not** an auto-generated cache

Don't confuse this directory with `modelblaster/examples/<model>/<quant>/cache/<backend>/`.

| | `modelblaster/kernels/` (this dir) | `modelblaster/examples/*/cache/*/` |
|---|---|---|
| **Content** | Curated, hand-written or post-LLM-promoted kernels | Per-model LLM-optimized kernels + reference seeds |
| **Origin** | Authored by humans, or promoted from a cache after a kernel proves itself across models | Generated/cached by `generate_kernels.py` for one specific model |
| **Scope** | Model-agnostic — implements the abstract kernel signature, must work for any shape | Tuned to one model's specific layer shapes |
| **Lifecycle** | Tracked in git, committed deliberately | Regenerable; the pipeline rewrites it if missing |
| **Role in LLM loop** | Optional **seed** for the LLM optimizer (`--seed-from-curated`) and a fast-path probe before LLM is invoked | Output of the LLM loop; gets promoted *into* this dir if it stays winning across models |

If you delete `modelblaster/examples/<m>/<q>/cache/`, the pipeline regenerates
it. If you delete `modelblaster/kernels/`, you lose the curated seeds — they
won't come back unless someone re-authors them.

## Directory layout

```
modelblaster/kernels/
  <target>/
    <backend>_<op>_<algorithm>.c    # one file per (target, op, algorithm)
```

The filename convention is identical to the per-model cache so the probe
logic in `generate_kernels.py` can resolve files by the same key:

```
rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c
rvv/rvv_conv2d_s8_rvv_widening_oc.c
scalar/scalar_conv2d_s8_direct.c
gemmini/gemmini_conv2d_s8_gemmini_tiled_conv.c     # hardware im2col via tiled_conv_auto (preferred)
gemmini/gemmini_conv2d_s8_gemmini_im2col_full_C.c  # software im2col + tiled_matmul_auto, bit-exact Q0.31
```

## How it integrates with the pipeline

Pass `--global-curated-dir modelblaster/kernels` (or set `GLOBAL_CURATED_DIR`) to
`generate_kernels.py` / any `run.sh`. The pipeline then:

1. For each op and each algorithm candidate, checks this directory first.
2. If a curated file exists, verifies it against the model's actual shapes on
   spike. If it passes, the file is promoted into the per-model cache and
   used — no LLM call needed.
3. Curated and LLM-cached kernels compete in the same fastest-wins ranking,
   so the faster of the two is always selected.
4. If the curated kernel fails verification for a particular model (e.g. a
   shape it doesn't support), the pipeline falls through to the per-model
   cache or LLM generation as normal.

## Adding a curated kernel

1. Write a `.c` file implementing the standard kernel signature (copy from
   `reference_kernels.py` `KernelSpec.signature` for the op).
2. Name it `<backend>_<op>_<algorithm>.c` under the matching target subdir.
3. The algorithm name must be registered as an `AlgorithmCandidate` in
   `reference_kernels.py` (add it if new) so the pipeline knows to probe it.
4. Test by running any model's `run.sh` with
   `BACKEND=llm GLOBAL_CURATED_DIR=modelblaster/kernels` and confirming the log
   shows `curated HIT` for the op.

## Source tagging

Each file should open with:
```c
/* source: curated */
/* algorithm: <algorithm_name> */
/* origin: <brief description, e.g. "XNNPACK-style vsmul/vnclip requantize"> */
```

The pipeline does not parse these comments; they are for human readers.
