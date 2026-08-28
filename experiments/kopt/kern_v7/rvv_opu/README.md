# Curated kernels — `rvv_opu` backend

RVV + Saturn OPU (Outer Product Unit, integer-only) hand-written kernels.

## Backend

See `modelblaster/pipeline/backends.py::RVV_OPU` and
`modelblaster/cores/saturn_opu/include/saturn_opu.h` for the programming
model. The OPU exposes 4 matrix registers (m0..m3) and four custom
.insn-encoded operations (`OPMVINBCAST`, `VOPACC`, `VMV_VR`, `VMV_RV`)
that overlay the V opcode (0x57).

The integer OPU computes `m += vs2 ⊗ vs1` (outer-product MAC) where
`m` is i32 and `vs1`/`vs2` are i8 — i.e. the kernel that maps cleanly
to this hardware is **gemm with int8 inputs and int32 accumulator**.

## File naming

```
modelblaster/kernels/rvv_opu/rvv_opu_<op>_<algorithm>.c
```

The pipeline picks these up when `--global-curated-dir modelblaster/kernels`
is passed and the `AlgorithmCandidate` in `reference_kernels.py`
declares `target_affinity=("rvv_opu",)` for the matching op/algorithm.

## Reference upstream kernels

Where to look for canonical OPU usage (Saturn submodule, branch
`origin/opu-fp8`, path `generators/saturn/benchmarks/`):

| upstream benchmark | purpose | translates to |
|---|---|---|
| `opu-gemm/kernel.h::i8_mm_bme_sq` | i8 square + general matmul with VOPACC | `rvv_opu_linear_s8_*.c`, `rvv_opu_matmul_s8_*.c` |
| `opu-mt-gemm/main.c` | multi-tile gemm | `rvv_opu_linear_s8_mt.c` |
| `opu-m4-memcopy/`, `opu-lm2-memcopy/` | mvin/mvout patterns | scratchpad-blocked variants |
| `opu-m4-transpose/` | matrix transpose via OPU | conv2d im2col / weight-pack helpers |

These live in chipyard's saturn submodule, not in this repo. Use
`git -C generators/saturn show origin/opu-fp8:<path>` to fetch them.

## Status

- **2026-05-16**: scaffolding only. No curated kernels yet — first
  candidate will be `rvv_opu_linear_s8_outerprod_acc.c` ported from
  `opu-gemm/kernel.h::i8_mm_bme_sq`.

## Build / verify

```
TARGET=rvv_opu BACKEND=reference RUNNER=firesim \
  GLOBAL_CURATED_DIR=/abs/path/to/zephyr-chipyard-sw/modelblaster/kernels \
  bash modelblaster/examples/<model>/run.sh
```

`RUNNER=spike` will currently fail on the first OPU instruction —
upstream spike has no OPU decoder. Run on a Saturn OPU bitstream
(e.g. `REFV256D128DualRocketSaturnOPUGemmini32x32Q31WsConfig` from
chipyard, or `OPUV256D128ShuttleConfig` for a Shuttle-side build).
