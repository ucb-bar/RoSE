# rose_nav_cosim — on-SoC drone nav flying the Isaac warehouse gate course

Reproducible bundle for the RoSE <-> IsaacLab spike lockstep co-sim in which the
on-SoC flight stack (**fused vision + EKF + TinyMPC -> 4 motor thrusts**) flies the
Isaac warehouse gate course.

**Full step-by-step guide:** [`../../docs/REPRODUCE_ROSE_NAV_COSIM.md`](../../docs/REPRODUCE_ROSE_NAV_COSIM.md)

## Contents
| Path | What |
|------|------|
| `model/rvv_f16/` | Committed fused RVV model kernels (int8 per-channel encoders + fp16 tail + fp16 low-dim). Zero-dependency guest input. |
| `model_src/` | `graph.json` + `weights.npz` — regen source for the RVV codegen step. |
| `calib/calib_real.pkl` | Real gate camera frames used for int8 calibration (regen only). |
| `env.sh` | Sources the in-tree Zephyr toolchain (conda env + SDK). |
| `rvv_nav.conf` | Zephyr Kconfig fragment enabling RISC-V Vector. |
| `build_guest.sh` | Builds the `rose_fused_mpc` guest ELF (spike_riscv64, RVV). |
| `run_cosim.sh` | Launches the co-sim (`MODE=short` sanity / `MODE=full` 4-gate course). |
| `regen_model.sh` | Provenance: how `model/rvv_f16` was generated (needs external checkpoint). |

## Quick start
```bash
source experiments/rose_nav_cosim/env.sh          # in-tree toolchain (bootstrap first)
experiments/rose_nav_cosim/build_guest.sh         # -> build_guest/zephyr/zephyr.elf
experiments/rose_nav_cosim/run_cosim.sh           # MODE=short by default
```

## Model provenance
The model is generated from an **external** collaborator checkpoint
(`fused_bc_warehouse_v12_mixed_cnn`) that is **not in this repo**. Because that
checkpoint is unavailable in a fresh clone, the generated kernels are committed
directly (`model/rvv_f16`) so the guest builds with no Python/PyTorch step.
