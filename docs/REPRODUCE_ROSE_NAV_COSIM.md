# Reproduce: on-SoC drone nav flying the Isaac warehouse gate course

Step-by-step guide to reproduce, **from a fresh git clone**, the RoSE ↔ IsaacLab
spike lockstep co-sim in which the **on-SoC flight stack** (fused vision + EKF +
TinyMPC → 4 motor thrusts) flies the Isaac warehouse **gate course**.

- **Architecture / wire contract / gotchas:** [`ROSE_FUSED_GATE_COSIM.md`](ROSE_FUSED_GATE_COSIM.md) (this doc is the *runbook*; that one is the *design*).
- **Reproducible bundle:** [`../experiments/rose_nav_cosim/`](../experiments/rose_nav_cosim/) — committed model + parameterized build/run scripts.
- **Companion memory:** `rose-fused-gate-cosim`, `rose-isaac-cosim-flow`, `rose-modelblaster-spike-build`, `rose-flight-controller-hw-validation`.

---

## 0. What flies

The **entire flight stack runs on the SoC (spike)**: the fused vision net
(int8 per-channel CNN/depth encoders + fp16 tail + fp16 low-dim) → an EKF state
estimate → TinyMPC (yawfix params) → 4 motor thrusts. Isaac owns only the
photoreal warehouse gate scene, the cameras, the rigid-body dynamics, and gate
scoring. Lockstep: every control tick the two sides exchange sensors ↔ thrusts.

```
Isaac warehouse (photoreal gates, cameras, dynamics)
   │  front_grey 0x11 (DMA, int8 60x90)   ▲  4 motor thrusts 0x20
   │  tof_cross  0x41 (reqrsp, int8[256]) │  (float32[4])
   ▼  lowdim     0x42 (reqrsp, f32[21])   │
SoC guest: fused_full  →  EKF  →  TinyMPC  →  motor mixer
```

---

## 1. Prerequisites (heavy, one-time — NOT in the repo)

These are large local/external installs. A fresh clone does **not** carry them;
install/point at them once. (All were pre-validated for this experiment; see the
memory notes for the exact provenance.)

**Prereq A — IsaacLab + `env_isaaclab` conda env** (the Isaac side).
- IsaacLab checkout (reference: `/scratch2/dima/IsaacLab`).
- conda env `env_isaaclab` with Isaac Sim (reference python: `/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python`).
- Our warehouse gate task lives in-tree at `soc/sw/xpu-rt/sims/isaaclab_tasks/warehouse_nav/`
  and the bridge env at `deploy/hephaestus/envs/warehouse_fused_nav/warehouse_thrust_env.py`.
- See memory `rose-isaac-cosim-flow` and `docs/ROSE_ISAAC_COSIM_FLOW.md`.

**Prereq B — Zephyr / RISC-V toolchain bootstrap** (the guest side).
- The in-tree bootstrap installs a `zephyr` conda env + west + the Zephyr SDK
  **inside** the `zephyr-chipyard-sw` submodule (`tools/miniforge3`, `tools-manual/`).
- Run the in-tree bootstrap (memory `rose-zephyr-fork-and-intree-build`; script
  `soc/sw/xpu-rt/zephyr-chipyard-sw/scripts/…`, plus `activate_conda.sh` /
  `set_envvars_sdk.sh`). This is a multi-GB download (conda + SDK).
- Sanity: `source experiments/rose_nav_cosim/env.sh` must print a `west` and
  `cmake` under `…/tools/miniforge3/envs/zephyr/bin` and a non-empty `ZEPHYR_BASE`.

**Prereq C — `rose_spike_sim`** (the RoSE spike bridge binary).
- Built from `soc/src/main/cc/rose_spike/build.sh` → `soc/sim/rose_spike_sim`
  (+ `librose_spike.so`). This pulls in Chipyard/Spike deps and is a heavy build.
- Reference binary in the working tree is ~176 MB. Gitignored (regenerable).

> **Disk:** keep clones and builds on a large volume (e.g. `/scratch`). Do **not**
> build on a full root filesystem.

---

## 2. Clone + submodules

```bash
git clone <RoSE-remote> RoSE            # or: git clone /path/to/local/RoSE
cd RoSE
git checkout rose-2-dev
git submodule update --init --recursive soc/sw/xpu-rt
# (soc/sim/chipyard and the Isaac/XNNPACK/executorch submodules are only needed
#  for Prereq A/C; init them per those flows.)
```

The committed submodule cascade tip for this experiment (verify with `git log`):

| Repo | Commit | Contains |
|------|--------|----------|
| RoSE (super) | *this branch tip* | `experiments/rose_nav_cosim/` (model+scripts), docs, xpu-rt pointer |
| `soc/sw/xpu-rt` | bump | zephyr-chipyard-sw pointer |
| `…/zephyr-chipyard-sw` | bump | `rose_fused_mpc` guest + **velocity-frame derotation (coordinated-turn) fix** |
| `…/tinympc` | `c1192f3` | `quadrotor_yawfix_params.hpp` |

---

## 3. Build the fused RVV model

**Nothing to do** — the generated RVV kernels are committed at
`experiments/rose_nav_cosim/model/rvv_f16/` and build with **zero** Python/PyTorch
dependency. (Provenance / how to regenerate from the external checkpoint:
`experiments/rose_nav_cosim/regen_model.sh`. The `fused_bc_warehouse_v12_mixed_cnn`
checkpoint and the collaborator model source live under `/scratch/agustin` and are
**not** in the repo, which is exactly why the kernels are committed.)

---

## 4. Build the guest ELF

```bash
source experiments/rose_nav_cosim/env.sh      # Prereq B toolchain
experiments/rose_nav_cosim/build_guest.sh     # -> experiments/rose_nav_cosim/build_guest/zephyr/zephyr.elf
```

This compiles `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_fused_mpc` for
`spike_riscv64` with RVV, consuming the committed model dir. Reproducible defaults:
gate-centerline altitude **2.0 m**, **20 Hz** vision (`FUSED_VISION_DIV=10`),
yaw-rate command, `CTRL_ITERS=5000 SETTLE_ITERS=200`. Expect a ~3.8 MB `zephyr.elf`
and `-march=…zfh…zvfh…` in the build.

---

## 5. Build the spike bridge (Prereq C, if not already built)

```bash
bash soc/src/main/cc/rose_spike/build.sh      # -> soc/sim/rose_spike_sim
```

---

## 6. Run the co-sim

```bash
# short sanity (settle + a few gate ticks):
ISAAC_PY=/path/to/env_isaaclab/bin/python \
  experiments/rose_nav_cosim/run_cosim.sh          # MODE=short (default)

# full 4-gate course:
MODE=full ISAAC_PY=/path/to/env_isaaclab/bin/python \
  experiments/rose_nav_cosim/run_cosim.sh
```

The script boots Isaac (`run_sync_only.py` on
`config_gym_WarehouseThrustEnv-v0.yaml`), waits for `listening on`, then runs the
guest under `run_spike_rose_lockstep.sh`. Key env (all set by the script):

| Var | Value | Meaning |
|-----|-------|---------|
| `ROSE_WH_SEED` | `1000` | warehouse gate-course seed (determinism) |
| `ROSE_FREEZE` | `1` | freeze Isaac while the SoC computes (true lockstep) |
| `ROSE_ISA` | `rv64gcv_zicntr_zihpm_zfh_zvfh` | spike ISA (RVV + fp16) |
| `ROSE_ISAAC_CAMERA` | `1` | real Isaac camera frames → SoC (chase + FPV dumps) |
| `ROSE_GYM_ENV` | `WarehouseThrustEnv-v0` | motor-thrust warehouse env |
| `ROSE_VIDEO_DECIM` | `2` | frame-dump decimation |

Artifacts land in `experiments/rose_nav_cosim/run_out/` (`sync.log`, `sim.log`,
`traj.csv`, `chase_frames/`, `fpv_frames/`).

---

## 7. Expected result

- **Controlled gate-nav.** The drone lifts to ~2.0 m, holds heading during the
  settle window, then does a **coordinated turn** toward each gate bearing and
  cruises through. With the committed coordinated-turn fix this reaches **3/4**
  gates on seed 1000 (`grep '\[GATE\]' run_out/sync.log`).
- **No traps.** `sim.log` shows the fused model + EKF + TinyMPC ticking with no
  `Illegal`/`mcause` faults; RVV kernels golden-verify at build.

---

## 8. Gotchas (why the config is what it is)

1. **Real-data calibration is mandatory.** The int8 encoders must be calibrated on
   **real gate frames** (`calib/calib_real.pkl`). Calibrating on synthetic/hover
   data saturates the head on the gate course.
2. **Pure-int8 fails; the winning config is int8 encoders + fp16 tail + fp16
   low-dim.** The low-dim optical-flow channel has an outlier that overflows int8,
   and the int8 head saturates on real data. fp16 for tail + low-dim recovers
   fp32-equivalent flight (see `ROSE_FUSED_GATE_COSIM.md` results).
3. **The 4 flight-control fixes** (cumulative, all committed in `rose_fused_mpc`):
   (a) yaw root-cause fix; (b) desired-velocity **alignment gate** + turn-before-go
   (don't translate until roughly pointed at the gate); (c) fly at **gate-centerline
   altitude (2.0 m)**, not spawn altitude; (d) **velocity-frame derotation
   (coordinated turn)** — the EKF emits *world*-frame velocity but TinyMPC's K is
   hover-linearized at yaw 0 (body frame), so the velocity feedback is derotated
   into the body-heading frame (`err[6]=vfwd-cmd`, `err[7]=vlat→0`). Without (d) the
   nose turns while the body keeps sliding the old world direction.
4. **Vision rate = 20 Hz** (`FUSED_VISION_DIV=10` at the 200 Hz control loop) — the
   deployable rate on the current SoC clock; faster starves the control loop.
5. **Determinism:** always `ROSE_WH_SEED=1000` + `ROSE_FREEZE=1`. Freeze makes the
   result independent of host wall-clock jitter.

---

## 9. Reproduction coverage (validated 2026-08-10)

**Validated from a fresh clone** (temp clone on `/scratch`, submodules sourced from
local so the unpushed cascade materializes):
- superproject clone carries the full `experiments/rose_nav_cosim/` bundle;
  `model/rvv_f16/weights.c` is byte-identical to source (md5 match).
- submodule cascade checks out the committed tips (xpu-rt `b6cb578`, zcs `c953690`,
  tinympc `c1192f3`) and the **coordinated-turn fix is present** in the fresh
  clone's `rose_fused_mpc/src/main.cpp`.
- the **guest ELF builds** from the fresh clone's sources + committed model dir
  (`-march=…zfh_zvfh`, ~3.88 MB), reusing the Prereq-B toolchain.
- the co-sim harness is wired: `run_cosim.sh` resolves all committed inputs and
  its preflight guard correctly flags the one missing heavy prereq.

**Deferred (heavy prereqs, documented above; reused not rebuilt):** Zephyr/SDK
bootstrap (Prereq B), `rose_spike_sim`/Chipyard build (Prereq C), IsaacLab +
`env_isaaclab` (Prereq A).

**Deferred — live short co-sim run:** not executed during this capture because the
single shared GPU was in active use by another Isaac workload; launching a
competing Isaac Sim would have risked disturbing it. The exact build+run recipe
here is the one that produced the controlled 3/4 result on seed 1000 (the
coordinated-turn commit `c953690`).
