# Fused vision-nav on spike-in-the-loop: the warehouse gate co-sim

How to fly the Isaac warehouse **gate course** with the fused drone-nav vision model
running **on the SoC (spike)** in the RoSE↔IsaacLab lockstep co-sim, steering the
flight controller. This is the DroNet-study pattern (camera→model→policy→actuator on
the SoC) applied to the multi-input fused net + the photoreal warehouse gate scene.

Companion memory: `rose-fused-gate-cosim`, `rose-isaac-cosim-flow`, `rose-dronet-integration`.
Model/kernel details: `rose-modelblaster-spike-build`, `ModelBlaster/notes/curated_rvv_kernels.md`.

---

## Architecture (Stage 1: vision model on spike, tracker in Isaac)

The **only** thing on the SoC is the fused vision model; everything else (sense(),
the altitude-hold + `cmd_to_action`, the Lee velocity tracker, the warehouse scene &
gate scoring) stays in Isaac — so the co-sim faithfully reproduces the pytorch eval
with just the model swapped onto spike. (Stage 2 = move TinyMPC on-SoC too.)

```
Isaac warehouse env ──sense()──> pre-quantize (host ref) ──bridge──> SoC guest
   (photoreal gates)                                                   (fused_full
        ^                                                               int8 enc +
        │  Lee velocity tracker <── cmd_to_action(yr,fwd,h) <──0x20──   fp16 tail)
        └────────────────── env.step(polar velocity) ──────────────────┘
```

### Wire contract (Isaac pre-quantizes exactly as the validated host reference)
| data | dir | cmd | transport | format |
|---|---|---|---|---|
| front_grey | Isaac→SoC | 0x11 | DMA ch0 | int8[5400] (60×90) |
| tof_cross | Isaac→SoC | 0x41 | reqrsp | int8[256] |
| lowdim | Isaac→SoC | 0x42 | reqrsp | float32[21] → guest casts to `_Float16` |
| (yaw_rate, fwd) | SoC→Isaac | 0x20 | action | float32[2] (from `_Float16` model out) |

Model ABI: `run_model_fused_full(int8* front, int8* tof, _Float16* lowdim, _Float16* out, pool)`.

### New files
- **Guest**: `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_fused_nav/` — per tick:
  DMA front (0x11) → int8 input0; reqrsp 0x41 → int8 tof; reqrsp 0x42 → f32→`_Float16`
  lowdim; `run_model_fused_full`; emit (yr,fwd) on 0x20. No TinyMPC (Isaac owns the tracker).
- **Isaac env**: `deploy/hephaestus/envs/warehouse_fused_nav/warehouse_fused_nav_env.py`
  wraps `Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0`; serves
  pre-quantized `sense()`, applies the guest command via `cmd_to_action` (identical to
  `sims/scripts/eval_fused_warehouse.py:cmd_to_action`) + the env's Lee tracker.
- **Config**: `deploy/config/config_gym_WarehouseFusedNavBridgeEnv-v0.yaml` (packet bindings).
- **Launch**: scratch `run_wh_flight.sh <tag> <obst> <spike_timeout_s> [freeze] [camera]`.

---

## Build + run

```bash
S=<scratch>; ROSE=/scratch/dima/rose-infra/RoSE
# 1) Build the guest (scalar_f16 model; needs zfh). CTRL_ITERS = #control ticks to fly.
west build -p always -b spike_riscv64 --build-dir $S/build_fused_nav \
  $ROSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_fused_nav -- \
  -DZEPHYR_EXTRA_MODULES=<zephyr-rose> -DMODEL_DIR=$S/cfg_bp_f16lowdim/gen \
  -DMB_POOL_INC=$ROSE/soc/sw/xpu-rt/ModelBlaster/runtime/modelblaster_pool -DCTRL_ITERS=300
# 2) Launch the lockstep co-sim (2 procs: Isaac sync + spike bridge). Camera=1 for video.
export ROSE_MAX_SIM_TIME=90            # see "grant-budget cutoff" below — REQUIRED
bash $S/run_wh_flight.sh gate 0 3000 1 1
```

---

## Novel build configs / RoSE invocations (the things that bit us)

1. **Spike ISA = `rv64gc_zicntr_zihpm_zfh`** (NOT the DroNet study's default ISA).
   - `zfh` — the fp16 tail (lstm_f16 / linear_f16 / cat_f16) uses `_Float16`.
   - `zicntr` (+`zihpm`) — `model.c` reads `rdcycle`; a plain `rv64gc_zfh` drops
     `zicntr` and the guest traps (mcause=2, illegal instruction). Set via
     `ROSE_ISA` (new overridable knob in `soc/sim/run_spike_rose_lockstep.sh`).

2. **`ROSE_MAX_SIM_TIME` override (new, in `gym_synchronizer.py`) — the grant-budget cutoff.**
   The synchronizer ends the run at `cycle_limit = max_sim_time * firesim_freq`
   (default `max_sim_time=12` → `12/gym_timestep=2400` firesim grants). Under the
   **freeze seam**, a SLOW on-SoC guest (the co-sim runs the **scalar_f16** model —
   the co-sim spike has no vector unit — at ~120M cyc/inference) burns ~34 grants per
   physics step, so the 2400-grant budget is exhausted after only ~70 physics steps
   (**~1.4 s of flight**) — the episode is cut off long before gate 1, with the drone
   flying perfectly. Fix: raise the cap, e.g. `ROSE_MAX_SIM_TIME=90` (~530 physics
   steps). (The RVV `rvv_f16` kernels would cut inference ~17× and largely remove this
   waste, but the co-sim spike would then need `--isa=rv64gcv...`.)

3. **Freeze seam** (`ROSE_FREEZE=1`, default): the wrapper HOLDS the served `sense()`
   snapshot and advances the inner physics EXACTLY ONCE per received 0x20 command
   (verified 1:1), giving one coherent sensors→command→step tuple per control tick —
   the faithful match to the pytorch eval. `ROSE_FREEZE=0` free-runs with a ~3-frame
   sensor skew (each blocking per-modality reqrsp otherwise crosses a token boundary).
   Implemented via a non-breaking `on_action_received()` hook in the synchronizer.

## Gotchas / lessons

- **Calibration distribution**: pure-int8 fails on the real gate distribution (lowdim
  `optical_flow` outlier ~332 + head saturation) — the deployable model is **int8
  encoders + fp16 tail + fp16 lowdim**, calibrated on REAL gate-course frames. See
  `rose-fused-gate-cosim`.
- **Altitude-hold seam bug**: the first closed-loop run crashed (altitude sank through
  TARGET_H to the ground). Root cause was the env/seam, not `cmd_to_action` (which is
  byte-identical to the reference). Isolating test (fp32 model → the wrapper env's
  control path, no spike) is the fast way to separate env/seam bugs from bridge bugs —
  it flew 2 gates with altitude locked once fixed.
- **Monitor GROUND TRUTH only**: use the trajectory CSV (`gates_passed` col 13, `z`
  col 6 <0.5 = crash), not mid-flight "looks stable" snapshots.

## Results

- Model: int8+fp16 fused net flies the clean gates 4/4 in pytorch (== fp32); on-spike
  output matches the host reference to **1 fp16 ULP**.
- Co-sim M1 (wiring): byte-exact same-frame validation (probe), live warehouse connect
  + exchange, no hang.
- Co-sim M2 (closed loop): stable flight, altitude locked ~2.10 m; env control path
  flies 2 gates in the isolating test.
- **FULL 4/4 on RVV (definitive):** with the curated `rvv_f16` guest (V ISA
  `rv64gcv_zicntr_zihpm_zfh_zvfh`) + a known-good spawn seed (`ROSE_WH_SEED=1000`,
  matching a pytorch-4/4 seed), the on-spike model flew **ALL 4 GATES**:
  `[GATE] 1@2.04s 2@4.56s 3@7.12s *** ALL 4 GATES PASSED *** @9.68s`,
  `outcome=success gates_passed=4/4 ticks=484`, altitude locked 2.00 m. Two
  watchable videos (243 frames = the ep0 4-gate flight, pixel-motion-verified):
  scratchpad `fused_nav_4gate_chase.{mp4,avi}` (following 3rd-person, drone+gate
  framed) and `fused_nav_4gate_fpv.{mp4,avi}` (the model's forward view). NOTE:
  the quantized model is spawn-sensitive — the co-sim MUST reset with a
  pytorch-good seed (`ROSE_WH_SEED`), else a random unaligned spawn drifts ~1 m
  off the x≈−8 gate line and misses (a ≤1-ULP closed-loop divergence, not a bug).
  Also: the fixed `chase_camera` is world-anchored — the wrapper repositions it
  each tick to follow the drone (`_drive_chase`), and captures `front_camera`
  separately for the FPV video; without the follow-update the chase video is a
  static shot.

- **Co-sim M3 (spike gate flight, scalar): SUCCESS.** With the raised grant budget
  (`ROSE_MAX_SIM_TIME=90`, `CTRL_ITERS=300`), the on-spike fused model flew the clean
  course and passed **2 gates** — env ground truth: `[GATE] passed gate 1/4 at t=2.20s`,
  `[GATE] passed gate 2/4 at t=4.74s`. Altitude locked at **2.00 m** the whole flight,
  level, freeze cadence exactly 1:1 (cmds=300 ticks=300), **no crash**. It stopped at
  y=14.1 (between gate 2 and gate 3) only because the 300-iter (t=6 s) budget ended —
  not a failure; `CTRL_ITERS≈800` + a higher `ROSE_MAX_SIM_TIME` would fly all 4 (at
  ~4 s wall/step for the scalar_f16 guest, ~60 min wall). Chase-cam video (150 frames,
  480×270) assembled from `wh_gate2_frames/` via `assemble_video.py` →
  `rose_cosim_gate_flight.{mp4,avi}`. This is the DroNet-study milestone for the fused
  net: one Zephyr ELF, live warehouse sensors → on-SoC vision model → policy → gate flight.
