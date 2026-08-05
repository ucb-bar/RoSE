# Integrating Agustin's warehouse fused-nav net with RoSE

How to bring the collaborator's learned warehouse local-planner (`FusedSensorNet`) onto the RoSE
side, in two paths: (A) **host net + RoSE tracker** (fastest, closes the loop with the real
warehouse env), and (B) **on-SoC net via ModelBlaster** (quantize + lower to int8, run on the
SoC like DroNet). Source of truth is Agustin's handover; this doc is the RoSE-side plan.

## Where their work lives (read-only, NOT in git)

Everything under `/scratch/agustin/projects/DIMA` (world-readable). Uncommitted working-tree
changes in a **separate** `XPU-RT` checkout + a `vitfly` checkout — copy into our space or ask
for a branch. Isaac python: `/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python`.

| thing | path |
|---|---|
| handover / RoSE handoff | `.../DIMA/HANDOVER.md`, `.../XPU-RT/sims/isaaclab_tasks/warehouse_nav/HANDOFF.md` |
| model class `FusedSensorNet` | `.../DIMA/vitfly/models/fused_model.py` |
| deployable ckpt (v12 CNN) | `.../DIMA/train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt` |
| env + sensors | `.../XPU-RT/sims/isaaclab_tasks/warehouse_nav/`, `.../forest_trail/{sensors,state_estimator}.py` |
| eval/glue driver | `.../XPU-RT/sims/scripts/eval_fused_warehouse.py` (`sense()`, `cmd_to_action()`) |
| seam action | `.../warehouse_nav/mdp_velocity_action.py` |
| HW co-design / DSE | `.../XPU-RT/docs/hw_codesign_model_spec.md`, `.../sims/scripts/{dse_pareto,hw_cycle_model}.py` |

## What it is

A learned local planner: fly a Crazyflie-class drone through 4 warehouse gate "checkpoints"
while avoiding racks, a dense tall-thin prop field, and patrolling people. `FusedSensorNet` maps
the full onboard suite → `(yaw_rate, forward_speed)` @ **10 Hz**; a low-level controller tracks
it @ **100 Hz** — **the seam where RoSE/TinyMPC plugs in**. Status: **100% (12/12)** clean gate
course; **~42%** complex collidable course (teacher-bounded, not a HW/net limit). Two encoders:
**CNN v12 (~1.55 M params, int8/Gemmini-friendly — the deployable one)** and a ViT (3.67 M, flies
the clean course, too heavy for the SoC). The reproducible demos/videos use the **CNN v12**.

## Deployable net (CNN v12) — I/O

**Input** = dict of range-normalized tensors (missing key → zeros): `front_grey (B,1,H,W)`
(interpolated to **60×90** inside the net), `tof_cross (B,4,8,8)` order **[N,E,S,W]**,
`optical_flow(2)`, `down_tof(1)`, `baro(2)` (fed `/10`), `quat(4)` (Madgwick), `body_rates(3)`,
`desired_vel(3)` (goal dir×cruise; mask → camera-only), `flags(6)`. Plus an **external 3-layer
LSTM hidden carry** (h=128) threaded step-to-step. **Output** = `(B,2)` = `(yaw_rate,
forward_speed)`, raw (no output norm). The speed head underfits (yaw is the reliable channel; a
fixed cruise is typical). Shape: CNN stem (4 conv/ReLU, 60×90→4×6) → `1536→512`; ToF branch (2
conv on 4×8×8) → `1024→64`; fuse `[512+64+15+6=597]` → LSTM(128) → head→2.

## The command seam (integration interface)

```
FusedSensorNet(sensors) @10Hz → (yaw_rate, forward_speed)
   → cmd_to_action  → 4-ch polar [a0,a1,a2,0]  (+ altitude-hold P-loop to TARGET_H=2.0 m)
   → VelocityCommandAction.process_actions → velocity setpoint [vx≥0, vy=0, vz, yawrate] (yaw frame)
   → low-level tracker @100Hz  ← RoSE/TinyMPC REPLACES THIS
```
- **Contract RoSE must satisfy:** track the 4-vector **`[vx, 0, vz, yawrate]` in the yaw frame**
  at 100 Hz (or take raw `(yaw_rate, forward_speed)` + own `vz`). Limits `max_speed=2.0`,
  `max_yawrate=π/3`, `max_inclination=π/4`. Their tracker is a Lee geometric velocity controller
  (vel=2, att=200, rate=20) — the thing TinyMPC swaps in for.
- **Altitude** is a separate P-loop to `TARGET_H=2.0 m` in `cmd_to_action`; the net does
  horizontal guidance only. Drop it if RoSE owns `vz`.
- **10 Hz/100 Hz split** = exactly the timer-driven decoupling we built (net @10 Hz emits the
  setpoint, TinyMPC @100 Hz tracks, ZOH between). Their eval models mixed cadences with a
  `SensorRateSampler` (ZOH); no explicit dual-rate loop in the eval itself.

## Sensors — the ToF cross maps 1:1 onto our RoSE ToF

4× VL53L5CX, order **[N,E,S,W] = [+x front, −y right, −x back, +y left]**, 8×8 zones, 63°
diagonal FoV, 0.02–4.0 m, no-return→4.0 m, normalized `[0,1]` via `normalize_range(x, 0.02,
4.0)` ("do not reorder"). This is **the same shape as our existing `rose_tof_zone`** (cmds
`0x30–0x33` = front/right/back/left = N/E/S/W, 64 zones = 8×8) — so our ToF plumbing already
fits; drop the analytic dummy geometry and match this layout + normalization exactly. Camera is
HM01B0 mono (native QVGA, net works at 60×90); flow/baro/down-ToF/Madgwick round out the suite.

## Path A — host net + RoSE tracker (fastest)

Run `FusedSensorNet` host-side in their warehouse env (`eval_fused_warehouse.py` is the reference
`sense()` + `cmd_to_action()`), and have **RoSE/TinyMPC replace `VelocityCommandAction`** as the
100 Hz setpoint tracker. Everything above the seam (net + env + sensors) is unchanged. This
validates the closed loop on the real collidable warehouse course with **no on-SoC net** — the
net stays torch-on-host, the RoSE co-sim provides the 100 Hz tracker. Smallest first move; proves
the seam and the warehouse env before any quantization work.

Steps: (1) copy/point at their env + eval; (2) expose the net's `(yaw_rate, forward_speed)` (or
the `[vx,0,vz,yawrate]` setpoint) to the RoSE bridge instead of the geometric tracker; (3) let
the RoSE guest's TinyMPC track it at 100 Hz and emit motor thrusts back into Isaac; (4) run the
`-Coll-Crowded-v0` course and measure gate success. Reuses our timer-driven controller's setpoint
path directly.

## Path B — on-SoC net via ModelBlaster (quantize + lower, run on spike)

Follow the DroNet procedure (`docs/ROSE_DRONET_INTEGRATION_PLAN.md` P0): PyTorch →
`extract_graph --quant int8` → `generate_skeleton`/`generate_kernels` → compile the generated C
into a zephyr sample → run on spike, compare to the PyTorch golden.

**The gap / risk:** there is **no real int8 artifact for the fused net today** — the "100% int8
on Gemmini-Q31" claim is *modeled* (DSE cycle/energy model), not produced. The only real
quantization path is DroNet's (`qat_dronet.py`), which **explicitly defers the LSTM** ("attention/
LSTM FX-quant is fragile — a follow-up"). So the hard parts for the DroNet-style lowering are:
1. **The LSTM** — ModelBlaster's int8 kernels are conv2d/linear/relu/etc.; there is no int8 LSTM
   kernel, and FX-quant of `nn.LSTM` is fragile. This is the primary blocker for the *full* net.
2. **The multi-input dict** — the DroNet path assumes a single input tensor; the fused net takes
   a 9-key dict.

**Plan for Path B (exploratory, "see how far we get"):**
1. Register the fused CNN in ModelBlaster (a `models/fused_gate.py` wrapper like `models/dronet.py`)
   pulling `FusedSensorNet(vision_encoder="cnn", out_dim=2)` from the vitfly checkout + the v12
   checkpoint; provide `get_sample_input()` for the input dict.
2. Attempt the full int8 `extract_graph`. If the LSTM/dict block it, **fall back to lowering the
   feed-forward sub-network** — the CNN vision stem + `vision_fc` + depth conv + `depth_fc` + the
   fuse Linear (all conv/linear/relu, DroNet-like) — as a single-input int8 model, and lower +
   spike-verify THAT (proves the vision/ToF encoder path runs int8 on the SoC).
3. Treat the LSTM as the remaining piece: either a separate int8/fixed-point LSTM kernel
   (ModelBlaster kernel work) or run the small LSTM+head in fp on the scalar/RVV path while the
   encoder GEMMs go to Gemmini (matches the hw_cycle_model's "LSTM gates on RVV" split).
4. Validate on spike exactly like DroNet: golden compare of the lowered sub-model's int8 output.

**Success criteria:** an int8-lowered piece of the fused net (at minimum the encoder+fuse)
verifying bit-exact vs its PyTorch golden on the co-sim spike, plus a clear statement of what the
LSTM needs. Full-net on-SoC inference is the stretch goal pending LSTM int8 support.

### Path B — results (2026-08-05)

Both feed-forward encoders of the v12 CNN were quantized + lowered to int8 via ModelBlaster
(extract_graph int8 → skeleton → kernels, scalar backend) and **verified bit-exact on the co-sim
spike** (ModelBlaster harness under rose_spike_sim + minimal_sync, golden compare):

| sub-model | input (int8) | output (int8) | in_scale | spike verify | wall cycles |
|---|---|---|---|---|---|
| **fused_vision** (front_grey→CNN stem→`vision_fc`→512) | 5400 (1×1×60×90) | 512 | 0.032945862 | **max_abs_err=0, n=512** | ~1.74 M |
| **fused_depth** (tof_cross→conv→`depth_fc`→64) | 256 (1×4×8×8) | 64 | 0.026229556 | **max_abs_err=0, n=64** | ~72 K |

Artifacts in `ModelBlaster/examples/{fused_vision,fused_depth}/int8/generated/scalar/`
(`run_model_fused_{vision,depth}`, DroNet-shaped 12-file set). Wrappers `models/fused_{vision,
depth}.py` + `models/_fused_loader.py` load `FusedSensorNet(cnn)` from the vitfly checkout + the
v12 ckpt, fold `spectral_norm`, and expose each encoder as a single-input conv/linear net.
Gotcha that mattered: use `torch.flatten(x, 1)` not `x.flatten(1)` (the latter traces as a
`call_method` the int8 extractor rejects).

**Full-net blockers (confirmed, three compounding):** (1) FX `symbolic_trace` fails on the
data-dependent `if img.shape[-2:] != (60,90)` guard in `forward`; (2) inlining past it hits
`int8 extract: get_attr _tensor_constant0 not supported` (tensor constants); (3) **`nn.LSTM` has
no int8 lowering path at all** — the extractor's allowlist is {Linear, ReLU, ReLU6, Conv2d,
MaxPool2d, AdaptiveAvgPool2d, BatchNorm2d, Upsample}; grep for lstm/rnn/gru in `extract_graph.py`
returns nothing. So the full net needs (a) a traceable single-tensor front-end and (b) a new int8
LSTM kernel + extractor support. The **encoder GEMMs are the bulk of the compute and now run int8
on the SoC**; the remaining LSTM+head (~636 K params, tiny GEMMs + elementwise gating) is the
piece to either add as an int8 kernel or run fp on the scalar/RVV path (matches the
hw_cycle_model "LSTM gates on RVV" split).
