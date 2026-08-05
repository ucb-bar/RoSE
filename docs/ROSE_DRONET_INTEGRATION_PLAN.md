# Integrating ModelBlaster's quantized DroNet with the HM01B0 camera + flight controller

Systematic plan to run ModelBlaster's **int8 DroNet** on the RoSE SoC, fed by a **live
HM01B0 camera** frame (320×240 8-bit grayscale over the bridge DMA path), with its
`(steer, collision)` output driving the **flight controller**. Builds on the three halves
that already exist and are validated in isolation:

- **Camera → SoC** — HM01B0 FPV frame lands byte-exact on the SoC via DMA ch0 / cmd 0x11
  (`docs/ROSE_CAMERA_DMA_PATH.md`; `ucbbar,rose-camera` video driver).
- **DroNet int8** — ModelBlaster PyTorch→per-op-C-kernel pipeline, symmetric per-tensor
  PTQ, validated bit-exact on spike (`ModelBlaster/notes/int8_quantization_flow.md`).
- **Flight controller** — threaded estimator + TinyMPC + `send_control` (0x20) in
  `zephyr-chipyard-sw/samples/rose_{flight,nav}_controller`
  (`docs/ROSE_FLIGHT_CONTROLLER_THREADING.md`).

The work is the three **seams** between them, none of which exist yet: (a) on-SoC
preprocessing camera-frame → DroNet int8 input, (b) DroNet output → control setpoint, and
(c) wiring the camera + model into the controller's threading/rate structure.

---

## 0. What each side actually is (the contract)

**DroNet int8 (ModelBlaster @ `dbbdcf0a`)**
- **Input:** `(1, 3, 112, 112)` NCHW, **RGB**, **int8** symmetric per-tensor, zero-point 0,
  `scale = max_abs/127`. Host preprocessing (`sims/training/dataset_sim.py:44`): `RGB →
  Resize(112,112) → ToTensor` (i.e. `/255`, **no** mean/std). The int8 quantization uses
  `in_scale = graph.json` input-tensor `quant.scale` (`extract_graph.py:1257`).
- **Output:** `(steer, collision)` — 2 fp scalars. `steer = linear1(x)` (raw), `collision
  = sigmoid(linear2(x))` ∈ [0,1]. **Caveat:** the shipped checkpoint trained only the
  **steering head**; `linear2` (collision) is random-init. Weights default to random
  unless `MODELBLASTER_DRONET_CKPT` points at the trained steering checkpoint.
- **Input injection point (the one load-bearing seam):** the model reads `s->input` /
  `run_model(model_test_input, …)` (`harness/src/main.c:48`,
  `generate_xpurt_main.py:150`), which today points at an `.incbin`'d rodata blob
  `model_dronet_test_input` (a baked `torch.randn` frame). **Redirect this pointer to a
  RAM buffer holding the live, preprocessed, int8-quantized camera frame.**
- **Build:** `examples/dronet/run.sh` (single model) or `examples/xpurt_demo/run.sh`
  (schedule-driven), knobs `TARGET`/`QUANT=int8`/`BACKEND`/`RUNNER`. Dronet-only schedule
  fixture exists: `ModelBlaster/schedule_fixtures/dronet_xpurt_mosek.json` (30 dispatches).

**HM01B0-ANA-00FT870 camera (color/RGB variant)**
- 320×240 **RGB** (the ANA-00FT870 is a color sensor), delivered to `dma_base` (0x90000000)
  and copied into a `video_buffer` by `ucbbar,rose-camera` on `video_dequeue`
  (`rose_camera.c:119`). **Only mismatch vs DroNet: resolution** (320×240 → 112×112) — no
  grayscale→RGB replication needed; the camera path now carries RGB frames end to end (the
  driver format + Isaac render switch from GREY to RGB; see §2).

**Flight controller (TinyMPC)**
- Threads `io / est / ctrl` (`rose_flight_controller/src/main.cpp`), `CTRL_DT=0.005`
  (200 Hz), ctrl at `200/ROSE_CTRL_DIV` Hz. `ROSE_THREADED` **defaults 0** (single loop)
  because of the co-sim lockstep hang — budget for that.
- **Setpoint injection point:** where `err[] = state − setpoint` is assembled just before
  `matsetv(work.x.vector[0], err)` / `tiny_solve` (`rose_flight_controller` `solve_control`
  L244-251; `rose_nav_controller` L220-252). `rose_nav_controller` VELOCITY mode already
  ramps `setpoint[6]=vx_cmd` and WAYPOINT mode ramps `setpoint[0]=x` — **this ramp is
  exactly what a DroNet-derived command replaces.**
- **Output:** `send_control(u)` → `rose_tx(0x20)` + 4 float32 motor thrusts.
- **Does not touch the camera today** — no camera node in the controller overlay (only
  `num-dma-channels=<1>` + `dma-base-address`). Adding the `ucbbar,rose-camera` node + an
  `fpv` alias to the controller overlay is required.

**Reference chain (host prototype, `play_dronet_mlp_scheduled.py`):** the exact mapping to
reproduce — `target_yaw_rate = steer · max_yaw_rate`; `target_velocity = (1−collision) ·
max_velocity`; last output held (ZOH) between DroNet runs; DroNet gated to run only inside
its scheduled window. Note that prototype then feeds a **trained MLP policy** (16→4), not
TinyMPC — see the two endpoints in §6.

---

## 1. Guiding decisions (resolve before Phase 3; recommendations given)

1. **Backend / accelerator target.** DroNet int8 kernels exist for `scalar`, `rvv`,
   `gemmini_q31`. **Spike models Saturn RVV by default** and can build **Gemmini via an
   external lib.** **Decision: target `rvv` int8 for the co-sim integration (Phases 0-3);**
   `gemmini_q31` is available via the extlib if we want the systolic path in the co-sim, and
   is the FireSim target in Phase 4.
2. **Control endpoint (§6).** Recommend **Endpoint A (DroNet → TinyMPC setpoint)** for the
   near-term integration — it *is* the current RoSE flight controller and needs no new
   trained model on the SoC. **Endpoint B (DroNet → scheduled MLP policy under
   `harness_xpurt`)** is the XPU-RT convergence and mirrors the host prototype exactly;
   sequence it after A.
3. **Collision head.** Untrained in the shipped ckpt. **Recommendation: use `steer` only
   at first** (yaw-rate command), hold `target_velocity` at a fixed cruise; wire
   `(1−collision)·v` only after the collision head is retrained/validated. Flag loudly so
   nobody flies on a random-init collision signal.
4. **Vision rate.** DroNet runs far slower than the 200 Hz control loop. Run it as a
   **low-rate producer** (e.g. 10-30 Hz) with **ZOH** of its command into the setpoint —
   the same multi-rate pattern as `ROSE_CTRL_DIV`, and what the host prototype's schedule
   gate does.

---

## 2. Phase 0 — Build & validate quantized DroNet standalone (no camera, no controller)

**Goal:** a DroNet int8 ELF that runs on the co-sim-compatible target and reproduces the
PyTorch golden, so the model is trusted before any wiring.

- Point `MODELBLASTER_DRONET_CKPT` at the trained steering checkpoint (the default paths in
  `models/dronet.py` are for another host).
- Build: `TARGET=scalar QUANT=int8 bash ModelBlaster/examples/dronet/run.sh` (spike runner).
  Confirm the built-in verifier passes against `io.npz` golden (`spike_runner`).
- Record the **input `in_scale`** from `generated/graph.json` (input tensor `quant.scale`) —
  Phase 1 needs it to quantize live frames identically.
- **Exit criteria:** DroNet int8 ELF verifies bit-exact (within the backend's `atol`) on the
  co-sim target; `in_scale` captured.

## 3. Phase 1 — On-SoC preprocessing: grayscale frame → DroNet int8 input

**Goal:** a small, testable C routine that turns a 320×240 u8 grayscale frame into the exact
`(1,3,112,112)` int8 tensor DroNet expects, **bit-identical to the host pipeline.**

- New guest module (in `zephyr-rose`, e.g. `lib/rose_vision/preproc.c`): 
  `rgb320x240_u8 → resize 112×112 (RGB) → /255 → symmetric int8 quantize with in_scale`.
  Resize: bilinear to match host `F.interpolate`/`Resize` (or document the exact
  interpolation and match it host-side). No channel replication — the ANA-00FT870 is RGB.
- **Validation (CRC parity, same method as the camera path):** run the identical frame
  through a host reference (the ModelBlaster preprocessing + quantize) and through the guest
  routine; compare **CRC32 of the int8 tensor**. A match proves the SoC feeds DroNet exactly
  what the host golden assumes.
- **Exit criteria:** guest int8 tensor CRC == host int8 tensor CRC for a fixed test frame.

## 4. Phase 2 — Camera → DroNet on the SoC (no controller yet)

**Goal:** prove the full sensor→model edge on the SoC in the co-sim.

- New sample `zephyr-rose/samples/dronet_cam` (camvalidate + Phase-0 ELF + Phase-1 preproc):
  `video_dequeue` a live HM01B0 frame → Phase-1 preproc → point `s->input`/`run_model` at
  the resulting RAM buffer → run DroNet → print `(steer, collision)`.
- Drive it in the RoSE lockstep with the FPV-camera env (`IsaacCrazyflieMultiSensorEnv-v0`
  + `ROSE_ISAAC_FPV=1`, camera on 0x11/dma/ch0). 
- **Validation:** feed the same served frame to host DroNet; assert on-SoC `steer` matches
  host `steer` (within int8 tolerance). This is the "legitimate camera sensing → model on
  SoC" milestone, one layer up from the camera CRC match.
- **Exit criteria:** live Isaac frame → on-SoC DroNet `steer` == host DroNet `steer` (±tol).

## 5. Phase 3 — Integrate into the flight controller (Endpoint A: DroNet → TinyMPC setpoint)

**Goal:** DroNet's output steers the TinyMPC-controlled drone in the co-sim, closing the
loop camera→model→policy→actuator on the SoC.

- **Overlay:** add the `ucbbar,rose-camera` node + `fpv` alias to
  `rose_{flight,nav}_controller/boards/spike_riscv64.overlay` (DMA ch0, `dma-base-address`
  0x90000000 — must match the runner's `ROSE_DMA_BASE`).
- **Vision producer:** add a low-rate vision step (single-loop first, given `ROSE_THREADED`
  defaults 0). Each vision tick: dequeue frame → preproc → DroNet → map to command:
  - `yaw_rate_cmd = steer · MAX_YAW_RATE`
  - `vx_cmd = (1 − collision) · CRUISE_VX`  *(Phase-3a: hold `vx_cmd` fixed, steer-only,
    until the collision head is trusted — decision #3)*
  - Publish into a mutex-protected `g_vision_cmd` (mirrors the `g_state`/`g_ctrl`
    latest-value + ZOH pattern), held between vision ticks.
- **Setpoint injection:** at the controller's setpoint-assembly point (`rose_nav_controller`
  L220-252), replace the ramp: set `setpoint[6] = vx_cmd` (VELOCITY mode) and fold
  `yaw_rate_cmd` into the yaw setpoint state, reading `g_vision_cmd` under its mutex — instead
  of `rose_nav_controller`'s hand-authored ramp. TinyMPC + `send_control(0x20)` unchanged.
- **Rate:** vision at ~10-30 Hz, control at 200/`ROSE_CTRL_DIV`; ZOH the command in between.
- **Threaded variant (later):** promote the vision step to its own thread only after the
  single-loop path is stable — it shares the lockstep-grant + keepalive machinery implicated
  in the co-sim hang (`ROSE_COSIM_HANG_INVESTIGATION.md`), and DMA ch0 is a shared resource.
- **Exit criteria:** in the RoSE co-sim, DroNet output visibly changes the drone's
  yaw/velocity via TinyMPC (e.g. steers along the Isaac hallway); no lockstep hang; control
  output on 0x20 well-formed.

## 6. Phase 4 — XPU-RT convergence (Endpoint B: scheduled DroNet + policy under harness_xpurt)

**Goal:** the on-SoC realization of `play_dronet_mlp_scheduled.py` — the "schedule a set of
models to HW + live sensor in + policy out" goal (`docs/ROSE_XPURT_MODELBLASTER_MAP.md` §6).

- ModelBlaster already ships an `mlp_control` model and `schedule_fixtures/
  dronet_mlp_control_periodic.json` (the `xpurt_demo` default `MODELS=dronet,mlp_control`).
  This is the RL-policy endpoint the host prototype uses (obs 16 → 4 thrust/moment actions).
- Add **RoSE source/sink dispatches** to ModelBlaster (map §6 step 1): a *source* op that
  fills DroNet's input from a live `video_dequeue`+preproc (Phases 1-2 as a kernel), and a
  *sink* op that routes the policy output via `send_control(0x20)`.
- Run `harness_xpurt` as the SoC guest in the RoSE lockstep (same setup the flight
  controller uses now); XPU-RT co-schedules DroNet (low-rate) + policy (high-rate) on the
  available cores. On FireSim, bring in `gemmini_q31`/`rvv` targets (decision #1).
- **Exit criteria:** one Zephyr ELF, schedule-driven, flies the Isaac drone from live camera
  frames — replacing the hand-authored threading with a generated schedule.

**Endpoint A vs B:** A is the shortest path to a closed camera→model→flight loop on the SoC
and reuses the existing TinyMPC controller. B is the strategic target (generated schedule,
matches the host prototype's DroNet→MLP chain), but needs the source/sink dispatches and a
co-sim-runnable `harness_xpurt`. **Do A first; it de-risks preproc + the camera/DroNet
seam that B also needs.**

---

## 7. Cross-cutting risks / notes

- **Backend availability in the co-sim spike (decision #1)** gates everything — verify V/Gemmini
  support before assuming a target; scalar int8 is the safe default and sets the vision rate.
- **Random-init collision head (decision #3)** — do not fly on `(1−collision)·v` until retrained.
- **Preprocessing must be bit-exact** — the RGB resize interpolation and the `/255`+int8
  quantize must match the host reference, or the golden comparison is meaningless. Phase 1's
  CRC gate exists precisely to lock this down.
- **DMA ch0 is shared** — camera capture and (future) any other DMA consumer arbitrate on it;
  the single-payload/fixed-landing-zone DMA model (`ROSE_CAMERA_DMA_PATH.md`) means one frame
  in flight at a time.
- **Co-sim lockstep hang** — keep the vision step in the single loop until proven; the threaded
  path interacts with the same grant/keepalive machinery that motivates `ROSE_THREADED=0`.
- **Weights provenance** — set `MODELBLASTER_DRONET_CKPT`; the default is random init.

## 7b. Implementation status — P0–P3 DONE + validated (2026-08-05)

Built and validated end-to-end on the co-sim (`rose_spike_sim` + synchronizer). DroNet int8
**scalar** used for the co-sim (Saturn RVV/Gemmini also codegen'd; scalar guarantees it runs
under the default-ISA lockstep spike). Camera path switched to **RGB** (ANA-00FT870 is color).

- **P0 — DroNet int8 built + golden-verified.** ModelBlaster codegen (scalar, int8, trained
  `best.pt` ckpt) → `run_model_dronet`, 24 dispatches. On the co-sim spike the output is
  **`[-56,127]`, `max_abs_err=0`** vs the PyTorch golden. `in_scale=0.032945861966591182`
  (randn-calibrated → a real `[0,1]` frame uses ~`[0,30]` of int8; recalibrate for accuracy).
- **P1 — On-SoC RGB preproc, CRC parity.** `lib/rose_vision/rose_preproc.c` (integer bilinear
  + symmetric int8 quant, `tools/gen_preproc_quant.py` from `in_scale`). **Three-way
  bit-exact**: native gcc = host `preproc_ref.py` = guest `crc32_ieee` on `rose_spike_sim`,
  all `crc32=0x4aa455b2`. Sample `samples/preproc_selftest`.
- **P2 — Live camera → DroNet on SoC.** `ucbbar,rose-camera` gains an `rgb` mode (RGB24);
  sample `samples/dronet_cam` (capture → preproc → DroNet). Guest reproduces the host oracle
  (`tools/host_oracle/`) exactly: served frame `crc=0x788fef76`, input `crc=0x719b62c4`,
  **`out_i8=[-26,127]`, steer=-0.007549, collision=0.478426** — every layer byte-identical.
- **P3 — DroNet → TinyMPC setpoint.** `rose_nav_controller` gains `-DROSE_DRONET_NAV=1`:
  low-rate FPV capture (ZOH) → DroNet → `vx=(1-collision)·CRUISE`, `yaw=steer·gain` at the
  setpoint-assembly point. In the co-sim (GPU-free `CrazyflieVisionNavProbeEnv-v0` serving the
  full sensor set + RGB FPV), **DroNet drives the setpoint**: `vx 0 → 0.208 m/s`
  (=(1−0.478)·0.4) after settle, yaw from steer, `0x20` control well-formed (4 hover thrusts
  ~0.418), no lockstep hang. The physics-closed flight is the real Isaac multisensor env (GPU)
  — the same guest ELF, its RGB-FPV render being the remaining env hook.

**P3b — threaded producer/consumer split (2026-08-05).** `rose_nav_controller -DROSE_THREADED=1`
splits into five threads with real-time priorities (lower = higher): **ctrl(3) > est(5) > io(7) >
vision(10) > keepalive(14)**. The pipeline is deterministic (io paces via a per-grant `sem_done`
barrier); **all bridge I/O — sensor reqrsp, camera capture, control TX — stays in the io thread**
so the RoSE transport is never touched concurrently. The vision thread is **compute-only**
(preproc + the ~455 ms DroNet inference) and runs at the lowest app priority, so control always
preempts it. `keepalive` busy-spins to stop the "all-threads-WFI → mtime halts → lockstep
deadlock" failure.

Because the reqrsp FIFO **busy-polls** (no RX IRQ), a lower-priority thread would be starved; the
fix is a Kconfig-gated **cooperative transport** (`CONFIG_ROSE_TRANSPORT_COOP`, threaded build
only — it needs the keepalive) that `k_usleep`s between polls so the ~80%/tick I/O-wait slack goes
to vision, plus a fine tick (`SYS_CLOCK_TICKS_PER_SEC=100000`). Control still preempts, so the
rate holds.

Validated (GPU-free probe env, A/B): **max actuator-command staleness 467 ms (single-loop, inline
inference stalls control) → 12 ms threaded** (~2 control periods) — a **~39× reduction**; control
runs at ~5 ms/tick throughout while vision completes **6 background inferences** over the run,
driving the setpoint (`cmd 0.208` from the real DroNet collision). Clean exit, no hang. The
compute-only vision thread is the on-SoC analogue of a dedicated vision core/NPU on real HW.

Repro: `tools/host_oracle/run_oracle.sh` (P2 host side); guest builds via the in-tree Zephyr
env with `-DZEPHYR_EXTRA_MODULES=<zephyr-rose> -DMODEL_DIR=<generated/scalar>`; co-sim via
`run_spike_rose_lockstep.sh` + a synchronizer (`minimal_sync.py` for P0/P1, the probe env for
P3).

## 7c. Timer-driven control — co-sim-agnostic decoupling (2026-08-05)

Replaced the lockstep pipeline with the **real-prototype paradigm**: the control loop is a
periodic `k_timer`-paced task (sensors → EKF → TinyMPC → actuate), with vision a lower-priority
background task. The app has **no co-sim awareness** — the same source runs on real HW. The
control rate is driven by the guest's own timer, NOT the simulator; if a control tick overruns,
the physics keeps stepping on the last (ZOH-latched) actuation — realistic degradation, not the
sim waiting for the SoC.

**Spike time-model facts (verified against `soc/sim/chipyard/.../riscv-isa-sim`):** this spike
uses the **deterministic internal clint** (`real_time_clint=false`, not the wall-clock
`--real-time-clint`), and `sim_t::step` ticks the clint by a **constant per round-robin**
(driven by the requested step count, not retired instructions). So **mtime advances during WFI**
(the FireSim/RTL digital-top-clock model) while `minstret`/mcycle stalls — exactly the intended
HW semantics. Empirically confirmed: a guest that only `k_msleep`s (pure WFI idle, **no
keepalive**) gets 15/15 timer wakeups with deterministic 200000-tick deltas. **The keepalive
busy-spin was a workaround for a misdiagnosed "WFI halts mtime" and is deleted; so is the
cooperative-transport hack** — with the control task WFI-idling between ticks, the idle CPU goes
to the vision task by ordinary preemption.

**Scheduling granularity (measured, both tick rates):** `k_msleep`/timeout waits **quantize to
the system tick** (+1-tick ceil: `k_msleep(1)` = 19.8 ms at 100 Hz, 1.0 ms at 10 kHz), but
**preemptive context switches are instruction-granular and tick-independent** (<100 ns at both).
This is why a thread pool parking workers on timed condvars shows ~20 ms switching at the 100 Hz
tick while directly-scheduled threads switch sub-ms. Design consequence: preemption for
vision↔control interleave; a periodic timer only for control pacing, with a fine tick
(`SYS_CLOCK_TICKS_PER_SEC=10000`, threaded-build only).

**Decoupling validated (GPU-free probe env):**
- Control period **tracks the guest timer**: CTRL_HZ=200 → `period_us=5000`, CTRL_HZ=100 →
  `period_us=9990`, while the grant quantum is 0.5 ms and Isaac runs at 2 kHz
  (`ROSE_FIRESIM_STEP=500000`, `gym_timestep=0.0005`, 1 env step/grant) — so control is paced by
  its own clock, independent of the sim step rate. The `action_latch` ZOH holds each command
  across the fine physics steps.
- Control-tick compute breakdown: **sensor reqrsp round-trip 0.5 ms (≈1 grant), EKF 1.9 ms,
  TinyMPC 1.0 ms** → ~3.4 ms floor (EKF-dominated, ~294 Hz max). Above that rate control becomes
  compute-limited (run-as-fast-as-possible) and the sim steps on stale actions.
- Vision completes in the background (no coop), DroNet drives the setpoint, no keepalive, no hang.

Open follow-up: **push-based sensors** (bridge DMAs a sensor buffer per env step + IRQ on new
data) would make sensor reads a plain memory load instead of a reqrsp round-trip — fully
non-blocking, the last coupling to remove for arbitrary control rates.

## 8. Milestone ladder (each independently validatable)

| # | Milestone | Validation | Status |
|---|-----------|------------|--------|
| 0 | DroNet int8 ELF on co-sim target | verifier bit-exact vs PyTorch golden | ✅ `[-56,127]` err=0 |
| 1 | On-SoC preproc RGB resize→int8 tensor | CRC32(guest) == CRC32(host) | ✅ `0x4aa455b2` (3-way) |
| 2 | Live camera → DroNet on SoC | on-SoC output == host oracle | ✅ `[-26,127]` byte-exact |
| 3 | DroNet → TinyMPC setpoint (collision→vx, steer→yaw) | DroNet drives setpoint, 0x20 ok, no hang | ✅ `vx 0→0.208`, 166×0x20 |
| 4 | Scheduled DroNet+policy under harness_xpurt (Isaac closed loop) | one ELF flies from live frames | ⬜ next |
