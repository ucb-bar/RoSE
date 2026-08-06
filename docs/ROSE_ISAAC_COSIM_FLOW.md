# RoSE ↔ IsaacLab co-simulation flow (Spike tier) + video rendering

Authoritative, **path-concrete** reference for running a RoSE lockstep co-sim against an
**NVIDIA Isaac Sim / IsaacLab** environment, and for rendering IsaacLab videos of the run.
This is the flow that had been reconstructed by memory before; the concrete environment
locations below are the part that keeps getting lost — they are recorded here so they don't.

```
IsaacLab Crazyflie (GPU physics) --sensors/state--> RoSE bridge --> Spike (rose_spike_sim) --> Zephyr guest
        ^                                                                                          |
        +----------------------- 4 normalized per-rotor thrusts <-- RoSE bridge <-----------------+
```

The Zephyr guest is **unchanged** between real hardware, the PyBullet CPU loop, and this Isaac
loop — only the `gym_env` id (and thus the physics backend + sensor synthesis) differs.

---

## 1. The concrete environment (the part that gets lost)

| Piece | Location (verified 2026-08-05) |
|---|---|
| **Conda env** with Isaac Sim | `/scratch2/dima/miniforge3/envs/env_isaaclab` — Python **3.11.15**; `isaacsim` + `isaaclab` (v0.54.3) import cleanly. |
| Env's python (use THIS to run the synchronizer) | `/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python` |
| **Active IsaacLab source** (editable install) | `/scratch2/dima/IsaacLab` (`source/isaaclab`, `source/isaaclab_assets`, …). This is what `import isaaclab` resolves to. |
| Bootstrap origin | The **FreshScheduler** IsaacLab submodule at `/scratch2/dima/misc_sw/FreshScheduler/sims/IsaacLab` — the env was bootstrapped from there, then migrated to the standalone `/scratch2/dima/IsaacLab` editable checkout. |
| Isaac Sim app / kit | `/scratch2/dima/IsaacLab/apps/isaaclab.python.headless.kit` (headless; loaded automatically). |
| GPU | `cuda:0` (host `garden`: NVIDIA TITAN RTX). |

> **Do NOT use `deploy/.venv-rose`** for the Isaac path — it is the light synchronizer venv and
> has **no** `isaacsim`/`isaaclab`. It is only for the CPU (PyBullet/AirSim) envs.
>
> **The repo's own submodule `soc/sw/xpu-rt/sims/IsaacLab` is EMPTY** (not checked out) and is
> **not** what runs. The env README pins `isaac-sim/IsaacLab@4df6560` there for provenance, but
> the live install is `/scratch2/dima/IsaacLab` inside `env_isaaclab`.

`isaaclab_assets` importing standalone raises `ModuleNotFoundError: pxr` — that is **normal**;
`pxr`/USD only exists after the Isaac app boots (via `AppLauncher`), which the env does in-process.

---

## 2. Environments (gym ids)

Registered in `deploy/hephaestus/register_envs.py`; Isaac envs are **lazily** imported (string
entry points) so the synchronizer stays light unless one is actually `gym.make()`d.

| gym id | Serves the SoC | Use |
|---|---|---|
| `IsaacCrazyflieMPCEnv-v0` | full ground-truth 12-DoF state | drive TinyMPC directly (`samples/rose/drone_control`) |
| **`IsaacCrazyflieSensorEnv-v0`** | **IMU accel+gyro, optical flow, down-ToF** (no ground truth) | **the flight controller** (`samples/rose_flight_controller`) — runs a state estimator |
| `IsaacCrazyflieMultiSensorEnv-v0` | + multizone ToF + FPV camera | WIP (vision nav) |

CPU counterparts (no GPU, `deploy/.venv-rose`): `PyBulletDroneMPCEnv-v0`, `PyBulletDroneEnv-v0`.

---

## 3. Config & selection

- **`deploy/config/config_deploy_gym.yaml`** — picks the env + SoC timing. Current default:
  ```yaml
  gym_env: 'IsaacCrazyflieSensorEnv-v0'
  firesim_freq: 1_000_000_000     # 1 GHz modeled SoC
  firesim_step: 5_000_000         # 5M cyc / 1e9 = 0.005 s = 200 Hz  (1 env step / SoC step)
  max_sim_time: 12.0              # simulated seconds
  ```
- **`deploy/config/config_gym_<gym_env>.yaml`** — per-env packet bindings (which obs field →
  which reqrsp cmd/channel, and the action-latch). For the sensor env: accel `0x12`(ch2),
  gyro `0x15`(ch2), flow `0x13`(ch1), tof `0x14`(ch1), thrust `0x20`(action_latch). These MUST
  match the guest overlay `boards/spike_riscv64.overlay`.
- **Env overrides** (no need to edit the committed yaml):
  `ROSE_GYM_ENV`, `ROSE_FIRESIM_STEP`, `ROSE_FIRESIM_FREQ`, `ROSE_SYNC_PORT` (default 10001),
  `ROSE_TRAJ_CSV`, `ROSE_ISAAC_CAMERA`, `ROSE_ENV_DEBUG=1` (per-step `[isaac-cf]` print).

`gym_timestep` in the per-env yaml MUST equal `firesim_step/firesim_freq` (→ 1 control per env step).

---

## 4. Run it (two terminals, Spike lockstep tier)

**Terminal 1 — Isaac synchronizer** (boots Isaac Sim in-process; ~1 min first launch, then listens):
```bash
cd /scratch/dima/rose-infra/RoSE/deploy/hephaestus
ROSE_DIR=/scratch/dima/rose-infra/RoSE \
ROSE_TRAJ_CSV=/tmp/traj_flightctrl.csv \
  /scratch2/dima/miniforge3/envs/env_isaaclab/bin/python run_sync_only.py
# wait for:  [run_sync_only] listening on localhost:10001 — waiting for 1 bridge connection(s)...
```

**Terminal 2 — Spike lockstep bridge with the guest elf:**
```bash
cd /scratch/dima/rose-infra/RoSE
# build the guest once (spike target, links the zephyr-rose module):
bash soc/sim/build_zephyr_rose.sh rose_flight_controller
# rose_spike_sim already built at soc/sim/rose_spike_sim (else: soc/src/main/cc/rose_spike/build.sh)
ROSE_SPIKE_TIMEOUT=300 bash soc/sim/run_spike_rose_lockstep.sh \
  soc/sim/zephyr_rose_builds/rose_flight_controller/zephyr/zephyr.elf 1
```

The bridge connects to the waiting synchronizer on `ROSE_SYNC_PORT`; the co-sim then steps in
lockstep until `max_sim_time` (sync side) or the guest's `CTRL_ITERS` (SoC side). When one side
ends, the other sees the closed connection and shuts down cleanly.

> **Wall-clock note.** Isaac headless steps at ≈0.5 s/step here, so `max_sim_time: 12.0` @ 200 Hz
> (2400 steps) is ~20 min wall. For a quick check, lower `max_sim_time` (e.g. 2–3 s) or just let
> `ROSE_SPIKE_TIMEOUT` cut it — the takeoff+settle is visible in the first ~1.5 s. Run both sides
> **backgrounded** for long/full runs (don't wrap the spike side in a 2-min foreground shell).

### Firesim-manager path (metasim/FPGA), for reference
`run_sync_only.py` is the manager-less launcher (pair with a directly-launched Spike or VFireSim).
`deploy/hephaestus/rose.py --task run` is the full path that couples the synchronizer to
`firesim runworkload`. The Spike two-terminal flow above is the day-to-day one.

---

## 5. Logging & video

The wire carries only what the SoC consumes; **ground truth** (world pose/vel, per-rotor forces,
target, sim time) rides in the gym `info` dict and is logged separately.

- **Trajectory CSV.** `ROSE_TRAJ_CSV=/path/traj.csv` (or the `traj_csv` gym_kwarg) → one row per
  control step. Columns: `t,x,y,z,qw..qz,vx..vz,wx..wz,f0..f3,u0..u3,tx,ty,tz`. Plot with:
  ```bash
  /scratch2/dima/miniforge3/envs/env_isaaclab/bin/python \
    deploy/hephaestus/envs/isaac_crazyflie/plot_trajectory.py /path/traj.csv   # -> /path/traj.png
  ```
- **IsaacLab video.** `ROSE_ISAAC_CAMERA=1` (or `camera: true` gym_kwarg) adds an offscreen chase
  camera; `env.render()` then returns an `(H,W,3)` uint8 RGB frame each step. The synchronizer's
  `GymLogger` collects frames and `save_video()`s on close. Outputs land in
  **`deploy/hephaestus/logs/`**: `recording-<base>.avi` (video), `runlog-<base>.csv`,
  `plot-<base>.png`. Camera must be enabled at Isaac boot (constructor flag, not a runtime toggle)
  and is **much** slower than headless — leave it off for timing runs, on for demo videos.
  Knobs: `camera_res` (default 640×480), `camera_focal` (35), `camera_follow`/`camera_offset`.

---

## 6. Validated result (2026-08-05, `IsaacCrazyflieSensorEnv-v0` + `rose_flight_controller`)

Full loop over the Spike tier at 1 GHz / 200 Hz. The SoC ran the complementary estimator
(Mahony + fixed gain) on IMU+flow+ToF (no ground truth) feeding TinyMPC; the RoSE virtual ToF
delivered valid height (`tof=1`). Ground-truth altitude (Isaac side), setpoint 1.0 m, start 0.9 m:

```
t=0.005 z=0.900   t=0.205 z=0.997   t=0.405 z=1.075   t=0.605 z=1.061   t=0.805 z=1.005
t=1.005 z=0.996   t=1.205 z=1.024      final z=1.024  (min 0.900, max 1.080)
```

Clean takeoff → mild overshoot to 1.08 m → damped settle at ~1.02 m. The ~0.02 m steady-state
offset is expected (proportional TinyMPC, no integral) and matches the PyBullet and MPC-env
results. Run was cut at step 244 (~1.2 s sim) by a 2-min foreground shell limit, not a fault —
the hover was already established. SoC side reported level attitude (roll/pitch ≈ 0) throughout.

---

## 7. Gotchas

- **Wrong python** → `ModuleNotFoundError: isaacsim`. Use the `env_isaaclab` python, not `.venv-rose`.
- **Isaac boot ~1 min**; don't assume the synchronizer hung — watch for the `listening on …` line.
- **`max_sim_time` vs wall time**: headless ≈0.5 s/step; a foreground shell with a short timeout
  will SIGTERM the spike process group mid-run (the sync then logs `bridge closed … co-sim ended`).
  Background both sides for full runs.
- **Port collisions**: set `ROSE_SYNC_PORT` per run if sharing the host (the parallel stress harness
  does this per cell).
- **`/tmp/isaaclab` owner clash** on shared hosts → set `log_dir`/`ROSE_*` per user (the env defaults
  to a per-uid `/tmp/isaaclab-<uid>`).
- **`pxr` import error** for `isaaclab_assets` outside a booted app is normal (see §1).

See also: `deploy/hephaestus/envs/isaac_crazyflie/README.md` (env internals, MPC-env flow),
`docs/ROSE_DRONE_MPC_DEMO.md`, `docs/ROSE_FUSED_NAV_INTEGRATION.md`, and the guest overlay
`soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_flight_controller/boards/spike_riscv64.overlay`.
