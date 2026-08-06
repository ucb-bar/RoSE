# IsaacLab Crazyflie RoSE environment (`IsaacCrazyflieMPCEnv-v0`)

The **Isaac Sim / IsaacLab** counterpart of `PyBulletDroneMPCEnv-v0`. It flies a Bitcraze
**Crazyflie** (`cf2x`) inside NVIDIA Isaac Sim and presents the **exact same RoSE contract**
as the PyBullet loop, so the *unchanged* Zephyr guest (`samples/rose/drone_control`, a
TinyMPC controller) and the *unchanged* loop wiring drive either simulator — only the
`gym_env` id differs.

```
Isaac Sim Crazyflie --full 12-DoF state--> RoSE bridge --> Zephyr TinyMPC controller
        ^                                                            |
        +----------- 4 normalized thrusts (per rotor) <-- RoSE bridge <-+
```

- **State served** (reqrsp, 12×f32, target-relative): `[x,y,z, r1,r2,r3, vx,vy,vz, dphi,dtheta,dpsi]`,
  Rodrigues attitude — byte-identical layout to the PyBullet MPC env.
- **Control consumed** (action-latch, 4×f32): normalized per-motor thrusts around the 0.583
  hover point. These are applied as **per-rotor z-forces at the 4 propeller bodies**
  (`m.*_prop`), exactly like IsaacLab's own `scripts/demos/quadcopter_fpv.py`. Per-rotor
  forces reproduce collective thrust + roll/pitch from the arm geometry; the propeller drag
  reaction (yaw) is added as a per-rotor z-torque with alternating spin sign.

This per-motor model (vs. the collective-thrust + body-moment RL env) is the natural match
for TinyMPC's 4-thrust output.

## Requirements

Needs the Isaac Sim toolchain. **Concrete live setup (host `garden`, verified 2026-08-05):**
run the synchronizer with the **`env_isaaclab` conda env** —
`/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python` (Python 3.11, `isaacsim` +
`isaaclab` v0.54.3 editable from **`/scratch2/dima/IsaacLab`**, bootstrapped from the
FreshScheduler IsaacLab submodule). **Not** `deploy/.venv-rose` (no Isaac). The repo's own
`soc/sw/xpu-rt/sims/IsaacLab` submodule (pinned `isaac-sim/IsaacLab@4df6560`) is provenance
only and is currently **unpopulated** — the live install is `/scratch2/dima/IsaacLab`. The env
is imported **lazily** (string entry_point), so the synchronizer stays dependency-light unless
this env is actually `gym.make()`d.

> **Full path-concrete flow (Spike co-sim + video rendering): [`docs/ROSE_ISAAC_COSIM_FLOW.md`](../../../../docs/ROSE_ISAAC_COSIM_FLOW.md).**

## Run (Spike path)

Point the deploy config at this env and use the 1 GHz timing (the setting that hovers):

```yaml
# deploy/config/config_deploy_gym.yaml
gym_env: 'IsaacCrazyflieMPCEnv-v0'
firesim_freq: 1_000_000_000     # 1 GHz modeled SoC
firesim_step: 20_000_000        # 20M cyc / 1e9 = 0.02 s = 50 Hz => 1 Isaac step / SoC step
max_sim_time: 3.0
```

```bash
# Terminal 1 — synchronizer (Isaac Sim boots in-process; ~1 min first launch)
cd deploy/hephaestus
ROSE_DIR=$(git rev-parse --show-toplevel) \
  /scratch2/dima/miniforge3/envs/env_isaaclab/bin/python run_sync_only.py   # waits: "listening on localhost:10001..."

# Terminal 2 — Spike lockstep bridge with the TinyMPC guest
soc/src/main/cc/rose_spike/build.sh                          # once (-> soc/sim/rose_spike_sim)
soc/sim/run_spike_rose_lockstep.sh \
    soc/sim/zephyr_rose_builds/drone_control/zephyr/zephyr.elf 1
```

Set `ROSE_ENV_DEBUG=1` for the per-step `[isaac-cf]` trajectory print.

## Validated result

Closed loop over the Spike tier at 1 GHz: the drone starts at z=0.50 and the SoC-resident
TinyMPC controller drives it to a **stable hover** at the 1.0 m setpoint —
`z 0.50 → 1.03 → settles 1.02` (Isaac side), `z_err −0.500 → +0.023 steady` (SoC side). The
~0.02 m steady-state offset is expected (proportional control, no integral), matching the
PyBullet loop. (As with PyBullet, a 10 MHz SoC misses the 50 Hz deadline and diverges.)

## Logging: ground-truth trajectory + camera video

The wire and the log are separate channels (see the repo-wide logging note): `obs` carries
only what the SoC consumes; the **ground truth** (absolute world pose/velocity, per-rotor
forces, target, sim time) is returned in the gym `info` dict, which is never serialized.

- **Trajectory CSV + plot.** Set `traj_csv` (or `$ROSE_TRAJ_CSV`) and the env records the
  ground truth every control step. Render it with:
  ```bash
  # during the run:  ROSE_TRAJ_CSV=/tmp/traj.csv on the synchronizer process
  python envs/isaac_crazyflie/plot_trajectory.py /tmp/traj.csv          # -> /tmp/traj.png
  ```
  The figure has the 3D path, altitude-vs-setpoint, horizontal drift, and per-rotor thrust.
  (The plotter is env-agnostic — any drone env that writes the same CSV columns works.)

- **Camera video.** Set `camera: true` (or `$ROSE_ISAAC_CAMERA=1`). The env adds an
  offscreen chase camera that keeps the Crazyflie framed; `render()` returns an `(H,W,3)`
  uint8 RGB frame each step (assemble frames into a video, e.g. via `imageio.mimsave`, or
  let `GymLogger.save_video` do it). Cameras must be enabled at Isaac boot, so this is a
  constructor-time flag, not a runtime toggle, and it is **much** slower than the headless
  no-camera path — leave it off for timing runs.

## Key knobs (`gym_kwargs`)

| kwarg | default | meaning |
|---|---|---|
| `ctrl_freq` | 50 | control rate (must match the TinyMPC problem data) |
| `phys_freq` | 100 | physics rate; `decimation = phys_freq // ctrl_freq` sub-steps per control tick |
| `headless` | true | boot Isaac Sim without a GUI |
| `target` | `[0,0,1]` | hover setpoint (state is served target-relative) |
| `start_height` | 0.5 | reset altitude |
| `traj_csv` | `$ROSE_TRAJ_CSV` | path to write the ground-truth trajectory CSV (off if unset) |
| `camera` | `$ROSE_ISAAC_CAMERA` | add the offscreen chase camera so `render()` returns RGB |
| `camera_res` / `camera_focal` | `(640,480)` / `35` | frame size / lens (higher focal = tighter framing) |
| `camera_follow` / `camera_offset` | `true` / `(0.45,0.45,0.18)` | chase the drone; eye = pos+offset |
| `log_dir` | `<tmp>/isaaclab-<uid>/logs` | per-user log dir (avoids the shared `/tmp/isaaclab` owner clash) |
