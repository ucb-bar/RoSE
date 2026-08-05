# RoSÉ Example: Closed-Loop Drone Control (TinyMPC over the RoSE bridge)

A worked, end-to-end RoSÉ example: a **quadrotor physics simulator** flies in a
closed loop with a **TinyMPC flight controller running on a simulated RISC-V SoC**,
coupled through the RoSE bridge. It exercises the full RoSÉ stack — environment ⇄
synchronizer ⇄ SoC RTL/ISA model ⇄ guest software — and is fully **headless** (no
GPU/display).

It is also a RoSÉ *result*, not just a plumbing demo: sweeping the modeled SoC clock
shows the controller **diverges at 10 MHz but hovers at 1 GHz**, because the SoC must
finish each TinyMPC solve within the control deadline. That HW/SW timing interaction
is exactly what RoSÉ exists to expose.

---

## 1. What the demo shows

```
   PyBullet quadrotor (CtrlAviary, DIRECT/headless)
        │  full 12-DoF state  [x y z, r1 r2 r3, vx vy vz, dφ dθ dψ]   (reqrsp cmd 0x12)
        ▼
   PyBulletDroneMPCEnv ──► gym synchronizer ──TCP :10001──► Spike SoC (lockstep tier)
        ▲                    (deploy/hephaestus)              └─ Zephyr guest: TinyMPC solve
        │  4 normalized motor thrusts  (action cmd 0x20)                     │
        └──────────────────────────────────────────────────────────────────┘
```

- The **SoC guest** requests the full simulator state each control step, solves a
  constrained TinyMPC problem, and returns 4 motor thrusts.
- The **synchronizer** serves state on request and applies the returned thrusts as the
  next environment action — a true closed loop.
- The controller is a RoSE port of `zephyr-chipyard-sw`'s `samples/drone_control` HIL
  controller: the physical-UART transport is replaced by the RoSE bridge, and it reads
  the **full simulator state directly** (rather than a per-sensor abstraction).

---

## 2. Component map

Repo legend: **[R]** = `ucb-bar/RoSE` (`chipyard-top`) · **[X]** = `ucb-bar/XPU-RT`
submodule (`dev`) · **[Z]** = `ucb-bar/zephyr-chipyard-sw` nested submodule
(`rose-2-dev`) · **[T]** = `ucb-bar/Accelerated-TinyMPC` (nested in Z).

### Physics side — simulator, environment, synchronizer
| Component | Path | Repo |
|---|---|---|
| PyBullet simulator (gym-pybullet-drones) | `soc/sw/xpu-rt/zephyr-chipyard-sw/tools/gym-pybullet-drones/` | [Z] |
| Drone MPC env (full-state serve, thrust action) | `deploy/hephaestus/envs/pybullet_drone/drone_mpc_env.py` | [R] |
| Env registration | `deploy/hephaestus/register_envs.py` | [R] |
| Synchronizer (core loop) | `deploy/hephaestus/gym_synchronizer.py` | [R] |
| Socket framing (buffered receiver) | `deploy/hephaestus/socket_thread.py` | [R] |
| Sync launcher | `deploy/hephaestus/run_sync_only.py` | [R] |
| Standalone env smoke test | `deploy/hephaestus/test_drone_env.py` | [R] |
| Socket framing stress/regression test | `deploy/hephaestus/tests/test_socket_thread_stress.py` | [R] |

### Config
| Config | Path | Repo |
|---|---|---|
| Drone MPC loop (reqrsp 0x12 state + action 0x20 control) | `deploy/config/config_gym_PyBulletDroneMPCEnv-v0.yaml` | [R] |
| Top-level env/timing selector (`gym_env`, `firesim_step`, `firesim_freq`) | `deploy/config/config_deploy_gym.yaml` | [R] |

### SoC side — RoSE bridge / Spike
| Component | Path | Repo |
|---|---|---|
| Lockstep harness (`rose_spike_sim`, owns the step loop) | `soc/src/main/cc/rose_spike/rose_spike_sim.cc` | [R] |
| Passive `--extlib` plugin (fast functional tier) | `soc/src/main/cc/rose_spike/rose_spike_device.cc` | [R] |
| Synchronizer-protocol client | `soc/src/main/cc/rose_sync_client.{cc,h}` | [R] |
| Spike `sim.h` step() patch (applied by `setup.sh`) | `soc/src/main/cc/rose_spike/rose_spike_sim_stepaccess.patch` | [R] |

### Guest software (runs on the SoC)
| Component | Path | Repo |
|---|---|---|
| Drone controller app (main.cpp + CMake + prj.conf + overlay) | `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose/drone_control/` | [Z] |
| TinyMPC controller (admm, matlib, problem_data) | `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/drone_control/tinympc/` | [T] |
| RoSE Zephyr driver + protocol layer | `soc/sw/zephyr-rose/{drivers,subsys,include}/rose/` | [R] |

### Build & run scripts
| Script | Path | Repo |
|---|---|---|
| Build the bridge (plugin + harness) | `soc/src/main/cc/rose_spike/build.sh` | [R] |
| Build the Zephyr guest elf(s) | `soc/sim/build_zephyr_rose.sh` | [R] |
| Run on the lockstep harness | `soc/sim/run_spike_rose_lockstep.sh` | [R] |
| Run on the `--extlib` plugin | `soc/sim/run_spike_rose.sh` | [R] |
| Spike-tier deep-dive docs | `soc/src/main/cc/rose_spike/README.md` | [R] |

### Build artifacts (gitignored — regenerated, not committed)
`soc/sim/rose_spike_sim`, `soc/sim/librose_spike.so`,
`soc/sim/zephyr_rose_builds/drone_control/zephyr/zephyr.elf`.

---

## 3. Prerequisites (one-time)

1. **Repo + chipyard setup** — follow the top-level `README.md` "Installation"
   (chipyard `build-setup.sh`, `./soc/setup.sh`, `source rose-setup.sh`).
2. **Guest-software submodules** (scoped — do NOT `--recursive` on `xpu-rt`):
   ```bash
   git submodule update --init soc/sw/xpu-rt
   git -C soc/sw/xpu-rt submodule update --init zephyr-chipyard-sw
   # TinyMPC controller (+ its matlib submodule):
   git -C soc/sw/xpu-rt/zephyr-chipyard-sw submodule update --init --recursive \
       samples/drone_control/tinympc
   ```
3. **Zephyr toolchain** — installed locally inside the `zephyr-chipyard-sw` submodule:
   ```bash
   ( cd soc/sw/xpu-rt/zephyr-chipyard-sw
     bash   scripts/install_submodules.sh        # west workspace (zephyr_ws)
     source scripts/install_conda.sh             # conda env 'zephyr' (provides west)
     bash   scripts/install_toolchain_sdk.sh )   # beta Zephyr SDK
   ```
4. **Synchronizer venv**:
   ```bash
   python -m venv deploy/.venv-rose && deploy/.venv-rose/bin/pip install -r deploy/requirements.txt
   ```

---

## 4. Build

```bash
# 4a. RoSE bridge (produces soc/sim/rose_spike_sim and librose_spike.so)
soc/src/main/cc/rose_spike/build.sh

# 4b. Zephyr guest elf -> soc/sim/zephyr_rose_builds/drone_control/zephyr/zephyr.elf
soc/sim/build_zephyr_rose.sh drone_control
```

---

## 5. Run the demo

The demo runs as two headless processes coupled over `localhost:10001`.

### 5a. Select the env + a 1 GHz SoC model
Edit `deploy/config/config_deploy_gym.yaml`:
```yaml
gym_env: 'PyBulletDroneMPCEnv-v0'
firesim_freq: 1_000_000_000     # 1 GHz model
firesim_step: 20_000_000        # 20M cycles/step => one 50 Hz control step per grant
```
Rationale: the 50 Hz TinyMPC solve (~7–8M instructions) must fit inside one control
period. At `firesim_freq=1e9`, `20M` cycles ≈ 0.02 s covers it, so control keeps up.

### 5b. Terminal 1 — synchronizer (physics)
```bash
cd deploy/hephaestus
ROSE_DIR=$(git rev-parse --show-toplevel) ROSE_ENV_DEBUG=1 \
    ../.venv-rose/bin/python run_sync_only.py
# waits: "listening on localhost:10001 — waiting for 1 bridge connection(s)..."
```
`ROSE_ENV_DEBUG=1` prints the drone trajectory so you can watch it hover.

### 5c. Terminal 2 — SoC (lockstep tier)
```bash
soc/sim/run_spike_rose_lockstep.sh \
    soc/sim/zephyr_rose_builds/drone_control/zephyr/zephyr.elf 1
```

---

## 6. Expected results & validation

**Guest side** (Terminal 2) — TinyMPC boots and the tracking error settles near zero:
```
[rose_sync] connected to localhost:10001
ROSE drone_control: TinyMPC ready, entering control loop
ROSE drone_control: iter=1   z_err=-0.887 ...
ROSE drone_control: iter=101 z_err=0.122  ...
ROSE drone_control: iter=201 z_err=0.095  ...      # error small & bounded
```

**Physics side** (Terminal 1) — the drone climbs to the setpoint (z = 1.0) and holds:
```
[drone] step   1  pos=(+0.00,+0.00,+0.11) z_err=-0.89 thrust=[0 0 0 0]
[drone] step  26  pos=(-0.00,+0.00,+1.03) z_err=+0.03 thrust=[-0.66  0.51 -0.58  0.46]
[drone] step 201  pos=(+0.00,-0.04,+1.10) z_err=+0.10 thrust=[-0.57  0.41 -0.93  0.47]
```

**Pass criteria:**
1. Guest prints `TinyMPC ready` and connects.
2. Altitude reaches ≈ 1.0 m and **stays bounded** (z_err ≲ 0.15) for hundreds of steps
   — it does **not** diverge.
3. Per-motor thrusts are **differential** (attitude actively controlled), not a constant
   vector.
4. Controls flow continuously — with `ROSE_RX_DEBUG=1` on the synchronizer, the received
   `0x20` count climbs in step with the guest's iterations (hundreds, not a dozen).

A small steady-state altitude offset (~0.1 m) is expected — the controller is
proportional (no integral term); it is a bounded hover, not drift.

### The HW/SW co-design result
Re-run with `firesim_freq: 10_000_000` (10 MHz) and the drone **diverges** (climbs away):
at 10 MHz the SoC cannot finish the TinyMPC solve within the 50 Hz control deadline, so
control lags the physics badly. This clock sweep — divergence at 10 MHz, stable hover at
1 GHz — is the headline RoSÉ finding: it quantifies the controller's compute-vs-deadline
requirement in closed loop, before any silicon exists.

---

## 7. Fast functional tier (optional)

For quick driver/app iteration (single-core, no cycle budget), swap the SoC command:
```bash
soc/sim/run_spike_rose.sh soc/sim/zephyr_rose_builds/drone_control/zephyr/zephyr.elf
```
This uses the passive `--extlib` plugin (`librose_spike.so`) instead of the lockstep
harness. It preserves transaction ordering but not cycle timing, so it will not reproduce
the clock-sweep result above.

## 7b. IsaacLab (Isaac Sim) variant — same loop, different physics

The identical guest, bridge, and loop wiring also fly a Bitcraze **Crazyflie** in NVIDIA
Isaac Sim via `IsaacCrazyflieMPCEnv-v0` — only the `gym_env` id changes. It serves the same
12-DoF target-relative state and consumes the same 4 per-motor thrusts (applied as per-rotor
forces on the propeller bodies, matching TinyMPC's per-motor output). It needs the Isaac Sim
toolchain (the same IsaacLab version pinned at `soc/sw/xpu-rt/sims/IsaacLab`); run the
synchronizer with an Isaac-capable Python (e.g. the `env_isaaclab` conda env). Setup, the
per-rotor model, and the validated hover result (z 0.50 → settles 1.02 at 1 GHz) are in
[`deploy/hephaestus/envs/isaac_crazyflie/README.md`](../deploy/hephaestus/envs/isaac_crazyflie/README.md).

---

## 8. Troubleshooting & notes

- **No display needed.** PyBullet runs in `p.DIRECT` mode (`gui=False`); Spike uses the
  HTIF console; the synchronizer is a TCP server. Observe via stdout/log files.
- **Bridge framing.** High-rate bidirectional traffic (state + control every step) relies
  on the buffered receiver in `socket_thread.py`. The regression test
  `deploy/hephaestus/tests/test_socket_thread_stress.py` guards it; run it after any
  change to the socket path:
  ```bash
  ROSE_DIR=$(git rev-parse --show-toplevel) deploy/.venv-rose/bin/python \
      deploy/hephaestus/tests/test_socket_thread_stress.py   # expect "ALL PASS"
  ```
- **Wall-clock speed.** At 1 GHz the harness runs ~20M guest instructions per grant, so
  the functional lockstep tier is wall-clock-slow (seconds per control step). Reaching a
  visible hover takes a few hundred steps; be patient or lower `firesim_freq`/`firesim_step`
  proportionally (which trades timing fidelity for speed).
- **Debug switches.** `ROSE_ENV_DEBUG=1` (env trajectory), `ROSE_RX_DEBUG=1`
  (synchronizer received-packet counts), `ROSE_SPIKE_DEBUG=1` (bridge TX/route/DMA/IRQ).
