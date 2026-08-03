# RoSE Flight Controller Stress-Test Plan

Design / implementation plan for stress-testing the sensor-based RoSE drone flight
controller against realistic sensor imperfections and harder flight scenarios. This is a
**planning document only** — no code is changed by adding it.

## 0. Baseline recap (what exists today)

The working loop (see `docs/ROSE_DRONE_MPC_DEMO.md` and `docs/ROSE_SENSOR_ABSTRACTION.md`):

- **Env** `deploy/hephaestus/envs/isaac_crazyflie/crazyflie_sensor_env.py` synthesizes
  sensors from Isaac Sim ground truth per control step:
  - accel = `R^T (a_world - g)` (`crazyflie_sensor_env.py:110-111`), where `a_world` is
    either Isaac's `body_lin_acc_w` or a finite-difference of `root_lin_vel_w`
    (`:100-108`, selected by `ROSE_ACCEL_SRC`).
  - gyro = `root_ang_vel_b` (`:113`).
  - optical flow = body-frame horizontal velocity `root_lin_vel_b[:2]` (`:118`).
  - ToF height = true `pos[2]`, refreshed every `_tof_period` steps and **held** in
    between (`:70-72`, `:115-118`), modeling a ~30 Hz VL53L1x. `ROSE_TOF_PERIOD=6` at
    200 Hz.
  - **Sensors are CLEAN** — no noise, bias, or transport delay (`:32-33`).
  - Full 12-DoF ground truth is returned in `info` (`:131-143`) and, if
    `ROSE_TRAJ_CSV`/`traj_csv` is set, written to a CSV by the base env
    (`crazyflie_mpc_env.py:286-303`, `_TRAJ_COLS` at `:60-69`). **This CSV is the primary
    scoring artifact for every experiment below.**
- **Physics base** `crazyflie_mpc_env.py`: per-rotor wrench via
  `permanent_wrench_composer.set_forces_and_torques(...)` at the 4 `m.*_prop` bodies
  (`:267-279`); `reset()` writes root pose/velocity and sets `start_height`
  (`:305-325`); `step()` sub-steps physics `decimation` times re-asserting the wrench
  (`:327-360`). Hover setpoint `DEFAULT_TARGET = [0,0,1.0]` (`:47`).
- **Loop wiring** `config_gym_IsaacCrazyflieSensorEnv-v0.yaml`: 200 Hz (`gym_timestep:
  0.005`), IMU on reqrsp ch2 (cmd `0x12`), flow on ch1 (cmd `0x13`), thrust
  `action_latch` (cmd `0x20`), `channel_bandwidth: [[0,0,0]]` (unlimited),
  `gym_kwargs.start_height: 0.9`. Top-level timing in `config_deploy_gym.yaml`
  (`firesim_freq`/`firesim_step`); the synchronizer enforces
  `firesim_period % gym_timestep == 0` (`gym_synchronizer.py:164-169`).
- **Guest** `samples/rose_flight_controller/`: requests IMU+flow each step, runs a
  modular estimator behind `IStateEstimator` (`src/estimator.hpp`) — EKF default
  (`estimator_ekf.*`), complementary fallback (`estimator_complementary.*`), shared
  Mahony attitude (`attitude_mahony.hpp`) — then TinyMPC. Notable current behaviors:
  - **Model-based delay compensation** already present: EKF predicts `delay_steps=1` ahead
    in `get_state()` (`estimator_ekf.cpp:56-86`, `.hpp:96`); complementary uses
    `lead`/`lead_att` (`estimator_complementary.cpp:58-82`).
  - EKF measurement/process variances are hard-coded in `init()`
    (`estimator_ekf.cpp:10-22`): `r_flow=4e-4` (~0.02 m/s std), `r_tof=1e-4` (~0.01 m
    std), process accel noise `sigma_a=2.0`.
  - Mahony gain `kp=0.5` is deliberately **low** and gravity-trim is gated to
    `|accel|≈g` (`attitude_mahony.hpp:20-49`); the comment explicitly says raise `kp`
    "once gyro noise/bias is modeled" (`:26-28`) — a direct hook for this plan.
  - x,y position error is **zeroed** before TinyMPC (`main.cpp:193-194`) because position
    is unobservable from this sensor set; only velocity is regulated. Altitude is
    observable via ToF.
  - ToF fused only every `ROSE_TOF_PERIOD` steps via `tof_valid` (`main.cpp:181-182`).

Everything below is designed to **preserve the existing RoSE contract** (same cmds/
channels/action) so the guest and synchronizer run unchanged unless a specific item
calls for a guest change.

---

## 1. Sensor noise / bias models (in `crazyflie_sensor_env.py`)

### 1.1 Where and how to inject
Add a small, self-contained **sensor-corruption layer** applied to the clean values in
`_obs_info()` **after** synthesis (`crazyflie_sensor_env.py:110-118`) and **before** the
`imu`/`flow` arrays are assembled (`:120-123`). Keep the clean values in `info` (add
`gt_imu`/`gt_flow` alongside the existing `gt_*`) so estimator error can be scored
against the truth the corruption was applied to.

**Parameterization (reproducible + sweepable):** accept a single `sensor_noise` dict in
`gym_kwargs` (parsed in `__init__`), with a top-level `seed` and per-modality
sub-dicts. Mirror each field to an env var (as `ROSE_ACCEL_SRC`/`ROSE_TOF_PERIOD`
already are, `:68`/`:72`) so sweep scripts can override without editing YAML. Draw all
randomness from a dedicated `np.random.default_rng(seed)` stored on the env and
**re-seeded in `reset()`** (extend `reset()` at `:154-158`) so each run is bit-for-bit
reproducible and a seed sweep gives independent trials. A master switch
`sensor_noise.enabled: false` keeps the current clean behavior the default (no
regression to the working demo).

### 1.2 Per-modality models and realistic Crazyflie-class magnitudes

All magnitudes are per-axis, for a Crazyflie 2.x (BMI088 IMU, PMW3901 flow, VL53L1x ToF)
at the 200 Hz control rate. Bias states evolve at the **control dt** (`self._ctrl_dt`,
`:60`).

| Sensor | Model | Nominal (`level 1`) | Aggressive (`level 2`) |
|---|---|---|---|
| **Accel** | white Gaussian + slow bias (1st-order random walk, `b += N(0,σ_bw)·√dt`, τ≈100 s) + small scale-factor err | white σ = 0.15 m/s²; bias walk σ_bw = 0.01 m/s²/√s, init bias ±0.1 m/s²; scale ±0.5% | white 0.4; σ_bw 0.03, init ±0.3; scale ±1.5% |
| **Gyro** | white Gaussian + bias random walk (the classic ARW+RRW) | white σ = 0.02 rad/s; bias walk σ_bw = 0.002 rad/s/√s, init bias ±0.01 rad/s | white 0.05; σ_bw 0.006, init ±0.03 |
| **Optical flow** | white Gaussian **scaled by 1/height** (flow SNR falls as ground recedes / low light) + Bernoulli **dropout** (hold last good sample, flag) + occasional outlier | base σ = 0.01 m/s at 1 m, ×(1/h clamp 0.3–3), dropout p = 0.5% | base σ 0.03, dropout p 3%, outlier p 0.2% (±0.5 m/s) |
| **ToF** | Gaussian + **quantization** (~1 mm LSB, use ~5 mm to be safe) + already-modeled low rate/hold; add per-sample **latency** (§2); range clip 0.03–4 m | σ = 0.008 m, q = 0.005 m | σ = 0.02 m, q = 0.01 m, dropout p 2% |

Implementation notes: flow noise multiplier uses the true height already available as
`pos[2]` (`:116`). Dropout = with prob `p`, re-emit the previously emitted (corrupted)
value and set an info flag `flow_valid=False` — this is the hook the guest needs in §4.
Keep corruption in float64, cast at the end (`:120-123`) as today.

---

## 2. Sensor delay / latency

### 2.1 Env-side transport/sampling delay
Model each modality's end-to-end delay with a **per-modality ring buffer** in the env.
On each `_obs_info()` call, push the freshly corrupted sample into that modality's deque
and **emit the sample `d_k` steps old** (`collections.deque(maxlen=d+1)`; emit
`buf[0]`). Configure integer delays (in control steps) per modality via the same
`sensor_noise` dict / env vars:

- accel/gyro: `d = 0–1` step (IMU is fast; ~1–5 ms → 0–1 step at 200 Hz).
- optical flow: `d = 1–2` steps (PMW3901 integration + SPI, ~5–10 ms).
- ToF: `d = 1–3` steps **on top of** the existing hold cadence (`_tof_period`, `:72`);
  VL53L1x measurement latency is real and comparable to its period.

Reset all ring buffers in `reset()` (extend `:154-158`). Sub-step note: sensors are
synthesized once per control step (not per physics sub-step), so delay is naturally in
control-step units — consistent with the synchronizer's one-control-per-step wiring
(`config_deploy_gym.yaml` step==gym_timestep).

### 2.2 Interaction with the guest's existing delay compensation
The guest **already** predicts forward to counter *actuation* latency (`delay_steps=1`
in `estimator_ekf.cpp:21,61`; `lead` in complementary). Adding *sensor* delay stacks a
second, opposite-direction lag: the estimate is now built from stale measurements, so the
effective compensation horizon should grow to `actuation_delay + sensor_delay`. Plan:
1. Treat the EKF's `delay_steps` / complementary `lead` as the knob that must cover the
   **total** loop delay. Expose it (build-time `-D` or a small compile-time constant) so a
   sweep can match it to the env's injected `d`.
2. Verify the model-based predictor degrades gracefully: prediction amplifies gyro/accel
   noise (already why `lead_att=0.5 < lead=1.0`, `estimator_complementary.cpp:20-22`), so
   larger horizons interact with §1 noise — this is exactly the coupling the stress matrix
   (§5) must probe, not tune away blindly.
3. Do **not** hand the guest a "measurement age" signal in v1 (keeps the wire contract
   fixed); if delay-robustness proves insufficient, §4 lists adding a timestamp.

---

## 3. Harder scenarios

All are injected on the **physics side** so the guest sees only sensors. Add a
`scenario` dict to `gym_kwargs` (again env-var mirrored, RNG-seeded).

### 3.1 Harder initial conditions
Extend `reset()` in `crazyflie_mpc_env.py:305-325` (the sensor env inherits it). Today it
forces level, at-rest, at `start_height`. Add optional randomized offsets to
`root_state` before `write_root_pose_to_sim`/`write_root_velocity_to_sim` (`:313-314`):
- position offset: ±0.2 m lateral, ±0.15 m in z about `start_height`.
- initial attitude: ±10–20° roll/pitch (build a quaternion, write into `root_state[:,3:7]`).
- initial velocity: ±0.3 m/s lateral, ±0.2 m/s vertical, ±0.5 rad/s body rate
  (`root_state[:,7:]`).
Note the guest's estimator `init()` assumes a **known** takeoff pose (`main.cpp:152`,
`START_Z=0.9`); a large IC mismatch is itself a stress input (estimator convergence from a
wrong prior). Keep offsets modest at first so the estimator still catches up.

### 3.2 External disturbances (wind / force impulses)
Add a **body-frame external wrench** as a *second* wrench, separate from the rotor wrench.
The base env's `_apply_wrench()` (`:267-279`) sets rotor forces at the prop bodies; add a
sibling that applies a force/torque at the **root/base body** (`self._base_id`, already
found in the sensor env `:63`) via the same
`permanent_wrench_composer.set_forces_and_torques(...)` API (or a dedicated composer
call) each sub-step in `step()` (`:336-340`). Disturbance profiles:
- **Steady wind:** constant world-frame force (rotate into body), e.g. 0.02–0.05 N
  (Crazyflie mass ≈ 0.027 kg → ~0.7–1.9 m/s² — a stiff breeze). Optionally add a drag
  term ∝ velocity for realism.
- **Gust / impulse:** a force of 0.1–0.3 N for 50–150 ms at scheduled or Poisson times.
- **Torque impulse:** small yaw/roll kick (tests attitude recovery + Mahony gating).
Parameterize by magnitude, direction, start time, duration, and (for wind) a slow
Ornstein–Uhlenbeck component so it is not a pure step.

### 3.3 Setpoint changes / step commands
The setpoint lives in **two** places: the env `target` (`crazyflie_mpc_env.py:47,95`,
used only for logging/`gt` in the sensor env) and the **guest** `setpoint`
(`main.cpp:155`, `TARGET_Z=1.0`). Because the sensor env sends raw sensors (not
target-relative state), setpoint changes must be commanded **in the guest**. Plan:
- v1 (no wire change): script a time-varying `setpoint[]` in `main.cpp` — e.g. a z step
  1.0→1.3 m at t≈8 s, and a commanded lateral velocity step (set `err[6]/err[7]` targets)
  since x,y position is unobservable (`main.cpp:193-194`). This exercises climb/descend
  and translation without new plumbing.
- v2 (optional): add a `setpoint` reqrsp cmd so the env can drive step commands; only if
  scripted setpoints prove insufficient.
Keep the env `target` in sync for correct error metrics (§5).

### 3.4 Mass / thrust mismatch (model error)
The controller's thrust mapping assumes fixed constants (`_HOVER_THRUST=0.583`,
`_MAX_THRUST_N`, `crazyflie_mpc_env.py:51-52`) and TinyMPC is tuned to CF2X mass. Inject a
mismatch **in the env only** (guest unchanged, so it must reject it):
- scale applied rotor force by `thrust_scale` (e.g. 0.9–1.1) in `_thrusts_to_forces()`
  (`:281-284`) or `_apply_wrench()`.
- change vehicle mass via the payload/mass API at build (`self._robot_mass`, `:193`) —
  e.g. +10–20% payload — which shifts the true hover thrust away from 0.583.
This directly stresses the integrator-free controller's steady-state altitude
(a persistent thrust bias → altitude offset the ToF/EKF must absorb).

---

## 4. Estimator / controller changes to cope (prioritized)

The estimator is modular (`IStateEstimator`), so each item is a drop-in behind the
interface or a localized change. **Priority order** (do the cheap, high-value ones first):

1. **Gyro/accel bias states in the EKF** *(highest value)*. Once §1 injects bias, the
   fixed Mahony `kp=0.5` and dead-reckoned velocity will drift. Add per-axis gyro bias
   estimation (augment the Mahony integrator or add bias states) and an accel-bias term in
   the vertical KF. This is the change the code comment at `attitude_mahony.hpp:26-28`
   anticipates ("raise kp again once gyro noise/bias is modeled"). Enables raising `kp`
   for faster attitude correction once bias is observable.
2. **Flow-dropout / validity handling** *(cheap, needed as soon as §1.2 dropout is on)*.
   The `update()` signature already gates ToF with `tof_valid` (`estimator.hpp:37-38`);
   add a symmetric `flow_valid` and skip `update_vel` on dropout (predict-only, let
   covariance grow) — mirrors the existing ToF path (`estimator_ekf.cpp:51-53`). Requires
   the env dropout flag from §1.2 to reach the guest — smallest wire change is to encode
   validity in a NaN/sentinel in the existing flow packet (no new cmd).
3. **Outlier rejection on ToF and flow** *(cheap)*. Add a normalized-innovation gate
   (χ² test: reject if `resid²/S > k`) in `Kf2::update_pos/update_vel`
   (`estimator_ekf.hpp:54-77`). Directly addresses the ToF/flow outliers in §1.2 and the
   ToF range clip.
4. **Measurement-noise tuning / adaptivity** *(medium)*. `r_flow`/`r_tof`
   (`estimator_ekf.cpp:19-20`) are hand-set for clean data; make them match the §1 std,
   and optionally scale `r_flow` by height (the flow-noise-vs-height model, §1.2) since the
   env makes flow noisier when high.
5. **Delay/latency budget** *(medium)*. Grow `delay_steps`/`lead` to the total loop delay
   (§2.2); only add a per-measurement timestamp to the wire if model-based prediction
   proves insufficient — deferred to avoid a contract change.
6. **Controller robustness** *(lower priority, only if §3.4 needs it)*. TinyMPC has no
   integrator, so a persistent thrust/mass bias yields a steady altitude offset. Options:
   a slow altitude-bias estimator feeding a thrust trim, or widen the constraints. Prefer
   fixing it in the estimator (accel-bias state, item 1) over changing the tuned MPC.

---

## 5. Metrics and pass/fail

### 5.1 Quantitative metrics (all from the ground-truth traj CSV + the guest boot/iter log)
Score offline from the `ROSE_TRAJ_CSV` columns (`crazyflie_mpc_env.py:60-69`), over the
**steady window** (exclude the first ~3 s takeoff transient):

| Metric | Definition | Source |
|---|---|---|
| Altitude RMS error | RMS(`z − target_z`) | traj `z`, `tz` |
| Altitude ripple | peak-to-peak `z` in steady window | traj `z` |
| Horizontal drift rate | ‖(x,y)‖ growth per second (drift expected, bound the rate) | traj `x,y` |
| Velocity RMS | RMS‖(vx,vy,vz)‖ | traj `vx,vy,vz` |
| Attitude excursion | max |roll|,|pitch| from quaternion | traj `qw..qz` |
| Estimator error | ‖est − gt‖ for z, v, attitude (needs guest to also emit its estimate, or reconstruct offline from the same corrupted sensors) | `gt_*` vs est |
| Survival time | time until crash (z<0.05) / divergence (‖pos‖>threshold) / NaN | traj |
| Control effort | RMS/‖u‖ and saturation fraction (u at ±limits) | traj `u0..u3` |
| Solve headroom | TinyMPC compute cycles vs deadline (already logged, `main.cpp:207-210`) | guest log |

### 5.2 Pass / fail thresholds (baseline hover, tighten per level)
- **PASS (nominal, level 1):** survives full run (≥10 s steady), altitude RMS < 0.03 m,
  ripple < 0.05 m, velocity RMS < 0.1 m/s, max tilt < 10°, no saturation > 5% of steps,
  solve within deadline.
- **DEGRADED:** survives but exceeds one PASS bound by <2×.
- **FAIL:** crash, divergence, NaN, or solve overrun.

### 5.3 Test matrix (run through the existing RoSE Spike flow)
Axes: **noise level** {clean, L1, L2} × **scenario** {baseline hover, hard IC, wind,
gust impulse, setpoint step, mass/thrust mismatch, combined} × **seed** {5 seeds for
stochastic cells}. Run each cell through the same flow used by the demo
(`docs/ROSE_DRONE_MPC_DEMO.md`): build the guest, launch the sync
(`deploy/hephaestus/run_sync_only.py`) and the Spike lockstep harness
(`soc/sim/run_spike_rose_lockstep.sh`), with `ROSE_TRAJ_CSV` set per cell. Drive the
matrix from a sweep script modeled on `deploy/scripts/rose-velocity-sweep.sh` — a loop
over env-var overrides (`ROSE_SENSOR_NOISE_*`, `ROSE_SCENARIO_*`, seed) writing a
per-cell CSV, then a small Python scorer that emits the metrics table and PASS/DEGRADED/
FAIL. Cheap tier first: validate every cell in a **pure-Python env smoke run** (env +
sync, no SoC) before spending Spike cycles.

---

## 6. Phasing (ordered rollout, cheap validation at each step)

Each phase gates on a cheap check before moving on. Keep `sensor_noise.enabled=false`
and `scenario` empty as the default so `main`/the demo stay green throughout.

- **Phase 0 — Harness & scoring (no dynamics change).** Add `ROSE_TRAJ_CSV` to the
  sensor-env run, write the offline scorer (§5.1) and the sweep driver skeleton (§5.3).
  *Validate:* re-score the current clean hover → confirms it PASSES level-1 thresholds
  and gives the baseline numbers. No env/guest code change yet.
- **Phase 1 — Noise (§1).** Add the corruption layer + `sensor_noise` plumbing + RNG
  reseed in `reset()`. Keep clean values in `info`. *Validate:* env-only run at L1;
  confirm distributions/magnitudes with `ROSE_SENSOR_DEBUG` (`:144-151`); then one Spike
  run to confirm the loop still hovers (likely DEGRADED at L2 — expected, motivates §4).
- **Phase 2 — Delay (§2).** Add per-modality ring buffers. *Validate:* env-only step
  response of emitted vs true sample confirms the exact delay; Spike run at
  `d=0` must equal Phase 1 (regression), then increase `d` and sweep the guest's
  `delay_steps`/`lead` to find the stable range.
- **Phase 3 — Estimator hardening (§4 items 1–3).** Add bias states, flow-validity, and
  innovation gating behind `IStateEstimator` (EKF first; complementary optional).
  *Validate:* re-run Phases 1–2 at L2 — hardened EKF should recover PASS/DEGRADED where
  the baseline FAILed; A/B against the complementary fallback (`-DROSE_USE_EKF=0`).
- **Phase 4 — Disturbances & harder IC (§3.1–3.2).** Add IC randomization + external
  wrench. *Validate:* single-shot wind/gust/IC runs; check recovery time and attitude
  excursion; confirm the added wrench does not perturb the baseline when magnitude=0.
- **Phase 5 — Setpoint steps & model mismatch (§3.3–3.4).** Scripted guest setpoints +
  env thrust/mass scale. *Validate:* z-step tracking (rise time, overshoot, steady
  offset — the offset exposes the no-integrator limitation → §4 item 6).
- **Phase 6 — Combined matrix (§5.3).** Full noise×scenario×seed sweep through Spike;
  produce the metrics table + pass/fail summary as the deliverable stress-test result.

---

## Appendix — key file references

- Env (sensor synth, add noise/delay here): `deploy/hephaestus/envs/isaac_crazyflie/crazyflie_sensor_env.py`
- Env base (reset/wrench/traj CSV, add IC/disturbance/mass here): `deploy/hephaestus/envs/isaac_crazyflie/crazyflie_mpc_env.py`
- Loop config: `deploy/config/config_gym_IsaacCrazyflieSensorEnv-v0.yaml`, `deploy/config/config_deploy_gym.yaml`
- Synchronizer: `deploy/hephaestus/gym_synchronizer.py`
- Guest: `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_flight_controller/src/{main,estimator*,attitude_mahony}.{cpp,hpp}`
- Run flow: `docs/ROSE_DRONE_MPC_DEMO.md`, `deploy/hephaestus/run_sync_only.py`, `soc/sim/run_spike_rose_lockstep.sh`, sweep model `deploy/scripts/rose-velocity-sweep.sh`
