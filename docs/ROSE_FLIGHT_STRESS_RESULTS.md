# RoSE Flight Controller Stress-Test — Results & Status

Execution log for the plan in [`ROSE_FLIGHT_STRESS_PLAN.md`](ROSE_FLIGHT_STRESS_PLAN.md).
Every result below comes from the real RoSE co-sim loop (Isaac Sim physics ⇄ Spike SoC
running the shared `rose_flight_controller` guest), scored offline from the ground-truth
trajectory CSV by `deploy/scripts/rose_stress_score.py`.

## Harness (plan Phase 0) — DONE

Parallel evaluation harness: `deploy/scripts/rose_stress_harness.py` runs many co-sim cells
concurrently, each an independent (synchronizer + `rose_spike_sim`) pair on its own sync
port (`ROSE_SYNC_PORT` → synchronizer bind + `rose_spike_sim --rose-port`). Cells are driven
to a fixed **sim time** (a target CSV row count = duration / control-dt), not wall-clock, so
every cell flies the same length regardless of host load. A cell is a name + a dict of env
overrides (`ROSE_SENSOR_NOISE_*`, `ROSE_SENSOR_DELAY_*`, `ROSE_IC_*`, seed); matrices are
defined in the harness (`baseline`, `noise`, `delay`, `scenario`, `combined`, …).

Scorer `rose_stress_score.py` computes the plan §5.1 metrics over the steady window and
classifies **PASS / DEGRADED / FAIL** against the §5.2 level-1 bounds (alt RMS < 0.03 m,
ripple < 0.05 m, vel RMS < 0.1 m/s, tilt < 10°; >2× a bound or crash/diverge/NaN/died-early
= FAIL; a genuinely short intended run is annotated, not failed).

**Validated:** 3 concurrent cells with distinct ports, slot/port reuse as cells complete,
fixed-sim-time termination, clean re-score of the baseline hover to PASS. On this
single-GPU host Isaac is the shared bottleneck, so concurrency mainly buys amortized boot,
fail-fast cells, and linear scaling on multi-GPU hosts (not a 3× wall-clock speedup here).

Usage:
```bash
deploy/scripts/rose_stress_harness.py --matrix noise --jobs 3 --duration 12 --out-dir <dir>
deploy/scripts/rose_stress_score.py <dir>
```

## Sensor noise / bias (plan Phase 1) — DONE (env)

`crazyflie_sensor_env.py` gained a `SensorCorruptor` applied after clean synthesis (clean
values kept in `info` as `gt_*` for offline estimator-error scoring). Models per the plan
§1.2 table: accel/gyro white noise + slow bias random-walk + fixed scale-factor error;
optical-flow white noise scaled by 1/height + Bernoulli dropout (+ outliers at L2); ToF
Gaussian + quantization + dropout. Selected by `ROSE_SENSOR_NOISE_LEVEL` ∈ {0,1,2} (0 =
clean, identical to the validated hover), reproducible via `ROSE_SENSOR_NOISE_SEED`
(re-seeded each `reset()`). Validity flags (`flow_valid`, `tof_valid`) are exposed in `info`
as the hook for the guest-side handling in Phase 3.

**Result (12 s, seed 0, steady window t ≥ 2.5 s):**

| cell | verdict | alt_rms_err_m | ripple_m | vel_rms_ms | tilt_deg | horiz_drift_ms | failure |
|---|---|---|---|---|---|---|---|
| L0 clean | **PASS** | 0.023 | 0.000 | 0.000 | 0.0 | 1e-5 | — |
| L1 nominal | **FAIL** | 0.066 | 0.024 | 0.032 | 0.27 | 0.017 | altitude offset only (>2× bound) |
| L2 aggressive | **FAIL** | 0.116 | 0.056 | 0.182 | 0.77 | 0.172 | offset + ripple + velocity |

L1 fails on **altitude alone** — ripple/velocity/tilt/drift all stay within the level-1
bounds; the accel bias produces a ~66 mm steady offset (2.2× the 30 mm bound) that the
no-integrator controller cannot reject. L2 additionally fails on ripple and velocity RMS
(flow noise) and drifts at 0.17 m/s. This scopes Phase 3 precisely: an **accel-bias state**
recovers L1; **flow innovation gating + noise-matched `r_flow`** are needed for L2.

Headline finding: an accelerometer **bias** turns into a **steady altitude offset** because
TinyMPC has no integrator, and **flow noise** produces growing **horizontal drift**. Clean →
PASS, nominal (L1) → DEGRADED (altitude offset), aggressive (L2) → FAIL (offset > 2× +
drift). This directly motivates Phase 3 (bias states + innovation gating), exactly as the
plan anticipated.

## Sensor delay (plan Phase 2) — DONE (env), validation pending

`SensorDelay`: per-modality ring buffer (`ROSE_SENSOR_DELAY_{ACCEL,GYRO,FLOW,TOF}` in
control steps), applied after corruption. Stacks against the guest's forward
delay-compensation, so the total compensation horizon must cover actuation + sensor delay.
Default 0 = no latency. Matrix `delay` sweeps d ∈ {0,1,2,3}. Validation run pending GPU.

## Harder initial conditions (plan Phase 3.1) — DONE (env), validation pending

`crazyflie_mpc_env.py` `reset()` applies seeded IC offsets (`ROSE_IC_POS`, `ROSE_IC_Z`,
`ROSE_IC_TILT_DEG`, `ROSE_IC_VEL`, `ROSE_IC_VZ`, `ROSE_IC_RATE`; seed `ROSE_SCENARIO_SEED`).
No-op unless set. Matrix `scenario` sweeps tilt / offset / velocity starts. Validation run
pending GPU.

## External disturbances (plan Phase 3.2) — DEFERRED (deliberately)

The rotor wrench goes through a custom `robot.permanent_wrench_composer` whose set-vs-
accumulate semantics I could not verify from source. Rather than risk the validated hover, a
disturbance wrench (steady wind + gust/torque impulse) is deferred until it can be added with
a **magnitude-0 regression gate** (confirm the baseline is bit-identical with the disturbance
path present but zero) on a free GPU. Design is in the plan §3.2.

## Estimator hardening (plan Phase 4/§4) — NEXT

Motivated directly by the Phase-1 result. Priority: (1) accel-bias state in the vertical KF
(kills the altitude offset — the dominant L1/L2 failure) + gyro-bias in the attitude filter;
(2) `flow_valid`/`tof_valid` dropout handling (predict-only on dropout); (3) innovation
gating (χ²) on ToF/flow for the L2 outliers. All behind `IStateEstimator`. Requires a guest
rebuild + re-run of the noise matrix to show recovery (PASS/DEGRADED where the baseline
FAILed), A/B EKF vs complementary.
