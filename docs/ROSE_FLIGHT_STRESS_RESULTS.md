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
Default 0 = no latency. Matrix `delay` sweeps d ∈ {0,1,2,3} (at noise L1).

**Result (10 s, steady t ≥ 2.5 s):** delay0 alt_rms 0.0646 ≈ Phase-1 L1 (0.066) — the d=0
**regression holds**. delay1/2/3 add only mild ripple (0.020→0.026 m) and tilt
(0.23→0.26°); all four FAIL, but purely on the delay-*independent* L1 accel-bias altitude
offset (~0.065 m). Takeaway: **1–3-step sensor delay is well tolerated** here (the guest's
forward delay-compensation absorbs it at 200 Hz); the altitude offset dominates → Phase 3.

## Harder initial conditions (plan Phase 3.1) — DONE (env), validation pending

`crazyflie_mpc_env.py` `reset()` applies seeded IC offsets (`ROSE_IC_POS`, `ROSE_IC_Z`,
`ROSE_IC_TILT_DEG`, `ROSE_IC_VEL`, `ROSE_IC_VZ`, `ROSE_IC_RATE`; seed `ROSE_SCENARIO_SEED`).
No-op unless set. Matrix `scenario` sweeps tilt / offset / velocity starts.

**Result (10 s, clean sensors):** the loop **recovers from hard ICs** — attitude, velocity,
and altitude all return to nominal:
- `ic_tilt` (~8° roll/pitch start) → levels to 0.0°, velocity damps to ~0.01 m/s, z→1.025;
  drifts ~0.55 m horizontally *during* the recovery, which is then permanent.
- `ic_offset` (z=0.76, lateral offset start) → climbs back to z≈1.02 via ToF, velocity 0,
  the initial horizontal offset stays frozen (does not grow).

Confirms the documented limitation: attitude/velocity/altitude are observable and recovered;
**horizontal position excursions are permanent** (no GPS/mocap/position reference).

**Scored (10 s, steady t ≥ 3.5 s):**

| cell | verdict | alt_rms_m | ripple_m | vel_rms_ms | max_tilt_deg |
|---|---|---|---|---|---|
| ic_none | PASS | 0.023 | 0.000 | 0.000 | 0.0 |
| ic_offset ×2 | PASS | 0.024 | 0.002 | 0.000 | 0.0 |
| ic_tilt ×2 | PASS | 0.025–0.028 | 0.006–0.016 | 0.017–0.030 | 0.15–0.26 |
| ic_velocity s0 | **FAIL** | 0.095 | 0.204 | 0.607 | **48.3** |
| ic_velocity s1 | FAIL | 0.072 | 0.092 | 0.049 | 1.39 |

Robust to position offset and ≤15° tilt starts; a hard initial **velocity + body-rate** combo
(±0.3 m/s + ±0.5 rad/s) can drive a large tilt excursion (48° on seed 0 — a near-flip) — a
genuine controller robustness limit worth revisiting alongside Phase 3.

## Sim throughput — bottleneck & speedups

Profiling showed the **GPU idle (1% util, 7/24 GB)** — the 1-drone scene is trivial. Each
`rose_spike_sim` sits at 100% of one core, but a **controlled A/B disproved the spike-cycle
theory**: cell at 5M cycles @1 GHz = **1.97 steps/s** vs 1.5M @300 MHz = **2.10 steps/s** —
only ~7% faster for 3.3× fewer SoC cycles. So spike's 100% is a **busy-wait spin** (the guest
blocks in `rose_rx`), **not** the bottleneck. The real long pole is **Isaac Sim's per-step
CPU cost** (~0.5 s/step, single drone, on the CPU — GPU idle).

Implications for speeding up experimentation:
- **`--jobs` (parallelism) is the real win** — now defaults to 8. Each cell is ~2 steps/s
  regardless of clock and uses ~1–2 CPU cores (Isaac + the spike spin), so on a 48-core host
  ~10–14 cells run concurrently → ~5–7× matrix throughput (per-cell rate unchanged). Leave
  headroom on a shared box.
- **SoC clock (`--firesim-freq/--firesim-step`) does NOT speed things up** (~7%, A/B above);
  kept only as an optional co-design knob, default = inherit config (1 GHz). `ROSE_FIRESIM_*`
  overrides in `gym_synchronizer.load_config` implement it.
- **Isaac per-step CPU** is where further single-cell speedup would come from (fewer physics
  substeps, no rendering, leaner scene/extensions) — not pursued here.
- Fixed a harness bug where a completed cell **orphaned its `rose_spike_sim`** (100%-CPU
  spin, `ppid=1`) because cleanup waited on the bash wrapper; `_kill_group` now always
  SIGKILLs the whole process group. This had been compounding the slowdown across runs.

## External disturbances (plan Phase 3.2) — DONE

Implemented in `crazyflie_mpc_env.py` (`_disturbance_cfg` / `_apply_disturbance`): a body-frame
external wrench on the BASE body via `permanent_wrench_composer.set_forces_and_torques(...,
body_ids=[base])`. The composer's `set` is per-body, so setting the base body's wrench leaves
the rotor forces on the prop bodies intact (verified against the WrenchComposer source).
Configured by env vars (all default off): steady wind (`ROSE_WIND_N`/`_DIR_DEG`, world-frame,
rotated into the body), a timed gust (`ROSE_GUST_N`/`_START`/`_DUR`), and a timed yaw-torque
impulse (`ROSE_TORQUE_IMP`/`_START`/`_DUR`). Harness matrix `disturbance` sweeps them.

**Magnitude-0 gate (by construction):** with no disturbance env var set, `_disturbance_cfg`
returns `None`, so `_apply_disturbance` returns before touching the composer — the baseline
physics path is bit-for-bit unchanged.

**Validated:** under a 0.03 N +x wind the drone drifts +x (velocity-regulated controller →
steady drift), tilts ~6° into the wind to resist, holds altitude (z=1.02), and stays stable.

## Estimator hardening (plan §4) — DONE (item 1: accel-bias state), validated

Item 1 implemented: the per-axis translational KF in the EKF grew from `Kf2` [pos, vel] to
`Kf3` [pos, vel, **accel_bias**] (`estimator_ekf.hpp`). A constant IMU bias — the root cause
of the altitude offset — is now estimated (observable via ToF position for z, optical flow
for x/y) and subtracted in predict; the delay-comp lead uses the bias-corrected accel too.
Built with `-DROSE_USE_EKF=1` (`rose_flight_controller_ekf`).

**A/B result (noise matrix, 12 s, steady t ≥ 2.5 s) — EKF+bias vs the complementary baseline:**

| cell | complementary (baseline) | EKF + accel-bias | outcome |
|---|---|---|---|
| L0 clean | PASS, alt_rms 0.023 | PASS, alt_rms 0.023 | no regression |
| L1 nominal | **FAIL**, alt_rms 0.066 | **PASS**, alt_rms 0.024, ripple 0.021, vel 0.047 | **recovered FAIL→PASS** |
| L2 aggressive | **FAIL**, alt_rms 0.116 | FAIL, alt_rms **0.026**, ripple 0.057, vel 0.201 | altitude offset eliminated; residual = flow noise |

The accel-bias state **kills the altitude offset** (the dominant failure): L1 recovers fully
to PASS, and L2's alt_rms drops 0.116→0.026. L2 still FAILs, but now purely on flow-noise
**velocity/ripple** (vel 0.201, ripple 0.057).

### Items 3 & 4 — innovation gating + noise-matched `r_flow` (done)

Added a χ² (normalized-innovation) outlier gate to the `Kf3` position/velocity updates
(reject when `resid²/S` > gate; flow gate 9, ToF 25) and raised `r_flow` 4e-4→9e-4 to match
the aggressive flow std (~0.03 m/s).

| cell | item-1 (accel-bias) | + items 3&4 (gating + r_flow) |
|---|---|---|
| L0 | PASS, alt_rms 0.023 | PASS, alt_rms 0.023 (no regression) |
| L1 | PASS, ripple 0.021, vel 0.047 | PASS, ripple 0.021, vel 0.047 |
| L2 | FAIL, vel 0.201, ripple 0.057 | FAIL, **vel 0.201 (unchanged)**, ripple 0.053 |

Honest result: gating rejects the rare ±0.5 m/s flow spikes and slightly tightens ripple, but
**does not recover L2** — its velocity RMS is dominated by *broadband* white flow noise plus
the 3% flow **dropout** (the env holds the last value, and the guest cannot tell a sample is
stale without a `flow_valid` bit on the wire). ### Item 2 — flow-validity dropout handling (done)

Wire: on a flow dropout the env now sends a **NaN sentinel** in the existing flow packet (no
new cmd/channel); the guest driver path detects NaN → `flow_valid=false`, and both estimators
run the velocity step **predict-only** on dropout (mirrors the ToF path). Interface gained a
`flow_valid` arg (EKF + complementary).

Result (EKF, noise matrix): L0/L1 still PASS (no regression), **L2 vel 0.201 unchanged**. So
the 3% flow dropout was NOT L2's driver either — confirming L2's residual velocity is
*broadband* white noise (dominated by the aggressive accel σ=0.4 propagating through
predict + the delay-comp lead), not stale samples or outliers.

**Conclusion on L2:** item 1 (accel-bias) eliminated the altitude offset — the dominant
failure — and recovered L1 to PASS; items 2 & 3 are correct, non-regressing robustness
architecture (dropout handling + outlier rejection) that real hardware needs, but none
recover L2's broadband-noise velocity. Fully closing L2 would require **retuning the TinyMPC
gains or a better sensor suite** — deliberately NOT done, to avoid overtuning to the sim.
This is an honest capability boundary of the current sensor-based stack at the aggressive
noise level, surfaced by the harness. Gyro-bias in Mahony remains a further item (attitude
stayed <1°, low priority).

Harness note: fixed the `rose_spike_sim` orphan leak at the root (`timeout --foreground`);
confirmed 0 orphans after a completed run.
