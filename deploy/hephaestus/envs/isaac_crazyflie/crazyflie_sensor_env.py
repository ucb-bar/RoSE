"""RoSE IsaacLab Crazyflie env that serves REAL SENSOR MODALITIES (not ground truth).

This is the environment half of the "prep a real flight controller" flow. Where
`IsaacCrazyflieMPCEnv` hands the SoC the full ground-truth 12-DoF state (which a real
vehicle never has), this env exposes only what onboard sensors would measure:

    imu  = [ax, ay, az, gx, gy, gz]   accelerometer specific force + rate gyro (BODY frame)
    flow = [vx, vy]                    optical-flow horizontal velocity (BODY frame)
    tof  = [h]                         downward ToF height above ground (LOW-RATE)

The optical-flow modality models a real Crazyflie **Flow deck v2** (PMW3901 optical flow
+ VL53L1x downward ToF), which reports horizontal velocity AND height-above-ground in one
packet. The ToF height makes altitude *locally* observable (so the hover holds); there is
still no absolute x/y position or heading reference, so those drift over time — expected.
(IMU + flow-velocity ALONE leaves altitude unobservable: the vehicle hovers, but at a
frozen-in altitude offset from the climb transient. The ToF fixes that.)

The SoC (`samples/rose_flight_controller`) must run a **state estimator** on these to
reconstruct the state TinyMPC needs. There is no absolute position/attitude reference in
this set (no GPS/mocap/magnetometer), so estimated pose drifts over time — expected.

Observations are structured per modality (a dict of named sensors), and the RoSE routing
serves each on its own reqrsp channel (see config_gym_IsaacCrazyflieSensorEnv-v0.yaml):
imu -> cmd 0x12 (ch2), flow -> cmd 0x13 (ch1), tof -> cmd 0x14 (ch1). The action is 4 normalized
per-rotor thrusts (cmd 0x20), so the same Crazyflie physics and camera/logging apply.

Sensor synthesis from Isaac ground truth (per control step, dt = 1/ctrl_freq):
  - accelerometer measures specific force f = R^T (a_world - g),  g = (0,0,-9.81);
    a_world is obtained by finite-differencing the world linear velocity. At hover
    a_world=0 so f -> (0,0,+9.81) in a level body, as a real accel reads.
  - gyro measures body angular velocity (root_ang_vel_b).
  - optical flow measures body-frame horizontal velocity (root_lin_vel_b[:2]).
Sensors are modeled clean (no bias/noise) for a first bring-up; add noise later to study
estimator robustness. Full ground truth is still returned in `info` for logging/plots.
"""

import os

import numpy as np
from gymnasium import spaces

from .crazyflie_mpc_env import IsaacCrazyflieMPCEnv

_G_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)

# ---- Sensor corruption models (stress plan section 1) --------------------------------
# Per-modality, per-axis magnitudes for a Crazyflie-class stack (BMI088 IMU, PMW3901 flow,
# VL53L1x ToF) at the 200 Hz control rate. Level 0 = clean (identity: preserves the
# validated hover). Level 1 = nominal, Level 2 = aggressive. Selected by
# ROSE_SENSOR_NOISE_LEVEL; runs are reproducible via ROSE_SENSOR_NOISE_SEED.
_NOISE_LEVELS = {
    0: None,  # clean
    1: dict(
        accel=dict(white=0.15, bias_rw=0.01, bias0=0.10, scale=0.005),
        gyro=dict(white=0.02, bias_rw=0.002, bias0=0.01, scale=0.0),
        flow=dict(white=0.01, dropout=0.005, outlier=0.0, outlier_mag=0.5),
        tof=dict(white=0.008, quant=0.005, dropout=0.0),
    ),
    2: dict(
        accel=dict(white=0.40, bias_rw=0.03, bias0=0.30, scale=0.015),
        gyro=dict(white=0.05, bias_rw=0.006, bias0=0.03, scale=0.0),
        flow=dict(white=0.03, dropout=0.03, outlier=0.002, outlier_mag=0.5),
        tof=dict(white=0.02, quant=0.01, dropout=0.02),
    ),
}


class SensorCorruptor:
    """Applies realistic imperfections to the clean synthesized sensors (stress plan section 1).

    White Gaussian noise + a slow bias random-walk (b += N(0,bias_rw)*sqrt(dt)) + a fixed
    per-run scale-factor error on the inertial channels; optical flow noise scales as 1/height
    (SNR falls as the ground recedes) with Bernoulli dropout + occasional outliers; ToF adds
    Gaussian noise, quantization, and dropout. All randomness comes from one seeded RNG, reset
    per episode, so a run is bit-for-bit reproducible and a seed sweep gives independent trials.
    Returns corrupted values plus per-modality validity flags (the hook the guest uses in the
    plan's estimator-hardening phase).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = None
        self._init_states(np.random.default_rng(0))

    def _init_states(self, rng):
        self.rng = rng
        c = self.cfg
        self.accel_bias = rng.uniform(-c["accel"]["bias0"], c["accel"]["bias0"], 3)
        self.gyro_bias = rng.uniform(-c["gyro"]["bias0"], c["gyro"]["bias0"], 3)
        self.accel_scale = 1.0 + rng.uniform(-c["accel"]["scale"], c["accel"]["scale"], 3)
        self.gyro_scale = 1.0 + rng.uniform(-c["gyro"]["scale"], c["gyro"]["scale"], 3)
        self._last_flow = None
        self._last_tof = None

    def reset(self, seed):
        self._init_states(np.random.default_rng(seed))

    def apply(self, accel, gyro, flow, tof_h, height, dt):
        c, rng = self.cfg, self.rng
        sdt = np.sqrt(max(dt, 1e-9))
        # inertial: scale-factor * true + slow bias walk + white noise
        self.accel_bias += rng.normal(0.0, c["accel"]["bias_rw"], 3) * sdt
        self.gyro_bias += rng.normal(0.0, c["gyro"]["bias_rw"], 3) * sdt
        accel_c = (self.accel_scale * accel + self.accel_bias
                   + rng.normal(0.0, c["accel"]["white"], 3))
        gyro_c = (self.gyro_scale * gyro + self.gyro_bias
                  + rng.normal(0.0, c["gyro"]["white"], 3))

        # optical flow: white noise * (1/height clamp), dropout (hold last), rare outlier
        hclamp = float(np.clip(1.0 / max(height, 1e-3), 0.3, 3.0))
        flow_valid = True
        if self._last_flow is not None and rng.random() < c["flow"]["dropout"]:
            flow_c = self._last_flow.copy()
            flow_valid = False
        else:
            flow_c = flow + rng.normal(0.0, c["flow"]["white"] * hclamp, 2)
            if c["flow"]["outlier"] > 0 and rng.random() < c["flow"]["outlier"]:
                flow_c = flow_c + rng.uniform(-1, 1, 2) * c["flow"]["outlier_mag"]
            self._last_flow = flow_c.copy()

        # ToF: gaussian + quantization + dropout (hold last)
        tof_valid = True
        if self._last_tof is not None and rng.random() < c["tof"]["dropout"]:
            tof_c = self._last_tof
            tof_valid = False
        else:
            tof_c = tof_h + rng.normal(0.0, c["tof"]["white"])
            q = c["tof"]["quant"]
            if q > 0:
                tof_c = round(tof_c / q) * q
            self._last_tof = tof_c
        return (accel_c, gyro_c, flow_c, np.array([tof_c], dtype=np.float64),
                flow_valid, tof_valid)


class SensorDelay:
    """Per-modality transport/sampling latency (stress plan section 2).

    Each modality's freshly-corrupted sample is pushed into its own ring buffer and the sample
    d steps old is emitted (deque(maxlen=d+1) -> emit buf[0]). Delays are integer control steps
    (200 Hz): IMU is fast (0-1), optical flow ~1-2 (PMW3901 integration+SPI), ToF 1-3 on top of
    its hold cadence. This STACKS against the guest's forward delay-compensation, so the loop's
    total compensation horizon must cover actuation + sensor delay -- exactly the coupling the
    stress matrix probes. All delays default 0 (no latency = the validated behavior).
    """

    def __init__(self, delays):
        self.delays = delays            # {modality: int steps}
        self.buf = {}

    def reset(self):
        self.buf = {}

    def push(self, name, val):
        d = self.delays.get(name, 0)
        if d <= 0:
            return val
        import collections
        q = self.buf.get(name)
        if q is None or q.maxlen != d + 1:
            q = collections.deque(maxlen=d + 1)
            self.buf[name] = q
        q.append(np.array(val, dtype=np.float64))
        return q[0].copy()              # oldest available (== d steps old once warmed up)


def _quat_to_matrix(qw, qx, qy, qz):
    """(w,x,y,z) unit quaternion -> body->world rotation matrix (v_world = R @ v_body)."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


class IsaacCrazyflieSensorEnv(IsaacCrazyflieMPCEnv):
    """IsaacLab Crazyflie serving IMU + optical-flow sensors instead of ground-truth state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctrl_dt = 1.0 / self.ctrl_freq
        self._prev_vel_w = None
        # base ("body") link index for Isaac's instantaneous body acceleration
        self._base_id = self.robot.find_bodies("body")[0][0]
        # accelerometer source: "isaac" (instantaneous body_lin_acc_w, low-lag) or
        # "finitediff" (dv/dt, 1-step lag). Isaac's is smoother -> better estimator vz.
        # Isaac's body_lin_acc_w matches dv/dt in flight (it is computed by finite-diff
        # internally), so "finitediff" is the verified default; "isaac" is available too.
        self._accel_src = os.environ.get("ROSE_ACCEL_SRC", "finitediff")
        # Low-rate ToF: a real downward rangefinder (e.g. VL53L1x) samples every ~20-40 ms,
        # far slower than the IMU/flow. Refresh the reported height only every _tof_period
        # control steps (at 200 Hz, 6 -> ~30 ms); hold it in between.
        self._tof_period = int(os.environ.get("ROSE_TOF_PERIOD", "6"))
        self._tof_held = None
        self._tof_ctr = 0
        # Sensor-corruption layer (stress plan section 1). Level 0 (default) is clean and
        # identical to the validated hover; 1/2 add nominal/aggressive noise+bias. Seed makes
        # each run reproducible; sweep the seed for independent stochastic trials.
        self._noise_level = int(os.environ.get("ROSE_SENSOR_NOISE_LEVEL", "0"))
        self._noise_seed = int(os.environ.get("ROSE_SENSOR_NOISE_SEED", "0"))
        cfg = _NOISE_LEVELS.get(self._noise_level)
        self._corruptor = SensorCorruptor(cfg) if cfg is not None else None
        if self._corruptor is not None:
            self._corruptor.reset(self._noise_seed)
        # Sensor transport delay (stress plan section 2), integer control steps per modality.
        delays = {m: int(os.environ.get("ROSE_SENSOR_DELAY_" + m.upper(), "0"))
                  for m in ("accel", "gyro", "flow", "tof")}
        self._delay = SensorDelay(delays) if any(delays.values()) else None
        f32 = np.float32
        big = np.finfo(f32).max
        # Structured per-modality observation (each modality is one reqrsp packet).
        self.observation_space = spaces.Dict({
            # Separate accel + gyro packets, matching the real BMI088's two I2C devices
            # (the guest issues them as independent, pipelined reqrsp requests).
            "accel": spaces.Box(-big, big, (3,), f32),  # [ax,ay,az] body specific force
            "gyro":  spaces.Box(-big, big, (3,), f32),  # [gx,gy,gz] body rate
            "flow":  spaces.Box(-big, big, (2,), f32),  # [vx,vy] body-frame optical flow
            "tof":   spaces.Box(-big, big, (1,), f32),  # [h] downward ToF height (low-rate)
            # ground-truth pose kept for logging (never routed onto the wire)
            "pos":  spaces.Box(-big, big, (3,), f32),
            "quat": spaces.Box(-1.0, 1.0, (4,), f32),
        })

    def _obs_info(self, forces_z, action_norm):
        d = self.robot.data
        pos = d.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        quat = d.root_quat_w[0].detach().cpu().numpy().astype(np.float64)   # (w,x,y,z)
        vel_w = d.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
        angv_w = d.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float64)
        vel_b = d.root_lin_vel_b[0].detach().cpu().numpy().astype(np.float64)
        angv_b = d.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float64)

        R = _quat_to_matrix(quat[0], quat[1], quat[2], quat[3])

        # world linear acceleration. Two sources (a real accelerometer measures
        # instantaneous specific force; finite-differencing velocity adds a 1-step lag +
        # noise amplification that degrades the estimator's vz and the loop's margin):
        a_fd = (np.zeros(3) if self._prev_vel_w is None
                else (vel_w - self._prev_vel_w) / self._ctrl_dt)
        self._prev_vel_w = vel_w.copy()
        a_isaac = d.body_lin_acc_w[0, self._base_id].detach().cpu().numpy().astype(np.float64)
        # Isaac reports ~0 on the pre-force reset frame; fall back to finite-diff there.
        if self._accel_src == "isaac" and np.linalg.norm(a_isaac) > 1e-6:
            a_world = a_isaac
        else:
            a_world = a_fd

        # accelerometer specific force in body frame: f = R^T (a_world - g)
        accel_body = R.T @ (a_world - _G_WORLD)
        gyro_body = angv_b                        # rate gyro (body)
        # Separate sensors, matching real hardware: optical flow (fast, body-frame
        # horizontal velocity) and a downward ToF rangefinder (low-rate: the reported
        # height is held between samples to model the real ~20-40 ms cadence).
        flow = np.array([vel_b[0], vel_b[1]], dtype=np.float64)
        if self._tof_held is None or (self._tof_ctr % self._tof_period) == 0:
            self._tof_held = float(pos[2])
        self._tof_ctr += 1
        tof = np.array([self._tof_held], dtype=np.float64)

        # keep the CLEAN values (post-synthesis) for offline estimator-error scoring
        clean_accel, clean_gyro = accel_body.copy(), gyro_body.copy()
        clean_flow, clean_tof = flow.copy(), tof.copy()
        flow_valid = tof_valid = True
        if self._corruptor is not None:
            accel_body, gyro_body, flow, tof, flow_valid, tof_valid = self._corruptor.apply(
                accel_body, gyro_body, flow, self._tof_held, float(pos[2]), self._ctrl_dt)

        if self._delay is not None:
            accel_body = self._delay.push("accel", accel_body)
            gyro_body = self._delay.push("gyro", gyro_body)
            flow = self._delay.push("flow", flow)
            tof = self._delay.push("tof", tof)

        obs = {
            "accel": accel_body.astype(np.float32),
            "gyro": gyro_body.astype(np.float32),
            "flow": flow.astype(np.float32),
            "tof": tof.astype(np.float32),
            "pos": pos.astype(np.float32),
            "quat": np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32),
        }
        # Ground truth for logging/plots — includes the true 12-DoF state so the estimator
        # error can be scored offline.
        r = quat[1:4] / (quat[0] if abs(quat[0]) > 1e-9 else 1e-9)
        gt_state = np.concatenate([pos, r, vel_w, angv_w]).astype(np.float32)
        info = {
            "gt_pos": pos.astype(np.float32),
            "gt_quat": quat.astype(np.float32),
            "gt_vel": vel_w.astype(np.float32),
            "gt_angvel": angv_w.astype(np.float32),
            "gt_state": gt_state,
            "accel": accel_body.astype(np.float32),
            "gyro": gyro_body.astype(np.float32),
            "flow": flow.astype(np.float32),
            "tof": tof.astype(np.float32),
            # clean (pre-corruption) sensors + validity flags for offline scoring / the
            # guest-side validity hook (stress plan sections 1.2 & 4).
            "gt_accel": clean_accel.astype(np.float32),
            "gt_gyro": clean_gyro.astype(np.float32),
            "gt_flow": clean_flow.astype(np.float32),
            "gt_tof": clean_tof.astype(np.float32),
            "flow_valid": bool(flow_valid),
            "tof_valid": bool(tof_valid),
            "noise_level": np.int32(self._noise_level),
            "rotor_force_N": np.asarray(forces_z, dtype=np.float32),
            "action_norm": np.asarray(action_norm, dtype=np.float32),
            "target": self.target.astype(np.float32),
            "sim_time": np.float32(self._t),
        }
        if os.environ.get("ROSE_SENSOR_DEBUG"):
            self._dbg = getattr(self, "_dbg", 0) + 1
            if self._dbg % 25 == 1:
                f_fd = R.T @ (a_fd - _G_WORLD)
                f_bl = R.T @ (a_isaac - _G_WORLD)
                print(f"[sensor] t={self._t:5.2f} accel_z used={accel_body[2]:+.2f} "
                      f"(fd={f_fd[2]:+.2f} isaac={f_bl[2]:+.2f}) "
                      f"flow=({flow[0]:+.3f},{flow[1]:+.3f}) z_true={pos[2]:.3f}", flush=True)
        return obs, info, (pos, quat, vel_w, angv_w)

    def reset(self, *, seed=None, options=None):
        self._prev_vel_w = None
        self._tof_held = None
        self._tof_ctr = 0
        if self._corruptor is not None:
            # re-seed per episode so each run is bit-for-bit reproducible (seed sweep -> IID trials)
            self._corruptor.reset(self._noise_seed)
        if self._delay is not None:
            self._delay.reset()
        return super().reset(seed=seed, options=options)

