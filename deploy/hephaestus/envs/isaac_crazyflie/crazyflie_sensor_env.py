"""RoSE IsaacLab Crazyflie env that serves REAL SENSOR MODALITIES (not ground truth).

This is the environment half of the "prep a real flight controller" flow. Where
`IsaacCrazyflieMPCEnv` hands the SoC the full ground-truth 12-DoF state (which a real
vehicle never has), this env exposes only what onboard sensors would measure:

    imu  = [ax, ay, az, gx, gy, gz]   accelerometer specific force + rate gyro (BODY frame)
    flow = [vx, vy, h]                 optical-flow horizontal velocity (BODY) + ToF height

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
imu -> cmd 0x12 (ch2), flow -> cmd 0x13 (ch1). The action is unchanged: 4 normalized
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
        self._prev_vel_w = None
        self._ctrl_dt = 1.0 / self.ctrl_freq
        f32 = np.float32
        big = np.finfo(f32).max
        # Structured per-modality observation (each modality is one reqrsp packet).
        self.observation_space = spaces.Dict({
            "imu":  spaces.Box(-big, big, (6,), f32),   # [ax,ay,az, gx,gy,gz] body frame
            "flow": spaces.Box(-big, big, (3,), f32),   # [vx,vy] body flow + h (ToF height)
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

        # world linear acceleration by finite difference of world velocity
        if self._prev_vel_w is None:
            a_world = np.zeros(3, dtype=np.float64)
        else:
            a_world = (vel_w - self._prev_vel_w) / self._ctrl_dt
        self._prev_vel_w = vel_w.copy()

        # accelerometer specific force in body frame: f = R^T (a_world - g)
        accel_body = R.T @ (a_world - _G_WORLD)
        gyro_body = angv_b                        # rate gyro (body)
        # flow deck: body-frame horizontal velocity + downward ToF height above ground
        # (modeled as vertical height; near hover the tilt correction is negligible).
        flow = np.array([vel_b[0], vel_b[1], pos[2]], dtype=np.float64)

        imu = np.concatenate([accel_body, gyro_body]).astype(np.float32)
        obs = {
            "imu": imu,
            "flow": flow.astype(np.float32),
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
            "imu": imu,
            "flow": flow.astype(np.float32),
            "rotor_force_N": np.asarray(forces_z, dtype=np.float32),
            "action_norm": np.asarray(action_norm, dtype=np.float32),
            "target": self.target.astype(np.float32),
            "sim_time": np.float32(self._t),
        }
        if os.environ.get("ROSE_SENSOR_DEBUG"):
            self._dbg = getattr(self, "_dbg", 0) + 1
            if self._dbg % 25 == 1:
                print(f"[sensor] t={self._t:5.2f} accel=({accel_body[0]:+.2f},{accel_body[1]:+.2f},"
                      f"{accel_body[2]:+.2f}) gyro=({gyro_body[0]:+.2f},{gyro_body[1]:+.2f},"
                      f"{gyro_body[2]:+.2f}) flow=({flow[0]:+.3f},{flow[1]:+.3f}) "
                      f"z_true={pos[2]:.3f}", flush=True)
        return obs, info, (pos, quat, vel_w, angv_w)

    def reset(self, *, seed=None, options=None):
        self._prev_vel_w = None
        return super().reset(seed=seed, options=options)
