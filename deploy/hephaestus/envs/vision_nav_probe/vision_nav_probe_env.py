"""CrazyflieVisionNavProbeEnv — GPU-free co-sim env for the DroNet vision-nav controller (P3).

Serves the FULL flight-sensor set the rose_nav_controller consumes (IMU accel+gyro, optical
flow, downward ToF, and the 4 horizontal VL53L5CX multizone ToFs) AND the forward FPV RGB
camera, all headless (no isaaclab/GPU) from a fixed hover pose in a corridor. This lets the
DroNet-nav controller run end-to-end in the RoSE lockstep so we can observe DroNet's
(steer, collision) driving the TinyMPC setpoint (yaw / forward velocity) and a well-formed
0x20 control output — deterministically, without Isaac.

It is a *bench* stand-in (hover sensors, static scene), not a physics-closed flight — physical
flight is the real Isaac multisensor env (GPU). Here the point is to validate the on-SoC
camera->DroNet->setpoint->control integration deterministically. Camera synthesis reuses the
exact analytic FPV path (synth_fpv_analytic) the camera-DMA validation uses.
"""

import os
import zlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.isaac_crazyflie.crazyflie_multisensor_env import (
    synth_fpv_analytic, _pixel_ray_dirs, _maze_walls, FPV_W, FPV_H, FPV_FOV_DEG,
)
from envs.cam_probe.cam_probe_env import CrazyflieCamProbeEnv, frame_checksum

GRAVITY = 9.81
ZONES = 64


class CrazyflieVisionNavProbeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        self._w = int(os.environ.get("ROSE_FPV_W", str(FPV_W)))
        self._h = int(os.environ.get("ROSE_FPV_H", str(FPV_H)))
        if (self._h * self._w * 3) % 4 != 0:
            raise ValueError("FPV RGB H*W*3 must be a multiple of 4 (DMA word packing)")
        self._fov = float(os.environ.get("ROSE_FPV_FOV", str(FPV_FOV_DEG)))
        self._dirs = _pixel_ray_dirs(self._w, self._h, self._fov)
        self._z = float(os.environ.get("ROSE_CAMPROBE_Z", "0.9"))

        # Corridor model (matches CamProbe defaults) for the analytic FPV.
        rx = float(os.environ.get("ROSE_TOF_ROOM_X", "8.0"))
        ry = float(os.environ.get("ROSE_TOF_ROOM_Y", "2.0"))
        rz = float(os.environ.get("ROSE_TOF_ROOM_Z", "2.5"))
        self._room_min = np.array((-rx, -ry, 0.0), np.float64)
        self._room_max = np.array((rx, ry, rz), np.float64)
        self._walls = [(np.asarray(a, np.float64), np.asarray(b, np.float64))
                       for a, b in _maze_walls(os.environ.get("ROSE_MAZE", "hallway"))]

        # Symmetric corridor distances (m): front/back open, left/right walls -> x,y centered.
        self._d_front = float(os.environ.get("ROSE_PROBE_FRONT", "2.0"))
        self._d_back = float(os.environ.get("ROSE_PROBE_BACK", "2.0"))
        self._d_left = float(os.environ.get("ROSE_PROBE_LEFT", "1.0"))
        self._d_right = float(os.environ.get("ROSE_PROBE_RIGHT", "1.0"))

        self.observation_space = spaces.Dict({
            "accel": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "gyro":  spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "flow":  spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "tof":   spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "front": spaces.Box(0, np.inf, (ZONES,), np.float32),
            "right": spaces.Box(0, np.inf, (ZONES,), np.float32),
            "back":  spaces.Box(0, np.inf, (ZONES,), np.float32),
            "left":  spaces.Box(0, np.inf, (ZONES,), np.float32),
            "fpv":   spaces.Box(0, 255, (self._h * self._w * 3,), np.uint8),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        print("[VisionNavProbe] %dx%d RGB fov=%.0f hover z=%.2f corridor F/B/L/R=%.1f/%.1f/%.1f/%.1f"
              % (self._w, self._h, self._fov, self._z, self._d_front, self._d_back,
                 self._d_left, self._d_right), flush=True)

    def _fpv_rgb(self):
        pos = np.array([0.0, 0.0, self._z], np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], np.float64)   # level, facing +x
        gray = synth_fpv_analytic(pos, quat, self._room_min, self._room_max,
                                  self._walls, self._dirs, self._w, self._h)
        return CrazyflieCamProbeEnv.colorize_rgb(gray)

    def _obs(self):
        rgb = self._fpv_rgb()
        obs = {
            "accel": np.array([0.0, 0.0, GRAVITY], np.float32),   # level hover specific force
            "gyro":  np.zeros(3, np.float32),
            "flow":  np.zeros(2, np.float32),
            "tof":   np.array([self._z], np.float32),
            "front": np.full(ZONES, self._d_front, np.float32),
            "right": np.full(ZONES, self._d_right, np.float32),
            "back":  np.full(ZONES, self._d_back, np.float32),
            "left":  np.full(ZONES, self._d_left, np.float32),
            "fpv":   rgb.reshape(-1),
        }
        info = {"fpv_checksum": frame_checksum(rgb)}
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs()

    def step(self, action):
        obs, info = self._obs()
        return obs, 0.0, False, False, info
