"""CrazyflieCamProbeEnv — a GPU-free stand-in for validating the HM01B0 FPV camera DMA path.

The real FPV camera lives on IsaacCrazyflieMultiSensorEnv, which needs Isaac Sim (GPU). This
env produces the SAME analytic scene-derived frames (`synth_fpv_analytic`) from a scripted
drone pose flying down the hallway, with no isaaclab dependency — so the full camera path
(env frame -> synchronizer DMA serve -> Spike bridge DMA engine -> SoC guest) can be validated
deterministically and headlessly, exactly the way PatternEnv validates the reqrsp/DMA plumbing.

The frame it serves is legitimate camera sensing data (a pinhole projection of the corridor
walls from the drone's pose, with parallax as it moves) — identical synthesis to the Isaac
env's headless FPV path; on a GPU the multisensor env serves the real rendered frame instead.
"""

import os
import zlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.isaac_crazyflie.crazyflie_multisensor_env import (
    synth_fpv_analytic, _pixel_ray_dirs, _maze_walls, FPV_W, FPV_H, FPV_FOV_DEG,
)


def frame_checksum(frame_u8):
    """CRC32 (IEEE 802.3) of a uint8 frame — matches the synchronizer's zlib.crc32 and the
    guest's Zephyr crc32_ieee(). Matching host+guest checksums prove the exact frame landed."""
    return zlib.crc32(np.asarray(frame_u8, dtype=np.uint8).reshape(-1).tobytes()) & 0xffffffff


class CrazyflieCamProbeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        self._w = int(os.environ.get("ROSE_FPV_W", str(FPV_W)))
        self._h = int(os.environ.get("ROSE_FPV_H", str(FPV_H)))
        if (self._w * self._h) % 4 != 0:
            raise ValueError("FPV W*H must be a multiple of 4 (DMA word packing); got %dx%d"
                             % (self._w, self._h))
        self._fov = float(os.environ.get("ROSE_FPV_FOV", str(FPV_FOV_DEG)))
        self._dirs = _pixel_ray_dirs(self._w, self._h, self._fov)
        # RGB (HM01B0-ANA-00FT870 is a color sensor) -> flat H*W*3; else 8-bit GREY H*W.
        self._rgb = int(os.environ.get("ROSE_FPV_RGB", "0")) != 0
        # STATIC freezes the served frame at the reset pose (deterministic single-frame oracle).
        self._static = int(os.environ.get("ROSE_CAMPROBE_STATIC", "0")) != 0
        if self._rgb and (self._h * self._w * 3) % 4 != 0:
            raise ValueError("FPV RGB H*W*3 must be a multiple of 4 (DMA word packing)")

        # Same room + hallway model as the multisensor env (ROSE_MAZE, ROSE_HALL_X0/X1).
        rx = float(os.environ.get("ROSE_TOF_ROOM_X", "8.0"))
        ry = float(os.environ.get("ROSE_TOF_ROOM_Y", "2.0"))
        rz = float(os.environ.get("ROSE_TOF_ROOM_Z", "2.5"))
        self._room_min = np.array((-rx, -ry, 0.0), dtype=np.float64)
        self._room_max = np.array((rx, ry, rz), dtype=np.float64)
        self._walls = [(np.asarray(a, np.float64), np.asarray(b, np.float64))
                       for a, b in _maze_walls(os.environ.get("ROSE_MAZE", "hallway"))]

        # Scripted forward flight down the corridor (+x), level, at a fixed height.
        self._dt = float(os.environ.get("ROSE_CAMPROBE_DT", "0.02"))
        self._vx = float(os.environ.get("ROSE_CAMPROBE_VX", "0.5"))
        self._x0 = float(os.environ.get("ROSE_CAMPROBE_X0", "-2.0"))
        self._z = float(os.environ.get("ROSE_CAMPROBE_Z", "0.9"))
        self._t = 0

        fpv_len = self._h * self._w * (3 if self._rgb else 1)
        self.observation_space = spaces.Dict({
            "fpv": spaces.Box(0, 255, (fpv_len,), np.uint8),
            "pose": spaces.Box(-np.inf, np.inf, (7,), np.float32),
        })
        # action ignored (scripted flight); present so the synchronizer can latch one.
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        print("[CamProbe] %dx%d fov=%.0f maze=%s vx=%.2f x0=%.2f z=%.2f"
              % (self._w, self._h, self._fov, os.environ.get("ROSE_MAZE", "hallway"),
                 self._vx, self._x0, self._z), flush=True)

    def _pose(self):
        t = 0 if self._static else self._t
        x = self._x0 + self._vx * t * self._dt
        return (np.array([x, 0.0, self._z], np.float64),
                np.array([1.0, 0.0, 0.0, 0.0], np.float64))   # level, facing +x

    @staticmethod
    def colorize_rgb(gray):
        """Deterministic gray (H,W) -> RGB888 (H,W,3): a depth FPV given color so a real
        (RGB) HM01B0-ANA feed is exercised. R=depth, G=dimmed depth, B=inverse depth. Purely
        deterministic (integer), so the host oracle reproduces it exactly. Not a substitute
        for the real Isaac RGB render (that's the GPU path); this is the headless stand-in."""
        g = gray.astype(np.uint16)
        r = gray
        gg = ((g * 180) // 255).astype(np.uint8)
        b = (255 - g).astype(np.uint8)
        return np.stack([r, gg, b], axis=-1).astype(np.uint8)   # (H,W,3)

    def _obs(self):
        pos, quat = self._pose()
        gray = synth_fpv_analytic(pos, quat, self._room_min, self._room_max,
                                  self._walls, self._dirs, self._w, self._h)
        frame = self.colorize_rgb(gray) if self._rgb else gray
        obs = {"fpv": frame.reshape(-1),
               "pose": np.concatenate([pos, quat]).astype(np.float32)}
        info = {"fpv_checksum": frame_checksum(frame), "fpv_x": float(pos[0])}
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs()

    def step(self, action):
        self._t += 1
        obs, info = self._obs()
        return obs, 0.0, False, False, info

    def render(self):
        pos, quat = self._pose()
        return synth_fpv_analytic(pos, quat, self._room_min, self._room_max,
                                  self._walls, self._dirs, self._w, self._h)
