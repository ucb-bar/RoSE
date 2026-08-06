"""FusedNavProbeEnv — GPU-free deterministic bench for the on-SoC fused-vision nav model (M1).

Serves ONE fixed, pre-quantized sensor frame (baked by scratchpad/prep_frame0.py from a real
warehouse sense() capture, quantized EXACTLY as the host reference int8_fused_model_f16.py):
  - front_q  int8[5400]  (0x11 DMA)   — pre-quantized 60x90 front_grey
  - tof_q    int8[256]   (0x41 reqrsp) — pre-quantized 4x8x8 cross ToF
  - lowdim   float32[21] (0x42 reqrsp) — assembled lowdim vector (guest casts each to fp16)

The guest runs run_model_fused_full on these BYTE-IDENTICAL inputs, so its (yaw_rate,
forward_speed) must equal the host .so run on the same bytes (m1_frame0/expected.txt) within
fp16 tolerance. This isolates the transport + zfh model execution on spike with NO Isaac/GPU.
Pair with config_gym_FusedNavProbeEnv-v0.yaml.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

FRAME_DIR = os.environ.get(
    "ROSE_M1_FRAME_DIR",
    "/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/d4827fc8-b516-4227-b71d-79192ba241cd/scratchpad/m1_frame0")


class FusedNavProbeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        self._front = np.fromfile(os.path.join(FRAME_DIR, "front_q.i8"), dtype=np.int8)
        self._tof = np.fromfile(os.path.join(FRAME_DIR, "tof_q.i8"), dtype=np.int8)
        self._lowdim = np.fromfile(os.path.join(FRAME_DIR, "lowdim.f32"), dtype=np.float32)
        assert self._front.shape == (5400,), self._front.shape
        assert self._tof.shape == (256,), self._tof.shape
        assert self._lowdim.shape == (21,), self._lowdim.shape
        self.observation_space = spaces.Dict({
            "front":  spaces.Box(-128, 127, (5400,), np.int8),
            "tof":    spaces.Box(-128, 127, (256,), np.int8),
            "lowdim": spaces.Box(-np.inf, np.inf, (21,), np.float32),
        })
        # Bounded so default_action_for_space -> 0 (not nan).
        self.action_space = spaces.Box(-10.0, 10.0, (2,), np.float32)
        print("[FusedNavProbe] serving baked frame from %s (front int8[5400], tof int8[256], "
              "lowdim f32[21])" % FRAME_DIR, flush=True)

    def _obs(self):
        return {"front": self._front.copy(), "tof": self._tof.copy(),
                "lowdim": self._lowdim.copy()}, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs()

    def step(self, action):
        obs, info = self._obs()
        return obs, 0.0, False, False, info
