import gymnasium as gym
from gymnasium import spaces
import numpy as np

# A trivial no-simulator env that serves a KNOWN, predictable uint32 pattern so the
# SoC side can validate exactly what it received (vs the camera env, whose pixel data
# is hard to predict). Pattern = [PATTERN_BASE + 0, +1, ..., +PATTERN_LEN-1].
PATTERN_BASE = 0xC0DE0000
PATTERN_LEN = 16   # serve more than the SoC reads, so the RX read never blocks


class PatternEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, **kwargs):
        self.observation_space = spaces.Dict({
            "pattern": spaces.Box(low=0, high=np.iinfo(np.uint32).max,
                                  shape=(PATTERN_LEN,), dtype=np.uint32),
        })
        self.action_space = spaces.Discrete(1)

    def _obs(self):
        return {"pattern": (PATTERN_BASE + np.arange(PATTERN_LEN, dtype=np.uint32)).astype(np.uint32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}

    def render(self):
        return None

    def close(self):
        pass
