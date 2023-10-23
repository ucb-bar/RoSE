import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }

    def __init__(self, mujoco_timestep=None, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Get the directory of the current script
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Join the directory with the XML filename to get the absolute path
        model_path = os.path.join(dir_path, "inverted_pendulum_fine.xml")

        MujocoEnv.__init__(
            self,
            model_path,
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        if mujoco_timestep is not None:
            self.model.opt.timestep = mujoco_timestep
            self.metadata['render_fps'] = int(1.0 / (mujoco_timestep * self.frame_skip))

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2 or np.abs(ob[0]) > 0.3))
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def reset_model(self):
        angle_range = np.array([[-0.1, -0.05], [0.05, 0.1]])
        selected_range = angle_range[self.np_random.choice(len(angle_range))]
        angle = self.np_random.uniform(low=selected_range[0], high=selected_range[1])
        
        qpos = self.init_qpos.copy()
        qpos[0] = angle  
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
    
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.array(np.concatenate([self.data.qpos, self.data.qvel]).ravel(), dtype=np.float32)
