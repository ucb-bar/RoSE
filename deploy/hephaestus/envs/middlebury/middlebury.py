import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np

class MiddleBuryEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.image_dim = 128
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3), dtype=np.uint8),
        })
        self.action_space = spaces.Discrete(1)

    def calc_observation(self):
        # read image
        left_png = cv2.imread("envs/middlebury/left.png")
        right_png = cv2.imread("envs/middlebury/right.png")

        left_grey_img = cv2.cvtColor(left_png, cv2.COLOR_BGR2GRAY)
        right_grey_img = cv2.cvtColor(right_png, cv2.COLOR_BGR2GRAY)

        left_resized_grey_img = cv2.resize(left_grey_img, (self.image_dim, self.image_dim))
        right_resized_grey_img = cv2.resize(right_grey_img, (self.image_dim, self.image_dim))

        camera_observation = np.empty((self.image_dim * 2, self.image_dim), dtype=np.uint8)
        camera_observation[0::2, :] = left_resized_grey_img
        camera_observation[1::2, :] = right_resized_grey_img

        observation = {
            "camera": camera_observation,
        }

        return observation

    def reset(self):
        return self.calc_observation(), {}

    def step(self, action):
        reward = 0
        done = False
        info = {}

        return self.calc_observation(), reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def configure_logs(self):
        pass