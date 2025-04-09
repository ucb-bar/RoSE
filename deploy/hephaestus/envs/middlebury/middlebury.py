import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np

class MiddleBuryEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.image_dim = 256
        self.observation_space = spaces.Dict({
            "camera_lr": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim*2), dtype=np.uint8),
            "camera_l": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim), dtype=np.uint8),
            "camera_r": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim), dtype=np.uint8),
            "dummy": spaces.Box(low=0, high=255, shape=(4,), dtype=np.uint8),
            "lat_test": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
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

        camera_lr_observation = np.empty((self.image_dim * 2, self.image_dim), dtype=np.uint8)
        camera_lr_observation[0::2, :] = left_resized_grey_img
        camera_lr_observation[1::2, :] = right_resized_grey_img

        camera_left_observation = np.empty((self.image_dim, self.image_dim), dtype=np.uint8)
        camera_left_observation = left_resized_grey_img

        camera_right_observation = np.empty((self.image_dim, self.image_dim), dtype=np.uint8)
        camera_right_observation = right_resized_grey_img

        dummy = np.array([1, 2, 3, 4])

        lat_test = np.array([1])

        observation = {
            "camera_l": camera_left_observation,
            "camera_lr": camera_lr_observation,
            "camera_r": camera_right_observation,
            "dummy": dummy,
            "lat_test": lat_test
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
