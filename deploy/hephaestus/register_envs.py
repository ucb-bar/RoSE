from gymnasium.envs.registration import register
from envs.airsim import airsim_gym 
#import envs.customized_env.LQR_gym_env as LQR_gym_env
from envs.mujoco import inverted_pendulum_fine_v4

import gymnasium as gym

# id: the name the gym env is called (e.g. gym make)
# entry_point='[name of python file (included above):name of gym class]
register(
    id='AirSimEnv-v0',
    entry_point='envs.airsim.airsim_gym:AirSimEnv',
)

register(
    id='LQR_gym_env-v0',
    entry_point='envs.customized_env.LQR_gym_env:LQREnvironment',
)

register(
    id='InvertedPendulumFine-v4',
    entry_point='envs.mujoco.inverted_pendulum_fine_v4:InvertedPendulumEnv',
)