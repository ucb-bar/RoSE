from gymnasium.envs.registration import register
import airsim_gym 

import gymnasium as gym 

register(
    id='AirSimEnv-v0',
    entry_point='airsim_gym:AirSimEnv',
)
