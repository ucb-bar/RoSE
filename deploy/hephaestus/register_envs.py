from gymnasium.envs.registration import register
# NOTE: registration uses string entry_points (lazy import), so we do NOT eagerly
# import the env modules here. This keeps loading the synchronizer dependency-light:
# optional envs (airsim -> needs `airsim`; mujoco -> needs `gymnasium[mujoco]`) are
# only imported if you actually gym.make() them. Dummy envs (MiddleBury/LQR) need
# only numpy + opencv.

import gymnasium as gym

# id: the name the gym env is called (e.g. gym make)
# entry_point='[name of python file (included above):name of gym class]
register(
    id='AirSimEnv-v0',
    entry_point='envs.airsim.airsim_gym:AirSimEnv',
)

register(
    id='AirSimEnv-stereo-v0',
    entry_point='envs.airsim-stereo.airsim_gym:AirSimEnv',
)

register(
    id='LQR_gym_env-v0',
    entry_point='envs.customized_env.LQR_gym_env:LQREnvironment',
)

register(
    id='InvertedPendulumFine-v4',
    entry_point='envs.mujoco.inverted_pendulum_fine_v4:InvertedPendulumEnv',
)

register(
    id='MiddleBuryEnv-v0',
    entry_point='envs.middlebury.middlebury:MiddleBuryEnv',
)

# Known-pattern no-simulator env for bridge RX validation
register(
    id='PatternEnv-v0',
    entry_point='envs.pattern.pattern_env:PatternEnv',
)

# gym-pybullet-drones quadrotor: serves IMU state over the bridge, consumes motor RPMs
register(
    id='PyBulletDroneEnv-v0',
    entry_point='envs.pybullet_drone.drone_env:PyBulletDroneEnv',
)