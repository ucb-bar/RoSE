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

# gym-pybullet-drones quadrotor for the TinyMPC drone_control loop: serves the full
# 12-DoF linearized state and consumes 4 normalized motor thrusts (closed-loop control).
register(
    id='PyBulletDroneMPCEnv-v0',
    entry_point='envs.pybullet_drone.drone_mpc_env:PyBulletDroneMPCEnv',
)

# IsaacLab (Isaac Sim) Crazyflie for the same TinyMPC drone_control loop: identical
# RoSE contract as PyBulletDroneMPCEnv-v0 (full 12-DoF state, 4 per-rotor thrusts), so
# the same Zephyr guest runs unchanged. Needs the Isaac Sim toolchain (env_isaaclab);
# imported lazily so the synchronizer stays dependency-light unless gym.make()'d.
register(
    id='IsaacCrazyflieMPCEnv-v0',
    entry_point='envs.isaac_crazyflie.crazyflie_mpc_env:IsaacCrazyflieMPCEnv',
)

# Same Crazyflie physics, but serves REAL SENSOR MODALITIES (IMU accel+gyro, optical
# flow) instead of ground-truth state. The SoC (samples/rose_flight_controller) runs a
# state estimator on these before TinyMPC. Structured per-modality obs; pose drifts since
# there is no absolute position/attitude reference.
register(
    id='IsaacCrazyflieSensorEnv-v0',
    entry_point='envs.isaac_crazyflie.crazyflie_sensor_env:IsaacCrazyflieSensorEnv',
)

# docs/ROSE_FUTURE_SENSORS_PLAN.md: adds 4× VL53L5CX multizone ToF (F/R/B/L) + an HM01B0 FPV
# camera on top of the sensor env. Not used by the stress loop (that env id is unchanged); this
# is the navigation/obstacle-sensing env.
register(
    id='IsaacCrazyflieMultiSensorEnv-v0',
    entry_point='envs.isaac_crazyflie.crazyflie_multisensor_env:IsaacCrazyflieMultiSensorEnv',
)