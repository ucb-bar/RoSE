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

# GPU-free stand-in for validating the HM01B0 FPV camera DMA path headlessly: serves the SAME
# analytic scene-derived frames as the Isaac multisensor env, from a scripted hallway flight,
# with no isaaclab dependency. Pair with config_gym_CrazyflieCamProbeEnv-v0.yaml.
register(
    id='CrazyflieCamProbeEnv-v0',
    entry_point='envs.cam_probe.cam_probe_env:CrazyflieCamProbeEnv',
)
# GPU-free bench env for the DroNet vision-nav controller (P3): serves the full flight sensor
# set (IMU/flow/ToF/4x multizone) + forward FPV RGB from a fixed hover, so the on-SoC
# camera->DroNet->setpoint->control integration runs headlessly. Pair with
# config_gym_CrazyflieVisionNavProbeEnv-v0.yaml.
register(
    id='CrazyflieVisionNavProbeEnv-v0',
    entry_point='envs.vision_nav_probe.vision_nav_probe_env:CrazyflieVisionNavProbeEnv',
)

# M1 (fused-vision nav on spike-in-the-loop): GPU-free deterministic bench that serves ONE
# baked pre-quantized frame so the guest's model output can be byte-compared to the host
# reference. Pair with config_gym_FusedNavProbeEnv-v0.yaml.
register(
    id='FusedNavProbeEnv-v0',
    entry_point='envs.warehouse_fused_nav.fused_nav_probe_env:FusedNavProbeEnv',
)

# M1 real path: instantiates the SAME photoreal warehouse-gate IsaacLab env as
# sims/scripts/eval_fused_warehouse.py, serving the real onboard sense() suite pre-quantized
# per the host reference over the RoSE bridge, and applying the guest's (yaw_rate,
# forward_speed) via the warehouse velocity action. Needs env_isaaclab (GPU); imported lazily.
# Pair with config_gym_WarehouseFusedNavBridgeEnv-v0.yaml.
register(
    id='WarehouseFusedNavBridgeEnv-v0',
    entry_point='envs.warehouse_fused_nav.warehouse_fused_nav_env:WarehouseFusedNavBridgeEnv',
)
