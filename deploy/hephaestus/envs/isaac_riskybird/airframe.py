"""RiskyBird airframe for the RoSE Isaac envs: asset, plant constants and sensor poses.

Everything the Isaac RiskyBird envs need to know about the vehicle lives here, so the env
code stays a diff of the Crazyflie env. Source of the numbers: the scad RiskyBird model
(assets/, see assets/SOURCE.txt), whose manifest.json gives each one's status. In short:
mass is documented, CoM/inertia are inferred from a component model, and the thrust curve,
yaw-drag ratio and spin directions are NOT measured. The constants marked PLACEHOLDER must
be replaced once the motors/props are characterized.
"""

import os
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))

# RoSE-layout URDF (root `body`, props m1..m4 = FR, RR, RL, FL); see assets/make_rose_urdf.py.
URDF_PATH = os.path.join(_HERE, "assets", "riskybird_rose.urdf")

MASS_KG = 0.0585               # scad total (MEASURED-DOC); the env checks PhysX against it
GRAVITY = 9.81

# Wire convention shared by every RoSE drone guest: the SoC sends u = duty - DUTY_OFFSET per
# motor, so motor duty = u + DUTY_OFFSET, clamped to [0, 1].
DUTY_OFFSET = 0.583
# PLACEHOLDER (no RiskyBird thrust-stand data yet): per-motor thrust vs. duty is the model the
# riskybirdv3 flight controller (zephyr-chipyard-sw samples/rose_flight_controller,
# controller_pid.cpp forceToVoltage) inverts, so the plant and the controller agree: Bitcraze's
# published 7x16 + 45-35 TOTAL-thrust fit, thrust[g] = A*duty^2 + B*duty, scaled by PROP_GAIN for
# RiskyBird's 7x20 motors + 51MMX2 props, divided over 4 motors. Full duty gives 0.204 N/motor,
# thrust-to-weight 1.43; hover sits at duty ~0.78.
THRUST_A_G = 26.919633         # g per duty^2, whole-vehicle fit
THRUST_B_G = 35.754861         # g per duty, whole-vehicle fit
PROP_GAIN = 1.33
N_PER_GRAM = 9.80665e-3


def thrust_n(duty):
    """Per-motor thrust (N) at duty in [0, 1] (works elementwise on numpy arrays)."""
    return PROP_GAIN * (THRUST_A_G * duty * duty + THRUST_B_G * duty) * N_PER_GRAM / 4.0
# PLACEHOLDER: propeller drag/thrust ratio, CF2X value (KM=7.94e-12, KF=3.16e-10).
KM_OVER_KF = 7.94e-12 / 3.16e-10
# Yaw reaction sign per motor m1..m4 (FR, RR, RL, FL): reaction torque about +z is
# SPIN_SIGN * KM_OVER_KF * thrust. From the scad model's spin_signs (copied from dronesim,
# UNVERIFIED on the real drone); also the gym-pybullet-drones CF2X convention the
# drone_control TinyMPC was tuned against. A -1 motor's prop turns CCW seen from above.
SPIN_SIGN = (-1.0, +1.0, -1.0, +1.0)
# Visual prop spin (rad/s) for the m*_joint init velocities: CCW (+z) for SPIN_SIGN -1.
PROP_SPIN_RAD_S = 200.0

# Sensor poses, body FLU metres from the board-centre datum (= `body` origin), measured from
# the PCB (dronesim_plant.json). Orientations are identity except the down ToF (faces -z);
# the camera tilt is not in the model.
CAMERA_POS = (0.0355, 0.0, -0.003)
IMU_POS = (-0.000004, 0.016, -0.002075)
FLOW_POS = (-0.02615, 0.0001, -0.00396)
TOF_DOWN_POS = (-0.021572, 0.010537, -0.00316)


def robot_cfg(prim_path="{ENV_REGEX_NS}/Robot"):
    """ArticulationCfg for RiskyBird. Import only after the Isaac Sim app has booted.

    Keeps every CRAZYFLIE_CFG setting (rigid/articulation properties, the zero-gain dummy
    actuator) and swaps the spawn for the RiskyBird URDF, converted to USD on first use and
    cached per user (the shared /tmp/IsaacLab default is often owned by someone else).
    """
    import isaaclab.sim as sim_utils
    from isaaclab_assets import CRAZYFLIE_CFG

    cf = CRAZYFLIE_CFG.spawn
    spawn = sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        usd_dir=os.path.join(tempfile.gettempdir(), f"isaaclab-{os.getuid()}", "riskybird_usd"),
        fix_base=False,
        merge_fixed_joints=False,
        root_link_name="body",
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="none",
        ),
        rigid_props=cf.rigid_props,
        articulation_props=cf.articulation_props,
    )
    spin = {f"m{i + 1}_joint": -s * PROP_SPIN_RAD_S for i, s in enumerate(SPIN_SIGN)}
    return CRAZYFLIE_CFG.replace(
        prim_path=prim_path,
        spawn=spawn,
        init_state=CRAZYFLIE_CFG.init_state.replace(joint_vel=spin),
    )
