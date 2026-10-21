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

# Normalized hover command per motor. Kept equal to the CF2X value the unchanged drone_control
# TinyMPC guest linearizes around, so the same ELF can fly this airframe.
HOVER_THRUST = 0.583
# PLACEHOLDER (no RiskyBird thrust curve yet): per-motor force at full normalized command,
# chosen so HOVER_THRUST exactly carries MASS_KG. Thrust-to-weight = 1 / HOVER_THRUST = 1.72,
# the same margin as the CF2X model. The Crazyflie's 0.58/4 N would give 1.01 here.
MAX_THRUST_N = MASS_KG * GRAVITY / (4.0 * HOVER_THRUST)
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
