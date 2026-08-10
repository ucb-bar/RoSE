# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates a quadcopter (Crazyflie) with a forward-facing FPV camera on the airframe.

The camera is a :class:`~isaaclab.sensors.camera.Camera` sensor parented to the ``body`` link, with a
ROS-style offset so the optical axis looks along the vehicle forward direction.

.. code-block:: bash

    # Usage (cameras require the flag)
    ./isaaclab.sh -p scripts/demos/quadcopter_fpv.py --enable_cameras

    # Headless / no extra window
    ./isaaclab.sh -p scripts/demos/quadcopter_fpv.py --enable_cameras --headless
    ./isaaclab.sh -p scripts/demos/quadcopter_fpv.py --enable_cameras --no_fpv_plot

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Quadcopter demo with an onboard FPV camera sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--no_fpv_plot",
    action="store_true",
    help="Do not open a matplotlib window for live FPV (useful if you only want Isaac Sim).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip


@configclass
class QuadcopterFpvSceneCfg(InteractiveSceneCfg):
    """Scene with Crazyflie and an FPV camera on the main body link."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot (matches default init pose from CRAZYFLIE_CFG)
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # FPV camera: slightly forward on the body, ROS optical frame (+Z forward, -Y up)
    fpv_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/fpv_cam",
        update_period=0.04,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.01),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


def _tensor_rgb_to_uint8_hwc(rgb: torch.Tensor, env_idx: int) -> np.ndarray:
    """Convert camera RGB tensor (N, H, W, C) to uint8 (H, W, 3) for display."""
    img = rgb[env_idx, ..., :3].detach().cpu().numpy()
    if img.dtype == np.uint8:
        return img
    if img.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    img_max = float(np.nanmax(img))
    if img_max <= 1.0:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def _tensor_depth_to_vis(depth: torch.Tensor, env_idx: int, clip_m: float = 15.0) -> np.ndarray:
    """Convert depth to a float32 (H, W) array in [0, 1] for colormap display."""
    d = depth[env_idx].detach().cpu().numpy()
    if d.ndim == 3:
        d = d[..., 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(d / clip_m, 0.0, 1.0)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    scene_cfg = QuadcopterFpvSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot = scene["robot"]
    fpv = scene["fpv_camera"]
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    print("[INFO]: Setup complete (FPV camera on /Robot/body/fpv_cam).")

    show_fpv_plot = not args_cli.headless and not args_cli.no_fpv_plot
    fpv_fig = None
    im_rgb = im_depth = None

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    while simulation_app.is_running():
        if count % 2000 == 0:
            sim_time = 0.0
            count = 0
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            scene.reset()
            print(">>>>>>>> Reset!")

        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = robot_mass * gravity / 4.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=prop_body_ids,
        )

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        rgb = fpv.data.output["rgb"]
        depth = fpv.data.output["distance_to_image_plane"]

        # Live matplotlib window (~50 Hz redraw; full physics still runs every step)
        if show_fpv_plot and rgb.shape[0] > 0 and count % 4 == 0:
            if fpv_fig is not None and not plt.fignum_exists(fpv_fig.number):
                fpv_fig = None
                im_rgb = im_depth = None
            rgb_np = _tensor_rgb_to_uint8_hwc(rgb, 0)
            depth_vis = _tensor_depth_to_vis(depth, 0)
            if fpv_fig is None:
                plt.ion()
                fpv_fig, axes = plt.subplots(1, 2, num="Quadcopter FPV", figsize=(10.5, 4.0))
                im_rgb = axes[0].imshow(rgb_np)
                axes[0].set_title("RGB (env 0)")
                axes[0].axis("off")
                im_depth = axes[1].imshow(depth_vis, cmap="turbo", vmin=0.0, vmax=1.0)
                axes[1].set_title("Depth (normalized, ~15 m max)")
                axes[1].axis("off")
                plt.tight_layout()
            else:
                im_rgb.set_data(rgb_np)
                im_depth.set_data(depth_vis)
            if fpv_fig is not None and plt.fignum_exists(fpv_fig.number):
                fpv_fig.canvas.draw_idle()
                fpv_fig.canvas.flush_events()
                plt.pause(0.001)

        if count % 120 == 0:
            print(
                f"[FPV] t={sim_time:.2f}s rgb {tuple(rgb.shape)} "
                f"depth {tuple(depth.shape)} rgb_mean={rgb.float().mean().item():.3f}"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
