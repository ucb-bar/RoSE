#!/usr/bin/env python
"""Headless, GPU-free registration check for the RoSE nav co-sim gym envs.

Run with the env_isaaclab conda python:
    /path/to/env_isaaclab/bin/python experiments/rose_nav_cosim/isaaclab_overlay/check_registration.py

Imports ONLY the registration modules (which hard-import just gymnasium; the
env_cfg entry points are lazy strings) and asserts the co-sim's gym env ids are
present in the gymnasium registry. It never calls gym.make() or AppLauncher, so it
does NOT open an Omniverse app or touch the GPU -- safe to run alongside a live
Isaac Sim workload on a shared GPU.

Resolves the RoSE repo root from this file's location, so it works from any clone
path (override with ROSE_DIR if the layout differs).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../RoSE/experiments/rose_nav_cosim/isaaclab_overlay -> up 3 = RoSE
_DEFAULT_ROSE = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
ROSE = os.environ.get("ROSE_DIR", _DEFAULT_ROSE)
XPURT = os.path.join(ROSE, "soc/sw/xpu-rt")
HEPH = os.path.join(ROSE, "deploy/hephaestus")
for p in (XPURT, HEPH):
    if p not in sys.path:
        sys.path.insert(0, p)

import gymnasium as gym  # noqa: E402

need_task = [
    "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0",  # co-sim TASK_ID
    "Isaac-Drone-Warehouse-Nav-Crazyflie-v0",
    "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-Crowded-v0",
]
need_bridge = ["WarehouseThrustEnv-v0", "WarehouseFusedNavBridgeEnv-v0", "FusedNavProbeEnv-v0"]

# 1) custom IsaacLab task ids (sims.isaaclab_tasks, tracked in the xpu-rt submodule)
import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: F401,E402
# 2) RoSE bridge envs (deploy/hephaestus)
import register_envs  # noqa: F401,E402

keys = set(gym.registry.keys())
ok = True
print(f"ROSE_DIR = {ROSE}")
print("=== task ids (sims.isaaclab_tasks, xpu-rt submodule) ===")
for i in need_task:
    hit = i in keys
    ok &= hit
    print(f"  [{'OK' if hit else 'MISSING'}] {i}")
print("=== bridge ids (deploy/hephaestus/register_envs) ===")
for i in need_bridge:
    hit = i in keys
    ok &= hit
    print(f"  [{'OK' if hit else 'MISSING'}] {i}")

n_isaac = sum(1 for k in keys if k.startswith("Isaac-Drone-Warehouse"))
print(f"\nTotal Isaac-Drone-Warehouse* ids registered: {n_isaac}")
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
