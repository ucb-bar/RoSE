#!/usr/bin/env python3
"""Validate the ROSE_MAZE navigation environment (hallway/maze) for the multisensor env.

Checks that (1) the named wall set is spawned as visible geometry and (2) the horizontal
multizone ToFs range against the corridor walls: at the corridor center the LEFT/RIGHT sensors
read the ~0.7 m wall while FRONT/BACK stay open. Also captures chase-camera frames (drone
hovering in the hallway) as a visual record of the environment.

Run (from deploy/hephaestus, Isaac python):
    ROSE_MAZE=hallway ROSE_ISAAC_CAMERA=1 ROSE_TOF_ROOM_X=2.5 ROSE_TOF_ROOM_Y=2 \
        ROSE_ISAAC_FRAMEDIR=<dir> <env_isaaclab python> validate_hallway.py
"""
import os
import numpy as np
import gymnasium as gym
import register_envs  # noqa: F401


def cz(grid, z):
    return float(np.asarray(grid).reshape(z, z)[z // 2, z // 2])


def main():
    os.environ.setdefault("ROSE_MAZE", "hallway")
    os.environ.setdefault("ROSE_ISAAC_CAMERA", "1")
    env = gym.make("IsaacCrazyflieMultiSensorEnv-v0", render_mode="rgb_array")
    u = env.unwrapped
    z = u._tof_zones
    print("[hall] maze=%s obstacles=%d" % (u._maze, len(u._obstacles)), flush=True)
    assert len(u._obstacles) >= 2, "hallway walls not registered as analytic obstacles"

    env.reset()

    # ToF geometry: level drone along the corridor centerline (y=0). Left/right bores hit the
    # ~0.7 m walls; front/back stay open (>1.5 m).
    q = np.array([1.0, 0.0, 0.0, 0.0])
    ok = True
    for x in (-1.0, 0.0, 1.0):
        g = u._synth_multizone_tof(np.array([x, 0.0, 1.0]), q)
        fr, bk, lf, rt = cz(g["front"], z), cz(g["back"], z), cz(g["left"], z), cz(g["right"], z)
        print("[hall] x=%+.1f  front=%.2f back=%.2f left=%.2f right=%.2f" % (x, fr, bk, lf, rt),
              flush=True)
        ok &= (0.5 < lf < 0.9) and (0.5 < rt < 0.9) and (fr > 1.5) and (bk > 1.5)

    # capture a few chase-cam frames (hover) as a visual record of the hallway
    frames = 0
    for i in range(40):
        env.step(np.zeros(4, dtype=np.float32))
        frames += 1
    print("[hall] captured ~%d hover frames (ROSE_ISAAC_FRAMEDIR)" % frames, flush=True)

    env.close()
    print("[hall] RESULT=%s" % ("PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
