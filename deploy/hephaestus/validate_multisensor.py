#!/usr/bin/env python3
"""Runtime validation for IsaacCrazyflieMultiSensorEnv (task: sensors + environment work).

Boots the env in Isaac (headless, camera on), then checks:
  1. The observation dict carries the 4 horizontal VL53L5CX zone grids + the FPV frame.
  2. Multizone-ToF geometry is correct in a live env: at the room center every sensor reads
     ~half-room to its facing wall; offsetting the drone toward a wall makes that sensor read
     closer and the opposite sensor farther.
  3. The FPV camera produces a non-trivial (non-zero, varied) frame.

Run (from deploy/hephaestus, Isaac-capable python):
    ROSE_ISAAC_CAMERA=1 ROSE_TOF_ROOM_X=2 ROSE_TOF_ROOM_Y=2 \
        <env_isaaclab python> validate_multisensor.py
"""
import os
import numpy as np
import gymnasium as gym
import register_envs  # noqa: F401  (registers the env ids)

ZC = None


def center_zone(grid, zones):
    return float(np.asarray(grid).reshape(zones, zones)[zones // 2, zones // 2])


def main():
    os.environ.setdefault("ROSE_ISAAC_CAMERA", "1")
    env = gym.make("IsaacCrazyflieMultiSensorEnv-v0", render_mode="rgb_array")
    u = env.unwrapped
    zones = u._tof_zones
    rx = u._room_max[0]
    print("[val] room_max=%s zones=%d" % (u._room_max.tolist(), zones), flush=True)

    obs, info = env.reset()
    need = {"front", "left", "back", "right", "fpv"}
    missing = need - set(obs.keys())
    assert not missing, "obs missing sensor keys: %s" % missing
    print("[val] obs keys OK: %s" % sorted(obs.keys()), flush=True)

    # --- geometry probe at known poses (level quat) ---
    q = np.array([1.0, 0.0, 0.0, 0.0])
    checks = [
        ("center",      [0.0, 0.0, 1.0]),
        ("near +x wall",[1.5, 0.0, 1.0]),
        ("near -y wall",[0.0, -1.5, 1.0]),
    ]
    ok = True
    for label, p in checks:
        g = u._synth_multizone_tof(np.array(p, float), q)
        fr, bk = center_zone(g["front"], zones), center_zone(g["back"], zones)
        lf, rt = center_zone(g["left"], zones), center_zone(g["right"], zones)
        print("[val] %-12s front=%.2f back=%.2f left=%.2f right=%.2f"
              % (label, fr, bk, lf, rt), flush=True)
        if label == "center":
            ok &= all(abs(v - rx) < 0.25 for v in (fr, bk, lf, rt))
        if label == "near +x wall":
            ok &= (fr < 0.7) and (bk > 3.0)          # closer to +x, farther from -x
        if label == "near -y wall":
            ok &= (rt < 0.7) and (lf > 3.0)          # right bore = -y

    # --- runtime: step a few hover ticks, read the live obs + FPV ---
    fpv_ok = False
    for i in range(6):
        obs, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    fpv = np.asarray(obs["fpv"])
    frac_nonzero = float((fpv > 0).mean())
    nunique = int(np.unique(fpv).size)
    fpv_ok = frac_nonzero > 0.05 and nunique > 5
    print("[val] live obs front.center=%.2f  fpv: nonzero=%.1f%% unique=%d"
          % (center_zone(obs["front"], zones), 100 * frac_nonzero, nunique), flush=True)

    env.close()
    print("[val] GEOMETRY=%s  FPV=%s" % ("PASS" if ok else "FAIL",
                                         "PASS" if fpv_ok else "FAIL"), flush=True)
    print("[val] RESULT=%s" % ("PASS" if (ok and fpv_ok) else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
