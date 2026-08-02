#!/usr/bin/env python3
"""Plot a ground-truth drone trajectory logged by the RoSE drone envs.

The IsaacCrazyflie / PyBullet drone MPC envs can log ground truth to CSV (set the
`traj_csv` gym_kwarg or $ROSE_TRAJ_CSV). This renders that CSV into a single figure:
a 3D flight path, altitude-vs-time against the setpoint, horizontal drift, and the
per-rotor thrust commands.

    python plot_trajectory.py <traj.csv> [-o out.png]

Headless-safe (Agg backend); writes a PNG next to the CSV by default.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    return {k: data[:, i] for i, k in enumerate(header)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.csv)[0] + ".png"

    d = load(args.csv)
    t = d["t"]
    tx, ty, tz = d["tx"][-1], d["ty"][-1], d["tz"][-1]

    fig = plt.figure(figsize=(13, 9))

    # 3D path
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(d["x"], d["y"], d["z"], lw=1.6, color="#2b6cb0", label="flight path")
    ax.scatter([d["x"][0]], [d["y"][0]], [d["z"][0]], c="#38a169", s=40, label="start")
    ax.scatter([tx], [ty], [tz], c="#e53e3e", marker="*", s=120, label="setpoint")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Ground-truth 3D trajectory"); ax.legend(loc="upper left", fontsize=8)

    # altitude vs time
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, d["z"], color="#2b6cb0", label="z (truth)")
    ax.axhline(tz, ls="--", color="#e53e3e", label=f"setpoint z={tz:.2f}")
    ax.set_xlabel("control time [s]"); ax.set_ylabel("altitude z [m]")
    ax.set_title("Altitude tracking"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # horizontal drift vs time
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, d["x"] - tx, color="#805ad5", label="x - x*")
    ax.plot(t, d["y"] - ty, color="#dd6b20", label="y - y*")
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("control time [s]"); ax.set_ylabel("horizontal error [m]")
    ax.set_title("Horizontal drift"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # per-rotor thrust
    ax = fig.add_subplot(2, 2, 4)
    for i in range(4):
        ax.plot(t, d[f"f{i}"], lw=1.0, label=f"rotor {i}")
    ax.set_xlabel("control time [s]"); ax.set_ylabel("per-rotor force [N]")
    ax.set_title("Applied per-rotor thrust"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    zf = d["z"][-1]
    fig.suptitle(f"RoSE drone_control (TinyMPC over the bridge) — "
                 f"final z={zf:.3f} m (err {zf - tz:+.3f}), {len(t)} control steps",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  ({len(t)} steps, final z={zf:.3f}, z_err={zf - tz:+.3f})")


if __name__ == "__main__":
    main()
