"""Build riskybird_rose.urdf: the scad RiskyBird airframe in the link layout RoSE's Isaac envs use.

The scad export (riskybird.urdf, vendored here unchanged) is one rigid `base_link` with massless
fixed rotor and sensor frames. The RoSE envs instead expect what the Isaac `cf2x.usd` Crazyflie
provides (see the RiskyBird ROSE_HANDOFF.md):

  - an articulation whose root link is named `body` (disturbances, cameras, contact sensor);
  - four dynamic links matching `m.*_prop`, where per-rotor thrust is applied;
  - continuous joints `m1_joint`..`m4_joint` about +z (the props spin freely, as on cf2x).

Motor order is the PCB wiring, FR, RR, RL, FL = MOTOR1..4, which is also the gym-pybullet-drones
CF2X order the drone_control TinyMPC guest was built for. Each prop link gets a small mass so
PhysX accepts it; `body` is recomputed so the assembly keeps the scad total mass, CoM and inertia
(checked below). Sensor frames are dropped: massless links would become zero-mass bodies, and the
envs carry the sensor poses as constants instead.

usage: python3 make_rose_urdf.py   (rewrites riskybird_rose.urdf next to this file)
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "riskybird.urdf")
DST = os.path.join(HERE, "riskybird_rose.urdf")

ROTORS = ["FR", "RR", "RL", "FL"]           # m1..m4, PCB MOTOR1..4
PROP_MASS = 1.0e-4                          # kg; placeholder so PhysX has a dynamic body
PROP_RADIUS = 0.0255                        # m; dronesim ground_effect.rotor_radius_m


def _vec(s):
    return np.array([float(x) for x in s.split()], dtype=np.float64)


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def _shift(m, d):
    """Parallel-axis term m * (d.d E - d d^T)."""
    d = np.asarray(d, dtype=np.float64)
    return m * (d @ d * np.eye(3) - np.outer(d, d))


def _read_inertial(link):
    inert = link.find("inertial")
    m = float(inert.find("mass").get("value"))
    c = _vec(inert.find("origin").get("xyz"))
    a = inert.find("inertia").attrib
    I = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                  [a["ixy"], a["iyy"], a["iyz"]],
                  [a["ixz"], a["iyz"], a["izz"]]], dtype=np.float64)
    return m, c, I


def _write_inertial(link, m, c, I):
    inert = link.find("inertial")
    if inert is None:
        inert = ET.SubElement(link, "inertial")
    inert.clear()
    ET.SubElement(inert, "origin", xyz=_fmt(c), rpy="0 0 0")
    ET.SubElement(inert, "mass", value=f"{m:.9g}")
    ET.SubElement(inert, "inertia", ixx=f"{I[0, 0]:.9g}", ixy=f"{I[0, 1]:.9g}",
                  ixz=f"{I[0, 2]:.9g}", iyy=f"{I[1, 1]:.9g}", iyz=f"{I[1, 2]:.9g}",
                  izz=f"{I[2, 2]:.9g}")


def main():
    tree = ET.parse(SRC)
    robot = tree.getroot()
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = {j.get("name"): j for j in robot.findall("joint")}
    base = links["base_link"]
    M, c, I_total = _read_inertial(base)

    # rotor positions from the scad fixed rotor frames (prop plane, body FLU)
    rotor_pos = [_vec(joints[f"rotor_{r}_joint"].find("origin").get("xyz")) for r in ROTORS]

    # drop every massless fixed frame (rotors and sensors)
    for name, j in joints.items():
        if j.get("type") == "fixed":
            robot.remove(j)
            robot.remove(links[j.find("child").get("link")])

    # prop link inertia: thin disk about its own centre
    I_prop = PROP_MASS * PROP_RADIUS ** 2 * np.diag([0.25, 0.25, 0.5])

    # body = total minus the props, so the assembly matches the scad totals
    m_b = M - 4 * PROP_MASS
    c_b = (M * c - PROP_MASS * np.sum(rotor_pos, axis=0)) / m_b
    I_b = I_total - _shift(m_b, c_b - c)
    for p in rotor_pos:
        I_b -= I_prop + _shift(PROP_MASS, p - c)
    assert np.all(np.linalg.eigvalsh(I_b) > 0), "body inertia not positive definite"

    base.set("name", "body")
    _write_inertial(base, m_b, c_b, I_b)

    # move each prop's visual onto its prop link so it spins with the joint
    prop_visuals = {v.get("name"): v for v in base.findall("visual") if v.get("name", "").startswith("Prop_")}
    for i, (r, p) in enumerate(zip(ROTORS, rotor_pos), start=1):
        link = ET.SubElement(robot, "link", name=f"m{i}_prop")
        _write_inertial(link, PROP_MASS, np.zeros(3), I_prop)
        vis = prop_visuals.get(f"Prop_{r}")
        if vis is not None:
            base.remove(vis)
            # scad meshes are authored in body coordinates; re-express in the prop frame
            vis.find("origin").set("xyz", _fmt(-p))
            link.append(vis)
        joint = ET.SubElement(robot, "joint", name=f"m{i}_joint", type="continuous")
        ET.SubElement(joint, "parent", link="body")
        ET.SubElement(joint, "child", link=f"m{i}_prop")
        ET.SubElement(joint, "origin", xyz=_fmt(p), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="0 0 1")
        robot.append(ET.Comment(f" m{i} = {r} (PCB MOTOR{i}) "))

    # closure check: the assembly must reproduce the scad totals
    parts = [(m_b, c_b, I_b)] + [(PROP_MASS, p, I_prop) for p in rotor_pos]
    M2 = sum(m for m, _, _ in parts)
    c2 = sum(m * ci for m, ci, _ in parts) / M2
    I2 = sum(Ii + _shift(m, ci - c2) for m, ci, Ii in parts)
    assert abs(M2 - M) < 1e-12 and np.allclose(c2, c, atol=1e-12) and np.allclose(I2, I_total, rtol=1e-9, atol=1e-15)

    ET.indent(tree, space="  ")
    header = ("<!-- GENERATED by make_rose_urdf.py from riskybird.urdf (scad export); do not edit.\n"
              "     RoSE layout: root link `body`, props m1..m4 = FR, RR, RL, FL on continuous +z joints. -->\n")
    with open(DST, "w") as f:
        f.write('<?xml version="1.0"?>\n' + header)
        f.write(ET.tostring(robot, encoding="unicode"))
        f.write("\n")
    print(f"wrote {DST}: total {M2 * 1e3:.2f} g, body {m_b * 1e3:.2f} g, "
          f"principal I {np.round(np.linalg.eigvalsh(I2) * 1e6, 3)} e-6 kg m^2")


if __name__ == "__main__":
    main()
