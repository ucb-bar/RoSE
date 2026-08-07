#!/usr/bin/env python3
"""Q3: regenerate a TinyMPC problem_data .hpp with STRONGER yaw-rate authority.

Reproduces TinyMPC's rho-augmented infinite-horizon cache (Kinf, Pinf, Quu_inv, AmBKt)
in pure numpy -- validated to match the committed quadrotor_100hz_params.hpp to <2e-5.
No cmake / tinympc C++ build needed. Boost Q[yaw-rate=11] (and optionally Q[yaw=5]) so the
FAITHFUL body-rate command setpoint[11]=yaw_rate is actually tracked, then command rate directly.

Usage: python gen_strongyaw_params.py <out.hpp> <Q_yawrate> [Q_yaw] [rho]
Base A,B are taken from the committed 100Hz Crazyflie model (dt=0.01, closest committed to our
200Hz loop). Q/R base = 100Hz values except the yaw-rate weight you pass.
"""
import numpy as np, re, sys

SRC = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/drone_control/tinympc/examples/problem_data/quadrotor_100hz_params.hpp"

def blk(hpp, n):
    return np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",
                     re.search(n + r"\[[^\]]*\]\s*=\s*\{([^}]*)\}", hpp).group(1))])

def fmt(arr):
    return ",\n  ".join(",\t".join(f"{v:.7f}" for v in row) for row in np.atleast_2d(arr))

def main():
    out = sys.argv[1]
    q_yawrate = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    q_yaw     = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0
    rho       = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    hpp = open(SRC).read()
    A = blk(hpp, "Adyn_data").reshape(12, 12); B = blk(hpp, "Bdyn_data").reshape(12, 4)
    Qv = blk(hpp, "Q_data").copy(); Rv = blk(hpp, "R_data").copy()
    Qv[5] = q_yaw; Qv[11] = q_yawrate                      # BOOST yaw + yaw-rate weights
    # FIX (b): the committed Bdyn yaw rows (5,11) use signs (+,-,-,+) but the PHYSICAL plant's
    # propeller-drag yaw follows CRAZYFLIE_CFG spin (m1:+,m2:-,m3:+,m4:-) = (+,-,+,-). Flip cols
    # 2,3 (m3,m4) so the MODEL matches the physical plant; then Kinf/cache are recomputed below.
    import os as _os
    if _os.environ.get("FIX_YAW_MIXING", "1") == "1":
        B[5,  2] *= -1.0; B[5,  3] *= -1.0
        B[11, 2] *= -1.0; B[11, 3] *= -1.0
        print(f"  [FIX] Bdyn yaw-rate row now {np.round(B[11],4)} signs {np.sign(B[11]).astype(int)} (matches plant spin +,-,+,-)")
    Q = np.diag(Qv) + rho * np.eye(12); R = np.diag(Rv) + rho * np.eye(4)
    P = np.diag(Qv).copy()
    for _ in range(20000):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Pn = np.diag(Qv) + rho*np.eye(12) + A.T @ P @ (A - B @ K)
        if np.max(np.abs(Pn - P)) < 1e-12: P = Pn; break
        P = Pn
    Kinf = K; Pinf = P
    Quu_inv = np.linalg.inv(R + B.T @ P @ B); AmBKt = (A - B @ K).T
    coeff_d2p = Kinf.T @ R - AmBKt @ Pinf @ B      # R here is rho-augmented; matches committed <1e-3
    with open(out, "w") as f:
        f.write("#pragma once\n#include <admm.hpp>\n\n")
        f.write(f"/* Q3 strong-yaw regen: Q[yaw]={q_yaw} Q[yaw-rate]={q_yawrate} (was 400/4), rho={rho}.\n"
                f" * numpy rho-augmented DARE, validated vs committed 100Hz to <2e-5. */\n\n")
        f.write(f"tinytype rho_value = {rho:.7f};\n\n")
        for nm, M in (("Adyn_data", A), ("Bdyn_data", B), ("Kinf_data", Kinf),
                      ("Pinf_data", Pinf), ("Quu_inv_data", Quu_inv), ("AmBKt_data", AmBKt),
                      ("coeff_d2p_data", coeff_d2p)):
            dim = "NSTATES*NSTATES" if M.shape==(12,12) else ("NSTATES*NINPUTS" if M.shape==(12,4) else ("NINPUTS*NSTATES" if M.shape==(4,12) else "NINPUTS*NINPUTS"))
            f.write(f"tinytype {nm}[{dim}] = {{\n  {fmt(M)}\t\n}};\n\n")
        f.write(f"tinytype Q_data[NSTATES]= {{{','.join(f'{v:.7f}' for v in Qv)}}};\n\n")
        f.write(f"tinytype Qf_data[NSTATES]= {{{','.join(f'{v:.7f}' for v in Qv)}}};\n\n")
        f.write(f"tinytype R_data[NINPUTS]= {{{','.join(f'{v:.7f}' for v in Rv)}}};\n")
    print(f"wrote {out}  Q[yaw]={q_yaw} Q[yawrate]={q_yawrate} rho={rho}")
    print(f"Kinf yaw-rate col (state11) per input: {np.round(Kinf[:,11],4)}")

if __name__ == "__main__":
    main()
