#!/usr/bin/env python3
"""Sim-throughput characterization of the successful 3-gate FPGA gate-nav run.

Parses the harness wall-clock heartbeat log (traces/fpga_gatenav_3gate_heartbeat.log:
lines like "[HH:MM:SS]   t+<wall>s grant_iter=<N> gates=<g> | physics: <t>s x=..") plus
the physics trajectory CSV (traces/fpga_gatenav_3gate_traj.csv). Emits the co-sim
throughput table.

Model / constants (from config_hwdb_local.yaml + gatenav_flight.sh env):
  FPGA host clock  = 30 MHz   (RoseTLRocketSaturnMMIOOnlyConfig, bitstream 2026-08-13--14-28-37)
  target modeled   = 1 GHz    (ROSE_FIRESIM_FREQ=1e9)
  cycles / grant   = 5e6      (ROSE_FIRESIM_STEP)  -> 5 ms target-time / grant
  physics timestep = 5 ms     (gym_timestep, WarehouseThrustEnv-v0)
  FireSim maps 1 target cycle -> 1 host FPGA cycle (single-clock target).
"""
import re, sys, pathlib

HERE = pathlib.Path(__file__).parent
HB   = HERE / "traces/fpga_gatenav_3gate_heartbeat.log"
TRAJ = HERE / "traces/fpga_gatenav_3gate_traj.csv"

F_FPGA_HZ   = 30e6
F_TARGET_HZ = 1e9
CYC_PER_GRANT = 5e6
TARGET_S_PER_GRANT = CYC_PER_GRANT / F_TARGET_HZ   # 5 ms

hb = []  # (wall_s, grant_iter, physics_s, gates)
for ln in HB.read_text().splitlines():
    m = re.search(r"t\+(\d+)s grant_iter=(\d+) gates=(\d+) \| physics: ([\d.]+)s", ln)
    if m:
        hb.append((int(m[1]), int(m[2]), float(m[4]), int(m[3])))

# physics steps from the traj CSV (authoritative physics count)
traj = [l for l in TRAJ.read_text().splitlines()[1:] if l.strip()]
n_phys_steps = len(traj)
phys_dt = 0.005

# ---- steady productive window: first heartbeat with physics>0 -> last physics advance ----
prod = [h for h in hb if h[2] > 0.0]
# find last row where physics is still advancing (drop the tail plateau)
last_adv = len(prod) - 1
while last_adv > 0 and prod[last_adv][2] == prod[last_adv-1][2]:
    last_adv -= 1
a, b = prod[0], prod[last_adv]

d_wall   = b[0] - a[0]
d_grant  = b[1] - a[1]
d_phys_s = b[2] - a[2]
d_phys_steps = round(d_phys_s / phys_dt)
d_target_s = d_grant * TARGET_S_PER_GRANT

grant_rate   = d_grant / d_wall
phys_rate    = d_phys_steps / d_wall
rtf          = d_phys_s / d_wall                      # physics real-time factor
eff_clock    = (d_target_s * F_TARGET_HZ) / d_wall    # effective host cycles/s
duty         = eff_clock / F_FPGA_HZ                   # FPGA-productive fraction
overhead     = 1 - duty
grants_per_step = d_grant / d_phys_steps
wall_per_step   = d_wall / d_phys_steps
wall_per_grant  = d_wall / d_grant
fpga_ms_per_grant  = 1e3 * CYC_PER_GRANT / F_FPGA_HZ  # pure FPGA time per grant
stall_ms_per_grant = 1e3 * wall_per_grant - fpga_ms_per_grant

# gate wall-times (relative to runworkload start = heartbeat t+0)
gate_wall = {}
for w, gi, ps, g in hb:
    if g not in gate_wall and g > 0:
        gate_wall[g] = w

def row(k, v): print(f"  {k:<34} {v}")

print("="*68)
print("  FPGA gate-nav — sim throughput (successful 3-gate run, seed 1000)")
print("="*68)
print(f"\n  Bitstream : RoseTLRocketSaturnMMIOOnlyConfig @ 30 MHz (2026-08-13--14-28-37)")
print(f"  Split     : FPGA firesim1(U250)  <->  IsaacLab garden(GPU)  over LAN")
print(f"\n  Productive window (physics {a[2]:.1f}s -> {b[2]:.1f}s, excl. boot + tail plateau):")
row("wall time",            f"{d_wall} s  ({d_wall/60:.1f} min)")
row("grant-iters",          f"{d_grant}")
row("physics steps",        f"{d_phys_steps}  ({d_phys_s:.2f} physics-s @ {phys_dt*1e3:.0f} ms)")
row("target-time simulated",f"{d_target_s:.1f} s @ {F_TARGET_HZ/1e9:.0f} GHz")
print(f"\n  Rates:")
row("grant-iter throughput",   f"{grant_rate:.2f} grants/s")
row("physics-step throughput", f"{phys_rate:.2f} steps/s   ({wall_per_step*1e3:.0f} ms/step)")
row("physics real-time factor",f"{rtf:.4f} x  (1 physics-s / {1/rtf:.0f} wall-s)")
row("grants per physics step", f"{grants_per_step:.2f}  (FREEZE + 12-sensor exchange)")
print(f"\n  FPGA utilization (the co-sim seam cost):")
row("effective host clock",  f"{eff_clock/1e6:.1f} MHz  of 30 MHz fabric")
row("FPGA-productive duty",  f"{duty*100:.1f} %")
row("co-sim sync overhead",  f"{overhead*100:.1f} %")
row("  FPGA compute / grant",f"{fpga_ms_per_grant:.1f} ms")
row("  sync stall / grant",  f"{stall_ms_per_grant:.1f} ms  (Isaac step + bridge + LAN RTT)")
print(f"\n  Gate wall-clock (from runworkload start):")
for g in sorted(gate_wall):
    row(f"gate {g}/4", f"{gate_wall[g]} s  ({gate_wall[g]/60:.1f} min)")
print("="*68)
