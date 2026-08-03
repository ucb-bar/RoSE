# RoSE Drone — Milestone Videos

Record of notable co-sim milestones captured as video. Each was rendered from the Isaac Sim
chase camera during a real RoSE co-sim run (physics in Isaac Sim ⇄ flight controller on a
Spike-simulated SoC over the RoSE bridge) and published as a private claude.ai artifact.

| Milestone | Description | Artifact | Source frames/traj |
|---|---|---|---|
| **Near-flip & recovery** | Hard velocity+rate initial condition drives a 48.3° tilt (near-flip) at t=7.65 s; the sensor-based TinyMPC controller recovers to level. The `ic_velocity_s0` stress cell. | https://claude.ai/code/artifact/799e5d2d-25e8-4b27-9a90-93e306667c71 | `nearflip_traj.csv`, `nearflip.mp4` |
| **Waypoint flight by feel** | Corridor waypoint navigation with NO GPS/mocap: 4× horizontal VL53L5CX ToF give obstacle-relative position (`fuse_walls` in the EKF), so TinyMPC holds the corridor center + altitude and advances ~1.3 m forward. `IsaacCrazyflieMultiSensorEnv`, `ROSE_MAZE=hallway`. | https://claude.ai/code/artifact/a8a3ab1f-0043-46e2-be87-6fe2063b75d7 | `nav_traj.csv`, `nav_hallway.mp4` |

Notes:
- Videos render at real-time from 200 Hz frame capture (`ROSE_ISAAC_CAMERA=1` +
  `ROSE_ISAAC_FRAMEDIR`). The mp4/CSV sources are written to the run's scratch dir (session-
  temporary); the artifacts above are the durable record. Re-capture any run by setting those
  env vars on the sync process.
- **Waypoint-flight refinement (RESOLVED):** the first waypoint run overshot (true x→1.4 vs
  setpoint 1.0) because the front/back sensor's *nearest* zone read the side-wall corner in a
  narrow corridor, so the longitudinal estimate under-read (guest 0.79 vs true 1.32). Fix: the
  `ucbbar,rose-tof-zone` driver now also exposes the perpendicular **center (bore) zone**
  (`ROSE_SENSOR_CHAN_TOF_ZONE_CENTER`), and `fuse_walls` uses it for position. Re-run: the guest
  estimate now matches truth (0.856 vs 0.858 at the setpoint) and x approaches 1.0 monotonically
  with **no overshoot** — a clean waypoint arrival. (The near-flip video predates this; the
  corridor video shows the pre-fix forward flight, which is still a valid nav demo.)
