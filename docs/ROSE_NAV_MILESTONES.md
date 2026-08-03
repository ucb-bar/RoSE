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
- **Waypoint-flight caveat (tracked for estimator refinement):** the forward setpoint
  overshoots slightly because the front/back sensor's *nearest* zone reads the side-wall corner
  in a narrow corridor, so longitudinal position lags. Fix: use the perpendicular center zone
  for longitudinal wall fusion (lateral min-zone is already clean).
