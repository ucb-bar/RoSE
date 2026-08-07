# Host-only quadrotor + TinyMPC prototype (Stage-2 yaw debugging & tuning)

Fast (ms/run) pure-numpy replicas of the on-SoC control problem — no spike, no Isaac bridge — used
to find/fix the Stage-2 yaw root cause and to tune the yaw loop before paying co-sim cost.

- `host_quad_proto.py`   — 6-DOF quad (exact MotorThrustAction wrench) + host TinyMPC (Kinf+clamp),
                           GT state; test (yaw_rate, forward_speed) commands.
- `host_yaw_rootcause.py`— isolates the yaw-mixing model/plant mismatch (before/after Kinf).
- `host_weave_validate.py`— validates the corrected params track a sign-changing weave yaw profile.
- `gen_strongyaw_params.py`— regenerates a TinyMPC problem_data .hpp (numpy rho-augmented DARE,
                           validated <2e-5 vs committed) with corrected yaw mixing + yaw weights.
                           FIX_YAW_MIXING=1 flips Bdyn yaw cols 2,3 to match the physical spin.

Params live in the Accelerated-TinyMPC submodule examples/problem_data/. Plant/model constants
(mass, km/kf, spin, thrust gain) mirror mdp_motor_thrust_action.py / crazyflie_mpc_env.py.
