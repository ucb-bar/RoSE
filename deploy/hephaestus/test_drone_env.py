"""Standalone smoke test for PyBulletDroneEnv-v0 (no metasim / bridge).

Validates that the RoSE drone env:
  - registers + gym.make()s,
  - returns the IMU-friendly observation dict with correct fields/shapes,
  - produces physically sane IMU while hovering (accel ~ [0,0,g], gyro ~ 0),
  - serializes obs fields the way the synchronizer forwards them (uint32 words).
"""

import numpy as np
import gymnasium as gym

import register_envs  # noqa: F401  (registers PyBulletDroneEnv-v0)

env = gym.make("PyBulletDroneEnv-v0", render_mode="rgb_array")
obs, info = env.reset()

print("obs fields:", {k: (v.shape, v.dtype) for k, v in obs.items()})
for field in ("imu_accel", "imu_gyro", "state", "pos", "quat"):
    assert field in obs, f"missing obs field {field}"

# step with hover (no action -> env defaults to hover RPM)
last = obs
for _ in range(240):  # ~1 s at 240 Hz
    last, rew, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))

accel = np.asarray(last["imu_accel"], dtype=np.float64)
gyro = np.asarray(last["imu_gyro"], dtype=np.float64)
pos = np.asarray(last["pos"], dtype=np.float64)
print(f"hover accel (m/s^2): {accel}")
print(f"hover gyro  (rad/s): {gyro}")
print(f"hover pos   (m):     {pos}")

# how the synchronizer serializes imu_accel over the bridge (3 float32 -> 3 uint32 words)
words = np.frombuffer(np.asarray(last["imu_accel"], dtype=np.float32).tobytes(), dtype=np.uint32)
print(f"imu_accel as bridge words: {[hex(int(w)) for w in words]} (float bits)")

# physical sanity while hovering: |accel| ~ g, gyro small, drone roughly in place
g = 9.8
accel_mag = float(np.linalg.norm(accel))
gyro_mag = float(np.linalg.norm(gyro))
ok = (abs(accel_mag - g) < 3.0) and (gyro_mag < 1.0) and (len(words) == 3)
print(f"accel_mag={accel_mag:.3f} (~{g}), gyro_mag={gyro_mag:.4f}, words={len(words)}")
print("DRONE_ENV_SMOKE:", "PASS" if ok else "FAIL")
env.close()
