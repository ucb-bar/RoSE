# Reproduce: the IsaacLab prereq for the RoSE nav co-sim (Prereq A)

The heavy Isaac side of the RoSE ↔ IsaacLab spike co-sim: a pinned **IsaacLab**
checkout + an **`env_isaaclab`** conda env with **Isaac Sim**, plus the RoSE custom
overlay and gym-env registration. Companion to
[`REPRODUCE_ROSE_NAV_COSIM.md`](REPRODUCE_ROSE_NAV_COSIM.md) (the co-sim runbook) and
[`ROSE_ISAAC_COSIM_FLOW.md`](ROSE_ISAAC_COSIM_FLOW.md) (the flow).

**Reproducible bundle:** [`../experiments/rose_nav_cosim/isaaclab_overlay/`](../experiments/rose_nav_cosim/isaaclab_overlay/)
+ [`../experiments/rose_nav_cosim/setup_isaaclab.sh`](../experiments/rose_nav_cosim/setup_isaaclab.sh).

---

## 1. What is custom, and where it lives

The reference install `/scratch2/dima/IsaacLab` is a **clean git checkout** of
`isaac-sim/IsaacLab` — not a fork, no tangled core edits. Verified:

```
remote  git@github.com:isaac-sim/IsaacLab.git
branch  main
commit  4df6560e187f2cc66685b41b21b259f4485d0c22
VERSION 2.3.2
```

Its **entire** working-tree delta vs upstream is two files:
- `source/isaaclab_contrib/.../multirotor/multirotor.py` — a 2-line thruster-mapping fix (modified, tracked-upstream).
- `scripts/demos/quadcopter_fpv.py` — an untracked demo.

Both are captured in the overlay (`isaaclab_overlay/patches/`,
`isaaclab_overlay/scripts/demos/`). **Everything else custom is used from this repo
via `PYTHONPATH`, not from the IsaacLab tree:**

| Custom content | Location | Tracked by |
|---|---|---|
| Warehouse gate task, crazyflie thrust/vision/sensor envs, procedural gates, gym `Isaac-Drone-Warehouse-*` registration, forest_trail sensors + small USDAs | `soc/sw/xpu-rt/sims/isaaclab_tasks/` (imported as `sims.isaaclab_tasks`) | **xpu-rt submodule** |
| RoSE bridge envs (`WarehouseThrustEnv-v0`, …), `run_sync_only.py`, config yaml | `deploy/hephaestus/`, `deploy/config/` | **RoSE repo** |
| isaaclab_contrib multirotor fix + `quadcopter_fpv.py` demo | IsaacLab install tree | **this overlay** |

The co-sim's `WarehouseThrustEnv-v0` (in `deploy/hephaestus/envs/warehouse_fused_nav/warehouse_thrust_env.py`)
does `sys.path.insert` for `soc/sw/xpu-rt` + `IsaacLab/source/*`, then imports
`sims.isaaclab_tasks.warehouse_nav.config.crazyflie` and `gym.make(
"Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0")`.

> **Note (portability):** `warehouse_thrust_env.py` currently hardcodes
> `_XPURT = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt"` and the
> `/scratch2/dima/IsaacLab/source/*` paths. On a differently-located clone these must
> be edited (or set via env). Flagged, not yet parameterized.

---

## 2. Versions to reproduce (pinned)

See [`isaaclab_overlay/PINNED_VERSION.txt`](../experiments/rose_nav_cosim/isaaclab_overlay/PINNED_VERSION.txt).
Key: IsaacLab `4df6560e187…` (v2.3.2), Isaac Sim **5.1.0.0** (PyPI wheels, *not* an
Omniverse-launcher install), Python **3.11.15**, torch **2.7.0+cu128**, gymnasium
1.2.1, rsl-rl-lib 5.0.1, warp-lang 1.12.1.

Authoritative package manifests (generated from the live env):
- [`isaaclab_overlay/pip_freeze_isaaclab.txt`](../experiments/rose_nav_cosim/isaaclab_overlay/pip_freeze_isaaclab.txt) — 260 pkgs; the IsaacLab subprojects appear as `-e git+…@4df6560…` and Isaac Sim as `isaacsim==5.1.0.0` + all `isaacsim-*` extensions.
- [`isaaclab_overlay/environment_isaaclab.yml`](../experiments/rose_nav_cosim/isaaclab_overlay/environment_isaaclab.yml) — `conda env export --no-builds`.

Because Isaac Sim is installed from PyPI wheels, `pip install isaacsim==5.1.0.0 …`
reproduces it (multi-GB) — no Omniverse Launcher / nucleus server required.

---

## 3. Automated setup

```bash
SCRATCH_BASE=/scratch2/$USER \
  bash experiments/rose_nav_cosim/setup_isaaclab.sh
```

Steps (each printed, stops on error): checks conda → clones/checks out IsaacLab at
the pinned commit → creates `env_isaaclab` from `environment_isaaclab.yml` →
`pip install -r pip_freeze_isaaclab.txt` (Isaac Sim wheels + IsaacLab editable) →
`apply_overlay.sh` (multirotor patch + demo) → runs the GPU-free registration check.

Parameters: `ISAACLAB_DIR`, `ENV_NAME`, `CONDA_BASE`, `SCRATCH_BASE`, `ROSE_DIR`,
`PIP_ISAACSIM`, `FORCE` (see the script header). It will not clobber an existing env
unless `FORCE=1`.

> Torch `cu128` and the Isaac Sim wheels may need `--extra-index-url`
> (`https://download.pytorch.org/whl/cu128` and NVIDIA's Isaac Sim wheel index). If a
> bare `pip install -r` cannot resolve them, run those two with their index URLs
> first, then re-run. Keep all installs on `/scratch*` — never a full root fs.

Overlay only (onto an existing IsaacLab checkout):
```bash
ISAACLAB_DIR=/path/to/IsaacLab bash experiments/rose_nav_cosim/isaaclab_overlay/apply_overlay.sh
```

---

## 4. Validation

**GPU-free registration check — VALIDATED 2026-08-10.** In the live `env_isaaclab`:

```bash
/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python \
  experiments/rose_nav_cosim/isaaclab_overlay/check_registration.py
```

This imports only the registration modules (they hard-import just `gymnasium`; the
`env_cfg` entry points are lazy strings) and asserts the gym ids exist. It never calls
`gym.make()`/`AppLauncher`, so it opens **no** Omniverse app and touches **no** GPU —
safe to run alongside a live Isaac workload. Observed result:

```
=== task ids (sims.isaaclab_tasks, xpu-rt submodule) ===
  [OK] Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0   <- co-sim TASK_ID
  [OK] Isaac-Drone-Warehouse-Nav-Crazyflie-v0
  [OK] Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-Crowded-v0
=== bridge ids (deploy/hephaestus/register_envs) ===
  [OK] WarehouseThrustEnv-v0
  [OK] WarehouseFusedNavBridgeEnv-v0
  [OK] FusedNavProbeEnv-v0
Total Isaac-Drone-Warehouse* ids registered: 9
RESULT: PASS
```

**Manifest completeness — VALIDATED.** `pip_freeze_isaaclab.txt` (260 pkgs) and
`environment_isaaclab.yml` were generated directly from the live `env_isaaclab`; the
IsaacLab editable subprojects and `isaacsim==5.1.0.0` (+ all `isaacsim-*` extensions)
are present and pin to the same commit as the checkout.

## 5. Coverage — validated vs deferred (honest note)

**Validated (no GPU, on this shared box):**
- IsaacLab install is a clean checkout; upstream commit/version pinned.
- The custom delta is fully captured (2-file overlay here; tasks already tracked in
  the xpu-rt submodule + `deploy/`).
- The co-sim gym ids **register** in `env_isaaclab` (check above), incl. the exact
  co-sim `TASK_ID` and `WarehouseThrustEnv-v0`.
- The env manifests are complete and pin Isaac Sim 5.1.0.0 + IsaacLab @4df6560.

**Deferred to a GPU box (NOT run here):**
- A **fresh `setup_isaaclab.sh` run from scratch** — creating a new conda env and
  pulling the multi-GB Isaac Sim/torch wheels — was not executed on this shared
  machine (disk `/` ~full; must stay on `/scratch*`, and the box's GPU was in active
  use by a protected Isaac workload). The script replays the captured pinned
  manifests, but a clean-machine run is unverified.
- **`gym.make()` / any Isaac Sim GPU session** (constructing the env, loading the
  warehouse USD, cameras, dynamics) was intentionally NOT run, to avoid competing
  with the live Isaac instances for the shared GPU.
- The **end-to-end live co-sim** run is deferred for the same reason (see
  `REPRODUCE_ROSE_NAV_COSIM.md` §9).

**Asset provenance:** warehouse scene + props resolve via `ISAAC_NUCLEUS_DIR`
(built-in Isaac Sim content, ships with the install); gates are procedural cuboids;
the crowded-obstacle prop path (`/scratch/agustin/.../rb_props`, `mdp_obstacles.py`)
is a collaborator external and is **off the co-sim path** (`ROSE_WH_OBST=0`, TASK_ID
`…WithSensors-v0`, not the `-Coll*/-Crowded` variants). forest_trail is not exercised
by the warehouse co-sim (only its `sensors.py` is imported).
