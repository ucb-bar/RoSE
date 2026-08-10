# IsaacLab overlay for the RoSE nav co-sim

This directory captures **everything custom about the Isaac side** of the RoSE ↔
IsaacLab spike co-sim that is *not already tracked elsewhere in the repo*, plus the
exact upstream versions it must overlay onto. Goal: a fresh machine can rebuild the
Isaac prereq (`env_isaaclab`) deterministically.

## Where the custom content actually lives

The install at `/scratch2/dima/IsaacLab` is a **clean git checkout** of
`isaac-sim/IsaacLab` (see `PINNED_VERSION.txt`). Our custom co-sim content is split:

| Custom content | Where it lives | Tracked by |
|----------------|----------------|------------|
| Warehouse gate task, crazyflie thrust/vision/sensor envs, gym `Isaac-Drone-Warehouse-*` registration, procedural gates, forest_trail sensors | `soc/sw/xpu-rt/sims/isaaclab_tasks/` (imported as the `sims.isaaclab_tasks` package via `PYTHONPATH`) | **xpu-rt submodule** (already tracked) |
| RoSE bridge envs incl. `WarehouseThrustEnv-v0`, `run_sync_only.py`, config yaml | `deploy/hephaestus/` + `deploy/config/` | **RoSE repo** (already tracked) |
| Committed small forest USDA assets | `soc/sw/xpu-rt/sims/isaaclab_tasks/forest_trail/assets/` | **xpu-rt submodule** (already tracked) |
| `isaaclab_contrib` multirotor thruster-mapping fix | IsaacLab install tree (`source/isaaclab_contrib/...`) | **this overlay** (`patches/`) |
| `quadcopter_fpv.py` demo | IsaacLab install tree (`scripts/demos/`) | **this overlay** (`scripts/demos/`) |

So the **only untracked delta living inside the IsaacLab install tree** is the
2-file overlay in `patches/` + `scripts/demos/` here. The tasks themselves are used
straight from this repo via `PYTHONPATH` — they are never copied into IsaacLab.

## Contents

- `PINNED_VERSION.txt` — the IsaacLab commit/branch/VERSION + Isaac Sim / torch / python versions to reproduce.
- `pip_freeze_isaaclab.txt` — authoritative `pip freeze` of the live `env_isaaclab` (260 pkgs; Isaac Sim from PyPI wheels).
- `environment_isaaclab.yml` — `conda env export --no-builds` of `env_isaaclab` (conda-level + pip section).
- `patches/isaaclab_contrib-multirotor-thruster-mapping.patch` — the one IsaacLab-core delta in the install.
- `scripts/demos/quadcopter_fpv.py` — untracked demo captured for completeness.
- `apply_overlay.sh` — apply the 2-file overlay onto a stock IsaacLab checkout.
- `check_registration.py` — **GPU-free** headless check that the co-sim gym ids register.

## Reproduce

Use `../setup_isaaclab.sh` (parameterized). It clones/checks out the pinned
IsaacLab, creates `env_isaaclab` from `environment_isaaclab.yml` / `pip_freeze`,
installs Isaac Sim wheels + IsaacLab (editable), applies this overlay, then runs
`check_registration.py`. See `docs/REPRODUCE_ISAACLAB.md` for the full runbook and
the honest coverage note (what was validated vs deferred to a GPU box).

## Assets / provenance

- **Warehouse scene + props** (`full_warehouse.usd`, pallets/crates/cones, …):
  built-in Isaac Sim content resolved via `ISAAC_NUCLEUS_DIR` — ships with the Isaac
  Sim install (the `isaacsim-asset` wheel / nucleus). Not bundled here; not custom.
- **Gates:** procedural (`sim_utils.CuboidCfg`) — no external asset.
- **Obstacle props** (`mdp_obstacles.py` → `/scratch/agustin/.../rb_props`): a
  collaborator external path, **off the co-sim path** (the co-sim runs `ROSE_WH_OBST=0`
  and TASK_ID `...WithSensors-v0`, not the `-Coll*/-Crowded` variants). Only needed for
  the crowded-obstacle course variants, which the co-sim does not use.
- **forest_trail USDs:** small procedural USDAs committed in the xpu-rt submodule;
  the PBR sapling `.usdc` is fetched by `forest_trail/assets/download_trees.py`.
  forest_trail is not exercised by the warehouse gate co-sim (only its `sensors.py`
  is imported).
