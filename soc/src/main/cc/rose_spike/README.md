# RoSE co-simulation on Spike

Run a RoSE closed-loop co-simulation (physics/environment ⇄ SoC) with **Spike** as
the SoC simulator — a fast functional tier alongside the FireSim metasim and FPGA
flows, sharing the *same* synchronizer, configs, and Zephyr guest software.

Two Spike tiers are provided:

| Tier | Binary | Sync fidelity | Cores | Use for |
|---|---|---|---|---|
| **Plugin** | stock `spike --extlib=librose_spike.so` | transaction ordering; grants acked immediately | single | fastest inner-loop driver/app dev |
| **Lockstep** | `rose_spike_sim` (owns the step loop) | exact `step_size` cycles/step, two-sided barrier | multicore | timing-faithful rate + multicore |

Design notes: `../../../../../ROSE_SPIKE_BRIDGE_PLAN.md` and
`../../../../../ROSE_SPIKE_LOCKSTEP_PLAN.md`.

---

## 1. One-time setup (from a fresh clone)

```bash
# 0. Clone + submodules (chipyard, xpu-rt -> zephyr-chipyard-sw -> zephyr_ws)
git submodule update --init --recursive

# 1. RoSE injections + the spike sim.h step() patch (idempotent)
./soc/setup.sh

# 2. Chipyard toolchains — builds spike's libriscv/libfesvr/... that the bridge links.
#    (Standard chipyard build-setup; env.sh must exist under soc/sim/chipyard.)
#    If spike libs are missing later, soc/sim/build_spike.sh rebuilds just spike.

# 3. Zephyr toolchain — all LOCAL to the zephyr-chipyard-sw submodule (no external SDK):
cd soc/sw/xpu-rt/zephyr-chipyard-sw
source scripts/install_conda.sh           # conda env 'zephyr' (provides west) -> tools/miniforge3
bash   scripts/install_toolchain_sdk.sh   # beta Zephyr SDK -> tools-manual/zephyr-sdk-1.0.0-beta1
cd -

# 4. Synchronizer Python venv (physics side)
python -m venv deploy/.venv-rose && deploy/.venv-rose/bin/pip install -r deploy/requirements.txt
```

## 2. Build

```bash
# RoSE bridge: both librose_spike.so (plugin) + rose_spike_sim (lockstep harness)
soc/src/main/cc/rose_spike/build.sh            # or: build.sh plugin | build.sh harness

# Zephyr guest apps (board spike_riscv64) -> soc/sim/zephyr_rose_builds/<app>/zephyr/zephyr.elf
soc/sim/build_zephyr_rose.sh                    # all: reqrsp dma protocol selftest
soc/sim/build_zephyr_rose.sh selftest          # or a subset
```

All scripts derive their own repo root, so the checkout can live anywhere.

## 3. Run (two processes, over TCP :10001)

**Terminal 1 — physics synchronizer** (env selected by `deploy/config/config_deploy_gym.yaml`,
default `PatternEnv-v0`):
```bash
cd deploy/hephaestus
ROSE_DIR=$(git -C ../.. rev-parse --show-toplevel) ../.venv-rose/bin/python run_sync_only.py
# waits: "listening on localhost:10001 — waiting for 1 bridge connection(s)..."
```

**Terminal 2 — Spike** (pick a tier):
```bash
# Lockstep harness (two-sided barrier, multicore):  [elf] [nprocs]
soc/sim/run_spike_rose_lockstep.sh soc/sim/zephyr_rose_builds/selftest/zephyr/zephyr.elf 1

# — or — fast plugin (single-core):                 [elf]
soc/sim/run_spike_rose.sh          soc/sim/zephyr_rose_builds/selftest/zephyr/zephyr.elf
```

Expected: Zephyr boots, the bridge connects (`[rose_sync] connected to localhost:10001`),
and e.g. `ROSE selftest: dma=PASS reqrsp=PASS => PASS`.

Set `ROSE_SPIKE_DEBUG=1` for per-event tracing (routes, DMA writes, IRQ, grant budget).

---

## Files

| Path | Role |
|---|---|
| `rose_sync_client.{cc,h}` | synchronizer-protocol client (socket + framing + `CS_*`) |
| `rose_spike/rose_spike_device.cc` | passive `--extlib` MMIO plugin (`librose_spike.so`) |
| `rose_spike/rose_spike_sim.cc` | lockstep harness (`rose_spike_sim`) — subclasses `sim_t`, owns `idle()` |
| `rose_spike/rose_spike_sim_stepaccess.patch` | minimal spike patch (`step()` → protected), applied by `setup.sh` |
| `rose_spike/build.sh` | builds both flavors |
| `soc/sim/build_zephyr_rose.sh` | builds the `samples/rose` guest elfs |
| `soc/sim/run_spike_rose{,_lockstep}.sh` | launch a guest elf on each tier |
| `soc/sw/zephyr-rose/` | Zephyr rose driver + `subsys/rose` protocol layer |
| `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose/` | guest test apps (reqrsp/dma/protocol/selftest) |

## Notes
- Build artifacts (`librose_spike.so`, `rose_spike_sim`, `zephyr_rose_builds/`) are
  gitignored — rebuild via the scripts above.
- No single-command launcher yet: the synchronizer and Spike run in two terminals.
- The lockstep budget is gated on the global CLINT `mtime`, so it stays correct with
  multiple harts (`-p N`); the plugin tier is single-core only.
