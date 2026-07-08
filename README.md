# RoSÉ: A Hardware-Software Co-Simulation Infrastructure for Pre-Silicon Full-Stack Robotics SoC Evaluation

RoSÉ is an open-source hardware-software co-simulation infrastructure for full-stack,
pre-silicon, hardware-in-the-loop evaluation of robotics SoCs. It closes the loop
between a **physics/environment simulator** and an **SoC simulator**: sensor data flows
from the environment into the SoC to trigger the hardware/software pipeline, and the
computed actuation commands flow back to the environment — with timing and data transfer
synchronized between the two sides. This captures the closed-loop interactions across
environment, algorithm, and hardware that isolated benchmarks miss, enabling
design-space exploration of robotic SoCs without a tape-out.

> This branch (`chipyard-top`) is the modernized infrastructure: **chipyard-as-top**
> (Chisel 6 / chipyard 1.14), OpenAI-Gym-style environments (not only AirSim), a Zephyr
> RTOS software stack, and a new **Spike** functional-simulation tier alongside FireSim.
> For the original ISCA'23 artifact (AirSim + AWS + firesim-as-top), see
> [§ Historical artifact](#historical-isca23-artifact).

## Architecture

Every run is the same two-process shape, coupled over TCP (`localhost:10001`):

```
[ physics side ]  <-- synchronizer protocol -->  [ SoC side ]
 gym env + synchronizer                           one of three fidelity tiers
```

The **physics side is identical across all flows**; you choose an **SoC tier**:

| Tier | Fidelity / speed | Build | Run |
|---|---|---|---|
| **Spike** (ISA sim) | functional, seconds | `soc/src/main/cc/rose_spike/build.sh` | `run_spike_rose.sh` / `run_spike_rose_lockstep.sh` |
| **FireSim metasim** (Verilator) | RTL, slow (~kHz) | `soc/sim/build_metasim.sh` | `run_metasim_selftest.sh`, `run_z*.sh` |
| **FireSim FPGA** (Xilinx U250) | cycle-accurate, fast | `soc/sim/run_buildbitstream.sh` | `run_fpga.sh {infrasetup\|runworkload\|kill}` |

All three consume the same synchronizer, configs, and guest software (Zephyr elf / Linux
image), so a workload developed on Spike runs unchanged on metasim and FPGA.

Repository layout:
- `deploy/` — physics side: the synchronizer (`hephaestus/`), gym environments, configs.
- `soc/src/main/scala/` — RoSE RTL (RoseAdapter, bridge, DMA) injected into chipyard by `soc/setup.sh`.
- `soc/src/main/cc/` — host-side bridge drivers + the Spike bridge (`rose_spike/`).
- `soc/sw/` — guest software: `zephyr-rose` (driver + protocol), `xpu-rt` (apps/samples), `dnn`.
- `soc/sim/` — build/run scripts and the `chipyard` submodule (chipyard-as-top; firesim nested inside).

## Installation

RoSÉ is **chipyard-as-top**. Do **not** run a blanket `git submodule update --init
--recursive` — chipyard curates its own submodules through `build-setup.sh` (which also
builds conda, the RISC-V toolchain, precompiles Scala, and installs firesim + CIRCT).
Initialize per-path:

```bash
git clone https://github.com/ucb-bar/RoSE.git && cd RoSE
git checkout chipyard-top

# 1. Chipyard: checkout the pinned commit, then run ITS OWN setup (~15 min).
git submodule update --init soc/sim/chipyard
( cd soc/sim/chipyard && ./build-setup.sh --skip-marshal )   # drop --skip-marshal for Linux images

# 2. RoSE injections + spike sim.h patch (idempotent), then env/config wiring.
./soc/setup.sh
source rose-setup.sh          # sources chipyard env.sh + firesim; re-source per new shell

# 3. Guest-software submodule. Do NOT use --recursive here either: xpu-rt nests its
#    own hw/chipyard (-> ara -> llvm-project) + IsaacLab, which would explode. Init
#    only zephyr-chipyard-sw:
git submodule update --init soc/sw/xpu-rt
git -C soc/sw/xpu-rt submodule update --init zephyr-chipyard-sw

# 4. Synchronizer Python venv (physics side).
python -m venv deploy/.venv-rose && deploy/.venv-rose/bin/pip install -r deploy/requirements.txt
```

Building the **Zephyr** guest software needs the west workspace + toolchain, all
installed **locally** inside the `zephyr-chipyard-sw` submodule (no external SDK):

```bash
( cd soc/sw/xpu-rt/zephyr-chipyard-sw
  bash   scripts/install_submodules.sh        # west workspace (zephyr_ws) + python deps
  source scripts/install_conda.sh             # conda env 'zephyr' (provides west)
  bash   scripts/install_toolchain_sdk.sh )   # beta Zephyr SDK -> tools-manual/
```

## Running a co-simulation

Two processes in two terminals. **Terminal 1 — the synchronizer** (env chosen by
`deploy/config/config_deploy_gym.yaml`, default `PatternEnv-v0`):

```bash
cd deploy/hephaestus
ROSE_DIR=$(git rev-parse --show-toplevel) ../.venv-rose/bin/python run_sync_only.py
```

**Terminal 2 — one SoC tier:**

```bash
# Spike (functional). Build once, then run a guest elf:
soc/src/main/cc/rose_spike/build.sh
soc/sim/build_zephyr_rose.sh selftest
soc/sim/run_spike_rose_lockstep.sh soc/sim/zephyr_rose_builds/selftest/zephyr/zephyr.elf 1

# FireSim metasim (Verilator):
soc/sim/build_metasim.sh
soc/sim/run_metasim_selftest.sh

# FireSim FPGA (U250): build a bitstream, then drive it with the FireSim manager:
soc/sim/run_buildbitstream.sh
soc/sim/run_fpga.sh infrasetup && soc/sim/run_fpga.sh runworkload
```

Expected: the guest boots, the bridge connects (`connected to localhost:10001`), and a
sample prints e.g. `ROSE selftest: dma=PASS reqrsp=PASS => PASS`.

The **Spike tier** (both the fast `--extlib` plugin and the cycle-lockstep harness) is
documented in depth in [`soc/src/main/cc/rose_spike/README.md`](soc/src/main/cc/rose_spike/README.md).

## Worked example
[`ROSE_DRONE_MPC_DEMO.md`](ROSE_DRONE_MPC_DEMO.md) — a full closed-loop example: a
PyBullet quadrotor flown by a TinyMPC controller running on a simulated SoC over the
RoSE bridge. Includes the component map, build/run steps, validation criteria, and the
HW/SW timing result (diverges at 10 MHz, hovers at 1 GHz).

## Environments and workloads
- **Environments** live in `deploy/hephaestus/envs/` and are registered in
  `register_envs.py`; select one via `deploy/config/config_deploy_gym.yaml`. Options
  include `PatternEnv` (bridge validation), `PyBulletDroneEnv`, `AirSimEnv`,
  `MiddleBuryEnv`, `InvertedPendulum`, and `LQR`.
- **Guest software:** the Zephyr `rose` driver + `subsys/rose` protocol layer
  (`soc/sw/zephyr-rose`) with test samples in
  `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose/` (reqrsp / dma / protocol / selftest);
  plus baremetal packet tests and the ONNX DNN controllers used in the paper.

## Historical ISCA'23 artifact

The original artifact — AirSim on Unreal Engine + AWS EC2 + firesim-as-top, reproducing
the paper's figures — is preserved on the pre-migration `main` branch. Its AWS/AirSim
install and `deploy/scripts/*` figure-reproduction flow differ from the modernized flow
above.

## Citing RoSÉ

```
@inproceedings{rose-isca,
  title={RoS{\'E}: A Hardware-Software Co-Simulation Infrastructure Enabling Pre-Silicon Full-Stack Robotics SoC Evaluation},
  author={Nikiforov, Dima and Dong, Shengjun Kris and Zhang, Chengyi Lux and Kim, Seah and Nikolic, Borivoje and Shao, Yakun Sophia},
  booktitle={Proceedings of the 50th Annual International Symposium on Computer Architecture},
  pages={1--15},
  year={2023}
}
```
