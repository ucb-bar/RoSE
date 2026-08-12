# Reproduce: the RoSE co-sim throughput ablation

Step-by-step guide to reproduce, **from a fresh git clone**, the ablation over every
simulation-throughput lever in the RoSE ↔ IsaacLab / FireSim co-sim: which knobs
actually move throughput, and by how much.

- **Reproducible bundle:** this directory (`experiments/rose_throughput_ablation/`) — harness + analysis + results CSV + self-contained HTML report.
- **Shareable artifact:** [`report.html`](report.html) — open in a browser (self-contained; no external deps).
- **Companion memory / prior captures:** `rose-cosim-hang-soc-ruled-out`, `rose-fpga-firesim-bringup`; the raw prior logs are cited per-row in `data/measured_prior.json`.

---

## 0. What the ablation covers

Six independent throughput levers, each a concrete toggle (no rebuild except the datapath bitstream):

| # | Lever | Toggle | Old → New |
|---|-------|--------|-----------|
| 1 | Sync granularity | `ROSE_FIRESIM_STEP` | 1M … 500M cyc/grant |
| 2 | Synchronizer socket poll | `ROSE_SYNC_RECV_TIMEOUT` | `0.1` → `0.001` |
| 3 | Lazy camera render | `ROSE_RENDER_HZ` (+`ROSE_ISAAC_CAMERA`) | `0` → `10` |
| 4 | Datapath | bitstream | MMIO-Saturn → DMA-Saturn |
| 5 | Sync mode | script | `minimal_sync` (free) vs `run_sync_only` (barrier) |
| 6 | Environment | `ROSE_GYM_ENV` | PatternEnv-v0 → WarehouseThrustEnv-v0 |

Designed as a proper ablation: **baseline (all-old) → single-lever flips → all-new**, plus the granularity sweep — 18 configs, not a full cross-product.

---

## 1. Prerequisites

Split by which rows they gate. The **socket-fix axis runs with only Prereq A+B** (no FPGA, no GPU).

**Prereq A — the deploy Python venv** (`deploy/.venv-rose`, or any Python 3.10+ with the RoSE synchronizer deps). Only the stdlib + the tracked `deploy/hephaestus` modules are used for the barrier measurement.

**Prereq B — `rose_spike_sim`** (the functional RoSE bridge) + a camera-free guest ELF.
- Binary: `soc/sim/rose_spike_sim` (+ `librose_spike.so`), built from `soc/src/main/cc/rose_spike/build.sh` (heavy; gitignored, regenerable). ~176 MB reference binary.
- Guest: `soc/sim/zephyr_rose_builds/reqrsp/zephyr/zephyr.elf` (the `samples/rose/bench` reqrsp mode). Any camera-free guest works — the barrier round-trip is guest-independent at `firesim_step=1000`.

**Prereq C — U250 FPGA + Saturn+RoSE bitstreams** (gates the on-silicon effective-MHz rows only).
- MMIO: `RoseTLRocketSaturnMMIOOnlyConfig` / hwdb `alveo_u250_firesim-rocket-saturn-with-rose-fast`.
- DMA: `RoseTLRocketSaturnDMAMMIOOnlyConfig` (30 MHz built; 60 MHz building).
- These rows are **scripted, not run** in this capture (see §5).

**Prereq D — IsaacLab + `env_isaaclab`** (gates the render-axis env.step timing only). See `docs/REPRODUCE_ROSE_NAV_COSIM.md` Prereq A. **Deferred** here (shared GPU busy with a live capstone flight); the render numbers are reused measured-prior.

---

## 2. Clone + submodules

```bash
git clone <RoSE-remote> RoSE && cd RoSE
git checkout rose-2-dev
git submodule update --init --recursive soc/sw/xpu-rt   # for the guest, if rebuilding
```

The committed socket / render / DMA toggles this ablation exercises: `72fa33e` (socket reorder + `ROSE_SYNC_RECV_TIMEOUT`), `b4c946c` (lazy render), `d5778bd` (DMA datapath). This bundle **only reads** those; it does not modify them.

---

## 3. The one command (runnable now, no FPGA/GPU)

```bash
export ROSE_ROOT=$PWD
bash experiments/rose_throughput_ablation/run_ablation.sh --now
```

This:
1. Runs the socket-fix barrier axis on the Spike bridge — both `ROSE_SYNC_RECV_TIMEOUT`
   regimes (`0.001` new, `0.1` old) via `lib/spike_barrier.sh`, which drives the **real
   tracked** `gym_synchronizer` + `socket_thread` and timestamps every grant→RSP_STALL
   round-trip (`lib/seam_instr_run2.py`, non-invasive — tracked files untouched).
2. Runs `lib/analyze.py` to assemble `results/ablation_results.csv` + `results/waterfall.json`,
   folding the measured-now barrier into the effective-MHz model and the measured-prior /
   computed rows (every value carries a `source`).

> **Sandbox note.** The co-sim needs loopback TCP between the synchronizer and the bridge.
> If your shell sandbox blocks loopback sockets, run detached / with the sandbox relaxed.

Paths (overridable env): `ROSE_PY`, `ROSE_SPIKE_BIN`, `ROSE_GUEST_ELF`, `ROSE_PATTERN_YAML`.

---

## 4. Expected output

```
== LEVER 2: socket-fix barrier axis  (Spike bridge, no FPGA/GPU) ==
[spike_barrier] tag=newsock recv_timeout=0.001s ...
SUMMARY newsock 0.001 16.40 37
SUMMARY oldsock 0.1  115.44 37
== ANALYZE -> results/ablation_results.csv + results/waterfall.json ==
MODEL SELF-CHECK (computed vs measured-prior FPGA anchors):
  @5M  old socket: model 24.2 MHz  vs measured 24.6 MHz
  @10M old socket: model 34.5 MHz  vs measured 34.3 MHz
spike barrier NEW socket: 16.40 ms/grant  ->  7.0x
wrote results/ablation_results.csv  (18 configs)
```

- **Socket fix: ~115 → ~16 ms/grant (7.0×)**, send term 105 → 6 ms. Numbers within a few ms
  of the reference are a pass (localhost jitter). Component CSVs land in `run_out/seam_*.csv`.
- The model self-check must reproduce the measured FPGA anchors (24.6 / 34.3 MHz) to ~1%.
- Open `report.html` for the waterfall, granularity curve, barrier decomposition, and full table.

---

## 5. The FPGA / GPU rows (scripted, not run)

Print the exact recipes without touching the FPGA/GPU:

```bash
bash experiments/rose_throughput_ablation/run_ablation.sh --fpga     # granularity sweep + datapath recipe
bash experiments/rose_throughput_ablation/run_ablation.sh --render   # env.step recipe (auto-defers if GPU busy)
```

To actually populate the deferred columns later, on a free machine:

```bash
# on firesim1/garden when the FPGA is free:
ROSE_ALLOW_FPGA=1 bash experiments/rose_throughput_ablation/run_ablation.sh --fpga
# on a free GPU box:
ISAAC_PY=/path/to/env_isaaclab/bin/python ROSE_ALLOW_GPU=1 \
  bash experiments/rose_throughput_ablation/run_ablation.sh --render
```

`analyze.py` will automatically prefer any freshly-measured CSVs over the prior/computed values.

---

## 6. Honesty / provenance

Every row in `results/ablation_results.csv` carries a `source`:

- **measured-now** — the socket-fix barrier (regenerated by `--now` on this machine).
- **measured-prior** — free ceiling 59.9 MHz, barrier @5M/@10M 24.6/34.3 MHz (FPGA recon),
  render 31.0/8.1/6.06 ms (direct GPU timing), DMA byte-exact. Reused, not re-run; each cites
  its log in `data/measured_prior.json`.
- **computed(FPGA-run-later)** — the effective-MHz model
  `wall = 13.8 s + (tcyc/step)·per_grant + ticks·env_step`, validated against the three
  measured anchors (≤1% error), used for the granularity sweep, socket-fixed FPGA, and DMA rows.

**Headline:** the big throughput wins are host-side (socket poll, 7.0×) and env-side (lazy
render, 3.9×). The FPGA datapath (MMIO→DMA) moves the `num_bytes=0` per-grant barrier by ~0 ms —
DMA's value is camera-frame payload bandwidth, a different axis than co-sim throughput.
