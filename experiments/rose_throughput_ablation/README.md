# RoSE co-sim throughput ablation

A fully-automated, reproducible ablation over every **simulation-throughput** architectural
lever in the RoSE ↔ IsaacLab / FireSim SoC co-sim. Answers, with sourced numbers: *which knobs
actually move throughput, and by how much?*

**Headline:** the two biggest wins are **host-side** (synchronizer socket poll, **7.0×** on the
per-grant barrier — measured now) and **environment-side** (lazy camera render, **3.9×** on Isaac
`env.step`). The FPGA **datapath** (MMIO→DMA) moves the `num_bytes=0` per-grant barrier by **~0 ms** —
its value is on the orthogonal camera-payload axis, not co-sim throughput.

## Layout

```
run_ablation.sh          # entry point: --now | --render | --fpga | --analyze | --all
report.html              # self-contained, theme-aware HTML report (the shareable artifact)
REPRODUCE.md             # fresh-clone runbook
lib/
  spike_barrier.sh       # one socket-fix barrier point on the Spike bridge (runnable now)
  seam_instr_run2.py     # non-invasive per-grant timestamping (real tracked synchronizer)
  analyze.py             # assembles the ablation CSV + waterfall JSON from model + measured data
data/
  measured_prior.json    # anchors + model params + measured-prior values, each with provenance
results/
  ablation_results.csv   # THE ablation matrix (18 configs; every row labeled by source)
  waterfall.json         # run-level cost waterfall + granularity curve (drives report.html)
run_out/                 # raw measured-now evidence (seam_*.csv, sync/spike logs)
```

## The levers (6 axes)

| # | Lever | Toggle | Substrate | Status |
|---|-------|--------|-----------|--------|
| 1 | Sync granularity | `ROSE_FIRESIM_STEP` 1M…500M | FPGA eff-MHz | 5M/10M measured, rest computed+scripted |
| 2 | Socket poll fix | `ROSE_SYNC_RECV_TIMEOUT` 0.1↔0.001 | Spike barrier | **measured now** |
| 3 | Lazy camera render | `ROSE_RENDER_HZ` 0↔10 | Isaac `env.step` | measured-prior (GPU busy) |
| 4 | Datapath | MMIO↔DMA bitstream | FPGA eff-MHz | scripted, not run |
| 5 | Sync mode | free (`minimal_sync`) ↔ barrier | FPGA eff-MHz | measured-prior (ceiling) |
| 6 | Environment | PatternEnv ↔ WarehouseThrust | run-level | measured-prior + computed |

## Run it

```bash
export ROSE_ROOT=/path/to/RoSE
bash experiments/rose_throughput_ablation/run_ablation.sh --now   # socket-fix axis + assemble CSV
```

No FPGA/GPU needed for `--now`. See [`REPRODUCE.md`](REPRODUCE.md) for prereqs, the FPGA/GPU-gated
rows, and expected output. Open [`report.html`](report.html) in a browser (no external deps).

## Model

Effective-MHz rows use one transparent model, validated against the measured FPGA anchors (≤1%):

```
grants(step) = tcyc / step                       # tcyc = 827,967,947 counted target cycles
wall_s       = 13.8 + grants·per_grant + ticks·env_step
effective_MHz = (tcyc/1e6) / wall_s              # free-run 13.8 s => 59.9 MHz = FMR 1.0 = 60 MHz host clock
```

Self-check: @5M model 24.2 vs measured 24.6 MHz; @10M 34.5 vs 34.3; baseline warehouse 16.0 vs 16.2.
