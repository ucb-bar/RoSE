# FPGA gate-nav — exact experimental setup & reproduce guide

Full, self-contained recipe for the **successful 3-gate warehouse navigation on FPGA**
(Saturn+RoSE on FireSim/U250, in lockstep with IsaacLab), matching the spike reference
(`gate 1@3.27s, 2@7.07s, 3@9.92s`; gate 4 missed, exactly as spike). This documents every
host, config file, environment variable, and command used for the recorded run.

- **Result / evidence:** `experiments/rose_arb_deadlock/` (traces, stills, throughput).
- **Recorded video (chase + FPV):** `docs/media/fpga_gatenav_3gate.mp4` (combined PiP), plus
  `…_chase.mp4` and `…_fpv.mp4`. `docs/media/` is the repo's on-disk media store (gitignored,
  like the other `stage2_*` gate-nav videos) — the clips live there but are not git-tracked;
  regenerate them from the deterministic seed-1000 flight with the ffmpeg recipe in §5.
- **Root-cause writeup:** `docs/FPGA_GATENAV_DEBUG_LOG.md`.
- **Live results page:** https://claude.ai/code/artifact/e8079a95-9fbc-4050-9c94-5231cee5d414
- **One-shot harness** that runs the whole thing: `experiments/rose_arb_deadlock/gatenav_flight.sh`.

---

## 1. Topology — split FPGA / GPU co-sim

The compute SoC runs on an FPGA on one host; the physics + rendering run on a GPU host; they
are bridged over the LAN and advance in lockstep (1 IsaacLab `env.step` per motor-thrust the
guest emits).

```
  garden (GPU, 136.152.139.10)                 firesim1.millennium (FPGA host)
  ┌─────────────────────────────┐              ┌──────────────────────────────────┐
  │ IsaacLab WarehouseThrustEnv  │              │ FireSim manager → U250 slot 0    │
  │ run_sync_only.py (sync loop) │   TCP/LAN    │ Saturn+RoSE @ 30 MHz             │
  │  • renders FPV camera        │◀────────────▶│  • Zephyr nav guest (zephyr.elf) │
  │  • serves 12 sensor packets  │ sync :10071  │  • fused-vision RVV policy       │
  │  • applies 4 motor thrusts   │ data :60002  │  • RoSE bridge adapter           │
  └─────────────────────────────┘              └──────────────────────────────────┘
     ROSE_SYNC_HOST=136.152.139.10                RoSE bridge dials back to the sync
```

- **Sync/handshake port** `10071` (`ROSE_SYNC_PORT`); **data port** `60002` (hard-coded in
  `soc/src/main/cc/rosebridge.cc`). Both must be free on garden.
- The RoSE env (chipyard/firesim/conda) lives on **garden**; **firesim1** is only a remote
  run-farm host (see memory `rose-firesim1-run-farm.md`). The FireSim manager runs on garden
  and drives firesim1 over SSH.

---

## 2. Prerequisites (one-time)

| Component | Location / identity |
|-----------|---------------------|
| Bitstream | hwdb key `alveo_u250_firesim-rocket-saturn-with-rose-fast-no-nic-l2-llc4mb-ddr3` → `results-build/2026-08-13--14-28-37-.../FireSim-RoseTLRocketSaturnMMIOOnlyConfig-.../firesim.tar.gz` (**30 MHz**, WNS +0.060 ns; deeper 2048-deep rx FIFO, commit `64ed2bb`) |
| Guest ELF | workload `rose-nav.json` → `common_bootbinary: zephyr.elf` = the `rose_fused_mpc` Zephyr app built with the **DMA-address fix** (`dma-base-address = 0x88000000`). Build per `docs/ROSE_ISAAC_COSIM_FLOW.md` + memory `rose-modelblaster-spike-build.md` (in-tree zephyr env: `activate_conda` + `set_envvars_sdk` + west build for `spike_riscv64`, then deploy the ELF into the workload dir). |
| Isaac env | conda `env_isaaclab` at `/scratch2/dima/miniforge3/envs/env_isaaclab`; IsaacLab at `/scratch2/dima/IsaacLab`; warehouse gate task in tracked `isaaclab_tasks` (memory `rose-warehouse-gate-nav-task.md`). |
| FireSim env | `soc/sim/chipyard/sims/firesim` — `source env.sh; source sourceme-manager.sh --skip-ssh-setup`. |
| Run-farm host | firesim1 reachable as `vnikiforov@firesim1.millennium.berkeley.edu`, 1×U250 free; paramiko 2.12 / fpga-util version-match / LD_LIBRARY_PATH gates per memory `rose-firesim1-run-farm.md`. |

---

## 3. Config files (exact)

### 3a. `soc/sim/chipyard/sims/firesim/deploy/config_runtime_firesim1.yaml`
Key fields (full file in-tree):
```yaml
run_farm:
  recipe_arg_overrides:
    default_simulation_dir: /scratch/vnikiforov/rose-fsim-run/
    default_fpga_db:        /scratch/vnikiforov/rose-fsim/fpga_target.json
    run_farm_hosts_to_use:
      - "vnikiforov@firesim1.millennium.berkeley.edu": one_fpga_spec
metasimulation:
  metasimulation_enabled: false                 # real FPGA, not metasim
target_config:
  topology: no_net_config
  no_net_num_nodes: 1
  default_hw_config: alveo_u250_firesim-rocket-saturn-with-rose-fast-no-nic-l2-llc4mb-ddr3
  plusarg_passthrough: "+partitioned=1"
workload:
  workload_name: rose-nav.json                  # -> zephyr.elf guest
  terminate_on_completion: no
```

### 3b. `deploy/config/config_gym_WarehouseThrustEnv-v0.yaml` — the sensor/actuator contract
`gym_timestep: 0.005` (5 ms physics); 1 `env.step` per received thrust (`action_latch`).
The 12 served packets + their bridge channels (this is exactly what the guest reads):

| id | name | type | ch | id | name | type | ch |
|----|------|------|----|----|------|------|----|
| 0x11 | cam_front | **dma** | 0 | 0x30 | tof_front | reqrsp | 1 |
| 0x12 | accel | reqrsp | 2 | 0x31 | tof_right | reqrsp | 2 |
| 0x15 | gyro | reqrsp | 2 | 0x32 | tof_back | reqrsp | 1 |
| 0x13 | flow | reqrsp | 1 | 0x33 | tof_left | reqrsp | 2 |
| 0x14 | tof (down) | reqrsp | 1 | 0x41 | tof_cross | reqrsp | 1 |
| 0x16 | state | reqrsp | 1 | 0x42 | lowdim | reqrsp | 2 |
| 0x20 | thrust | action_latch (guest→env) | — | | | | |

### 3c. Guest device-tree — THE FIX
`soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_fused_mpc/boards/spike_riscv64.overlay`:
```dts
rose0: rose@2000 {
    num-reqrsp-channels = <2>;
    num-dma-channels    = <1>;
    dma-base-address = <0x88000000>;   /* MUST match the RTL DMA write addr (see 3d) */
};
rose_cam: rose-cam { rose-cmd = <0x11>; dma-channel = <0>; width = <90>; height = <60>; };
```
The camera lands via **DMA ch0** at `0x88000000`; the driver
(`soc/sw/zephyr-rose/drivers/rose/rose_adapter.c`) selects the just-filled ping-pong half
from the `curr_counter` register (race-free).

### 3d. RTL DMA address (must equal 3c)
`soc/src/main/scala/RoSEConfigs.scala`: `DstParams(port_type="DMA", DMA_address = 0x88000000L, …)`.
**Invariant:** the guest DT `dma-base-address` (3c) and the RTL `DMA_address` (3d) must be the
same value, or the camera freezes (the entire bug — see `docs/FPGA_GATENAV_DEBUG_LOG.md`).
Long-term both should move to `0x90000000` (above SRAM) together + rebuild.

---

## 4. Run it — exact command sequence

The whole flight is one script: `experiments/rose_arb_deadlock/gatenav_flight.sh`. It performs
the three steps below with the exact environment. **Always `infrasetup` immediately before
`runworkload`** — chaining runworkloads without re-infrasetup silently hard-resets the U250
(AER off; memory `rose-fpga-host-crash-mode.md`).

**Step 1 — start the sync/physics on garden** (exact env):
```bash
cd deploy/hephaestus
ROSE_GYM_ENV=WarehouseThrustEnv-v0 ROSE_VISION=1 ROSE_WH_OBST=0 ROSE_FREEZE=1 ROSE_WH_SEED=1000 \
ROSE_ISAAC_CAMERA=1 ROSE_VIDEO_DIR=$CHASE_DIR ROSE_FPV_DIR=$FPV_DIR ROSE_VIDEO_DECIM=2 \
ROSE_RENDER_HZ=10 ROSE_TRAJ_CSV=$TRAJ ROSE_CAM_CHUNK=1 ROSE_SERVE_DEBUG=0 \
ROSE_SYNC_RECV_TIMEOUT=0.001 ROSE_FIRESIM_STEP=5000000 ROSE_FIRESIM_FREQ=1000000000 \
ROSE_SYNC_HOST=136.152.139.10 ROSE_SYNC_PORT=10071 PYTHONPATH=$PWD \
  /scratch2/dima/miniforge3/envs/env_isaaclab/bin/python -u run_sync_only.py \
  --yaml_path $PWD/../config/config_gym_WarehouseThrustEnv-v0.yaml
# wait for: "[run_sync_only] listening on 136.152.139.10:10071"
```
Key knobs: `ROSE_FREEZE=1` (1 physics step per thrust — deterministic), `ROSE_WH_SEED=1000`
(the reference course), `ROSE_WH_OBST=0` (clean gate course), `ROSE_VISION=1` (fused-vision
policy), `ROSE_FIRESIM_STEP=5000000`+`ROSE_FIRESIM_FREQ=1e9` (5 ms target-time/grant),
`ROSE_VIDEO_DIR`/`ROSE_FPV_DIR` (chase + FPV JPEG capture, decim 2), `ROSE_TRAJ_CSV` (real
physics — grant-iters ≠ physics under FREEZE, so measure physics here).

**Step 2 — infrasetup the FPGA** (loads the 30 MHz Saturn+RoSE bitstream to firesim1):
```bash
cd soc/sim/chipyard/sims/firesim && source env.sh && source sourceme-manager.sh --skip-ssh-setup
cd deploy
ROSE_SYNC_HOST=136.152.139.10 ROSE_SYNC_PORT=10071 \
  firesim infrasetup -c config_runtime_firesim1.yaml     # wait for "infrasetup OK"
```

**Step 3 — runworkload** (the guest boots, dials the sync, the flight begins):
```bash
ROSE_SYNC_HOST=136.152.139.10 ROSE_SYNC_PORT=10071 \
  firesim runworkload -c config_runtime_firesim1.yaml
```
Watch `[GATE] passed gate N/4` in the sync stdout and physics time in the traj CSV.
Clean up with `firesim kill -c config_runtime_firesim1.yaml`.

---

## 5. Record the video (chase + FPV picture-in-picture)

The run writes chase-cam JPEGs to `$ROSE_VIDEO_DIR` and the drone-POV FPV JPEGs to
`$ROSE_FPV_DIR` (frame-aligned, decim 2). Compose them into the deliverable:
```bash
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ffmpeg -y -framerate 20 -i $CHASE_DIR/f_%05d.jpg -framerate 20 -i $FPV_DIR/f_%05d.jpg \
 -filter_complex "\
[0:v]scale=960:540,setsar=1[bg];\
[1:v]scale=304:228:flags=neighbor,pad=310:234:3:3:white[fpv];\
[bg][fpv]overlay=W-w-20:H-h-20[v1];\
[v1]drawtext=fontfile=$FONT:text='chase cam':x=24:y=20:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8[v2];\
[v2]drawtext=fontfile=$FONT:text='FPV — policy vision input':x=W-tw-26:y=H-234-52:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8[v3];\
[v3]drawtext=fontfile=$FONT:text='RoSE Saturn+RoSE on FPGA (U250) — warehouse 3-gate nav — seed 1000':x=24:y=H-40:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8" \
 -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart -an docs/media/fpga_gatenav_3gate.mp4
```
(A chase-only variant is `experiments/rose_arb_deadlock/make_gatenav_video.sh`.)

---

## 6. Sim-throughput characterization

```bash
python3 experiments/rose_arb_deadlock/analyze_throughput.py
```
parses the committed wall-clock heartbeat log + physics trajectory and reports the co-sim
throughput. See `docs/FPGA_GATENAV_THROUGHPUT.md` for the numbers from the recorded run
(≈85% FPGA duty, ≈15% co-sim seam overhead).

---

## 7. Expected output

```
[GATE] passed gate 1/4 at t=3.27s xy=(-8.27,+8.13)     # spike 3.33s
[GATE] passed gate 2/4 at t=7.07s xy=(-8.57,+12.15)    # spike 6.65s
[GATE] passed gate 3/4 at t=9.92s xy=(-7.95,+16.13)    # spike 10.14s
# gate 4 missed (spike misses it too) — exact match to the reference
```
Camera is unfrozen (126 unique frame hashes vs 1 when frozen). If the drone flies diagonally
and `gates=0`, the camera is frozen → check the §3c/§3d address invariant first.
```
