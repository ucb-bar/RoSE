# RoSE HM01B0 FPV camera over the bridge DMA path (co-sim)

How a legitimate FPV camera frame gets from the Isaac env onto the SoC in a Spike RoSE
co-sim, end to end, and how it is validated. Implemented + validated 2026-08-04 (goal:
"validate getting legitimate Isaac camera sensing data on the SoC from a Spike RoSE sim";
the real himax,hm01b0 HW driver is intentionally stubbed — the FPGA camera-interface IP
does not exist yet).

## Data path (end to end)

```
 Isaac env (GPU) or CamProbe (CPU)        gym_synchronizer.py            rose_spike_sim.cc            guest (Zephyr)
 ─────────────────────────────────        ───────────────────            ─────────────────            ──────────────
 obs["fpv"] = uint8 (H*W,)   ──serve──►  retrieve_obs_push_dma_frame  ──TCP──►  deliver() ch0:            ucbbar,rose-camera
 forward pinhole projection              (one payload, cmd 0x11)                memcpy(addr_to_mem(         video dequeue():
 of the room from the drone              route 0x11 -> ch0 (CFG_ROUTE)          dma_base), frame)           arm DMA0, rose_tx(0x11),
 pose (or real render)                   crc32 logged                          set ST_DMA0_DONE, IRQ       wait IRQ, copy dma_base
                                                                                                           -> video_buffer
```

- **Command / routing:** the camera frame is requested with `CS_CAMERA_LEFT (0x11)` (see
  `rose_packet.h`). The synchronizer already emits `CFG_ROUTE 0x11 -> channel 0` for any
  binding with `channel: 0`, so the bridge routes it to the DMA engine. Same command
  dmavalidate/PatternEnv exercise.
- **Bridge DMA engine** (`soc/src/main/cc/rose_spike/rose_spike_sim.cc`, mirrored in the
  `--extlib` plugin `rose_spike_device.cc`): a data packet routed to channel 0 is
  `memcpy`'d into guest DRAM at a fixed `dma_base` (`--rose-dma-base`, default `0x88000000`),
  then `ST_DMA0_DONE` is set and the completion IRQ raised. Registers: DMA_CFG (arm/size),
  DMA_CUR (progress), INT_PEND (W1C). **This engine already existed and was validated** with
  PatternEnv/dmavalidate — no new engine was needed; see "DMA engine notes" for its model
  and limits.
- **Guest:** `ucbbar,rose-camera` (a Zephyr **video** driver) captures on dequeue —
  `rose_dma_arm(nbytes)`, `rose_tx(0x11)`, `rose_dma_wait(IRQ)`, then copies the landing
  buffer (`rose_dma_buffer()` == `dma_base`) into the app's `video_buffer`. The same app
  binds `himax,hm01b0` on real HW (only the DT compatible differs).

## Pieces

Guest (in `ucb-bar/zephyr-rose`):
- `drivers/video/rose_camera.c` — `ucbbar,rose-camera` video driver (GREY 8-bit, capture-on-dequeue).
- `drivers/video/hm01b0_stub.c` — `himax,hm01b0` stub (advertises format, binds the node, capture returns -ENOSYS; real bring-up waits on the camera IP).
- `dts/bindings/video/{ucbbar,rose-camera,himax,hm01b0}.yaml`.
- `samples/camvalidate/` — captures one frame via the video API, CRC32s it, prints stats + an ASCII thumbnail.

Host (in RoSE):
- `deploy/hephaestus/gym_synchronizer.py` — `retrieve_obs_push_dma_frame()` + a `type: dma` serve branch: serves a flat frame as ONE payload and logs `[dma-serve] ... crc32=`.
- `deploy/hephaestus/envs/isaac_crazyflie/crazyflie_multisensor_env.py` — `synth_fpv_analytic()` (GPU-free pinhole projection of the room from pose; real Isaac render used when `ROSE_ISAAC_CAMERA=1`). Size via `ROSE_FPV_W/H`.
- `deploy/hephaestus/envs/cam_probe/cam_probe_env.py` + `config_gym_CrazyflieCamProbeEnv-v0.yaml` — GPU-free stand-in that serves the same analytic frames from a scripted flight (deterministic validation, no isaaclab).
- `config_gym_IsaacCrazyflieMultiSensorEnv-v0.yaml` — `fpv` now on `0x11 / type: dma / channel: 0`.

## Validation

The frame's CRC32 is consistent across all three layers (env `frame_checksum` = `zlib.crc32`
= guest `crc32_ieee`), so a byte-exact match proves the exact frame landed on the SoC.

- **GPU-free (deterministic):** `CrazyflieCamProbeEnv-v0` + `camvalidate` on `rose_spike_sim`:
  guest `crc32=0x60646b1a` == synchronizer `[dma-serve] crc32=0x60646b1a`; the SoC thumbnail
  shows the hallway (dark corridor ahead, bright side walls).
- **Real Isaac env (Titan RTX):** `IsaacCrazyflieMultiSensorEnv-v0` through the identical path:
  guest `crc32=0x63b3be64` == served `crc32=0x63b3be64` — a frame from the real
  Isaac-simulated pose delivered byte-exact to the SoC.

Repro:
```
# GPU-free:
ROSE_GYM_ENV=CrazyflieCamProbeEnv-v0 ROSE_FPV_W=64 ROSE_FPV_H=48 ROSE_MAZE=hallway \
  python run_sync_only.py --yaml_path ../config/config_gym_CrazyflieCamProbeEnv-v0.yaml &
soc/sim/run_spike_rose_lockstep.sh soc/sim/zephyr_rose_builds/camvalidate/zephyr/zephyr.elf
# Real Isaac (env_isaaclab python): ROSE_GYM_ENV=IsaacCrazyflieMultiSensorEnv-v0 ... same guest.
```

## Forward FPV camera (real Isaac render)

`ROSE_ISAAC_FPV=1` mounts a **body-fixed forward camera** on the drone (`crazyflie_mpc_env.py`,
prim `{ENV}/Robot/body/rose_fpv`, `OffsetCfg` looking down body +x, up +z), with properties
matched to a real **HM01B0**: 320×240 QVGA, 8-bit monochrome (rendered RGB → luma in
`render_fpv_gray()`), pixel pitch 3.6 µm → physical sensor width `W*3.6µm`, and a pinhole
focal length derived so the horizontal FoV = `ROSE_FPV_FOV` (default 70°, ≈ the AI-deck lens).
`_synth_fpv` uses this real render when present, else the GPU-free analytic projection. The
frame flows through the exact same DMA serve path.

## DMA engine notes (model + fixes)

The bridge DMA is a **functional** engine (not a descriptor DMA). Two correctness properties
were fixed while bringing up the camera:

- **Page-chunked writes (fix).** Spike's `mem_t` is *sparse* (a per-4 KB-page host
  allocation), so `addr_to_mem()` returns a pointer valid for only ONE page. The original
  `deliver()` did a single `memcpy` of the whole payload, which ran off the page into the
  host heap for frames > 4 KB (silent for the 16-word dmavalidate, **host heap corruption**
  for a 76,800-byte camera frame). `deliver()` now writes page by page, re-resolving
  `addr_to_mem()` per page (`dma_write()` in `rose_spike_sim.cc` / `rose_spike_device.cc`).
- **Landing zone above the guest heap (fix).** `dma_base` is now **0x90000000** — the first
  address *above* the Zephyr guest's 256 MB SRAM (`0x80000000..0x90000000`) yet inside
  spike's 2 GB physical RAM (no MMU/PMP, so flat-addressable). At the old `0x88000000`
  (mid-SRAM) a DMA'd frame overwrote the kernel heap. The guest overlay's
  `dma-base-address` and the runner's `--rose-dma-base` (via `ROSE_DMA_BASE`, default
  0x90000000) must agree.
- **Fixed landing zone / single payload.** Frames land at `dma_base` and the guest copies
  them into its video buffer; `deliver()` writes one payload, so a DMA frame must be one
  contiguous serve (`retrieve_obs_push_dma_frame` enforces a 1-D obs; a 2-D per-row serve
  would overwrite `dma_base`). A future enhancement (a guest-programmed target-address
  register) would allow zero-copy DMA into the video buffer and multiple DMA buffers.

## Validation (updated)

CRC32 consistent across env (`frame_checksum`), synchronizer (`zlib.crc32`), and guest
(`crc32_ieee`); a match proves the exact frame reached the SoC. All at HM01B0 QVGA 320×240
(76,800 B) unless noted:
- **GPU-free analytic (deterministic):** CamProbe → guest `crc32=0x21968170` == served.
- **Real Isaac forward FPV (Titan RTX):** body-mounted HM01B0 camera → guest
  `crc32=0x54e06b9e` == served; the SoC's ASCII thumbnail shows the rendered corridor
  (ceiling/walls/floor), fully structured (76,800/76,800 non-zero).
