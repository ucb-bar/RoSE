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

## DMA engine notes (model + limits)

The bridge DMA is a **functional** engine, not a descriptor DMA:
- **Fixed landing zone.** Frames always land at `dma_base`; the guest reads there. `arm()`
  programs only the size, not a target address. Fine for a single camera; the guest copies
  `dma_base` -> its video buffer. A future enhancement (a guest-programmed target-address
  register threaded into `deliver()`) would enable zero-copy DMA straight into the video
  buffer and multiple concurrent DMA buffers.
- **Single payload per frame.** `deliver()` `memcpy`s one payload to `dma_base`, so a DMA
  frame must be served as ONE contiguous payload (`retrieve_obs_push_dma_frame` enforces a
  1-D obs). A 2-D per-row serve (as AirSim stereo uses on a reqrsp channel) would overwrite
  `dma_base` row by row and is rejected.
- **Size.** `dma_base` (0x88000000) sits in guest DRAM (256 MB region); the camera frame and
  the guest's `dma_base` reservation must both fit. Validation uses 64×48; HM01B0 QVGA
  (320×240 = 76,800 B) also fits.
