# RoSE bridge DMA+reqrsp stress test (non-Isaac, metasim + FPGA)

A small, fully-controllable stress harness for the **RoSE bridge host→FPGA datapath** —
no Isaac, no GPU. It exists to reproduce and debug a mid-flight bridge stall that the
IsaacLab gate-nav co-sim hits on real hardware, without paying the cost (or crash risk) of
the full FPGA flight.

## The bug it reproduces

The gate-nav flight freezes mid-flight: the guest goes WFI-forever on a **reqrsp response
that follows a camera DMA** (vision tick = `0x11` DMA + `0x41/0x42/0x14` reqrsp). It flies
correctly for ~60 control iters, then the vision tick stalls.

This harness isolates it: `guest/main.c` loops **interleaved DMA + reqrsp** with a marker
before each blocking op. Run through the Verilator metasim (same `rosebridge.cc` + RTL as the
FPGA), it **hangs at the 3rd DMA** (`STRESS iter 2: dma...`, `rose_dma_wait` never returns) —
the RTL sim keeps advancing cycles while the guest is frozen.

**Localization (2026-08-12):** the stall is in the **`ROSE_DMA_RX` shared beat-stream**
(`soc/src/main/cc/rosebridge.cc` ~L733–780, commit d5778bd) — the host→FPGA stream carries
*both* DMA and reqrsp data as 512-bit beats (`[count, payload×15]`) and **desyncs after ~3
interleaved packets**, so the FPGA never fires the 3rd DMA's completion IRQ. Ruled out: host
framing (`net_read_full` + `deploy/hephaestus/tests/test_socket_thread_stress.py` pass) and
the guest driver (`rose_adapter.c` arm/wait/ISR are correct). Suspect: the beat framing or the
RTL demux (`RoseStreamToRxAdapter`).

## Run it (two terminals, ~30–60 min for the Verilator rebuild+run)

```bash
# 1) build the stress guest (durable source lives here; installs into the submodule)
experiments/rose_bridge_stress/install_guest.sh

# 2) PatternEnv sync (non-Isaac serve env) on :10001
cd deploy/hephaestus && ROSE_GYM_ENV=PatternEnv-v0 \
  ../.venv-rose/bin/python -u run_sync_only.py \
  --yaml_path ../config/config_gym_PatternEnv-v0.yaml

# 3) drive the stress guest through the DMA metasim (VFireSim + ROSE_DMA_RX)
experiments/rose_bridge_stress/run_metasim_stress.sh
```

**PASS** = uartlog `ROSE stress: 300 iters 0 fails => PASS`.
**Bug reproduced** = a HANG at `STRESS iter K: dma...` / `reqrsp...` (guest frozen, VFireSim
still burning CPU). `grep 'STRESS iter' <log>` shows how far it got.

## Prereqs
- Verilator metasim buildable: needs the `-Wno-SYNCASYNCNET` fix in
  `soc/sim/chipyard/.../rtlsim/Makefrag-verilator` (else the DMA rebuild aborts on that lint).
- `rose_spike_sim` not required (this is the Verilator/RTL tier, not spike).
- Host framing unit test (independent, always-green baseline):
  `python deploy/hephaestus/tests/test_socket_thread_stress.py`.

## Files
- `guest/` — durable stress-guest source (loops interleaved DMA+reqrsp, per-op markers).
- `install_guest.sh` — install source into the submodule + build.
- `run_metasim_stress.sh` — drive it through the DMA-enabled Verilator metasim.
