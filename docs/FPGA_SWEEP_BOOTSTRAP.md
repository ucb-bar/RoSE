# Getting to the point where the FPGA sweep can run

`soc/sw/xpu-rt/scripts/repro_fpga_sweep.sh` assumes a working AWS manager, a
built bitstream in the FireSim HWDB, and an `fq` queue with at least one F2
run host. This document is everything upstream of that assumption. It is the
"how did this environment come to exist" companion to
`soc/sw/xpu-rt/runs/sweeps/fpga_20260829-195805/RUNBOOK.md` (how to run the
sweep) and `docs/FPGA_QUEUE_USAGE.md` (how to drive the queue).

Nothing here is fast. The bitstream build is hours; budget a day for a
cold start.

---

## 0. AWS account and access

Account 025690631703, region us-east-1. F1 is retired; everything is **F2**
(`f2.6xlarge`, VU47P). Key `firesim.pem` — there is one copy, guard it.

Quota: "Running On-Demand F instances" must cover 24 vCPUs per f2.6xlarge.
Ours is 384, i.e. 16 instances. Check before scaling:

    aws service-quotas get-service-quota --region us-east-1 \
        --service-code ec2 --quota-code L-74FC7D96

## 1. Manager instance

The manager is a small always-on box (m5.xlarge) holding the chipyard tree,
the FireSim manager, the HWDB and the `fq` daemon. It is NOT an FPGA host.
Ours is shared (`firesim-manager-chrisdong`) — treat it as shared
infrastructure: do not reap it, and drain the queue before restarting the
daemon.

It carries `~/chipyard-rose` (the chipyard+FireSim tree the sweep names as
`--tree`) and `~/firesim.pem` for reaching run hosts on the private subnet.

## 2. Bitstream

The sweep runs `f2_dual_small_norose_tacit_q31_60mhz`, which is
`SatGemDualSmallTacitMMIOOnlyConfig` ->
`chipyard.config.SatGemDualSmallTacitConfig`:

    tacit.WithTraceSinkRawByte(0) ++ chipyard.WithTacitEncoder ++
    gemmini.Q31WsGemminiConfig ++
    saturn.rocket.WithRocketVectorUnit(256, 128, VectorParams.refParams) ++
    freechips.rocketchip.rocket.WithNBigCores(2) ++
    chipyard.config.WithSystemBusWidth(128) ++ AbstractConfig

Two Rocket cores; hart 0 also carries the Gemmini Q0.31 RoCC. **Both harts
get the Saturn vector unit** — `WithRocketVectorUnit` hardcodes `vfh=true`
and `minFLen=16`, so Zfh and Zvfh are present on every tile regardless of
the config; there is no flag to disable them. The registry's
"hart 1 = saturn-RVV-only" is a role split, not a hardware difference.

Build on a z1d build-farm instance driven by the manager:

    cd ~/chipyard-rose/sims/firesim/deploy
    firesim buildbitstream -b <config_build_*.yaml>

Traps:
* **espresso must be on PATH**, before AND after sourcing the env scripts.
  Without it Rocket's decode-table minimisation falls back to
  Quine-McCluskey and the build OOMs in elaboration. This cost a whole
  failed build.
* GoldenGate needs headroom: guard on >=9 GB free RAM and >=25 GB disk.
* Vivado synthesis + P&R is hours. `buildbitstream` farms to a z1d; the
  manager only orchestrates.
* On success, add the AGFI to the HWDB so `--hw-config` resolves it.

One-hart-per-kind is what forces `pool_sizes = 0` and disables intra-op
sharding (see `experiments/spike_intraop/README.md`). A multi-hart-per-kind
bitstream (`SatGemQuadHeteroTacitMMIOOnlyConfig`) is the prerequisite for
enabling it on FPGA.

## 3. Run hosts and the fq queue

Launch F2 hosts from the prepared AMI, same subnet/SG/key, tagged
`fsimcluster=rosef2run`:

    aws ec2 run-instances --region us-east-1 \
      --image-id ami-017bd23ff95264395 --instance-type f2.6xlarge --count N \
      --key-name firesim --subnet-id subnet-061e6fb07d6ea44da \
      --security-group-ids sg-0fcc5bf0afef3f0e5 \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=fsimcluster,Value=rosef2run}]'

`fq` binds lanes to hosts by **explicit private IP** in `/var/lib/fq/pool.yaml`,
one host per lane (`firesim kill` is host-wide `pkill`; `infrasetup` reflashes
every slot, so two jobs on one host destroy each other). After changing the
pool:

    # queue MUST be idle first
    kill $(pgrep -f 'fq.cli daemon'); sleep 5
    cd ~/fpga_queue && (setsid python3 -m fq.cli daemon --pool /var/lib/fq/pool.yaml \
        > ~/fq_daemon.log 2>&1 < /dev/null &)

Traps:
* The daemon reads pool.yaml only at startup; there is no reload subcommand.
* Backgrounding the relaunch inside the same ssh that killed it does not
  survive the disconnect — use the subshell form above and verify with
  `fq lanes`, not `pgrep` (a `pgrep -f` pattern matches your own ssh command
  and reports a phantom daemon).
* Terminating a host leaves its lane pointing at a dead IP: drain the lane,
  terminate, rewrite pool.yaml, restart.
* `lspci | grep xilinx` returns 0 even on a healthy host — not a useful probe.

## 4. Profiles and dispatch graphs

The scheduler needs, per model and per backend, a measured `results.csv` plus
a dispatch graph. Both are now committed for the sweep's five models, so a
clone needs no regeneration. To add a model:

    # dispatch graph
    python -m modelblaster.pipeline.emit_dispatch_graph --ir <graph.json> \
        --out-root zephyr-chipyard-sw/gen/vmfb --target firesim_f2_armB --hw gemmini_q31

    # profile: build single-backend, run on FPGA, convert the uartlog
    python3 scripts/uartlog_to_profile.py --uartlog U --model M --quant int8 \
        --backend {gemmini_q31|V256D128_rvv} --cpu firesim_f2_armB \
        --cores 0 --clock-mhz 1.0 --out-root gen/profile

Then add the model to `data/banks/model_bank.json` under `firesim_f2_armB`
with `role`, `count`, and (if periodic) `period_ms`/`window_ms`.

**Trace cycle units**: the XPURT_TRACE columns are Zephyr `k_cycle_get_64()`
ticks. `CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC=1000000`, so 1 tick = 1 us and the
ms divisor is 1000 (`--trace-clock-mhz 1.0`). Do NOT use 60 (that is the host
FPGA frequency) and do NOT assume 1 GHz target cycles.

## 5. Then run the sweep

    bash soc/sw/xpu-rt/scripts/repro_fpga_sweep.sh --dry-run --seeds 0-7 --arms baseline,fused
    bash soc/sw/xpu-rt/scripts/repro_fpga_sweep.sh          --seeds 0-7 --arms baseline,fused

See RUNBOOK.md for flags, the per-step detail, and the workarounds currently
carried in the tree (per-dispatch IRQ guard, bounded profile dump,
`rvv_f16.conf`).

## What is NOT yet reproducible from a clone

The commits described here are LOCAL. A real fresh clone cannot fetch them
until `modelblaster`, `zephyr-chipyard-sw`, `xpu-rt` and `RoSE` are pushed,
in that order (each parent records the child's new SHA). Validation so far
used a clone with submodule URLs overridden to the local repos, which proves
the committed *content* is complete but not that a remote clone works.
