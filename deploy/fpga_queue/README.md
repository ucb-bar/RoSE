# `fq` — FireSim job queue for a shared FPGA pool

Submit `(hardware config + guest ELF + options)`, and `fq` queues it and runs it
on whichever FPGA is free. Multiple users and unrelated processes share the pool
safely.

**Full design rationale: [`docs/FPGA_QUEUE_DESIGN.md`](../../docs/FPGA_QUEUE_DESIGN.md).**
Read §2 (the lane model) and §5 (locking) before changing anything.

## Quick start

```bash
./bin/fq check examples/pool.example.yaml       # validate a pool, start nothing
sudo FQ_POOL=/etc/fq/pool.yaml ./bin/fq daemon  # root => jobs run as submitter

export FQ_SOCKET=/var/lib/fq/fq.sock
./bin/fq submit --tree ~/chipyard-rose \
                --hw-config f2_dual_small_rose_tacit_q31_60mhz \
                --elf ./zephyr.elf --timeout 1800 \
                --results ./out --wait
./bin/fq status      # queue, with the reason each job is blocked
./bin/fq lanes       # pool, incl. lanes held by orphaned simulations
```

## Tests

```bash
./run_tests.sh          # everything (~9 min; starts real daemons, mock FPGAs)
./run_tests.sh -f       # unit tests only (~0.1s)
```

No AWS, no FireSim and no FPGA required — the mock backend's "FPGAs" are
directories and its "simulations" are real processes.

## The three things most likely to surprise you

1. **A lane is whole hosts, never a sub-slot.** `firesim kill` runs a host-wide
   `pkill FireSim-f2` and `infrasetup` reflashes every slot on the host, so two
   jobs sharing a host destroy each other. Config load refuses it.
2. **"Instance running" ≠ "FPGA free".** A guest that never exits leaves the
   simulation attached to the FPGA forever. `fq` probes for the live driver
   process; a lane with a simulation nobody owns is `FOREIGN` and is not
   scheduled until reclaimed. Do not disable probing in production.
3. **The flock is the only authority on who owns a lane.** The DB and each
   job's `status.json` are records, not permissions.
