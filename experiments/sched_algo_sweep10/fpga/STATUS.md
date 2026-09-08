# Build status (live)

`driver_rest.sh` is running **detached** and finishes the whole 1056-row set
unattended. It waits for the tranche-1 build pass to release
`examples/xpurt_s10_w1..6`, waits out the CP-SAT emissions, re-emits anything
missing, dedupes on schedule content, then builds every distinct schedule that
has no ELF yet (`--skip-existing`, so it is safe to re-run).

Watch it:

```bash
tail -f fpga/logs/driver_rest.log            # stage transitions
tail -1 fpga/logs/build_t1.log               # tranche-1 build pass
tail -2 fpga/logs/emit_rest_cpsat.log        # the 222 CP-SAT emissions
ls elf | wc -l                               # ELFs on disk
ls fpga/schedules/*.meta.json | wc -l        # emitted schedules (of 1056)
```

When it is done it writes `fpga/SUMMARY.txt`, the full `fpga/elf_plan.json` and
`fpga/manifest.json` (all 1056 rows -> ELF), `fpga/winners_local.json` and
`fpga/dispatch_order.txt`.

Dispatch is a separate step and is never run by the build pass:

```bash
bash fpga/submit_cell.sh <tag>       # one ELF -> one fq job
cut -f2 fpga/dispatch_order.txt      # tags, winners first, then greedy, then the rest
```
