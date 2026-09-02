# Experiment log

`kernel_opt_log.jsonl` is the durable record of every measured experiment.
Append-only, one JSON object per line, 385 entries. Nothing else in this tree
is authoritative: `results/` holds raw uartlogs that get overwritten by the next
run with the same tag, and an arm's identity (which env vars, which partition,
which bitstream) otherwise lives only in a shell invocation that is gone once
the terminal scrolls.

## Appending

```
python3 shard_dim/scripts/log_entry.py entry.json     # or: ... - <<<'{...}'
```

It fills in `ts`, and **refuses to append if a path under `raw`/`plots` does not
exist** — an entry pointing at an artifact that was never written is worse than
no entry, because it reads as evidence.

## Schema

Required: `ts`, `experiment`, `platform`, `job`, `question`.
Strongly expected: `gates` (what had to hold for the run to count), `raw`
(artifact dirs), and one of `result` / `verdict` / `headline`.
Everything else is free-form — the log has never had a fixed result shape and
should not grow one.

Three conventions worth keeping:

- **`job` carries the fq ids**, e.g. `"fq 470 (zc2A), 471 (zc2B)"`. That is the
  only join key back to the FPGA queue, and it is how you find out whether a
  number came from hardware or a model.
- **`gates` is not decoration.** Every trap this campaign hit was silent: a
  stale ELF, `MB_DRIFT_ATOL` unset (which swaps hardware im2col for a 3.45x
  slower software one), a registry that does not resolve the schedule's slots,
  `--global-curated-dir` passed as an env var so every op fell back to scalar.
  Record the gate that would have caught it.
- **Log the corrections too.** `shard_dim_gemmini_split_ceiling_rca` exists to
  retract a wrong mechanism ("per-dispatch call setup") that a prior entry
  implied. A log that only records successes cannot be used to reason.

## When

Append when a job's results are collected — not at the end of a session. A
result that is not logged before the next `mk_split` overwrites its tree is
unrecoverable except from the uartlog, and only if the tag was not reused.
