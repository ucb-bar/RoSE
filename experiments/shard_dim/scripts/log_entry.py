#!/usr/bin/env python3
"""Append one entry to experiments/kernel_opt_log.jsonl.

The log is APPEND-ONLY and shared with 378 prior entries, so this never
rewrites and never reorders. Schema follows the existing entries: `ts`,
`experiment`, `platform`, `job`, `question`, `gates`, then free-form result
fields, then `raw` (artifact paths) and `plots`.

  log_entry.py entry.json            # append the object in entry.json
  cat entry.json | log_entry.py -    # ...or from stdin

`ts` is filled in if absent. Paths under `raw`/`plots` are checked to exist,
because an entry pointing at an artifact that was never written is worse than
no entry -- it reads as evidence.
"""
import datetime, json, os, sys

LOG = "/scratch/dima/rose-infra/RoSE/experiments/kernel_opt_log.jsonl"
ROOT = "/scratch/dima/rose-infra/RoSE"


def append(entry: dict, strict: bool = True) -> None:
    entry.setdefault("ts", datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    missing = [p for k in ("raw", "plots") for p in entry.get(k, [])
               if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        msg = "artifact paths do not exist: " + ", ".join(missing)
        if strict:
            sys.exit(f"[log_entry] refusing to log -- {msg}")
        print(f"[log_entry] WARNING {msg}")
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[log_entry] appended {entry.get('experiment')} ({entry.get('job')})")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "-"
    obj = json.load(sys.stdin if src == "-" else open(src))
    for e in (obj if isinstance(obj, list) else [obj]):
        append(e)
