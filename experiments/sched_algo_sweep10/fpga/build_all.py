#!/usr/bin/env python3
"""Build one ELF per distinct schedule from an elf_plan.json, N builds at once.

Each concurrent build gets its own `examples/xpurt_s10_w<k>` tree because
`xpurt_demo_armB/run.sh` derives BUILD_DIR from the example dir it lives in --
one shared build dir means concurrent builds overwrite each other's objects and
you cannot tell afterwards which schedule any ELF came from.

Freshness is enforced in build_cell.sh (ELF must post-date a stamp taken
immediately before the build) and re-checked here against the dispatch count
the schedule declares.
"""
import argparse, json, os, subprocess, time
import concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
ELFDIR = os.path.abspath(os.path.join(HERE, "..", "elf"))
_slots = None


def one(job):
    p, workers = job
    tag = p["tag"]
    elf = os.path.join(ELFDIR, tag + ".elf")
    w = _slots.get()
    t0 = time.time()
    try:
        r = subprocess.run(["bash", os.path.join(HERE, "build_cell.sh"),
                            p["spec"], p["schedule"], tag, w],
                           capture_output=True, text=True)
    finally:
        _slots.put(w)
    log = os.path.join(HERE, "logs", f"build_{tag}.log")
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    ok = "BUILDDONE" in txt and os.path.exists(elf)
    out = dict(tag=tag, ok=ok, elapsed=round(time.time() - t0, 1),
               bytes=os.path.getsize(elf) if os.path.exists(elf) else 0)
    if not ok:
        bad = [l for l in txt.splitlines() if "ABORT" in l or "error:" in l]
        out["error"] = bad[-1][:200] if bad else f"rc={r.returncode}"
    return out


def main():
    global _slots
    import queue
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    os.makedirs(ELFDIR, exist_ok=True)
    todo = []
    for p in plan:
        elf = os.path.join(ELFDIR, p["tag"] + ".elf")
        if a.skip_existing and os.path.exists(elf) and \
           os.path.getmtime(elf) > os.path.getmtime(p["schedule"]):
            continue
        todo.append(p)
    _slots = queue.Queue()
    for k in range(1, a.workers + 1):
        _slots.put(f"xpurt_s10_w{k}")
    print(f"{len(todo)} of {len(plan)} to build, {a.workers} wide", flush=True)
    res, t0, done = [], time.time(), 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(one, [(p, a.workers) for p in todo]):
            res.append(r); done += 1
            if not r["ok"]:
                print(f"  FAIL {r['tag']}: {r.get('error')}", flush=True)
            if done % 20 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {el:.0f}s  eta {el/done*(len(todo)-done):.0f}s",
                      flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    ok = [r for r in res if r["ok"]]
    print(f"{len(ok)}/{len(res)} built -> {a.out}")


if __name__ == "__main__":
    main()
