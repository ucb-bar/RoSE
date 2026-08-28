#!/usr/bin/env python3
"""Run many ELFs on the F2 pool CONCURRENTLY and drop their uartlogs locally.

usage: fqbatch.py <tag>=<elf> [<tag>=<elf> ...]

Wraps modelblaster.optimize.firesim_eval.fq_transport.run_fq_many, which
fans the jobs out over a thread pool so N candidates cost ~1 job-time
instead of N. Logs land in experiments/kopt/logs/<tag>.uartlog.
"""
import os, sys
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
from modelblaster.optimize.firesim_eval.fq_transport import run_fq_many, FqConfig

OUT = "/scratch/dima/rose-infra/RoSE/experiments/kopt/logs"
# tag prefix -> the model banner the uartlog must carry
MODEL_OF = {"dronet": "dronet", "fused": "fused_full",
            "mlp": "mlp_generic", "yolo": "yolov8_nano", "yolov8": "yolov8_nano"}

def main():
    jobs = []
    for a in sys.argv[1:]:
        tag, _, elf = a.partition("=")
        if not os.path.exists(elf):
            print(f"SKIP {tag}: no elf {elf}"); continue
        jobs.append((tag, elf))
    if not jobs:
        sys.exit("no jobs")
    os.makedirs(OUT, exist_ok=True)
    cfg = FqConfig()
    cfg.max_parallel = min(8, max(2, len(jobs)))
    print(f"submitting {len(jobs)} job(s), {cfg.max_parallel}-way parallel: "
          f"{[t for t,_ in jobs]}", flush=True)
    models = {MODEL_OF.get(t.split("_")[0]) for t, _ in jobs}
    guard = models.pop() if len(models) == 1 else None
    if guard is None:
        print("NOTE: mixed models in batch -> per-job guard via tag prefix")
    res = {}
    from concurrent.futures import ThreadPoolExecutor
    from modelblaster.optimize.firesim_eval.fq_transport import run_fq
    import time
    def _one(item):
        i, (tag, elf) = item
        if i and cfg.stagger_sec:
            time.sleep(i * cfg.stagger_sec)
        try:
            return tag, run_fq(elf, tag=tag, cfg=cfg,
                               expected_model=MODEL_OF.get(tag.split("_")[0]))
        except Exception as e:
            return tag, e
    with ThreadPoolExecutor(max_workers=cfg.max_parallel) as ex:
        res = dict(ex.map(_one, list(enumerate(jobs))))
    for tag, r in res.items():
        p = os.path.join(OUT, f"{tag}.uartlog")
        if isinstance(r, Exception):
            print(f"FAIL {tag}: {r}")
            continue
        open(p, "w").write(r)
        err = [l for l in r.splitlines() if "max_abs_err" in l]
        print(f"OK   {tag} -> {p}  {err[0].strip() if err else '(no verify line)'}")

if __name__ == "__main__":
    main()
