#!/usr/bin/env python3
"""Job lists and the dedupe -> ELF plan for the sched_algo_sweep10 FPGA set.

  plan.py jobs   --stage {winners,greedy,rest,all}   -> JSON list of [arm, wl, solver]
  plan.py dedupe --emitted <emit_out.json> ...       -> elf_plan.json + manifest.json

Dedupe is on the schedule's CONTENT hash (`emit_schedule._sched_hash`: every
dispatch's target, start, duration, deps), never on the objective. On
`depth_nav_gempair` eight solvers all report makespan 592.416 and yet form FOUR
distinct assignments -- collapsing those on the number would have thrown away
three quarters of that cell's coverage.
"""
import argparse, collections, hashlib, json, os

SOLVERS = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
           "heft", "heft_edf", "pso", "sa", "best-of-fast",
           "cpsat", "cpsat:warm", "cpsat:warmbest"]
PREF = {s: i for i, s in enumerate(SOLVERS)}
HERE = os.path.dirname(os.path.abspath(__file__))


def cid(s):
    """SCHED_NAME becomes a C identifier and a header guard, so ':' and '-'
    cannot survive. Leading digits break the guard too, hence the s10_ prefix
    the callers add."""
    return s.replace(":", "_").replace("-", "_")


def cells(results):
    out = collections.OrderedDict()
    for r in json.load(open(results)):
        out.setdefault((r["arm"], r["workload"]), {})[r["solver"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["jobs", "dedupe"])
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results", "all_results.json"))
    ap.add_argument("--winners", default=os.path.join(HERE, "winners_recorded.json"))
    ap.add_argument("--stage", default="all")
    ap.add_argument("--emitted", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cl = cells(a.results)
    if a.cmd == "jobs":
        win = {(w["arm"], w["workload"]): w["winner"] for w in json.load(open(a.winners))}
        jobs = []
        if a.stage in ("winners", "all"):
            jobs += [[arm, wl, win[(arm, wl)]] for (arm, wl) in cl]
        if a.stage in ("greedy", "all"):
            jobs += [[arm, wl, "greedy"] for (arm, wl) in cl]
        if a.stage in ("rest", "all"):
            jobs += [[arm, wl, s] for (arm, wl) in cl for s in SOLVERS]
        seen, uniq = set(), []
        for j in jobs:
            k = tuple(j)
            if k not in seen:
                seen.add(k); uniq.append(j)
        json.dump(uniq, open(a.out, "w"))
        print(f"{len(uniq)} jobs -> {a.out}")
        return

    # dedupe
    em = {}
    for f in a.emitted:
        for r in json.load(open(f)):
            if "error" in r:
                em[(r["arm"], r["workload"], r["solver"])] = r
            else:
                em[(r["arm"], r["workload"], r["solver"])] = r
    groups = collections.OrderedDict()   # (arm, wl, hash) -> [solvers]
    failed = []
    for (arm, wl), m in cl.items():
        for s in SOLVERS:
            r = em.get((arm, wl, s))
            if r is None:
                failed.append(dict(arm=arm, workload=wl, solver=s, reason="not emitted"))
                continue
            if "error" in r:
                failed.append(dict(arm=arm, workload=wl, solver=s, reason=r["error"]))
                continue
            groups.setdefault((arm, wl, r["sched_hash"]), []).append(s)

    plan, manifest = [], []
    win = {(w["arm"], w["workload"]): w for w in json.load(open(a.winners))}
    for (arm, wl, h), svs in groups.items():
        svs.sort(key=lambda s: PREF[s])
        fam = wl.replace("networks_", "").rsplit("_", 1)[0]
        cfg = wl.rsplit("_", 1)[1]
        arm_tag = "base" if arm == "wl_sweep" else "shard"
        tag = f"s10_{fam}_{cfg}_{arm_tag}_{cid(svs[0])}"
        rep = em[(arm, wl, svs[0])]
        spec = f"/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/data/toplevel/{arm}/{wl}.json"
        plan.append(dict(tag=tag, arm=arm, workload=wl, family=fam, config=cfg,
                         arm_tag=arm_tag, sched_hash=h, solvers=svs,
                         primary=svs[0], spec=spec, schedule=rep["schedule"],
                         dispatches=rep["dispatches"], objective=rep["objective"],
                         all_ops=rep["all_ops"], misses=rep["misses"],
                         ops=rep["ops"], periodic_ops=rep["periodic_ops"]))
        for s in svs:
            r = em[(arm, wl, s)]
            rec = cl[(arm, wl)][s]
            manifest.append(dict(
                arm=arm, workload=wl, family=fam, config=cfg, solver=s, elf_tag=tag,
                sched_hash=h, shared_with=[x for x in svs if x != s],
                emit_objective=r["objective"], emit_misses=r["misses"],
                emit_all_ops=r["all_ops"], emit_wall_s=r["wall_s"],
                sweep10_objective=rec.get("objective"), sweep10_misses=rec.get("misses"),
                sweep10_error=rec.get("error"),
                reproduces=(rec.get("objective") is not None
                            and abs(r["objective"] - rec["objective"]) < 1e-6),
                is_recorded_winner=(win[(arm, wl)]["winner"] == s),
                is_greedy_baseline=(s == "greedy")))
    json.dump(plan, open(a.out, "w"), indent=1)
    mpath = os.path.join(os.path.dirname(os.path.abspath(a.out)), "manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"rows {len(manifest)} distinct schedules {len(plan)} failed {len(failed)}")
    print(f"  plan -> {a.out}\n  manifest -> {mpath}")
    if failed:
        fpath = os.path.join(os.path.dirname(os.path.abspath(a.out)), "unbuildable.json")
        json.dump(failed, open(fpath, "w"), indent=1)
        print(f"  unbuildable -> {fpath}")
        for x in failed[:12]:
            print("   ", x["arm"], x["workload"], x["solver"], x["reason"][:90])


if __name__ == "__main__":
    main()
