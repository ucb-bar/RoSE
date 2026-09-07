"""Turn the 880 per-job JSONs into the tables and plots the study reports.

Ranking rule, applied everywhere: FEASIBLE-THEN-FASTEST. A schedule that beats
greedy on makespan while overrunning a periodic window has not beaten greedy —
it has moved work out of the objective and into a missed deadline. So the
primary key is `misses == 0` and only the secondary key is `objective`.

`tight_loop` is reported apart from every aggregate: its declared periods are
infeasible by construction (one dronet_sa instance's own critical path is
35.3 ms base / 16.9 ms shard against 2.3 / 3.6 ms windows), so every solver
misses there and the misses say nothing about the solver.
"""
import argparse, json, os, glob, math
from collections import defaultdict
import numpy as np

# The ten the study was commissioned to compare.
TEN = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
       "heft", "heft_edf", "pso", "sa", "cpsat", "cpsat:warm"]
# Two SUPPLEMENTARY arms the ten-solver data motivated, reported apart from the
# ranking of the ten: `best-of-fast` = best feasible-then-fastest of the six
# sub-second heuristics (a virtual solver, costed as the sum of all six);
# `cpsat:warmbest` = CP-SAT hinted from that portfolio instead of from heft_edf
# unconditionally.
EXTRA = ["best-of-fast", "cpsat:warmbest"]
SOLVERS = TEN + EXTRA
CHEAP6 = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
          "heft", "heft_edf"]
FAMILIES = ["bimodal", "control_mix", "depth_chain", "depth_contended",
            "depth_nav", "perception_heavy", "saturation", "scale_ladder",
            "tight_loop", "vint_intro", "vint_multi"]
PAIRS = ["gempair", "hetero", "quad", "rvvpair"]


def load(outdir):
    recs = []
    for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        try:
            recs.append(json.load(open(f)))
        except Exception as e:
            recs.append({"error": f"unreadable {os.path.basename(f)}: {e}"})
    return recs


def famof(name):
    base = name[len("networks_"):] if name.startswith("networks_") else name
    for p in PAIRS:
        if base.endswith("_" + p):
            return base[: -len(p) - 1], p
    return base, "?"


def key(r):
    """Feasible-then-fastest sort key. Failures sort last."""
    if r.get("error") or r.get("objective") is None:
        return (2, math.inf, math.inf)
    return (1 if r["misses"] > 0 else 0, r["misses"], r["objective"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--extra-outdir", default="", help="extra job dirs, comma separated")
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    recs = load(a.outdir)
    for d in (x for x in a.extra_outdir.split(",") if x):
        recs += load(d)

    tbl = {}
    for r in recs:
        if "workload" not in r:
            continue
        tbl[(r["arm"], r["workload"], r["solver"])] = r

    # Synthesise the cheap portfolio. It is a real deployable policy -- run the
    # six sub-second heuristics and keep the best FEASIBLE one -- so it is
    # costed at the sum of all six walls, not the winner's.
    for (arm, wl) in sorted({(k[0], k[1]) for k in tbl}):
        cands = [tbl[(arm, wl, s)] for s in CHEAP6
                 if tbl.get((arm, wl, s)) and tbl[(arm, wl, s)].get("objective") is not None]
        if not cands:
            continue
        b = min(cands, key=key)
        tbl[(arm, wl, "best-of-fast")] = dict(
            b, solver="best-of-fast", picked=b["solver"],
            wall_s=round(sum(tbl[(arm, wl, s)].get("wall_s", 0.0) for s in CHEAP6
                             if tbl.get((arm, wl, s))), 3))
    arms = sorted({k[0] for k in tbl})
    wls = sorted({(k[0], k[1]) for k in tbl})
    print(f"loaded {len(tbl)} records over {len(wls)} workload-arms, "
          f"{len({k[2] for k in tbl})} solvers")

    json.dump([tbl[k] for k in sorted(tbl)], open(f"{a.dest}/all_results.json", "w"),
              indent=1)

    # ---------------- flat CSV ----------------
    with open(f"{a.dest}/results.csv", "w") as fh:
        fh.write("arm,family,pair,workload,ops,periodic_ops,combos,solver,objective,"
                 "all_ops,misses,wall_s,cpsat_status,cpsat_gap,prec_viol,overlap_viol,"
                 "inf_dur_assign,error\n")
        for (arm, wl, s) in sorted(tbl):
            r = tbl[(arm, wl, s)]
            fam, pair = famof(wl)
            v = r.get("validation") or {}
            c = r.get("cpsat") or {}
            gap = c.get("gap")
            fh.write(",".join(str(x) for x in [
                arm, fam, pair, wl, r.get("ops", ""), r.get("periodic_ops", ""),
                r.get("combos", ""), s,
                r.get("objective", ""), r.get("all_ops", ""), r.get("misses", ""),
                r.get("wall_s", ""), c.get("status", ""),
                ("" if gap is None else round(gap, 6)),
                v.get("prec_viol", ""), v.get("overlap_viol", ""),
                v.get("inf_dur_assign", ""),
                (r.get("error", "") or "").replace(",", ";")[:150]]) + "\n")

    # ---------------- failures ----------------
    fails = [r for r in recs if r.get("error")]
    with open(f"{a.dest}/failures.txt", "w") as fh:
        fh.write(f"{len(fails)} failed jobs of {len(recs)}\n\n")
        for r in fails:
            fh.write(f"{r.get('arm')} {r.get('workload')} {r.get('solver')}\n"
                     f"  {r.get('error')}\n")
            if r.get("traceback"):
                fh.write("  " + r["traceback"].replace("\n", "\n  ")[-1200:] + "\n")
            fh.write("\n")

    # ---------------- validation audit ----------------
    bad = []
    for k, r in sorted(tbl.items()):
        v = r.get("validation")
        if not v:
            continue
        if v["prec_viol"] or v["overlap_viol"] or v["inf_dur_assign"] or v["neg_start"] \
           or v["before_min_start"]:
            bad.append((k, v))
    with open(f"{a.dest}/validation.txt", "w") as fh:
        fh.write(f"{len(bad)} of {len([1 for r in tbl.values() if r.get('validation')])} "
                 f"schedules have a precedence/overlap/assignment violation\n\n")
        for k, v in bad:
            fh.write(f"{k}\n  {v}\n")

    # ---------------- per-workload winner + improvement over greedy ----
    improv = defaultdict(list)      # solver -> [pct improvement vs greedy]
    improv_ex = defaultdict(list)   # excluding tight_loop
    wins = defaultdict(int)
    wins_ex = defaultdict(int)
    feas = defaultdict(int)
    total = defaultdict(int)
    misses_tot = defaultdict(int)
    misses_tot_ex = defaultdict(int)
    walls = defaultdict(list)
    per_wl_rank = []

    for (arm, wl) in wls:
        fam, pair = famof(wl)
        rows = {s: tbl.get((arm, wl, s)) for s in SOLVERS}
        g = rows.get("greedy")
        gobj = g.get("objective") if g else None
        ordered = sorted([s for s in SOLVERS if rows.get(s)], key=lambda s: key(rows[s]))
        best = ordered[0] if ordered else None
        if best:
            wins[best] += 1
            if fam != "tight_loop":
                wins_ex[best] += 1
        per_wl_rank.append((arm, fam, pair, wl, best,
                            [(s, rows[s].get("objective"), rows[s].get("misses"))
                             for s in ordered]))
        for s in SOLVERS:
            r = rows.get(s)
            if not r:
                continue
            total[s] += 1
            if r.get("error") or r.get("objective") is None:
                continue
            walls[s].append(r["wall_s"])
            misses_tot[s] += r["misses"]
            if r["misses"] == 0:
                feas[s] += 1
            if fam != "tight_loop":
                misses_tot_ex[s] += r["misses"]
            if gobj:
                pct = (gobj - r["objective"]) / gobj * 100.0
                improv[s].append(pct)
                if fam != "tight_loop":
                    improv_ex[s].append(pct)

    def stat(v):
        return (float(np.mean(v)), float(np.median(v)), float(np.min(v)),
                float(np.max(v))) if v else (0, 0, 0, 0)

    summary = {}
    for s in SOLVERS:
        m, md, lo, hi = stat(improv_ex[s])
        ma, mda, _, _ = stat(improv[s])
        summary[s] = dict(
            n=total[s], solved=len(improv[s]),
            feasible=feas[s], feas_pct=round(100.0 * feas[s] / max(total[s], 1), 1),
            mean_impr_ex_tight=round(m, 3), median_impr_ex_tight=round(md, 3),
            min_impr_ex_tight=round(lo, 3), max_impr_ex_tight=round(hi, 3),
            mean_impr_all=round(ma, 3), median_impr_all=round(mda, 3),
            wins=wins[s], wins_ex_tight=wins_ex[s],
            misses_total=misses_tot[s], misses_ex_tight=misses_tot_ex[s],
            mean_wall_s=round(float(np.mean(walls[s])) if walls[s] else 0, 3),
            median_wall_s=round(float(np.median(walls[s])) if walls[s] else 0, 3),
            max_wall_s=round(float(np.max(walls[s])) if walls[s] else 0, 3))
    json.dump(summary, open(f"{a.dest}/summary.json", "w"), indent=1)

    # ---------------- feasible-first headline ranking ----------------
    # Rank on mean improvement over greedy computed ONLY over the workload-arms
    # where that solver is feasible AND greedy is feasible: crediting a solver
    # for a makespan it bought with a missed window is the error this whole
    # study exists to avoid.
    fair = defaultdict(list)
    for (arm, wl) in wls:
        fam, _ = famof(wl)
        if fam == "tight_loop":
            continue
        g = tbl.get((arm, wl, "greedy"))
        if not g or g.get("objective") is None or g["misses"] > 0:
            continue
        for s in SOLVERS:
            r = tbl.get((arm, wl, s))
            if not r or r.get("objective") is None:
                continue
            if r["misses"] > 0:
                fair[s].append(None)     # counted as "not usable here"
            else:
                fair[s].append((g["objective"] - r["objective"]) / g["objective"] * 100)
    headline = {}
    for s in SOLVERS:
        vals = fair[s]
        ok = [v for v in vals if v is not None]
        headline[s] = dict(
            usable=len(ok), of=len(vals),
            usable_pct=round(100.0 * len(ok) / max(len(vals), 1), 1),
            mean=round(float(np.mean(ok)), 3) if ok else None,
            median=round(float(np.median(ok)), 3) if ok else None,
            p25=round(float(np.percentile(ok, 25)), 3) if ok else None,
            p75=round(float(np.percentile(ok, 75)), 3) if ok else None,
            worst=round(float(np.min(ok)), 3) if ok else None,
            best=round(float(np.max(ok)), 3) if ok else None)
    json.dump(headline, open(f"{a.dest}/headline.json", "w"), indent=1)

    # ---------------- markdown tables ----------------
    with open(f"{a.dest}/RESULTS.md", "w") as fh:
        fh.write("# Ten-solver scheduler bench: wl_sweep x {base, shard}\n\n")
        fh.write(f"{len(wls)} workload-arms x {len(SOLVERS)} solvers = {len(tbl)} solves.\n\n")
        fh.write("## Headline (feasible-first, tight_loop excluded)\n\n")
        fh.write("`usable` = workload-arms where the solver returned a schedule with ZERO "
                 "missed periodic windows (and greedy also had zero, so the comparison is "
                 "defined). `mean/median` = %% makespan improvement over greedy on those.\n\n")
        fh.write("| solver | usable / of | usable %% | mean impr %% | median impr %% | "
                 "p25 | p75 | worst | best | mean wall s |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        order = sorted(SOLVERS, key=lambda s: (-headline[s]["usable"],
                                               -(headline[s]["mean"] or -1e9)))
        for s in order:
            h = headline[s]
            fh.write(f"| {s} | {h['usable']}/{h['of']} | {h['usable_pct']} | "
                     f"{h['mean']} | {h['median']} | {h['p25']} | {h['p75']} | "
                     f"{h['worst']} | {h['best']} | {summary[s]['mean_wall_s']} |\n")
        fh.write("\n## Per-solver summary (all 88 workload-arms)\n\n")
        fh.write("| solver | feasible/88 | wins | wins (ex tight_loop) | total misses | "
                 "misses ex tight_loop | mean impr vs greedy %% | median wall s | max wall s |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for s in SOLVERS:
            v = summary[s]
            fh.write(f"| {s} | {v['feasible']}/{v['n']} | {v['wins']} | {v['wins_ex_tight']} | "
                     f"{v['misses_total']} | {v['misses_ex_tight']} | "
                     f"{v['mean_impr_ex_tight']} | {v['median_wall_s']} | {v['max_wall_s']} |\n")

        fh.write("\n## Per workload-arm: full table\n\n")
        for (arm, wl) in wls:
            fam, pair = famof(wl)
            r0 = next((tbl[(arm, wl, s)] for s in SOLVERS if tbl.get((arm, wl, s))), {})
            fh.write(f"\n### {arm} / {wl}  ({r0.get('ops','?')} ops, "
                     f"{r0.get('periodic_ops','?')} periodic, {r0.get('combos','?')} combos, "
                     f"lanes {r0.get('lanes','?')})\n\n")
            fh.write("| solver | objective ms | all-ops ms | misses | wall s | cpsat status | "
                     "gap | prec/overlap viol |\n|---|---|---|---|---|---|---|---|\n")
            for s in sorted(SOLVERS, key=lambda s: key(tbl.get((arm, wl, s), {"error": 1}))):
                r = tbl.get((arm, wl, s))
                if not r:
                    continue
                if r.get("error"):
                    fh.write(f"| {s} | FAILED | | | {r.get('wall_s','')} | | | "
                             f"{str(r['error'])[:80]} |\n")
                    continue
                v = r.get("validation", {})
                c = r.get("cpsat", {})
                gap = c.get("gap")
                fh.write(f"| {s} | {r['objective']:.3f} | {r['all_ops']:.3f} | {r['misses']} | "
                         f"{r['wall_s']:.2f} | {c.get('status','')} | "
                         f"{'' if gap is None else round(gap,4)} | "
                         f"{v.get('prec_viol',0)}/{v.get('overlap_viol',0)} |\n")

    # ---------------- plots ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots(tbl, wls, a.dest, summary, headline)
    except Exception as e:
        print("plotting failed:", type(e).__name__, e)

    print(f"wrote {a.dest}/RESULTS.md, results.csv, summary.json, headline.json, "
          f"failures.txt, validation.txt")
    print("\nHEADLINE (feasible-first, ex tight_loop):")
    for s in order:
        h = headline[s]
        print(f"  {s:16s} usable {h['usable']:3d}/{h['of']:3d}  mean {str(h['mean']):>8}%  "
              f"median {str(h['median']):>8}%  wall {summary[s]['mean_wall_s']:7.2f}s")


def plots(tbl, wls, dest, summary, headline):
    import matplotlib.pyplot as plt
    C = {s: plt.cm.tab20(i / max(len(SOLVERS) - 1, 1)) for i, s in enumerate(SOLVERS)}

    # 1. normalised makespan by solver, per family (ex tight_loop shown apart)
    fams = [f for f in FAMILIES]
    fig, axes = plt.subplots(3, 4, figsize=(22, 12), sharey=False)
    for ax, fam in zip(axes.ravel(), fams):
        xs, labels = [], []
        for si, s in enumerate(SOLVERS):
            vals = []
            for (arm, wl) in wls:
                f, _ = famof(wl)
                if f != fam:
                    continue
                g = tbl.get((arm, wl, "greedy"))
                r = tbl.get((arm, wl, s))
                if not g or not r or r.get("objective") is None or not g.get("objective"):
                    continue
                vals.append(r["objective"] / g["objective"])
            xs.append(vals)
            labels.append(s)
        ax.boxplot(xs, labels=labels, showfliers=True)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(fam + (" (INFEASIBLE BY CONSTRUCTION)" if fam == "tight_loop" else ""),
                     fontsize=10)
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.set_ylabel("objective / greedy")
        ax.grid(alpha=.3)
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    fig.suptitle("Makespan relative to greedy, by family (both arms pooled; <1 is better)")
    fig.tight_layout()
    fig.savefig(f"{dest}/makespan_by_family.png", dpi=110)
    plt.close(fig)

    # 2. solve time vs quality
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in SOLVERS:
        h = headline[s]
        if h["mean"] is None:
            continue
        w = max(summary[s]["mean_wall_s"], 5e-3)
        ax.scatter(w, h["mean"], s=90, color=C[s], zorder=3)
        ax.annotate(f"{s}\n{h['usable']}/{h['of']} usable", (w, h["mean"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("mean solver wall-clock (s, log)")
    ax.set_ylabel("mean makespan improvement over greedy (%), feasible cases only")
    ax.set_title("Quality vs cost (tight_loop excluded)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{dest}/quality_vs_cost.png", dpi=110)
    plt.close(fig)

    # 3. misses by solver
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (title, kf) in zip(axes, [
            ("workload-arms with >=1 missed window", None),
            ("total missed windows (log)", "tot")]):
        base, shard = [], []
        for s in SOLVERS:
            b = sum(1 for (arm, wl) in wls
                    if famof(wl)[0] != "tight_loop" and arm == "wl_sweep"
                    and (tbl.get((arm, wl, s)) or {}).get("misses", 0) > 0) if kf is None else \
                sum((tbl.get((arm, wl, s)) or {}).get("misses", 0) or 0
                    for (arm, wl) in wls if famof(wl)[0] != "tight_loop" and arm == "wl_sweep")
            sh = sum(1 for (arm, wl) in wls
                     if famof(wl)[0] != "tight_loop" and arm == "wl_sweep_shard"
                     and (tbl.get((arm, wl, s)) or {}).get("misses", 0) > 0) if kf is None else \
                sum((tbl.get((arm, wl, s)) or {}).get("misses", 0) or 0
                    for (arm, wl) in wls if famof(wl)[0] != "tight_loop" and arm == "wl_sweep_shard")
            base.append(b)
            shard.append(sh)
        x = np.arange(len(SOLVERS))
        ax.bar(x - .2, base, .4, label="wl_sweep")
        ax.bar(x + .2, shard, .4, label="wl_sweep_shard")
        ax.set_xticks(x)
        ax.set_xticklabels(SOLVERS, rotation=70, fontsize=8)
        ax.set_title(title + " (tight_loop excluded)")
        ax.legend()
        ax.grid(alpha=.3, axis="y")
        if kf == "tot":
            ax.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(f"{dest}/misses_by_solver.png", dpi=110)
    plt.close(fig)

    # 4. per-workload improvement heat strip
    fig, ax = plt.subplots(figsize=(20, 7))
    order = [(arm, wl) for (arm, wl) in wls]
    M = np.full((len(SOLVERS), len(order)), np.nan)
    for j, (arm, wl) in enumerate(order):
        g = tbl.get((arm, wl, "greedy"))
        if not g or not g.get("objective"):
            continue
        for i, s in enumerate(SOLVERS):
            r = tbl.get((arm, wl, s))
            if r and r.get("objective") is not None:
                M[i, j] = (g["objective"] - r["objective"]) / g["objective"] * 100
    v = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v)
    ax.set_yticks(range(len(SOLVERS)))
    ax.set_yticklabels(SOLVERS, fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{'S' if a=='wl_sweep_shard' else 'B'}:{famof(w)[0]}.{famof(w)[1]}"
                        for a, w in order], rotation=90, fontsize=5)
    fig.colorbar(im, label="% makespan improvement over greedy (ignores misses)")
    ax.set_title("Improvement over greedy per workload-arm  (B=base arm, S=shard arm)")
    fig.tight_layout()
    fig.savefig(f"{dest}/improvement_heatmap.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
