#!/usr/bin/env python3
"""Pareto view of the ten-solver sweep: quality against cost, and against feasibility.

TWO PARETO QUESTIONS, TWO PANELS. Makespan alone is the wrong axis to optimise:
heft has the best raw objective on 13 workload-arms while missing deadlines on
30 of 80, and a schedule that misses its window has not won. So the left panel
asks "how much quality per second of solve time" and the right asks "how much
quality per deadline missed". A solver has to do well on both to be usable.

Colour encodes solver FAMILY, not solver identity: twelve categorical hues is
past the point where anyone can tell them apart, and the skill's rule is that a
9th series folds into composite encoding rather than inventing a hue. Family
(hue) plus a direct label (identity) reads better than twelve colours. Hues are
the reference palette's slots 1-4 in fixed order.

Baseline is greedy, per workload-arm: it is the incumbent that every one of the
88 measured FPGA cells was scheduled with.
"""
import csv, os, statistics, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "results")

# reference palette, categorical slots 1-4, light mode
FAMILY = {
    "greedy": ("greedy family", "#2a78d6"), "greedy_periodic": ("greedy family", "#2a78d6"),
    "greedy_reserved": ("greedy family", "#2a78d6"), "decomposed": ("greedy family", "#2a78d6"),
    "heft": ("list heuristic", "#eb6834"), "heft_edf": ("list heuristic", "#eb6834"),
    "cheap_portfolio": ("list heuristic", "#eb6834"),
    "pso": ("metaheuristic", "#1baf7a"), "sa": ("metaheuristic", "#1baf7a"),
    "cpsat": ("CP-SAT", "#eda100"), "cpsat:warm": ("CP-SAT", "#eda100"),
    "cpsat:warmbest": ("CP-SAT", "#eda100"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

rows = list(csv.DictReader(open(f"{R}/results.csv")))
# tight_loop's periods were infeasible by construction when this swept, so every
# solver misses there for the workload's reasons, not its own. Excluded from the
# aggregate exactly as the sweep's own analysis excluded it.
rows = [r for r in rows if r["family"] != "tight_loop"]

base = {}
for r in rows:
    if r["solver"] == "greedy":
        try: base[(r["arm"], r["workload"])] = float(r["objective"])
        except (TypeError, ValueError): pass

imp = collections.defaultdict(list)
wall = collections.defaultdict(list)
miss = collections.Counter()          # raw total, kept for the printout
miss_pct = collections.defaultdict(list)   # per-cell % of periodic ops missed
hit = collections.Counter()           # cells with ANY miss
clean = collections.Counter()
cells = collections.Counter()
for r in rows:
    k = (r["arm"], r["workload"]); s = r["solver"]
    b = base.get(k)
    try: obj = float(r["objective"])
    except (TypeError, ValueError): continue
    if not b or obj <= 0: continue
    imp[s].append((b - obj) / b * 100.0)
    try: wall[s].append(float(r["wall_s"]))
    except (TypeError, ValueError): pass
    m = int(float(r["misses"] or 0)); miss[s] += m; cells[s] += 1
    clean[s] += (m == 0); hit[s] += (m > 0)
    # NORMALISED, not the raw total. A total conflates "fails a little
    # everywhere" with "fails catastrophically once": decomposed's 130 misses
    # are all in ONE cell while heft's 2930 are spread over 30, so on totals
    # decomposed looks 3x worse than greedy while actually hitting fewer cells.
    # Percent of that cell's own periodic operations is comparable across
    # workloads of very different size.
    po = int(float(r["periodic_ops"] or 0))
    if po:
        miss_pct[s].append(m / po * 100.0)

S = sorted(imp, key=lambda s: -statistics.mean(imp[s]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4))


def frontier(ax, xs, ys, names, lower_x_better=True):
    """Draw the Pareto set: nothing beats these on both axes at once."""
    pts = sorted(zip(xs, ys, names), key=lambda p: (p[0], -p[1]))
    best, front = -1e9, []
    for x, y, n in pts:
        if y > best:
            front.append((x, y, n)); best = y
    ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=INK2,
            lw=1.2, alpha=0.45, zorder=1)
    return {p[2] for p in front}


for ax, xs, xlabel, scale in (
        (ax1, [statistics.median(wall[s]) for s in S],
         "median solve time (s, log scale)  →  cheaper is left", "log"),
        (ax2, [statistics.mean(miss_pct[s]) for s in S],
         "periodic operations that missed their window (%, symlog)  →  fewer is left",
         "symlog")):
    ys = [statistics.mean(imp[s]) for s in S]
    front = frontier(ax, xs, ys, S)
    # Label placement. A fixed offset overprints (several solvers land within a
    # fraction of a percent of each other, and in the right panel seven share
    # x=0 exactly). A naive downward cascade is worse -- it pushes a label onto
    # the NEXT point's label. So place greedily in data space against a running
    # floor: each label sits at its point, or just below the last one placed,
    # whichever is lower. That cannot collide by construction.
    lo, hi = min(ys), max(ys)
    gap = (hi - lo) * 0.052            # ~one label height in data units
    label_y, floor = {}, None
    for i in sorted(range(len(S)), key=lambda i: -ys[i]):
        y = ys[i] if floor is None else min(ys[i], floor)
        label_y[i] = y
        floor = y - gap
    for i, (x, y, s) in enumerate(zip(xs, ys, S)):
        fam, col = FAMILY[s]
        on = s in front
        ax.scatter([x], [y], s=150 if on else 95, color=col, zorder=3,
                   edgecolor="white", linewidth=1.6 if on else 0.9,
                   alpha=1.0 if on else 0.72)
        ly = label_y[i]
        # a hairline leader only when the label had to move off its point
        if abs(ly - y) > gap * 0.35:
            ax.plot([x, x], [y, ly], color=GRID, lw=0.7, zorder=2)
        ax.annotate(s, (x, ly), textcoords="offset points", xytext=(10, -3),
                    fontsize=8.5, color=INK if on else INK2,
                    fontweight="bold" if on else "normal", zorder=4)
    # symlog, not log, on the misses axis: six solvers sit at EXACTLY zero
    # misses, and a linear axis crushes them into one pile against heft's 2930
    # so no frontier can form. symlog keeps zero as a real, plottable value.
    ax.set_xscale(scale, **({"linthresh": 0.1} if scale == "symlog" else {}))
    ax.axhline(0, color=GRID, lw=1.2, zorder=0)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    ax.grid(alpha=0.22, lw=0.6, color=GRID)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)

ax1.set_ylabel("mean makespan improvement over greedy (%)", fontsize=9.5, color=INK2)
ax1.set_title("quality vs cost", fontsize=11, color=INK, loc="left")
ax2.set_title("quality vs feasibility", fontsize=11, color=INK, loc="left")

seen, handles = set(), []
from matplotlib.lines import Line2D
for s in S:
    fam, col = FAMILY[s]
    if fam in seen: continue
    seen.add(fam)
    handles.append(Line2D([], [], marker="o", ls="", color=col, markersize=8,
                          markeredgecolor="white", label=fam))
handles.append(Line2D([], [], ls="-", color=INK2, alpha=0.45, label="Pareto frontier"))
ax1.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
           labelcolor=INK2)

fig.suptitle("Scheduler algorithms on wl_sweep — 12 solvers x 80 workload-arms, "
             "greedy as baseline", fontsize=12.5, color=INK, x=0.005, ha="left")
fig.text(0.005, 0.935,
         "Predicted makespan from the measured-cost model, not hardware. "
         "tight_loop excluded (its periods were infeasible by construction at sweep time). "
         "Swept before the window retune.",
         fontsize=8.5, color=INK2, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(os.path.dirname(HERE), "plots", "solver_pareto.png")
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print("wrote", out)
print(f"\n  {'solver':<18}{'mean %':>9}{'med wall':>10}{'miss%':>10}{'total':>9}{'cells hit':>11}")
for s in S:
    print(f"  {s:<18}{statistics.mean(imp[s]):>9.2f}{statistics.median(wall[s]):>10.2f}"
          f"{statistics.mean(miss_pct[s]):>9.1f}%{miss[s]:>9}{hit[s]:>6}/{cells[s]}")
