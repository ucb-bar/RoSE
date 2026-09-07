#!/usr/bin/env python3
"""One-panel Pareto, dense enough for a single paper column.

WHY ONE PANEL. The two-panel version spends a whole axis on feasibility, but
that dimension is nearly binary: seven of twelve solvers miss EXACTLY zero
deadlines and the other five miss 0.3-22.9%. An axis is the wrong encoding for
a variable with two clusters -- a mark is right. So a filled marker means
deadline-clean and a hollow one means it misses windows, with the percentage
printed beside it. That frees the second panel entirely and the figure drops
into one column, or two side by side.

Reading it: up is better quality, left is cheaper, and hollow is disqualified
however good its position looks. heft is the case in point -- third-best
makespan, and it misses 22.9% of periodic operations.

  plot_pareto_compact.py [--wide]
"""
import csv, os, statistics, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "results")
WIDE = "--wide" in sys.argv

FAMILY = {
    "greedy": "#2a78d6", "greedy_periodic": "#2a78d6",
    "greedy_reserved": "#2a78d6", "decomposed": "#2a78d6",
    "heft": "#eb6834", "heft_edf": "#eb6834", "cheap_portfolio": "#eb6834",
    "pso": "#1baf7a", "sa": "#1baf7a",
    "cpsat": "#eda100", "cpsat:warm": "#eda100", "cpsat:warmbest": "#eda100",
}
FAMNAME = {"#2a78d6": "greedy family", "#eb6834": "list heuristic",
           "#1baf7a": "metaheuristic", "#eda100": "CP-SAT"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
FIG = (7.1, 3.4) if WIDE else (3.5, 3.4)
FT, FA, FK, FL, FG = (13, 11, 10, 10, 9.5) if WIDE else (8.5, 7.5, 7, 6.6, 6.2)

rows = [r for r in csv.DictReader(open(f"{R}/results.csv")) if r["family"] != "tight_loop"]
base = {}
for r in rows:
    if r["solver"] == "greedy":
        try: base[(r["arm"], r["workload"])] = float(r["objective"])
        except (TypeError, ValueError): pass

imp, wall, mpct = (collections.defaultdict(list) for _ in range(3))
for r in rows:
    k, s = (r["arm"], r["workload"]), r["solver"]
    b = base.get(k)
    try: obj = float(r["objective"])
    except (TypeError, ValueError): continue
    if not b or obj <= 0: continue
    imp[s].append((b - obj) / b * 100.0)
    try: wall[s].append(float(r["wall_s"]))
    except (TypeError, ValueError): pass
    po = int(float(r["periodic_ops"] or 0))
    if po: mpct[s].append(int(float(r["misses"] or 0)) / po * 100.0)

S = sorted(imp, key=lambda s: -statistics.mean(imp[s]))
X = {s: statistics.median(wall[s]) for s in S}
Y = {s: statistics.mean(imp[s]) for s in S}
M = {s: statistics.mean(mpct[s]) for s in S}

fig, ax = plt.subplots(figsize=FIG)

# Pareto frontier over the DEADLINE-CLEAN solvers only. A frontier that admits
# infeasible points recommends them, which is exactly the error the two-panel
# version existed to prevent.
clean = [s for s in S if M[s] < 0.05]
pts = sorted(((X[s], Y[s], s) for s in clean), key=lambda p: (p[0], -p[1]))
best, front = -1e9, []
for x, y, n in pts:
    if y > best: front.append((x, y, n)); best = y
ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=INK2, lw=1.2,
        alpha=0.45, zorder=1)
FRONT = {p[2] for p in front}

lo, hi = min(Y.values()), max(Y.values())
gap = (hi - lo) * 0.078
label_y, floor = {}, None
for s in sorted(S, key=lambda s: -Y[s]):
    y = Y[s] if floor is None else min(Y[s], floor)
    label_y[s] = y; floor = y - gap

for s in S:
    x, y, col, ok = X[s], Y[s], FAMILY[s], M[s] < 0.05
    on = s in FRONT
    ax.scatter([x], [y], s=(60 if WIDE else 40) * (1.5 if on else 1.0),
               facecolor=col if ok else "white", edgecolor=col,
               linewidth=1.0 if ok else 1.6, zorder=3)
    ly = label_y[s]
    if abs(ly - y) > gap * 0.35:
        ax.plot([x, x], [y, ly], color=GRID, lw=0.7, zorder=2)
    txt = s if ok else f"{s}  {M[s]:.1f}%" if M[s] < 1 else f"{s}  {M[s]:.0f}%"
    right = x > 10 ** ((sum(map(lambda v: __import__('math').log10(v), X.values())) / len(X)))
    ax.annotate(txt, (x, ly), textcoords="offset points",
                xytext=(-9 if right else 9, -3), ha="right" if right else "left",
                fontsize=FL, color=INK if on else INK2,
                fontweight="bold" if on else "normal", zorder=4)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.margins(x=0.18, y=0.14)
ax.axhline(0, color=GRID, lw=1.1, zorder=0)
ax.grid(alpha=0.22, lw=0.6, color=GRID); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=FK)
ax.set_xlabel("median solve time (s)", fontsize=FA, color=INK2)
ax.set_ylabel("makespan gain over greedy (%)", fontsize=FA, color=INK2)
ax.set_title("Scheduler algorithms on wl_sweep", fontsize=FT, color=INK, loc="left")

h = [Line2D([], [], marker="o", ls="", markerfacecolor=c, markeredgecolor=c,
            markersize=5, label=n) for c, n in FAMNAME.items()]
h.append(Line2D([], [], marker="o", ls="", markerfacecolor="white",
                markeredgecolor=INK2, markersize=5, label="misses deadlines"))
ax.legend(handles=h, loc="lower right", frameon=False, fontsize=FG,
          labelcolor=INK2, handletextpad=0.3, borderpad=0.15, labelspacing=0.25)
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "plots",
                   "solver_pareto_compact_wide.png" if WIDE else "solver_pareto_compact.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb")
print("wrote", out, f"({FIG[0]}x{FIG[1]} in)")
print("  frontier (deadline-clean only):", " → ".join(p[2] for p in front))
