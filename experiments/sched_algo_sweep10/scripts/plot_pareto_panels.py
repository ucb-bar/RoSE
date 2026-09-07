#!/usr/bin/env python3
"""Two panels: Pareto optimality on the left, distributions on the right.

The single-panel version had to carry twelve floating labels on top of twelve
overlapping interquartile rectangles, and the labels and the shading fought
each other. Splitting the two questions fixes that structurally rather than by
tuning offsets:

  LEFT   which solvers are Pareto optimal in quality against solve time. Only
         the frontier is labelled -- four names instead of twelve -- because
         the right panel already names everything in reading order.
  RIGHT  what the spread actually is. Solver names live on a CATEGORICAL AXIS,
         so they are tick labels rather than annotations and cannot collide by
         construction. Rows are sorted by mean gain, so the reader maps left to
         right by colour and by rank.

Deadline misses ride along the right panel as a trailing column, which keeps
the "is it even usable" question attached to the solver it belongs to instead
of in a separate strip.

  plot_pareto_panels.py [--wide]
"""
import csv, math, os, statistics, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "results")
WIDE = "--wide" in sys.argv

FAMILY = {"greedy": "#2a78d6", "greedy_periodic": "#2a78d6",
          "greedy_reserved": "#2a78d6", "decomposed": "#2a78d6",
          "heft": "#eb6834", "heft_edf": "#eb6834", "best-of-fast": "#eb6834",
          "pso": "#1baf7a", "sa": "#1baf7a",
          "cpsat": "#eda100", "cpsat:warm": "#eda100", "cpsat:warmbest": "#eda100"}
FAMNAME = {"#2a78d6": "greedy family", "#eb6834": "list heuristic",
           "#1baf7a": "metaheuristic", "#eda100": "CP-SAT"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
FIG = (11.0, 4.6) if WIDE else (7.6, 4.3)
FT, FA, FK, FL, FG = (14, 12, 11, 11, 10) if WIDE else (11, 9.5, 8.5, 8.5, 8)

rows = [r for r in csv.DictReader(open(f"{R}/results.csv")) if r["family"] != "tight_loop"]
base = {}
for r in rows:
    if r["solver"] == "greedy":
        try: base[(r["arm"], r["workload"])] = float(r["objective"])
        except (TypeError, ValueError): pass
imp, wall, mp = (collections.defaultdict(list) for _ in range(3))
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
    if po: mp[s].append(int(float(r["misses"] or 0)) / po * 100.0)

S = sorted(imp, key=lambda s: -statistics.mean(imp[s]))
X = {s: statistics.median(wall[s]) for s in S}
Y = {s: statistics.mean(imp[s]) for s in S}
M = {s: statistics.mean(mp[s]) for s in S}
Q = {s: statistics.quantiles(sorted(imp[s]), n=4) for s in S}   # p25, p50, p75
# 10th-90th percentile, NOT min-max. The extremes here are single cells --
# heft, heft_edf and decomposed each have one workload where they land near
# -119% -- and letting those set the axis compresses every interquartile box
# into an unreadable smear around zero. The outliers are real and belong in the
# text, not in charge of the scale.
W = {s: (statistics.quantiles(sorted(imp[s]), n=10)[0],
         statistics.quantiles(sorted(imp[s]), n=10)[8]) for s in S}

fig, (ax, bx) = plt.subplots(1, 2, figsize=FIG,
                             gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.32))

# ---------------- left: Pareto ----------------
clean = [s for s in S if M[s] < 0.05]
pts = sorted(((X[s], Y[s], s) for s in clean), key=lambda p: (p[0], -p[1]))
best, front = -1e9, []
for x, y, n in pts:
    if y > best: front.append((x, y, n)); best = y
FRONT = {p[2] for p in front}
ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=INK2, lw=1.3,
        alpha=0.5, zorder=1)
for s in S:
    on, ok = s in FRONT, M[s] < 0.05
    # Hollow marks the solvers that miss deadlines. Without this cue the
    # frontier looks wrong: heft sits at 0.04 s / 3.81%, above AND left of
    # heft_edf, so the eye says the line should start there. It is excluded
    # because it misses 22.9% of periodic operations, and the reader needs to
    # be able to see that from this panel rather than infer it from the other.
    # Fill carries ONE thing (feasible or not) and size carries the other (on
    # the frontier or not). Using opacity for the second made a faded fill read
    # as hollow, which is the very distinction the panel turns on.
    ax.scatter([X[s]], [Y[s]], s=(95 if WIDE else 66) if on else (42 if WIDE else 30),
               facecolor=FAMILY[s] if ok else "none",
               edgecolor="white" if (on and ok) else FAMILY[s],
               linewidth=1.2 if ok else 1.4, zorder=3 if on else 2)
# only the frontier is named here; the right panel names everything
lab = sorted(FRONT, key=lambda s: -Y[s])
for i, s in enumerate(lab):
    dy = 10 if i % 2 == 0 else -15
    ax.annotate(s, (X[s], Y[s]), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=FL, color=INK, fontweight="bold", zorder=4)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.margins(x=0.22, y=0.20)
ax.axhline(0, color=GRID, lw=1.1, zorder=0)
ax.grid(alpha=0.22, lw=0.6, color=GRID); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=FK)
ax.set_xlabel("median solve time (s)", fontsize=FA, color=INK2)
ax.set_ylabel("mean makespan gain over greedy (%)", fontsize=FA, color=INK2)
ax.set_title("Pareto: quality vs solve time", fontsize=FT, color=INK, loc="left")
_h = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=n)
      for c, n in FAMNAME.items()]
_h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none",
                 markeredgecolor=INK2, markersize=5, label="misses deadlines"))
ax.legend(handles=_h,
          loc="lower right", frameon=False, fontsize=FG, labelcolor=INK2,
          handletextpad=0.3, borderpad=0.15, labelspacing=0.25)

# ---------------- right: distributions ----------------
ypos = list(range(len(S)))[::-1]
for yi, s in zip(ypos, S):
    col = FAMILY[s]
    lo, hi = W[s]
    bx.plot([lo, hi], [yi, yi], color=col, lw=0.9, alpha=0.35, zorder=1)   # p10-p90
    bx.plot([Q[s][0], Q[s][2]], [yi, yi], color=col, lw=5.0, alpha=0.42,
            solid_capstyle="butt", zorder=2)                                # IQR
    bx.plot([Q[s][1]], [yi], marker="|", color=col, ms=9, mew=1.6, zorder=3)  # median
    bx.scatter([Y[s]], [yi], s=30, color=col, edgecolor="white", lw=0.8, zorder=4)
bx.axvline(0, color=GRID, lw=1.1, zorder=0)
bx.set_yticks(ypos)
bx.set_yticklabels(S, fontsize=FL, color=INK2)
for t, s in zip(bx.get_yticklabels(), S):
    if s in FRONT: t.set_color(INK); t.set_fontweight("bold")
bx.set_ylim(-0.8, len(S) - 0.2)
bx.grid(axis="x", alpha=0.22, lw=0.6, color=GRID); bx.set_axisbelow(True)
for sp in ("top", "right", "left"): bx.spines[sp].set_visible(False)
bx.spines["bottom"].set_color(GRID)
bx.tick_params(colors=INK2, labelsize=FK, length=2)
bx.set_xlabel("makespan gain over greedy (%)", fontsize=FA, color=INK2)
bx.set_title("Distribution over 80 workload-arms", fontsize=FT, color=INK, loc="left")


# trailing column: the miss rate, attached to its own solver
xr = bx.get_xlim()[1]
bx.text(xr * 1.02, len(S) - 0.35, "misses", fontsize=FG, color=INK2, ha="left",
        va="center", fontweight="bold")
for yi, s in zip(ypos, S):
    txt = "—" if M[s] < 0.05 else (f"{M[s]:.1f}%" if M[s] < 1 else f"{M[s]:.0f}%")
    bx.text(xr * 1.02, yi, txt, fontsize=FG, ha="left", va="center",
            color=INK2 if M[s] < 0.05 else "#b3452a")
x0 = bx.get_xlim()[0]
bx.set_xlim(x0, xr * 1.34)

# One key line for the whole figure: per-panel subtitles collided with the
# titles and with each other at this width.
fig.text(0.5, 0.012,
         "thin bar = p10–p90    block = interquartile    | median    ● mean",
         fontsize=FG - 0.5, color=INK2, ha="center")
fig.subplots_adjust(left=0.085 if WIDE else 0.105, right=0.955, top=0.90, bottom=0.185)
out = os.path.join(os.path.dirname(HERE), "plots",
                   "solver_pareto_panels_wide.png" if WIDE else "solver_pareto_panels.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb")
print("wrote", out, f"({FIG[0]}x{FIG[1]} in)")
print("  frontier:", " → ".join(p[2] for p in front))
