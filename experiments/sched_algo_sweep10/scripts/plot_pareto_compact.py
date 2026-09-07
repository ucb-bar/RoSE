#!/usr/bin/env python3
"""Quality-vs-cost Pareto, with deadline misses as a short strip beneath.

The scatter answers ONE question -- how much makespan does a solver buy, and
what does it cost in solve time. Folding feasibility into the marks made a
single frame carry two arguments and read as clutter, so misses now live in
their own strip: one bar per solver that misses anything, and a single "other"
bar at zero standing for every solver that misses nothing at all. Seven of
twelve are in that bar, which is the point -- most solvers here are deadline
clean and only a handful are not.

The frontier is still computed over the deadline-clean solvers only. heft has
the third-best makespan and misses 23% of periodic operations; a frontier that
admits it is recommending it, and the strip below is what explains its absence.

  plot_pareto_compact.py [--wide]
"""
import csv, math, os, statistics, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "results")
WIDE = "--wide" in sys.argv

FAMILY = {"greedy": "#2a78d6", "greedy_periodic": "#2a78d6",
          "greedy_reserved": "#2a78d6", "decomposed": "#2a78d6",
          "heft": "#eb6834", "heft_edf": "#eb6834", "cheap_portfolio": "#eb6834",
          "pso": "#1baf7a", "sa": "#1baf7a",
          "cpsat": "#eda100", "cpsat:warm": "#eda100", "cpsat:warmbest": "#eda100"}
FAMNAME = {"#2a78d6": "greedy family", "#eb6834": "list heuristic",
           "#1baf7a": "metaheuristic", "#eda100": "CP-SAT"}
INK, INK2, GRID, MUTE = "#0b0b0b", "#52514e", "#d8d7d2", "#9a9992"
FIG = (9.0, 4.2) if WIDE else (5.4, 4.0)
FT, FA, FK, FL, FG = (14, 12, 11, 11, 10) if WIDE else (11, 9.5, 8.5, 8.5, 8)


def series(rows):
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
    return imp, wall, mp


rows = [r for r in csv.DictReader(open(f"{R}/results.csv")) if r["family"] != "tight_loop"]
imp, wall, mpct = series(rows)
S = sorted(imp, key=lambda s: -statistics.mean(imp[s]))
X = {s: statistics.median(wall[s]) for s in S}
Y = {s: statistics.mean(imp[s]) for s in S}
M = {s: statistics.mean(mpct[s]) for s in S}


def iqr(v):
    """25th/75th percentile. Each marker summarises 80 workload-arms and a bare
    point throws that sampling away -- two solvers with the same mean can have
    completely different spread, and on this data they do."""
    q = statistics.quantiles(sorted(v), n=4)
    return q[0], q[2]


XQ = {s: iqr(wall[s]) for s in S}
YQ = {s: iqr(imp[s]) for s in S}

fig, (ax, bx) = plt.subplots(2, 1, figsize=FIG, sharex=False,
                             gridspec_kw=dict(height_ratios=[4.4, 0.85], hspace=0.52))

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
mid = 10 ** (sum(math.log10(v) for v in X.values()) / len(X))

for s in S:
    x, y, col, on = X[s], Y[s], FAMILY[s], s in FRONT
    # Interquartile region as a soft box rather than crossed bars: twelve
    # overlapping crosses read as noise, whereas a filled area is legible even
    # where solvers overlap, and it says the same thing -- the middle half of
    # this solver's 80 workload-arms lands in here.
    (xa, xb), (ya, yb) = XQ[s], YQ[s]
    ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, facecolor=col,
                           alpha=0.12, edgecolor=col, linewidth=0.5,
                           joinstyle="round", zorder=1))
    ax.scatter([x], [y], s=(60 if WIDE else 40) * (1.5 if on else 1.0),
               color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ly = label_y[s]
    if abs(ly - y) > gap * 0.35:
        ax.plot([x, x], [y, ly], color=GRID, lw=0.7, zorder=2)
    right = x > mid
    ax.annotate(s, (x, ly), textcoords="offset points",
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
ax.set_xlabel("median solve time (s)", fontsize=FA, color=INK2, labelpad=1)
ax.set_ylabel("makespan gain over greedy (%)", fontsize=FA, color=INK2)
ax.set_title("Scheduler algorithms on wl_sweep", fontsize=FT, color=INK,
             loc="left", pad=13)
ax.text(0.0, 1.012, "shaded = interquartile range over 80 workload-arms",
        transform=ax.transAxes, fontsize=FG - 0.5, color=INK2, va="bottom")
h = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=n)
     for c, n in FAMNAME.items()]
ax.legend(handles=h, loc="lower right", frameon=False, fontsize=FG,
          labelcolor=INK2, handletextpad=0.3, borderpad=0.15, labelspacing=0.25)

# ---- the strip: only solvers that miss anything, plus one bar for the rest ----
bad = sorted((s for s in S if M[s] >= 0.05), key=lambda s: -M[s])
names = bad + [f"other ({len(S) - len(bad)})"]
vals = [M[s] for s in bad] + [0.0]
cols = [FAMILY[s] for s in bad] + [MUTE]
bx.bar(range(len(vals)), vals, color=cols, width=0.62, zorder=3)
for i, v in enumerate(vals):
    bx.text(i, v + max(vals) * 0.06, f"{v:.1f}" if 0 < v < 1 else f"{v:.0f}",
            ha="center", fontsize=FL - 0.4, color=INK2)
bx.set_xticks(range(len(names)))
bx.set_xticklabels(names, fontsize=FL - 0.4, color=INK2, rotation=18, ha="right")
bx.set_ylabel("misses\n(% of ops)", fontsize=FA - 1, color=INK2)
bx.set_ylim(0, max(vals) * 1.35)
bx.set_yticks([0, 10, 20])
bx.tick_params(colors=INK2, labelsize=FK - 1, length=2)
bx.grid(axis="y", alpha=0.22, lw=0.6, color=GRID); bx.set_axisbelow(True)
for sp in ("top", "right", "left"): bx.spines[sp].set_visible(False)
bx.spines["bottom"].set_color(GRID)

# tight_layout refuses this gridspec (it warns and then clips the y-label), so
# the margins are set explicitly: room on the left for the axis title and at
# the bottom for the strip's rotated tick labels.
fig.subplots_adjust(left=0.125 if not WIDE else 0.082, right=0.965,
                    top=0.925, bottom=0.135)
out = os.path.join(os.path.dirname(HERE), "plots",
                   "solver_pareto_compact_wide.png" if WIDE else "solver_pareto_compact.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb")
print("wrote", out, f"({FIG[0]}x{FIG[1]} in)")
print("  frontier:", " → ".join(p[2] for p in front))
print("  strip:", ", ".join(f"{s} {M[s]:.1f}%" for s in bad), f"| other={len(S)-len(bad)}")
