#!/usr/bin/env python3
"""Does the solver ranking hold when the machine is heterogeneous?

Splits the same sweep two ways and re-derives everything within each half:

  HOMOGENEOUS    rvvpair (2 Saturn RVV harts), gempair (2 Gemmini harts)
  HETEROGENEOUS  hetero (1 + 1), quad (2 + 2)

Each half gets its OWN greedy baseline, because comparing a hetero solver's
makespan against a homogeneous greedy would fold the machine's own advantage
into the solver's score. The question here is only whether the ORDERING of the
algorithms survives the change of machine.

  plot_pareto_configs.py [--wide]
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
GROUPS = [("homogeneous", {"rvvpair", "gempair"}, "one backend kind"),
          ("heterogeneous", {"hetero", "quad"}, "Gemmini + Saturn together")]
FAMILY = {"greedy": "#2a78d6", "greedy_periodic": "#2a78d6",
          "greedy_reserved": "#2a78d6", "decomposed": "#2a78d6",
          "heft": "#eb6834", "heft_edf": "#eb6834", "best-of-fast": "#eb6834",
          "pso": "#1baf7a", "sa": "#1baf7a",
          "cpsat": "#eda100", "cpsat:warm": "#eda100", "cpsat:warmbest": "#eda100"}
FAMNAME = {"#2a78d6": "greedy family", "#eb6834": "list heuristic",
           "#1baf7a": "metaheuristic", "#eda100": "CP-SAT"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
FIG = (10.0, 4.2) if WIDE else (5.4, 6.4)
FT, FA, FK, FL, FG = (13.5, 11.5, 10.5, 10.5, 9.5) if WIDE else (11, 9.5, 8.5, 8.5, 8)

rows = [r for r in csv.DictReader(open(f"{R}/results.csv")) if r["family"] != "tight_loop"]


def stats(sub):
    base = {}
    for r in sub:
        if r["solver"] == "greedy":
            try: base[(r["arm"], r["workload"])] = float(r["objective"])
            except (TypeError, ValueError): pass
    imp, wall, mp = (collections.defaultdict(list) for _ in range(3))
    for r in sub:
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
    def q(v):
        a = statistics.quantiles(sorted(v), n=4)
        return a[0], a[2]
    return ({s: statistics.mean(v) for s, v in imp.items()},
            {s: statistics.median(wall[s]) for s in imp},
            {s: statistics.mean(mp[s]) if mp[s] else 0.0 for s in imp},
            {s: q(imp[s]) for s in imp}, {s: q(wall[s]) for s in imp})


G = [(name, note) + stats([r for r in rows if r["pair"] in pairs])
     for name, pairs, note in GROUPS]
allY = [v for _, _, Y, _, _, _, _ in G for v in Y.values()]
allX = [v for _, _, _, X, _, _, _ in G for v in X.values()]
allYQ = [b for _, _, _, _, _, YQ, _ in G for t in YQ.values() for b in t]
allXQ = [b for _, _, _, _, _, _, XQ in G for t in XQ.values() for b in t]
ylim = (min(allY + allYQ) - 1.2, max(allY + allYQ) + 2.0)
xlim = (min(allX + allXQ) * 0.55, max(allX + allXQ) * 2.2)

fig, axes = plt.subplots(1, 2, figsize=FIG) if WIDE else plt.subplots(2, 1, figsize=FIG)
for ax, (name, note, Y, X, M, YQ, XQ) in zip(axes, G):
    S = sorted(Y, key=lambda s: -Y[s])
    clean = [s for s in S if M[s] < 0.05]
    pts = sorted(((X[s], Y[s], s) for s in clean), key=lambda p: (p[0], -p[1]))
    best, front = -1e9, []
    for x, y, n in pts:
        if y > best: front.append((x, y, n)); best = y
    ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=INK2, lw=1.1,
            alpha=0.45, zorder=1)
    FRONT = {p[2] for p in front}
    gap = (ylim[1] - ylim[0]) * 0.070
    label_y, floor = {}, None
    for s in S:
        y = Y[s] if floor is None else min(Y[s], floor)
        label_y[s] = y; floor = y - gap
    mid = 10 ** (sum(math.log10(v) for v in X.values()) / len(X))
    for s in S:
        on = s in FRONT
        # interquartile region as a soft box (see plot_pareto_compact)
        (xa, xb), (ya, yb) = XQ[s], YQ[s]
        ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, facecolor=FAMILY[s],
                               alpha=0.12, edgecolor=FAMILY[s], linewidth=0.5,
                               joinstyle="round", zorder=1))
        ax.scatter([X[s]], [Y[s]], s=(52 if WIDE else 34) * (1.5 if on else 1.0),
                   color=FAMILY[s], edgecolor="white", linewidth=0.8, zorder=3)
        ly = label_y[s]
        if abs(ly - Y[s]) > gap * 0.35:
            ax.plot([X[s], X[s]], [Y[s], ly], color=GRID, lw=0.7, zorder=2)
        right = X[s] > mid
        ax.annotate(s, (X[s], ly), textcoords="offset points",
                    xytext=(-8 if right else 8, -3), ha="right" if right else "left",
                    fontsize=FL, color=INK if on else INK2,
                    fontweight="bold" if on else "normal", zorder=4)
    ax.set_xscale("log"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(0, color=GRID, lw=1.1, zorder=0)
    ax.grid(alpha=0.22, lw=0.6, color=GRID); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=FK)
    ax.set_xlabel("median solve time (s)", fontsize=FA, color=INK2, labelpad=1)
    ax.set_title(f"{name} — {note}", fontsize=FT, color=INK, loc="left")
axes[0].set_ylabel("makespan gain over that\nconfig's greedy (%)", fontsize=FA, color=INK2)
if not WIDE:
    axes[1].set_ylabel("makespan gain over that\nconfig's greedy (%)", fontsize=FA, color=INK2)
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=c, markersize=5, label=n)
                        for c, n in FAMNAME.items()],
               loc="lower right", frameon=False, fontsize=FG, labelcolor=INK2,
               handletextpad=0.3, borderpad=0.15, labelspacing=0.25)
fig.subplots_adjust(left=0.135 if not WIDE else 0.075, right=0.975,
                    top=0.93, bottom=0.10, hspace=0.42, wspace=0.16)
out = os.path.join(os.path.dirname(HERE), "plots",
                   "solver_pareto_configs_wide.png" if WIDE else "solver_pareto_configs.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb")
print("wrote", out)
for name, note, Y, X, M, _YQ, _XQ in G:
    top = sorted(Y, key=lambda s: -Y[s])[:4]
    print(f"  {name:<15} top: " + ", ".join(f"{s} {Y[s]:.1f}%" for s in top))
