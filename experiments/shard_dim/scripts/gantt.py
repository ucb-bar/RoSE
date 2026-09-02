#!/usr/bin/env python3
"""Gantt charts from an xpurt uartlog: one row per hart, one bar per dispatch.

The uartlog's per-dispatch record carries the hart it ran on and its actual
start/end cycles, which is everything a Gantt needs. Bars are coloured by
BACKEND (which hart kind executed it), because the question these charts exist
to answer is where the work went and how much of the timeline is idle.

  gantt.py OUT.png "Title" <tag>[:label] [<tag>[:label] ...]

Tags are resolved under experiments/sweep3net and experiments/3net, and the
uartlog's `xpurt-runner: schedule=` is checked against the tag -- fq copies
outputs from the run host's sim_slot_*/ and those survive between jobs, so a
cell can otherwise silently plot the previous job's run.
"""
import glob, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOTS = ["/scratch/dima/rose-infra/RoSE/experiments/sweep3net",
         "/scratch/dima/rose-infra/RoSE/experiments/3net"]
# hart -> backend on the quad hetero bitstream: 0/1 gemmini, 2/3 saturn rvv.
# Used only for the row LABEL: which backend ran a dispatch is already carried
# by the track it sits on, so the colour is spent on the network instead.
BE = {0: "gemmini", 1: "gemmini", 2: "rvv", 3: "rvv"}

#: Stable per-network palette, so a network keeps its colour across every
#: figure -- dronet reads the same in its own chart and inside the 3net mix.
NET_COL = {"dronet": "#2f6f9f", "mlp_control": "#e0a458",
           "yolov8_nano": "#7d5ba6", "vint": "#4c9f70"}
FALLBACK = "#8a8a8a"


def load(tag):
    for root in ROOTS:
        for x in (glob.glob(f"{root}/res_{tag}/**/uartlog", recursive=True)
                  + [f"{root}/res_{tag}/uartlog"]):
            if not os.path.exists(x):
                continue
            t = open(x, errors="ignore").read()
            m = re.search(r"xpurt-runner: schedule=(\S+)", t)
            if not m or m.group(1) != tag:
                continue          # stale uartlog from a previous job on the lane
            out = []
            for l in t.split("\n"):
                f = l.split(",")
                # EXACTLY 14 fields. The uartlog mixes two record formats:
                # the 14-field per-dispatch trace, and a 12-field summary line
                # whose columns mean something else entirely. A `>=` filter
                # swept the latter in, where f[1] is an integer rather than a
                # network name -- which is where the bogus "0"/"1" legend
                # entries and the "hart ?" row labels came from.
                if not re.match(r"^\d+,", l) or len(f) != 14:
                    continue
                try:
                    # field 1 is the network; a 3net log carries three of them
                    out.append((int(f[-3]), int(f[-2]) / 1000.0,
                                int(f[-1]) / 1000.0, f[5], f[4], f[1]))
                except ValueError:
                    pass
            if out:
                return out
    return None


def main():
    out, title, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    panels = []
    for s in specs:
        tag, _, lab = s.partition(":")
        d = load(tag)
        if d:
            panels.append((lab or tag, d))
        else:
            print(f"  (skip {tag}: no matching uartlog)")
    if not panels:
        sys.exit("no data")
    span = max(max(e for _, _, e, _, _, _ in d) - min(s for _, s, _, _, _, _ in d)
               for _, d in panels)
    nets_seen = sorted({n for _, d in panels for *_, n in d})
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 1.9 * len(panels) + 1.2),
                             squeeze=False)
    for ax, (lab, d) in zip(axes[:, 0], panels):
        t0 = min(s for _, s, _, _, _, _ in d)
        harts = sorted({h for h, _, _, _, _, _ in d})
        busy = {h: 0.0 for h in harts}
        for h, s, e, nm, op, net in d:
            ax.barh(harts.index(h), max(e - s, span * 0.0004), left=s - t0, height=0.62,
                    color=NET_COL.get(net, FALLBACK),
                    edgecolor="white", linewidth=0.25)
            busy[h] += e - s
        mk = max(e for _, _, e, _, _, _ in d) - t0
        ax.set_yticks(range(len(harts)))
        ax.set_yticklabels([f"hart {h}\n{BE.get(h,'?')}" for h in harts], fontsize=7)
        util = ", ".join(f"h{h} {100*busy[h]/mk:.0f}%" for h in harts)
        ax.set_title(f"{lab}   makespan {mk:.3f} ms   busy: {util}",
                     fontsize=9, loc="left")
        ax.set_xlim(0, span * 1.02)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[-1, 0].set_xlabel("ms (all panels share one scale)", fontsize=9)
    fig.legend(handles=[Patch(facecolor=NET_COL.get(n, FALLBACK), label=n)
                        for n in nets_seen],
               loc="upper right", fontsize=8, frameon=False, ncol=len(nets_seen))
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")


main()
