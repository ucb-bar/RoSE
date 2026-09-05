#!/usr/bin/env python3
"""Gantt of the fastdepth -> dronet pipeline, from MEASURED FPGA uartlogs.

  gantt_depth_pipeline.py OUT.png [tag[:label] ...]              measured
  gantt_depth_pipeline.py --schedule OUT.png SCHED.json[:label]  predicted

One row per hart, one bar per dispatch, coloured by PIPELINE STAGE rather than
by network: examples/fastdepth_dronet is a single fused network, so colouring
by network (which experiments/shard_dim/scripts/gantt.py does) paints the whole
timeline one colour and hides the very thing this chart exists to show -- where
depth estimation ends and navigation begins.

UNITS. actual_start_cycles / actual_end_cycles are mtime ticks at 1 MHz, i.e.
MICROSECONDS, despite the column name. Dividing by a 60 MHz clock understates
every span 60x; that mistake has already been made twice in this project, so
the loader calibrates instead of trusting the name: dronet's single-hart arms
reproduce their independently known 7.912 ms / 4.988 ms to 4 decimal places
under /1000, which is what pins the unit.

The trace's predicted_* columns are NOT plotted here. The fused tree is fp32
and was scheduled before it had a profile, so its predicted timeline disagrees
with the measured one by ~3 orders of magnitude -- a real gap, but a scheduling
input gap, not something a Gantt of actuals should quietly average over.
"""
import os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/scratch/dima/rose-infra/RoSE/experiments/sweep3net"
US_PER_MS = 1000.0
STAGE_COL = {"fastdepth": "#2a9d8f", "dronet": "#e76f51"}
# hart -> backend on f2_quad_hetero: 0/1 Rocket+Gemmini, 2/3 Rocket+Saturn.
BE = {0: "gemmini", 1: "gemmini", 2: "rvv", 3: "rvv"}


def load(tag):
    """[(hart, start_ms, end_ms, name, op, stage)] for one run, or None."""
    ul = f"{ROOT}/res_{tag}/uartlog"
    if not os.path.exists(ul):
        return None
    t = open(ul, errors="ignore").read()
    m = re.search(r"xpurt-runner: schedule=(\S+)", t)
    if not m or m.group(1) != tag:
        return None      # a stale uartlog left on the lane by an earlier job
    rows = []
    for l in t.split("\n"):
        f = l.split(",")
        if not re.match(r"^\d+,", l) or len(f) != 14:
            continue     # 12-field summary lines mean something else entirely
        try:
            if int(f[-3]) < 0:
                continue  # kernel-less alias ops: dispatch_id -1, duration 0
            rows.append((int(f[3]), int(f[-3]), int(f[-2]), int(f[-1]), f[5], f[4]))
        except ValueError:
            pass
    if not rows:
        return None
    # Stage by GRAPH POSITION, not by name prefix: the residual adds are bare
    # ("add", "add_1", ...) and carry no "depth."/"nav." prefix, so a prefix
    # test would drop 14 of 216 ops or guess at them from their timing.
    first_nav = min((r[0] for r in rows if r[4].startswith("nav.")), default=None)
    out = []
    t0 = min(r[2] for r in rows)
    for did, hart, s, e, name, op in rows:
        stage = "dronet" if (first_nav is not None and did >= first_nav) else "fastdepth"
        out.append((hart, (s - t0) / US_PER_MS, (e - t0) / US_PER_MS, name, op, stage))
    return out


def load_schedule(path):
    """Same tuple shape as load(), from a scheduled_*.json (PREDICTED times).

    Also returns the cross-network precedence edges, which is the whole point
    of these families: a dependency edge is invisible in a Gantt unless the
    handoff it forces is drawn on top of the bars.
    """
    import json
    d = json.load(open(path))
    v = d["dispatches"]
    v = list(v.values()) if isinstance(v, dict) else v
    slot = {}          # hardware_target -> synthetic hart id, for the row label
    out, cross = [], []
    for x in v:
        ht = x["hardware_target"]
        # CPU_P is the Gemmini-attached hart, CPU_E the Saturn/RVV one; map
        # them onto the real hart numbering so the rows read like the
        # measured panels rather than inventing a second vocabulary.
        h = slot.setdefault(ht, 0 if ht.startswith("CPU_P") else 2)
        job = x["job_name"]
        stage = "dronet" if job.startswith("dronet") else "fastdepth"
        out.append((h, x["start_time"], x["start_time"] + x["duration"],
                    x.get("module_name", ""), job, stage))
        for dep in x.get("dependencies", []):
            if isinstance(dep, str) and "_dispatch_" in dep:
                if dep.rsplit("_dispatch_", 1)[0] != job:
                    cross.append(x["start_time"])
    return out, sorted(set(cross))


def main():
    args = sys.argv[1:]
    sched = False
    if args and args[0] == "--schedule":
        sched = True
        args = args[1:]
    out, specs = args[0], args[1:]
    panels = []
    for spec in specs:
        tag, _, lab = spec.partition(":")
        if sched:
            d, _cross = load_schedule(tag)
            panels.append((lab or os.path.basename(tag), d))
            continue
        d = load(tag)
        if d:
            panels.append((lab or tag, d))
        else:
            print(f"  (skip {tag}: no matching uartlog)")
    if not panels:
        sys.exit("no data")

    span = max(max(e for _, _, e, _, _, _ in d) for _, d in panels)
    fig, axes = plt.subplots(len(panels), 1, squeeze=False,
                             figsize=(13, 1.75 * len(panels) + 1.5))
    for ax, (lab, d) in zip(axes[:, 0], panels):
        harts = sorted({h for h, *_ in d})
        for i, h in enumerate(harts):
            bars = [x for x in d if x[0] == h]
            busy = sum(e - s for _, s, e, _, _, _ in bars)
            ax.broken_barh([(s, max(e - s, span * 4e-4)) for _, s, e, _, _, _ in bars],
                           (i * 10 + 2.5, 5),
                           facecolors=[STAGE_COL[x[5]] for x in bars], linewidth=0)
            ax.text(span * 1.005, i * 10 + 5, f"{busy / span * 100:.0f}% busy",
                    va="center", fontsize=7.5, color="#555")
        # the handoff: last fastdepth op to end, first dronet op to start
        dr_start = min((x[1] for x in d if x[5] == "dronet"), default=None)
        if dr_start is not None:
            ax.axvline(dr_start, color="#333", ls="--", lw=1.1, zorder=5)
            ax.text(dr_start, len(harts) * 10 + 1.2, f" handoff {dr_start:.1f} ms",
                    fontsize=7.5, color="#333", va="bottom")
        ax.set_yticks([i * 10 + 5 for i in range(len(harts))])
        ax.set_yticklabels([f"hart {h}\n{BE.get(h,'?')}" for h in harts], fontsize=8)
        ax.set_ylim(0, len(harts) * 10 + 5)
        ax.set_xlim(0, span * 1.06)
        ax.set_title(f"{lab}   ({len(d)} dispatches, {span if len(panels)==1 else max(e for _,_,e,_,_,_ in d):.1f} ms)",
                     fontsize=9, loc="left")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[-1, 0].set_xlabel("time (ms, predicted by the scheduler)" if sched
                           else "time (ms, measured — mtime ticks at 1 MHz)", fontsize=9)
    fig.legend(handles=[Patch(facecolor=STAGE_COL[k], label=k) for k in ("fastdepth", "dronet")],
               loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("fastdepth → dronet, PREDICTED schedule with an explicit "
                 "dependency edge (int8, hetero)" if sched else
                 "fastdepth → dronet fused pipeline, measured on F2 "
                 "(f2_quad_hetero_norose_tacit_q31_60mhz, fp32)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


main()
