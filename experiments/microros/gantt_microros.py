#!/usr/bin/env python3
"""Gantt of a micro-ROS multi-network run, straight from the uartlog trace.

  gantt_microros.py <uartlog> [-o out.png] [--title T]

The trace block between MODELBLASTER_ROS_TRACE_BEGIN/END carries one row per
dispatch:  entry_id,network,instance,dispatch_id,op,name,kind,hart,start,end
with start/end in target CYCLES (not the 14-field xpurt schema shard_dim/gantt.py
reads, hence a separate reader).

The point of the plot is the thing the whole campaign was about: whether two
executors pinned to the SAME hart actually interleave, or whether one owns the
core. So harts are the rows, colour is the network, and the per-hart occupancy
is annotated -- a starved network is a colour that never appears on its hart.
"""
import argparse, collections, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NET_COLOR = {"yolov8_nano": "#1971C2", "dronet": "#E8590C", "mlp_control": "#2F9E44"}
MHZ = 60.0                      # f2_quad_hetero_norose_tacit_q31_60mhz


def rows(path):
    out, inblock = [], False
    for ln in open(path, errors="ignore"):
        ln = ln.strip()
        if "MODELBLASTER_ROS_TRACE_BEGIN" in ln:
            inblock = True; continue
        if "MODELBLASTER_ROS_TRACE_END" in ln:
            break
        if not inblock or not re.match(r"^\d+,", ln):
            continue
        f = ln.split(",")
        if len(f) < 10:
            continue
        try:
            out.append(dict(net=f[1], inst=int(f[2]), did=int(f[3]), kind=f[6],
                            hart=int(f[7]), t0=int(f[8]), t1=int(f[9])))
        except ValueError:
            continue
    return out


def main(path, out, title):
    r = rows(path)
    if not r:
        raise SystemExit(f"{path}: no trace rows between the BEGIN/END markers")
    base = min(x["t0"] for x in r)
    span = (max(x["t1"] for x in r) - base) / MHZ / 1000.0        # ms
    harts = sorted({x["hart"] for x in r})
    nets = [n for n in NET_COLOR if any(x["net"] == n for x in r)]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 6.6),
                                  gridspec_kw={"height_ratios": [2.6, 1]})
    busy = collections.defaultdict(float)
    per_net_hart = collections.defaultdict(float)
    for x in r:
        y = harts.index(x["hart"])
        a = (x["t0"] - base) / MHZ / 1000.0
        w = max((x["t1"] - x["t0"]) / MHZ / 1000.0, span * 4e-4)   # keep 1-cycle ops visible
        ax.barh(y, w, left=a, height=.62, color=NET_COLOR[x["net"]],
                edgecolor="none", zorder=3)
        busy[x["hart"]] += (x["t1"] - x["t0"]) / MHZ / 1000.0
        per_net_hart[(x["hart"], x["net"])] += (x["t1"] - x["t0"]) / MHZ / 1000.0
    ax.set_yticks(range(len(harts)))
    ax.set_yticklabels([f"hart {h}\n{'gemmini' if h < 2 else 'rvv'}\n{busy[h]/span:.0%} busy"
                        for h in harts], fontsize=9)
    ax.set_xlim(0, span); ax.invert_yaxis()
    ax.set_xlabel("target time (ms)")
    ax.grid(axis="x", alpha=.25, zorder=0)
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[Patch(facecolor=NET_COLOR[n], label=n) for n in nets],
              fontsize=9, ncol=len(nets), loc="upper right")

    # Which networks share a hart, and how the hart's time divides between them.
    labels, vals, cols = [], [], []
    for h in harts:
        for n in nets:
            v = per_net_hart[(h, n)]
            if v > 0:
                labels.append(f"h{h}\n{n.replace('_',' ')}"); vals.append(v); cols.append(NET_COLOR[n])
    b = ax2.bar(range(len(vals)), vals, color=cols, edgecolor="k", linewidth=.5)
    ax2.bar_label(b, fmt="%.2f ms", fontsize=8, padding=2)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("busy time (ms)")
    ax2.set_ylim(0, max(vals) * 1.28)
    ax2.set_title("Per-hart time by network — two entries on one hart means they interleaved, "
                  "not that one starved", fontsize=10)
    ax2.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out, dpi=135)
    print(f"wrote {out}   span={span:.2f} ms, {len(r)} dispatches")
    for h in harts:
        share = ", ".join(f"{n}={per_net_hart[(h,n)]:.2f}ms" for n in nets if per_net_hart[(h, n)])
        print(f"  hart {h}: {busy[h]/span:5.1%} busy   {share}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("uartlog")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default="micro-ROS 3-network concurrent run")
    a = ap.parse_args()
    main(a.uartlog, a.out or os.path.splitext(a.uartlog)[0] + "_gantt.png", a.title)
