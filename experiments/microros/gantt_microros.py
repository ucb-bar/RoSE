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
import argparse, collections, json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NET_COLOR = {"yolov8_nano": "#1971C2", "dronet": "#E8590C", "mlp_control": "#2F9E44"}

# The trace column is HEADED "start_cycles"/"end_cycles" and does not contain
# cycles: it is the mtime tick counter, which runs at 1 MHz, so one tick is one
# MICROSECOND regardless of the 60 MHz core clock. Dividing by 60e3 as if they
# were core cycles makes every duration 60x too small -- a 319.55 ms run reads
# as 5.33 ms, which is the give-away that something is wrong, since yolov8_nano
# alone is known to take ~319 ms here.
#
# The check below is not decoration. dronet is launched with a 40 ms period, so
# its measured inter-iteration spacing pins the unit: 39400/40512/39488/40601
# ticks is 39-41 ms as microseconds and 0.66 ms as core cycles. The same field
# name burned this repo once already on the profile CSV, whose "cycles" column
# holds nanoseconds (uartlog_to_profile.py --clock-mhz 1000).
US_PER_TICK = 1.0


MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
QUANT = {"mlp_control": "fp32"}


def op_map():
    """{network: {dispatch_id: op kind}}.

    The trace's `op` and `name` columns are EMPTY in every row -- the harness
    writes the header but never fills them -- so operator identity has to be
    recovered by joining dispatch_id against each model's graph.json. Verified
    to cover 700/700 rows of cfg3xCANON with no misses.
    """
    m = {}
    for net in ("dronet", "mlp_control", "yolov8_nano"):
        g = f"{MB}/examples/{net}/{QUANT.get(net, 'int8')}/generated/graph.json"
        if not os.path.exists(g):
            continue
        ir = json.load(open(g))
        m[net] = {int(o["dispatch_id"]): o["op"]
                  for o in ir["ops"] if o.get("dispatch_id") is not None}
    return m


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
                            hart=int(f[7]), t0=int(f[8]), t1=int(f[9]),
                            op=None))
        except ValueError:
            continue
    return out


def main(path, out, title, zoom=None):
    r = rows(path)
    if not r:
        raise SystemExit(f"{path}: no trace rows between the BEGIN/END markers")
    om = op_map()
    unresolved = 0
    for x in r:
        x["op"] = (om.get(x["net"]) or {}).get(x["did"])
        if x["op"] is None:
            x["op"] = "?"; unresolved += 1
    if unresolved:
        print(f"  warning: {unresolved}/{len(r)} dispatches had no graph.json entry")
    # Unit tripwire: recover a periodic network's period from the trace and
    # refuse to plot if it is not a plausible millisecond-scale period. Getting
    # this wrong does not look like an error, it looks like a fast run.
    for net in ("dronet", "mlp_control"):
        st = sorted({x["t0"] for x in r if x["net"] == net and x["did"] == 0})
        if len(st) >= 3:
            med = sorted(st[i + 1] - st[i] for i in range(len(st) - 1))[len(st) // 2]
            if not (1_000 <= med <= 1_000_000):
                raise SystemExit(
                    f"{net}: median inter-iteration spacing {med} ticks is not a "
                    f"plausible period in microseconds. The tick unit assumption "
                    f"is wrong -- do not trust the time axis.")
            break
    base = min(x["t0"] for x in r)
    span = (max(x["t1"] for x in r) - base) * US_PER_TICK / 1000.0   # ms
    harts = sorted({x["hart"] for x in r})
    nets = [n for n in NET_COLOR if any(x["net"] == n for x in r)]

    # Colour by OPERATOR KIND, not by network: at operator granularity the
    # question is what each hart is actually executing, and the network is
    # already given by the row's hart in this pinning.
    kinds = [k for k, _ in collections.Counter(x["op"] for x in r).most_common()]
    cmap = plt.get_cmap("tab20")
    kcol = {k: cmap(i % 20) for i, k in enumerate(kinds)}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 8.4),
                                  gridspec_kw={"height_ratios": [1, 1.25]})

    busy = collections.defaultdict(float)
    per_net_hart = collections.defaultdict(float)
    for x in r:
        y = harts.index(x["hart"])
        a = (x["t0"] - base) * US_PER_TICK / 1000.0
        w = max((x["t1"] - x["t0"]) * US_PER_TICK / 1000.0, span * 4e-4)
        ax.barh(y, w, left=a, height=.62, color=kcol[x["op"]], edgecolor="none", zorder=3)
        busy[x["hart"]] += (x["t1"] - x["t0"]) * US_PER_TICK / 1000.0
        per_net_hart[(x["hart"], x["net"])] += (x["t1"] - x["t0"]) * US_PER_TICK / 1000.0
    ax.set_yticks(range(len(harts)))
    ax.set_yticklabels([f"hart {h}\n{'gemmini' if h < 2 else 'rvv'}\n{busy[h]/span:.0%} busy"
                        for h in harts], fontsize=9)
    ax.set_xlim(0, span); ax.invert_yaxis()
    ax.set_xlabel("target time (ms)")
    ax.grid(axis="x", alpha=.25, zorder=0)
    ax.set_title(title + "  —  full run, coloured by operator", fontsize=11)
    if zoom is None:
        # Default window: the first dronet iteration that starts after every
        # network is live, so the zoom shows genuine three-way overlap rather
        # than the startup staircase.
        live = max(min(x["t0"] for x in r if x["net"] == n)
                   for n in {y["net"] for y in r})
        cand = sorted(x["t0"] for x in r if x["net"] == "dronet" and x["t0"] >= live)
        z0 = (cand[0] - base) * US_PER_TICK / 1000.0 if cand else 0.0
        zoom = (max(z0 - 1.0, 0.0), max(z0 - 1.0, 0.0) + 42.0)
    ax.axvspan(zoom[0], zoom[1], color="#212529", alpha=.10, zorder=1)

    inwin = [x for x in r
             if (x["t1"] - base) * US_PER_TICK / 1000.0 >= zoom[0]
             and (x["t0"] - base) * US_PER_TICK / 1000.0 <= zoom[1]]
    lab = collections.Counter()
    for x in inwin:
        y = harts.index(x["hart"])
        a = (x["t0"] - base) * US_PER_TICK / 1000.0
        w = max((x["t1"] - x["t0"]) * US_PER_TICK / 1000.0, (zoom[1] - zoom[0]) * 1.2e-3)
        ax2.barh(y, w, left=a, height=.55, color=kcol[x["op"]],
                 edgecolor="k", linewidth=.25, zorder=3)
        # Label only ops wide enough to carry text, once each per hart.
        if w > (zoom[1] - zoom[0]) * .022 and lab[(y, x["op"])] < 2:
            ax2.text(a + w / 2, y - .34, x["op"].replace("2d", "").replace("_s8", ""),
                     ha="center", va="bottom", fontsize=6.4, rotation=32, zorder=5)
            lab[(y, x["op"])] += 1
    ax2.set_yticks(range(len(harts)))
    ax2.set_yticklabels([f"hart {h}" for h in harts], fontsize=9)
    ax2.set_xlim(*zoom); ax2.invert_yaxis()
    ax2.set_xlabel("target time (ms)")
    ax2.grid(axis="x", alpha=.25, zorder=0)
    ax2.set_title(f"zoom {zoom[0]:.0f}–{zoom[1]:.0f} ms — every dispatch, {len(inwin)} of {len(r)}",
                  fontsize=11)
    top = [k for k in kinds if k != "?"][:14]
    ax2.legend(handles=[Patch(facecolor=kcol[k], label=k) for k in top],
               fontsize=7.5, ncol=7, loc="upper center", bbox_to_anchor=(.5, -.22))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}   span={span:.2f} ms, {len(r)} dispatches, {len(kinds)} op kinds")
    for h in harts:
        share = ", ".join(f"{n}={per_net_hart[(h,n)]:.2f}ms"
                          for n in sorted({x['net'] for x in r}) if per_net_hart[(h, n)])
        print(f"  hart {h}: {busy[h]/span:5.1%} busy   {share}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("uartlog")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default="micro-ROS 3-network concurrent run")
    ap.add_argument("--zoom", default=None,
                    help="zoom window in ms, e.g. 60,102 (default: one dronet period once all nets are live)")
    a = ap.parse_args()
    z = tuple(float(v) for v in a.zoom.split(",")) if a.zoom else None
    main(a.uartlog, a.out or os.path.splitext(a.uartlog)[0] + "_gantt.png", a.title, z)
