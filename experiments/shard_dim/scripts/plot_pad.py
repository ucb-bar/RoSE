#!/usr/bin/env python3
"""Attribution: how much of the OH work-ratio win is the SPLIT, and how much is
the pre-padding the split's tiles happen to do anyway.

  plot_pad.py <out_dir>

Left  -- per conv, the pre-padding-only ratio (k=1, no split) against the OH k=2
         ratio. The gap between them IS the row split's marginal contribution.
Right -- the end-to-end makespans, measured, against the scheduler's prediction.
"""
import csv, glob, json, os, re, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ul import parse_uartlog, conv_rows

RES = "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/results"
MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
C_PAD, C_SPLIT, C_OC = "#1baf7a", "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#d9d8d4"


def style(ax):
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def ul(d):
    return max(glob.glob(f"{d}/**/uartlog", recursive=True), key=os.path.getsize)


def main():
    out = sys.argv[1]
    b0 = conv_rows(parse_uartlog(ul(f"{RES}/aln/res_alnB0E")))
    pd = conv_rows(parse_uartlog(ul(f"{RES}/oh/res_padE")))
    cells = {(r["op"], r["axis"]): float(r["work_ratio"])
             for r in csv.DictReader(open(f"{RES}/ohvsoc_cells.csv"))
             if r["backend"] == "rvv" and r["k"] == "2" and r["work_ratio"]}
    geom = {o["name"]: o["shape"] for o in
            json.load(open(f"{MB}/examples/dronet_armB/int8/generated/graph.json"))["ops"]
            if o.get("op") == "conv2d_s8"}
    ops = sorted(pd, key=lambda n: int(n.split(".")[-1]))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6),
                             gridspec_kw={"width_ratios": [1.75, 1]})
    ax = axes[0]
    w = 0.36
    for i, nm in enumerate(ops):
        rp = pd[nm][0][1] / b0[nm][0][1]
        roh = cells.get((nm, "OH"))
        roc = cells.get((nm, "OC"))
        ax.bar(i - w / 2, rp, w, color=C_PAD, edgecolor="white", lw=0.6)
        ax.text(i - w / 2, rp + 0.03, f"{rp:.2f}", ha="center", va="bottom",
                fontsize=7, color=INK)
        if roh:
            ax.bar(i + w / 2, rp, w, color=C_PAD, edgecolor="white", lw=0.6)
            ax.bar(i + w / 2, roh - rp, w, bottom=rp, color=C_SPLIT,
                   edgecolor="white", lw=0.6)
            ax.text(i + w / 2, max(roh, rp) + 0.03, f"{roh:.2f}", ha="center",
                    va="bottom", fontsize=7, color=INK)
        if roc:
            ax.plot([i - 0.44, i + 0.44], [roc, roc], color=C_OC, lw=1.5, ls="--")
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([f"{n.split('.')[-1]}\nK{geom[n]['KH']} S{geom[n]['SH']}"
                        for n in ops], fontsize=7.5, color=INK2)
    ax.set_ylabel("work ratio vs unsplit", fontsize=8.5, color=INK2)
    ax.set_title("Almost the whole OH win is PRE-PADDING, not the row split  "
                 "(rvv, serial)", fontsize=10, color=INK, loc="left", pad=8)
    ax.legend(handles=[Patch(facecolor=C_PAD, label="pre-padding alone (k=1, NO split)"),
                       Patch(facecolor=C_SPLIT, label="the row split's marginal cost"),
                       plt.Line2D([], [], color=C_OC, ls="--", lw=1.5,
                                  label="OC split, k=2")],
              frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_ylim(0, 2.6)
    style(ax)
    # right: measured e2e
    ax = axes[1]
    import analyze_e2e as A  # reuse the trace parser
    def span(tag):
        t = A.trace(ul(f"{RES}/oh/res_{tag}"))
        return t["span"]
    mu, mm = span("e2eU"), span("e2eM")
    pu = json.load(open("/scratch/dima/rose-infra/RoSE/experiments/shard_dim/"
                        "ohsched/e2eU.json"))["metadata"]["makespan"] * 1e6
    pm = json.load(open("/scratch/dima/rose-infra/RoSE/experiments/shard_dim/"
                        "ohsched/e2eM.json"))["metadata"]["makespan"] * 1e6
    xs = [0, 1]
    ax.bar([x - 0.19 for x in xs], [pu, pm], 0.36, color="#c9d7ea",
           edgecolor="white", lw=0.6)
    ax.bar([x + 0.19 for x in xs], [mu, mm], 0.36, color=C_SPLIT,
           edgecolor="white", lw=0.6)
    for x, (p, m) in zip(xs, ((pu, mu), (pm, mm))):
        ax.text(x - 0.19, p + 25, f"{p:.0f}", ha="center", fontsize=7.5, color=INK2)
        ax.text(x + 0.19, m + 25, f"{m:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(["unsplit", "best per-axis\n(all OC)"], fontsize=8.5, color=INK2)
    ax.set_ylabel("makespan (us)", fontsize=8.5, color=INK2)
    ax.set_title(f"End-to-end, 4 harts: {mu/mm:.2f}x measured",
                 fontsize=10, color=INK, loc="left", pad=8)
    ax.legend(handles=[Patch(facecolor="#c9d7ea", label="predicted (measured per-op costs)"),
                       Patch(facecolor=C_SPLIT, label="measured on FPGA")],
              frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, max(mu, pu) * 1.22)
    style(ax)
    fig.tight_layout()
    p = os.path.join(out, "oh_padding_attribution.png")
    fig.savefig(p, dpi=160, facecolor="white")
    print(f"  -> {p}")


main()
