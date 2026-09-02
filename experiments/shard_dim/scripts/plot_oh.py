#!/usr/bin/env python3
"""Figures for the OH (spatial) vs OC (output-channel) split-axis study.

  plot_oh.py <cells.csv> <plots_dir>

  1. oh_vs_oc_work_ratio.png -- the headline. Cost of a 2-way split relative to
     not splitting, per conv, per backend, one bar per AXIS. 1.0 is free.
  2. oh_split_anatomy.png    -- why. (a) measured OH overhead against the halo
     model's row duplication; (b) how each axis scales with split degree on the
     one conv where both are available at k=2 and k=4.

Two series, so colour is categorical: slots 1 and 2 of the documented theme
(blue / orange), inside the three-slot subset that validates all-pairs in both
modes. Hatch repeats the distinction so identity is never colour alone, and
every bar is direct-labelled.
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_OC, C_OH = "#eb6834", "#2a78d6"          # slot 2 orange, slot 1 blue
H_OC, H_OH = "///", None
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#d9d8d4"
BACKENDS = [("rvv", "rvv  (Saturn V256, OC quantum 32)"),
            ("gemmini_q31", "gemmini_q31  (systolic DIM 16)")]
ORDER = [f"conv_modules.{i}" for i in range(10)]


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def best(rows, be, op, axis, k):
    v = [float(r["work_ratio"]) for r in rows
         if r["backend"] == be and r["op"] == op and r["axis"] == axis
         and int(r["k"]) == k and r["work_ratio"]]
    return min(v) if v else None


def fig_work_ratio(rows, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=False)
    for ax, (be, title) in zip(axes, BACKENDS):
        ops = [o for o in ORDER
               if best(rows, be, o, "OC", 2) or best(rows, be, o, "OH", 2)]
        x = range(len(ops))
        oc = [best(rows, be, o, "OC", 2) for o in ops]
        oh = [best(rows, be, o, "OH", 2) for o in ops]
        w = 0.36
        for i, (a, b) in enumerate(zip(oc, oh)):
            if a is not None:
                ax.bar(i - w / 2, a, w, color=C_OC, hatch=H_OC, edgecolor="white",
                       linewidth=0.6)
                ax.text(i - w / 2, a + 0.03, f"{a:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
            if b is not None:
                ax.bar(i + w / 2, b, w, color=C_OH, edgecolor="white", linewidth=0.6)
                ax.text(i + w / 2, b + 0.03, f"{b:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
        ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=0)
        # Left-anchored: the rightmost bars are the tall ones, and a
        # right-anchored note lands on top of them.
        ax.text(-0.48, 1.03, "1.0 = split is free", fontsize=7.5,
                color=MUTED, ha="left", va="bottom")
        # geometry annotation, so the reader can see WHY an op behaves as it does
        lab = []
        for o in ops:
            r = next(r for r in rows if r["op"] == o and r["backend"] == be)
            lab.append(f"{o.split('.')[-1]}\nOC{r['OC']} OH{r['OH']}\nK{r['KH']} S{r['SH']}")
        ax.set_xticks(list(x)); ax.set_xticklabels(lab, fontsize=7.5, color=INK2)
        ax.set_ylabel("work ratio\n(sum of tile times / unsplit)", fontsize=8.5,
                      color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        top = max([v for v in oc + oh if v is not None] + [2.1])
        ax.set_ylim(0, top * 1.16)
        style(ax)
    axes[0].legend(handles=[Patch(facecolor=C_OC, hatch=H_OC, edgecolor="white",
                                  label="OC split (output channels)"),
                            Patch(facecolor=C_OH, edgecolor="white",
                                  label="OH split (output rows)")],
                   loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    fig.suptitle("Splitting a conv 2 ways: spatial (OH) vs output-channel (OC)  "
                 "— dronet int8, AWS F2, tiles serial on one hart",
                 fontsize=11.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"  -> {out}")


def fig_anatomy(rows, out):
    """Where an OH split's cost actually goes, and how it scales with degree."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    C_HALO, C_GATH, C_SCAT = "#eb6834", "#2a78d6", "#1baf7a"
    # (a) bytes moved per OH split, halo broken out
    ax = axes[0]
    sub = [r for r in rows if r["axis"] == "OH" and int(r["k"]) == 2
           and r["overhead_us"] != ""]
    sub.sort(key=lambda r: (r["backend"], int(r["op"].split(".")[-1])))
    labs, y = [], []
    for i, r in enumerate(sub):
        g, h, sc = int(r["gather_B"]), max(0, int(r["halo_B"])), int(r["scatter_B"])
        base = (g - h) / 1024.0
        ax.barh(i, base, color=C_GATH, edgecolor="white", lw=0.6)
        ax.barh(i, h / 1024.0, left=base, color=C_HALO, edgecolor="white", lw=0.6)
        ax.barh(i, sc / 1024.0, left=base + h / 1024.0, color=C_SCAT,
                edgecolor="white", lw=0.6)
        tot = (g + sc) / 1024.0
        ax.text(tot + 2, i, f"{float(r['overhead_us']):+.0f} us "
                            f"({100*(float(r['work_ratio'])-1):+.0f}%)",
                va="center", fontsize=7.5, color=INK)
        labs.append(f"{r['op'].split('.')[-1]}  {r['backend'][:3]}  "
                    f"K{r['KH']}S{r['SH']}")
        y.append(i)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.5, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("bytes the split moves that the unsplit conv does not  (kB)",
                  fontsize=8.5, color=INK2)
    ax.set_xlim(0, max((int(r["gather_B"]) + int(r["scatter_B"])) / 1024.0
                       for r in sub) * 1.42)
    ax.set_title("The halo is not the cost — the NCHW gather/scatter is",
                 fontsize=9.5, color=INK, loc="left", pad=8)
    ax.legend(handles=[Patch(facecolor=C_GATH, label="gather: one pass over the input"),
                       Patch(facecolor=C_HALO, label="gather: HALO (rows read twice)"),
                       Patch(facecolor=C_SCAT, label="scatter: output row bands")],
              frameon=False, fontsize=7.5, loc="lower right")
    style(ax)
    # (b) degree scaling on the flagship conv
    ax = axes[1]
    ks, w = [2, 4], 0.36
    for j, (be, c) in enumerate((("rvv", C_OH), ("gemmini_q31", C_SCAT))):
        for i, k in enumerate(ks):
            v = best(rows, be, "conv_modules.0", "OH", k)
            if v is None:
                continue
            ax.bar(i + (j - 0.5) * w, v, w, color=c, edgecolor="white", lw=0.6)
            ax.text(i + (j - 0.5) * w, v + 0.02, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8, color=INK)
    # the OC alternative, which is a STAIRCASE in k for an OC=32 conv
    for i, k in enumerate(ks):
        pred = k / 1.0        # ceil(32/32)=1 slab unsplit, k slabs split
        ax.plot([i - 0.48, i + 0.48], [pred, pred], color=C_OC, lw=1.7, ls="--")
        ax.text(i, pred + 0.06, f"OC slab model: {pred:.0f}x", fontsize=7.5,
                color=C_OC, ha="center")
    m = best(rows, "rvv", "conv_modules.0", "OC", 2)
    if m:
        ax.scatter([0], [m], marker="D", s=34, color=C_OC, zorder=4,
                   edgecolor="white", lw=0.8)
        ax.text(0.05, m, f" OC k=2 measured {m:.2f}", fontsize=7.5, color=C_OC,
                va="center")
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k = {k}" for k in ks], fontsize=9, color=INK2)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylabel("work ratio", fontsize=8.5, color=INK2)
    ax.set_title("conv_modules.0 (OC 32, OH 56): cost vs split degree",
                 fontsize=9.5, color=INK, loc="left", pad=8)
    ax.legend(handles=[Patch(facecolor=C_OH, label="OH on rvv"),
                       Patch(facecolor=C_SCAT, label="OH on gemmini_q31")],
              frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"  -> {out}")


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1]))]
    d = sys.argv[2]; os.makedirs(d, exist_ok=True)
    fig_work_ratio(rows, os.path.join(d, "oh_vs_oc_work_ratio.png"))
    fig_anatomy(rows, os.path.join(d, "oh_split_anatomy.png"))


main()
