#!/usr/bin/env python3
"""What the 36-cell workload sweep says, and why.

  plot_wl_sweep.py [--out-dir DIR]

Left  what sharding bought, per workload family and machine pair, against the
      unsharded schedule on the SAME two harts.  Families whose base arm is
      distorted are marked rather than dropped: the reader should see them.
Right the mechanism.  Sharding cannot speed up a schedule that is already
      balanced -- it buys back idle hart.  So plot each cell's measured
      speedup against how idle the base schedule's quieter hart was, and the
      relationship should be a line, not a cloud.
"""
import argparse, collections, glob, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/scratch/dima/rose-infra/RoSE/experiments/wl_sweep"
FAMS = ["tight_loop", "control_mix", "scale_ladder", "saturation",
        "vint_intro", "vint_multi"]
PAIRS = [("rvvpair", "rvv + rvv", "#1971C2"),
         ("gempair", "gemmini + gemmini", "#E8590C"),
         ("hetero", "rvv + gemmini", "#6741D9")]
# Families whose base arm we know to be distorted, and by what.
CAVEAT = {"scale_ladder": "contains dronet_sb:\nscalar kernels",
          "tight_loop": "contains dronet_sa:\nscalar kernels",
          "vint_intro": "ViNT owns 100%\nof the makespan",
          "vint_multi": "ViNT owns 100%\nof the makespan"}


def cell(tag):
    """(makespan ms, {hart: busy fraction}) for one run, from its trace."""
    d = f"{ROOT}/res_wl_{tag}"
    ul = next((c for c in [f"{d}/uartlog"] + glob.glob(f"{d}/**/uartlog", recursive=True)
               if os.path.exists(c)), None)
    if not ul:
        return None
    r = [l.split(",") for l in open(ul, errors="ignore").read().split("\n")
         if re.match(r"^\d+,", l) and len(l.split(",")) == 14]
    r = [x for x in r if int(x[11]) >= 0]
    if not r:
        return None
    iv = [(int(x[12]), int(x[13])) for x in r]
    span = max(b for _, b in iv) - min(a for a, _ in iv)
    busy = collections.Counter()
    for x in r:
        busy[int(x[11])] += int(x[13]) - int(x[12])
    return span / 1000.0, {h: v / span for h, v in sorted(busy.items())}


def main(out_dir):
    data = {}
    for f in FAMS:
        for p, _lbl, _c in PAIRS:
            b, s = cell(f"{f}_{p}_greedy_base"), cell(f"{f}_{p}_greedy_shard")
            if b and s:
                data[(f, p)] = (b, s)
    print(f"{len(data)} cells with both arms")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.4),
                                 gridspec_kw={"width_ratios": [1.45, 1]})
    w, xs = .26, list(range(len(FAMS)))
    for i, (p, lbl, c) in enumerate(PAIRS):
        pos, val = [], []
        for x, f in zip(xs, FAMS):
            if (f, p) in data:
                (bm, _), (sm, _) = data[(f, p)]
                pos.append(x + (i - 1) * w); val.append(bm / sm)
        bars = a1.bar(pos, val, w, color=c, label=lbl, edgecolor="k", linewidth=.5,
                      alpha=.45 if False else 1.0)
        a1.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2, rotation=90)
    for x, f in zip(xs, FAMS):
        if f in CAVEAT:
            a1.axvspan(x - .45, x + .45, color="#C92A2A", alpha=.07, zorder=0)
            top = max((data[(f, p)][0][0] / data[(f, p)][1][0]
                       for p, _l, _c in PAIRS if (f, p) in data), default=1.0)
            a1.text(x, top + .17, CAVEAT[f], ha="center", va="bottom", fontsize=7.4,
                    color="#C92A2A", linespacing=1.25)
    a1.axhline(1.0, ls="--", c="#C92A2A", lw=1.4)
    a1.set_xticks(xs); a1.set_xticklabels(FAMS, fontsize=9.5)
    a1.set_ylim(0, 2.16); a1.set_ylabel("makespan speedup, sharded vs unsharded")
    a1.set_title("What sharding bought, per workload and hart pair\n"
                 "(shaded families have a distorted baseline — see the label)",
                 fontsize=11.5)
    a1.legend(fontsize=8.5, loc="upper left", ncol=1); a1.grid(axis="y", alpha=.25)

    for (f, p), ((bm, bb), (sm, sb)) in data.items():
        c = dict((q, col) for q, _l, col in PAIRS)[p]
        idle = 1.0 - min(bb.values())          # how idle the base's quieter hart was
        a2.scatter(idle, bm / sm, s=95, c=c, edgecolors="k", linewidths=.5,
                   marker="D" if f in CAVEAT else "o", zorder=3)
        a2.annotate(f.replace("_", "\n"), (idle, bm / sm), fontsize=6.6,
                    xytext=(6, -2), textcoords="offset points", color="#495057")
    a2.axhline(1.0, ls="--", c="#C92A2A", lw=1.4)
    a2.set_xlabel("idle time on the base schedule's QUIETER hart\n"
                  "(1 − busy fraction)")
    a2.set_ylabel("makespan speedup")
    a2.set_title("The mechanism: sharding buys back idle hart", fontsize=11.5)
    a2.grid(alpha=.25)
    a2.legend(handles=[Patch(facecolor=c, label=l) for _p, l, c in PAIRS]
              + [plt.Line2D([], [], ls="", marker="o", c="#495057", label="clean baseline"),
                 plt.Line2D([], [], ls="", marker="D", c="#495057", label="distorted baseline")],
              fontsize=8, loc="upper left")
    fig.suptitle("Sharded against unsharded across 6 workload families × 3 hart pairs "
                 "— 36 FPGA runs, all bit-matched to their base arm", fontsize=13, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .94])
    os.makedirs(out_dir, exist_ok=True)
    o = f"{out_dir}/wl_sweep_speedup.png"; fig.savefig(o, dpi=135); print("wrote", o)

    # ---- utilisation, the same data seen as occupancy ----
    fig2, ax = plt.subplots(figsize=(15, 5.6))
    labels, xs2 = [], []
    for i, (f, p) in enumerate(sorted(data, key=lambda k: (FAMS.index(k[0]), k[1]))):
        (bm, bb), (sm, sb) = data[(f, p)]
        c = dict((q, col) for q, _l, col in PAIRS)[p]
        for j, (busy, alpha, hatch) in enumerate([(bb, .38, None), (sb, 1.0, None)]):
            for k, h in enumerate(sorted(busy)):
                ax.bar(i + (j - .5) * .38 + (k - .5) * .17, busy[h], .16, color=c,
                       alpha=alpha, edgecolor="k", linewidth=.4, hatch=hatch)
        labels.append(f"{f}\n{p}"); xs2.append(i)
    ax.axhline(1.0, ls=":", c="#2F9E44", lw=1.5)
    ax.set_xticks(xs2); ax.set_xticklabels(labels, fontsize=7.4, rotation=30, ha="center")
    ax.set_ylabel("hart busy fraction of the makespan")
    ax.set_ylim(0, 1.30)
    for i in range(1, len(xs2)):
        ax.axvline(i - .5, c="#DEE2E6", lw=.8, zorder=0)
    ax.set_title("Per-hart occupancy: pale = unsharded, solid = sharded, two bars per arm "
                 "(one per hart)", fontsize=12)
    ax.grid(axis="y", alpha=.25)
    ax.legend(handles=[Patch(facecolor=c, label=l) for _p, l, c in PAIRS]
              + [Patch(facecolor="#495057", alpha=.38, label="unsharded"),
                 Patch(facecolor="#495057", label="sharded")],
              fontsize=8.5, loc="upper left", ncol=5)
    fig2.tight_layout()
    o2 = f"{out_dir}/wl_sweep_utilisation.png"; fig2.savefig(o2, dpi=135); print("wrote", o2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{ROOT}/plots")
    main(ap.parse_args().out_dir)
