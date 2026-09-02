#!/usr/bin/env python3
"""DroNet sharded vs unsharded across every hardware configuration measured.

  plot_dronet_hw.py [--out <png>]

Reads the fq uartlogs directly (never a transcribed table) and checks each
one's `xpurt-runner: schedule=<tag>` line before using it -- fq copies results
off the run host's sim_slot_*, whose contents SURVIVE between jobs, so a cell
can silently collect a previous job's log.

PROVENANCE IS PLOTTED, not assumed. Cells measured before the 2026-09-02
per-hart gemmini kernel fixes and cells measured after are drawn differently,
because they are not the same binary: BACKENDS=gemmini_q31,rvv links BOTH
backends into every image, so a gemmini-only kernel change moves the .bss of a
run that never executes a gemmini kernel.
"""
import argparse, glob, os, re, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/scratch/dima/rose-infra/RoSE/experiments"
# The kernel-fix cutover. Cells older than this were built against the
# pre-per-hart gemmini kernels.
CUTOVER = datetime.datetime(2026, 9, 2, 12, 0)

HW = [("serialE",  "1x rvv\n(single hart)"),
      ("serialP",  "1x gemmini\n(single hart)"),
      ("rvvpair",  "rvv + rvv"),
      ("gempair",  "gemmini + gemmini"),
      ("hetero",   "rvv + gemmini")]
ARMS = [("base",    "unsharded",              "#4C6EF5"),
        ("shard",   "sharded: conv/linear",   "#F59F00"),
        ("shardec", "+ pointwise & pool",     "#2F9E44")]


def cell(tag):
    """(ms, err, mtime) for one tag, from any archive, or None."""
    for root in (f"{ROOT}/sweep3net", f"{ROOT}/sweep3net_prekfix"):
        cands = glob.glob(f"{root}/res_{tag}/**/uartlog", recursive=True)
        cands += [f"{root}/res_{tag}/uartlog"]
        for x in cands:
            if not os.path.exists(x):
                continue
            t = open(x, errors="ignore").read()
            if not re.search(r"xpurt-runner: schedule=" + re.escape(tag) + r"\b", t):
                continue
            rows = [l.split(",") for l in t.split("\n")
                    if re.match(r"^\d+,", l) and len(l.split(",")) == 14]
            if not rows:
                continue
            iv = [(int(y[-2]), int(y[-1])) for y in rows]
            e = re.search(r"max_abs_err=([0-9.eE+-]+)", t)
            return (max(b for _, b in iv) - min(a for a, _ in iv)) / 1000.0, \
                   (e.group(1) if e else "?"), \
                   datetime.datetime.fromtimestamp(os.path.getmtime(x))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/sweep3net/plots/dronet_hw_sharding.png")
    a = ap.parse_args()

    data = {(p, arm): cell(f"dronet_{p}_{arm}")
            for p, _ in HW for arm, _, _ in ARMS}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.5, 9.6),
                                  gridspec_kw={"height_ratios": [3, 2]})
    n_arm, w = len(ARMS), 0.26
    for j, (arm, label, colour) in enumerate(ARMS):
        xs, ys, hatches, edges = [], [], [], []
        for i, (p, _) in enumerate(HW):
            c = data.get((p, arm))
            if not c:
                continue
            xs.append(i + (j - (n_arm - 1) / 2) * w)
            ys.append(c[0])
            pre = c[2] < CUTOVER
            hatches.append("//" if pre else "")
            edges.append("#B00020" if pre else "#222222")
        if not xs:
            continue
        bars = ax.bar(xs, ys, w * 0.92, label=label, color=colour,
                      edgecolor=edges, linewidth=1.4)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.07, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(HW)))
    ax.set_xticklabels([l for _, l in HW])
    ax.set_ylabel("wall time (ms)  — lower is better")
    ax.set_title("DroNet int8: sharded vs unsharded across every measured hardware configuration\n"
                 "AWS F2, f2_quad_hetero_norose_tacit_q31_60mhz", fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="white", edgecolor="#B00020", hatch="//"))
    l.append("measured PRE kernel-fix (different binary)")
    ax.legend(h, l, loc="upper right", fontsize=9)

    # --- speedup vs unsharded on the SAME hardware -------------------------
    for j, (arm, label, colour) in enumerate(ARMS[1:], start=1):
        xs, ys = [], []
        for i, (p, _) in enumerate(HW):
            b, c = data.get((p, "base")), data.get((p, arm))
            if not (b and c):
                continue
            xs.append(i + (j - 1.5) * w * 1.15)
            ys.append(b[0] / c[0])
        if not xs:
            continue
        bars = ax2.bar(xs, ys, w * 1.05, label=label, color=colour,
                       edgecolor="#222222", linewidth=1.0)
        for x, y in zip(xs, ys):
            ax2.text(x, y + 0.015, f"{y:.2f}x", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    ax2.axhline(1.0, color="#B00020", lw=1.2, ls="--")
    ax2.text(len(HW) - 0.45, 1.005, "no gain", color="#B00020", fontsize=8.5)
    ax2.set_xticks(range(len(HW)))
    ax2.set_xticklabels([l for _, l in HW])
    ax2.set_ylabel("speedup vs unsharded\non the same hardware")
    ax2.set_xlim(ax.get_xlim())
    ax2.grid(axis="y", alpha=0.25); ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title("Single-hart configurations have no sharded arm by definition — "
                  "there is no second hart to shard onto.", fontsize=9.5, loc="left")

    fig.text(0.012, 0.010,
             "ACCURACY: every arm equals its own single-hart baseline (rvv 0 / gemmini 2).\n"
             "CAVEAT: dronet's rvv result is now known LAYOUT-SENSITIVE — a fresh 1x rvv build measures 4, and swapping only\n"
             "unused gemmini code flips it back to 0. So the rvv-involving bars are correct-looking but NOT yet trustworthy.\n"
             "The gemmini+gemmini column is the only one measured entirely post-kernel-fix.",
             fontsize=8.2, va="bottom", linespacing=1.45)
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=155)
    print(f"wrote {a.out}")
    for (p, arm), c in sorted(data.items()):
        if c:
            print(f"  {p:<9}{arm:<9}{c[0]:8.3f} ms  err={c[1]:<8}"
                  f"{'PRE-fix' if c[2] < CUTOVER else 'post-fix'}  {c[2]:%m-%d %H:%M}")


main()
