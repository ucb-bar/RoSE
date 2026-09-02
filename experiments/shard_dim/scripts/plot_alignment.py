#!/usr/bin/env python3
"""Figures for the ALIGNMENT axis of OC operator splitting.

  plot_alignment.py <cells.csv> <tiles.csv> <plots_dir>

Three figures, in the order the argument is made:
  1. alignment_work_ratio.png   -- split cost / unsplit cost, by partition.
     The headline: aligned partitions cost 1.0x on rvv whether or not they are
     EVEN; misaligned ones cost the extra slab. Gemmini is flat.
  2. alignment_latency_us.png   -- the same thing in absolute microseconds,
     summed tile latency against the unsplit baseline bar.
  3. alignment_tile_slabs.png   -- the confound-free within-binary check:
     one tile's time against its own slab count, no baseline involved.

Colour encodes the alignment class and MARKER SHAPE repeats it, so the three
classes are never distinguished by colour alone.  Palette = slots 1/3/2 of the
documented categorical theme (blue / aqua / orange), which is the validated
all-pairs-safe three-slot subset.
"""
import csv, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_ALIGNED_EVEN = "#2a78d6"   # slot 1 blue
C_ALIGNED_UNEV = "#1baf7a"   # slot 3 aqua
C_MISALIGNED   = "#eb6834"   # slot 2 orange
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
M_EVEN, M_UNEV, M_MIS = "o", "^", "s"

BACKENDS = [("rvv", "rvv (Saturn V256) — quantum 32"),
            ("gemmini_q31", "gemmini_q31 — quantum 16")]


def cls(row):
    a = row["aligned"] == "True"
    e = row["even"] == "True"
    if a and e:   return "aligned, even", C_ALIGNED_EVEN, M_EVEN
    if a:         return "aligned, UNEVEN", C_ALIGNED_UNEV, M_UNEV
    return "misaligned", C_MISALIGNED, M_MIS


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)


def legend(ax, extra=()):
    h = [Line2D([], [], color=C_ALIGNED_EVEN, marker=M_EVEN, ls="", ms=7, label="aligned, even"),
         Line2D([], [], color=C_ALIGNED_UNEV, marker=M_UNEV, ls="", ms=7, label="aligned, UNEVEN"),
         Line2D([], [], color=C_MISALIGNED, marker=M_MIS, ls="", ms=7, label="misaligned")]
    h.extend(extra)
    ax.legend(handles=h, frameon=False, fontsize=8.2, labelcolor=INK2,
              loc="upper left", ncol=2)


def load(p):
    return list(csv.DictReader(open(p)))


def order_partitions(rows):
    """x order: group by OC, then aligned-even, aligned-uneven, misaligned."""
    seen = {}
    for r in rows:
        seen.setdefault((int(r["OC"]), r["partition"]), cls(r)[0])
    rank = {"aligned, even": 0, "aligned, UNEVEN": 1, "misaligned": 2}
    return [k for k in sorted(seen, key=lambda k: (k[0], rank[seen[k]], k[1]))]


def fig_work_ratio(cells, out):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.8), sharex=True)
    xs = order_partitions(cells)
    xi = {k: i for i, k in enumerate(xs)}
    for ax, (be, title) in zip(axes, BACKENDS):
        sub = [r for r in cells if r["backend"] == be and r["work_ratio"]]
        style(ax)
        ax.axhline(1.0, color=INK2, lw=1.2, ls="--", zorder=1)
        for k in {(int(r["OC"]), r["partition"]): r for r in sub}.values():
            x = xi[(int(k["OC"]), k["partition"])]
            ax.plot([x - .36, x + .36], [float(k["pred_slab"])] * 2,
                    color="#9a9a95", lw=2.6, solid_capstyle="butt", zorder=2)
        # deterministic fan so co-located measurements stay countable
        seen = {}
        for r in sub:
            key = (int(r["OC"]), r["partition"])
            n = seen.get(key, 0); seen[key] = n + 1
            x = xi[key] + (n - 1) * 0.15
            _, c, m = cls(r)
            ax.plot(x, float(r["work_ratio"]), marker=m, color=c, ms=7.5,
                    mec="white", mew=1.0, ls="", zorder=4)
        for i in range(1, len(xs)):
            if xs[i][0] != xs[i - 1][0]:
                ax.axvline(i - .5, color=GRID, lw=1.0)
        ax.set_ylabel("sum(tile time) / unsplit time", color=INK2, fontsize=9)
        ax.set_title(title + "   —   colour = alignment against THIS backend's quantum",
                     color=INK, fontsize=10.5, loc="left", pad=22)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.04)
        for i, (oc, _) in enumerate(xs):
            if i == 0 or xs[i - 1][0] != oc:
                j = max(k for k in range(len(xs)) if xs[k][0] == oc)
                ax.text((i + j) / 2, ax.get_ylim()[1], f"OC = {oc}", ha="center",
                        va="bottom", fontsize=9.5, color=INK)
    legend(axes[1], [Line2D([], [], color="#9a9a95", lw=2.6,
                            label="predicted = Σceil(w/Q) / ceil(OC/Q)"),
                     Line2D([], [], color=INK2, lw=1.2, ls="--",
                            label="1.0 = split is free")])
    axes[-1].set_xticks(range(len(xs)))
    axes[-1].set_xticklabels([f"{p}" for _, p in xs], rotation=32,
                             ha="right", fontsize=8.8)
    fig.suptitle("Split cost vs the unsplit baseline: ALIGNMENT, not evenness, "
                 "sets the rvv price of a partition",
                 fontsize=13.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.945, "dronet on AWS F2, every tile serial on one hart; one "
             "point per (operator, binary) measurement, fanned right where cells coincide",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out)


def fig_latency(cells, out):
    ops = sorted({(int(r["OC"]), r["op"]) for r in cells})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8.6))
    for ax, (be, title) in zip(axes, BACKENDS):
        sub = [r for r in cells if r["backend"] == be and r["work_ratio"]]
        style(ax)
        ax.grid(axis="y", lw=0)
        ax.grid(axis="x", color=GRID, lw=0.7)
        y, ylab = 0, []
        for oc, op in ops:
            rs = [r for r in sub if r["op"] == op]
            if not rs:
                continue
            base = float(rs[0]["t_unsplit_us"])
            rank = {"aligned, even": 0, "aligned, UNEVEN": 1, "misaligned": 2}
            rs.sort(key=lambda r: (rank[cls(r)[0]], r["partition"], r["binary"]))
            for r in rs:
                _, c, m = cls(r)
                ax.barh(y, float(r["sum_tiles_us"]), height=.66, color=c, zorder=2)
                ylab.append((y, f"{op}  {r['partition']}"))
                y += 1
            ax.plot([base, base], [y - len(rs) - .55, y - .45], color=INK,
                    lw=1.8, ls="--", zorder=5)
            ax.text(base, y - len(rs) - .75, f"unsplit {base:.0f}µs", fontsize=7.4,
                    color=INK, va="bottom", ha="center")
            y += 1.6
        ax.set_yticks([p for p, _ in ylab])
        ax.set_yticklabels([t for _, t in ylab], fontsize=7.0)
        ax.invert_yaxis()
        ax.set_xlabel("summed tile latency (µs, all tiles serial on one hart)",
                      color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=6)
        ax.margins(y=0.012)
    h = [Line2D([], [], color=C_ALIGNED_EVEN, marker="s", ls="", ms=8, label="aligned, even"),
         Line2D([], [], color=C_ALIGNED_UNEV, marker="s", ls="", ms=8, label="aligned, UNEVEN"),
         Line2D([], [], color=C_MISALIGNED, marker="s", ls="", ms=8, label="misaligned"),
         Line2D([], [], color=INK, lw=1.8, ls="--", label="unsplit baseline")]
    fig.legend(handles=h, frameon=False, fontsize=9, labelcolor=INK2, ncol=4,
               loc="upper left", bbox_to_anchor=(0.012, 0.945))
    fig.suptitle("Absolute split latency against the unsplit baseline "
                 "(a bar past the dashed line is work the split invented)",
                 fontsize=13.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out)


def fig_tiles(tiles, out):
    """Per-tile time against its OWN slab count -- one binary, no baseline,
    so neither the BSS layout nor the unsplit run can enter."""
    import statistics as st
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, (be, title) in zip(axes, BACKENDS):
        sub = [r for r in tiles if r["backend"] == be]
        style(ax)
        Q = 32 if be == "rvv" else 16
        xs = list(range(1, 129))
        ax.step(xs, [math.ceil(x / Q) for x in xs], where="post", color="#9a9a95",
                lw=2.0, zorder=1)
        devs = []
        for op in sorted({r["op"] for r in sub}):
            rs = [r for r in sub if r["op"] == op]
            u = st.median(float(r["us"]) / math.ceil(int(r["width"]) / Q) for r in rs)
            for r in rs:
                w = int(r["width"]); yv = float(r["us"]) / u
                al = (w % Q == 0)
                devs.append(abs(yv - math.ceil(w / Q)) / math.ceil(w / Q))
                ax.plot(w, yv, marker=(M_EVEN if al else M_MIS),
                        color=(C_ALIGNED_EVEN if al else C_MISALIGNED),
                        ms=6.5, mec="white", mew=.8, ls="", zorder=3)
        ax.set_xlabel(f"tile width w (output channels)", color=INK2, fontsize=9)
        ax.set_ylabel("tile time / that operator's one-slab time", color=INK2, fontsize=9)
        ax.set_title(f"{title}    median |deviation from ceil(w/Q)| = "
                     f"{100 * st.median(devs):.1f}%",
                     color=INK, fontsize=10, loc="left", pad=6)
        ax.set_xlim(0, 125)
    h = [Line2D([], [], color="#9a9a95", lw=2.0, label="ceil(w / Q)  — the slab model"),
         Line2D([], [], color=C_ALIGNED_EVEN, marker=M_EVEN, ls="", ms=7,
                label="w is a multiple of Q"),
         Line2D([], [], color=C_MISALIGNED, marker=M_MIS, ls="", ms=7,
                label="w is not")]
    axes[0].legend(handles=h, frameon=False, fontsize=8.2, labelcolor=INK2, loc="upper left")
    fig.suptitle("The mechanism, measured inside one binary with no baseline: "
                 "an rvv tile costs ceil(w/32) slabs, whatever w is",
                 fontsize=13.5, color=INK, x=0.012, ha="left", y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor="white")
    print("  ->", out)


def main():
    cells, tiles, d = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    os.makedirs(d, exist_ok=True)
    fig_work_ratio(cells, os.path.join(d, "alignment_work_ratio.png"))
    fig_latency(cells, os.path.join(d, "alignment_latency_us.png"))
    fig_tiles(tiles, os.path.join(d, "alignment_tile_slabs.png"))


main()
