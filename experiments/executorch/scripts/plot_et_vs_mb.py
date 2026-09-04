#!/usr/bin/env python3
"""ExecuTorch vs ModelBlaster on one RVV hart — end-to-end and per-operator.

  plot_et_vs_mb.py [--out-dir DIR]

Numbers are transcribed from experiments/executorch/FINDINGS.md sections 3 and
4, which is the vetted source: the end-to-end column comes from the
profiling-OFF build, and the per-operator column from the profiling-ON build
(fq 859), because XNNPACK's per-op logging inflates the whole-model bracket
~500x on an HTIF console. Mixing the two would be wrong, so they are kept
separate here as they are there. The per-op attribution reproduced across two
independent builds (fq 859 vs fq 861) to within 0.8% in every category.

Both sides are rdcycle on the same bitstream, one rvv hart:
ET = rdcycle around method->execute(); MB = the rdcycle sum of the per-op
profile from experiments/sweep3net/res_<model>_serialE_base.
"""
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ET, MB = "#7048E8", "#12B886"

# FINDINGS.md section 3
E2E = [("mlp_control\nfp32",   1_512_815,     578_541),
       ("dronet\nint8",       13_941_927,   7_855_964),
       ("yolov8_nano\nint8", 303_143_017, 167_132_539),
       ("vint\nint8",        673_136_658, 17_019_615_052)]

# FINDINGS.md section 4, "Op-for-op against ModelBlaster".
# None on the MB side = ExecuTorch-only cost with no ModelBlaster counterpart.
PEROP = {
    "dronet int8": [("convolutions\n(10 v 10)",   4_063_025,  7_209_291),
                    ("max pool",                  2_097_137,    239_338),
                    ("NCHW<->NHWC\ntransposes",   1_147_513,       None),
                    ("clamp / convert",             837_853,        451),
                    ("residual adds",                55_882,    356_525),
                    ("outside\ndelegates",        5_651_516,          0)],
    "yolov8_nano int8": [("convolutions\n(63 v 63)", 77_864_014, 155_470_762),
                         ("SiLU\n(Sigmoid+Mul)",     80_226_946,   4_794_103),
                         ("max pool",                   160_741,   3_655_642),
                         ("transposes /\nconverts",   9_776_784,        None),
                         ("outside\ndelegates",     133_194_950,           0)],
}


def fig_e2e(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(E2E)); w = .36
    a1.bar(x - w/2, [e for _, e, _ in E2E], w, color=ET, label="ExecuTorch",
           edgecolor="k", linewidth=.5)
    a1.bar(x + w/2, [m for _, _, m in E2E], w, color=MB, label="ModelBlaster",
           edgecolor="k", linewidth=.5)
    a1.set_yscale("log")
    a1.set_xticks(x); a1.set_xticklabels([n for n, _, _ in E2E], fontsize=9)
    a1.set_ylabel("warm execute, rdcycle (log)")
    a1.set_title("End to end, one rvv hart, same bitstream", fontsize=11.5)
    a1.legend(fontsize=9); a1.grid(axis="y", alpha=.25)
    for i, (_, e, m) in enumerate(E2E):
        a1.text(i, max(e, m) * 1.5, f"{e/m:.2f}x" if e > m else f"{e/m:.3f}x",
                ha="center", fontsize=9,
                color="#C92A2A" if e > m else "#2B8A3E", fontweight="bold")

    r = [e / m for _, e, m in E2E]
    cols = ["#C92A2A" if v > 1 else "#2B8A3E" for v in r]
    a2.barh(x, r, .5, color=cols, edgecolor="k", linewidth=.5)
    a2.axvline(1.0, ls="--", c="#212529", lw=1.4)
    a2.set_xscale("log"); a2.set_yticks(x)
    a2.set_yticklabels([n.replace("\n", " ") for n, _, _ in E2E], fontsize=9)
    a2.invert_yaxis()
    a2.set_xlabel("ET / MB   (>1 = ExecuTorch slower, log)")
    a2.set_title("Ratio. vint is not an ET win —\nMB's conv2d_s8_pc is kernel-limited there",
                 fontsize=10.5)
    a2.grid(axis="x", alpha=.25)
    for i, v in enumerate(r):
        a2.text(v * (1.15 if v > 1 else .85), i, f"{v:.2f}x" if v > .1 else f"{v:.3f}x",
                va="center", ha="left" if v > 1 else "right", fontsize=9)
    fig.suptitle("ExecuTorch vs ModelBlaster — end to end "
                 "(FINDINGS.md §3, profiling-off build)", fontsize=13, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(out, dpi=135)
    print("wrote", out)


def fig_perop(out):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for row, (model, rows) in enumerate(PEROP.items()):
        a, b = axes[row]
        y = np.arange(len(rows)); h = .36
        a.barh(y - h/2, [e for _, e, _ in rows], h, color=ET, edgecolor="k",
               linewidth=.5, label="ExecuTorch")
        a.barh(y + h/2, [(m if m else 0) for _, _, m in rows], h, color=MB,
               edgecolor="k", linewidth=.5, label="ModelBlaster")
        for i, (_, e, m) in enumerate(rows):
            if m is None:
                a.text(e * 1.15, i + h/2, "no MB counterpart", va="center",
                       fontsize=7.5, color="#495057", style="italic")
        a.set_xscale("log"); a.set_yticks(y)
        a.set_yticklabels([n for n, _, _ in rows], fontsize=8.5)
        a.invert_yaxis(); a.grid(axis="x", alpha=.25)
        a.set_xlabel("rdcycle (log)")
        a.set_title(f"{model} — where the time goes", fontsize=11)
        if row == 0:
            a.legend(fontsize=8.5, loc="lower right")

        # signed delta: the thing the log plot cannot show
        d = [(n, e - (m or 0)) for n, e, m in rows]
        cols = ["#C92A2A" if v > 0 else "#2B8A3E" for _, v in d]
        b.barh(y, [v / 1e6 for _, v in d], .5, color=cols, edgecolor="k", linewidth=.5)
        b.axvline(0, c="#212529", lw=1.3)
        b.set_yticks(y); b.set_yticklabels([n for n, _, _ in rows], fontsize=8.5)
        b.invert_yaxis(); b.grid(axis="x", alpha=.25)
        b.set_xlabel("ET − MB, millions of cycles\n(left = ExecuTorch ahead)")
        net = sum(v for _, v in d) / 1e6
        b.set_title(f"net {net:+.1f} M cycles", fontsize=11)
    fig.legend(handles=[Patch(facecolor="#2B8A3E", label="ExecuTorch ahead"),
                        Patch(facecolor="#C92A2A", label="ExecuTorch behind")],
               fontsize=9, ncol=2, loc="lower center", frameon=False,
               bbox_to_anchor=(.5, -.004))
    fig.suptitle("Per operator (FINDINGS.md §4, profiling-on build fq 859) — "
                 "the convolutions are not the problem", fontsize=13, y=.985)
    fig.tight_layout(rect=[0, .035, 1, .95]); fig.savefig(out, dpi=135)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default="/scratch/dima/rose-infra/RoSE/experiments/executorch/plots")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    fig_e2e(f"{a.out_dir}/et_vs_mb_e2e.png")
    fig_perop(f"{a.out_dir}/et_vs_mb_perop.png")
