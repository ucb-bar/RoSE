#!/usr/bin/env python3
"""Two views of what sharding buys: end-to-end, and per operator.

  plot_shard_benefit.py [--out-dir DIR]

E2E panel: measured wall time per (network, machine pair), unsharded against
each sharded arm, on the SAME two harts -- so the bar is what sharding bought,
with the hardware held constant.

PER-OPERATOR panel: from single-hart runs of the unsplit and split trees, so
each operator's tiles are timed in isolation with no concurrency mixed in.
Two quantities, and they answer different questions:

  work ratio    sum(tile costs) / unsplit cost. >1 means splitting ADDED work
                -- halo re-reads on OH, a re-read of the whole input per OC
                tile, a per-call fixed cost paid twice. This is the tax.
  ideal 2-hart  unsplit / max(tile cost). The best a perfectly balanced
                2-hart placement could do. 2.0 is the ceiling.

An operator is worth sharding when the second is comfortably above 1 AND the
first is close to it. Plotting both is the point: a large ideal speedup on an
operator whose work ratio is 2.0 is an illusion.
"""
import argparse, glob, os, re, collections, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/scratch/dima/rose-infra/RoSE/experiments/sweep3net"


def rows(tag):
    for x in glob.glob(f"{ROOT}/res_{tag}/**/uartlog", recursive=True) + [f"{ROOT}/res_{tag}/uartlog"]:
        if not os.path.exists(x):
            continue
        t = open(x, errors="ignore").read()
        if not re.search(r"xpurt-runner: schedule=" + re.escape(tag) + r"\b", t):
            continue
        r = [l.split(",") for l in t.split("\n")
             if re.match(r"^\d+,", l) and len(l.split(",")) == 14]
        r = [x for x in r if int(x[11]) >= 0]          # drop kernel-less alias ops
        if r:
            return r
    return None


def span(tag):
    r = rows(tag)
    if not r:
        return None
    iv = [(int(x[-2]), int(x[-1])) for x in r]
    return (max(b for _, b in iv) - min(a for a, _ in iv)) / 1000.0


def per_op(tag):
    """{parent_op_name: (op_kind, [tile costs us])}"""
    r = rows(tag)
    if not r:
        return {}
    agg = collections.defaultdict(list)
    kind = {}
    for x in r:
        par = x[5].split(".tile_")[0]
        agg[par].append(int(x[-1]) - int(x[-2]))
        kind[par] = x[4]
    return {k: (kind[k], v) for k, v in agg.items()}


NETS = ["mlp_control", "dronet", "yolov8_nano", "vint"]
PAIRS = [("rvvpair", "rvv + rvv"), ("gempair", "gemmini + gemmini"), ("hetero", "rvv + gemmini")]
ARMS = [("shard", "conv/linear", "#F59F00"),
        ("shardec", "+ pointwise & pool", "#2F9E44"),
        ("bestax", "+ measured best axis", "#1971C2"),
        ("bestaxG", "+ measured best axis", "#1971C2")]


def e2e(out):
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    xs, labels, seen = [], [], {}
    x = 0
    for net in NETS:
        for p, plab in PAIRS:
            b = span(f"{net}_{p}_base")
            if b is None:
                continue
            got = [(a, l, c) for a, l, c in ARMS if span(f"{net}_{p}_{a}")]
            if not got:
                continue
            n = len(got)
            for j, (a, l, c) in enumerate(got):
                s = span(f"{net}_{p}_{a}")
                w = 0.8 / n
                bar = ax.bar(x + (j - (n - 1) / 2) * w, b / s, w * 0.92, color=c,
                             edgecolor="#222", linewidth=1.0)
                ax.text(x + (j - (n - 1) / 2) * w, b / s + 0.012, f"{b/s:.2f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")
                seen[l] = c
            labels.append(f"{net.replace('_control','').replace('v8_nano','v8')}\n{plab.replace(' + ','+')}")
            xs.append(x); x += 1
    ax.axhline(1.0, color="#B00020", lw=1.2, ls="--")
    ax.text(len(xs) - 0.4, 1.006, "no gain", color="#B00020", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("speedup vs UNSHARDED on the same two harts")
    ax.set_title("End-to-end benefit of sharding, per network and machine pair\n"
                 "AWS F2 f2_quad_hetero_norose_tacit_q31_60mhz — measured, hardware held constant",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#222", label=l) for l, c in seen.items()],
              fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=150); print(f"wrote {out}")


def perop(out):
    # unsplit vs split, single hart, so no concurrency is folded in
    SRC = [("dronet", "shardecH"), ("yolov8_nano", "shardecH")]
    BE = {"serialE": ("rvv", "#4C6EF5"), "serialP": ("gemmini", "#E8590C")}
    pts = collections.defaultdict(list)     # (kind, backend) -> [(cost, work, ideal)]
    for net, split in SRC:
        for ap, (belab, _) in BE.items():
            base, sp = per_op(f"{net}_{ap}_base"), per_op(f"{net}_{ap}_{split}")
            for par, (kind, tiles) in sp.items():
                if len(tiles) < 2 or par not in base:
                    continue
                u = base[par][1][0]
                if u < 5:                      # below the ~5us launch floor
                    continue
                pts[(kind, belab)].append((u, sum(tiles) / u, u / max(tiles)))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    for (kind, be), v in sorted(pts.items()):
        c = BE["serialE"][1] if be == "rvv" else BE["serialP"][1]
        ax.scatter([p[0] for p in v], [p[1] for p in v], s=34, alpha=.75, color=c,
                   marker="o" if be == "rvv" else "^", edgecolors="#222", linewidths=.5)
    ax.axhline(1.0, color="#B00020", lw=1.2, ls="--")
    ax.text(ax.get_xlim()[0], 1.02, "no added work", color="#B00020", fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlabel("unsplit operator cost (us, log)")
    ax.set_ylabel("work ratio:  sum(tiles) / unsplit\n(>1 = sharding ADDED work)")
    ax.set_title("The tax: what splitting costs, per operator", fontsize=11, loc="left")
    ax.grid(alpha=.25); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=BE['serialE'][1], label="rvv"),
                       Patch(facecolor=BE['serialP'][1], label="gemmini")], fontsize=9)

    kinds = sorted({k for k, _ in pts})
    w = 0.38
    for j, be in enumerate(("rvv", "gemmini")):
        ys, xs2 = [], []
        for i, k in enumerate(kinds):
            v = pts.get((k, be))
            if not v:
                continue
            xs2.append(i + (j - .5) * w); ys.append(statistics.median(p[2] for p in v))
        c = BE["serialE"][1] if be == "rvv" else BE["serialP"][1]
        ax2.bar(xs2, ys, w * .9, color=c, edgecolor="#222", linewidth=1.0, label=be)
        for xx, yy in zip(xs2, ys):
            ax2.text(xx, yy + .02, f"{yy:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.axhline(2.0, color="#2F9E44", lw=1.2, ls=":")
    ax2.text(len(kinds) - .6, 2.02, "2-hart ceiling", color="#2F9E44", fontsize=8.5)
    ax2.axhline(1.0, color="#B00020", lw=1.2, ls="--")
    ax2.set_xticks(range(len(kinds)))
    ax2.set_xticklabels([k.replace("2d", "").replace("_s8", "") for k in kinds],
                        rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("median ideal 2-hart speedup\nunsplit / max(tile)")
    ax2.set_title("The prize: best case if the tiles are balanced", fontsize=11, loc="left")
    ax2.grid(axis="y", alpha=.25); ax2.set_axisbelow(True); ax2.legend(fontsize=9)
    fig.suptitle("Per-operator characterisation of sharding — single-hart profiles of the unsplit "
                 "and split trees, so no concurrency is folded in", fontsize=12, x=.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(out, dpi=150); print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{ROOT}/plots")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    e2e(f"{a.out_dir}/shard_benefit_e2e.png")
    perop(f"{a.out_dir}/shard_benefit_perop.png")


main()
