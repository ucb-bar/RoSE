#!/usr/bin/env python3
"""<net> sharded vs unsharded across every hardware configuration measured.

  plot_hw_sharding.py <net> [--out <png>]

Reads the fq uartlogs directly and re-checks each one's
`xpurt-runner: schedule=<tag>` line before using it -- fq copies results off the
run host's sim_slot_*, whose contents survive between jobs, so a cell can
silently collect a previous job's log.

TWO SPEEDUPS ARE PLOTTED, because they answer different questions and the gap
between them is itself a result:

  vs unsplit  -- same two harts in both arms. "What did SHARDING buy me?"
  vs 1 core   -- against the single-hart run. "What did the SECOND CORE buy me?"

They diverge by however much inter-op parallelism the unsplit graph already
had. DroNet is a near-serial chain, so its second rvv hart is ~95% idle when
unsplit and the two framings nearly coincide; yolov8n keeps its second hart 31%
busy unsplit, so its "vs unsplit" understates the total two-core benefit.
"""
import argparse, glob, os, re, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/scratch/dima/rose-infra/RoSE/experiments"
HW = [("serialE", "1x rvv\n(single hart)"), ("serialP", "1x gemmini\n(single hart)"),
      ("rvvpair", "rvv + rvv"), ("gempair", "gemmini + gemmini"), ("hetero", "rvv + gemmini")]
ARMS = [("base", "unsharded", "#4C6EF5"), ("shard", "sharded: conv/linear", "#F59F00"),
        ("shardec", "+ pointwise & pool", "#2F9E44")]


def cell(tag):
    for root in (f"{ROOT}/sweep3net", f"{ROOT}/sweep3net_prekfix"):
        for x in glob.glob(f"{root}/res_{tag}/**/uartlog", recursive=True) + [f"{root}/res_{tag}/uartlog"]:
            if not os.path.exists(x):
                continue
            t = open(x, errors="ignore").read()
            if not re.search(r"xpurt-runner: schedule=" + re.escape(tag) + r"\b", t):
                continue
            r = [l.split(",") for l in t.split("\n")
                 if re.match(r"^\d+,", l) and len(l.split(",")) == 14]
            if not r:
                continue
            iv = [(int(y[-2]), int(y[-1])) for y in r]
            e = re.search(r"max_abs_err=([0-9.eE+-]+)", t)
            return dict(ms=(max(b for _, b in iv) - min(a for a, _ in iv)) / 1000.0,
                        err=(e.group(1) if e else "?"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{ROOT}/sweep3net/plots/{a.net}_hw_sharding.png"
    D = {(p, arm): cell(f"{a.net}_{p}_{arm}") for p, _ in HW for arm, _, _ in ARMS}
    # single-hart reference: the faster of the two serial runs is NOT the point;
    # each pair is compared against ITS OWN backend's single hart.
    ref = {"rvvpair": D.get(("serialE", "base")), "gempair": D.get(("serialP", "base")),
           "hetero": D.get(("serialP", "base"))}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9.8), gridspec_kw={"height_ratios": [3, 2]})
    w = 0.26
    for j, (arm, label, colour) in enumerate(ARMS):
        xs, ys = [], []
        for i, (p, _) in enumerate(HW):
            c = D.get((p, arm))
            if c:
                xs.append(i + (j - 1) * w); ys.append(c["ms"])
        if xs:
            ax.bar(xs, ys, w * 0.92, label=label, color=colour, edgecolor="#222", linewidth=1.2)
            for x, y in zip(xs, ys):
                ax.text(x, y + max(ys) * 0.012, f"{y:.2f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(HW))); ax.set_xticklabels([l for _, l in HW])
    ax.set_ylabel("wall time (ms)  — lower is better")
    ax.set_title(f"{a.net} int8: sharded vs unsharded across every measured hardware configuration\n"
                 f"AWS F2 f2_quad_hetero_norose_tacit_q31_60mhz — schedules placed on MEASURED tile costs",
                 fontsize=12.5)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True); ax.legend(fontsize=9)

    # ---- two speedups side by side -------------------------------------
    series = [("vs unsplit on the same 2 harts", "#2F9E44", lambda p, c: D[(p, "base")]["ms"] / c["ms"]
               if D.get((p, "base")) else None),
              ("vs 1 core (that backend's single hart)", "#7048E8",
               lambda p, c: ref[p]["ms"] / c["ms"] if ref.get(p) else None)]
    for j, (label, colour, fn) in enumerate(series):
        xs, ys = [], []
        for i, (p, _) in enumerate(HW):
            if p.startswith("serial"):
                continue
            c = D.get((p, "shardec")) or D.get((p, "shard"))
            if not c:
                continue
            v = fn(p, c)
            if v:
                xs.append(i + (j - 0.5) * w * 1.2); ys.append(v)
        if xs:
            ax2.bar(xs, ys, w * 1.1, label=label, color=colour, edgecolor="#222", linewidth=1.0)
            for x, y in zip(xs, ys):
                ax2.text(x, y + 0.02, f"{y:.2f}x", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")
    ax2.axhline(1.0, color="#B00020", lw=1.2, ls="--")
    ax2.set_xticks(range(len(HW))); ax2.set_xticklabels([l for _, l in HW])
    ax2.set_ylabel("speedup of the best\nsharded arm")
    ax2.set_xlim(ax.get_xlim()); ax2.grid(axis="y", alpha=0.25); ax2.set_axisbelow(True)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.set_title("The gap between the two bars is inter-op parallelism the unsplit graph already had.",
                  fontsize=9.5, loc="left")

    # Name the single-hart baseline per backend and flag ANY arm that differs
    # from it. Saying "each equals its baseline" when one does not is exactly
    # the kind of caption that makes a figure untrustworthy.
    baseE = (D.get(("serialE", "base")) or {}).get("err")
    baseP = (D.get(("serialP", "base")) or {}).get("err")
    expect = {"rvvpair": baseE, "gempair": baseP, "hetero": baseP,
              "serialE": baseE, "serialP": baseP}
    odd = [f"{p}/{arm}={c['err']}" for (p, arm), c in sorted(D.items())
           if c and expect.get(p) is not None and c["err"] != expect[p]]
    line = (f"Accuracy: single-hart baselines are rvv={baseE}, gemmini={baseP}. "
            + ("Every plotted arm equals its own backend's baseline."
               if not odd else
               "ALL arms equal their baseline EXCEPT: " + ", ".join(odd) + "."))
    fig.text(0.012, 0.012, line, fontsize=8.4, va="bottom")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for (p, arm), c in sorted(D.items()):
        if c:
            print(f"    {p:<9}{arm:<9}{c['ms']:9.3f} ms  err={c['err']}")


main()
