#!/usr/bin/env python3
"""What sharding actually delivers once both harts really run at once.

  plot_shard_concurrency.py [--out-dir DIR]

plot_shard_benefit.py's per-operator panel is measured on SINGLE-HART runs of
the unsplit and split trees, so its "ideal 2-hart speedup" (unsplit/max tile)
is a ceiling: the tiles never contend, and the dispatch that launches the
second one is free.  This script measures the same operators on the REAL
two-hart runs and subtracts the two.

Per parent operator, from four measured runs:

  U_iso   unsplit cost, one hart              res_<net>_serial{E,P}_base
  t_iso   tile costs, one hart                res_<net>_serial{E,P}_shardec{R,G,H}
  U_cc    unsplit duration, the hart PAIR     res_<net>_<pair>_base
  t_cc    tile durations, the hart PAIR       res_<net>_<pair>_shardec
  S       wall span of the tile set           max(end) - min(start) over t_cc

  ideal    = U_iso / max(t_iso)      what the isolated profile promises
  achieved = U_cc  / S               what the pair actually did

and the two are joined by an identity, not a model:

  achieved = ideal x (U_cc/U_iso) x (max t_iso / max t_cc) x (max t_cc / S)
                     ^baseline        ^contention            ^launch+sync

so every microsecond of the shortfall is attributed to a term that was
separately measured.  The baseline term is the unsharded operator's own
slowdown on a busy pair -- it is not a sharding cost, and it is the reason a
few operators come out ABOVE their isolated ideal.
"""
import argparse, collections, glob, os, math, re, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/scratch/dima/rose-infra/RoSE/experiments/sweep3net"
GEM_HARTS = {0, 1}                       # harts 0/1 Rocket+Gemmini, 2/3 Rocket+Saturn


def rows(tag):
    for x in [f"{ROOT}/res_{tag}/uartlog"] + glob.glob(f"{ROOT}/res_{tag}/**/uartlog", recursive=True):
        if not os.path.exists(x):
            continue
        t = open(x, errors="ignore").read()
        if not re.search(r"xpurt-runner: schedule=" + re.escape(tag) + r"\b", t):
            continue
        r = [l.split(",") for l in t.split("\n")
             if re.match(r"^\d+,", l) and len(l.split(",")) == 14]
        r = [x for x in r if int(x[11]) >= 0]      # kernel-less alias ops
        if r:
            return r
    return None


def ops(tag):
    """{parent: {"kind":.., "tiles":{tile_idx: (hart, start, end)}}}

    Keyed on the tile index, not appended in trace order: the isolated arms run
    every tile on one hart, so tile k there has to be matched to tile k here or
    a hetero split reads the wrong backend's cost.
    """
    r = rows(tag) or []
    out = collections.defaultdict(lambda: {"kind": None, "tiles": {}})
    for x in r:
        nm = x[5]
        p, _, t = nm.partition(".tile_")
        k = int(t) if t.isdigit() else 0
        out[p]["kind"] = x[4].replace("_s8", "").replace("2d", "")
        out[p]["tiles"][k] = (int(x[11]), int(x[12]), int(x[13]))
    return out


NETS = ["mlp_control", "dronet", "yolov8_nano", "vint"]
# pair -> (2-hart base, 2-hart shard, isolated unsplit by backend, isolated split by backend)
PAIRS = {
    "rvvpair": ("rvv + rvv",     "#1971C2", {"rvv": "serialE_base"},
                {"rvv": "serialE_shardecR"}),
    "gempair": ("gemmini + gemmini", "#E8590C", {"gem": "serialP_base"},
                {"gem": "serialP_shardecG"}),
    "hetero":  ("rvv + gemmini", "#6741D9", {"rvv": "serialE_base", "gem": "serialP_base"},
                {"rvv": "serialE_shardecH", "gem": "serialP_shardecH"}),
}


def collect():
    """One record per (net, pair, parent op) that is split and measurable."""
    recs = []
    for net in NETS:
        for pair, (_lbl, _col, iso_u_tags, iso_s_tags) in PAIRS.items():
            base, shard = ops(f"{net}_{pair}_base"), ops(f"{net}_{pair}_shardec")
            if not base or not shard:
                continue
            iso_u = {k: ops(f"{net}_{t}") for k, t in iso_u_tags.items()}
            iso_s = {k: ops(f"{net}_{t}") for k, t in iso_s_tags.items()}
            for p, e in shard.items():
                if len(e["tiles"]) < 2 or p not in base:
                    continue                       # not split on this pair
                tl = e["tiles"]
                t_cc = [b - a for _h, a, b in tl.values()]
                S = (max(b for _h, _a, b in tl.values())
                     - min(a for _h, a, _b in tl.values()))
                U_cc = base[p]["tiles"][0][2] - base[p]["tiles"][0][1]
                # isolated side: read tile k from the arm whose backend matches
                # the hart tile k landed on, so hetero is not averaged.
                bk = lambda h: "gem" if h in GEM_HARTS else "rvv"
                t_iso = []
                for k, (h, _a, _b) in sorted(tl.items()):
                    src = iso_s.get(bk(h)) or next(iter(iso_s.values()))
                    if p not in src or k not in src[p]["tiles"]:
                        t_iso = []
                        break
                    _hh, a, b = src[p]["tiles"][k]
                    t_iso.append(b - a)
                ubk = bk(base[p]["tiles"][0][0])
                usrc = iso_u.get(ubk) or next(iter(iso_u.values()))
                if not t_iso or p not in usrc or min(t_cc) <= 0 or S <= 0:
                    continue
                U_iso = usrc[p]["tiles"][0][2] - usrc[p]["tiles"][0][1]
                if U_iso <= 0 or max(t_iso) <= 0:
                    continue
                recs.append(dict(
                    net=net, pair=pair, op=p, kind=e["kind"], n=len(t_cc),
                    U_cc=U_cc, S=S, U_iso=U_iso,
                    max_t_cc=max(t_cc), max_t_iso=max(t_iso),
                    sum_t_cc=sum(t_cc), sum_t_iso=sum(t_iso),
                    work_iso=sum(t_iso) / U_iso, work_cc=sum(t_cc) / U_cc,
                    ideal=U_iso / max(t_iso), achieved=U_cc / S,
                    f_base=U_cc / U_iso,
                    f_cont=max(t_iso) / max(t_cc),
                    f_sync=max(t_cc) / S))
    return recs


MARK = {"mlp_control": "s", "dronet": "o", "yolov8_nano": "^", "vint": "D"}


def fig_concurrency(recs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.0),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for r in recs:
        a1.scatter(r["U_cc"], r["achieved"], s=42, alpha=.8,
                   c=PAIRS[r["pair"]][1], marker=MARK[r["net"]],
                   edgecolors="k", linewidths=.4, zorder=3)
    a1.set_xscale("log")
    a1.axhline(1.0, ls="--", c="#C92A2A", lw=1.4, zorder=2)
    a1.axhline(2.0, ls=":", c="#2F9E44", lw=1.6, zorder=2)
    a1.text(a1.get_xlim()[1], 2.0, " 2-hart ceiling", va="bottom", ha="right",
            fontsize=9, color="#2F9E44")
    a1.text(a1.get_xlim()[1], 1.0, " no benefit", va="top", ha="right",
            fontsize=9, color="#C92A2A")
    a1.axvspan(1, 12, color="#868E96", alpha=.13, zorder=1)
    a1.text(11, a1.get_ylim()[0] + .1, "below the ~5 us\nlaunch floor",
            fontsize=8.5, color="#495057", ha="right", va="bottom")
    a1.set_xlabel("unsharded operator duration on the same hart pair (us, log)")
    a1.set_ylabel("achieved speedup   U_cc / span(tiles)")
    a1.set_title("Measured on the two-hart runs: what sharding actually delivered",
                 fontsize=11.5)
    a1.grid(alpha=.25, zorder=0)
    a1.legend(handles=[Patch(facecolor=c, label=l) for _p, (l, c, _u, _s) in PAIRS.items()]
              + [plt.Line2D([], [], ls="", marker=MARK[n], c="#495057", label=n)
                 for n in NETS if any(r["net"] == n for r in recs)],
              fontsize=8.5, ncol=2, loc="lower right")

    # Keep the kinds with enough operators to have a stable median, and only
    # the ones every pair actually split -- a missing bar is not a zero.
    cnt = collections.Counter(r["kind"] for r in recs)
    kinds = [k for k, n in cnt.items() if n >= 8
             and all(any(r["kind"] == k and r["pair"] == p for r in recs) for p in PAIRS)]
    kinds.sort(key=lambda k: -cnt[k])          # the kinds that carry the runtime
    kinds = kinds[:9]
    w, xs = .26, list(range(len(kinds)))
    for i, (pair, (lbl, col, _u, _s)) in enumerate(PAIRS.items()):
        pos, vals = [], []
        for x, k in zip(xs, kinds):
            v = [r["achieved"] for r in recs if r["kind"] == k and r["pair"] == pair]
            if v:
                pos.append(x + (i - 1) * w); vals.append(statistics.median(v))
        b = a2.bar(pos, vals, w, color=col, label=lbl, edgecolor="k", linewidth=.5)
        a2.bar_label(b, fmt="%.2f", fontsize=7, padding=1, rotation=90)
    a2.axhline(1.0, ls="--", c="#C92A2A", lw=1.4)
    a2.axhline(2.0, ls=":", c="#2F9E44", lw=1.6)
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{k}\n(n={cnt[k]})" for k in kinds], fontsize=8.5,
                       rotation=18, ha="right")
    a2.set_ylim(0, 2.45)
    a2.set_ylabel("median achieved speedup")
    a2.set_title("By operator kind and hart pair", fontsize=11.5)
    a2.legend(fontsize=8.5, loc="upper right"); a2.grid(axis="y", alpha=.25)
    fig.suptitle("Concurrency-informed view of sharding — every quantity read off the "
                 "two-hart traces, no model", fontsize=13, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(out, dpi=135)
    print("wrote", out)


def fig_gap(recs, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.0),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    lim = max([r["ideal"] for r in recs] + [r["achieved"] for r in recs]) * 1.06
    a1.plot([0, lim], [0, lim], ls="--", c="#495057", lw=1.3, zorder=2)
    a1.text(lim * .97, lim * .97, "isolated profile\nkept its promise",
            fontsize=8.5, ha="right", va="top", color="#495057")
    for r in recs:
        a1.scatter(r["ideal"], r["achieved"], s=42, alpha=.8, c=PAIRS[r["pair"]][1],
                   marker=MARK[r["net"]], edgecolors="k", linewidths=.4, zorder=3)
    a1.set_xlim(0, lim); a1.set_ylim(0, lim)
    a1.set_xlabel("ideal, from single-hart profiles   U_iso / max(tile)")
    a1.set_ylabel("achieved, on the hart pair   U_cc / span")
    a1.set_title("Promise against delivery, per operator", fontsize=11.5)
    a1.grid(alpha=.25, zorder=0)
    a1.legend(handles=[Patch(facecolor=c, label=l) for _p, (l, c, _u, _s) in PAIRS.items()],
              fontsize=8.5, loc="upper left")

    # the identity: achieved = ideal x f_base x f_cont x f_sync
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
    terms = [("f_base", "unsharded op is slower on a busy pair", "#ADB5BD"),
             ("f_cont", "tiles contend with each other", "#E8590C"),
             ("f_sync", "launch + sync (span > slowest tile)", "#5F3DC4")]
    xs = list(range(len(PAIRS)))
    # Geometric means: log(achieved) = log(ideal)+log(f_base)+log(f_cont)+
    # log(f_sync) holds per operator, so aggregating in log space makes the
    # waterfall land exactly on the achieved value instead of near it.
    ideal = [gm([r["ideal"] for r in recs if r["pair"] == p]) for p in PAIRS]
    run = list(ideal); thin: dict[int, int] = {}
    a2.scatter(xs, ideal, marker="_", s=2600, c="#212529", zorder=5, linewidths=2.2)
    for x, v in zip(xs, ideal):
        a2.text(x, v + .012, f"{v:.2f} ideal, isolated", ha="center", va="bottom",
                fontsize=9, color="#212529")
    for key, lbl, col in terms:
        f = [gm([r[key] for r in recs if r["pair"] == p]) for p in PAIRS]
        nxt = [r * k for r, k in zip(run, f)]
        for x, hi, lo, k in zip(xs, run, nxt, f):
            a2.bar(x, abs(hi - lo), .46, bottom=min(hi, lo), color=col,
                   edgecolor="k", linewidth=.6, zorder=3)
            if abs(hi - lo) > .022:
                a2.text(x, (hi + lo) / 2, f"x{k:.3f}", ha="center", va="center",
                        fontsize=8.5, color="w" if col != "#ADB5BD" else "#212529",
                        zorder=4)
            else:                                  # too thin to label inside
                thin.setdefault(x, 0)
                a2.text(x + .26, (hi + lo) / 2 + .022 * thin[x], f"x{k:.3f}",
                        ha="left", va="center", fontsize=8, color=col, zorder=4)
                thin[x] += 1
        run = nxt
    a2.scatter(xs, run, marker="_", s=2600, c="#C92A2A", zorder=5, linewidths=2.6)
    for x, v, i in zip(xs, run, ideal):
        a2.text(x, v - .022, f"{v:.2f} achieved\n{v / i:.0%} of the promise",
                ha="center", va="top", fontsize=9.5, fontweight="bold",
                color="#C92A2A", linespacing=1.35)
    a2.set_xticks(xs)
    a2.set_xticklabels([PAIRS[p][0] for p in PAIRS])
    a2.set_xlim(-.6, len(xs) - .4)
    a2.set_ylim(min(run) - .26, max(ideal) + .11)
    a2.set_ylabel("per-operator speedup (geometric mean)")
    a2.set_title("Where the ideal goes \u2014 each factor separately measured",
                 fontsize=11.5)
    a2.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in terms],
              fontsize=8.5, loc="lower left")
    a2.grid(axis="y", alpha=.25, zorder=0)
    fig.suptitle("The difference between the isolated promise and the concurrent result",
                 fontsize=13, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(out, dpi=135)
    print("wrote", out)



# The homogeneous pairs are the ones that map cleanly onto a single backend,
# so they are what "rvv" and "gemmini" mean in the contended view.
BACKEND_PAIR = {"rvv": ("rvvpair", "#1971C2"), "gemmini": ("gempair", "#E8590C")}


def fig_benefit_contended(recs, out):
    """plot_shard_benefit's per-operator view, with concurrency folded in.

    Same two questions -- what does splitting COST, and what does it BUY --
    but each answered twice: once from the single-hart profiles, and once from
    the runs where both harts were actually busy.  The distance between the
    two pairs of numbers is what the isolated profile cannot see.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.2),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    for bk, (pair, col) in BACKEND_PAIR.items():
        sub = [r for r in recs if r["pair"] == pair]
        a1.scatter([r["U_cc"] for r in sub], [r["work_iso"] for r in sub],
                   s=34, marker="o", facecolors="none", edgecolors=col,
                   linewidths=1.1, alpha=.75, zorder=3)
        a1.scatter([r["U_cc"] for r in sub], [r["work_cc"] for r in sub],
                   s=34, marker="o", c=col, edgecolors="k", linewidths=.35,
                   alpha=.8, zorder=4)
        for r in sub:                       # join the two readings of one op
            a1.plot([r["U_cc"], r["U_cc"]], [r["work_iso"], r["work_cc"]],
                    c=col, lw=.6, alpha=.35, zorder=2)
    a1.set_xscale("log")
    a1.axhline(1.0, ls="--", c="#C92A2A", lw=1.4, zorder=1)
    a1.set_xlabel("unsharded operator duration on the hart pair (us, log)")
    a1.set_ylabel("work ratio   sum(tiles) / unsplit\n(>1 = sharding ADDED work)")
    a1.set_title("The tax, isolated (hollow) against contended (filled)", fontsize=11.5)
    a1.grid(alpha=.25, zorder=0)
    a1.legend(handles=[Patch(facecolor=c, label=b) for b, (_p, c) in BACKEND_PAIR.items()]
              + [plt.Line2D([], [], ls="", marker="o", mfc="none", mec="#495057",
                            label="one hart at a time"),
                 plt.Line2D([], [], ls="", marker="o", c="#495057",
                            label="both harts busy")],
              fontsize=8.5, loc="upper right")

    cnt = collections.Counter(r["kind"] for r in recs
                              if r["pair"] in ("rvvpair", "gempair"))
    kinds = [k for k, n in cnt.items() if n >= 8]
    kinds.sort(key=lambda k: -cnt[k]); kinds = kinds[:8]
    w, xs = .36, list(range(len(kinds)))
    for i, (bk, (pair, col)) in enumerate(BACKEND_PAIR.items()):
        off = (i - .5) * w * 1.06
        for x, k in zip(xs, kinds):
            v = [r for r in recs if r["kind"] == k and r["pair"] == pair]
            if not v:
                continue
            ide = statistics.median(r["ideal"] for r in v)
            ach = statistics.median(r["achieved"] for r in v)
            # the isolated promise as an outline, the contended result filled
            # inside it: the exposed hatching IS what concurrency cost.
            a2.bar(x + off, ide, w, facecolor="none", edgecolor=col,
                   hatch="///", linewidth=1.2, zorder=3)
            a2.bar(x + off, ach, w, color=col, edgecolor="k", linewidth=.5, zorder=4)
            a2.text(x + off, ide + .03, f"{ach:.2f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90, color=col, fontweight="bold")
            if ide - ach > .015:
                a2.text(x + off, ide + .30, f"\u2212{(1 - ach / ide):.0%}", ha="center",
                        va="bottom", fontsize=7, rotation=90, color="#C92A2A")
    a2.axhline(2.0, ls=":", c="#2F9E44", lw=1.6, zorder=1)
    a2.axhline(1.0, ls="--", c="#C92A2A", lw=1.4, zorder=1)
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{k}\n(n={cnt[k]})" for k in kinds], fontsize=8.5)
    a2.set_xlim(-.7, len(xs) - .3)
    a2.set_ylim(0, 2.75)
    a2.set_ylabel("median 2-hart speedup")
    a2.set_title("The prize: promised (hatched outline) vs delivered (filled)",
                 fontsize=11.5)
    a2.legend(handles=[Patch(facecolor=c, label=b) for b, (_p, c) in BACKEND_PAIR.items()]
              + [Patch(facecolor="none", edgecolor="#495057", hatch="///",
                       label="ideal, isolated profile"),
                 Patch(facecolor="#495057", label="achieved, both harts busy")],
              fontsize=7.8, ncol=2, loc="lower center")
    a2.grid(axis="y", alpha=.25, zorder=0)
    fig.suptitle("Per-operator sharding benefit with contention included \u2014 "
                 "homogeneous pairs, so a backend means one thing", fontsize=13, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(out, dpi=135)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{ROOT}/plots")
    a = ap.parse_args()
    recs = collect()
    print(f"{len(recs)} split operators with all four measurements")
    for p in PAIRS:
        s = [r for r in recs if r["pair"] == p]
        if s:
            g = lambda k: math.exp(sum(math.log(r[k]) for r in s) / len(s))
            print(f"  {p:8s} n={len(s):3d}  ideal={g('ideal'):.3f}"
                  f"  achieved={g('achieved'):.3f}"
                  f"  base=x{g('f_base'):.3f}"
                  f"  cont=x{g('f_cont'):.3f}"
                  f"  sync=x{g('f_sync'):.3f}"
                  f"   [median achieved {statistics.median(r['achieved'] for r in s):.2f}]")
    os.makedirs(a.out_dir, exist_ok=True)
    fig_concurrency(recs, f"{a.out_dir}/shard_concurrency_perop.png")
    fig_gap(recs, f"{a.out_dir}/shard_concurrency_gap.png")
    fig_benefit_contended(recs, f"{a.out_dir}/shard_benefit_perop_contended.png")
