#!/usr/bin/env python3
"""Side-by-side Gantt: unsharded on the left, sharded on the right, one row per
machine configuration, all six panels on ONE shared time axis.

  pair_gantt.py OUT.png "Title" "<label>|<base_tag>|<shard_tag>" ...

A blank tag leaves that half empty rather than dropping the row, so a missing
cell is visible instead of silently changing the layout.
"""
import sys, os, colorsys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_sp = importlib.util.spec_from_file_location(
    "g", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gantt.py"))
_g = importlib.util.module_from_spec(_sp)
_g.__name__ = "g"
src = open(_sp.origin).read().replace("\nmain()\n", "\n")
exec(compile(src, _sp.origin, "exec"), _g.__dict__)

def _pale(c, drop=0.68, lift=0.16):
    """Same HUE, much less saturation. The unsharded panel uses the pale form
    of whatever colour the op's tiles carry on the sharded side, so one op is
    followable across the two panels by hue alone."""
    h, sat, v = colorsys.rgb_to_hsv(*mcolors.to_rgb(c))
    return colorsys.hsv_to_rgb(h, sat * (1 - drop), min(1.0, v + lift))


out, title, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for s in specs:
    lab, b, sh = (s.split("|") + ["", ""])[:3]
    # An EMPTY tag means "this half cannot exist" (a 1-hart vanilla run has no
    # sharded counterpart); a NAMED tag that fails to load means the cell was
    # supposed to run and did not. Those are different facts and the chart must
    # not render them identically -- reading one as the other is exactly the
    # confusion this labelling caused.
    rows.append((lab,
                 (_g.load(b), bool(b)) if True else None,
                 (_g.load(sh), bool(sh))))
span = max((max(e for _, _, e, _, _, _ in d) - min(t for _, t, _, _, _, _ in d))
           for _, a, c in rows for d, _ in (a, c) if d)
nets = sorted({n for _, a, c in rows for d, _ in (a, c) if d for *_, n in d})
fig, axes = plt.subplots(len(rows), 2, figsize=(15, 1.75 * len(rows) + 2.4),
                         squeeze=False, sharex=True)
# ONE colour map for the WHOLE FIGURE, not per row. Building it per row keyed
# colours to sorted-index within that row's own split set -- and the sets differ
# per machine pair (dronet splits 14 ops on the rvv pair, 12 on the gemmini
# pair, 7 on hetero), so the same hue meant a different op in each row and the
# chart invited exactly the cross-config comparison it could not support.
# HATCHING extends the palette: with more split ops than colours the cycle
# wrapped and two ops in ONE panel shared a hue (rvvpair's purple was both
# conv_modules.0 and maxpool1).
_HATCH = ["", "///", "...", "xxx"]
_all_parents = sorted({nm.split(".tile_")[0]
                       for _, _a, _c in rows
                       for _d, _ in (_a, _c) if _d
                       for h, t, e, nm, op, net in _d if ".tile_" in nm})
pcol = {par: _g.TILE_CYCLE[i % len(_g.TILE_CYCLE)]
        for i, par in enumerate(_all_parents)}
phat = {par: _HATCH[(i // len(_g.TILE_CYCLE)) % len(_HATCH)]
        for i, par in enumerate(_all_parents)}

for r, (lab, a, c) in enumerate(rows):
    for col, (d, expected) in ((0, a), (1, c)):
        ax = axes[r][col]
        if not d:
            msg = ("MISSING — cell did not complete" if expected
                   else "n/a — single hart, nothing to shard")
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8,
                    color=("#b34" if expected else "#bbb"),
                    style=("normal" if expected else "italic"),
                    transform=ax.transAxes)
            ax.set_yticks([]); ax.set_xlim(0, span * 1.02)
            for sp2 in ("top", "right", "left"): ax.spines[sp2].set_visible(False)
            continue
        t0 = min(t for _, t, _, _, _, _ in d)
        harts = sorted({h for h, _, _, _, _, _ in d})
        busy = {h: 0.0 for h in harts}
        # TILES OF ONE OP SHARE A COLOUR, and are joined by a line when they
        # land on different harts. Without this every block is the same shade
        # and the chart cannot answer the question it exists to answer -- which
        # of these bars are the two halves of one split op, running at once?
        # Unsplit ops stay neutral grey so the split work is what stands out
        # (on a multi-network chart they keep their network colour instead,
        # because there identity matters more than splitness).
        multi_net = len(nets) > 1
        groups = {}
        for h, t, e, nm, op, net in d:
            if ".tile_" in nm:
                groups.setdefault(nm.split(".tile_")[0], []).append(
                    (harts.index(h), t - t0, e - t0))
        for h, t, e, nm, op, net in d:
            par = nm.split(".tile_")[0]
            hat = ""
            if ".tile_" in nm and par in pcol:          # a tile, sharded side
                bcol, ec, lw, hat = pcol[par], "#1a1a1a", 0.35, phat[par]
            elif par in pcol:                            # SAME op, unsharded side
                bcol, ec, lw, hat = _pale(pcol[par]), "white", 0.3, phat[par]
            else:                                        # never split ANYWHERE
                bcol = _g.NET_COL.get(net, _g.FALLBACK) if multi_net else "#C9CED4"
                ec, lw = "white", 0.25
            ax.barh(harts.index(h), max(e - t, span * 0.0004), left=t - t0,
                    height=0.62, color=bcol, edgecolor=ec, linewidth=lw, hatch=hat)
            busy[h] += e - t
        # Join the tiles of one parent across harts.
        for par, tiles in groups.items():
            if len(tiles) < 2 or len({y for y, _, _ in tiles}) < 2:
                continue
            tiles = sorted(tiles, key=lambda z: z[0])
            lcol = pcol[par]
            for (y0, s0, e0), (y1, s1, e1) in zip(tiles, tiles[1:]):
                ax.plot([(s0 + e0) / 2, (s1 + e1) / 2], [y0, y1],
                        color=lcol, lw=0.7, alpha=0.55, zorder=3,
                        solid_capstyle="round")
        mk = max(e for _, _, e, _, _, _ in d) - t0
        ax.set_yticks(range(len(harts)))
        ax.set_yticklabels([f"h{h} {_g.BE.get(h,'?')}" for h in harts], fontsize=7)
        ax.set_title(f"{'UNSHARDED' if col==0 else 'SHARDED'} — {lab}   "
                     f"{mk:.3f} ms   busy " +
                     ", ".join(f"{100*busy[h]/mk:.0f}%" for h in harts),
                     fontsize=8.5, loc="left")
        ax.set_xlim(0, span * 1.02)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        for sp2 in ("top", "right", "left"): ax.spines[sp2].set_visible(False)
for ax in axes[-1]: ax.set_xlabel("ms (all panels share one scale)", fontsize=9)
if len(nets) > 1:
    legend = [Patch(facecolor=_g.NET_COL.get(n, _g.FALLBACK), label=n) for n in nets]
else:
    legend = [
        Patch(facecolor="#C9CED4", edgecolor="white",
              label="op that is never split"),
        Patch(facecolor=_pale(_g.TILE_CYCLE[0]), edgecolor="white",
              label="pale = that op UNSHARDED (left)"),
        Patch(facecolor=_g.TILE_CYCLE[0], edgecolor="#1a1a1a",
              label="full = its tiles SHARDED (right), joined by a line"),
        Patch(facecolor=_pale(_g.TILE_CYCLE[1]), edgecolor="white", label=None),
        Patch(facecolor=_g.TILE_CYCLE[1], edgecolor="#1a1a1a", label=None)]
fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.005, 0.912),
           fontsize=8, frameon=False, ncol=len(legend))
fig.suptitle(title, fontsize=11, x=0.005, y=0.985, ha="left")
fig.text(0.005, 0.938, "hue identifies the OP and is consistent across every machine "
         "configuration; hatching extends the palette past 10 split ops",
         fontsize=8.5, ha="left", color="#444")
fig.tight_layout(rect=[0, 0, 1, 0.875])
fig.savefig(out, dpi=140)
print(f"  wrote {out}")
