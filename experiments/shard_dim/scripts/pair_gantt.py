#!/usr/bin/env python3
"""Side-by-side Gantt: unsharded on the left, sharded on the right, one row per
machine configuration, all six panels on ONE shared time axis.

  pair_gantt.py OUT.png "Title" "<label>|<base_tag>|<shard_tag>" ...

A blank tag leaves that half empty rather than dropping the row, so a missing
cell is visible instead of silently changing the layout.
"""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_sp = importlib.util.spec_from_file_location(
    "g", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gantt.py"))
_g = importlib.util.module_from_spec(_sp)
_g.__name__ = "g"
src = open(_sp.origin).read().replace("\nmain()\n", "\n")
exec(compile(src, _sp.origin, "exec"), _g.__dict__)

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
fig, axes = plt.subplots(len(rows), 2, figsize=(15, 1.75 * len(rows) + 1.3),
                         squeeze=False, sharex=True)
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
        for h, t, e, nm, op, net in d:
            ax.barh(harts.index(h), max(e - t, span * 0.0004), left=t - t0,
                    height=0.62, color=_g.NET_COL.get(net, _g.FALLBACK),
                    edgecolor="white", linewidth=0.25)
            busy[h] += e - t
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
fig.legend(handles=[Patch(facecolor=_g.NET_COL.get(n, _g.FALLBACK), label=n)
                    for n in nets],
           loc="upper right", fontsize=8, frameon=False, ncol=len(nets))
fig.suptitle(title, fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(out, dpi=140)
print(f"  wrote {out}")
