#!/usr/bin/env python3
"""Choose which pointwise (E) and pool (C) ops are WORTH splitting, and print
the `mk_split.py` arguments for them.

  mk_ec_args.py <net> <exdir> <quant> [--pair P] [--min-us N] [--nsplits K]

Why this is not just "split everything splittable". Registering the pointwise
and pool kinds made 47 + 1 op kinds tileable, but a dispatch is not free: on a
SERIAL run, where nothing overlaps and the whole span-minus-work gap is
handshake, the measured per-dispatch launch cost is 2.1-2.9 us across all four
networks and both backends (`experiments/sweep3net/res_*_serial*_base`):

    dronet/serialE   21 disp   gap    45 us -> 2.1 us/disp
    yolov8n/serialP 155 disp   gap   447 us -> 2.9 us/disp
    vint/serialE    605 disp   gap 1,681 us -> 2.8 us/disp

A K-way split replaces one dispatch with K, so it pays (K-1) launches to save
(K-1)/K of the op's cost. Break-even for K=2 is therefore an op costing about
2 x 2.5 = 5 us; below that the split is a guaranteed REGRESSION no matter how
perfectly it is scheduled. `--min-us` defaults to that.

The cost read is the MINIMUM over the two backends, not the maximum: an op is
only worth tiling if it clears the bar on the backend where it is already
cheapest, since that is the placement the scheduler will prefer.

THE ELASTICITY GATE, and why the cost floor alone is not enough. The floor
assumes an op's cost is proportional to `n`, so that halving `n` halves the
work. Measured on F2, yolov8n's `silu_s8`:

    rvv      57 calls  4.88 ms   85.6 us/call   ->  split: 114 calls  9.47 ms   83.1 us/call
    gemmini  57 calls 16.94 ms  297.1 us/call   ->  split: 114 calls 19.29 ms  169.2 us/call

On rvv the per-call cost did not move when the tile halved: the curated
`rvv_silu_s8_direct` rebuilds a 256-entry expf LUT on entry, and that rebuild
IS the cost for every `n` this network has (1,600 to 102,400 elements). So a
2-way split doubles silu's total work, and yolov8n on an rvv pair measured
146.0 ms sharded against 120.3 ms with the convs alone -- a 0.82x REGRESSION,
with bit-exact output. On gemmini the same op is element-bound and the same
split pays.

So a kind is proposed only if its cost actually varies with `n` on that
backend. The test is deliberately crude, because the alternative (a regression
over the kind's ops) fits garbage when the true slope is zero: over a >=4x
spread in `n`, the big op must cost at least `--elastic` times more than the
small one. Kinds with too few ops or too little spread are left alone -- the
gate can only reject on evidence, never on absence of it.
"""
import argparse, glob, json, os, re, sys

ROOT = "/scratch/dima/rose-infra/RoSE"
MB = f"{ROOT}/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
SWEEP = f"{ROOT}/experiments/sweep3net"

#: Kinds the E axis covers, mirrored from apply_split_hint._POINTWISE_KINDS.
#: Only the ones that actually appear in these networks are listed -- this is a
#: chooser, not a registry, and a kind it omits is simply never proposed.
E_KINDS = {"silu_s8", "silu_f16", "sigmoid_s8", "sigmoid_f16", "add_s8",
           "add_f16", "mul_s8", "mul_f16", "relu_s8", "relu_f16", "elu",
           "elu_f16", "gelu_s8", "gelu_f16", "relu6_s8", "tanh", "tanh_f16"}
C_KINDS = {"maxpool2d_s8"}


def serial_costs(net):
    """{dispatch_id: us} per backend, from the two single-hart cost runs."""
    out = {}
    for arm in ("serialE", "serialP"):
        p = next((x for x in glob.glob(f"{SWEEP}/res_{net}_{arm}_base/**/uartlog",
                                       recursive=True) if os.path.exists(x)), None)
        if not p:
            sys.exit(f"[mk_ec_args] no {arm} cost run for {net} -- run that arm first")
        d = {}
        for line in open(p, errors="ignore"):
            f = line.rstrip("\n").split(",")
            if re.match(r"^\d+,", line) and len(f) == 14:
                d[int(f[3])] = int(f[-1]) - int(f[-2])
        out[arm] = d
    return out


def per_call_bound_kinds(graph, cost, elastic):
    """{(kind, arm)} whose cost does NOT scale with `n` on that backend.

    See the module docstring. Only kinds with at least 4 measured ops and a 4x
    spread in `n` are judged; everything else is left alone, because this gate
    is allowed to reject on evidence and never on its absence.
    """
    by_kind = {}
    for o in graph["ops"]:
        did, sh = o.get("dispatch_id"), (o.get("shape") or {})
        if did is None or "n" not in sh:
            continue
        by_kind.setdefault(o["op"], []).append((int(sh["n"]), did))
    out = set()
    for kind, ops in by_kind.items():
        if len(ops) < 4:
            continue
        ops.sort()
        if ops[-1][0] < 4 * ops[0][0]:
            continue
        for arm in ("serialE", "serialP"):
            c_lo = cost[arm].get(ops[0][1])
            c_hi = cost[arm].get(ops[-1][1])
            if not c_lo or not c_hi:
                continue
            if c_hi < elastic * c_lo:
                out.add((kind, arm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net"); ap.add_argument("exdir"); ap.add_argument("quant")
    ap.add_argument("--min-us", type=float, default=5.0)
    ap.add_argument("--nsplits", type=int, default=2)
    ap.add_argument("--pair", choices=("rvvpair", "gempair", "hetero"),
                    default="hetero",
                    help="which machine pair the arm targets. It decides which "
                         "backends the elasticity gate consults: an rvv pair "
                         "has nowhere to put a tile that only scales on "
                         "gemmini, so a kind that is per-call bound on rvv is "
                         "refused outright there and allowed on a hetero pair "
                         "(default hetero, the permissive case)")
    ap.add_argument("--elastic", type=float, default=1.5,
                    help="over a >=4x spread in n, the largest op of a kind "
                         "must cost at least this multiple of the smallest, "
                         "or the kind is treated as per-CALL bound and not "
                         "split on that backend (default 1.5)")
    a = ap.parse_args()

    g = json.load(open(f"{MB}/examples/{a.exdir}/{a.quant}/generated/graph.json"))
    cost = serial_costs(a.net)
    arms = {"rvvpair": ("serialE",), "gempair": ("serialP",),
            "hetero": ("serialE", "serialP")}[a.pair]
    inelastic = per_call_bound_kinds(g, cost, a.elastic)
    if inelastic:
        print(f"# per-call bound: "
              f"{', '.join(f'{k} on {b}' for k, b in sorted(inelastic))} "
              f"(pair={a.pair} consults {'+'.join(arms)})", file=sys.stderr)
    args, skipped_cost, skipped_shape, skipped_flat = [], [], [], []
    for o in g["ops"]:
        did, kind, sh = o.get("dispatch_id"), o["op"], (o.get("shape") or {})
        if did is None:
            continue
        if kind in E_KINDS and "n" in sh:
            axis, total = "E", int(sh["n"])
        elif kind in C_KINDS and "C" in sh:
            axis, total = "C", int(sh["C"])
        else:
            continue
        if total % a.nsplits:
            # n_splits names a COUNT, so an indivisible axis is skipped rather
            # than given a hand-picked uneven partition here; `tile_sizes` is
            # how a caller asks for that deliberately.
            skipped_shape.append((did, kind, axis, total))
            continue
        if all((kind, arm) in inelastic for arm in arms):
            # Per-call bound on every backend THIS PAIR has, so no placement
            # can make the split pay. On a hetero pair, bound on one side only
            # is allowed through: the scheduler can put the tiles on the
            # backend where they scale.
            skipped_flat.append((did, kind))
            continue
        c = min(cost[arm].get(did, 0) for arm in arms)
        if c < a.min_us:
            skipped_cost.append((did, kind, c))
            continue
        w = total // a.nsplits
        args.append(f"{did}:{axis}:{','.join([str(w)] * a.nsplits)}")
    print(" ".join(args))
    print(f"# {len(args)} ops proposed; {len(skipped_cost)} below --min-us="
          f"{a.min_us} (launch-bound); {len(skipped_flat)} per-call bound; "
          f"{len(skipped_shape)} not divisible by {a.nsplits}", file=sys.stderr)


main()
