#!/usr/bin/env python3
"""Audit curated kernels for portability fixes present in one algorithm of an
op family but missing in its siblings.

Motivation: rvv_conv2d_s8_rvv_oc_blocked.c hoists its input row offset to
size_t, with a comment saying 32-bit index arithmetic wraps for BSS buffers
above 0x80000000. Its sibling rvv_conv2d_s8_rvv_vsmul_vnclip.c never got that
fix and was silently wrong on yolov8n for the whole campaign. Any file that
indexes a large tensor with pure int arithmetic is exposed to the same class.
"""
import os, re, sys, collections, glob
sys.path.insert(0,"/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")

ROOT = os.environ["GLOBAL_CURATED_DIR"]

# An index expression built only from ints, used directly to subscript a
# tensor pointer. This is the pattern that wraps.
INT_INDEX = re.compile(r"\b(input|weight|output|in_row|in_plane|out_plane|a|b)\s*\[\s*\(*\s*\(?[A-Za-z_][A-Za-z0-9_]*\s*\*")
SIZE_T_HOIST = re.compile(r"\(size_t\)")
SIZE_T_ROWOFF = re.compile(r"size_t\s+\w*(row_off|off|base|idx)")

from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
OPS = sorted(KERNEL_SPECS, key=len, reverse=True)   # longest op name first

def family(fn):
    """<backend>_<op>_<algo>.c -> (op, algo). Split on the REGISTERED op
    names, longest first -- a naive right-split turns conv2d_s8_pc_direct
    into op=conv2d_s8_pc/algo=direct and hides the sibling relationship."""
    stem = fn[:-2]
    for be in ("gemmini_q31_rvv", "rvv_f16", "rvv_opu", "gemmini_q31", "gemmini", "rvv"):
        if stem.startswith(be + "_"):
            rest = stem[len(be)+1:]
            break
    else:
        return None, None
    for op in OPS:
        if rest == op:
            return op, "direct"
        if rest.startswith(op + "_"):
            return op, rest[len(op)+1:]
    return rest, ""

rows = collections.defaultdict(list)
for target in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, target)
    if not os.path.isdir(d): continue
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".c"): continue
        src = open(os.path.join(d, fn)).read()
        op, algo = family(fn)
        rows[(target, op)].append({
            "algo": algo, "file": fn,
            "size_t": len(SIZE_T_HOIST.findall(src)),
            "hoist": bool(SIZE_T_ROWOFF.search(src)),
            "int_idx": len(INT_INDEX.findall(src)),
        })

print("="*92)
print("SIBLING AUDIT: size_t index hoisting across algorithms of the same op")
print("="*92)
flagged = []
for (target, op), impls in sorted(rows.items()):
    if len(impls) < 2:
        continue
    have = [i for i in impls if i["size_t"] > 0]
    lack = [i for i in impls if i["size_t"] == 0]
    if have and lack:
        print(f"\n!! {target}/{op}: fix present in some siblings, MISSING in others")
        for i in impls:
            mark = "OK " if i["size_t"] else "GAP"
            print(f"   [{mark}] {i['algo']:<22} (size_t casts={i['size_t']:>2}, hoisted_off={i['hoist']}, int-indexed={i['int_idx']})")
        flagged.append((target, op, [i["file"] for i in lack]))

print("\n" + "="*92)
print("ALL curated kernels with ZERO size_t casts that index a tensor with int math")
print("="*92)
solo = []
for (target, op), impls in sorted(rows.items()):
    for i in impls:
        if i["size_t"] == 0 and i["int_idx"] > 0:
            print(f"   {target}/{i['file']:<44} int-indexed exprs={i['int_idx']}")
            solo.append((target, i["file"]))
print(f"\nSUMMARY: {len(flagged)} op families with an intra-family gap; "
      f"{len(solo)} files total using int-only tensor indexing")
