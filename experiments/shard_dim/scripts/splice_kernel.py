#!/usr/bin/env python3
"""Replace ONE kernel's body inside an already-generated kernels.c, in place.

  splice_kernel.py <kernels.c> <algorithm> <curated-source.c>

WHY THIS EXISTS INSTEAD OF JUST REGENERATING. `generate_kernels` re-runs the
whole curated selection, and selection is not stable across time: on
2026-09-02 a regeneration of `dronet_armB/int8/generated/gemmini_q31` silently
demoted `add_s8` from the curated `gemmini_resadd` to `reference_impl`, because
that kernel now FAILS the spike verify harness (max_abs_err=67) on a PRISTINE
tree -- it is a pre-existing condition unrelated to whatever the caller wanted
to change, and the on-disk `kernel_picks.json` predates it. Every sweep arm
already measured used `gemmini_resadd`, so regenerating to pick up an unrelated
kernel edit would have quietly changed the baseline out from under the
comparison. `mk_split.py`'s KEEP list exists for the same reason.

So: splice, do not regenerate. `kernels.c` is a plain concatenation of curated
sources, each preceded by `/* source: ... */` and `/* algorithm: <name> */`, so
one block is addressable by algorithm name.

The tool refuses anything ambiguous: the algorithm must match exactly one
block, and the block count must be unchanged afterwards.
"""
import re, sys

HDR = re.compile(r"^/\* source: [^\n]*\*/\n/\* algorithm: ([A-Za-z0-9_]+) \*/\n",
                 re.M)


KERNEL_ID = re.compile(r"\bkernel_[A-Za-z0-9_]+")


def _mangle_like(body: str, old_block: str, path: str) -> str:
    """Rename `kernel_*` symbols in `body` the way the generator renamed them.

    `generate_kernels` mangles every kernel symbol by MODEL name -- the curated
    source defines `kernel_maxpool2d_s8` and the generated block defines
    `kernel_maxpool2d_s8_dronet` -- and the Zephyr staging step then mangles
    again by backend. Splicing the curated source in verbatim therefore builds
    a translation unit whose function nobody calls, and the link fails with
    `undefined reference to kernel_maxpool2d_s8_dronet_gemmini_q31`. (It fails
    LOUDLY, which is the only reason this is a footnote and not a measurement
    taken against the wrong kernel.)

    The suffix is recovered from the block being replaced rather than passed in,
    so this cannot be given a model name that disagrees with the file.
    """
    old_ids = set(KERNEL_ID.findall(old_block))
    suffixes = set()
    for n in sorted(set(KERNEL_ID.findall(body))):
        cand = {o[len(n):] for o in old_ids if o.startswith(n) and len(o) > len(n)}
        # `kernel_foo` is a prefix of `kernel_foobar`, so keep only suffixes
        # that start with the separator the mangler uses.
        cand = {c for c in cand if c.startswith("_")}
        if cand:
            suffixes |= cand
    if not suffixes:
        return body                      # generator did not mangle this block
    # Prefer the SHORTEST suffix: for `kernel_add_s8` in a block that also
    # mentions `kernel_add_s8_dronet`, `_dronet` is the mangle and anything
    # longer is a different symbol that merely shares the prefix.
    sfx = min(suffixes, key=len)
    if len(suffixes) > 1:
        print(f"[splice] {path}: note -- candidate suffixes {sorted(suffixes)}, "
              f"using {sfx!r} (shortest)")
    return KERNEL_ID.sub(lambda m: m.group(0) + sfx
                         if m.group(0) + sfx in old_ids else m.group(0), body)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    path, algo, src = sys.argv[1:4]
    text = open(path).read()
    hits = list(HDR.finditer(text))
    if not hits:
        sys.exit(f"[splice] {path}: no '/* algorithm: */' headers -- not a "
                 f"generated kernels.c?")
    idx = [i for i, m in enumerate(hits) if m.group(1) == algo]
    if len(idx) != 1:
        sys.exit(f"[splice] {path}: algorithm {algo!r} matched {len(idx)} "
                 f"blocks (need exactly 1). Present: "
                 f"{sorted({m.group(1) for m in hits})}")
    i = idx[0]
    start = hits[i].start()
    end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
    old_block = text[start:end]
    body = open(src).read()
    if not body.endswith("\n"):
        body += "\n"
    body = _mangle_like(body, old_block, path)
    out = text[:start] + body + text[end:]
    n_after = len(list(HDR.finditer(out)))
    if n_after != len(hits):
        sys.exit(f"[splice] {path}: block count changed {len(hits)} -> "
                 f"{n_after}; the replacement source carries its own "
                 f"'/* algorithm: */' header count mismatch. Refusing.")
    open(path, "w").write(out)
    print(f"[splice] {path}: {algo} <- {src} "
          f"({end - start} bytes -> {len(body)}), {n_after} blocks intact")


main()
