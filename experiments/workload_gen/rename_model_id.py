#!/usr/bin/env python3
"""Rename a generated ModelBlaster example tree's model id, in place.

  rename_model_id.py <example_dir>/<quant>/generated  <new_id> [old_id]

mk_split.py copies an example tree verbatim, so the SPLIT tree still calls
itself by the base model's name.  Two things key on that name and they pull in
opposite directions:

  * emit_dispatch_graph writes gen/vmfb/<graph name>/... , so an un-renamed
    split tree emits ON TOP OF the unsplit model's dispatch graph -- after
    which both arms of a sharded-vs-unsharded sweep schedule the SPLIT graph
    while the base arm builds unsplit ELFs.  Silent, and it invalidates the
    comparison the sweep exists to make.
  * generate_xpurt_main derives the C symbol prefix from the SCHEDULE's
    network name (`mid = _c_ident(net)`), so the schedule name and the tree's
    `model_<mid>_*` symbols have to agree or the link fails.

So the split tree needs a genuinely distinct id, in graph.json AND in the
generated C.  The model id is always an identifier PREFIX (`model_<mid>_...`,
`kernel_<op>_<mid>`, `<mid>_<weight>`) and never appears in a filename, so a
plain textual substitution over generated/**.{c,h,S} is exact.
"""
import json, os, re, sys


def rename(gen_dir: str, new_id: str, old_id: str | None = None) -> int:
    ir = os.path.join(gen_dir, "graph.json")
    g = json.load(open(ir))
    old = old_id or g["name"]
    if old == new_id:
        return 0                                  # idempotent: already renamed
    if not new_id.startswith(old):
        # A pure prefix extension keeps every derived identifier well-formed;
        # anything else would need the full symbol table to be safe.
        raise SystemExit(f"{new_id!r} is not an extension of {old!r}")
    g["name"] = new_id
    json.dump(g, open(ir, "w"), indent=1)

    # The id is an identifier PREFIX (model_<mid>_x, kernel_<op>_<mid>,
    # <mid>_<weight>), so it is followed by "_" as often as by a word break --
    # \b would skip exactly the occurrences that matter.  Match the literal
    # instead, and use a negative lookahead for the part being appended so a
    # re-run of a partially renamed tree cannot double-suffix it.
    tail = re.escape(new_id[len(old):])
    pat = re.compile(re.escape(old) + f"(?!{tail})")
    upat = re.compile(re.escape(old.upper()) + f"(?!{tail.upper()})")
    n = 0
    for root, _, files in os.walk(gen_dir):
        for f in files:
            if not f.endswith((".c", ".h", ".S")):
                continue
            p = os.path.join(root, f)
            s = open(p, errors="surrogateescape").read()
            t = upat.sub(new_id.upper(), pat.sub(new_id, s))
            if t != s:
                open(p, "w", errors="surrogateescape").write(t)
                n += 1
    return n


if __name__ == "__main__":
    d, nid = sys.argv[1], sys.argv[2]
    old = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"  {os.path.relpath(d)} -> {nid}: {rename(d, nid, old)} files rewritten")
