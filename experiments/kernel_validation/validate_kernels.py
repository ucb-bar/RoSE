#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# validate_kernels.py -- reproducible regression suite for the curated kernel
# library in ModelBlaster (<MB>/kernels/*.c) and the selections that ship it.
#
# WHY THIS EXISTS
#   Three classes of defect have shipped undetected in this tree:
#     1. STALE SELECTION. kernel_picks.json records what selection chose the
#        last time it ran. Re-running selection today can reject that kernel
#        and fall back to reference_impl -- so the shipped binary contains a
#        kernel that no longer passes its own gate, and nothing says so.
#     2. HART-UNSAFE WORKSPACE. A kernel with a shared mutable `static`
#        workspace is silently wrong the moment two harts run the same op
#        concurrently. gemmini_im2col_full_C produced max_abs_err=89 on
#        yolov8_nano's 2-gemmini-hart schedule with NOTHING split, for as
#        long as that arm existed. Found by accident.
#     3. STALE HEADER. Kernel headers carry measured claims ("CONFIRMED
#        BROKEN", "max_abs_err=0", cycle counts). They are not re-checked
#        when the kernel is rewritten, and a stale one caused a wrong
#        attribution.
#
# TWO TIERS
#   Tier 1 -- no FPGA, deterministic. Seconds for the static checks, ~25 min
#   for the full run with selection regen + isolation.
#     selftest  does the hart scanner actually FAIL the known-bad kernels?
#               (checks 5 pre-fix blobs out of git and 5 post-fix ones)
#     hart      static hart-safety scan of every kernels/*/*.c: which
#               mutable statics are per-hart, which are shared, and whether
#               a shared one is a data workspace / a parameter-keyed memo /
#               a write-once table + init flag. Cross-referenced against
#               recorded uartlogs to separate racy-and-reached from
#               racy-but-latent from reachability-not-established.
#     seed      the pre-seed baseline must be layout-correct. Calls the
#               pipeline's OWN curated_seed_for() but judges with an
#               independent predicate, so neutering the pipeline's filter
#               turns this check red instead of blinding it.
#     stale     re-run curated selection in a SCRATCH copy of each example
#               tree and diff the picks against the shipped
#               kernel_picks.json, capturing per-kernel PASS/FAIL and
#               max_abs_err.
#     isolate   attribute error to a KERNEL rather than to the chain: one
#               spike run per kernel with that kernel the only curated one
#               and every other op on reference_impl, plus one
#               all-reference run per (network, backend) for the floor.
#     class     accuracy_class declared vs. ISOLATED measurement.
#     headers   header claims vs. current measurements.
#   Tier 2 -- FPGA, the only tier that can catch concurrency.
#     For each (network, pair, arm): asserts the pair arm's max_abs_err
#     EQUALS that network's single-hart (serialE/serialP) value EXACTLY.
#     Not "smaller", not "PASS". One hart cannot race itself, so the serial
#     arm is ground truth and any difference is concurrency-dependent
#     output -- a defect even when it is small.
#
# WHY ATTRIBUTION NEEDS ISOLATION (the trap this suite is built around)
#   generate_kernels' curated verify is a WHOLE-MODEL check: it swaps ONE
#   candidate into a baseline dict, builds the whole graph and compares the
#   final output to the PyTorch golden. Its max_abs_err therefore belongs to
#   the CHAIN, not to the kernel under test. Two defects found by this suite
#   are both instances of forgetting that:
#     * the pre-seed installed an NHWC batchnorm into an NCHW graph, and
#       add_s8 -- alphabetically first -- was blamed at err=67 and demoted,
#       for as long as the pre-seed has existed;
#     * a naive accuracy_class check reads every one of vint's 21 curated
#       kernels as violating bit_exact at 0.328, when 0.328 is what the
#       ALL-REFERENCE model reads.
#   So: whole-model err == 0 is sound evidence a kernel is exact; whole-model
#   err > 0 is INDETERMINATE and needs an isolation run.
#
# HOW TO RUN
#   Analysis python (no build env needed for the static checks):
#     P=/scratch2/dima/miniforge3/envs/xpurt/bin/python
#     cd /scratch/dima/rose-infra/RoSE
#
#   Static Tier 1 (seconds, no build, safe to run any time):
#     $P experiments/kernel_validation/validate_kernels.py \
#         --tier1 --no-regen --checks selftest,hart,seed
#
#   Full Tier 1 (BUILDS -- see TREE SAFETY. ~25 min with -j 6):
#     $P experiments/kernel_validation/validate_kernels.py \
#         --tier1 -j 6 --backends gemmini_q31,rvv,rvv_f16
#
#   Re-score without rebuilding anything (uses results/regen_*.log,
#   results/iso_*.log and results/floors.json):
#     $P experiments/kernel_validation/validate_kernels.py --tier1 \
#         --reuse-regen --reuse-isolate --backends gemmini_q31,rvv,rvv_f16
#
#   Tier 2 -- score collected uartlogs (no queue time; the logs must be
#   named <net>_<pair>_<arm>.uartlog, which is what the sweep produces):
#     $P experiments/kernel_validation/validate_kernels.py --tier2 \
#         --tier2-uartlogs 'experiments/sweep3net/.../*.uartlog'
#
#   Tier 2 -- emit the literal FPGA sweep commands WITHOUT submitting
#   anything (read the queue etiquette in the generated file first):
#     $P experiments/kernel_validation/validate_kernels.py \
#         --tier2-emit-plan experiments/kernel_validation/tier2_sweep.sh
#
#   Root override (default /scratch/dima/rose-infra/RoSE, or $ROSE_ROOT):
#     --root /path/to/RoSE
#
# EXIT CODE CONTRACT
#   0  every enabled check PASSed (or was explicitly SKIPped)
#   1  at least one check FAILed  (a defect: a hart-unsafe kernel, a stale
#      selection, a contradicted header claim, a pair arm whose error differs
#      from single-hart)
#   2  the suite itself could not run (missing tree, build env, bad args)
#   A check that could not be evaluated reports SKIP and does NOT fail the
#   run; SKIPs are always counted in the summary table so partial coverage is
#   visible rather than silently green.
#
# TREE SAFETY -- the traps this script is written around
#   * Selection regen is NEVER run in place. --stale copies the example tree
#     (generated/ + cache/) to <root>/experiments/kernel_validation/scratch/
#     and regenerates THERE. Regenerating in place silently rewrites
#     kernels.c with a different kernel (that is how gemmini_resadd got
#     demoted) and corrupts every subsequent measurement of that tree.
#   * MB_DRIFT_ATOL is recorded, never changed. It legitimately affects
#     selection (it is what keeps gemmini_tiled_conv selected over the
#     3.45x-slower software im2col), so every staleness finding carries the
#     value it was produced under, taken from the arm's own run.sh. Raising
#     it to make something pass is forbidden.
#   * The BACKEND NAME IS NOT THE DIRECTORY NAME. _run_lib.sh auto-promotes
#     rvv -> rvv_f16 for a graph with fp16 ops but still writes into
#     generated/rvv, so examples/vint/int8/generated/rvv was built as
#     rvv_f16. Regenerating it as `rvv` makes every curated kernel fail to
#     BUILD, which reads like a library-wide defect and is pure artifact.
#     resolve_target() reads the target from kernel_picks.json instead.
#   * File enumeration is done with glob, never `ls` (ANSI colour in ls
#     output has corrupted paths in this tree before).
#   * No `pkill -f`.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DEFAULT_ROOT = os.environ.get("ROSE_ROOT", "/scratch/dima/rose-infra/RoSE")


def paths(root: str) -> dict:
    zcs = os.path.join(root, "soc/sw/xpu-rt/zephyr-chipyard-sw")
    mb = os.path.join(zcs, "modelblaster")
    return {
        "root": root,
        "zcs": zcs,
        "mb": mb,
        "kernels": os.path.join(mb, "kernels"),
        "examples": os.path.join(mb, "examples"),
        "log": os.path.join(root, "experiments/kernel_opt_log.jsonl"),
        "self": os.path.join(root, "experiments/kernel_validation"),
        "scratch": os.path.join(root, "experiments/kernel_validation/scratch"),
        "results": os.path.join(root, "experiments/kernel_validation/results"),
    }


# The (network, example-dir, quant) tuples that are actually measured. The
# suite reports coverage against the full 221-file / 101-selection library,
# so this list is the *measured* subset, not a claim of completeness.
NETWORKS = [
    # name,           example dir,          quant,  MB_DRIFT_ATOL for this arm
    ("dronet",        "dronet_armB",        "int8", "2"),
    ("yolov8_nano",   "yolov8_nano_armB",   "int8", "2"),
    ("mlp_control",   "mlp_control_armB",   "fp32", "2"),
    ("vint",          "vint",               "int8", None),
]
BACKENDS = ["gemmini_q31", "rvv"]

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


@dataclass
class Finding:
    check: str
    subject: str        # kernel file / (net,backend,op) / arm
    status: str         # PASS | FAIL | SKIP | WARN
    detail: str
    evidence: str = ""  # what measurement establishes it
    data: dict = field(default_factory=dict)


# ==========================================================================
# C source utilities
# ==========================================================================

def strip_c(src: str) -> str:
    """Blank out comments and string/char literals, preserving newline count
    and byte offsets so brace-depth counting stays accurate."""
    out = []
    i, n = 0, len(src)
    while i < n:
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src[i] in "\"'":
            q = src[i]
            j = i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            out.append(" " * (min(j + 1, n) - i))
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


_ATTR_KW = re.compile(r"\b(__attribute__|__aligned|__align|_Alignas|"
                      r"__declspec)\b")


def strip_attributes(decl: str) -> str:
    """Remove `__attribute__((aligned(64)))` and friends with BALANCED paren
    matching.

    A non-greedy `\\(\\(.*?\\)\\)` regex stops at the FIRST `))`, so
    `__attribute__((aligned(64)))` leaves a stray `)` behind and the
    declaration then looks like it ends in a parameter list -- i.e. like a
    function prototype -- and gets skipped. That false negative hid
    `ws_pad/ws_p0/ws_p1` in gemmini_conv2d_s8_gemmini_im2col_full_C.c, the
    exact kernel whose shared workspace produced max_abs_err=89 on
    yolov8_nano's 2-gemmini-hart schedule. Caught by --selftest.
    """
    out = []
    i, n = 0, len(decl)
    while i < n:
        m = _ATTR_KW.match(decl, i)
        if m:
            j = m.end()
            while j < n and decl[j] in " \t\n":
                j += 1
            if j < n and decl[j] == "(":
                par = 0
                while j < n:
                    if decl[j] == "(":
                        par += 1
                    elif decl[j] == ")":
                        par -= 1
                        if par == 0:
                            j += 1
                            break
                    j += 1
                i = j
                out.append(" ")
                continue
        out.append(decl[i])
        i += 1
    return "".join(out)


def declarator_names(head: str) -> list[str]:
    """Every declared object name in a (possibly multi-declarator) head.

    `static int64_t ta[256], tb[256];` declares TWO workspaces. Taking only
    the last identifier reported `tb` and silently missed `ta` -- both were
    raced by gemmini_q31_add_s8_gemmini_resadd before commit fc0316c.
    """
    # drop the storage-class / type prefix by splitting on top-level commas
    # and taking the identifier that precedes the first `[` or `=` in each.
    parts, depth, cur = [], 0, []
    for ch in head:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    names = []
    for p in parts:
        p = p.split("=")[0]
        ids = re.findall(r"[A-Za-z_]\w*", p.split("[")[0])
        # last identifier before the first subscript is the declarator name
        # (everything before it is storage class / type / qualifiers)
        if ids:
            names.append(ids[-1])
    return names


def _decl_end(src: str, start: int) -> int:
    """Index of the `;` (or `{` opening a function body) that ends the
    declaration starting at `start`. Skips over parens and over brace-
    delimited initialisers."""
    i, n = start, len(src)
    par = 0
    brace = 0
    seen_eq = False
    while i < n:
        c = src[i]
        if c == "(":
            par += 1
        elif c == ")":
            par -= 1
        elif c == "{":
            if par == 0 and not seen_eq:
                return i          # function body
            brace += 1
        elif c == "}":
            brace -= 1
        elif c == "=" and par == 0 and brace == 0:
            seen_eq = True
        elif c == ";" and par == 0 and brace == 0:
            return i
        i += 1
    return n


def find_static_objects(src_stripped: str) -> list[dict]:
    """Every `static` that declares an OBJECT (not a function) with mutable
    storage, at any scope. A function-local `static` has exactly the same
    lifetime and sharing as a file-scope one -- both are shared by every hart
    -- so both are reported, with `scope` recording which."""
    objs = []
    n = len(src_stripped)
    for m in re.finditer(r"\bstatic\b", src_stripped):
        s = m.start()
        depth = (src_stripped.count("{", 0, s) - src_stripped.count("}", 0, s))
        e = _decl_end(src_stripped, s)
        raw = src_stripped[s:e]
        # A function *body* ends the scan at `{`; a function *prototype* ends
        # at `;` with a trailing parameter list.
        is_body = e < n and src_stripped[e] == "{"
        decl = " ".join(strip_attributes(raw).split())
        # prototype test AFTER balanced attribute removal -- otherwise
        # `static int8_t ws[..] __attribute__((aligned(64)));` reads as a
        # function because it ends in `)`.
        is_proto = re.search(r"\)\s*$", decl) is not None
        if is_body or is_proto:
            continue
        head = decl.split("=")[0]
        head_type = head.split("[")[0]
        if re.search(r"\bconst\b", head_type):
            continue           # static const -- read-only, not a hazard
        if re.search(r"\b(?:__thread|_Thread_local)\b", head):
            continue           # genuinely per-thread storage
        line = src_stripped.count("\n", 0, s) + 1
        for name in declarator_names(decl.split("=")[0]) or ["?"]:
            objs.append({
                "name": name,
                "decl": decl[:200],
                "scope": "file" if depth == 0 else "function",
                "line": line,
            })
    return objs


def slot_macros(src: str) -> dict[str, bool]:
    """`#define <NAME> ... arch_proc_id()` -> per-hart slot selectors.

    A slot macro is defined TWICE in every kernel that has one: the SMP
    branch expands to arch_proc_id(), the `#else` branch to 0. Only the SMP
    branch decides hart-safety, so definitions are OR-ed, not overwritten.
    (Overwriting was this scanner's first bug: it reported all 25 correctly
    slotted workspaces as racy because it kept the `#else ... 0` definition.)
    """
    out: dict[str, bool] = {}
    for m in re.finditer(r"#\s*define\s+(\w+)(?:\([^)]*\))?\s+(.+)", src):
        name, body = m.group(1), m.group(2)
        out[name] = out.get(name, False) or _is_hart_expr(body)
    # `static inline int read_mhartid(void) { ... csr_read(mhartid) ... }`
    # is the SAME thing as a slot macro. rvv_opu's three conv kernels index
    # their scratch with `read_mhartid() & 1`; a macro-only scan called them
    # racy, which is a false positive.
    for m in re.finditer(r"static\s+(?:inline\s+)?[\w\s\*]+?(\w+)\s*\([^)]*\)"
                         r"\s*\{", src):
        name = m.group(1)
        body = src[m.end(): m.end() + 400]
        out[name] = out.get(name, False) or _is_hart_expr(body)
    return out


def _is_hart_expr(s: str) -> bool:
    return bool(re.search(r"arch_proc_id|_current_cpu|z_smp_|mhartid|"
                          r"hartid|cpu_id|\bsmp_", s))


def classify_kernel_statics(path: str) -> dict:
    """Classify a kernel file's shared mutable statics.

    Classes, in the vocabulary the task set:
      per_hart      indexed by a slot selector that resolves to arch_proc_id
                    under SMP -- SAFE
      workspace     shared mutable data buffer, no per-hart slot -- RACY
      memo          parameter-keyed cache -- RACY across differing params
      init_flag     write-once table + guard flag; safe ONLY if both the
                    table and the flag are per-hart, or the flag has
                    release/acquire pairing. A per-hart table with a SHARED
                    flag is a real RVWMO bug (the maxpool case).
      none          no shared mutable statics -- CLEAN
    """
    src = open(path, errors="replace").read()
    st = strip_c(src)
    objs = find_static_objects(st)
    macros = slot_macros(src)
    per_hart_macros = {k for k, v in macros.items() if v}
    # SLOTS enum sized by CONFIG_MP_MAX_NUM_CPUS?
    sized_by_cpus = bool(
        re.search(r"enum\s*\{\s*\w*SLOTS\s*=\s*CONFIG_MP_MAX_NUM_CPUS", src))
    smp_guarded = "CONFIG_MP_MAX_NUM_CPUS" in src

    findings = []
    for o in objs:
        name = o["name"]
        # every subscript expression of this object in the file
        uses = re.findall(re.escape(name) + r"\s*\[([^\]\[]*)\]", st)
        # the declaration's own first dimension is a SLOTS bound, not a use
        decl_dims = re.findall(r"\[([^\]\[]*)\]", o["decl"])
        first_dim = decl_dims[0].strip() if decl_dims else None
        real_uses = {u.strip() for u in uses if u.strip() != first_dim}
        def _hart_idx(expr: str) -> bool:
            ids = set(re.findall(r"[A-Za-z_]\w*", expr))
            return any(i in per_hart_macros for i in ids)
        indexed_per_hart = bool(real_uses) and all(
            _hart_idx(u) for u in real_uses)
        # A hart index MASKED to fewer slots than the build's hart count
        # aliases. rvv_opu's `g_conv_scratch[read_mhartid() & 1]` is safe on
        # a 2-hart build and aliases hart 0 with hart 2 on the 4-hart
        # quad-hetero registry (harness/backends/
        # firesim_chipyard_quad_hetero_q31.conf: CONFIG_MP_MAX_NUM_CPUS=4).
        masked = indexed_per_hart and any(
            re.search(r"[&%]\s*\d+", u) for u in real_uses)
        literal_bound = None
        if first_dim and first_dim.strip().isdigit():
            literal_bound = int(first_dim.strip())

        # ROLE (what the object is for) is orthogonal to SAFETY (whether it
        # is per-hart). The task's vocabulary needs both: an idempotent
        # write-once table is safe *as data* and still unsafe if its INIT
        # FLAG is shared, because RVWMO lets another hart observe the flag
        # set before the table's stores are visible.
        lname = name.lower()
        if "memo" in lname or lname.endswith("_cache"):
            role = "memo"            # parameter-keyed: racy across params
        elif ("init" in lname or lname.endswith("_ready")
              or lname.endswith("_done") or lname.endswith("_valid")):
            role = "init_flag"
        else:
            role = "workspace"       # mutable data buffer
        if not indexed_per_hart:
            safety = "shared"
        elif masked or (literal_bound is not None and literal_bound < 4):
            safety = "per_hart_masked"
        else:
            safety = "per_hart"
        findings.append({**o, "role": role, "safety": safety,
                         "indices": sorted(real_uses),
                         "slot_bound": first_dim,
                         "slot_macro_ok": indexed_per_hart})

    # RVWMO cross-check: an init FLAG that is shared while the table it
    # guards is not (or vice versa). Either way the publish is unordered.
    shared_flags = [f for f in findings
                    if f["role"] == "init_flag" and f["safety"] == "shared"]
    has_release = bool(re.search(
        r"__atomic_store|atomic_store|__sync_synchronize|"
        r"memory_order_release|__ATOMIC_RELEASE|barrier\(\)", src))
    mixed_flag_bug = bool(shared_flags) and not has_release

    racy = [f for f in findings if f["safety"] == "shared"]
    masked_f = [f for f in findings if f["safety"] == "per_hart_masked"]
    return {
        "path": path,
        "file": os.path.basename(path),
        "backend": os.path.basename(os.path.dirname(path)),
        "statics": findings,
        "n_statics": len(findings),
        "n_racy": len(racy),
        "n_masked": len(masked_f),
        "roles": sorted({f["role"] for f in findings}),
        "smp_guarded": smp_guarded,
        "slots_sized_by_cpus": sized_by_cpus,
        "mixed_flag_bug": mixed_flag_bug,
        "has_release_barrier": has_release,
        "verdict": ("clean" if not findings else
                    ("racy" if racy else
                     ("per_hart_masked" if masked_f else "per_hart_ok"))),
    }


# ==========================================================================
# Check 1 -- hart safety
# ==========================================================================

def check_hart_safety(P: dict, reached: dict) -> list[Finding]:
    out = []
    files = sorted(glob.glob(os.path.join(P["kernels"], "*", "*.c")))
    for f in files:
        r = classify_kernel_statics(f)
        key = os.path.basename(f)
        if r["verdict"] == "clean":
            out.append(Finding(
                "hart", key, PASS,
                "no shared mutable statics",
                evidence="static scan: 0 mutable static objects "
                         "(static const / static inline are not hits)",
                data=r))
            continue
        if r["verdict"] == "per_hart_ok":
            roles = "+".join(r["roles"])
            det = (f"{r['n_statics']} static {roles}(s), all indexed by a "
                   f"per-hart slot selector")
            if not r["slots_sized_by_cpus"]:
                out.append(Finding(
                    "hart", key, WARN,
                    det + " but the SLOTS bound is not "
                          "CONFIG_MP_MAX_NUM_CPUS",
                    evidence="static scan", data=r))
            else:
                out.append(Finding("hart", key, PASS, det,
                                   evidence="static scan: every subscript of "
                                            "every mutable static is a macro "
                                            "expanding to arch_proc_id()",
                                   data=r))
            continue
        if r["verdict"] == "per_hart_masked":
            names = [f"{s['name']}[{s['indices']}] bound={s['slot_bound']}"
                     for s in r["statics"]
                     if s["safety"] == "per_hart_masked"]
            out.append(Finding(
                "hart", key, WARN,
                f"per-hart index MASKED to a fixed slot count: "
                f"{'; '.join(names)} -- safe only while at most that many "
                f"harts dispatch this op; the quad-hetero registry sets "
                f"CONFIG_MP_MAX_NUM_CPUS=4 "
                f"(harness/backends/firesim_chipyard_quad_hetero_q31.conf), "
                f"which would alias hart 0 with hart 2",
                evidence="static scan; no measured schedule runs this "
                         "backend on >2 harts, so this is latent",
                data=r))
            continue
        racy_names = [s["name"] + f"({s['role']})"
                      for s in r["statics"] if s["safety"] == "shared"]
        rc = reached.get(key)
        if rc is None:
            reach_txt = ("racy, reachability UNKNOWN -- no measured "
                         "selection ships this kernel, so no uartlog "
                         "covers it")
        elif rc["n"] > 0:
            reach_txt = (f"RACY AND REACHED: {rc['n']} same-op interval "
                         f"overlap(s) on different worker_hart "
                         f"[{'; '.join(rc['where'][:2])}]")
        elif not rc.get("backend_observed"):
            reach_txt = ("racy, reachability NOT ESTABLISHED -- this kernel "
                         "is shipped, but no recorded uartlog contains a "
                         "single dispatch row for its core_kind, so nothing "
                         "here can say whether it is co-scheduled")
        else:
            reach_txt = ("RACY BUT LATENT: this kernel is shipped, its "
                         "core_kind does appear in recorded uartlogs, and "
                         "its op never overlaps on two harts there")
        out.append(Finding(
            "hart", key, FAIL,
            f"shared mutable static(s) with no per-hart slot: "
            f"{', '.join(racy_names)}; {reach_txt}",
            evidence="static scan + per-(network,core_kind,op) uartlog "
                     "overlap count",
            data={**r, "reached": rc}))
    return out


# ==========================================================================
# Self-test -- does the hart scanner actually have power?
# ==========================================================================
# A static scan that passes everything is worthless unless it is shown to
# FAIL the kernels that were known to be broken. These five are the fixes
# that landed this week; the scanner must flag every pre-fix blob and clear
# every post-fix one. Both directions found a real bug in this scanner:
#   * balanced-paren attribute stripping (ws_pad/ws_p0/ws_p1 were invisible)
#   * multi-declarator `static int64_t ta[256], tb[256];` (ta was invisible)
SELFTEST_CASES = [
    # fix commit, path
    ("11d53f9", "kernels/gemmini/gemmini_conv2d_s8_gemmini_im2col_full_C.c"),
    ("79c1740", "kernels/gemmini_q31/"
                "gemmini_q31_maxpool2d_s8_gemmini_tiled_conv_pool.c"),
    ("fc0316c", "kernels/gemmini_q31/gemmini_q31_add_s8_gemmini_resadd.c"),
    ("19d1d3a", "kernels/rvv/rvv_silu_s8_direct.c"),
    ("19d1d3a", "kernels/rvv/rvv_conv2d_s8_pc_direct.c"),
]


def run_selftest(P: dict) -> list[Finding]:
    import tempfile
    out = []
    for commit, rel in SELFTEST_CASES:
        name = os.path.basename(rel)
        try:
            pre = subprocess.run(
                ["git", "-C", P["mb"], "show", f"{commit}^:{rel}"],
                capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            out.append(Finding("selftest", name, SKIP,
                               f"cannot fetch pre-fix blob {commit}^:{rel}",
                               evidence=str(e)[:200]))
            continue
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, name)
            open(p, "w").write(pre)
            r_pre = classify_kernel_statics(p)
        r_now = classify_kernel_statics(os.path.join(P["mb"], rel))
        ok_pre = r_pre["verdict"] == "racy"
        ok_now = r_now["verdict"] in ("per_hart_ok", "clean")
        if ok_pre and ok_now:
            out.append(Finding(
                "selftest", name, PASS,
                f"pre-{commit} blob classified racy "
                f"({r_pre['n_racy']}/{r_pre['n_statics']} shared); "
                f"current classified {r_now['verdict']}",
                evidence=f"git show {commit}^:{rel}"))
        else:
            out.append(Finding(
                "selftest", name, FAIL,
                f"scanner lacks power: pre-fix verdict={r_pre['verdict']} "
                f"(want racy), current verdict={r_now['verdict']} "
                f"(want per_hart_ok/clean)",
                evidence=f"git show {commit}^:{rel}",
                data={"pre": r_pre, "now": r_now}))
    return out


# ==========================================================================
# Check 2 -- accuracy_class consistency
# ==========================================================================

ACC_RE = re.compile(r"/\*\s*accuracy_class\s*:\s*([a-zA-Z_]+)\s*\*/")
CLASS_ATOL = {"bit_exact": 0.0, "numeric_drift": 2.0, "approximate": 8.0}


def log_resolution(x: float) -> float:
    """Half-ulp of a value as the pipeline PRINTS it.

    generate_kernels logs max_abs_err with '%.3g', so a parsed 0.328 could
    be anything in [0.3275, 0.3285). Subtracting a full-precision floor
    (vint's all-reference run reports 0.327884674) from a 3-significant-
    figure error invents an excess of 1.15e-4 out of nothing, and every one
    of vint's 21 curated kernels then reads as violating its bit_exact
    declaration. Anything at or below this resolution is not a measurement.
    """
    import math
    if x == 0:
        return 0.0
    return 0.5 * 10 ** (math.floor(math.log10(abs(x))) - 2)


def declared_class(path: str) -> Optional[str]:
    m = ACC_RE.search(open(path, errors="replace").read())
    return m.group(1).lower() if m else None


ISOLATE_SH_MARKER = "MB_KV_ISOLATE"


def reference_floor(P: dict, net, exdir, quant, backend) -> Optional[float]:  # noqa: C901
    """Whole-model max_abs_err with EVERY op on its reference_impl.

    This is the floor any isolation measurement has to be read against: the
    reference chain is not itself exact against the PyTorch golden (dronet
    reads 2 LSB, vint 0.328), so a kernel whose isolated run reads the floor
    has contributed nothing.
    """
    target = resolve_target(P, exdir, quant, backend)
    code = f'''
import json, os, sys
sys.path.insert(0, {P["zcs"]!r})
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
from modelblaster.pipeline import backends as B
from modelblaster.pipeline.profile_kernel import build_and_run
ir = json.load(open({os.path.join(P["scratch"], f"{net}_{backend}", "generated", "graph.json")!r}))
ops = sorted({{o["op"] for o in ir["ops"] if o["op"] in KERNEL_SPECS}})
specs = [KERNEL_SPECS[o] for o in ops]
impls = {{sp.op: sp.reference_impl for sp in specs}}
r = build_and_run(impls, specs, backend=B.get({target!r}),
    model_dir={os.path.join(P["scratch"], f"{net}_{backend}", "generated", backend)!r},
    build_dir={os.path.join(P["scratch"], f"{net}_{backend}", "build", backend)!r},
    repo_root={P["mb"]!r}, harness_dir={os.path.join(P["mb"], "harness")!r},
    pristine=False,
    io_path={os.path.join(P["scratch"], f"{net}_{backend}", "generated", "io.npz")!r},
    timeout=3600, atol=1e9)
print("FLOOR", r.golden_max_abs_err)
'''
    sh = (f'set -uo pipefail\n'
          f'source {P["zcs"]}/scripts/activate_conda.sh >/dev/null 2>&1\n'
          f'source {P["zcs"]}/scripts/set_envvars_sdk.sh >/dev/null 2>&1\n'
          f'export PYTHONPATH={P["zcs"]}\nexport PATH=/usr/bin:$PATH\n'
          f'cd {P["mb"]}\npython - <<\'MBPY\'\n{code}\nMBPY\n')
    try:
        p = subprocess.run(["bash", "-c", sh], capture_output=True,
                           text=True, timeout=5400)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"FLOOR ([0-9.eE+-]+)", p.stdout)
    return float(m.group(1)) if m else None


def run_isolation(P, net, exdir, quant, backend, op, algo,
                  timeout=5400) -> Optional[float]:
    """Whole-model max_abs_err with ONLY this kernel curated and every other
    op on reference_impl. This is the ONLY attribution that is sound: the
    pipeline's own verify is a whole-model check whose baseline is the
    pre-seed, so its number belongs to the chain, not to the kernel."""
    tag = f"{net}_{backend}_{op}_{algo}"
    sc_dir = os.path.join(P["scratch"], "iso", tag)
    src_ir = os.path.join(P["examples"], exdir, quant, "generated")
    src_gen = os.path.join(src_ir, backend)
    ir_d = os.path.join(sc_dir, "generated")
    gen = os.path.join(ir_d, backend)
    cache = os.path.join(sc_dir, "cache", backend)
    build = os.path.join(sc_dir, "build", backend)
    cur = os.path.join(sc_dir, "curated", backend)
    if os.path.isdir(sc_dir):
        shutil.rmtree(sc_dir)
    for d in (gen, cache, cur):
        os.makedirs(d, exist_ok=True)
    for fn in ("graph.json", "io.npz", "weights.npz"):
        q = os.path.join(src_ir, fn)
        if os.path.exists(q):
            shutil.copy2(q, os.path.join(ir_d, fn))
    for q in glob.glob(os.path.join(src_gen, "*")):
        shutil.copy2(q, os.path.join(gen, os.path.basename(q)))
    target = resolve_target(P, exdir, quant, backend)
    cur = os.path.join(sc_dir, "curated", target)
    os.makedirs(cur, exist_ok=True)
    kfile = f"{target}_{op}_{algo}.c"
    ksrc = os.path.join(P["kernels"], target, kfile)
    if not os.path.exists(ksrc):
        return None
    shutil.copy2(ksrc, os.path.join(cur, kfile))
    sh = REGEN_SH.format(zcs=P["zcs"], mb=P["mb"], ir=ir_d, gen=gen,
                         target=target, quant=quant, build=build,
                         cache=cache,
                         drift="unset MB_DRIFT_ATOL").replace(
        f'{P["mb"]}/kernels', os.path.join(sc_dir, "curated"))
    try:
        p = subprocess.run(["bash", "-c", sh], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    log = p.stdout + p.stderr
    with open(os.path.join(P["results"], f"iso_{tag}.log"), "w") as fh:
        fh.write(log)
    for (vop, valgo), v in parse_regen_log(log).items():
        if vop == op and valgo == algo:
            return v["max_abs_err"]
    return None


ISO_LOG = re.compile(r"^iso_(?P<net>.+?)_(?P<backend>gemmini_q31_rvv|"
                     r"gemmini_q31|gemmini|rvv_f16|rvv_opu|rvv|ime)_"
                     r"(?P<op>.+)_(?P<algo>[^_]+(?:_[^_]+)*)\.log$")


def load_isolation(P: dict) -> dict:
    """Rebuild isolation measurements from the per-run logs on disk.

    Each isolation run leaves results/iso_<net>_<backend>_<op>_<algo>.log
    containing the pipeline's own verify line for the one curated kernel it
    exposed; the all-reference floors are in results/floors.json (or, for a
    run that predates that file, in the run log). Re-scoring is then free,
    and the numbers in a report can be re-derived from committed artifacts
    rather than taken on trust.
    """
    floors: dict = {}
    fj = os.path.join(P["results"], "floors.json")
    if os.path.exists(fj):
        for k, v in json.load(open(fj)).items():
            n, b = k.split("/", 1)
            floors[(n, b)] = v
    for lg in glob.glob(os.path.join(P["results"], "*run*.log")):
        for m in re.finditer(r"^  floor (\S+?)/(\S+?): (\S+)$",
                             open(lg, errors="replace").read(), re.M):
            try:
                floors.setdefault((m.group(1), m.group(2)),
                                  float(m.group(3)))
            except ValueError:
                pass
    out: dict = {}
    for lg in sorted(glob.glob(os.path.join(P["results"], "iso_*.log"))):
        txt = open(lg, errors="replace").read()
        vr = parse_regen_log(txt)
        if not vr:
            continue
        base = os.path.basename(lg)[4:-4]      # strip 'iso_' and '.log'
        # the (op, algo) come from the log body, so only the net/backend
        # prefix has to be recovered from the filename
        (op, algo), v = next(iter(vr.items()))
        suffix = f"_{op}_{algo}"
        if not base.endswith(suffix):
            continue
        head = base[: -len(suffix)]
        net, backend = None, None
        for _n, _e, _q, _d in NETWORKS:
            if head.startswith(_n + "_"):
                net, backend = _n, head[len(_n) + 1:]
                break
        if net is None or v["max_abs_err"] is None:
            continue
        target = backend
        for _n, _e, _q, _d in NETWORKS:
            if _n == net:
                target = resolve_target(P, _e, _q, backend)
        fn = f"{target}_{op}_{algo}.c"
        floor = floors.get((net, backend))
        prov = (f"isolation run: {net}/{backend}, {op}/{algo} the ONLY "
                f"curated kernel, every other op on reference_impl; floor = "
                f"all-reference model ({floor}). log {lg}")
        def _exc(e_, f_):
            return e_ if f_ is None else max(0.0, e_ - f_)
        prev = out.get(fn)
        if prev is None or _exc(v["max_abs_err"], floor) > _exc(prev[0],
                                                               prev[1]):
            out[fn] = (v["max_abs_err"], floor, prov)
    return out


def check_accuracy_class(P: dict, measured: dict,
                         isolated: dict) -> list[Finding]:
    """A kernel declaring bit_exact that measures nonzero error is a defect
    in one or the other -- but ONLY an isolated measurement can say which
    kernel the error belongs to.

    `measured` holds the pipeline's own verify numbers, which are
    WHOLE-MODEL: _verify swaps one candidate into the pre-seed baseline,
    builds the whole graph and compares the final output to the golden. So a
    nonzero number there is the chain's error, not the kernel's -- treating
    it as the kernel's is precisely the mis-attribution that made
    gemmini_resadd look broken at 67 when it is exact. Therefore:

      whole-model err == 0        -> the kernel is exact IN THIS CHAIN; a
                                     bit_exact claim is confirmed (sound: a
                                     nonzero kernel error cannot cancel to
                                     zero at the output for every element)
      whole-model err  > 0        -> INDETERMINATE without isolation
      isolated err available      -> judged against the reference floor for
                                     that (network, backend)
    """
    out = []
    for f in sorted(glob.glob(os.path.join(P["kernels"], "*", "*.c"))):
        key = os.path.basename(f)
        cls = declared_class(f)
        if cls is None:
            continue
        atol = CLASS_ATOL.get(cls)
        iso = isolated.get(key)
        if iso is not None:
            err, floor, prov = iso
            excess = None if floor is None else max(0.0, err - floor)
            if atol is None:
                out.append(Finding("class", key, WARN,
                                   f"unknown accuracy_class {cls!r}",
                                   evidence=prov))
            elif excess is None:
                out.append(Finding("class", key, SKIP,
                                   f"isolated err={err:g} but no reference "
                                   f"floor for this (network, backend)",
                                   evidence=prov))
            elif excess > max(atol, log_resolution(err)):
                out.append(Finding(
                    "class", key, FAIL,
                    f"declares {cls} (atol {atol}) but ISOLATED "
                    f"max_abs_err={err:g} against a reference floor of "
                    f"{floor:g} -- the kernel itself contributes "
                    f"{excess:g}",
                    evidence=prov,
                    data={"declared": cls, "isolated": err, "floor": floor}))
            else:
                out.append(Finding(
                    "class", key, PASS,
                    f"declares {cls}; ISOLATED max_abs_err={err:g} vs "
                    f"reference floor {floor:g} -- excess {excess:g} is "
                    f"within the {log_resolution(err):g} print resolution "
                    f"of the logged figure, so this kernel contributes no "
                    f"error of its own",
                    evidence=prov))
            continue
        if key not in measured:
            out.append(Finding("class", key, SKIP,
                               f"declares {cls}; not selected by any "
                               f"measured network this run",
                               evidence="no measurement exists"))
            continue
        err, prov = measured[key]
        if err is None:
            out.append(Finding("class", key, SKIP,
                               f"declares {cls}; verify produced no error "
                               f"value", evidence=prov))
        elif err == 0:
            out.append(Finding("class", key, PASS,
                               f"declares {cls}; the whole model reads "
                               f"max_abs_err=0 with this kernel selected, "
                               f"so it is exact in this chain",
                               evidence=prov))
        else:
            out.append(Finding(
                "class", key, SKIP,
                f"declares {cls}; INDETERMINATE -- the whole model reads "
                f"max_abs_err={err:g} with this kernel selected, but that "
                f"is the CHAIN's error (the pipeline's verify is a "
                f"whole-model check against a pre-seeded baseline), not "
                f"this kernel's. Re-run with --checks isolate to attribute "
                f"it.",
                evidence=prov))
    return out


def run_isolation_campaign(P: dict, regen_results: list[dict], jobs: int,
                           only_indeterminate: bool = True) -> dict:
    """Isolate the kernels whose attribution the cheap evidence cannot
    settle: one spike build+run per kernel, with that kernel the ONLY
    curated one and every other op on reference_impl, plus one
    all-reference run per (network, backend) for the floor.

    Default is targeted, not exhaustive: a kernel whose whole-model verify
    already reads 0 needs no isolation (it cannot be contributing error),
    so only the nonzero ones are paid for. --isolate-all overrides.
    """
    want: list[tuple] = []
    for r in regen_results:
        picks = ((r.get("new") or {}).get("picks")
                 or (r.get("shipped") or {}).get("picks") or {})
        for op, pk in picks.items():
            if pk.get("source") != "curated" or not pk.get("algorithm"):
                continue
            algo = pk["algorithm"]
            v = r["verify"].get((op, algo))
            err = v["max_abs_err"] if v else None
            if only_indeterminate and (err == 0 or err is None):
                continue
            want.append((r["net"], r["exdir"], r["quant"], r["backend"],
                         op, algo))
    if not want:
        print("[isolate] nothing indeterminate to isolate", flush=True)
        return {}
    cells = sorted({(w[0], w[1], w[2], w[3]) for w in want})
    print(f"[isolate] {len(want)} kernel(s) need attribution across "
          f"{len(cells)} (network, backend) cell(s); measuring the "
          f"all-reference floor for each first", flush=True)
    floors: dict = {}
    with ThreadPoolExecutor(max_workers=min(jobs, len(cells))) as ex:
        futs = {ex.submit(reference_floor, P, n, e, q, b): (n, b)
                for n, e, q, b in cells}
        for f in futs:
            try:
                floors[futs[f]] = f.result()
            except Exception as exc:                        # noqa: BLE001
                print(f"  floor {futs[f]} raised "
                      f"{type(exc).__name__}: {exc}")
                floors[futs[f]] = None
    for k, v in sorted(floors.items()):
        print(f"  floor {k[0]}/{k[1]}: {v}", flush=True)
    with open(os.path.join(P["results"], "floors.json"), "w") as fh:
        json.dump({f"{k[0]}/{k[1]}": v for k, v in floors.items()}, fh,
                  indent=1)
    out: dict = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_isolation, P, n, e, q, b, op, algo):
                (n, e, q, b, op, algo) for n, e, q, b, op, algo in want}
        for f in futs:
            n, e, q, b, op, algo = futs[f]
            try:
                err = f.result()
            except Exception as exc:                        # noqa: BLE001
                print(f"  isolate {n}/{b}/{op} raised "
                      f"{type(exc).__name__}: {exc}")
                continue
            if err is None:
                print(f"  isolate {n}/{b}/{op}/{algo}: no result")
                continue
            fn = f"{resolve_target(P, e, q, b)}_{op}_{algo}.c"
            floor = floors.get((n, b))
            prov = (f"isolation run: {n}/{b}, {op}/{algo} the ONLY curated "
                    f"kernel, every other op on reference_impl; floor = "
                    f"all-reference model. log "
                    f"{P['results']}/iso_{n}_{b}_{op}_{algo}.log")
            prev = out.get(fn)
            # keep the worst excess over floor across networks
            def _exc(e_, f_):
                return e_ if f_ is None else max(0.0, e_ - f_)
            if prev is None or _exc(err, floor) > _exc(prev[0], prev[1]):
                out[fn] = (err, floor, prov)
            print(f"  isolate {n}/{b}/{op}/{algo}: err={err} floor={floor}",
                  flush=True)
    return out


# ==========================================================================
# Check 3 -- header claims vs. reality
# ==========================================================================

BROKEN_RE = re.compile(
    r"(CONFIRMED BROKEN|is BROKEN|known broken|BROKEN on real hardware)",
    re.I)
# A retraction QUOTES the claim it withdraws, so a naive scan re-flags the
# very header that was just corrected. Text from a RETRACTED/WITHDRAWN
# marker to the end of that paragraph is quoted history, not a live claim.
RETRACT_RE = re.compile(r"\b(RETRACTED|WITHDRAWN|NO LONGER TRUE|"
                        r"SUPERSEDED)\b")


def live_claims(hdr: str) -> list[str]:
    """Claims still being ASSERTED -- quoted-and-retracted text removed."""
    m = RETRACT_RE.search(hdr)
    live = hdr[:m.start()] if m else hdr
    return BROKEN_RE.findall(live)
ERRCLAIM_RE = re.compile(r"max_abs_err\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)")


def header_comment(path: str) -> str:
    """The leading block-comment run of the file -- where the claims live."""
    src = open(path, errors="replace").read()
    out, i, n = [], 0, len(src)
    while i < n:
        while i < n and src[i] in " \t\r\n":
            i += 1
        if not src.startswith("/*", i):
            break
        j = src.find("*/", i + 2)
        j = n if j < 0 else j + 2
        out.append(src[i:j])
        i = j
    return "\n".join(out)


def check_headers(P: dict, measured: dict,
                  isolated: Optional[dict] = None) -> list[Finding]:
    out = []
    for f in sorted(glob.glob(os.path.join(P["kernels"], "*", "*.c"))):
        key = os.path.basename(f)
        hdr = header_comment(f)
        if not hdr:
            continue
        claims_broken = live_claims(hdr)
        errs = [float(x) for x in ERRCLAIM_RE.findall(hdr)]
        basis = "this kernel's own contribution (isolated)"
        iso = (isolated or {}).get(key)
        if iso is not None:
            # Prefer the ISOLATED number: a header's quoted figure is a
            # claim about THIS kernel, and the pipeline's whole-model verify
            # number is not that. Comparing a per-kernel claim against a
            # chain measurement produced 5 of the 6 warnings this check
            # first emitted, all of them spurious.
            err, floor, prov = iso
            err = err if floor is None else max(0.0, err - floor)
            if floor is not None and err <= log_resolution(iso[0]):
                err = 0.0
        elif key in measured:
            err, prov = measured[key]
            basis = ("the WHOLE MODEL with this kernel selected -- no "
                     "isolation run exists, so this number is the chain's, "
                     "not the kernel's")
        else:
            if claims_broken:
                out.append(Finding("headers", key, SKIP,
                                   f"header asserts {claims_broken[0]!r}; "
                                   f"no current measurement to confirm or "
                                   f"refute", evidence="unmeasured this run"))
            continue
        if err is None:
            continue
        # A header that calls something BROKEN while the current build
        # measures 0 is stale and actively misleading.
        if claims_broken and err == 0:
            out.append(Finding(
                "headers", key, FAIL,
                f"header asserts {claims_broken[0]!r} but the current "
                f"selection measures max_abs_err=0",
                evidence=prov,
                data={"claim": claims_broken[0], "measured": err}))
        elif errs and err not in errs:
            out.append(Finding(
                "headers", key, WARN,
                f"REVIEW (not a defect on its own): header quotes "
                f"max_abs_err {sorted(set(errs))}; {basis} measures "
                f"{err:g}. Headers legitimately quote historical and "
                f"pre-fix figures, so this only flags the file for a human "
                f"to read.",
                evidence=prov,
                data={"claimed": sorted(set(errs)), "measured": err}))
        else:
            out.append(Finding("headers", key, PASS,
                               "no header claim contradicted by this run's "
                               "measurement", evidence=prov))
    return out


# ==========================================================================
# Check 4 -- selection staleness (regen into scratch, never in place)
# ==========================================================================

REGEN_SH = r"""
set -uo pipefail
source {zcs}/scripts/activate_conda.sh >/dev/null 2>&1
source {zcs}/scripts/set_envvars_sdk.sh >/dev/null 2>&1
export PYTHONPATH={zcs}
export PATH=/usr/bin:$PATH
{drift}
cd {mb}
exec python -m modelblaster.pipeline.generate_kernels \
  --ir {ir}/graph.json \
  --out-dir {gen} \
  --backend reference \
  --target {target} \
  --quant {quant} \
  --io {ir}/io.npz \
  --repo-root {mb} \
  --build-dir {build} \
  --harness-dir {mb}/harness \
  --cache-dir {cache} \
  --algorithms all \
  --global-curated-dir {mb}/kernels
"""


def resolve_target(P, exdir, quant, backend) -> str:
    """The BACKEND NAME IS NOT THE DIRECTORY NAME.

    examples/_run_lib.sh auto-promotes GEN_TARGET rvv -> rvv_f16 when the IR
    contains any fp16 op, but still writes into generated/${TARGET}. So
    examples/vint/int8/generated/rvv/ was generated with --target rvv_f16.
    Regenerating that directory as `rvv` compiles fp16 intrinsics without
    zvfh and every curated kernel fails to BUILD -- which reads exactly like
    a library-wide defect and is entirely an artifact of the harness.
    Caught by the vint/rvv regen log: 'argument type vfloat16m1_t requires
    the zvfhmin or zvfh ISA extension'.

    kernel_picks.json records the target the pipeline actually used, so it
    is the authority; the directory name is only a fallback.
    """
    pf = os.path.join(P["examples"], exdir, quant, "generated", backend,
                      "kernel_picks.json")
    if os.path.exists(pf):
        try:
            t = json.load(open(pf)).get("target")
            if t:
                return t
        except Exception:                                    # noqa: BLE001
            pass
    gp = os.path.join(P["examples"], exdir, quant, "generated", "graph.json")
    if backend in ("rvv", "scalar") and os.path.exists(gp):
        try:
            g = json.load(open(gp))
            if any("f16" in n["op"] for n in g.get("ops", [])):
                return f"{backend}_f16"
        except Exception:                                    # noqa: BLE001
            pass
    return backend


def _prep_scratch(P, net, exdir, quant, backend) -> dict:
    """Copy the IR + the per-model cache into a scratch tree. NOTHING is
    written back into the example tree."""
    src_ir = os.path.join(P["examples"], exdir, quant, "generated")
    src_gen = os.path.join(src_ir, backend)
    src_cache = os.path.join(P["examples"], exdir, quant, "cache", backend)
    dst = os.path.join(P["scratch"], f"{net}_{backend}")
    ir = os.path.join(dst, "generated")
    gen = os.path.join(ir, backend)
    cache = os.path.join(dst, "cache", backend)
    build = os.path.join(dst, "build", backend)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(gen, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    for fn in ("graph.json", "io.npz", "weights.npz"):
        p = os.path.join(src_ir, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(ir, fn))
    # the skeleton (model.c/h, weights.c/h, buffers.c, test_*) is what
    # generate_kernels' verify builds against -- copy it, do not regenerate,
    # so the ONLY thing that varies between shipped and scratch is selection.
    if os.path.isdir(src_gen):
        for p in glob.glob(os.path.join(src_gen, "*")):
            shutil.copy2(p, os.path.join(gen, os.path.basename(p)))
    if os.path.isdir(src_cache):
        for p in glob.glob(os.path.join(src_cache, "*")):
            shutil.copy2(p, os.path.join(cache, os.path.basename(p)))
    return {"dst": dst, "ir": ir, "gen": gen, "cache": cache, "build": build,
            "shipped_picks": os.path.join(src_gen, "kernel_picks.json")}


VERIFY_LINE = re.compile(
    r"\[(?P<op>[\w]+)/(?P<algo>[\w]+)\]\s+(?P<src>curated|cache)"
    r"(?:\[[\w]+\])?\s+verify\s+(?P<res>PASS|FAIL)\s+[-—]+\s*(?P<msg>.*)")
ERR_IN_MSG = re.compile(r"max_abs_err=([0-9.eE+-]+)")


def parse_regen_log(text: str) -> dict:
    """Per-(op, algorithm) PASS/FAIL + max_abs_err out of a regen log."""
    res = {}
    for m in VERIFY_LINE.finditer(text):
        op, algo, res_, msg = (m.group("op"), m.group("algo"),
                               m.group("res"), m.group("msg"))
        e = ERR_IN_MSG.search(msg)
        err = float(e.group(1)) if e else None
        res[(op, algo)] = {"result": res_, "max_abs_err": err,
                           "msg": msg.strip()[:300]}
    return res


def run_regen(P, net, exdir, quant, backend, drift, timeout=7200) -> dict:
    sc = _prep_scratch(P, net, exdir, quant, backend)
    target = resolve_target(P, exdir, quant, backend)
    drift_line = f'export MB_DRIFT_ATOL="{drift}"' if drift else \
        "unset MB_DRIFT_ATOL"
    sh = REGEN_SH.format(zcs=P["zcs"], mb=P["mb"], ir=sc["ir"], gen=sc["gen"],
                         target=target, quant=quant, build=sc["build"],
                         cache=sc["cache"], drift=drift_line)
    t0 = time.time()
    p = subprocess.run(["bash", "-c", sh], capture_output=True, text=True,
                       timeout=timeout)
    log = p.stdout + "\n" + p.stderr
    logp = os.path.join(P["results"], f"regen_{net}_{backend}.log")
    with open(logp, "w") as fh:
        fh.write(log)
    new_picks = None
    np_path = os.path.join(sc["gen"], "kernel_picks.json")
    if os.path.exists(np_path):
        new_picks = json.load(open(np_path))
    shipped = None
    if os.path.exists(sc["shipped_picks"]):
        shipped = json.load(open(sc["shipped_picks"]))
    return {"net": net, "backend": backend, "target": target,
            "quant": quant, "exdir": exdir,
            "drift": drift, "rc": p.returncode, "log": logp,
            "secs": round(time.time() - t0, 1),
            "verify": parse_regen_log(log),
            "new": new_picks, "shipped": shipped}


def load_regen(P: dict, net, exdir, quant, backend, drift) -> Optional[dict]:
    """Rebuild a regen result from artifacts on disk instead of rebuilding.

    The regen log and the regenerated kernel_picks.json are the whole
    result, so re-scoring is free. Useful for iterating on the analysis
    without spending another 25 minutes of spike time, and it means the
    numbers in a report can be re-derived from committed artifacts.
    """
    logp = os.path.join(P["results"], f"regen_{net}_{backend}.log")
    newp = os.path.join(P["scratch"], f"{net}_{backend}", "generated",
                        backend, "kernel_picks.json")
    shipped = os.path.join(P["examples"], exdir, quant, "generated",
                           backend, "kernel_picks.json")
    if not os.path.exists(logp):
        return None
    return {
        "net": net, "backend": backend,
        "target": resolve_target(P, exdir, quant, backend),
        "quant": quant, "exdir": exdir,
        "drift": drift, "rc": 0, "log": logp, "secs": None,
        "verify": parse_regen_log(open(logp, errors="replace").read()),
        "new": json.load(open(newp)) if os.path.exists(newp) else None,
        "shipped": (json.load(open(shipped))
                    if os.path.exists(shipped) else None),
    }


def check_staleness(P: dict, results: list[dict]) -> tuple[list[Finding], dict]:
    out = []
    measured: dict[str, tuple[Optional[float], str]] = {}
    for r in results:
        tag = f"{r['net']}/{r['backend']}"
        if r["rc"] != 0 or r["new"] is None:
            out.append(Finding("stale", tag, SKIP,
                               f"regen did not complete (rc={r['rc']}); "
                               f"see {r['log']}",
                               evidence=f"MB_DRIFT_ATOL={r['drift']}"))
            continue
        ship = (r["shipped"] or {}).get("picks", {})
        new = (r["new"] or {}).get("picks", {})
        ops = sorted(set(ship) | set(new))
        for op in ops:
            s = ship.get(op) or {}
            n = new.get(op) or {}
            sa = (s.get("source"), s.get("algorithm"))
            na = (n.get("source"), n.get("algorithm"))
            subj = f"{tag}/{op}"
            if sa == na:
                out.append(Finding("stale", subj, PASS,
                                   f"shipped == regenerated: {sa[0]}/{sa[1]}",
                                   evidence=f"regen MB_DRIFT_ATOL={r['drift']}"))
            else:
                kind = ("shipped-but-would-NOT-be-selected"
                        if sa[0] == "curated" and na[0] != "curated"
                        else ("would-now-be-selected-but-is-NOT-shipped"
                              if na[0] == "curated" and sa[0] != "curated"
                              else "algorithm changed"))
                # why?
                why = ""
                for (vop, valgo), v in r["verify"].items():
                    if vop == op and valgo == sa[1] and v["result"] == "FAIL":
                        why = f" -- regen rejects it: {v['msg']}"
                out.append(Finding(
                    "stale", subj, FAIL,
                    f"{kind}: shipped {sa[0]}/{sa[1]} vs regenerated "
                    f"{na[0]}/{na[1]}{why}",
                    evidence=f"regen with MB_DRIFT_ATOL={r['drift']}, "
                             f"log {r['log']}",
                    data={"shipped": s, "regenerated": n}))
        # feed the class/header checks with per-kernel measurements
        for (op, algo), v in r["verify"].items():
            fn = f"{r.get('target', r['backend'])}_{op}_{algo}.c"
            prov = (f"{tag} regen verify {v['result']} "
                    f"(MB_DRIFT_ATOL={r['drift']}, {r['log']})")
            if v["max_abs_err"] is not None:
                prev = measured.get(fn)
                # keep the WORST measurement across networks -- a kernel that
                # is exact on one model and wrong on another is wrong.
                if prev is None or prev[0] is None or \
                        v["max_abs_err"] > prev[0]:
                    measured[fn] = (v["max_abs_err"], prov)
    return out, measured


# ==========================================================================
# Check 4b -- pre-seed layout invariant (pure python, seconds, no build)
# ==========================================================================
# generate_kernels' verify is a WHOLE-MODEL check: it swaps ONE candidate
# into a baseline `impls` dict and compares the model's final output to the
# PyTorch golden. So every kernel in that baseline is part of the
# experiment's control, and a wrong one is attributed to whatever op is
# under test.
#
# The baseline is built by the PRE-SEED pass, which picks, per op, the
# curated file with the lowest accuracy_class. This check asserts the
# invariant that pass must hold: a seeded kernel must be able to read the
# activation layout this graph actually declares. It runs in seconds and
# needs no build, and it is the check that would have caught the
# gemmini_resadd demotion immediately.

def check_seed_layout(P: dict, nets, backends) -> list[Finding]:
    """Assert the pre-seed baseline is layout-correct, by calling the
    PIPELINE'S OWN seed function -- not a copy of it. If someone deletes the
    activation-layout filter from generate_kernels.curated_seed_for, this
    check goes red; a reimplementation here would not notice."""
    sys.path.insert(0, P["zcs"])
    try:
        from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
        from modelblaster.pipeline import backends as backends_mod
        from modelblaster.pipeline.generate_kernels import curated_seed_for
        from modelblaster.pipeline import act_layout as alm
    except Exception as e:                                   # noqa: BLE001
        return [Finding("seed", "-", SKIP,
                        f"cannot import the modelblaster pipeline "
                        f"({type(e).__name__}: {e}); run with "
                        f"PYTHONPATH={P['zcs']}", evidence="")]
    # INDEPENDENT layout predicate. Deliberately NOT
    # generate_kernels.act_layout_candidate_ok: if this check borrowed the
    # pipeline's own predicate, neutering that predicate would neuter the
    # check too, and the suite would stay green through exactly the
    # regression it exists to catch. (Verified: monkeypatching
    # act_layout_candidate_ok to return True left the borrowed version at
    # 109 PASS / 0 FAIL.)
    def layout_ok(ir, op, algo, target) -> bool:
        if op in alm.RELAYOUT_OPS or op in alm.LAYOUT_AGNOSTIC_OPS:
            return True
        want = alm.kernel_layout_for_kind(ir, op, target.name)
        declared = tuple(getattr(algo, "act_layouts", None) or (alm.NCHW,))
        return want in declared

    out = []
    for name, exdir, quant, drift in NETWORKS:
        if nets and name not in nets:
            continue
        gpath = os.path.join(P["examples"], exdir, quant, "generated",
                             "graph.json")
        if not os.path.exists(gpath):
            continue
        ir = json.load(open(gpath))
        ops = sorted({o["op"] for o in ir.get("ops", [])
                      if o["op"] in KERNEL_SPECS})
        specs = [KERNEL_SPECS[o] for o in ops]
        for b in backends:
            try:
                target = backends_mod.get(b)
            except Exception:                                # noqa: BLE001
                continue
            try:
                seed = curated_seed_for(ir, specs, target, P["kernels"])
            except SystemExit as e:
                out.append(Finding("seed", f"{name}/{b}", FAIL,
                                   f"curated_seed_for raised SystemExit: {e}",
                                   evidence=gpath))
                continue
            for spec in specs:
                subj = f"{name}/{b}/{spec.op}"
                got = seed.get(spec.op)
                # what WOULD have been seeded with no layout filter
                naive = None
                for a in sorted(spec.algorithms,
                                key=lambda x: (getattr(x, "accuracy_class", 0)
                                               or 0)):
                    fp = os.path.join(P["kernels"], target.name,
                                      f"{target.name}_{spec.op}_{a.name}.c")
                    if os.path.exists(fp):
                        naive = a
                        break
                if naive is None:
                    continue          # no curated file for this op/backend
                want = alm.kernel_layout_for_kind(ir, spec.op, target.name)
                if got is None:
                    out.append(Finding(
                        "seed", subj, WARN,
                        f"a curated file exists ({naive.name}) but nothing "
                        f"is seeded for this op; graph needs {want!r}",
                        evidence=gpath))
                elif not layout_ok(
                        ir, spec.op,
                        next(a for a in spec.algorithms
                             if a.name == got[0]), target):
                    out.append(Finding(
                        "seed", subj, FAIL,
                        f"pre-seed installs {got[0]} whose act_layouts do "
                        f"not match the graph's {want!r}. A layout-wrong "
                        f"seed is size-identical, so nothing downstream "
                        f"catches it; it poisons the verify baseline and "
                        f"the blame lands on whichever op is verified "
                        f"first.",
                        evidence=f"{gpath} + {os.path.basename(got[1])}",
                        data={"op": spec.op, "seeded": got[0],
                              "needs": want}))
                elif naive.name != got[0]:
                    out.append(Finding(
                        "seed", subj, PASS,
                        f"pre-seed correctly SKIPPED {naive.name} "
                        f"(act_layouts="
                        f"{getattr(naive, 'act_layouts', ('nchw',))}, graph "
                        f"needs {want!r}) and seeded {got[0]} instead -- the "
                        f"layout filter is load-bearing here",
                        evidence=f"{gpath} + {os.path.basename(got[1])}"))
                else:
                    out.append(Finding(
                        "seed", subj, PASS,
                        f"pre-seed installs {got[0]}, layout {want!r} ok",
                        evidence=f"{gpath} + {os.path.basename(got[1])}"))
    return out


def load_floors(P: dict) -> dict:
    """All-reference whole-model errors, keyed (network, backend)."""
    floors: dict = {}
    fj = os.path.join(P["results"], "floors.json")
    if os.path.exists(fj):
        for k, v in json.load(open(fj)).items():
            n, b = k.split("/", 1)
            floors[(n, b)] = v
    for lg in glob.glob(os.path.join(P["results"], "*run*.log")):
        for m in re.finditer(r"^  floor (\S+?)/(\S+?): (\S+)$",
                             open(lg, errors="replace").read(), re.M):
            try:
                floors.setdefault((m.group(1), m.group(2)),
                                  float(m.group(3)))
            except ValueError:
                pass
    return floors


def check_gate(P: dict, regen_results: list[dict], floors: dict
               ) -> list[Finding]:
    """Is the curated-verify gate even PASSABLE for this (network, backend)?

    generate_kernels' gate compares the WHOLE MODEL against the PyTorch
    golden. When the all-reference chain itself already exceeds the
    tolerance, no curated kernel can pass however good it is, and the tree
    silently ships 100% reference_impl while the logs read like a
    library-wide correctness failure.

    MEASURED: vint's all-reference model reads max_abs_err=0.327884674 on
    BOTH gemmini_q31 and rvv_f16. gemmini_q31 carries atol_override=1.0
    (pipeline/backends.py) so it passes and 21 curated kernels get selected;
    rvv and rvv_f16 carry atol_override=None, so their gate is
    max(1e-5, 1e-4*sqrt(K)) -- two to three orders of magnitude BELOW the
    floor -- and all 29 curated candidates are rejected, every one of them
    at the floor value. That is a gate defect, not a kernel defect.
    """
    sys.path.insert(0, P["zcs"])
    try:
        from modelblaster.pipeline import backends as backends_mod
    except Exception as e:                                   # noqa: BLE001
        return [Finding("gate", "-", SKIP,
                        f"cannot import backends ({type(e).__name__}: {e})",
                        evidence="")]
    out = []
    for r in regen_results:
        key = (r["net"], r["backend"])
        subj = f"{r['net']}/{r['backend']}"
        floor = floors.get(key)
        if floor is None:
            out.append(Finding("gate", subj, SKIP,
                               "no all-reference floor measured for this "
                               "cell", evidence=""))
            continue
        try:
            tgt = backends_mod.get(r.get("target", r["backend"]))
        except Exception:                                    # noqa: BLE001
            continue
        ov = getattr(tgt, "atol_override", None)
        drift = float(r["drift"]) if r["drift"] else 0.0
        # The most generous gate any op in this model can get.
        gate = max(ov if ov is not None else 1e-2, drift)
        if floor <= gate:
            out.append(Finding(
                "gate", subj, PASS,
                f"all-reference floor {floor:g} <= the loosest gate "
                f"{gate:g} (atol_override={ov}, MB_DRIFT_ATOL={r['drift']}) "
                f"-- a good kernel can pass",
                evidence=r["log"]))
            continue
        n_fail = sum(1 for v in r["verify"].values() if v["result"] == "FAIL")
        out.append(Finding(
            "gate", subj, FAIL,
            f"UNPASSABLE GATE: the ALL-REFERENCE model already reads "
            f"max_abs_err={floor:g}, above the loosest gate {gate:g} "
            f"(atol_override={ov}, MB_DRIFT_ATOL={r['drift']}). No curated "
            f"kernel can be selected here however good it is -- "
            f"{n_fail}/{len(r['verify'])} candidates were rejected this "
            f"run. This tree ships 100% reference_impl for a reason that "
            f"is not about any kernel.",
            evidence=f"measured floor + {r['log']}",
            data={"floor": floor, "gate": gate, "atol_override": ov}))
    return out


# ==========================================================================
# Check 5 -- reachability: is a racy kernel actually co-scheduled?
# ==========================================================================
# uartlog dispatch rows are 14 comma-separated fields:
#   entry_id,network,instance,dispatch_id,op,name,core_kind,hart,
#   pred_start_ms,pred_dur_ms,worker_kind_idx,worker_hart,
#   actual_start,actual_end          (last two = mtime ticks, us)
# A shared-workspace kernel is only *reached* where two dispatches of the
# SAME op overlap in time on DIFFERENT worker_hart.

def overlaps_by_op(uartlog: str) -> Counter:
    """Counter keyed (network, core_kind, op).

    The key MUST carry core_kind. Keying on the op alone over-attributes: a
    yolov8_nano gempair arm splits silu_s8 across two gemmini harts (56
    overlaps on shardec), and an op-only key would then report the RVV
    silu kernel as 'reached' -- which it is not. The rvv-pair arms run 57
    silu calls with ZERO concurrent on different harts.
    """
    rows = []
    for line in open(uartlog, errors="replace"):
        f = line.strip().split(",")
        if len(f) != 14:
            continue
        try:
            rows.append(((f[1], f[6], f[4]), f[11], int(f[12]), int(f[13])))
        except ValueError:
            continue
    by_key = defaultdict(list)
    for key, hart, a, b in rows:
        by_key[key].append((hart, a, b))
    c = Counter()
    for key, iv in by_key.items():
        iv.sort(key=lambda x: x[1])
        for i in range(len(iv)):
            for j in range(i + 1, len(iv)):
                if iv[j][1] >= iv[i][2]:
                    break
                if iv[j][0] != iv[i][0]:
                    c[key] += 1
    return c


def build_reachability(P: dict, uartlogs: list[str],
                       picks_files: list[str]) -> dict:
    """kernel basename -> {'n': cross-hart same-op overlaps observed,
    'where': the (network, core_kind, op) keys that produced them}.

    A kernel is credited with an overlap only when the overlapping
    dispatches ran on a core_kind whose curated directory that kernel lives
    in -- i.e. the backend actually executing the op.
    """
    ov = Counter()
    src: dict = defaultdict(set)
    kinds_seen: set = set()          # core_kinds with ANY dispatch row
    for u in uartlogs:
        c = overlaps_by_op(u)
        ov.update(c)
        for k, v in c.items():
            if v:
                src[k].add(os.path.basename(u))
        for line in open(u, errors="replace"):
            f = line.strip().split(",")
            if len(f) == 14 and f[6] != "core_kind":
                kinds_seen.add(f[6])
    kern: dict = {}
    for pf in picks_files:
        try:
            d = json.load(open(pf))
        except Exception:
            continue
        backend = os.path.basename(os.path.dirname(pf))
        for op, p in (d.get("picks") or {}).items():
            path = p.get("path")
            if not path:
                continue
            base = os.path.basename(path)
            if os.path.basename(os.path.dirname(path)) != backend:
                continue        # aliased backend; credited under its own dir
            hits = [(k, v) for k, v in ov.items()
                    if k[2] == op and k[1] == backend and v]
            n = max([v for _, v in hits], default=0)
            cur = kern.setdefault(
                base, {"n": 0, "where": [],
                       # Distinguish "measured not to overlap" from "no
                       # measurement exists". Every 14-field dispatch row in
                       # every uartlog in this repo carries
                       # core_kind=gemmini_q31 -- there is not a single rvv
                       # dispatch row -- so an rvv kernel's reachability is
                       # NOT ESTABLISHED here, which is a different claim
                       # from LATENT.
                       "backend_observed": backend in kinds_seen})
            if n > cur["n"]:
                cur["n"] = n
                cur["where"] = [f"{k[0]}/{k[1]}/{k[2]} x{v} "
                                f"({', '.join(sorted(src[k])[:2])})"
                                for k, v in hits]
            cur["backend_observed"] = (cur["backend_observed"]
                                       or backend in kinds_seen)
    return kern


# ==========================================================================
# Tier 2 -- FPGA pair arms vs single-hart ground truth
# ==========================================================================

TIER2_PAIRS = ["rvvpair", "gempair", "hetero"]
TIER2_SERIAL = ["serialE", "serialP"]
TIER2_ARMS = ["base", "shard", "shardec"]

VERIFY_UART = re.compile(
    r"MODELBLASTER_VERIFY\s*\[(?P<net>[\w]+)\]\s*===\s*"
    r"max_abs_err=(?P<abs>[0-9.eE+-]+)\s+max_rel_err=(?P<rel>[0-9.eE+-]+)")
# the sweep names its logs <net>_<pair>_<arm>.uartlog
LOGNAME = re.compile(r"^(?P<net>.+?)_(?P<pair>rvvpair|gempair|hetero|serialE|"
                     r"serialP)_(?P<arm>\w+)\.uartlog$")


def emit_tier2_plan(P: dict, path: str) -> None:
    """Write the exact, literal commands Tier 2 needs. Someone else runs
    these; this script never submits an FPGA job itself."""
    S = os.path.join(P["root"], "experiments/shard_dim/scripts")
    lines = [
        "#!/usr/bin/env bash",
        "# Tier 2 of the kernel validation suite -- FPGA, the only tier that",
        "# can catch concurrency. GENERATED by validate_kernels.py",
        "# --tier2-emit-plan; nothing here has been run by that script.",
        "#",
        "# FPGA QUEUE ETIQUETTE (non-negotiable, SHARED infrastructure):",
        "#   manager ubuntu@3.88.218.39, key ~/.ssh/firesim.pem, 4 lanes.",
        "#   NEVER terminate the manager, never restart its daemon, never",
        "#   `firesim kill`, never cancel a job you did not submit. If the",
        "#   lanes are busy, wait. Check with:",
        "#     ssh -i ~/.ssh/firesim.pem ubuntu@3.88.218.39 \\",
        "#       \"cd /home/ubuntu/fpga_queue 2>/dev/null||cd /home/ubuntu;"
        " ./bin/fq status\"",
        "#",
        "# WHAT IT ASSERTS: for each (network, pair) the pair arm's",
        "# max_abs_err must EQUAL the network's single-hart value EXACTLY.",
        "# Not 'smaller', not 'PASS'. One hart cannot race itself, so the",
        "# serial arm is ground truth; any difference is concurrency-",
        "# dependent output, which is a defect even when it is small.",
        "#",
        "# COST: 4 networks x (2 serial + 3 pairs x 3 arms) = 44 jobs of",
        "# several minutes each. Budget it. Tier 1 tells you where to aim:",
        "# run the arms that have never been validated and any kernel Tier 1",
        "# flags racy-and-reached, before brute-forcing the matrix.",
        "set -euo pipefail",
        f"cd {S}",
        "",
    ]
    for net, exdir, quant, drift in NETWORKS:
        lines.append(f"# ---- {net} ({exdir}, {quant}) ----")
        for pair in TIER2_SERIAL:
            lines.append(f"SWEEP_FORCE_SERIAL_COSTS=1 bash sweep_pair.sh "
                         f"{net} {exdir} {quant} {pair} base")
        for pair in TIER2_PAIRS:
            for arm in TIER2_ARMS:
                lines.append(f"SWEEP_FORCE_SERIAL_COSTS=1 bash sweep_pair.sh "
                             f"{net} {exdir} {quant} {pair} {arm}")
        lines.append("")
    lines += [
        "# submit, then collect:",
        "#   bash sweep_submit.sh <tag>",
        f"#   {sys.executable} analyze_sweep3net.py mlp_control dronet "
        "yolov8_nano vint",
        "#     (pass the network names explicitly -- auto-discovery",
        "#      mis-splits mlp_control and yolov8_nano)",
        "#",
        "# then score the collected uartlogs:",
        f"#   {sys.executable} "
        f"{P['self']}/validate_kernels.py --tier2 \\",
        "#       --tier2-uartlogs '<dir>/*.uartlog'",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    os.chmod(path, 0o755)


def check_tier2(P: dict, uartlog_glob: Optional[str]) -> list[Finding]:
    """Score Tier 2 from collected uartlogs.

    Each run prints one
      === MODELBLASTER_VERIFY [<net>] === max_abs_err=.. max_rel_err=.. ..
    line, and the sweep names its logs <net>_<pair>_<arm>.uartlog, so the
    whole tier is scoreable from a directory of logs with no manual
    transcription. The assertion is EQUALITY with the single-hart
    (serialE/serialP) value for the same network -- one hart cannot race
    itself, so that is ground truth. 'smaller' is not the test and 'PASS'
    is not the test.
    """
    if not uartlog_glob:
        return [Finding("tier2", "-", SKIP,
                        "not run: pass --tier2-uartlogs '<dir>/*.uartlog' "
                        "with logs collected from the sweep "
                        "(--tier2-emit-plan writes the exact commands)",
                        evidence="")]
    logs = sorted(glob.glob(uartlog_glob))
    if not logs:
        return [Finding("tier2", "-", SKIP,
                        f"no uartlogs matched {uartlog_glob!r}",
                        evidence="")]
    rows = []
    for p in logs:
        m = LOGNAME.match(os.path.basename(p))
        if not m:
            continue
        txt = open(p, errors="replace").read()
        for v in VERIFY_UART.finditer(txt):
            if v.group("net") != m.group("net"):
                continue
            rows.append({"net": m.group("net"), "pair": m.group("pair"),
                         "arm": m.group("arm"),
                         "max_abs_err": float(v.group("abs")),
                         "log": p})
    ground: dict = {}
    for r in rows:
        if r["pair"] in TIER2_SERIAL:
            ground.setdefault(r["net"], []).append(r)
    out = []
    for r in sorted(rows, key=lambda x: (x["net"], x["pair"], x["arm"])):
        subj = f"{r['net']}/{r['pair']}/{r['arm']}"
        if r["pair"] in TIER2_SERIAL:
            out.append(Finding("tier2", subj, PASS,
                               f"single-hart ground truth "
                               f"max_abs_err={r['max_abs_err']:g}",
                               evidence=r["log"]))
            continue
        g = ground.get(r["net"])
        if not g:
            out.append(Finding(
                "tier2", subj, SKIP,
                f"max_abs_err={r['max_abs_err']:g} but no single-hart "
                f"(serialE/serialP) run for {r['net']} in this log set, so "
                f"there is nothing to compare against",
                evidence=r["log"]))
            continue
        gv = {x["max_abs_err"] for x in g}
        if len(gv) > 1:
            out.append(Finding("tier2", subj, SKIP,
                               f"single-hart runs disagree ({sorted(gv)}); "
                               f"no unambiguous ground truth",
                               evidence=", ".join(x["log"] for x in g)))
        elif r["max_abs_err"] == next(iter(gv)):
            out.append(Finding("tier2", subj, PASS,
                               f"max_abs_err={r['max_abs_err']:g} == "
                               f"single-hart {next(iter(gv)):g}",
                               evidence=f"{r['log']} vs {g[0]['log']}"))
        else:
            out.append(Finding(
                "tier2", subj, FAIL,
                f"max_abs_err={r['max_abs_err']:g} != single-hart "
                f"{next(iter(gv)):g} -- output depends on how the work was "
                f"scheduled across harts",
                evidence=f"{r['log']} vs {g[0]['log']}",
                data={"arm": r, "ground": g}))
    return out


# ==========================================================================
# Reporting
# ==========================================================================

def coverage(P: dict, findings: list[Finding]) -> str:
    """Honest coverage accounting against the WHOLE library, not just what
    the suite happened to touch. Partial coverage that is labelled is worth
    more than a green table over everything."""
    files = sorted(glob.glob(os.path.join(P["kernels"], "*", "*.c")))
    all_names = {os.path.basename(f) for f in files}
    sel_files, sel_triples = set(), set()
    for pf in glob.glob(os.path.join(P["examples"], "**",
                                     "kernel_picks.json"), recursive=True):
        try:
            d = json.load(open(pf))
        except Exception:
            continue
        for op, pk in (d.get("picks") or {}).items():
            if pk.get("algorithm"):
                sel_triples.add((d.get("target"), op, pk["algorithm"]))
                if pk.get("path"):
                    sel_files.add(os.path.basename(pk["path"]))
    def subj(check, statuses=(PASS, FAIL, WARN)):
        return {f.subject for f in findings
                if f.check == check and f.status in statuses}
    hart = subj("hart")
    measured = {f.subject for f in findings
                if f.check in ("class", "headers") and f.status != SKIP}
    stale_cells = {f.subject for f in findings if f.check == "stale"
                   and f.status != SKIP}
    # Which of the 101 distinct (target, op, algorithm) selections do those
    # cells actually cover? Counting cells double-counts a selection that
    # several networks share, and can exceed the 101 total.
    audited_nets = {f.subject.split("/")[0] for f in findings
                    if f.check == "stale" and f.status != SKIP}
    covered_triples = set()
    for name, exdir, quant, _d in NETWORKS:
        if name not in audited_nets:
            continue
        for pf in glob.glob(os.path.join(P["examples"], exdir, quant,
                                         "generated", "*",
                                         "kernel_picks.json")):
            try:
                d = json.load(open(pf))
            except Exception:
                continue
            for op, pk in (d.get("picks") or {}).items():
                if pk.get("algorithm"):
                    covered_triples.add((d.get("target"), op,
                                         pk["algorithm"]))
    t2 = {f.subject for f in findings if f.check == "tier2"
          and f.status != SKIP}
    L = [
        "COVERAGE",
        "--------",
        f"  kernel .c files in kernels/*/*.c        {len(all_names):4d}",
        f"    scanned by the hart-safety check      {len(hart & all_names):4d}"
        f"   ({100*len(hart & all_names)//max(1,len(all_names))}%)",
        f"    numerically measured this run         {len(measured):4d}"
        f"   (only kernels some measured network actually selects)",
        f"    never selected by ANY example tree    "
        f"{len(all_names - sel_files):4d}   (static checks only; no "
        f"numeric evidence exists for these)",
        f"  distinct (target, op, algorithm) picks  {len(sel_triples):4d}",
        f"    re-selected + diffed this run         "
        f"{len(covered_triples & sel_triples):4d}   "
        f"(across {len(stale_cells)} (network, backend, op) cells)",
        f"  Tier 2 arms scored                      {len(t2):4d}",
    ]
    return "\n".join(L)


def summarise(findings: list[Finding]) -> str:
    by = defaultdict(Counter)
    for f in findings:
        by[f.check][f.status] += 1
    w = max([len(k) for k in by] + [8])
    lines = [f"{'check'.ljust(w)}  {'PASS':>5} {'FAIL':>5} {'WARN':>5} "
             f"{'SKIP':>5}", "-" * (w + 26)]
    tot = Counter()
    for k in sorted(by):
        c = by[k]
        tot.update(c)
        lines.append(f"{k.ljust(w)}  {c[PASS]:>5} {c[FAIL]:>5} "
                     f"{c[WARN]:>5} {c[SKIP]:>5}")
    lines.append("-" * (w + 26))
    lines.append(f"{'TOTAL'.ljust(w)}  {tot[PASS]:>5} {tot[FAIL]:>5} "
                 f"{tot[WARN]:>5} {tot[SKIP]:>5}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validation suite for the curated kernel library.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--tier1", action="store_true")
    ap.add_argument("--tier2", action="store_true")
    ap.add_argument("--tier2-uartlogs", default=None,
                    help="glob of uartlogs collected from the FPGA sweep, "
                         "named <net>_<pair>_<arm>.uartlog")
    ap.add_argument("--tier2-emit-plan", default=None,
                    help="write the literal sweep commands Tier 2 needs to "
                         "this path and exit (submits nothing)")
    ap.add_argument("--reuse-isolate", action="store_true",
                    help="re-score isolation from results/iso_*.log + "
                         "results/floors.json without rebuilding")
    ap.add_argument("--isolate-all", action="store_true",
                    help="isolate EVERY curated selection, not just the "
                         "ones the cheap whole-model evidence leaves "
                         "indeterminate")
    ap.add_argument("--reuse-regen", action="store_true",
                    help="re-score the staleness audit from the regen logs "
                         "and scratch picks already on disk, without "
                         "rebuilding anything")
    ap.add_argument("--no-regen", action="store_true",
                    help="Tier 1 static checks only -- skip the selection "
                         "staleness regen (which builds and needs the tree)")
    ap.add_argument("--checks",
                    default="selftest,hart,seed,stale,isolate,gate,class,headers",
                    help="comma list of Tier 1 checks to run")
    ap.add_argument("--nets", default=None,
                    help="comma list of network names (default: all four)")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--uartlogs", default=None,
                    help="glob of uartlogs for the reachability evidence")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not (args.tier1 or args.tier2 or args.tier2_emit_plan):
        args.tier1 = True
    P = paths(args.root)
    if not os.path.isdir(P["kernels"]):
        print(f"ERROR: no kernel library at {P['kernels']}", file=sys.stderr)
        return 2
    if args.tier2_emit_plan:
        os.makedirs(os.path.dirname(os.path.abspath(args.tier2_emit_plan)),
                    exist_ok=True)
        emit_tier2_plan(P, args.tier2_emit_plan)
        print(f"tier 2 plan -> {args.tier2_emit_plan} (nothing submitted)")
        if not (args.tier1 or args.tier2):
            return 0
    os.makedirs(P["results"], exist_ok=True)
    os.makedirs(P["scratch"], exist_ok=True)
    checks = set(args.checks.split(","))
    findings: list[Finding] = []

    # ---- reachability evidence (cheap, feeds the hart check) ----
    ulogs = sorted(glob.glob(args.uartlogs)) if args.uartlogs else \
        sorted(glob.glob(os.path.join(P["root"], "experiments", "**",
                                      "*.uartlog"), recursive=True))
    pfiles = sorted(glob.glob(os.path.join(P["examples"], "**",
                                           "kernel_picks.json"),
                              recursive=True))
    reached = build_reachability(P, ulogs, pfiles) if ulogs else {}
    _r = sum(1 for v in reached.values() if v.get("n"))
    _obs = sum(1 for v in reached.values() if v.get("backend_observed"))
    print(f"[reach] {len(ulogs)} uartlog(s); {len(reached)} kernel(s) are "
          f"shipped by some selection, {_obs} of those have their core_kind "
          f"present in a log at all, {_r} are observed co-scheduled on >1 "
          f"hart", flush=True)

    regen_results: list[dict] = []
    measured: dict[str, tuple[Optional[float], str]] = {}

    if args.tier1:
        if "stale" in checks and args.reuse_regen:
            nets = args.nets.split(",") if args.nets else None
            for name, exdir, quant, drift in NETWORKS:
                if nets and name not in nets:
                    continue
                for b in args.backends.split(","):
                    r = load_regen(P, name, exdir, quant, b, drift)
                    if r:
                        regen_results.append(r)
            print(f"[stale] re-scored {len(regen_results)} regen(s) from "
                  f"artifacts on disk (nothing rebuilt)", flush=True)
            sf, measured = check_staleness(P, regen_results)
            findings += sf
        elif "stale" in checks and not args.no_regen:
            nets = args.nets.split(",") if args.nets else None
            todo = []
            for name, exdir, quant, drift in NETWORKS:
                if nets and name not in nets:
                    continue
                for b in args.backends.split(","):
                    if not os.path.isdir(os.path.join(
                            P["examples"], exdir, quant, "generated", b)):
                        continue
                    todo.append((name, exdir, quant, b, drift))
            print(f"[stale] regenerating selection for {len(todo)} "
                  f"(network, backend) pair(s) in "
                  f"{P['scratch']} -- NEVER in place", flush=True)
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_regen, P, n, e, q, b, d)
                        for n, e, q, b, d in todo]
                for f in futs:
                    try:
                        r = f.result()
                    except Exception as exc:      # noqa: BLE001
                        print(f"  regen raised {type(exc).__name__}: {exc}")
                        continue
                    regen_results.append(r)
                    print(f"  {r['net']}/{r['backend']}: rc={r['rc']} "
                          f"{r['secs']}s "
                          f"{len(r['verify'])} verify result(s)", flush=True)
            sf, measured = check_staleness(P, regen_results)
            findings += sf
        elif "stale" in checks:
            findings.append(Finding("stale", "-", SKIP,
                                    "--no-regen: selection staleness not "
                                    "evaluated", evidence=""))
        if "selftest" in checks:
            findings += run_selftest(P)
        if "seed" in checks:
            findings += check_seed_layout(
                P, args.nets.split(",") if args.nets else None,
                args.backends.split(","))
        if "hart" in checks:
            findings += check_hart_safety(P, reached)
        isolated: dict = {}
        if "isolate" in checks and args.reuse_isolate:
            isolated = load_isolation(P)
            print(f"[isolate] re-scored {len(isolated)} isolation "
                  f"measurement(s) from artifacts on disk (nothing rebuilt)",
                  flush=True)
        elif "isolate" in checks:
            isolated = run_isolation_campaign(
                P, regen_results, args.jobs,
                only_indeterminate=not args.isolate_all)
        if "gate" in checks:
            findings += check_gate(P, regen_results, load_floors(P))
        if "class" in checks:
            findings += check_accuracy_class(P, measured, isolated)
        if "headers" in checks:
            findings += check_headers(P, measured, isolated)

    if args.tier2:
        findings += check_tier2(P, args.tier2_uartlogs)

    print()
    print(summarise(findings))
    print()
    print(coverage(P, findings))
    print()
    fails = [f for f in findings if f.status == FAIL]
    if fails:
        print(f"{len(fails)} FAILING check(s):")
        for f in fails:
            print(f"  [{f.check}] {f.subject}: {f.detail}")
            if f.evidence:
                print(f"      evidence: {f.evidence}")
    out = args.json_out or os.path.join(P["results"], "findings.json")
    with open(out, "w") as fh:
        json.dump([asdict(f) for f in findings], fh, indent=1, default=str)
    print(f"\nfull findings -> {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
